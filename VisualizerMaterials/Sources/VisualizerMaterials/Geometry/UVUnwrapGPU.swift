import Foundation
import simd

//  UVUnwrapGPU.swift — LSCM with the whole conjugate-gradient solve in Metal (ticket item M3).
//
//  Guarantees: `UVUnwrap.lscm(_:island:options:backend:)` solves the same Jacobi-preconditioned CG on the normal
//  equations as `UVUnwrap.lscm`, started from the same planar projection with the same two pins, then makes the same
//  layout (un-mirrored, area-normalised, min-rectangle aligned, at the origin). On the GPU: the per-triangle
//  conformal weights, their area/normal reductions, the matrix-free MᵀM·v (a per-triangle pass then a deterministic
//  per-vertex gather over a CSR corner list — no float atomics, so a run is repeatable), the Jacobi diagonal, the
//  dot products (threadgroup reductions into partials, folded by a one-group kernel that also computes α and β, so
//  no scalar ever visits the CPU inside a batch), axpy and the direction update. One command buffer carries
//  `batch` iterations; the CPU looks at the residual between batches only. Each batch starts by recomputing the true
//  residual (a restart), which keeps Float CG at ~1e-6 of its start where a pure recurrence stalls.
//  Doesn't: reach 1e-10 (Float; the tolerance is floored at 1e-6, and the stop is also "stalled"), so UVs agree with
//  the CPU's to ~1e-3 of the island's size on well-conditioned islands, not to 1e-9. Stays on the CPU: the
//  corner→CSR counting sort, the pin search (a farthest-pair scan over boundary vertices needs Double), the planar
//  start vector (one dot per vertex, uploaded once), and the final area scale. All O(n) with no per-iteration cost.

public extension UVUnwrap {
    static func lscm(_ mesh: Mesh3, island: UVIsland, options: UVUnwrapOptions = .init(), backend: UVBackend) -> UVIslandLayout {
        #if canImport(Metal)
        if backend.useGPU(work: island.triangles.count, threshold: 30_000), let g = UVGPU.shared {
            let mb = UVGPUMesh(g, mesh)
            if let r = UVUnwrapGPU.lscm(mb, mesh, island, options) { return r }
        }
        #endif
        return lscm(mesh, island: island, options: options)
    }
}

public extension Mesh3 {
    /// `unwrappedUVs(seams:options:pack:)` with every island's solve and the packing run on `backend`.
    func unwrappedUVs(seams: UVSeams, options: UVUnwrapOptions = .init(), pack: UVPackOptions? = UVPackOptions(),
                      backend: UVBackend) -> UVUnwrapResult {
        let topo = UVTopology(self)
        let islands = topo.islands(seams)
        var layouts: [UVIslandLayout]
        #if canImport(Metal)
        let mb = backend == .cpu ? nil : UVGPU.shared.map { UVGPUMesh($0, self) }
        layouts = islands.map { isl in
            if let mb, backend.useGPU(work: isl.triangles.count, threshold: 30_000), let r = UVUnwrapGPU.lscm(mb, self, isl, options) { return r }
            return UVUnwrap.lscm(self, island: isl, options: options)
        }
        #else
        layouts = islands.map { UVUnwrap.lscm(self, island: $0, options: options) }
        #endif
        if let pack { layouts = UVPackGPU.pack(layouts, mesh: self, options: pack, backend: backend) }
        let out = applying(layouts)
        return UVUnwrapResult(mesh: out, layouts: layouts, report: UVReport(out, backend: backend))
    }

    /// `packedUVs(_:)` on `backend`.
    func packedUVs(_ options: UVPackOptions = UVPackOptions(), backend: UVBackend) -> UVUnwrapResult {
        let layouts = UVPackGPU.pack(uvLayouts(), mesh: self, options: options, backend: backend)
        let out = applying(layouts)
        return UVUnwrapResult(mesh: out, layouts: layouts, report: UVReport(out, backend: backend))
    }
}

enum UVUnwrapGPU {
    static var source: String {
        #if canImport(Metal)
        kernelSource
        #else
        ""
        #endif
    }
}

#if canImport(Metal)
import Metal

/// A mesh's positions and indices on the device, shared by every island of one unwrap.
final class UVGPUMesh: @unchecked Sendable {
    let g: UVGPU, pos: MTLBuffer, idx: MTLBuffer
    init(_ g: UVGPU, _ m: Mesh3) { self.g = g; pos = g.buffer(UVGPU.float4s(m.positions)); idx = g.buffer(m.indices) }
}

extension UVUnwrapGPU {
    struct LP { var n: UInt32 = 0, T: UInt32 = 0, groups: UInt32 = 0, first: UInt32 = 0; var tol2: Float = 0 }
    static let batch = 256

    static func lscm(_ mb: UVGPUMesh, _ mesh: Mesh3, _ island: UVIsland, _ options: UVUnwrapOptions) -> UVIslandLayout? {
        let g = mb.g
        let n = island.vertexCount, T = island.triangles.count
        guard n > 2, T > 0 else { return nil }
        let gv = (n + 255) / 256, gt = (T + 255) / 256

        // --- GPU: per-triangle weights and the area / normal sums ---
        let tris = g.buffer(island.triangles.map { UInt32($0) }), tl = g.buffer(island.corners.map { UInt32($0) })
        let wr = g.empty(T * 16), wi = g.empty(T * 16), setupPart = g.empty(gt * 16)
        var lp = LP(n: UInt32(n), T: UInt32(T), groups: UInt32(gt))
        guard g.run({ enc in
            call(g, enc, "lscm_setup", T, [mb.pos, mb.idx, tris, wr, wi, setupPart], &lp)
        }) else { return nil }
        let wrv = UVGPU.read(wr, SIMD4<Float>.self, count: T)
        let sp = UVGPU.read(setupPart, SIMD4<Float>.self, count: gt)
        let area3D = sp.reduce(0.0) { $0 + Double($1.x) }
        let nsum = sp.reduce(Vec3.zero) { $0 + Vec3(Double($1.y), Double($1.z), Double($1.w)) }
        guard area3D > 0 else { return nil }

        // --- CPU: pins and the planar start (O(n) bookkeeping, needs Double) ---
        var pos = [Vec3](repeating: .zero, count: n)
        var used = [Bool](repeating: false, count: n)
        var keys: [UInt64] = []; keys.reserveCapacity(3 * T)
        for (k, t) in island.triangles.enumerated() {
            let l = (0..<3).map { island.corners[3 * k + $0] }
            for c in 0..<3 {
                pos[l[c]] = mesh.positions[Int(mesh.indices[3 * t + c])]
                let a = l[c], b = l[(c + 1) % 3]
                if a != b { keys.append(UInt64(min(a, b)) << 32 | UInt64(max(a, b))) }
                if wrv[k].w > 0 { used[l[c]] = true }
            }
        }
        keys.sort()
        var cand = [Bool](repeating: false, count: n), anyCand = false
        var i0 = 0
        while i0 < keys.count {
            var j = i0; while j < keys.count, keys[j] == keys[i0] { j += 1 }
            if j - i0 == 1 { cand[Int(keys[i0] >> 32)] = true; cand[Int(keys[i0] & 0xffff_ffff)] = true; anyCand = true }
            i0 = j
        }
        var pool = (0..<n).filter { used[$0] && (!anyCand || cand[$0]) }
        if pool.isEmpty { pool = (0..<n).filter { used[$0] } }
        guard let first = pool.first else { return nil }
        func farthest(_ from: Int) -> Int { pool.max { len3(pos[$0] - pos[from]) < len3(pos[$1] - pos[from]) }! }
        let pa = farthest(farthest(first)), pb = farthest(pa)
        let dpin = max(len3(pos[pa] - pos[pb]), 1e-12)
        let nAvg = len3(nsum) > 1e-14 ? normalize3(nsum) : Vec3(0, 0, 1)
        var xa = pos[pb] - pos[pa]; xa -= nAvg * dot3(xa, nAvg)
        xa = len3(xa) > 1e-12 ? normalize3(xa) : normalize3(cross3(nAvg, abs(nAvg.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0)))
        let ya = cross3(nAvg, xa)
        var x0 = [SIMD2<Float>](repeating: .zero, count: n)
        var fixed = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            if !used[i] { fixed[i] = 1; continue }
            x0[i] = SIMD2(Float(dot3(pos[i] - pos[pa], xa)), Float(dot3(pos[i] - pos[pa], ya)))
        }
        x0[pa] = .zero; x0[pb] = SIMD2(Float(dpin), 0); fixed[pa] = 1; fixed[pb] = 1

        // --- CSR of corners per vertex (counting sort) ---
        var start = [UInt32](repeating: 0, count: n + 1)
        for c in island.corners { start[c + 1] += 1 }
        for i in 0..<n { start[i + 1] += start[i] }
        var cursor = start, items = [UInt32](repeating: 0, count: island.corners.count)
        for (e, c) in island.corners.enumerated() { items[Int(cursor[c])] = UInt32(e); cursor[c] += 1 }

        // --- GPU: the solve ---
        let csrS = g.buffer(start), csrI = g.buffer(items), fx = g.buffer(fixed)
        let x = g.buffer(x0), res = g.empty(n * 8), z = g.empty(n * 8), p = g.empty(n * 8), ap = g.empty(n * 8)
        let diag = g.empty(n * 4), ab = g.empty(T * 8)
        let pA = g.empty(Swift.max(gv, 1) * 4), pB = g.empty(Swift.max(gv, 1) * 4), scal = g.empty(32 * 4)
        memset(scal.contents(), 0, 32 * 4)
        let tol = Float(Swift.max(options.tolerance, 1e-6))
        lp = LP(n: UInt32(n), T: UInt32(T), groups: UInt32(gv), first: 1, tol2: tol * tol)
        let maxIt = options.maxIterations ?? Swift.min(20_000, 50 * n + 200)
        var done = 0, lastIters = -1, verifies = 0, needRestart = true
        guard g.run({ enc in call(g, enc, "lscm_diag", n, [csrS, csrI, wr, wi, diag], &lp) }) else { return nil }
        func restart(_ enc: MTLComputeCommandEncoder) {
            // res = -A x (masked), z, p, rz, rr — the true residual, recomputed from x
            call(g, enc, "lscm_tri", T, [x, tl, wr, wi, ab], &lp)
            call(g, enc, "lscm_gather", n, [csrS, csrI, ab, wr, wi, fx, x, ap, pA], &lp)
            call(g, enc, "lscm_res0", n, [ap, fx, diag, res, z, p, pA, pB], &lp)
            groupsCall(g, enc, "lscm_scalar", [pA, pB, scal], &lp, mode: 0)
            lp.first = 0
        }
        while done < maxIt {
            let its = Swift.min(batch, maxIt - done)
            let ok = g.run { enc in
                if needRestart { restart(enc) }
                for _ in 0..<its {
                    call(g, enc, "lscm_tri", T, [p, tl, wr, wi, ab], &lp)
                    call(g, enc, "lscm_gather", n, [csrS, csrI, ab, wr, wi, fx, p, ap, pA], &lp)
                    groupsCall(g, enc, "lscm_scalar", [pA, pB, scal], &lp, mode: 1)
                    call(g, enc, "lscm_update", n, [x, res, z, ap, p, diag, fx, scal, pA, pB], &lp)
                    groupsCall(g, enc, "lscm_scalar", [pA, pB, scal], &lp, mode: 2)
                    call(g, enc, "lscm_pdir", n, [z, p, scal], &lp)
                }
            }
            guard ok else { return nil }
            needRestart = false
            var s = UVGPU.read(scal, Float.self, count: 8)
            done = Int(s[5])
            if !(s[3] > s[4]) || done == lastIters {
                // the recurrence says converged (or stopped moving): trust it only after the true residual agrees
                guard g.run({ restart($0) }) else { return nil }
                s = UVGPU.read(scal, Float.self, count: 8)
                verifies += 1
                if s[3] <= s[4] * 4 || verifies >= 4 { break }
            }
            lastIters = done
            if its < batch { break }
        }
        var uv = UVGPU.read(x, SIMD2<Float>.self, count: n).map { Vec2(Double($0.x), Double($0.y)) }

        // --- layout: same steps as the CPU ---
        var layout = UVIslandLayout(island: island, uv: uv)
        var a = layout.uvArea(mesh)
        if a < 0 { uv = uv.map { Vec2($0.x, -$0.y) }; layout.uv = uv; a = -a }
        if a > 1e-300 { let s = (area3D / a).squareRoot(); layout = layout.mapped { $0 * s } }
        if options.alignToAxes, let r = UVPackGPU.aligned([layout], wide: false, g) { return r[0] }
        if options.alignToAxes { layout = UVGeometry.alignedToMinimumRectangle(layout) }
        return UVUnwrapResult.normalizedToOrigin(layout)
    }

    /// Encode one 1-D kernel with `bufs` at 0…, then the params.
    static func call(_ g: UVGPU, _ enc: MTLComputeCommandEncoder, _ name: String, _ threads: Int,
                     _ bufs: [MTLBuffer], _ lp: inout LP) {
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        enc.setBytes(&lp, length: MemoryLayout<LP>.stride, index: bufs.count)
        g.dispatch(enc, name, threads)
    }
    static func groupsCall(_ g: UVGPU, _ enc: MTLComputeCommandEncoder, _ name: String, _ bufs: [MTLBuffer],
                           _ lp: inout LP, mode: UInt32) {
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        enc.setBytes(&lp, length: MemoryLayout<LP>.stride, index: bufs.count)
        var m = mode
        enc.setBytes(&m, length: 4, index: bufs.count + 1)
        g.dispatchGroups(enc, name, groups: 1)
    }

    static let kernelSource = """
    struct LP { uint n, T, groups, first; float tol2; };
    #define SHAREDSUM threadgroup float sh[32]

    kernel void lscm_setup(device const float4* pos [[buffer(0)]], device const uint* idx [[buffer(1)]],
                           device const uint* tris [[buffer(2)]], device float4* wr [[buffer(3)]], device float4* wi [[buffer(4)]],
                           device float4* part [[buffer(5)]], constant LP& P [[buffer(6)]],
                           uint k [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        SHAREDSUM;
        float area = 0; float3 ch = float3(0);
        if (k < P.T) {
            uint t = tris[k];
            float3 p0 = pos[idx[3*t]].xyz, p1 = pos[idx[3*t+1]].xyz, p2 = pos[idx[3*t+2]].xyz;
            float3 e1 = p1 - p0, cr = cross(e1, p2 - p0);
            float a = length(cr) / 2;
            if (a > 1e-14f) {
                float l1 = length(e1); float3 xa = e1 / l1; float3 ya = normalize(cross(normalize(cr), xa));
                float2 q0 = float2(0), q1 = float2(l1, 0), q2 = float2(dot(p2 - p0, xa), dot(p2 - p0, ya));
                float sq = sqrt(l1 * q2.y);
                float2 w0 = (q2 - q1) / sq, w1 = (q0 - q2) / sq, w2 = (q1 - q0) / sq;
                wr[k] = float4(w0.x, w1.x, w2.x, 1); wi[k] = float4(w0.y, w1.y, w2.y, 0);
                area = a; ch = cr / 2;
            } else { wr[k] = float4(0); wi[k] = float4(0); }
        }
        float4 r = float4(tg_sum(area, sh, tid, tgs), tg_sum(ch.x, sh, tid, tgs), tg_sum(ch.y, sh, tid, tgs), tg_sum(ch.z, sh, tid, tgs));
        if (tid == 0) part[grp] = r;
    }

    kernel void lscm_diag(device const uint* cs [[buffer(0)]], device const uint* ci [[buffer(1)]],
                          device const float4* wr [[buffer(2)]], device const float4* wi [[buffer(3)]],
                          device float* diag [[buffer(4)]], constant LP& P [[buffer(5)]], uint i [[thread_position_in_grid]]) {
        if (i >= P.n) return;
        float d = 0;
        for (uint e = cs[i]; e < cs[i + 1]; e++) { uint k = ci[e] / 3, j = ci[e] % 3; d += wr[k][j] * wr[k][j] + wi[k][j] * wi[k][j]; }
        diag[i] = d;
    }

    kernel void lscm_tri(device const float2* v [[buffer(0)]], device const uint* tl [[buffer(1)]],
                         device const float4* wr [[buffer(2)]], device const float4* wi [[buffer(3)]],
                         device float2* ab [[buffer(4)]], constant LP& P [[buffer(5)]], uint k [[thread_position_in_grid]]) {
        if (k >= P.T) return;
        float a = 0, b = 0;
        for (uint j = 0; j < 3; j++) {
            float2 q = v[tl[3*k + j]];
            a += wr[k][j] * q.x - wi[k][j] * q.y;
            b += wi[k][j] * q.x + wr[k][j] * q.y;
        }
        ab[k] = float2(a, b);
    }

    kernel void lscm_gather(device const uint* cs [[buffer(0)]], device const uint* ci [[buffer(1)]],
                            device const float2* ab [[buffer(2)]], device const float4* wr [[buffer(3)]],
                            device const float4* wi [[buffer(4)]], device const uchar* fixedm [[buffer(5)]],
                            device const float2* p [[buffer(6)]], device float2* ap [[buffer(7)]],
                            device float* part [[buffer(8)]], constant LP& P [[buffer(9)]],
                            uint i [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        SHAREDSUM;
        float dp = 0;
        if (i < P.n) {
            float2 acc = float2(0);
            if (!fixedm[i]) {
                for (uint e = cs[i]; e < cs[i + 1]; e++) {
                    uint k = ci[e] / 3, j = ci[e] % 3;
                    float a = ab[k].x, b = ab[k].y, r = wr[k][j], m = wi[k][j];
                    acc += float2(r * a + m * b, -m * a + r * b);
                }
            }
            ap[i] = acc; dp = dot(p[i], acc);
        }
        float s = tg_sum(dp, sh, tid, tgs);
        if (tid == 0) part[grp] = s;
    }

    kernel void lscm_res0(device const float2* ap [[buffer(0)]], device const uchar* fixedm [[buffer(1)]],
                          device const float* diag [[buffer(2)]], device float2* res [[buffer(3)]], device float2* z [[buffer(4)]],
                          device float2* p [[buffer(5)]], device float* pA [[buffer(6)]], device float* pB [[buffer(7)]],
                          constant LP& P [[buffer(8)]], uint i [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        SHAREDSUM;
        float rz = 0, rr = 0;
        if (i < P.n) {
            bool f = fixedm[i] != 0;
            float2 r = f ? float2(0) : -ap[i];
            float2 zz = (f || diag[i] < 1e-30f) ? float2(0) : r / diag[i];
            res[i] = r; z[i] = zz; p[i] = zz; rz = dot(r, zz); rr = dot(r, r);
        }
        float a = tg_sum(rz, sh, tid, tgs), b = tg_sum(rr, sh, tid, tgs);
        if (tid == 0) { pA[grp] = a; pB[grp] = b; }
    }

    // One threadgroup: mode 0 init (rz, rr, threshold on the first call), 1 alpha, 2 beta.
    // scal: 0 rz, 1 alpha, 2 beta, 3 rr, 4 threshold, 5 iterations, 6 pAp.
    kernel void lscm_scalar(device const float* pA [[buffer(0)]], device const float* pB [[buffer(1)]],
                            device float* s [[buffer(2)]], constant LP& P [[buffer(3)]], constant uint& mode [[buffer(4)]],
                            uint tid [[thread_index_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
        SHAREDSUM;
        float a = 0, b = 0;
        for (uint g = tid; g < P.groups; g += tgs) { a += pA[g]; b += pB[g]; }
        a = tg_sum(a, sh, tid, tgs);
        b = tg_sum(b, sh, tid, tgs);
        if (tid != 0) return;
        if (mode == 0) { s[0] = a; s[3] = b; if (P.first != 0) s[4] = P.tol2 * b; }
        else if (mode == 1) {
            s[6] = a;
            s[1] = (s[3] <= s[4] || !(a > 1e-30f)) ? 0.0f : s[0] / a;
        } else {
            float beta = (s[1] == 0.0f || !(s[0] > 0)) ? 0.0f : a / s[0];
            s[2] = beta; s[0] = a; s[3] = b; if (s[1] != 0.0f) s[5] += 1.0f;
        }
    }

    kernel void lscm_update(device float2* x [[buffer(0)]], device float2* res [[buffer(1)]], device float2* z [[buffer(2)]],
                            device const float2* ap [[buffer(3)]], device const float2* p [[buffer(4)]],
                            device const float* diag [[buffer(5)]], device const uchar* fixedm [[buffer(6)]],
                            device const float* s [[buffer(7)]], device float* pA [[buffer(8)]], device float* pB [[buffer(9)]],
                            constant LP& P [[buffer(10)]], uint i [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        SHAREDSUM;
        float rz = 0, rr = 0;
        if (i < P.n) {
            float alpha = s[1];
            bool f = fixedm[i] != 0;
            float2 xv = x[i] + alpha * p[i];
            float2 r = res[i] - alpha * ap[i];
            float2 zz = (f || diag[i] < 1e-30f) ? float2(0) : r / diag[i];
            x[i] = xv; res[i] = r; z[i] = zz; rz = dot(r, zz); rr = dot(r, r);
        }
        float a = tg_sum(rz, sh, tid, tgs), b = tg_sum(rr, sh, tid, tgs);
        if (tid == 0) { pA[grp] = a; pB[grp] = b; }
    }

    kernel void lscm_pdir(device const float2* z [[buffer(0)]], device float2* p [[buffer(1)]], device const float* s [[buffer(2)]],
                          constant LP& P [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= P.n) return;
        p[i] = z[i] + s[2] * p[i];
    }
    """
}
#endif
