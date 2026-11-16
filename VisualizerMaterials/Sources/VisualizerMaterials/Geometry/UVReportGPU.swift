import Foundation
import simd

//  UVReportGPU.swift — the UV distortion / texel-density report as Metal kernels (ticket item M3).
//
//  Guarantees: `UVReport(_:backend:)` fills the same fields as `UVReport(_:)`: per triangle the 2×2 Jacobian, σ1 ≥ σ2
//  (σ2 = |det|/σ1, no cancellation), anisotropy, density √|det|, the flipped test, all in one kernel, and the sums,
//  maxima, minima and UV bounds reduced on the GPU by threadgroup (one 256-triangle chunk each); a second kernel
//  does the area-weighted variance about the global mean (two-pass, as the CPU does). Matches the CPU to ~1e-4
//  relative (Float). Doesn't: find islands — `UVTopology` (CPU hashing) still does; and the last merge of one partial
//  per 256 triangles into island totals is a CPU sum (3 000 numbers for a 800k-triangle mesh).

public extension UVReport {
    init(_ mesh: Mesh3, tolerance: Double = 1e-6, backend: UVBackend) {
        #if canImport(Metal)
        if backend.useGPU(work: mesh.triangleCount), let g = UVGPU.shared {
            let topo = UVTopology(mesh)
            let isl = topo.islands(topo.seams(uvBoundaryOf: mesh, tolerance: tolerance))
            if let r = UVReportGPU.build(mesh, isl, g) { self = r; return }
        }
        #endif
        self.init(mesh, tolerance: tolerance)
    }
}

enum UVReportGPU {
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

extension UVReportGPU {
    /// The report for known islands (so a benchmark can time the kernels apart from the topology).
    static func build(_ mesh: Mesh3, _ islands: [UVIsland], _ g: UVGPU) -> UVReport? {
        let chunk = 256
        var order: [UInt32] = [], chunks: [SIMD2<UInt32>] = [], chunkIsland: [Int] = []
        for (ii, isl) in islands.enumerated() {
            let base = order.count
            order += isl.triangles.map { UInt32($0) }
            var s = 0
            while s < isl.triangles.count {
                chunks.append(SIMD2(UInt32(base + s), UInt32(Swift.min(chunk, isl.triangles.count - s)))); chunkIsland.append(ii); s += chunk
            }
        }
        let hasUV: UInt32 = mesh.uvs.count == mesh.positions.count ? 1 : 0
        let uvf = hasUV == 1 ? mesh.uvs.map { SIMD2<Float>(Float($0.x), Float($0.y)) } : [SIMD2<Float>(0, 0)]
        let pos = g.buffer(UVGPU.float4s(mesh.positions)), uv = g.buffer(uvf), idx = g.buffer(mesh.indices)
        let ord = g.buffer(order), ch = g.buffer(chunks)
        let out = g.empty(chunks.count * 16 * 4), out2 = g.empty(chunks.count * 4)
        var hu = hasUV
        guard !chunks.isEmpty, g.run({ enc in
            for (i, b) in [pos, uv, idx, ord, ch, out].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
            enc.setBytes(&hu, length: 4, index: 6)
            g.dispatchGroups(enc, "rep_chunk", groups: chunks.count)
        }) else { return nil }
        let raw = UVGPU.read(out, Float.self, count: chunks.count * 16)
        // merge chunks per island
        var tot = [[Double]](repeating: [Double](repeating: 0, count: 16), count: islands.count)
        var lo = [Double](repeating: .infinity, count: islands.count), hi = [Double](repeating: 0, count: islands.count)
        var ma = [Double](repeating: 1, count: islands.count)
        var blo = [Vec2](repeating: Vec2(repeating: .infinity), count: islands.count)
        var bhi = [Vec2](repeating: Vec2(repeating: -.infinity), count: islands.count)
        for (c, ii) in chunkIsland.enumerated() {
            let r = (0..<16).map { Double(raw[c * 16 + $0]) }
            for k in [0, 1, 2, 3, 5, 8] { tot[ii][k] += r[k] }
            ma[ii] = Swift.max(ma[ii], r[4]); lo[ii] = Swift.min(lo[ii], r[6]); hi[ii] = Swift.max(hi[ii], r[7])
            blo[ii] = simd.min(blo[ii], Vec2(r[9], r[10])); bhi[ii] = simd.max(bhi[ii], Vec2(r[11], r[12]))
        }
        let totW = tot.reduce(0) { $0 + $1[2] }, totSD = tot.reduce(0) { $0 + $1[5] }
        let mean = totW > 0 ? totSD / totW : 0
        var mp = Float(mean)
        guard g.run({ enc in
            for (i, b) in [pos, uv, idx, ord, ch, out2].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
            enc.setBytes(&hu, length: 4, index: 6); enc.setBytes(&mp, length: 4, index: 7)
            g.dispatchGroups(enc, "rep_var", groups: chunks.count)
        }) else { return nil }
        let varSum = UVGPU.read(out2, Float.self, count: chunks.count).reduce(0.0) { $0 + Double($1) }

        var reports: [UVIslandReport] = []
        var totA = 0.0, maxA = 1.0, flips = 0, uvTotal = 0.0
        for i in 0..<islands.count {
            let t = tot[i], ok = t[2] > 0
            reports.append(UVIslandReport(
                triangleCount: islands[i].triangles.count, area3D: t[0], areaUV: t[1], bounds: (blo[i], bhi[i]),
                meanAnisotropy: ok ? t[3] / t[2] : 1, maxAnisotropy: ma[i], meanDensity: ok ? t[5] / t[2] : 0,
                minDensity: ok ? lo[i] : 0, maxDensity: hi[i], flipped: Int(t[8].rounded())))
            totA += t[3]; maxA = Swift.max(maxA, ma[i]); flips += Int(t[8].rounded()); uvTotal += t[1]
        }
        return UVReport(islands: reports, meanAnisotropy: totW > 0 ? totA / totW : 1, maxAnisotropy: maxA, meanDensity: mean,
                        densityVariation: mean > 0 && totW > 0 ? (varSum / totW).squareRoot() / mean : 0,
                        flipped: flips, coverage: uvTotal)
    }

    static let kernelSource = """
    struct TS { float area, uvA, d, aniso; bool valid, flipped; };

    static TS tri_stats(uint t, device const float4* pos, device const float2* uv, device const uint* idx, bool hasUV,
                        thread float2& lo, thread float2& hi) {
        uint i0 = idx[3*t], i1 = idx[3*t+1], i2 = idx[3*t+2];
        float3 p0 = pos[i0].xyz, p1 = pos[i1].xyz, p2 = pos[i2].xyz;
        float2 u0 = hasUV ? uv[i0] : float2(0), u1 = hasUV ? uv[i1] : float2(0), u2 = hasUV ? uv[i2] : float2(0);
        lo = min(u0, min(u1, u2)); hi = max(u0, max(u1, u2));
        float3 e1 = p1 - p0, e2 = p2 - p0, cr = cross(e1, e2);
        TS r; r.area = length(cr) / 2;
        r.uvA = ((u1.x - u0.x) * (u2.y - u0.y) - (u1.y - u0.y) * (u2.x - u0.x)) / 2;
        r.valid = false; r.flipped = false; r.d = 0; r.aniso = 1;
        if (!(r.area > 1e-14f)) return r;
        if (r.uvA <= 1e-14f * max(r.area, 1e-12f)) { r.flipped = true; return r; }
        float l1 = length(e1); float3 xa = e1 / l1; float3 ya = normalize(cross(normalize(cr), xa));
        float2 q1 = float2(l1, 0), q2 = float2(dot(e2, xa), dot(e2, ya));
        float det = q1.x * q2.y - q2.x * q1.y;
        float2 d1 = u1 - u0, d2 = u2 - u0;
        float2 jx = float2((d1.x * q2.y - d2.x * q1.y) / det, (-d1.x * q2.x + d2.x * q1.x) / det);
        float2 jy = float2((d1.y * q2.y - d2.y * q1.y) / det, (-d1.y * q2.x + d2.y * q1.x) / det);
        float f = dot(jx, jx) + dot(jy, jy), dj = fabs(jx.x * jy.y - jx.y * jy.x);
        float s1 = (sqrt(f + 2 * dj) + sqrt(max(f - 2 * dj, 0.0f))) / 2;
        float s2 = s1 > 0 ? dj / s1 : 0;
        r.aniso = s2 > 1e-30f ? s1 / s2 : INFINITY;
        r.d = sqrt(dj); r.valid = true;
        return r;
    }

    // One 256-triangle chunk per threadgroup. out[16]: 0 area3D, 1 |uvArea|, 2 w, 3 Σaniso·w, 4 max aniso, 5 Σd·w,
    // 6 min d, 7 max d, 8 flipped, 9–12 uv bounds.
    kernel void rep_chunk(device const float4* pos [[buffer(0)]], device const float2* uv [[buffer(1)]],
                          device const uint* idx [[buffer(2)]], device const uint* order [[buffer(3)]],
                          device const uint2* chunks [[buffer(4)]], device float* out [[buffer(5)]], constant uint& hasUV [[buffer(6)]],
                          uint tid [[thread_index_in_threadgroup]], uint grp [[threadgroup_position_in_grid]],
                          uint tgs [[threads_per_threadgroup]]) {
        threadgroup float sh[32];
        uint2 c = chunks[grp];
        float area = 0, auv = 0, w = 0, sa = 0, ma = 1, sd = 0, lo = INFINITY, hi = 0, fl = 0;
        float2 blo = float2(INFINITY), bhi = float2(-INFINITY);
        if (tid < c.y) {
            TS r = tri_stats(order[c.x + tid], pos, uv, idx, hasUV != 0, blo, bhi);
            area = r.area;
            auv = fabs(r.uvA);
            if (r.flipped) fl = 1;
            if (r.valid) { w = r.area; sa = r.aniso * r.area; ma = max(1.0f, r.aniso); sd = r.d * r.area; lo = r.d; hi = r.d; }
        }
        float v[9] = { area, auv, w, sa, 0, sd, 0, 0, fl };
        float s[9];
        for (int k = 0; k < 9; k++) s[k] = (k == 4 || k == 6 || k == 7) ? 0 : tg_sum(v[k], sh, tid, tgs);
        float mx = tg_max(ma, sh, tid, tgs), mn = tg_min(lo, sh, tid, tgs), mh = tg_max(hi, sh, tid, tgs);
        float b0 = tg_min(blo.x, sh, tid, tgs), b1 = tg_min(blo.y, sh, tid, tgs);
        float b2 = tg_max(bhi.x, sh, tid, tgs), b3 = tg_max(bhi.y, sh, tid, tgs);
        if (tid == 0) {
            out[grp*16+0] = s[0]; out[grp*16+1] = s[1]; out[grp*16+2] = s[2]; out[grp*16+3] = s[3]; out[grp*16+4] = mx;
            out[grp*16+5] = s[5]; out[grp*16+6] = mn; out[grp*16+7] = mh; out[grp*16+8] = s[8];
            out[grp*16+9] = b0; out[grp*16+10] = b1; out[grp*16+11] = b2; out[grp*16+12] = b3;
        }
    }

    kernel void rep_var(device const float4* pos [[buffer(0)]], device const float2* uv [[buffer(1)]],
                        device const uint* idx [[buffer(2)]], device const uint* order [[buffer(3)]],
                        device const uint2* chunks [[buffer(4)]], device float* out [[buffer(5)]], constant uint& hasUV [[buffer(6)]],
                        constant float& mean [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
                        uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        threadgroup float sh[32];
        uint2 c = chunks[grp];
        float v = 0;
        if (tid < c.y) {
            float2 a, b;
            TS r = tri_stats(order[c.x + tid], pos, uv, idx, hasUV != 0, a, b);
            if (r.valid) v = (r.d - mean) * (r.d - mean) * r.area;
        }
        float s = tg_sum(v, sh, tid, tgs);
        if (tid == 0) out[grp] = s;
    }
    """
}

extension UVReport {
    init(islands: [UVIslandReport], meanAnisotropy: Double, maxAnisotropy: Double, meanDensity: Double,
         densityVariation: Double, flipped: Int, coverage: Double) {
        self.islands = islands; self.meanAnisotropy = meanAnisotropy; self.maxAnisotropy = maxAnisotropy
        self.meanDensity = meanDensity; self.densityVariation = densityVariation; self.flipped = flipped
        self.coverage = coverage
    }
}
#endif
