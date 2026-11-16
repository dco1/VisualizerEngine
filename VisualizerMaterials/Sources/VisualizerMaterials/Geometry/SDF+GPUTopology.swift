//  SDF+GPUTopology.swift — surface nets entirely on the GPU (ticket item M4; the CPU `SDF.mesh` stays the oracle).
//
//  `SDF.meshGPU` runs, with no CPU step between the SDF evaluation and the final readback:
//    1. grid          `sdf_grid` (SDFGPU's kernel) → samples
//    2. boundary      samples on the box's faces forced outside (closed surface)
//    3. saddles       the CPU's sequential `disambiguate` becomes rounds of  detect (each ambiguous grid face marks the
//                     sample nearest the surface) → apply (flip marked samples). A tiny kernel reads the previous round's
//                     flip count and writes the next round's INDIRECT dispatch size, so a mesh with no saddles runs
//                     one empty round, not 16.
//    4. classify      cell sign mask → vertex flag; per (sample, axis) grid edge → quad flag (all four cells exist)
//    5. scan          exclusive prefix sums of both flag arrays = vertex index per cell, first-quad slot per edge
//    6. emit          vertices (mean of edge crossings), quads (two triangles on the shorter diagonal, wound outward)
//    7. refine        `sdf_refine` (projection + gradient normal), then optional box-projected UVs.
//  The ONE synchronisation: the two totals (vertex count, quad count) are read back between steps 5 and 6 (8 bytes) so
//  the output buffers can be allocated exactly. The final readback converts Float → Mesh3's Double on the CPU.
//  Guarantees: same mesh as the CPU path wherever the CPU finds no saddle (same cell order, same winding, same
//  diagonals); where saddles exist the parallel rounds flip a (slightly) different set of samples than the CPU's
//  sequential sweep, so vertex counts / volume agree to a fraction of a percent, not bit for bit; the result is closed and
//  edge-manifold either way (tests). `nil` (use the CPU) without a device or when the tree has no GPU program.
//  Not guaranteed: more than 16 saddle rounds leaves residual saddles (never seen).

import Foundation
import simd

#if canImport(Metal)
import Metal

struct SDFNetTimings: Sendable { var gridAndTopologyMS = 0.0, emitAndRefineMS = 0.0, readbackMS = 0.0 }

enum SDFNetGPU {
    struct NetParams {
        var nx, ny, nz, count: UInt32
        var cx, cy, cz, pad: UInt32
        var eps, cell: Float
        var pad2: SIMD2<Float> = .zero
        var lo: SIMD4<Float>
    }
    struct RefineParams { var cell, h: Float; var iterations, count, ninst: UInt32 }

    static let maxSaddleRounds = 16

    static func mesh(_ sdf: SDF, lo: Vec3, cell: Double, cells: SIMD3<Int>, options: SDFMeshOptions,
                     timings: UnsafeMutablePointer<SDFNetTimings>? = nil) -> Mesh3? {
        guard let g = ModellingGPU.shared, let program = sdf.gpuProgram() else { return nil }
        let nx = cells.x + 1, ny = cells.y + 1, nz = cells.z + 1
        let ns = nx * ny * nz, nc = cells.x * cells.y * cells.z
        let eps = Float(cell * 1e-3)
        var P = NetParams(nx: UInt32(nx), ny: UInt32(ny), nz: UInt32(nz), count: UInt32(program.count),
                          cx: UInt32(cells.x), cy: UInt32(cells.y), cz: UInt32(cells.z), pad: 0, eps: eps, cell: Float(cell),
                          lo: SIMD4(Float(lo.x), Float(lo.y), Float(lo.z), Float(cell)))
        let prog = g.buffer(program.insts)
        let values = g.buffer(length: ns * 4), flip = g.buffer(length: ns * 4)
        let cflag = g.buffer(length: (nc + 1) * 4), cidx = g.buffer(length: (nc + 1) * 4)
        let qflag = g.buffer(length: (3 * ns + 1) * 4), qidx = g.buffer(length: (3 * ns + 1) * 4)
        let state = g.buffer([UInt32(1), 0, 0, 0])
        let args = g.buffer(length: 16)
        let groups = SIMD3<UInt32>(UInt32((nx + 7) / 8), UInt32((ny + 7) / 8), UInt32((nz + 3) / 4))
        let t0 = Date()

        let okA = g.run { enc in
            g.dispatch3D(enc, "sdf_grid", nx, ny, nz, [.buf(values), .buf(prog), .value(GridParams(origin: P.lo, nx: P.nx, ny: P.ny, nz: P.nz, count: P.count))])
            g.dispatch3D(enc, "net_force_boundary", nx, ny, nz, [.buf(values), .value(P)])
            g.fill(enc, flip, count: ns, value: 0)
            for _ in 0..<maxSaddleRounds {
                g.dispatch1D(enc, "net_round_args", 1, [.buf(state), .buf(args), .value(groups)])
                g.dispatchIndirect(enc, "net_saddle_detect", [.buf(values), .buf(flip), .buf(state), .value(P)], indirect: args)
                g.dispatchIndirect(enc, "net_saddle_apply", [.buf(values), .buf(flip), .value(P)], indirect: args)
            }
            g.fill(enc, cflag, count: nc + 1, value: 0)
            g.fill(enc, qflag, count: 3 * ns + 1, value: 0)
            g.dispatch3D(enc, "net_classify", cells.x, cells.y, cells.z, [.buf(values), .buf(cflag), .value(P)])
            g.dispatch3D(enc, "net_quad_flags", nx, ny, nz, [.buf(values), .buf(cflag), .buf(qflag), .value(P)])
            g.scan(enc, input: cflag, output: cidx, n: nc + 1)
            g.scan(enc, input: qflag, output: qidx, n: 3 * ns + 1)
        }
        guard okA else { return nil }
        let nv = Int(cidx.contents().assumingMemoryBound(to: UInt32.self)[nc])
        let nq = Int(qidx.contents().assumingMemoryBound(to: UInt32.self)[3 * ns])
        let t1 = Date()
        if nv == 0 || nq == 0 { return Mesh3() }

        let pos = g.buffer(length: nv * 16), cmin = g.buffer(length: nv * 16), nrm = g.buffer(length: nv * 16)
        let idx = g.buffer(length: nq * 6 * 4), uv = g.buffer(length: nv * 8)
        var R = RefineParams(cell: Float(cell), h: Float(cell * 0.25), iterations: UInt32(options.projectIterations),
                             count: UInt32(nv), ninst: UInt32(program.count))
        var uvScale: Float = 0
        if case .boxProjected(let s) = options.uvs { uvScale = Float(s) }
        let okB = g.run { enc in
            g.dispatch3D(enc, "net_vertices", cells.x, cells.y, cells.z, [.buf(values), .buf(cflag), .buf(cidx), .buf(pos), .buf(cmin), .value(P)])
            g.dispatch1D(enc, "net_emit", 3 * ns, [.buf(values), .buf(cidx), .buf(qflag), .buf(qidx), .buf(pos), .buf(idx), .value(P)])
            g.dispatch1D(enc, "sdf_refine", nv, [.buf(pos), .buf(nrm), .buf(cmin), .buf(prog), .value(R)])
            if uvScale > 0 { g.dispatch1D(enc, "net_uv", nv, [.buf(pos), .buf(nrm), .buf(uv), .value(uvScale), .value(UInt32(nv))]) }
        }
        guard okB else { return nil }
        let t2 = Date()

        var m = Mesh3()
        let pp = pos.contents().assumingMemoryBound(to: SIMD4<Float>.self), nn = nrm.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        m.positions = (0..<nv).map { Vec3(Double(pp[$0].x), Double(pp[$0].y), Double(pp[$0].z)) }
        m.normals = (0..<nv).map { Vec3(Double(nn[$0].x), Double(nn[$0].y), Double(nn[$0].z)) }
        if uvScale > 0 {
            let u = uv.contents().assumingMemoryBound(to: SIMD2<Float>.self)
            m.uvs = (0..<nv).map { Vec2(Double(u[$0].x), Double(u[$0].y)) }
        } else { m.uvs = [Vec2](repeating: .zero, count: nv) }
        m.indices = Array(UnsafeBufferPointer(start: idx.contents().assumingMemoryBound(to: UInt32.self), count: nq * 6))
        if let timings {
            let t3 = Date()
            timings.pointee = SDFNetTimings(gridAndTopologyMS: t1.timeIntervalSince(t0) * 1000, emitAndRefineMS: t2.timeIntervalSince(t1) * 1000,
                                            readbackMS: t3.timeIntervalSince(t2) * 1000)
        }
        return m
    }

    private struct GridParams { var origin: SIMD4<Float>; var nx, ny, nz, count: UInt32 }

    static let source = """
    struct NetParams { uint nx, ny, nz, count; uint cx, cy, cz, pad; float eps, cell; float2 pad2; float4 lo; };
    constant uint2 NET_EDGES[12] = { uint2(0,1), uint2(2,3), uint2(4,5), uint2(6,7), uint2(0,2), uint2(1,3),
                                     uint2(4,6), uint2(5,7), uint2(0,4), uint2(1,5), uint2(2,6), uint2(3,7) };

    static inline uint net_s(uint x, uint y, uint z, constant NetParams& p) { return (z * p.ny + y) * p.nx + x; }
    static inline uint net_c(uint x, uint y, uint z, constant NetParams& p) { return (z * p.cy + y) * p.cx + x; }

    kernel void net_force_boundary(device float* v [[buffer(0)]], constant NetParams& p [[buffer(1)]], uint3 id [[thread_position_in_grid]]) {
        if (id.x >= p.nx || id.y >= p.ny || id.z >= p.nz) return;
        bool edge = id.x == 0 || id.y == 0 || id.z == 0 || id.x == p.nx - 1 || id.y == p.ny - 1 || id.z == p.nz - 1;
        uint s = net_s(id.x, id.y, id.z, p);
        if (edge && v[s] <= 0) v[s] = p.eps;
    }

    // One thread: the previous round's flip count decides whether this round runs at all (indirect dispatch size).
    kernel void net_round_args(device atomic_uint* state [[buffer(0)]], device uint* args [[buffer(1)]], constant uint3& groups [[buffer(2)]]) {
        uint prev = atomic_exchange_explicit(&state[0], 0u, memory_order_relaxed);
        bool go = prev != 0;
        args[0] = go ? groups.x : 0; args[1] = go ? groups.y : 0; args[2] = go ? groups.z : 0;
    }

    static inline bool net_interior(uint3 s, constant NetParams& p) {
        return s.x > 0 && s.y > 0 && s.z > 0 && s.x < p.nx - 1 && s.y < p.ny - 1 && s.z < p.nz - 1;
    }

    // Every grid face whose four samples alternate in sign marks the interior sample nearest the surface.
    kernel void net_saddle_detect(device const float* v [[buffer(0)]], device uint* flip [[buffer(1)]], device atomic_uint* state [[buffer(2)]],
                                  constant NetParams& p [[buffer(3)]], uint3 id [[thread_position_in_grid]]) {
        if (id.x >= p.nx || id.y >= p.ny || id.z >= p.nz) return;
        uint c[3] = { p.cx, p.cy, p.cz };
        uint s[3] = { id.x, id.y, id.z };
        for (uint a = 0; a < 3; a++) {
            uint u = (a + 1) % 3, w = (a + 2) % 3;
            if (s[u] >= c[u] || s[w] >= c[w]) continue;
            uint3 s0 = id, s1 = id, s2, s3 = id;
            if (u == 0) s1.x++; else if (u == 1) s1.y++; else s1.z++;
            s2 = s1;
            if (w == 0) { s2.x++; s3.x++; } else if (w == 1) { s2.y++; s3.y++; } else { s2.z++; s3.z++; }
            uint i0 = net_s(s0.x, s0.y, s0.z, p), i1 = net_s(s1.x, s1.y, s1.z, p), i2 = net_s(s2.x, s2.y, s2.z, p), i3 = net_s(s3.x, s3.y, s3.z, p);
            bool n0 = v[i0] < 0;
            if (!(n0 == (v[i2] < 0) && n0 != (v[i1] < 0) && (v[i1] < 0) == (v[i3] < 0))) continue;
            int best = -1; float bestAbs = INFINITY;
            if (net_interior(s0, p) && fabs(v[i0]) < bestAbs) { best = int(i0); bestAbs = fabs(v[i0]); }
            if (net_interior(s1, p) && fabs(v[i1]) < bestAbs) { best = int(i1); bestAbs = fabs(v[i1]); }
            if (net_interior(s2, p) && fabs(v[i2]) < bestAbs) { best = int(i2); bestAbs = fabs(v[i2]); }
            if (net_interior(s3, p) && fabs(v[i3]) < bestAbs) { best = int(i3); }
            if (best < 0) continue;
            flip[best] = 1;
            atomic_fetch_add_explicit(&state[0], 1u, memory_order_relaxed);
        }
    }

    kernel void net_saddle_apply(device float* v [[buffer(0)]], device uint* flip [[buffer(1)]], constant NetParams& p [[buffer(2)]],
                                 uint3 id [[thread_position_in_grid]]) {
        if (id.x >= p.nx || id.y >= p.ny || id.z >= p.nz) return;
        uint s = net_s(id.x, id.y, id.z, p);
        if (flip[s] == 0) return;
        flip[s] = 0;
        float x = v[s];
        v[s] = x < 0 ? max(-x, p.eps) : -max(x, p.eps);
    }

    kernel void net_classify(device const float* v [[buffer(0)]], device uint* cflag [[buffer(1)]], constant NetParams& p [[buffer(2)]],
                             uint3 id [[thread_position_in_grid]]) {
        if (id.x >= p.cx || id.y >= p.cy || id.z >= p.cz) return;
        uint neg = 0;
        for (uint b = 0; b < 8; b++) neg += v[net_s(id.x + (b & 1), id.y + ((b >> 1) & 1), id.z + ((b >> 2) & 1), p)] < 0 ? 1 : 0;
        cflag[net_c(id.x, id.y, id.z, p)] = (neg != 0 && neg != 8) ? 1 : 0;
    }

    // Quad flag per (axis, sample): the grid edge from the sample along the axis changes sign and its four cells exist.
    kernel void net_quad_flags(device const float* v [[buffer(0)]], device const uint* cflag [[buffer(1)]], device uint* qflag [[buffer(2)]],
                               constant NetParams& p [[buffer(3)]], uint3 id [[thread_position_in_grid]]) {
        if (id.x >= p.nx || id.y >= p.ny || id.z >= p.nz) return;
        uint n[3] = { p.nx, p.ny, p.nz }, c[3] = { p.cx, p.cy, p.cz }, s[3] = { id.x, id.y, id.z };
        uint ns = p.nx * p.ny * p.nz, si = net_s(id.x, id.y, id.z, p);
        for (uint a = 0; a < 3; a++) {
            uint u = (a + 1) % 3, w = (a + 2) % 3;
            if (s[a] >= c[a] || s[u] < 1 || s[w] < 1 || s[u] > c[u] - 1 || s[w] > c[w] - 1) continue;
            uint e[3] = { s[0], s[1], s[2] }; e[a] += 1;
            if ((v[si] < 0) == (v[net_s(e[0], e[1], e[2], p)] < 0)) continue;
            bool ok = true;
            for (uint k = 0; k < 4; k++) {
                uint q[3] = { s[0], s[1], s[2] };
                q[u] -= (k == 0 || k == 3) ? 1 : 0; q[w] -= (k == 0 || k == 1) ? 1 : 0;
                if (cflag[net_c(q[0], q[1], q[2], p)] == 0) ok = false;
            }
            if (ok) qflag[a * ns + si] = 1;
        }
    }

    kernel void net_vertices(device const float* v [[buffer(0)]], device const uint* cflag [[buffer(1)]], device const uint* cidx [[buffer(2)]],
                             device float4* pos [[buffer(3)]], device float4* cmin [[buffer(4)]], constant NetParams& p [[buffer(5)]],
                             uint3 id [[thread_position_in_grid]]) {
        if (id.x >= p.cx || id.y >= p.cy || id.z >= p.cz) return;
        uint ci = net_c(id.x, id.y, id.z, p);
        if (cflag[ci] == 0) return;
        float cv[8];
        for (uint b = 0; b < 8; b++) cv[b] = v[net_s(id.x + (b & 1), id.y + ((b >> 1) & 1), id.z + ((b >> 2) & 1), p)];
        float3 acc = 0; float cnt = 0;
        for (uint e = 0; e < 12; e++) {
            uint a = NET_EDGES[e].x, b = NET_EDGES[e].y;
            if ((cv[a] < 0) == (cv[b] < 0)) continue;
            float t = cv[a] / (cv[a] - cv[b]);
            float3 pa = float3(a & 1, (a >> 1) & 1, (a >> 2) & 1), pb = float3(b & 1, (b >> 1) & 1, (b >> 2) & 1);
            acc += pa + (pb - pa) * t; cnt += 1;
        }
        float3 lo = p.lo.xyz + float3(id) * p.cell;
        uint o = cidx[ci];
        pos[o] = float4(lo + acc / cnt * p.cell, 0);
        cmin[o] = float4(lo, 0);
    }

    // One thread per (axis, sample) slot; 6 indices at its scan position.
    kernel void net_emit(device const float* v [[buffer(0)]], device const uint* cidx [[buffer(1)]], device const uint* qflag [[buffer(2)]],
                         device const uint* qidx [[buffer(3)]], device const float4* pos [[buffer(4)]], device uint* out [[buffer(5)]],
                         constant NetParams& p [[buffer(6)]], uint q [[thread_position_in_grid]]) {
        uint ns = p.nx * p.ny * p.nz;
        if (q >= 3 * ns || qflag[q] == 0) return;
        uint a = q / ns, si = q % ns;
        uint s[3] = { si % p.nx, (si / p.nx) % p.ny, si / (p.nx * p.ny) };
        uint u = (a + 1) % 3, w = (a + 2) % 3;
        uint e[3] = { s[0], s[1], s[2] }; e[a] += 1;
        float v0 = v[si], v1 = v[net_s(e[0], e[1], e[2], p)];
        uint o[4];
        for (uint k = 0; k < 4; k++) {
            uint c[3] = { s[0], s[1], s[2] };
            c[u] -= (k == 0 || k == 3) ? 1 : 0; c[w] -= (k == 0 || k == 1) ? 1 : 0;
            o[k] = cidx[net_c(c[0], c[1], c[2], p)];
        }
        if (v0 >= 0) { uint t = o[0]; o[0] = o[3]; o[3] = t; t = o[1]; o[1] = o[2]; o[2] = t; }
        float3 p0 = pos[o[0]].xyz, p1 = pos[o[1]].xyz, p2 = pos[o[2]].xyz, p3 = pos[o[3]].xyz;
        uint b = qidx[q] * 6;
        if (length(p0 - p2) <= length(p1 - p3)) {
            out[b] = o[0]; out[b+1] = o[1]; out[b+2] = o[2]; out[b+3] = o[0]; out[b+4] = o[2]; out[b+5] = o[3];
        } else {
            out[b] = o[1]; out[b+1] = o[2]; out[b+2] = o[3]; out[b+3] = o[1]; out[b+4] = o[3]; out[b+5] = o[0];
        }
    }

    kernel void net_uv(device const float4* pos [[buffer(0)]], device const float4* nrm [[buffer(1)]], device float2* uv [[buffer(2)]],
                       constant float& scale [[buffer(3)]], constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float3 p = pos[i].xyz, a = fabs(nrm[i].xyz);
        float2 r = (a.x >= a.y && a.x >= a.z) ? float2(p.z, p.y) : (a.y >= a.z ? float2(p.x, p.z) : float2(p.x, p.y));
        uv[i] = r / scale;
    }
    """
}

public extension SDF {
    /// The whole surface-nets pipeline on the GPU (see `SDF+GPUTopology.swift`), or nil when there is no Metal device
    /// or the tree has no GPU program (`.mesh`, `.custom`, twist, bend). `SDF.mesh(options:)` is the CPU-capable entry.
    func meshGPU(lo: Vec3, hi: Vec3, cell: Double, options: SDFMeshOptions = .init()) -> Mesh3? {
        let cells = SIMD3<Int>(Int(((hi.x - lo.x) / cell).rounded(.up)), Int(((hi.y - lo.y) / cell).rounded(.up)),
                               Int(((hi.z - lo.z) / cell).rounded(.up)))
        precondition(cells.x >= 2 && cells.y >= 2 && cells.z >= 2, "SDF.meshGPU: box smaller than two cells")
        return SDFNetGPU.mesh(self, lo: lo, cell: cell, cells: cells, options: options)
    }
}
#else
public extension SDF {
    func meshGPU(lo: Vec3, hi: Vec3, cell: Double, options: SDFMeshOptions = .init()) -> Mesh3? { nil }
}
#endif
