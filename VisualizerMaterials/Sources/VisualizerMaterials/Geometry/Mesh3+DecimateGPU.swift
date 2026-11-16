//  Mesh3+DecimateGPU.swift — quadric-error edge-collapse decimation on the GPU (ticket item M1).
//  `Mesh3.decimate(targetTriangles:creaseDegrees:backend:)`.
//
//  Guarantees: the same rules and the same answer shape as the CPU `decimate` (`Mesh3+Decimate.swift`): reaches
//  `targetTriangles` (or stops when no legal collapse is left), open boundaries / creases / UV seams are features a
//  vertex never leaves, corners stay, link condition and no folded triangle, winding kept, UVs per corner, normals
//  rebuilt smooth within `creaseDegrees`. Vertices that never moved keep their exact input doubles.
//  What runs where: weld, triangle prep, per-face plane quadrics and per-vertex accumulation (with boundary / crease
//  constraint planes), every edge's legality + optimal position + quadric cost in parallel, the independent-set
//  selection, the merge of quadrics, the relabel / UV fix-up / face compaction: all kernels (`Mesh3+CollapseKernels*`).
//  The CPU does: normalising the input into a unit box (Float has no room for metres), the budget cut in the last
//  round (a sort of the few selected keys), and the output assembly (`TriTopology.toMesh3`: vertices split per smooth
//  group and UV: dictionary work on the output, which is the small mesh).
//  Differs from the CPU: costs are Float quadrics (noise ~1e-7 of the mesh's extent squared times the planes' area, so
//  near-flat regions order arbitrarily), creases are re-detected each round, and collapses are taken in parallel
//  rounds of locally cheapest edges rather than one global heap, so the result is a different, equally valid mesh
//  (a slightly larger error at equal count is normal). `Report.maxError` is the CPU's measure: sqrt(cost) rescaled to
//  the input's units.

import simd

#if canImport(Metal)
import Metal

extension TriTopology {
    /// A topology over already-welded arrays (GPU output). Creases are not remembered: `toMesh3` splits by angle.
    init(welded pos: [Vec3], tri: [SIMD3<Int>], cuv: [Vec2]) {
        self.pos = pos
        self.tri = tri
        self.cuv = cuv
        alive = [Bool](repeating: true, count: tri.count)
        aliveCount = tri.count
        var lists = [[Int]](repeating: [], count: pos.count)
        for (f, t) in tri.enumerated() { lists[t.x].append(f); lists[t.y].append(f); lists[t.z].append(f) }
        vf = lists
    }
}

public extension Mesh3 {

    /// `decimate` on the chosen backend (`.gpu` falls back to the CPU without a Metal device).
    func decimate(targetTriangles: Int, creaseDegrees: Double = 40, backend: MeshBackend) -> (mesh: Mesh3, report: DecimationReport) {
        if backend.useGPU(triangles: triangleCount), let r = decimateGPU(targetTriangles: targetTriangles, creaseDegrees: creaseDegrees) { return r }
        return decimate(targetTriangles: targetTriangles, creaseDegrees: creaseDegrees)
    }
    func decimated(targetTriangles: Int, creaseDegrees: Double = 40, backend: MeshBackend) -> Mesh3 {
        decimate(targetTriangles: targetTriangles, creaseDegrees: creaseDegrees, backend: backend).mesh
    }
    func decimated(ratio: Double, creaseDegrees: Double = 40, backend: MeshBackend) -> Mesh3 {
        decimated(targetTriangles: Int((Double(triangleCount) * Swift.max(0, Swift.min(1, ratio))).rounded()),
                  creaseDegrees: creaseDegrees, backend: backend)
    }

    internal func decimateGPU(targetTriangles: Int, creaseDegrees: Double) -> (mesh: Mesh3, report: DecimationReport)? {
        guard let g = ModelGPU.shared, let b = bounds, !indices.isEmpty else { return nil }
        let n = positions.count, T = indices.count / 3
        let center = (b.min + b.max) * 0.5, diag = Swift.max(len3(b.max - b.min), 1e-9)
        let target = Swift.max(4, targetTriangles)
        let hasUV = uvs.count == n

        // 1. weld by position (the CPU's TriTopology does the same, at 1e-7 of the diagonal) and build the face arrays
        let posN = g.buffer(positions.map { p in let q = (p - center) / diag; return SIMD4<Float>(Float(q.x), Float(q.y), Float(q.z), 0) })
        let uvB: MTLBuffer? = hasUV ? g.buffer(uvs.map { SIMD2<Float>(Float($0.x), Float($0.y)) }) : nil
        guard let core = WeldCore.make(g, pos: posN, uv: nil, nrm: nil, n: n, distance: 1e-7, uvTolerance: 0, normalAngleDegrees: 180) else { return nil }
        let V = core.nreps
        let idxB = g.buffer(indices), flag = g.buffer(count: T + 1, UInt32.self)
        let wpos = g.buffer(count: V, SIMD4<Float>.self), triB = g.buffer(count: 3 * T, UInt32.self)
        let cuvB: MTLBuffer? = hasUV ? g.buffer(count: 3 * T, SIMD2<Float>.self) : nil
        guard g.run({ s in
            s.go("dec_prep_pos", V, [posN, core.repList, wpos], GP(a: UInt32(V)))
            s.go("dec_prep_flag", T + 1, [idxB, core.remap, wpos, flag], GP(a: UInt32(T)))
            s.scan(flag, T + 1)
            s.go("dec_prep_scatter", T, [idxB, core.remap, flag, uvB ?? posN, triB, cuvB ?? posN], GP(a: UInt32(T), c: hasUV ? 1 : 0))
        }) else { return nil }
        let F0 = Int(flag.ptr(UInt32.self)[T])
        guard F0 > 0 else { return nil }

        // 2. quadrics once, then rounds of parallel collapses
        let m = GPUTriMesh(g, V: V, F: F0, capV: V, capF: F0, pos: wpos, tri: triB, cuv: cuvB)
        let cp = CollapseParams(hasUV: hasUV, remesh: false, cosCrease: Float(cos(creaseDegrees * .pi / 180)),
                                cosTurn: Float(cos(creaseDegrees * .pi / 180)))
        guard m.buildQuadrics(cp) else { return nil }
        var report = DecimationReport()
        var maxCost: Float = 0
        while m.F > target {
            let budget = Swift.max(1, (m.F - target + 1) / 2)
            guard let r = m.collapseRound(cp, maxCollapses: budget) else { return nil }
            if r.count == 0 { break }
            report.collapses += r.count
            maxCost = Swift.max(maxCost, r.maxCost)
        }

        // 3. read back and assemble
        let F = m.F
        let pp = m.pos.ptr(SIMD4<Float>.self), mv = m.moved.ptr(UInt32.self), rl = core.repList.ptr(UInt32.self)
        let tp = m.tri.ptr(UInt32.self), cu = m.cuv.ptr(SIMD2<Float>.self)
        let outPos: [Vec3] = (0 ..< V).map { v in
            mv[v] == 0 ? positions[Int(rl[v])]
                       : center + Vec3(Double(pp[v].x), Double(pp[v].y), Double(pp[v].z)) * diag
        }
        let tris = (0 ..< F).map { SIMD3<Int>(Int(tp[3 * $0]), Int(tp[3 * $0 + 1]), Int(tp[3 * $0 + 2])) }
        let cuvs: [Vec2] = hasUV ? (0 ..< 3 * F).map { Vec2(Double(cu[$0].x), Double(cu[$0].y)) } : [Vec2](repeating: .zero, count: 3 * F)
        let t = TriTopology(welded: outPos, tri: tris, cuv: cuvs)
        report.triangles = F
        report.maxError = Double(maxCost).squareRoot() * diag * diag
        return (t.toMesh3(creaseDegrees: creaseDegrees), report)
    }
}

extension ModelGPU {
    static let decimateSource = """
    kernel void dec_prep_pos(device const float4* posN [[buffer(0)]], device const uint* rl [[buffer(1)]], device float4* out [[buffer(2)]],
                             constant GP& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
        if (i < p.a) out[i] = posN[rl[i]];
    }
    static bool prep_ok(device const uint* idx, device const uint* remap, device const float4* wp, uint t) {
        uint a = remap[idx[3 * t]], b = remap[idx[3 * t + 1]], c = remap[idx[3 * t + 2]];
        if (a == b || b == c || a == c) return false;
        return length(cross(wp[b].xyz - wp[a].xyz, wp[c].xyz - wp[a].xyz)) > 1e-15f;
    }
    kernel void dec_prep_flag(device const uint* idx [[buffer(0)]], device const uint* remap [[buffer(1)]], device const float4* wp [[buffer(2)]],
                              device uint* flag [[buffer(3)]], constant GP& p [[buffer(15)]], uint t [[thread_position_in_grid]]) {
        flag[t] = (t < p.a && prep_ok(idx, remap, wp, t)) ? 1 : 0;
    }
    kernel void dec_prep_scatter(device const uint* idx [[buffer(0)]], device const uint* remap [[buffer(1)]], device const uint* sc [[buffer(2)]],
                                 device const float2* uv [[buffer(3)]], device uint* tri [[buffer(4)]], device float2* cuv [[buffer(5)]],
                                 constant GP& p [[buffer(15)]], uint t [[thread_position_in_grid]]) {
        if (t >= p.a || sc[t + 1] == sc[t]) return;
        uint o = sc[t];
        for (uint k = 0; k < 3; k++) {
            tri[3 * o + k] = remap[idx[3 * t + k]];
            if (p.c & 1) cuv[3 * o + k] = uv[idx[3 * t + k]];
        }
    }
    """
}
#else
public extension Mesh3 {
    func decimate(targetTriangles: Int, creaseDegrees: Double = 40, backend: MeshBackend) -> (mesh: Mesh3, report: DecimationReport) {
        decimate(targetTriangles: targetTriangles, creaseDegrees: creaseDegrees)
    }
    func decimated(targetTriangles: Int, creaseDegrees: Double = 40, backend: MeshBackend) -> Mesh3 { decimated(targetTriangles: targetTriangles, creaseDegrees: creaseDegrees) }
    func decimated(ratio: Double, creaseDegrees: Double = 40, backend: MeshBackend) -> Mesh3 { decimated(ratio: ratio, creaseDegrees: creaseDegrees) }
}
#endif
