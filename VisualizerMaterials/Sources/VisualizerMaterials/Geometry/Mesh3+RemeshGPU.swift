//  Mesh3+RemeshGPU.swift — isotropic remeshing on the GPU (ticket item M1).
//  `Mesh3.isotropicRemeshed(targetEdgeLength:iterations:creaseDegrees:backend:)`.
//
//  Guarantees: Botsch–Kobbelt's loop as in the CPU `isotropicRemeshed` (`Mesh3+Remesh.swift`): split edges longer than
//  4/3 L, collapse edges shorter than 4/5 L (never leaving an edge longer than 4/3 L), flip toward valence 6 (4 on a
//  boundary), tangentially smooth and project back onto the ORIGINAL surface. Open boundaries and creases keep their
//  vertices fixed, winding is kept, a closed mesh stays closed, the result has no UVs.
//  What runs where: all of it is kernels. Splitting is one parallel sweep (every long edge at once, red/green face
//  patterns, new vertex ids by prefix scan, midpoints projected onto the surface through the BVH); collapsing is the
//  decimator's parallel independent-set rounds with edge length as the cost; flipping is a parallel sweep in which an
//  edge flips when it wins its two faces and its opposite-vertex pair by atomic min; smoothing is a Jacobi step per
//  vertex followed by a closest-point query on the GPU BVH (`SurfaceBVHGPU`, built on the CPU once, queried in Metal).
//  CPU: the BVH build (one Morton sort), reading three counts per phase back, and the output assembly
//  (`TriTopology.toMesh3` splitting vertices per smooth group).
//  Differs from the CPU: parallel sweeps replace sequential longest-first order (split sweeps up to 6, collapse rounds
//  until none is left (at most 64), one flip sweep per iteration), smoothing is Jacobi not Gauss–Seidel, creases are re-detected each pass,
//  so the tessellation differs while the edge-length distribution, the surface error and topology guarantees match.

import simd

#if canImport(Metal)
import Metal

extension GPUTriMesh {
    /// The same mesh in buffers with room for `capV` vertices and `capF` faces.
    func grown(capV: Int, capF: Int) -> GPUTriMesh {
        let np = g.buffer(count: capV, SIMD4<Float>.self), nt = g.buffer(count: 3 * capF, UInt32.self)
        memcpy(np.contents(), pos.contents(), V * 16); memcpy(nt.contents(), tri.contents(), F * 12)
        return GPUTriMesh(g, V: V, F: F, capV: capV, capF: capF, pos: np, tri: nt, cuv: nil)
    }
}

private struct RemeshRun {
    let g: ModelGPU
    var m: GPUTriMesh
    let bvh: SurfaceBVHGPU
    var cp: CollapseParams
    let L: Float

    mutating func splitSweep() -> Int? {
        let F0 = m.F, V0 = m.V
        let flag = g.buffer(count: 3 * F0 + 1, UInt32.self), cnt = g.buffer(count: F0 + 1, UInt32.self)
        let p = GP(a: UInt32(V0), b: UInt32(F0), x: cp.cosCrease, w: cp.hi)
        guard g.run({ s in
            m.encodeAdjacency(s)
            s.go("rm_split_flag", 3 * F0 + 1, [m.tri, m.pos, m.vs, m.vl, flag], p)
            s.scan(flag, 3 * F0 + 1)
            s.go("rm_split_count", F0 + 1, [m.tri, m.pos, m.vs, m.vl, flag, cnt], p)
            s.scan(cnt, F0 + 1)
        }) else { return nil }
        let nNew = Int(flag.ptr(UInt32.self)[3 * F0]), newF = Int(cnt.ptr(UInt32.self)[F0])
        if nNew == 0 { return 0 }
        if V0 + nNew > m.capV || newF > m.capF { m = m.grown(capV: Swift.max(m.capV, (V0 + nNew) * 3 / 2), capF: Swift.max(m.capF, newF * 3 / 2)) }
        guard g.run({ s in
            m.encodeAdjacency(s)
            s.go("rm_split_mid", 3 * F0, [m.tri, m.pos, m.vs, m.vl, flag] + bvh.buffers + [m.pos], p)
            s.go("rm_split_emit", F0, [m.tri, m.pos, m.vs, m.vl, flag, cnt, m.triOut], p)
        }) else { return nil }
        swap(&m.tri, &m.triOut)
        m.V = V0 + nNew; m.F = newF
        return nNew
    }

    mutating func flipSweep() -> Int? {
        let F = m.F, V = m.V
        let fc = g.buffer(count: 3 * F, SIMD4<UInt32>.self), val = g.buffer(count: V, SIMD2<UInt32>.self)
        let claimF = g.buffer(count: F, UInt32.self), claimV = g.buffer(count: V, UInt32.self), done = g.buffer(count: 1, UInt32.self)
        done.ptr(UInt32.self).pointee = 0
        let p = m.gp(cp)
        guard g.run({ s in
            m.encodeAdjacency(s)
            m.encodeFeatures(s, cp)
            s.go("rm_val", V, [m.tri, m.pos, m.vs, m.vl, val], p)
            s.go("rm_flip_eval", 3 * F, [m.tri, m.pos, m.vs, m.vl, m.vfeat, val, fc], p)
            s.go("fill_u32", F, [claimF], GP(a: UInt32(F), b: 0xFFFF_FFFF))
            s.go("fill_u32", V, [claimV], GP(a: UInt32(V), b: 0xFFFF_FFFF))
            s.go("rm_flip_claim", 3 * F, [fc, claimF, claimV], p)
            s.go("rm_flip_apply", 3 * F, [m.tri, fc, claimF, claimV, done], p)
        }) else { return nil }
        return Int(done.ptr(UInt32.self).pointee)
    }

    mutating func smooth() -> Bool {
        var p = m.gp(cp)
        p.x = 0.8
        guard g.run({ s in
            m.encodeAdjacency(s)
            m.encodeFeatures(s, cp)
            s.go("rm_smooth", m.V, [m.tri, m.pos, m.vs, m.vl, m.vfeat] + bvh.buffers + [m.posAlt], p)
        }) else { return false }
        swap(&m.pos, &m.posAlt)
        return true
    }
}

public extension Mesh3 {

    /// `isotropicRemeshed` on the chosen backend (`.gpu` falls back to the CPU without a Metal device).
    func isotropicRemeshed(targetEdgeLength L: Double, iterations: Int = 5, creaseDegrees: Double = 40, backend: MeshBackend) -> Mesh3 {
        if backend.useGPU(triangles: triangleCount),
           let r = isotropicRemeshedGPU(targetEdgeLength: L, iterations: iterations, creaseDegrees: creaseDegrees) { return r }
        return isotropicRemeshed(targetEdgeLength: L, iterations: iterations, creaseDegrees: creaseDegrees)
    }

    internal func isotropicRemeshedGPU(targetEdgeLength L: Double, iterations: Int, creaseDegrees: Double) -> Mesh3? {
        guard L > 0, let g = ModelGPU.shared, let prep = gpuPrep(g, keepUV: false, capFactor: 1) else { return nil }
        let center = prep.center, diag = prep.diag
        // the original surface, for the projection
        let m0 = prep.mesh
        let p0 = m0.pos.ptr(SIMD4<Float>.self), t0 = m0.tri.ptr(UInt32.self)
        let tris = (0 ..< m0.F).map { (p0[Int(t0[3 * $0])], p0[Int(t0[3 * $0 + 1])], p0[Int(t0[3 * $0 + 2])]) }
        guard let bvh = SurfaceBVHGPU(g, tris: tris) else { return nil }
        let Ln = Float(L / diag)
        let cosC = Float(cos(creaseDegrees * .pi / 180))
        var run = RemeshRun(g: g, m: m0.grown(capV: m0.V * 2, capF: m0.F * 2), bvh: bvh,
                            cp: CollapseParams(hasUV: false, remesh: true, cosCrease: cosC, cosTurn: cosC, lo: Ln * 0.8, hi: Ln * 4 / 3), L: Ln)
        for _ in 0 ..< iterations {
            for _ in 0 ..< 6 { guard let n = run.splitSweep() else { return nil }; if n == 0 { break } }
            for _ in 0 ..< 64 {
                guard let r = run.m.collapseRound(run.cp, maxCollapses: Int.max) else { return nil }
                if r.count == 0 { break }
            }
            guard run.flipSweep() != nil, run.smooth() else { return nil }
        }
        let m = run.m
        let pp = m.pos.ptr(SIMD4<Float>.self), tp = m.tri.ptr(UInt32.self)
        let outPos = (0 ..< m.V).map { center + Vec3(Double(pp[$0].x), Double(pp[$0].y), Double(pp[$0].z)) * diag }
        let tr = (0 ..< m.F).map { SIMD3<Int>(Int(tp[3 * $0]), Int(tp[3 * $0 + 1]), Int(tp[3 * $0 + 2])) }
        let t = TriTopology(welded: outPos, tri: tr, cuv: [Vec2](repeating: .zero, count: 3 * m.F))
        return t.toMesh3(creaseDegrees: creaseDegrees)
    }
}
#else
public extension Mesh3 {
    func isotropicRemeshed(targetEdgeLength L: Double, iterations: Int = 5, creaseDegrees: Double = 40, backend: MeshBackend) -> Mesh3 {
        isotropicRemeshed(targetEdgeLength: L, iterations: iterations, creaseDegrees: creaseDegrees)
    }
}
#endif
