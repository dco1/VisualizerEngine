//  Mesh3+CollapseGPU.swift — `GPUTriMesh`: an indexed triangle mesh that lives in Metal buffers and is edited by
//  rounds of parallel edge collapses (decimation and the remesher's collapse step share it).
//
//  Guarantees: after every round `tri` / `pos` / `cuv` hold exactly the live faces (compacted), so the CPU can read
//  them back at any time; vertices removed by a collapse keep their slot (no faces reference them). Each round is
//  two command buffers: A (adjacency, features, candidates, selection) and B (apply, relabel, compact), with one
//  CPU look between them at the number selected (and, when over budget, the keys of the selected candidates, to
//  pick the cheapest ones). Everything else, including the quadric build, cost evaluation, legality and the
//  independent-set selection, is a kernel.
//  Why a CPU look at all: the round count and the budget cut need a number back; reading 4 bytes costs one
//  synchronisation, ~20 of them for a 10× decimation.

import simd

#if canImport(Metal)
import Metal

struct CollapseParams {
    var hasUV = false, remesh = false
    var cosCrease: Float = 0, cosTurn: Float = 0, lo: Float = 0, hi: Float = 0
}

final class GPUTriMesh {
    let g: ModelGPU
    var V: Int, F: Int
    let hasUV: Bool
    var pos: MTLBuffer, tri: MTLBuffer, cuv: MTLBuffer
    var capV: Int, capF: Int
    var q: MTLBuffer, moved: MTLBuffer, anchor: MTLBuffer, posAlt: MTLBuffer
    var triOut: MTLBuffer, cuvOut: MTLBuffer
    var vs: MTLBuffer, cur: MTLBuffer, vl: MTLBuffer, vfeat: MTLBuffer
    var m1: MTLBuffer, m2: MTLBuffer, vmap: MTLBuffer, rec: MTLBuffer, vuv: MTLBuffer
    var cand: MTLBuffer, sel: MTLBuffer, keepF: MTLBuffer

    /// Takes `pos` (float4 × capV), `tri` (uint × 3capF) and `cuv` (float2 × 3capF, or nil) as they are.
    init(_ g: ModelGPU, V: Int, F: Int, capV: Int, capF: Int, pos: MTLBuffer, tri: MTLBuffer, cuv: MTLBuffer?) {
        self.g = g; self.V = V; self.F = F; self.capV = capV; self.capF = capF
        hasUV = cuv != nil
        self.pos = pos; self.tri = tri
        self.cuv = cuv ?? g.buffer(count: 3 * capF, SIMD2<Float>.self)
        q = g.buffer(count: 3 * capV, SIMD4<Float>.self); moved = g.buffer(count: capV, UInt32.self)
        triOut = g.buffer(count: 3 * capF, UInt32.self); cuvOut = g.buffer(count: hasUV ? 3 * capF : 1, SIMD2<Float>.self)
        vs = g.buffer(count: capV + 1, UInt32.self); cur = g.buffer(count: capV, UInt32.self)
        vl = g.buffer(count: 3 * capF, UInt32.self); vfeat = g.buffer(count: capV, SIMD4<UInt32>.self)
        m1 = g.buffer(count: capV, UInt64.self); m2 = g.buffer(count: capV, UInt64.self)
        vmap = g.buffer(count: capV, UInt32.self); rec = g.buffer(count: capV, SIMD2<UInt32>.self)
        vuv = g.buffer(count: capV, SIMD4<Float>.self)
        cand = g.buffer(count: 3 * capF, Cand.self); sel = g.buffer(count: 3 * capF + 1, UInt32.self)
        keepF = g.buffer(count: capF + 1, UInt32.self)
        memset(moved.contents(), 0, capV * 4)
        anchor = g.buffer(count: capV, SIMD4<Float>.self)
        posAlt = g.buffer(count: capV, SIMD4<Float>.self)
        memcpy(anchor.contents(), pos.contents(), V * 16)
    }

    struct Cand { var p = SIMD4<Float>(); var cost: Float = 0; var keep: UInt32 = 0, remove: UInt32 = 0; var lerp: Float = 0
                  var sh0: UInt32 = 0, sh1: UInt32 = 0, pad0: UInt32 = 0, pad1: UInt32 = 0 }

    func gp(_ cp: CollapseParams) -> GP {
        GP(a: UInt32(V), b: UInt32(F), c: (cp.hasUV ? 1 : 0) | (cp.remesh ? 2 : 0), d: UInt32(F),
           x: cp.cosCrease, y: cp.cosTurn, z: cp.lo, w: cp.hi)
    }

    // MARK: encoders

    func encodeAdjacency(_ s: ModelPass) {
        s.go("zero_u32", V + 1, [vs], GP(a: UInt32(V + 1)))
        s.go("vf_count", 3 * F, [tri, vs], GP(b: UInt32(F)))
        s.scan(vs, V + 1)
        s.go("vf_cursor", V, [vs, cur], GP(a: UInt32(V)))
        s.go("vf_fill", 3 * F, [tri, cur, vl], GP(b: UInt32(F)))
    }
    func encodeFeatures(_ s: ModelPass, _ cp: CollapseParams) {
        s.go("vtx_feat", V, [tri, pos, vs, vl, cuv, vfeat], gp(cp))
    }
    func encodeCandidates(_ s: ModelPass, _ cp: CollapseParams) {
        let p = gp(cp)
        s.go("cand_eval", 3 * F, [tri, pos, vs, vl, cuv, vfeat, q, cand, anchor], p)
        s.go("edge_min1", V, [tri, pos, vs, vl, cand, m1], p)
        s.go("edge_min2", V, [tri, pos, vs, vl, m1, m2], p)
    }
    func encodeSelect(_ s: ModelPass, _ cp: CollapseParams, threshold: UInt64 = .max) {
        var p = gp(cp)
        p.c = UInt32(threshold >> 32); p.d = UInt32(threshold & 0xFFFF_FFFF)
        s.go("edge_select", 3 * F + 1, [cand, m2, sel], p)
        s.scan(sel, 3 * F + 1)
    }

    /// Plane quadrics (area weighted) and constraint planes along boundary / crease edges, per vertex.
    func buildQuadrics(_ cp: CollapseParams) -> Bool {
        g.run { s in
            encodeAdjacency(s)
            s.go("quad_init", V, [tri, pos, vs, vl, q], gp(cp))
        }
    }

    /// One round of collapses, at most `maxCollapses` of them (the cheapest). Returns how many were done and the
    /// largest cost among them; nil on a GPU error.
    func collapseRound(_ cp: CollapseParams, maxCollapses: Int) -> (count: Int, maxCost: Float)? {
        guard F > 4 else { return (0, 0) }
        guard g.run({ s in
            encodeAdjacency(s)
            encodeFeatures(s, cp)
            encodeCandidates(s, cp)
            encodeSelect(s, cp)
        }) else { return nil }
        var count = Int(sel.ptr(UInt32.self)[3 * F])
        if count == 0 { return (0, 0) }
        if count > maxCollapses {
            let sc = sel.ptr(UInt32.self), cd = cand.ptr(Cand.self)
            var keys: [UInt64] = []
            for c in 0 ..< 3 * F where sc[c + 1] != sc[c] { keys.append(UInt64(cd[c].cost.bitPattern) << 32 | UInt64(c)) }
            keys.sort()
            let thr = keys[Swift.max(0, maxCollapses - 1)]
            guard g.run({ s in encodeSelect(s, cp, threshold: thr) }) else { return nil }
            count = Int(sel.ptr(UInt32.self)[3 * F])
        }
        var maxCost: Float = 0
        let sc = sel.ptr(UInt32.self), cd = cand.ptr(Cand.self)
        for c in 0 ..< 3 * F where sc[c + 1] != sc[c] { maxCost = Swift.max(maxCost, cd[c].cost) }
        let p = gp(cp)
        guard g.run({ s in
            s.go("vmap_init", V, [vmap, vuv], p)
            s.go("collapse_apply", 3 * F, [pos, q, cand, sel, vmap, rec, vuv, tri, cuv, moved, anchor], p)
            s.go("relabel", 3 * F, [tri, cuv, vmap, rec, vuv, triOut, cuvOut], p)
            s.go("face_flag", F + 1, [triOut, keepF], p)
            s.scan(keepF, F + 1)
            s.go("face_scatter", F, [triOut, cuvOut, keepF, tri, cuv], p)
        }) else { return nil }
        F = Int(keepF.ptr(UInt32.self)[F])
        return (count, maxCost)
    }
}
#endif
