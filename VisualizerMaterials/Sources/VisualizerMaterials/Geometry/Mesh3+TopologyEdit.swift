//  TriTopology edits: edge collapse, split and flip, with the feature rules (boundary, crease, UV seam) and the
//  validity tests (link condition, no flipped triangles) that decimation and remeshing share.
//
//  Guarantees: each edit keeps `vf`, `creases` and per-corner UVs consistent, and keeps the original winding.
//  A collapse that passes `collapseIsValid` keeps a manifold mesh manifold. Does not guarantee geometric
//  quality — callers pick which edges to edit and where the survivor goes.

import simd

extension TriTopology {

    // MARK: features

    func isSeam(_ a: Int, _ b: Int, _ f1: Int, _ f2: Int) -> Bool {
        let d1 = uv(f1, a) - uv(f2, a), d2 = uv(f1, b) - uv(f2, b)
        return Swift.max(abs(d1.x), abs(d1.y), abs(d2.x), abs(d2.y)) > 1e-6
    }

    /// Boundary / non-manifold / crease / UV-seam edge: one the surface must keep.
    func isFeatureEdge(_ a: Int, _ b: Int) -> Bool {
        let sh = facesOn(a, b)
        if sh.count != 2 { return true }
        return creases.contains(TriTopology.key(a, b)) || isSeam(a, b, sh[0], sh[1])
    }

    func featureNeighbors(_ v: Int) -> [Int] { neighbors(v).filter { isFeatureEdge(v, $0) } }

    /// A vertex on exactly two feature edges that turn by more than `cosTurn` (cosine of the allowed bend) is a corner.
    func isCorner(_ v: Int, featureNeighbors fn: [Int], cosTurn: Double) -> Bool {
        guard fn.count == 2 else { return true }
        let d1 = normalize3(pos[v] - pos[fn[0]]), d2 = normalize3(pos[fn[1]] - pos[v])
        return dot3(d1, d2) < cosTurn
    }

    struct CollapseRule { var free = false, removeU = false, removeV = false }

    /// Which way an edge may collapse. `free`: neither end is on a feature, so the survivor may move.
    /// Otherwise a feature vertex can only be removed along a feature edge into a neighbour (it never leaves the
    /// feature), and a corner (≠ 2 feature edges, or a sharp bend) is never removed.
    func collapseRule(_ u: Int, _ v: Int, cosTurn: Double) -> CollapseRule {
        let fu = featureNeighbors(u), fv = featureNeighbors(v)
        var r = CollapseRule()
        if fu.isEmpty, fv.isEmpty { r.free = true; r.removeU = true; r.removeV = true; return r }
        let edgeFeat = isFeatureEdge(u, v)
        r.removeU = fu.isEmpty || (edgeFeat && !isCorner(u, featureNeighbors: fu, cosTurn: cosTurn))
        r.removeV = fv.isEmpty || (edgeFeat && !isCorner(v, featureNeighbors: fv, cosTurn: cosTurn))
        return r
    }

    // MARK: validity

    private func flips(_ f: Int, replacing v: Int, with p: Vec3) -> Bool {
        let t = tri[f]
        let n0 = rawNormal(f)
        let q = { (w: Int) -> Vec3 in w == v ? p : self.pos[w] }
        let n1 = cross3(q(t.y) - q(t.x), q(t.z) - q(t.x))
        let l0 = len3(n0), l1 = len3(n1)
        if l0 < 1e-18 { return false }
        return l1 < l0 * 1e-3 || dot3(n0, n1) < 0.25 * l0 * l1
    }

    /// Link condition (the two ends share exactly the neighbours the shared faces imply) and no triangle flipped or
    /// squashed by moving the survivor `keep` to `p`.
    func collapseIsValid(keep: Int, remove: Int, to p: Vec3) -> Bool {
        let sh = facesOn(keep, remove)
        guard sh.count == 1 || sh.count == 2, aliveCount > 4 else { return false }
        let nk = neighbors(keep), nr = neighbors(remove)
        if nr.filter({ nk.contains($0) }).count != sh.count { return false }
        for f in vf[remove] where !sh.contains(f) { if flips(f, replacing: remove, with: p) { return false } }
        if p != pos[keep] { for f in vf[keep] where !sh.contains(f) { if flips(f, replacing: keep, with: p) { return false } } }
        return true
    }

    // MARK: edits

    /// Remove `remove`, keep `keep` at `p`. `lerpT` (free collapses) blends the UV along the edge; nil takes the
    /// survivor's own UV from the shared face on the same side (so seams stay put). Caller checks `collapseIsValid`.
    mutating func collapse(keep v: Int, remove u: Int, to p: Vec3, lerpT: Double? = nil) {
        let sh = facesOn(u, v)
        let nu = neighbors(u)
        if let t = lerpT, let s = sh.first {
            let uvNew = uv(s, u) * (1 - t) + uv(s, v) * t
            for f in vf[u] { cuv[f * 3 + corner(f, u)] = uvNew }
            for f in vf[v] { cuv[f * 3 + corner(f, v)] = uvNew }
        } else {
            for f in vf[u] where !sh.contains(f) {
                let c = uv(f, u)
                if let s = sh.first(where: { let d = uv($0, u) - c; return Swift.max(abs(d.x), abs(d.y)) < 1e-6 }) {
                    cuv[f * 3 + corner(f, u)] = uv(s, v)
                }
            }
        }
        for f in sh { kill(f) }
        for f in vf[u] {
            let k = corner(f, u)
            tri[f][k] = v
            vf[v].append(f)
        }
        vf[u] = []
        pos[v] = p
        for w in nu where w != v {
            let old = TriTopology.key(u, w)
            if creases.remove(old) != nil { creases.insert(TriTopology.key(v, w)) }
        }
        creases.remove(TriTopology.key(u, v))
    }

    /// Split edge (a, b) at `m`; returns the new vertex. Both faces keep their winding.
    @discardableResult
    mutating func split(_ a: Int, _ b: Int, at m: Vec3) -> Int {
        let w = addVertex(m)
        for f in facesOn(a, b) {
            let ca = corner(f, a), cb = corner(f, b)
            let uvm = (cuv[f * 3 + ca] + cuv[f * 3 + cb]) * 0.5
            let t = tri[f]
            var uvs = [cuv[f * 3], cuv[f * 3 + 1], cuv[f * 3 + 2]]
            // g: the old face with corner `a` replaced by w (it keeps b); f itself keeps a and takes w for b.
            var tg = t; tg[ca] = w
            var uvg = uvs; uvg[ca] = uvm
            uvs[cb] = uvm
            tri[f][cb] = w
            cuv[f * 3 + cb] = uvm
            if let i = vf[b].firstIndex(of: f) { vf[b].remove(at: i) }
            vf[w].append(f)
            addFace(tg.x, tg.y, tg.z, uvg[0], uvg[1], uvg[2])
        }
        if creases.remove(TriTopology.key(a, b)) != nil {
            creases.insert(TriTopology.key(a, w)); creases.insert(TriTopology.key(w, b))
        }
        return w
    }

    /// Flip the interior edge (a, b). Returns false (and changes nothing) for a boundary edge, a feature edge,
    /// an edge whose opposite corners are already joined, or a flip that would fold a triangle.
    @discardableResult
    mutating func flip(_ a: Int, _ b: Int) -> Bool {
        let sh = facesOn(a, b)
        guard sh.count == 2, !isFeatureEdge(a, b) else { return false }
        var f1 = sh[0], f2 = sh[1]
        // order so f1 runs a→b
        let t1 = tri[f1]
        if !(t1[(corner(f1, a) + 1) % 3] == b) { swap(&f1, &f2) }
        let x1 = third(f1, a, b), x2 = third(f2, a, b)
        guard x1 != x2, !vf[x1].contains(where: { contains($0, x2) }) else { return false }
        func area(_ p: Vec3, _ q: Vec3, _ r: Vec3) -> Vec3 { cross3(q - p, r - p) }
        let n1 = area(pos[a], pos[x2], pos[x1]), n2 = area(pos[x2], pos[b], pos[x1])
        let o = rawNormal(f1) + rawNormal(f2)
        guard dot3(n1, o) > 0, dot3(n2, o) > 0 else { return false }
        let l1 = len3(n1), l2 = len3(n2)
        guard l1 > 1e-14, l2 > 1e-14, dot3(n1, n2) > 0.5 * l1 * l2 else { return false }
        let ua = uv(f1, a), ub = uv(f1, b), u1 = uv(f1, x1), u2 = uv(f2, x2)
        tri[f1] = SIMD3(a, x2, x1); cuv[f1 * 3] = ua; cuv[f1 * 3 + 1] = u2; cuv[f1 * 3 + 2] = u1
        tri[f2] = SIMD3(x2, b, x1); cuv[f2 * 3] = u2; cuv[f2 * 3 + 1] = ub; cuv[f2 * 3 + 2] = u1
        if let i = vf[a].firstIndex(of: f2) { vf[a].remove(at: i) }
        if let i = vf[b].firstIndex(of: f1) { vf[b].remove(at: i) }
        vf[x1].append(f2); vf[x2].append(f1)
        return true
    }
}
