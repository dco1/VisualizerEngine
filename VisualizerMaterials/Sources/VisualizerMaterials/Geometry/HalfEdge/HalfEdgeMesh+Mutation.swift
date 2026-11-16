//  Mutation primitives for HalfEdgeMesh: the vocabulary a topology tool (extrude, inset, loop cut, bridge,
//  knife) is written in. Each returns nil / false and leaves the mesh untouched when it cannot be done
//  without breaking the mesh. Ids: `splitEdge` and `splitFace` keep every existing id; `dissolveEdge`,
//  `flipEdge` and `collapseEdge` re-add the faces they touch into their own slots, so face ids are stable too
//  (edge ids of touched faces are renumbered). After any run of edits `validate()` should be empty.
//  Corner data (uv, normal) is carried: split points interpolate, so seams stay seams.

import simd

public extension HalfEdgeMesh {

    // MARK: split edge

    /// Put a vertex on edge `e` at parameter `t` (0 = its first endpoint). Adjacent faces gain one corner
    /// (a triangle becomes a quad). Returns the new vertex; the old edge id keeps the first half.
    @discardableResult
    mutating func splitEdge(_ e: Int, at t: Double = 0.5) -> Int? {
        guard isAlive(edge: e) else { return nil }
        let h = edgeHE[e], tw = heTwin[h]
        let a = heOrigin[h], b = heOrigin[heNext[h]]
        let m = addVertex(positions[a] + (positions[b] - positions[a]) * t)
        let yh = insertCorner(after: h, vertex: m, fraction: t)
        let e2 = edgeHE.count
        if tw >= 0 {
            let yt = insertCorner(after: tw, vertex: m, fraction: 1 - t)
            heTwin[h] = yt; heTwin[yt] = h; heEdge[yt] = e
            heTwin[yh] = tw; heTwin[tw] = yh; heEdge[tw] = e2
            edgeHE.append(yh)
        } else { edgeHE.append(yh) }
        heEdge[yh] = e2
        edgeHE[e] = h
        vertexHE[m] = yh
        return m
    }

    /// `x` (p -> q) becomes p -> m, and a new halfedge m -> q follows it in the same face.
    private mutating func insertCorner(after x: Int, vertex m: Int, fraction s: Double) -> Int {
        let q = heOrigin[heNext[x]], p = heOrigin[x], nx = heNext[x], f = heFace[x]
        let y = heOrigin.count
        let nn = normalize3(heNormal[x] * (1 - s) + heNormal[nx] * s)
        heOrigin.append(m); heNext.append(nx); hePrev.append(x); heTwin.append(-1); heFace.append(f); heEdge.append(-1)
        heUV.append(heUV[x] * (1 - s) + heUV[nx] * s); heNormal.append(len3(nn) > 0 ? nn : heNormal[x])
        hePrev[nx] = y; heNext[x] = y
        directed[Self.key(p, q)] = nil
        directed[Self.key(p, m)] = x; directed[Self.key(m, q)] = y
        return y
    }

    // MARK: split face

    /// Cut face `f` along a new edge between two of its corners, given as halfedges `ha`, `hb` of `f` (the
    /// corner is the halfedge's origin). They must not be adjacent. `f` keeps the loop `ha ... prev(hb)`; the
    /// returned new face is `hb ... prev(ha)`.
    @discardableResult
    mutating func splitFace(_ f: Int, from ha: Int, to hb: Int) -> Int? {
        guard isAlive(face: f), ha != hb, heFace[ha] == f, heFace[hb] == f,
              heNext[ha] != hb, heNext[hb] != ha else { return nil }
        let A = heOrigin[ha], B = heOrigin[hb]
        guard directed[Self.key(A, B)] == nil, directed[Self.key(B, A)] == nil else { return nil }
        let pa = hePrev[ha], pb = hePrev[hb]
        let g = faceHE.count; faceHE.append(hb)
        let d1 = heOrigin.count, d2 = d1 + 1, e = edgeHE.count      // d1: A -> B (in g), d2: B -> A (in f)
        heOrigin += [A, B]; heNext += [hb, ha]; hePrev += [pa, pb]; heTwin += [d2, d1]; heFace += [g, f]
        heEdge += [e, e]; heUV += [heUV[ha], heUV[hb]]; heNormal += [heNormal[ha], heNormal[hb]]
        edgeHE.append(d1)
        heNext[pb] = d2; hePrev[ha] = d2; heNext[pa] = d1; hePrev[hb] = d1
        var h = hb
        repeat { heFace[h] = g; h = heNext[h] } while h != hb
        faceHE[f] = ha
        directed[Self.key(A, B)] = d1; directed[Self.key(B, A)] = d2
        return g
    }

    /// `splitFace` by vertex ids (the face's corners at `a` and `b`).
    @discardableResult
    mutating func splitFace(_ f: Int, between a: Int, _ b: Int) -> Int? {
        let hs = halfedges(of: f)
        guard let ha = hs.first(where: { heOrigin[$0] == a }), let hb = hs.first(where: { heOrigin[$0] == b }) else { return nil }
        return splitFace(f, from: ha, to: hb)
    }

    // MARK: dissolve / flip / collapse

    private struct Poly { var v: [Int]; var uv: [Vec2]; var n: [Vec3] }
    private func poly(of hs: [Int]) -> Poly {
        Poly(v: hs.map { heOrigin[$0] }, uv: hs.map { heUV[$0] }, n: hs.map { heNormal[$0] })
    }
    private func rotated(_ hs: [Int], startingAfter h: Int) -> [Int] {
        let i = hs.firstIndex(of: h)!
        return Array(hs[(i + 1)...] + hs[..<i])               // every halfedge but h, starting after it
    }

    /// Remove an interior edge, joining its two faces into one n-gon (the face id of `h`'s side survives).
    /// nil on a boundary edge, an edge with the same face on both sides, or when the faces share another vertex.
    @discardableResult
    mutating func dissolveEdge(_ e: Int) -> Int? {
        guard isAlive(edge: e) else { return nil }
        let h = edgeHE[e], t = heTwin[h]
        guard t >= 0, heFace[h] != heFace[t] else { return nil }
        let f1 = heFace[h], f2 = heFace[t]
        let loop = rotated(halfedges(of: f1), startingAfter: h) + rotated(halfedges(of: f2), startingAfter: t)
        let p = poly(of: loop)
        guard Set(p.v).count == p.v.count else { return nil }
        removeFaces([f1, f2], scan: false)
        return addFace(p.v, uvs: p.uv, normals: p.n, slot: f1)
    }

    /// Swap the diagonal of two triangles sharing `e`. false (unchanged) unless both are triangles and the new diagonal is free.
    @discardableResult
    mutating func flipEdge(_ e: Int) -> Bool {
        guard isAlive(edge: e) else { return false }
        let h = edgeHE[e], t = heTwin[h]
        guard t >= 0, heFace[h] != heFace[t], heNext[heNext[heNext[h]]] == h, heNext[heNext[heNext[t]]] == t else { return false }
        let f1 = heFace[h], f2 = heFace[t]
        let c = heOrigin[hePrev[h]], d = heOrigin[hePrev[t]]
        guard c != d, directed[Self.key(c, d)] == nil, directed[Self.key(d, c)] == nil else { return false }
        // f1 = (a b c), f2 = (b a d) -> (c a d), (d b c)
        let a = heOrigin[h], b = heOrigin[t]
        let hb = heNext[h], hc = hePrev[h], ta = heNext[t], td = hePrev[t]
        // each new corner keeps the data of the old halfedge that leaves it along a surviving edge
        let (uc, nc) = (heUV[hc], heNormal[hc]), (ub, nb) = (heUV[hb], heNormal[hb])
        let (ua, na) = (heUV[ta], heNormal[ta]), (ud, nd) = (heUV[td], heNormal[td])
        removeFaces([f1, f2], scan: false)
        addFace([c, a, d], uvs: [uc, ua, ud], normals: [nc, na, nd], slot: f1)
        addFace([d, b, c], uvs: [ud, ub, uc], normals: [nd, nb, nc], slot: f2)
        return true
    }

    /// Merge an edge's endpoints into the first one, at `t` along the edge (0.5 = midpoint). Faces that
    /// shrink below three corners vanish. nil (nothing changed) unless it keeps the mesh a clean 2-manifold:
    /// the endpoints' shared neighbours must be exactly the apexes of the triangles on the edge, an interior
    /// edge between two boundary vertices is refused, and no face may fold onto itself or onto another.
    @discardableResult
    mutating func collapseEdge(_ e: Int, at t: Double = 0.5) -> Int? {
        guard isAlive(edge: e) else { return nil }
        let h = edgeHE[e]
        let a = heOrigin[h], b = heOrigin[heNext[h]]
        let na = Set(neighbours(of: a)), nb = Set(neighbours(of: b))
        var apex = Set<Int>()
        for f in faces(ofEdge: e) { let hs = halfedges(of: f); if hs.count == 3 { apex.insert(heOrigin[hs.first { heOrigin[$0] != a && heOrigin[$0] != b }!]) } }
        guard na.intersection(nb) == apex, !(isClosed && vertexCount <= 4) else { return nil }   // a tetrahedron would flatten to a pillow
        if heTwin[h] >= 0, isBoundary(vertex: a), isBoundary(vertex: b) { return nil }

        let affected = Array(Set(faces(around: a) + faces(around: b)))
        var rebuilt: [(slot: Int, p: Poly)] = []
        var newEdges = Set<UInt64>()
        for f in affected {
            let hs = halfedges(of: f)
            var p = Poly(v: [], uv: [], n: [])
            for x in hs { p.v.append(heOrigin[x] == b ? a : heOrigin[x]); p.uv.append(heUV[x]); p.n.append(heNormal[x]) }
            var i = 0                                          // drop the first corner of each (a, a) pair
            while i < p.v.count && p.v.count > 1 {
                let j = (i + 1) % p.v.count
                if p.v[i] == p.v[j] { p.v.remove(at: i); p.uv.remove(at: i); p.n.remove(at: i) } else { i += 1 }
            }
            if p.v.count < 3 { continue }
            guard Set(p.v).count == p.v.count else { return nil }
            for k in 0..<p.v.count {
                let key = Self.key(p.v[k], p.v[(k + 1) % p.v.count])
                guard newEdges.insert(key).inserted else { return nil }
                if let other = directed[key], !affected.contains(heFace[other]) { return nil }
            }
            rebuilt.append((f, p))
        }
        removeFaces(affected, scan: false)
        positions[a] = positions[a] + (positions[b] - positions[a]) * t
        vertexHE[b] = -1
        for r in rebuilt { addFace(r.p.v, uvs: r.p.uv, normals: r.p.n, slot: r.slot) }
        return a
    }

    // MARK: orientation

    /// Reverse the winding of faces (corner data stays with its vertex). Neighbouring faces must be flipped
    /// along with them or the shared edges would have the same direction twice; returns false (unchanged) then.
    @discardableResult
    mutating func flip(faces fs: some Sequence<Int>) -> Bool {
        let list = Array(Set(fs)).filter { isAlive(face: $0) }
        let saved = list.map { f -> (Int, Poly) in (f, poly(of: halfedges(of: f))) }
        let backup = self
        removeFaces(list, scan: false)
        for (f, p) in saved {
            let idx = [0] + Array((1..<p.v.count).reversed())
            if addFace(idx.map { p.v[$0] }, uvs: idx.map { p.uv[$0] }, normals: idx.map { p.n[$0] }, slot: f) == nil {
                self = backup; return false
            }
        }
        return true
    }

    /// Flip every closed component that encloses negative volume so all closed shells face outward.
    /// Returns how many components were flipped.
    @discardableResult
    mutating func orientOutward() -> Int {
        var flipped = 0
        for comp in connectedComponents() where isClosedComponent(comp) && signedVolume(of: comp) < 0 {
            if flip(faces: comp) { flipped += 1 }
        }
        return flipped
    }
    func isClosedComponent(_ comp: [Int]) -> Bool {
        comp.allSatisfy { f in halfedges(of: f).allSatisfy { heTwin[$0] >= 0 } }
    }
}
