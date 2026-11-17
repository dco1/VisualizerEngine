//  HalfEdge+Bevel — bevel (chamfer) selected edges on the topology, one segment: each selected edge becomes a
//  strip of quads `width` wide on each side, measured in the adjacent faces (the idea of Mesh3+Bevel, but on
//  the half-edge mesh, so it works on any n-gon mesh and returns the strip faces).
//
//  How: at every touched vertex v the ring of edges is walked. A face between two selected edges gets a mitre
//  point; a face between a selected and an unselected edge gets a point on the unselected edge (`width / sin`
//  of the angle between them, capped at 45% of that edge); an unselected edge next to a selected one has one
//  such point shared by both faces on it. Each face's corner is replaced by the list of those points (and v
//  itself where an unselected edge without a point remains, so a lone edge ending at a valence >= 4 vertex
//  closes with a triangle at v, and at valence 3 v disappears). A strip quad fills each selected edge and a
//  polygon each vertex where three or more points surround it. `addFace` proves every edge matched.
//  Refuses (nil, unchanged): no interior selected edge, a touched boundary vertex, or a vertex pattern the
//  rules above cannot close. Gaps: one segment only (no round profile), no per-edge widths.

import simd

public extension HalfEdgeMesh {

    @discardableResult
    mutating func bevel(_ sel: HESelection, width: Double) -> HESelection? {
        let chosen = Set(sel.edges.filter { isAlive(edge: $0) && !isBoundary(edge: $0) })
        guard !chosen.isEmpty, width > 0 else { return nil }
        var touched = Set<Int>()
        for e in chosen { let v = vertices(ofEdge: e); touched.insert(v.0); touched.insert(v.1) }
        guard touched.allSatisfy({ !isBoundary(vertex: $0) && valence(of: $0) >= 3 }) else { return nil }
        let backup = self

        struct Ring { var hs: [Int]; var sel: [Bool]; var bp: [Int?]; var mit: [Int?]; var persists: Bool }
        var rings: [Int: Ring] = [:]
        var slot: [Int: (v: Int, i: Int)] = [:]
        for v in touched.sorted() {
            let hs = outgoing(of: v), k = hs.count
            let s = hs.map { chosen.contains(heEdge[$0]) }
            var r = Ring(hs: hs, sel: s, bp: [Int?](repeating: nil, count: k), mit: [Int?](repeating: nil, count: k), persists: false)
            let p0 = positions[v]
            let dirs = hs.map { normalize3(positions[heOrigin[heNext[$0]]] - p0) }
            for i in 0..<k where !s[i] {
                let nb = [(i + k - 1) % k, (i + 1) % k].filter { s[$0] }
                if nb.isEmpty { r.persists = true; continue }
                let ds = nb.map { width / Swift.max(len3(cross3(dirs[i], dirs[$0])), 0.2) }
                let d = Swift.min(ds.reduce(0, +) / Double(ds.count), 0.45 * dist3(p0, positions[heOrigin[heNext[hs[i]]]]))
                r.bp[i] = addVertex(p0 + dirs[i] * d)
            }
            for i in 0..<k where s[i] && s[(i + k - 1) % k] {         // the face between two selected edges: mitre
                let f = heFace[hs[i]], n = faceNormal(f)
                let a = normalize3(cross3(n, normalize3(p0 - positions[heOrigin[hePrev[hs[i]]]]))), b = normalize3(cross3(n, dirs[i]))
                let kk = 1 + dot3(a, b)
                r.mit[i] = addVertex(p0 + (kk > 1e-6 ? (a + b) * (width / kk) : a * width))
            }
            rings[v] = r
            for (i, h) in hs.enumerated() { slot[h] = (v, i) }
        }
        /// What the corner at halfedge h turns into.
        func corner(_ h: Int) -> [Int] {
            guard let (v, i) = slot[h], let r = rings[v] else { return [heOrigin[h]] }
            let k = r.hs.count, a = r.sel[(i + k - 1) % k], b = r.sel[i]
            if a && b { return [r.mit[i]!] }
            if b { return [r.bp[(i + k - 1) % k]!] }
            if a { return [r.bp[i]!] }
            var out: [Int] = []
            if let x = r.bp[(i + k - 1) % k] { out.append(x) }
            if r.persists { out.append(v) }
            if let x = r.bp[i] { out.append(x) }
            return out
        }
        var affected = Set<Int>()
        for v in touched { for h in rings[v]!.hs { affected.insert(heFace[h]) } }
        var adds: [EditPoly] = []
        for f in affected.sorted() {
            var p = EditPoly(v: [], uv: [], n: [], slot: f)
            for h in halfedges(of: f) { for x in corner(h) { p.v.append(x); p.uv.append(heUV[h]); p.n!.append(heNormal[h]) } }
            adds.append(p)
        }
        func single(_ h: Int) -> Int? { let c = corner(h); return c.count == 1 ? c[0] : nil }
        var strips: [EditPoly] = []
        for e in chosen.sorted() {
            let h = edgeHE[e], t = heTwin[h]
            guard let aF = single(h), let bF = single(heNext[h]), let bG = single(t), let aG = single(heNext[t]) else { self = backup; return nil }
            strips.append(EditPoly(v: HalfEdgeMesh.dedupeCyclic([bF, aF, aG, bG])))
        }
        var caps: [EditPoly] = []
        for v in touched.sorted() {
            let r = rings[v]!, k = r.hs.count
            var pts: [Int] = []
            for i in 0..<k {
                if r.sel[i] { pts.append(corner(r.hs[i]).last!); pts.append(corner(r.hs[(i + 1) % k]).first!) }
                else { pts.append(r.bp[i] ?? v) }
            }
            let c = HalfEdgeMesh.dedupeCyclic(pts)
            if c.count >= 3 { caps.append(EditPoly(v: c.reversed())) }
        }
        guard let ids = rebuild(removing: Array(affected), adding: adds + strips.filter { $0.v.count >= 3 } + caps) else { self = backup; return nil }
        return selection(.face, ids[adds.count..<(adds.count + strips.filter { $0.v.count >= 3 }.count)])
    }

    /// Drop consecutive repeats, cyclically.
    static func dedupeCyclic(_ a: [Int]) -> [Int] {
        var out: [Int] = []
        for x in a where out.last != x { out.append(x) }
        while out.count > 1, out.first == out.last { out.removeLast() }
        return out
    }
}
