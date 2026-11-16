//  HalfEdge+LoopCut — loop cut (N parallel cuts along an edge ring) and edge slide.
//
//  Loop cut: the ring through `edge` (`edgeRing`: the parallel edges across a strip of quads) gets `cuts`
//  points on every edge, evenly spaced and shifted by `slide` (-1...1: 0 = even; at one cut, +/-0.5 of the
//  edge toward the ring's first / last end), then every quad of the ring is split between the matching points.
//  Faces at an open ring's end that are not quads only gain corners (stay n-gons). The new edges (the cut
//  loops, outermost first) are returned as an edge selection. A box ring cut once: V 8 -> 12, E 12 -> 20, F 6 -> 10.
//  Edge slide: the selected edges' vertices move along the edges leaving the loop to one side (`factor` > 0
//  toward the left of the loop's walking direction, < 0 toward the right) by `|factor|` of that edge. Topology
//  is untouched. Gaps: a vertex whose side edge is missing (boundary) stays; no clamping against other loops.

import simd

public extension HalfEdgeMesh {

    // MARK: loop cut

    @discardableResult
    mutating func loopCut(ringFrom startEdge: Int, cuts: Int = 1, slide: Double = 0) -> HESelection? {
        guard isAlive(edge: startEdge), cuts >= 1 else { return nil }
        let ring = edgeRing(from: startEdge)
        guard ring.edges.count >= 2 else { return nil }
        let backup = self
        // orient every ring edge: (a_i, b_i) with a_{i+1} the neighbour of a_i inside their shared quad
        var ends: [(Int, Int)] = []
        for (i, e) in ring.edges.enumerated() {
            if i == 0 { ends.append(vertices(ofEdge: e)); continue }
            let (pa, pb) = ends[i - 1], (x, y) = vertices(ofEdge: e)
            let shared = Set(faces(ofEdge: ring.edges[i - 1])).intersection(faces(ofEdge: e)).first { valence(ofFace: $0) == 4 }
            guard let q = shared else { return nil }
            let vs = vertices(of: q), n = 4
            func adjacent(_ u: Int, _ w: Int) -> Bool { let i = vs.firstIndex(of: u)!, j = vs.firstIndex(of: w)!; return (i + 1) % n == j || (j + 1) % n == i }
            ends.append(adjacent(pa, x) && adjacent(pb, y) ? (x, y) : (y, x))
        }
        let sl = Swift.max(-0.98, Swift.min(0.98, slide))
        let us = (1...cuts).map { Swift.max(0.01, Swift.min(0.99, Double($0) / Double(cuts + 1) + sl / Double(cuts + 1))) }
        var chains: [[Int]] = []                                   // per ring edge: [a, p1 ... pN, b]
        for (e0, (a, b)) in zip(ring.edges, ends) {
            var chain = [a], prev = 0.0
            var cur = e0
            for u in us {
                let f = (u - prev) / (1 - prev)
                let (o, _) = vertices(ofEdge: cur)
                guard let m = splitEdge(cur, at: o == chain.last! ? f : 1 - f) else { self = backup; return nil }
                chain.append(m); prev = u
                guard let rest = self.edge(between: m, b) else { self = backup; return nil }
                cur = rest
            }
            chain.append(b); chains.append(chain)
        }
        var newEdges: [Int] = []
        let pairs = ring.isClosed ? chains.count : chains.count - 1
        for i in 0..<pairs {
            let c0 = chains[i], c1 = chains[(i + 1) % chains.count]
            for j in 1...cuts {
                guard let f = Set(faces(around: c0[j])).intersection(faces(around: c1[j])).first(where: { valence(ofFace: $0) > 3 }),
                      splitFace(f, between: c0[j], c1[j]) != nil, let ne = edge(between: c0[j], c1[j]) else { self = backup; return nil }
                newEdges.append(ne)
            }
        }
        return selection(.edge, newEdges)
    }

    // MARK: edge slide

    @discardableResult
    mutating func slideEdges(_ sel: HESelection, factor: Double) -> HESelection {
        let es = Set(sel.edges.filter { isAlive(edge: $0) })
        guard !es.isEmpty, factor != 0 else { return sel }
        var moved: [Int: Vec3] = [:]
        for path in vertexPaths(of: es) {
            for i in 0..<path.verts.count {
                let v = path.verts[i]
                guard moved[v] == nil else { continue }
                let (a, b) = i + 1 < path.verts.count ? (v, path.verts[i + 1]) : (path.verts[i - 1], v)
                var target: Int?
                if factor > 0 {
                    if let h = halfedge(from: a, to: b) { target = (a == v) ? heOrigin[hePrev[h]] : heOrigin[heNext[heNext[h]]] }
                    else if let t = halfedge(from: b, to: a) { target = (a == v) ? heOrigin[heNext[heNext[t]]] : heOrigin[hePrev[t]] }
                } else {
                    if let t = halfedge(from: b, to: a) { target = (a == v) ? heOrigin[heNext[heNext[t]]] : heOrigin[hePrev[t]] }
                    else if let h = halfedge(from: a, to: b) { target = (a == v) ? heOrigin[hePrev[h]] : heOrigin[heNext[heNext[h]]] }
                }
                // the side: "left" of a -> b is the face owning halfedge a -> b; its edge at v goes to `target`
                if let t = target, t != v, edge(between: v, t) != nil, !es.contains(edge(between: v, t)!) {
                    moved[v] = positions[v] + (positions[t] - positions[v]) * abs(factor)
                }
            }
        }
        let flat = flatFaces(around: moved.keys)
        for (v, p) in moved { positions[v] = p }
        reshade(flat)
        return sel
    }

    /// Ordered vertex walks through a set of edges (open paths, or cycles with the first vertex not repeated).
    func vertexPaths(of es: Set<Int>) -> [(verts: [Int], closed: Bool)] {
        var adj: [Int: [Int]] = [:]
        for e in es { let (a, b) = vertices(ofEdge: e); adj[a, default: []].append(b); adj[b, default: []].append(a) }
        var seen = Set<Int>(), out: [(verts: [Int], closed: Bool)] = []
        func walk(from s: Int) -> ([Int], Bool) {
            var path = [s], cur = s, prev = -1, closed = false
            seen.insert(s)
            while let nx = adj[cur]?.first(where: { $0 != prev && !(seen.contains($0) && $0 != s) }) {
                if nx == s { closed = true; break }
                path.append(nx); seen.insert(nx); prev = cur; cur = nx
            }
            return (path, closed)
        }
        for s in adj.keys.sorted() where adj[s]!.count == 1 && !seen.contains(s) { let (p, c) = walk(from: s); out.append((p, c)) }
        for s in adj.keys.sorted() where !seen.contains(s) { let (p, c) = walk(from: s); out.append((p, c)) }
        return out
    }
}
