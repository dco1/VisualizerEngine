//  Edge loops and edge rings (Alt-click / Ctrl-Alt-click in a modeller).
//   * loop: from an edge, keep going straight through every interior vertex of valence 4 (the edge opposite
//     in the vertex's rotation); stops at poles (valence != 4) and non-manifold vertices. A boundary edge
//     follows the boundary instead. Closed when it returns to its start.
//   * ring: the parallel edges across a strip of quads: in each quad the edge opposite the one entered,
//     crossing into the neighbour through it. Stops at the first non-quad face or boundary.
//  Edges are returned in walking order, the start edge once.

public struct HEEdgeWalk: Sendable, Equatable {
    public var edges: [Int]
    public var isClosed: Bool
}

public extension HalfEdgeMesh {

    func edgeLoop(from e: Int) -> HEEdgeWalk {
        guard isAlive(edge: e) else { return HEEdgeWalk(edges: [], isClosed: false) }
        let (a, b) = vertices(ofEdge: e)
        var seen: Set<Int> = [e]
        var forward = [e], closed = false
        var state: (Int, Int)? = (e, b)
        while let (cur, head) = state, let nxt = loopStep(edge: cur, head: head) {
            if nxt.edge == e { closed = true; break }
            if !seen.insert(nxt.edge).inserted { break }
            forward.append(nxt.edge); state = (nxt.edge, nxt.head)
        }
        if closed { return HEEdgeWalk(edges: forward, isClosed: true) }
        var back: [Int] = []
        state = (e, a)
        while let (cur, head) = state, let nxt = loopStep(edge: cur, head: head), seen.insert(nxt.edge).inserted {
            back.append(nxt.edge); state = (nxt.edge, nxt.head)
        }
        return HEEdgeWalk(edges: back.reversed() + forward, isClosed: false)
    }

    /// Having walked `edge` and arrived at `head`, the edge that continues straight and the vertex it leads to.
    private func loopStep(edge: Int, head: Int) -> (edge: Int, head: Int)? {
        let ring = edges(around: head)
        guard let i = ring.firstIndex(of: edge) else { return nil }
        var next: Int
        if isBoundary(edge: edge) {
            let bnd = ring.filter { isBoundary(edge: $0) && $0 != edge }
            guard bnd.count == 1 else { return nil }
            next = bnd[0]
        } else {
            guard ring.count == 4, !isBoundary(vertex: head) else { return nil }
            next = ring[(i + 2) % 4]
        }
        let (p, q) = vertices(ofEdge: next)
        return (next, p == head ? q : p)
    }

    func edgeRing(from e: Int) -> HEEdgeWalk {
        guard isAlive(edge: e) else { return HEEdgeWalk(edges: [], isClosed: false) }
        let h0 = edgeHE[e]
        var seen: Set<Int> = [e]
        var forward = [e], closed = false
        ringWalk(from: h0, e, &seen, &forward, &closed)
        if closed { return HEEdgeWalk(edges: forward, isClosed: true) }
        var back: [Int] = []
        if heTwin[h0] >= 0 { var c = false; ringWalk(from: heTwin[h0], e, &seen, &back, &c) }
        return HEEdgeWalk(edges: back.reversed() + forward, isClosed: false)
    }

    private func ringWalk(from start: Int, _ e0: Int, _ seen: inout Set<Int>, _ out: inout [Int], _ closed: inout Bool) {
        var h = start
        while heNext[heNext[heNext[heNext[h]]]] == h {              // current face is a quad
            let o = heNext[heNext[h]]
            if heEdge[o] == e0 { closed = true; return }
            if !seen.insert(heEdge[o]).inserted { return }
            out.append(heEdge[o])
            let t = heTwin[o]
            if t < 0 { return }
            h = t
        }
    }

    /// The faces a ring runs through (each quad once), for face-loop style selections.
    func faces(alongRing ring: HEEdgeWalk) -> [Int] {
        var out: [Int] = [], seen = Set<Int>()
        for e in ring.edges { for f in faces(ofEdge: e) where valence(ofFace: f) == 4 && seen.insert(f).inserted { out.append(f) } }
        return out
    }
}
