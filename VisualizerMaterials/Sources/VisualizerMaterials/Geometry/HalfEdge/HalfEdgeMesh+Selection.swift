//  Selection sets for HalfEdgeMesh: vertices, edges and faces, kept consistent with each other, with
//  grow / shrink / invert and region boundary queries. A selection is a value (a set of ids per kind) plus
//  the mode that is authoritative; the other two sets are derived from it ("flushed"), the way a modeller's
//  vertex / edge / face modes behave:
//    vertex mode: edges = both ends selected, faces = every corner selected
//    edge mode:   verts = their ends, faces = every edge selected
//    face mode:   edges and verts = those of the faces
//  Ids refer to the mesh they were made on; edits that renumber ids (compacted()) invalidate them.

public struct HESelection: Sendable, Equatable {
    public enum Mode: Sendable { case vertex, edge, face }
    public var mode: Mode
    public var vertices: Set<Int> = []
    public var edges: Set<Int> = []
    public var faces: Set<Int> = []
    public init(mode: Mode) { self.mode = mode }
    public var isEmpty: Bool { vertices.isEmpty && edges.isEmpty && faces.isEmpty }
}

public extension HalfEdgeMesh {

    /// A flushed selection from the authoritative ids of `mode`.
    func selection(_ mode: HESelection.Mode, _ ids: some Sequence<Int>) -> HESelection {
        var s = HESelection(mode: mode)
        switch mode {
        case .vertex: s.vertices = Set(ids).filter { $0 >= 0 && $0 < positions.count && vertexHE[$0] >= 0 }
        case .edge:   s.edges = Set(ids).filter { isAlive(edge: $0) }
        case .face:   s.faces = Set(ids).filter { isAlive(face: $0) }
        }
        return flushed(s)
    }
    func selectAll(_ mode: HESelection.Mode) -> HESelection {
        selection(mode, mode == .vertex ? vertices : mode == .edge ? edges : faces)
    }

    /// Recompute the two derived sets from the authoritative one.
    func flushed(_ sel: HESelection) -> HESelection {
        var s = sel
        switch sel.mode {
        case .vertex:
            s.edges = Set(edges.filter { let v = vertices(ofEdge: $0); return sel.vertices.contains(v.0) && sel.vertices.contains(v.1) })
            s.faces = Set(faces.filter { vertices(of: $0).allSatisfy(sel.vertices.contains) })
        case .edge:
            s.vertices = Set(sel.edges.flatMap { e -> [Int] in let v = vertices(ofEdge: e); return [v.0, v.1] })
            s.faces = Set(faces.filter { edges(of: $0).allSatisfy(sel.edges.contains) })
        case .face:
            s.edges = Set(sel.faces.flatMap { edges(of: $0) })
            s.vertices = Set(sel.faces.flatMap { vertices(of: $0) })
        }
        return s
    }

    /// One ring bigger: vertices gain their neighbours, edges those touching a selected vertex, faces those
    /// sharing a vertex with the selection.
    func grow(_ sel: HESelection) -> HESelection {
        var s = HESelection(mode: sel.mode)
        switch sel.mode {
        case .vertex: s.vertices = sel.vertices.union(sel.vertices.flatMap { neighbours(of: $0) })
        case .edge:   s.edges = sel.edges.union(sel.vertices.flatMap { edges(around: $0) })
        case .face:   s.faces = sel.faces.union(sel.vertices.flatMap { faces(around: $0) })
        }
        return flushed(s)
    }

    /// One ring smaller (the inverse of `grow` on a convex blob): drops whatever touches the unselected.
    /// Mesh boundaries count as the selection's edge only where an unselected element is there.
    func shrink(_ sel: HESelection) -> HESelection {
        var s = HESelection(mode: sel.mode)
        switch sel.mode {
        case .vertex: s.vertices = sel.vertices.filter { v in neighbours(of: v).allSatisfy(sel.vertices.contains) }
        case .edge:
            s.edges = sel.edges.filter { e in
                let v = vertices(ofEdge: e)
                return [v.0, v.1].allSatisfy { edges(around: $0).allSatisfy(sel.edges.contains) }
            }
        case .face: s.faces = sel.faces.filter { f in vertices(of: f).allSatisfy { faces(around: $0).allSatisfy(sel.faces.contains) } }
        }
        return flushed(s)
    }

    func inverted(_ sel: HESelection) -> HESelection {
        switch sel.mode {
        case .vertex: return selection(.vertex, Set(vertices).subtracting(sel.vertices))
        case .edge:   return selection(.edge, Set(edges).subtracting(sel.edges))
        case .face:   return selection(.face, Set(faces).subtracting(sel.faces))
        }
    }

    /// Edges on the border of a face region: one side in `region`, the other outside it or absent.
    func boundaryEdges(ofFaces region: Set<Int>) -> [Int] {
        edges.filter { e in faces(ofEdge: e).filter(region.contains).count == 1 }
    }
    /// Every face reachable from `f` through shared edges, stopping at `stop` edges (e.g. seams).
    func linkedFaces(from f: Int, stoppingAt stop: Set<Int> = []) -> Set<Int> {
        var seen: Set<Int> = [f], stack = [f]
        while let x = stack.popLast() {
            for h in halfedges(of: x) where heTwin[h] >= 0 && !stop.contains(heEdge[h]) {
                let n = heFace[heTwin[h]]
                if seen.insert(n).inserted { stack.append(n) }
            }
        }
        return seen
    }
}
