//  HalfEdge+Dissolve — dissolve vertices, edges or faces (remove them and heal the mesh, instead of leaving a
//  hole), and merge vertices (at the centre, at the first or last, at a position, or by distance).
//
//  Dissolve faces: each connected region of selected faces becomes one n-gon over its outer loop (nil, nothing
//  changed, when a region is not a disk or touches the mesh's own boundary). Edges: interior edges join their two
//  faces (`dissolveEdge`); with `cleanVertices` the valence-2 vertices that leaves are removed too, so a
//  dissolved loop cut returns the original faces. Vertices: a valence-2 vertex just leaves its two faces'
//  polygons (the edges join); any other interior vertex merges its fan into one n-gon. Every step is skipped
//  rather than half-done: the returned selection holds what changed.
//  Merge: the chosen vertices collapse onto one (or each cluster onto its own, by distance); faces that shrink
//  below three corners vanish; refused (nil) if two faces would end up on the same directed edge.

import simd

public enum HEMergeTarget: Sendable { case center, first, last, position(Vec3) }

public extension HalfEdgeMesh {

    // MARK: dissolve

    @discardableResult
    mutating func dissolve(_ sel: HESelection, cleanVertices: Bool = true) -> HESelection? {
        switch sel.mode {
        case .face: return dissolveFaces(selectedFaces(sel))
        case .edge:
            var out = Set<Int>()
            for e in sel.edges.sorted() where isAlive(edge: e) { if let f = dissolveEdge(e) { out.insert(f) } }
            guard !out.isEmpty else { return nil }
            if cleanVertices {
                for v in sel.vertices.sorted() where vertexHE[v] >= 0 && valence(of: v) == 2 { _ = dissolveVertex(v) }
            }
            return selection(.face, out.filter { isAlive(face: $0) })
        case .vertex:
            var out = Set<Int>()
            for v in sel.vertices.sorted() where vertexHE[v] >= 0 { if let fs = dissolveVertex(v) { out.formUnion(fs) } }
            return out.isEmpty ? nil : selection(.face, out.filter { isAlive(face: $0) })
        }
    }

    private mutating func dissolveFaces(_ region: [Int]) -> HESelection? {
        guard !region.isEmpty else { return nil }
        var seen = Set<Int>(), out: [Int] = []
        let backup = self
        let inRegion = Set(region)
        for f in region where !seen.contains(f) {
            var comp = [f], stack = [f]; seen.insert(f)
            while let x = stack.popLast() { for n in neighbours(ofFace: x) where inRegion.contains(n) && seen.insert(n).inserted { comp.append(n); stack.append(n) } }
            let rim = Set(boundaryEdges(ofFaces: Set(comp)))
            guard rim.allSatisfy({ !isBoundary(edge: $0) }) else { self = backup; return nil }
            deleteFaces(comp); repairVertexLinks()
            let loops = boundaryLoops(within: rim).filter { l in l.indices.allSatisfy { edge(between: l[$0], l[($0 + 1) % l.count]).map(rim.contains) ?? false } }
            guard loops.count == 1, let ids = fill(loop: loops[0], mode: .ngon) else { self = backup; return nil }
            out += ids
        }
        return selection(.face, out)
    }

    /// Remove a vertex: valence 2 -> its corner leaves both faces; otherwise its fan becomes one polygon.
    /// Returns the faces it touched afterwards, or nil (unchanged).
    @discardableResult
    mutating func dissolveVertex(_ v: Int) -> [Int]? {
        guard vertexHE[v] >= 0 else { return nil }
        let hs = outgoing(of: v), fs = hs.map { heFace[$0] }
        if valence(of: v) == 2 {
            var adds: [EditPoly] = []
            for f in Set(fs).sorted() {
                var p = editPoly(of: f)
                let i = p.v.firstIndex(of: v)!
                p.v.remove(at: i); p.uv.remove(at: i); p.n!.remove(at: i)
                guard p.v.count >= 3 else { return nil }
                adds.append(p)
            }
            return rebuild(removing: Array(Set(fs)), adding: adds).map { _ in Array(Set(fs)) }
        }
        guard !isBoundary(vertex: v) else { return nil }
        var paths: [Int: [Int]] = [:]
        for h in hs {
            let vs = vertices(of: heFace[h]), i = vs.firstIndex(of: v)!
            let rot = Array(vs[(i + 1)...] + vs[..<i])                // the face's loop without v, starting after it
            paths[rot.first!] = rot
        }
        var poly = paths.values.first!, used = 1
        while used < paths.count, let nxt = paths[poly.last!] {
            if poly.count > 1 && nxt.first! == poly.first! { break }
            poly += nxt.dropFirst(); used += 1
        }
        if poly.first == poly.last { poly.removeLast() }
        guard Set(poly).count == poly.count, poly.count >= 3 else { return nil }
        return rebuild(removing: Array(Set(fs)), adding: [EditPoly(v: poly)])
    }

    // MARK: merge

    /// Collapse the selection's vertices onto one.
    @discardableResult
    mutating func merge(_ sel: HESelection, at target: HEMergeTarget = .center) -> HESelection? {
        let vs = sel.vertices.filter { vertexHE[$0] >= 0 }.sorted()
        guard vs.count >= 2 else { return nil }
        let to: Vec3
        let keep: Int
        switch target {
        case .center: to = vs.reduce(Vec3.zero) { $0 + positions[$1] } / Double(vs.count); keep = vs[0]
        case .first: to = positions[vs[0]]; keep = vs[0]
        case .last: to = positions[vs.last!]; keep = vs.last!
        case .position(let p): to = p; keep = vs[0]
        }
        var map: [Int: Int] = [:]
        for v in vs { map[v] = keep }
        guard collapse(map, positions: [keep: to]) else { return nil }
        return selection(.vertex, [keep])
    }

    /// Weld vertices closer than `distance` (all of them, or the selection's), each cluster to its centre.
    @discardableResult
    mutating func mergeByDistance(_ sel: HESelection? = nil, distance: Double) -> HESelection? {
        let vs = (sel?.vertices.filter { vertexHE[$0] >= 0 }.sorted()) ?? vertices
        guard vs.count >= 2, distance >= 0 else { return nil }
        let cell = Swift.max(distance, 1e-12)
        struct K: Hashable { var x, y, z: Int }
        func key(_ p: Vec3) -> K { K(x: Int((p.x / cell).rounded(.down)), y: Int((p.y / cell).rounded(.down)), z: Int((p.z / cell).rounded(.down))) }
        var grid: [K: [Int]] = [:]
        for v in vs { grid[key(positions[v]), default: []].append(v) }
        var parent = Dictionary(uniqueKeysWithValues: vs.map { ($0, $0) })
        func find(_ x: Int) -> Int { var r = x; while parent[r]! != r { r = parent[r]! }; var c = x; while parent[c]! != r { let n = parent[c]!; parent[c] = r; c = n }; return r }
        for v in vs {
            let k = key(positions[v])
            for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
                for w in grid[K(x: k.x + dx, y: k.y + dy, z: k.z + dz)] ?? [] where w > v && dist3(positions[v], positions[w]) <= distance {
                    let a = find(v), b = find(w); if a != b { parent[Swift.max(a, b)] = Swift.min(a, b) }
                }
            } } }
        }
        var clusters: [Int: [Int]] = [:]
        for v in vs { clusters[find(v), default: []].append(v) }
        var map: [Int: Int] = [:], pos: [Int: Vec3] = [:]
        for (root, members) in clusters where members.count > 1 {
            for m in members { map[m] = root }
            pos[root] = members.reduce(Vec3.zero) { $0 + positions[$1] } / Double(members.count)
        }
        guard !map.isEmpty else { return sel ?? selectAll(.vertex) }
        guard collapse(map, positions: pos) else { return nil }
        return selection(.vertex, pos.keys)
    }

    /// Re-add every face round the mapped vertices on the mapped corners; degenerate faces vanish.
    mutating func collapse(_ map: [Int: Int], positions newPos: [Int: Vec3]) -> Bool {
        let affected = Array(Set(map.keys.flatMap { faces(around: $0) })).sorted()
        var adds: [EditPoly] = [], gone: [Int] = []
        for f in affected {
            var p = editPoly(of: f)
            p.v = p.v.map { map[$0] ?? $0 }
            var i = 0                                                // drop consecutive repeats (cyclically)
            while p.v.count > 1 && i < p.v.count {
                let j = (i + 1) % p.v.count
                if p.v[i] == p.v[j] { p.v.remove(at: i); p.uv.remove(at: i); p.n!.remove(at: i); if i >= p.v.count { i = 0 } } else { i += 1 }
            }
            if p.v.count < 3 { continue }
            guard Set(p.v).count == p.v.count else { return false }
            adds.append(p); gone.append(f)
        }
        let backup = self
        for (v, p) in newPos { positions[v] = p }
        // faces that vanish keep their slot dead; the rest are re-added in place
        guard rebuild(removing: affected, adding: adds) != nil else { self = backup; return false }
        for v in map.keys where map[v] != v { vertexHE[v] = -1 }
        return true
    }
}
