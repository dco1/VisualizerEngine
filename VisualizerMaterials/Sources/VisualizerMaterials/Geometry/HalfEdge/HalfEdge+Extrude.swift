//  HalfEdge+Extrude — extrude faces (a region with shared walls, or each face on its own), boundary edges and
//  vertices; push/pull faces along their normals; translate a selection.
//
//  Faces: the region's vertices are duplicated at the offset, the region's faces are re-added on the copies
//  (same face ids), and one wall quad stands on every region-boundary edge (so a region shares its walls, and
//  an interior edge gets none). The original ring of vertices stays with the surrounding mesh: extruding is a
//  pop-up, the base is not left behind (the result of extruding a closed mesh's face is still closed).
//  `individual: true` extrudes each face separately (a wall on every one of its edges).
//  Edges: only boundary edges (a strip of quads out of an open mesh's rim; offset in the face's plane, away
//  from it, unless `vector` is given); interior edges are skipped, there is no loose-edge geometry here.
//  Vertices: a lone interior vertex becomes a spike: a small cap polygon is cut round it (`fraction` of each
//  incident edge) and that cap is extruded. Boundary vertices are skipped.
//  Push/pull is the same offset without any new topology: it moves the faces' vertices (volume changes by
//  area × distance for a flat face). Offsets run on the GPU above `HEEdit.gpuThreshold` vertices.

import simd

public extension HalfEdgeMesh {

    // MARK: faces

    /// Extrude the selection's faces by `distance` along their normals (or by `vector`). Returns the moved faces.
    @discardableResult
    mutating func extrude(faces sel: HESelection, distance: Double = 0, vector: Vec3? = nil,
                          normals: HENormalMode = .averaged, individual: Bool = false) -> HESelection? {
        let region = selectedFaces(sel)
        guard !region.isEmpty else { return nil }
        let backup = self
        let groups: [[Int]] = individual ? region.map { [$0] } : [region]
        var out: [Int] = []
        for g in groups {
            guard let ids = extrudeRegion(g, distance: distance, vector: vector, normals: normals) else { self = backup; return nil }
            out += ids
        }
        return selection(.face, out)
    }

    private mutating func extrudeRegion(_ region: [Int], distance: Double, vector: Vec3?, normals: HENormalMode) -> [Int]? {
        let inRegion = Set(region)
        let entries = areaVectorsByVertex(region)
        let offs = vector.map { v in entries.map { _ in v } } ?? normalOffsets(entries, distance: distance, mode: normals, useGPU: nil)
        var dup: [Int: Int] = [:]
        for (i, e) in entries.enumerated() { dup[e.vertex] = addVertex(positions[e.vertex] + offs[i]) }
        var adds: [EditPoly] = []
        for f in region { var p = editPoly(of: f); p.v = p.v.map { dup[$0]! }; adds.append(p) }
        for f in region { for h in halfedges(of: f) {
            let t = heTwin[h]
            if t >= 0 && inRegion.contains(heFace[t]) { continue }
            let a = heOrigin[h], b = heOrigin[heNext[h]], ua = heUV[h], ub = heUV[heNext[h]]
            adds.append(EditPoly(v: [a, b, dup[b]!, dup[a]!], uv: [ua, ub, ub, ua]))
        } }
        guard let ids = rebuild(removing: region, adding: adds) else { return nil }
        return Array(ids.prefix(region.count))
    }

    // MARK: edges

    /// Extrude the selection's boundary edges into a strip of quads. Returns the new outer edges.
    @discardableResult
    mutating func extrude(edges sel: HESelection, distance: Double = 0, vector: Vec3? = nil) -> HESelection? {
        let es = sel.edges.filter { isAlive(edge: $0) && isBoundary(edge: $0) }.sorted()
        guard !es.isEmpty else { return nil }
        let backup = self
        var push: [Int: Vec3] = [:]
        for e in es {
            let h = edgeHE[e], a = heOrigin[h], b = heOrigin[heNext[h]]
            let out = vector ?? normalize3(cross3(positions[b] - positions[a], faceNormal(heFace[h]))) * distance
            push[a, default: .zero] += out; push[b, default: .zero] += out
        }
        var dup: [Int: Int] = [:]
        for v in push.keys.sorted() {
            var o = push[v]!
            if vector == nil { let l = len3(o); if l > 1e-12 { o = o / l * abs(distance) } }
            dup[v] = addVertex(positions[v] + o)
        }
        var adds: [EditPoly] = []
        for e in es {
            let h = edgeHE[e], a = heOrigin[h], b = heOrigin[heNext[h]], ua = heUV[h], ub = heUV[heNext[h]]
            adds.append(EditPoly(v: [b, a, dup[a]!, dup[b]!], uv: [ub, ua, ua, ub]))
        }
        guard rebuild(removing: [], adding: adds) != nil else { self = backup; return nil }
        return selection(.edge, es.compactMap { e in
            let h = backup.edgeHE[e]; return edge(between: dup[backup.heOrigin[h]]!, dup[backup.heOrigin[backup.heNext[h]]]!)
        })
    }

    // MARK: vertices

    /// Pull each selected interior vertex out as a spike `distance` high (along its normal). Returns the spikes' tip faces.
    @discardableResult
    mutating func extrude(vertices sel: HESelection, distance: Double, fraction: Double = 0.25) -> HESelection? {
        let backup = self
        var tips: [Int] = []
        for v in sel.vertices.sorted() {
            guard vertexHE[v] >= 0, let cap = capVertex(v, fraction: fraction),
                  let ids = extrudeRegion([cap], distance: distance, vector: nil, normals: .averaged) else { continue }
            tips += ids
        }
        guard !tips.isEmpty else { self = backup; return nil }
        return selection(.face, tips)
    }

    /// Cut a small polygon round interior vertex `v` (corner of each incident face replaced by two points on its
    /// edges); returns the cap face. nil for boundary vertices or valence < 3.
    mutating func capVertex(_ v: Int, fraction: Double) -> Int? {
        let hs = outgoing(of: v)
        guard hs.count >= 3, !isBoundary(vertex: v) else { return nil }
        let k = hs.count
        let u = hs.map { addVertex(positions[v] + (positions[heOrigin[heNext[$0]]] - positions[v]) * fraction) }
        var adds: [EditPoly] = []
        for (i, h) in hs.enumerated() {
            let f = heFace[h]
            var p = editPoly(of: f)
            let at = halfedges(of: f).firstIndex(of: h)!
            p.v.replaceSubrange(at...at, with: [u[(i + k - 1) % k], u[i]])
            p.uv.insert(p.uv[at], at: at); p.n = p.n.map { var n = $0; n.insert(n[at], at: at); return n }
            adds.append(p)
        }
        adds.append(EditPoly(v: u.reversed()))
        guard let ids = rebuild(removing: hs.map { heFace[$0] }, adding: adds) else { return nil }
        return ids.last
    }

    // MARK: push / pull and translate

    /// Move the selected faces' vertices along their normals by `distance` (positive = outward). No topology
    /// change. With a vertex-mode selection that holds no whole face, every face round the vertex counts.
    @discardableResult
    mutating func pushPull(_ sel: HESelection, distance: Double, normals: HENormalMode = .averaged, useGPU: Bool? = nil) -> HESelection {
        var fs = selectedFaces(sel)
        if fs.isEmpty { fs = Array(Set(sel.vertices.flatMap { faces(around: $0) })).sorted() }
        let entries = areaVectorsByVertex(fs)
        let offs = normalOffsets(entries, distance: distance, mode: normals, useGPU: useGPU)
        let flat = flatFaces(around: entries.map { $0.vertex })
        for (i, e) in entries.enumerated() { positions[e.vertex] += offs[i] }
        reshade(flat)
        return sel
    }

    /// Move everything the selection touches by `vector`.
    @discardableResult
    mutating func translate(_ sel: HESelection, by vector: Vec3) -> HESelection {
        let flat = flatFaces(around: sel.vertices)
        for v in sel.vertices { positions[v] += vector }
        reshade(flat)
        return sel
    }
}
