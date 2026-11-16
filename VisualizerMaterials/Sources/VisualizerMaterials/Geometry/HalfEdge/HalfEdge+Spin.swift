//  HalfEdge+Spin — spin (lathe) a selection round an axis, and build solids of revolution from a profile.
//
//  `spin`: the selected faces (a region: their border edges sweep a wall of quads per step, the faces end up
//  on the last copy, like a multi-step extrude along an arc) or boundary edges (a strip of quads out of the
//  open edge) are rotated `angle` radians about the axis in `steps` steps. Vertices on the axis stay single
//  (their quads become triangles). A sweep of a full turn would close on edges that already have a face, so
//  it is refused (nil): `|angle|` must stay under a full turn minus a step; build full solids with `lathe`.
//  Winding follows the selection (its neighbours decide it), so the sign of `angle` chooses which side the
//  swept surface faces. Returns the end faces / end edges.
//  `lathe(profile:)`: a new mesh from a polyline spun about the axis; `closed` joins the last point to the
//  first (a closed profile off the axis is a torus). A full turn closes into itself; a profile whose ends sit
//  on the axis is a closed solid. Closed shells are wound outward.

import simd

public extension HalfEdgeMesh {

    static func rotate(_ p: Vec3, about o: Vec3, axis a: Vec3, angle: Double) -> Vec3 {
        let q = p - o, c = cos(angle), s = sin(angle)
        return o + q * c + cross3(a, q) * s + a * (dot3(a, q) * (1 - c))
    }

    @discardableResult
    mutating func spin(_ sel: HESelection, axisOrigin: Vec3, axisDirection: Vec3, angle: Double, steps: Int) -> HESelection? {
        let axis = normalize3(axisDirection)
        guard len3(axis) > 0, steps >= 1, abs(angle) < 2 * .pi * (1 - 0.5 / Double(steps)) else { return nil }
        let step = angle / Double(steps)
        let backup = self
        let faceMode = sel.mode == .face
        let region = faceMode ? selectedFaces(sel) : []
        let inRegion = Set(region)
        // border halfedges and the vertices that need every ring
        var border: [Int] = []
        if faceMode {
            for f in region { for h in halfedges(of: f) where heTwin[h] < 0 || !inRegion.contains(heFace[heTwin[h]]) { border.append(h) } }
        } else {
            border = sel.edges.filter { isAlive(edge: $0) && isBoundary(edge: $0) }.sorted().map { edgeHE[$0] }
        }
        guard !border.isEmpty else { return nil }
        var verts = Set<Int>()
        for h in border { verts.insert(heOrigin[h]); verts.insert(heOrigin[heNext[h]]) }
        let moving = faceMode ? Set(region.flatMap { vertices(of: $0) }) : verts
        func onAxis(_ v: Int) -> Bool { len3(cross3(axis, positions[v] - axisOrigin)) < 1e-9 }
        var ring: [Int: [Int]] = [:]                                // v -> [v_0 ... v_N]
        for v in moving.sorted() {
            if onAxis(v) { ring[v] = [Int](repeating: v, count: steps + 1); continue }
            var r = [v]
            for s in 1...steps { r.append(addVertex(HalfEdgeMesh.rotate(positions[v], about: axisOrigin, axis: axis, angle: step * Double(s)))) }
            ring[v] = r
        }
        var adds: [EditPoly] = []
        if faceMode { for f in region { var p = editPoly(of: f); p.v = p.v.map { ring[$0]![steps] }; adds.append(p) } }
        for h in border {
            let a = ring[heOrigin[h]]!, b = ring[heOrigin[heNext[h]]]!
            for s in 0..<steps {
                let q = faceMode ? [a[s], b[s], b[s + 1], a[s + 1]] : [b[s], a[s], a[s + 1], b[s + 1]]
                let c = HalfEdgeMesh.dedupeCyclic(q)
                if c.count >= 3 { adds.append(EditPoly(v: c)) }
            }
        }
        guard let ids = rebuild(removing: region, adding: adds) else { self = backup; return nil }
        if faceMode { return selection(.face, ids.prefix(region.count)) }
        return selection(.edge, border.compactMap { h in edge(between: ring[heOrigin[h]]![steps], ring[heOrigin[heNext[h]]]![steps]) })
    }

    /// A solid (or surface) of revolution. nil when the profile is degenerate.
    static func lathe(profile: [Vec3], closed: Bool = false, axisOrigin: Vec3 = .zero, axisDirection: Vec3 = Vec3(0, 1, 0),
                      angle: Double = 2 * .pi, steps: Int = 32) -> HalfEdgeMesh? {
        let axis = normalize3(axisDirection)
        guard profile.count >= 2, steps >= 3, len3(axis) > 0 else { return nil }
        let full = abs(angle) >= 2 * .pi - 1e-9
        let rings = full ? steps : steps + 1
        var m = HalfEdgeMesh()
        var id = [[Int]](repeating: [], count: rings)
        for s in 0..<rings {
            for p in profile {
                if len3(cross3(axis, p - axisOrigin)) < 1e-9 {
                    id[s].append(s == 0 ? m.addVertex(p) : id[0][id[s].count])
                } else {
                    id[s].append(m.addVertex(rotate(p, about: axisOrigin, axis: axis, angle: angle * Double(s) / Double(steps))))
                }
            }
        }
        let n = profile.count, segs = closed ? n : n - 1
        for s in 0..<(full ? steps : steps) {
            let s2 = (s + 1) % rings
            for i in 0..<segs {
                let j = (i + 1) % n
                let c = dedupeCyclic([id[s][i], id[s][j], id[s2][j], id[s2][i]])
                if c.count >= 3, m.addFace(c) == nil { return nil }
            }
        }
        m.orientOutward()
        return m
    }
}
