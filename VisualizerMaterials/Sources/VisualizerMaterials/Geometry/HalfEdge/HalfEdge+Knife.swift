//  HalfEdge+Knife — cut the mesh with a plane: new vertices where the plane crosses an edge, new edges where it
//  crosses a face, optionally split into two capped halves.
//
//  Edges that cross the plane (endpoints strictly on opposite sides, beyond `tolerance`) are split at the
//  crossing; vertices within `tolerance` of the plane count as on it. A face is cut between its on-plane corners
//  when it has corners strictly on both sides (more than two on-plane corners are paired along the cut line,
//  so a concave face is cut in several pieces). `faces` restricts the knife to some faces; their neighbours only
//  gain the split vertices. Returns the cut edges (every edge on the plane between faces that lie on opposite
//  sides) as an edge selection.
//  `split: true` then separates the two sides along the cut: the cut vertices are duplicated for the back
//  faces and each resulting hole is capped with `capMode` (default one n-gon), so each half of a closed mesh is
//  closed; the result is the cap faces. Not guaranteed: a face lying in the plane has no side (it stays with the front).

import simd

public extension HalfEdgeMesh {

    @discardableResult
    mutating func knife(planeOrigin o: Vec3, normal: Vec3, split: Bool = false, faces restrict: HESelection? = nil,
                        tolerance eps: Double = 1e-9, capMode: HEFillMode = .ngon) -> HESelection? {
        let n = normalize3(normal)
        guard len3(n) > 0 else { return nil }
        let backup = self
        let candidates = restrict.map { selectedFaces($0) } ?? faces
        func side(_ p: Vec3) -> Double { dot3(p - o, n) }
        // 1. split crossing edges
        var crossing = Set<Int>()
        for f in candidates { for e in edges(of: f) {
            let (a, b) = vertices(ofEdge: e)
            let da = side(positions[a]), db = side(positions[b])
            if (da > eps && db < -eps) || (da < -eps && db > eps) { crossing.insert(e) }
        } }
        var onPlane = Set<Int>()
        for e in crossing.sorted() {
            let (a, b) = vertices(ofEdge: e)
            let da = side(positions[a]), db = side(positions[b])
            guard let m = splitEdge(e, at: da / (da - db)) else { self = backup; return nil }
            positions[m] -= n * side(positions[m])
            onPlane.insert(m)
        }
        for v in vertices where abs(side(positions[v])) <= eps { onPlane.insert(v) }
        // 2. cut the faces
        var cutAny = !crossing.isEmpty
        for f0 in candidates {
            let vs = vertices(of: f0)
            let ds = vs.map { side(positions[$0]) }
            guard ds.contains(where: { $0 > eps }), ds.contains(where: { $0 < -eps }) else { continue }
            var on = vs.filter { onPlane.contains($0) }
            guard on.count >= 2 else { continue }
            let line = normalize3(cross3(n, faceNormal(f0)))
            on.sort { dot3(positions[$0], line) < dot3(positions[$1], line) }
            var i = 0
            while i + 1 < on.count {
                let a = on[i], b = on[i + 1]; i += 2
                guard let f = Set(faces(around: a)).intersection(faces(around: b)).first(where: { valence(ofFace: $0) > 3 }) else { continue }
                if splitFace(f, between: a, b) != nil { cutAny = true }
            }
        }
        // 3. the cut edges
        func sideOf(_ f: Int) -> Int { let d = side(faceCentroid(f)); return d > eps ? 1 : (d < -eps ? -1 : 0) }
        var cut: [Int] = []
        for e in edges {
            let (a, b) = vertices(ofEdge: e)
            guard onPlane.contains(a), onPlane.contains(b) else { continue }
            let fs = faces(ofEdge: e)
            if fs.count == 2 && sideOf(fs[0]) * sideOf(fs[1]) < 0 { cut.append(e) }
        }
        guard cutAny || !cut.isEmpty else { self = backup; return nil }
        guard split else { return selection(.edge, cut) }
        // 4. separate and cap
        let cutVerts = Set(cut.flatMap { e -> [Int] in let v = vertices(ofEdge: e); return [v.0, v.1] })
        guard !cutVerts.isEmpty else { self = backup; return nil }
        var dup: [Int: Int] = [:]
        for v in cutVerts.sorted() { dup[v] = addVertex(positions[v]) }
        let back = faces.filter { sideOf($0) < 0 }
        var adds: [EditPoly] = []
        for f in back { var p = editPoly(of: f); p.v = p.v.map { dup[$0] ?? $0 }; adds.append(p) }
        guard rebuild(removing: back, adding: adds) != nil else { self = backup; return nil }
        let frontRim = cutVerts, backRim = Set(dup.values)
        var caps: [Int] = []
        for loop in boundaryLoops() where Set(loop).isSubset(of: frontRim) || Set(loop).isSubset(of: backRim) {
            guard let ids = fill(loop: loop, mode: capMode) else { self = backup; return nil }
            caps += ids
        }
        return selection(.face, caps)
    }
}
