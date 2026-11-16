//  Conversion between Mesh3 (triangle soup with per-triangle vertices) and HalfEdgeMesh.
//
//  Mesh3 -> HalfEdgeMesh: vertices are welded by position (quantised to `weldTolerance`; two points straddling
//  a quantisation boundary stay apart). Every triangle becomes a face. The Mesh3's uvs and normals are kept
//  per corner, so seams and hard edges survive as corners that differ across an edge. Degenerate triangles
//  (two corners welded together, or zero area) are dropped. A triangle whose directed edge is already taken is
//  retried reversed (so a mesh built from independently wound triangles imports consistently); if that clashes
//  too (a third face on an edge: non-manifold) it is dropped. Both counts are in `ImportReport`.
//  `mergeCoplanarFaces` then turns triangle pairs back into quads / n-gons.
//
//  HalfEdgeMesh -> Mesh3: every n-gon is triangulated (3: as is, 4: along the better diagonal, more: ear clip
//  in the face's own plane through `Triangulation.earClip`, fan fallback), closed shells that enclose negative
//  volume are emitted reversed so triangles face outward, and corners that agree on position, uv and normal
//  share a vertex (so shading and seams come out exactly as stored). Not guaranteed: planar n-gons only
//  triangulate well; a strongly non-planar polygon is triangulated against its Newell normal.

import simd

public extension HalfEdgeMesh {

    struct ImportReport: Sendable, Equatable {
        public var degenerate = 0
        public var reversed = 0
        public var rejected = 0
    }

    init(_ mesh: Mesh3, weldTolerance: Double = 1e-6) { self = Self.imported(mesh, weldTolerance: weldTolerance).mesh }

    static func imported(_ mesh: Mesh3, weldTolerance: Double = 1e-6) -> (mesh: HalfEdgeMesh, report: ImportReport) {
        var m = HalfEdgeMesh(), report = ImportReport()
        struct Q: Hashable { var x, y, z: Int64 }
        let inv = 1 / max(weldTolerance, 1e-12)
        var ids: [Q: Int] = [:]
        func vid(_ p: Vec3) -> Int {
            let q = Q(x: Int64((p.x * inv).rounded()), y: Int64((p.y * inv).rounded()), z: Int64((p.z * inv).rounded()))
            if let v = ids[q] { return v }
            let v = m.addVertex(p); ids[q] = v; return v
        }
        let hasN = mesh.normals.count == mesh.positions.count, hasUV = mesh.uvs.count == mesh.positions.count
        var t = 0
        while t + 2 < mesh.indices.count {
            let i = (0..<3).map { Int(mesh.indices[t + $0]) }; t += 3
            let p = i.map { mesh.positions[$0] }
            let v = p.map(vid)
            if Set(v).count < 3 || len3(HalfEdgeMesh.newell(p)) < 1e-14 { report.degenerate += 1; continue }
            let uv = i.map { hasUV ? mesh.uvs[$0] : Vec2.zero }
            let nn = i.map { hasN ? mesh.normals[$0] : nil as Vec3? }
            let flat = normalize3(HalfEdgeMesh.newell(p))
            let ns = nn.map { $0 ?? flat }
            if m.addFace(v, uvs: uv, normals: ns) != nil { continue }
            if m.addFace(v.reversed(), uvs: uv.reversed(), normals: ns.reversed()) != nil { report.reversed += 1 } else { report.rejected += 1 }
        }
        return (m, report)
    }

    /// Dissolve the edges between faces that lie in one plane (normals within `degrees`), skipping seams and hard
    /// edges, so the triangles of a flat face become one n-gon. Returns how many edges were dissolved.
    @discardableResult
    mutating func mergeCoplanarFaces(withinDegrees degrees: Double = 0.01) -> Int {
        let c = cos(degrees * .pi / 180)
        var n = 0
        for e in edges where isAlive(edge: e) && !isBoundary(edge: e) && !isSeam(edge: e) && !isHard(edge: e) {
            let fs = faces(ofEdge: e)
            guard fs.count == 2, fs[0] != fs[1], dot3(faceNormal(fs[0]), faceNormal(fs[1])) >= c else { continue }
            if dissolveEdge(e) != nil { n += 1 }
        }
        return n
    }

    /// Triangle list of the whole mesh, see the file header. `flatNormals` ignores the stored corner normals.
    func toMesh3(flatNormals: Bool = false) -> Mesh3 {
        var flipped = Set<Int>()
        for comp in connectedComponents() where isClosedComponent(comp) && signedVolume(of: comp) < 0 { flipped.formUnion(comp) }
        var out = Mesh3()
        struct K: Hashable { var v: Int; var u, w: SIMD2<Int32>; var n: SIMD3<Int32> }
        func q(_ x: Double) -> Int32 { Int32(truncatingIfNeeded: Int((x * 1e6).rounded())) }
        var cache: [K: UInt32] = [:]
        for f in faces {
            var hs = halfedges(of: f)
            if flipped.contains(f) { hs = [hs[0]] + hs.dropFirst().reversed() }
            let fn = flipped.contains(f) ? -faceNormal(f) : faceNormal(f)
            func corner(_ h: Int) -> UInt32 {
                let n = flatNormals ? fn : (flipped.contains(f) ? -heNormal[h] : heNormal[h])
                let k = K(v: heOrigin[h], u: SIMD2(q(heUV[h].x), q(heUV[h].y)), w: .zero, n: SIMD3(q(n.x), q(n.y), q(n.z)))
                if let i = cache[k] { return i }
                let i = UInt32(out.positions.count)
                out.positions.append(positions[heOrigin[h]]); out.normals.append(n); out.uvs.append(heUV[h])
                cache[k] = i; return i
            }
            let idx = hs.map(corner)
            for tri in triangulate(hs.map { positions[heOrigin[$0]] }, normal: fn) {
                out.indices += [idx[tri.0], idx[tri.1], idx[tri.2]]
            }
        }
        return out
    }

    /// Triangles of one polygon as corner-index triples, wound like the polygon.
    private func triangulate(_ p: [Vec3], normal n: Vec3) -> [(Int, Int, Int)] {
        let c = p.count
        if c == 3 { return [(0, 1, 2)] }
        func fan() -> [(Int, Int, Int)] { (1..<(c - 1)).map { (0, $0, $0 + 1) } }
        func good(_ a: Int, _ b: Int, _ d: Int) -> Bool { dot3(cross3(p[b] - p[a], p[d] - p[a]), n) > 0 }
        if c == 4 {
            let opts: [(Int, Int, Int, Int, Int, Int)] = [(0, 1, 2, 0, 2, 3), (1, 2, 3, 1, 3, 0)]
            let ok = opts.filter { good($0.0, $0.1, $0.2) && good($0.3, $0.4, $0.5) }
            let pick = ok.min { dist3(p[$0.0], p[$0.2]) < dist3(p[$1.0], p[$1.2]) } ?? opts[0]
            return [(pick.0, pick.1, pick.2), (pick.3, pick.4, pick.5)]
        }
        let nn = normalize3(n)
        let u = normalize3(abs(nn.x) < 0.9 ? cross3(nn, Vec3(1, 0, 0)) : cross3(nn, Vec3(0, 1, 0)))
        let w = cross3(nn, u)
        let tris = Triangulation.earClip(p.map { Vec2(dot3($0, u), dot3($0, w)) })
        return tris.count == c - 2 ? tris : fan()
    }
}
