//  Starting shapes and uniform refinement for HalfEdgeMesh (also what the tests and later tools model from).
//  `box` is all quads, wound outward, flat normals, each face uv 0...1. `subdividedQuads` is the topology of
//  one Catmull-Clark step without any smoothing: every n-gon becomes n quads around its centre, all positions
//  stay on the original surface; corner data is interpolated, and seams are preserved.

import simd

public extension HalfEdgeMesh {

    static func box(size: Vec3 = Vec3(repeating: 1), center: Vec3 = .zero) -> HalfEdgeMesh {
        var m = HalfEdgeMesh()
        let h = size / 2
        for i in 0..<8 {
            m.addVertex(center + Vec3((i & 1) == 0 ? -h.x : h.x, (i & 2) == 0 ? -h.y : h.y, (i & 4) == 0 ? -h.z : h.z))
        }
        // vertex index = x + 2y + 4z (bit set = +); quads counter-clockwise seen from outside
        let quads = [[1, 3, 7, 5], [0, 4, 6, 2], [2, 6, 7, 3], [0, 1, 5, 4], [4, 5, 7, 6], [0, 2, 3, 1]]
        let uv = [Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)]
        for q in quads { m.addFace(q, uvs: uv) }
        return m
    }

    /// Each n-gon split into n quads about its centroid (new vertices at edge midpoints and face centres).
    func subdividedQuads() -> HalfEdgeMesh {
        var out = HalfEdgeMesh()
        out.positions = positions; out.vertexHE = [Int](repeating: -1, count: positions.count)
        var mids: [UInt64: Int] = [:]
        func mid(_ a: Int, _ b: Int) -> Int {
            let k = a < b ? Self.key(a, b) : Self.key(b, a)
            if let v = mids[k] { return v }
            let v = out.addVertex((positions[a] + positions[b]) / 2); mids[k] = v; return v
        }
        for f in faces {
            let hs = halfedges(of: f), n = hs.count
            let vs = hs.map { heOrigin[$0] }
            let centre = out.addVertex(faceCentroid(f))
            let uvC = hs.reduce(Vec2.zero) { $0 + heUV[$1] } / Double(n)
            let nC = normalize3(hs.reduce(Vec3.zero) { $0 + heNormal[$1] })
            for i in 0..<n {
                let j = (i + 1) % n, p = (i + n - 1) % n
                let uvs = [heUV[hs[i]], (heUV[hs[i]] + heUV[hs[j]]) / 2, uvC, (heUV[hs[p]] + heUV[hs[i]]) / 2]
                let nm = [heNormal[hs[i]], normalize3(heNormal[hs[i]] + heNormal[hs[j]]), nC, normalize3(heNormal[hs[p]] + heNormal[hs[i]])]
                out.addFace([vs[i], mid(vs[i], vs[j]), centre, mid(vs[p], vs[i])], uvs: uvs, normals: nm)
            }
        }
        return out
    }
}
