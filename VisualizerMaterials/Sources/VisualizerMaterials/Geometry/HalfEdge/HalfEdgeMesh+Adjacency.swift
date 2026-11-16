//  Adjacency and manifold / boundary queries for HalfEdgeMesh. All pure reads, O(valence) unless noted.
//  Vertex rings are visited in one rotational order; a boundary vertex's ring starts and ends at the boundary.
//  Guarantee: on a manifold mesh each query returns every incident element exactly once. On a bow-tie vertex
//  only the fan holding `vertexHE` is visited (see `nonManifoldVertices`).

import simd

public extension HalfEdgeMesh {

    // MARK: face

    /// The face's halfedges in winding order.
    func halfedges(of face: Int) -> [Int] {
        var out: [Int] = []
        let start = faceHE[face]
        guard start >= 0 else { return out }
        var h = start
        repeat {
            out.append(h); h = heNext[h]
            if out.count > heOrigin.count { break }          // corrupt loop: never spin
        } while h != start && h >= 0
        return out
    }
    func vertices(of face: Int) -> [Int] { halfedges(of: face).map { heOrigin[$0] } }
    func edges(of face: Int) -> [Int] { halfedges(of: face).map { heEdge[$0] } }
    func corners(of face: Int) -> [Vec3] { vertices(of: face).map { positions[$0] } }
    func valence(ofFace f: Int) -> Int { halfedges(of: f).count }
    /// Faces sharing an edge with `f`.
    func neighbours(ofFace f: Int) -> [Int] {
        halfedges(of: f).compactMap { heTwin[$0] >= 0 ? heFace[heTwin[$0]] : nil }
    }
    func faceNormal(_ f: Int) -> Vec3 { normalize3(HalfEdgeMesh.newell(corners(of: f))) }
    func faceArea(_ f: Int) -> Double { len3(HalfEdgeMesh.newell(corners(of: f))) / 2 }
    func faceCentroid(_ f: Int) -> Vec3 {
        let c = corners(of: f)
        return c.reduce(Vec3.zero, +) / Double(c.count)
    }

    // MARK: edge

    /// The edge's two endpoints (the halfedge's direction).
    func vertices(ofEdge e: Int) -> (Int, Int) {
        let h = edgeHE[e]
        return (heOrigin[h], heOrigin[heNext[h]])
    }
    func edgeLength(_ e: Int) -> Double { let v = vertices(ofEdge: e); return dist3(positions[v.0], positions[v.1]) }
    func midpoint(ofEdge e: Int) -> Vec3 { let v = vertices(ofEdge: e); return (positions[v.0] + positions[v.1]) / 2 }
    /// One face, or two.
    func faces(ofEdge e: Int) -> [Int] {
        let h = edgeHE[e]
        return heTwin[h] >= 0 ? [heFace[h], heFace[heTwin[h]]] : [heFace[h]]
    }
    func isBoundary(edge e: Int) -> Bool { heTwin[edgeHE[e]] < 0 }
    /// The edge between two vertices, if any.
    func edge(between a: Int, _ b: Int) -> Int? {
        (directed[Self.key(a, b)] ?? directed[Self.key(b, a)]).map { heEdge[$0] }
    }
    /// The halfedge a -> b, if any.
    func halfedge(from a: Int, to b: Int) -> Int? { directed[Self.key(a, b)] }

    /// An edge whose corner uvs differ across it (the uv island's border).
    func isSeam(edge e: Int) -> Bool {
        let h = edgeHE[e], t = heTwin[h]
        guard t >= 0 else { return false }
        let tol = 1e-9
        return len2(heUV[h] - heUV[heNext[t]]) > tol || len2(heUV[heNext[h]] - heUV[t]) > tol
    }
    /// An edge whose corner normals differ across it (a hard edge in the shading).
    func isHard(edge e: Int, tolerance: Double = 1e-5) -> Bool {
        let h = edgeHE[e], t = heTwin[h]
        guard t >= 0 else { return false }
        return len3(heNormal[h] - heNormal[heNext[t]]) > tolerance || len3(heNormal[heNext[h]] - heNormal[t]) > tolerance
    }
    private func len2(_ v: Vec2) -> Double { (v.x * v.x + v.y * v.y).squareRoot() }

    // MARK: vertex

    /// Halfedges leaving `v`, in rotational order (boundary: from one boundary to the other).
    func outgoing(of v: Int) -> [Int] {
        let h0 = vertexHE[v]
        guard h0 >= 0 else { return [] }
        var start = h0, steps = 0
        while true {                                         // rotate back to the fan's start (or all the way round)
            let p = heTwin[hePrev[start]]
            if p < 0 || p == h0 { if p == h0 { start = h0 }; break }
            start = p; steps += 1
            if steps > heOrigin.count { break }
        }
        var out: [Int] = [], h = start
        repeat {
            out.append(h)
            let t = heTwin[h]
            if t < 0 { break }
            h = heNext[t]
            if out.count > heOrigin.count { break }
        } while h != start
        return out
    }
    func valence(of v: Int) -> Int { edges(around: v).count }
    func neighbours(of v: Int) -> [Int] {
        var out = outgoing(of: v).map { heOrigin[heNext[$0]] }
        if isBoundary(vertex: v), let first = outgoing(of: v).first { out.insert(heOrigin[hePrev[first]], at: 0) }
        return out
    }
    func faces(around v: Int) -> [Int] { outgoing(of: v).map { heFace[$0] } }
    /// Edges at `v` in rotational order, boundary vertices including the boundary edge that only arrives at `v`.
    func edges(around v: Int) -> [Int] {
        let out = outgoing(of: v)
        guard let first = out.first else { return [] }
        var es = out.map { heEdge[$0] }
        if heTwin[hePrev[first]] < 0 { es.insert(heEdge[hePrev[first]], at: 0) }
        return es
    }
    /// True when `v`'s fan is open (touches a boundary edge).
    func isBoundary(vertex v: Int) -> Bool {
        guard let first = outgoing(of: v).first, let last = outgoing(of: v).last else { return false }
        return heTwin[hePrev[first]] < 0 || heTwin[last] < 0
    }

    // MARK: mesh-wide

    func boundaryEdges() -> [Int] { edges.filter { isBoundary(edge: $0) } }
    /// Boundary vertex loops (each a closed ring of vertices along boundary edges).
    func boundaryLoops() -> [[Int]] {
        var seen = Set<Int>(), loops: [[Int]] = []
        for e in boundaryEdges() where !seen.contains(e) {
            var loop: [Int] = [], h = edgeHE[e]
            repeat {
                seen.insert(heEdge[h]); loop.append(heOrigin[h])
                // step to the next boundary halfedge: rotate round the head vertex until a boundary one leaves it
                var n = heNext[h], guardN = 0
                while heTwin[n] >= 0 && guardN < heOrigin.count { n = heNext[heTwin[n]]; guardN += 1 }
                h = n
            } while heEdge[h] != e && !seen.contains(heEdge[h]) && loop.count <= heOrigin.count
            loops.append(loop)
        }
        return loops
    }
    /// No boundary edges (and not empty).
    var isClosed: Bool { edgeCount > 0 && !edgeHE.contains { $0 >= 0 && heTwin[$0] < 0 } }
    /// Vertices whose faces form more than one fan (bow-ties). O(H).
    func nonManifoldVertices() -> [Int] {
        var count = [Int](repeating: 0, count: positions.count)
        for h in 0..<heOrigin.count where heFace[h] >= 0 { count[heOrigin[h]] += 1 }
        return vertices.filter { outgoing(of: $0).count != count[$0] }
    }
    var isManifold: Bool { nonManifoldVertices().isEmpty }
    /// V - E + F over the used vertices.
    var eulerCharacteristic: Int { vertexCount - edgeCount + faceCount }
    /// Face groups connected through shared edges.
    func connectedComponents() -> [[Int]] {
        var seen = Set<Int>(), out: [[Int]] = []
        for f in faces where !seen.contains(f) {
            var comp: [Int] = [], stack = [f]; seen.insert(f)
            while let x = stack.popLast() {
                comp.append(x)
                for n in neighbours(ofFace: x) where seen.insert(n).inserted { stack.append(n) }
            }
            out.append(comp.sorted())
        }
        return out
    }
    /// Signed volume enclosed by `faces` (positive = wound outward), via fan triangles.
    func signedVolume(of fs: some Sequence<Int>) -> Double {
        var v = 0.0
        for f in fs {
            let c = corners(of: f)
            for i in 1..<(c.count - 1) { v += dot3(c[0], cross3(c[i], c[i + 1])) / 6 }
        }
        return v
    }
}
