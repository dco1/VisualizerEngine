import XCTest
@testable import VisualizerMaterials

final class HalfEdgeMeshTests: XCTestCase {

    // MARK: fixtures (independently wound triangles, like the rest of Mesh3's builders)

    private func boxMesh(_ w: Double = 2, _ h: Double = 3, _ d: Double = 4) -> Mesh3 {
        var m = Mesh3()
        let (x, y, z) = (w, h, d)
        m.addQuad(Vec3(x,0,0), Vec3(x,y,0), Vec3(x,y,z), Vec3(x,0,z), outward: Vec3( 1,0,0))
        m.addQuad(Vec3(0,0,0), Vec3(0,0,z), Vec3(0,y,z), Vec3(0,y,0), outward: Vec3(-1,0,0))
        m.addQuad(Vec3(0,y,0), Vec3(0,y,z), Vec3(x,y,z), Vec3(x,y,0), outward: Vec3(0, 1,0))
        m.addQuad(Vec3(0,0,0), Vec3(x,0,0), Vec3(x,0,z), Vec3(0,0,z), outward: Vec3(0,-1,0))
        m.addQuad(Vec3(0,0,z), Vec3(x,0,z), Vec3(x,y,z), Vec3(0,y,z), outward: Vec3(0,0, 1))
        m.addQuad(Vec3(0,0,0), Vec3(0,y,0), Vec3(x,y,0), Vec3(x,0,0), outward: Vec3(0,0,-1))
        return m
    }

    private func sphereMesh(lon: Int = 24, lat: Int = 12, r: Double = 1.5) -> Mesh3 {
        func p(_ i: Int, _ j: Int) -> Vec3 {
            let th = Double.pi * Double(j) / Double(lat), ph = 2 * Double.pi * Double(i % lon) / Double(lon)
            return Vec3(r * sin(th) * cos(ph), r * cos(th), r * sin(th) * sin(ph))
        }
        var m = Mesh3()
        for j in 0..<lat { for i in 0..<lon {
            let a = p(i, j), b = p(i + 1, j), c = p(i + 1, j + 1), d = p(i, j + 1)
            let out = (a + b + c + d) / 4
            if j == 0 { m.addTriangle(a, c, d, outward: out) }
            else if j == lat - 1 { m.addTriangle(a, b, c, outward: out) }
            else { m.addTriangle(a, b, c, outward: out); m.addTriangle(a, c, d, outward: out) }
        } }
        return m
    }

    private func cylinderMesh(n: Int = 16, r: Double = 1, h: Double = 2) -> Mesh3 {
        func p(_ i: Int, _ y: Double) -> Vec3 {
            let a = 2 * Double.pi * Double(i % n) / Double(n); return Vec3(r * cos(a), y, r * sin(a))
        }
        var m = Mesh3()
        for i in 0..<n {
            let out = (p(i, 0) + p(i + 1, 0)) / 2
            m.addQuad(p(i, 0), p(i + 1, 0), p(i + 1, h), p(i, h), outward: Vec3(out.x, 0, out.z))
        }
        for i in 1..<(n - 1) {
            m.addTriangle(p(0, h), p(i, h), p(i + 1, h), outward: Vec3(0, 1, 0))
            m.addTriangle(p(0, 0), p(i, 0), p(i + 1, 0), outward: Vec3(0, -1, 0))
        }
        return m
    }

    // MARK: audit helpers

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            v += dot3(m.positions[Int(m.indices[t])], cross3(m.positions[Int(m.indices[t + 1])], m.positions[Int(m.indices[t + 2])])) / 6
            t += 3
        }
        return v
    }
    /// Undirected welded edges not used by exactly two triangles, plus edges used twice in the same direction.
    private func openEdges(_ m: Mesh3) -> Int {
        func key(_ p: Vec3) -> SIMD3<Int64> { SIMD3(Int64((p.x * 1e6).rounded()), Int64((p.y * 1e6).rounded()), Int64((p.z * 1e6).rounded())) }
        struct D: Hashable { let a, b: SIMD3<Int64> }
        var dir: [D: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            let k = (0..<3).map { key(m.positions[Int(m.indices[t + $0])]) }
            for i in 0..<3 { dir[D(a: k[i], b: k[(i + 1) % 3]), default: 0] += 1 }
            t += 3
        }
        return dir.filter { $0.value != 1 || dir[D(a: $0.key.b, b: $0.key.a)] != 1 }.count
    }
    private func invertedTriangles(_ m: Mesh3) -> Int {
        var bad = 0, t = 0
        while t + 2 < m.indices.count {
            let i = (0..<3).map { Int(m.indices[t + $0]) }
            let n = cross3(m.positions[i[1]] - m.positions[i[0]], m.positions[i[2]] - m.positions[i[0]])
            if dot3(n, m.normals[i[0]] + m.normals[i[1]] + m.normals[i[2]]) <= 0 { bad += 1 }
            t += 3
        }
        return bad
    }
    private func assertSound(_ h: HalfEdgeMesh, file: StaticString = #filePath, line: UInt = #line) {
        let issues = h.validate()
        XCTAssertTrue(issues.isEmpty, "\(issues)", file: file, line: line)
    }

    // MARK: round trips

    func testBoxRoundTrip() {
        let src = boxMesh()
        let (he, report) = HalfEdgeMesh.imported(src)
        XCTAssertEqual(report, .init())
        XCTAssertEqual(he.vertexCount, 8); XCTAssertEqual(he.faceCount, 12); XCTAssertEqual(he.edgeCount, 18)
        XCTAssertEqual(he.eulerCharacteristic, 2); XCTAssertTrue(he.isClosed); XCTAssertTrue(he.isManifold)
        assertSound(he)

        var quads = he
        XCTAssertEqual(quads.mergeCoplanarFaces(), 6)
        XCTAssertEqual(quads.faceCount, 6)
        XCTAssertTrue(quads.faces.allSatisfy { quads.valence(ofFace: $0) == 4 })
        assertSound(quads)

        for h in [he, quads] {
            let out = h.toMesh3()
            XCTAssertEqual(volume(out), volume(src), accuracy: 1e-9)
            XCTAssertEqual(volume(out), 24, accuracy: 1e-9)
            XCTAssertEqual(out.triangleCount, 12)
            XCTAssertEqual(openEdges(out), 0)
            XCTAssertEqual(invertedTriangles(out), 0)
        }
    }

    func testSphereRoundTrip() {
        let src = sphereMesh()
        let (he, report) = HalfEdgeMesh.imported(src)
        XCTAssertEqual(report, .init())
        XCTAssertEqual(he.vertexCount, 24 * 11 + 2)
        XCTAssertEqual(he.eulerCharacteristic, 2)
        XCTAssertTrue(he.isClosed && he.isManifold)
        assertSound(he)
        let out = he.toMesh3()
        XCTAssertEqual(volume(out), volume(src), accuracy: 1e-9)
        XCTAssertEqual(openEdges(out), 0)
        XCTAssertEqual(invertedTriangles(out), 0)
        XCTAssertEqual(out.triangleCount, src.triangleCount)
        XCTAssertEqual(he.signedVolume(of: he.faces), volume(src), accuracy: 1e-9)
    }

    func testCylinderRoundTripWithNgonCaps() {
        let src = cylinderMesh()
        var he = HalfEdgeMesh(src)
        assertSound(he)
        XCTAssertEqual(he.eulerCharacteristic, 2)
        he.mergeCoplanarFaces()
        assertSound(he)
        XCTAssertEqual(he.faceCount, 16 + 2)
        XCTAssertEqual(he.faces.filter { he.valence(ofFace: $0) == 16 }.count, 2)
        let out = he.toMesh3()
        XCTAssertEqual(volume(out), volume(src), accuracy: 1e-9)
        XCTAssertEqual(openEdges(out), 0)
        XCTAssertEqual(invertedTriangles(out), 0)
        XCTAssertEqual(out.triangleCount, 32 + 2 * 14)
    }

    func testInsideOutShellIsEmittedOutward() {
        var inside = boxMesh()
        for t in stride(from: 0, to: inside.indices.count, by: 3) { inside.indices.swapAt(t + 1, t + 2) }
        XCTAssertLessThan(volume(inside), 0)
        var he = HalfEdgeMesh(inside)
        XCTAssertEqual(he.toMesh3().triangleCount, 12)
        XCTAssertEqual(volume(he.toMesh3()), 24, accuracy: 1e-9)
        XCTAssertEqual(he.orientOutward(), 1)
        XCTAssertEqual(he.signedVolume(of: he.faces), 24, accuracy: 1e-9)
        assertSound(he)
    }

    func testImportFlipsInconsistentTrianglesAndWeldsPositions() {
        var m = Mesh3()
        m.addTriangle(Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), outward: Vec3(0,0,1))
        m.addTriangle(Vec3(1,0,0), Vec3(1,1,0), Vec3(0,1,0), outward: Vec3(0,0,-1))   // wound the other way
        let (he, report) = HalfEdgeMesh.imported(m)
        XCTAssertEqual(he.vertexCount, 4); XCTAssertEqual(he.edgeCount, 5)
        XCTAssertEqual(report.reversed, 1)
        XCTAssertEqual(he.boundaryLoops().count, 1)
        assertSound(he)
    }

    func testSeamsAndHardEdgesSurviveRoundTrip() {
        let he = HalfEdgeMesh.box()
        XCTAssertEqual(he.edges.filter { he.isSeam(edge: $0) }.count, 12)
        XCTAssertEqual(he.edges.filter { he.isHard(edge: $0) }.count, 12)
        let out = he.toMesh3()
        XCTAssertEqual(out.vertexCount, 24)                       // 4 corners x 6 faces, nothing shared across hard edges
        XCTAssertEqual(out.triangleCount, 12)
        XCTAssertEqual(volume(out), 1, accuracy: 1e-12)
        XCTAssertEqual(invertedTriangles(out), 0)
        XCTAssertEqual(openEdges(out), 0)
    }

    // MARK: adjacency

    func testAdjacencyOnBox() {
        let he = HalfEdgeMesh.box()
        for v in he.vertices {
            XCTAssertEqual(he.valence(of: v), 3); XCTAssertEqual(he.neighbours(of: v).count, 3)
            XCTAssertEqual(he.faces(around: v).count, 3); XCTAssertFalse(he.isBoundary(vertex: v))
        }
        for f in he.faces { XCTAssertEqual(he.neighbours(ofFace: f).count, 4); XCTAssertEqual(he.faceArea(f), 1, accuracy: 1e-12) }
        for e in he.edges { XCTAssertEqual(he.faces(ofEdge: e).count, 2); XCTAssertEqual(he.edgeLength(e), 1, accuracy: 1e-12) }
        XCTAssertEqual(he.edge(between: 0, 1) != nil, true)
        XCTAssertNil(he.edge(between: 0, 7))
    }

    func testBoundaryQueries() {
        var he = HalfEdgeMesh.box()
        he.deleteFace(0)
        assertSound(he)
        XCTAssertFalse(he.isClosed)
        XCTAssertEqual(he.boundaryEdges().count, 4)
        XCTAssertEqual(he.boundaryLoops().count, 1)
        XCTAssertEqual(he.boundaryLoops()[0].count, 4)
        let onBoundary = he.vertices.filter { he.isBoundary(vertex: $0) }
        XCTAssertEqual(onBoundary.count, 4)
        for v in onBoundary { XCTAssertEqual(he.valence(of: v), 3); XCTAssertEqual(he.faces(around: v).count, 2) }
        XCTAssertEqual(he.connectedComponents().count, 1)
    }

    // MARK: loops and rings

    private func subdividedBox() -> HalfEdgeMesh { HalfEdgeMesh.box().subdividedQuads() }

    func testLoopsAndRingsOnSubdividedBox() {
        let he = subdividedBox()
        assertSound(he)
        XCTAssertEqual(he.faceCount, 24); XCTAssertEqual(he.vertexCount, 26); XCTAssertEqual(he.eulerCharacteristic, 2)
        // an edge on the mid-height belt, on the +x face
        let mid = he.edges.first { e in
            let v = he.vertices(ofEdge: e)
            let a = he.positions[v.0], b = he.positions[v.1]
            return abs(a.y) < 1e-9 && abs(b.y) < 1e-9 && abs(a.x - 0.5) < 1e-9 && abs(b.x - 0.5) < 1e-9
        }!
        let loop = he.edgeLoop(from: mid)
        XCTAssertTrue(loop.isClosed); XCTAssertEqual(loop.edges.count, 8)
        XCTAssertEqual(Set(loop.edges).count, 8)
        for e in loop.edges { let v = he.vertices(ofEdge: e); XCTAssertEqual(he.positions[v.0].y, 0, accuracy: 1e-9); XCTAssertEqual(he.positions[v.1].y, 0, accuracy: 1e-9) }
        XCTAssertEqual(he.edgeLoop(from: loop.edges[3]).edges.count, 8)

        // a ring: the edges crossing that belt's neighbours, one slab round the box
        let ring = he.edgeRing(from: mid)
        XCTAssertTrue(ring.isClosed); XCTAssertEqual(ring.edges.count, 8)
        XCTAssertEqual(Set(ring.edges).count, 8)
        XCTAssertTrue(ring.edges.contains(mid))
        XCTAssertEqual(he.faces(alongRing: ring).count, 8)
        XCTAssertEqual(Set(ring.edges).intersection(loop.edges).count, 2)   // the edge and its antipode round the box
    }

    func testLoopsStopAtPolesAndFollowBoundaries() {
        let plain = HalfEdgeMesh.box()
        let e = plain.edges[0]
        XCTAssertEqual(plain.edgeLoop(from: e).edges, [e])               // corners have valence 3
        XCTAssertFalse(plain.edgeLoop(from: e).isClosed)
        let ring = plain.edgeRing(from: e)                                // straight round the cube's quads
        XCTAssertTrue(ring.isClosed); XCTAssertEqual(ring.edges.count, 4)

        var open = plain.subdividedQuads()
        open.deleteFace(open.faces[0])
        let b = open.boundaryEdges()
        XCTAssertEqual(b.count, 4 + 0 * 1 - 0)
        let walk = open.edgeLoop(from: b[0])
        XCTAssertTrue(walk.isClosed); XCTAssertEqual(walk.edges.count, 4)
        XCTAssertEqual(Set(walk.edges), Set(b))
    }

    // MARK: selection

    func testSelectionGrowShrinkAndFlush() {
        let he = subdividedBox().subdividedQuads()                       // 96 quads, 4x4 per face
        let f0 = he.faces.first { f in let c = he.faceCentroid(f); return he.faceNormal(f).x > 0.9 && abs(c.y - 0.125) < 1e-9 && abs(c.z - 0.125) < 1e-9 }!
        let s = he.selection(.face, [f0])
        XCTAssertEqual(s.vertices.count, 4); XCTAssertEqual(s.edges.count, 4)
        let g = he.grow(s)
        XCTAssertEqual(g.faces.count, 9)
        XCTAssertEqual(he.shrink(g), s)
        XCTAssertEqual(he.inverted(s).faces.count, 95)
        XCTAssertEqual(he.boundaryEdges(ofFaces: g.faces).count, 12)
        XCTAssertEqual(he.linkedFaces(from: f0).count, 96)
        let stop = Set(he.edges.filter { he.isHard(edge: $0) })          // box corners are hard: one cube face = 16 quads
        XCTAssertEqual(he.linkedFaces(from: f0, stoppingAt: stop).count, 16)

        let v = he.selection(.vertex, g.vertices)
        XCTAssertEqual(v.faces, g.faces)
        let ed = he.selection(.edge, he.edges(of: f0))
        XCTAssertEqual(ed.faces, [f0])
        XCTAssertEqual(he.shrink(he.grow(ed)), ed)
        XCTAssertEqual(he.selectAll(.face).faces.count, 96)
        XCTAssertEqual(he.shrink(he.selectAll(.vertex)), he.selectAll(.vertex))
    }

    // MARK: mutation primitives

    func testSplitEdgeKeepsShapeAndStaysValid() {
        var he = HalfEdgeMesh.box()
        let e = he.edges[4]
        let (a, b) = he.vertices(ofEdge: e)
        let m = he.splitEdge(e, at: 0.25)!
        XCTAssertEqual(he.positions[m], he.positions[a] * 0.75 + he.positions[b] * 0.25)
        assertSound(he)
        XCTAssertEqual(he.vertexCount, 9); XCTAssertEqual(he.edgeCount, 13); XCTAssertEqual(he.faceCount, 6)
        XCTAssertEqual(he.valence(of: m), 2)
        XCTAssertEqual(he.faces(around: m).map { he.valence(ofFace: $0) }.sorted(), [5, 5])
        XCTAssertEqual(he.eulerCharacteristic, 2)
        XCTAssertEqual(he.signedVolume(of: he.faces), 1, accuracy: 1e-12)
        XCTAssertTrue(he.isSeam(edge: he.edge(between: a, m)!))          // the box's uv seams stay seams
        // a boundary edge splits too
        var open = HalfEdgeMesh.box(); open.deleteFace(0)
        _ = open.splitEdge(open.boundaryEdges()[0])
        assertSound(open); XCTAssertEqual(open.boundaryEdges().count, 5)
    }

    func testSplitFaceThenDissolveRestores() {
        var he = HalfEdgeMesh.box()
        let f = he.faces[0], hs = he.halfedges(of: f)
        XCTAssertNil(he.splitFace(f, from: hs[0], to: hs[1]))
        let g = he.splitFace(f, from: hs[0], to: hs[2])!
        assertSound(he)
        XCTAssertEqual(he.faceCount, 7); XCTAssertEqual(he.valence(ofFace: f), 3); XCTAssertEqual(he.valence(ofFace: g), 3)
        XCTAssertEqual(he.faceNormal(f), he.faceNormal(g))
        XCTAssertEqual(he.signedVolume(of: he.faces), 1, accuracy: 1e-12)
        let diagonal = he.edge(between: he.origin(of: hs[0]), he.origin(of: hs[2]))!
        XCTAssertNotNil(he.dissolveEdge(diagonal))
        assertSound(he)
        XCTAssertEqual(he.faceCount, 6)
        XCTAssertTrue(he.faces.allSatisfy { he.valence(ofFace: $0) == 4 })
        let hex = he.dissolveEdge(he.edges[0])!
        XCTAssertEqual(he.valence(ofFace: hex), 6)                        // two quads make a hexagon
        assertSound(he)
    }

    func testFlipCollapseAndDelete() {
        var he = HalfEdgeMesh(sphereMesh(lon: 12, lat: 6))
        let before = he.vertexCount
        let e = he.edges.first { e in he.faces(ofEdge: e).allSatisfy { he.valence(ofFace: $0) == 3 } && !he.isBoundary(edge: e) }!
        XCTAssertTrue(he.flipEdge(e))
        assertSound(he); XCTAssertEqual(he.faceCount, 120)
        XCTAssertEqual(he.eulerCharacteristic, 2)
        let nf = he.faceCount
        var collapsed = 0
        for e in he.edges where he.isAlive(edge: e) && he.edgeLength(e) < 0.9 {
            if he.collapseEdge(e) != nil { collapsed += 1; assertSound(he) }
            if collapsed == 6 { break }
        }
        XCTAssertGreaterThan(collapsed, 0)
        XCTAssertEqual(he.vertexCount, before - collapsed)
        XCTAssertLessThan(he.faceCount, nf)
        XCTAssertEqual(he.eulerCharacteristic, 2)
        XCTAssertTrue(he.isClosed && he.isManifold)

        // the tetrahedron must refuse to collapse into nothing
        var tet = HalfEdgeMesh()
        for p in [Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)] { tet.addVertex(p) }
        for f in [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]] { XCTAssertNotNil(tet.addFace(f)) }
        assertSound(tet)
        XCTAssertNil(tet.collapseEdge(tet.edges[0]))
        assertSound(tet)
    }

    func testAddFaceRefusesWhatWouldBreakTheMesh() {
        var m = HalfEdgeMesh()
        for p in [Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(1,1,0)] { m.addVertex(p) }
        XCTAssertNotNil(m.addFace([0, 1, 2]))
        XCTAssertNil(m.addFace([0, 1, 3]))              // 0->1 already used
        XCTAssertNil(m.addFace([0, 1, 1]))
        XCTAssertNil(m.addFace([0, 1]))
        XCTAssertNotNil(m.addFace([1, 0, 3]))
        assertSound(m)
        let (c, remap) = m.compacted()
        XCTAssertEqual(c.faceCount, 2); XCTAssertEqual(remap.vertex.filter { $0 >= 0 }.count, 4)
        assertSound(c)
    }

    // MARK: validate

    func testValidateCatchesCorruption() {
        let good = HalfEdgeMesh(sphereMesh(lon: 8, lat: 4))
        assertSound(good)
        func corrupt(_ name: String, _ edit: (inout HalfEdgeMesh) -> Void) {
            var m = good; edit(&m)
            XCTAssertTrue(m.validate().contains { $0.severity == .error }, "\(name) went unnoticed")
            XCTAssertFalse(m.isValid, name)
        }
        corrupt("twin one-way") { $0.heTwin[3] = $0.heTwin[3] == 5 ? 6 : 5 }
        corrupt("twin dropped") { m in let t = m.heTwin[2]; m.heTwin[2] = -1; _ = t }
        corrupt("next swapped") { $0.heNext.swapAt(0, 1) }
        corrupt("prev broken") { $0.hePrev[4] = $0.hePrev[5] }
        corrupt("face id wrong") { $0.heFace[7] = ($0.heFace[7] + 1) % $0.faceHE.count }
        corrupt("origin changed") { $0.heOrigin[9] = ($0.heOrigin[9] + 1) % $0.positions.count }
        corrupt("directed table") { m in m.directed[m.directed.keys.first!] = nil }
        corrupt("vertex halfedge") { $0.vertexHE[3] = $0.vertexHE[4] }
        corrupt("vertex isolated") { $0.vertexHE[2] = -1 }
        corrupt("edge dead") { $0.edgeHE[5] = -1 }
        corrupt("edge shared wrongly") { $0.heEdge[1] = $0.heEdge[40] }
        corrupt("face pointer") { $0.faceHE[2] = $0.faceHE[3] }
        corrupt("array length") { $0.heUV.removeLast() }
        corrupt("loop does not close") { m in m.heNext[m.faceHE[1]] = m.faceHE[1] }
        var nan = good; nan.positions[0].x = .nan
        XCTAssertTrue(nan.validate().contains { $0.severity == .warning })
        XCTAssertTrue(nan.isValid)
        var flat = good; flat.positions = flat.positions.map { _ in Vec3(1, 2, 3) }
        XCTAssertTrue(flat.validate().contains { $0.message.contains("zero area") })
    }

    func testBowTieIsReportedNotRejected() {
        var m = HalfEdgeMesh()
        for p in [Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(-1,0,0), Vec3(0,-1,0)] { m.addVertex(p) }
        m.addFace([0, 1, 2]); m.addFace([0, 3, 4])
        XCTAssertEqual(m.nonManifoldVertices(), [0]); XCTAssertFalse(m.isManifold)
        XCTAssertTrue(m.isValid)
        XCTAssertTrue(m.validate().contains { $0.message.contains("bow-tie") })
    }
}
