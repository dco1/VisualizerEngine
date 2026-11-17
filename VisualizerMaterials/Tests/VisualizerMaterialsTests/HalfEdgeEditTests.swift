import XCTest
@testable import VisualizerMaterials

final class HalfEdgeEditTests: XCTestCase {

    // MARK: helpers

    private func box() -> HalfEdgeMesh { HalfEdgeMesh.box() }

    private func face(_ m: HalfEdgeMesh, _ n: Vec3) -> Int { m.faces.first { dot3(m.faceNormal($0), n) > 0.99 }! }
    private func faces(_ m: HalfEdgeMesh, _ n: Vec3, at coord: Int, _ value: Double) -> [Int] {
        m.faces.filter { dot3(m.faceNormal($0), n) > 0.99 && abs(m.faceCentroid($0)[coord] - value) < 1e-9 }
    }

    private func assertSound(_ m: HalfEdgeMesh, closed: Bool = true, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(m.validate().filter { $0.severity == .error }, [], file: file, line: line)
        XCTAssertEqual(m.validate().count, 0, "\(m.validate())", file: file, line: line)
        if closed { XCTAssertTrue(m.isClosed, "closed", file: file, line: line) }
    }

    /// Directed edges of the welded triangle list that lack their opposite or repeat.
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
    private func assertWatertight(_ m: HalfEdgeMesh, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(openEdges(m.toMesh3()), 0, "round trip to a watertight Mesh3", file: file, line: line)
    }
    /// Move the components of a split mesh away from each other (a split leaves two coincident caps).
    private func pullApart(_ m: inout HalfEdgeMesh) -> HalfEdgeMesh {
        var out = m
        for comp in m.connectedComponents() {
            let vs = Set(comp.flatMap { m.vertices(of: $0) })
            let side: Double = vs.reduce(0) { $0 + m.positions[$1].x } >= 0 ? 1 : -1
            for v in vs { out.positions[v].x += side }
        }
        return out
    }
    private func volume(_ m: HalfEdgeMesh) -> Double { m.signedVolume(of: m.faces) }
    private func area(_ m: HalfEdgeMesh) -> Double { m.faces.reduce(0) { $0 + m.faceArea($1) } }
    private func counts(_ m: HalfEdgeMesh) -> [Int] { [m.vertexCount, m.edgeCount, m.faceCount] }

    // MARK: extrude

    func testExtrudeBoxFaceVolume() {
        var m = box()
        let top = face(m, Vec3(0, 1, 0))
        let s = m.extrude(faces: m.selection(.face, [top]), distance: 0.5)!
        XCTAssertEqual(s.faces, [top], "the moved face keeps its id")
        assertSound(m)
        XCTAssertEqual(counts(m), [12, 20, 10])
        XCTAssertEqual(m.eulerCharacteristic, 2)
        XCTAssertEqual(volume(m), 1.5, accuracy: 1e-9)
        XCTAssertEqual(m.faceCentroid(top).y, 1.0, accuracy: 1e-9)
        assertWatertight(m)
    }

    func testExtrudeRegionSharesWallsAndIndividualDoesNot() {
        var region = box().subdividedQuads()
        let topFaces = faces(region, Vec3(0, 1, 0), at: 1, 0.5)
        XCTAssertEqual(topFaces.count, 4)
        let before = region.faceCount
        region.extrude(faces: region.selection(.face, topFaces), distance: 0.3)
        assertSound(region)
        XCTAssertEqual(region.faceCount - before, 8, "walls only on the 8 rim edges")
        XCTAssertEqual(volume(region), 1.3, accuracy: 1e-9)
        assertWatertight(region)

        var solo = box().subdividedQuads()
        let ids = faces(solo, Vec3(0, 1, 0), at: 1, 0.5)
        solo.extrude(faces: solo.selection(.face, ids), distance: 0.2, individual: true)
        assertSound(solo)
        XCTAssertEqual(solo.faceCount - before, 16, "four walls each")
        XCTAssertEqual(volume(solo), 1.2, accuracy: 1e-9)
    }

    func testExtrudeByVectorAndEdgesAndVertices() {
        var m = box()
        let right = face(m, Vec3(1, 0, 0))
        m.extrude(faces: m.selection(.face, [right]), vector: Vec3(0.4, 0, 0))
        XCTAssertEqual(volume(m), 1.4, accuracy: 1e-9)
        assertSound(m)

        // open quad: its boundary edge grows a strip
        var q = HalfEdgeMesh()
        let v = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0)].map { q.addVertex($0) }
        q.addFace(v)
        let bottom = q.edge(between: v[0], v[1])!
        let s = q.extrude(edges: q.selection(.edge, [bottom]), distance: 0.5)!
        XCTAssertEqual(q.faceCount, 2)
        XCTAssertEqual(area(q), 1.5, accuracy: 1e-9)
        XCTAssertEqual(s.edges.count, 1)
        XCTAssertTrue(q.isValid)
        XCTAssertLessThan(q.positions.map { $0.y }.min()!, -0.49, "the strip grows away from the face")

        // a spike out of a closed mesh stays closed
        var b = box().subdividedQuads()
        let centre = b.vertices.first { len3(b.positions[$0] - Vec3(0, 0, 0.5)) < 1e-9 }!
        let tip = b.extrude(vertices: b.selection(.vertex, [centre]), distance: 0.4)!
        XCTAssertEqual(tip.faces.count, 1)
        assertSound(b)
        XCTAssertEqual(b.eulerCharacteristic, 2)
        XCTAssertGreaterThan(volume(b), 1.0)
        assertWatertight(b)
    }

    // MARK: push / pull

    func testPushPullChangesVolumeByAreaTimesDistance() {
        for mode in [HENormalMode.averaged, .perFace] {
            var m = box()
            let right = face(m, Vec3(1, 0, 0))
            m.pushPull(m.selection(.face, [right]), distance: 0.25, normals: mode)
            XCTAssertEqual(volume(m), 1.25, accuracy: 1e-9)
            m.pushPull(m.selection(.face, [right]), distance: -0.5, normals: mode)
            XCTAssertEqual(volume(m), 0.75, accuracy: 1e-9)
            assertSound(m)
            XCTAssertEqual(len3(m.heNormal[m.halfedge(of: right)] - Vec3(1, 0, 0)), 0, accuracy: 1e-9)
        }
        // a whole subdivided top patch
        var m = box().subdividedQuads()
        m.pushPull(m.selection(.face, faces(m, Vec3(0, 1, 0), at: 1, 0.5)), distance: 0.2)
        XCTAssertEqual(volume(m), 1.2, accuracy: 1e-9)
    }

    func testPushPullAveragedVersusPerFaceAtACorner() {
        // two faces meeting at 90 degrees: the shared vertices move diagonally; per-face mitres keep d on each face
        let top = HalfEdgeMesh.box()
        let sel = top.selection(.face, [face(top, Vec3(0, 1, 0)), face(top, Vec3(1, 0, 0))])
        var avg = top, mit = top
        avg.pushPull(sel, distance: 0.1, normals: .averaged)
        mit.pushPull(sel, distance: 0.1, normals: .perFace)
        let shared = 7
        let da = len3(avg.positions[shared] - top.positions[shared]), dm = len3(mit.positions[shared] - top.positions[shared])
        XCTAssertEqual(da, 0.1, accuracy: 1e-9)
        XCTAssertEqual(dm, 0.1 * 2.0.squareRoot(), accuracy: 1e-9)
    }

    func testPushPullAndSmoothGPUMatchCPU() throws {
        guard HalfEdgeEditGPU.shared != nil else { throw XCTSkip("no Metal device") }
        var big = box()
        for _ in 0..<5 { big = big.subdividedQuads() }
        XCTAssertGreaterThan(big.vertexCount, HEEdit.gpuThreshold)
        let all = big.selectAll(.face)
        var cpu = big, gpu = big
        cpu.pushPull(all, distance: 0.02, normals: .perFace, useGPU: false)
        gpu.pushPull(all, distance: 0.02, normals: .perFace, useGPU: true)
        for v in big.vertices { XCTAssertLessThan(len3(cpu.positions[v] - gpu.positions[v]), 1e-5) }

        // roughen, then smooth both ways
        var rough = cpu
        var seed: UInt64 = 7
        for v in rough.vertices { seed = seed &* 6364136223846793005 &+ 1442695040888963407
            rough.positions[v] += Vec3(Double(seed >> 40 & 0xff) / 255 - 0.5, Double(seed >> 32 & 0xff) / 255 - 0.5, Double(seed >> 24 & 0xff) / 255 - 0.5) * 0.004 }
        let verts = rough.selectAll(.vertex)
        var a = rough, b = rough
        a.smoothVertices(verts, iterations: 6, lambda: 0.5, useGPU: false)
        b.smoothVertices(verts, iterations: 6, lambda: 0.5, useGPU: true)
        var worst = 0.0
        for v in rough.vertices { worst = max(worst, len3(a.positions[v] - b.positions[v])) }
        XCTAssertLessThan(worst, 1e-5)
        assertSound(b)
    }

    func testSmoothReducesNoiseAndPinsBoundary() {
        var m = box().subdividedQuads().subdividedQuads()
        m.deleteFace(face(m, Vec3(0, 1, 0)))
        m.repairVertexLinks()
        let before = m
        var seed: UInt64 = 3
        for v in m.vertices { seed = seed &* 6364136223846793005 &+ 1; m.positions[v].y += Double(seed >> 40 & 0xff) / 255 * 0.01 }
        func rough(_ x: HalfEdgeMesh) -> Double {
            x.vertices.reduce(0) { acc, v in acc + len3(x.neighbours(of: v).reduce(Vec3.zero) { $0 + x.positions[$1] } / Double(x.neighbours(of: v).count) - x.positions[v]) }
        }
        let r0 = rough(m)
        let pinned = Set(m.vertices.filter { m.isBoundary(vertex: $0) })
        let at = Dictionary(uniqueKeysWithValues: pinned.map { ($0, m.positions[$0]) })
        m.smoothVertices(m.selectAll(.vertex), iterations: 4, lambda: 0.6)
        XCTAssertLessThan(rough(m), r0)
        for v in pinned { XCTAssertEqual(len3(m.positions[v] - at[v]!), 0, accuracy: 1e-12) }
        XCTAssertEqual(counts(m), counts(before))
        XCTAssertTrue(m.isValid)
    }

    // MARK: inset

    func testInsetAreaAndDepth() {
        var m = box()
        let top = face(m, Vec3(0, 1, 0))
        let s = m.inset(m.selection(.face, [top]), thickness: 0.1)!
        assertSound(m)
        XCTAssertEqual(counts(m), [12, 20, 10])
        XCTAssertEqual(m.faceArea(s.faces.first!), 0.64, accuracy: 1e-9)
        XCTAssertEqual(area(m), 6, accuracy: 1e-9, "a flat inset keeps the surface area")
        var d = box()
        let t2 = face(d, Vec3(0, 1, 0))
        let s2 = d.inset(d.selection(.face, [t2]), thickness: 0.2, depth: -0.25)!
        XCTAssertEqual(d.faceCentroid(s2.faces.first!).y, 0.25, accuracy: 1e-9)
        XCTAssertEqual(volume(d), 1 - 0.25 / 3 * (1 + 0.36 + 0.6), accuracy: 1e-9, "a frustum sunk out of the top")
        assertSound(d); assertWatertight(d)
    }

    func testInsetRegionKeepsInnerFacesJoinedAndIndividualDoesNot() {
        var region = box().subdividedQuads()
        let top = faces(region, Vec3(0, 1, 0), at: 1, 0.5)
        let s = region.inset(region.selection(.face, top), thickness: 0.1)!
        assertSound(region)
        XCTAssertEqual(s.faces.count, 4)
        XCTAssertEqual(s.faces.reduce(0) { $0 + region.faceArea($1) }, 0.64, accuracy: 1e-9, "one 0.8 x 0.8 patch")
        XCTAssertEqual(region.faceCount, 24 + 8, "a ring of 8 quads round the patch")

        var solo = box().subdividedQuads()
        let ids = faces(solo, Vec3(0, 1, 0), at: 1, 0.5)
        let t = solo.inset(solo.selection(.face, ids), thickness: 0.05, individual: true)!
        assertSound(solo)
        XCTAssertEqual(t.faces.reduce(0) { $0 + solo.faceArea($1) }, 4 * 0.16, accuracy: 1e-9)
        XCTAssertEqual(solo.faceCount, 24 + 16)
        XCTAssertEqual(area(solo), 6, accuracy: 1e-9)
    }

    // MARK: loop cut / slide

    func testLoopCutCountsOnABox() {
        var m = box()
        let e = m.edge(between: 0, 1)!                        // along x
        let s = m.loopCut(ringFrom: e)!
        assertSound(m)
        XCTAssertEqual(counts(m), [12, 20, 10])
        XCTAssertEqual(m.eulerCharacteristic, 2)
        XCTAssertEqual(s.edges.count, 4)
        for ed in s.edges { let (a, b) = m.vertices(ofEdge: ed); XCTAssertEqual(m.positions[a].x, 0, accuracy: 1e-12); XCTAssertEqual(m.positions[b].x, 0, accuracy: 1e-12) }
        XCTAssertTrue(m.faces.allSatisfy { m.valence(ofFace: $0) == 4 })
        XCTAssertEqual(volume(m), 1, accuracy: 1e-12)
        assertWatertight(m)

        var three = box()
        three.loopCut(ringFrom: three.edge(between: 0, 1)!, cuts: 3)
        assertSound(three)
        XCTAssertEqual(counts(three), [20, 36, 18])
        XCTAssertEqual(three.eulerCharacteristic, 2)
        let xs = Set(three.vertices.map { ($0, three.positions[$0].x) }.filter { abs($0.1) < 0.4999 }.map { ($0.1 * 1000).rounded() / 1000 })
        XCTAssertEqual(xs, [-0.25, 0, 0.25])
    }

    func testLoopCutSlideMovesAllCutsTheSameWay() {
        var m = box()
        m.loopCut(ringFrom: m.edge(between: 0, 1)!, cuts: 1, slide: 0.5)
        assertSound(m)
        let xs = m.vertices.map { m.positions[$0].x }.filter { abs($0) < 0.4999 }
        XCTAssertEqual(xs.count, 4)
        XCTAssertEqual(xs.map { abs($0) }.max()!, xs.map { abs($0) }.min()!, accuracy: 1e-12)
        XCTAssertEqual(Set(xs.map { $0 > 0 }).count, 1, "all four on one side of the middle")
        XCTAssertEqual(abs(xs[0]), 0.25, accuracy: 1e-12)
    }

    func testEdgeSlideMovesALoopAlongItsNeighbours() {
        var m = box()
        let cut = m.loopCut(ringFrom: m.edge(between: 0, 1)!)!
        let before = m.positions
        m.slideEdges(cut, factor: 0.5)
        XCTAssertTrue(m.isValid)
        let moved = m.vertices.filter { len3(m.positions[$0] - before[$0]) > 1e-12 }
        XCTAssertEqual(moved.count, 4)
        XCTAssertEqual(Set(moved.map { (m.positions[$0].x * 1000).rounded() / 1000 }).count, 1, "the whole loop slides to one x")
        XCTAssertEqual(abs(m.positions[moved[0]].x), 0.25, accuracy: 1e-12)
        let other = m.vertices.filter { len3(m.positions[$0] - before[$0]) > 1e-12 }
        XCTAssertEqual(other.count, 4)
        assertWatertight(m)
    }

    // MARK: bridge / fill

    private func tube() -> HalfEdgeMesh {
        var m = box()
        m.deleteFaces([face(m, Vec3(0, 1, 0)), face(m, Vec3(0, -1, 0))])
        m.repairVertexLinks()
        return m
    }

    func testBridgeTwoLoopsMakesATorus() {
        var m = tube()
        XCTAssertEqual(m.boundaryLoops().count, 2)
        let sel = m.selection(.edge, m.boundaryEdges())
        XCTAssertNil(m.bridge(sel), "straight across duplicates the tube's own edges")
        let s = m.bridge(sel, twist: 1)!
        assertSound(m)
        XCTAssertEqual(counts(m), [8, 16, 8])
        XCTAssertEqual(m.eulerCharacteristic, 0, "genus 1")
        XCTAssertEqual(s.faces.count, 4)

        var cut = tube()
        let s2 = cut.bridge(cut.selection(.edge, cut.boundaryEdges()), cuts: 2)!
        assertSound(cut)
        XCTAssertEqual(cut.eulerCharacteristic, 0)
        XCTAssertEqual(counts(cut), [16, 32, 16])
        // bow the band out so its surface stands clear of the tube, then it must still be watertight
        for v in s2.vertices where abs(cut.positions[v].y) < 0.49 { cut.positions[v] = Vec3(cut.positions[v].x * 1.6, cut.positions[v].y, cut.positions[v].z * 1.6) }
        assertWatertight(cut)
        var bad = tube()
        XCTAssertNil(bad.bridge(bad.selection(.edge, [bad.boundaryEdges()[0]])))
        XCTAssertEqual(counts(bad), counts(tube()))
    }

    func testFillHoleModesAndDeleteAndFill() {
        for (mode, faces) in [(HEFillMode.ngon, 6), (.triangles, 7), (.grid, 6)] {
            var m = box()
            m.deleteFace(face(m, Vec3(0, 1, 0))); m.repairVertexLinks()
            let s = m.fillHoles(m.selection(.edge, m.boundaryEdges()), mode: mode)!
            assertSound(m)
            XCTAssertEqual(m.faceCount, faces, "\(mode)")
            XCTAssertEqual(volume(m), 1, accuracy: 1e-12)
            XCTAssertEqual(s.faces.count, faces - 5)
            assertWatertight(m)
        }
        for (mode, n) in [(HEFillMode.ngon, 1), (.triangles, 6), (.grid, 4)] {
            var m = box().subdividedQuads()
            let top = faces(m, Vec3(0, 1, 0), at: 1, 0.5)
            m.deleteAndFill(m.selection(.face, top), mode: mode)
            assertSound(m)
            XCTAssertEqual(m.faceCount, 20 + n, "\(mode)")
            XCTAssertEqual(m.eulerCharacteristic, 2)
            XCTAssertEqual(volume(m), 1, accuracy: 1e-12)
            assertWatertight(m)
        }
    }

    // MARK: knife

    func testKnifeCutsAndSplitsIntoClosedHalves() {
        var m = box()
        let cut = m.knife(planeOrigin: Vec3(0.1, 0, 0), normal: Vec3(1, 0, 0))!
        assertSound(m)
        XCTAssertEqual(counts(m), [12, 20, 10])
        XCTAssertEqual(cut.edges.count, 4)
        XCTAssertEqual(volume(m), 1, accuracy: 1e-12)
        assertWatertight(m)

        for subdivided in [false, true] {
            var s = subdivided ? box().subdividedQuads() : box()
            let caps = s.knife(planeOrigin: Vec3(0, 0.0, 0), normal: Vec3(1, 0, 0), split: true)!
            assertSound(s)
            let parts = s.connectedComponents()
            XCTAssertEqual(parts.count, 2)
            for p in parts { XCTAssertTrue(s.isClosedComponent(p)); XCTAssertEqual(abs(s.signedVolume(of: p)), 0.5, accuracy: 1e-9) }
            XCTAssertEqual(caps.faces.count, 2)
            XCTAssertEqual(s.eulerCharacteristic, 4, "two spheres")
            let apart = pullApart(&s)
            XCTAssertEqual(openEdges(apart.toMesh3()), 0, "each half round-trips to a watertight Mesh3")
        }
    }

    func testKnifeAtAnAngleThroughATriangulatedSphereLikeShape() {
        var m = box().subdividedQuads()
        m.knife(planeOrigin: Vec3(0, 0, 0.1), normal: normalize3(Vec3(0.3, 0.2, 1)))
        XCTAssertTrue(m.isValid)
        XCTAssertEqual(m.eulerCharacteristic, 2)
        XCTAssertEqual(volume(m), 1, accuracy: 1e-12)
    }

    // MARK: spin / lathe

    func testLatheTorusAndCone() {
        let profile = [Vec3(1, 0, 0), Vec3(2, 0, 0), Vec3(2, 1, 0), Vec3(1, 1, 0)]
        let n = 48
        let torus = HalfEdgeMesh.lathe(profile: profile, closed: true, steps: n)!
        assertSound(torus)
        XCTAssertEqual(torus.eulerCharacteristic, 0)
        XCTAssertEqual(torus.vertexCount, 4 * n)
        XCTAssertEqual(volume(torus), Double(n) / 2 * sin(2 * .pi / Double(n)) * 3, accuracy: 1e-9)
        assertWatertight(torus)

        let cone = HalfEdgeMesh.lathe(profile: [Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 0)], steps: 24)!
        assertSound(cone)
        XCTAssertEqual(cone.eulerCharacteristic, 2)
        XCTAssertEqual(cone.vertexCount, 26)
        XCTAssertGreaterThan(volume(cone), 0)
        assertWatertight(cone)
    }

    func testSpinAFaceAlongAnArc() {
        var m = box()
        let right = face(m, Vec3(1, 0, 0))
        let s = m.spin(m.selection(.face, [right]), axisOrigin: Vec3(0.5, 2, 0), axisDirection: Vec3(0, 0, 1), angle: 0.4, steps: 32)!
        assertSound(m)
        XCTAssertEqual(s.faces, [right])
        XCTAssertEqual(m.eulerCharacteristic, 2)
        XCTAssertEqual(volume(m), 1.8, accuracy: 0.02)
        assertWatertight(m)
        var full = box()
        XCTAssertNil(full.spin(full.selection(.face, [face(full, Vec3(1, 0, 0))]), axisOrigin: Vec3(0.5, 2, 0), axisDirection: Vec3(0, 0, 1), angle: 2 * .pi, steps: 8))
    }

    // MARK: dissolve / merge

    func testDissolveReturnsALoopCutToTheBox() {
        var m = box()
        let cut = m.loopCut(ringFrom: m.edge(between: 0, 1)!)!
        let r = m.dissolve(cut)!
        assertSound(m)
        XCTAssertEqual(counts(m), [8, 12, 6])
        XCTAssertEqual(r.faces.count, 4)
        XCTAssertEqual(volume(m), 1, accuracy: 1e-12)
        assertWatertight(m)

        var e = box()
        let ed = e.edge(between: 0, 1)!
        e.dissolve(e.selection(.edge, [ed]), cleanVertices: false)
        assertSound(e)
        XCTAssertEqual(e.faceCount, 5)
        XCTAssertTrue(e.faces.contains { e.valence(ofFace: $0) == 6 })
        assertWatertight(e)
    }

    func testDissolveFacesAndVertices() {
        var m = box().subdividedQuads()
        m.dissolve(m.selection(.face, faces(m, Vec3(0, 1, 0), at: 1, 0.5)))
        assertSound(m)
        XCTAssertEqual(m.faceCount, 21)
        XCTAssertEqual(m.faces.filter { m.valence(ofFace: $0) == 8 }.count, 1)
        assertWatertight(m)

        var v = box().subdividedQuads()
        let centre = v.vertices.first { len3(v.positions[$0] - Vec3(0, 0.5, 0)) < 1e-9 }!
        v.dissolve(v.selection(.vertex, [centre]))
        assertSound(v)
        XCTAssertEqual(v.faceCount, 21)
        assertWatertight(v)
    }

    func testMergeAndMergeByDistance() {
        var m = box()
        let top = face(m, Vec3(0, 1, 0))
        m.extrude(faces: m.selection(.face, [top]), distance: 0)
        XCTAssertEqual(m.faceCount, 10)
        m.mergeByDistance(distance: 1e-6)
        assertSound(m)
        XCTAssertEqual(counts(m), [8, 12, 6])
        XCTAssertEqual(volume(m), 1, accuracy: 1e-12)

        var c = box()
        let s = c.merge(c.selection(.vertex, [0, 1]), at: .center)!
        assertSound(c)
        XCTAssertEqual(counts(c), [7, 11, 6])
        XCTAssertEqual(c.positions[s.vertices.first!], Vec3(0, -0.5, -0.5))
        var f = box()
        f.merge(f.selection(.vertex, [3, 7]), at: .first)
        XCTAssertEqual(f.positions[3], Vec3(-0.5 + 1, 0.5, -0.5))
        XCTAssertTrue(f.isValid)
    }

    // MARK: bevel

    func testBevelOneEdge() {
        var m = box()
        let e = m.edge(between: 7, 3)!
        let w = 0.2
        let s = m.bevel(m.selection(.edge, [e]), width: w)!
        assertSound(m)
        XCTAssertEqual(counts(m), [10, 15, 7])
        XCTAssertEqual(m.eulerCharacteristic, 2)
        XCTAssertEqual(s.faces.count, 1)
        XCTAssertEqual(volume(m), 1 - w * w / 2, accuracy: 1e-9)
        assertWatertight(m)
    }

    func testBevelATopRingAndATriple() {
        var m = box()
        let top = face(m, Vec3(0, 1, 0))
        let s = m.bevel(m.selection(.edge, m.edges(of: top)), width: 0.1)!
        assertSound(m)
        XCTAssertEqual(counts(m), [12, 20, 10])
        XCTAssertEqual(s.faces.count, 4)
        XCTAssertLessThan(volume(m), 1)
        XCTAssertGreaterThan(volume(m), 1 - 4 * 0.005)
        assertWatertight(m)

        var all = box()
        all.bevel(all.selectAll(.edge), width: 0.1)
        assertSound(all)
        XCTAssertEqual(all.eulerCharacteristic, 2)
        XCTAssertEqual(all.faceCount, 6 + 12 + 8, "faces, edge strips, corner triangles")
        assertWatertight(all)

        var open = tube()
        XCTAssertNil(open.bevel(open.selection(.edge, [open.boundaryEdges()[0]]), width: 0.1))
    }

    // MARK: every tool

    func testEveryToolLeavesAValidWatertightMesh() {
        var results: [(String, HalfEdgeMesh)] = []
        func run(_ name: String, _ body: (inout HalfEdgeMesh) -> Void) {
            var m = box().subdividedQuads(); body(&m); results.append((name, name.contains("split") ? pullApart(&m) : m))
        }
        let topOf: (HalfEdgeMesh) -> [Int] = { self.faces($0, Vec3(0, 1, 0), at: 1, 0.5) }
        run("extrude") { $0.extrude(faces: $0.selection(.face, topOf($0)), distance: 0.2) }
        run("extrude individual") { $0.extrude(faces: $0.selection(.face, topOf($0)), distance: 0.1, individual: true) }
        run("pushpull") { $0.pushPull($0.selection(.face, topOf($0)), distance: 0.1) }
        run("inset") { $0.inset($0.selection(.face, topOf($0)), thickness: 0.08, depth: 0.05) }
        run("loopcut") { $0.loopCut(ringFrom: $0.edge(between: 0, $0.neighbours(of: 0)[0])!, cuts: 2, slide: 0.3) }
        run("knife") { $0.knife(planeOrigin: Vec3(0.07, 0, 0), normal: Vec3(1, 0, 0)) }
        run("knife split") { $0.knife(planeOrigin: Vec3(0.07, 0, 0), normal: Vec3(1, 0, 0), split: true) }
        run("smooth") { $0.smoothVertices($0.selectAll(.vertex), iterations: 3) }
        run("dissolve faces") { $0.dissolve($0.selection(.face, topOf($0))) }
        run("delete fill") { $0.deleteAndFill($0.selection(.face, topOf($0)), mode: .grid) }
        run("merge distance") { $0.mergeByDistance(distance: 0.01) }
        run("bevel") { $0.bevel($0.selection(.edge, $0.edges(of: topOf($0)[0])), width: 0.05) }
        for (name, m) in results {
            XCTAssertEqual(m.validate().count, 0, "\(name): \(m.validate())")
            XCTAssertTrue(m.isClosed, name)
            if name != "extrude individual" {          // its neighbouring prisms share coincident walls
                XCTAssertEqual(openEdges(m.toMesh3()), 0, "\(name) round-trips to a watertight Mesh3")
            }
        }
    }

    func testRefusedToolsLeaveTheMeshUntouched() {
        var m = box()
        let before = m
        XCTAssertNil(m.extrude(faces: m.selection(.face, []), distance: 1))
        XCTAssertNil(m.bevel(m.selection(.edge, []), width: 0.1))
        XCTAssertNil(m.loopCut(ringFrom: 9999))
        XCTAssertNil(m.knife(planeOrigin: Vec3(5, 0, 0), normal: Vec3(1, 0, 0)))
        XCTAssertEqual(m.positions, before.positions)
        XCTAssertEqual(counts(m), counts(before))
    }
}
