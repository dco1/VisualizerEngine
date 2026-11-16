import XCTest
@testable import VisualizerMaterials

final class MeshDecimateTests: XCTestCase {

    func testSphere20kTo2kKeepsVolumeAndIsWatertight() {
        let sphere = MeshKit.icosphere(levels: 5)
        XCTAssertEqual(sphere.triangleCount, 20480)
        let v0 = MeshKit.volume(sphere)
        let t0 = Date()
        let (d, report) = sphere.decimate(targetTriangles: 2000)
        let dt = Date().timeIntervalSince(t0)
        print("decimate 20480 -> \(d.triangleCount) tris in \(dt)s, maxError \(report.maxError), collapses \(report.collapses)")
        XCTAssertLessThanOrEqual(d.triangleCount, 2010)
        XCTAssertGreaterThan(d.triangleCount, 1500)
        XCTAssertEqual(MeshKit.volume(d) / v0, 1, accuracy: 0.02)
        XCTAssertEqual(MeshKit.openEdges(d), 0, "watertight and consistently wound")
        // surface stays near the sphere
        let worst = d.positions.map { abs(len3($0) - 1) }.max()!
        XCTAssertLessThan(worst, 0.02)
        XCTAssertGreaterThan(report.maxError, 0)
        XCTAssertLessThan(report.maxError, 0.05)
        XCTAssertEqual(report.triangles, d.triangleCount)
    }

    func testRatioAPI() {
        let sphere = MeshKit.icosphere(levels: 3)
        let d = sphere.decimated(ratio: 0.25)
        XCTAssertEqual(Double(d.triangleCount), 320, accuracy: 20)
        XCTAssertEqual(MeshKit.openEdges(d), 0)
    }

    func testOpenPlaneKeepsBoundary() {
        let plane = MeshKit.plane(n: 16)
        let d = plane.decimated(targetTriangles: 40)
        XCTAssertLessThan(d.triangleCount, 60)
        XCTAssertEqual(MeshKit.area(d), 1, accuracy: 1e-9)
        let b = d.bounds!
        XCTAssertEqual(b.min, Vec3(0, 0, 0)); XCTAssertEqual(b.max, Vec3(1, 0, 1))
        // every boundary vertex lies on the square's perimeter, and all four corners are still there
        let bv = MeshKit.boundaryVertices(d)
        XCTAssertFalse(bv.isEmpty)
        for i in bv {
            let p = d.positions[i]
            XCTAssertTrue(p.x < 1e-12 || p.x > 1 - 1e-12 || p.z < 1e-12 || p.z > 1 - 1e-12, "boundary vertex left the boundary: \(p)")
        }
        for corner in [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1), Vec3(1, 0, 1)] {
            XCTAssertTrue(d.positions.contains { len3($0 - corner) < 1e-12 }, "corner \(corner) lost")
        }
        // interior is still a facing-up plane
        d.normals.forEach { XCTAssertEqual($0.y, 1, accuracy: 1e-9) }
    }

    func testCreaseBoxStaysABox() {
        let box = MeshKit.box(n: 8)
        XCTAssertEqual(box.triangleCount, 768)
        let (d, report) = box.decimate(targetTriangles: 48)
        XCTAssertLessThanOrEqual(d.triangleCount, 60)
        XCTAssertEqual(MeshKit.volume(d), 1, accuracy: 1e-9)
        XCTAssertEqual(MeshKit.area(d), 6, accuracy: 1e-9)
        XCTAssertEqual(MeshKit.openEdges(d.welded(distance: 1e-9)), 0)
        XCTAssertLessThan(report.maxError, 1e-6, "flat faces and straight creases cost nothing")
        let b = d.bounds!
        XCTAssertEqual(b.min, Vec3(0, 0, 0)); XCTAssertEqual(b.max, Vec3(1, 1, 1))
        for c in 0 ..< 8 {
            let corner = Vec3(Double(c & 1), Double((c >> 1) & 1), Double((c >> 2) & 1))
            XCTAssertTrue(d.positions.contains { len3($0 - corner) < 1e-12 }, "corner \(corner) lost")
        }
        // every vertex sits on the cube's surface; every triangle is axis aligned
        for p in d.positions { XCTAssertTrue((0 ..< 3).contains { p[$0] < 1e-12 || p[$0] > 1 - 1e-12 }) }
        var t = 0
        while t + 2 < d.indices.count {
            let a = d.positions[Int(d.indices[t])]
            let n = normalize3(cross3(d.positions[Int(d.indices[t + 1])] - a, d.positions[Int(d.indices[t + 2])] - a))
            XCTAssertGreaterThan(max(abs(n.x), abs(n.y), abs(n.z)), 0.999999)
            t += 3
        }
    }

    func testUVSeamIsKept() {
        // A plane cut by a UV seam down x = 0.5 (U jumps by 0.25 across it).
        var m = MeshKit.plane(n: 8)
        var seam = Mesh3()
        for t in 0 ..< m.triangleCount {
            let i = (0 ..< 3).map { Int(m.indices[t * 3 + $0]) }
            let cx = (m.positions[i[0]].x + m.positions[i[1]].x + m.positions[i[2]].x) / 3
            let off = cx > 0.5 ? 0.25 : 0
            seam.addTriangle(m.positions[i[0]], m.positions[i[1]], m.positions[i[2]], outward: Vec3(0, 1, 0),
                             uv: (m.uvs[i[0]] + Vec2(off, 0), m.uvs[i[1]] + Vec2(off, 0), m.uvs[i[2]] + Vec2(off, 0)))
        }
        m = seam
        let d = m.decimated(targetTriangles: 24)
        // the seam is still a straight line of position-coincident, UV-split vertices
        var onSeamUV: [Double] = []
        for (i, p) in d.positions.enumerated() where abs(p.x - 0.5) < 1e-9 { onSeamUV.append(d.uvs[i].x) }
        XCTAssertTrue(onSeamUV.contains { abs($0 - 0.5) < 1e-6 } && onSeamUV.contains { abs($0 - 0.75) < 1e-6 }, "seam uv split lost: \(onSeamUV)")
        // UVs still match positions on each side (u = x or x + 0.25)
        for (i, p) in d.positions.enumerated() {
            let du = d.uvs[i].x - p.x
            XCTAssertTrue(abs(du) < 1e-6 || abs(du - 0.25) < 1e-6, "uv drifted \(du) at \(p)")
            XCTAssertEqual(d.uvs[i].y, p.z, accuracy: 1e-6)
        }
        XCTAssertEqual(MeshKit.area(d), 1, accuracy: 1e-9)
    }
}
