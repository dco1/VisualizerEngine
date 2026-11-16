import XCTest
@testable import VisualizerMaterials

final class MeshWeldTests: XCTestCase {

    /// A box built as a soup, then jittered by < 1e-5 so positions are near, not equal.
    private func jitteredBox() -> Mesh3 {
        var m = MeshKit.box(n: 1)
        var s: UInt64 = 7
        func r() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 40) / Double(1 << 24) - 0.5 }
        m.positions = m.positions.map { $0 + Vec3(r(), r(), r()) * 1e-5 }
        return m
    }

    func testWeldClosesSoupCube() {
        let soup = jitteredBox()
        XCTAssertEqual(soup.vertexCount, 36)
        let w = soup.welded(distance: 1e-4)
        XCTAssertEqual(w.vertexCount, 8)
        XCTAssertEqual(w.triangleCount, 12)
        XCTAssertEqual(MeshKit.openEdges(w), 0)
        XCTAssertEqual(MeshKit.volume(w), 1, accuracy: 1e-3)
        w.normals.forEach { XCTAssertEqual(len3($0), 1, accuracy: 1e-9) }
    }

    func testWeldKeepsHardEdgesWhenAskedAndSeams() {
        let soup = jitteredBox()
        let hard = soup.welded(distance: 1e-4, normalAngleDegrees: 30)
        XCTAssertEqual(hard.vertexCount, 24, "flat-shaded cube keeps one vertex per face corner")
        // a plane with a UV seam down the middle: coincident vertices differing in UV stay split
        var p = MeshKit.plane(n: 2)
        p.uvs = p.uvs.map { $0 }
        let flatSoup = p.welded(distance: 1e-6)
        XCTAssertEqual(flatSoup.vertexCount, p.vertexCount)
        var s = Mesh3()
        s.addTriangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1), outward: Vec3(0, 1, 0), uv: (Vec2(0, 0), Vec2(1, 0), Vec2(0, 1)))
        s.addTriangle(Vec3(1, 0, 0), Vec3(1, 0, 1), Vec3(0, 0, 1), outward: Vec3(0, 1, 0), uv: (Vec2(0.5, 0), Vec2(1, 1), Vec2(0.5, 1)))
        XCTAssertEqual(s.welded(distance: 1e-6).vertexCount, 6, "UV seam survives")
        XCTAssertEqual(s.welded(distance: 1e-6, uvTolerance: 1).vertexCount, 4)
    }

    func testWeldDropsDegenerateTriangles() {
        var m = Mesh3()
        m.positions = [Vec3(0, 0, 0), Vec3(1e-9, 0, 0), Vec3(0, 0, 1), Vec3(1, 0, 0)]
        m.indices = [0, 1, 2, 0, 3, 2]
        let w = m.welded(distance: 1e-6)
        XCTAssertEqual(w.vertexCount, 3)
        XCTAssertEqual(w.triangleCount, 1)
    }
}
