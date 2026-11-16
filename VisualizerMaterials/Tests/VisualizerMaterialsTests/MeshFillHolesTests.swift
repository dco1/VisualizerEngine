import XCTest
@testable import VisualizerMaterials

final class MeshFillHolesTests: XCTestCase {

    /// A sphere with every triangle touching a cap around +Y (y > 0.8) removed.
    private func cutSphere() -> Mesh3 {
        let s = MeshKit.icosphere(levels: 3)
        var m = s
        m.indices = []
        var t = 0
        while t + 2 < s.indices.count {
            let c = (0 ..< 3).reduce(0.0) { $0 + s.positions[Int(s.indices[t + $1])].y } / 3
            if c < 0.8 { m.indices += [s.indices[t], s.indices[t + 1], s.indices[t + 2]] }
            t += 3
        }
        return m
    }

    func testFillClosesCutSphere() {
        let cut = cutSphere()
        XCTAssertGreaterThan(MeshKit.openEdges(cut), 0)
        let filled = cut.filledHoles(maxEdges: 64)
        XCTAssertEqual(MeshKit.openEdges(filled), 0)
        XCTAssertGreaterThan(filled.triangleCount, cut.triangleCount)
        XCTAssertGreaterThan(MeshKit.volume(filled), 0, "patch wound the right way (positive volume)")
        XCTAssertEqual(MeshKit.volume(filled), 4.0 / 3 * .pi, accuracy: 0.3)
    }

    func testLoopLongerThanMaxIsLeftOpen() {
        let cut = cutSphere()
        let same = cut.filledHoles(maxEdges: 4)
        XCTAssertGreaterThan(MeshKit.openEdges(same), 0)
    }

    func testClosedMeshUnchanged() {
        let s = MeshKit.icosphere(levels: 2)
        XCTAssertEqual(s.filledHoles(), s)
    }
}
