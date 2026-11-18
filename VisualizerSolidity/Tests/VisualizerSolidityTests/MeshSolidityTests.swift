import XCTest
import VisualizerMaterials
@testable import VisualizerSolidity

final class MeshSolidityTests: XCTestCase {
    private func mesh(_ p: [Vec3], _ i: [UInt32]) -> Mesh3 { Mesh3(positions: p, normals: [], uvs: [], indices: i) }

    /// Outward-wound tetrahedron.
    private var tetra: Mesh3 {
        mesh([Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0)],
             [0, 1, 2,  0, 3, 1,  1, 3, 2,  2, 3, 0].enumerated().map { $0.element })
    }

    /// An upward unit quad at height `y` (two triangles), offset in x.
    private func floor(y: Double, x: Double = 0) -> Mesh3 {
        mesh([Vec3(x, y, 0), Vec3(x, y, 1), Vec3(x + 1, y, 1), Vec3(x + 1, y, 0)], [0, 1, 2, 0, 2, 3])
    }

    func testClosedSolidPassesAndOpenSheetFails() {
        XCTAssertTrue(MeshSolidity.closedSolidFailures([floor(y: 0)], label: "sheet").count == 1)
        XCTAssertTrue(MeshSolidity.closedSolidFailures([], label: "none").isEmpty)
    }

    func testCoplanarSameFacingFacesOfDifferentPartsFight() {
        XCTAssertFalse(MeshSolidity.zFightFailures([floor(y: 0.5), floor(y: 0.5)], label: "two", cameraPitchDegrees: 35).isEmpty)
        XCTAssertTrue(MeshSolidity.zFightFailures([floor(y: 0.5), floor(y: 0.6)], label: "apart", cameraPitchDegrees: 35).isEmpty)
        XCTAssertTrue(MeshSolidity.zFightFailures([floor(y: 0.5), floor(y: 0.5, x: 2)], label: "beside", cameraPitchDegrees: 35).isEmpty)
    }

    func testUndersidesAreNeverVisible() {
        XCTAssertFalse(MeshSolidity.everVisible([0, -1, 0], cameraPitchDegrees: 35))
        XCTAssertTrue(MeshSolidity.everVisible([0, 1, 0], cameraPitchDegrees: 35))
    }
}
