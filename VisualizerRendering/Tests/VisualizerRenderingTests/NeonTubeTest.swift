import XCTest
import simd
import VisualizerCore
@testable import VisualizerRendering

@MainActor
final class NeonTubeTest: XCTestCase {
    func testNeonTubeWindingAndCounts() {
        // An L-shaped path with a turn — exercises the parallel-transport frame.
        let path: [SIMD3<Float>] = [
            SIMD3(-1, 0, 0), SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 2, 0)
        ]
        let built = NeonTube.buildGeometry(path: path, radius: 0.07,
                                           radialSegments: 12, closed: false)
        XCTAssertGreaterThan(built.positions.count, 0)
        XCTAssertEqual(built.positions.count, built.normals.count)
        XCTAssertEqual(built.indices.count % 3, 0)
        let frac = GeometryWinding.audit(positions: built.positions, normals: built.normals,
                                         indices: built.indices.map { UInt32(bitPattern: $0) },
                                         label: "NeonTube-test")
        print("NEON_INSIDE_OUT_FRACTION = \(frac)")
        XCTAssertLessThan(frac, 0.2, "NeonTube winding inverted (\(frac*100)% inside-out)")
    }
}
