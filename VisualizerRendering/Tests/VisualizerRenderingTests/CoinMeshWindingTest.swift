import XCTest
import simd
import VisualizerCore
@testable import VisualizerRendering

/// Diagnoses the "coins invisible" symptom: a custom SCNGeometry whose winding
/// disagrees with its normals renders inside-out (back-faces culled) under a
/// single-sided material. GeometryWinding.audit returns the inside-out triangle
/// fraction; a healthy mesh is ~0, an inverted mesh ~1.
@MainActor
final class CoinMeshWindingTest: XCTestCase {
    func testCoinMeshWinding() {
        let mesh = CoinMesh.make(radius: 0.12, halfThickness: 0.009, radialSegments: 28)
        let frac = GeometryWinding.audit(positions: mesh.positions,
                                         normals: mesh.normals,
                                         indices: mesh.indices.map { UInt32(bitPattern: $0) },
                                         label: "CoinMesh-test")
        print("COIN_MESH_INSIDE_OUT_FRACTION = \(frac)")
        XCTAssertLessThan(frac, 0.20, "CoinMesh winding is inverted (\(frac*100)% inside-out)")
    }
}
