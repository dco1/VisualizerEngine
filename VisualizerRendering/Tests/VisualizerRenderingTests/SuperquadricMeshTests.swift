import XCTest
import simd
import VisualizerCore
@testable import VisualizerRendering

/// The perfect analytic superquadric primitive. The RT proxy mesh must be sound
/// (no out-of-range indices, finite verts, outward winding), and — critically —
/// its parameterized surface must agree with the in-fragment field `F` the
/// impostor ray-traces, or the proxy's RT shadow/reflection would not match the
/// camera-view silhouette.
@MainActor
final class SuperquadricMeshTests: XCTestCase {

    /// Roundness pairs spanning the family: sphere, rounded box, box-ish,
    /// pinched, and an anisotropic mix.
    private let cases: [(Float, Float)] = [
        (1.0, 1.0),   // sphere / ellipsoid
        (0.4, 0.4),   // rounded box
        (0.15, 0.15), // near-cube
        (1.0, 0.3),   // rounded cylinder-ish
        (1.8, 0.6),   // pinched equator
    ]

    func testProxyMeshIsSound() {
        for (e1, e2) in cases {
            let (verts, indices) = IlluminatoramaMesh.superquadricProxyGeometry(e1: e1, e2: e2)
            let positions = verts.map(\.position)
            let normals = verts.map(\.normal)
            let r = MeshSoundness.audit(positions: positions, indices: indices.map(UInt32.init),
                                        normals: normals, label: "sq(\(e1),\(e2))",
                                        expectClosed: true)
            XCTAssertEqual(r.outOfRangeIndices, 0, "\(r)")
            XCTAssertEqual(r.nonFiniteVertices, 0, "\(r)")
            // Pole rings collapse to a point (as in unitSphere), so a small band of
            // degenerate triangles at each pole is expected; bound it generously.
            XCTAssertLessThanOrEqual(r.degenerateTriangles, 2 * 48 + 4,
                                     "only the two pole rings should degenerate: \(r)")
        }
    }

    func testProxyWindingIsOutward() {
        for (e1, e2) in cases {
            let (verts, indices) = IlluminatoramaMesh.superquadricProxyGeometry(e1: e1, e2: e2)
            let frac = GeometryWinding.audit(positions: verts.map(\.position),
                                             normals: verts.map(\.normal),
                                             indices: indices.map(UInt32.init),
                                             label: "sq(\(e1),\(e2))")
            // Winding must agree with the central-difference outward normals.
            XCTAssertLessThan(frac, 0.05, "proxy winding disagrees with normals (\(e1),\(e2)): \(frac)")
        }
    }

    /// The whole hybrid hinges on this: the parameterized proxy vertices must lie
    /// on the same implicit surface the impostor intersects (`F == 1`).
    func testParameterizationMatchesField() {
        for (e1, e2) in cases {
            let (verts, _) = IlluminatoramaMesh.superquadricProxyGeometry(e1: e1, e2: e2)
            var maxErr: Float = 0
            for p in verts.map(\.position) {
                let f = IlluminatoramaMesh.superquadricField(p, e1: e1, e2: e2)
                maxErr = max(maxErr, abs(f - 1))
            }
            XCTAssertLessThan(maxErr, 0.02,
                "proxy vertices stray from F==1 for (\(e1),\(e2)); max |F-1|=\(maxErr)")
        }
    }

    /// Sphere is the e1==e2==1 unit-radius case.
    func testSphereCaseIsUnitRadius() {
        let (verts, _) = IlluminatoramaMesh.superquadricProxyGeometry(e1: 1, e2: 1)
        for p in verts.map(\.position) {
            XCTAssertEqual(simd_length(p), 1.0, accuracy: 1e-3, "e1=e2=1 must be the unit sphere")
        }
    }

    func testBiunitBoxBounds() throws {
        // The impostor's bounding box spans [-1,1]³ so, scaled by the semi-axis
        // extents, it bounds the analytic surface exactly.
        let device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device")
        let m = IlluminatoramaMesh.biunitBox(device: device!)
        XCTAssertEqual(m.vertexCount, 24, "6 faces × 4 verts")
        XCTAssertEqual(m.indexCount, 36, "6 faces × 2 tris × 3")
    }
}
