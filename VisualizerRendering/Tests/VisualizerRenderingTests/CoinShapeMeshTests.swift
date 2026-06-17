import XCTest
import simd
import VisualizerCore
@testable import VisualizerRendering

/// Soundness of the instanced-renderer primitive meshes used by RigidPileField scenes
/// (`BoxMesh`, `SphereMesh`) — they must have in-range indices, finite vertices, and
/// the right size/winding, since a bad mesh renders inside-out or invisible.
final class CoinShapeMeshTests: XCTestCase {

    func testBoxMeshIsUnitAndSound() {
        let m = BoxMesh.unit()
        XCTAssertEqual(m.positions.count, 24, "6 faces × 4 verts (per-face normals)")
        XCTAssertEqual(m.indices.count, 36, "6 faces × 2 triangles × 3")
        // Unit cube: every vertex on the [-0.5, 0.5]³ boundary.
        for p in m.positions {
            XCTAssertEqual(max(abs(p.x), max(abs(p.y), abs(p.z))), 0.5, accuracy: 1e-5)
        }
        let r = MeshSoundness.audit(positions: m.positions,
                                    indices: m.indices.map { UInt32($0) },
                                    normals: m.normals, label: "BoxMesh")
        XCTAssertEqual(r.outOfRangeIndices, 0, "\(r)")
        XCTAssertEqual(r.nonFiniteVertices, 0, "\(r)")
        XCTAssertEqual(r.degenerateTriangles, 0, "a clean cube has no zero-area triangles")
    }

    func testBeveledBoxIsUnitAndSound() {
        let m = BoxMesh.unitBeveled(chamfer: 0.16)
        // Still within the unit cube ([-0.5, 0.5]³) so the per-instance transform scale holds.
        for p in m.positions {
            XCTAssertLessThanOrEqual(max(abs(p.x), max(abs(p.y), abs(p.z))), 0.5 + 1e-5)
        }
        let r = MeshSoundness.audit(positions: m.positions,
                                    indices: m.indices.map { UInt32($0) },
                                    normals: m.normals, label: "BoxMesh.beveled")
        XCTAssertEqual(r.outOfRangeIndices, 0, "\(r)")
        XCTAssertEqual(r.nonFiniteVertices, 0, "\(r)")
        XCTAssertEqual(r.degenerateTriangles, 0, "auto-winding chamfer has no zero-area facets")
    }

    func testSphereMeshIsCorrectRadiusAndSound() {
        let R: Float = 0.06
        let m = SphereMesh.make(radius: R)
        XCTAssertGreaterThan(m.positions.count, 100)
        for p in m.positions {
            XCTAssertEqual(simd_length(p), R, accuracy: 1e-4, "every vertex at the sphere radius")
        }
        let r = MeshSoundness.audit(positions: m.positions,
                                    indices: m.indices.map { UInt32($0) },
                                    normals: m.normals, label: "SphereMesh")
        XCTAssertEqual(r.outOfRangeIndices, 0, "\(r)")
        XCTAssertEqual(r.nonFiniteVertices, 0, "\(r)")
        // (A UV sphere has zero-area triangles at the two poles by construction — the
        //  same as IlluminatoramaMesh.unitSphere — so degenerateTriangles is expected
        //  there and not asserted; they render nothing and cause no artifact.)
    }
}
