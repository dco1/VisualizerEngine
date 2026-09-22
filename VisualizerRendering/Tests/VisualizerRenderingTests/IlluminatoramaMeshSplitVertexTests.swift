import XCTest
import SceneKit
import Metal
@testable import VisualizerRendering

/// DH-0742 — SceneKit represents a geometry whose sources have DIFFERENT `vectorCount`s (the
/// common USD/Blender-import shape: position is vertex-varying/shared, normal is authored
/// FACEVARYING — one value per triangle corner, for a hard edge) by widening each corner's
/// index entry to one index PER DISTINCT vectorCount group, interleaved per corner
/// ([posIndex, cornerIndex, posIndex, cornerIndex, …]) instead of the single shared index
/// every hand-built/procedural mesh in this engine had ever produced. `IlluminatoramaMesh.from`
/// read that as one flat index stream, pairing arbitrary position indices with what are
/// actually per-corner counters — shredding a real imported `.usdz` side table into a torn mess
/// (reported as "this looks bad" against a genuinely clean source file, confirmed by an
/// independent vanilla-SceneKit render and by the jagged render SHADOW proving the corrupted
/// positions were real, not a shading artifact).
///
/// This constructs the identical shape by hand — 4 shared positions, 2 triangles, 6 facevarying
/// normals, a 2-stream interleaved index buffer matching SceneKit's own layout exactly — and
/// checks the conversion recovers the CORRECT per-corner pairing, not the scrambled one.
final class IlluminatoramaMeshSplitVertexTests: XCTestCase {

    /// Two triangles sharing 4 positions (p0,p1,p2 and p0,p2,p3), with 6 DISTINCT per-corner
    /// normals (facevarying — SceneKit can't collapse them to 4 without losing the hard edge)
    /// and a UV whose `.x` carries the corner index (0...5) so a wrong pairing is detectable
    /// even where a normal collision could mask it.
    /// Tightly-packed (12-byte stride) Float triples — `SIMD3<Float>`'s own in-memory stride is
    /// PADDED to 16 bytes (SIMD alignment), so building `SCNGeometrySource` data straight from a
    /// `[SIMD3<Float>]` buffer with `dataStride: 12` reads garbage past the first element. Flatten
    /// to a plain `[Float]` first to get the tight packing USD/SceneKit sources actually use.
    private func packedTriples(_ vs: [SIMD3<Float>]) -> Data {
        var flat: [Float] = []
        flat.reserveCapacity(vs.count * 3)
        for v in vs { flat.append(v.x); flat.append(v.y); flat.append(v.z) }
        return flat.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func splitVertexQuad() -> SCNGeometry {
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1),
        ]
        // Corner order: triangle A = (p0,p1,p2), triangle B = (p0,p2,p3).
        let posIndexPerCorner: [UInt32] = [0, 1, 2, 0, 2, 3]
        let cornerNormals: [SIMD3<Float>] = (0..<6).map { SIMD3<Float>(0, 1, Float($0)) }

        let posSource = SCNGeometrySource(data: packedTriples(positions), semantic: .vertex, vectorCount: 4,
                                          usesFloatComponents: true, componentsPerVector: 3,
                                          bytesPerComponent: 4, dataOffset: 0, dataStride: 12)

        let normalSource = SCNGeometrySource(data: packedTriples(cornerNormals), semantic: .normal, vectorCount: 6,
                                             usesFloatComponents: true, componentsPerVector: 3,
                                             bytesPerComponent: 4, dataOffset: 0, dataStride: 12)

        // Interleave [posIndex, cornerIndex] per corner — SceneKit's own observed layout for
        // this exact case (verified against the real fixture's raw index buffer).
        var interleaved: [UInt32] = []
        interleaved.reserveCapacity(12)
        for k in 0..<6 { interleaved.append(posIndexPerCorner[Int(k)]); interleaved.append(UInt32(k)) }
        let indexData = interleaved.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(data: indexData, primitiveType: .triangles,
                                         primitiveCount: 2, bytesPerIndex: 4)

        return SCNGeometry(sources: [posSource, normalSource], elements: [element])
    }

    /// The ordinary case every OTHER SceneKit mesh in this engine has always shipped: normals
    /// share position's vectorCount (per-vertex, not facevarying) — one index per corner, no
    /// interleaving. Must keep working unchanged (this is the regression guard for the new
    /// branch: it must NOT fire here).
    private func sharedVertexQuad() -> SCNGeometry {
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1),
        ]
        let normals: [SIMD3<Float>] = Array(repeating: SIMD3<Float>(0, 1, 0), count: 4)
        let posSource = SCNGeometrySource(data: packedTriples(positions), semantic: .vertex, vectorCount: 4,
                                          usesFloatComponents: true, componentsPerVector: 3,
                                          bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let normalSource = SCNGeometrySource(data: packedTriples(normals), semantic: .normal, vectorCount: 4,
                                             usesFloatComponents: true, componentsPerVector: 3,
                                             bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(data: indexData, primitiveType: .triangles,
                                         primitiveCount: 2, bytesPerIndex: 4)
        return SCNGeometry(sources: [posSource, normalSource], elements: [element])
    }

    @MainActor
    func testSplitVertexFacevaryingNormalsAreCorrectlyPaired() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let mesh = try XCTUnwrap(IlluminatoramaMesh.from(scnGeometry: splitVertexQuad(), device: device),
                                 "split-vertex geometry must still convert")

        XCTAssertEqual(mesh.indexCount, 6, "2 triangles × 3 corners — no triangle dropped")
        XCTAssertEqual(mesh.vertexCount, 6,
                       "facevarying normals force a per-CORNER vertex list (un-welded), not 4")

        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1),
        ]
        let posIndexPerCorner: [Int] = [0, 1, 2, 0, 2, 3]
        let verts = mesh.vertexBuffer.contents().bindMemory(to: IlluminatoramaVertex.self, capacity: 6)
        for k in 0..<6 {
            let v = verts[k]
            XCTAssertEqual(v.position, positions[posIndexPerCorner[k]],
                           "corner \(k): position must match its OWN face-vertex index, not the corner counter")
            XCTAssertEqual(v.normal, SIMD3<Float>(0, 1, Float(k)),
                           "corner \(k): normal must be its own facevarying value — a scrambled pairing was the bug")
        }
    }

    @MainActor
    func testOrdinarySharedVertexMeshIsUnaffected() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let mesh = try XCTUnwrap(IlluminatoramaMesh.from(scnGeometry: sharedVertexQuad(), device: device))
        XCTAssertEqual(mesh.indexCount, 6)
        XCTAssertEqual(mesh.vertexCount, 4, "no facevarying source present — must stay welded, unchanged behaviour")
    }
}
