import XCTest
import SceneKit
import Metal
@testable import VisualizerRendering

/// `IlluminatoramaMesh.from(scnGeometry:...)` used to stamp `(0, 0)` onto EVERY vertex when a
/// geometry had no texcoord source — harmless while the renderer never sampled textures, but the
/// material system now does, so a mesh with no authored UV sampled one shared texel for its whole
/// surface: a wood grain (or any other textured material) rendered as a single flat colour on an
/// imported custom object with no UV unwrap (Daydream Home, DH-0742 follow-up, 2026-09-10 — "only
/// getting COLORS ... not the actual wood grain"). This checks the box-projected fallback that
/// replaced it: real variation across the surface, at the SAME "one metre per repeat" convention
/// every other instanced placeable's own generated mesh already bakes.
final class IlluminatoramaMeshTriplanarUVTests: XCTestCase {

    private func packedTriples(_ vs: [SIMD3<Float>]) -> Data {
        var flat: [Float] = []
        flat.reserveCapacity(vs.count * 3)
        for v in vs { flat.append(v.x); flat.append(v.y); flat.append(v.z) }
        return flat.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func packedPairs(_ vs: [SIMD2<Float>]) -> Data {
        var flat: [Float] = []
        flat.reserveCapacity(vs.count * 2)
        for v in vs { flat.append(v.x); flat.append(v.y) }
        return flat.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// A flat quad in the XZ plane (normal +Y on every vertex, positions spanning 2×2 m) with NO
    /// texcoord source — the shape a simple imported box/table top with no UV unwrap has.
    private func unUVedQuad(positions: [SIMD3<Float>], normal: SIMD3<Float>) -> SCNGeometry {
        let normals = Array(repeating: normal, count: positions.count)
        let posSource = SCNGeometrySource(data: packedTriples(positions), semantic: .vertex,
                                          vectorCount: positions.count, usesFloatComponents: true,
                                          componentsPerVector: 3, bytesPerComponent: 4,
                                          dataOffset: 0, dataStride: 12)
        let normalSource = SCNGeometrySource(data: packedTriples(normals), semantic: .normal,
                                             vectorCount: normals.count, usesFloatComponents: true,
                                             componentsPerVector: 3, bytesPerComponent: 4,
                                             dataOffset: 0, dataStride: 12)
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(data: indexData, primitiveType: .triangles,
                                         primitiveCount: 2, bytesPerIndex: 4)
        return SCNGeometry(sources: [posSource, normalSource], elements: [element])
    }

    @MainActor
    func testAMeshWithNoUVGetsRealVariationNotOneSharedTexel() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        // A 2×2 m quad, dominant normal +Y (a tabletop), corners far enough apart that a
        // per-metre box projection must place them at visibly different UVs.
        let positions: [SIMD3<Float>] = [
            SIMD3(-1, 0, -1), SIMD3(1, 0, -1), SIMD3(1, 0, 1), SIMD3(-1, 0, 1),
        ]
        let mesh = try XCTUnwrap(IlluminatoramaMesh.from(scnGeometry: unUVedQuad(positions: positions, normal: SIMD3(0, 1, 0)), device: device))
        let verts = mesh.vertexBuffer.contents().bindMemory(to: IlluminatoramaVertex.self, capacity: mesh.vertexCount)
        var uvs: [SIMD2<Float>] = []
        for i in 0..<mesh.vertexCount { uvs.append(verts[i].uv) }

        XCTAssertFalse(uvs.allSatisfy { $0 == SIMD2<Float>(0, 0) },
                       "every vertex still shares the placeholder (0,0) UV — a textured material "
                       + "would sample one texel for the whole mesh, the exact bug this closes")
        // Every corner should land at a DISTINCT uv — the projection must vary continuously
        // with position, not collapse the whole face to a repeated constant.
        XCTAssertEqual(Set(uvs.map { "\($0.x),\($0.y)" }).count, uvs.count,
                       "corners of a 2×2 m face must resolve to distinct UVs")
    }

    @MainActor
    func testUVMatchesTheOneMetrePerRepeatConventionFurnitureAlreadyUses() throws {
        // A face with normal +Z (dominant Z), so the projection reads (x, y) directly — pick
        // positions where the expected UV is easy to state and check it exactly, matching
        // `ElementUVScale.furnitureSlice` / `SurfaceTiling.prebaked(1.0)`'s "1 UV unit = 1 metre"
        // convention: no additional scale factor belongs in the projection.
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let positions: [SIMD3<Float>] = [
            SIMD3(0, 0, 5), SIMD3(2, 0, 5), SIMD3(2, 3, 5), SIMD3(0, 3, 5),
        ]
        let mesh = try XCTUnwrap(IlluminatoramaMesh.from(scnGeometry: unUVedQuad(positions: positions, normal: SIMD3(0, 0, 1)), device: device))
        let verts = mesh.vertexBuffer.contents().bindMemory(to: IlluminatoramaVertex.self, capacity: mesh.vertexCount)
        XCTAssertEqual(verts[0].uv, SIMD2<Float>(0, 0))
        XCTAssertEqual(verts[1].uv, SIMD2<Float>(2, 0))
        XCTAssertEqual(verts[2].uv, SIMD2<Float>(2, 3))
    }

    /// A mesh that DOES carry a real texcoord source must be completely unaffected — the
    /// projection only ever fires when there is no usable UV data at all.
    @MainActor
    func testAMeshWithRealUVsIsUnaffected() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let positions: [SIMD3<Float>] = [
            SIMD3(-1, 0, -1), SIMD3(1, 0, -1), SIMD3(1, 0, 1), SIMD3(-1, 0, 1),
        ]
        let normals = Array(repeating: SIMD3<Float>(0, 1, 0), count: 4)
        let authoredUVs: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)]
        let posSource = SCNGeometrySource(data: packedTriples(positions), semantic: .vertex,
                                          vectorCount: 4, usesFloatComponents: true,
                                          componentsPerVector: 3, bytesPerComponent: 4,
                                          dataOffset: 0, dataStride: 12)
        let normalSource = SCNGeometrySource(data: packedTriples(normals), semantic: .normal,
                                             vectorCount: 4, usesFloatComponents: true,
                                             componentsPerVector: 3, bytesPerComponent: 4,
                                             dataOffset: 0, dataStride: 12)
        let uvSource = SCNGeometrySource(data: packedPairs(authoredUVs), semantic: .texcoord,
                                         vectorCount: 4, usesFloatComponents: true,
                                         componentsPerVector: 2, bytesPerComponent: 4,
                                         dataOffset: 0, dataStride: 8)
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        let indexData = indices.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(data: indexData, primitiveType: .triangles,
                                         primitiveCount: 2, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [posSource, normalSource, uvSource], elements: [element])

        let mesh = try XCTUnwrap(IlluminatoramaMesh.from(scnGeometry: geometry, device: device))
        let verts = mesh.vertexBuffer.contents().bindMemory(to: IlluminatoramaVertex.self, capacity: 4)
        for i in 0..<4 {
            XCTAssertEqual(verts[i].uv, authoredUVs[i],
                           "an authored UV must pass through untouched — no projection substituted")
        }
    }
}
