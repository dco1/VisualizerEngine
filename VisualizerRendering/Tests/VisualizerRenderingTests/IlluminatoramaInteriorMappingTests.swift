import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Interior mapping (`IlluminatoramaRenderer.interiorMappedMeshKinds`,
/// IlluminatoramaGBuffer.metal `interiorMapRadiance`): the room box behind a pane, ray-traced per
/// fragment. Probed directly from the shader source.
final class IlluminatoramaInteriorMappingTests: XCTestCase {

    static let kernel = """
    kernel void imapProbe(device float4* out [[buffer(0)]], device const float4* cams [[buffer(1)]],
                          constant InteriorMapParams& m [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
        // A pane in the z = 0 plane facing +Z (the room behind it at −Z), tangent +X, its
        // bottom-left corner at the origin; probe its centre.
        float3 P = float3(0.475, 0.65, 0.0);
        float3 rad = interiorMapRadiance(P, float3(0, 0, 1), float4(1, 0, 0, 1), float2(0.5 + m.pane.z, 0.5), cams[tid].xyz, m);
        out[tid] = float4(rad, 0.0);
    }
    """

    @MainActor
    private func probe(_ cams: [SIMD3<Float>], _ m: IlluminatoramaInteriorMapping) throws -> [SIMD3<Float>] {
        let engine = SimEngine.shared
        let device = engine.device
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaGBuffer.metal")
        let src = try MetalSourceLoader.source(contentsOf: url)
        let opts = MTLCompileOptions()
        let lib = try device.makeLibrary(source: src + "\n" + Self.kernel, options: opts)
        guard let fn = lib.makeFunction(name: "imapProbe") else { throw XCTSkip("probe missing") }
        let pso = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        var c4 = cams.map { SIMD4<Float>($0, 0) }
        var g = m.gpu
        let n = cams.count
        guard let cb4 = device.makeBuffer(bytes: &c4, length: n * 16, options: .storageModeShared),
              let ob = device.makeBuffer(length: n * 16, options: .storageModeShared),
              let cb = engine.commandQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw XCTSkip("no resources") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(ob, offset: 0, index: 0)
        enc.setBuffer(cb4, offset: 0, index: 1)
        enc.setBytes(&g, length: MemoryLayout<IlluminatoramaInteriorMapping.GPU>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: n, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        if let e = cb.error { throw e }
        let o = ob.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
        return (0..<n).map { SIMD3(o[$0].x, o[$0].y, o[$0].z) }
    }

    @MainActor
    func testRoomBoxHasDepthAndParallax() throws {
        let m = IlluminatoramaInteriorMapping(paneSize: SIMD2(0.95, 1.3), depthJitter: 0)
        // Straight on from 30 m, from the left, from above, from below.
        let cams: [SIMD3<Float>] = [SIMD3(0.475, 0.65, 30), SIMD3(-25, 0.65, 15), SIMD3(0.475, 20, 25),
                                    SIMD3(0.475, -15, 25), SIMD3(0.475, 0.65, -5)]
        let r = try probe(cams, m)
        for (i, c) in r.prefix(4).enumerated() {
            XCTAssertTrue(c.x.isFinite && c.x > 0, "view \(i): a lit room, finite: \(c)")
            XCTAssertLessThan(c.x, 5, "view \(i): normalised near 1 at the back wall: \(c)")
        }
        // Different views see different surfaces of the room (parallax): not all equal.
        XCTAssertGreaterThan(abs(r[0].x - r[1].x) + abs(r[0].x - r[2].x) + abs(r[0].x - r[3].x), 0.05)
        // Looking down (camera above) sees the FLOOR — darker (floor scale 0.45) than the ceiling
        // seen from below.
        XCTAssertLessThan(r[2].y, r[3].y, "floor darker than ceiling: \(r[2]) vs \(r[3])")
        // From behind the pane (inside the room) nothing: the variant is one-sided.
        XCTAssertEqual(r[4], .zero)
        // Hue is the wall albedo's (warm off-white), luminance-normalised.
        XCTAssertGreaterThan(r[0].x, r[0].z)
    }
}
