import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// GPU test for `lbs_skin` (LinearBlendSkin.metal): identity bones reproduce the rest pose,
/// a 50/50 blend of two translations lands halfway, and a NON-UNIFORM scale keeps the
/// normal perpendicular to the scaled surface (the cofactor transform) — the case a naive
/// `M · n` gets wrong for a stretched tooth.
///
/// The SwiftPM CLI doesn't build the package metallib, so the kernel is compiled from
/// source at runtime (same approach as `IlluminatoramaMeshSynthTests`).
final class LinearBlendSkinTests: XCTestCase {

    private func run(positions: [SIMD3<Float>], normals: [SIMD3<Float>],
                     boneIndex: [SIMD4<UInt16>], boneWeight: [SIMD4<Float>],
                     bones: [simd_float4x4]) throws -> ([SIMD3<Float>], [SIMD3<Float>]) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let src = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/LinearBlendSkin.metal")
        let lib = try device.makeLibrary(source: try String(contentsOf: src, encoding: .utf8), options: nil)
        let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "lbs_skin")!)   // gpu-ok: test-only
        let n = positions.count
        let flat: ([SIMD3<Float>]) -> [Float] = { $0.flatMap { [$0.x, $0.y, $0.z] } }
        let rp = device.makeBuffer(bytes: flat(positions), length: n * 12)!
        let rn = device.makeBuffer(bytes: flat(normals), length: n * 12)!
        let bi = device.makeBuffer(bytes: boneIndex, length: n * 8)!
        let bw = device.makeBuffer(bytes: boneWeight, length: n * 16)!
        let bb = device.makeBuffer(bytes: bones, length: bones.count * 64)!
        let op = device.makeBuffer(length: n * 12)!
        let on = device.makeBuffer(length: n * 12)!
        let queue = device.makeCommandQueue()!   // gpu-ok: test-only
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (i, b) in [rp, rn, bi, bw, bb].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        var c = UInt32(n)
        enc.setBytes(&c, length: 4, index: 5)
        enc.setBuffer(op, offset: 0, index: 6)
        enc.setBuffer(on, offset: 0, index: 7)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: test-only readback
        let pp = op.contents().bindMemory(to: Float.self, capacity: n * 3)
        let nn = on.contents().bindMemory(to: Float.self, capacity: n * 3)
        return ((0 ..< n).map { SIMD3(pp[$0 * 3], pp[$0 * 3 + 1], pp[$0 * 3 + 2]) },
                (0 ..< n).map { SIMD3(nn[$0 * 3], nn[$0 * 3 + 1], nn[$0 * 3 + 2]) })
    }

    func testIdentityAndBlend() throws {
        var t = matrix_identity_float4x4
        t.columns.3 = SIMD4(0, 2, 0, 1)
        let (p, _) = try run(positions: [SIMD3(1, 2, 3), SIMD3(1, 0, 0)],
                             normals: [SIMD3(0, 1, 0), SIMD3(0, 1, 0)],
                             boneIndex: [SIMD4(0, 0, 0, 0), SIMD4(0, 1, 0, 0)],
                             boneWeight: [SIMD4(1, 0, 0, 0), SIMD4(0.5, 0.5, 0, 0)],
                             bones: [matrix_identity_float4x4, t])
        XCTAssertLessThan(simd_distance(p[0], SIMD3(1, 2, 3)), 1e-5)
        XCTAssertLessThan(simd_distance(p[1], SIMD3(1, 1, 0)), 1e-5)
    }

    func testNonUniformScaleKeepsNormalPerpendicular() throws {
        // A 45° slope in the XY plane, stretched 3× along Y: the surface tangent (1,1,0)
        // becomes (1,3,0); the normal must stay perpendicular to it.
        let s = simd_float4x4(diagonal: SIMD4(1, 3, 1, 1))
        let n0 = simd_normalize(SIMD3<Float>(1, -1, 0))
        let (_, n) = try run(positions: [SIMD3(0, 0, 0)], normals: [n0],
                             boneIndex: [SIMD4(0, 0, 0, 0)], boneWeight: [SIMD4(1, 0, 0, 0)],
                             bones: [s])
        XCTAssertLessThan(abs(simd_dot(n[0], simd_normalize(SIMD3(1, 3, 0)))), 1e-5)
        XCTAssertEqual(simd_length(n[0]), 1, accuracy: 1e-5)
    }
}
