import Metal
import XCTest
import simd
@testable import VisualizerHumans

@MainActor
final class GPUSkinningTests: XCTestCase {

    // The GPU kernel must reproduce the CPU reference (PosedHuman.skin) exactly
    // (within float tolerance) — same inputs, same matrix palette.
    func testGPUMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("no Metal device")
        }
        let human = GeneratedHuman(spec: HumanSpec(ageYears: 33, gender: 0.7, seed: 3))
        let pose = WalkCycle.pose(phase: 0.37, human: human, running: true)
        let posed = PosedHuman(human: human, pose: pose)

        let gpu = try GPUSkinnedHuman(device: device, human: human)
        guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
        gpu.encode(pose: posed, into: cb)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let (cpuPos, cpuNrm) = posed.skinned(human)
        let outP = gpu.positionBuffer.contents()
        let outN = gpu.normalBuffer.contents()
        var maxPosErr: Float = 0
        var maxNrmErr: Float = 0
        for i in 0..<human.positions.count {
            let p = outP.load(fromByteOffset: i * 12, as: (Float, Float, Float).self)
            let n = outN.load(fromByteOffset: i * 12, as: (Float, Float, Float).self)
            maxPosErr = max(maxPosErr, simd_distance(SIMD3(p.0, p.1, p.2), cpuPos[i]))
            maxNrmErr = max(maxNrmErr, simd_distance(SIMD3(n.0, n.1, n.2), cpuNrm[i]))
        }
        XCTAssertLessThan(maxPosErr, 1e-4, "GPU positions diverge from CPU reference")
        XCTAssertLessThan(maxNrmErr, 1e-3, "GPU normals diverge from CPU reference")
    }

    func testGarmentSkinsOnGPU() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("no Metal device")
        }
        let human = GeneratedHuman(spec: HumanSpec(ageYears: 40, gender: 0.2, seed: 9))
        guard let shirt = ClothingBuilder.garments(for: human, outfit: Outfit()).first else {
            return XCTFail("no shirt garment")
        }
        let gpu = try GPUSkinnedHuman(device: device, garment: shirt, boneCount: human.bones.count)
        let posed = PosedHuman(human: human, pose: WalkCycle.pose(phase: 0.1, human: human))
        guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
        gpu.encode(pose: posed, into: cb)
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let (cpuPos, _) = posed.skin(
            positions: shirt.positions, normals: shirt.normals,
            joints: shirt.skinJoints, weights: shirt.skinWeights)
        let outP = gpu.positionBuffer.contents()
        var maxErr: Float = 0
        for i in 0..<shirt.positions.count {
            let p = outP.load(fromByteOffset: i * 12, as: (Float, Float, Float).self)
            maxErr = max(maxErr, simd_distance(SIMD3(p.0, p.1, p.2), cpuPos[i]))
        }
        XCTAssertLessThan(maxErr, 1e-4)
    }
}
