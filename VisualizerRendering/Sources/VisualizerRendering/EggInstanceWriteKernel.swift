import Foundation
import Metal
import simd

// ── EggInstanceWriteKernel ───────────────────────────────────────────────────
//
// GPU-resident instance write. One thread per egg reads the EggMotion transform
// buffer, builds a full `Instance` struct (model + normal matrix + material),
// computes the 1/r² rail-bulb irradiance tint as the emission term, and writes
// it straight into the renderer's instance buffer at `groupStart + i`. This
// replaces the CPU transform readback + per-egg tint loop + instance assembly
// in EggsControllerUltra — the egg render data is now produced entirely on the
// GPU with no round-trip.
//
// Driven from `IlluminatoramaRenderer.onEncodeGPUInstances`, which hands the
// live instance buffer + the egg group's slot range to the scene each frame.
// See the `eggs_write_instances` kernel in Illuminatorama.metal.

/// Mirror of `EggInstanceWriteUniforms` in Illuminatorama.metal (8 × 4 bytes).
public struct EggInstanceWriteUniforms {
    public var count: UInt32
    public var bulbCount: UInt32
    public var groupStart: UInt32
    public var minDistSq: Float
    public var tintScale: Float
    public var tintCap: Float
    public var metallic: Float
    public var roughness: Float

    public init(
        count: UInt32, bulbCount: UInt32, groupStart: UInt32,
        minDistSq: Float, tintScale: Float, tintCap: Float,
        metallic: Float, roughness: Float
    ) {
        self.count = count
        self.bulbCount = bulbCount
        self.groupStart = groupStart
        self.minDistSq = minDistSq
        self.tintScale = tintScale
        self.tintCap = tintCap
        self.metallic = metallic
        self.roughness = roughness
    }
}

@MainActor
public final class EggInstanceWriteKernel {

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    public init?(engine: SimEngine = .shared) {
        self.engine = engine
        guard let pipeline = engine.pipeline("eggs_write_instances") else { return nil }
        self.pipeline = pipeline
        self.threadgroupSize = min(64, pipeline.maxTotalThreadsPerThreadgroup)
    }

    /// Encode the instance write into `commandBuffer`.
    ///
    /// - transforms: EggMotion output buffer (the completed/previous-frame buffer
    ///   the CPU path would have read — no in-flight hazard).
    /// - colors: per-egg albedo (`SIMD4<Float>`, xyz used).
    /// - bulbPositions / bulbColors: rail-bulb world positions + LUT colours for
    ///   the irradiance tint (xyz used). Pass `bulbCount = 0` to skip the tint.
    /// - outInstances: the renderer's instance buffer (from the GPU-instance hook).
    public func dispatch(
        uniforms: EggInstanceWriteUniforms,
        transforms: MTLBuffer,
        colors: MTLBuffer,
        bulbPositions: MTLBuffer,
        bulbColors: MTLBuffer,
        outInstances: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        guard uniforms.count > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "EggInstanceWriteKernel.dispatch"
        enc.setComputePipelineState(pipeline)
        var u = uniforms
        enc.setBytes(&u, length: MemoryLayout<EggInstanceWriteUniforms>.stride, index: 0)
        enc.setBuffer(transforms, offset: 0, index: 1)
        enc.setBuffer(colors, offset: 0, index: 2)
        enc.setBuffer(bulbPositions, offset: 0, index: 3)
        enc.setBuffer(bulbColors, offset: 0, index: 4)
        enc.setBuffer(outInstances, offset: 0, index: 5)
        let grid = MTLSize(width: Int(uniforms.count), height: 1, depth: 1)
        let group = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}
