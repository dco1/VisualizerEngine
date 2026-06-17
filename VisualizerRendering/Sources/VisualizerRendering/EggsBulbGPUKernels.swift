import Foundation
import Metal
import simd

// ── EggsBulbGPUKernels ───────────────────────────────────────────────────────
//
// GPU-resident bulb work for EggsUltra:
//   • BulbInstanceWriteKernel  — writes the marquee bulb sphere Instances
//     (static position + per-tick LUT emission; dark bulbs collapse to scale 0).
//   • BulbPointLightCullKernel — atomically selects the camera-near lit bulbs
//     and writes them as real point lights so the chrome rails catch the colour.
//
// Both read the SAME GPU buffers the egg path uses (bulb positions, LUT colour
// buffer) and write straight into the renderer's instance / point-light buffers
// via the renderer's GPU hooks — no CPU per-bulb loop, no readback.

/// Mirror of `BulbInstanceWriteUniforms` in Illuminatorama.metal (8 × 4 bytes).
public struct BulbInstanceWriteUniforms {
    public var bulbCount: UInt32
    public var groupStart: UInt32
    public var displayRadius: Float
    public var emissionScale: Float
    public var darkThreshold: Float
    public var _pad0: Float = 0
    public var _pad1: Float = 0
    public var _pad2: Float = 0

    public init(bulbCount: UInt32, groupStart: UInt32, displayRadius: Float,
                emissionScale: Float, darkThreshold: Float) {
        self.bulbCount = bulbCount
        self.groupStart = groupStart
        self.displayRadius = displayRadius
        self.emissionScale = emissionScale
        self.darkThreshold = darkThreshold
    }
}

/// Mirror of `BulbCullUniforms` in Illuminatorama.metal (48 bytes).
public struct BulbCullUniforms {
    public var cameraCull: SIMD4<Float>   // xyz = camera pos, w = cullRadius²
    public var bulbCount: UInt32
    public var maxLights: UInt32
    public var lightOffset: UInt32
    public var darkThreshold: Float
    public var lightRadius: Float
    public var gain: Float
    public var _pad0: Float = 0
    public var _pad1: Float = 0

    public init(cameraCull: SIMD4<Float>, bulbCount: UInt32, maxLights: UInt32,
                lightOffset: UInt32, darkThreshold: Float, lightRadius: Float, gain: Float) {
        self.cameraCull = cameraCull
        self.bulbCount = bulbCount
        self.maxLights = maxLights
        self.lightOffset = lightOffset
        self.darkThreshold = darkThreshold
        self.lightRadius = lightRadius
        self.gain = gain
    }
}

/// Mirror of `GlowInstanceWriteUniforms` in Illuminatorama.metal (4 × 4 bytes).
public struct GlowInstanceWriteUniforms {
    public var sliceCount: UInt32
    public var bulbCount: UInt32
    public var groupStart: UInt32
    public var bloomScale: Float

    public init(sliceCount: UInt32, bulbCount: UInt32, groupStart: UInt32, bloomScale: Float) {
        self.sliceCount = sliceCount
        self.bulbCount = bulbCount
        self.groupStart = groupStart
        self.bloomScale = bloomScale
    }
}

@MainActor
public final class GlowInstanceWriteKernel {
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    public init?(engine: SimEngine = .shared) {
        guard let p = engine.pipeline("glow_write_instances") else { return nil }
        self.pipeline = p
        self.threadgroupSize = min(32, p.maxTotalThreadsPerThreadgroup)
    }

    public func dispatch(
        uniforms: GlowInstanceWriteUniforms,
        bulbColors: MTLBuffer,
        outInstances: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        guard uniforms.sliceCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "GlowInstanceWriteKernel.dispatch"
        enc.setComputePipelineState(pipeline)
        var u = uniforms
        enc.setBytes(&u, length: MemoryLayout<GlowInstanceWriteUniforms>.stride, index: 0)
        enc.setBuffer(bulbColors, offset: 0, index: 1)
        enc.setBuffer(outInstances, offset: 0, index: 2)
        let grid = MTLSize(width: Int(uniforms.sliceCount), height: 1, depth: 1)
        let group = MTLSize(width: min(threadgroupSize, Int(uniforms.sliceCount)), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}

@MainActor
public final class BulbInstanceWriteKernel {
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    public init?(engine: SimEngine = .shared) {
        guard let p = engine.pipeline("bulbs_write_instances") else { return nil }
        self.pipeline = p
        self.threadgroupSize = min(64, p.maxTotalThreadsPerThreadgroup)
    }

    public func dispatch(
        uniforms: BulbInstanceWriteUniforms,
        bulbPositions: MTLBuffer,
        bulbColors: MTLBuffer,
        outInstances: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        guard uniforms.bulbCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "BulbInstanceWriteKernel.dispatch"
        enc.setComputePipelineState(pipeline)
        var u = uniforms
        enc.setBytes(&u, length: MemoryLayout<BulbInstanceWriteUniforms>.stride, index: 0)
        enc.setBuffer(bulbPositions, offset: 0, index: 1)
        enc.setBuffer(bulbColors, offset: 0, index: 2)
        enc.setBuffer(outInstances, offset: 0, index: 3)
        let grid = MTLSize(width: Int(uniforms.bulbCount), height: 1, depth: 1)
        let group = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}

@MainActor
public final class BulbPointLightCullKernel {
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int
    /// 4-byte atomic counter, reset to 0 (blit fill) before each dispatch.
    public let counterBuffer: MTLBuffer?

    public init?(engine: SimEngine = .shared) {
        guard let p = engine.pipeline("bulbs_write_pointlights") else { return nil }
        self.pipeline = p
        self.threadgroupSize = min(64, p.maxTotalThreadsPerThreadgroup)
        self.counterBuffer = engine.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride, options: .storageModePrivate)
        self.counterBuffer?.label = "BulbPointLightCull.counter"
    }

    public func dispatch(
        uniforms: BulbCullUniforms,
        bulbPositions: MTLBuffer,
        bulbColors: MTLBuffer,
        outLights: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        guard uniforms.bulbCount > 0, let counter = counterBuffer else { return }
        // Reset the atomic counter for this frame before the cull appends.
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "BulbPointLightCull.resetCounter"
            blit.fill(buffer: counter, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
            blit.endEncoding()
        }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "BulbPointLightCullKernel.dispatch"
        enc.setComputePipelineState(pipeline)
        var u = uniforms
        enc.setBytes(&u, length: MemoryLayout<BulbCullUniforms>.stride, index: 0)
        enc.setBuffer(bulbPositions, offset: 0, index: 1)
        enc.setBuffer(bulbColors, offset: 0, index: 2)
        enc.setBuffer(counter, offset: 0, index: 3)
        enc.setBuffer(outLights, offset: 0, index: 4)
        let grid = MTLSize(width: Int(uniforms.bulbCount), height: 1, depth: 1)
        let group = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}
