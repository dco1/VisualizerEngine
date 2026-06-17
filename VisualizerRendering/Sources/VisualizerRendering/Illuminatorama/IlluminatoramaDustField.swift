import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

/// GPU dust-mote field for the Illuminatorama Room (issue #28). Owns the
/// position + colour buffers, advects motes on a curl-noise flow via the
/// `illumi_dust_step` compute kernel each tick, and exposes a
/// `ParticleFieldSource` the renderer draws as additive HDR point sprites.
/// Light reactivity (motes brightening in the sunbeam) is a real window
/// shadow test in the kernel, not a faked texture.
@MainActor
public final class IlluminatoramaDustField {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatoramaDust")

    /// Mirror of `DustUniforms` in IlluminatoramaDust.metal.
    private struct DustUniforms {
        var dt: Float; var time: Float; var count: UInt32; var curlScale: Float
        var boundsMin: SIMD3<Float>; var drift: Float
        var boundsMax: SIMD3<Float>; var windowX: Float
        var sunDir: SIMD3<Float>; var shaftGlow: Float
        var sunColor: SIMD3<Float>; var baseBrightness: Float
        var winY0: Float; var winY1: Float; var winZ0: Float; var winZ1: Float
        var feather: Float; var skyFill: Float; var twinkleAmt: Float; var outdoor: Float = 0
    }

    public let capacity: Int
    public let positionBuffer: MTLBuffer
    public let colorBuffer: MTLBuffer

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let boundsMin: SIMD3<Float>
    private let boundsMax: SIMD3<Float>

    // Per-tick inputs (set by the host before `update`).
    public var curlScale: Float = 0.35
    public var drift: Float = 0.5
    public var shaftGlow: Float = 6.0
    public var baseBrightness: Float = 0.5
    public var skyFill: Float = 1.0
    public var twinkleAmt: Float = 0.25
    public var feather: Float = 0.12
    public var sunDir: SIMD3<Float> = simd_normalize(SIMD3(1, 1, 0))
    public var sunColor: SIMD3<Float> = SIMD3(1.0, 0.95, 0.85)
    public var windowX: Float = 4.0
    public var winRect: SIMD4<Float> = SIMD4(0.9, 2.5, -1.6, 1.6)  // y0,y1,z0,z1
    /// Outdoor mode: no window plane — motes are sun-lit via a drifting canopy-gap
    /// dapple instead of the window-rectangle shadow test (for open-air scenes
    /// like the Forest, where the sun isn't coming through a +X window).
    public var outdoor: Bool = false

    public init?(engine: SimEngine, capacity: Int,
                 boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        guard capacity > 0,
              let pipe = engine.pipelineCache.pipelineState(name: "illumi_dust_step",
                                                            device: engine.device) else {
            Self.log.error("illumi_dust_step pipeline unavailable")
            return nil
        }
        let bytes = MemoryLayout<SIMD4<Float>>.stride * capacity
        guard let pb = engine.device.makeBuffer(length: bytes, options: .storageModeShared),
              let cb = engine.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            Self.log.error("dust buffer allocation failed")
            return nil
        }
        pb.label = "Illuminatorama.dust.positions"
        cb.label = "Illuminatorama.dust.colors"
        self.engine = engine
        self.pipeline = pipe
        self.capacity = capacity
        self.positionBuffer = pb
        self.colorBuffer = cb
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax

        // Seed motes uniformly through the volume with a random per-mote phase.
        let pp = pb.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity)
        let cc = cb.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<capacity {
            let x = Float.random(in: boundsMin.x...boundsMax.x, using: &rng)
            let y = Float.random(in: boundsMin.y...boundsMax.y, using: &rng)
            let z = Float.random(in: boundsMin.z...boundsMax.z, using: &rng)
            pp[i] = SIMD4(x, y, z, Float.random(in: 0...1, using: &rng))
            cc[i] = SIMD4(0.2, 0.2, 0.2, 1)
        }
    }

    /// Advance the simulation one tick. Commits its own command buffer on the
    /// shared queue; ordering guarantees it completes before the renderer's
    /// frame command buffer (committed afterward) draws the buffers.
    public func update(dt: Float, time: Float) {
        var u = DustUniforms(
            dt: max(0, min(0.05, dt)), time: time, count: UInt32(capacity), curlScale: curlScale,
            boundsMin: boundsMin, drift: drift,
            boundsMax: boundsMax, windowX: windowX,
            sunDir: simd_normalize(sunDir), shaftGlow: shaftGlow,
            sunColor: sunColor, baseBrightness: baseBrightness,
            winY0: winRect.x, winY1: winRect.y, winZ0: winRect.z, winZ1: winRect.w,
            feather: feather, skyFill: skyFill, twinkleAmt: twinkleAmt,
            outdoor: outdoor ? 1 : 0)

        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "Illuminatorama.dust"
        enc.label = "Illuminatorama.dust.step"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(positionBuffer, offset: 0, index: 0)
        enc.setBuffer(colorBuffer, offset: 0, index: 1)
        enc.setBytes(&u, length: MemoryLayout<DustUniforms>.stride, index: 2)
        let tg = min(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth * 8)
        let groups = MTLSize(width: (capacity + tg - 1) / tg, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    public func source(ownerScene: ObjectIdentifier, pointSize: Float, colorScale: Float = 1.0)
        -> ParticleFieldSource {
        ParticleFieldSource(
            positionBuffer: positionBuffer,
            colorBuffer: colorBuffer,
            vertexCount: capacity,
            ownerScene: ownerScene,
            positionStrideFloats: 4, colorStrideFloats: 4,
            positionOffsetFloats: 0, colorOffsetFloats: 0,
            pointSize: pointSize, colorScale: colorScale)
    }
}
