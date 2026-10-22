import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

/// GPU falling-snow field (Christmas Cabin). Owns the position / velocity / colour buffers, steps
/// them with the `illumi_snow_step` compute kernel each tick, and exposes a `ParticleFieldSource`
/// the renderer draws as additive HDR point sprites, depth-tested against the opaque scene (a
/// flake behind the tree is hidden by it).
///
/// Honest physics, not a scrolling texture: every flake is an inertial particle with its own
/// terminal fall speed that relaxes toward a gusting wind + curl turbulence + tumbling flutter
/// (see the kernel's header). Shading uses the real lights the host passes in each tick.
@MainActor
public final class IlluminatoramaSnowField {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatoramaSnow")

    /// Mirror of `SnowUniforms` in IlluminatoramaSnow.metal (every vector is float4 — no packed
    /// float3 stride traps).
    private struct SnowUniforms {
        var dt: Float; var time: Float; var count: UInt32; var relaxTau: Float
        var boundsMin: SIMD4<Float>
        var boundsMax: SIMD4<Float>
        var wind: SIMD4<Float>
        var cam: SIMD4<Float>
        var moon: SIMD4<Float>
        var pointPx: SIMD4<Float>
        var lightPos: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                       SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var lightCol: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                       SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    }

    /// A point light that shades flakes (position in world metres, radiant intensity pre-multiplied); the first 8 of `lights` are used.
    public struct FlakeLight {
        public var position: SIMD3<Float>
        public var color: SIMD3<Float>
        public init(position: SIMD3<Float>, color: SIMD3<Float>) { self.position = position; self.color = color }
    }

    public let capacity: Int
    public let positionBuffer: MTLBuffer
    public let colorBuffer: MTLBuffer
    private let velocityBuffer: MTLBuffer

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let boundsMin: SIMD3<Float>
    private let boundsMax: SIMD3<Float>

    // Per-tick tunables (set by the host).
    /// Mean terminal fall speed, m/s (dry powder ~0.5, ordinary flakes ~1, wet aggregates ~1.4).
    public var fallSpeed: Float = 1.0
    /// Base wind, m/s.
    public var wind: SIMD3<Float> = SIMD3(0.6, 0, 0.2)
    /// Gust depth — fraction of the base wind the slow gust field adds or removes.
    public var gustAmount: Float = 0.7
    /// Curl-turbulence speed, m/s.
    public var turbulence: Float = 0.25
    /// Tumbling flutter, m/s.
    public var flutter: Float = 0.18
    /// Drag time constant — how quickly a flake follows the air, s.
    public var relaxTau: Float = 0.45
    /// Flake diameter, m (clumped aggregates read bigger than single crystals).
    public var flakeDiameter: Float = 0.006
    /// Moonlit flake radiance (linear HDR).
    public var moonRadiance: SIMD3<Float> = SIMD3(0.05, 0.06, 0.09)
    public var sparkle: Float = 0.5
    public var edgeFade: Float = 1.5
    public var pointSizePx: Float = 3.0

    public init?(engine: SimEngine, capacity: Int,
                 boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        guard capacity > 0,
              let pipe = engine.pipelineCache.pipelineState(name: "illumi_snow_step",
                                                            device: engine.device) else {
            Self.log.error("illumi_snow_step pipeline unavailable")
            return nil
        }
        let bytes = MemoryLayout<SIMD4<Float>>.stride * capacity
        guard let pb = engine.device.makeBuffer(length: bytes, options: .storageModeShared),
              let vb = engine.device.makeBuffer(length: bytes, options: .storageModeShared),
              let cb = engine.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            Self.log.error("snow buffer allocation failed")
            return nil
        }
        pb.label = "Illuminatorama.snow.positions"
        vb.label = "Illuminatorama.snow.velocities"
        cb.label = "Illuminatorama.snow.colors"
        self.engine = engine; self.pipeline = pipe; self.capacity = capacity
        self.positionBuffer = pb; self.velocityBuffer = vb; self.colorBuffer = cb
        self.boundsMin = boundsMin; self.boundsMax = boundsMax

        let pp = pb.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity)
        let vv = vb.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity)
        let cc = cb.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<capacity {
            pp[i] = SIMD4(Float.random(in: boundsMin.x...boundsMax.x, using: &rng),
                          Float.random(in: boundsMin.y...boundsMax.y, using: &rng),
                          Float.random(in: boundsMin.z...boundsMax.z, using: &rng),
                          Float.random(in: 0...1, using: &rng))
            vv[i] = SIMD4(0, -1, 0, 0)
            cc[i] = SIMD4(0, 0, 0, 1)
        }
    }

    /// Advance one tick. `pixelAngle` = radians per canvas pixel at the camera (fovY / height).
    /// Commits its own command buffer on the shared queue (ordered ahead of the renderer's frame).
    public func update(dt: Float, time: Float, camera: SIMD3<Float>, pixelAngle: Float,
                       lights: [FlakeLight] = []) {
        func lp(_ i: Int) -> SIMD4<Float> { i < lights.count ? SIMD4(lights[i].position, 0) : .zero }
        func lc(_ i: Int) -> SIMD4<Float> { i < lights.count ? SIMD4(lights[i].color, 0) : .zero }
        var u = SnowUniforms(
            dt: max(0, min(0.05, dt)), time: time, count: UInt32(capacity), relaxTau: relaxTau,
            boundsMin: SIMD4(boundsMin, fallSpeed),
            boundsMax: SIMD4(boundsMax, turbulence),
            wind: SIMD4(wind, gustAmount),
            cam: SIMD4(camera, pixelAngle),
            moon: SIMD4(moonRadiance, flakeDiameter),
            pointPx: SIMD4(pointSizePx, edgeFade, flutter, sparkle),
            lightPos: (lp(0), lp(1), lp(2), lp(3), lp(4), lp(5), lp(6), lp(7)),
            lightCol: (lc(0), lc(1), lc(2), lc(3), lc(4), lc(5), lc(6), lc(7)))

        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "Illuminatorama.snow"
        enc.label = "Illuminatorama.snow.step"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(positionBuffer, offset: 0, index: 0)
        enc.setBuffer(velocityBuffer, offset: 0, index: 1)
        enc.setBuffer(colorBuffer, offset: 0, index: 2)
        enc.setBytes(&u, length: MemoryLayout<SnowUniforms>.stride, index: 3)
        let tg = min(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth * 8)
        enc.dispatchThreadgroups(MTLSize(width: (capacity + tg - 1) / tg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    public func source(ownerScene: ObjectIdentifier, colorScale: Float = 1.0) -> ParticleFieldSource {
        ParticleFieldSource(
            positionBuffer: positionBuffer, colorBuffer: colorBuffer, vertexCount: capacity,
            ownerScene: ownerScene,
            positionStrideFloats: 4, colorStrideFloats: 4,
            positionOffsetFloats: 0, colorOffsetFloats: 0,
            pointSize: pointSizePx, colorScale: colorScale)
    }
}
