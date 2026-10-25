import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

/// The near-camera crystal snow: a small GPU pool of flakes drawn as real six-fold ice-crystal shapes
/// (see `IlluminatoramaCrystals.metal`). Owns one 64-byte-per-flake buffer that the compute kernel
/// advances and the renderer's crystal pass reads, plus the per-frame camera/light uniforms.
///
/// The pool lives in a frustum-shaped volume in front of the camera (a few hundred flakes between
/// ~0.5 m and ~5 m), recycled as the camera moves, so the nearest snow is always modelled flakes.
/// Far snow stays on the cheap point-sprite `IlluminatoramaSnowField`.
@MainActor
public final class IlluminatoramaCrystalFlakeField {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatoramaCrystals")

    /// Mirror of `CrystalStepUniforms` (every vector float4).
    private struct StepUniforms {
        var dt: Float; var time: Float; var count: UInt32; var relaxTau: Float
        var camPos: SIMD4<Float>
        var camFwd: SIMD4<Float>
        var camRightT: SIMD4<Float>
        var camUpT: SIMD4<Float>
        var wind: SIMD4<Float>
        var dyn: SIMD4<Float>
        var size: SIMD4<Float>
        var moon: SIMD4<Float>
        var lightPos: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                       SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var lightCol: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
                       SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    }

    public let capacity: Int
    public let flakeBuffer: MTLBuffer
    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState

    // Tunables (host sets before `update`).
    public var fallSpeed: Float = 0.7            // big dendrites fall slowly
    public var wind: SIMD3<Float> = SIMD3(0.6, 0, 0.2)
    public var gustAmount: Float = 0.7
    public var turbulence: Float = 0.30
    public var flutter: Float = 0.22
    public var tumbleAmplitude: Float = 0.95     // rad — the pendulum rock of a falling plate
    public var relaxTau: Float = 0.5
    public var nearDistance: Float = 0.5
    public var farDistance: Float = 4.5
    /// Flake radius range, metres (real big dendrites are 3–10 mm across the arms; a touch generous
    /// so the crystal reads at a few metres).
    public var minRadius: Float = 0.010
    public var maxRadius: Float = 0.026
    public var groundY: Float = 0
    public var moonRadiance: SIMD3<Float> = SIMD3(0.20, 0.23, 0.30)
    public var fadeIn: Float = 0.5

    public struct FlakeLight {
        public var position: SIMD3<Float>; public var color: SIMD3<Float>
        public init(position: SIMD3<Float>, color: SIMD3<Float>) { self.position = position; self.color = color }
    }

    public init?(engine: SimEngine, capacity: Int) {
        guard capacity > 0,
              let pipe = engine.pipelineCache.pipelineState(name: "illumi_crystal_step", device: engine.device),
              let buf = engine.device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * 4 * capacity,
                                                 options: .storageModeShared) else {
            Self.log.error("crystal flake field unavailable")
            return nil
        }
        buf.label = "Illuminatorama.crystals"
        self.engine = engine; self.pipeline = pipe; self.capacity = capacity; self.flakeBuffer = buf
        // Everything starts far below the ground so the first step recycles it into the pool.
        let p = buf.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity * 4)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<capacity {
            p[i * 4 + 0] = SIMD4(0, -1000, 0, 0.01)
            p[i * 4 + 1] = SIMD4(0, 0, 0, -2)
            p[i * 4 + 2] = SIMD4(0, 1, 0, Float.random(in: 0...6.28, using: &rng))
            p[i * 4 + 3] = SIMD4(0, 0, 0, Float(Int.random(in: 0...3, using: &rng)) + Float.random(in: 0...0.99, using: &rng))
        }
    }

    /// Advance the pool. `camera` basis: `right`/`up` are scaled by tan(half-fov) of the SAME camera
    /// the renderer draws with, so the pool fills exactly the visible frustum.
    public func update(dt: Float, time: Float, camPos: SIMD3<Float>, forward: SIMD3<Float>,
                       right: SIMD3<Float>, up: SIMD3<Float>, tanHalfFovY: Float, aspect: Float,
                       lights: [FlakeLight] = []) {
        func lp(_ i: Int) -> SIMD4<Float> { i < lights.count ? SIMD4(lights[i].position, 0) : .zero }
        func lc(_ i: Int) -> SIMD4<Float> { i < lights.count ? SIMD4(lights[i].color, 0) : .zero }
        var u = StepUniforms(
            dt: max(0, min(0.05, dt)), time: time, count: UInt32(capacity), relaxTau: relaxTau,
            camPos: SIMD4(camPos, farDistance),
            camFwd: SIMD4(simd_normalize(forward), nearDistance),
            camRightT: SIMD4(simd_normalize(right) * tanHalfFovY * aspect, 0),
            camUpT: SIMD4(simd_normalize(up) * tanHalfFovY, 0),
            wind: SIMD4(wind, gustAmount),
            dyn: SIMD4(fallSpeed, turbulence, flutter, tumbleAmplitude),
            size: SIMD4(minRadius, maxRadius, groundY, fadeIn),
            moon: SIMD4(moonRadiance, 0),
            lightPos: (lp(0), lp(1), lp(2), lp(3), lp(4), lp(5), lp(6), lp(7)),
            lightCol: (lc(0), lc(1), lc(2), lc(3), lc(4), lc(5), lc(6), lc(7)))
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "Illuminatorama.crystals"
        enc.label = "Illuminatorama.crystals.step"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(flakeBuffer, offset: 0, index: 0)
        enc.setBytes(&u, length: MemoryLayout<StepUniforms>.stride, index: 1)
        let tg = min(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth * 4)
        enc.dispatchThreadgroups(MTLSize(width: (capacity + tg - 1) / tg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }
}
