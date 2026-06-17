import Foundation
import Metal
import simd

// ── LightPatternKernel ───────────────────────────────────────────────────────
//
// Drives an HDR colour pattern over N slots into a 1×N `rgba32Float`
// MTLTexture, one thread per slot. Replaces the per-tick CPU loop that
// used to compute pattern colours for Eggs' rail lights.
//
// One kernel per (device, bulb-count) is fine: pattern is a uniform, so
// switching between glow / chase / marquee / rainbow / sparkle is just a
// new uniform write and a fresh dispatch.
//
// Reusable for any future "string of lights" asset — neon signs, marquee
// bulbs, popcorn lamps, etc. The slot index is opaque; consumers decide
// what each slot represents.

/// Shared with the kernel — keep in lockstep with `LightPattern.metal`'s
/// `LightPatternUniforms`. Same alignment rule as the rest of the project:
/// float4 only, no bare `SIMD3<Float>` in shared structs.
public struct LightPatternUniforms {
    public var baseColor:   SIMD4<Float>
    public var accentColor: SIMD4<Float>
    public var time:        Float
    public var speed:       Float
    public var intensity:   Float
    public var emissionCap: Float
    public var pattern:     UInt32
    public var bulbCount:   UInt32
    public var _pad0:       Float
    public var _pad1:       Float

    public init(
        baseColor: SIMD3<Float>,
        accentColor: SIMD3<Float>,
        time: Float,
        speed: Float,
        intensity: Float,
        emissionCap: Float,
        pattern: UInt32,
        bulbCount: UInt32
    ) {
        self.baseColor   = SIMD4<Float>(baseColor, 0)
        self.accentColor = SIMD4<Float>(accentColor, 0)
        self.time = time
        self.speed = speed
        self.intensity = intensity
        self.emissionCap = emissionCap
        self.pattern = pattern
        self.bulbCount = bulbCount
        self._pad0 = 0
        self._pad1 = 0
    }
}

/// Pattern enum mirrors the kernel's switch arms. Renumbering breaks the
/// kernel; add new patterns at the end.
public enum LightPatternKind: UInt32 {
    case glow    = 0
    case chase   = 1
    case marquee = 2
    case rainbow = 3
    case sparkle = 4
}

@MainActor
public final class LightPatternKernel {

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    public init?(engine: SimEngine = .shared) {
        self.engine = engine
        guard let pipeline = engine.pipeline("light_pattern_write") else {
            return nil
        }
        self.pipeline = pipeline
        self.threadgroupSize = min(64, pipeline.maxTotalThreadsPerThreadgroup)
    }

    /// Dispatch the kernel into `lut`. `lut.width` must equal
    /// `uniforms.bulbCount`; `lut.height` must be 1.
    public func dispatch(
        uniforms: LightPatternUniforms,
        lut: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        guard uniforms.bulbCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "LightPatternKernel.dispatch"
        enc.setComputePipelineState(pipeline)
        var u = uniforms
        enc.setBytes(&u, length: MemoryLayout<LightPatternUniforms>.stride, index: 0)
        enc.setTexture(lut, index: 0)
        let grid = MTLSize(width: Int(uniforms.bulbCount), height: 1, depth: 1)
        let group = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}
