import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── ILLUMINATORAMA PARTICLES (Phase 4.11 — Metal-native particle subsystem) ──
//
// `SCNParticleSystem` is silently dropped by the scene extractor (it's
// CPU-driven, lifetime-managed by SceneKit, and impossible to extract from
// outside). Per the project's "always Metal" rule, the replacement is a
// compute-driven particle subsystem that lives inside Illuminatorama: a
// caller creates an `IlluminatoramaParticleEmitter`, hands it the renderer,
// and per-frame the renderer integrates positions via compute and draws the
// surviving particles as additive-HDR point sprites into the post-lighting
// composite. Bloom + TAA + tonemap process them naturally because they
// land in the same `hdrCompositeTexture` the deferred lighting writes to.
//
// First-cut design (this turn):
//   • Fixed-capacity ring of particles. Caller calls `emit(...)` to push
//     a burst of N new particles; the kernel decrements `life` per tick
//     and tags dead particles via `life <= 0`. Burst-based for now —
//     continuous emission is just per-frame bursts on the host side.
//   • Point primitives via `.point` element. Simpler than billboard
//     quads — single draw call, GPU expands. Looks great for sparks,
//     embers, dust motes, fireflies.
//   • Per-particle: position, velocity, color (HDR, premultiplied), life
//     (0–1, 1 = freshly emitted), size (in NDC-relative). 48 bytes per
//     particle; matches `Particle` in Illuminatorama.metal.
//   • Gravity + drag are scalar host knobs (no per-particle forces yet).
//     Adequate for sparks / embers; per-emitter wind would land in 4.12.
//
// What this does NOT yet do:
//   • Billboard quads (point primitives only). Add when the first scene
//     wants soft particles with a texture.
//   • SCNParticleSystem extraction. Per "always Metal" we don't try.
//     Scenes that want particles instantiate an
//     `IlluminatoramaParticleEmitter` directly from their controller.

/// Per-particle state. Matches the Metal `Particle` struct byte-for-byte.
/// 48 bytes; stride 48. Fields are laid out so SIMD3<Float> (16-byte stride)
/// pads correctly against the scalar `life`/`size` slots without explicit
/// trailing padding.
public struct IlluminatoramaParticle {
    public var position: SIMD3<Float>
    public var life: Float            // 0–1; 1 = just emitted, 0 = dead
    public var velocity: SIMD3<Float>
    public var size: Float            // point sprite size in pixels at distance 1
    public var color: SIMD3<Float>    // HDR linear RGB, premultiplied with intensity
    public var _pad: Float = 0

    public init(position: SIMD3<Float>,
                velocity: SIMD3<Float>,
                color: SIMD3<Float>,
                life: Float = 1.0,
                size: Float = 8.0) {
        self.position = position
        self.life = life
        self.velocity = velocity
        self.size = size
        self.color = color
    }
}

/// Per-frame uniforms the integration kernel reads. Matches Metal-side
/// `ParticleFrameUniforms`.
public struct IlluminatoramaParticleFrameUniforms {
    public var dt: Float
    public var gravity: SIMD3<Float>
    public var drag: Float
    public var capacity: UInt32
    public var _pad0: Float = 0
    public var _pad1: Float = 0
    public var _pad2: Float = 0

    public init(dt: Float,
                gravity: SIMD3<Float> = SIMD3(0, -9.81, 0),
                drag: Float = 0.0,
                capacity: UInt32) {
        self.dt = dt
        self.gravity = gravity
        self.drag = drag
        self.capacity = capacity
    }
}

@MainActor
public final class IlluminatoramaParticleEmitter {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                     category: "illuminatoramaParticles")

    /// Maximum number of in-flight particles. Fixed at init; a re-emit
    /// past capacity overwrites the oldest slot via the ring index.
    public let capacity: Int

    /// Per-particle MTLBuffer. Bound by the integration kernel
    /// (read-write) and the particle render pass (read-only vertex/frag).
    public let particleBuffer: MTLBuffer

    /// Gravity vector applied each step. Default Earth (-9.81 m/s² down).
    public var gravity: SIMD3<Float> = SIMD3(0, -9.81, 0)

    /// Per-second exponential damping on velocity. 0 = no drag; 1 = full
    /// stop in one second. Useful for embers fading in place.
    public var drag: Float = 0.0

    /// Toggle. When false the renderer skips the integration kernel and
    /// the draw pass for this emitter.
    public var enabled: Bool = true

    /// When set, the DDGI trace kernel treats this emitter as an analytic
    /// point light so probe irradiance (and therefore ALL scene surfaces)
    /// pick up the particles' glow.
    /// - `position`: world-space centroid of the particle cloud.
    /// - `color`: aggregate pre-multiplied HDR emission for the whole field.
    /// - `radius`: falloff distance in metres; contribution reaches zero at this distance.
    public var ddgiLight: (position: SIMD3<Float>, color: SIMD3<Float>, radius: Float)? = nil

    private let device: MTLDevice
    private var ringIndex: Int = 0

    public init(device: MTLDevice, capacity: Int = 4096) throws {
        precondition(capacity > 0, "Particle emitter capacity must be > 0")
        let bytes = MemoryLayout<IlluminatoramaParticle>.stride * capacity
        guard let buf = device.makeBuffer(length: bytes,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("particleBuffer")
        }
        // Zero out so untouched slots have `life = 0` and the render pass
        // skips them.
        memset(buf.contents(), 0, bytes)
        buf.label = "Illuminatorama.particles"
        self.particleBuffer = buf
        self.capacity = capacity
        self.device = device
    }

    /// Write `count` freshly-emitted particles into the ring, overwriting
    /// the oldest slots if we wrap. Caller supplies positions, velocities
    /// and colors as parallel arrays of length `count`.
    public func emit(count: Int,
                     positions: [SIMD3<Float>],
                     velocities: [SIMD3<Float>],
                     colors: [SIMD3<Float>],
                     size: Float = 8.0,
                     life: Float = 1.0) {
        precondition(positions.count >= count
                  && velocities.count >= count
                  && colors.count >= count,
                     "emit: parallel arrays must contain at least `count` elements")
        guard count > 0 else { return }
        let stride = MemoryLayout<IlluminatoramaParticle>.stride
        let base = particleBuffer.contents().bindMemory(to: IlluminatoramaParticle.self,
                                                          capacity: capacity)
        for i in 0..<count {
            let slot = (ringIndex + i) % capacity
            base[slot] = IlluminatoramaParticle(
                position: positions[i],
                velocity: velocities[i],
                color: colors[i],
                life: life,
                size: size
            )
            _ = stride // silence unused warning when count is 0
        }
        ringIndex = (ringIndex + count) % capacity
    }
}
