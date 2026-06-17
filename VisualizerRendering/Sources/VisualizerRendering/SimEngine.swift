import Foundation
import Metal
import OSLog
import VisualizerCore

// ── SIM ENGINE ───────────────────────────────────────────────────────────────
//
// The shared substrate every GPU simulation in the project consumes:
//   • the MTLDevice the simulation runs on
//   • a single MTLCommandQueue (lazily created — cheap, but no point making
//     more than one per scene)
//   • the default MTLLibrary the kernels were compiled into
//   • a `SimPipelineCache` for memoised pipeline-state lookups
//
// SimEngine is deliberately a thin holder, not a Hibernate-style "framework."
// The READMEs in /Reference/ (and ChatGPT's pitch for a "unified simulation
// runtime") were right that simulations share infrastructure — but wrong that
// the shared layer should be a protocol-heavy monolith. The shared layer that
// actually pays off is: device + queue + compiled pipelines + buffer storage
// conventions. Everything above that (constraint solver, MLS-MPM, SPH, grass)
// stays specialised, because the math really is different and a uniform
// "Constraint" protocol becomes awkward the moment fluids show up.
//
// USAGE
//
//   • Existing PBD callers don't need to thread an engine through — they
//     keep calling `PBDSolver(device:)`, which uses `SimPipelineCache.shared`
//     under the hood. Cache hits survive across spawns / scene rebuilds
//     exactly like they did when the cache was named `PBDPipelineCache`.
//
//   • New solvers (MLS-MPM fluid, SPH foam, grass, …) take a `SimEngine` in
//     their initialiser:
//
//         let engine = SimEngine.shared            // or SimEngine(device: d)
//         let fluid  = try MLSMPMSolver(engine: engine, config: …)
//         scene.rootNode.addChildNode(fluid.node)
//
//   • One engine per scene is the recommended granularity. The pipeline
//     cache is process-wide (`SimPipelineCache.shared`) so multiple scenes
//     share compiled pipelines automatically; the *queue* is per-engine so
//     each scene's command buffers don't serialise against each other any
//     more than Metal already requires.
//
// LIFECYCLE
//
// SimEngine itself doesn't allocate anything beyond the lazy queue — the
// device is borrowed, the library and pipelines live in the shared cache.
// It is safe to create and discard engines freely; the expensive work
// (library load, pipeline compilation) is done once per device and shared.

@MainActor
public final class SimEngine {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "SimEngine")

    /// Convenience engine wrapping `MTLCreateSystemDefaultDevice()` plus the
    /// process-wide `SimPipelineCache.shared`. Use this in the common single-
    /// device case where you don't need per-scene queue isolation.
    public static let shared: SimEngine = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            preconditionFailure("MTLCreateSystemDefaultDevice() returned nil — Metal-incapable host")
        }
        return SimEngine(device: device, cache: .shared)
    }()

    public let device: MTLDevice
    public let pipelineCache: SimPipelineCache

    public init(device: MTLDevice, cache: SimPipelineCache = .shared) {
        self.device = device
        self.pipelineCache = cache
    }

    /// Lazily-created per-engine command queue. Solvers that batch multiple
    /// dispatches into a single command buffer should reuse this queue rather
    /// than making one of their own — saves the queue-allocation cost on
    /// every spawn at high spawn rates.
    public var commandQueue: MTLCommandQueue {
        if let q = _commandQueue { return q }
        guard let q = device.makeCommandQueue() else {
            preconditionFailure("MTLDevice.makeCommandQueue() returned nil")
        }
        q.label = "SimEngine"
        _commandQueue = q
        return q
    }
    private var _commandQueue: MTLCommandQueue?

    /// The default Metal library for this engine's device. Nil if the bundle
    /// ships no `.metallib` (programmer error — surface up to the caller).
    public var library: MTLLibrary? {
        pipelineCache.library(for: device)
    }

    /// Sugar over `pipelineCache.pipelineState(name:device:)` so callers can
    /// write `engine.pipeline("mls_p2g")` without restating the device.
    public func pipeline(_ name: String) -> MTLComputePipelineState? {
        pipelineCache.pipelineState(name: name, device: device)
    }

    // ── Particle-field registry ───────────────────────────────────────────────
    //
    // The shared handoff for GPU particle buffers that need to render in BOTH
    // render paths (SceneKit's forward draw AND the Illuminatorama overlay).
    //
    // A solver's renderer-shim (FoamRenderer, FireworkParticleRenderer) already
    // wraps the solver's MTLBuffer in an SCNGeometry so SceneKit can draw it.
    // Illuminatorama doesn't go through SceneKit's draw path, so it needs its
    // own way to reach the same buffer. Rather than launder the buffer back
    // out through a SceneKit object (the earlier `illuminatoramaExternalPoint`
    // associated-object bridge), the shim registers a `ParticleFieldSource`
    // here — package-internal, no round-trip — and the Illuminatorama extractor
    // reads it each frame.
    //
    // SCOPING. The registry is process-wide (`SimEngine.particleFields`) so it
    // works regardless of which engine instance a scene used. Controllers are
    // CACHED across scene switches in `AppModel` (an inactive scene's
    // controller and its shims stay alive, tick-timer off), so deinit can't be
    // relied on to clear stale entries. Each source therefore carries its
    // owning SCNScene's `ObjectIdentifier`; the Illuminatorama extractor reads
    // only the active scene's sources. Re-registration is idempotent (keyed by
    // the position buffer's identity), so a re-attach replaces in place.
    public static let particleFields = ParticleFieldRegistry()
}

/// One GPU particle buffer published for cross-renderer consumption. The
/// fields mirror what the Illuminatorama point pipeline needs to draw the
/// buffer as additive HDR sprites; `ownerScene` scopes the source to the
/// SCNScene that owns it so the overlay only draws the active scene's fields.
public struct ParticleFieldSource {
    public let positionBuffer: MTLBuffer
    public let colorBuffer: MTLBuffer
    public let vertexCount: Int
    /// Position / colour strides in *floats* (e.g. 8 for a 32-byte particle).
    public let positionStrideFloats: UInt32
    public let colorStrideFloats: UInt32
    /// Float offsets of the position / colour's first component within each
    /// stride slot. Non-zero `colorOffsetFloats` lets ONE interleaved buffer
    /// feed both position and colour (FireworkParticle's `displayColor` sits
    /// 8 floats into the 24-float struct).
    public let positionOffsetFloats: UInt32
    public let colorOffsetFloats: UInt32
    public let pointSize: Float
    public let colorScale: Float
    /// Identity of the SCNScene that owns this field — the overlay's
    /// active-scene filter key.
    public let ownerScene: ObjectIdentifier
    /// When set, the DDGI trace kernel treats this field as an analytic point
    /// light so all scene surfaces pick up the particles' glow.
    public let ddgiLight: (position: SIMD3<Float>, color: SIMD3<Float>, radius: Float)?

    public init(positionBuffer: MTLBuffer,
                colorBuffer: MTLBuffer,
                vertexCount: Int,
                ownerScene: ObjectIdentifier,
                positionStrideFloats: UInt32 = 4,
                colorStrideFloats: UInt32 = 4,
                positionOffsetFloats: UInt32 = 0,
                colorOffsetFloats: UInt32 = 0,
                pointSize: Float = 2.5,
                colorScale: Float = 1.0,
                ddgiLight: (position: SIMD3<Float>, color: SIMD3<Float>, radius: Float)? = nil) {
        self.positionBuffer = positionBuffer
        self.colorBuffer = colorBuffer
        self.vertexCount = vertexCount
        self.ownerScene = ownerScene
        self.positionStrideFloats = positionStrideFloats
        self.colorStrideFloats = colorStrideFloats
        self.positionOffsetFloats = positionOffsetFloats
        self.colorOffsetFloats = colorOffsetFloats
        self.pointSize = pointSize
        self.colorScale = colorScale
        self.ddgiLight = ddgiLight
    }
}

/// Process-wide table of `ParticleFieldSource`, keyed by position-buffer
/// identity so re-registration replaces in place. `@MainActor` because both
/// the producing shims and the consuming extractor run on the main actor.
@MainActor
public final class ParticleFieldRegistry {
    private var sources: [ObjectIdentifier: ParticleFieldSource] = [:]

    public init() {}

    public func register(_ source: ParticleFieldSource) {
        sources[ObjectIdentifier(source.positionBuffer)] = source
    }

    public func unregister(positionBuffer: MTLBuffer) {
        sources.removeValue(forKey: ObjectIdentifier(positionBuffer))
    }

    /// All sources owned by the given SCNScene identity. The Illuminatorama
    /// extractor passes `ObjectIdentifier(activeScene)` so inactive scenes'
    /// still-registered fields (cached controllers) are filtered out.
    public func sources(forScene scene: ObjectIdentifier) -> [ParticleFieldSource] {
        sources.values.filter { $0.ownerScene == scene }
    }
}
