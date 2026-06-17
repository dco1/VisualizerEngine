import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── FIREWORK PARTICLE SOLVER ────────────────────────────────────────────────
//
// GPU-resident particle simulation for the Fireworks+ scene. One big shared
// `SimBuffer<FireworkParticle>` (capacity ≈ 32k) holds every live spark; the
// host manages a free-slot ring allocator that hands out contiguous ranges to
// each burst, and dispatches `fwIntegrate` every frame.
//
// USAGE
//
//   let engine = SimEngine.shared
//   guard let solver = FireworkParticleSolver(engine: engine, capacity: 32_000) else { ... }
//   scene.rootNode.addChildNode(renderer.node)   // see FireworkParticleRenderer
//
//   // Per frame, inside the controller's tick:
//   let cb = engine.commandQueue.makeCommandBuffer()!
//   solver.encodeStep(into: cb, dt: dt, time: t, gravity: g, wind: w)
//   cb.commit()
//
//   // To spawn a burst:
//   solver.spawnBurst(center: pos, kind: .sphere, baseColor: c, …)
//
// LIFECYCLE
//
// All slots are always "drawn" — dead particles get parked far below the
// scene (y = −10000) so the renderer doesn't need to dynamically resize the
// draw call. Free-slot bookkeeping is a simple ring cursor: bursts append
// at `nextSlot` and wrap, which is fine because the buffer is sized larger
// than `maxAliveParticles × peakBurstRate × peakLife`.

// ── Particle struct ─────────────────────────────────────────────────────────
//
// 96 bytes, all float4 — see ALIGNMENT RULE in PBDSolver.swift. The first
// 12 bytes (positionAge.xyz) are read by SceneKit as the vertex position; the
// next 32 bytes are skipped; bytes 32..<44 (displayColor.rgb) are read as
// the vertex color.
//
// `colorTransition` (last 16 bytes) enables real-fireworks color-change
// stars: xyz holds the SECONDARY colour the star transitions into part-way
// through its life; w holds `transitionAt` ∈ [0, 1] — the age fraction
// where the chemistry crosses over. `w == 0` is the sentinel for "no
// transition, stay on baseDrag.rgb the whole life."

public struct FireworkParticle {
    public var positionAge:     SIMD4<Float>
    public var velocityLife:    SIMD4<Float>
    public var displayColor:    SIMD4<Float>
    public var baseDrag:        SIMD4<Float>
    public var paramsExtra:     SIMD4<Float>
    public var colorTransition: SIMD4<Float>

    public init() {
        self.positionAge     = SIMD4(0, -10000, 0, 0)
        self.velocityLife    = SIMD4(0, 0, 0, 0)
        self.displayColor    = SIMD4(0, 0, 0, 0)
        self.baseDrag        = SIMD4(0, 0, 0, 0)
        self.paramsExtra     = SIMD4(0, 0, 0, 0)
        self.colorTransition = SIMD4(0, 0, 0, 0)
    }
}

struct FWIntegrateUniforms {
    var dtTimeCount: SIMD4<Float>
    var gravity:     SIMD4<Float>
    var wind:        SIMD4<Float>
}

struct FWSpawnUniforms {
    var centerKind:         SIMD4<Float>
    var baseColor:          SIMD4<Float>
    var speedSizeLife:      SIMD4<Float>
    var dragTurbGravLife:   SIMD4<Float>
    var firstCountSeedTime: SIMD4<Float>
    var ringAxis:           SIMD4<Float>
    var secondaryColor:     SIMD4<Float>  // rgb = secondary, a = transition chance
}

struct FWStreakUniforms {
    var cameraCount:  SIMD4<Float>
    var streakParams: SIMD4<Float>
}

struct FWSparkUniforms {
    var baseColorJitter:    SIMD4<Float>
    var speedLifeJitter:    SIMD4<Float>
    var dragTurbGrav:       SIMD4<Float>
    var firstCountSeedTime: SIMD4<Float>
}

/// Per-spark emit record used by `spawnSparks` — the host fills one per
/// climbing shell (or N per shell at higher trail densities) and dispatches
/// `fwSpawnSparks`. Single float4 keeps the buffer compact and the alignment
/// rules in PBDSolver.swift trivially satisfied.
public struct FWSparkEmit {
    public var positionSeed: SIMD4<Float>

    public init(position: SIMD3<Float>, seed: UInt32) {
        self.positionSeed = SIMD4(position.x, position.y, position.z,
                                  Float(bitPattern: seed))
    }
}

/// Parameters for a single spark-spawn batch — typically reused frame after
/// frame for a particular trail effect. The host calls `spawnSparks` with
/// these + the array of emit positions; the kernel handles everything else.
public struct FireworkSparkParams {
    /// Base colour pre-scaled to land inside HDR after `fwIntegrate`'s
    /// brightness profile. For a soft warm trail, ~0.20 × (1.0, 0.82, 0.45)
    /// keeps the bloom present but not blinding.
    public var baseColor: SIMD3<Float>
    public var colorJitter: Float
    public var speed: Float
    public var speedJitter: Float
    public var lifespan: Float
    public var lifeJitter: Float
    public var drag: Float
    public var turbulence: Float
    public var gravityMultiplier: Float
    public var size: Float

    public init(
        baseColor: SIMD3<Float>,
        colorJitter: Float = 0.18,
        speed: Float = 0.5,
        speedJitter: Float = 0.35,
        lifespan: Float = 0.55,
        lifeJitter: Float = 0.18,
        drag: Float = 0,
        turbulence: Float = 0,
        gravityMultiplier: Float = 0.22,
        size: Float = 0.18
    ) {
        self.baseColor = baseColor
        self.colorJitter = colorJitter
        self.speed = speed
        self.speedJitter = speedJitter
        self.lifespan = lifespan
        self.lifeJitter = lifeJitter
        self.drag = drag
        self.turbulence = turbulence
        self.gravityMultiplier = gravityMultiplier
        self.size = size
    }
}

/// Per-vertex streak-quad vertex written by `fwBuildStreaks` and read by
/// SceneKit as the vertex / color sources of the streak geometry. 32 bytes,
/// 4 vertices per particle, 6 indices per particle (two triangles).
public struct FWStreakVertex {
    public var position: SIMD4<Float>
    public var color:    SIMD4<Float>

    public init() {
        self.position = SIMD4(0, -10000, 0, 1)
        self.color    = SIMD4(0, 0, 0, 0)
    }
}

// ── Burst parameters ────────────────────────────────────────────────────────

/// What geometric shape the spawn kernel produces for a burst's initial
/// velocity field. Differences between chrysanthemum / peony / willow live in
/// the integrator parameters (drag, gravity, life) — those three all share
/// `kind = .sphere`.
public enum FireworkBurstKind: UInt32 {
    case sphere = 0
    case ring   = 1
    case heart  = 2
    /// US-flag tableau. Particles teleport to a 30 × 16 grid in the plane
    /// perpendicular to `orientationAxis`, coloured red/white stripes with
    /// a blue canton + white "stars" in the upper-left. `speed` is reused
    /// as the flag's half-width in metres (the grid maps from normalised
    /// [-1, 1] × [-1, 1] coords); set `gravityMultiplier` near 0 and
    /// `speedJitter`/`turbulence`/`drag` to 0 so the tableau holds.
    case flag   = 3
}

/// Everything the spawn kernel needs to populate a contiguous slot range.
/// Built by the scene controller from its higher-level burst-pattern enum.
public struct FireworkBurstParams {
    public var center: SIMD3<Float>
    public var kind: FireworkBurstKind
    public var baseColor: SIMD3<Float>
    /// Random "nudge each spark toward white" amount in 0...1. Larger values
    /// give a "core + halo" feel as some sparks burn hotter than the base hue.
    public var colorJitter: Float
    /// Initial outward speed in m/s.
    public var speed: Float
    public var speedJitter: Float
    /// Lifespan in seconds. The actual life of each particle is
    /// `lifespan ± lifeJitter`.
    public var lifespan: Float
    public var lifeJitter: Float
    /// Visual size hint passed through to the renderer for any future
    /// per-particle sizing. Unused by the point-primitive renderer but reserved
    /// so the API is stable when we move to billboards.
    public var size: Float
    /// Per-second exponential drag. 0 = no drag; larger = faster slowdown.
    public var drag: Float
    /// Per-second turbulence amplitude applied to velocity. Adds the wandery
    /// motion that willows and long-lived embers need to read as organic.
    public var turbulence: Float
    /// Gravity multiplier per-particle. 1 = full gravity; 2.5 = willow droop;
    /// 0.0 = ring/heart hovering.
    public var gravityMultiplier: Float
    /// Number of particles to spawn. Capped at the solver's remaining capacity.
    public var particleCount: Int
    /// Axis used for ring/heart orientation. Ignored for `.sphere`. Default
    /// points the ring's plane normal at the camera (so the ring/heart faces
    /// the viewer) — controller picks this from camera position.
    public var orientationAxis: SIMD3<Float>
    /// Out-of-plane thickness for ring/heart in 0..1. Larger = thicker tube.
    public var ringThickness: Float
    /// Probability ∈ [0, 1] that any individual spark in this burst
    /// transitions to `secondaryColor` mid-life. Real fireworks colour-
    /// changers usually transition almost-all stars together (single
    /// chemistry layer burning out), so the host typically sets this to
    /// either 0 (no transition) or 0.85+ (whole burst transitions).
    public var secondaryColorChance: Float
    /// Color the spark transitions INTO when chosen. Ignored when
    /// `secondaryColorChance == 0`.
    public var secondaryColor: SIMD3<Float>

    public init(
        center: SIMD3<Float>,
        kind: FireworkBurstKind = .sphere,
        baseColor: SIMD3<Float>,
        colorJitter: Float = 0.25,
        speed: Float = 9,
        speedJitter: Float = 1.6,
        lifespan: Float = 2.0,
        lifeJitter: Float = 0.6,
        size: Float = 0.3,
        drag: Float = 0.4,
        turbulence: Float = 0,
        gravityMultiplier: Float = 1.0,
        particleCount: Int = 260,
        orientationAxis: SIMD3<Float> = SIMD3(0, 0, 1),
        ringThickness: Float = 0.1,
        secondaryColorChance: Float = 0,
        secondaryColor: SIMD3<Float> = SIMD3(1, 1, 1)
    ) {
        self.center = center
        self.kind = kind
        self.baseColor = baseColor
        self.colorJitter = colorJitter
        self.speed = speed
        self.speedJitter = speedJitter
        self.lifespan = lifespan
        self.lifeJitter = lifeJitter
        self.size = size
        self.drag = drag
        self.turbulence = turbulence
        self.gravityMultiplier = gravityMultiplier
        self.particleCount = particleCount
        self.orientationAxis = orientationAxis
        self.ringThickness = ringThickness
        self.secondaryColorChance = secondaryColorChance
        self.secondaryColor = secondaryColor
    }
}

// ── Solver ──────────────────────────────────────────────────────────────────

@MainActor
public final class FireworkParticleSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "FireworkParticleSolver")

    public let device: MTLDevice
    public let engine: SimEngine

    /// Per-particle state. SceneKit aliases this buffer in
    /// `FireworkParticleRenderer` — do not reallocate after init.
    public let particleBuffer: SimBuffer<FireworkParticle>

    /// Per-frame streak-quad vertices. 4 vertices per particle, rewritten by
    /// `fwBuildStreaks` every tick. SceneKit reads this buffer directly as the
    /// streak geometry's vertex source — do not reallocate after init.
    public let streakBuffer: SimBuffer<FWStreakVertex>

    /// Per-frame streak-quad texture coordinates. 4 UVs per particle (matching
    /// `streakBuffer`'s layout), rewritten by `fwBuildStreaks` every tick. The
    /// kernel writes V from 0 (tail) to `streakLen / streakWidth` (head) so
    /// the Gaussian-disc sprite tiles repeatedly along the velocity axis when
    /// the streak is long. With the sprite material's wrap mode set to
    /// `.repeat`, this reads as a chain of soft round dots fading along the
    /// trail instead of one stretched ellipse — exactly the "collection of
    /// dying light particles" look. SceneKit binds this as the geometry's
    /// texcoord source — do not reallocate after init.
    public let streakUVBuffer: SimBuffer<SIMD2<Float>>

    private let integratePipeline:    MTLComputePipelineState
    private let spawnPipeline:        MTLComputePipelineState
    private let spawnSparksPipeline:  MTLComputePipelineState
    private let buildStreaksPipeline: MTLComputePipelineState

    private let integrateUniformBuf:    MTLBuffer
    private let buildStreaksUniformBuf: MTLBuffer

    // ── Spawn N-buffering ─────────────────────────────────────────────────────
    //
    // `spawnBurst` / `spawnSparks` each commit their own command buffer WITHOUT
    // waiting for completion, so several spawns can be queued before the GPU
    // runs any of them (compound bursts firing 3–5 children in one tick, a
    // finale barrage, etc.). A single shared uniform buffer would be overwritten
    // by every call, so every queued dispatch would read the LAST burst's
    // colour / centre / slot range and they'd all collapse into one burst.
    //
    // Fix: a ring of `inFlightSpawns` uniform slots inside one MTLBuffer. Each
    // spawn writes its uniforms into the next slot and binds `buffer(1)` at that
    // slot's offset — the kernel is unchanged, it just reads a different region.
    // The spark emit buffer is partitioned the same way (one region per slot)
    // so back-to-back `spawnSparks` calls don't clobber each other's emit
    // positions either. Depth 16 comfortably covers any realistic per-tick spawn
    // count (and however many command buffers the queue keeps in flight) without
    // any per-spawn CPU↔GPU sync — that stays on the off-tick capture path only.
    private static let inFlightSpawns = 16

    /// Constant-buffer offsets must be aligned; 256 is universally safe across
    /// the Metal devices we target.
    private static let uniformOffsetAlignment = 256

    private static func alignUp(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }

    private let spawnUniformBuf:       MTLBuffer
    private let spawnUniformStride:    Int
    private var spawnUniformCursor:    Int = 0

    private let spawnSparksUniformBuf: MTLBuffer
    private let spawnSparksStride:     Int
    private var spawnSparksCursor:     Int = 0

    /// Shared per-tick emit buffer for spark spawning, partitioned into
    /// `inFlightSpawns` regions (one per uniform-ring slot) so concurrent
    /// in-flight spawns don't overwrite each other's emit positions. Each region
    /// holds `sparkEmitCapacity` records; emits are transient and rewritten each
    /// call.
    private let sparkEmitBuffer: SimBuffer<FWSparkEmit>
    private let sparkEmitRegionCapacity: Int

    /// Ring cursor over `particleBuffer`. Bursts allocate from here and wrap
    /// around. Slots that haven't been spawned yet hold an "already dead"
    /// particle (life = 0) so the integrator parks them at y = −10000.
    private var nextSlot: Int = 0
    /// Cumulative seed counter — incremented per burst so successive bursts
    /// see different per-particle random streams even when their params match.
    private var burstSeed: UInt32 = 0x9E3779B9

    public init?(engine: SimEngine, capacity: Int, sparkEmitCapacity: Int = 512) {
        let device = engine.device
        guard
            let integrate    = engine.pipeline("fwIntegrate"),
            let spawn        = engine.pipeline("fwSpawnBurst"),
            let spawnSparks  = engine.pipeline("fwSpawnSparks"),
            let buildStreaks = engine.pipeline("fwBuildStreaks")
        else {
            Self.log.error("FireworkParticleSolver pipeline lookup failed — is Fireworks.metal in VisualizerRendering/Shaders/?")
            return nil
        }
        let spawnStride = Self.alignUp(MemoryLayout<FWSpawnUniforms>.stride,
                                       to: Self.uniformOffsetAlignment)
        let sparkStride = Self.alignUp(MemoryLayout<FWSparkUniforms>.stride,
                                       to: Self.uniformOffsetAlignment)
        guard
            let pBuf = SimBuffer<FireworkParticle>(device: device,
                                                   capacity: capacity,
                                                   label: "Fireworks.particles"),
            let sBuf = SimBuffer<FWStreakVertex>(device: device,
                                                 capacity: capacity * 4,
                                                 label: "Fireworks.streakVerts"),
            let uvBuf = SimBuffer<SIMD2<Float>>(device: device,
                                                capacity: capacity * 4,
                                                label: "Fireworks.streakUVs"),
            // One emit region per in-flight spawn slot, so concurrent spawns
            // don't overwrite each other's positions.
            let emitBuf = SimBuffer<FWSparkEmit>(device: device,
                                                 capacity: sparkEmitCapacity * Self.inFlightSpawns,
                                                 label: "Fireworks.sparkEmits"),
            let intU = device.makeBuffer(length: MemoryLayout<FWIntegrateUniforms>.stride,
                                          options: .storageModeShared),
            let spnU = device.makeBuffer(length: spawnStride * Self.inFlightSpawns,
                                          options: .storageModeShared),
            let spkU = device.makeBuffer(length: sparkStride * Self.inFlightSpawns,
                                          options: .storageModeShared),
            let bsU  = device.makeBuffer(length: MemoryLayout<FWStreakUniforms>.stride,
                                          options: .storageModeShared)
        else {
            Self.log.error("FireworkParticleSolver buffer allocation failed")
            return nil
        }
        intU.label = "Fireworks.integrateUniforms"
        spnU.label = "Fireworks.spawnUniforms"
        spkU.label = "Fireworks.spawnSparksUniforms"
        bsU.label  = "Fireworks.streakUniforms"

        // Seed the whole buffer with dead particles so the very first integrate
        // pass parks them all at y = −10000 instead of leaving uninitialised
        // memory at the visible origin.
        let dead = FireworkParticle()
        var seeded = Array(repeating: dead, count: capacity)
        seeded.withUnsafeMutableBufferPointer { p in
            pBuf.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<FireworkParticle>.stride * capacity
            )
        }
        // SimBuffer.count tracks "intended population"; for this solver every
        // slot is drawn regardless, so we set it to capacity once and never
        // touch it again.
        pBuf.write(seeded)

        // Seed the streak buffer with offscreen vertices so the first frame
        // (before fwBuildStreaks has run) doesn't render a sheet of zero-
        // initialised triangles at the world origin.
        let deadVert = FWStreakVertex()
        var seededVerts = Array(repeating: deadVert, count: capacity * 4)
        seededVerts.withUnsafeMutableBufferPointer { p in
            sBuf.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<FWStreakVertex>.stride * capacity * 4
            )
        }

        self.device                 = device
        self.engine                 = engine
        self.particleBuffer         = pBuf
        self.streakBuffer           = sBuf
        self.streakUVBuffer         = uvBuf
        self.sparkEmitBuffer        = emitBuf
        self.integratePipeline      = integrate
        self.spawnPipeline          = spawn
        self.spawnSparksPipeline    = spawnSparks
        self.buildStreaksPipeline   = buildStreaks
        self.integrateUniformBuf    = intU
        self.spawnUniformBuf        = spnU
        self.spawnUniformStride     = spawnStride
        self.spawnSparksUniformBuf  = spkU
        self.spawnSparksStride      = sparkStride
        self.sparkEmitRegionCapacity = sparkEmitCapacity
        self.buildStreaksUniformBuf = bsU
    }

    // ── Step ────────────────────────────────────────────────────────────────

    /// Encode one integration pass into the given command buffer. The caller
    /// is responsible for committing the buffer (typically once per frame,
    /// batched with any other GPU work).
    public func encodeStep(
        into cb: MTLCommandBuffer,
        dt: Float,
        time: Float,
        gravity: SIMD3<Float>,
        wind: SIMD3<Float>
    ) {
        let count = particleBuffer.capacity
        guard count > 0 else { return }

        // Pack count into the uniforms via bitcast. `as_type<uint>(float)` on
        // the GPU side undoes this.
        let countAsFloat = Float(bitPattern: UInt32(count))
        var u = FWIntegrateUniforms(
            dtTimeCount: SIMD4(dt, time, countAsFloat, 0),
            gravity:     SIMD4(gravity, 0),
            wind:        SIMD4(wind, 0)
        )
        integrateUniformBuf.contents()
            .copyMemory(from: &u, byteCount: MemoryLayout<FWIntegrateUniforms>.stride)

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Fireworks.integrate"
        enc.setComputePipelineState(integratePipeline)
        enc.setBuffer(particleBuffer.buffer,  offset: 0, index: 0)
        enc.setBuffer(integrateUniformBuf,    offset: 0, index: 1)
        let tg = min(count, integratePipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    // ── Streak build ────────────────────────────────────────────────────────

    /// Encode the per-frame streak-quad build pass. Reads the particle buffer
    /// (must be encoded after `encodeStep` in the same command buffer so the
    /// quads track the integrated positions, not the previous frame's) and
    /// writes 4 vertices per particle into `streakBuffer`. SceneKit draws
    /// straight out of that buffer via `FireworkParticleRenderer`.
    public func encodeBuildStreaks(
        into cb: MTLCommandBuffer,
        cameraPosition: SIMD3<Float>,
        streakDuration: Float,
        streakWidth: Float,
        minLength: Float
    ) {
        let count = particleBuffer.capacity
        guard count > 0 else { return }

        let countAsFloat = Float(bitPattern: UInt32(count))
        var u = FWStreakUniforms(
            cameraCount:  SIMD4(cameraPosition, countAsFloat),
            streakParams: SIMD4(streakDuration, streakWidth, minLength, 0)
        )
        buildStreaksUniformBuf.contents()
            .copyMemory(from: &u, byteCount: MemoryLayout<FWStreakUniforms>.stride)

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Fireworks.buildStreaks"
        enc.setComputePipelineState(buildStreaksPipeline)
        enc.setBuffer(particleBuffer.buffer,    offset: 0, index: 0)
        enc.setBuffer(streakBuffer.buffer,      offset: 0, index: 1)
        enc.setBuffer(streakUVBuffer.buffer,    offset: 0, index: 2)
        enc.setBuffer(buildStreaksUniformBuf,   offset: 0, index: 3)
        let tg = min(count, buildStreaksPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    // ── Burst spawn ─────────────────────────────────────────────────────────

    /// Dispatch a spawn-burst kernel for the given burst parameters. Allocates
    /// a contiguous slot range from the ring cursor; if the range would wrap
    /// past the end, it wraps to 0 (overwriting the oldest live particles —
    /// the buffer is sized large enough that this only happens to long-dead
    /// or long-faded sparks).
    public func spawnBurst(_ params: FireworkBurstParams) {
        let count = max(0, params.particleCount)
        guard count > 0 else { return }
        let cap = particleBuffer.capacity
        // Wrap if the requested range straddles the end.
        if nextSlot + count > cap {
            nextSlot = 0
        }
        let firstIndex = nextSlot
        nextSlot += count
        if nextSlot >= cap { nextSlot = 0 }

        // Seeds advance per-burst. Mixed with a large prime so consecutive
        // bursts see well-separated streams.
        burstSeed &+= 0x9E3779B9

        let firstAsFloat = Float(bitPattern: UInt32(firstIndex))
        let countAsFloat = Float(bitPattern: UInt32(count))
        let seedAsFloat  = Float(bitPattern: burstSeed)
        let kindAsFloat  = Float(bitPattern: params.kind.rawValue)

        var u = FWSpawnUniforms(
            centerKind: SIMD4(params.center, kindAsFloat),
            baseColor:  SIMD4(params.baseColor, params.colorJitter),
            speedSizeLife: SIMD4(params.speed,
                                 params.speedJitter,
                                 params.size,
                                 params.lifespan),
            dragTurbGravLife: SIMD4(params.drag,
                                    params.turbulence,
                                    params.gravityMultiplier,
                                    params.lifeJitter),
            firstCountSeedTime: SIMD4(firstAsFloat,
                                      countAsFloat,
                                      seedAsFloat,
                                      0),
            ringAxis: SIMD4(simd_normalize(params.orientationAxis),
                            params.ringThickness),
            secondaryColor: SIMD4(params.secondaryColor,
                                  params.secondaryColorChance)
        )

        // Rotate to the next uniform-ring slot so a back-to-back spawn (compound
        // burst, finale barrage) queued before the GPU drains this dispatch
        // doesn't read this burst's params overwritten by the next one.
        let slot = spawnUniformCursor
        spawnUniformCursor = (spawnUniformCursor + 1) % Self.inFlightSpawns
        let uniformOffset = slot * spawnUniformStride
        spawnUniformBuf.contents().advanced(by: uniformOffset)
            .copyMemory(from: &u, byteCount: MemoryLayout<FWSpawnUniforms>.stride)

        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder()
        else { return }
        cb.label = "Fireworks.spawnBurst"
        enc.label = "Fireworks.spawnBurst"
        enc.setComputePipelineState(spawnPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(spawnUniformBuf,        offset: uniformOffset, index: 1)
        let tg = min(count, spawnPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
    }

    // ── Spark spawn (trails) ────────────────────────────────────────────────

    /// Spawn N sparks from N world positions in a single dispatch — the
    /// shell-rise trail equivalent of `spawnBurst`. Allocates a contiguous
    /// slot range, writes per-emit positions + seeds into the shared emit
    /// buffer, then dispatches `fwSpawnSparks` to populate the corresponding
    /// particle slots. Trail particles integrate, brighten, and streak via
    /// the same `fwIntegrate` + `fwBuildStreaks` passes the burst particles
    /// use — no separate renderer code.
    ///
    /// Caller is responsible for keeping `positions.count` within the spark
    /// emit buffer's capacity (default 512); excess emits are silently
    /// truncated. At typical trail-density settings this is comfortably above
    /// the realistic ceiling (e.g. 30 active shells × 4 emits/tick = 120).
    public func spawnSparks(at positions: [SIMD3<Float>], params: FireworkSparkParams) {
        let n = min(positions.count, sparkEmitRegionCapacity)
        guard n > 0 else { return }

        let particleCap = particleBuffer.capacity
        if nextSlot + n > particleCap {
            nextSlot = 0
        }
        let firstIndex = nextSlot
        nextSlot += n
        if nextSlot >= particleCap { nextSlot = 0 }

        burstSeed &+= 0x9E3779B9
        let globalSeed = burstSeed

        // Rotate to the next ring slot — same rationale as `spawnBurst`: a
        // second `spawnSparks` queued before the GPU drains this dispatch must
        // not overwrite this call's uniforms OR its emit positions. The emit
        // buffer is partitioned into one region per slot.
        let slot = spawnSparksCursor
        spawnSparksCursor = (spawnSparksCursor + 1) % Self.inFlightSpawns
        let uniformOffset = slot * spawnSparksStride
        let emitFirst = slot * sparkEmitRegionCapacity
        let emitByteOffset = emitFirst * MemoryLayout<FWSparkEmit>.stride

        // Fill this slot's emit region in shared memory. Per-emit seed is a quick
        // hash of (globalSeed, index, position.x bits) so two emits at the
        // same world position get well-separated random streams.
        let emitPtr = sparkEmitBuffer.buffer.contents()
            .advanced(by: emitByteOffset)
            .bindMemory(to: FWSparkEmit.self, capacity: n)
        for i in 0..<n {
            let pos = positions[i]
            let h = globalSeed &+ UInt32(i) &* 0x9E3779B9
            emitPtr[i] = FWSparkEmit(position: pos, seed: h)
        }

        let firstAsFloat = Float(bitPattern: UInt32(firstIndex))
        let countAsFloat = Float(bitPattern: UInt32(n))
        let seedAsFloat  = Float(bitPattern: globalSeed)

        var u = FWSparkUniforms(
            baseColorJitter: SIMD4(params.baseColor, params.colorJitter),
            speedLifeJitter: SIMD4(params.speed,
                                   params.speedJitter,
                                   params.lifespan,
                                   params.lifeJitter),
            dragTurbGrav:    SIMD4(params.drag,
                                   params.turbulence,
                                   params.gravityMultiplier,
                                   params.size),
            firstCountSeedTime: SIMD4(firstAsFloat,
                                      countAsFloat,
                                      seedAsFloat,
                                      0)
        )
        spawnSparksUniformBuf.contents().advanced(by: uniformOffset)
            .copyMemory(from: &u, byteCount: MemoryLayout<FWSparkUniforms>.stride)

        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder()
        else { return }
        cb.label = "Fireworks.spawnSparks"
        enc.label = "Fireworks.spawnSparks"
        enc.setComputePipelineState(spawnSparksPipeline)
        enc.setBuffer(particleBuffer.buffer,      offset: 0, index: 0)
        enc.setBuffer(spawnSparksUniformBuf,      offset: uniformOffset, index: 1)
        enc.setBuffer(sparkEmitBuffer.buffer,     offset: emitByteOffset, index: 2)
        let tg = min(n, spawnSparksPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
    }

    /// Kill every live particle: reset the ring cursor and overwrite the
    /// whole buffer with default-constructed (dead) records. Cheap — one
    /// 80-byte struct copy per slot. Use when the scene reloads.
    public func clearAll() {
        nextSlot = 0
        let dead = FireworkParticle()
        var seeded = Array(repeating: dead, count: particleBuffer.capacity)
        seeded.withUnsafeMutableBufferPointer { p in
            particleBuffer.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<FireworkParticle>.stride * particleBuffer.capacity
            )
        }
    }
}
