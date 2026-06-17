import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── FOAM SOLVER ──────────────────────────────────────────────────────────────
//
// Secondary-particle layer that rides on top of an MLSMPMSolver. Spawns
// "spray / foam / mist" droplets from energetic fluid particles, then
// integrates them ballistically (gravity + drag) until they expire or fall
// out of the simulation bounds.
//
// Two kernels, both per-frame (not per-substep — the foam doesn't care about
// CFL and integrating it 4× as often would just burn GPU for no visible win):
//
//   1. foamSpawn    — one thread per fluid particle. Hash-based RNG decides
//                     whether to emit a foam droplet this frame.
//   2. foamAdvect   — one thread per foam slot. Ballistic integration; kills
//                     anything that expires or escapes.
//
// Pairs with `FoamRenderer` — the renderer aliases the same particle buffer
// SceneKit reads each frame. Zero-copy, same contract as FluidParticleRenderer.
//
// SHAPE NOTES
// ───────────
// • Ring buffer. `nextSlot` is an atomic uint that wraps mod capacity. When
//   the user has the spawn rate cranked the oldest live particles get
//   overwritten — fine for foam, which is visual, not load-bearing.
// • Foam dynamics are decoupled from the fluid grid. After birth a foam
//   particle is just a small projectile. Coupling them back to the grid
//   velocity would look slightly nicer in calm regions but would also
//   prevent the splashes from actually escaping the fluid surface — which
//   is the whole point.

/// One secondary particle. 32 bytes; matches `FoamParticle` in Foam.metal.
///
///   positionLife : xyz = world position, w = remaining life (s, ≤0 = dead)
///   velocityAge  : xyz = velocity,       w = age (s)
public struct FoamParticle {
    public var positionLife: SIMD4<Float>
    public var velocityAge:  SIMD4<Float>

    public init(position: SIMD3<Float> = .zero,
                velocity: SIMD3<Float> = .zero,
                life: Float = 0,
                age: Float = 0) {
        self.positionLife = SIMD4(position, life)
        self.velocityAge  = SIMD4(velocity, age)
    }
}

/// Internal uniforms. Layout MUST match `FoamUniforms` in Foam.metal.
struct FoamUniforms {
    var counts:    SIMD4<UInt32>   // x = fluidCount, y = foamCapacity, z = randSeed
    var dtTime:    SIMD4<Float>    // x = dt, y = elapsed
    var gravity:   SIMD4<Float>    // xyz = gravity, w = drag (1/s)
    var bounds:    SIMD4<Float>    // xyz = boundsMin, w = lifeMax (s)
    var boundsMax: SIMD4<Float>    // xyz = boundsMax, w = parkY (dead-particle park height)
    var spawn:    SIMD4<Float>     // x = minSpeed, y = maxDensity, z = spawnProb, w = velJitter
}

@MainActor
public final class FoamSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "FoamSolver")

    // ── GPU plumbing ──────────────────────────────────────────────────────────

    public let device: MTLDevice
    private let spawnPipeline:  MTLComputePipelineState
    private let advectPipeline: MTLComputePipelineState

    // ── Storage ───────────────────────────────────────────────────────────────

    /// The foam-particle ring buffer. `FoamRenderer` reads positions from this
    /// buffer directly.
    public let particleBuffer: SimBuffer<FoamParticle>

    /// Atomic ring-buffer cursor. One uint; we let it wrap naturally and the
    /// kernel reduces mod capacity. Shared-storage so the kernel can write to
    /// it across multiple command buffers without a re-bind dance.
    private let nextSlotBuffer: MTLBuffer

    private let uniformBuffer: MTLBuffer

    /// Held weakly so the foam solver doesn't keep the fluid solver alive —
    /// the controller owns both and they share the same lifetime.
    public weak var fluidSolver: MLSMPMSolver?

    // ── Tunables ──────────────────────────────────────────────────────────────

    /// Maximum lifetime of a foam particle in seconds. The actual lifetime per
    /// particle is randomised in [0.4, 1.0] × this so they don't all expire
    /// at the same moment after a spawn burst.
    public var lifeMax: Float = 1.4

    /// Air drag coefficient (1/s). The advect kernel uses `vel *= exp(-drag·dt)`
    /// each step, so this is the *rate constant* — at drag = 2.0, velocity
    /// halves every ~0.35 s in the absence of gravity. Higher = stickier
    /// foam, lower = ballistic.
    public var drag: Float = 1.8

    /// Minimum particle speed (m/s) that can trigger a foam emission.
    /// Calm regions sit at < 0.5 m/s; splashes are 4–10 m/s.
    public var spawnMinSpeed: Float = 2.0

    /// Maximum particle *density estimate* that can trigger a foam emission.
    /// Particles deep inside the bulk register density ≈ restDensity (~8 by
    /// default). Surface and splash particles have fewer neighbours and read
    /// closer to 2–4. Tune below ~6 to restrict emission to the surface.
    public var spawnMaxDensity: Float = 5.0

    /// Base probability per fluid particle per frame of emitting a foam
    /// droplet (gated by the speed + density criteria above). Upward-moving
    /// particles get an additional boost in the kernel.
    public var spawnProb: Float = 0.015

    /// Magnitude of random velocity jitter (m/s) added to spawned foam so it
    /// doesn't track the parent fluid particle exactly. 0 = foam moves with
    /// the parent; ~0.5 reads as "droplet broke off and is fanning out."
    public var spawnVelJitter: Float = 0.4

    /// Magnitude of gravity along -Y (m/s²). Mirrors the fluid solver's
    /// gravity by default; can diverge if you want lazy / floaty foam.
    public var gravity: Float = 9.8

    /// Where to park dead particles so the renderer never sees them. The
    /// renderer always draws `capacity` points; dead ones live here, which
    /// the host sets to well below the simulation bounds.
    private var parkY: Float

    /// World-space domain. Defaults to the fluid solver's bounds; the advect
    /// kernel kills any foam that escapes here.
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>

    private var elapsed: Float = 0
    private var frameCounter: UInt32 = 0

    // ── Init ─────────────────────────────────────────────────────────────────

    /// Build a foam solver wired to a fluid simulation.
    ///
    /// - Parameters:
    ///   - engine: shared SimEngine (device + pipeline cache).
    ///   - fluidSolver: the MLS-MPM solver whose particles we sample to spawn
    ///                  foam. Held weakly.
    ///   - capacity: maximum number of simultaneously-live foam particles. The
    ///               ring buffer wraps when the user cranks the spawn rate
    ///               beyond what the lifetime can absorb.
    public init?(engine: SimEngine,
                 fluidSolver: MLSMPMSolver,
                 capacity: Int = 60_000) {
        let device = engine.device

        guard
            let spawnP  = engine.pipeline("foamSpawn"),
            let advectP = engine.pipeline("foamAdvect")
        else {
            Self.log.error("Foam pipeline lookup failed — check Foam.metal is in VisualizerRendering/Shaders/")
            return nil
        }

        guard
            let pBuf = SimBuffer<FoamParticle>(device: device,
                                               capacity: capacity,
                                               label: "Foam.particles"),
            let slotBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared),
            let uBuf = device.makeBuffer(length: MemoryLayout<FoamUniforms>.stride,
                                         options: .storageModeShared)
        else {
            Self.log.error("Foam buffer allocation failed (capacity = \(capacity))")
            return nil
        }
        slotBuf.label = "Foam.nextSlot"
        uBuf.label    = "Foam.uniforms"

        // The buffer count is the *renderable* count and matches capacity for
        // the ring buffer — every slot is potentially live every frame.
        pBuf.write(Array(repeating: FoamParticle(), count: capacity))

        // Zero the slot cursor.
        slotBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0

        self.device          = device
        self.spawnPipeline   = spawnP
        self.advectPipeline  = advectP
        self.particleBuffer  = pBuf
        self.nextSlotBuffer  = slotBuf
        self.uniformBuffer   = uBuf
        self.fluidSolver     = fluidSolver
        self.boundsMin       = fluidSolver.boundsMin
        self.boundsMax       = fluidSolver.boundsMax
        // Park dead particles well below the floor — far enough that any
        // accidental near-camera framing still hides them. The renderer
        // doesn't fade dead foam (it just keeps drawing every slot), so
        // visual hygiene depends on parking being deep enough that the
        // camera never sees the parking lot.
        self.parkY           = fluidSolver.boundsMin.y - 1000
    }

    // ── Step ─────────────────────────────────────────────────────────────────

    /// Encode one foam spawn + advect pass into the given command buffer.
    /// Must be encoded AFTER the fluid solver in the same command buffer so
    /// it reads up-to-date fluid particle positions / velocities.
    public func encode(to commandBuffer: MTLCommandBuffer, wallDt: Float) {
        guard let fluid = fluidSolver else { return }

        let fluidCount = fluid.particleBuffer.count
        let capacity   = particleBuffer.capacity
        elapsed += wallDt
        frameCounter &+= 1

        // 1. Write uniforms (shared memory; cheap struct write).
        let uPtr = uniformBuffer.contents().bindMemory(to: FoamUniforms.self, capacity: 1)
        uPtr.pointee = FoamUniforms(
            counts:    SIMD4(UInt32(fluidCount), UInt32(capacity), frameCounter, 0),
            dtTime:    SIMD4(wallDt, elapsed, 0, 0),
            gravity:   SIMD4(0, -gravity, 0, drag),
            bounds:    SIMD4(boundsMin, lifeMax),
            boundsMax: SIMD4(boundsMax, parkY),
            spawn:     SIMD4(spawnMinSpeed, spawnMaxDensity, spawnProb, spawnVelJitter)
        )

        // 2. Spawn pass — one thread per FLUID particle. Skipped if there are
        //    no fluid particles yet (e.g. first frame after a reseed before
        //    seedBox() has uploaded).
        if fluidCount > 0 {
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "Foam.spawn"
                enc.setComputePipelineState(spawnPipeline)
                enc.setBuffer(fluid.particleBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(particleBuffer.buffer,       offset: 0, index: 1)
                enc.setBuffer(nextSlotBuffer,              offset: 0, index: 2)
                enc.setBuffer(uniformBuffer,               offset: 0, index: 3)
                let w = min(fluidCount, spawnPipeline.maxTotalThreadsPerThreadgroup)
                enc.dispatchThreads(
                    MTLSize(width: fluidCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
                )
                enc.endEncoding()
            }
        }

        // 3. Advect pass — one thread per FOAM slot. Runs every frame so dead
        //    particles get parked even after the fluid solver is paused.
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "Foam.advect"
            enc.setComputePipelineState(advectPipeline)
            enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(uniformBuffer,         offset: 0, index: 1)
            let w = min(capacity, advectPipeline.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreads(
                MTLSize(width: capacity, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
            )
            enc.endEncoding()
        }
    }

    /// Clear all foam particles by parking them under the floor and resetting
    /// the ring cursor. Useful when the fluid is reseeded.
    public func clear() {
        let cap = particleBuffer.capacity
        let dead = FoamParticle(position: SIMD3<Float>(0, parkY, 0),
                                velocity: .zero, life: 0, age: 0)
        particleBuffer.write(Array(repeating: dead, count: cap))
        nextSlotBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
    }
}
