import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── MLS-MPM TYPES ────────────────────────────────────────────────────────────
//
// Shared between Swift and Metal — layouts MUST match the matching structs in
// MLSMPM.metal exactly. See the ALIGNMENT RULE block in PBDSolver.swift for the
// general rule (no bare float3 in shared structs).
//
// All padding here is explicit. `simd_float3x3` in Swift has stride 48 (three
// columns padded to 16 bytes each), matching Metal `float3x3` byte-for-byte.

/// One simulation particle. 144 bytes; matches `MLSParticle` in MLSMPM.metal.
///
///   positionMass : xyz = world position, w = mass
///   velocityJp   : xyz = velocity,       w = Jp (det of plastic def. gradient)
///   C            : APIC affine velocity matrix (3×3, 48 B)
///   F            : elastic deformation gradient (3×3, 48 B) — for fluid we
///                  reset to an isotropic matrix encoding only J each step
///   misc         : x = density estimate, y = age (s), z/w = pad
public struct MLSParticle {
    public var positionMass: SIMD4<Float>
    public var velocityJp:   SIMD4<Float>
    public var C:            simd_float3x3
    public var F:            simd_float3x3
    public var misc:         SIMD4<Float>

    public init(position: SIMD3<Float>,
                velocity: SIMD3<Float> = .zero,
                mass: Float = 1.0) {
        self.positionMass = SIMD4(position, mass)
        self.velocityJp   = SIMD4(velocity, 1.0)   // Jp starts at 1 (no compression)
        self.C            = simd_float3x3(0)
        self.F            = matrix_identity_float3x3
        self.misc         = SIMD4(0, 0, 0, 0)
    }
}

/// One MAC-grid node — 32 bytes. Written by the grid-update kernel, read by g2p.
public struct MLSGridNode {
    public var momentumMass: SIMD4<Float>  // xyz = momentum,  w = mass
    public var velocityPad:  SIMD4<Float>  // xyz = velocity, w = pad
}

/// Uniforms for the capsule push-out kernel. Internal — only the solver builds these.
/// Layout must match `MLSColliderUniforms` in MLSMPM.metal exactly.
struct MLSColliderUniforms {
    var particleCount:   UInt32
    var colliderCount:   UInt32
    var skinDepth:       Float
    var restitution:     Float
    var boost:           Float
    var tangentFriction: Float = 0
    var _pad1:           Float = 0
    var _pad2:           Float = 0
}

/// Layout must match `MLSGripUniforms` in MLSMPM.metal exactly. Drives the
/// kinematic-grip pass (bonded pull): two cap rigid displacements + velocities.
struct MLSGripUniforms {
    var particleCount: UInt32
    var enabled:       UInt32
    var _pad0:         Float = 0
    var _pad1:         Float = 0
    var dispA:         SIMD4<Float>
    var dispB:         SIMD4<Float>
    var velA:          SIMD4<Float>
    var velB:          SIMD4<Float>
}

/// Layout must match `MLSRecycleUniforms` in MLSMPM.metal exactly. Drives the
/// opt-in conveyor-recycle pass (endless upward column).
struct MLSRecycleUniforms {
    var particleCount: UInt32
    var frame:         UInt32
    var lipY:          Float
    var wrapHeight:    Float
    var driveSpeed:    Float
    var driveRelax:    Float
    var _pad0:         Float = 0
    var _pad1:         Float = 0
}

/// Per-frame uniforms uploaded to every kernel. Internal — callers don't build these.
struct MLSUniforms {
    var gridRes:    SIMD4<UInt32>   // xyz = gridX/Y/Z, w = particleCount
    var dxParams:   SIMD4<Float>    // x = dx, y = invDx, z = dt, w = time
    var matParams:  SIMD4<Float>    // x = bulkModulus, y = gamma, z = restDensity, w = viscosity
    var boundsMin:  SIMD4<Float>    // xyz = boundsMin, w = friction
    var boundsMax:  SIMD4<Float>    // xyz = boundsMax, w = pad
    var gravity:    SIMD4<Float>    // xyz = gravity,    w = pad
    // Elastoplastic "droopy sauce" params (ignored when materialMode == 0).
    var plasticA:   SIMD4<Float>    // x = mu0, y = lambda0, z = xi, w = thetaC
    var plasticB:   SIMD4<Float>    // x = thetaS, y = materialMode, zw = pad
}

// ── MLS-MPM SOLVER ───────────────────────────────────────────────────────────
//
// Moving-Least-Squares Material Point Method (Hu et al. 2018, "A Moving Least
// Squares Material Point Method with Displacement Discontinuity and Two-Way
// Rigid Body Coupling"). One particle buffer, one Eulerian grid, three kernels
// per step:
//
//   1.  mlsGridClear     — zero grid mass + momentum atomics
//   2.  mlsP2G           — scatter mass + APIC momentum + stress contribution
//                          to the grid via quadratic B-spline weights
//   3.  mlsGridUpdate    — normalise momentum to velocity, add gravity, clamp
//                          to box bounds, write a read-only GridNode array
//   4.  mlsG2P           — gather grid velocity back to particles, update C
//                          and F, advect position
//
// Same SimEngine convention as PBDSolver: device + pipeline cache come from
// the engine; one solver per scene; encode() takes a caller-owned command
// buffer so the scene can batch with other GPU work.
//
// For the MVP this is a weakly-compressible Neo-Hookean fluid with simple
// box-AABB boundaries. Secondary-particle spawn paths from the reference
// FluidSim.metal are deliberately not ported yet — they belong in a follow-up
// once the core dynamics are visually verified.

@MainActor
public final class MLSMPMSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "MLSMPMSolver")

    // ── GPU plumbing ──────────────────────────────────────────────────────────

    public let device: MTLDevice

    // Bin-and-gather P2G pipeline. The old single-shot scatter P2G was the
    // bottleneck at high particle counts (108 atomic_fetch_add ops per
    // particle); replaced with a 5-step pipeline that sorts particles into
    // per-cell bins and gathers in cell-major order. See the PIPELINE NOTE
    // block in MLSMPM.metal for the full breakdown.
    private let cellClearPipeline:    MTLComputePipelineState
    private let cellCountPipeline:    MTLComputePipelineState
    private let offsetsScanPipeline:  MTLComputePipelineState
    private let scatterPipeline:      MTLComputePipelineState
    private let p2gGatherPipeline:    MTLComputePipelineState
    private let gridUpdatePipeline:   MTLComputePipelineState
    private let g2pPipeline:          MTLComputePipelineState
    private let collideCapsulesPipeline: MTLComputePipelineState
    private let gripPinPipeline:      MTLComputePipelineState
    /// Optional — only present if the conveyor kernel is in the library. nil
    /// leaves `conveyorEnabled` inert (older libraries / safety).
    private let conveyorRecyclePipeline: MTLComputePipelineState?

    // ── Storage ───────────────────────────────────────────────────────────────

    /// Per-particle state. SceneKit reads `positionMass.xyz` from this buffer
    /// directly via FluidParticleRenderer (zero copy).
    public let particleBuffer: SimBuffer<MLSParticle>

    /// Atomic-float buffer of length `gridCellCount`. Holds accumulated mass per
    /// voxel during P2G; cleared each step.
    ///
    /// Exposed as the SCALAR DENSITY FIELD for downstream consumers — the
    /// MarchingCubesBridge reads this to extract an isosurface. Treated as
    /// `device const float*` on the read side; the atomic-vs-non-atomic
    /// distinction is purely about the writer.
    public let gridMassBuffer: MTLBuffer

    /// Float buffer of length `gridCellCount * 3` (interleaved xyz). Holds
    /// per-cell momentum after the gather P2G writes it; no atomics needed
    /// because each cell is owned by a single thread in the gather pass.
    private let gridMomBuffer: MTLBuffer

    /// `MLSGridNode` array, written by gridUpdate, read by g2p. No atomics —
    /// each thread owns its voxel during the gridUpdate pass.
    private let gridNodeBuffer: MTLBuffer

    /// Per-cell particle counts. Atomic-incremented during the count phase;
    /// re-zeroed during the offsets scan so the scatter phase can reuse it
    /// as a write-cursor.
    private let cellCountsBuffer: MTLBuffer

    /// Per-cell exclusive prefix-sum of `cellCountsBuffer`. Length
    /// `gridCellCount + 1`; the final entry holds the total particle count
    /// so the gather pass can clamp `end` to a valid index.
    private let cellOffsetsBuffer: MTLBuffer

    /// Particle indices sorted by centre cell. The gather pass iterates
    /// `sortedIndices[cellOffsets[c] ..< cellOffsets[c+1]]` to find every
    /// particle whose centre is bin `c`.
    private let sortedIndicesBuffer: MTLBuffer

    private let uniformBuffer: MTLBuffer

    /// Per-particle CAPTURED rest position for the kinematic-grip pass
    /// (xyz = rest pos at seed time, w = unused). Filled by
    /// `seedCylinderAlongX` in lockstep with the particle buffer; read by
    /// `mlsGripPin`. Shared storage so the seeder can write it CPU-side.
    private let gripRestBuffer: SimBuffer<SIMD4<Float>>
    private let gripUniformBuffer: MTLBuffer
    private let recycleUniformBuffer: MTLBuffer
    private var conveyorFrame: UInt32 = 0

    // ── Kinematic grip (bonded pull) ───────────────────────────────────────────
    /// When true, particles tagged in `misc.z` (1 = cap A, 2 = cap B) are
    /// snapped each substep to their captured rest position + the cap
    /// displacement below, and given the cap velocity — a moving boundary the
    /// free middle necks between. No-op when false (existing scenes untouched).
    public var gripEnabled: Bool = false
    /// Rigid displacement of each cap from the captured rest pose (world).
    public var gripDispA: SIMD3<Float> = .zero
    public var gripDispB: SIMD3<Float> = .zero
    /// Velocity of each cap (world) — drives the P2G pull on free neighbours.
    public var gripVelA: SIMD3<Float> = .zero
    public var gripVelB: SIMD3<Float> = .zero

    // ── Conveyor recycle (opt-in endless column) ───────────────────────────────
    /// When true, after each `encode()` any particle that has risen past
    /// `conveyorLipY` is teleported back into the [`conveyorSeedMin`,
    /// `conveyorSeedMax`] band at the bottom with fresh material state — a
    /// treadmill that keeps an upward-pushed column flowing forever at constant
    /// particle count. No-op when false (existing scenes untouched).
    public var conveyorEnabled: Bool = false
    /// Particles above this Y wrap down by `conveyorLipY − conveyorFloorY`.
    public var conveyorLipY: Float = 0
    public var conveyorFloorY: Float = 0
    /// Uniform upward conveyor speed (m/s) the whole column relaxes toward — the
    /// "invisible push". 0 disables the drive (pure body-force mode).
    public var conveyorDriveSpeed: Float = 0
    /// Per-frame relaxation toward `conveyorDriveSpeed` (0..1). ~0.15 = smooth.
    public var conveyorDriveRelax: Float = 0.15

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Number of cells along each axis of the MAC grid.
    public let gridResolution: SIMD3<UInt32>
    public let cellSize: Float
    public let boundsMin: SIMD3<Float>
    public let boundsMax: SIMD3<Float>

    /// Total cell count = gridResolution.x * .y * .z.
    public var gridCellCount: Int {
        Int(gridResolution.x) * Int(gridResolution.y) * Int(gridResolution.z)
    }

    // ── Tunables ──────────────────────────────────────────────────────────────

    /// Magnitude of gravity along `gravityDirection` (m/s²).
    public var gravity: Float = 9.8
    /// Unit direction of gravity, applied at this magnitude. Default is
    /// world-down `(0, -1, 0)`; callers that animate the simulation domain
    /// (e.g. the FluidTest auto-rotating-cube mode) rotate this vector to
    /// match where "down" should be in the sim's axis-aligned frame, so
    /// the fluid sloshes toward the visual low side without moving the
    /// grid bounds.
    public var gravityDirection: SIMD3<Float> = SIMD3(0, -1, 0)
    /// Weakly-compressible fluid bulk modulus (stiffness).
    ///
    /// Real water sits around k = 2.2×10⁹ Pa, but using that at our spatial
    /// + temporal scales would demand CFL substeps in the microseconds. The
    /// standard MPM trick is to detune k by 6+ orders of magnitude so sound
    /// speed sqrt(k/ρ) is a few m/s, keeping CFL inside an animation-grade
    /// dt. k = 4 with γ = 3 reads as "wobbly water" at 4 substeps × 1/60 s.
    public var bulkModulus: Float = 4.0
    /// Pressure exponent in p = k((ρ/ρ₀)^γ − 1). γ = 7 is the real-water Tait
    /// exponent; once compressed even a little it produces pressures large
    /// enough to launch particles to absurd speeds in one substep. γ = 3 is
    /// the "play water" value that stays well-behaved at our parameters.
    public var gamma: Float = 3.0
    /// Rest density, in mass-per-cell-volume units.
    ///
    /// MLS-MPM's pressure EOS is ρ-relative — if `restDensity` is set lower
    /// than the actual density produced by the seed, the EOS will think the
    /// fluid is *compressed* at rest and push particles apart on frame 1
    /// before anything else happens. `seedBox()` recomputes this automatically
    /// from particle mass × particles-per-cell-volume; callers seeding by
    /// other means should set it explicitly afterwards.
    public var restDensity: Float = 8.0
    /// Dynamic viscosity μ. A little goes a long way — the bulk-modulus
    /// detune already produces a wobbly fluid; viscosity is mainly here to
    /// dissipate the kinetic energy of the initial drop.
    public var viscosity: Float = 0.2
    /// Coulomb friction coefficient at the box walls.
    public var boundaryFriction: Float = 0.3

    /// Per-substep velocity damping factor for slow-moving fluid. 1.0 =
    /// no damping (fluid keeps micro-jittering from EOS pressure, causing
    /// caustic flicker even when the body is "settled"). 0.95 = aggressive
    /// damping that brings residual motion to zero in ~1 s. The kernel
    /// smoothstep-blends damping out for fast cells so real splashes
    /// preserve momentum — only slow cells get damped.
    public var settleDamping: Float = 0.92
    /// Substeps per visual frame. MLS-MPM is conditionally stable; 4 substeps
    /// at dt = 1/60 ≈ 4.2 ms per substep is the working compromise on Apple
    /// Silicon for the default (bulkModulus 50, dx 0.1) tuning.
    public var substepsPerFrame: Int = 4

    // ── Elastoplastic "droopy sauce" material (issue #44) ─────────────────────
    //
    // A reusable mode that turns the solver from a weakly-compressible FLUID into
    // a soft ELASTOPLASTIC solid (the Stomakhin 2013 snow model): firm enough to
    // hold a shape and stretch into strands, soft enough to yield and droop under
    // gravity. Cheese, chocolate sauce, mustard, etc. are `SaucePreset`s.
    // Defaults to `.fluid`, so every existing fluid scene is byte-for-byte
    // unchanged — the kernels branch on a flag that is 0 unless `material` is set.

    public enum Material { case fluid, elastoplastic }

    /// Material model. `.fluid` (default) = the existing weakly-compressible path.
    public var material: Material = .fluid

    /// Young's modulus E. NOT in physical Pa — the affine-momentum formula
    /// implicitly pre-scales stress by 1/dx³ via `(dt/rho0)·stress·D_inv`
    /// where rho0 is in cell-mass units (not kg/m³). Realistic-Pa values
    /// (10³–10⁵) overshoot CFL by orders of magnitude here, exploding the
    /// sim within 1–2 substeps. Values comparable to FluidTest's
    /// `bulkModulus = 22.8` (i.e. tens, not thousands) are the stable
    /// range. See the `SaucePreset` doc-comment for the working numbers.
    public var youngsModulus: Float = 30
    /// Poisson ratio ν (incompressibility). ~0.3 for a soft food.
    public var poissonRatio: Float = 0.30
    /// Snow hardening ξ — worked material near the yield surface stiffens/softens.
    public var hardening: Float = 6.0
    /// Critical compression θc — yields (compacts/flows) once compressed past this.
    public var thetaCompress: Float = 0.025
    /// Critical stretch θs — the "stringiness" lever: how far it stretches
    /// elastically before the strand yields. Low = smooth queso, high = mozzarella.
    public var thetaStretch: Float = 0.0075

    /// Lamé μ₀ from (E, ν). Used only in `.elastoplastic` mode.
    var lameMu: Float { youngsModulus / (2 * (1 + poissonRatio)) }
    /// Lamé λ₀ from (E, ν).
    var lameLambda: Float {
        youngsModulus * poissonRatio / ((1 + poissonRatio) * (1 - 2 * poissonRatio))
    }

    /// A full droopy-sauce recipe. Reusable across scenes — CheeseTest
    /// runs the cheese through the `.fluid` material path and reads
    /// only `tint`, `transparency`, and the three `recipe*` defaults;
    /// GuacMash still runs `.elastoplastic` and reads the snow-MPM
    /// block (`youngsModulus` … `viscosity`) via `solver.apply(_:)`.
    /// The two field groups are documented separately so it's clear
    /// which scenes consume which.
    public struct SaucePreset: Sendable {
        public var name: String
        /// Snow-MPM elastoplastic stress params. Consumed by
        /// `MLSMPMSolver.apply(_:)`, which is currently only called
        /// by `GuacMashController`. CheeseTest's fluid path ignores
        /// these — leave them in place to keep GuacMash working.
        public var youngsModulus, poissonRatio, hardening: Float
        public var thetaCompress, thetaStretch: Float
        public var viscosity: Float
        /// Surface tint (linear RGB). Read by both CheeseTest + GuacMash.
        public var tint: SIMD3<Float>
        /// 0..1 alpha. 1 = fully opaque (cheese, mustard, chocolate);
        /// values below 1 (honey ≈ 0.18) make the MC surface
        /// translucent. Read by CheeseTest via the bridge's
        /// `setSauceAlpha`; ignored by scenes that don't care.
        public var transparency: Float
        /// Recipe-slider defaults so the CheeseTest preset picker can
        /// pre-fill Droopyness / Stretchyness / Stickyness sliders to
        /// the behaviour expected for this sauce. Each scene that
        /// uses the recipe pattern reads these on preset change.
        public var recipeDroopyness: Float
        public var recipeStretchyness: Float
        public var recipeStickyness: Float

        public init(name: String, youngsModulus: Float, poissonRatio: Float,
                    hardening: Float, thetaCompress: Float, thetaStretch: Float,
                    viscosity: Float, tint: SIMD3<Float>,
                    transparency: Float = 1.0,
                    recipeDroopyness: Float = 0.65,
                    recipeStretchyness: Float = 0.75,
                    recipeStickyness: Float = 0.85) {
            self.name = name; self.youngsModulus = youngsModulus
            self.poissonRatio = poissonRatio; self.hardening = hardening
            self.thetaCompress = thetaCompress; self.thetaStretch = thetaStretch
            self.viscosity = viscosity; self.tint = tint
            self.transparency = transparency
            self.recipeDroopyness = recipeDroopyness
            self.recipeStretchyness = recipeStretchyness
            self.recipeStickyness = recipeStickyness
        }

        // Stability note: the affine-momentum contribution per substep is
        //   (dt · stress · D_inv) / restDensity  where D_inv = 4/dx² and
        //   restDensity is in CELL-MASS units (=particleMass·ppc=8 here),
        // NOT kg/m³. That means stress is implicitly pre-scaled by 1/dx³
        // relative to a physical-Pa formulation, so realistic-Pa values for
        // E (Pa range 10³–10⁵) overshoot CFL by orders of magnitude — the
        // sim explodes (vel saturates at the 50 m/s safety cap and the
        // box fills wall-to-wall in 1–2 substeps).
        //
        // Comparable scale to the working fluid path: FluidTest's
        // `bulkModulus = 22.8` is at the upper edge of what's stable at
        // dt~5ms / dx~0.1m. Snow-stress for E=30 produces a comparable
        // 2μ·strain magnitude (~3) at 1 % strain, with the λ·J·(J-1) term
        // adding another ~0.6 — same order as fluid stress, so the same
        // substep count is fine. The numbers below are tuned against this
        // scale; the firmness ordering (cheese > chocolate > mustard) is
        // preserved.  Avocado-paste is the firmest so the pestle has
        // something to grind against.
        // The cheese preset is the lab's hero recipe — tuned for the
        // "drop / slide / gloop / droop" behaviour. With the (now wired-
        // up) viscous term in `snow_stress`, viscosity actually
        // resists strain RATE, letting the cheese flow over the chip
        // edge instead of holding a rigid shape past yield. E lowered
        // significantly (was 160 → 35) so the cheese deforms under
        // gravity in a visible second or two; yield thresholds lowered
        // so the blob actually flows past elasticity into plastic
        // strain on contact.
        public static let cheese = SaucePreset(
            name: "Cheese", youngsModulus: 35, poissonRatio: 0.30, hardening: 6,
            thetaCompress: 0.012, thetaStretch: 0.004, viscosity: 1.5,
            tint: SIMD3(0.96, 0.74, 0.26),
            recipeDroopyness: 0.65, recipeStretchyness: 0.75, recipeStickyness: 0.85)
        public static let chocolate = SaucePreset(
            name: "Chocolate", youngsModulus: 20, poissonRatio: 0.32, hardening: 4,
            thetaCompress: 0.010, thetaStretch: 0.002, viscosity: 2.5,
            tint: SIMD3(0.30, 0.16, 0.07),
            recipeDroopyness: 0.55, recipeStretchyness: 0.70, recipeStickyness: 0.75)
        public static let mustard = SaucePreset(
            name: "Mustard", youngsModulus: 10, poissonRatio: 0.30, hardening: 3,
            thetaCompress: 0.008, thetaStretch: 0.0015, viscosity: 3.5,
            tint: SIMD3(0.85, 0.69, 0.10),
            recipeDroopyness: 0.80, recipeStretchyness: 0.40, recipeStickyness: 0.55)
        /// Honey — thick, slow, very stretchy, sticks to every surface.
        /// Translucent amber so light passes through the sauce. Recipe
        /// defaults are at one extreme of the droopyness scale (slow
        /// flow) and the upper end of stretchyness + stickyness.
        public static let honey = SaucePreset(
            name: "Honey", youngsModulus: 4, poissonRatio: 0.30, hardening: 2,
            thetaCompress: 0.020, thetaStretch: 0.040, viscosity: 5.0,
            // Warm amber: rich orange-gold (R 0.95, G 0.55, B 0.10) —
            // reads as honey under the warm IBL while still surviving
            // the spec wash.
            tint: SIMD3(0.95, 0.55, 0.10),
            // Genuinely translucent. 0.18 reads as actual see-through
            // honey — the chip's golden brown blends through the honey
            // and the floor's tone bleeds in too. 0.55 / 0.30 were too
            // opaque to read as honey.
            transparency: 0.18,
            recipeDroopyness: 0.20, recipeStretchyness: 0.95, recipeStickyness: 0.95)
        /// Firm avocado FLESH for the mash: stiffer (holds a chunk shape) with a
        /// higher compaction yield so it resists until the pestle presses hard,
        /// then crushes/breaks up and flows into paste. Low stretch (it tears,
        /// doesn't string like cheese). Tint = avocado green.
        public static let avocado = SaucePreset(
            name: "Avocado (mash)", youngsModulus: 200, poissonRatio: 0.30, hardening: 9,
            thetaCompress: 0.040, thetaStretch: 0.006, viscosity: 0.9,
            // More saturated yellow-green than a literal photo-sampled
            // avocado (which is ~(0.62, 0.72, 0.30)) — the warm IBL +
            // PBR spec wash desaturated values toward bright yellow,
            // which read as "cheese" not "guac." Darker, more vibrant
            // values like (0.40, 0.60, 0.15) survive the lighting and
            // read unambiguously green.
            tint: SIMD3(0.40, 0.60, 0.15))

        public static let all: [SaucePreset] = [.cheese, .chocolate, .mustard, .honey, .avocado]
    }

    /// Load a sauce recipe and switch into elastoplastic mode.
    public func apply(_ p: SaucePreset) {
        material = .elastoplastic
        youngsModulus = p.youngsModulus
        poissonRatio = p.poissonRatio
        hardening = p.hardening
        thetaCompress = p.thetaCompress
        thetaStretch = p.thetaStretch
        viscosity = p.viscosity
    }

    /// External rigid-body colliders that push fluid particles around. Same
    /// `PBDCollider` layout the PBD solvers consume, so a host that already
    /// builds a per-substep collider buffer (e.g. the Hotdog Drop+ scene
    /// feeding spine capsules through `PBDTubeRenderer.dispatchBuildColliders`)
    /// can hand the same buffer here for one-way fluid coupling.
    /// `nil` (or `count == 0`) skips the collide pass.
    public struct ColliderBinding {
        public let buffer: MTLBuffer
        public let count:  Int
        public init(buffer: MTLBuffer, count: Int) {
            self.buffer = buffer; self.count = count
        }
    }
    public var colliders: ColliderBinding?

    /// Skin offset added to each capsule's radius when pushing fluid out.
    /// A tiny positive value keeps particles from sitting exactly on the
    /// surface (where round-off can sneak them back inside next substep).
    public var colliderSkin: Float = 0.005

    /// Bounce coefficient applied to the inward velocity component when
    /// a fluid particle hits a collider. 0 = no bounce (water sticks and
    /// slides); 1 = full mirror reflection. Mid values produce visible
    /// splash/spray seed velocity that the foam solver can pick up on.
    public var colliderRestitution: Float = 0.5

    /// Outward velocity kick added to a particle that's been pushed out of
    /// a collider, proportional to the depth it had penetrated. Empirical
    /// scale, default 2.5; needed for FLUID scenes so a static-immersed
    /// object continuously sloshes the surface around it (without this,
    /// a settled collider produces zero motion — fluid just deforms once
    /// and freezes). For ELASTOPLASTIC scenes (cheese on a ledge, paste
    /// in a bowl) this constant kick prevents the sauce from ever resting
    /// — particles scatter off the ledge indefinitely. Set to 0 in those
    /// scenes so the collider acts as a true static surface.
    public var colliderDisplacementBoost: Float = 2.5

    /// Per-substep fraction of TANGENTIAL velocity removed at capsule
    /// contact. 0 = frictionless slide (water — default). 1 = particles
    /// stick to the surface and stop sliding entirely. Models the
    /// adhesion / surface-tension force that bulk-fluid MPM doesn't
    /// have natively. Bulk viscosity in fluid_stress damps with
    /// VELOCITY GRADIENT, so a small remaining blob has little gradient
    /// and accelerates freely off a static surface; this tangent
    /// friction damps with VELOCITY directly, so the per-volume drag
    /// stays the same regardless of blob size — exactly what's needed
    /// for "honey clings to the chip as it drains."
    public var colliderTangentFriction: Float = 0.0

    private var elapsed: Float = 0

    // ── Init ─────────────────────────────────────────────────────────────────

    /// Build a solver for a box-bounded fluid simulation.
    ///
    /// - Parameters:
    ///   - engine: shared SimEngine (device + pipeline cache).
    ///   - boundsMin/boundsMax: world-space AABB of the simulation domain.
    ///   - cellSize: MAC-grid cell size in metres. Smaller = more detail, much
    ///     more cost (memory cubic, kernel work cubic).
    ///   - maxParticles: capacity of the particle buffer.
    public init?(engine: SimEngine,
                 boundsMin: SIMD3<Float>,
                 boundsMax: SIMD3<Float>,
                 cellSize: Float,
                 maxParticles: Int) {
        let device = engine.device

        guard
            let cellClear   = engine.pipeline("mlsCellClear"),
            let cellCount   = engine.pipeline("mlsCellCount"),
            let offsetsScan = engine.pipeline("mlsCellOffsetsScan"),
            let scatter     = engine.pipeline("mlsScatterParticles"),
            let p2gGather   = engine.pipeline("mlsP2GGather"),
            let gridUpdate  = engine.pipeline("mlsGridUpdate"),
            let g2p         = engine.pipeline("mlsG2P"),
            let collideCap  = engine.pipeline("mlsCollideCapsules"),
            let gripPin     = engine.pipeline("mlsGripPin")
        else {
            Self.log.error("MLS-MPM pipeline lookup failed — check MLSMPM.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        // Optional — opt-in conveyor recycle. Absent → conveyorEnabled stays inert.
        let conveyorRecycle = engine.pipeline("mlsConveyorRecycle")

        // Grid resolution = ceil(extents / cellSize) + 1 ring of padding so
        // particles can never touch the absolute edge cell (where the quadratic
        // B-spline stencil would read out of bounds).
        let extents = boundsMax - boundsMin
        let resX = UInt32((extents.x / cellSize).rounded(.up)) + 2
        let resY = UInt32((extents.y / cellSize).rounded(.up)) + 2
        let resZ = UInt32((extents.z / cellSize).rounded(.up)) + 2
        let res = SIMD3<UInt32>(resX, resY, resZ)
        let cells = Int(resX) * Int(resY) * Int(resZ)

        guard
            let pBuf = SimBuffer<MLSParticle>(device: device,
                                              capacity: maxParticles,
                                              label: "MLS.particles"),
            let gMass = device.makeBuffer(length: MemoryLayout<Float>.stride * cells,
                                          options: .storageModePrivate),
            let gMom  = device.makeBuffer(length: MemoryLayout<Float>.stride * cells * 3,
                                          options: .storageModePrivate),
            let gNode = device.makeBuffer(length: MemoryLayout<MLSGridNode>.stride * cells,
                                          options: .storageModePrivate),
            let cCount = device.makeBuffer(length: MemoryLayout<UInt32>.stride * cells,
                                           options: .storageModePrivate),
            let cOff   = device.makeBuffer(length: MemoryLayout<UInt32>.stride * (cells + 1),
                                           options: .storageModePrivate),
            let sIdx   = device.makeBuffer(length: MemoryLayout<UInt32>.stride * max(maxParticles, 1),
                                           options: .storageModePrivate),
            let uBuf  = device.makeBuffer(length: MemoryLayout<MLSUniforms>.stride,
                                          options: .storageModeShared),
            let gripRest = SimBuffer<SIMD4<Float>>(device: device,
                                                   capacity: maxParticles,
                                                   label: "MLS.gripRest"),
            let gripU = device.makeBuffer(length: MemoryLayout<MLSGripUniforms>.stride,
                                          options: .storageModeShared),
            let recycleU = device.makeBuffer(length: MemoryLayout<MLSRecycleUniforms>.stride,
                                             options: .storageModeShared)
        else {
            Self.log.error("MLS-MPM buffer allocation failed (cells = \(cells))")
            return nil
        }
        gripU.label = "MLS.gripUniforms"
        recycleU.label = "MLS.recycleUniforms"
        gMass.label  = "MLS.gridMass"
        gMom.label   = "MLS.gridMom"
        gNode.label  = "MLS.gridNode"
        cCount.label = "MLS.cellCounts"
        cOff.label   = "MLS.cellOffsets"
        sIdx.label   = "MLS.sortedIndices"
        uBuf.label   = "MLS.uniforms"

        self.device                = device
        self.cellClearPipeline     = cellClear
        self.cellCountPipeline     = cellCount
        self.offsetsScanPipeline   = offsetsScan
        self.scatterPipeline       = scatter
        self.p2gGatherPipeline     = p2gGather
        self.gridUpdatePipeline    = gridUpdate
        self.g2pPipeline           = g2p
        self.collideCapsulesPipeline = collideCap
        self.gripPinPipeline       = gripPin
        self.conveyorRecyclePipeline = conveyorRecycle
        self.gripRestBuffer        = gripRest
        self.gripUniformBuffer     = gripU
        self.recycleUniformBuffer  = recycleU
        self.particleBuffer        = pBuf
        self.gridMassBuffer        = gMass
        self.gridMomBuffer         = gMom
        self.gridNodeBuffer        = gNode
        self.cellCountsBuffer      = cCount
        self.cellOffsetsBuffer     = cOff
        self.sortedIndicesBuffer   = sIdx
        self.uniformBuffer         = uBuf
        self.gridResolution        = res
        self.cellSize              = cellSize
        self.boundsMin             = boundsMin
        self.boundsMax             = boundsMax
    }

    // ── Particle seeding ─────────────────────────────────────────────────────

    /// Fill an axis-aligned box with particles on a regular grid plus a small
    /// random jitter. Useful for the canonical "cube of water dropped into a
    /// box" test.
    public func seedBox(min: SIMD3<Float>,
                        max: SIMD3<Float>,
                        particlesPerCellAxis: Int = 2,
                        jitter: Float = 0.3,
                        rngSeed: UInt64 = 0x12345678) {
        var rng = SplitMix64(state: rngSeed)
        var particles: [MLSParticle] = []
        let extents = max - min
        let spacing = cellSize / Float(particlesPerCellAxis)
        let nx = Int((extents.x / spacing).rounded(.down))
        let ny = Int((extents.y / spacing).rounded(.down))
        let nz = Int((extents.z / spacing).rounded(.down))
        particles.reserveCapacity(nx * ny * nz)
        let cap = particleBuffer.capacity
        for ix in 0..<nx {
            for iy in 0..<ny {
                for iz in 0..<nz {
                    if particles.count >= cap { break }
                    let basePos = min + spacing * SIMD3<Float>(
                        Float(ix) + 0.5,
                        Float(iy) + 0.5,
                        Float(iz) + 0.5
                    )
                    let jx = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let jy = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let jz = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let pos = basePos + SIMD3<Float>(jx, jy, jz)
                    particles.append(MLSParticle(position: pos))
                }
            }
        }
        particleBuffer.write(particles)

        // Derive rest density from the actual seed so the EOS reads "at rest"
        // == zero pressure on frame 1. Otherwise the seed itself looks
        // compressed (or expanded) and the very first substep tries to fix it
        // — which at our timestep and detuned k still produces visible
        // explosions before any real physics has happened.
        let particleMass: Float = 1.0   // matches MLSParticle.init default
        let ppc = Float(particlesPerCellAxis * particlesPerCellAxis * particlesPerCellAxis)
        restDensity = particleMass * ppc
    }

    /// Fill an upright cylinder (axis +Y) with particles on the same regular
    /// grid + jitter as `seedBox`, rejecting any sample whose XZ distance from
    /// `center` exceeds `radius`. Use for a circular pool footprint. Rest
    /// density is derived from the seed exactly as in `seedBox` so frame 1 reads
    /// as "at rest."
    public func seedCylinder(center: SIMD2<Float>,
                             radius: Float,
                             bottomY: Float,
                             topY: Float,
                             particlesPerCellAxis: Int = 2,
                             jitter: Float = 0.3,
                             rngSeed: UInt64 = 0x12345678) {
        var rng = SplitMix64(state: rngSeed)
        var particles: [MLSParticle] = []
        let spacing = cellSize / Float(particlesPerCellAxis)
        let minP = SIMD3<Float>(center.x - radius, bottomY, center.y - radius)
        let extents = SIMD3<Float>(radius * 2, topY - bottomY, radius * 2)
        let nx = Int((extents.x / spacing).rounded(.down))
        let ny = Int((extents.y / spacing).rounded(.down))
        let nz = Int((extents.z / spacing).rounded(.down))
        particles.reserveCapacity(nx * ny * nz)
        let cap = particleBuffer.capacity
        let r2 = radius * radius
        outer: for ix in 0..<nx {
            for iy in 0..<ny {
                for iz in 0..<nz {
                    if particles.count >= cap { break outer }
                    let basePos = minP + spacing * SIMD3<Float>(
                        Float(ix) + 0.5,
                        Float(iy) + 0.5,
                        Float(iz) + 0.5
                    )
                    let dx0 = basePos.x - center.x
                    let dz0 = basePos.z - center.y
                    if dx0 * dx0 + dz0 * dz0 > r2 { continue }
                    let jx = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let jy = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let jz = (rng.nextFloat01() - 0.5) * jitter * spacing
                    particles.append(MLSParticle(position: basePos + SIMD3<Float>(jx, jy, jz)))
                }
            }
        }
        particleBuffer.write(particles)

        let particleMass: Float = 1.0
        let ppc = Float(particlesPerCellAxis * particlesPerCellAxis * particlesPerCellAxis)
        restDensity = particleMass * ppc
    }

    /// Fill a horizontal cylinder (axis +X) with particles — the `seedCylinder`
    /// sibling for assets whose long axis lies along X (the fried-mozzarella
    /// stick's cheese core). Rejects samples whose YZ distance from
    /// `center` (y, z) exceeds `radius`. Optionally tags two end bands with a
    /// pin flag in `misc.z` (1 = left/−X band, 2 = right/+X band) so the
    /// kinematic-grip pass can carry them with the crust halves; pass
    /// `gripBandFraction == 0` to leave every particle free.
    public func seedCylinderAlongX(center: SIMD2<Float>,   // (y, z)
                                   radius: Float,
                                   leftX: Float,
                                   rightX: Float,
                                   particlesPerCellAxis: Int = 2,
                                   jitter: Float = 0.3,
                                   gripBandFraction: Float = 0.0,
                                   rngSeed: UInt64 = 0x12345678) {
        var rng = SplitMix64(state: rngSeed)
        var particles: [MLSParticle] = []
        var rest: [SIMD4<Float>] = []
        let spacing = cellSize / Float(particlesPerCellAxis)
        let minP = SIMD3<Float>(leftX, center.x - radius, center.y - radius)
        let extents = SIMD3<Float>(rightX - leftX, radius * 2, radius * 2)
        let nx = Int((extents.x / spacing).rounded(.down))
        let ny = Int((extents.y / spacing).rounded(.down))
        let nz = Int((extents.z / spacing).rounded(.down))
        particles.reserveCapacity(nx * ny * nz)
        let cap = particleBuffer.capacity
        let r2 = radius * radius
        let band = max(0, gripBandFraction) * (rightX - leftX)
        let leftBandEnd  = leftX + band
        let rightBandStart = rightX - band
        outer: for ix in 0..<nx {
            for iy in 0..<ny {
                for iz in 0..<nz {
                    if particles.count >= cap { break outer }
                    let basePos = minP + spacing * SIMD3<Float>(
                        Float(ix) + 0.5, Float(iy) + 0.5, Float(iz) + 0.5
                    )
                    let dy0 = basePos.y - center.x
                    let dz0 = basePos.z - center.y
                    if dy0 * dy0 + dz0 * dz0 > r2 { continue }
                    let jx = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let jy = (rng.nextFloat01() - 0.5) * jitter * spacing
                    let jz = (rng.nextFloat01() - 0.5) * jitter * spacing
                    var p = MLSParticle(position: basePos + SIMD3<Float>(jx, jy, jz))
                    if band > 0 {
                        if p.positionMass.x <= leftBandEnd       { p.misc.z = 1 }
                        else if p.positionMass.x >= rightBandStart { p.misc.z = 2 }
                    }
                    particles.append(p)
                    // Capture the rest position for the grip pass (used only
                    // for pinned particles; harmless for free ones).
                    rest.append(SIMD4<Float>(p.positionMass.x, p.positionMass.y,
                                             p.positionMass.z, 0))
                }
            }
        }
        particleBuffer.write(particles)
        gripRestBuffer.write(rest)

        let particleMass: Float = 1.0
        let ppc = Float(particlesPerCellAxis * particlesPerCellAxis * particlesPerCellAxis)
        restDensity = particleMass * ppc
    }

    /// Reset to an empty particle buffer (count = 0). Doesn't free memory.
    public func clear() {
        particleBuffer.write([])
    }

    // ── Step ─────────────────────────────────────────────────────────────────

    /// Encode `substepsPerFrame` MLS-MPM steps into the given command buffer.
    /// `wallDt` is the wall-clock interval since the last call; the solver
    /// splits it into `substepsPerFrame` equal substeps internally.
    public func encode(to commandBuffer: MTLCommandBuffer, wallDt: Float) {
        guard particleBuffer.count > 0 else { return }
        let substeps = max(1, substepsPerFrame)
        let dt = wallDt / Float(substeps)
        for _ in 0..<substeps {
            encodeOneStep(into: commandBuffer, dt: dt)
            elapsed += dt
        }
        // Conveyor recycle runs ONCE per frame (not per substep): teleport any
        // particle that cleared the lip back to the bottom band. No-op unless enabled.
        encodeConveyorRecycle(into: commandBuffer)
    }

    private func encodeConveyorRecycle(into cb: MTLCommandBuffer) {
        guard conveyorEnabled,
              let pipeline = conveyorRecyclePipeline,
              particleBuffer.count > 0,
              let enc = cb.makeComputeCommandEncoder()
        else { return }
        enc.label = "MLS.conveyorRecycle"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        conveyorFrame &+= 1
        var u = MLSRecycleUniforms(
            particleCount: UInt32(particleBuffer.count),
            frame: conveyorFrame,
            lipY: conveyorLipY,
            wrapHeight: max(0.01, conveyorLipY - conveyorFloorY),
            driveSpeed: conveyorDriveSpeed,
            driveRelax: max(0, min(1, conveyorDriveRelax)))
        enc.setBytes(&u, length: MemoryLayout<MLSRecycleUniforms>.stride, index: 1)
        let count = particleBuffer.count
        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeOneStep(into cb: MTLCommandBuffer, dt: Float) {
        // 1. Write uniforms (shared memory; cheap struct write).
        let uPtr = uniformBuffer.contents().bindMemory(to: MLSUniforms.self, capacity: 1)
        uPtr.pointee = MLSUniforms(
            gridRes:   SIMD4(gridResolution, UInt32(particleBuffer.count)),
            dxParams:  SIMD4(cellSize, 1.0 / cellSize, dt, elapsed),
            matParams: SIMD4(bulkModulus, gamma, restDensity, viscosity),
            boundsMin: SIMD4(boundsMin, boundaryFriction),
            boundsMax: SIMD4(boundsMax, max(0, min(1, settleDamping))),
            gravity:   SIMD4(gravityDirection.x * gravity,
                             gravityDirection.y * gravity,
                             gravityDirection.z * gravity,
                             0),
            plasticA:  SIMD4(lameMu, lameLambda, hardening, thetaCompress),
            plasticB:  SIMD4(thetaStretch,
                             material == .elastoplastic ? 1 : 0,
                             0, 0)
        )

        // ── Binning pipeline (replaces the old scatter P2G) ──────────────
        // Steps 2-5 build the per-cell particle bins so step 6 can run a
        // gather-mode P2G without atomics. Steps 7-8 are unchanged.

        // 2. Zero cellCounts.
        encodeCellPass(into: cb,
                       pipeline: cellClearPipeline,
                       label: "MLS.cellClear",
                       buffers: [(cellCountsBuffer, 0)])

        // 3. Count particles per centre cell.
        encodeParticlePass(into: cb,
                           pipeline: cellCountPipeline,
                           label: "MLS.cellCount",
                           extraBuffers: [(cellCountsBuffer, 1)],
                           uniformIndex: 2)

        // 4. Exclusive prefix sum → cellOffsets. Re-zeros cellCounts as a
        //    side effect so the scatter pass can reuse it as a cursor.
        encodeScanPass(into: cb)

        // 5. Scatter particle indices into per-cell bins.
        encodeParticlePass(into: cb,
                           pipeline: scatterPipeline,
                           label: "MLS.scatter",
                           extraBuffers: [
                               (cellCountsBuffer,    1),
                               (cellOffsetsBuffer,   2),
                               (sortedIndicesBuffer, 3),
                           ],
                           uniformIndex: 4)

        // 6. Gather-mode P2G — one thread per grid cell, walks 27 neighbour
        //    bins, writes mass + momentum directly (no atomics).
        encodeGridUpdate3D(into: cb,
                           pipeline: p2gGatherPipeline,
                           label: "MLS.p2gGather",
                           buffers: [
                               (particleBuffer.buffer,  0),
                               (sortedIndicesBuffer,    1),
                               (cellOffsetsBuffer,      2),
                               (gridMassBuffer,         3),
                               (gridMomBuffer,          4),
                               (uniformBuffer,          5),
                           ])

        // 7. Grid update — unchanged. One thread per voxel.
        encodeGridUpdate(into: cb)

        // 8. G2P — unchanged. Gather grid velocity, advect particles.
        encodeParticlePass(into: cb,
                           pipeline: g2pPipeline,
                           label: "MLS.g2p",
                           extraBuffers: [(gridNodeBuffer, 1)],
                           uniformIndex: 2)

        // 9. Capsule push-out — if the host bound any colliders, push
        //    fluid particles out of their volumes and kill the inward
        //    component of velocity. Cheap when bound (one thread per
        //    fluid particle × small collider count), no-op otherwise.
        encodeCapsuleCollide(into: cb)

        // 10. Kinematic grip — snap pinned end-bands to their cap's rigid
        //     pose + velocity (bonded pull). No-op unless gripEnabled.
        encodeGripPin(into: cb)
    }

    private func encodeGripPin(into cb: MTLCommandBuffer) {
        guard gripEnabled,
              particleBuffer.count > 0,
              let enc = cb.makeComputeCommandEncoder()
        else { return }
        enc.label = "MLS.gripPin"
        enc.setComputePipelineState(gripPinPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(gripRestBuffer.buffer, offset: 0, index: 1)
        var u = MLSGripUniforms(
            particleCount: UInt32(particleBuffer.count),
            enabled: 1,
            dispA: SIMD4(gripDispA, 0),
            dispB: SIMD4(gripDispB, 0),
            velA:  SIMD4(gripVelA, 0),
            velB:  SIMD4(gripVelB, 0))
        enc.setBytes(&u, length: MemoryLayout<MLSGripUniforms>.stride, index: 2)
        let count = particleBuffer.count
        let w = min(count, gripPinPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeCapsuleCollide(into cb: MTLCommandBuffer) {
        guard let binding = colliders,
              binding.count > 0,
              particleBuffer.count > 0,
              let enc = cb.makeComputeCommandEncoder()
        else { return }
        enc.label = "MLS.collideCapsules"
        enc.setComputePipelineState(collideCapsulesPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(binding.buffer,        offset: 0, index: 1)
        var u = MLSColliderUniforms(
            particleCount:   UInt32(particleBuffer.count),
            colliderCount:   UInt32(binding.count),
            skinDepth:       colliderSkin,
            restitution:     colliderRestitution,
            boost:           colliderDisplacementBoost,
            tangentFriction: colliderTangentFriction
        )
        enc.setBytes(&u, length: MemoryLayout<MLSColliderUniforms>.stride, index: 2)
        let count = particleBuffer.count
        let w = min(count, collideCapsulesPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    // ── Encode helpers ───────────────────────────────────────────────────────

    /// Dispatch a 1-D kernel over the cell count. Shared by the cell-clear
    /// pass and any other "one thread per cell, no extra inputs" kernel.
    private func encodeCellPass(into cb: MTLCommandBuffer,
                                pipeline: MTLComputePipelineState,
                                label: String,
                                buffers: [(MTLBuffer, Int)]) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers {
            enc.setBuffer(buf, offset: 0, index: idx)
        }
        let count = gridCellCount
        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    /// Single-threadgroup, single-thread prefix-sum scan. For 32k-cell grids
    /// the sequential reduce is ~32 µs and well below the noise floor of
    /// the rest of the pipeline. Bigger grids should swap in a multi-pass
    /// parallel scan — see the PIPELINE NOTE in MLSMPM.metal.
    private func encodeScanPass(into cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MLS.cellOffsetsScan"
        enc.setComputePipelineState(offsetsScanPipeline)
        enc.setBuffer(cellCountsBuffer,  offset: 0, index: 0)
        enc.setBuffer(cellOffsetsBuffer, offset: 0, index: 1)
        enc.setBuffer(uniformBuffer,     offset: 0, index: 2)
        enc.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    /// Dispatch a 3-D kernel over the cell grid with caller-specified
    /// buffer bindings. Used by the gather P2G; the legacy grid-update
    /// kernel has its own dedicated helper because its bindings are fixed.
    private func encodeGridUpdate3D(into cb: MTLCommandBuffer,
                                    pipeline: MTLComputePipelineState,
                                    label: String,
                                    buffers: [(MTLBuffer, Int)]) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        for (buf, idx) in buffers {
            enc.setBuffer(buf, offset: 0, index: idx)
        }
        let res = MTLSize(width:  Int(gridResolution.x),
                          height: Int(gridResolution.y),
                          depth:  Int(gridResolution.z))
        let tile = MTLSize(width: 4, height: 4, depth: 4)
        enc.dispatchThreads(res, threadsPerThreadgroup: tile)
        enc.endEncoding()
    }

    private func encodeGridUpdate(into cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MLS.gridUpdate"
        enc.setComputePipelineState(gridUpdatePipeline)
        enc.setBuffer(gridMassBuffer,  offset: 0, index: 0)
        enc.setBuffer(gridMomBuffer,   offset: 0, index: 1)
        enc.setBuffer(gridNodeBuffer,  offset: 0, index: 2)
        enc.setBuffer(uniformBuffer,   offset: 0, index: 3)
        // 3-D dispatch — one thread per voxel.
        let res = MTLSize(width:  Int(gridResolution.x),
                          height: Int(gridResolution.y),
                          depth:  Int(gridResolution.z))
        // Modest tile size; the kernel is light (per-voxel arithmetic).
        let tile = MTLSize(width: 4, height: 4, depth: 4)
        enc.dispatchThreads(res, threadsPerThreadgroup: tile)
        enc.endEncoding()
    }

    private func encodeParticlePass(into cb: MTLCommandBuffer,
                                    pipeline: MTLComputePipelineState,
                                    label: String,
                                    extraBuffers: [(MTLBuffer, Int)],
                                    uniformIndex: Int) {
        let count = particleBuffer.count
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        for (buf, idx) in extraBuffers {
            enc.setBuffer(buf, offset: 0, index: idx)
        }
        enc.setBuffer(uniformBuffer, offset: 0, index: uniformIndex)
        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        enc.endEncoding()
    }
}

// ── SplitMix64 — tiny deterministic RNG for particle seeding ────────────────
//
// 64-bit state, 64-bit output. Good enough for jitter; doesn't need to be a
// secure or statistically rigorous RNG. Mirrors the same simple seed-based
// PRNG used in a few other scenes (Eggs, FlyingCats) so seeded box layouts
// are reproducible across runs.

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z &>> 27)) &* 0x94D049BB133111EB
        return z ^ (z &>> 31)
    }
    mutating func nextFloat01() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
