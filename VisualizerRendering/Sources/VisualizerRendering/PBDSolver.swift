import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── PBD TYPES ────────────────────────────────────────────────────────────────
//
// These structs are shared between Swift and Metal. They MUST match the Metal
// struct layouts in PBD.metal exactly. See the ALIGNMENT RULE below.
//
// ALIGNMENT RULE: Never use a bare float3 in a Swift↔Metal shared struct.
// Metal's `float3` is 16-byte aligned (same as float4), so MemoryLayout in
// Swift would be 12 bytes while the Metal shader expects 16. Use float4 in
// Metal + SIMD4<Float> in Swift, packing extra data into the w component.
// OR use packed_float3 in Metal + SIMD3<Float> in Swift for vertex arrays
// where you want compact storage (DynamicMesh positions).

// One simulation particle: current position + previous position (Verlet).
// invMass = 0 pins the particle in place (use for anchored rope ends, grass
// roots, cloth corners).
public struct PBDParticle {
    // xyz = world position, w = invMass (0 = pinned)
    public var positionAndInvMass: SIMD4<Float>
    // xyz = previous world position, w = unused padding
    public var prevPositionAndPad: SIMD4<Float>

    public init(position: SIMD3<Float>, invMass: Float = 1.0) {
        positionAndInvMass = SIMD4(position, invMass)
        prevPositionAndPad = SIMD4(position, 0)
    }

    public var position: SIMD3<Float> {
        SIMD3(positionAndInvMass.x, positionAndInvMass.y, positionAndInvMass.z)
    }
    public var invMass: Float { positionAndInvMass.w }
}

// One distance constraint between particles i and j.
//
// XPBD compliance formulation (Müller 2020). `compliance = α` has units of
// inverse stiffness (m/N for distance constraints). Physically:
//   • α = 0           → perfectly rigid (no extension regardless of load)
//   • α ≈ 1e-6 … 1e-4 → very stiff cloth / rope
//   • α ≈ 1e-3 … 1e-2 → noticeably stretchy
//   • α ≫ 1           → mush / no effective constraint
//
// XPBD makes stiffness *iteration-count independent*: more iterations refine
// convergence toward the same equilibrium, rather than progressively stiffening
// the same constraint as in classic PBD. Equally important for our use case:
// soft XPBD constraints actually *yield* when fighting another constraint
// (e.g. an SDF contact push), so long-range chain constraints can be present
// for shape-holding without dominating inter-body collisions.
//
// Most callers build the public API in terms of an ergonomic stiffness ∈ [0,1]
// — see `stiffnessToCompliance(_:)` in PBDSolver for the mapping into the
// physical compliance.
public struct PBDConstraint {
    public var i: UInt32
    public var j: UInt32
    public var restLength: Float
    public var compliance: Float

    public init(i: UInt32, j: UInt32, restLength: Float, compliance: Float) {
        self.i = i; self.j = j; self.restLength = restLength; self.compliance = compliance
    }
}

// SDF collider used by the Phase 3 inter-body collision pass.
//
// One struct, three shape kinds — keeps the GPU buffer homogeneous and lets the
// kernel iterate a single array without per-shape buffer juggling.
//
//   .sphere(center, radius)            → a.xyz = center,    b.xyz = unused,       b.w = radius
//   .capsule(a, b, radius)             → a.xyz = endpoint0, b.xyz = endpoint1,    b.w = radius
//   .box(center, halfExtents)          → a.xyz = center,    b.xyz = halfExtents,  b.w = 0
//
// `a.w` carries the type tag (0 = sphere, 1 = capsule, 2 = box) cast through
// `Float(bitPattern:)` so a single 16-byte float4 holds both centre and tag.
//
// `ownerID` lets a particle skip colliders that belong to its own body. Hot dogs
// emit one capsule per spine segment and tag every capsule with the tube's ID;
// the kernel skips matching IDs so the tube never collides with itself.
//
// Layout: 48 bytes (16 + 16 + 16). Matches `struct PBDCollider` in PBD.metal —
// keep the two definitions in lockstep.
public struct PBDCollider {
    public var a:    SIMD4<Float>   // xyz = primary point, w = type tag (bit-cast)
    public var b:    SIMD4<Float>   // xyz = secondary point / extents, w = radius
    public var meta: SIMD4<UInt32>  // x = ownerID, yzw = reserved

    public enum Kind: UInt32 { case sphere = 0, capsule = 1, box = 2 }

    public init(a: SIMD4<Float>, b: SIMD4<Float>, ownerID: UInt32) {
        self.a = a; self.b = b
        self.meta = SIMD4(ownerID, 0, 0, 0)
    }

    public static func sphere(center: SIMD3<Float>, radius: Float,
                              ownerID: UInt32 = .max) -> PBDCollider {
        PBDCollider(
            a: SIMD4(center.x, center.y, center.z,
                     Float(bitPattern: Kind.sphere.rawValue)),
            b: SIMD4(0, 0, 0, radius),
            ownerID: ownerID
        )
    }

    public static func capsule(a: SIMD3<Float>, b: SIMD3<Float>, radius: Float,
                               ownerID: UInt32 = .max) -> PBDCollider {
        PBDCollider(
            a: SIMD4(a.x, a.y, a.z, Float(bitPattern: Kind.capsule.rawValue)),
            b: SIMD4(b.x, b.y, b.z, radius),
            ownerID: ownerID
        )
    }

    public static func box(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                           ownerID: UInt32 = .max) -> PBDCollider {
        PBDCollider(
            a: SIMD4(center.x, center.y, center.z,
                     Float(bitPattern: Kind.box.rawValue)),
            b: SIMD4(halfExtents.x, halfExtents.y, halfExtents.z, 0),
            ownerID: ownerID
        )
    }
}

// Per-frame uniforms uploaded to every kernel. Internal — callers never build these.
struct PBDUniforms {
    var dt: Float
    var gravity: Float
    var damping: Float
    var floorY: Float
    var particleCount: UInt32
    var constraintCount: UInt32
    var floorBand: Float = 0   // tolerance zone above floorY; kill upward velocity here
    var colliderCount: UInt32 = 0
    var ownerID: UInt32 = .max     // colliders with this ID are skipped (self-collision)
    var collideStiffness: Float = 1.0  // 0…1 scale on SDF push-out per substep
    var collideFriction: Float = 0.4   // 0…1 RETENTION of tangential vel on contact
    var selfRadius: Float = 0.0        // particles' own swept radius — added to collider radius
    var velRestitution: Float = 0.2    // XPBD velocity-solve restitution along constraint gradient
    var floorRestitution: Float = 0.2  // bounce coefficient against the floor
    // ── Endless-feed ring treadmill (SoftServeCoilSim) ─────────────────────────
    // When ringCount > 0 the chain is a RING of `ringCount` particles and the
    // distance/bend constraints crossing the open seam (between ringHead-1 and
    // ringHead) are skipped in pbdConstraint, so the ring behaves as an OPEN
    // rope whose cut point moves as the treadmill recycles. ringCount == 0
    // disables the gate entirely (every existing PBD caller — chains, not rings).
    var ringCount: UInt32 = 0
    var ringHead:  UInt32 = 0
    // > 0 → clamp per-substep SDF push-out to this many metres (anti-tear for thin
    // sheets buried in solid boxes). 0 = unlimited; every non-cloth caller leaves
    // it 0 and is unaffected. See `maxSDFPush` in PBD.metal.
    var maxSDFPush: Float = 0
}

// The PBD compute pipelines (integrate / constraint / floor / SDF collide /
// velocity / collider-build / tube-expand) used to live in a PBD-specific
// `PBDPipelineCache` here. They've been promoted to `SimPipelineCache` in
// SimPipelineCache.swift so other GPU solvers (MLS-MPM fluid, SPH foam, …)
// can share the same memoised pipeline-state machinery. PBD now asks the
// shared cache for its named bundle via `SimPipelineCache.shared.pbdPipelines(for:)`.

// Uniforms for the GPU collider-rebuild kernel. Internal — only
// PBDTubeRenderer drives the dispatch. Layout matches the matching Metal struct
// in PBD.metal (4 × 4-byte fields, no padding).
struct BuildCollidersUniforms {
    var particleCount: UInt32
    var targetOffset:  UInt32
    var ownerID:       UInt32
    var radius:        Float
}

// Uniforms for the particle self-collision kernel (liquid-rope coiling).
// Internal — driven by PBDSolver's dispatchSelfCollide. Layout matches
// `struct PBDSelfCollideUniforms` in PBD.metal (8 × 4-byte fields).
struct PBDSelfCollideUniforms {
    var particleCount: UInt32
    var selfRadius:    Float
    var stiffness:     Float
    var friction:      Float
    var selfSkip:      UInt32
    // Cup-wall confinement (inside-out cylinder). See PBD.metal.
    var wallEnabled:   UInt32 = 0
    var wallRadius:    Float = 0
    var wallCx:        Float = 0
    var wallCz:        Float = 0
    var wallBottomY:   Float = 0
}

// Uniforms for the endless-feed recycle (treadmill) kernel. Internal — driven
// by PBDSolver.recycleStep. Layout matches `struct PBDRecycleUniforms` in
// PBD.metal (8 × 4-byte fields).
struct PBDRecycleUniforms {
    var oldAnchor:  UInt32
    var newAnchor:  UInt32
    var spoolX:     Float
    var spoolY:     Float
    var spoolZ:     Float
    var seedAngle:  Float
    var seedRadius: Float
    var pad0:       Float = 0
}

// Uniforms for the sinking-floor conveyor kernel. Internal — driven by
// PBDSolver.encodeConveyorSink. Layout matches `struct PBDConveyorUniforms`
// in PBD.metal (4 × 4-byte fields).
struct PBDConveyorUniforms {
    var particleCount: UInt32
    var sinkDelta:     Float
    var sinkFloorY:    Float
    var pad0:          Float = 0
}

// Uniforms for the column-depth probe kernel. Layout matches
// `struct PBDProbeUniforms` in PBD.metal (4 × 4-byte fields).
struct PBDProbeUniforms {
    var particleCount: UInt32
    var recycleY:      Float
    var pad0:          Float = 0
    var pad1:          Float = 0
}

// Column-depth probe readback. Layout matches `struct PBDColumnProbe` in
// PBD.metal (4 × 4-byte fields). Read back ASYNCHRONOUSLY in the tick command
// buffer's completion handler — never via a per-frame waitUntilCompleted.
public struct PBDColumnProbe: Sendable {
    public var columnTopY:      Float = 0
    public var buriedTipMinY:   Float = 0
    public var liveCount:       UInt32 = 0
    public var belowPlaneCount: UInt32 = 0
    public init() {}
}

// Uniforms for the tube-expand kernel. Internal — only PBDTubeRenderer uses these.
struct TubeExpandUniforms {
    var ringSegments: UInt32
    var spineCount: UInt32
    var radius: Float
    // Visual rings emitted per spine segment. Kernel evaluates Catmull-Rom at
    // subSegments evenly-spaced t values per segment, giving smooth curves from
    // a low physics particle count. Total visual rings = (spineCount-1)*subSegments+1.
    var subSegments: UInt32 = 1
    // Optional caller-supplied "up" axis for the ring-frame seed (see PBD.metal
    // frameFromTangent). When `useFixedUp == 0`, the kernel uses its legacy
    // world-Y default with the Z fallback near vertical tangents. When nonzero,
    // it uses (fixedUpX, fixedUpY, fixedUpZ) — Hotdog Font passes Z to keep
    // glyph cross-sections from twisting as a stroke curves in the XY plane.
    var useFixedUp: UInt32 = 0
    var fixedUpX: Float = 0
    var fixedUpY: Float = 0
    var fixedUpZ: Float = 0
    // Ring-treadmill logical order. spineWrap != 0 → spine is a ring; logical
    // ring sample k reads particles[(spineOffset + k) % spineCount]. Keeps the
    // open seam at the rope ENDS so no spurious segment is drawn across the
    // buffer wrap. spineWrap == 0 = plain linear chain (every other caller).
    var spineWrap: UInt32 = 0
    var spineOffset: UInt32 = 0
    /// Tip-taper fraction: 0 = perfect cylinder; 0.25 = tips 25% narrower than
    /// the body centre. Applied as a sin(π·bodyT) profile in the tube-expand
    /// kernel so the maximum cross-section coincides with the tube midpoint and
    /// both hemisphere caps scale down accordingly.
    var taperFactor: Float = 0
    /// Straight-line render mode (HotdogDropUltra).
    ///
    /// When 1, the kernel ignores the Catmull-Rom spline and renders a perfect
    /// straight cylinder from spine[0] to spine[N-1], with hemispherical caps
    /// on the same axis. This eliminates two artifacts that appear when a
    /// constant-radius tube is swept along a bent piecewise-cubic centerline:
    ///   1. Silhouette lobes — any bend creates a region where more outer-radius
    ///      surface is visible, reading as a bulge at each interior knot.
    ///   2. Frame seam artifacts — at Catmull-Rom segment boundaries the
    ///      independent frameFromTangent computation may produce a tiny twist.
    /// Both are invisible for a straight chain; the PBD physics chain is still
    /// used for collision (capsules built from actual particle positions), only
    /// the VISUAL mesh is straight. Matches the SCNCapsule shape from HotdogDrop+.
    /// Default 0 = Catmull-Rom mode (SoftServe, HotdogFont, etc.).
    var useStraightLine: UInt32 = 0
    /// Dip-coat waterline (see PBD.metal TubeExpandUniforms — keep in lockstep).
    /// When dipStrength > 0, tube vertices below dipY within dipRadius of the
    /// world Y axis blend their vertex RGB toward (dipR, dipG, dipB), a
    /// per-channel albedo multiplier the host derives from fluid ÷ body colour.
    var dipY: Float = 0
    var dipRadius: Float = 0
    var dipStrength: Float = 0
    var dipR: Float = 1
    var dipG: Float = 1
    var dipB: Float = 1
    /// Per-frank BODY colour multiplier written into the tube's vertex-colour
    /// stream (multiplied into albedo at shading). Default (1,1,1) = the legacy
    /// white-identity colour, so unbatched/per-instance scenes are unchanged. The
    /// BATCHED tube path sets this to the frank's per-frank albedo so a SINGLE
    /// white InstanceRef + the shared colour buffer reproduce per-frank tone in
    /// one draw call. The dip-coat blends bodyColor → (dipR,dipG,dipB) below the
    /// waterline, so a dip still reads on top of the per-frank body colour.
    var bodyR: Float = 1
    var bodyG: Float = 1
    var bodyB: Float = 1
    /// Rotation-minimizing frame mode (see PBD.metal — keep in lockstep). When 1
    /// and useStraightLine == 0, the ring frame is propagated along the centerline
    /// instead of rebuilt per-ring, eliminating the pinch/twist a bent Catmull-Rom
    /// tube gets from frameFromTangent's up-axis switch. Default 0 = legacy frame.
    var useRMF: UInt32 = 0
}

// ── PBD SOLVER (XPBD core) ───────────────────────────────────────────────────
//
// Extended Position Based Dynamics solver for chains of connected particles
// (Müller 2020). All computation happens on the GPU via Metal compute; the
// Swift side manages buffers, pipelines, and the fixed-timestep accumulator.
//
// Each distance constraint stores a `compliance` (α, units m/N). The kernel
// accumulates a Lagrange multiplier λ per constraint within each substep
// (blit-reset to 0 between substeps), and applies:
//
//     α̃    = α / dt²
//     Δλ   = (−C − α̃·λ) / (w_i + w_j + α̃)
//     λ   += Δλ
//     Δx_i =  w_i · n · Δλ      Δx_j = −w_j · n · Δλ
//
// XPBD gives us three things that classic PBD couldn't:
//   • Iteration-count-independent stiffness (4 iters and 40 iters reach the
//     same equilibrium; classic PBD's "stiffness 0.3" became near-rigid by
//     ~10 iters).
//   • Soft constraints that actually yield to other constraints — long-range
//     chain straightening with α > 0 lets the SDF contact push win in piles
//     instead of being negotiated to a partial-overlap compromise.
//   • Cleanly separated position vs. velocity corrections — the implicit-
//     velocity ghost impulse from classic PBD is gone.
//
// See docs/known-issues/pbd-end-on-kinking.md for the residual tuning knobs.
//
// CONSTRAINT: the floor kernel must stay AFTER all constraint iterations —
// running it inside the loop or before constraints was tried twice and caused
// resonance / artificial upward velocity. See `floorBand` comment below.
//
// ROADMAP — FUTURE EXTENSIONS
//
// True angle-based bending constraints
//   The current skip-one distance constraints approximate bending stiffness but
//   fight axial compression. Add a `pbdBendConstraint` kernel on triples
//   (i-1, i, i+1) that penalises angle deviation from the rest angle at the
//   hinge, using the same XPBD compliance machinery. Reference: Müller 2007
//   "Position Based Dynamics", §3.4.
//
// Shape matching
//   For blob / soft-body simulations (gummy bear deformation, jelly cube),
//   add a shape-matching kernel: compute the polar decomposition of the
//   deformation gradient relative to a rest pose, then pull each particle
//   toward its target position. This prevents volume collapse that pure
//   distance constraints suffer from. Pair with the XPBD compliance machinery
//   for the per-substep λ accumulation.
//
// Self-collision via spatial hash
//   Once particle count > ~256, O(n²) self-contact is too slow. Add a
//   buildSpatialHash kernel that bins particles into a uniform grid, and a
//   selfCollide kernel that queries only nearby cells. The same hash grid is
//   reused for SPH fluid neighbour searches.
//
// SPH fluid / MLS-MPM water
//   See PBDTubeRenderer.swift for the full roadmap.
//
// ─────────────────────────────────────────────────────────────────────────────
// FUTURE PERF — one shared solver per scene
// ─────────────────────────────────────────────────────────────────────────────
//
// We currently allocate one PBDSolver per body, batch their encode() calls into
// a single command buffer, and rely on pooling to avoid steady-state churn. The
// remaining cost at high body counts is GPU command-encoder overhead: each
// solver issues its own encoder for every constraint sub-pass, so per-tick
// encoder count grows ~linearly with body count.
//
// The next perf tier collapses N solvers into one shared solver with one big
// particle buffer, one big constraint buffer, and a "chain offset" table:
//
//   - One MTLBuffer of [PBDParticle], sized for maxBodies × maxParticlesPerBody.
//   - One MTLBuffer of [PBDConstraint] storing GLOBAL particle indices.
//   - A per-chain "slots" array on the Swift side: each body holds
//     `(particleStart, particleCount, constraintStart, constraintCount)`.
//   - PBDUniforms.ownerID becomes a PER-PARTICLE field. Stash it in the
//     currently-unused `prevPositionAndPad.w` slot — `as_type<uint>` on the
//     GPU side, no extra buffer needed.
//   - integrate / floor / SDF dispatch ONCE across all particles.
//   - constraint dispatch ONCE per colour group across ALL bodies' constraints
//     of that colour (the existing per-body even/odd primary + 3-colour
//     bending split extends naturally — colour each global constraint by
//     `(i % stride)` within its own chain).
//   - tubeExpand still dispatches per body, since each body writes to its own
//     position/normal MTLBuffer (those can't easily share because SceneKit
//     reads them as separate vertex sources).
//
// Trade-offs:
//   • Encoder count goes from O(bodies × passes) to O(passes) per substep.
//     With 100 bodies at 4 iterations × ~6 sub-passes that's ~2400 → ~24.
//   • Body lifecycle (spawn / cull) becomes slot management — a free-slot
//     bitmap, fragmentation worries, optional compaction pass.
//   • Per-body settings (ownerID, friction, selfRadius) move from solver-
//     wide uniforms to per-particle fields. PBDUniforms slims down to a
//     genuinely scene-wide struct.
//   • Tube renderer changes: it currently reads `solver.particleBuffer.contents`
//     for cull / collider building; would need to read its own slice.
//
// Defer until pool-driven steady-state is exhausted and encoder count is
// measured to still be the bottleneck. The current path (pipeline cache +
// index/material sharing + free pool) buys a lot of headroom before that.

@MainActor
public final class PBDSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "PBDSolver")

    // ── GPU plumbing ──────────────────────────────────────────────────────────
    /// The MTLDevice that owns every buffer and pipeline this solver uses.
    /// Exposed so PBDTubeRenderer (and the controller's batched-encode path)
    /// can read it without going through a dedicated command queue.
    public let device: MTLDevice

    /// Command queue is lazily created on first use of the *standalone* paths
    /// (`advance(wallDt:)` / PBDTubeRenderer.tick). The Visualizer controller
    /// batches all tubes through its own shared queue and never touches this —
    /// so at 4 spawns/sec we avoid 4 MTLCommandQueue allocations per second.
    public var commandQueue: MTLCommandQueue {
        if let q = _commandQueue { return q }
        guard let q = device.makeCommandQueue() else {
            preconditionFailure("MTLDevice.makeCommandQueue() returned nil")
        }
        q.label = "PBDSolver"
        _commandQueue = q
        return q
    }
    private var _commandQueue: MTLCommandQueue?

    private let integratePipeline:  MTLComputePipelineState
    private let constraintPipeline: MTLComputePipelineState
    private let floorPipeline:      MTLComputePipelineState
    private let collidePipeline:    MTLComputePipelineState
    private let velocityPipeline:   MTLComputePipelineState
    private let selfCollidePipeline: MTLComputePipelineState
    private let recyclePipeline:     MTLComputePipelineState
    private let conveyorPipeline:    MTLComputePipelineState
    private let probePipeline:       MTLComputePipelineState

    // ── Particle / constraint storage ─────────────────────────────────────────
    public let particleBuffer:   SimBuffer<PBDParticle>
    public let constraintBuffer: SimBuffer<PBDConstraint>

    /// One float per constraint — the Lagrange multiplier λ that XPBD
    /// accumulates *within a single substep* and *resets between substeps*.
    /// Allocated to match `constraintBuffer.capacity`; the per-substep reset
    /// is a single MTLBlitCommandEncoder.fill (byte 0 = 0.0f exactly).
    private let lambdaBuffer: MTLBuffer

    private let uniformBuffer: MTLBuffer

    // ── Tunable physics ───────────────────────────────────────────────────────
    public var gravity: Float = 9.8
    public var damping: Float = 0.99
    public var floorY:  Float = 0.0
    // Band above floorY within which the floor kernel also kills upward velocity.
    // Set to the tube radius so constraint corrections that push a particle
    // slightly above the floor don't become ε/dt artificial upward velocity.
    public var floorBand: Float = 0.0
    // PERF vs PILE STABILITY: dispatch overhead per encoder is the dominant
    // cost once >5 tubes are live (~6 sub-passes per iteration × N tubes),
    // and the chain-vs-SDF tug-of-war converges more reliably with more
    // alternations. Bumped 6 → 10 once we measured 13 % sustained
    // interpenetration in dense piles (12 + tubes) — at 6 iterations the
    // SDF push didn't have enough chances to win against neighbouring
    // chains' constraint corrections per substep. 10 lands well inside the
    // GPU budget on M-series and resolves the worst pile overlaps within
    // a handful of frames after settling.
    public var constraintIterations: Int = 10

    // ── SDF colliders (Phase 3) ───────────────────────────────────────────────
    //
    // A binding into a controller-owned MTLBuffer of `PBDCollider` records. The
    // controller rebuilds this buffer each substep with the world-space shapes
    // every body should see (capsules for other hot dogs, a bowl, etc.). The
    // solver reads it during the SDF pass and skips colliders whose `ownerID`
    // matches `self.ownerID` so a body never collides with itself.
    //
    // Left nil → SDF pass is skipped (back-compat with rope / cloth callers).
    public struct ColliderBinding {
        public var buffer: MTLBuffer
        public var count:  Int
        public init(buffer: MTLBuffer, count: Int) {
            self.buffer = buffer; self.count = count
        }
    }
    public var colliders: ColliderBinding?
    /// Identifier for this body. Particles in this solver skip colliders whose
    /// `ownerID` matches. Defaults to `.max` (matches nothing) so a single-body
    /// scene still gets self-skipping for free if you tag your shapes accordingly.
    public var ownerID: UInt32 = .max
    /// Scale on the per-substep SDF push-out. 1.0 = full correction in one step.
    public var collideStiffness: Float = 0.8
    /// Fraction of tangential velocity *retained* on every SDF contact. Lower
    /// values dissipate sliding/grinding energy faster, but starve a settled
    /// pile of the lateral velocity it needs to unwind interpenetration: a
    /// stacked-tubes contact has a near-vertical SDF gradient and depends on
    /// tangential drift to slide neighbours apart over many substeps.
    /// 1.0 = frictionless contacts; 0.0 = particle sticks to the contact point.
    /// 0.70 (≈30 % friction) is the working compromise — quick enough to kill
    /// the writhing-pile feedback loop, slippery enough to let piles relax.
    public var collideFriction: Float = 0.70
    /// Effective tube radius of *this* body. Added to every collider's own
    /// radius inside `pbdSDFCollide` to give the true particle-vs-tube contact
    /// distance — particles are spine centres of an `selfRadius`-thick tube,
    /// not infinitely thin points. Leave 0 for rope / chain bodies where the
    /// particles really are point masses.
    public var selfRadius: Float = 0.0

    // ── Particle self-collision (liquid-rope coiling) ──────────────────────────
    //
    // The XPBD distance/bend constraints give a rope its shape but no contact:
    // a falling viscous rope passes through its own pile instead of stacking.
    // `pbdSelfCollide` (KERNEL 4b) is the missing inter-particle contact term —
    // every particle is a swept sphere of `selfCollideRadius`, and two
    // particles closer than 2·radius are pushed apart UNLESS they're chain
    // neighbours within `selfCollideSkip` indices (those are held tight by the
    // distance constraints on purpose). That's exactly the coil-on-coil and
    // coil-on-pile contact that produces liquid-rope coiling.
    //
    // O(n²) per dispatch — fine for a few-hundred-particle rope; see the kernel
    // docstring for the spatial-hash scale-up. Disabled (radius 0) for ropes /
    // chains that don't need self-contact, so existing callers are unaffected.
    public var selfCollideEnabled: Bool = false
    /// Swept-sphere radius per particle. Contact occurs at 2× this separation.
    /// Set to ≈ the rope's visual radius so coils rest tangent to each other.
    public var selfCollideRadius: Float = 0.0
    /// 0…1 push scale per dispatch. < 1 lets the rigid distance constraints win
    /// over the contact push within the iteration loop (no explosion).
    public var selfCollideStiffness: Float = 0.5
    /// 0…1 tangential-velocity retention on a self-contact (1 = frictionless).
    /// Lower values settle the pile faster but can starve coils of the lateral
    /// slip they need to nest. ≈0.6 is the working compromise.
    public var selfCollideFriction: Float = 0.6
    /// Exclude particle pairs within this many chain indices from self-contact.
    /// Must be ≥ 1 so immediate neighbours (held at segLength < 2·radius by the
    /// distance constraints) don't fight the contact term. Larger values let
    /// the rope coil more tightly before a turn counts as "touching itself".
    public var selfCollideSkip: Int = 2

    // ── Cup-wall confinement (inside-out cylinder SDF) ─────────────────────────
    //
    // Keeps the rope INSIDE a vertical cylinder so coils stack into a tall
    // column instead of spreading flat — the soft-serve cup. Evaluated in the
    // self-collide pass (same per-particle dispatch, O(1) extra), so it only
    // costs anything when `wallEnabled` is set. The push is one-sided: only
    // particles that escape past `wallRadius` feel an inward correction.
    public var wallEnabled: Bool = false
    /// Inner radius of the cup (m). Particles are kept within this of the axis.
    public var wallRadius: Float = 0
    /// Cup-axis position in XZ (the nozzle/column centre).
    public var wallCenterX: Float = 0
    public var wallCenterZ: Float = 0
    /// Confinement applies only above this Y (the cup floor); below it the rope
    /// is free (e.g. the open floor outside/under the cup).
    public var wallBottomY: Float = 0

    // ── Endless-feed ring treadmill ────────────────────────────────────────────
    //
    // Set by `configureRing`. `ringCount` > 0 turns on the seam gate in
    // pbdConstraint (the constraint(s) crossing the open seam between
    // ringHead-1 and ringHead are skipped, so the index-ring acts as an open
    // rope). `ringHead` is the current pinned-anchor index; `recycleStep`
    // advances it as it respools the buried tip back to the nozzle. 0 = off
    // (plain chain) for every existing caller.
    public private(set) var ringCount: Int = 0
    public private(set) var ringHead: Int = 0

    /// XPBD velocity post-solve restitution coefficient (Müller-2020 §3.5).
    /// After the position pass each substep, every distance constraint's
    /// normal-component relative velocity is rescaled to
    /// `−velocityRestitution · v_N_before`. Paired with non-zero primary
    /// compliance, this gives the chain a controlled elastic response to
    /// impact instead of the implicit-velocity ghost that produces end-on
    /// kinking.
    ///   • 0.0  → fully inelastic, oscillation dies in one substep (piles)
    ///   • 0.2  → chain flexes on impact and rings briefly (default)
    ///   • ≥0.8 → perpetual ring; don't use for distance constraints
    public var velocityRestitution: Float = 0.2
    /// Coefficient of restitution against the floor. Reflects a particle's
    /// downward velocity on contact instead of zeroing it. Without bounce,
    /// impact energy has nowhere to go but into the chain's bending DOF —
    /// which on end-on landings reads as visible kinking.
    ///   • 0.0 → fully inelastic (impact energy absorbed, the original
    ///           behaviour, kink-prone on end-on landings)
    ///   • 0.2 → real-world hot-dog elasticity (default)
    ///   • 1.0 → perfectly elastic, hot dog bounces forever
    public var floorRestitution: Float = 0.2
    // All constraints are stored in race-condition-free colour groups so every
    // GPU sub-pass writes to disjoint particles and can run fully in parallel.
    //
    // Primary (1-D chain, 2-colourable):
    //   even: (0,1),(2,3),(4,5),…  — no two share a particle
    //   odd:  (1,2),(3,4),(5,6),…  — no two share a particle
    //
    // Skip-one bending (each particle appears in up to 2 constraints → 3 colours):
    //   A: i%3==0 → (0,2),(3,5),(6,8),…  — no two share a particle
    //   B: i%3==1 → (1,3),(4,6),(7,9),…  — no two share a particle
    //   C: i%3==2 → (2,4),(5,7),(8,10),… — no two share a particle
    //
    // Buffer layout: [even primary][odd primary][bend A][bend B][bend C][long-range]
    public private(set) var primaryEvenCount: Int = 0
    public private(set) var primaryConstraintCount: Int = 0
    public private(set) var bendACount: Int = 0
    public private(set) var bendBCount: Int = 0
    public private(set) var bendCCount: Int = 0

    // ── Fixed timestep ────────────────────────────────────────────────────────
    // NEXT STEP (XPBD): use compliance-based constraints that are truly
    // dt-independent. Then fixedDt can be coarser without visual softening.
    private let fixedDt: Float = 1.0 / 120.0
    public var accumulator: Float = 0.0

    // ── Init ─────────────────────────────────────────────────────────────────

    public init?(device: MTLDevice, maxParticles: Int, maxConstraints: Int) {
        // PERF: pipelines and the Metal library are expensive to build (10s of ms
        // each, especially the first pipeline on a cold device). At a 4/sec spawn
        // rate this dominated CPU time. SimPipelineCache hands out shared
        // pipeline-state instances keyed by device — every solver after the first
        // is essentially free to construct on the Metal side.
        guard let pipelines = SimPipelineCache.shared.pbdPipelines(for: device) else {
            Self.log.error("PBD pipeline cache failed — check PBD.metal is in VisualizerRendering/Shaders/")
            return nil
        }

        let integPipeline  = pipelines.integrate
        let constrPipeline = pipelines.constraint
        let flrPipeline    = pipelines.floor
        let collPipeline   = pipelines.collide
        let velPipeline    = pipelines.velocity
        let selfCollPipe   = pipelines.selfCollide
        let recyclePipe    = pipelines.recycle
        let conveyorPipe   = pipelines.conveyorSink
        let probePipe      = pipelines.probeColumn

        let lambdaBytes = MemoryLayout<Float>.stride * max(maxConstraints, 1)
        guard
            let pBuf = SimBuffer<PBDParticle>(device: device,
                                              capacity: maxParticles,
                                              label: "PBD.particles"),
            let cBuf = SimBuffer<PBDConstraint>(device: device,
                                                capacity: maxConstraints,
                                                label: "PBD.constraints"),
            let uBuf = device.makeBuffer(length: MemoryLayout<PBDUniforms>.stride,
                                         options: .storageModeShared),
            let scBuf = device.makeBuffer(length: MemoryLayout<PBDSelfCollideUniforms>.stride,
                                          options: .storageModeShared),
            let rcBuf = device.makeBuffer(length: MemoryLayout<PBDRecycleUniforms>.stride,
                                          options: .storageModeShared),
            let cvBuf = device.makeBuffer(length: MemoryLayout<PBDConveyorUniforms>.stride,
                                          options: .storageModeShared),
            let prBuf = device.makeBuffer(length: MemoryLayout<PBDColumnProbe>.stride,
                                          options: .storageModeShared),
            let puBuf = device.makeBuffer(length: MemoryLayout<PBDProbeUniforms>.stride,
                                          options: .storageModeShared),
            let lBuf = device.makeBuffer(length: lambdaBytes,
                                         options: .storageModePrivate)
        else {
            Self.log.error("PBD buffer allocation failed")
            return nil
        }
        uBuf.label = "PBD.uniforms"
        scBuf.label = "PBD.selfCollideUniforms"
        rcBuf.label = "PBD.recycleUniforms"
        cvBuf.label = "PBD.conveyorUniforms"
        prBuf.label = "PBD.columnProbe"
        puBuf.label = "PBD.probeUniforms"
        lBuf.label = "PBD.lambda"

        self.device              = device
        self.integratePipeline   = integPipeline
        self.constraintPipeline  = constrPipeline
        self.floorPipeline       = flrPipeline
        self.collidePipeline     = collPipeline
        self.velocityPipeline    = velPipeline
        self.selfCollidePipeline = selfCollPipe
        self.recyclePipeline     = recyclePipe
        self.conveyorPipeline    = conveyorPipe
        self.probePipeline       = probePipe
        self.particleBuffer      = pBuf
        self.constraintBuffer    = cBuf
        self.lambdaBuffer        = lBuf
        self.uniformBuffer       = uBuf
        self.selfCollideUniformBuffer = scBuf
        self.recycleUniformBuffer = rcBuf
        self.conveyorUniformBuffer = cvBuf
        self.columnProbeBuffer = prBuf
        self.probeUniformBuffer = puBuf
    }

    private let selfCollideUniformBuffer: MTLBuffer
    // Jacobi snapshot of particle positions for the self-collide pass (lazily
    // sized to the live particle count). Frozen-read source so contact pushes
    // are symmetric — see pbdSelfCollide in PBD.metal.
    private var selfCollideSnapshot: MTLBuffer?
    private let recycleUniformBuffer: MTLBuffer
    private let conveyorUniformBuffer: MTLBuffer
    private let probeUniformBuffer: MTLBuffer
    /// Shared-memory readback target for the column-depth probe. Safe to read
    /// from Swift only inside a completed command buffer's completion handler.
    /// `nonisolated(unsafe)` because it's an immutable reference to a shared
    /// MTLBuffer; the completion-handler reader (off the main actor) only reads
    /// its bytes after the writing command buffer has completed, so there's no
    /// data race despite MTLBuffer not being Sendable.
    nonisolated(unsafe) private let columnProbeBuffer: MTLBuffer

    // ── Stiffness ↔ compliance bridge ────────────────────────────────────────
    //
    // XPBD wants `compliance` in physical units (1/k, m/N for a distance
    // constraint), but most callers think in terms of an ergonomic 0…1 stiffness
    // slider. This is a deliberately gentle quadratic mapping:
    //
    //   s = 1.0  → α = 0                  (rigid)
    //   s = 0.7  → α ≈ 0.09 · scale       (firm, yields to strong contacts)
    //   s = 0.5  → α ≈ 0.25 · scale       (noticeably soft)
    //   s = 0.0  → α = scale              (essentially no constraint)
    //
    // `scale` is sized so that "fully soft" still produces a non-degenerate
    // constraint — large enough to read as compliant at our chain scales
    // (~10 cm segments, ~1 N forces) but small enough that even s=0.3 keeps
    // chain shape recognisably.
    //
    // Callers that want direct physical compliance can build PBDConstraint
    // values themselves; this helper is purely for the public stiffness-style
    // API on `configureChain`.
    public static func stiffnessToCompliance(_ stiffness: Float) -> Float {
        let s = max(0, min(1, stiffness))
        let scale: Float = 1e-3   // ≈ k of 1000 N/m at s=0 — fairly soft
        let t = 1 - s
        return t * t * scale
    }

    // ── Shared open-chain constraint topology ────────────────────────────────
    //
    // The canonical XPBD chain colouring used by BOTH the single-chain solver
    // (`configureChain`, base 0) and the field solver (`PBDFieldSolver`, one
    // call per chain with base = b·M). Emitting it from one place means the
    // five colour groups — and the disjoint-particle invariant each GPU
    // sub-pass relies on — can never drift between the two solvers. (This is
    // the family that has harboured the SIMD3-stride alignment bug twice; a
    // single generator keeps the index math honest.)
    //
    // Topology for a chain of `count` particles whose GLOBAL indices begin at
    // `base`:
    //   • even primary : (base+0, base+1), (base+2, base+3), …   compliance = primaryCompliance
    //   • odd  primary : (base+1, base+2), (base+3, base+4), …   compliance = primaryCompliance
    //   • skip-one bend: (base+i, base+i+2) for i in 0..<count-2, restLength ×2,
    //                    3-coloured by i%3 into (bendA, bendB, bendC)
    //
    // Returned in solver dispatch order. Caller owns concatenation, count
    // bookkeeping, and any solver-specific extras (long-range straightening,
    // ring wrap) — those stay out of the shared core on purpose.
    static func openChainColourGroups(
        count: Int, base: UInt32, segLength: Float,
        primaryCompliance: Float, bendCompliance: Float
    ) -> (even: [PBDConstraint], odd: [PBDConstraint],
          bendA: [PBDConstraint], bendB: [PBDConstraint], bendC: [PBDConstraint]) {
        var even = [PBDConstraint](), odd = [PBDConstraint]()
        var ba = [PBDConstraint](), bb = [PBDConstraint](), bc = [PBDConstraint]()
        // Even group: (0,1),(2,3),… — no two share a particle.
        for i in stride(from: 0, to: count - 1, by: 2) {
            even.append(PBDConstraint(i: base + UInt32(i), j: base + UInt32(i + 1),
                                      restLength: segLength, compliance: primaryCompliance))
        }
        // Odd group: (1,2),(3,4),… — no two share a particle.
        for i in stride(from: 1, to: count - 1, by: 2) {
            odd.append(PBDConstraint(i: base + UInt32(i), j: base + UInt32(i + 1),
                                     restLength: segLength, compliance: primaryCompliance))
        }
        // Skip-one bending: 3-colour graph colouring so no two constraints in
        // the same group share a particle — each group dispatches race-free.
        for i in 0..<(count - 2) {
            let c = PBDConstraint(i: base + UInt32(i), j: base + UInt32(i + 2),
                                  restLength: segLength * 2.0, compliance: bendCompliance)
            switch i % 3 {
            case 0:  ba.append(c)
            case 1:  bb.append(c)
            default: bc.append(c)
            }
        }
        return (even, odd, ba, bb, bc)
    }

    // ── Chain setup ──────────────────────────────────────────────────────────

    public func configureChain(count: Int, segLength: Float, anchorFirst: Bool = false,
                               bendStiffness: Float = 0.4,
                               longRangeStiffness: Float = 0.0,
                               primaryCompliance: Float = 5e-5) {
        var particles   = [PBDParticle]()
        var constraints = [PBDConstraint]()
        particles.reserveCapacity(count)
        constraints.reserveCapacity(count * 4)

        for i in 0..<count {
            var p = PBDParticle(position: SIMD3(Float(i) * segLength, 0, 0))
            if anchorFirst && i == 0 { p.positionAndInvMass.w = 0 }
            particles.append(p)
        }

        // Primary distance constraints use a SMALL non-zero compliance so the
        // chain can elastically compress on impact and rebound, rather than
        // accordion-buckling into the visible "hourglass" wave that pure-rigid
        // primary constraints produce on end-on landings. Default 5e-5:
        //   • Under gravity load (≈1.7 N per particle): segment stretch ≈
        //     1.7 · 5e-5 = 0.08 mm. Invisible.
        //   • Under impact load (≈200 N momentary): stretch ≈ 1 cm. Visible
        //     squish-and-rebound that reads as natural elasticity.
        // Iteration-count independent (the whole point of XPBD), so 6 iters
        // is plenty to reach equilibrium each substep.
        //
        // Bend and long-range constraints are SOFT via XPBD compliance, derived
        // from the caller's ergonomic stiffness-0-to-1 by `stiffnessToCompliance`.
        // A soft long-range constraint can hold the chain's general shape while
        // still yielding to a (rigid) SDF contact push — which is the whole
        // reason we moved off classic PBD.

        // Even/odd primary + skip-one bend share one generator with the field
        // solver — see `openChainColourGroups`. Single-chain base index = 0.
        let bendCompliance = PBDSolver.stiffnessToCompliance(bendStiffness)
        let groups = PBDSolver.openChainColourGroups(
            count: count, base: 0, segLength: segLength,
            primaryCompliance: primaryCompliance, bendCompliance: bendCompliance)
        constraints += groups.even
        let evenCount = constraints.count
        constraints += groups.odd
        let primaryCount = constraints.count
        let ba = groups.bendA, bb = groups.bendB, bc = groups.bendC
        constraints += ba + bb + bc

        // Skip-two and skip-three: optional long-range straightening (disabled by
        // default). These are appended after the coloured bending groups and run
        // as a single uncoloured pass; with XPBD compliance > 0 they yield to
        // any rigid contact push, so residual races at high body counts are
        // cosmetically harmless.
        if longRangeStiffness > 0 {
            let longCompliance = PBDSolver.stiffnessToCompliance(longRangeStiffness)
            // Skip-three is slightly softer than skip-two (×1.43 compliance ≈
            // ×0.7 stiffness in the legacy classic-PBD interpretation) so the
            // chain stays bendy at long range while keeping near-range firm.
            let skip3Compliance = longCompliance * 1.43
            for i in 0..<(count - 3) {
                constraints.append(PBDConstraint(i: UInt32(i), j: UInt32(i + 3),
                                                 restLength: segLength * 3.0,
                                                 compliance: longCompliance))
            }
            for i in 0..<(count - 4) {
                constraints.append(PBDConstraint(i: UInt32(i), j: UInt32(i + 4),
                                                 restLength: segLength * 4.0,
                                                 compliance: skip3Compliance))
            }
        }

        particleBuffer.write(particles)
        constraintBuffer.write(constraints)
        primaryEvenCount       = evenCount
        primaryConstraintCount = primaryCount
        bendACount             = ba.count
        bendBCount             = bb.count
        bendCCount             = bc.count
    }

    // ── Ring treadmill (endless feed) ──────────────────────────────────────────

    /// Build a RING of `count` particles for the endless-feed treadmill
    /// (SoftServeCoilSim). Identical constraint topology to `configureChain`
    /// (2-colour primary distance + 3-colour skip-one bend) but the links close
    /// the ring (i ↔ (i+1)%N and i ↔ (i+2)%N). One seam is held open at runtime
    /// by the `ringHead` gate in pbdConstraint, so the ring behaves as an OPEN
    /// rope whose cut point moves as `recycleStep` rotates the pin. `count` MUST
    /// be even so the ring's primary distance constraints 2-colour cleanly
    /// (even/odd) with no shared particle in a sub-pass.
    ///
    /// Particle positions are NOT written here — the caller seeds the compressed
    /// spool (all particles stacked just below the nozzle, particle 0 pinned).
    public func configureRing(count: Int, segLength: Float,
                              bendStiffness: Float = 0.4,
                              primaryCompliance: Float = 5e-5) {
        precondition(count >= 6 && count % 2 == 0,
                     "configureRing requires an even count ≥ 6")
        let n = count
        var constraints = [PBDConstraint]()
        constraints.reserveCapacity(n * 2)

        // Primary distance ring, 2-coloured by the lower index's parity.
        var even = [PBDConstraint](), odd = [PBDConstraint]()
        for i in 0..<n {
            let j = (i + 1) % n
            let c = PBDConstraint(i: UInt32(i), j: UInt32(j),
                                  restLength: segLength, compliance: primaryCompliance)
            if i % 2 == 0 { even.append(c) } else { odd.append(c) }
        }
        constraints += even
        let evenCount = even.count
        constraints += odd
        let primaryCount = constraints.count

        // Skip-one bend ring, 3-coloured by i%3. With a ring whose count isn't a
        // multiple of 3 the wrap can leave one same-colour pair adjacent at the
        // seam; bend is soft (XPBD compliance > 0) so a residual seam race is
        // cosmetically harmless (same rationale as long-range in configureChain).
        let bendCompliance = PBDSolver.stiffnessToCompliance(bendStiffness)
        var ba = [PBDConstraint](), bb = [PBDConstraint](), bc = [PBDConstraint]()
        for i in 0..<n {
            let j = (i + 2) % n
            let c = PBDConstraint(i: UInt32(i), j: UInt32(j),
                                  restLength: segLength * 2.0, compliance: bendCompliance)
            switch i % 3 {
            case 0:  ba.append(c)
            case 1:  bb.append(c)
            default: bc.append(c)
            }
        }
        constraints += ba + bb + bc

        constraintBuffer.write(constraints)
        primaryEvenCount       = evenCount
        primaryConstraintCount = primaryCount
        bendACount             = ba.count
        bendBCount             = bb.count
        bendCCount             = bc.count

        ringCount = n
        ringHead  = 0
    }

    /// Advance the treadmill one notch (call once per cadence, inside the tick
    /// command buffer, AFTER `encodeConstraintsAndContacts`). Releases the
    /// current anchor (`ringHead`) as fresh rope and respools the buried tip
    /// (`ringHead-1` in ring order) to a tight spool slot at `spool`, pinning
    /// it as the new anchor. O(1): one GPU thread, a couple of buffer writes,
    /// plus advancing the CPU `ringHead`. No readback, no CPU particle loop.
    public func encodeRecycle(into commandBuffer: MTLCommandBuffer,
                              spool: SIMD3<Float>,
                              seedAngle: Float,
                              seedRadius: Float) {
        guard ringCount > 0, particleBuffer.count > 0 else { return }
        let n = ringCount
        let oldAnchor = ringHead
        let newAnchor = (ringHead + n - 1) % n   // the buried tip

        let uPtr = recycleUniformBuffer.contents()
            .bindMemory(to: PBDRecycleUniforms.self, capacity: 1)
        uPtr.pointee = PBDRecycleUniforms(
            oldAnchor:  UInt32(oldAnchor),
            newAnchor:  UInt32(newAnchor),
            spoolX:     spool.x, spoolY: spool.y, spoolZ: spool.z,
            seedAngle:  seedAngle, seedRadius: seedRadius
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBD.recycle"
        encoder.setComputePipelineState(recyclePipeline)
        encoder.setBuffer(particleBuffer.buffer,    offset: 0, index: 0)
        encoder.setBuffer(recycleUniformBuffer,     offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()

        // The tip becomes the new pinned anchor; the open seam moves one notch
        // back with it. The constraint that WAS the seam (tip↔oldAnchor) becomes
        // a live link; the next link up becomes the new seam — all via the
        // ringHead gate, no constraint-buffer mutation.
        ringHead = newAnchor
    }

    // ── Sinking-floor conveyor (endless tall column) ─────────────────────────

    /// Lower the whole settled column by `sinkDelta` metres this substep,
    /// opening headroom below the nozzle so the active feed always has somewhere
    /// to coil (the world-space sinking-floor conveyor). Pinned particles and
    /// any already at/below `sinkFloorY` (the recycle plane) are left in place.
    /// Encode AFTER the constraint/contact loop so it slides the settled state.
    public func encodeConveyorSink(into commandBuffer: MTLCommandBuffer,
                                   sinkDelta: Float,
                                   sinkFloorY: Float) {
        let count = particleBuffer.count
        guard count > 0, sinkDelta > 0 else { return }
        let uPtr = conveyorUniformBuffer.contents()
            .bindMemory(to: PBDConveyorUniforms.self, capacity: 1)
        uPtr.pointee = PBDConveyorUniforms(
            particleCount: UInt32(count),
            sinkDelta:     sinkDelta,
            sinkFloorY:    sinkFloorY
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBD.conveyorSink"
        encoder.setComputePipelineState(conveyorPipeline)
        encoder.setBuffer(particleBuffer.buffer,    offset: 0, index: 0)
        encoder.setBuffer(conveyorUniformBuffer,    offset: 0, index: 1)
        let w = min(count, conveyorPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // ── Column-depth probe (closed-loop rate control) ────────────────────────

    /// Encode the column-depth reduction into `commandBuffer`. Reads the live
    /// particle column and writes (columnTopY, buriedTipMinY, liveCount,
    /// belowPlaneCount) into the shared probe buffer; `recycleY` is the plane
    /// below which particles are counted as owed a respool. Read the result back
    /// via `readColumnProbe(from:)` from the command buffer's completion handler
    /// — NEVER block on it. Encode near the END of the tick so it sees the
    /// settled, post-sink state.
    public func encodeProbeColumn(into commandBuffer: MTLCommandBuffer, recycleY: Float) {
        let count = particleBuffer.count
        guard count > 0 else { return }
        let uPtr = probeUniformBuffer.contents()
            .bindMemory(to: PBDProbeUniforms.self, capacity: 1)
        uPtr.pointee = PBDProbeUniforms(particleCount: UInt32(count), recycleY: recycleY)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBD.probeColumn"
        encoder.setComputePipelineState(probePipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(probeUniformBuffer,    offset: 0, index: 1)
        encoder.setBuffer(columnProbeBuffer,     offset: 0, index: 2)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// The shared-memory buffer the column probe writes into. Exposed so a
    /// controller can read it from a command-buffer completion handler (which
    /// runs off the main actor) via `PBDSolver.readColumnProbe(from:)` without a
    /// main-actor hop. `nonisolated` because it never touches mutable state.
    public nonisolated var columnProbeBufferRef: MTLBuffer { columnProbeBuffer }

    /// Read a column probe out of a shared-memory probe buffer. `nonisolated` +
    /// static so it's safe to call from a Metal completion handler (off the main
    /// actor). Only valid once the command buffer that ran `encodeProbeColumn`
    /// has completed — which is exactly when the completion handler fires.
    public nonisolated static func readColumnProbe(from buffer: MTLBuffer) -> PBDColumnProbe {
        buffer.contents().bindMemory(to: PBDColumnProbe.self, capacity: 1).pointee
    }

    /// Manually publish the colour-group split for callers that write the
    /// `constraintBuffer` themselves (instead of going through
    /// `configureChain`). The Hot Dog Font scene uses this so it can
    /// place spine particles along curved glyph strokes while still
    /// driving them with the same 2-colour primary + 3-colour bend layout
    /// the encode loop expects.
    ///
    /// The caller is responsible for matching the layout to the actual
    /// `constraintBuffer` contents — start offsets are derived as
    /// `[0, primaryEven, primaryTotal, +bendA, +bendB, +bendC]`.
    public func configureLayout(primaryEven: Int,
                                primaryTotal: Int,
                                bendA: Int,
                                bendB: Int,
                                bendC: Int) {
        primaryEvenCount       = primaryEven
        primaryConstraintCount = primaryTotal
        bendACount             = bendA
        bendBCount             = bendB
        bendCCount             = bendC
    }

    // ── Advance (standalone) ─────────────────────────────────────────────────

    public func advance(wallDt: Float) {
        accumulator += wallDt
        var steps = 0
        while accumulator >= fixedDt && steps < 4 {
            guard let cb = commandQueue.makeCommandBuffer() else { break }
            cb.label = "PBD.step"
            encode(to: cb, dt: fixedDt)
            cb.commit()
            accumulator -= fixedDt
            steps += 1
        }
    }

    // ── Encode into an external command buffer ───────────────────────────────
    //
    // `encode` is the convenience path — it runs the full substep in one call.
    // For multi-body scenes that need to rebuild SDF colliders from the GPU's
    // own freshly-integrated particle state, call the phase-split variants
    // explicitly:
    //
    //     for each body:  encodeIntegrate(cb, dt)
    //     for each body:  (use buildColliders pipeline to write capsules into
    //                      the shared collider buffer at this body's offset)
    //     for each body:  encodeConstraintsAndContacts(cb, dt)
    //
    // The phase split is what eliminates the asymmetric stale-collider read
    // that drove the "later-spawned tube creeps into earlier-spawned tube
    // after settling" bug. See the kernel docstring on pbdBuildCapsuleColliders
    // for the diagnosis.

    /// Convenience: full substep in one call. Equivalent to
    /// `encodeIntegrate` + `encodeConstraintsAndContacts`, but does NOT rebuild
    /// colliders between the two — so it sees whatever's in `colliders.buffer`
    /// at the moment of the call. Use the split form when multiple bodies share
    /// a collider buffer that must be rebuilt mid-substep.
    public func encode(to commandBuffer: MTLCommandBuffer, dt: Float) {
        encodeIntegrate(to: commandBuffer, dt: dt)
        encodeConstraintsAndContacts(to: commandBuffer, dt: dt)
    }

    /// Phase 1: write uniforms, integrate particles, reset XPBD lambda.
    /// After this kernel completes, `particleBuffer` holds the post-gravity
    /// positions that the collider-build kernel should read from.
    public func encodeIntegrate(to commandBuffer: MTLCommandBuffer, dt: Float) {
        let uPtr = uniformBuffer.contents().bindMemory(to: PBDUniforms.self, capacity: 1)
        uPtr.pointee = PBDUniforms(
            dt: dt, gravity: gravity, damping: damping, floorY: floorY,
            particleCount: UInt32(particleBuffer.count),
            constraintCount: UInt32(constraintBuffer.count),
            floorBand: floorBand,
            colliderCount: UInt32(colliders?.count ?? 0),
            ownerID: ownerID,
            collideStiffness: collideStiffness,
            collideFriction: collideFriction,
            selfRadius: selfRadius,
            velRestitution: velocityRestitution,
            floorRestitution: floorRestitution,
            ringCount: UInt32(ringCount),
            ringHead: UInt32(ringHead)
        )

        encodePass(commandBuffer, pipeline: integratePipeline,
                   count: particleBuffer.count, label: "PBD.integrate")

        // XPBD: reset every constraint's Lagrange multiplier to zero before the
        // iteration loop. Lambda accumulates within a substep and must restart
        // each substep — otherwise old-substep λ values are interpreted as the
        // current substep's accumulated impulse and the constraint behaves like
        // it's already partially satisfied. Byte 0 = 0.0f for IEEE float, so a
        // blit fill is correct and avoids a dedicated reset kernel.
        if constraintBuffer.count > 0,
           let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "PBD.lambdaReset"
            let bytes = MemoryLayout<Float>.stride * constraintBuffer.count
            blit.fill(buffer: lambdaBuffer, range: 0..<bytes, value: 0)
            blit.endEncoding()
        }
    }

    /// Phase 2: constraint iteration loop, floor pass, post-floor SDF cleanup.
    /// Reads from `colliders.buffer` whatever the caller has had written into
    /// it before this call (typically by a per-body buildColliders dispatch).
    public func encodeConstraintsAndContacts(to commandBuffer: MTLCommandBuffer,
                                             dt: Float) {
        // The buildColliders pass between integrate and this method may have
        // changed the bound collider count if bodies entered or left the
        // simulation. Re-write the uniforms with the up-to-date count so the
        // SDF kernel sees the right iteration limit. dt and tunables are still
        // accurate from the integrate-phase write — preserve them verbatim.
        let uPtr = uniformBuffer.contents().bindMemory(to: PBDUniforms.self, capacity: 1)
        var current = uPtr.pointee
        current.colliderCount = UInt32(colliders?.count ?? 0)
        uPtr.pointee = current

        // Five ordered sub-passes per iteration — all constraint groups are fully
        // graph-coloured so every sub-pass is race-condition-free on the GPU:
        //   1. Even primary  — (0,1),(2,3),…        2-colour, no shared particles
        //   2. Odd  primary  — (1,2),(3,4),…        2-colour, no shared particles
        //   3. Bend A        — (0,2),(3,5),(6,8),…  3-colour, no shared particles
        //   4. Bend B        — (1,3),(4,6),(7,9),…  3-colour, no shared particles
        //   5. Bend C        — (2,4),(5,7),(8,10),… 3-colour, no shared particles
        //   6. Long-range    — uncoloured, disabled by default (stiffness 0)
        let primaryOddStart  = primaryEvenCount
        let primaryOddCount  = primaryConstraintCount - primaryEvenCount
        let bendAStart       = primaryConstraintCount
        let bendBStart       = bendAStart + bendACount
        let bendCStart       = bendBStart + bendBCount
        let longRangeStart   = bendCStart + bendCCount
        let longRangeCount   = constraintBuffer.count - longRangeStart
        for i in 0..<constraintIterations {
            if primaryEvenCount > 0 {
                encodeConstraintSubpass(commandBuffer, start: 0,
                                        count: primaryEvenCount,
                                        label: "PBD.even[\(i)]")
            }
            if primaryOddCount > 0 {
                encodeConstraintSubpass(commandBuffer, start: primaryOddStart,
                                        count: primaryOddCount,
                                        label: "PBD.odd[\(i)]")
            }
            if bendACount > 0 {
                encodeConstraintSubpass(commandBuffer, start: bendAStart,
                                        count: bendACount,
                                        label: "PBD.bendA[\(i)]")
            }
            if bendBCount > 0 {
                encodeConstraintSubpass(commandBuffer, start: bendBStart,
                                        count: bendBCount,
                                        label: "PBD.bendB[\(i)]")
            }
            if bendCCount > 0 {
                encodeConstraintSubpass(commandBuffer, start: bendCStart,
                                        count: bendCCount,
                                        label: "PBD.bendC[\(i)]")
            }
            if longRangeCount > 0 {
                encodeConstraintSubpass(commandBuffer, start: longRangeStart,
                                        count: longRangeCount,
                                        label: "PBD.longRange[\(i)]")
            }

            // SDF inter-body collision (Phase 3). Run *inside* the constraint
            // loop so the constraint solver can react to the pushed positions
            // on the next iteration — otherwise the constraint pass spends all
            // its iterations re-pulling pairs that the single end-of-loop SDF
            // call would have pushed apart, and clipping shows through. Safe
            // to run here (unlike the floor) because collisions are between
            // bodies, not against a hard kinematic boundary, so no resonance.
            if let bind = colliders, bind.count > 0 {
                encodeCollidePass(commandBuffer, binding: bind)
            }

            // Particle self-collision is NO LONGER run here (was once per
            // constraint iteration = 14×/substep). Applying the contact push 14
            // times per substep over-corrected catastrophically — a particle in a
            // dense pile got shoved ~200 mm/substep and the rope stretched/exploded
            // (flung metres up; measured via stats CSV). It now runs ONCE per
            // substep after the loop (see below), which is the standard stable
            // cadence for contacts and cuts the injected energy ~14×.
        }

        // Particle self-collision — ONCE per substep, after the constraint loop.
        // (Liquid-rope coiling: coils rest on the coils below.) Overlaps resolve
        // across consecutive SUBSTEPS, which is stable; in-loop it was ~14× too hot.
        if selfCollideEnabled, selfCollideRadius > 0 {
            encodeSelfCollidePass(commandBuffer)
        }

        // XPBD velocity post-solve — DISABLED.
        //
        // Theory was that damping relative velocity along each chain
        // constraint's gradient would kill angular ringing without affecting
        // bulk motion. In practice this is wrong for distance constraints on
        // a near-straight chain: the gradient is parallel to the chain axis,
        // so the kernel damps AXIAL relative velocity — the exact channel
        // that propagates bounce energy from the floor-contact particle up
        // the chain.
        //
        // Worked example with the bottom particle bouncing at +0.9 m/s while
        // upper particles descend at −5 m/s:
        //   • Primary (0,1):  vRel·n = −5.9, restitution 0.2 → p0 reversed
        //     to −2.64 m/s (bounce inverted, no upward motion).
        //   • Bend    (0,2):  parallel gradient, same maths, same kill.
        //
        // The XPBD position solve with non-zero compliance already provides
        // elastic shock absorption along the chain axis. Floor restitution
        // handles the contact impulse. Integrate-kernel damping (0.99×) does
        // bulk energy dissipation. No velocity post-solve needed.
        //
        // Kernel + slider + dispatch helper left wired for a future
        // angle-based bend constraint, where the gradient really is the
        // angular DOF and damping it damps only angular wobble — the right
        // use for this machinery.

        // Single floor pass after all constraint iterations. The floorBand
        // tolerance in the kernel kills upward velocity for particles pushed
        // slightly above floorY by the last constraint round, preventing ε/dt
        // artificial velocity without the resonance of running floor every iter.
        encodePass(commandBuffer, pipeline: floorPipeline,
                   count: particleBuffer.count, label: "PBD.floor")

        // Post-floor SDF cleanup. The floor pass can clamp a particle to
        // floorY that is also currently inside another tube's capsule — without
        // this final separation pass, the overlap waits one full substep for
        // the next loop's first iteration to fix it. That one-substep lag is
        // exactly the steady-state overlap visible in settled piles. Running
        // SDF after floor catches the case and the subsequent particles enter
        // the next substep already separated.
        if let bind = colliders, bind.count > 0 {
            encodeCollidePass(commandBuffer, binding: bind)

            // ── Second floor clamp ─────────────────────────────────────────
            //
            // The post-floor SDF pass can push a bottom-body particle DOWN
            // (when another body sits directly above it): the SDF push
            // direction is from B's collider toward A's particle, which for a
            // stacked pair points straight into the floor. Without a second
            // floor clamp here, that downward push lands A's particle below
            // floorY for the rest of the substep, and the tube-expand kernel
            // reads it from there — bottom hot dog visibly clips through the
            // ground (the symptom QA reported as "bottom hot dog gets driven
            // into the floor to make space for the top hot dog").
            //
            // Safe to add at the very end: it's NOT inside the constraint
            // iteration loop (the floorBand resonance issue documented above
            // only applies there). At worst this re-imposes a contact the
            // post-floor SDF was trying to widen — but the next substep's
            // SDF passes will resolve the residue, with the body now starting
            // from a physically-valid above-floor position.
            encodePass(commandBuffer, pipeline: floorPipeline,
                       count: particleBuffer.count, label: "PBD.floor.final")
        }
    }

    // ── Batched-dispatch primitives (one encoder shared across many tubes) ───
    //
    // The convenience `encode*` methods above each open + end their own encoder
    // (and the integrate path also opens its own blit). That's clean for a
    // single-body caller, but at N tubes per substep it produces ~49 × N
    // encoder allocations — encoder overhead, not GPU work, is the dominant
    // cost at N > ~10 (kernels are ≤8 threads; the GPU work itself is tiny).
    //
    // These primitives bind + dispatch into a caller-owned encoder without
    // allocating one. Pair with `writeIntegrateUniforms` to keep all of a
    // substep's per-phase work in a small fixed number of encoders regardless
    // of N. See `HotdogDropPlusController.tickPBD` for the calling pattern.

    /// Write this solver's per-substep `PBDUniforms` into its uniform buffer.
    /// Cheap struct write into `.storageModeShared` memory; call once per
    /// substep before any of this solver's dispatches. Pulled out of the old
    /// `encodeIntegrate` so a batched-encoder caller doesn't have to round-trip
    /// through encoder allocation to land the uniforms.
    ///
    /// Picks up `colliders?.count` at call time — so the controller should set
    /// `solver.colliders = binding` *before* calling this. (The old
    /// `encodeConstraintsAndContacts` works around stale collider counts with
    /// a read-back/re-write of the uniform buffer; that's no longer needed in
    /// the batched path because the controller writes once with the final
    /// count.)
    public func writeIntegrateUniforms(dt: Float) {
        let uPtr = uniformBuffer.contents().bindMemory(to: PBDUniforms.self, capacity: 1)
        uPtr.pointee = PBDUniforms(
            dt: dt, gravity: gravity, damping: damping, floorY: floorY,
            particleCount: UInt32(particleBuffer.count),
            constraintCount: UInt32(constraintBuffer.count),
            floorBand: floorBand,
            colliderCount: UInt32(colliders?.count ?? 0),
            ownerID: ownerID,
            collideStiffness: collideStiffness,
            collideFriction: collideFriction,
            selfRadius: selfRadius,
            velRestitution: velocityRestitution,
            floorRestitution: floorRestitution
        )
    }

    /// Bind buffers + dispatch the integrate kernel into a caller-owned encoder.
    /// The pipeline state is set on each call — redundant `setComputePipelineState`
    /// to the same value is dedupe-cheap on Metal, much cheaper than allocating
    /// a fresh encoder.
    public func dispatchIntegrate(into encoder: MTLComputeCommandEncoder) {
        let count = particleBuffer.count
        guard count > 0 else { return }
        encoder.setComputePipelineState(integratePipeline)
        encoder.setBuffer(particleBuffer.buffer,   offset: 0, index: 0)
        encoder.setBuffer(constraintBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer,           offset: 0, index: 2)
        let w = min(count, integratePipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    /// Fill this solver's lambda buffer with zeros via a caller-owned blit
    /// encoder. XPBD requires λ = 0 at the start of every substep — see the
    /// LAMBDA BUFFER LIFECYCLE block in PBD.metal.
    public func dispatchLambdaReset(into blit: MTLBlitCommandEncoder) {
        guard constraintBuffer.count > 0 else { return }
        let bytes = MemoryLayout<Float>.stride * constraintBuffer.count
        blit.fill(buffer: lambdaBuffer, range: 0..<bytes, value: 0)
    }

    /// Bind buffers + dispatch one constraint colour-group subpass into a
    /// caller-owned encoder. `start`/`count` index into this solver's
    /// `constraintBuffer`; use `constraintLayout` for the per-colour offsets.
    public func dispatchConstraintSubpass(into encoder: MTLComputeCommandEncoder,
                                          start: Int, count: Int) {
        guard count > 0 else { return }
        encoder.setComputePipelineState(constraintPipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        let cByteOffset = start * MemoryLayout<PBDConstraint>.stride
        encoder.setBuffer(constraintBuffer.buffer, offset: cByteOffset, index: 1)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        let lByteOffset = start * MemoryLayout<Float>.stride
        encoder.setBuffer(lambdaBuffer, offset: lByteOffset, index: 3)
        let w = min(count, constraintPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    /// Bind buffers + dispatch one velocity post-solve subpass into a
    /// caller-owned encoder. Mirrors `dispatchConstraintSubpass` but binds the
    /// `velocityPipeline` and skips the lambda buffer (the velocity kernel
    /// reads neither lambda nor restLength — only the constraint endpoints +
    /// normal).
    public func dispatchVelocitySubpass(into encoder: MTLComputeCommandEncoder,
                                        start: Int, count: Int) {
        guard count > 0 else { return }
        encoder.setComputePipelineState(velocityPipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        let cByteOffset = start * MemoryLayout<PBDConstraint>.stride
        encoder.setBuffer(constraintBuffer.buffer, offset: cByteOffset, index: 1)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        let w = min(count, velocityPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    // ── Flat-batched constraint/velocity (PBDTubeBatch flat-solve path) ───────
    // These reuse this solver's compiled `constraintPipeline` / `velocityPipeline`
    // (the kernel is identical regardless of which solver instance owns the
    // pipeline) but bind CALLER-PROVIDED flat buffers: ONE particle buffer and
    // ONE constraint buffer spanning every tube, with constraint indices already
    // remapped to global flat particle indices. One dispatch per colour group
    // replaces the per-tube loop — see PBDTubeBatch.runFlatConstraintColour. The
    // math is identical: within a colour no two constraints share a particle, and
    // distinct tubes never share particles, so the corrections are independent of
    // dispatch granularity.

    /// Pipelines exposed so a batched caller can size threadgroups.
    public var constraintMaxThreads: Int { constraintPipeline.maxTotalThreadsPerThreadgroup }
    public var velocityMaxThreads:   Int { velocityPipeline.maxTotalThreadsPerThreadgroup }

    /// Dispatch one constraint colour group over a flat particle+constraint+lambda
    /// buffer. `constraintByteOffset` / `lambdaByteOffset` select the colour's
    /// contiguous slice; `uniform` carries dt + ringCount(0) + a constraintCount
    /// ≥ `count`.
    public func dispatchConstraintFlat(into enc: MTLComputeCommandEncoder,
                                       particles: MTLBuffer,
                                       constraints: MTLBuffer, constraintByteOffset: Int,
                                       lambda: MTLBuffer, lambdaByteOffset: Int,
                                       uniform: MTLBuffer, count: Int) {
        guard count > 0 else { return }
        enc.setComputePipelineState(constraintPipeline)
        enc.setBuffer(particles,   offset: 0,                    index: 0)
        enc.setBuffer(constraints, offset: constraintByteOffset, index: 1)
        enc.setBuffer(uniform,     offset: 0,                    index: 2)
        enc.setBuffer(lambda,      offset: lambdaByteOffset,     index: 3)
        let w = min(count, constraintPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Dispatch one velocity post-solve colour group over the flat buffers.
    public func dispatchVelocityFlat(into enc: MTLComputeCommandEncoder,
                                     particles: MTLBuffer,
                                     constraints: MTLBuffer, constraintByteOffset: Int,
                                     uniform: MTLBuffer, count: Int) {
        guard count > 0 else { return }
        enc.setComputePipelineState(velocityPipeline)
        enc.setBuffer(particles,   offset: 0,                    index: 0)
        enc.setBuffer(constraints, offset: constraintByteOffset, index: 1)
        enc.setBuffer(uniform,     offset: 0,                    index: 2)
        let w = min(count, velocityPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Bind buffers + dispatch the SDF inter-body collide kernel into a
    /// caller-owned encoder. No-op when there are no colliders or no particles.
    public func dispatchSDFCollide(into encoder: MTLComputeCommandEncoder) {
        guard let binding = colliders, binding.count > 0 else { return }
        let count = particleBuffer.count
        guard count > 0 else { return }
        encoder.setComputePipelineState(collidePipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(binding.buffer,        offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer,         offset: 0, index: 2)
        let w = min(count, collidePipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    /// Copy this solver's particles into a shared flat buffer at `offset`,
    /// stamping `ownerID` into `prevPositionAndPad.w` for the batch SDF kernel.
    public func dispatchCopyToFlat(into encoder: MTLComputeCommandEncoder,
                                   flatBuffer: MTLBuffer,
                                   offset: Int,
                                   pipelines: SimPipelineCache.PBDPipelines) {
        let count = particleBuffer.count
        guard count > 0 else { return }
        var base = UInt32(offset)
        var oid  = ownerID
        encoder.setComputePipelineState(pipelines.copyToFlat)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(flatBuffer,            offset: 0, index: 1)
        encoder.setBytes(&base, length: 4,                  index: 2)
        encoder.setBytes(&oid,  length: 4,                  index: 3)
        let w = min(count, pipelines.copyToFlat.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Copy updated particles back from the flat buffer at `offset` into this
    /// solver's particle buffer, clearing the ownerID stamp in `.w`.
    public func dispatchCopyFromFlat(into encoder: MTLComputeCommandEncoder,
                                     flatBuffer: MTLBuffer,
                                     offset: Int,
                                     pipelines: SimPipelineCache.PBDPipelines) {
        let count = particleBuffer.count
        guard count > 0 else { return }
        var base = UInt32(offset)
        encoder.setComputePipelineState(pipelines.copyFromFlat)
        encoder.setBuffer(flatBuffer,            offset: 0, index: 0)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 1)
        encoder.setBytes(&base, length: 4,                  index: 2)
        let w = min(count, pipelines.copyFromFlat.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Bind buffers + dispatch the particle self-collision kernel into a
    /// caller-owned encoder. No-op unless `selfCollideEnabled` and a positive
    /// radius are set. One dispatch = one full O(n²) self-contact sweep; call
    /// it once per constraint iteration so the distance constraints can react
    /// to the pushed positions on the next iteration (same rationale as the SDF
    /// pass — run inside the loop, not once at the end). Writes its own uniform
    /// buffer each call; cheap shared-memory struct write.
    public func dispatchSelfCollide(into encoder: MTLComputeCommandEncoder,
                                    snapshot: MTLBuffer? = nil) {
        guard selfCollideEnabled, selfCollideRadius > 0 else { return }
        let count = particleBuffer.count
        guard count > 0 else { return }
        let uPtr = selfCollideUniformBuffer.contents()
            .bindMemory(to: PBDSelfCollideUniforms.self, capacity: 1)
        uPtr.pointee = PBDSelfCollideUniforms(
            particleCount: UInt32(count),
            selfRadius:    selfCollideRadius,
            stiffness:     selfCollideStiffness,
            friction:      selfCollideFriction,
            selfSkip:      UInt32(max(1, selfCollideSkip)),
            wallEnabled:   wallEnabled ? 1 : 0,
            wallRadius:    wallRadius,
            wallCx:        wallCenterX,
            wallCz:        wallCenterZ,
            wallBottomY:   wallBottomY
        )
        encoder.setComputePipelineState(selfCollidePipeline)
        encoder.setBuffer(particleBuffer.buffer,       offset: 0, index: 0)
        encoder.setBuffer(selfCollideUniformBuffer,    offset: 0, index: 1)
        // Frozen-read snapshot for symmetric Jacobi contacts. Falls back to the
        // live buffer (old racy behaviour) only if a caller didn't supply one.
        encoder.setBuffer(snapshot ?? particleBuffer.buffer, offset: 0, index: 2)
        let w = min(count, selfCollidePipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    /// Bind buffers + dispatch the floor clamp kernel into a caller-owned encoder.
    public func dispatchFloor(into encoder: MTLComputeCommandEncoder) {
        let count = particleBuffer.count
        guard count > 0 else { return }
        encoder.setComputePipelineState(floorPipeline)
        encoder.setBuffer(particleBuffer.buffer,   offset: 0, index: 0)
        encoder.setBuffer(constraintBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer,           offset: 0, index: 2)
        let w = min(count, floorPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    /// Per-colour-group offsets within this solver's `constraintBuffer`.
    /// Computed from the counts written by `configureChain`; cheap to call but
    /// a batched-encoder caller can cache it across the substep since chain
    /// topology doesn't change inside the constraint loop.
    public struct ConstraintLayout {
        public let primaryEvenStart: Int
        public let primaryEvenCount: Int
        public let primaryOddStart:  Int
        public let primaryOddCount:  Int
        public let bendAStart: Int
        public let bendACount: Int
        public let bendBStart: Int
        public let bendBCount: Int
        public let bendCStart: Int
        public let bendCCount: Int
        public let longRangeStart: Int
        public let longRangeCount: Int
    }

    public var constraintLayout: ConstraintLayout {
        let oddStart   = primaryEvenCount
        let bendAStart = primaryConstraintCount
        let bendBStart = bendAStart + bendACount
        let bendCStart = bendBStart + bendBCount
        let longStart  = bendCStart + bendCCount
        return ConstraintLayout(
            primaryEvenStart: 0,
            primaryEvenCount: primaryEvenCount,
            primaryOddStart:  oddStart,
            primaryOddCount:  primaryConstraintCount - primaryEvenCount,
            bendAStart: bendAStart, bendACount: bendACount,
            bendBStart: bendBStart, bendBCount: bendBCount,
            bendCStart: bendCStart, bendCCount: bendCCount,
            longRangeStart: longStart,
            longRangeCount: constraintBuffer.count - longStart
        )
    }

    // ── Internal dispatch helpers ─────────────────────────────────────────────

    private func encodePass(
        _ commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        count: Int,
        label: String
    ) {
        guard count > 0, let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = label
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(particleBuffer.buffer,   offset: 0, index: 0)
        encoder.setBuffer(constraintBuffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer,            offset: 0, index: 2)

        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeSelfCollidePass(_ commandBuffer: MTLCommandBuffer) {
        let count = particleBuffer.count
        guard count > 0 else { return }
        // JACOBI double-buffer: copy current positions into a frozen snapshot so
        // every thread reads its neighbours from the SAME pre-pass state. Without
        // this the kernel read live, mutating neighbours → asymmetric contact
        // pushes → the pile buzzed (neon heatmap diagnosis, 2026-06-06). The blit
        // and the compute dispatch are separate encoders in this command buffer,
        // so Metal orders the copy strictly before the kernel reads it.
        let bytes = count * MemoryLayout<PBDParticle>.stride
        if (selfCollideSnapshot?.length ?? 0) < bytes {
            selfCollideSnapshot = device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let snapshot = selfCollideSnapshot else { return }
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "PBD.selfCollideSnapshot"
            blit.copy(from: particleBuffer.buffer, sourceOffset: 0,
                      to: snapshot, destinationOffset: 0, size: bytes)
            blit.endEncoding()
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBD.selfCollide"
        dispatchSelfCollide(into: encoder, snapshot: snapshot)
        encoder.endEncoding()
    }

    private func encodeCollidePass(_ commandBuffer: MTLCommandBuffer,
                                   binding: ColliderBinding) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBD.sdfCollide"
        encoder.setComputePipelineState(collidePipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(binding.buffer,        offset: 0, index: 1)
        encoder.setBuffer(uniformBuffer,         offset: 0, index: 2)
        let count = particleBuffer.count
        let w = min(count, collidePipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeVelocitySubpass(
        _ commandBuffer: MTLCommandBuffer,
        start: Int, count: Int, label: String
    ) {
        guard count > 0, let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = label
        encoder.setComputePipelineState(velocityPipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        // The constraint buffer is offset by `start` × stride so the kernel's
        // `constraints[id]` indexes into this colour group's slice. Mirrors
        // encodeConstraintSubpass — dispatched thread count `count` already
        // bounds the kernel; u.constraintCount stays at the global total and
        // its `id >= u.constraintCount` guard is satisfied trivially.
        let cByteOffset = start * MemoryLayout<PBDConstraint>.stride
        encoder.setBuffer(constraintBuffer.buffer, offset: cByteOffset, index: 1)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        let w = min(count, velocityPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encodeConstraintSubpass(
        _ commandBuffer: MTLCommandBuffer,
        start: Int, count: Int, label: String
    ) {
        guard count > 0, let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = label
        encoder.setComputePipelineState(constraintPipeline)
        encoder.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        let cByteOffset = start * MemoryLayout<PBDConstraint>.stride
        encoder.setBuffer(constraintBuffer.buffer, offset: cByteOffset, index: 1)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        // Lambda buffer slice must mirror the constraint buffer's slice — the
        // kernel sees lambda[id] alongside constraint[id], so both need to be
        // offset by `start` for the per-colour-group dispatches.
        let lByteOffset = start * MemoryLayout<Float>.stride
        encoder.setBuffer(lambdaBuffer, offset: lByteOffset, index: 3)
        let w = min(count, constraintPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}
