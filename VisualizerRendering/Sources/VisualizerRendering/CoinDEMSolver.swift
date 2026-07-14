import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── CoinDEMSolver ─────────────────────────────────────────────────────────────
//
// GPU rigid-body solver for a coin-pusher pile. The first non-soft-body, non-
// fluid solver in the project — see CoinDEM.metal for the contact model (full
// ORIENTED rigid body: position + quaternion + linear & angular velocity, real
// disk inertia, analytic oriented-cylinder SAT contacts resolved at contact
// points so they carry torque, plus a gentle support-leveling pass) and the
// per-substep kernel order.
//
// Mirrors the PBDSolver conventions: `init?` pulls memoised pipelines from the
// shared SimEngine, `SimBuffer<T>` for CPU-readable storage, `.storageModePrivate`
// for GPU-only scratch (hash + delta), and a single-command-buffer `encode(to:)`
// with NO per-substep `waitUntilCompleted`.
//
// ALIGNMENT RULE (PBDSolver.swift:12): every shared struct below uses only
// SIMD4 lanes; the uniform struct is scalars-only (like PBDUniforms).

// One coin's rigid state. 96 bytes (6 × float4). Matches `CoinBody` in CoinDEM.metal.
// Full oriented rigid body: position + quaternion + linear & angular velocity.
//
// TODO(general-rigid-body): the integrator, oriented-disk inertia, torque-bearing
// contacts, and GPU plumbing here generalize to arbitrary convex rigid bodies —
// the only thing pinning this to coins is the DYNAMIC collider shape (a capped
// cylinder defined by `prevPos.w` = radius, `vel.w` = half-thickness). To get a
// generic "drop a cube and let it tip onto a face" solver outside SceneKit:
//   1. add a shape tag + half-extents to CoinBody (spare SIMD4 lanes exist), and
//   2. add box–box / box–plane SAT to the contact kernel (see CoinDEM.metal).
// Static colliders are already boxes + planes, so box-vs-environment is half done.
public struct CoinBody {
    public var posInvMass: SIMD4<Float>   // xyz = COM, w = invMass (0 = inactive slot)
    public var prevPos:    SIMD4<Float>   // xyz = previous COM (scratch), w = collider RADIUS (bounding for a box)
    public var vel:        SIMD4<Float>   // xyz = linear velocity, w = collider HALF-THICKNESS (min half-extent for a box)
    public var orient:     SIMD4<Float>   // physical orientation quaternion (x,y,z,w)
    public var prevOrient: SIMD4<Float>   // orientation at substep start (for deriving ω)
    public var angVel:     SIMD4<Float>   // xyz = angular velocity (world, rad/s), w = support flag
    // Shape: w = tag (0 = disc / capped cylinder, the default; 1 = box; 2 = sphere).
    // For a box, xyz = the three half-extents; for a sphere, x = radius. Appended
    // last so every existing field keeps its offset — existing kernels and CPU
    // readbacks are byte-identical for discs.
    public var shapeExtents: SIMD4<Float>

    public init(position: SIMD3<Float>,
                invMass: Float = 1.0,
                velocity: SIMD3<Float> = .zero,
                orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                angVel: SIMD3<Float> = .zero,
                shapeExtents: SIMD4<Float> = SIMD4(0, 0, 0, 0)) {
        self.posInvMass = SIMD4(position, invMass)
        self.prevPos    = SIMD4(position, 0)
        self.vel        = SIMD4(velocity, 0)
        self.orient     = orient
        self.prevOrient = orient
        self.angVel     = SIMD4(angVel, 0)
        self.shapeExtents = shapeExtents
    }

    public static var inactive: CoinBody {
        CoinBody(position: SIMD3(0, -100_000, 0), invMass: 0)
    }
}

// Static (or kinematic) environment collider. 80 bytes. Matches CoinDEM.metal.
public struct CoinStaticCollider {
    public var a:      SIMD4<Float>   // plane: xyz=normal, w=tag(0) ; box: xyz=centre, w=tag(1)
    public var b:      SIMD4<Float>   // plane: w=offset d ; box: xyz=halfExtents
    public var vel:    SIMD4<Float>   // xyz = kinematic surface velocity, w = friction (reserved)
    public var meta:   SIMD4<UInt32>  // x = flags (bit0 = one-way ledge)
    public var orient: SIMD4<Float>   // oriented-box (kind 3) quaternion (x,y,z,w); identity otherwise

    enum Kind: UInt32 { case plane = 0, box = 1, pusherPlate = 2, orientedBox = 3, cylinder = 4 }

    private static let identityQuat = SIMD4<Float>(0, 0, 0, 1)

    /// A CYLINDER-interior segment — the marble is constrained INSIDE a cylinder of
    /// `radius` whose axis passes through `center` along `axis` for ±`halfLength`.
    /// `lowerHalfOnly` makes it a half-pipe (only the lower hemisphere relative to
    /// `up` pushes; the top is open). Smooth curved surface → no corners to stick on.
    /// Resolved on the constraint path's sphere case (CoinDEM.metal kind 4).
    public static func cylinder(center: SIMD3<Float>, axis: SIMD3<Float>, radius: Float,
                                up: SIMD3<Float>, halfLength: Float,
                                lowerHalfOnly: Bool) -> CoinStaticCollider {
        CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.cylinder.rawValue)),
            b: SIMD4(simd_normalize(axis), radius),
            vel: SIMD4(simd_normalize(up), halfLength),
            meta: SIMD4(lowerHalfOnly ? 1 : 0, 0, 0, 0),
            orient: identityQuat)
    }

    /// A reciprocating pusher plate that only ever pushes coins FORWARD (+Z),
    /// never down/up — so coins at the paddle/shelf corner can't be squeezed into
    /// the floor and ejected. `center`/`halfExtents` describe the solid plate;
    /// coins within its X/Y footprint that the plate has overtaken are pushed to
    /// just in front of its +Z face — but no faster than `velocity.z` (the plate's
    /// true forward speed), so the contact can't launch coins (see CoinDEM.metal).
    public static func pusherPlate(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                                   velocity: SIMD3<Float> = .zero) -> CoinStaticCollider {
        CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.pusherPlate.rawValue)),
            b: SIMD4(halfExtents, 0),
            vel: SIMD4(velocity, 0), meta: SIMD4(0, 0, 0, 0),
            orient: identityQuat)
    }

    /// Half-space `n · x ≥ d`. Coins are pushed back to the surface.
    public static func plane(normal: SIMD3<Float>, offset: Float) -> CoinStaticCollider {
        let n = simd_normalize(normal)
        return CoinStaticCollider(
            a: SIMD4(n, Float(bitPattern: Kind.plane.rawValue)),
            b: SIMD4(0, 0, 0, offset),
            vel: .zero, meta: SIMD4(0, 0, 0, 0),
            orient: identityQuat)
    }

    /// Axis-aligned box. `oneWay` lets coins fall off the top but not be shoved
    /// back up (the overhang lip). `velocity` is the kinematic surface velocity
    /// (the reciprocating pusher) — but note the shove is mostly implicit via the
    /// box's per-frame advance (see CoinDEM.metal).
    public static func box(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                           oneWay: Bool = false,
                           velocity: SIMD3<Float> = .zero) -> CoinStaticCollider {
        CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.box.rawValue)),
            b: SIMD4(halfExtents, 0),
            vel: SIMD4(velocity, 0),
            meta: SIMD4(oneWay ? 1 : 0, 0, 0, 0),
            orient: identityQuat)
    }

    /// An ORIENTED box — a box rotated by `orientation` (local→world). The local
    /// axes are X/Y/Z with the given `halfExtents`. Resolved on the constraint
    /// path only (the path the ball/plank scenes use); proper sphere-vs-OBB
    /// contact for marbles rolling on a tilted plank. See CoinDEM.metal kind 3.
    public static func orientedBox(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                                   orientation: simd_quatf) -> CoinStaticCollider {
        let q = orientation.normalized
        return CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.orientedBox.rawValue)),
            b: SIMD4(halfExtents, 0),
            vel: .zero, meta: SIMD4(0, 0, 0, 0),
            orient: SIMD4(q.imag.x, q.imag.y, q.imag.z, q.real))
    }
}

// 3×4 transform per coin, consumed by CoinInstancedRenderer. Matches CoinDEM.metal.
public struct CoinTransform {
    public var col0: SIMD4<Float>
    public var col1: SIMD4<Float>
    public var col2: SIMD4<Float>
    public var col3: SIMD4<Float>
}

// One contact-constraint (a manifold point) — the constraint solver's currency.
// 112 bytes (7 × {u,f}16). Mirrors `CoinContact` in CoinDEM.metal exactly.
public struct CoinContact {
    public var meta: SIMD4<UInt32>   // x=A, y=B (0xFFFFFFFF=static), z=collider|feature, w=pairKey
    public var nrm:  SIMD4<Float>    // xyz=normal (B→A), w=depth
    public var rA:   SIMD4<Float>    // xyz=cp−comA, w=normalImpulse
    public var rB:   SIMD4<Float>    // xyz=cp−comB, w=tangent1Impulse
    public var tan1: SIMD4<Float>    // xyz=tangent1, w=tangent2Impulse
    public var tan2: SIMD4<Float>    // xyz=tangent2, w=colour
    public var aux:  SIMD4<Float>    // x=pre-solve approach vn₀ (restitution target), w=captured flag
}

// One generic joint (constraint path). 80 bytes (5 × {u,f}16). Mirrors `CoinJoint`
// in CoinDEM.metal exactly. Built via the addBallJoint / addHingeJoint /
// addDistanceJoint APIs — not constructed by hand.
public struct CoinJoint {
    public var meta:    SIMD4<UInt32>  // x=type(0 ball,1 hinge,2 distance), y=A, z=B(0xFFFFFFFF=world), w=enabled
    public var anchorA: SIMD4<Float>   // xyz=A-local anchor, w=rest length (distance)
    public var anchorB: SIMD4<Float>   // xyz=B-local anchor (WORLD if z==world)
    public var axisA:   SIMD4<Float>   // xyz=A-local hinge axis, w=limit lo (rad)
    public var axisB:   SIMD4<Float>   // xyz=B-local hinge axis (WORLD if world), w=limit hi
}

// Per-substep uniforms. Scalars only (no float3) — alignment-safe. 132 bytes (33 × 4).
struct CoinUniforms {
    var dt: Float = 1.0 / 240.0
    var gravity: Float = 9.8
    var linDamping: Float = 0.999
    var coinRadius: Float = 0.05
    var halfThickness: Float = 0.0037
    var contactRelax: Float = 1.0
    var friction: Float = 0.82
    var restitution: Float = 0.0
    var frictionCoeff: Float = 0.0       // Coulomb μ for positional friction-with-torque (0 = off)
    var rollingResistance: Float = 0.0   // resists rolling spin at a contact (0 = off)
    var floorY: Float = 0.0
    var sleepLinVel: Float = 0.01
    var angFriction: Float = 0.86      // angular velocity retention on contact (was settleStrength)
    var angDamping: Float = 0.999      // per-substep angular velocity retention (was settleDamp)
    var gridMinX: Float = 0
    var gridMinY: Float = 0
    var gridMinZ: Float = 0
    var invCell: Float = 1
    var coinCount: UInt32 = 0
    var colliderCount: UInt32 = 0
    var gridResX: UInt32 = 1
    var gridResY: UInt32 = 1
    var gridResZ: UInt32 = 1
    var maxHSpeed: Float = 1.2     // coin-pusher default; raised for fast-ballistic scenes
    var maxSpeed: Float = 6.0
    var maxOmega: Float = 11.0
    var contactSlop: Float = 0.002      // constraint solver: allowed penetration (m)
    var baumgarteBeta: Float = 0.2      // constraint solver: position-bias gain
    var restThreshold: Float = 0.5      // constraint solver: restitution approach-speed gate (m/s)
    var restitutionVelFalloff: Float = 0.0  // COR drop per m/s impact speed (0 = constant COR)
    var restitutionMinE: Float = 0.0        // floor for the velocity-faded COR
    var quadraticDrag: Float = 0.0          // ∝v² aerodynamic drag (accel = −k·|v|·v); 0 = off
    var dragRefRadius: Float = 0.0          // radius where quadraticDrag is calibrated; drag ∝1/r per body. 0 = flat k
    var speculativeMargin: Float = 0.0      // emit near-contacts within this gap (anti-tunneling); 0 = off
}

@MainActor
public final class CoinDEMSolver: PenetrationProbing {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "CoinDEMSolver")

    // ── GPU plumbing ──────────────────────────────────────────────────────────
    public let engine: SimEngine
    public var device: MTLDevice { engine.device }

    private let integratePipeline:  MTLComputePipelineState
    private let cellClearPipeline:  MTLComputePipelineState
    private let cellCountPipeline:  MTLComputePipelineState
    private let scanPipeline:       MTLComputePipelineState
    private let scatterPipeline:    MTLComputePipelineState
    private let contactPipeline:    MTLComputePipelineState
    private let applyPipeline:      MTLComputePipelineState
    private let finalizePipeline:   MTLComputePipelineState
    private let orientPipeline:     MTLComputePipelineState
    private let transformPipeline:  MTLComputePipelineState
    private let jointPipeline:      MTLComputePipelineState
    private let measurePipeline:    MTLComputePipelineState   // coinMeasurePenetration (diagnostic)
    private let generatePipeline:   MTLComputePipelineState   // coinGenerateContacts (constraint solver)
    private let clearBodyPipeline:  MTLComputePipelineState
    private let buildBodyPipeline:  MTLComputePipelineState
    private let colorRoundPipeline: MTLComputePipelineState
    private let intVelCSPipeline:   MTLComputePipelineState
    private let solveVelCSPipeline: MTLComputePipelineState
    private let intPosCSPipeline:   MTLComputePipelineState
    private let finalizeCSPipeline: MTLComputePipelineState
    private let clearHashPipeline:  MTLComputePipelineState
    private let snapshotPipeline:   MTLComputePipelineState
    private let warmMatchPipeline:  MTLComputePipelineState
    private let warmApplyPipeline:  MTLComputePipelineState
    private let islandInitPipeline: MTLComputePipelineState
    private let islandUnionPipeline: MTLComputePipelineState
    private let islandJumpPipeline: MTLComputePipelineState
    private let sleepTickPipeline:  MTLComputePipelineState
    private let islandMinPipeline:  MTLComputePipelineState
    private let sleepMarkPipeline:  MTLComputePipelineState
    private let gjkEPAPipeline:     MTLComputePipelineState
    private let writeSolveArgsPipeline: MTLComputePipelineState
    private let jointSolveCSPipeline: MTLComputePipelineState
    private let islandUnionJointsPipeline: MTLComputePipelineState
    private let solveArgsBuffer: MTLBuffer         // MTLDispatchThreadgroupsIndirectArguments (3×UInt32)
    private let solveTGSizeBuffer: MTLBuffer       // 1×UInt32 (threadgroup size for the indirect sizing)
    private static let solveTGSize = 64

    // ── Storage ───────────────────────────────────────────────────────────────
    public let coinBuffer: SimBuffer<CoinBody>     // shared — CPU reads for cull
    public let transformBuffer: MTLBuffer          // CoinTransform per coin (render input)
    /// Per-body asset type (UInt32 per slot). A MIXED pile holds coins/jewels/franks
    /// in one solver; each asset's renderer expands only its own type (the rest are
    /// parked). Written at spawn, read by `coin_expand_instances`.
    public let bodyTypeBuffer: MTLBuffer
    /// Per-body articulation link (SIMD2<Int32> per slot): x = partner slot (−1 =
    /// none), y = which of this body's ends joins (±1 → ±Y). Drives `coinJointSolve`
    /// so a frankfurter's two segments bend at their joint. Default (−1, 0).
    private let linkBuffer: MTLBuffer
    /// Per-body material (SIMD2<Float> per slot): x = Coulomb friction μ, y =
    /// restitution e. A NEGATIVE lane means "inherit the global uniform"
    /// (`frictionCoeff` / `restitution`) — the default, so untouched scenes are
    /// unchanged. Constraint path only; combined per contact as μ=√(μA·μB),
    /// e=max(eA,eB). Set at spawn (`friction:`/`restitution:`) or via `setMaterial`.
    public let materialBuffer: MTLBuffer
    private let colliderBuffer: SimBuffer<CoinStaticCollider>
    private let coinDeltaBuffer: MTLBuffer         // private — per-coin Jacobi delta
    private let cellCounts: MTLBuffer              // private
    private let cellOffsets: MTLBuffer             // private
    private let sortedIndices: MTLBuffer           // private
    private let uniformBuffer: MTLBuffer
    // Diagnostic readback for coinMeasurePenetration: [0]=maxDepth µm, [1]=pairCount.
    private let penetrationResult: MTLBuffer       // shared, 2×UInt32
    private let penetrationThreshold: MTLBuffer    // shared, 1×Float

    // ── Constraint solver storage (Stage 1+) ──────────────────────────────────
    /// The per-substep contact buffer: `coinGenerateContacts` bump-appends a
    /// `CoinContact` per manifold point; the colour / velocity / position passes
    /// consume it. Shared so the host can read the count + warm-start.
    public let contactBuffer: MTLBuffer            // CoinContact[maxContacts]
    public let contactCountBuffer: MTLBuffer       // 1×UInt32 (atomic append cursor)
    private let maxContactsBuffer: MTLBuffer       // 1×UInt32 (constant)
    /// Per-body contact-index lists (maxCoins × 48) + per-body count — built each
    /// substep so the colouring pass can find a contact's neighbours cheaply.
    private let bodyContactsBuffer: MTLBuffer      // maxCoins × CD_MAX_BODY_CONTACTS UInt32
    private let bodyContactCountBuffer: MTLBuffer  // maxCoins UInt32
    private static let maxBodyContacts = 48
    /// Per-body split-impulse BIAS (pseudo) velocity: 2 × float4 per body
    /// (linear, angular). Cleared each substep; moves position then is discarded.
    private let biasBuffer: MTLBuffer              // maxCoins × 2 × SIMD4<Float>
    /// Warm starting: last substep's solved contacts + an open-addressing hash
    /// (pairKey → prev slot) so a re-found contact resumes from its converged impulse.
    private let prevContactBuffer: MTLBuffer       // CoinContact[maxContacts]
    private let pairHashBuffer: MTLBuffer          // UInt32[hashSize] (power of two)
    private let hashSizeBuffer: MTLBuffer          // 1×UInt32 (constant)
    private let hashSize: Int
    /// Island sleeping: connected-component label + per-island min sleep-timer
    /// (scratch, recomputed each frame), a persistent per-body slow-frame counter,
    /// and the asleep flag the solver kernels gate on (shared → host can read the count).
    private let islandLabelBuffer: MTLBuffer       // maxCoins UInt32
    private let islandMinBuffer: MTLBuffer         // maxCoins UInt32
    private let sleepTimerBuffer: MTLBuffer        // maxCoins UInt32 (persists)
    public let asleepBuffer: MTLBuffer             // maxCoins UInt32 (0/1)
    /// Capacity of `contactBuffer`. A dense mixed pile emits up to ~tens of contacts
    /// per body (box manifolds + statics), so size generously (maxCoins × 64).
    public let maxContacts: Int

    // ── Generic joints (constraint path) ──────────────────────────────────────
    /// Joint slots (CoinJoint each; meta.w == 0 = disabled/free). Solved serially
    /// (true Gauss-Seidel) once per velocity iteration, interleaved with the
    /// contact colours; joint edges union into the sleep islands so an articulated
    /// assembly sleeps and wakes as one.
    public let jointBuffer: MTLBuffer
    public static let maxJoints = 1024
    private var freeJointSlots: [Int] = []
    private var jointHighWater: Int = 0
    /// Number of joint slots the solve loops over (disabled slots are skipped
    /// in-kernel). For tests / instrumentation.
    public var jointCount: Int { jointHighWater }
    /// Active (enabled) joints.
    public var activeJointCount: Int {
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        var n = 0
        for i in 0..<jointHighWater where j[i].meta.w != 0 { n += 1 }
        return n
    }

    public let maxCoins: Int

    // ── Tunables (defaults from the Coin Pusher research) ─────────────────────
    public var gravity: Float = 9.8
    public var linDamping: Float = 0.999      // light: keep impacts/jostle lively
    // Friction RETENTION (PBD convention: lower = stickier). Coins are slippery
    // flat metal — high retention lets a coin that lands tilted SHEAR across its
    // neighbours and settle at a calm, shallow angle instead of locking propped.
    public var friction: Float = 0.94         // tangential linear retention — slippery, coins shear flat
    public var restitution: Float = 0.12      // bounce on a real (falling) impact; low so landings don't prop
    /// Coulomb friction coefficient μ for the POSITIONAL friction-with-torque pass —
    /// a tangential correction applied AT the contact point (so it carries torque:
    /// a body sliding on a contact picks up roll, and a heap's angle of repose is set
    /// by μ, not just geometry). Bounded by μ·(normal correction) per the friction
    /// cone. Default 0 (OFF): coins are deliberately slippery flat metal and settle
    /// by shearing flat, and the `friction` velocity-retention damp above already
    /// gives them their light tangential drag — turning μ up makes them grip and pile
    /// steeper. Scenes that want real grip (rolling balls, drifting chips) opt in.
    public var frictionCoeff: Float = 0.0
    /// Rolling resistance: a small torque opposing the spin of a body rolling on a
    /// contact, so a coin on its edge slows and stops instead of rolling forever under
    /// the near-frictionless global angular damping. Default 0 (off).
    public var rollingResistance: Float = 0.0
    public var contactRelax: Float = 0.7      // Jacobi relaxation on the AVERAGED per-coin correction
    public var sleepLinVel: Float = 0.03      // sleep a slow CONTACTING coin so a settled heap goes quiet (no micro-jitter)
    public var angFriction: Float = 0.78      // contact angular retention → spin bleeds off fast (no whirling pile)
    public var angDamping: Float = 0.999      // per-substep angular velocity retention
    /// Optional support-leveling assist (coinOrient). OFF — and it must stay off for
    /// this scene. It was added to flatten coins, but A/B with the instability
    /// tracker showed it does the OPPOSITE on a real shallow pile: perpetually
    /// rotating resting coins a hair into their neighbours kept ~150 of 600 coins
    /// twitching (asleep 346→548 and micro-jitter 194→41 when disabled) AND, because
    /// they never slept, left them frozen mid-twitch at random tilts. With it off the
    /// slippery coins settle FLAT on their own on the wide shelves (settled-tilt
    /// mean 3.6°, pile KE 0.9→0.001). Kept (gated) only for a hypothetical deeply-
    /// jammed bin where coins can't flatten passively — not this scene.
    public var levelingEnabled: Bool = false
    public var floorY: Float = 0.0

    // ── Velocity caps (finalize) ──────────────────────────────────────────────
    // Coin-pusher defaults: a pusher has no legitimate fast lateral motion, so a
    // dense pile shoved out of overlap is clamped hard. A fast-BALLISTIC scene
    // (Tennis Ball Painter) raises these so a thrown ball flies across the play
    // volume and rebounds off the wall instead of being clamped to a crawl.
    public var maxHSpeed: Float = 1.2     // horizontal de-penetration speed cap (m/s)
    public var maxSpeed: Float = 6.0      // global linear speed cap (m/s)
    public var maxOmega: Float = 11.0     // angular speed cap (rad/s)

    // ── Constraint solver (Stage 3+) ──────────────────────────────────────────
    /// Which solver runs in `encode(to:wallDt:)`: the legacy Jacobi position solver
    /// (default, proven for all shipping scenes) or the new constraint solver
    /// (graph-colored sequential impulse + split-impulse — opt-in per scene).
    public enum SolverMode { case legacy, constraint }
    public var solverMode: SolverMode = .legacy
    public var contactSlop: Float = 0.002       // allowed penetration before recovery (m)
    public var baumgarteBeta: Float = 0.2       // split-impulse position-bias gain
    public var restThreshold: Float = 0.5       // restitution approach-speed gate (m/s)
    public var velocityIterations: Int = 8      // sequential-impulse passes per substep

    // ── Realistic-bounce extensions (opt-in; all 0 ⇒ legacy constant-COR/linear) ──
    /// Coefficient-of-restitution falloff per m/s of impact speed. Real materials —
    /// a fuzzy tennis ball especially (Cross 2002) — rebound with a smaller fraction
    /// of their speed as the hit gets harder. `eEff = clamp(restitution − falloff·v,
    /// restitutionMinE, restitution)`. 0 (default) ⇒ constant COR, exactly as before.
    public var restitutionVelFalloff: Float = 0.0
    /// Lower bound for the velocity-faded COR (only consulted when falloff > 0).
    public var restitutionMinE: Float = 0.0
    /// ∝v² aerodynamic drag coefficient k (acceleration = −k·|v|·v, units 1/m, so
    /// terminal speed ≈ √(g/k)). 0 (default) ⇒ off. A scene using this sets
    /// `linDamping` ≈ 1 so airborne energy is shed only by this physical term.
    public var quadraticDrag: Float = 0.0
    /// Radius at which `quadraticDrag` is calibrated. When > 0, each body's drag is
    /// scaled by `dragRefRadius / bodyRadius` (physical A/m ∝ 1/r) — a bigger ball
    /// drags LESS and flies further, a smaller one drags MORE. 0 (default) ⇒ flat k
    /// for every body (no size scaling).
    public var dragRefRadius: Float = 0.0
    /// Speculative-contact margin (metres, constraint path, default 0 = off).
    /// When > 0, sphere/capsule pairs and plane/cylinder statics also emit
    /// NEAR-contacts within this gap (negative depth); the normal constraint then
    /// only limits the approach speed to gap/dt — the standard anti-tunneling
    /// scheme for small fast bodies. Keep ≤ the broadphase cell size (pairs
    /// beyond one cell are never even found).
    public var speculativeMargin: Float = 0.0
    /// Colour passes per velocity iteration. The colouring uses ≤ this many colours in
    /// practice (≈8 for a dense pile); contacts in higher colours wait a substep.
    public var solveColors: Int = 16
    /// Warm starting: carry each contact's converged impulse across substeps so a
    /// resting stack resumes from its solution instead of re-solving from zero.
    /// OPT-IN (default off): it's the biggest convergence win for TALL single-column
    /// stacks, but on a dense MIXED heap the colored Gauss-Seidel already settles dead-
    /// still on its own, and a stale warm seed (a contact whose feature shifted between
    /// substeps) can occasionally leave the heap un-settled. Scenes that build towers
    /// opt in; the loose-heap scenes (Pile of Mess) leave it off — GS alone is enough.
    public var warmStart: Bool = false
    private var needsHashClear = true           // lazily zero the pair hash on first use
    /// Island sleeping: freeze a connected island once every body in it has been slow
    /// for `sleepFrames`, so a settled heap skips the whole solve. On by default for the
    /// constraint path (it's pure perf + the per-island gate prevents premature freezing).
    public var sleepEnabled: Bool = true
    public var sleepFrames: UInt32 = 30         // ~0.5 s of stillness before an island sleeps
    /// True if the last `encode(to:)` skipped all solver work (the whole pile was
    /// frozen). For tests / perf instrumentation.
    public private(set) var didSkipLastFrame = false
    public var islandUnionRounds: Int = 12      // connected-components convergence rounds/frame

    // ── Geometry / grid ───────────────────────────────────────────────────────
    public let coinRadius: Float
    public let halfThickness: Float
    private let gridMin: SIMD3<Float>
    private let gridRes: SIMD3<UInt32>
    private let cellSize: Float
    private var numCells: Int { Int(gridRes.x) * Int(gridRes.y) * Int(gridRes.z) }
    /// Spatial-hash grid dimensions + total cell count — exposed for perf
    /// instrumentation (the prefix-sum scan cost scales with cell count).
    public var gridCellCount: Int { numCells }
    public var gridResolution: SIMD3<UInt32> { gridRes }

    // ── Substepping ───────────────────────────────────────────────────────────
    public var fixedDt: Float = 1.0 / 240.0
    public var maxSubsteps: Int = 6
    public var iterations: Int = 16           // contact passes per substep
    public var accumulator: Float = 0

    // ── Slot lifecycle ────────────────────────────────────────────────────────
    private var freeSlots: [Int] = []
    private var nextSlot: Int = 0
    private(set) public var highWater: Int = 0
    public var activeCount: Int { highWater - freeSlots.count }
    /// Substeps encoded on the LAST `encode(...)` call — for instrumentation /
    /// slow-motion detection (`steps × fixedDt / wallDt` ≈ realtime ratio).
    private(set) public var lastStepCount: Int = 0

    private var colliders: [CoinStaticCollider] = []

    // ── Init ──────────────────────────────────────────────────────────────────

    struct Pipelines {
        let integrate, cellClear, cellCount, scan, scatter: MTLComputePipelineState
        let contact, apply, finalize, orient, transform: MTLComputePipelineState
        let joint: MTLComputePipelineState
        let measure: MTLComputePipelineState   // coinMeasurePenetration (diagnostic)
        let generate: MTLComputePipelineState  // coinGenerateContacts (constraint solver)
        let clearBody, buildBody, colorRound: MTLComputePipelineState   // graph colouring
        let intVelCS, solveVelCS, intPosCS, finalizeCS: MTLComputePipelineState  // sequential impulse
        let clearHash, snapshot, warmMatch, warmApply: MTLComputePipelineState   // warm starting
        let islandInit, islandUnion, islandJump: MTLComputePipelineState          // island sleeping
        let sleepTick, islandMinReduce, sleepMark: MTLComputePipelineState
        let gjkEPA: MTLComputePipelineState                                        // GJK/EPA probe
        let writeSolveArgs: MTLComputePipelineState                                // indirect-dispatch sizing
        let jointSolveCS, islandUnionJoints: MTLComputePipelineState               // generic joints
    }

    /// Resolve the kernels through an arbitrary lookup (engine cache in production; a
    /// runtime-compiled library in the headless tests).
    static func makePipelines(_ resolve: (String) -> MTLComputePipelineState?) -> Pipelines? {
        guard
            let p0 = resolve("coinIntegrate"),
            let p1 = resolve("coinCellClear"),
            let p2 = resolve("coinCellCount"),
            let p3 = resolve("coinCellOffsetsScan"),
            let p4 = resolve("coinScatter"),
            let p5 = resolve("coinContactSolve"),
            let p6 = resolve("coinApplyDelta"),
            let p7 = resolve("coinFinalize"),
            let p8 = resolve("coinOrient"),
            let p9 = resolve("coinDeriveTransforms"),
            let p10 = resolve("coinJointSolve"),
            let p11 = resolve("coinMeasurePenetration"),
            let p12 = resolve("coinGenerateContacts"),
            let p13 = resolve("coinClearBodyContacts"),
            let p14 = resolve("coinBuildBodyContacts"),
            let p15 = resolve("coinColorRound"),
            let p16 = resolve("coinIntegrateVelocityCS"),
            let p17 = resolve("coinSolveVelocityColor"),
            let p18 = resolve("coinIntegratePositionCS"),
            let p19 = resolve("coinFinalizeCS"),
            let p20 = resolve("coinClearHash"),
            let p21 = resolve("coinSnapshotContacts"),
            let p22 = resolve("coinWarmStartMatch"),
            let p23 = resolve("coinWarmStartApply"),
            let p24 = resolve("coinIslandInit"),
            let p25 = resolve("coinIslandUnion"),
            let p26 = resolve("coinIslandJump"),
            let p27 = resolve("coinSleepTick"),
            let p28 = resolve("coinIslandMinReduce"),
            let p29 = resolve("coinSleepMark"),
            let p30 = resolve("coinGJKEPAProbe"),
            let p31 = resolve("coinWriteSolveArgs"),
            let p32 = resolve("coinJointSolveCS"),
            let p33 = resolve("coinIslandUnionJoints")
        else { return nil }
        return Pipelines(integrate: p0, cellClear: p1, cellCount: p2, scan: p3, scatter: p4,
                         contact: p5, apply: p6, finalize: p7, orient: p8, transform: p9, joint: p10,
                         measure: p11, generate: p12, clearBody: p13, buildBody: p14, colorRound: p15,
                         intVelCS: p16, solveVelCS: p17, intPosCS: p18, finalizeCS: p19,
                         clearHash: p20, snapshot: p21, warmMatch: p22, warmApply: p23,
                         islandInit: p24, islandUnion: p25, islandJump: p26,
                         sleepTick: p27, islandMinReduce: p28, sleepMark: p29, gjkEPA: p30,
                         writeSolveArgs: p31, jointSolveCS: p32, islandUnionJoints: p33)
    }

    /// Production init: pipelines come from the engine's memoised cache (the
    /// package's compiled metallib via `Bundle.module`).
    public convenience init?(engine: SimEngine = .shared,
                             maxCoins: Int,
                             coinRadius: Float,
                             halfThickness: Float,
                             boundsMin: SIMD3<Float>,
                             boundsMax: SIMD3<Float>) {
        guard let pipelines = CoinDEMSolver.makePipelines({ engine.pipeline($0) }) else {
            CoinDEMSolver.log.error("Coin pipeline cache failed — check CoinDEM.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        self.init(engine: engine, pipelines: pipelines, maxCoins: maxCoins,
                  coinRadius: coinRadius, halfThickness: halfThickness,
                  boundsMin: boundsMin, boundsMax: boundsMax)
    }

    /// Test seam: pipelines built from a runtime-compiled library (the SwiftPM
    /// CLI doesn't produce a metallib, so the headless tests compile CoinDEM.metal
    /// from source and inject the library here).
    convenience init?(engine: SimEngine, library: MTLLibrary,
                      maxCoins: Int, coinRadius: Float, halfThickness: Float,
                      boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        let dev = engine.device
        guard let pipelines = CoinDEMSolver.makePipelines({ name in
            guard let fn = library.makeFunction(name: name) else { return nil }
            return try? dev.makeComputePipelineState(function: fn)  // gpu-ok: test-time pipeline from runtime-compiled library
        }) else { return nil }
        self.init(engine: engine, pipelines: pipelines, maxCoins: maxCoins,
                  coinRadius: coinRadius, halfThickness: halfThickness,
                  boundsMin: boundsMin, boundsMax: boundsMax)
    }

    init?(engine: SimEngine,
          pipelines: Pipelines,
          maxCoins: Int,
          coinRadius: Float,
          halfThickness: Float,
          boundsMin: SIMD3<Float>,
          boundsMax: SIMD3<Float>) {
        let p0 = pipelines.integrate, p1 = pipelines.cellClear, p2 = pipelines.cellCount
        let p3 = pipelines.scan, p4 = pipelines.scatter, p5 = pipelines.contact
        let p6 = pipelines.apply, p7 = pipelines.finalize, p8 = pipelines.orient
        let p9 = pipelines.transform, p10 = pipelines.joint, p11 = pipelines.measure
        let p12 = pipelines.generate
        let p13 = pipelines.clearBody, p14 = pipelines.buildBody, p15 = pipelines.colorRound
        let p16 = pipelines.intVelCS, p17 = pipelines.solveVelCS
        let p18 = pipelines.intPosCS, p19 = pipelines.finalizeCS
        let p20 = pipelines.clearHash, p21 = pipelines.snapshot
        let p22 = pipelines.warmMatch, p23 = pipelines.warmApply
        let p24 = pipelines.islandInit, p25 = pipelines.islandUnion, p26 = pipelines.islandJump
        let p27 = pipelines.sleepTick, p28 = pipelines.islandMinReduce, p29 = pipelines.sleepMark
        let p30 = pipelines.gjkEPA, p31 = pipelines.writeSolveArgs
        let p32 = pipelines.jointSolveCS, p33 = pipelines.islandUnionJoints

        // Grid: one cell ≈ one contact diameter so a 3×3×3 scan covers every
        // possible body–body contact and box contacts. For a MIXED pile the cell
        // must cover the LARGEST body's extent (a long frankfurter's half-length
        // exceeds a coin's radius), so size it off max(radius, halfThickness) — the
        // controller constructs the solver with the biggest asset's dimensions.
        let cell = max(2.0 * max(coinRadius, halfThickness), 1e-4)
        let span = boundsMax - boundsMin
        let res = SIMD3<UInt32>(
            UInt32(max(1, Int((span.x / cell).rounded(.up)) + 1)),
            UInt32(max(1, Int((span.y / cell).rounded(.up)) + 1)),
            UInt32(max(1, Int((span.z / cell).rounded(.up)) + 1)))
        let cellCount = Int(res.x) * Int(res.y) * Int(res.z)

        let dev = engine.device
        let contactCap = maxCoins * 64                       // CoinContact buffer capacity
        var hashCap = 1; while hashCap < contactCap * 2 { hashCap <<= 1 }   // next power of two ≥ 2·cap
        guard
            let coins = SimBuffer<CoinBody>(device: dev, capacity: maxCoins, label: "Coin.bodies"),
            let cols  = SimBuffer<CoinStaticCollider>(device: dev, capacity: 256, label: "Coin.colliders"),
            let xform = dev.makeBuffer(length: MemoryLayout<CoinTransform>.stride * maxCoins,
                                       options: .storageModeShared),
            // Four float4 per coin: [Δpos.xyz, contactCount], [Δrot.xyz, supportFlag],
            // [supportNormal.xyz, _] (the leveling target), [contactNormal.xyz, _]
            // (Σ all contact normals, depth-weighted — the restitution impact axis).
            let delta = dev.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * maxCoins * 4,
                                       options: .storageModePrivate),
            let counts = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * cellCount,
                                        options: .storageModePrivate),
            let offsets = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * (cellCount + 1),
                                         options: .storageModePrivate),
            let sorted = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins,
                                        options: .storageModePrivate),
            let btype = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins,
                                       options: .storageModeShared),
            let linkB = dev.makeBuffer(length: MemoryLayout<SIMD2<Int32>>.stride * maxCoins,
                                       options: .storageModeShared),
            let matB = dev.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * maxCoins,
                                      options: .storageModeShared),
            let uni = dev.makeBuffer(length: MemoryLayout<CoinUniforms>.stride,
                                     options: .storageModeShared),
            let penResult = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * 2,
                                           options: .storageModeShared),
            let penThresh = dev.makeBuffer(length: MemoryLayout<Float>.stride,
                                           options: .storageModeShared),
            let contacts = dev.makeBuffer(length: MemoryLayout<CoinContact>.stride * maxCoins * 64,
                                          options: .storageModeShared),
            let contactCount = dev.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                              options: .storageModeShared),
            let maxContactsBuf = dev.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                                options: .storageModeShared),
            let bodyContacts = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins * CoinDEMSolver.maxBodyContacts,
                                              options: .storageModePrivate),
            let bodyContactCount = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins,
                                                  options: .storageModePrivate),
            let biasBuf = dev.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * maxCoins * 2,
                                         options: .storageModePrivate),
            let prevContacts = dev.makeBuffer(length: MemoryLayout<CoinContact>.stride * contactCap,
                                              options: .storageModePrivate),
            let pairHash = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * hashCap,
                                          options: .storageModePrivate),
            let hashSizeBuf = dev.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
            let islandLabel = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModePrivate),
            let islandMin = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModePrivate),
            let sleepTimer = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModeShared),
            let asleep = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModeShared),
            let solveArgs = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * 4, options: .storageModePrivate),
            let solveTG = dev.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
            let jointsBuf = dev.makeBuffer(length: MemoryLayout<CoinJoint>.stride * CoinDEMSolver.maxJoints,
                                           options: .storageModeShared)
        else {
            Self.log.error("Coin buffer allocation failed (cells=\(cellCount))")
            return nil
        }
        xform.label = "Coin.transforms"
        btype.label = "Coin.bodyType"
        linkB.label = "Coin.links"
        delta.label = "Coin.delta"
        counts.label = "Coin.cellCounts"
        offsets.label = "Coin.cellOffsets"
        sorted.label = "Coin.sorted"
        uni.label = "Coin.uniforms"

        self.engine = engine
        self.integratePipeline = p0
        self.cellClearPipeline = p1
        self.cellCountPipeline = p2
        self.scanPipeline = p3
        self.scatterPipeline = p4
        self.contactPipeline = p5
        self.applyPipeline = p6
        self.finalizePipeline = p7
        self.orientPipeline = p8
        self.transformPipeline = p9
        self.jointPipeline = p10
        self.measurePipeline = p11
        self.generatePipeline = p12
        self.clearBodyPipeline = p13
        self.buildBodyPipeline = p14
        self.colorRoundPipeline = p15
        self.intVelCSPipeline = p16
        self.solveVelCSPipeline = p17
        self.intPosCSPipeline = p18
        self.finalizeCSPipeline = p19
        self.clearHashPipeline = p20
        self.snapshotPipeline = p21
        self.warmMatchPipeline = p22
        self.warmApplyPipeline = p23
        self.islandInitPipeline = p24
        self.islandUnionPipeline = p25
        self.islandJumpPipeline = p26
        self.sleepTickPipeline = p27
        self.islandMinPipeline = p28
        self.sleepMarkPipeline = p29
        self.gjkEPAPipeline = p30
        self.writeSolveArgsPipeline = p31
        self.jointSolveCSPipeline = p32
        self.islandUnionJointsPipeline = p33
        self.coinBuffer = coins
        self.transformBuffer = xform
        self.bodyTypeBuffer = btype
        self.linkBuffer = linkB
        matB.label = "Coin.material"
        self.materialBuffer = matB
        self.colliderBuffer = cols
        self.coinDeltaBuffer = delta
        self.cellCounts = counts
        self.cellOffsets = offsets
        self.sortedIndices = sorted
        penResult.label = "Coin.penResult"
        penThresh.label = "Coin.penThreshold"
        self.penetrationResult = penResult
        self.penetrationThreshold = penThresh
        self.uniformBuffer = uni
        contacts.label = "Coin.contacts"
        contactCount.label = "Coin.contactCount"
        self.contactBuffer = contacts
        self.contactCountBuffer = contactCount
        self.maxContactsBuffer = maxContactsBuf
        bodyContacts.label = "Coin.bodyContacts"
        bodyContactCount.label = "Coin.bodyContactCount"
        self.bodyContactsBuffer = bodyContacts
        self.bodyContactCountBuffer = bodyContactCount
        biasBuf.label = "Coin.bias"
        self.biasBuffer = biasBuf
        prevContacts.label = "Coin.prevContacts"
        pairHash.label = "Coin.pairHash"
        self.prevContactBuffer = prevContacts
        self.pairHashBuffer = pairHash
        self.hashSizeBuffer = hashSizeBuf
        self.hashSize = hashCap
        hashSizeBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(hashCap)
        islandLabel.label = "Coin.islandLabel"; islandMin.label = "Coin.islandMin"
        sleepTimer.label = "Coin.sleepTimer"; asleep.label = "Coin.asleep"
        self.islandLabelBuffer = islandLabel
        self.islandMinBuffer = islandMin
        self.sleepTimerBuffer = sleepTimer
        self.asleepBuffer = asleep
        solveArgs.label = "Coin.solveArgs"
        self.solveArgsBuffer = solveArgs
        self.solveTGSizeBuffer = solveTG
        jointsBuf.label = "Coin.joints"
        self.jointBuffer = jointsBuf
        solveTG.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(CoinDEMSolver.solveTGSize)
        // Persistent + read-during-substep buffers start cleared (all awake, no slow frames).
        sleepTimer.contents().bindMemory(to: UInt32.self, capacity: maxCoins).update(repeating: 0, count: maxCoins)
        asleep.contents().bindMemory(to: UInt32.self, capacity: maxCoins).update(repeating: 0, count: maxCoins)
        self.maxContacts = maxCoins * 64
        maxContactsBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(maxCoins * 64)
        self.maxCoins = maxCoins
        self.coinRadius = coinRadius
        self.halfThickness = halfThickness
        self.gridMin = boundsMin
        self.gridRes = res
        self.cellSize = cell

        // Park every slot off-screen so unused instances are invisible from frame 0,
        // and clear every articulation link (no joints until a frank is spawned).
        // Materials start at (−1, −1) = inherit the global friction/restitution.
        let ptr = coins.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let lptr = linkB.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)
        let mptr = matB.contents().bindMemory(to: SIMD2<Float>.self, capacity: maxCoins)
        for i in 0..<maxCoins { ptr[i] = .inactive; lptr[i] = SIMD2(-1, 0); mptr[i] = SIMD2(-1, -1) }
    }

    /// Set (or clear) a body's material override. A negative component inherits the
    /// solver-wide `frictionCoeff` / `restitution`. Constraint path only; combined
    /// per contact as μ=√(μA·μB), e=max(eA,eB).
    public func setMaterial(_ slot: Int, friction: Float?, restitution: Float?) {
        guard slot >= 0, slot < maxCoins else { return }
        let m = materialBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: maxCoins)
        m[slot] = SIMD2(friction ?? -1, restitution ?? -1)
    }

    /// Articulate two slots: `slot`'s `mySign` end (±1 → ±Y) joins its partner's
    /// opposite end. Call after spawning both bodies (see CoinPusher's frankfurter).
    public func setLink(_ slot: Int, partner: Int, mySign: Int32) {
        guard slot >= 0, slot < maxCoins else { return }
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(Int32(partner), mySign)
    }

    private func clearLink(_ slot: Int, _ lptr: UnsafeMutablePointer<SIMD2<Int32>>) {
        // Freeing a linked body unlinks its partner too, so the survivor becomes a
        // lone capsule and the recycled slot is never falsely joint-solved.
        let partner = lptr[slot].x
        if partner >= 0, Int(partner) < maxCoins { lptr[Int(partner)] = SIMD2(-1, 0) }
        lptr[slot] = SIMD2(-1, 0)
    }

    // ── Colliders ─────────────────────────────────────────────────────────────

    /// Replace the static/kinematic collider set. Call once at setup for the
    /// fixed cabinet, then again each frame only to move the pusher box.
    public func setColliders(_ list: [CoinStaticCollider]) {
        colliders = list
        colliderBuffer.write(list)
    }

    // ── Spawn / cull ──────────────────────────────────────────────────────────

    /// Activate a body in a free slot. Returns the slot index, or nil if full.
    /// `radius`/`halfThickness` set the body's oriented-cylinder collider (default to
    /// the solver's nominal coin dims); `type` tags it for the mixed-pile renderers.
    @discardableResult
    public func spawn(at position: SIMD3<Float>,
                      velocity: SIMD3<Float> = .zero,
                      spin: Float = 0,
                      orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                      tumble: SIMD3<Float> = .zero,
                      radius: Float? = nil,
                      halfThickness: Float? = nil,
                      mass: Float = 1,
                      friction: Float? = nil,
                      restitution: Float? = nil,
                      type: UInt32 = 0) -> Int? {
        // `spin` is folded into the initial angular velocity about the body's own
        // axis; `tumble` is the world-frame initial angular velocity.
        let slot: Int
        if let reused = freeSlots.popLast() {
            slot = reused
        } else if nextSlot < maxCoins {
            slot = nextSlot
            nextSlot += 1
        } else {
            return nil
        }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        // Initial ω = world-frame tumble + spin about the body's own (oriented) axis.
        let axis = simd_act(simd_quatf(ix: orient.x, iy: orient.y, iz: orient.z, r: orient.w),
                            SIMD3<Float>(0, 1, 0))
        // Per-body mass → invMass (0 only for static; a real positive mass otherwise).
        // Default mass 1 reproduces the historical unit-mass pile exactly; a heavier
        // body (mass > 1) resists both translation and rotation proportionally.
        let invMass: Float = mass > 1e-6 ? 1.0 / mass : 1.0
        ptr[slot] = CoinBody(position: position, invMass: invMass, velocity: velocity,
                             orient: orient, angVel: tumble + axis * spin)
        // Per-body collision dimensions: radius → prevPos.w, halfThickness → vel.w.
        ptr[slot].prevPos.w = radius ?? coinRadius
        ptr[slot].vel.w     = halfThickness ?? self.halfThickness
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        // A freshly-spawned body has no joint until setLink wires one (it may be
        // reusing a slot whose previous link wasn't cleared).
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        // A reused slot must not inherit the previous body's sleep state, or a fresh
        // body could spawn frozen mid-air while the rest of the pile is asleep.
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    /// Activate a BOX body (half-extents `halfExtents`) in a free slot. Box bodies
    /// integrate, tip onto a face, and rest against the static environment with the
    /// same machinery as discs; their contact uses box corners + box inertia. (Box
    /// vs. other DYNAMIC bodies is not yet resolved — see CoinDEM.metal Phase 1.) The
    /// broadphase cell is sized off the solver's `coinRadius`, so construct the solver
    /// with `coinRadius ≥ length(halfExtents)` when spawning boxes.
    @discardableResult
    public func spawnBox(at position: SIMD3<Float>,
                         halfExtents: SIMD3<Float>,
                         velocity: SIMD3<Float> = .zero,
                         orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                         tumble: SIMD3<Float> = .zero,
                         mass: Float = 1,
                         friction: Float? = nil,
                         restitution: Float? = nil,
                         type: UInt32 = 0) -> Int? {
        let slot: Int
        if let reused = freeSlots.popLast() {
            slot = reused
        } else if nextSlot < maxCoins {
            slot = nextSlot; nextSlot += 1
        } else {
            return nil
        }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let invMass: Float = mass > 1e-6 ? 1.0 / mass : 1.0
        ptr[slot] = CoinBody(position: position, invMass: invMass, velocity: velocity,
                             orient: orient, angVel: tumble,
                             shapeExtents: SIMD4(halfExtents, 1))   // w = 1 → box
        // Bounding radius rides prevPos.w (broadphase reach); the SMALLEST half-extent
        // rides vel.w (the floor-safety / per-apply-clamp scale, like a disc's halfThick).
        ptr[slot].prevPos.w = simd_length(halfExtents)
        ptr[slot].vel.w     = min(halfExtents.x, min(halfExtents.y, halfExtents.z))
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        // A reused slot must not inherit the previous body's sleep state, or a fresh
        // body could spawn frozen mid-air while the rest of the pile is asleep.
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    /// Activate a SPHERE body of `radius` in a free slot. A sphere collides with an
    /// orientation-independent single-point contact (center − R·n) against planes,
    /// boxes, and other spheres, so it rests at exactly R in every orientation — no
    /// rim-corner float, no cap-flat clip (the failure modes of approximating a ball
    /// with the disc/capped-cylinder collider). The broadphase cell is sized off the
    /// solver's `coinRadius`, so construct the solver with `coinRadius ≥ radius`.
    @discardableResult
    public func spawnSphere(at position: SIMD3<Float>,
                            radius: Float,
                            velocity: SIMD3<Float> = .zero,
                            orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                            tumble: SIMD3<Float> = .zero,
                            mass: Float = 1,
                            friction: Float? = nil,
                            restitution: Float? = nil,
                            type: UInt32 = 0) -> Int? {
        let slot: Int
        if let reused = freeSlots.popLast() {
            slot = reused
        } else if nextSlot < maxCoins {
            slot = nextSlot; nextSlot += 1
        } else {
            return nil
        }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let invMass: Float = mass > 1e-6 ? 1.0 / mass : 1.0
        ptr[slot] = CoinBody(position: position, invMass: invMass, velocity: velocity,
                             orient: orient, angVel: tumble,
                             shapeExtents: SIMD4(radius, 0, 0, 2))   // w = 2 → sphere
        // radius rides BOTH prevPos.w (broadphase reach + cdRadiusOf) and vel.w
        // (the per-apply / floor-safety min-extent), so a sphere reads R uniformly.
        ptr[slot].prevPos.w = radius
        ptr[slot].vel.w     = radius
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        // A reused slot must not inherit the previous body's sleep state, or a fresh
        // body could spawn frozen mid-air while the rest of the pile is asleep.
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    /// Activate a CAPSULE body (cross-section `radius`, segment half-length
    /// `halfLength` — cap centres at ±halfLength along local +Y, so the full
    /// height is 2·(halfLength + radius)). On the constraint path a capsule
    /// collides EXACTLY (segment + radius probes: smooth round caps, a 2-point
    /// manifold when lying on its side); on the legacy path it degrades to its
    /// bounding capped cylinder (the historical frank approximation). The
    /// broadphase cell is sized off the solver's construction dims, so construct
    /// with `halfThickness ≥ halfLength + radius` when spawning capsules.
    @discardableResult
    public func spawnCapsule(at position: SIMD3<Float>,
                             radius: Float,
                             halfLength: Float,
                             velocity: SIMD3<Float> = .zero,
                             orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                             tumble: SIMD3<Float> = .zero,
                             mass: Float = 1,
                             friction: Float? = nil,
                             restitution: Float? = nil,
                             type: UInt32 = 0) -> Int? {
        let slot: Int
        if let reused = freeSlots.popLast() {
            slot = reused
        } else if nextSlot < maxCoins {
            slot = nextSlot; nextSlot += 1
        } else {
            return nil
        }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let invMass: Float = mass > 1e-6 ? 1.0 / mass : 1.0
        ptr[slot] = CoinBody(position: position, invMass: invMass, velocity: velocity,
                             orient: orient, angVel: tumble,
                             shapeExtents: SIMD4(halfLength, 0, 0, 3))   // w = 3 → capsule
        // Disc-compatible lanes: cross-section radius rides prevPos.w, the FULL
        // half-height (hl + r) rides vel.w — so the legacy capped-cylinder path,
        // the broadphase reach, and the floor backstop all see the true bounds.
        ptr[slot].prevPos.w = radius
        ptr[slot].vel.w     = halfLength + radius
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        // A reused slot must not inherit the previous body's sleep state.
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    // ── Generic joints (constraint path) ──────────────────────────────────────

    private func claimJointSlot() -> Int? {
        if let reused = freeJointSlots.popLast() { return reused }
        guard jointHighWater < Self.maxJoints else { return nil }
        defer { jointHighWater += 1 }
        return jointHighWater
    }

    private func writeJoint(_ slot: Int, _ joint: CoinJoint) {
        jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)[slot] = joint
        wakeAll()   // constraint topology changed — sleeping islands must re-evaluate
    }

    /// A-local coordinates of a world point (for anchor construction).
    private func toLocal(_ slot: Int, _ world: SIMD3<Float>) -> SIMD3<Float> {
        let b = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)[slot]
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        return simd_act(q.inverse, world - SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z))
    }

    private func toLocalDir(_ slot: Int, _ worldDir: SIMD3<Float>) -> SIMD3<Float> {
        let b = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)[slot]
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        return simd_act(q.inverse, worldDir)
    }

    /// BALL joint: the world point `worldAnchor` on body A coincides with the same
    /// point on body B (or with that fixed world point when `bodyB` is nil).
    /// Anchors are captured from the bodies' CURRENT poses. Returns a handle for
    /// `removeJoint`, or nil when the joint table is full. Constraint path only.
    @discardableResult
    public func addBallJoint(bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        writeJoint(slot, CoinJoint(
            meta: SIMD4(0, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), 1),
            anchorA: SIMD4(toLocal(bodyA, worldAnchor), 0),
            anchorB: SIMD4(world ? worldAnchor : toLocal(bodyB!, worldAnchor), 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0)))
        return slot
    }

    /// HINGE joint: ball at `worldAnchor` + the bodies may only rotate relative to
    /// each other about `worldAxis`. `limits` (radians, lo < hi, measured from the
    /// CURRENT relative pose = 0) adds angle stops; nil = free spin.
    @discardableResult
    public func addHingeJoint(bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                              worldAxis: SIMD3<Float>,
                              limits: ClosedRange<Float>? = nil) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        let axis = simd_normalize(worldAxis)
        // lo == hi disables the limit branch in-kernel.
        let lo = limits?.lowerBound ?? 0, hi = limits?.upperBound ?? 0
        writeJoint(slot, CoinJoint(
            meta: SIMD4(1, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), 1),
            anchorA: SIMD4(toLocal(bodyA, worldAnchor), 0),
            anchorB: SIMD4(world ? worldAnchor : toLocal(bodyB!, worldAnchor), 0),
            axisA: SIMD4(toLocalDir(bodyA, axis), lo),
            axisB: SIMD4(world ? axis : toLocalDir(bodyB!, axis), hi)))
        return slot
    }

    /// DISTANCE joint: the two world anchor points keep their CURRENT separation
    /// (or `restLength` when given) — a rigid tether/rod, free to swing.
    @discardableResult
    public func addDistanceJoint(bodyA: Int, bodyB: Int?,
                                 worldAnchorA: SIMD3<Float>, worldAnchorB: SIMD3<Float>,
                                 restLength: Float? = nil) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        let rest = restLength ?? simd_length(worldAnchorB - worldAnchorA)
        writeJoint(slot, CoinJoint(
            meta: SIMD4(2, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), 1),
            anchorA: SIMD4(toLocal(bodyA, worldAnchorA), rest),
            anchorB: SIMD4(world ? worldAnchorB : toLocal(bodyB!, worldAnchorB), 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0)))
        return slot
    }

    /// Remove (disable + recycle) a joint created by the add*Joint APIs.
    public func removeJoint(_ handle: Int) {
        guard handle >= 0, handle < jointHighWater else { return }
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        guard j[handle].meta.w != 0 else { return }   // already free
        j[handle].meta.w = 0
        freeJointSlots.append(handle)
        wakeAll()
    }

    /// Disable every joint that references `slot` (called when the body despawns —
    /// a recycled slot must never inherit a stale constraint).
    private func removeJoints(referencing slot: Int) {
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        let s = UInt32(slot)
        for i in 0..<jointHighWater where j[i].meta.w != 0 && (j[i].meta.y == s || j[i].meta.z == s) {
            j[i].meta.w = 0
            freeJointSlots.append(i)
        }
    }

    /// Recycle a coin slot (e.g. it left through the trough or a side gap).
    public func despawn(_ slot: Int) {
        guard slot >= 0, slot < maxCoins else { return }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        guard ptr[slot].posInvMass.w != 0 else { return }   // already free
        let lptr = linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)
        clearLink(slot, lptr)
        removeJoints(referencing: slot)
        ptr[slot] = .inactive
        freeSlots.append(slot)
    }

    /// Recycle every active coin whose COM has fallen below `y` (collected at the
    /// payout / fell out a side gap). Returns how many were recycled. Reads the
    /// shared buffer — call after the frame's command buffer has completed (the
    /// one-frame-stale read in a live tick is harmless).
    @discardableResult
    public func cull(belowY y: Float) -> Int {
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let lptr = linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)
        var n = 0
        for slot in 0..<highWater where ptr[slot].posInvMass.w != 0 {
            if ptr[slot].posInvMass.y < y {
                clearLink(slot, lptr)
                removeJoints(referencing: slot)
                ptr[slot] = .inactive
                freeSlots.append(slot)
                n += 1
            }
        }
        return n
    }

    /// Recycle every active body at once (e.g. a drop-asset toggle was switched
    /// off, so its whole pile should vanish). Mutates the shared buffer — call
    /// after the frame's command buffer has completed, like `cull`.
    public func clearAll() {
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let lptr = linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)
        for slot in 0..<highWater where ptr[slot].posInvMass.w != 0 {
            lptr[slot] = SIMD2(-1, 0)
            ptr[slot] = .inactive
        }
        // Fully reset the slot bookkeeping so `encode` (which dispatches over
        // `highWater`) does NO work on an emptied field — otherwise a cleared
        // field keeps paying the full per-frame solver cost forever.
        freeSlots.removeAll(keepingCapacity: true)
        nextSlot = 0
        highWater = 0
        // Joints reference the cleared bodies — drop them all with the field.
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        for i in 0..<jointHighWater { j[i].meta.w = 0 }
        freeJointSlots.removeAll(keepingCapacity: true)
        jointHighWater = 0
    }

    /// Read-only snapshot of an active coin's COM (safe after the frame's command
    /// buffer completes). Used by the controller's cull pass.
    public func position(of slot: Int) -> SIMD3<Float>? {
        guard slot >= 0, slot < highWater else { return nil }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let b = ptr[slot]
        return b.posInvMass.w == 0 ? nil : SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z)
    }

    /// Read-only snapshot of an active body's linear velocity.
    public func velocity(of slot: Int) -> SIMD3<Float>? {
        guard slot >= 0, slot < highWater else { return nil }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let b = ptr[slot]
        return b.posInvMass.w == 0 ? nil : SIMD3(b.vel.x, b.vel.y, b.vel.z)
    }

    /// Read-only snapshot of an active body's orientation quaternion.
    public func orientation(of slot: Int) -> simd_quatf? {
        guard slot >= 0, slot < highWater else { return nil }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let b = ptr[slot]
        guard b.posInvMass.w != 0 else { return nil }
        return simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
    }

    /// Directly set an active body's linear velocity (e.g. a scripted impulse —
    /// the Tennis Ball Painter's puddle-bounce "kick toward the wall"). Mutates
    /// the shared buffer; safe between frames (the next `encode` reads it).
    public func setVelocity(ofSlot slot: Int, to v: SIMD3<Float>) {
        guard slot >= 0, slot < highWater else { return }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        guard ptr[slot].posInvMass.w != 0 else { return }
        ptr[slot].vel.x = v.x; ptr[slot].vel.y = v.y; ptr[slot].vel.z = v.z
    }

    /// Directly set an active body's COM (world space). Used to DRIVE a body the
    /// user is dragging by the mouse — pair with `setKinematicHold` so the body
    /// is frozen-immovable while the host writes its position each frame. Mutates
    /// the shared buffer between frames; the next `encode` reads it.
    public func setPosition(ofSlot slot: Int, to p: SIMD3<Float>) {
        guard slot >= 0, slot < highWater else { return }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        guard ptr[slot].posInvMass.w != 0 else { return }
        ptr[slot].posInvMass.x = p.x; ptr[slot].posInvMass.y = p.y; ptr[slot].posInvMass.z = p.z
    }

    /// Hold (or release) a body KINEMATICALLY — a real infinite-mass moving
    /// collider, not a faked freeze. A held body is what a mouse-grabbed object
    /// should be: the integrator applies no gravity and never moves it, and the
    /// contact solve treats it as immovable (invMass 0), so the awake pile
    /// genuinely collides against it while the host drives its COM with
    /// `setPosition`. Releasing returns it to a normal dynamic body — set its
    /// throw with `setVelocity` first if you want it to fly on let-go.
    ///
    /// IMPLEMENTATION: this reuses the per-body `asleep` flag, which the
    /// constraint-path kernels already honour as exactly "no gravity / no move /
    /// immovable neighbour". That requires the **constraint** solver
    /// (`solverMode = .constraint`) and **sleep disabled** (`sleepEnabled =
    /// false`) so the island-sleep kernels don't overwrite the flag the host
    /// owns. Both are asserted so a future caller can't silently lose the hold.
    public func setKinematicHold(_ slot: Int, _ held: Bool) {
        guard slot >= 0, slot < highWater else { return }
        assert(solverMode == .constraint,
               "setKinematicHold requires solverMode == .constraint (the legacy path ignores asleep[])")
        assert(!sleepEnabled,
               "setKinematicHold requires sleepEnabled == false (host owns asleep[])")
        let a = asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        a[slot] = held ? 1 : 0
    }

    // ── Encode ────────────────────────────────────────────────────────────────

    /// Standalone advance using the engine's own queue. Used by the isolation
    /// harness; the scene controller calls `encode(to:wallDt:)` on its own buffer.
    public func advance(wallDt: Float) {
        guard let cb = engine.commandQueue.makeCommandBuffer() else { return }
        cb.label = "Coin.advance"
        encode(to: cb, wallDt: wallDt)
        cb.commit()
    }

    /// Encode a frame's worth of fixed-dt substeps + orientation + transforms
    /// into `cb`. No `waitUntilCompleted` here — the caller commits once.
    public func encode(to cb: MTLCommandBuffer, wallDt: Float) {
        let bound = highWater
        guard bound > 0 else { return }

        // Whole-pile sleep (constraint path): if every ACTIVE body is frozen, skip ALL
        // solver work this frame — generation, coloring, the solve, integration, even the
        // sleep update. Positions are unchanged, so the last-derived transforms stay
        // valid and the renderer keeps drawing the settled heap. A wake (spawn clears the
        // slot's flag; wakeAll on a buoyancy/gravity flip or reset) drops the asleep count
        // and resumes work next frame. This is what makes a settled scene cost ~nothing.
        if solverMode == .constraint && sleepEnabled {
            let active = activeCount
            if active > 0 && asleepCount >= active { didSkipLastFrame = true; return }
        }
        didSkipLastFrame = false

        // Spiral-of-death guard: bank at most one frame's worth of substeps. Without
        // this, a hitch (GC pause, window drag, sleep/wake) inflates the accumulator
        // past maxSubsteps·fixedDt; every following frame then runs the full substep
        // cap trying to repay the debt, which slows the sim and can never catch up. We
        // run real time up to the cap and DROP the unrepayable remainder — the sim
        // briefly runs slightly slow-motion through the hitch instead of death-
        // spiralling, which is the standard fixed-timestep degradation.
        accumulator = min(accumulator + wallDt, fixedDt * Float(maxSubsteps))
        var steps = 0
        while accumulator >= fixedDt && steps < maxSubsteps {
            writeUniforms(dt: fixedDt, coinCount: bound)
            switch solverMode {
            case .legacy:     encodeSubstep(cb, coinCount: bound)
            case .constraint: encodeConstraintSubstep(cb, coinCount: bound)
            }
            accumulator -= fixedDt
            steps += 1
        }
        lastStepCount = steps
        if steps == 0 { return }   // not enough wall time accumulated for a substep

        // Island detection + sleeping (constraint path): freeze settled islands so a
        // resting heap skips its whole solve next frame. Uses the last substep's contacts.
        if solverMode == .constraint && sleepEnabled {
            encodeSleepUpdate(cb, coinCount: bound)
        }

        // Gentle support-leveling once per frame. With the distributed face-contact
        // manifold doing the flattening by emergent physics, this is now an OPTIONAL
        // quieting assist (default off — set `levelingEnabled` to re-enable).
        if levelingEnabled {
            dispatch(cb, orientPipeline, threads: bound, label: "Coin.level") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 1)
                enc.setBuffer(self.coinDeltaBuffer, offset: 0, index: 2)
            }
        }
        dispatch(cb, transformPipeline, threads: bound, label: "Coin.transforms") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.transformBuffer, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
    }

    private func encodeSubstep(_ cb: MTLCommandBuffer, coinCount: Int) {
        // 1. integrate (predict COM)
        dispatch(cb, integratePipeline, threads: coinCount, label: "Coin.integrate") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 1)
        }
        // 2. spatial hash (counting sort)
        dispatch(cb, cellClearPipeline, threads: numCells, label: "Coin.cellClear") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
        }
        dispatch(cb, cellCountPipeline, threads: coinCount, label: "Coin.cellCount") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, scanPipeline, threads: 1, label: "Coin.scan") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, scatterPipeline, threads: coinCount, label: "Coin.scatter") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
            enc.setBuffer(self.sortedIndices, offset: 0, index: 3)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
        }
        // 3. contact iterations (Jacobi: solve → apply)
        for _ in 0..<iterations {
            dispatch(cb, contactPipeline, threads: coinCount, label: "Coin.contact") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.sortedIndices, offset: 0, index: 1)
                enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
                enc.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 3)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
                enc.setBuffer(self.coinDeltaBuffer, offset: 0, index: 5)
                enc.setBuffer(self.linkBuffer, offset: 0, index: 6)
            }
            dispatch(cb, applyPipeline, threads: coinCount, label: "Coin.apply") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.coinDeltaBuffer, offset: 0, index: 1)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
            }
            // Articulation: solve the frank bend joints right after each contact apply
            // so the joint and contacts converge together (a sausage segment pressed
            // by a coin both de-penetrates AND bends at its joint in the same iter).
            dispatch(cb, jointPipeline, threads: coinCount, label: "Coin.joint") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.linkBuffer, offset: 0, index: 1)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
            }
        }
        // 4. finalize (velocity, friction, sleep, floor safety)
        dispatch(cb, finalizePipeline, threads: coinCount, label: "Coin.finalize") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.coinDeltaBuffer, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
    }

    // ── Constraint substep (Stage 3): graph-colored sequential impulse ──────────
    //
    // predict velocity → generate+colour contacts ONCE → iterate the velocity solve
    // over colours (Gauss-Seidel) → integrate position from (real + bias) velocity →
    // finalize. No position-to-velocity feedback, so a settled pile carries zero
    // restoring velocity and goes still regardless of density.
    private func encodeConstraintSubstep(_ cb: MTLCommandBuffer, coinCount: Int) {
        dispatch(cb, intVelCSPipeline, threads: coinCount, label: "Coin.cs.intVel") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
            enc.setBuffer(self.asleepBuffer, offset: 0, index: 3)
        }
        encodeBroadphase(cb, coinCount: coinCount)
        encodeGenerateContacts(cb, coinCount: coinCount)
        // Size the indirect dispatch to the ACTUAL contact count (≈600 in a settled
        // 176-pile vs ~12k capacity → ~19× fewer GPU threads launched per solve pass).
        dispatch(cb, writeSolveArgsPipeline, threads: 1, label: "Coin.cs.solveArgs") { enc in
            enc.setBuffer(self.contactCountBuffer, offset: 0, index: 0)
            enc.setBuffer(self.solveArgsBuffer, offset: 0, index: 1)
            enc.setBuffer(self.solveTGSizeBuffer, offset: 0, index: 2)
        }
        encodeColoring(cb)
        // Warm start: seed fresh contacts with last substep's converged impulses and
        // apply them before iterating (a resting stack resumes from its solution).
        if warmStart {
            if needsHashClear {                       // zero the hash on first use (private buffer)
                dispatch(cb, clearHashPipeline, threads: hashSize, label: "Coin.cs.clearHash0") { enc in
                    enc.setBuffer(self.pairHashBuffer, offset: 0, index: 0)
                }
                needsHashClear = false
            }
            dispatch(cb, warmMatchPipeline, threads: maxContacts, label: "Coin.cs.warmMatch") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.prevContactBuffer, offset: 0, index: 2)
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 3)
                enc.setBuffer(self.hashSizeBuffer, offset: 0, index: 4)
            }
            for color in 0..<solveColors {
                var cc = UInt32(color)
                dispatchIndirect(cb, warmApplyPipeline, label: "Coin.cs.warmApply") { enc in
                    enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                    enc.setBuffer(self.contactBuffer, offset: 0, index: 1)
                    enc.setBuffer(self.contactCountBuffer, offset: 0, index: 2)
                    enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 3)   // coinWarmStartApply doesn't read u; the old uniform@3 bind was unused (aborts under Metal API validation)
                }
            }
        }
        for _ in 0..<velocityIterations {
            for color in 0..<solveColors {
                var cc = UInt32(color)
                dispatchIndirect(cb, solveVelCSPipeline, label: "Coin.cs.solve") { enc in
                    enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                    enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                    enc.setBuffer(self.contactBuffer, offset: 0, index: 2)
                    enc.setBuffer(self.contactCountBuffer, offset: 0, index: 3)
                    enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
                    enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 5)
                    enc.setBuffer(self.asleepBuffer, offset: 0, index: 6)
                    enc.setBuffer(self.materialBuffer, offset: 0, index: 7)
                }
            }
            // Generic joints: one serial Gauss-Seidel pass over all joints per
            // velocity iteration, interleaved with the contact colours so joints
            // and contacts converge together (a jointed body pressed by a pile
            // both holds its anchor AND de-penetrates in the same substep).
            if jointHighWater > 0 {
                var jc = UInt32(jointHighWater)
                dispatch(cb, jointSolveCSPipeline, threads: 1, label: "Coin.cs.joints") { enc in
                    enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                    enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                    enc.setBuffer(self.jointBuffer, offset: 0, index: 2)
                    enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 3)
                    enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
                    enc.setBuffer(self.asleepBuffer, offset: 0, index: 5)
                }
            }
        }
        // Snapshot the solved contacts + rebuild the pair hash for next substep.
        if warmStart {
            dispatch(cb, clearHashPipeline, threads: hashSize, label: "Coin.cs.clearHash") { enc in
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 0)
            }
            dispatch(cb, snapshotPipeline, threads: maxContacts, label: "Coin.cs.snapshot") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.prevContactBuffer, offset: 0, index: 2)
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 3)
                enc.setBuffer(self.hashSizeBuffer, offset: 0, index: 4)
            }
        }
        dispatch(cb, intPosCSPipeline, threads: coinCount, label: "Coin.cs.intPos") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
            enc.setBuffer(self.asleepBuffer, offset: 0, index: 3)
        }
        dispatch(cb, finalizeCSPipeline, threads: coinCount, label: "Coin.cs.finalize") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 1)
            enc.setBuffer(self.asleepBuffer, offset: 0, index: 2)
        }
    }

    // ── Island detection + sleeping (Stage 5), once per frame ───────────────────
    // Label connected islands (union-find), tick per-body slow counters, and freeze a
    // whole island once its slowest body passes `sleepFrames`. Uses the LAST substep's
    // contacts (still in the buffer). Sets `asleep` for the NEXT frame's substeps.
    private func encodeSleepUpdate(_ cb: MTLCommandBuffer, coinCount: Int) {
        dispatch(cb, islandInitPipeline, threads: coinCount, label: "Coin.cs.islandInit") { enc in
            enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 1)   // coinIslandInit reads only label + u; the old coins@1 bind was unused (aborts under Metal API validation)
        }
        for _ in 0..<islandUnionRounds {
            dispatch(cb, islandUnionPipeline, threads: maxContacts, label: "Coin.cs.islandUnion") { enc in
                enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactBuffer, offset: 0, index: 1)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 2)
            }
            // Joint edges join the same islands, so an articulated assembly
            // sleeps and wakes as one body.
            if jointHighWater > 0 {
                var jc = UInt32(jointHighWater)
                dispatch(cb, islandUnionJointsPipeline, threads: jointHighWater, label: "Coin.cs.islandUnionJoints") { enc in
                    enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
                    enc.setBuffer(self.jointBuffer, offset: 0, index: 1)
                    enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 2)
                }
            }
            dispatch(cb, islandJumpPipeline, threads: coinCount, label: "Coin.cs.islandJump") { enc in
                enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 1)
            }
        }
        dispatch(cb, sleepTickPipeline, threads: coinCount, label: "Coin.cs.sleepTick") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.sleepTimerBuffer, offset: 0, index: 1)
            enc.setBuffer(self.islandMinBuffer, offset: 0, index: 2)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
        }
        dispatch(cb, islandMinPipeline, threads: coinCount, label: "Coin.cs.islandMin") { enc in
            enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
            enc.setBuffer(self.sleepTimerBuffer, offset: 0, index: 1)
            enc.setBuffer(self.islandMinBuffer, offset: 0, index: 2)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
        }
        dispatch(cb, sleepMarkPipeline, threads: coinCount, label: "Coin.cs.sleepMark") { enc in
            var sf = self.sleepFrames
            enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
            enc.setBuffer(self.islandMinBuffer, offset: 0, index: 1)
            enc.setBuffer(self.asleepBuffer, offset: 0, index: 2)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
            enc.setBytes(&sf, length: MemoryLayout<UInt32>.size, index: 4)
        }
    }

    /// How many ACTIVE bodies are currently asleep (read after a frame). For tests /
    /// perf. Checks invMass so a despawned slot's stale flag isn't counted.
    public var asleepCount: Int {
        let a = asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        let c = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        var n = 0
        for i in 0..<highWater where c[i].posInvMass.w != 0 && a[i] != 0 { n += 1 }
        return n
    }

    /// Wake every body (clear asleep + slow counters). Call on any change the island
    /// system can't see locally — a global gravity flip (buoyancy toggle), a reset, or
    /// an external impulse — so frozen bodies respond. (A new dropped body wakes its
    /// island on contact automatically, so spawning doesn't strictly need this.)
    public func wakeAll() {
        let a = asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        let s = sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        a.update(repeating: 0, count: maxCoins)
        s.update(repeating: 0, count: maxCoins)
    }

    /// Representative body radius for scale-relative penetration thresholds.
    public var characteristicBodyScale: Float { coinRadius }

    /// Measure interpenetration in the pile's CURRENT state without advancing
    /// the sim. Rebuilds the spatial-hash broadphase from current positions
    /// (no integrate step) and runs `coinMeasurePenetration`, which reuses the
    /// exact disk-vs-disk SAT the contact solver de-penetrates with — so the
    /// reported depth is consistent with what the solver itself sees. One-shot
    /// diagnostic: call after the pile settles, not per frame.
    public func measurePenetration(threshold: Float) -> PenetrationStats {
        let coinCount = highWater   // inactive slots (invMass 0) are skipped in-kernel
        guard coinCount > 0 else {
            return PenetrationStats(source: "CoinDEM", bodyCount: 0,
                                    penetratingPairs: 0, maxPenetration: 0, threshold: threshold)
        }
        let res = penetrationResult.contents().bindMemory(to: UInt32.self, capacity: 2)
        res[0] = 0; res[1] = 0
        penetrationThreshold.contents().bindMemory(to: Float.self, capacity: 1).pointee = threshold
        writeUniforms(dt: fixedDt, coinCount: coinCount)

        guard let cb = engine.commandQueue.makeCommandBuffer() else {
            return PenetrationStats(source: "CoinDEM", bodyCount: activeCount,
                                    penetratingPairs: 0, maxPenetration: 0, threshold: threshold)
        }
        // Rebuild broadphase from current positions — NO integrate, so the sim
        // does not advance; this is a pure read of the settled state.
        dispatch(cb, cellClearPipeline, threads: numCells, label: "Coin.measure.clear") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
        }
        dispatch(cb, cellCountPipeline, threads: coinCount, label: "Coin.measure.count") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, scanPipeline, threads: 1, label: "Coin.measure.scan") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, scatterPipeline, threads: coinCount, label: "Coin.measure.scatter") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
            enc.setBuffer(self.sortedIndices, offset: 0, index: 3)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)   // coinScatter reads grid dims from u[0] — was unbound (aborts under Metal API validation; matches encodeBroadphase/encodeSubstep)
        }
        dispatch(cb, measurePipeline, threads: coinCount, label: "Coin.measure") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.sortedIndices, offset: 0, index: 1)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
            enc.setBuffer(self.linkBuffer, offset: 0, index: 4)
            enc.setBuffer(self.penetrationResult, offset: 0, index: 5)
            enc.setBuffer(self.penetrationThreshold, offset: 0, index: 6)
        }
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: one-shot diagnostic readback, not the per-frame render loop

        let maxDepthMicrometres = res[0]
        let pairs = Int(res[1])
        return PenetrationStats(
            source: "CoinDEM",
            bodyCount: activeCount,
            penetratingPairs: pairs,
            maxPenetration: Float(maxDepthMicrometres) * 1e-6,
            threshold: threshold)
    }

    // ── Constraint solver (Stage 1+) ────────────────────────────────────────────

    /// Encode the broadphase (counting-sort spatial hash) from CURRENT positions —
    /// no integrate. Shared by the diagnostic measure and the constraint passes.
    private func encodeBroadphase(_ cb: MTLCommandBuffer, coinCount: Int) {
        dispatch(cb, cellClearPipeline, threads: numCells, label: "Coin.cs.cellClear") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
        }
        dispatch(cb, cellCountPipeline, threads: coinCount, label: "Coin.cs.cellCount") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, scanPipeline, threads: 1, label: "Coin.cs.scan") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, scatterPipeline, threads: coinCount, label: "Coin.cs.scatter") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
            enc.setBuffer(self.sortedIndices, offset: 0, index: 3)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
        }
    }

    /// Encode contact generation: zero the append cursor (CPU, shared), then append
    /// all contacts from the broadphase (must already be built this command buffer).
    private func encodeGenerateContacts(_ cb: MTLCommandBuffer, coinCount: Int) {
        contactCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        dispatch(cb, generatePipeline, threads: coinCount, label: "Coin.cs.generate") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.sortedIndices, offset: 0, index: 1)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
            enc.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 3)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
            enc.setBuffer(self.linkBuffer, offset: 0, index: 5)
            enc.setBuffer(self.contactBuffer, offset: 0, index: 6)
            enc.setBuffer(self.contactCountBuffer, offset: 0, index: 7)
            enc.setBuffer(self.maxContactsBuffer, offset: 0, index: 8)
        }
    }

    /// Run GJK+EPA on two active bodies and return whether they overlap, the
    /// penetration depth (m), and the contact normal (world, from B toward A). The
    /// general convex narrowphase — verified against the analytic routines.
    public func probeGJKEPA(_ a: Int, _ b: Int) -> (hit: Bool, depth: Float, normal: SIMD3<Float>) {
        guard let res = engine.device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * 2,
                                                 options: .storageModeShared),
              let cb = engine.commandQueue.makeCommandBuffer() else { return (false, 0, .zero) }
        var pair = SIMD2<UInt32>(UInt32(a), UInt32(b))
        dispatch(cb, gjkEPAPipeline, threads: 1, label: "Coin.gjkEPA") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBytes(&pair, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
            enc.setBuffer(res, offset: 0, index: 2)
        }
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: one-shot diagnostic probe
        let p = res.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)
        return (p[0].x > 0.5, p[0].y, SIMD3(p[1].x, p[1].y, p[1].z))
    }

    /// Current contact count (clamped to capacity). Read after a generate pass.
    public var contactCount: Int {
        min(Int(contactCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee), maxContacts)
    }

    /// Number of graph-colouring rounds encoded per substep. Jones-Plassmann with a
    /// random priority colours ~half the frontier per round, so ~log₂(#contacts)
    /// rounds suffice; 24 is generous headroom for a dense pile (the leftover, if any,
    /// stay colour −1 and are skipped — handled atomically in a later stage).
    public var colorRounds: Int = 24

    /// Encode the graph colouring of the current contact buffer: build per-body
    /// contact lists, then run `colorRounds` Jones-Plassmann rounds — all in `cb`,
    /// no per-round readback. Assumes contacts were generated this command buffer.
    private func encodeColoring(_ cb: MTLCommandBuffer) {
        dispatch(cb, clearBodyPipeline, threads: maxCoins, label: "Coin.cs.clearBody") { enc in
            enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 0)
        }
        dispatch(cb, buildBodyPipeline, threads: maxContacts, label: "Coin.cs.buildBody") { enc in
            enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
            enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
            enc.setBuffer(self.bodyContactsBuffer, offset: 0, index: 2)
            enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 3)
        }
        for _ in 0..<colorRounds {
            dispatch(cb, colorRoundPipeline, threads: maxContacts, label: "Coin.cs.color") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.bodyContactsBuffer, offset: 0, index: 2)
                enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 3)
            }
        }
    }

    /// One-shot: build the broadphase + generate contacts (and optionally colour them)
    /// from the CURRENT state; return how many contacts were produced. Does NOT advance
    /// the sim. For tests / diagnostics.
    @discardableResult
    public func generateContactsNow(color: Bool = false) -> Int {
        let coinCount = highWater
        guard coinCount > 0 else { return 0 }
        writeUniforms(dt: fixedDt, coinCount: coinCount)
        guard let cb = engine.commandQueue.makeCommandBuffer() else { return 0 }
        encodeBroadphase(cb, coinCount: coinCount)
        encodeGenerateContacts(cb, coinCount: coinCount)
        if color { encodeColoring(cb) }
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: one-shot diagnostic readback
        return contactCount
    }

    private func writeUniforms(dt: Float, coinCount: Int) {
        let u = CoinUniforms(
            dt: dt, gravity: gravity, linDamping: linDamping,
            coinRadius: coinRadius, halfThickness: halfThickness,
            contactRelax: contactRelax, friction: friction, restitution: restitution,
            frictionCoeff: frictionCoeff, rollingResistance: rollingResistance,
            floorY: floorY, sleepLinVel: sleepLinVel,
            angFriction: angFriction, angDamping: angDamping,
            gridMinX: gridMin.x, gridMinY: gridMin.y, gridMinZ: gridMin.z,
            invCell: 1.0 / cellSize,
            coinCount: UInt32(coinCount), colliderCount: UInt32(colliders.count),
            gridResX: gridRes.x, gridResY: gridRes.y, gridResZ: gridRes.z,
            maxHSpeed: maxHSpeed, maxSpeed: maxSpeed, maxOmega: maxOmega,
            contactSlop: contactSlop, baumgarteBeta: baumgarteBeta, restThreshold: restThreshold,
            restitutionVelFalloff: restitutionVelFalloff, restitutionMinE: restitutionMinE,
            quadraticDrag: quadraticDrag, dragRefRadius: dragRefRadius,
            speculativeMargin: speculativeMargin)
        uniformBuffer.contents().bindMemory(to: CoinUniforms.self, capacity: 1).pointee = u
    }

    private func dispatch(_ cb: MTLCommandBuffer,
                          _ pipeline: MTLComputePipelineState,
                          threads: Int,
                          label: String,
                          _ bind: (MTLComputeCommandEncoder) -> Void) {
        guard threads > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        bind(enc)
        let w = min(threads, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Dispatch a contact-iterating kernel over only the ACTUAL contacts (the GPU wrote
    /// the threadgroup count into `solveArgsBuffer` this substep) instead of the buffer
    /// capacity — the kernel's `cid >= contactCount` guard covers the partial tail.
    private func dispatchIndirect(_ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                                  label: String, _ bind: (MTLComputeCommandEncoder) -> Void) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        bind(enc)
        enc.dispatchThreadgroups(indirectBuffer: solveArgsBuffer, indirectBufferOffset: 0,
                                 threadsPerThreadgroup: MTLSize(width: Self.solveTGSize, height: 1, depth: 1))
        enc.endEncoding()
    }
}
