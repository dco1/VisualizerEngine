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
    // Shape: w = tag (0 = disc / capped cylinder, the default; 1 = box; 2 =
    // sphere; 3 = capsule; 4 = convex hull; 5 = ovoid/egg). For a box, xyz =
    // the three half-extents; sphere, x = radius; capsule, x = segment
    // half-length; ovoid, x = fat radius, y = tip radius, z = tip-centre offset.
    // Appended so every existing field keeps its offset — existing kernels and
    // CPU readbacks are byte-identical for discs.
    public var shapeExtents: SIMD4<Float>
    // Convex hull (tag 4): x = registered hull index (as float). Ovoid (tag 5):
    // x = fat-sphere centre offset from the COM. Both: yzw = the per-unit-mass
    // INVERSE inertia diagonal in the principal frame (I⁻¹ = invMass · yzw).
    // Zero for every other shape.
    public var hullRef: SIMD4<Float>

    public init(position: SIMD3<Float>,
                invMass: Float = 1.0,
                velocity: SIMD3<Float> = .zero,
                orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                angVel: SIMD3<Float> = .zero,
                shapeExtents: SIMD4<Float> = SIMD4(0, 0, 0, 0),
                hullRef: SIMD4<Float> = SIMD4(0, 0, 0, 0)) {
        self.posInvMass = SIMD4(position, invMass)
        self.prevPos    = SIMD4(position, 0)
        self.vel        = SIMD4(velocity, 0)
        self.orient     = orient
        self.prevOrient = orient
        self.angVel     = SIMD4(angVel, 0)
        self.shapeExtents = shapeExtents
        self.hullRef = hullRef
    }

    public static var inactive: CoinBody {
        CoinBody(position: SIMD3(0, -100_000, 0), invMass: 0)
    }
}

// Static (or kinematic) environment collider. 80 bytes. Matches CoinDEM.metal.
public struct CoinStaticCollider {
    public var a:      SIMD4<Float>   // plane: xyz=normal, w=tag(0) ; box: xyz=centre, w=tag(1)
    public var b:      SIMD4<Float>   // plane: w=offset d ; box: xyz=halfExtents
    public var vel:    SIMD4<Float>   // xyz = kinematic surface velocity; w = cylinder (kind 4) half-length only
    // x = flags (bit0 = one-way ledge). y/z = per-collider (μ, e), bit-cast float
    // — negative = inherit the global uniform, same convention + combine rule
    // (μ=√(μA·μB), e=max(eA,eB)) as per-body material (`friction:`/`restitution:`
    // on every factory below). w reserved.
    public var meta:   SIMD4<UInt32>
    public var orient: SIMD4<Float>   // oriented-box (kind 3) quaternion (x,y,z,w); identity otherwise

    enum Kind: UInt32 { case plane = 0, box = 1, pusherPlate = 2, orientedBox = 3, cylinder = 4 }

    private static let identityQuat = SIMD4<Float>(0, 0, 0, 1)

    private static func matMeta(_ flags: UInt32, friction: Float?, restitution: Float?) -> SIMD4<UInt32> {
        SIMD4(flags, (friction ?? -1).bitPattern, (restitution ?? -1).bitPattern, 0)
    }

    /// A CYLINDER-interior segment — the marble is constrained INSIDE a cylinder of
    /// `radius` whose axis passes through `center` along `axis` for ±`halfLength`.
    /// `lowerHalfOnly` makes it a half-pipe (only the lower hemisphere relative to
    /// `up` pushes; the top is open). Smooth curved surface → no corners to stick on.
    /// Resolved on the constraint path's sphere case (CoinDEM.metal kind 4).
    public static func cylinder(center: SIMD3<Float>, axis: SIMD3<Float>, radius: Float,
                                up: SIMD3<Float>, halfLength: Float,
                                lowerHalfOnly: Bool,
                                friction: Float? = nil, restitution: Float? = nil) -> CoinStaticCollider {
        CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.cylinder.rawValue)),
            b: SIMD4(simd_normalize(axis), radius),
            vel: SIMD4(simd_normalize(up), halfLength),
            meta: matMeta(lowerHalfOnly ? 1 : 0, friction: friction, restitution: restitution),
            orient: identityQuat)
    }

    /// A reciprocating pusher plate that only ever pushes coins FORWARD (+Z),
    /// never down/up — so coins at the paddle/shelf corner can't be squeezed into
    /// the floor and ejected. `center`/`halfExtents` describe the solid plate;
    /// coins within its X/Y footprint that the plate has overtaken are pushed to
    /// just in front of its +Z face — but no faster than `velocity.z` (the plate's
    /// true forward speed), so the contact can't launch coins (see CoinDEM.metal).
    public static func pusherPlate(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                                   velocity: SIMD3<Float> = .zero,
                                   friction: Float? = nil, restitution: Float? = nil) -> CoinStaticCollider {
        CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.pusherPlate.rawValue)),
            b: SIMD4(halfExtents, 0),
            vel: SIMD4(velocity, 0), meta: matMeta(0, friction: friction, restitution: restitution),
            orient: identityQuat)
    }

    /// Half-space `n · x ≥ d`. Coins are pushed back to the surface.
    public static func plane(normal: SIMD3<Float>, offset: Float,
                             friction: Float? = nil, restitution: Float? = nil) -> CoinStaticCollider {
        let n = simd_normalize(normal)
        return CoinStaticCollider(
            a: SIMD4(n, Float(bitPattern: Kind.plane.rawValue)),
            b: SIMD4(0, 0, 0, offset),
            vel: .zero, meta: matMeta(0, friction: friction, restitution: restitution),
            orient: identityQuat)
    }

    /// Axis-aligned box. `oneWay` lets coins fall off the top but not be shoved
    /// back up (the overhang lip). `velocity` is the kinematic surface velocity
    /// (the reciprocating pusher) — but note the shove is mostly implicit via the
    /// box's per-frame advance (see CoinDEM.metal).
    public static func box(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                           oneWay: Bool = false,
                           velocity: SIMD3<Float> = .zero,
                           friction: Float? = nil, restitution: Float? = nil) -> CoinStaticCollider {
        CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.box.rawValue)),
            b: SIMD4(halfExtents, 0),
            vel: SIMD4(velocity, 0),
            meta: matMeta(oneWay ? 1 : 0, friction: friction, restitution: restitution),
            orient: identityQuat)
    }

    /// An ORIENTED box — a box rotated by `orientation` (local→world). The local
    /// axes are X/Y/Z with the given `halfExtents`. Resolved on the constraint
    /// path only (the path the ball/plank scenes use); proper sphere-vs-OBB
    /// contact for marbles rolling on a tilted plank. See CoinDEM.metal kind 3.
    /// `velocity` is its kinematic surface velocity when the host moves it every frame
    /// (a running character's body) — the constraint solve targets it, so a moving box
    /// bats what it hits away at the right speed instead of only pushing it out of overlap.
    public static func orientedBox(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                                   orientation: simd_quatf,
                                   velocity: SIMD3<Float> = .zero,
                                   friction: Float? = nil, restitution: Float? = nil) -> CoinStaticCollider {
        let q = orientation.normalized
        return CoinStaticCollider(
            a: SIMD4(center, Float(bitPattern: Kind.orientedBox.rawValue)),
            b: SIMD4(halfExtents, 0),
            vel: SIMD4(velocity, 0), meta: matMeta(0, friction: friction, restitution: restitution),
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
// 128 bytes (8 × {u,f}16). Mirrors `CoinContact` in CoinDEM.metal exactly.
public struct CoinContact {
    public var meta: SIMD4<UInt32>   // x=A, y=B (0xFFFFFFFF=static), z=collider|feature, w=pairKey
    public var nrm:  SIMD4<Float>    // xyz=normal (B→A), w=depth
    public var rA:   SIMD4<Float>    // xyz=cp−comA, w=normalImpulse
    public var rB:   SIMD4<Float>    // xyz=cp−comB, w=tangent1Impulse
    public var tan1: SIMD4<Float>    // xyz=tangent1, w=tangent2Impulse
    public var tan2: SIMD4<Float>    // xyz=tangent2, w=colour
    // x = pre-solve approach vn₀ (restitution target), yz = accumulated rolling impulse (opt-in),
    // w = captured flag (1 after the first solve pass); BEFORE it, the manifold marker of
    // `manifoldSolve`: 0 ungrouped, −n the head of an n-point manifold, −0.25 a later point.
    public var aux:  SIMD4<Float>
    /// Stage B2 (torsional friction, `setPatchRadius`): x = accumulated torsional impulse
    /// about the normal (N·m·s, + on A), y = the torsion capacity per unit normal impulse
    /// the contact was solved with (μ·r_patch·share, m; 0 = no torsion row), zw reserved.
    public var ext:  SIMD4<Float>
}

// One generic joint (constraint path). 96 bytes (6 × {u,f}16). Mirrors `CoinJoint`
// in CoinDEM.metal exactly. Built via the addBallJoint / addHingeJoint /
// addDistanceJoint / addPrismaticJoint / addWeldJoint APIs (or the pooled enable*
// in CoinDEMSolver+Actuation) — not constructed by hand.
public struct CoinJoint {
    // w bit 0 = enabled, bit 1 = collideConnected (default off — the two jointed
    // bodies don't generate contacts against each other; set to keep them
    // colliding, e.g. a hinge whose bodies are meant to stay physically apart).
    public var meta:    SIMD4<UInt32>  // x=type(0 ball,1 hinge,2 distance,3 prismatic,4 weld), y=A, z=B(0xFFFFFFFF=world), w=enabled|collideConnected bits
    public var anchorA: SIMD4<Float>   // xyz=A-local anchor; w=rest length (distance) or motor target (hinge rad/s, prismatic m/s)
    public var anchorB: SIMD4<Float>   // xyz=B-local anchor (WORLD if z==world); w=motor max torque / force, >0 enables
                                       //   (distance: w = swing-friction arm, m — setDistanceSwingFriction, VZ-0168)
    public var axisA:   SIMD4<Float>   // xyz=A-local hinge/slide axis, w=limit lo (rad / m)
    public var axisB:   SIMD4<Float>   // xyz=B-local hinge axis (WORLD if world), w=limit hi
    /// Relative orientation at creation, conj(qA)·qB (world joint: conj(qA)), as
    /// (x, y, z, w). The weld / prismatic rotation lock and the hinge twist (its limits)
    /// are measured against it (VZ-0156), so a joint holds the pose it was made at.
    public var ref:     SIMD4<Float>
}

// Per-substep uniforms. Scalars only (no float3) — alignment-safe. 136 bytes (34 × 4).
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
    var solverFlags: UInt32 = 0             // CD_FLAG_* opt-in bits (see CoinDEM.metal); 0 = legacy
    var maxBodyBound: Float = 0             // largest tight bounding radius spawned (broadphase reach, VZ-0161)
    var manifoldPasses: UInt32 = 4          // Gauss–Seidel passes over a grouped manifold (manifoldSolve only)
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
    // Hierarchical cell-offset scan (block sums ∥ → short serial block scan →
    // offsets apply ∥) — the serial whole-grid scan tripped the GPU watchdog on
    // house-scale grids (see CoinDEM.metal § cell-offset prefix sum).
    private let blockSumsPipeline:  MTLComputePipelineState
    private let blockScanPipeline:  MTLComputePipelineState
    private let offsetsApplyPipeline: MTLComputePipelineState
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
    private let colorInitPipeline:  MTLComputePipelineState
    private let clearContactCountPipeline: MTLComputePipelineState
    private let colorWritebackPipeline: MTLComputePipelineState
    private let bucketClearPipeline: MTLComputePipelineState
    private let bucketCountPipeline: MTLComputePipelineState
    private let bucketScanPipeline:  MTLComputePipelineState
    private let bucketScatterPipeline: MTLComputePipelineState
    private let writeColorArgsPipeline: MTLComputePipelineState
    private let writeContactArgsPipeline: MTLComputePipelineState
    private let jointSolveCSPipeline: MTLComputePipelineState
    private let jointPreparePipeline: MTLComputePipelineState     // coinJointPrepare (VZ-0150)
    private let jointListUploadPipeline: MTLComputePipelineState  // coinJointListUpload (VZ-0150)
    private let islandUnionJointsPipeline: MTLComputePipelineState
    private let markJointedPipeline: MTLComputePipelineState      // coinMarkJointedBodies (VZ-0153)
    private let islandSleepHubPipeline: MTLComputePipelineState   // coinIslandSleepHub (VZ-0152)
    private let solveTailPipeline: MTLComputePipelineState        // coinSolveVelocityTail (VZ-0154)
    private let polyArgsPipeline: MTLComputePipelineState         // coinWritePolyArgs (stage B1)
    private let polyNarrowPipeline: MTLComputePipelineState       // coinPolyNarrow: one thread per pair
    private let captureApproachPipeline: MTLComputePipelineState  // coinCaptureApproach (manifoldSolve)
    private let colorTentativePipeline: MTLComputePipelineState   // coinColorTentative (VZ-0160, .speculative)
    private let colorResolvePipeline: MTLComputePipelineState     // coinColorResolve   (VZ-0160, .speculative)
    /// The OPTIONAL stage-B3 pipelines — coinSmallWorldFrame and the prepared contact rows' three —
    /// are resolved on FIRST USE (the first frame that opts in), not at init. coinSmallWorldFrame
    /// inlines a whole substep; its cold backend compile is ≈ 6 s on an M1 Max (5.9–6.8 s measured
    /// with the shader cache defeated; the prepared rows' three ≈ 0.25 s together), and resolving
    /// it in init made EVERY solver pay that on the main actor after an engine change — Daydream
    /// Home's egg rain and every scene, though only the Digital Clock takes the path. A kernel the
    /// library lacks, or a device cannot build (the small-world kernel's 25 KB of threadgroup memory
    /// on a 16 KB GPU), resolves to nil once and that path is simply never taken.
    private let resolveOptionalPipeline: (String) -> MTLComputePipelineState?
    private var optionalPipelines: [String: MTLComputePipelineState?] = [:]
    private func optionalPipeline(_ name: String) -> MTLComputePipelineState? {
        if let cached = optionalPipelines[name] { return cached }
        let p = resolveOptionalPipeline(name)
        optionalPipelines[name] = .some(p)
        return p
    }
    /// coinSmallWorldFrame — a whole small-world frame in one dispatch (stage B3; nil when the
    /// library lacks it, and then the small-world path is never taken). Resolved on first access.
    var smallWorldPipeline: MTLComputePipelineState? { optionalPipeline("coinSmallWorldFrame") }
    /// The prepared contact rows (stage B3, opt-in `preparedContactSolve`; Shaders/CoinDEMPreparedSolve.h):
    /// the prepare pass and the prepared solve's per-colour / tail kernels. Optional like the
    /// small-world pipeline — a library without them simply never prepares. Resolved on first access.
    private var prepareContactsPipeline: MTLComputePipelineState? { optionalPipeline("coinPrepareContacts") }
    /// The dormant carry (opt-in `keepSleepingContacts`, CD_FLAG_KEEP_ASLEEP_CONTACTS).
    private var carryDormantPipeline: MTLComputePipelineState? { optionalPipeline("coinCarryDormant") }
    private var solveColorPrepPipeline: MTLComputePipelineState? { optionalPipeline("coinSolveVelocityColorPrep") }
    private var solveTailPrepPipeline: MTLComputePipelineState? { optionalPipeline("coinSolveVelocityTailPrep") }
    /// Per contact point, CD_PREP_F4 (12) float4 — allocated on the first prepared substep.
    private var contactPrepBuffer: MTLBuffer?
    static let contactPrepStride = 192
    /// The fused multi-dispatch substep kernels (stage B3, plan 3b — Shaders/CoinDEMFusedKernels.h):
    /// 15 of the constraint substep's dispatches folded into 5, the same simulation.
    private let substepBeginPipeline: MTLComputePipelineState      // coinSubstepBegin
    private let broadphaseTGPipeline: MTLComputePipelineState      // coinBroadphaseTG
    private let colorPrepareTGPipeline: MTLComputePipelineState    // coinColorPrepareTG
    private let colorFinishTGPipeline: MTLComputePipelineState     // coinColorFinishTG
    private let intPosFinalizePipeline: MTLComputePipelineState    // coinIntegratePositionFinalize
    /// TEST SEAM (stage B3): encode the constraint substep with the one-step kernels the fused
    /// ones replace (A/B parity and cost). Never set in a scene.
    var fusedSubstepKernelsDisabledForTesting = false
    /// The speculative colouring's per-contact bid (UInt32 per contact slot), allocated the
    /// first time `coloringScheme == .speculative` colours anything — a Jones–Plassmann
    /// world (every existing scene, Daydream Home) never allocates it.
    private var colorBidBuffer: MTLBuffer?
    /// The polytope narrowphase's per-substep pair list (uint4 per candidate piece pair),
    /// its append cursor and the indirect args of the per-pair pass — allocated with the
    /// first box / hull / compound, so a world without them (Daydream Home's egg rain)
    /// never pays for it.
    private var polyPairBuffer: MTLBuffer?
    var polyPairCountBuffer: MTLBuffer?               // shared: `polyPairCount` reads it back
    private var polyArgsBuffer: MTLBuffer?
    var maxPolyPairs: Int { maxCoins * 64 }
    /// Piece pairs the last generate pass listed for the polytope narrowphase, UNCLAMPED —
    /// above `maxPolyPairs` the surplus was dropped (those pairs got no contact). Tests /
    /// diagnostics; 0 while no polytope body exists.
    var polyPairCount: Int {
        guard let b = polyPairCountBuffer else { return 0 }
        return Int(b.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
    /// Bound in place of the pair list while none exists (coinGenerateContacts never writes it then).
    private lazy var polyDummyBuffer: MTLBuffer? = device.makeBuffer(length: 32, options: .storageModePrivate)
    private let warmApplyTailPipeline: MTLComputePipelineState    // coinWarmStartApplyTail (VZ-0154)
    /// One MTLDispatchThreadgroupsIndirectArguments per colour, 16-byte stride (the GPU
    /// sizes each colour's solve to that colour's slice; an empty colour gets 0 groups).
    private let solveArgsBuffer: MTLBuffer         // 4×UInt32 × maxColors
    private let solveTGSizeBuffer: MTLBuffer       // 1×UInt32 (threadgroup size for the indirect sizing)
    private static let solveTGSize = 64
    // ── Colouring working set (constraint path) ───────────────────────────────
    private let contactPriorityBuffer: MTLBuffer   // UInt32[maxContacts] — identity-hashed priority
    private let colorBuffers: [MTLBuffer]          // 2 × UInt32[maxContacts] — ping-pong colours
    private let colorCountBuffer: MTLBuffer        // UInt32[maxColors + 1] (count, then scatter cursor; +1 = uncoloured)
    private let colorOffsetBuffer: MTLBuffer       // UInt32[maxColors + 3] — bucket starts (+ uncoloured bucket, + total, + its sub-colour count)
    /// Sub-colour of each uncoloured-bucket entry (by position in `colorContactsBuffer`),
    /// written by coinWriteColorArgs, read by the tail passes (VZ-0154).
    private let uncolSubBuffer: MTLBuffer          // UInt32[maxContacts]
    private let colorContactsBuffer: MTLBuffer     // UInt32[maxContacts] — contacts sorted by colour
    let colorStatsBuffer: MTLBuffer                // UInt32[5]: overflow / uncoloured / maxColourUsed / beyondSweep / uncolouredUnsolved
    /// How many colours the solve sweep encodes as separate parallel dispatches. The
    /// colouring takes the lowest free colour, so the live colours are 0…maxUsed with
    /// nothing above; dispatching all `maxColors` would pay ~50 empty dispatch bubbles
    /// per velocity iteration. Sized from the highest colour the GPU has reported so far
    /// (`colorStats.maxColorUsed`) plus headroom; starts at 0 (no contact seen yet).
    /// Colours at or above it are solved by the one-threadgroup tail pass, so the
    /// sweep never drops a contact (VZ-0154).
    private var colorSweep = 0
    /// Spare colours dispatched beyond the worst seen, so a moment of denser packing
    /// (which needs one colour per extra contact on the busiest body) is already covered.
    /// Was 8: each empty colour is a 0-threadgroup indirect dispatch (≈3.3 µs on M1 Max)
    /// per velocity iteration per substep, ≈0.7–1.1 ms/frame at 1/180 s × 6 — and with
    /// the serial tail pass behind the sweep, headroom is a speed bet, not a safety
    /// margin (VZ-0154).
    private static let colorSweepHeadroom = 1
    private let contactArgsBuffer: MTLBuffer       // threadgroup count for contact-sized passes

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
    let linkBuffer: MTLBuffer
    /// Per-body material (SIMD2<Float> per slot): x = Coulomb friction μ, y =
    /// restitution e. A NEGATIVE lane means "inherit the global uniform"
    /// (`frictionCoeff` / `restitution`) — the default, so untouched scenes are
    /// unchanged. Constraint path only; combined per contact as μ=√(μA·μB),
    /// e=max(eA,eB). Set at spawn (`friction:`/`restitution:`) or via `setMaterial`.
    public let materialBuffer: MTLBuffer
    /// Per-body contact PATCH radius (Float per slot, m; 0 = a point contact, the default):
    /// the effective moment arm of torsional friction about each of the body's contact
    /// normals (stage B2, engine plan 5a — `setPatchRadius`, CoinDEMSolver+Torsion.swift).
    /// Bound to the solve kernels always; read only while `torsionPatchSlots` is non-empty.
    let patchRadiusBuffer: MTLBuffer
    /// Slots with a patch radius > 0 (CD_FLAG_TORSION is set iff non-empty).
    var torsionPatchSlots: Set<Int> = []
    let colliderBuffer: SimBuffer<CoinStaticCollider>
    private let coinDeltaBuffer: MTLBuffer         // private — per-coin Jacobi delta
    private let cellCounts: MTLBuffer              // private
    private let cellOffsets: MTLBuffer             // private
    /// Per-block partial sums for the hierarchical cell-offset scan (one uint per
    /// CD_SCAN_BLOCK=1024 cells).
    private let cellBlockSums: MTLBuffer           // private
    private var numScanBlocks: Int { (numCells + 1023) / 1024 }
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
    /// Mirrors `CD_MAX_BODY_CONTACTS` in CoinDEM.metal — must exceed the true per-body
    /// contact degree or the colouring races (see the shader comment).
    private static let maxBodyContacts = 64
    /// `maxBodyContacts` for the small-world arena layout (CoinDEMSolver+SmallWorld.swift).
    static var bodyContactCapacity: Int { maxBodyContacts }
    /// Per-body split-impulse BIAS (pseudo) velocity: 2 × float4 per body
    /// (linear, angular). Cleared each substep; moves position then is discarded.
    let biasBuffer: MTLBuffer                      // maxCoins × 2 × SIMD4<Float>
    /// Warm starting: last substep's solved contacts + an open-addressing hash
    /// (pairKey → prev slot) so a re-found contact resumes from its converged impulse.
    let prevContactBuffer: MTLBuffer               // CoinContact[maxContacts]
    let pairHashBuffer: MTLBuffer                  // UInt32[hashSize] (power of two)
    private let hashSizeBuffer: MTLBuffer          // 1×UInt32 (constant)
    let hashSize: Int
    /// Island sleeping: connected-component label + per-island min sleep-timer
    /// (scratch, recomputed each frame), a persistent per-body slow-frame counter,
    /// and the asleep flag the solver kernels gate on (shared → host can read the count).
    private let islandLabelBuffer: MTLBuffer       // maxCoins UInt32
    private let islandMinBuffer: MTLBuffer         // maxCoins UInt32
    let sleepTimerBuffer: MTLBuffer                // maxCoins UInt32 (persists)
    public let asleepBuffer: MTLBuffer             // maxCoins UInt32 (0/1)
    /// Sleep groups (VZ-0152): the island label a body fell asleep under (0xFFFF_FFFF
    /// while awake; persists), and the per-group hub rebuilt each sleep update. Asleep
    /// pairs no longer generate contacts, so these carry an asleep heap's connectivity:
    /// a body that touches any member wakes the whole group in the same frame.
    let sleepKeyBuffer: MTLBuffer                  // maxCoins UInt32 (persists; shared for the init fill)
    private let sleepHubBuffer: MTLBuffer          // maxCoins UInt32 (scratch)
    /// Per body: 1 iff it is an end of an enabled, non-collideConnected two-body joint
    /// (coinMarkJointedBodies, once per frame) — gates the generate kernel's joint scan.
    private let jointedBodyBuffer: MTLBuffer       // maxCoins UInt32
    /// Capacity of `contactBuffer`. A dense mixed pile emits up to ~tens of contacts
    /// per body (box manifolds + statics), so size generously (maxCoins × 64).
    public let maxContacts: Int

    // ── Convex hulls (constraint path) ────────────────────────────────────────
    /// Registered hull vertices (principal frame, COM at origin), all hulls
    /// packed into one buffer; per-hull (offset, count) ranges beside it.
    /// The SHAPE TABLE: every registered hull's principal-frame vertices followed by its
    /// face/edge topology (CoinHullTopology.pack), and every registered compound's child
    /// boxes (3 slots each) — one float4 buffer, so every kernel that already binds it
    /// sees both. `hullRangeBuffer[i]` = (first slot, vertex / child count) of entry i.
    public let hullVertexBuffer: MTLBuffer      // SIMD4<Float> × maxHullVertices (table slots)
    public let hullRangeBuffer: MTLBuffer       // SIMD2<UInt32> × maxHulls (table entries)
    /// Shape-table ENTRIES (hulls + compounds share the numbering).
    public static let maxHulls = 64
    /// Shape-table float4 SLOTS: a hull takes its vertices + 1 header + its packed
    /// topology (the Digital Clock bar: 32 + 1 + 136); a compound 3 per child box.
    public static let maxHullVertices = 16384
    /// Entry i of the shape table: a hull (spawnHull) or a compound (spawnCompound).
    enum ShapeTableEntry {
        case hull(CoinHullMath.Prepared)
        case compound(CoinCompoundMath.Prepared)
    }
    var shapeTable: [ShapeTableEntry] = []
    var hullVertexCursor = 0
    /// Frame change applied at registration, returned so callers can express
    /// their render mesh in the same (principal, COM-origin) frame the solver
    /// simulates in: stored = principalRotation⁻¹ · (input − comOffset).
    public struct HullHandle {
        public let index: Int
        public let vertices: [SIMD3<Float>]          // the stored principal-frame hull
        public let comOffset: SIMD3<Float>
        public let principalRotation: simd_quatf
        public let boundingRadius: Float
    }

    // ── Generic joints (constraint path) ──────────────────────────────────────
    /// Joint slots (CoinJoint each; meta.w == 0 = disabled/free). Host-owned: the GPU
    /// only reads it. Each ENABLED joint is solved as one block (all of its rows in one
    /// LDLᵀ solve, lever coupling included) with its impulses warm-started across
    /// substeps, serially in slot order once per velocity iteration, interleaved with
    /// the contact colours; joint edges union into the sleep islands so an articulated
    /// assembly sleeps and wakes as one (VZ-0150).
    public let jointBuffer: MTLBuffer
    public static let maxJoints = 1024
    private var freeJointSlots: [Int] = []
    private var jointHighWater: Int = 0
    /// Joint-table high-water mark (enabled and disabled slots). The joint kernels read
    /// only the ENABLED ones (the per-frame active list), so reserved-but-disabled pool
    /// slots cost nothing (VZ-0150 item 4c). For tests / instrumentation.
    public var jointCount: Int { jointHighWater }
    /// FOOT HOOK — spring feet (CoinDEMSolver+Foot.swift, Shaders/CoinDEMFoot.h): nil until the
    /// first `addFoot`; a solver without an active foot never dispatches anything for them.
    var footSystem: CoinFootSystem?
    /// SMALL-WORLD HOOK — the one-dispatch frame path (CoinDEMSolver+SmallWorld.swift,
    /// Shaders/CoinDEMSmallWorld.h, stage B3): its opt-in flag, limits and scratch.
    let smallWorld = CoinSmallWorldState()
    /// Joint passes per velocity iteration (≥ 1): the serial joint loop runs this many
    /// times inside its one dispatch, so a long jointed chain can converge further than
    /// the contacts without paying another round of contact-colour dispatches. Each
    /// joint is already exact on its own; the passes only speed up joint-to-joint
    /// Gauss–Seidel. Default 1.
    public var jointInnerPasses: Int = 1
    /// The compact list of enabled joint slots for the current frame (private; written
    /// by coinJointListUpload from the host's scan) and the per-slot block-solve scratch
    /// + warm-start impulses (private; allocated with the first enabled joint, so a
    /// solver that never uses joints never pays for it).
    let jointListBuffer: MTLBuffer                  // UInt32 × (4·maxJoints + 4): slots, TG-cache pairs, TG-cache bodies, active list
    var jointPrepBuffer: MTLBuffer?                 // jointPrepStride × maxJoints
    /// `sizeof(CoinJointPrep)` in CoinDEM.metal (static_assert there).
    static let jointPrepStride = 496
    /// Geometry of each slot at the last list upload. A slot that was not enabled then,
    /// or whose bodies / anchors / axes / reference orientation changed since, is uploaded
    /// with the reset flag so its warm-start impulses start from zero (a pooled slot
    /// re-enabled between two other bodies must not inherit the old pair's load).
    private struct JointKey: Equatable {
        var meta: SIMD3<UInt32>
        var anchorA, anchorB, axisA, axisB: SIMD3<Float>
        var ref: SIMD4<Float>
        init(_ j: CoinJoint) {
            meta = SIMD3(j.meta.x, j.meta.y, j.meta.z)
            anchorA = SIMD3(j.anchorA.x, j.anchorA.y, j.anchorA.z)
            anchorB = SIMD3(j.anchorB.x, j.anchorB.y, j.anchorB.z)
            axisA = SIMD3(j.axisA.x, j.axisA.y, j.axisA.z)
            axisB = SIMD3(j.axisB.x, j.axisB.y, j.axisB.z)
            ref = j.ref
        }
    }
    private var jointKeys: [Int: JointKey] = [:]
    /// Active joints uploaded for the frame being encoded (0 ⇒ no joint work at all).
    var frameJointCount = 0
    /// Distinct bodies the frame's active joints join, when joints and bodies fit the
    /// joint solve's threadgroup cache (CD_JSOLVE_MAXJ / CD_JSOLVE_MAXB in CoinDEM.metal);
    /// 0 ⇒ the direct (device-memory) path.
    var frameJointTGBodies = 0
    static let jointTGMaxJoints = 40
    static let jointTGMaxBodies = 80
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
    /// Constraint path, OPT-IN (VZ-0155): clamp the ACCUMULATED rolling-resistance
    /// impulse of each contact over the substep's velocity iterations (standard
    /// sequential impulse, exactly like the friction rows), so `rollingResistance` is
    /// a physical coefficient μr — resisting torque ≤ μr·N·|r| — whatever
    /// `velocityIterations` is. OFF (legacy): each iteration's impulse is clamped on its
    /// own, so the per-substep cap is velocityIterations × μr·N·|r| and a rolling body
    /// parks/stops as if μr were iterations× larger. Every shipping scene's
    /// `rollingResistance` was tuned against the legacy behaviour (Marbles 0.11 × 12
    /// iterations, SuperquadricLab 0.04 × 16, Vintage Diner Ultra 0.035 × 8, Daydream
    /// Home egg rain 0.02 × 4), so turning this on there needs μr retuned ≈ ×iterations.
    public var accumulatedRollingResistance: Bool = false
    /// Constraint path, OPT-IN (VZ-0149): the finalize dead-stop and the island-sleep
    /// "slow" test compare the body's fastest SURFACE speed |v| + |ω|·R_bound with
    /// `sleepLinVel` (dead-stop) / 2.5·`sleepLinVel` (sleep), instead of the legacy
    /// |v| < sleepLinVel && |ω| < 0.6 (1.0 for sleep) — an absolute angular threshold
    /// that zeroes any body turning slower than 0.6 rad/s about its own COM (a 0.52 m
    /// crane jib at 0.4 rad/s has a 0.1 m/s tip speed). Off by default: the sleep-off
    /// scenes (Vintage Diner Ultra, SuperquadricLab) rely on the legacy dead-stop to
    /// quiet large resting bodies. Independent of this flag, a body driven by an
    /// enabled hinge/prismatic MOTOR with a non-zero target is never dead-stopped and
    /// never counts as slow for sleep — unless it is pushing into a limit it has already
    /// reached (holding, not moving) — so set the target to 0 (a brake) to let it rest.
    public var scaleAwareDeadStop: Bool = false
    /// Constraint path, OPT-IN (VZ-0157, VZ-0163): solve every multi-point contact MANIFOLD
    /// (≤ 4 points of one body pair — a box on a table, a bar in a V, a cube on a cube) as
    /// ONE unit. Its points are emitted contiguously, coloured as one node, and solved by one
    /// thread with the pair's velocities in registers: `manifoldInnerPasses` Gauss–Seidel
    /// passes over its normal / split-impulse / friction rows per velocity iteration, then
    /// one rolling-resistance pass. OFF (default): each point is its own colouring node and
    /// gets one visit per iteration — which leaves a 4-point manifold (a redundant, rank-3
    /// system on a rigid body) asymmetric at low iteration counts: a 10 mm cube resting on a
    /// cube got ω = 0.48 rad/s from its first cold 6-iteration solve (a 5-cube tower fell at
    /// frame 3), and a lone resting box yawed ~1°/s. Manifolds also cost one colour node
    /// instead of four, so the colour sweep shrinks. It also captures EVERY contact's
    /// restitution approach speed before the substep's first impulse (coinCaptureApproach)
    /// — the solve's lazy capture, in colour order, gave a flat landing's later corners a
    /// speed their predecessors had already cut, and the one-sided bounce tipped a dropped
    /// L 0.5°. Every existing scene keeps it off.
    public var manifoldSolve: Bool = false
    /// Constraint path, OPT-IN: the split-impulse BIAS row only ever pushes a contact's bodies
    /// apart (CD_FLAG_SEPARATING_BIAS, CoinDEM.metal). Off (the default, every existing world): the
    /// row drives the relative bias velocity to its target both ways, so a SPECULATIVE contact
    /// (two bodies within `speculativeMargin` but not touching) or one within the slop PULLS its
    /// bodies together in the position channel whenever another contact's recovery moves one of
    /// them away. With stacked bodies and a margin wider than the gaps between neighbours (a 2 × 5
    /// on-edge bar stack on a dynamic stake pallet, 0.5 mm between rows and stakes, margin 2.5 mm)
    /// that glue kept the stack jiggling and walking and it never slept; the flag makes the row
    /// what its documentation always said it was ("the pseudo impulse stays separating"), and a
    /// speculative contact resists a bias approach only beyond its own gap.
    public var separatingSplitImpulse: Bool = false
    /// Constraint path + `warmStart`, OPT-IN: a sleeping island keeps its warm start
    /// (CD_FLAG_KEEP_ASLEEP_CONTACTS, CoinDEM.metal `coinCarryDormant`). Off (the default, every
    /// existing world): VZ-0152 skips generating the contacts of asleep pairs (and asleep vs static),
    /// so they drop out of the warm-start table the substep after an island falls asleep, and every
    /// WAKE starts all of its contacts COLD. A tall stack does not survive that: Digital Clock's
    /// 2 × 5 stock of on-edge bars, woken after a sleep (the crane's magnet touching its top bar),
    /// sank at 14–48 mm/s in its first frame (a cold 6-iteration Gauss–Seidel pass cannot hold a
    /// 6-deep chain) and crept 0.7–1.4 mm; woken while its contacts were still in the table it moved
    /// 0.0003 mm. With the flag, each substep carries the last substep's contacts whose ends are
    /// both still inert into the new list as DORMANT copies (no narrowphase — VZ-0152 still skips
    /// the pairs; never coloured or solved), so the table keeps them for as long as the island
    /// sleeps and a wake resumes from its converged impulses. Cost: a copy of the dormant records
    /// per substep (one thread per hash slot).
    public var keepSleepingContacts: Bool = false
    /// Gauss–Seidel passes over a grouped manifold's points per velocity iteration
    /// (`manifoldSolve` only). ≥ 1; default 4.
    public var manifoldInnerPasses: Int = 4
    /// TEST SEAM (VZ-0151): skip coinGenerateContacts' per-collider bounding reject, so a
    /// test can prove the reject leaves the contact set bit-identical. Never set in a scene.
    var colliderCullDisabledForTesting: Bool = false
    /// Tests only: false registers compounds WITHOUT culling clusters (every child walked, as a
    /// compound of ≤ 16 children is) — the contact set must not change (CoinDEMSolver+Compound).
    var compoundClustersForTesting: Bool = true
    /// TEST SEAMS (stage B1 cost attribution): scan ±1 cell whatever the bodies' size, and
    /// skip the polytope narrowphase dispatch. Never set in a scene.
    var broadphaseClassicScanForTesting: Bool = false
    var polyKernelDisabledForTesting: Bool = false
    /// TEST SEAM (stage B3, VZ-0198): run the polytope narrowphase's separating-axis test on ONE
    /// thread (cdPolySAT, lane 0) instead of spread over the SIMD group (cdPolySATSG) — the A/B
    /// that proves the two bit-identical. Never set in a scene.
    var polySATSerialForTesting: Bool = false
    /// TEST SEAM (VZ-0150): force coinJointSolveCS's direct (device-memory) path, so a
    /// test can prove the threadgroup-cached path gives the same result. Never set in a scene.
    var jointThreadgroupCacheDisabledForTesting: Bool = false
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
    /// The colouring's colour cap — mirrors `CD_MAX_COLORS` in CoinDEM.metal, which is
    /// the source of truth (a colour has to fit the shader's `ulong` usedMask). It must
    /// clear the max per-body contact degree, since a body of degree d forces d distinct
    /// colours among its contacts; the shipped dense mixed pile peaks at 13. Only the
    /// colours actually in use are dispatched — see `colorSweep`.
    public static let maxColors = 64

    /// Warm starting: carry each contact's converged impulse across substeps so a
    /// resting stack resumes from its solution instead of re-solving from zero.
    /// OPT-IN (default off): it's the biggest convergence win for TALL single-column
    /// stacks, but on a dense MIXED heap the colored Gauss-Seidel already settles dead-
    /// still on its own, and a stale warm seed (a contact whose feature shifted between
    /// substeps) can occasionally leave the heap un-settled. Scenes that build towers
    /// opt in; the loose-heap scenes (Pile of Mess) leave it off — GS alone is enough.
    public var warmStart: Bool = false
    var needsHashClear = true                   // lazily zero the pair hash on first use

    /// OPT-IN (default off; stage B3 — Shaders/CoinDEMPreparedSolve.h): compute every contact row's
    /// pose-constant factors (effective masses, each body's angular response I⁻¹(r×d)) ONCE per
    /// substep, in a prepare pass, instead of on every row of every velocity iteration and manifold
    /// pass. The same rows, bounds, targets and order — but the arithmetic is regrouped
    /// (I⁻¹(r×(λd)) → λ·I⁻¹(r×d)) and Metal's fast math rounds it differently, so it is opt-in and
    /// every world that does not set it steps exactly as before. Measured on the Digital Clock's perf
    /// fence (small-world path, M1 Max, the two interleaved in one process): the contact solve of a
    /// frame where the crane's box manifold touches went from ≈ 233 µs per substep to ≈ 136 µs (plus
    /// ≈ 11 µs for the prepare pass); frames with 1–4 single contacts 69 → 52 µs; the fence scene
    /// configuration's physics p95 3.83 → 3.31 ms (see docs/coindem-constraint-solver.md, stage B3).
    /// `VIZ_COINDEM_PREPARED=1` makes it every solver's default for a test run.
    public var preparedContactSolve: Bool = CoinDEMSolver.preparedEnvironmentDefault
    static let preparedEnvironmentDefault = ProcessInfo.processInfo.environment["VIZ_COINDEM_PREPARED"] == "1"
    /// The prepared rows are in use this substep (the knob, the constraint path, the pipelines and
    /// the buffer all present).
    var preparedContactsLive: Bool {
        preparedContactSolve && solverMode == .constraint && contactPrepBuffer != nil
            && prepareContactsPipeline != nil && solveColorPrepPipeline != nil && solveTailPrepPipeline != nil
    }
    /// Make the prepare buffer before a frame that needs it (so the uniforms' flag and the bound
    /// buffer agree for the whole frame).
    func ensureContactPrepBuffer() {
        guard preparedContactSolve, contactPrepBuffer == nil, prepareContactsPipeline != nil else { return }
        contactPrepBuffer = device.makeBuffer(length: Self.contactPrepStride * maxContacts, options: .storageModePrivate)
        contactPrepBuffer?.label = "Coin.contactPrep"
    }
    /// The prepare buffer for the small-world kernel's binding (nil until first used).
    var contactPrepBufferForBinding: MTLBuffer? { contactPrepBuffer }
    /// Whether an optional (stage-B3) pipeline has been resolved yet — tests: a solver must not
    /// compile a path it has not opted into.
    func optionalPipelineResolvedForTesting(_ name: String) -> Bool { optionalPipelines[name] != nil }
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

    var colliders: [CoinStaticCollider] = []

    /// Largest TIGHT bounding radius of any body spawned since the last `clearAll` (a
    /// sphere's R, a box's |he|, a capsule's hl + r, a disc's √(R² + h²), an egg's / hull's /
    /// compound's bounding radius). The broadphase scans ±1 cell while this fits the cell
    /// (every correctly sized scene: bit-identical to before); a body that outgrew the
    /// cell widens the neighbourhood instead of silently missing pairs two cells apart
    /// (VZ-0161: a 0.08 m capsule in a 0.12 m cell, 20–30 mm overlaps with no contact).
    private(set) var maxBodyBound: Float = 0
    /// Any box / hull / compound spawned since the last `clearAll` — only then are the
    /// polytope pair list allocated and its narrowphase (coinPolyNarrow) dispatched.
    private(set) var hasPolyBodies = false
    func noteBodyBound(_ r: Float, poly: Bool = true) {
        maxBodyBound = max(maxBodyBound, r)
        if poly { hasPolyBodies = true }
    }

    // ── Init ──────────────────────────────────────────────────────────────────

    struct Pipelines {
        let integrate, cellClear, cellCount, scatter: MTLComputePipelineState
        let blockSums, blockScan, offsetsApply: MTLComputePipelineState
        let contact, apply, finalize, orient, transform: MTLComputePipelineState
        let joint: MTLComputePipelineState
        let measure: MTLComputePipelineState   // coinMeasurePenetration (diagnostic)
        let generate: MTLComputePipelineState  // coinGenerateContacts (constraint solver)
        let clearBody, buildBody, colorRound: MTLComputePipelineState   // graph colouring
        let colorInit, colorWriteback: MTLComputePipelineState           // ping-pong colouring
        let clearContactCount: MTLComputePipelineState
        let bucketClear, bucketCount, bucketScan, bucketScatter: MTLComputePipelineState  // colour compaction
        let writeColorArgs, writeContactArgs: MTLComputePipelineState
        let intVelCS, solveVelCS, intPosCS, finalizeCS: MTLComputePipelineState  // sequential impulse
        let clearHash, snapshot, warmMatch, warmApply: MTLComputePipelineState   // warm starting
        let islandInit, islandUnion, islandJump: MTLComputePipelineState          // island sleeping
        let sleepTick, islandMinReduce, sleepMark: MTLComputePipelineState
        let gjkEPA: MTLComputePipelineState                                        // GJK/EPA probe
        let jointSolveCS, islandUnionJoints: MTLComputePipelineState               // generic joints
        let jointPrepare, jointListUpload: MTLComputePipelineState                 // VZ-0150
        let markJointed, islandSleepHub: MTLComputePipelineState                   // VZ-0153 / VZ-0152
        let solveTail, warmApplyTail: MTLComputePipelineState                      // VZ-0154
        let polyArgs, polyNarrow: MTLComputePipelineState                          // stage B1 polytope narrowphase
        let captureApproach: MTLComputePipelineState                               // stage B1 restitution capture
        let colorTentative, colorResolve: MTLComputePipelineState                  // VZ-0160 speculative colouring (B2)
        /// The OPTIONAL stage-B3 kernels (coinSmallWorldFrame, the prepared contact rows) are not
        /// resolved here: the solver keeps this lookup and resolves each on first use (see
        /// `optionalPipeline`). A device or library without one simply never takes that path.
        var resolveOptional: (String) -> MTLComputePipelineState? = { _ in nil }
        /// The fused multi-dispatch substep kernels (stage B3, plan 3b; Shaders/CoinDEMFusedKernels.h).
        let substepBegin, broadphaseTG, colorPrepareTG, colorFinishTG, intPosFinalize: MTLComputePipelineState
    }

    /// Resolve the kernels through an arbitrary lookup (engine cache in production; a
    /// runtime-compiled library in the headless tests).
    static func makePipelines(_ resolve: @escaping (String) -> MTLComputePipelineState?) -> Pipelines? {
        guard
            let p0 = resolve("coinIntegrate"),
            let p1 = resolve("coinCellClear"),
            let p2 = resolve("coinCellCount"),
            let pS1 = resolve("coinCellBlockSums"),
            let pS2 = resolve("coinCellBlockScan"),
            let pS3 = resolve("coinCellOffsetsApply"),
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
            let p31 = resolve("coinWriteColorArgs"),
            let p34 = resolve("coinColorInit"),
            let p35 = resolve("coinColorWriteback"),
            let p36 = resolve("coinColorBucketClear"),
            let p37 = resolve("coinColorBucketCount"),
            let p38 = resolve("coinColorBucketScan"),
            let p39 = resolve("coinColorBucketScatter"),
            let p40 = resolve("coinWriteContactArgs"),
            let p41 = resolve("coinClearContactCount"),
            let p32 = resolve("coinJointSolveCS"),
            let p33 = resolve("coinIslandUnionJoints"),
            let p42 = resolve("coinMarkJointedBodies"),
            let p43 = resolve("coinIslandSleepHub"),
            let p44 = resolve("coinSolveVelocityTail"),
            let p45 = resolve("coinWarmStartApplyTail"),
            let p46 = resolve("coinJointPrepare"),
            let p47 = resolve("coinJointListUpload"),
            let p49 = resolve("coinWritePolyArgs"),
            let p50 = resolve("coinPolyNarrow"),
            let p51 = resolve("coinCaptureApproach"),
            let p52 = resolve("coinColorTentative"),
            let p53 = resolve("coinColorResolve"),
            let p54 = resolve("coinSubstepBegin"),
            let p55 = resolve("coinBroadphaseTG"),
            let p56 = resolve("coinColorPrepareTG"),
            let p57 = resolve("coinColorFinishTG"),
            let p58 = resolve("coinIntegratePositionFinalize")
        else { return nil }
        var pipelines = Pipelines(integrate: p0, cellClear: p1, cellCount: p2, scatter: p4,
                         blockSums: pS1, blockScan: pS2, offsetsApply: pS3,
                         contact: p5, apply: p6, finalize: p7, orient: p8, transform: p9, joint: p10,
                         measure: p11, generate: p12, clearBody: p13, buildBody: p14, colorRound: p15,
                         colorInit: p34, colorWriteback: p35, clearContactCount: p41,
                         bucketClear: p36, bucketCount: p37, bucketScan: p38, bucketScatter: p39,
                         writeColorArgs: p31, writeContactArgs: p40,
                         intVelCS: p16, solveVelCS: p17, intPosCS: p18, finalizeCS: p19,
                         clearHash: p20, snapshot: p21, warmMatch: p22, warmApply: p23,
                         islandInit: p24, islandUnion: p25, islandJump: p26,
                         sleepTick: p27, islandMinReduce: p28, sleepMark: p29, gjkEPA: p30,
                         jointSolveCS: p32, islandUnionJoints: p33,
                         jointPrepare: p46, jointListUpload: p47,
                         markJointed: p42, islandSleepHub: p43, solveTail: p44, warmApplyTail: p45,
                         polyArgs: p49, polyNarrow: p50, captureApproach: p51,
                         colorTentative: p52, colorResolve: p53,
                         substepBegin: p54, broadphaseTG: p55, colorPrepareTG: p56, colorFinishTG: p57,
                         intPosFinalize: p58)
        pipelines.resolveOptional = resolve
        return pipelines
    }

    /// Production init: pipelines come from the engine's memoised cache (the
    /// package's compiled metallib via `Bundle.module`).
    /// `maxColliders` sizes the static/kinematic collider buffer (`setColliders`): a scene
    /// whose surfaces are many facets (a lofted roof, polygon-prism poles) plus per-frame
    /// moving colliders can need more than the default 256. The kernels loop over the live
    /// count, so capacity costs only memory.
    public convenience init?(engine: SimEngine = .shared,
                             maxCoins: Int,
                             coinRadius: Float,
                             halfThickness: Float,
                             boundsMin: SIMD3<Float>,
                             boundsMax: SIMD3<Float>,
                             maxColliders: Int = 256) {
        guard let pipelines = CoinDEMSolver.makePipelines({ engine.pipeline($0) }) else {
            CoinDEMSolver.log.error("Coin pipeline cache failed — check CoinDEM.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        self.init(engine: engine, pipelines: pipelines, maxCoins: maxCoins,
                  coinRadius: coinRadius, halfThickness: halfThickness,
                  boundsMin: boundsMin, boundsMax: boundsMax, maxColliders: maxColliders)
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
          boundsMax: SIMD3<Float>,
          maxColliders: Int = 256) {
        let p0 = pipelines.integrate, p1 = pipelines.cellClear, p2 = pipelines.cellCount
        let p4 = pipelines.scatter, p5 = pipelines.contact
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
        let p30 = pipelines.gjkEPA, p31 = pipelines.writeColorArgs
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
            let cols  = SimBuffer<CoinStaticCollider>(device: dev, capacity: max(1, maxColliders), label: "Coin.colliders"),
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
            let blockSums = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * ((cellCount + 1023) / 1024 + 1),
                                           options: .storageModePrivate),
            let sorted = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins,
                                        options: .storageModePrivate),
            let btype = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins,
                                       options: .storageModeShared),
            let linkB = dev.makeBuffer(length: MemoryLayout<SIMD2<Int32>>.stride * maxCoins,
                                       options: .storageModeShared),
            let matB = dev.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * maxCoins,
                                      options: .storageModeShared),
            let patchB = dev.makeBuffer(length: MemoryLayout<Float>.stride * maxCoins,
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
            let sleepKey = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModeShared),
            let sleepHub = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModePrivate),
            let jointedBody = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * maxCoins, options: .storageModePrivate),
            let solveArgs = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * 4 * CoinDEMSolver.maxColors,
                                           options: .storageModePrivate),
            let contactPriority = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * contactCap,
                                                 options: .storageModePrivate),
            let colorA = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * contactCap,
                                        options: .storageModePrivate),
            let colorB = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * contactCap,
                                        options: .storageModePrivate),
            let colorCount = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * (CoinDEMSolver.maxColors + 1),
                                            options: .storageModePrivate),
            let colorOffset = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * (CoinDEMSolver.maxColors + 3),
                                             options: .storageModePrivate),
            let uncolSub = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * contactCap,
                                          options: .storageModePrivate),
            let colorContacts = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * contactCap,
                                               options: .storageModePrivate),
            let colorStats = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * 5,
                                            options: .storageModeShared),
            let contactArgs = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * 4,
                                             options: .storageModePrivate),
            let solveTG = dev.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
            let jointsBuf = dev.makeBuffer(length: MemoryLayout<CoinJoint>.stride * CoinDEMSolver.maxJoints,
                                           options: .storageModeShared),
            // slots, TG-cache pairs, TG-cache bodies, the substep's active list (+ its count)
            let jointList = dev.makeBuffer(length: MemoryLayout<UInt32>.stride * (4 * CoinDEMSolver.maxJoints + 4),
                                           options: .storageModePrivate),
            let hullV = dev.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * CoinDEMSolver.maxHullVertices,
                                       options: .storageModeShared),
            let hullR = dev.makeBuffer(length: MemoryLayout<SIMD2<UInt32>>.stride * CoinDEMSolver.maxHulls,
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
        self.blockSumsPipeline = pipelines.blockSums
        self.blockScanPipeline = pipelines.blockScan
        self.offsetsApplyPipeline = pipelines.offsetsApply
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
        self.writeColorArgsPipeline = p31
        self.writeContactArgsPipeline = pipelines.writeContactArgs
        self.colorInitPipeline = pipelines.colorInit
        self.clearContactCountPipeline = pipelines.clearContactCount
        self.colorWritebackPipeline = pipelines.colorWriteback
        self.bucketClearPipeline = pipelines.bucketClear
        self.bucketCountPipeline = pipelines.bucketCount
        self.bucketScanPipeline = pipelines.bucketScan
        self.bucketScatterPipeline = pipelines.bucketScatter
        self.jointSolveCSPipeline = p32
        self.jointPreparePipeline = pipelines.jointPrepare
        self.jointListUploadPipeline = pipelines.jointListUpload
        self.islandUnionJointsPipeline = p33
        self.markJointedPipeline = pipelines.markJointed
        self.islandSleepHubPipeline = pipelines.islandSleepHub
        self.solveTailPipeline = pipelines.solveTail
        self.warmApplyTailPipeline = pipelines.warmApplyTail
        self.polyArgsPipeline = pipelines.polyArgs
        self.polyNarrowPipeline = pipelines.polyNarrow
        self.captureApproachPipeline = pipelines.captureApproach
        self.colorTentativePipeline = pipelines.colorTentative
        self.colorResolvePipeline = pipelines.colorResolve
        self.resolveOptionalPipeline = pipelines.resolveOptional
        self.substepBeginPipeline = pipelines.substepBegin
        self.broadphaseTGPipeline = pipelines.broadphaseTG
        self.colorPrepareTGPipeline = pipelines.colorPrepareTG
        self.colorFinishTGPipeline = pipelines.colorFinishTG
        self.intPosFinalizePipeline = pipelines.intPosFinalize
        self.coinBuffer = coins
        self.transformBuffer = xform
        self.bodyTypeBuffer = btype
        self.linkBuffer = linkB
        matB.label = "Coin.material"
        self.materialBuffer = matB
        patchB.label = "Coin.patchRadius"
        memset(patchB.contents(), 0, patchB.length)
        self.patchRadiusBuffer = patchB
        self.colliderBuffer = cols
        self.coinDeltaBuffer = delta
        self.cellCounts = counts
        self.cellOffsets = offsets
        blockSums.label = "Coin.cellBlockSums"
        self.cellBlockSums = blockSums
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
        sleepKey.label = "Coin.sleepKey"; sleepHub.label = "Coin.sleepHub"; jointedBody.label = "Coin.jointedBody"
        self.sleepKeyBuffer = sleepKey
        self.sleepHubBuffer = sleepHub
        self.jointedBodyBuffer = jointedBody
        solveArgs.label = "Coin.solveArgs"
        self.solveArgsBuffer = solveArgs
        self.contactPriorityBuffer = contactPriority
        self.colorBuffers = [colorA, colorB]
        self.colorCountBuffer = colorCount
        self.colorOffsetBuffer = colorOffset
        uncolSub.label = "Coin.uncolouredSub"
        self.uncolSubBuffer = uncolSub
        self.colorContactsBuffer = colorContacts
        self.colorStatsBuffer = colorStats
        self.contactArgsBuffer = contactArgs
        colorStats.contents().bindMemory(to: UInt32.self, capacity: 5).update(repeating: 0, count: 5)
        self.solveTGSizeBuffer = solveTG
        jointsBuf.label = "Coin.joints"
        self.jointBuffer = jointsBuf
        jointList.label = "Coin.jointList"
        self.jointListBuffer = jointList
        hullV.label = "Coin.hullVertices"
        hullR.label = "Coin.hullRanges"
        self.hullVertexBuffer = hullV
        self.hullRangeBuffer = hullR
        solveTG.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(CoinDEMSolver.solveTGSize)
        // Persistent + read-during-substep buffers start cleared (all awake, no slow frames).
        sleepTimer.contents().bindMemory(to: UInt32.self, capacity: maxCoins).update(repeating: 0, count: maxCoins)
        asleep.contents().bindMemory(to: UInt32.self, capacity: maxCoins).update(repeating: 0, count: maxCoins)
        sleepKey.contents().bindMemory(to: UInt32.self, capacity: maxCoins).update(repeating: 0xFFFF_FFFF, count: maxCoins)
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
        noteBodyBound((ptr[slot].prevPos.w * ptr[slot].prevPos.w + ptr[slot].vel.w * ptr[slot].vel.w).squareRoot(), poly: false)
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
        noteBodyBound(simd_length(halfExtents))
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
        noteBodyBound(radius, poly: false)
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
        noteBodyBound(halfLength + radius, poly: false)
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        // A reused slot must not inherit the previous body's sleep state.
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    /// Activate an OVOID (egg) body — the convex hull of two spheres on the
    /// local +Y axis: the FAT sphere (radius `fatRadius`, toward −Y) and the
    /// TIP sphere (radius `tipRadius`, toward +Y) with centres `centerDistance`
    /// apart, joined by the tangent-cone flank. Smooth everywhere, so it rolls
    /// and wobbles like a real egg — no facets. The exact solid-of-revolution
    /// COM and inertia are integrated here at spawn, so the body's position is
    /// its true (fat-end-biased) centre of mass and the settle-to-the-fat-end
    /// wobble is real physics. Constraint path only (like hulls): against a
    /// plane the two end-sphere probes are exact; everything else resolves via
    /// swept-radius segment probes + GJK/EPA support.
    @discardableResult
    public func spawnEgg(at position: SIMD3<Float>,
                         fatRadius: Float,
                         tipRadius: Float,
                         centerDistance: Float,
                         velocity: SIMD3<Float> = .zero,
                         orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                         tumble: SIMD3<Float> = .zero,
                         mass: Float = 1,
                         friction: Float? = nil,
                         restitution: Float? = nil,
                         type: UInt32 = 0) -> Int? {
        assert(solverMode == .constraint,
               "spawnEgg requires solverMode == .constraint (legacy has no ovoid narrowphase)")
        guard fatRadius > 1e-4, tipRadius > 1e-4, centerDistance >= 0 else { return nil }
        let slot: Int
        if let reused = freeSlots.popLast() {
            slot = reused
        } else if nextSlot < maxCoins {
            slot = nextSlot; nextSlot += 1
        } else {
            return nil
        }
        // COM + per-unit-mass inertia of the swept-cone solid, memoised per
        // (fatRadius, tipRadius, centerDistance) — a rain of identical eggs
        // integrates once.
        let props = Self.eggProperties(fatRadius: fatRadius, tipRadius: tipRadius,
                                       centerDistance: centerDistance)
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let invMass: Float = mass > 1e-6 ? 1.0 / mass : 1.0
        ptr[slot] = CoinBody(position: position, invMass: invMass, velocity: velocity,
                             orient: orient, angVel: tumble,
                             shapeExtents: SIMD4(fatRadius, tipRadius, props.yTip, 5),  // w = 5 → ovoid
                             hullRef: SIMD4(props.yFat,
                                            props.invInertiaK.x, props.invInertiaK.y, props.invInertiaK.z))
        // Bounding radius rides prevPos.w (broadphase reach), the full half-height
        // rides vel.w (floor backstop + reach), mirroring the capsule convention.
        let maxExtent = max(abs(props.yFat) + fatRadius, abs(props.yTip) + tipRadius)
        ptr[slot].prevPos.w = maxExtent
        ptr[slot].vel.w     = maxExtent
        noteBodyBound(maxExtent, poly: false)
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        // A reused slot must not inherit the previous body's sleep state.
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    /// COM offset + per-unit-mass inverse inertia of the ovoid solid. Sphere
    /// centres pre-integration sit at y = 0 (fat) and y = `centerDistance`
    /// (tip); the returned yFat/yTip are those centres re-expressed relative to
    /// the integrated COM (the body origin the solver simulates around).
    /// Numeric disc-stack integration over the true swept-union profile — exact
    /// to sampling, no closed-form composite approximation to get subtly wrong.
    struct EggBodyProperties {
        var yFat: Float          // fat-sphere centre offset from COM (< 0)
        var yTip: Float          // tip-sphere centre offset from COM (> 0)
        var invInertiaK: SIMD3<Float>   // per-unit-mass inverse inertia diagonal
    }
    private static var eggPropsCache: [SIMD3<Float>: EggBodyProperties] = [:]
    static func eggProperties(fatRadius r1: Float, tipRadius r2: Float,
                              centerDistance d: Float) -> EggBodyProperties {
        let key = SIMD3(r1, r2, d)
        if let hit = eggPropsCache[key] { return hit }
        // Cross-section radius of the swept union at height y: the max over the
        // sphere family c(t) = t·d, r(t) = mix(r1,r2,t). 256 slices × 96 sweep
        // samples is exact to ~1e-4 of the size — far inside DEM tolerance.
        let yMin = Double(-r1), yMax = Double(d + r2)
        let nY = 256, nT = 96
        var v = 0.0, vy = 0.0                       // Σ r²·dy, Σ y·r²·dy (π cancels)
        var slices = [Double](repeating: 0, count: nY)   // r² per slice
        var ys     = [Double](repeating: 0, count: nY)
        let dy = (yMax - yMin) / Double(nY)
        for i in 0..<nY {
            let y = yMin + (Double(i) + 0.5) * dy
            var r2max = 0.0
            for k in 0...nT {
                let t = Double(k) / Double(nT)
                let rt = Double(r1) + (Double(r2) - Double(r1)) * t
                let dyc = y - t * Double(d)
                let rr = rt * rt - dyc * dyc
                if rr > r2max { r2max = rr }
            }
            slices[i] = r2max; ys[i] = y
            v += r2max; vy += y * r2max
        }
        let yc = v > 1e-12 ? vy / v : 0.0
        // Disc-stack inertia per unit mass: dI_axis = dm·(r²/2), dI_diam = dm·(r²/4 + z²).
        var iAxis = 0.0, iDiam = 0.0
        for i in 0..<nY {
            let z = ys[i] - yc
            iAxis += slices[i] * (slices[i] * 0.5)
            iDiam += slices[i] * (slices[i] * 0.25 + z * z)
        }
        let norm = max(v, 1e-12)
        let props = EggBodyProperties(
            yFat: Float(0.0 - yc),
            yTip: Float(Double(d) - yc),
            invInertiaK: SIMD3(Float(norm / max(iDiam, 1e-12)),
                               Float(norm / max(iAxis, 1e-12)),
                               Float(norm / max(iDiam, 1e-12))))
        eggPropsCache[key] = props
        return props
    }

    // ── Convex hulls (constraint path) ────────────────────────────────────────

    /// Register a convex shape from a point cloud. Interior points are dropped
    /// (a real incremental convex hull runs on the CPU, once); the EXACT solid
    /// COM + inertia are integrated over the hull's boundary, diagonalized, and
    /// the vertices re-expressed in the principal frame — the returned handle
    /// carries that frame change so the render mesh can match. nil when the
    /// input is degenerate or the hull tables are full.
    public func registerHull(vertices: [SIMD3<Float>]) -> HullHandle? {
        guard shapeTable.count < Self.maxHulls,
              let prep = CoinHullMath.prepare(vertices),
              prep.vertices.count >= 4 else { return nil }
        // The vertices, then ALWAYS a topology header (flags 0 = none: the kernels fall
        // back to the vertex-probe / GJK paths), then the packed faces/edges when the
        // hull fits the GPU narrowphase's caps.
        let topo = prep.topology.flatMap { $0.fitsGPU ? $0.pack(vertices: prep.vertices) : nil } ?? [CoinHullTopology.emptyHeader]
        guard hullVertexCursor + prep.vertices.count + topo.count <= Self.maxHullVertices else { return nil }
        let index = shapeTable.count
        let vp = hullVertexBuffer.contents().bindMemory(to: SIMD4<Float>.self,
                                                        capacity: Self.maxHullVertices)
        for (i, v) in prep.vertices.enumerated() { vp[hullVertexCursor + i] = SIMD4(v, 0) }
        for (i, t) in topo.enumerated() { vp[hullVertexCursor + prep.vertices.count + i] = t }
        hullRangeBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: Self.maxHulls)[index] =
            SIMD2(UInt32(hullVertexCursor), UInt32(prep.vertices.count))
        hullVertexCursor += prep.vertices.count + topo.count
        shapeTable.append(.hull(prep))
        return HullHandle(index: index, vertices: prep.vertices,
                          comOffset: prep.comOffset,
                          principalRotation: prep.principalRotation,
                          boundingRadius: prep.boundingRadius)
    }

    /// Whether registered hull `index` carries GPU face topology (the exact SAT + face-
    /// clipping narrowphase); false ⇒ it uses the vertex-probe / GJK fallback paths.
    public func hullHasTopology(_ index: Int) -> Bool {
        guard index >= 0, index < shapeTable.count, case .hull(let p) = shapeTable[index] else { return false }
        return p.topology?.fitsGPU ?? false
    }

    /// Activate a CONVEX HULL body (a shape registered with `registerHull`).
    /// Constraint path only: hull pairs resolve through live GJK+EPA with a
    /// clipped support-polygon manifold; statics via the true hull vertices.
    /// (On the legacy path a hull degrades to a coarse bounding disc — assert.)
    @discardableResult
    public func spawnHull(at position: SIMD3<Float>,
                          hull: HullHandle,
                          velocity: SIMD3<Float> = .zero,
                          orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                          tumble: SIMD3<Float> = .zero,
                          mass: Float = 1,
                          friction: Float? = nil,
                          restitution: Float? = nil,
                          type: UInt32 = 0) -> Int? {
        assert(solverMode == .constraint,
               "spawnHull requires solverMode == .constraint (legacy has no hull narrowphase)")
        guard hull.index >= 0, hull.index < shapeTable.count,
              case .hull(let prep) = shapeTable[hull.index] else { return nil }
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
                             shapeExtents: SIMD4(0, 0, 0, 4),   // w = 4 → hull
                             hullRef: SIMD4(Float(hull.index),
                                            prep.invInertiaK.x, prep.invInertiaK.y, prep.invInertiaK.z))
        ptr[slot].prevPos.w = prep.boundingRadius
        ptr[slot].vel.w     = prep.minHalfExtent
        noteBodyBound(prep.boundingRadius, poly: prep.topology?.fitsGPU ?? false)
        setMaterial(slot, friction: friction, restitution: restitution)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
        return slot
    }

    /// Claim a free body slot (the shared slot logic of every spawn*), or nil if full.
    func claimBodySlot() -> Int? {
        if let reused = freeSlots.popLast() { return reused }
        guard nextSlot < maxCoins else { return nil }
        defer { nextSlot += 1 }
        return nextSlot
    }

    /// The per-slot bookkeeping every spawn* finishes with: material, render type, no
    /// articulation link, awake with a zero slow-frame timer, and the high-water mark.
    func finishSpawn(_ slot: Int, friction: Float?, restitution: Float?, type: UInt32) {
        setMaterial(slot, friction: friction, restitution: restitution)
        clearPatchRadius(slot)                // a reused slot starts as a point contact (stage B2)
        bodyTypeBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = type
        linkBuffer.contents().bindMemory(to: SIMD2<Int32>.self, capacity: maxCoins)[slot] = SIMD2(-1, 0)
        asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] = 0
        highWater = max(highWater, slot + 1)
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

    /// `CoinJoint.ref` for a joint made NOW between A and B (or A and the world):
    /// conj(qA)·qB — B's orientation in A's frame (world: conj(qA)), unit (x, y, z, w).
    func jointReference(_ bodyA: Int, _ bodyB: Int?) -> SIMD4<Float> {
        let p = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        func q(_ s: Int) -> simd_quatf {
            let o = p[s].orient
            return simd_quatf(ix: o.x, iy: o.y, iz: o.z, r: o.w).normalized
        }
        let qb = bodyB.map(q) ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        let r = (q(bodyA).inverse * qb).normalized
        return SIMD4(r.imag, r.real)
    }
    static let identityReference = SIMD4<Float>(0, 0, 0, 1)

    /// BALL joint: the world point `worldAnchor` on body A coincides with the same
    /// point on body B (or with that fixed world point when `bodyB` is nil).
    /// Anchors are captured from the bodies' CURRENT poses. Returns a handle for
    /// `removeJoint`, or nil when the joint table is full. Constraint path only.
    /// `collideConnected`: false (default) suppresses contacts between the two
    /// jointed bodies — otherwise a body pair that touches at the anchor fights
    /// its own contact. Set true to keep them colliding (e.g. bodies meant to
    /// stay physically apart despite the joint).
    @discardableResult
    public func addBallJoint(bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                             collideConnected: Bool = false) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        writeJoint(slot, CoinJoint(
            meta: SIMD4(0, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), Self.jointMeta(collideConnected: collideConnected)),
            anchorA: SIMD4(toLocal(bodyA, worldAnchor), 0),
            anchorB: SIMD4(world ? worldAnchor : toLocal(bodyB!, worldAnchor), 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0),
            ref: jointReference(bodyA, bodyB)))
        return slot
    }

    /// WELD joint: A and B (or A and the world when `bodyB` is nil) are held rigidly at
    /// their CURRENT relative pose — ball at `worldAnchor` + all three rotations locked,
    /// solved as ONE 6-row block (VZ-0150). The rotation lock is the vector part of the
    /// relative orientation against the one captured here (`CoinJoint.ref`),
    /// sign-canonicalised to w ≥ 0, so a weld made at any relative orientation holds it
    /// through any number of turns and never wraps. One slot (the old two-hinge weld
    /// needed two, with redundant rows). `collideConnected`: see `addBallJoint`.
    @discardableResult
    public func addWeldJoint(bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                             collideConnected: Bool = false) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        writeJoint(slot, CoinJoint(
            meta: SIMD4(4, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), Self.jointMeta(collideConnected: collideConnected)),
            anchorA: SIMD4(toLocal(bodyA, worldAnchor), 0),
            anchorB: SIMD4(world ? worldAnchor : toLocal(bodyB!, worldAnchor), 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0),
            ref: jointReference(bodyA, bodyB)))
        return slot
    }

    /// HINGE joint: ball at `worldAnchor` + the bodies may only rotate relative to
    /// each other about `worldAxis`. `limits` (radians, lo < hi, measured from the
    /// CURRENT relative pose = 0 — the twist is taken against the relative orientation
    /// captured here and sign-canonicalised, so it spans (−π, π] and the stops hold for
    /// bodies at ANY relative orientation, VZ-0156) adds angle stops; nil = free spin. `motor`
    /// drives toward `targetVelocity` (rad/s about `worldAxis`), bounded by
    /// `maxTorque` (N·m) each substep — the impulse accumulated over the velocity
    /// iterations is what is clamped, so this is the physical bound at any
    /// `velocityIterations` (VZ-0155) — turns the hinge into a motorized wheel
    /// or door; nil = no motor (the default free/limited-swing hinge). Combine
    /// with `limits` for a motorized door that stops at an angle.
    /// For a WORLD hinge (`bodyB == nil`, the common wheel/fan/door case),
    /// `targetVelocity` is body A's own angular velocity — the intuitive
    /// framing. The in-kernel constraint is actually on B's velocity relative
    /// to A (the same convention the limit's twist angle already uses), which
    /// for a world hinge — B is the velocity-less world — would otherwise
    /// target `-A`'s own spin, so it's negated here to keep the public API
    /// intuitive. For a real two-body hinge `targetVelocity` is the RELATIVE
    /// spin of B about A (B's own spin plus this, roughly, when A is also
    /// moving) — not independently A's own spin.
    /// `collideConnected`: see `addBallJoint`.
    @discardableResult
    public func addHingeJoint(bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                              worldAxis: SIMD3<Float>,
                              limits: ClosedRange<Float>? = nil,
                              motor: (targetVelocity: Float, maxTorque: Float)? = nil,
                              collideConnected: Bool = false) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        let axis = simd_normalize(worldAxis)
        // lo == hi disables the limit branch in-kernel.
        let lo = limits?.lowerBound ?? 0, hi = limits?.upperBound ?? 0
        // maxTorque <= 0 disables the motor branch in-kernel. The kernel's
        // constraint target is "B relative to A"; negate for a world hinge so
        // targetVelocity means A's own spin (see doc comment above).
        let motorTarget = (motor?.targetVelocity ?? 0) * (world ? -1 : 1)
        writeJoint(slot, CoinJoint(
            meta: SIMD4(1, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), Self.jointMeta(collideConnected: collideConnected)),
            anchorA: SIMD4(toLocal(bodyA, worldAnchor), motorTarget),
            anchorB: SIMD4(world ? worldAnchor : toLocal(bodyB!, worldAnchor), motor?.maxTorque ?? 0),
            axisA: SIMD4(toLocalDir(bodyA, axis), lo),
            axisB: SIMD4(world ? axis : toLocalDir(bodyB!, axis), hi),
            ref: jointReference(bodyA, bodyB)))
        return slot
    }

    /// PRISMATIC/SLIDER joint: the bodies may only translate relative to each
    /// other along `worldAxis`, through `worldAnchor` — rotation is fully
    /// locked (unlike the hinge, which leaves rotation about its axis free) at the
    /// relative orientation captured here (VZ-0156: any relative orientation holds).
    /// `limits` (meters, lo < hi, measured from the CURRENT relative position
    /// = 0) adds travel stops; nil = unbounded slide. `motor` drives toward
    /// `targetVelocity` (m/s along the axis), bounded by `maxForce` (N) each
    /// substep (accumulated over the velocity iterations, like the hinge motor)
    /// — a piston or elevator instead of a free/limited slide; nil =
    /// no motor. `targetVelocity` is body A's own velocity along the axis for
    /// BOTH a world joint and a two-body one (unlike `addHingeJoint`'s
    /// `motor:`, which needs a world-case sign correction — the two kernels'
    /// motor sub-constraints are written against opposite relative-velocity
    /// conventions; verified by direct measurement for each, not assumed from
    /// the other).
    /// `collideConnected`: see `addBallJoint`.
    @discardableResult
    public func addPrismaticJoint(bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                                  worldAxis: SIMD3<Float>,
                                  limits: ClosedRange<Float>? = nil,
                                  motor: (targetVelocity: Float, maxForce: Float)? = nil,
                                  collideConnected: Bool = false) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        let axis = simd_normalize(worldAxis)
        // lo == hi disables the limit branch in-kernel.
        let lo = limits?.lowerBound ?? 0, hi = limits?.upperBound ?? 0
        // maxForce <= 0 disables the motor branch in-kernel. UNLIKE the hinge
        // motor, no world-case negation: the prismatic kernel's along-axis
        // motor is written against vpA−vpB (A relative to B), so for a world
        // joint (vpB=0) it already converges directly to A's own velocity —
        // confirmed by a single-body trace, not assumed from the hinge's
        // pattern (which uses the opposite wB−wA convention and DOES need it).
        let motorTarget = motor?.targetVelocity ?? 0
        writeJoint(slot, CoinJoint(
            meta: SIMD4(3, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), Self.jointMeta(collideConnected: collideConnected)),
            anchorA: SIMD4(toLocal(bodyA, worldAnchor), motorTarget),
            anchorB: SIMD4(world ? worldAnchor : toLocal(bodyB!, worldAnchor), motor?.maxForce ?? 0),
            axisA: SIMD4(toLocalDir(bodyA, axis), lo),
            axisB: SIMD4(world ? axis : toLocalDir(bodyB!, axis), hi),
            ref: jointReference(bodyA, bodyB)))
        return slot
    }

    /// DISTANCE joint: the two world anchor points keep their CURRENT separation
    /// (or `restLength` when given) — a rigid tether/rod, free to swing.
    /// `collideConnected`: see `addBallJoint`.
    @discardableResult
    public func addDistanceJoint(bodyA: Int, bodyB: Int?,
                                 worldAnchorA: SIMD3<Float>, worldAnchorB: SIMD3<Float>,
                                 restLength: Float? = nil,
                                 collideConnected: Bool = false) -> Int? {
        guard let slot = claimJointSlot() else { return nil }
        let world = bodyB == nil
        let rest = restLength ?? simd_length(worldAnchorB - worldAnchorA)
        writeJoint(slot, CoinJoint(
            meta: SIMD4(2, UInt32(bodyA), world ? 0xFFFF_FFFF : UInt32(bodyB!), Self.jointMeta(collideConnected: collideConnected)),
            anchorA: SIMD4(toLocal(bodyA, worldAnchorA), rest),
            anchorB: SIMD4(world ? worldAnchorB : toLocal(bodyB!, worldAnchorB), 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0),
            ref: Self.identityReference))
        return slot
    }

    /// Packs the `CoinJoint.meta.w` bit flags: bit 0 = enabled (always set for a
    /// live joint), bit 1 = collideConnected.
    private static func jointMeta(collideConnected: Bool) -> UInt32 {
        1 | (collideConnected ? 2 : 0)
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
        clearPatchRadius(slot)
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
                clearPatchRadius(slot)
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
        maxBodyBound = 0
        hasPolyBodies = false
        for slot in torsionPatchSlots { clearPatchRadius(slot) }
        // Joints reference the cleared bodies — drop them all with the field.
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        for i in 0..<jointHighWater { j[i].meta.w = 0 }
        freeJointSlots.removeAll(keepingCapacity: true)
        jointHighWater = 0
        jointKeys = [:]          // a joint added after this starts cold, even in a reused slot
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

    /// Read-only snapshot of an active body's angular velocity (rad/s, world frame).
    public func angularVelocity(of slot: Int) -> SIMD3<Float>? {
        guard slot >= 0, slot < highWater else { return nil }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let b = ptr[slot]
        return b.posInvMass.w == 0 ? nil : SIMD3(b.angVel.x, b.angVel.y, b.angVel.z)
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
        ensureContactPrepBuffer()                 // (opt-in prepared rows: before the uniforms say so)

        // Re-size the colour sweep from what the GPU has actually needed so far (read of
        // a shared buffer written by completed frames — no stall). The sweep is only a
        // SPLIT POINT: colours below it get one parallel dispatch each, and the serial
        // tail pass (coinSolveVelocityTail) solves every colour at or above it in the
        // same order — so a sweep that is too small costs time, never a skipped contact
        // (VZ-0154). A world that has never had a contact sweeps nothing.
        if solverMode == .constraint {
            let used = Int(colorStatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 4)[2])
            colorSweep = used == 0 ? 0 : min(Self.maxColors, used + Self.colorSweepHeadroom)
        }

        // Spiral-of-death guard: bank at most one frame's worth of substeps. Without
        // this, a hitch (GC pause, window drag, sleep/wake) inflates the accumulator
        // past maxSubsteps·fixedDt; every following frame then runs the full substep
        // cap trying to repay the debt, which slows the sim and can never catch up. We
        // run real time up to the cap and DROP the unrepayable remainder — the sim
        // briefly runs slightly slow-motion through the hitch instead of death-
        // spiralling, which is the standard fixed-timestep degradation.
        accumulator = min(accumulator + wallDt, fixedDt * Float(maxSubsteps))
        // The frame's active joint list (constraint path): one scan of the host-owned joint
        // table, uploaded once, read by every joint kernel below (VZ-0150 item 4c).
        frameJointCount = (solverMode == .constraint && accumulator >= fixedDt) ? encodeJointListUpload(cb) : 0
        // SMALL-WORLD HOOK (CoinDEMSolver+SmallWorld.swift): a small world's whole frame —
        // every substep, the sleep update and the transforms — as ONE dispatch, the same
        // simulation as the loop below; a world that outgrows it takes the loop by itself.
        if smallWorldEligible(bodies: bound) {
            var steps = 0
            while accumulator >= fixedDt && steps < maxSubsteps { accumulator -= fixedDt; steps += 1 }
            lastStepCount = steps
            smallWorld.lastFrameUsed = steps > 0
            guard steps > 0 else { return }
            writeUniforms(dt: fixedDt, coinCount: bound)
            encodeSmallWorldFrame(cb, coinCount: bound, steps: steps)
            return
        }
        smallWorld.lastFrameUsed = false
        var steps = 0
        while accumulator >= fixedDt && steps < maxSubsteps {
            writeUniforms(dt: fixedDt, coinCount: bound)
            // Which bodies the generate kernel's collideConnected joint scan applies to
            // (VZ-0153). Joints change only between frames (host writes): once a frame,
            // after the uniforms it reads are written.
            if steps == 0 && solverMode == .constraint { encodeMarkJointedBodies(cb, coinCount: bound) }
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

    /// Encode the hierarchical cell-offset scan (block sums ∥ → serial scan over
    /// BLOCKS only → per-block apply ∥). One call replaces the old single-thread
    /// whole-grid scan, whose cost was O(cells) on one GPU lane — seconds per
    /// substep on a house-scale grid, and the cause of the 2026-08-27 GPU
    /// watchdog timeouts. Contract unchanged: exclusive prefix sum + grand total
    /// in `cellOffsets[numCells]`, `cellCounts` zeroed for the scatter cursor.
    private func encodeCellScan(_ cb: MTLCommandBuffer, label: String) {
        dispatch(cb, blockSumsPipeline, threads: numScanBlocks, label: "\(label).blockSums") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
            enc.setBuffer(self.cellBlockSums, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        dispatch(cb, blockScanPipeline, threads: 1, label: "\(label).blockScan") { enc in
            enc.setBuffer(self.cellBlockSums, offset: 0, index: 0)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 1)
        }
        dispatch(cb, offsetsApplyPipeline, threads: numScanBlocks, label: "\(label).apply") { enc in
            enc.setBuffer(self.cellCounts, offset: 0, index: 0)
            enc.setBuffer(self.cellOffsets, offset: 0, index: 1)
            enc.setBuffer(self.cellBlockSums, offset: 0, index: 2)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
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
        encodeCellScan(cb, label: "Coin.scan")
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
        // Every colour the colouring can emit is solved. Each colour's dispatch is sized
        // by the GPU to that colour's slice of the compacted contact list, so an empty
        // colour costs one 0-threadgroup dispatch and the total thread count per velocity
        // iteration is the contact count — not colours × contacts.
        // Size this substep's colour sweep from the worst colour count the GPU has
        // reported (plus headroom for a denser moment). `colorStats.beyondSweep` is the
        // fail-closed counter if this is ever too small.
        let colors = colorSweep
        // Stage B3 (plan 3b): the substep's bookkeeping kernels run fused — the same step
        // functions in the same order (Shaders/CoinDEMFusedKernels.h), 15 fewer dependent
        // dispatches per substep. The one-step kernels stay for the diagnostics
        // (generateContactsNow, measurePenetration) and as the A/B seam.
        let fused = !fusedSubstepKernelsDisabledForTesting
        ensurePolyPairBuffers()
        if fused {
            // Velocity integration + every clear the substep needs before generation (cells,
            // the append cursors, per-body contact counts, colour buckets): coinSubstepBegin.
            let threads = max(coinCount, numCells, maxCoins, Self.maxColors + 1)
            dispatch(cb, substepBeginPipeline, threads: threads, label: "Coin.cs.begin") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 3)
                enc.setBuffer(self.cellCounts, offset: 0, index: 4)
                var cells = UInt32(self.numCells)
                enc.setBytes(&cells, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 6)
                // The polytope pair cursor (else the contact cursor again).
                enc.setBuffer(self.polyPairCountBuffer ?? self.contactCountBuffer, offset: 0, index: 7)
                enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 8)
                var slots = UInt32(self.maxCoins)
                enc.setBytes(&slots, length: MemoryLayout<UInt32>.size, index: 9)
                enc.setBuffer(self.colorCountBuffer, offset: 0, index: 10)
            }
        } else {
            dispatch(cb, intVelCSPipeline, threads: coinCount, label: "Coin.cs.intVel") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 3)
            }
        }
        // FOOT HOOK (CoinDEMSolver+Foot.swift): each active spring foot's leg contact and its
        // exactly integrated spring impulse, after gravity and before contact generation.
        encodeFootSubstep(cb, colliders: colliderBuffer.buffer, colliderCount: colliders.count, bias: biasBuffer)
        encodeBroadphase(cb, coinCount: coinCount, fused: fused && fusedBroadphaseFits, clearCells: !fused)
        encodeGenerateContacts(cb, coinCount: coinCount, clearCursors: !fused)
        // Opt-in warm-start memory through a sleep: the sleeping pairs' last contacts, carried as
        // dormant records (coinCarryDormant) — before the colouring, whose cursor clamp covers them.
        if keepSleepingContacts && warmStart && !needsHashClear, let pso = carryDormantPipeline {
            dispatch(cb, pso, threads: hashSize, label: "Coin.cs.carryDormant") { enc in
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 0)
                enc.setBuffer(self.hashSizeBuffer, offset: 0, index: 1)
                enc.setBuffer(self.prevContactBuffer, offset: 0, index: 2)
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 3)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 4)
                enc.setBuffer(self.contactBuffer, offset: 0, index: 5)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 6)
                enc.setBuffer(self.maxContactsBuffer, offset: 0, index: 7)
            }
        }
        encodeColoring(cb, fused: fused)
        // Manifold solve: every contact's restitution approach speed from the PRE-solve
        // velocities (coinCaptureApproach), not lazily in colour order.
        if manifoldSolve {
            dispatchOverContacts(cb, captureApproachPipeline, label: "Coin.cs.captureApproach") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 2)
                enc.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 3)
            }
        }
        // Prepared contact rows (opt-in, stage B3): every contact's pose-constant factors, once.
        let prepared = preparedContactsLive
        if prepared, let prepPSO = prepareContactsPipeline, let prep = contactPrepBuffer {
            dispatchOverContacts(cb, prepPSO, label: "Coin.cs.prepareContacts") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 2)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 4)
                enc.setBuffer(self.materialBuffer, offset: 0, index: 5)
                enc.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 6)
                enc.setBuffer(self.patchRadiusBuffer, offset: 0, index: 7)
                enc.setBuffer(prep, offset: 0, index: 8)
            }
        }
        // Warm start: seed fresh contacts with last substep's converged impulses and
        // apply them before iterating (a resting stack resumes from its solution).
        if warmStart {
            if needsHashClear {                       // zero the hash on first use (private buffer)
                dispatch(cb, clearHashPipeline, threads: hashSize, label: "Coin.cs.clearHash0") { enc in
                    enc.setBuffer(self.pairHashBuffer, offset: 0, index: 0)
                }
                needsHashClear = false
            }
            dispatchOverContacts(cb, warmMatchPipeline, label: "Coin.cs.warmMatch") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.prevContactBuffer, offset: 0, index: 2)
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 3)
                enc.setBuffer(self.hashSizeBuffer, offset: 0, index: 4)
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 5)   // position-match tolerance scale
                // Carry the torsion impulse only while CD_FLAG_TORSION is set (see the kernel).
                var torsionOn: UInt32 = self.torsionPatchSlots.isEmpty ? 0 : 1
                enc.setBytes(&torsionOn, length: MemoryLayout<UInt32>.size, index: 6)
            }
            dispatchPerColor(cb, warmApplyPipeline, tail: warmApplyTailPipeline, colors: colors, label: "Coin.cs.warmApply") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.contactBuffer, offset: 0, index: 1)
                enc.setBuffer(self.colorContactsBuffer, offset: 0, index: 2)
                enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 4)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 6)      // an asleep end is immovable (VZ-0162)
            }
        }
        // Joints: rows, block factors and re-projected impulses for this substep's poses,
        // then their warm start — before the first contact colour, so the contacts
        // iterate against an assembly that already carries its load (VZ-0150).
        encodeJointPrepare(cb)
        for _ in 0..<velocityIterations {
            if prepared, let colorPSO = solveColorPrepPipeline, let tailPSO = solveTailPrepPipeline, let prep = contactPrepBuffer {
                // The prepared rows (their μ, e and patch radius come from the prepare buffer).
                dispatchPerColor(cb, colorPSO, tail: tailPSO, colors: colors, label: "Coin.cs.solvePrep") { enc in
                    enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                    enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                    enc.setBuffer(self.contactBuffer, offset: 0, index: 2)
                    enc.setBuffer(self.colorContactsBuffer, offset: 0, index: 3)
                    enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
                    enc.setBuffer(self.asleepBuffer, offset: 0, index: 6)
                    enc.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 8)
                    enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 9)
                    enc.setBuffer(prep, offset: 0, index: 12)
                }
            } else {
            dispatchPerColor(cb, solveVelCSPipeline, tail: solveTailPipeline, colors: colors, label: "Coin.cs.solve") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                enc.setBuffer(self.contactBuffer, offset: 0, index: 2)
                enc.setBuffer(self.colorContactsBuffer, offset: 0, index: 3)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 6)
                enc.setBuffer(self.materialBuffer, offset: 0, index: 7)
                enc.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 8)
                enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 9)
                enc.setBuffer(self.patchRadiusBuffer, offset: 0, index: 11)   // torsion (stage B2)
            }
            }
            // Generic joints: one serial Gauss-Seidel pass over the active joints per
            // velocity iteration (each joint an exact block), interleaved with the
            // contact colours so joints and contacts converge together (a jointed body
            // pressed by a pile both holds its anchor AND de-penetrates in the same substep).
            if frameJointCount > 0, let prep = jointPrepBuffer {
                var jc = UInt32(frameJointCount)
                var passes = UInt32(max(1, jointInnerPasses))
                var tgBodies = UInt32(frameJointTGBodies)
                // One group of two SIMD groups: the real-velocity chain on thread 0, the bias
                // chain on thread 32 (they never exchange a value — cdJointSolveReal); the 64
                // threads also stage the threadgroup cache (tgBodies > 0).
                dispatch(cb, jointSolveCSPipeline, threads: 64, label: "Coin.cs.joints") { enc in
                    enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                    enc.setBuffer(self.biasBuffer, offset: 0, index: 1)
                    enc.setBuffer(self.jointListBuffer, offset: 0, index: 2)
                    enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 3)
                    enc.setBuffer(prep, offset: 0, index: 4)
                    enc.setBytes(&passes, length: MemoryLayout<UInt32>.size, index: 5)
                    enc.setBytes(&tgBodies, length: MemoryLayout<UInt32>.size, index: 6)
                }
            }
            // FOOT HOOK (CoinDEMSolver+Foot.swift): planted pads' torsion / yaw-motor rows; on
            // the last iteration, the legs' exact position correction (split-impulse bias).
            encodeFootIteration(cb, colliders: colliderBuffer.buffer, colliderCount: colliders.count, bias: biasBuffer)
        }
        // Snapshot the solved contacts + rebuild the pair hash for next substep.
        if warmStart {
            dispatch(cb, clearHashPipeline, threads: hashSize, label: "Coin.cs.clearHash") { enc in
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 0)
            }
            dispatchOverContacts(cb, snapshotPipeline, label: "Coin.cs.snapshot") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.prevContactBuffer, offset: 0, index: 2)
                enc.setBuffer(self.pairHashBuffer, offset: 0, index: 3)
                enc.setBuffer(self.hashSizeBuffer, offset: 0, index: 4)
            }
        }
        if fused {
            // Position integration + finalize, both per body (coinIntegratePositionFinalize).
            dispatch(cb, intPosFinalizePipeline, threads: coinCount, label: "Coin.cs.intPosFinalize") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.biasBuffer, offset: 0, index: 1)   // + the motor-driven marker (no dead-stop)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 3)
            }
            return
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
            enc.setBuffer(self.biasBuffer, offset: 0, index: 3)   // motor-driven marker (no dead-stop)
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
            enc.setBuffer(self.sleepHubBuffer, offset: 0, index: 2)
        }
        // Sleep groups (VZ-0152): asleep pairs generate no contacts any more, so each
        // asleep body is unioned with its group's hub every round instead.
        dispatch(cb, islandSleepHubPipeline, threads: coinCount, label: "Coin.cs.islandSleepHub") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.asleepBuffer, offset: 0, index: 1)
            enc.setBuffer(self.sleepKeyBuffer, offset: 0, index: 2)
            enc.setBuffer(self.sleepHubBuffer, offset: 0, index: 3)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
        }
        for _ in 0..<islandUnionRounds {
            dispatch(cb, islandUnionPipeline, threads: max(maxContacts, coinCount), label: "Coin.cs.islandUnion") { enc in
                enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactBuffer, offset: 0, index: 1)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 2)
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 3)
                enc.setBuffer(self.asleepBuffer, offset: 0, index: 4)
                enc.setBuffer(self.sleepKeyBuffer, offset: 0, index: 5)
                enc.setBuffer(self.sleepHubBuffer, offset: 0, index: 6)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 7)
            }
            // Joint edges join the same islands, so an articulated assembly
            // sleeps and wakes as one body (the active list only).
            if frameJointCount > 0 {
                var jc = UInt32(frameJointCount)
                dispatch(cb, islandUnionJointsPipeline, threads: frameJointCount, label: "Coin.cs.islandUnionJoints") { enc in
                    enc.setBuffer(self.islandLabelBuffer, offset: 0, index: 0)
                    enc.setBuffer(self.jointBuffer, offset: 0, index: 1)
                    enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 2)
                    enc.setBuffer(self.jointListBuffer, offset: 0, index: 3)
                }
            }
            // FOOT HOOK (CoinDEMSolver+Foot.swift): a planted foot joins its body's island to a
            // dynamic support's.
            encodeFootIslandUnion(cb, label: islandLabelBuffer)
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
            enc.setBuffer(self.biasBuffer, offset: 0, index: 4)   // last substep's motor-driven marker
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
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 5)   // freezing zeroes v/ω
            enc.setBuffer(self.sleepKeyBuffer, offset: 0, index: 6)      // the sleep group it froze in
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
        encodeCellScan(cb, label: "Coin.measure.scan")
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
            enc.setBuffer(self.hullVertexBuffer, offset: 0, index: 7)
            enc.setBuffer(self.hullRangeBuffer, offset: 0, index: 8)
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
    private func encodeBroadphase(_ cb: MTLCommandBuffer, coinCount: Int, fused: Bool = false, clearCells: Bool = true) {
        if fused {
            // Count, the block scan and the scatter in ONE threadgroup (coinBroadphaseTG); the
            // counts were zeroed by coinSubstepBegin.
            dispatchOneGroup(cb, broadphaseTGPipeline, label: "Coin.cs.broadphase") { enc in
                enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
                enc.setBuffer(self.cellCounts, offset: 0, index: 1)
                enc.setBuffer(self.cellOffsets, offset: 0, index: 2)
                enc.setBuffer(self.sortedIndices, offset: 0, index: 3)
                enc.setBuffer(self.uniformBuffer, offset: 0, index: 4)
            }
            return
        }
        if clearCells {                               // (coinSubstepBegin already zeroed them when fused)
            dispatch(cb, cellClearPipeline, threads: numCells, label: "Coin.cs.cellClear") { enc in
                enc.setBuffer(self.cellCounts, offset: 0, index: 0)
            }
        }
        dispatch(cb, cellCountPipeline, threads: coinCount, label: "Coin.cs.cellCount") { enc in
            enc.setBuffer(self.coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(self.cellCounts, offset: 0, index: 1)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        encodeCellScan(cb, label: "Coin.cs.scan")
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
    private func encodeGenerateContacts(_ cb: MTLCommandBuffer, coinCount: Int, clearCursors: Bool = true) {
        // Reset the append cursor as a DISPATCH — a CPU write would land at encode time,
        // before any substep has run, leaving every later substep appending to the
        // previous one's contacts (see coinClearContactCount). The fused substep's
        // coinSubstepBegin has already reset both cursors (clearCursors false).
        ensurePolyPairBuffers()
        if clearCursors {
            dispatch(cb, clearContactCountPipeline, threads: 1, label: "Coin.cs.clearContactCount") { enc in
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 0)
                // The polytope pair cursor rides the same dispatch (else the contact count again).
                enc.setBuffer(self.polyPairCountBuffer ?? self.contactCountBuffer, offset: 0, index: 1)
            }
        }
        var jc = UInt32(frameJointCount)
        let live = hasPolyBodies && !polyKernelDisabledForTesting && polyPairBuffer != nil
        // ONE serial compute encoder: the generate dispatch, then (with polytope bodies) the
        // pair pass's indirect args and the per-pair narrowphase — Metal runs a serial
        // encoder's dispatches in order, so this is the same barrier semantics as three
        // encoders, minus two encoder setups per substep.
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Coin.cs.generate"
        enc.setComputePipelineState(generatePipeline)
        enc.setBuffer(coinBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(sortedIndices, offset: 0, index: 1)
        enc.setBuffer(cellOffsets, offset: 0, index: 2)
        enc.setBuffer(colliderBuffer.buffer, offset: 0, index: 3)
        enc.setBuffer(uniformBuffer, offset: 0, index: 4)
        enc.setBuffer(linkBuffer, offset: 0, index: 5)
        enc.setBuffer(contactBuffer, offset: 0, index: 6)
        enc.setBuffer(contactCountBuffer, offset: 0, index: 7)
        enc.setBuffer(maxContactsBuffer, offset: 0, index: 8)
        enc.setBuffer(hullVertexBuffer, offset: 0, index: 9)
        enc.setBuffer(hullRangeBuffer, offset: 0, index: 10)
        enc.setBuffer(jointBuffer, offset: 0, index: 11)
        enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBuffer(asleepBuffer, offset: 0, index: 13)
        enc.setBuffer(jointedBodyBuffer, offset: 0, index: 14)
        enc.setBuffer(jointListBuffer, offset: 0, index: 15)
        // The polytope pair list it appends to (a dummy with capacity 0 while none exists).
        var maxP = UInt32(live ? maxPolyPairs : 0)
        enc.setBuffer(live ? polyPairBuffer : polyDummyBuffer, offset: 0, index: 16)
        enc.setBuffer(live ? polyPairCountBuffer : polyDummyBuffer, offset: 0, index: 17)
        enc.setBytes(&maxP, length: MemoryLayout<UInt32>.size, index: 18)
        let w = min(coinCount, generatePipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: coinCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        // The polytope narrowphase (boxes, topology hulls, compounds — and those against
        // swept spheres): coinGenerateContacts listed exactly the pairs cdPairIsPoly owns
        // (and generated nothing for them); one thread per listed pair now generates them,
        // appending to the same contact buffer. A world without such bodies never pays it.
        if live, let pairs = polyPairBuffer, let pairCount = polyPairCountBuffer, let args = polyArgsBuffer {
            var maxPairs = UInt32(maxPolyPairs)
            // One SIMD-group-wide threadgroup per listed pair (stage B3, VZ-0198): the pairs
            // that run the separating-axis test spread it over the group's lanes
            // (cdPolyNarrowSGBody); a swept / GJK pair runs on lane 0.
            var pairsPerGroup: UInt32 = 1
            enc.setComputePipelineState(polyArgsPipeline)
            enc.setBuffer(pairCount, offset: 0, index: 0)
            enc.setBuffer(args, offset: 0, index: 1)
            enc.setBytes(&pairsPerGroup, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&maxPairs, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.setComputePipelineState(polyNarrowPipeline)
            enc.setBuffer(coinBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(colliderBuffer.buffer, offset: 0, index: 1)
            enc.setBuffer(uniformBuffer, offset: 0, index: 2)
            enc.setBuffer(contactBuffer, offset: 0, index: 3)
            enc.setBuffer(contactCountBuffer, offset: 0, index: 4)
            enc.setBuffer(maxContactsBuffer, offset: 0, index: 5)
            enc.setBuffer(hullVertexBuffer, offset: 0, index: 6)
            enc.setBuffer(hullRangeBuffer, offset: 0, index: 7)
            enc.setBuffer(pairs, offset: 0, index: 8)
            enc.setBuffer(pairCount, offset: 0, index: 9)
            enc.setBytes(&maxPairs, length: MemoryLayout<UInt32>.size, index: 10)
            enc.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0,
                                     threadsPerThreadgroup: MTLSize(width: polyNarrowPipeline.threadExecutionWidth, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// The polytope narrowphase's pair list, made on first use (a world without polytope
    /// bodies never pays for it) — before anything in the substep binds its cursor.
    private func ensurePolyPairBuffers() {
        guard hasPolyBodies && !polyKernelDisabledForTesting && polyPairBuffer == nil else { return }
        polyPairBuffer = device.makeBuffer(length: MemoryLayout<SIMD4<UInt32>>.stride * maxPolyPairs, options: .storageModePrivate)
        polyPairCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        polyArgsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 3, options: .storageModePrivate)
        polyPairBuffer?.label = "Coin.polyPairs"; polyPairCountBuffer?.label = "Coin.polyPairCount"; polyArgsBuffer?.label = "Coin.polyArgs"
    }

    /// Mark every body that is one end of an enabled, non-collideConnected two-body
    /// joint (coinMarkJointedBodies). Nothing to mark without active joints — the generate
    /// kernel then never reads the buffer (jointCount 0).
    private func encodeMarkJointedBodies(_ cb: MTLCommandBuffer, coinCount: Int) {
        guard frameJointCount > 0 else { return }
        var jc = UInt32(frameJointCount)
        dispatch(cb, markJointedPipeline, threads: coinCount, label: "Coin.cs.markJointed") { enc in
            enc.setBuffer(self.jointBuffer, offset: 0, index: 0)
            enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 1)
            enc.setBuffer(self.jointedBodyBuffer, offset: 0, index: 2)
            enc.setBuffer(self.uniformBuffer, offset: 0, index: 3)
            enc.setBuffer(self.jointListBuffer, offset: 0, index: 4)
        }
    }

    /// Scan the host-owned joint table (the GPU never writes it, so this is safe with a
    /// frame in flight) and upload the compact list of ENABLED slots for the frame being
    /// encoded; returns its length. A slot that was not enabled at the previous upload, or
    /// whose bodies / anchors / axes / reference changed, carries the reset flag and
    /// starts cold (its warm-start impulses are cleared on the GPU in the same pass).
    private func encodeJointListUpload(_ cb: MTLCommandBuffer) -> Int {
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        var list: [UInt32] = []
        var keys: [Int: JointKey] = [:]
        for i in 0..<jointHighWater where (j[i].meta.w & 1) != 0 {
            let key = JointKey(j[i])
            list.append(UInt32(i) | (jointKeys[i] == key ? 0 : 0x8000_0000))
            keys[i] = key
        }
        frameJointTGBodies = 0
        guard !list.isEmpty else { jointKeys = [:]; return 0 }
        // Threadgroup-cache tables for the joint solve: each joint's two local body indices
        // (0xFFFF = the world) and the distinct bodies, when both fit.
        var pairs: [UInt32] = [], bodies: [UInt32] = []
        if list.count <= Self.jointTGMaxJoints && !jointThreadgroupCacheDisabledForTesting {
            var local: [UInt32: UInt32] = [:]
            func index(_ b: UInt32) -> UInt32 {
                if b == 0xFFFF_FFFF { return 0xFFFF }
                if let l = local[b] { return l }
                let l = UInt32(bodies.count)
                local[b] = l
                bodies.append(b)
                return l
            }
            for e in list {
                let m = j[Int(e & 0x7FFF_FFFF)].meta
                let la = index(m.y), lb = index(m.z)
                pairs.append(la | (lb << 16))
            }
            if bodies.count <= Self.jointTGMaxBodies { frameJointTGBodies = bodies.count } else { pairs = []; bodies = [] }
        }
        if jointPrepBuffer == nil {
            jointPrepBuffer = device.makeBuffer(length: Self.jointPrepStride * Self.maxJoints, options: .storageModePrivate)
            jointPrepBuffer?.label = "Coin.jointPrep"
        }
        guard let prep = jointPrepBuffer else { return 0 }
        jointKeys = keys
        var n = UInt32(list.count), nb = UInt32(frameJointTGBodies)
        if pairs.isEmpty { pairs = [0]; bodies = [0] }                        // bound, unread (nb 0)
        dispatch(cb, jointListUploadPipeline, threads: list.count, label: "Coin.cs.jointList") { enc in
            list.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }   // ≤ 4 KB (maxJoints × 4)
            enc.setBuffer(self.jointListBuffer, offset: 0, index: 1)
            enc.setBuffer(prep, offset: 0, index: 2)
            enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
            pairs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 4) }
            bodies.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 5) }
            enc.setBytes(&nb, length: MemoryLayout<UInt32>.size, index: 6)
        }
        return list.count
    }

    /// Per substep: every active joint's rows, block factor and re-projected impulses
    /// (parallel), then their warm start (one thread) — one threadgroup (coinJointPrepare).
    private func encodeJointPrepare(_ cb: MTLCommandBuffer) {
        guard frameJointCount > 0, let prep = jointPrepBuffer,
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Coin.cs.jointPrepare"
        enc.setComputePipelineState(jointPreparePipeline)
        var jc = UInt32(frameJointCount)
        enc.setBuffer(coinBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(biasBuffer, offset: 0, index: 1)
        enc.setBuffer(jointBuffer, offset: 0, index: 2)
        enc.setBuffer(jointListBuffer, offset: 0, index: 3)
        enc.setBytes(&jc, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBuffer(uniformBuffer, offset: 0, index: 5)
        enc.setBuffer(asleepBuffer, offset: 0, index: 6)
        enc.setBuffer(prep, offset: 0, index: 7)
        // Up to 8 SIMD groups: joint k on group k mod 8, lane k / 8, so every joint of the frame is
        // prepared in one round (cdJointPrepareTG — the prepare is instruction-fetch bound, and a
        // group's lanes share its one instruction stream).
        let simd = max(1, jointPreparePipeline.threadExecutionWidth)
        let tg = max(simd, min(frameJointCount * simd, 8 * simd, (jointPreparePipeline.maxTotalThreadsPerThreadgroup / simd) * simd))
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
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
            enc.setBuffer(self.hullVertexBuffer, offset: 0, index: 3)
            enc.setBuffer(self.hullRangeBuffer, offset: 0, index: 4)
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
    /// rounds suffice; 24 is generous headroom for a dense pile.
    ///
    /// Contacts left over stay colour −1. They used to be **never solved** (a cube fell
    /// through another at gravity speed in the dense hull pile test, 68 mm deep); the
    /// solve's tail pass now solves them after every colour, in deterministic sub-colours
    /// on one threadgroup — up to 256 per substep. A larger leftover is still left
    /// UNSOLVED for that substep (`uncoloredUnsolved` counts it): solving it one contact at
    /// a time cost a Daydream-Home-scale egg rain 5–7× its frame. Jones–Plassmann colours a
    /// body's d contacts at best one per round, so a body of degree ≳ colorRounds leaves
    /// some behind — the budget still has to be enough (VZ-0160).
    /// Measured on the shipped Pile of Mess config: 24 rounds leaves 0 uncoloured across
    /// a 900-frame settle, 16 rounds leaves a handful per frame. `colorStats.uncolored`
    /// counts them — a scene paying for the tail shows up there as a number.
    /// (`.jonesPlassmann` only; `.speculative` runs `speculativeColorRounds` instead.)
    public var colorRounds: Int = 24

    /// How each substep's contacts are coloured (VZ-0160, stage B2 — see `ColoringScheme`,
    /// CoinDEMSolver+Coloring.swift). `.jonesPlassmann` is the default and colours exactly
    /// as before; `.speculative` colours a busy body's contacts several per round, so a
    /// heap whose busiest bodies touch 20–40 others is fully coloured in ~6 rounds, where
    /// Jones–Plassmann needs ~2.5 rounds per contact on the busiest body.
    public var coloringScheme: ColoringScheme = .jonesPlassmann
    /// Rounds encoded per substep by `.speculative` (two dispatches each). 6 coloured every
    /// contact of every heap measured for VZ-0160 (CoinDEMColoringTests).
    public var speculativeColorRounds: Int = 6

    /// Encode the graph colouring of the current contact buffer: build per-body
    /// contact lists, then run `colorRounds` Jones-Plassmann rounds — all in `cb`,
    /// no per-round readback. Assumes contacts were generated this command buffer.
    private func encodeColoring(_ cb: MTLCommandBuffer, fused: Bool = false) {
        if fused {
            // The cursor clamp + contact args, the per-body lists and the colouring seed in
            // one threadgroup (coinColorPrepareTG; the list counts were zeroed by coinSubstepBegin).
            dispatchOneGroup(cb, colorPrepareTGPipeline, label: "Coin.cs.colorPrepare") { enc in
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactArgsBuffer, offset: 0, index: 1)
                enc.setBuffer(self.solveTGSizeBuffer, offset: 0, index: 2)
                enc.setBuffer(self.maxContactsBuffer, offset: 0, index: 3)   // clamps the cursor (overflow)
                enc.setBuffer(self.contactBuffer, offset: 0, index: 4)
                enc.setBuffer(self.bodyContactsBuffer, offset: 0, index: 5)
                enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 6)
                enc.setBuffer(self.colorStatsBuffer, offset: 0, index: 7)
                enc.setBuffer(self.contactPriorityBuffer, offset: 0, index: 8)
                enc.setBuffer(self.colorBuffers[0], offset: 0, index: 9)
            }
        } else {
            dispatch(cb, writeContactArgsPipeline, threads: 1, label: "Coin.cs.contactArgs") { enc in
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactArgsBuffer, offset: 0, index: 1)
                enc.setBuffer(self.solveTGSizeBuffer, offset: 0, index: 2)
                enc.setBuffer(self.maxContactsBuffer, offset: 0, index: 3)   // clamps the cursor (overflow)
            }
            dispatch(cb, clearBodyPipeline, threads: maxCoins, label: "Coin.cs.clearBody") { enc in
                enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 0)
            }
            dispatchOverContacts(cb, buildBodyPipeline, label: "Coin.cs.buildBody") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.bodyContactsBuffer, offset: 0, index: 2)
                enc.setBuffer(self.bodyContactCountBuffer, offset: 0, index: 3)
                enc.setBuffer(self.colorStatsBuffer, offset: 0, index: 4)
            }
            dispatchOverContacts(cb, colorInitPipeline, label: "Coin.cs.colorInit") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(self.contactPriorityBuffer, offset: 0, index: 2)
                enc.setBuffer(self.colorBuffers[0], offset: 0, index: 3)
            }
        }
        // Ping-pong the rounds: each reads the previous round's colours and writes the
        // next, so a round is a pure function and the colouring can't depend on which
        // neighbour's write landed first (see coinColorRound). All rounds ride ONE serial
        // encoder — Metal orders dispatches within a compute encoder, so this is the same
        // barrier semantics as an encoder each, minus ~`colorRounds` encoder setups.
        var src = 0
        if coloringScheme == .speculative {
            // VZ-0160: tentative bids + conflict resolution, both pure functions of the
            // previous round's colours (the bids have their own buffer), ping-ponged the same
            // way — see coinColorTentative / coinColorResolve.
            if speculativeColorRounds > 0, let bid = speculativeBidBuffer(),
               let enc = cb.makeComputeCommandEncoder() {                // (never an empty encoder)
                enc.label = "Coin.cs.colorSpeculative"
                enc.setBuffer(contactBuffer, offset: 0, index: 0)
                enc.setBuffer(contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(bodyContactsBuffer, offset: 0, index: 2)
                enc.setBuffer(bodyContactCountBuffer, offset: 0, index: 3)
                enc.setBuffer(contactPriorityBuffer, offset: 0, index: 4)
                enc.setBuffer(bid, offset: 0, index: 6)
                let tg = MTLSize(width: Self.solveTGSize, height: 1, depth: 1)
                for _ in 0..<speculativeColorRounds {
                    enc.setComputePipelineState(colorTentativePipeline)
                    enc.setBuffer(colorBuffers[src], offset: 0, index: 5)
                    enc.dispatchThreadgroups(indirectBuffer: contactArgsBuffer, indirectBufferOffset: 0,
                                             threadsPerThreadgroup: tg)
                    enc.setComputePipelineState(colorResolvePipeline)
                    enc.setBuffer(colorBuffers[1 - src], offset: 0, index: 7)
                    enc.dispatchThreadgroups(indirectBuffer: contactArgsBuffer, indirectBufferOffset: 0,
                                             threadsPerThreadgroup: tg)
                    src = 1 - src
                }
                enc.endEncoding()
            }
        } else if colorRounds > 0, let enc = cb.makeComputeCommandEncoder() {   // (an empty encoder aborts under validation)
            enc.label = "Coin.cs.color"
            enc.setComputePipelineState(colorRoundPipeline)
            enc.setBuffer(contactBuffer, offset: 0, index: 0)
            enc.setBuffer(contactCountBuffer, offset: 0, index: 1)
            enc.setBuffer(bodyContactsBuffer, offset: 0, index: 2)
            enc.setBuffer(bodyContactCountBuffer, offset: 0, index: 3)
            enc.setBuffer(contactPriorityBuffer, offset: 0, index: 4)
            let tg = MTLSize(width: Self.solveTGSize, height: 1, depth: 1)
            for _ in 0..<colorRounds {
                enc.setBuffer(colorBuffers[src], offset: 0, index: 5)
                enc.setBuffer(colorBuffers[1 - src], offset: 0, index: 6)
                enc.dispatchThreadgroups(indirectBuffer: contactArgsBuffer, indirectBufferOffset: 0,
                                         threadsPerThreadgroup: tg)
                src = 1 - src
            }
            enc.endEncoding()
        }
        let final = colorBuffers[src]
        var sweep = UInt32(colorSweep)
        if fused {
            // Writeback, the colour buckets (count, scan, scatter — counts zeroed by
            // coinSubstepBegin), the uncoloured bucket and the per-colour args in one
            // threadgroup (coinColorFinishTG).
            dispatchOneGroup(cb, colorFinishTGPipeline, label: "Coin.cs.colorFinish") { enc in
                enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
                enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
                enc.setBuffer(final, offset: 0, index: 2)
                enc.setBuffer(self.colorStatsBuffer, offset: 0, index: 3)
                enc.setBytes(&sweep, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBuffer(self.colorCountBuffer, offset: 0, index: 5)
                enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 6)
                enc.setBuffer(self.colorContactsBuffer, offset: 0, index: 7)
                enc.setBuffer(self.contactPriorityBuffer, offset: 0, index: 8)
                enc.setBuffer(self.uncolSubBuffer, offset: 0, index: 9)
                enc.setBuffer(self.solveArgsBuffer, offset: 0, index: 10)
                enc.setBuffer(self.solveTGSizeBuffer, offset: 0, index: 11)
            }
            return
        }
        dispatchOverContacts(cb, colorWritebackPipeline, label: "Coin.cs.colorWriteback") { enc in
            enc.setBuffer(self.contactBuffer, offset: 0, index: 0)
            enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
            enc.setBuffer(final, offset: 0, index: 2)
            enc.setBuffer(self.colorStatsBuffer, offset: 0, index: 3)
            enc.setBytes(&sweep, length: MemoryLayout<UInt32>.size, index: 4)
        }
        // Counting-sort the contacts by colour so each colour's solve dispatches only its
        // own slice (coinCellCount/coinCellOffsetsScan/coinScatter, one level up).
        dispatch(cb, bucketClearPipeline, threads: Self.maxColors + 1, label: "Coin.cs.bucketClear") { enc in
            enc.setBuffer(self.colorCountBuffer, offset: 0, index: 0)
        }
        dispatchOverContacts(cb, bucketCountPipeline, label: "Coin.cs.bucketCount") { enc in
            enc.setBuffer(final, offset: 0, index: 0)
            enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
            enc.setBuffer(self.colorCountBuffer, offset: 0, index: 2)
        }
        dispatch(cb, bucketScanPipeline, threads: 1, label: "Coin.cs.bucketScan") { enc in
            enc.setBuffer(self.colorCountBuffer, offset: 0, index: 0)
            enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 1)
        }
        dispatchOverContacts(cb, bucketScatterPipeline, label: "Coin.cs.bucketScatter") { enc in
            enc.setBuffer(final, offset: 0, index: 0)
            enc.setBuffer(self.contactCountBuffer, offset: 0, index: 1)
            enc.setBuffer(self.colorCountBuffer, offset: 0, index: 2)   // zeroed by the scan → cursor
            enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 3)
            enc.setBuffer(self.colorContactsBuffer, offset: 0, index: 4)
            enc.setBuffer(self.contactBuffer, offset: 0, index: 5)        // a manifold head's point count
        }
        // maxColors threads write the per-colour args; the first threadgroup (up to 256
        // threads) also orders + sub-colours the uncoloured bucket (coinWriteColorArgs).
        dispatch(cb, writeColorArgsPipeline, threads: Self.colorArgsThreads, label: "Coin.cs.colorArgs") { enc in
            enc.setBuffer(self.colorOffsetBuffer, offset: 0, index: 0)
            enc.setBuffer(self.solveArgsBuffer, offset: 0, index: 1)
            enc.setBuffer(self.solveTGSizeBuffer, offset: 0, index: 2)
            enc.setBuffer(self.colorContactsBuffer, offset: 0, index: 3)   // thread 0 sorts the uncoloured bucket
            enc.setBuffer(self.contactPriorityBuffer, offset: 0, index: 4)
            enc.setBuffer(self.contactBuffer, offset: 0, index: 5)
            enc.setBuffer(self.uncolSubBuffer, offset: 0, index: 6)
            enc.setBuffer(self.colorStatsBuffer, offset: 0, index: 7)       // [4] += a bucket left unsolved
        }
    }

    /// Adjacency/colouring overflow counters, accumulated since the last `resetColorStats()`.
    /// `listOverflow` MUST stay 0: a body whose contact list overflowed is missing
    /// neighbours from the colouring (two of its contacts can then share a colour and
    /// race). `uncolored` should: an uncoloured contact is solved by the tail pass after
    /// the colours (it used to be skipped), so it costs time — and past 256 in a substep
    /// it is not solved at all (`uncoloredUnsolved`). `beyondSweep` counts coloured
    /// contacts the per-colour dispatches did not cover (also tail-solved). Gated by
    /// CoinDEMSolverTests.
    public var colorStats: (listOverflow: Int, uncolored: Int, maxColorUsed: Int, beyondSweep: Int) {
        let p = colorStatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 5)
        return (Int(p[0]), Int(p[1]), Int(p[2]), Int(p[3]))
    }

    /// Uncoloured contacts LEFT UNSOLVED since the last `resetColorStats()`: a substep
    /// whose leftover bucket exceeded the tail's 256-contact sub-colouring cap (or needed
    /// ≥ 64 sub-colours) skips it, as every uncoloured contact was skipped before the tail
    /// existed. Non-zero means the scene needs more `colorRounds` (VZ-0160).
    public var uncoloredUnsolved: Int {
        Int(colorStatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 5)[4])
    }

    /// Zero the counters (they accumulate over frames). The colour sweep is re-derived
    /// from `maxColorUsed`, so it restarts from nothing — safe, since colours beyond the
    /// sweep are solved by the serial tail pass.
    public func resetColorStats() {
        colorStatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 5).update(repeating: 0, count: 5)
        colorSweep = 0      // re-sized from the fresh stats; the tail pass covers the gap
    }

    /// The `.speculative` colouring's bid buffer, made on first use (VZ-0160). Private: the
    /// kernels write every uncoloured contact's bid before any read of it, so it is never
    /// cleared.
    private func speculativeBidBuffer() -> MTLBuffer? {
        if let b = colorBidBuffer { return b }
        let b = device.makeBuffer(length: MemoryLayout<UInt32>.stride * max(maxContacts, 1), options: .storageModePrivate)
        b?.label = "Coin.colorBid"
        colorBidBuffer = b
        return b
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
        frameJointCount = encodeJointListUpload(cb)
        encodeMarkJointedBodies(cb, coinCount: coinCount)
        encodeBroadphase(cb, coinCount: coinCount)
        encodeGenerateContacts(cb, coinCount: coinCount)
        if color { encodeColoring(cb) }
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: one-shot diagnostic readback
        return contactCount
    }

    private func writeUniforms(dt: Float, coinCount: Int) {
        uniformBuffer.contents().bindMemory(to: CoinUniforms.self, capacity: 1).pointee = makeUniforms(dt: dt, coinCount: coinCount)
    }

    /// The per-substep uniforms (also the small-world frame kernel's, over its one-cell grid).
    func makeUniforms(dt: Float, coinCount: Int) -> CoinUniforms {
        CoinUniforms(
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
            speculativeMargin: speculativeMargin,
            solverFlags: (scaleAwareDeadStop ? 1 : 0) | (accumulatedRollingResistance ? 2 : 0)
                       | (colliderCullDisabledForTesting ? 4 : 0) | (manifoldSolve ? 8 : 0)
                       | (torsionPatchSlots.isEmpty ? 0 : 16) | (polySATSerialForTesting ? 32 : 0)
                       | (preparedContactsLive ? 64 : 0) | (separatingSplitImpulse ? 128 : 0)
                       | (keepSleepingContacts ? 256 : 0),
            maxBodyBound: broadphaseClassicScanForTesting ? 0 : maxBodyBound,
            manifoldPasses: UInt32(max(1, manifoldInnerPasses)))
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

    /// One threadgroup of a single-threadgroup kernel (the fused substep kernels): as wide as
    /// the pipeline allows, up to 1024, in whole SIMD groups; the kernel loops over any count.
    private func dispatchOneGroup(_ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                                  label: String, _ bind: (MTLComputeCommandEncoder) -> Void) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        bind(enc)
        let simd = max(1, pipeline.threadExecutionWidth)
        let w = max(simd, (min(Self.fusedTGSize, pipeline.maxTotalThreadsPerThreadgroup) / simd) * simd)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Threads in a fused single-threadgroup kernel's group (capped by the pipeline): wide, so a
    /// big world's per-contact loops take few passes (an 800-egg rain's ~5 700 contacts: 6).
    private static let fusedTGSize = 1024

    /// The fused broadphase (coinBroadphaseTG) only while the grid's scan fits ONE pass of its group:
    /// ≤ one scan block (1 024 cells) per thread, ≤ 1 048 576 cells at 1 024 threads. Past that its
    /// scan runs chunk after chunk on the ONE core its threadgroup occupies, where the one-step kernels
    /// (coinCellBlockSums / coinCellOffsetsApply) spread the blocks over the whole GPU — measured by the
    /// stage-B3 verifier (M1 Max, a 400-egg rain at 1/120 s × 4, fused vs one-step interleaved in one
    /// process, GPU ms per frame p50): 0.75 M cells 18.7 vs 19.1, 0.96 M 21.6 vs 22.1 (fused ahead),
    /// then 1.28 M 27.3 vs 24.1, 1.58 M 32.9 vs 24.6, 2.06 M 40.3 vs 24.6, 7.98 M 137 vs 47.5. Both
    /// broadphases build the same cell offsets and sorted list (integer scans), so the choice changes
    /// no bit of the simulation (CoinDEMSmallWorldTests.testFusedBroadphaseScansAHouseScaleGrid).
    var fusedBroadphaseFits: Bool {
        if let m = fusedBroadphaseMaxBlocksForTesting { return numScanBlocks <= m }
        let simd = max(1, broadphaseTGPipeline.threadExecutionWidth)
        let group = max(simd, (min(Self.fusedTGSize, broadphaseTGPipeline.maxTotalThreadsPerThreadgroup) / simd) * simd)
        return numScanBlocks <= group
    }
    /// TEST SEAM: overrides the fused broadphase's block limit (the multi-chunk scan's parity test).
    var fusedBroadphaseMaxBlocksForTesting: Int?

    /// Dispatch a contact-indexed kernel over the LIVE contacts (threadgroup count
    /// written by `coinWriteContactArgs`) rather than the buffer capacity — a settled
    /// 176-body pile has ~2.7k contacts against a 12k capacity, and the colouring runs
    /// `colorRounds` passes over them.
    private func dispatchOverContacts(_ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                                      label: String, _ bind: (MTLComputeCommandEncoder) -> Void) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        bind(enc)
        enc.dispatchThreadgroups(indirectBuffer: contactArgsBuffer, indirectBufferOffset: 0,
                                 threadsPerThreadgroup: MTLSize(width: Self.solveTGSize, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Dispatch one colour-slice kernel per colour, all on ONE serial compute encoder.
    /// Each colour's threadgroup count was written by `coinWriteColorArgs` (16-byte
    /// stride), sized to that colour's slice of `colorContacts` — so an empty colour
    /// costs a 0-threadgroup dispatch, and the whole sweep costs one encoder instead of
    /// `maxColors` of them. Metal executes dispatches in a compute encoder in order, so
    /// the colour-to-colour Gauss-Seidel dependency is preserved.
    ///
    /// Then ONE threadgroup of `tail` solves every colour from `colors` up, a colour at a
    /// time (VZ-0154): the sweep is sized from the colours seen so far (+1), and a
    /// denser moment than that used to leave its extra colours unsolved for the substep
    /// (`colorStats.beyondSweep`). With the tail, the split point only trades parallel
    /// dispatches for serial work — every coloured contact is solved, in colour order.
    /// Last, it solves the contacts the colouring left UNCOLOURED (they used to be dropped
    /// — never solved), sub-colour by sub-colour (coinWriteColorArgs greedy-colours that
    /// bucket deterministically, up to 256 per substep; a larger one is left unsolved and
    /// counted in `uncoloredUnsolved`). An empty tail costs one tiny dispatch that returns
    /// at once.
    private func dispatchPerColor(_ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                                  tail: MTLComputePipelineState,
                                  colors: Int, label: String,
                                  _ bind: (MTLComputeCommandEncoder) -> Void) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        bind(enc)
        if colors > 0 {
            enc.setComputePipelineState(pipeline)
            let tg = MTLSize(width: Self.solveTGSize, height: 1, depth: 1)
            for color in 0..<colors {
                var cc = UInt32(color)
                enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 5)
                enc.dispatchThreadgroups(indirectBuffer: solveArgsBuffer,
                                         indirectBufferOffset: color * 16, threadsPerThreadgroup: tg)
            }
        }
        do {   // always: the uncoloured bucket lives in the tail too
            enc.setComputePipelineState(tail)
            enc.setBuffer(uncolSubBuffer, offset: 0, index: 10)
            var first = UInt32(colors)
            enc.setBytes(&first, length: MemoryLayout<UInt32>.size, index: 5)
            let w = min(Self.tailTGSize, tail.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// Threads in the serial tail pass's single threadgroup (capped by the pipeline).
    private static let tailTGSize = 256
    /// coinWriteColorArgs grid: ≥ maxColors (one args triple each) and wide enough that
    /// its first threadgroup spreads the uncoloured bucket's ordering + greedy scan.
    private static let colorArgsThreads = 256
}
