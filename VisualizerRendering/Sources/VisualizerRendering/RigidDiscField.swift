import Metal
import simd

/// RigidDiscField — a drop-in "go-to" field for a pile of flat DISCS (coins, poker
/// chips, tortilla chips, checkers, tokens) on the shared CoinDEM GPU solver.
///
/// It is the disc sibling of `RigidBallField` (spheres): a THIN convenience over
/// `CoinDEMSolver`, **not a new solver**. Where `RigidBallField` sizes the dynamic
/// collider as a sphere (radius == halfThickness), this one keeps the true thin
/// oriented cylinder and ships the "slippery flat metal that shears flat and rests
/// calm" tunables that Coin Pusher and Tavern were each hand-rolling — plus the two
/// disc-pile mechanics those scenes add on top:
///
///   • BUOYANCY — flip to a negative-gravity rise with heavy drag so the discs float
///     up and tumble (Tavern's "lighter than air" chips). `setBuoyant(_:)` + `kick()`.
///   • PUSHER — a reciprocating plate that shoves the pile forward (Coin Pusher).
///     `setPusher(...)` updates the kinematic collider each frame.
///
/// ```swift
/// field = RigidDiscField(engine: .shared, config: .init(
///     maxDiscs: 600, discRadius: 0.12, discHalfThickness: 0.009,
///     bounds: (SIMD3(-1, -0.5, -1), SIMD3(1, 2, 1))),
///   colliders: RigidDiscField.bin(innerHalf: SIMD2(0.8, 0.8), floorY: 0))
/// let id = field.drop(at: SIMD3(0, 1.2, 0), tumble: SIMD3(2, 0, 1))
/// // …per frame…
/// field.encode(to: cb, dt: dt)
/// if let m = field.modelMatrix(of: id, scale: 1) { /* feed renderer */ }
/// ```
///
/// Plugs into the shared runtime: takes a `SimEngine`, encodes on the engine's queue
/// (via the wrapped solver), no parallel device stack, no per-frame
/// `waitUntilCompleted`. The wrapped solver conforms to `PenetrationProbing`, so a
/// scene can return `field.solver` from `penetrationProbers()`.
///
/// plausibility: real — delegates entirely to the real GPU DEM solver
///   (`CoinDEMSolver`): real gravity / restitution / friction / disc–disc oriented
///   contact + analytic plane/box collision. Buoyancy is real negative gravity, the
///   pusher a real kinematic collider — no faked motion, no bounding-volume stand-in.
@MainActor
public final class RigidDiscField {

    /// Opaque handle to one simulated disc (the underlying solver slot index).
    public typealias BodyID = Int

    /// One-shot configuration. The tunables map onto `CoinDEMSolver`'s knobs; the
    /// defaults are the "slippery metal disc that shears flat and rests calm"
    /// baseline both disc scenes converged on. `discRadius` sizes the broadphase
    /// cell — set it to the LARGEST disc the field will ever spawn.
    public struct Config {
        public var maxDiscs: Int
        public var discRadius: Float
        public var discHalfThickness: Float
        public var bounds: (min: SIMD3<Float>, max: SIMD3<Float>)
        public var gravity: Float = 9.8
        public var restitution: Float = 0.1
        /// Tangential velocity RETENTION (PBD convention: lower = stickier). High by
        /// default so a disc that lands tilted shears flat across its neighbours.
        public var friction: Float = 0.94
        /// Coulomb μ for friction-WITH-TORQUE. 0 (default) keeps discs slippery (they
        /// shear flat instead of gripping); raise it for discs that should grip a
        /// slope or roll on edge.
        public var frictionCoeff: Float = 0.0
        public var angularFriction: Float = 0.78
        public var linearDamping: Float = 0.99
        public var angularDamping: Float = 0.97
        public var contactRelax: Float = 0.6
        public var sleepLinearVelocity: Float = 0.03
        public var floorY: Float?
        public var maxHorizontalSpeed: Float = 1.6
        public var maxSpeed: Float = 6.0
        public var maxAngularSpeed: Float = 12.0
        public var fixedDt: Float = 1.0 / 240.0
        public var maxSubsteps: Int = 6
        public var iterations: Int = 14

        public init(maxDiscs: Int, discRadius: Float, discHalfThickness: Float,
                    bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
                    gravity: Float = 9.8) {
            self.maxDiscs = maxDiscs
            self.discRadius = discRadius
            self.discHalfThickness = discHalfThickness
            self.bounds = bounds
            self.gravity = gravity
        }
    }

    /// The wrapped GPU solver. Exposed so a scene can return it from
    /// `penetrationProbers()` (it conforms to `PenetrationProbing`) and for advanced
    /// needs; prefer the `RigidDiscField` API for everyday use.
    public let solver: CoinDEMSolver

    /// Gravity magnitude requested for the upright (non-buoyant) state — restored
    /// when buoyancy is turned back off.
    private let restGravity: Float
    private var staticColliders: [CoinStaticCollider]
    private var pusher: CoinStaticCollider?

    /// Production init: the solver pulls its pipelines from the engine's memoised
    /// cache (the package's compiled metallib).
    public convenience init?(engine: SimEngine, config: Config,
                             colliders: [CoinStaticCollider] = []) {
        guard let s = CoinDEMSolver(engine: engine, maxCoins: config.maxDiscs,
                                    coinRadius: config.discRadius,
                                    halfThickness: config.discHalfThickness,
                                    boundsMin: config.bounds.min,
                                    boundsMax: config.bounds.max) else { return nil }
        self.init(solver: s, config: config, colliders: colliders)
    }

    /// Test seam: the solver is built from a runtime-compiled library (the SwiftPM
    /// CLI produces no metallib), mirroring `CoinDEMSolver`'s own test seam.
    convenience init?(engine: SimEngine, library: MTLLibrary, config: Config,
                      colliders: [CoinStaticCollider] = []) {
        guard let s = CoinDEMSolver(engine: engine, library: library, maxCoins: config.maxDiscs,
                                    coinRadius: config.discRadius,
                                    halfThickness: config.discHalfThickness,
                                    boundsMin: config.bounds.min,
                                    boundsMax: config.bounds.max) else { return nil }
        self.init(solver: s, config: config, colliders: colliders)
    }

    /// Designated init — applies the config's tunables onto an already-built solver.
    private init(solver s: CoinDEMSolver, config: Config,
                 colliders: [CoinStaticCollider]) {
        self.solver = s
        self.restGravity = config.gravity
        self.staticColliders = colliders
        self.pusher = nil
        s.gravity = config.gravity
        s.restitution = config.restitution
        s.friction = config.friction
        s.frictionCoeff = config.frictionCoeff
        s.angFriction = config.angularFriction
        s.linDamping = config.linearDamping
        s.angDamping = config.angularDamping
        s.contactRelax = config.contactRelax
        s.sleepLinVel = config.sleepLinearVelocity
        s.maxHSpeed = config.maxHorizontalSpeed
        s.maxSpeed = config.maxSpeed
        s.maxOmega = config.maxAngularSpeed
        s.fixedDt = config.fixedDt
        s.maxSubsteps = config.maxSubsteps
        s.iterations = config.iterations
        s.levelingEnabled = false           // slippery discs flatten via the contact manifold
        s.floorY = config.floorY ?? config.bounds.min.y
        s.setColliders(colliders)
    }

    // ── Live tunables (pass-through, for sliders) ──────────────────
    public var gravity: Float { get { solver.gravity } set { solver.gravity = newValue } }
    public var restitution: Float { get { solver.restitution } set { solver.restitution = newValue } }
    public var friction: Float { get { solver.friction } set { solver.friction = newValue } }
    public var frictionCoeff: Float { get { solver.frictionCoeff } set { solver.frictionCoeff = newValue } }
    /// Live body count (active slots).
    public var activeCount: Int { solver.activeCount }

    // ── Colliders ──────────────────────────────────────────────────

    /// Replace the static collider set (the pusher, if any, is re-appended).
    public func setColliders(_ list: [CoinStaticCollider]) {
        staticColliders = list
        rebuildColliders()
    }

    private func rebuildColliders() {
        solver.setColliders(pusher.map { staticColliders + [$0] } ?? staticColliders)
    }

    /// An open-top rectangular bin: a floor plane plus four upright wall planes at
    /// ±`innerHalf` from `center`. Matches the floor + 4-wall layout Coin Pusher and
    /// Tavern hand-write. Wall planes are infinite half-spaces (discs can't escape
    /// over the top); for a finite cabinet with a payout gap, build box walls.
    public static func bin(center: SIMD3<Float> = .zero,
                           innerHalf: SIMD2<Float>,
                           floorY: Float) -> [CoinStaticCollider] {
        // A plane is the half-space n·x ≥ d (the body is kept on the +normal side),
        // so each wall's d is the signed distance of that wall along its inward normal.
        [
            .plane(normal: SIMD3(0, 1, 0), offset: floorY),
            .plane(normal: SIMD3( 1, 0, 0), offset:  center.x - innerHalf.x),   // left  wall, keep x ≥
            .plane(normal: SIMD3(-1, 0, 0), offset: -(center.x + innerHalf.x)),  // right wall, keep x ≤
            .plane(normal: SIMD3(0, 0,  1), offset:  center.z - innerHalf.y),   // near  wall, keep z ≥
            .plane(normal: SIMD3(0, 0, -1), offset: -(center.z + innerHalf.y)),  // far   wall, keep z ≤
        ]
    }

    /// A floor at `y` (re-export of the plane half-space `y' ≥ y`).
    public static func floor(y: Float) -> CoinStaticCollider {
        .plane(normal: SIMD3(0, 1, 0), offset: y)
    }

    // ── Drop / spawn ───────────────────────────────────────────────

    /// Drop a disc with an explicit initial velocity. Returns its `BodyID`, or nil if
    /// the field is full. `spin` is about the disc's own axis; `tumble` is world-frame
    /// initial angular velocity. `type` tags it for a mixed-asset renderer.
    @discardableResult
    public func drop(at position: SIMD3<Float>,
                     velocity: SIMD3<Float> = .zero,
                     spin: Float = 0,
                     orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                     tumble: SIMD3<Float> = .zero,
                     radius: Float? = nil,
                     halfThickness: Float? = nil,
                     mass: Float = 1,
                     type: UInt32 = 0) -> BodyID? {
        solver.spawn(at: position, velocity: velocity, spin: spin, orient: orient,
                     tumble: tumble, radius: radius, halfThickness: halfThickness,
                     mass: mass, type: type)
    }

    /// Recycle a disc's slot (e.g. it left through a gap). The handle is invalid after.
    public func remove(_ id: BodyID) { solver.despawn(id) }

    /// Recycle every disc whose COM has fallen below `y` (the payout line). Returns
    /// how many were recycled.
    @discardableResult
    public func cull(belowY y: Float) -> Int { solver.cull(belowY: y) }

    // ── Buoyancy (the Tavern "float up" mechanic) ──────────────────

    /// Flip the field between resting under gravity and rising under BUOYANCY. When
    /// buoyant, gravity inverts to `-strength·restGravity` (a gentle upward pull) so
    /// the discs float up and tumble; the heavy linear drag in the config eases them
    /// into a slow terminal-velocity rise rather than accelerating. Off restores the
    /// resting gravity.
    public func setBuoyant(_ on: Bool, strength: Float = 0.12) {
        solver.gravity = on ? -strength * restGravity : restGravity
    }

    /// Give every active disc a SMALL, distinct random velocity so a packed pile
    /// floats up as INDIVIDUALS rather than rising as one rigid clump (the Tavern
    /// kick). Deterministic per-slot RNG → a different gentle drift per disc.
    public func kick(horizontal: Float = 0.18, vertical: Float = 0.12) {
        let ptr = solver.coinBuffer.buffer.contents()
            .bindMemory(to: CoinBody.self, capacity: solver.highWater)
        for slot in 0..<solver.highWater where ptr[slot].posInvMass.w != 0 {
            var h = UInt64(bitPattern: Int64(slot)) &* 0x9E3779B97F4A7C15 &+ 0xD1B54A32D192ED03
            func rnd() -> Float {
                h ^= h >> 30; h &*= 0xBF58476D1CE4E5B9; h ^= h >> 27
                return Float(h >> 40) * (2.0 / 16_777_216.0) - 1.0
            }
            ptr[slot].vel.x += rnd() * horizontal
            ptr[slot].vel.y += rnd() * vertical
            ptr[slot].vel.z += rnd() * horizontal
        }
    }

    // ── Pusher (the Coin Pusher reciprocating plate) ───────────────

    /// Install / move the reciprocating pusher plate. Call each frame with the plate's
    /// current centre and forward velocity; it only ever shoves discs forward (+Z) at
    /// most `velocity.z`, never down, so a disc at the paddle corner can't be squeezed
    /// into the floor and launched (see CoinDEM.metal's forward-only pusher).
    public func setPusher(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                          velocity: SIMD3<Float>) {
        pusher = .pusherPlate(center: center, halfExtents: halfExtents, velocity: velocity)
        rebuildColliders()
    }

    /// Remove the pusher plate.
    public func clearPusher() { pusher = nil; rebuildColliders() }

    // ── Readback (one-frame-stale; fine for a live tick) ───────────
    public func position(of id: BodyID) -> SIMD3<Float>? { solver.position(of: id) }
    public func velocity(of id: BodyID) -> SIMD3<Float>? { solver.velocity(of: id) }
    public func orientation(of id: BodyID) -> simd_quatf? { solver.orientation(of: id) }

    /// A full TRS model matrix for a disc (translation · orientation · uniform
    /// scale), ready to drop into a renderer instance. nil if inactive.
    public func modelMatrix(of id: BodyID, scale: Float) -> simd_float4x4? {
        guard let p = solver.position(of: id), let q = solver.orientation(of: id) else { return nil }
        let r = simd_matrix3x3(q)
        let c0 = r.columns.0 * scale, c1 = r.columns.1 * scale, c2 = r.columns.2 * scale
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4(c0.x, c0.y, c0.z, 0)
        m.columns.1 = SIMD4(c1.x, c1.y, c1.z, 0)
        m.columns.2 = SIMD4(c2.x, c2.y, c2.z, 0)
        m.columns.3 = SIMD4(p.x, p.y, p.z, 1)
        return m
    }

    // ── Advance ────────────────────────────────────────────────────

    /// Advance the simulation by `dt` seconds, encoding onto the caller's command
    /// buffer (the shared engine queue). Order it before any read of the transforms
    /// in the same frame.
    public func encode(to cb: MTLCommandBuffer, dt: Float) {
        solver.encode(to: cb, wallDt: dt)
    }
}
