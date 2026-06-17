import Metal
import simd

/// RigidPileField — the general "pile of mess" facade over the shared CoinDEM GPU
/// solver: drop a heap of MIXED rigid bodies (coins, chips, dice, marbles, pegs,
/// crates, debris) into a container and let them settle into a real heap, with an
/// honest angle of repose and micro-avalanches.
///
/// It is the superset sibling of `RigidBallField` (spheres only) and `RigidDiscField`
/// (flat discs): a THIN convenience over `CoinDEMSolver`, **not a new solver**. Where
/// those wrap one shape, this exposes everything the solver now collides —
///
///   • `dropDisc`   flat capped cylinder (coin / chip / token / checker)
///   • `dropSphere` radius == half-thickness (marble / ball / pellet)
///   • `dropRod`    elongated capped cylinder, half-length ≫ radius (peg / rod / log)
///   • `dropBox`    oriented box, three half-extents (die / crate / brick / block)
///
/// — and they all share ONE pile, colliding with each other (box–box, box–disc,
/// disc–disc) and with the static environment. Each drop takes a `type` tag, so a
/// mixed pile draws each asset through its own type-filtered renderer. Plus the two
/// mechanics piles want on top: BUOYANCY (`setBuoyant`/`kick`) and a reciprocating
/// PUSHER (`setPusher`), and a payout/recycle line (`cull`).
///
/// ```swift
/// pile = RigidPileField(engine: .shared, config: .init(
///     maxBodies: 800, bodyScale: 0.15,                 // ← cell sized to the BIGGEST body
///     bounds: (SIMD3(-1,-0.5,-1), SIMD3(1, 2, 1))),
///   colliders: RigidPileField.bin(innerHalf: SIMD2(0.8, 0.8), floorY: 0))
/// pile.dropBox(at: SIMD3(0, 1.2, 0), halfExtents: SIMD3(0.08,0.08,0.08), tumble: SIMD3(2,0,1))
/// pile.dropDisc(at: SIMD3(0.1, 1.4, 0), radius: 0.12, halfThickness: 0.01)
/// // …per frame…
/// pile.encode(to: cb, dt: dt)
/// if let m = pile.modelMatrix(of: id, scale: 1) { /* feed renderer */ }
/// ```
///
/// Plugs into the shared runtime: takes a `SimEngine`, encodes on the engine's queue
/// (via the wrapped solver), no parallel device stack, no per-frame
/// `waitUntilCompleted`. The wrapped solver conforms to `PenetrationProbing`, so a
/// scene returns `pile.solver` from `penetrationProbers()`.
///
/// NOTE on stacking: this is a Jacobi position solver, so it makes a believable HEAP
/// (the realistic case) but not a tall *stable tower* — a body squeezed between
/// contacts above and below doesn't converge (see docs/coindem-ecosystem.md). Great
/// for mess; not for Jenga.
///
/// plausibility: real — delegates entirely to the real GPU DEM solver
///   (`CoinDEMSolver`): real gravity / restitution / friction / oriented box–box,
///   box–disc, and disc–disc contact + analytic plane/box collision. Buoyancy is real
///   negative gravity, the pusher a real kinematic collider — no faked motion.
@MainActor
public final class RigidPileField {

    /// Opaque handle to one simulated body (the underlying solver slot index).
    public typealias BodyID = Int

    /// One-shot configuration. The tunables map onto `CoinDEMSolver`'s knobs; the
    /// defaults are a calm "general rigid heap" baseline. **`bodyScale` sizes the
    /// broadphase grid cell — set it to the bounding radius of the LARGEST body the
    /// pile will ever hold** (the biggest box's `length(halfExtents)`, the biggest
    /// disc's `√(R²+h²)`), or contacts on oversized bodies will be missed.
    public struct Config {
        public var maxBodies: Int
        public var bodyScale: Float
        public var bounds: (min: SIMD3<Float>, max: SIMD3<Float>)
        public var gravity: Float = 9.8
        public var restitution: Float = 0.1
        /// Tangential velocity RETENTION (PBD convention: lower = stickier).
        public var friction: Float = 0.9
        /// Coulomb μ for friction-WITH-TORQUE (0 = off → slippery, bodies slide; raise
        /// it so bodies grip a slope, roll, and the heap's repose angle climbs).
        public var frictionCoeff: Float = 0.0
        public var angularFriction: Float = 0.8
        public var linearDamping: Float = 0.99
        public var angularDamping: Float = 0.985
        public var contactRelax: Float = 0.7
        public var sleepLinearVelocity: Float = 0.03
        public var rollingResistance: Float = 0.0
        public var floorY: Float?
        public var maxHorizontalSpeed: Float = 2.0
        public var maxSpeed: Float = 8.0
        public var maxAngularSpeed: Float = 14.0
        public var fixedDt: Float = 1.0 / 240.0
        public var maxSubsteps: Int = 6
        public var iterations: Int = 14
        /// Opt into the new graph-colored sequential-impulse CONSTRAINT solver (real
        /// Gauss-Seidel convergence + split-impulse rest stability) instead of the
        /// legacy Jacobi position solver. Off by default (the proven path for shipping
        /// scenes); Pile of Mess turns it on.
        public var useConstraintSolver: Bool = false
        /// Sequential-impulse velocity iterations per substep (constraint solver only).
        public var velocityIterations: Int = 8

        public init(maxBodies: Int, bodyScale: Float,
                    bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
                    gravity: Float = 9.8) {
            self.maxBodies = maxBodies
            self.bodyScale = bodyScale
            self.bounds = bounds
            self.gravity = gravity
        }
    }

    /// The wrapped GPU solver — exposed so a scene returns it from
    /// `penetrationProbers()` (it conforms to `PenetrationProbing`) and for advanced
    /// needs; prefer the `RigidPileField` API for everyday use.
    public let solver: CoinDEMSolver

    private let restGravity: Float
    private var staticColliders: [CoinStaticCollider]
    private var pusher: CoinStaticCollider?

    // ── Init ───────────────────────────────────────────────────────

    /// Production init: pipelines come from the engine's memoised cache.
    public convenience init?(engine: SimEngine, config: Config,
                             colliders: [CoinStaticCollider] = []) {
        guard let s = CoinDEMSolver(engine: engine, maxCoins: config.maxBodies,
                                    coinRadius: config.bodyScale,
                                    halfThickness: config.bodyScale,
                                    boundsMin: config.bounds.min,
                                    boundsMax: config.bounds.max) else { return nil }
        self.init(solver: s, config: config, colliders: colliders)
    }

    /// Test seam: solver built from a runtime-compiled library (SwiftPM CLI has no
    /// metallib), mirroring `CoinDEMSolver` / `RigidDiscField`.
    convenience init?(engine: SimEngine, library: MTLLibrary, config: Config,
                      colliders: [CoinStaticCollider] = []) {
        guard let s = CoinDEMSolver(engine: engine, library: library, maxCoins: config.maxBodies,
                                    coinRadius: config.bodyScale,
                                    halfThickness: config.bodyScale,
                                    boundsMin: config.bounds.min,
                                    boundsMax: config.bounds.max) else { return nil }
        self.init(solver: s, config: config, colliders: colliders)
    }

    private init(solver s: CoinDEMSolver, config: Config, colliders: [CoinStaticCollider]) {
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
        s.rollingResistance = config.rollingResistance
        s.maxHSpeed = config.maxHorizontalSpeed
        s.maxSpeed = config.maxSpeed
        s.maxOmega = config.maxAngularSpeed
        s.fixedDt = config.fixedDt
        s.maxSubsteps = config.maxSubsteps
        s.iterations = config.iterations
        s.solverMode = config.useConstraintSolver ? .constraint : .legacy
        s.velocityIterations = config.velocityIterations
        s.levelingEnabled = false
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
    /// ±`innerHalf` from `center`. Wall planes are infinite half-spaces (bodies can't
    /// escape over the top).
    public static func bin(center: SIMD3<Float> = .zero,
                           innerHalf: SIMD2<Float>,
                           floorY: Float) -> [CoinStaticCollider] {
        // Each wall's offset d is the signed distance of that wall along its inward
        // normal (a plane is the half-space n·x ≥ d).
        [
            .plane(normal: SIMD3(0, 1, 0), offset: floorY),
            .plane(normal: SIMD3( 1, 0, 0), offset:  center.x - innerHalf.x),
            .plane(normal: SIMD3(-1, 0, 0), offset: -(center.x + innerHalf.x)),
            .plane(normal: SIMD3(0, 0,  1), offset:  center.z - innerHalf.y),
            .plane(normal: SIMD3(0, 0, -1), offset: -(center.z + innerHalf.y)),
        ]
    }

    /// A floor plane at height `y`.
    public static func floor(y: Float) -> CoinStaticCollider {
        .plane(normal: SIMD3(0, 1, 0), offset: y)
    }

    // ── Drop one of each shape (all return the BodyID, nil if the pile is full) ──

    /// Drop a flat DISC (coin / chip / token / checker).
    @discardableResult
    public func dropDisc(at position: SIMD3<Float>, radius: Float, halfThickness: Float,
                         velocity: SIMD3<Float> = .zero, spin: Float = 0,
                         orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                         tumble: SIMD3<Float> = .zero, mass: Float = 1,
                         type: UInt32 = 0) -> BodyID? {
        solver.spawn(at: position, velocity: velocity, spin: spin, orient: orient,
                     tumble: tumble, radius: radius, halfThickness: halfThickness,
                     mass: mass, type: type)
    }

    /// Drop a SPHERE (marble / ball / pellet) — a capped cylinder with radius ==
    /// half-thickness, so it collides as a near-sphere.
    @discardableResult
    public func dropSphere(at position: SIMD3<Float>, radius: Float,
                           velocity: SIMD3<Float> = .zero,
                           tumble: SIMD3<Float> = .zero, mass: Float = 1,
                           type: UInt32 = 0) -> BodyID? {
        solver.spawn(at: position, velocity: velocity, tumble: tumble,
                     radius: radius, halfThickness: radius, mass: mass, type: type)
    }

    /// Drop a ROD / PEG (elongated capped cylinder, `halfLength` along its local +Y).
    @discardableResult
    public func dropRod(at position: SIMD3<Float>, radius: Float, halfLength: Float,
                        velocity: SIMD3<Float> = .zero,
                        orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                        tumble: SIMD3<Float> = .zero, mass: Float = 1,
                        type: UInt32 = 0) -> BodyID? {
        solver.spawn(at: position, velocity: velocity, orient: orient, tumble: tumble,
                     radius: radius, halfThickness: halfLength, mass: mass, type: type)
    }

    /// Drop an oriented BOX (die / crate / brick / block).
    @discardableResult
    public func dropBox(at position: SIMD3<Float>, halfExtents: SIMD3<Float>,
                        velocity: SIMD3<Float> = .zero,
                        orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                        tumble: SIMD3<Float> = .zero, mass: Float = 1,
                        type: UInt32 = 0) -> BodyID? {
        solver.spawnBox(at: position, halfExtents: halfExtents, velocity: velocity,
                        orient: orient, tumble: tumble, mass: mass, type: type)
    }

    /// Recycle a body's slot (it left the frame / dropped out a gap). Handle invalid after.
    public func remove(_ id: BodyID) { solver.despawn(id) }

    /// Recycle every body whose COM has fallen below `y` (a payout line). Returns the count.
    @discardableResult
    public func cull(belowY y: Float) -> Int { solver.cull(belowY: y) }

    /// Recycle the whole pile at once (e.g. a reset).
    public func clearAll() { solver.clearAll() }

    // ── Buoyancy ───────────────────────────────────────────────────

    /// Flip between resting under gravity and rising under BUOYANCY (gravity inverts
    /// to `-strength·restGravity`, a gentle upward pull). Heavy linear damping in the
    /// config makes it a slow drift, not a launch. Wakes the whole pile — a global
    /// gravity flip is exactly the change island-sleeping can't detect locally.
    public func setBuoyant(_ on: Bool, strength: Float = 0.12) {
        solver.gravity = on ? -strength * restGravity : restGravity
        solver.wakeAll()
    }

    /// Wake every sleeping body (call on any global change — buoyancy flip, reset, a
    /// gravity-slider change — so frozen bodies respond).
    public func wakeAll() { solver.wakeAll() }

    /// Give every active body a small, distinct random velocity so a packed pile
    /// floats up (or scatters) as INDIVIDUALS rather than as one rigid clump.
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

    // ── Pusher ─────────────────────────────────────────────────────

    /// Install / move the reciprocating pusher plate (forward-only +Z shove). Call
    /// each frame with the plate's current centre and forward velocity.
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

    /// A full TRS model matrix for a body (translation · orientation · uniform
    /// scale), ready to drop into a renderer instance. nil if inactive.
    ///
    /// For a BOX, the solver already bakes the per-instance extents into the transform
    /// feed used by `CoinInstancedRenderer` (draw a unit-cube mesh); this convenience
    /// applies a uniform `scale` for disc/sphere/rod meshes authored at unit size.
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
    /// buffer (the shared engine queue). Order it before reading transforms this frame.
    public func encode(to cb: MTLCommandBuffer, dt: Float) {
        solver.encode(to: cb, wallDt: dt)
    }
}
