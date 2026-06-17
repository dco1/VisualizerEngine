import Metal
import simd

/// RigidBallField — a drop-in "go-to" rigid-body physics field: bodies under
/// gravity that bounce off analytic floor / wall / box colliders with real
/// restitution + friction and resolve body–body non-penetration on the GPU.
///
/// SHAPE-AGNOSTIC with a smart default. Every `launch` takes a `Shape` and defaults
/// to `.sphere` (the field is named for balls, and a true sphere is the honest model
/// for one — orientation-independent contact, so it rests at exactly its radius and
/// never floats on a tilted rim or clips through a flat face). Pass `.disc(…)` or
/// `.box(…)` to reuse the same field for coins, tiles, or crates without a new
/// wrapper. (Disc/box piles that want their own ergonomics still have `RigidDiscField`
/// / `RigidPileField`; this one is the general entry point.)
///
/// It is a THIN convenience over `CoinDEMSolver` (the shared GPU DEM runtime),
/// **not a new solver**. It packages the assembly every native physics scene was
/// hand-repeating — make the solver, set tunables, build the collider list, throw
/// bodies on a ballistic arc, advance, read transforms — into one object, so a
/// scene can express *"bouncing balls, restitution X, off this floor and this
/// wall"* in a few lines and own only its own contact / gameplay logic:
///
/// ```swift
/// field = RigidBallField(engine: .shared, config: .init(
///     maxBodies: 44, bodyRadius: 0.75,
///     bounds: (SIMD3(-4, -3, -4), SIMD3(4, 6, 4)),
///     gravity: 9.81, restitution: 0.55, friction: 0.86),
///   colliders: [RigidBallField.floor(y: 0),
///               RigidBallField.wall(normal: SIMD3(0, 0, 1), offset: -3)])
/// let id = field.launch(from: spawn, toward: target, flightTime: 0.9, radius: 0.11)
/// // …per frame…
/// field.encode(to: cb, dt: dt)
/// if let m = field.modelMatrix(of: id, scale: 0.11) { /* feed renderer */ }
/// ```
///
/// Plugs into the shared runtime: takes a `SimEngine`, encodes on the engine's
/// command queue (via the wrapped solver), no parallel device stack, no per-frame
/// `waitUntilCompleted`. The wrapped solver conforms to `PenetrationProbing`, so a
/// scene can return `field.solver` from `penetrationProbers()`.
///
/// plausibility: real — delegates entirely to the real GPU DEM solver
///   (`CoinDEMSolver`): real gravity / restitution / friction / ball–ball contact
///   + analytic plane/box collision, and a gravity-correct ballistic launch. No
///   faked physics, no bounding-volume stand-in, no new parallel pipeline.
@MainActor
public final class RigidBallField {

    /// Opaque handle to one simulated body (the underlying solver slot index).
    public typealias BodyID = Int

    /// One-shot configuration. The tunables map straight onto `CoinDEMSolver`'s
    /// knobs; the defaults are a sensible "thrown ball that bounces but isn't a
    /// superball" baseline. `bodyRadius` sizes the broadphase grid cell — set it to
    /// the LARGEST radius the field will ever spawn (individual `launch` calls pass
    /// their own per-body radius).
    public struct Config {
        public var maxBodies: Int
        public var bodyRadius: Float
        public var bounds: (min: SIMD3<Float>, max: SIMD3<Float>)
        public var gravity: Float = 9.81
        public var restitution: Float = 0.5
        public var friction: Float = 0.9
        /// Coulomb μ for velocity-level friction-with-torque at contacts. 0 (default)
        /// keeps the legacy isotropic tangential damp (balls slide, never roll); a
        /// positive value makes a ball spin up toward rolling-without-slipping as it
        /// runs along the floor and sets its angle of repose against a slope.
        public var frictionCoeff: Float = 0.0
        /// Resists the rolling component of spin at a contact (a ball coasting on the
        /// floor slows instead of rolling forever). 0 = off.
        public var rollingResistance: Float = 0.0
        public var angularFriction: Float = 0.8
        public var linearDamping: Float = 0.999
        public var contactRelax: Float = 0.9
        public var sleepLinearVelocity: Float = 0.04
        /// The solver's built-in ground plane height. Defaults to the bounds floor;
        /// set it to match your floor collider (or far below the volume to rely
        /// solely on collider planes).
        public var floorY: Float?
        public var maxSpeed: Float = 20
        public var maxHorizontalSpeed: Float = 14
        public var maxAngularSpeed: Float = 24
        public var fixedDt: Float = 1.0 / 240.0
        public var maxSubsteps: Int = 8
        public var iterations: Int = 12

        public init(maxBodies: Int, bodyRadius: Float,
                    bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
                    gravity: Float = 9.81, restitution: Float = 0.5,
                    friction: Float = 0.9) {
            self.maxBodies = maxBodies
            self.bodyRadius = bodyRadius
            self.bounds = bounds
            self.gravity = gravity
            self.restitution = restitution
            self.friction = friction
        }
    }

    /// The wrapped GPU solver. Exposed so a scene can return it from
    /// `penetrationProbers()` (it conforms to `PenetrationProbing`) and for
    /// advanced needs; prefer the `RigidBallField` API for everyday use.
    public let solver: CoinDEMSolver

    public init?(engine: SimEngine, config: Config,
                 colliders: [CoinStaticCollider] = []) {
        // coinRadius == halfThickness == bodyRadius here only sizes the broadphase
        // grid cell (≈ one body diameter); the per-body shape + dims come from each
        // `launch` (a sphere gets exact orientation-independent contact via
        // `spawnSphere`, not this nominal coin geometry).
        guard let s = CoinDEMSolver(engine: engine, maxCoins: config.maxBodies,
                                    coinRadius: config.bodyRadius,
                                    halfThickness: config.bodyRadius,
                                    boundsMin: config.bounds.min,
                                    boundsMax: config.bounds.max) else { return nil }
        self.solver = s
        s.gravity = config.gravity
        s.restitution = config.restitution
        s.friction = config.friction
        s.frictionCoeff = config.frictionCoeff
        s.rollingResistance = config.rollingResistance
        s.angFriction = config.angularFriction
        s.linDamping = config.linearDamping
        s.contactRelax = config.contactRelax
        s.sleepLinVel = config.sleepLinearVelocity
        s.maxSpeed = config.maxSpeed
        s.maxHSpeed = config.maxHorizontalSpeed
        s.maxOmega = config.maxAngularSpeed
        s.fixedDt = config.fixedDt
        s.maxSubsteps = config.maxSubsteps
        s.iterations = config.iterations
        s.levelingEnabled = false           // spheres need no disc-leveling
        s.floorY = config.floorY ?? config.bounds.min.y
        s.setColliders(colliders)
    }

    // ── Live tunables (pass-through, for sliders) ──────────────────
    public var gravity: Float { get { solver.gravity } set { solver.gravity = newValue } }
    public var restitution: Float { get { solver.restitution } set { solver.restitution = newValue } }
    public var friction: Float { get { solver.friction } set { solver.friction = newValue } }
    /// Live body count (active slots).
    public var activeCount: Int { solver.activeCount }

    // ── Colliders ──────────────────────────────────────────────────
    public func setColliders(_ list: [CoinStaticCollider]) { solver.setColliders(list) }

    /// An upward-facing floor plane at height `y` (the half-space `y' ≥ y`).
    public static func floor(y: Float) -> CoinStaticCollider {
        .plane(normal: SIMD3(0, 1, 0), offset: y)
    }
    /// A ceiling plane: bodies stay below `y` (the half-space `-y' ≥ -y`).
    public static func ceiling(y: Float) -> CoinStaticCollider {
        .plane(normal: SIMD3(0, -1, 0), offset: -y)
    }
    /// A wall / arbitrary half-space `normal · x ≥ offset` (re-export of `.plane`).
    public static func wall(normal: SIMD3<Float>, offset: Float) -> CoinStaticCollider {
        .plane(normal: normal, offset: offset)
    }

    // ── Shape ──────────────────────────────────────────────────────

    /// The collider shape a launched body takes. `.sphere` is the field's default
    /// (the honest model for a ball — orientation-independent contact); `.disc` and
    /// `.box` let the same field stand in for coins / tiles / crates.
    public enum Shape: Equatable {
        case sphere(radius: Float)
        case disc(radius: Float, halfThickness: Float)
        case box(halfExtents: SIMD3<Float>)

        /// Bounding radius — what the broadphase reach and a scene's render scale key off.
        public var boundingRadius: Float {
            switch self {
            case .sphere(let r):      return r
            case .disc(let r, let h): return (r * r + h * h).squareRoot()
            case .box(let he):        return simd_length(he)
            }
        }
    }

    // ── Throw / launch ─────────────────────────────────────────────

    /// Launch a body of `shape` with an explicit initial velocity. Returns its
    /// `BodyID`, or nil if the field is full. `tumble` is the initial world-frame
    /// angular velocity.
    @discardableResult
    public func launch(from position: SIMD3<Float>, velocity: SIMD3<Float>,
                       shape: Shape, tumble: SIMD3<Float> = .zero) -> BodyID? {
        switch shape {
        case .sphere(let r):
            return solver.spawnSphere(at: position, radius: r, velocity: velocity, tumble: tumble)
        case .disc(let r, let h):
            return solver.spawn(at: position, velocity: velocity, tumble: tumble,
                                radius: r, halfThickness: h)
        case .box(let he):
            return solver.spawnBox(at: position, halfExtents: he, velocity: velocity, tumble: tumble)
        }
    }

    /// Convenience: launch a SPHERE of `radius` (the field's smart default).
    @discardableResult
    public func launch(from position: SIMD3<Float>, velocity: SIMD3<Float>,
                       radius: Float, tumble: SIMD3<Float> = .zero) -> BodyID? {
        launch(from: position, velocity: velocity, shape: .sphere(radius: radius), tumble: tumble)
    }

    /// Launch a body on a gravity-correct ballistic arc that passes through
    /// `target` after `flightTime` seconds — a real lob, not a faked path.
    /// `extraVelocity` adds scatter/spread on top of the exact solve. `shape`
    /// defaults to a sphere of `radius`.
    @discardableResult
    public func launch(from position: SIMD3<Float>, toward target: SIMD3<Float>,
                       flightTime: Float, radius: Float,
                       extraVelocity: SIMD3<Float> = .zero,
                       tumble: SIMD3<Float> = .zero,
                       shape: Shape? = nil) -> BodyID? {
        let v = Self.lobVelocity(from: position, to: target,
                                 flightTime: flightTime, gravity: solver.gravity)
        return launch(from: position, velocity: v + extraVelocity,
                      shape: shape ?? .sphere(radius: radius), tumble: tumble)
    }

    /// The exact launch velocity so a projectile from `from` passes through
    /// `target` after `t` seconds under downward gravity `g`. Pure function —
    /// usable for aiming independently of a field.
    public static func lobVelocity(from: SIMD3<Float>, to target: SIMD3<Float>,
                                   flightTime t: Float, gravity g: Float) -> SIMD3<Float> {
        let disp = target - from
        let tt = max(1e-3, t)
        return SIMD3(disp.x / tt,
                     (disp.y + 0.5 * g * tt * tt) / tt,
                     disp.z / tt)
    }

    public func setVelocity(of id: BodyID, to v: SIMD3<Float>) {
        solver.setVelocity(ofSlot: id, to: v)
    }

    /// Recycle a body's slot (e.g. it left the frame). The handle is invalid after.
    public func remove(_ id: BodyID) { solver.despawn(id) }

    // ── Readback (one-frame-stale; fine for a live tick) ───────────
    public func position(of id: BodyID) -> SIMD3<Float>? { solver.position(of: id) }
    public func velocity(of id: BodyID) -> SIMD3<Float>? { solver.velocity(of: id) }
    public func orientation(of id: BodyID) -> simd_quatf? { solver.orientation(of: id) }

    /// A full TRS model matrix for a body (translation · orientation · uniform
    /// scale), ready to drop straight into a renderer instance. nil if inactive.
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
