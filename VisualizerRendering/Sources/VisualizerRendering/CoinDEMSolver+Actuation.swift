import Foundation
import Metal
import simd

// ── CoinDEMSolver + Actuation ─────────────────────────────────────────────────
//
// Host-side actuation, per-body wake, pooled joints and read-backs for scenes that
// DRIVE a constraint-path CoinDEMSolver rather than only letting it settle (a toy
// crane on joints, self-righting "Weeble" workers that hop, electromagnet grips and
// socket latches). Everything here is a CPU write or read of the solver's SHARED
// buffers — no kernel, pipeline, queue or library is added, and no existing API
// changes behaviour. Every lane written below was checked against the kernel that
// reads it; the citations use the HEAD line numbers of CoinDEM.metal (`M@HEAD:n`)
// and CoinDEMSolver.swift (`S@HEAD:n`) at engine b7613a3.
//
// CALL RULE — every method is @MainActor and touches `.storageModeShared` buffers the
// GPU also writes. Call them ONLY while no command buffer that encodes this solver is
// in flight (after the last one has `.completed`; the scene gates on its rigid
// world's `isIdle`). A write racing an in-flight frame is silently overwritten by the
// kernels, and a read of one is torn.
//
// CAVEATS (read before using the pooled joints):
//   • Never call `clearAll()` on a solver that owns a `CoinJointPool`: it resets the
//     joint high-water mark (S@HEAD:1577), so the kernels stop iterating the pool.
//   • Never `despawn` a body that an ENABLED pooled slot references:
//     `removeJoints(referencing:)` (S@HEAD:1515) would push that slot onto the
//     solver's own free list, where a later add*Joint can claim it under the pool.
//     Disable the slot first; a scene whose population is fixed never despawns.
//   • Motor bounds are PHYSICAL per substep (N·m / N): the kernel clamps the motor
//     impulse accumulated over the velocity iterations (VZ-0155, fixed), so they hold
//     across a `velocityIterations` change without re-issuing the setter.
//   • Hinge LIMITS and the prismatic / weld rotation lock are measured against the
//     relative orientation captured when the joint is made (`CoinJoint.ref`,
//     sign-canonicalised to w ≥ 0 — VZ-0156, fixed): the twist is zero at creation,
//     spans (−π, π] and never jumps by 4π, so bodies at ANY relative orientation may
//     be hinged with stops, slid or welded. (It used to read the ABSOLUTE relative
//     orientation qb·conj(qa), so only bodies spawned with the same orientation worked.)
//   • AXIS-ALIGNMENT SIGN (VZ-0147, fixed). The kernel's hinge / prismatic
//     axis-alignment bias used to drive a misalignment φ AWAY from parallel (×(1 + β)
//     per substep), so this file stored axisB anti-parallel as a workaround. The kernel
//     now restores ×(1 − β) per substep and `axisBSign` is +1: pooled and
//     addHingeJoint / addPrismaticJoint joints use the same, stable, parallel storage.
//   • Dead-stop (VZ-0149, fixed): a body driven by a motor with a non-zero target is
//     never dead-stopped by finalize nor counted as slow for sleep, so a slow slew is
//     not frozen (unless the motor is pushing into a limit it has reached — then it is
//     holding, and may rest and sleep). A body turning slowly about its own COM WITHOUT a motor (a carried
//     bar on a braked wrist) still needs `scaleAwareDeadStop = true` on the solver, or
//     the legacy |v| < sleepLinVel && |ω| < 0.6 test zeroes it.
//   • Joints are BLOCK-solved and warm-started (VZ-0150, fixed): each joint's rows —
//     ball core, axis rows, and its motor (or, at a stop, its limit) — are one LDLᵀ
//     solve with the lever coupling through both bodies' inertia, and every row's
//     impulse carries over to the next substep. A lever-loaded hinge (the §A1 jib, COM
//     0.17 m off the pivot) is exact in one pass and sleeps; it used to keep 1.8 mm/s of
//     residual velocity at 6 iterations and never sleep, because the rows were solved
//     one after another from zero every substep. Joint-to-joint coupling along a chain
//     is still Gauss–Seidel (serial, in slot order; see `jointInnerPasses`).

// ── E1: ballasted egg ─────────────────────────────────────────────────────────

/// A hollow ovoid (the convex hull of a fat sphere and a tip sphere on local +Y —
/// the same outer shape as `spawnEgg`) with a shell of `shellThickness` and a dense
/// ballast poured into the bottom of its cavity: a "Weeble". All lengths in metres,
/// densities in kg/m³.
public struct CoinBallastedEgg: Sendable, Hashable {
    public var fatRadius: Float
    public var tipRadius: Float
    public var centerDistance: Float
    /// Wall thickness (m). 0 ⇒ a SOLID body of `shellDensity`; a ballast fill then
    /// models an insert that REPLACES that much wall material (adds ρ_b − ρ_s).
    /// Must stay below `tipRadius` (the inner cavity is the same two-sphere hull
    /// with both radii shrunk by it).
    public var shellThickness: Float
    public var shellDensity: Float
    /// Height of the ballast fill (m): a spherical cap of the INNER fat sphere
    /// (radius fatRadius − shellThickness) measured up from its bottom. Clamped to
    /// that radius (≤ a hemisphere), so the cap never reaches the flank.
    public var ballastFillHeight: Float
    public var ballastDensity: Float

    public init(fatRadius: Float, tipRadius: Float, centerDistance: Float, shellThickness: Float,
                shellDensity: Float, ballastFillHeight: Float, ballastDensity: Float) {
        self.fatRadius = fatRadius
        self.tipRadius = tipRadius
        self.centerDistance = centerDistance
        self.shellThickness = shellThickness
        self.shellDensity = shellDensity
        self.ballastFillHeight = ballastFillHeight
        self.ballastDensity = ballastDensity
    }
}

/// Composite mass properties of a `CoinBallastedEgg`, in the lane convention the
/// ovoid kernels read (M@HEAD:347–365, 388–400): offsets are signed along body +Y
/// from the COM.
public struct CoinEggMassProperties: Sendable, Equatable {
    /// kg.
    public let mass: Float
    /// d: how far the COM sits BELOW the fat-sphere centre (m). Positive for a
    /// ballasted Weeble — the property that makes it self-right; negative for a
    /// uniform egg, whose COM is pulled toward the tip.
    public let comBelowFatCenter: Float
    /// Fat-sphere centre offset from the COM along body +Y (= d). → `hullRef.x`.
    public let yFat: Float
    /// Tip-sphere centre offset from the COM along body +Y (= c + d). → `shapeExtents.z`.
    public let yTip: Float
    /// max(|yFat| + r1, |yTip| + r2). → `prevPos.w` and `vel.w`.
    public let boundingRadius: Float
    /// Per-unit-mass inverse inertia diagonal (m/I_diam, m/I_axis, m/I_diam), principal
    /// frame (axis = local Y). → `hullRef.yzw`; the kernel uses invMass · this.
    public let invInertiaK: SIMD3<Float>
}

// ── E6: read-back types ───────────────────────────────────────────────────────

/// One body's state as the solver left it after the last completed frame.
public struct CoinBodyState: Sendable {
    public var position: SIMD3<Float>          // COM (world)
    public var orientation: simd_quatf
    public var velocity: SIMD3<Float>          // COM linear velocity (world)
    public var angularVelocity: SIMD3<Float>   // world frame, rad/s
    public var asleep: Bool
    public init(position: SIMD3<Float>, orientation: simd_quatf, velocity: SIMD3<Float>,
                angularVelocity: SIMD3<Float>, asleep: Bool) {
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angularVelocity = angularVelocity
        self.asleep = asleep
    }
}

/// A pooled rigid weld: ONE joint slot enabled as a native WELD (type 4 — ball + all
/// three rotations locked, one 6-row block; see `enableWeld`). It was two unlimited
/// hinges (two slots, redundant rows) before VZ-0150.
public struct CoinWeld: Sendable, Hashable {
    public let slot: Int
    public init(slot: Int) { self.slot = slot }
}

// ── The extension ─────────────────────────────────────────────────────────────

@MainActor
extension CoinDEMSolver {

    /// CD_STATIC, the world / static sentinel in `CoinJoint.meta.z` and `CoinContact.meta.y`
    /// (M@HEAD:2034; `addBallJoint` writes the same 0xFFFF_FFFF, S@HEAD:1389).
    private static let worldBody: UInt32 = 0xFFFF_FFFF

    private var bodies: UnsafeMutablePointer<CoinBody> {
        coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
    }
    private var joints: UnsafeMutablePointer<CoinJoint> {
        jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
    }
    private func isActiveBody(_ slot: Int) -> Bool {
        slot >= 0 && slot < highWater && bodies[slot].posInvMass.w != 0
    }

    // ── E1: ballasted egg ─────────────────────────────────────────────────────

    private static var ballastedEggCache: [CoinBallastedEgg: CoinEggMassProperties] = [:]

    /// Composite mass properties of a ballasted egg (§B1), memoised per spec.
    ///
    /// Disc-stack integration along the axis with the origin at the fat-sphere
    /// centre: for each solid, M = ρ∫πr²dy, F = ρ∫πr²y dy, A = ρ∫(πr⁴/2)dy,
    /// D = ρ∫π(r⁴/4 + r²y²)dy. Solids: the outer two-sphere hull O(r1, r2, c); the
    /// cavity I(r1−t, r2−t, c) — exact, since an inward offset of a sphere-swept
    /// solid keeps the centres and shrinks the radii; the shell S = O − I; and the
    /// ballast B, a spherical cap of the inner fat sphere. Composite: m = M_S + M_B,
    /// y_c = (F_S + F_B)/m, I_axis = A_S + A_B, I_diam = D_S + D_B − m·y_c².
    ///
    /// Each profile is piecewise analytic (sphere cap / tangent frustum / sphere
    /// cap), so the integration splits at every breakpoint and uses 5-point
    /// Gauss–Legendre per piece — exact for these degree-≤4 integrands, not a
    /// sampling approximation.
    public static func ballastedEggProperties(_ s: CoinBallastedEgg) -> CoinEggMassProperties {
        if let hit = ballastedEggCache[s] { return hit }
        let r1 = Double(max(s.fatRadius, 1e-6)), r2 = Double(max(s.tipRadius, 1e-6))
        let c = Double(max(s.centerDistance, 0))
        let t = min(Double(max(s.shellThickness, 0)), 0.999 * min(r1, r2))
        let solid = t <= 0
        let ri1 = r1 - t, ri2 = r2 - t
        let rhoS = Double(s.shellDensity)
        // Ballast: a cap of the inner fat sphere (radius ri1), from y = −ri1 up by h.
        // A solid body (t = 0) has no cavity, so the fill is an insert replacing wall.
        let h = min(Double(max(s.ballastFillHeight, 0)), ri1)
        let rhoB = Double(s.ballastDensity) - (solid ? rhoS : 0)

        // Breakpoints of every piecewise profile involved.
        var cuts: [Double] = [-r1, c + r2]
        cuts += sweptBreakpoints(r1, r2, c)
        if !solid { cuts += [-ri1, c + ri2] + sweptBreakpoints(ri1, ri2, c) }
        if h > 0 { cuts += [-ri1, -ri1 + h] }
        cuts = Array(Set(cuts.filter { $0 >= -r1 && $0 <= c + r2 })).sorted()

        let gx: [Double] = [0, -0.5384693101056831, 0.5384693101056831,
                            -0.9061798459386640, 0.9061798459386640]
        let gw: [Double] = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
                            0.2369268850561891, 0.2369268850561891]
        var M = 0.0, F = 0.0, A = 0.0, D = 0.0
        for i in 0..<(cuts.count - 1) {
            let a = cuts[i], b = cuts[i + 1]
            guard b > a else { continue }
            let half = 0.5 * (b - a), mid = 0.5 * (a + b)
            for k in 0..<5 {
                let y = mid + half * gx[k], w = gw[k] * half
                let ro2 = sweptRadiusSq(y, r1, r2, c)
                let ri2Sq = solid ? 0 : sweptRadiusSq(y, ri1, ri2, c)
                // Shell: density ρ_s over the annulus ro² − ri².
                let s2 = max(ro2 - ri2Sq, 0), s4 = max(ro2 * ro2 - ri2Sq * ri2Sq, 0)
                M += w * rhoS * .pi * s2
                F += w * rhoS * .pi * s2 * y
                A += w * rhoS * .pi * s4 * 0.5
                D += w * rhoS * .pi * (s4 * 0.25 + s2 * y * y)
                // Ballast cap: full discs of the inner fat sphere below −ri1 + h.
                if h > 0, y > -ri1, y < -ri1 + h {
                    let b2 = max(ri1 * ri1 - y * y, 0)
                    M += w * rhoB * .pi * b2
                    F += w * rhoB * .pi * b2 * y
                    A += w * rhoB * .pi * b2 * b2 * 0.5
                    D += w * rhoB * .pi * (b2 * b2 * 0.25 + b2 * y * y)
                }
            }
        }
        let m = max(M, 1e-15)
        let yc = F / m
        let iAxis = max(A, 1e-18)
        let iDiam = max(D - m * yc * yc, 1e-18)
        let yFat = -yc, yTip = c - yc
        let props = CoinEggMassProperties(
            mass: Float(m),
            comBelowFatCenter: Float(-yc),
            yFat: Float(yFat),
            yTip: Float(yTip),
            boundingRadius: Float(max(abs(yFat) + r1, abs(yTip) + r2)),
            invInertiaK: SIMD3(Float(m / iDiam), Float(m / iAxis), Float(m / iDiam)))
        ballastedEggCache[s] = props
        return props
    }

    /// Cross-section radius² at height y of the convex hull of two spheres on the
    /// axis — radius ra centred at 0, rb centred at c (y measured from ra's centre).
    /// Exact piecewise form: ra's cap below its tangency circle, the tangent
    /// frustum, rb's cap above its tangency circle. When one sphere contains the
    /// other (|ra − rb| ≥ c) the hull is just the union.
    nonisolated static func sweptRadiusSq(_ y: Double, _ ra: Double, _ rb: Double, _ c: Double) -> Double {
        if c <= 1e-12 || abs(ra - rb) >= c {
            return max(ra * ra - y * y, rb * rb - (y - c) * (y - c), 0)
        }
        let s = (ra - rb) / c, k = (1 - s * s).squareRoot()
        let ya = ra * s, yb = c + rb * s
        if y <= ya { return max(ra * ra - y * y, 0) }
        if y >= yb { return max(rb * rb - (y - c) * (y - c), 0) }
        let r = ra * k + (y - ya) * (rb * k - ra * k) / (yb - ya)
        return r * r
    }

    /// The heights where `sweptRadiusSq` changes analytic form.
    nonisolated static func sweptBreakpoints(_ ra: Double, _ rb: Double, _ c: Double) -> [Double] {
        if c <= 1e-12 { return [] }
        if abs(ra - rb) >= c { return [(ra * ra - rb * rb + c * c) / (2 * c)] }   // union crossover
        let s = (ra - rb) / c
        return [ra * s, c + rb * s]
    }

    /// Spawn a ballasted egg with its FAT-SPHERE CENTRE at `fatCenter`.
    ///
    /// `spawnEgg(at: COM, …, mass: m)` (S@HEAD:1171) activates the slot with the
    /// uniform-solid lanes; this then overwrites exactly the lanes S@HEAD:1202–1209
    /// wrote for that solid, each checked against its reader:
    ///   • shapeExtents.z = yTip — cdEggSegment's tip centre (M@HEAD:364).
    ///   • hullRef.x = yFat — cdEggSegment's fat centre (M@HEAD:363), a SIGNED offset:
    ///     positive (fat centre above the COM) is fine, every egg path goes through
    ///     cdEggSegment / cdSweptSegment (M@HEAD:2364, 2703) or reads hullRef.x as a
    ///     signed y (sphere↔egg, M@HEAD:2516).
    ///   • hullRef.yzw = invInertiaK — cdBodyInvInertia returns invMass · hullRef.yzw
    ///     for an egg (M@HEAD:395).
    ///   • prevPos.w / vel.w = boundingRadius — cdRadiusOf / cdHalfThickOf
    ///     (M@HEAD:426–427): the pair bounding reject (M@HEAD:2338) and the finalize
    ///     backstop `floorY + ½·min(...)` (M@HEAD:3695). That backstop props an egg's
    ///     COM at half its bounding radius — set `floorY` below the scene.
    /// The COM is placed at fatCenter − q·(0, yFat, 0).
    @discardableResult
    public func spawnBallastedEgg(_ s: CoinBallastedEgg, fatCenter: SIMD3<Float>,
                                  orient: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                                  friction: Float? = nil, restitution: Float? = nil,
                                  type: UInt32 = 0) -> Int? {
        let p = Self.ballastedEggProperties(s)
        guard p.mass.isFinite, p.mass > 0,
              p.invInertiaK.x.isFinite, p.invInertiaK.y.isFinite else { return nil }
        let q = orient.normalized
        let com = fatCenter - simd_act(q, SIMD3<Float>(0, p.yFat, 0))
        guard let slot = spawnEgg(at: com, fatRadius: s.fatRadius, tipRadius: s.tipRadius,
                                  centerDistance: s.centerDistance,
                                  orient: SIMD4(q.imag, q.real),
                                  mass: p.mass, friction: friction, restitution: restitution,
                                  type: type) else { return nil }
        let b = bodies
        b[slot].shapeExtents.z = p.yTip
        b[slot].hullRef = SIMD4(p.yFat, p.invInertiaK.x, p.invInertiaK.y, p.invInertiaK.z)
        b[slot].prevPos.w = p.boundingRadius
        b[slot].vel.w = p.boundingRadius
        return slot
    }

    // ── E2: actuation ─────────────────────────────────────────────────────────

    /// Set an active body's angular velocity (world frame, rad/s): writes
    /// `angVel.xyz`, keeps `angVel.w` (the support flag). Wakes the body — a write to
    /// an asleep body is discarded by coinIntegrateVelocityCS (M@HEAD:3332). Bodies
    /// joined to it by enabled joints are NOT woken: wake the whole assembly first
    /// (a sleeping joint partner acts as infinite mass for one frame).
    public func setAngularVelocity(ofSlot slot: Int, to w: SIMD3<Float>) {
        guard isActiveBody(slot) else { return }
        wake(CollectionOfOne(slot))
        let b = bodies
        b[slot].angVel.x = w.x; b[slot].angVel.y = w.y; b[slot].angVel.z = w.z
    }

    /// Give `slots` the velocity field of ONE rigid body moving with linear velocity
    /// `v` at `pivot` and angular velocity `w`: v_i = v + w × (x_i − pivot), ω_i = w,
    /// where x_i is each body's COM. With `pivot` = the assembly COM this is an
    /// impulse through that COM, and it is consistent with every weld/joint between
    /// the listed bodies, so no internal stress is injected. Writes `vel.xyz` and
    /// `angVel.xyz` only (the `.w` lanes carry shape/support data). Wakes every listed
    /// body (see `setAngularVelocity`) — list the WHOLE jointed assembly.
    public func setRigidVelocity(slots: [Int], linear v: SIMD3<Float>, angular w: SIMD3<Float>,
                                 about pivot: SIMD3<Float>) {
        let live = slots.filter { isActiveBody($0) }
        wake(live)
        let b = bodies
        for s in live {
            let x = SIMD3<Float>(b[s].posInvMass.x, b[s].posInvMass.y, b[s].posInvMass.z)
            let vi = v + simd_cross(w, x - pivot)
            b[s].vel.x = vi.x; b[s].vel.y = vi.y; b[s].vel.z = vi.z
            b[s].angVel.x = w.x; b[s].angVel.y = w.y; b[s].angVel.z = w.z
        }
    }

    /// Mass-weighted centre of mass Σ m_i x_i / Σ m_i of the active `slots`
    /// (m_i = 1/posInvMass.w). `.zero` if none is active.
    public func centerOfMass(of slots: [Int]) -> SIMD3<Float> {
        let b = bodies
        var sum = SIMD3<Double>.zero, mass = 0.0
        for s in slots where isActiveBody(s) {
            let m = 1.0 / Double(b[s].posInvMass.w)
            sum += m * SIMD3<Double>(Double(b[s].posInvMass.x), Double(b[s].posInvMass.y),
                                     Double(b[s].posInvMass.z))
            mass += m
        }
        guard mass > 0 else { return .zero }
        let c = sum / mass
        return SIMD3(Float(c.x), Float(c.y), Float(c.z))
    }

    // ── E3: per-body wake ─────────────────────────────────────────────────────

    /// Wake exactly these bodies: clear `asleep[s]` AND zero `sleepTimer[s]`.
    /// Clearing only the flag is not a wake — coinSleepTick (M@HEAD:4138–4143) keeps
    /// counting a long-asleep body's timer up while the rest of the world runs, and
    /// coinSleepMark (M@HEAD:4182) re-freezes it at the end of the SAME frame. Nothing
    /// else is woken now; at the frame's end the island pass (contacts M@HEAD:4097–4113,
    /// non-world joints M@HEAD:4057–4072)
    /// takes each island's minimum timer, so anything in contact or jointed (non-world)
    /// with a woken body wakes too — one frame late, during which a still-asleep
    /// partner acts as immovable. Wake whole assemblies before actuating them.
    public func wake<S: Sequence>(_ slots: S) where S.Element == Int {
        let a = asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        let t = sleepTimerBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        for s in slots where s >= 0 && s < maxCoins {
            a[s] = 0
            t[s] = 0
        }
    }

    /// True if the body is frozen by island sleeping (as of the last completed frame).
    public func isAsleep(_ slot: Int) -> Bool {
        guard slot >= 0, slot < maxCoins else { return false }
        return asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)[slot] != 0
    }

    // ── E4: joint drive setters ───────────────────────────────────────────────
    //
    // Each validates the slot (below the high-water mark, enabled bit set, right
    // type) and is a no-op otherwise; each wakes the joint's two bodies.

    private func liveJoint(_ joint: Int, types: Set<UInt32>) -> CoinJoint? {
        guard joint >= 0, joint < jointCount else { return nil }
        let j = joints[joint]
        guard (j.meta.w & 1) != 0, types.contains(j.meta.x) else { return nil }
        return j
    }

    private func wakeBodies(of j: CoinJoint) {
        if j.meta.z == Self.worldBody { wake(CollectionOfOne(Int(j.meta.y))) }
        else { wake([Int(j.meta.y), Int(j.meta.z)]) }
    }

    /// Hinge motor in PHYSICAL units: drive the hinge's angular velocity toward
    /// `targetVelocity` (rad/s) with at most `maxTorque` (N·m); maxTorque ≤ 0 turns
    /// the motor off. A target of 0 with a torque is a brake/servo hold.
    ///
    /// Lanes: anchorA.w = target, anchorB.w = bound (N·m, stored as given). The kernel
    /// clamps the motor impulse ACCUMULATED over the substep's velocity iterations to
    /// ±anchorB.w·dt (VZ-0155), so the per-substep ceiling is the physical maxTorque·dt
    /// at any `velocityIterations` — no ÷ iterations here, and changing the iteration
    /// count later does not change the motor's strength. The kernel drives
    /// dot(ωB − ωA, aW) — B relative to A about A's axis — so for a WORLD hinge the
    /// target is negated, exactly as addHingeJoint does: `targetVelocity` is then A's
    /// own spin about the axis. For a two-body hinge it is B's spin relative to A.
    /// A non-zero target keeps the joint's bodies out of the finalize dead-stop and the
    /// sleep test (VZ-0149) — except while it pushes into a limit it has reached; set the
    /// target to 0 (a brake) when the move is done.
    public func setHingeMotor(_ joint: Int, targetVelocity: Float, maxTorque: Float) {
        guard let j = liveJoint(joint, types: [1]) else { return }
        let world = j.meta.z == Self.worldBody
        let p = joints
        p[joint].anchorA.w = targetVelocity * (world ? -1 : 1)
        p[joint].anchorB.w = max(maxTorque, 0)
        wakeBodies(of: j)
    }

    /// Prismatic motor in PHYSICAL units: drive the slide velocity toward
    /// `targetVelocity` (m/s along the joint axis) with at most `maxForce` (N);
    /// maxForce ≤ 0 turns it off. Lanes: anchorA.w = target, anchorB.w = maxForce
    /// (accumulated per-substep clamp, as the hinge). The kernel drives dot(vA − vB, aW),
    /// so NO world-case negation: the target is A's velocity along the axis relative to B.
    public func setPrismaticMotor(_ joint: Int, targetVelocity: Float, maxForce: Float) {
        guard let j = liveJoint(joint, types: [3]) else { return }
        let p = joints
        p[joint].anchorA.w = targetVelocity
        p[joint].anchorB.w = max(maxForce, 0)
        wakeBodies(of: j)
    }

    /// Distance-joint rest length (m): anchorA.w, read as `C = len − anchorA.w`
    /// (cdJointPrepareOne). The rod recovers a rest-length change through the
    /// split-impulse BIAS only — its real row drives the anchors' velocity along the rod
    /// to 0 — so a ramping winch lags its rest length by ≈ rate·dt/β and the hoisted
    /// body reads as "slow" to the sleep test. This setter wakes both bodies,
    /// which is what keeps a hoist that is re-targeted every tick from falling asleep
    /// mid-move.
    public func setDistanceRestLength(_ joint: Int, _ rest: Float) {
        guard let j = liveJoint(joint, types: [2]) else { return }
        joints[joint].anchorA.w = max(rest, 0)
        wakeBodies(of: j)
    }

    /// Hinge (radians) or prismatic (metres) limits: axisA.w = lo, axisB.w = hi; nil (or
    /// lo == hi) turns them off — the kernel enables them only for lo < hi. A PRISMATIC
    /// limit bounds the slide dot(pA − pB, aW), zero at creation. A HINGE limit bounds
    /// `hingeTwist` — the twist since the joint was made (VZ-0156), which for a world
    /// hinge is the NEGATIVE of A's own turn. A stop is solved as a one-sided row of the
    /// joint's block while it is within reach (speculative: the pair closes the gap and
    /// no further), with the motor then a scalar row ahead of the block.
    public func setJointLimits(_ joint: Int, _ limits: ClosedRange<Float>?) {
        guard let j = liveJoint(joint, types: [1, 3]) else { return }
        let p = joints
        p[joint].axisA.w = limits?.lowerBound ?? 0
        p[joint].axisB.w = limits?.upperBound ?? 0
        wakeBodies(of: j)
    }

    // ── E5: pooled joints ─────────────────────────────────────────────────────
    //
    // add*Joint / removeJoint call wakeAll() (S@HEAD:1359, 1510), which also zeroes
    // every sleep timer — one grip would wake every latched bar in the scene. A pooled
    // slot is instead RESERVED once (see CoinJointPool) and toggled here by writing the
    // whole CoinJoint, or just meta.w = 0, directly. Only the joint's own two bodies
    // are woken. `slot` must be a reserved slot below the high-water mark. The solver
    // re-scans the table at every encode and hands the kernels only the ENABLED slots
    // (the joint solve, generate's collideConnected check and the island union), and a
    // slot enabled anew — or re-enabled between other bodies / at another anchor — starts
    // with zero warm-start impulse (VZ-0150).

    /// A-local coordinates of a world point — the same maths as the private
    /// CoinDEMSolver.toLocal (S@HEAD:1363), re-derived from `coinBuffer`.
    private func localPoint(_ slot: Int, _ world: SIMD3<Float>) -> SIMD3<Float> {
        let b = bodies[slot]
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        return simd_act(q.inverse, world - SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z))
    }

    /// A-local direction of a world direction (S@HEAD:1369).
    private func localDir(_ slot: Int, _ worldDir: SIMD3<Float>) -> SIMD3<Float> {
        let b = bodies[slot]
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        return simd_act(q.inverse, worldDir)
    }

    private func poolSlotUsable(_ slot: Int, _ a: Int, _ b: Int?) -> Bool {
        guard slot >= 0, slot < jointCount, isActiveBody(a) else { return false }
        if let b { return isActiveBody(b) && b != a }
        return true
    }

    private static func metaW(_ collideConnected: Bool) -> UInt32 { 1 | (collideConnected ? 2 : 0) }

    /// Enable a pooled slot as a BALL joint: `worldAnchor` on A coincides with the
    /// same point on B (or that fixed world point when B is nil), captured from the
    /// current poses. Writes the whole CoinJoint exactly as addBallJoint does
    /// (S@HEAD:1384–1393) minus the wakeAll; wakes A and B.
    public func enableBall(slot: Int, bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                           collideConnected: Bool = false) {
        guard poolSlotUsable(slot, bodyA, bodyB) else { return }
        joints[slot] = CoinJoint(
            meta: SIMD4(0, UInt32(bodyA), bodyB.map { UInt32($0) } ?? Self.worldBody,
                        Self.metaW(collideConnected)),
            anchorA: SIMD4(localPoint(bodyA, worldAnchor), 0),
            anchorB: SIMD4(bodyB.map { localPoint($0, worldAnchor) } ?? worldAnchor, 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0),
            ref: jointReference(bodyA, bodyB))
        wake(bodyB.map { [bodyA, $0] } ?? [bodyA])
    }

    /// The sign `enableHinge` / `enablePrismatic` give axisB relative to axisA: +1, the
    /// same PARALLEL convention as addHingeJoint / addPrismaticJoint. It was −1 while the
    /// kernel's axis-alignment bias was sign-inverted (VZ-0147: jAb = (−β·errT/dt −
    /// bwrel)/k drove parallel axes apart, so anti-parallel was the stable storage); the
    /// kernel now uses +β·errT/dt and parallel axes restore ×(1 − β) per substep.
    /// Only the HINGE's two axis-alignment rows read axisB.xyz (their error is aW × bW);
    /// its motor and limits use aW = qa·axisA and the twist from `ref`, and since VZ-0150
    /// the prismatic does not read axisB at all (its rotation lock is the whole relative
    /// orientation against `ref`). Flipping this back to −1 would make every pooled hinge
    /// unstable — `testAxisAlignmentConventionPooledHoldsAddFlips` / CoinDEMCorrectnessTests
    /// pin it.
    static let axisBSign: Float = 1

    /// Enable a pooled slot as a free HINGE (no limits, no motor): ball at
    /// `worldAnchor` + rotation only about `worldAxis`. Lanes: meta = (1, A, B|world,
    /// 1 | cc<<1); anchorA = (A-local anchor, motor target 0); anchorB = (B-local anchor
    /// or the world point, maxTorque 0 ⇒ motor off); axisA = (A-local axis, lo 0);
    /// axisB = (`axisBSign` × the B-local axis or the world axis, hi 0) — lo == hi ⇒
    /// limit off; ref = the relative orientation now, which the twist (and so any
    /// limits added later) is measured from. Motor and limits can be added afterwards
    /// with `setHingeMotor` / `setJointLimits` (the rig's slew and wrist). Wakes A and B.
    public func enableHinge(slot: Int, bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                            worldAxis: SIMD3<Float>, collideConnected: Bool = false) {
        writePooledAxisJoint(type: 1, slot: slot, bodyA: bodyA, bodyB: bodyB, worldAnchor: worldAnchor,
                             worldAxis: worldAxis, collideConnected: collideConnected)
    }

    /// Enable a pooled slot as a free PRISMATIC joint (no limits, no motor): A may only
    /// translate relative to B (or the world) along `worldAxis`, rotation fully locked.
    /// Same lane layout as `enableHinge` with meta.x = 3: the slide s = dot(pA − pB, aW)
    /// is 0 now, limits (`setJointLimits`, metres) are measured from here, and the motor
    /// (`setPrismaticMotor`) drives A's velocity along aW relative to B. The rotation
    /// lock holds the relative orientation captured now (`ref`, VZ-0156), whatever it
    /// is. Wakes A and B.
    ///
    /// Not in the §F.1 API: added so the crane's trolley and plunger (prismatic) can be
    /// pooled slots toggled without wakeAll, like the hinges.
    public func enablePrismatic(slot: Int, bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                                worldAxis: SIMD3<Float>, collideConnected: Bool = false) {
        writePooledAxisJoint(type: 3, slot: slot, bodyA: bodyA, bodyB: bodyB, worldAnchor: worldAnchor,
                             worldAxis: worldAxis, collideConnected: collideConnected)
    }

    private func writePooledAxisJoint(type: UInt32, slot: Int, bodyA: Int, bodyB: Int?,
                                      worldAnchor: SIMD3<Float>, worldAxis: SIMD3<Float>,
                                      collideConnected: Bool) {
        guard poolSlotUsable(slot, bodyA, bodyB) else { return }
        let len = simd_length(worldAxis)
        let axis = len > 1e-9 ? worldAxis / len : SIMD3<Float>(0, 1, 0)
        joints[slot] = CoinJoint(
            meta: SIMD4(type, UInt32(bodyA), bodyB.map { UInt32($0) } ?? Self.worldBody,
                        Self.metaW(collideConnected)),
            anchorA: SIMD4(localPoint(bodyA, worldAnchor), 0),
            anchorB: SIMD4(bodyB.map { localPoint($0, worldAnchor) } ?? worldAnchor, 0),
            axisA: SIMD4(localDir(bodyA, axis), 0),
            axisB: SIMD4(Self.axisBSign * (bodyB.map { localDir($0, axis) } ?? axis), 0),
            ref: jointReference(bodyA, bodyB))
        wake(bodyB.map { [bodyA, $0] } ?? [bodyA])
    }

    /// Disable a pooled slot in place: meta.w = 0 (it leaves the next encode's active
    /// list: the joint solve, generate's collideConnected check and the island union no
    /// longer see it). NOT removeJoint (S@HEAD:1504), which would put the slot
    /// on the solver's own free list and wakeAll. Wakes the two bodies it joined, so
    /// a released body responds at once instead of hanging frozen.
    public func disableJoint(slot: Int) {
        guard slot >= 0, slot < jointCount else { return }
        let j = joints[slot]
        guard j.meta.w != 0 else { return }
        joints[slot].meta.w = 0
        if Int(j.meta.y) < maxCoins { wakeBodies(of: j) }
    }

    /// Enable a pooled RIGID WELD: the slot becomes a native WELD joint (type 4) holding
    /// A and B (or A and the world) at their CURRENT relative pose — the ball core at
    /// `worldAnchor` plus all three rotations, one 6-row block solve (VZ-0150). The
    /// rotation lock is the vector part of the relative orientation against the one
    /// captured now (`ref`), sign-canonicalised to w ≥ 0: it holds ANY relative
    /// orientation, through any number of turns, and cannot wrap. Lanes: meta = (4, A,
    /// B|world, 1 | cc<<1); anchorA/anchorB as `enableBall`; axes unused; ref. Wakes A
    /// and B. (It was two perpendicular hinges on two slots before VZ-0150.)
    public func enableWeld(_ w: CoinWeld, bodyA: Int, bodyB: Int?, worldAnchor: SIMD3<Float>,
                           collideConnected: Bool = false) {
        guard poolSlotUsable(w.slot, bodyA, bodyB) else { return }
        joints[w.slot] = CoinJoint(
            meta: SIMD4(4, UInt32(bodyA), bodyB.map { UInt32($0) } ?? Self.worldBody,
                        Self.metaW(collideConnected)),
            anchorA: SIMD4(localPoint(bodyA, worldAnchor), 0),
            anchorB: SIMD4(bodyB.map { localPoint($0, worldAnchor) } ?? worldAnchor, 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0),
            ref: jointReference(bodyA, bodyB))
        wake(bodyB.map { [bodyA, $0] } ?? [bodyA])
    }

    public func disableWeld(_ w: CoinWeld) {
        disableJoint(slot: w.slot)
    }

    // ── E6: read-backs (idle only; the kernels' own formulas) ─────────────────

    /// One `CoinBodyState` per requested slot, in order (an out-of-range slot yields
    /// a zero state; a despawned one its parked `.inactive` pose). Replaces `out`'s
    /// contents, keeping its capacity.
    public func readStates(_ slots: [Int], into out: inout [CoinBodyState]) {
        out.removeAll(keepingCapacity: true)
        let b = bodies
        let a = asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: maxCoins)
        for s in slots {
            guard s >= 0, s < maxCoins else {
                out.append(CoinBodyState(position: .zero, orientation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                                         velocity: .zero, angularVelocity: .zero, asleep: false))
                continue
            }
            let c = b[s]
            out.append(CoinBodyState(
                position: SIMD3(c.posInvMass.x, c.posInvMass.y, c.posInvMass.z),
                orientation: simd_quatf(ix: c.orient.x, iy: c.orient.y, iz: c.orient.z, r: c.orient.w),
                velocity: SIMD3(c.vel.x, c.vel.y, c.vel.z),
                angularVelocity: SIMD3(c.angVel.x, c.angVel.y, c.angVel.z),
                asleep: a[s] != 0))
        }
    }

    /// World anchors (pA on A, pB on B or the world point) of an enabled joint —
    /// M@HEAD:3789–3790: pA = xA + qA·anchorA.xyz, pB = world ? anchorB.xyz : xB + qB·anchorB.xyz.
    public func jointWorldAnchors(_ joint: Int) -> (a: SIMD3<Float>, b: SIMD3<Float>)? {
        guard joint >= 0, joint < jointCount else { return nil }
        let j = joints[joint]
        guard (j.meta.w & 1) != 0 else { return nil }
        return anchors(j)
    }

    private func anchors(_ j: CoinJoint) -> (a: SIMD3<Float>, b: SIMD3<Float>)? {
        let ia = Int(j.meta.y)
        guard isActiveBody(ia) else { return nil }
        let a = bodies[ia]
        let pA = SIMD3(a.posInvMass.x, a.posInvMass.y, a.posInvMass.z)
            + quatRotate(a.orient, SIMD3(j.anchorA.x, j.anchorA.y, j.anchorA.z))
        if j.meta.z == Self.worldBody { return (pA, SIMD3(j.anchorB.x, j.anchorB.y, j.anchorB.z)) }
        let ib = Int(j.meta.z)
        guard isActiveBody(ib) else { return nil }
        let b = bodies[ib]
        let pB = SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z)
            + quatRotate(b.orient, SIMD3(j.anchorB.x, j.anchorB.y, j.anchorB.z))
        return (pA, pB)
    }

    /// The kernel's twist for a hinge (or prismatic) joint (cdJointPrepareOne): the
    /// relative orientation against the one at creation, qe = (conj(qa)·qb)·conj(ref)
    /// (world: qb = identity), sign-canonicalised to w ≥ 0, and twist = 2·atan2(qe.xyz ·
    /// axisA, qe.w) about the A-local axis. This is the quantity `setJointLimits` bounds:
    /// 0 at creation, in (−π, π], no 4π jump (VZ-0156), and for a world hinge the
    /// NEGATIVE of A's own turn since creation (B relative to A).
    public func hingeTwist(_ joint: Int) -> Float? {
        guard let j = liveJoint(joint, types: [1, 3]) else { return nil }
        let ia = Int(j.meta.y)
        guard isActiveBody(ia) else { return nil }
        let qa = bodies[ia].orient
        var qb = SIMD4<Float>(0, 0, 0, 1)
        if j.meta.z != Self.worldBody {
            let ib = Int(j.meta.z)
            guard isActiveBody(ib) else { return nil }
            qb = bodies[ib].orient
        }
        var qe = quatMul(quatMul(SIMD4(-qa.x, -qa.y, -qa.z, qa.w), qb), SIMD4(-j.ref.x, -j.ref.y, -j.ref.z, j.ref.w))
        if qe.w < 0 { qe = -qe }
        let axis = simd_normalize(SIMD3(j.axisA.x, j.axisA.y, j.axisA.z))
        return 2 * atan2(simd_dot(SIMD3(qe.x, qe.y, qe.z), axis), qe.w)
    }

    /// Prismatic slide s = dot(pA − pB, aW), aW = normalize(qa·axisA) — M@HEAD:3832,
    /// 3925: zero at creation, the quantity its limits bound.
    public func prismaticSlide(_ joint: Int) -> Float? {
        guard let j = liveJoint(joint, types: [3]), let p = anchors(j) else { return nil }
        let qa = bodies[Int(j.meta.y)].orient
        let aW = simd_normalize(quatRotate(qa, SIMD3(j.axisA.x, j.axisA.y, j.axisA.z)))
        return simd_dot(p.a - p.b, aW)
    }

    /// Distance-joint length |pA − pB| — M@HEAD:3797–3798 (compare with its rest
    /// length anchorA.w; they differ by the bias lag while a winch ramps).
    public func distanceJointLength(_ joint: Int) -> Float? {
        guard let j = liveJoint(joint, types: [2]), let p = anchors(j) else { return nil }
        return simd_length(p.a - p.b)
    }

    /// Every contact of the LAST SUBSTEP of the last encoded frame that involves a
    /// body in `slots` (contactBuffer / contactCountBuffer, S@HEAD:359–360; the count
    /// is clamped to capacity). meta.y == 0xFFFF_FFFF marks a static contact, whose
    /// meta.z is the collider index. A frame skipped because the whole world slept
    /// leaves the previous frame's contacts in place. An ASLEEP body's contacts with
    /// statics and with other asleep bodies are not generated (VZ-0152) — only its
    /// contacts with awake bodies appear — so read resting contacts while it is awake.
    public func contacts(touching slots: Set<Int>) -> [CoinContact] {
        let n = contactCount
        guard n > 0, !slots.isEmpty else { return [] }
        let p = contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: maxContacts)
        var out: [CoinContact] = []
        for i in 0..<n {
            let c = p[i]
            if slots.contains(Int(c.meta.x))
                || (c.meta.y != Self.worldBody && slots.contains(Int(c.meta.y))) {
                out.append(c)
            }
        }
        return out
    }

    // ── Quaternion helpers matching the kernels' (x, y, z, w) maths ───────────

    /// cdQuatRotate (M@HEAD:276–280): v + w·t + q.xyz × t, t = 2·(q.xyz × v).
    private func quatRotate(_ q: SIMD4<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
        let u = SIMD3(q.x, q.y, q.z)
        let t = 2 * simd_cross(u, v)
        return v + q.w * t + simd_cross(u, t)
    }

    /// cdQuatMul (M@HEAD:267–272): Hamilton product a ⊗ b.
    private func quatMul(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> SIMD4<Float> {
        let av = SIMD3(a.x, a.y, a.z), bv = SIMD3(b.x, b.y, b.z)
        let v = a.w * bv + b.w * av + simd_cross(av, bv)
        return SIMD4(v, a.w * b.w - simd_dot(av, bv))
    }
}

// ── CoinJointPool ─────────────────────────────────────────────────────────────

/// A host-side free list over joint slots reserved ONCE at setup. Reservation calls
/// `addBallJoint` on a placeholder body (each add wakes everything once — at setup,
/// before anything sleeps) and then disables every slot in place (meta.w = 0) rather
/// than `removeJoint`, so the slots stay below the solver's high-water mark and OFF
/// its own free list: no later add*Joint can claim them, and toggling them through
/// `enableBall` / `enableHinge` / `enableWeld` / `disableJoint` never calls wakeAll.
/// Cost: a reserved slot is one CPU check per encode while disabled; the GPU kernels
/// read only enabled slots (VZ-0150 item 4c), so a generous reserve is free.
@MainActor
public final class CoinJointPool {
    public let solver: CoinDEMSolver
    /// Every slot this pool owns, in reservation order.
    public let reservedSlots: [Int]
    private var free: [Int]
    private var inUse: Set<Int> = []

    public var freeCount: Int { free.count }

    public init?(solver: CoinDEMSolver, reserve count: Int, placeholderBody: Int) {
        guard count > 0, let anchor = solver.position(of: placeholderBody) else { return nil }
        var got: [Int] = []
        got.reserveCapacity(count)
        for _ in 0..<count {
            guard let s = solver.addBallJoint(bodyA: placeholderBody, bodyB: nil, worldAnchor: anchor) else {
                for s in got { solver.removeJoint(s) }   // table full: hand back what we took
                return nil
            }
            got.append(s)
        }
        let j = solver.jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: CoinDEMSolver.maxJoints)
        for s in got { j[s].meta.w = 0 }                  // disabled in place, still reserved
        self.solver = solver
        self.reservedSlots = got
        self.free = got.reversed()                        // takeSlot hands out the lowest first
    }

    public func takeSlot() -> Int? {
        guard let s = free.popLast() else { return nil }
        inUse.insert(s)
        return s
    }

    public func takeWeld() -> CoinWeld? {
        takeSlot().map { CoinWeld(slot: $0) }
    }

    /// Return a slot: disables it (waking its two bodies) if it is still enabled.
    public func give(_ slot: Int) {
        guard inUse.remove(slot) != nil else { return }
        solver.disableJoint(slot: slot)
        free.append(slot)
    }

    public func give(_ w: CoinWeld) {
        give(w.slot)
    }
}
