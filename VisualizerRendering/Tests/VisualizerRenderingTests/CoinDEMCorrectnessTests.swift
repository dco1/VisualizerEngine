import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Correctness gate for the CoinDEM engine fixes of the Digital Clock work plan §6
/// items 1a–1d (VZ-0147, VZ-0148, VZ-0149, VZ-0155). Each test is the regression proof
/// of one root cause; measured numbers are PRINTed with an `A1_` prefix so a run's log
/// is the record. Same runtime-compiled-library seam and toy-scale §G configuration
/// (1/180 s, 6 velocity iterations, sleep on, sleepLinVel 2e-4) as
/// CoinDEMActuationTests.
@MainActor
final class CoinDEMCorrectnessTests: XCTestCase {

    // ── Harness ──────────────────────────────────────────────────────────────

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard FileManager.default.fileExists(atPath: shader.path) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader), queue)
        cached = c
        return c
    }

    static let g: Float = 9.81

    /// The Digital Clock worker (§B1) — same spec as CoinDEMActuationTests.worker.
    static let worker = CoinBallastedEgg(fatRadius: 0.0115, tipRadius: 0.0075, centerDistance: 0.017,
                                         shellThickness: 0.0009, shellDensity: 1050,
                                         ballastFillHeight: 0.0065, ballastDensity: 11300)

    /// §G knobs (CoinDEMActuationTests.applyClockConfig), with the opt-in flags explicit.
    private func makeSolver(maxCoins: Int = 16, maxRadius: Float = 0.03,
                            boundsMin: SIMD3<Float> = SIMD3(-0.4, -0.1, -0.4),
                            boundsMax: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4),
                            dt: Float = 1.0 / 180, iterations: Int = 6,
                            rollingResistance: Float = 0.004,
                            accumulatedRolling: Bool = false, scaleAwareDeadStop: Bool = false,
                            sleep: Bool = true, gravity: Float = CoinDEMCorrectnessTests.g,
                            colliders: [CoinStaticCollider]? = nil) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: maxRadius, halfThickness: maxRadius,
                                    boundsMin: boundsMin, boundsMax: boundsMax)
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = gravity
        s.fixedDt = dt
        s.maxSubsteps = 10
        s.velocityIterations = iterations
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = false
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.rollingResistance = rollingResistance
        s.accumulatedRollingResistance = accumulatedRolling
        s.scaleAwareDeadStop = scaleAwareDeadStop
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2e-4
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = sleep
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.setColliders(colliders ?? [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)])
        return (s, queue)
    }

    private func step(_ s: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
                      wallDt: Float = 1.0 / 60, perFrame: ((Int) -> Void)? = nil) {
        for f in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            perFrame?(f)
        }
    }

    static func deg(_ d: Float) -> Float { d * .pi / 180 }
    static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }
    /// Signed tilt of body +Y about +X (the plane the Weeble tests rock in).
    static func signedTiltDeg(_ q: simd_quatf) -> Float {
        let u = simd_act(q, SIMD3<Float>(0, 1, 0))
        return atan2(u.z, u.y) * 180 / .pi
    }

    /// Rotate a live body in place (orientation + substep-start orientation).
    private func seedRotation(_ s: CoinDEMSolver, _ b: Int, _ r: simd_quatf) {
        let p = s.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: s.maxCoins)
        let q1 = r * s.orientation(of: b)!
        p[b].orient = Self.v4(q1)
        p[b].prevOrient = p[b].orient
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1a — VZ-0147: hinge / prismatic axis-alignment bias sign
    // ══════════════════════════════════════════════════════════════════════════

    /// A hinge (and a prismatic) through the body's OWN COM — so the ball core / ⟂
    /// translation rows have zero lever arm and only the axis-alignment BIAS rows act —
    /// seeded 1° off its axis at rest, gravity off, no contacts, one substep per frame.
    /// Split-impulse bias rotates the misalignment φ by −β·sin φ per substep, so φ must
    /// decay by exactly (1 − β) = 0.8 per substep — through addHingeJoint /
    /// addPrismaticJoint AND through the pooled enableHinge / enablePrismatic.
    /// (Before the fix: ×1.2 per substep through add*, i.e. parallel axes were the
    /// unstable equilibrium and the pooled path stored axisB anti-parallel to cope.)
    func testAxisBiasRestoresThroughCOMForAddAndPooledJoints() throws {
        let dt: Float = 1.0 / 180
        let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05, dt: dt, sleep: false, gravity: 0, colliders: [])
        let axis = SIMD3<Float>(0, 1, 0)
        let he = SIMD3<Float>(0.02, 0.01, 0.015)
        var bodies: [Int] = []
        for i in 0..<4 {
            bodies.append(try XCTUnwrap(s.spawnBox(at: SIMD3(Float(i) * 0.1 - 0.15, 0.2, 0), halfExtents: he, mass: 0.01)))
        }
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: bodies[0]))
        let comOf = { (b: Int) in s.position(of: b)! }
        let jHingeAdd = try XCTUnwrap(s.addHingeJoint(bodyA: bodies[0], bodyB: nil, worldAnchor: comOf(bodies[0]), worldAxis: axis))
        let jPrisAdd = try XCTUnwrap(s.addPrismaticJoint(bodyA: bodies[1], bodyB: nil, worldAnchor: comOf(bodies[1]), worldAxis: axis))
        let jHingePool = try XCTUnwrap(pool.takeSlot())
        s.enableHinge(slot: jHingePool, bodyA: bodies[2], bodyB: nil, worldAnchor: comOf(bodies[2]), worldAxis: axis)
        let jPrisPool = try XCTUnwrap(pool.takeSlot())
        s.enablePrismatic(slot: jPrisPool, bodyA: bodies[3], bodyB: nil, worldAnchor: comOf(bodies[3]), worldAxis: axis)
        let joints = [jHingeAdd, jPrisAdd, jHingePool, jPrisPool]
        let names = ["hinge add", "prismatic add", "hinge pooled", "prismatic pooled"]

        func dev(_ i: Int) -> Float {
            let j = s.jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: CoinDEMSolver.maxJoints)[joints[i]]
            let aW = simd_normalize(simd_act(s.orientation(of: bodies[i])!, SIMD3(j.axisA.x, j.axisA.y, j.axisA.z)))
            return acos(min(max(simd_dot(aW, axis), -1), 1)) * 180 / .pi
        }
        // Seed 1° about +X (⟂ the axis), after the joints captured their local axes.
        for b in bodies { seedRotation(s, b, simd_quatf(angle: Self.deg(1), axis: SIMD3(1, 0, 0))) }
        var traces = [[Float]](repeating: [], count: 4)
        for i in 0..<4 { traces[i].append(dev(i)) }
        step(s, q, frames: 30, wallDt: dt) { _ in for i in 0..<4 { traces[i].append(dev(i)) } }
        for i in 0..<4 {
            let t = traces[i]
            let ratios = (1...8).map { t[$0] / max(t[$0 - 1], 1e-9) }
            let meanRatio = ratios.reduce(0, +) / Float(ratios.count)
            print("A1_1a \(names[i]): φ0=\(t[0])° φ1..5=\(t[1...5].map { String(format: "%.4f", $0) }) φ30=\(t[30])° mean ratio(1..8)=\(meanRatio)")
            XCTAssertEqual(ratios[0], 0.8, accuracy: 0.01, "\(names[i]): first substep decays ×(1−β)")
            XCTAssertEqual(meanRatio, 0.8, accuracy: 0.01, "\(names[i]): ×0.8 per substep")
            XCTAssertLessThan(t[30], 0.01, "\(names[i]): restored to parallel")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1c — VZ-0149: finalize dead-stop
    // ══════════════════════════════════════════════════════════════════════════

    /// §A1 jib (520 × 12 × 14 mm, 30 g) pivoted on a WORLD hinge through its COM and
    /// slewed by its motor at 0.1 rad/s — a 26 mm/s tip speed, but its COM is still, so
    /// the legacy dead-stop (|v| < sleepLinVel && |ω| < 0.6) zeroed it every substep.
    /// Sleep ON (§G): a motor-driven body must not count as "slow" either, or the island
    /// would freeze mid-slew after sleepFrames. Checked with the dead-stop in both modes.
    func testMotorDrivenJibSlewsAtPointOneRadPerSecond() throws {
        for scaleAware in [false, true] {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.27, scaleAwareDeadStop: scaleAware)
            let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
            let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 1, placeholderBody: jib))
            let j = try XCTUnwrap(pool.takeSlot())
            s.enableHinge(slot: j, bodyA: jib, bodyB: nil, worldAnchor: SIMD3(0, 0.2, 0), worldAxis: SIMD3(0, 1, 0))
            s.setHingeMotor(j, targetVelocity: 0.1, maxTorque: 0.01)
            var omegas: [Float] = []
            var everAsleep = false
            let q0 = s.orientation(of: jib)!
            step(s, q, frames: 120) { _ in
                omegas.append(s.angularVelocity(of: jib)!.y)
                everAsleep = everAsleep || s.isAsleep(jib)
            }
            let yaw = 2 * asin(min(1, abs((s.orientation(of: jib)! * q0.inverse).imag.y)))
            let tail = omegas.dropFirst(18)
            let worst = tail.map { abs($0 - 0.1) }.max() ?? 1
            print("A1_1c jib slew (scaleAware=\(scaleAware)): ω@0.1s=\(omegas[5]) ω@1s=\(omegas[59]) ω@2s=\(omegas[119]) worst|ω−0.1| after 0.3 s=\(worst) yaw(2 s)=\(yaw) rad everAsleep=\(everAsleep)")
            XCTAssertLessThan(worst, 0.005, "scaleAware=\(scaleAware): the motor holds 0.1 rad/s")
            XCTAssertEqual(yaw, 0.2, accuracy: 0.02, "scaleAware=\(scaleAware): ≈ 0.2 rad in 2 s")
            XCTAssertFalse(everAsleep, "scaleAware=\(scaleAware): a motor-driven body never counts as slow")
        }
    }

    /// The motor-driven exemption must not keep a motor that has REACHED its stop awake:
    /// a hinge door (limits ±0.5 rad) and a horizontal prismatic slider (±20 mm) driven
    /// into their stops sleep under the legacy dead-stop at sleepLinVel 0.03 (as they did
    /// before VZ-0149 — the unconditional marker kept the door buzzing awake forever at
    /// 0.12 rad/s, found by the A1 verifier), while the same joints commanded slowly
    /// AWAY from any stop (0.1 rad/s, 10 mm/s — both "slow" by the legacy test) keep
    /// moving at their target and never sleep.
    func testMotorIntoItsStopStillSleepsButSlowMotorsMove() throws {
        let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05, colliders: [])
        s.sleepLinVel = 0.03
        func door(_ z: Float, target: Float, limit: Float) throws -> (Int, Int) {
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.03, 0.2, z), halfExtents: SIMD3(0.03, 0.02, 0.003), mass: 0.02))
            let j = try XCTUnwrap(s.addHingeJoint(bodyA: b, bodyB: nil, worldAnchor: SIMD3(0, 0.2, z), worldAxis: SIMD3(0, 1, 0),
                                                  limits: -limit...limit, motor: (targetVelocity: target, maxTorque: 0.01)))
            return (b, j)
        }
        func slider(_ z: Float, target: Float, limit: Float) throws -> (Int, Int) {
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.2, 0.2, z), halfExtents: SIMD3(0.01, 0.01, 0.01), mass: 0.02))
            let j = try XCTUnwrap(s.addPrismaticJoint(bodyA: b, bodyB: nil, worldAnchor: SIMD3(-0.2, 0.2, z), worldAxis: SIMD3(1, 0, 0),
                                                      limits: -limit...limit, motor: (targetVelocity: target, maxForce: 0.05)))
            return (b, j)
        }
        let parked = try door(-0.2, target: 1.0, limit: 0.5), slewing = try door(0.2, target: 0.1, limit: 3)
        let pushed = try slider(-0.2, target: 0.05, limit: 0.02), creeping = try slider(0.2, target: 0.01, limit: 0.5)
        let tw0 = s.hingeTwist(slewing.1)!, sl0 = s.prismaticSlide(creeping.1)!
        var asleepAt = [Int: Int]()
        step(s, q, frames: 120) { f in
            for b in [parked.0, slewing.0, pushed.0, creeping.0] where asleepAt[b] == nil && s.isAsleep(b) { asleepAt[b] = f + 1 }
        }
        let doorTurn = abs(s.hingeTwist(slewing.1)! - tw0), slide = abs(s.prismaticSlide(creeping.1)! - sl0)
        print("A1_1c stall: door into its stop asleep at frame \(asleepAt[parked.0] ?? -1) (twist \(s.hingeTwist(parked.1)!)), slider into its stop asleep at \(asleepAt[pushed.0] ?? -1) (slide \(s.prismaticSlide(pushed.1)! * 1000) mm); slow door turned \(doorTurn) rad in 2 s (asleep \(asleepAt[slewing.0] ?? -1)), slow slider moved \(slide * 1000) mm (asleep \(asleepAt[creeping.0] ?? -1))")
        XCTAssertNotNil(asleepAt[parked.0], "a door motor holding it against its stop lets it sleep")
        XCTAssertNotNil(asleepAt[pushed.0], "a slider motor holding it against its stop lets it sleep")
        XCTAssertEqual(abs(s.hingeTwist(parked.1)!), 0.5, accuracy: 0.01, "the door reached its stop")
        XCTAssertEqual(abs(s.prismaticSlide(pushed.1)!), 0.02, accuracy: 0.001, "the slider reached its stop")
        XCTAssertNil(asleepAt[slewing.0], "a slow door slew (0.1 rad/s) is never frozen")
        XCTAssertNil(asleepAt[creeping.0], "a slow slide (10 mm/s) is never frozen")
        XCTAssertEqual(doorTurn, 0.2, accuracy: 0.02, "0.1 rad/s × 2 s")
        XCTAssertEqual(slide, 0.02, accuracy: 0.002, "10 mm/s × 2 s")
    }

    /// Without any motor: a free jib coasting at 0.1 rad/s (gravity off, sleep off) and a
    /// 10 mm box spinning at 0.5 rad/s. The scale-aware dead-stop keeps both (their
    /// surface speeds, 26 mm/s and 4.3 mm/s, are ≫ sleepLinVel = 0.2 mm/s); the legacy
    /// one zeroes both in the first substep (recorded — it is the default).
    func testScaleAwareDeadStopKeepsSlowSpinAboutCOM() throws {
        for scaleAware in [false, true] {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.27, scaleAwareDeadStop: scaleAware,
                                        sleep: false, gravity: 0, colliders: [])
            let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
            let box = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0.3), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.005))
            s.setAngularVelocity(ofSlot: jib, to: SIMD3(0, 0.1, 0))
            s.setAngularVelocity(ofSlot: box, to: SIMD3(0, 0.5, 0))
            step(s, q, frames: 60)
            let wj = s.angularVelocity(of: jib)!.y, wb = s.angularVelocity(of: box)!.y
            print("A1_1c coast 1 s (scaleAware=\(scaleAware)): jib ω=\(wj) (from 0.1), 10 mm box ω=\(wb) (from 0.5)")
            if scaleAware {
                XCTAssertGreaterThan(wj, 0.09, "the 0.52 m jib keeps coasting")
                XCTAssertGreaterThan(wb, 0.45, "the 10 mm box keeps spinning")
            } else {
                XCTAssertEqual(wj, 0, "legacy (default): |ω| < 0.6 about a still COM is dead-stopped")
                XCTAssertEqual(wb, 0, "legacy (default)")
            }
        }
    }

    /// Resting bodies still come to rest with the scale-aware test — a 10 mm box and the
    /// 0.52 m jib lying on the floor:
    ///  (a) the §G world (sleepLinVel 2e-4, sleep ON): both islands fall asleep within
    ///      sleepFrames + a few;
    ///  (b) sleep OFF with a dead-stop threshold above the solver's resting residual
    ///      (sleepLinVel 5e-4): an exact dead stop (v = ω = 0 at every frame end), and the
    ///      box's trajectory is IDENTICAL to the legacy dead-stop's — the flag changes
    ///      what slow bodies may keep, not how a resting body behaves.
    /// FINDING recorded alongside (printed, independent of the flag; filed as VZ-0157): at
    /// 1/180 s × 6 iterations with μ = 0.5 the tangent rows of the box's 4-point manifold
    /// (friction → the slide, legacy rolling resistance → most of the yaw) leave a
    /// per-substep residual (|v| ≈ 1.6e-4 m/s, a 0.015 rad/s yaw) that intPos integrates
    /// BEFORE finalize runs, so with sleep OFF the box creeps ≈ 1°/s under EITHER
    /// dead-stop (legacy included); it vanishes at 12 iterations or μ = 0, and island
    /// sleep stops it after sleepFrames.
    func testScaleAwareDeadStopStillStopsRestingBodies() throws {
        struct Run { var worstTail: Float = 0; var asleepAt = [-1, -1]; var boxDriftMM: Float = 0; var boxRotDeg: Float = 0 }
        func run(sleep: Bool, sleepLinVel: Float, frames: Int, scaleAware: Bool = true) throws -> Run {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.27, scaleAwareDeadStop: scaleAware, sleep: sleep)
            s.sleepLinVel = sleepLinVel
            let box = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.0051, 0.3), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.005))
            let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.0061, 0), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
            var r = Run()
            var p0 = SIMD3<Float>.zero, q0 = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            step(s, q, frames: frames) { f in
                if f == 29 { p0 = s.position(of: box)!; q0 = s.orientation(of: box)! }   // after the landing
                if f >= 30 {
                    for b in [box, jib] {
                        r.worstTail = max(r.worstTail, simd_length(s.velocity(of: b)!), simd_length(s.angularVelocity(of: b)!))
                    }
                    r.boxDriftMM = simd_length(s.position(of: box)! - p0) * 1000
                    r.boxRotDeg = 2 * acos(min(1, abs((s.orientation(of: box)! * q0.inverse).real))) * 180 / .pi
                }
                for (i, b) in [box, jib].enumerated() where r.asleepAt[i] < 0 && s.isAsleep(b) { r.asleepAt[i] = f + 1 }
            }
            return r
        }
        let slept = try run(sleep: true, sleepLinVel: 2e-4, frames: 60)
        let stopped = try run(sleep: false, sleepLinVel: 5e-4, frames: 300)
        let legacy = try run(sleep: false, sleepLinVel: 5e-4, frames: 300, scaleAware: false)
        let tight = try run(sleep: false, sleepLinVel: 2e-4, frames: 300)
        print("A1_1c resting (scaleAware): §G sleep on → asleep at frame box=\(slept.asleepAt[0]) jib=\(slept.asleepAt[1]); sleep off @5e-4 → worst |v|,|ω| at frame ends 30–300 = \(stopped.worstTail), box moved \(stopped.boxDriftMM) mm / \(stopped.boxRotDeg)° in 4.5 s (legacy dead-stop, same world: \(legacy.boxDriftMM) mm / \(legacy.boxRotDeg)°); sleep off @2e-4 → frame-end residual \(tight.worstTail), box \(tight.boxDriftMM) mm / \(tight.boxRotDeg)°. FINDING (VZ-0157): the ≈1°/s creep is the 6-iteration tangent-row residual integrated before finalize — identical under the legacy dead-stop")
        XCTAssertGreaterThan(slept.asleepAt[0], 0, "§G: the resting box falls asleep")
        XCTAssertLessThanOrEqual(slept.asleepAt[0], 25)
        XCTAssertGreaterThan(slept.asleepAt[1], 0, "§G: the resting jib falls asleep")
        XCTAssertLessThanOrEqual(slept.asleepAt[1], 25)
        XCTAssertEqual(stopped.worstTail, 0, "sleep off, threshold above the residual: exact dead stop at every frame end")
        XCTAssertEqual(stopped.boxDriftMM, legacy.boxDriftMM, accuracy: 1e-4, "same resting trajectory as the legacy dead-stop")
        XCTAssertEqual(stopped.boxRotDeg, legacy.boxRotDeg, accuracy: 1e-3, "same resting trajectory as the legacy dead-stop")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1d — VZ-0155: accumulated clamps (joint motors always; rolling opt-in)
    // ══════════════════════════════════════════════════════════════════════════

    /// Motor bounds are PHYSICAL per substep in the kernel itself: a hinge flap pivoted
    /// 5 mm off its COM (the bound, not Gauss–Seidel dilution, decides) and a prismatic
    /// slider under gravity, built with the RAW addHingeJoint / addPrismaticJoint motor
    /// lanes (no ÷ iterations anywhere), hold at 1.1 × the static load and slip at 0.9 ×
    /// at 6 and 12 iterations. A setter issued at 6 iterations keeps its meaning after
    /// `velocityIterations` is raised to 12 without re-issuing it.
    func testMotorBoundIsPhysicalAtAnyIterationCount() throws {
        for iters in [6, 12] {
            let (s, q) = try makeSolver(maxCoins: 12, maxRadius: 0.05, iterations: iters, colliders: [])
            let m: Float = 0.01, arm: Float = 0.005
            let tauG = m * Self.g * arm, fG = m * Self.g
            func flap(_ z: Float, _ ratio: Float) throws -> Int {
                let b = try XCTUnwrap(s.spawnBox(at: SIMD3(arm, 0.2, z), halfExtents: SIMD3(0.03, 0.004, 0.01), mass: m))
                _ = try XCTUnwrap(s.addHingeJoint(bodyA: b, bodyB: nil, worldAnchor: SIMD3(0, 0.2, z), worldAxis: SIMD3(0, 0, 1),
                                                  motor: (targetVelocity: 0, maxTorque: ratio * tauG)))
                return b
            }
            func slider(_ z: Float, _ ratio: Float) throws -> Int {
                let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.2, 0.2, z), halfExtents: SIMD3(0.01, 0.01, 0.01), mass: m))
                _ = try XCTUnwrap(s.addPrismaticJoint(bodyA: b, bodyB: nil, worldAnchor: SIMD3(0.2, 0.2, z), worldAxis: SIMD3(0, 1, 0),
                                                      motor: (targetVelocity: 0, maxForce: ratio * fG)))
                return b
            }
            let hold = try flap(-0.1, 1.1), slip = try flap(0.1, 0.9)
            let pHold = try slider(-0.1, 1.1), pSlip = try slider(0.1, 0.9)
            // Setter path, issued at 6 iterations, then the iteration count is changed.
            let placeholder = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.2, 0.05, 0.35), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.01))
            let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: placeholder))
            let sHold = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.2 + arm, 0.2, -0.1), halfExtents: SIMD3(0.03, 0.004, 0.01), mass: m))
            let sSlip = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.2 + arm, 0.2, 0.1), halfExtents: SIMD3(0.03, 0.004, 0.01), mass: m))
            let jsh = try XCTUnwrap(pool.takeSlot()), jss = try XCTUnwrap(pool.takeSlot())
            s.enableHinge(slot: jsh, bodyA: sHold, bodyB: nil, worldAnchor: SIMD3(-0.2, 0.2, -0.1), worldAxis: SIMD3(0, 0, 1))
            s.enableHinge(slot: jss, bodyA: sSlip, bodyB: nil, worldAnchor: SIMD3(-0.2, 0.2, 0.1), worldAxis: SIMD3(0, 0, 1))
            s.velocityIterations = 6
            s.setHingeMotor(jsh, targetVelocity: 0, maxTorque: 1.1 * tauG)
            s.setHingeMotor(jss, targetVelocity: 0, maxTorque: 0.9 * tauG)
            s.velocityIterations = iters
            func drop(_ b: Int) -> Float { asin(min(1, abs(simd_act(s.orientation(of: b)!, SIMD3<Float>(1, 0, 0)).y))) * 180 / .pi }
            step(s, q, frames: 60)
            let sag = { (b: Int) in (0.2 - s.position(of: b)!.y) * 1000 }
            print("A1_1d motor iters=\(iters): addHinge 1.1× drop=\(drop(hold))° 0.9× drop=\(drop(slip))°; addPrismatic 1.1× sag=\(sag(pHold)) mm 0.9× sag=\(sag(pSlip)) mm; setter@6→\(iters) 1.1× drop=\(drop(sHold))° 0.9× drop=\(drop(sSlip))°")
            XCTAssertLessThan(drop(hold), 1, "iters \(iters): raw hinge lane 1.1 × τ_g holds")
            XCTAssertGreaterThan(drop(slip), 10, "iters \(iters): raw hinge lane 0.9 × τ_g slips")
            XCTAssertLessThan(sag(pHold), 1, "iters \(iters): raw prismatic lane 1.1 × mg holds")
            XCTAssertGreaterThan(sag(pSlip), 5, "iters \(iters): raw prismatic lane 0.9 × mg slips")
            XCTAssertLessThan(drop(sHold), 1, "iters \(iters): setter 1.1 × τ_g holds after an iteration change")
            XCTAssertGreaterThan(drop(sSlip), 10, "iters \(iters): setter 0.9 × τ_g slips after an iteration change")
        }
    }

    /// Largest release tilt the Weeble HOLDS: bisection over release angles, 0.6 s each
    /// (CoinDEMActuationTests.parkTiltDeg).
    private func parkTiltDeg(iterations: Int, muR: Float, accumulated: Bool, scaleAware: Bool) throws -> Float {
        func holds(_ deg: Float) throws -> Bool {
            let (s, q) = try makeSolver(iterations: iterations, rollingResistance: muR,
                                        accumulatedRolling: accumulated, scaleAwareDeadStop: scaleAware)
            let w = try XCTUnwrap(s.spawnBallastedEgg(Self.worker, fatCenter: SIMD3(0, Self.worker.fatRadius, 0),
                                                      orient: simd_quatf(angle: Self.deg(deg), axis: SIMD3(1, 0, 0)),
                                                      friction: 0.6, restitution: 0.15))
            var worst: Float = 0
            step(s, q, frames: 36) { _ in
                worst = max(worst, abs(Self.signedTiltDeg(s.orientation(of: w)!) - deg))
            }
            return worst < max(0.1 * deg, 0.15)
        }
        var lo: Float = 0.1, hi: Float = 25
        if try holds(hi) { return hi }
        for _ in 0..<9 {
            let mid = 0.5 * (lo + hi)
            if try holds(mid) { lo = mid } else { hi = mid }
        }
        return lo
    }

    /// With the accumulated rolling clamp the Weeble's park tilt is a property of μr,
    /// not of the solver's iteration count: identical (±0.2°) at 6 and 12 iterations for
    /// a PHYSICAL μr = 0.024, and near asin(μr·h_c/d) — the static-rolling-friction
    /// balance. The legacy clamp is recorded alongside (asin(iterations·μr·h_c/d)).
    func testAccumulatedRollingParkTiltIndependentOfIterations() throws {
        let p = CoinDEMSolver.ballastedEggProperties(Self.worker)
        let d = p.comBelowFatCenter, hc = Self.worker.fatRadius - d
        func formula(_ muEff: Float) -> Float { asin(min(1, muEff * hc / d)) * 180 / .pi }
        let muR: Float = 0.024
        var acc: [Int: Float] = [:], accLegacyStop: [Int: Float] = [:], legacy: [Int: Float] = [:]
        for iters in [6, 12] {
            acc[iters] = try parkTiltDeg(iterations: iters, muR: muR, accumulated: true, scaleAware: true)
            accLegacyStop[iters] = try parkTiltDeg(iterations: iters, muR: muR, accumulated: true, scaleAware: false)
            legacy[iters] = try parkTiltDeg(iterations: iters, muR: muR / Float(iters), accumulated: false, scaleAware: false)
        }
        print("A1_1d park tilt μr=\(muR) [formula asin(μr·h_c/d)=\(formula(muR))°]: accumulated+scaleAware 6it=\(acc[6]!)° 12it=\(acc[12]!)°; accumulated+legacy dead-stop 6it=\(accLegacyStop[6]!)° 12it=\(accLegacyStop[12]!)°; legacy μr/iters 6it=\(legacy[6]!)° 12it=\(legacy[12]!)°")
        XCTAssertEqual(acc[6]!, acc[12]!, accuracy: 0.2, "accumulated: park tilt is iteration-independent (±0.2°)")
        XCTAssertEqual(accLegacyStop[6]!, accLegacyStop[12]!, accuracy: 0.2, "also with the legacy dead-stop")
        XCTAssertEqual(acc[6]!, formula(muR), accuracy: formula(muR) * 0.25, "≈ the static balance asin(μr·h_c/d)")
    }

    /// "By how much" for the scenes that set rollingResistance: a 20 mm sphere rolling
    /// at 2 m/s on a μ = 0.5 floor, deceleration over its first 0.1 s (1/240 s substeps —
    /// short enough that no configuration stops it). With the accumulated clamp the
    /// deceleration is set by μr alone (μr·g/1.4 for a rolling solid sphere); with the
    /// legacy per-iteration clamp it grows with the iteration count. Recorded for each
    /// shipping scene's (μr, iterations): Daydream Home egg rain (0.02, 4), Vintage Diner
    /// Ultra (0.035, 8), Marbles (0.11, 12), SuperquadricLab (0.04, 16).
    func testRollingResistanceLegacyScalesWithIterationsAccumulatedDoesNot() throws {
        let v0: Float = 2, seconds: Float = 0.1
        func decel(iters: Int, muR: Float, accumulated: Bool) throws -> Float {
            let (s, q) = try makeSolver(maxCoins: 4, maxRadius: 0.03, boundsMin: SIMD3(-0.5, -0.1, -0.2),
                                        boundsMax: SIMD3(0.5, 0.2, 0.2), dt: 1.0 / 240, iterations: iters,
                                        rollingResistance: muR, accumulatedRolling: accumulated, sleep: false)
            s.maxOmega = 200   // §G's 60 rad/s cap would clip the 100 rad/s rolling spin → slip
            let r: Float = 0.02
            let b = try XCTUnwrap(s.spawnSphere(at: SIMD3(-0.4, r, 0), radius: r, velocity: SIMD3(v0, 0, 0),
                                                tumble: SIMD3(0, 0, -v0 / r), mass: 0.03, friction: 0.5))
            step(s, q, frames: Int(seconds * 60))
            return (v0 - simd_length(s.velocity(of: b)!)) / seconds
        }
        var lines: [String] = []
        var acc: [Float] = [], leg: [Float] = []
        for iters in [4, 8, 12, 16] {
            let a = try decel(iters: iters, muR: 0.035, accumulated: true)
            let l = try decel(iters: iters, muR: 0.035, accumulated: false)
            acc.append(a); leg.append(l)
            lines.append("\(iters)it: accumulated \(a) legacy \(l) m/s² (×\(l / max(a, 1e-6)))")
        }
        let ideal: Float = 0.035 * Self.g / 1.4
        var scenes: [String] = []
        for (name, muR, iters) in [("DaydreamHome eggRain", Float(0.02), 4), ("VintageDinerUltra", 0.035, 8),
                                   ("Marbles", 0.11, 12), ("SuperquadricLab", 0.04, 16)] {
            let l = try decel(iters: iters, muR: muR, accumulated: false)
            let a = try decel(iters: iters, muR: muR, accumulated: true)
            scenes.append("\(name) μr=\(muR)×\(iters)it: legacy \(l) m/s² vs accumulated \(a) m/s² (×\(l / max(a, 1e-6)))")
        }
        print("A1_1d rolling sphere decel, μr=0.035 [ideal μr·g/1.4 = \(ideal) m/s²]: " + lines.joined(separator: "; "))
        print("A1_1d shipping scenes (legacy = today; accumulated = if they opted in with μr unchanged): " + scenes.joined(separator: "; "))
        XCTAssertLessThan((acc.max()! - acc.min()!) / acc.max()!, 0.1, "accumulated: iteration-independent within 10%")
        XCTAssertEqual(acc[1], ideal, accuracy: ideal * 0.3, "accumulated ≈ μr·g/1.4")
        XCTAssertGreaterThan(leg[3], 2 * leg[0], "legacy: grows with iterations (the N1 mechanism)")
    }

    /// The accumulated rolling impulse of every solved contact stays inside its own disc
    /// |aux.yz| ≤ μr·jₙ·|rA| at the end of the substep — including contacts a later
    /// iteration unloaded to jₙ = 0 (a box's manifold point whose neighbours took the
    /// weight, a separating bounce), which must hand their rolling impulse back exactly
    /// like the friction rows. A tumbling pile of 12 boxes + 12 spheres, 2 s.
    /// (A1 verifier: with the branch gated on jₙ > 0, 76 of 7415 contacts ended at jₙ = 0
    /// still carrying up to 1.5e-7 N·m·s — ≈ 70 % of a resting 10 g body's whole
    /// per-substep bound.)
    func testAccumulatedRollingImpulseStaysInsideItsBound() throws {
        let muR: Float = 0.05
        let (s, q) = try makeSolver(maxCoins: 32, maxRadius: 0.02, iterations: 6, rollingResistance: muR,
                                    accumulatedRolling: true, sleep: false)
        for i in 0..<24 {
            let fi = Float(i)
            let p = SIMD3<Float>(Float(i % 6) * 0.03 - 0.075, 0.02 + Float(i / 6) * 0.03, 0.02 * sin(fi * 1.7))
            let t = 3 * SIMD3<Float>(sin(fi * 2.3), cos(fi * 1.1), sin(fi * 0.7 + 1))
            if i % 2 == 0 {
                _ = try XCTUnwrap(s.spawnBox(at: p, halfExtents: SIMD3(0.012, 0.006, 0.009), tumble: t, mass: 0.01))
            } else {
                _ = try XCTUnwrap(s.spawnSphere(at: p, radius: 0.008, velocity: SIMD3(0.2, 0, 0), tumble: t,
                                                mass: 0.01, friction: 0.5))
            }
        }
        var seen = 0, unloaded = 0, overBound = 0
        var worstExcess: Float = 0
        step(s, q, frames: 120) { _ in
            let c = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
            for k in 0..<s.contactCount where c[k].tan2.w >= 0 {          // coloured ⇒ solved
                let cc = c[k]
                seen += 1
                if cc.rA.w <= 0 { unloaded += 1 }
                let bound = muR * max(cc.rA.w, 0) * max(simd_length(SIMD3(cc.rA.x, cc.rA.y, cc.rA.z)), 1e-4)
                let acc = simd_length(SIMD2(cc.aux.y, cc.aux.z))
                if acc > bound * 1.0001 + 1e-12 { overBound += 1; worstExcess = max(worstExcess, acc - bound) }
            }
        }
        print("A1_1d rolling accumulator bound: \(seen) solved contacts (\(unloaded) ended unloaded, jₙ = 0); over their bound: \(overBound), worst excess \(worstExcess) N·m·s")
        XCTAssertGreaterThan(seen, 1000, "the pile made contacts")
        XCTAssertEqual(overBound, 0, "every contact's accumulated rolling impulse is inside μr·jₙ·|rA|")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 1b — VZ-0148: CoinHullMath at millimetre scale
    // ══════════════════════════════════════════════════════════════════════════

    /// The §A1 segment-bar hull (4 rings × 8 plan stations), in metres × `scale`.
    /// `midRingsFirst: false` is the natural ring order that dropped 3 vertices before.
    static func barPoints(scale: Float, midRingsFirst: Bool) -> [SIMD3<Float>] {
        let st: [SIMD2<Float>] = [SIMD2(12.558, -0.344), SIMD2(12.558, 0.344), SIMD2(9.47, 3.43),
                                  SIMD2(-9.47, 3.43), SIMD2(-12.558, 0.344), SIMD2(-12.558, -0.344),
                                  SIMD2(-9.47, -3.43), SIMD2(9.47, -3.43)]
        let mm = 0.001 * scale
        var pts: [SIMD3<Float>] = []
        let rings: [(Float, Float)] = midRingsFirst ? [(-2.25, 0), (2.25, 0), (-2.75, 0.5), (2.75, 0.5)]
                                                    : [(-2.75, 0.5), (-2.25, 0), (2.25, 0), (2.75, 0.5)]
        for (z, off) in rings {
            for p in inset(st, off) { pts.append(SIMD3(p.x * mm, p.y * mm, z * mm)) }
        }
        return pts
    }
    /// Mitred inward offset of a CCW polygon by `d` (edges stay parallel).
    static func inset(_ st: [SIMD2<Float>], _ d: Float) -> [SIMD2<Float>] {
        let n = st.count
        return (0..<n).map { i in
            let p0 = st[(i + n - 1) % n], p1 = st[i], p2 = st[(i + 1) % n]
            let e0 = simd_normalize(p1 - p0), e1 = simd_normalize(p2 - p1)
            let n0 = SIMD2<Float>(-e0.y, e0.x), n1 = SIMD2<Float>(-e1.y, e1.x)
            return p1 + d * (n0 + n1) / (1 + simd_dot(n0, n1))
        }
    }
    static func shoelace(_ p: [SIMD2<Float>]) -> Double {
        var a = 0.0
        for i in 0..<p.count {
            let u = p[i], v = p[(i + 1) % p.count]
            a += Double(u.x) * Double(v.y) - Double(v.x) * Double(u.y)
        }
        return 0.5 * abs(a)
    }
    /// Exact bar volume (mm³): a prism over z ∈ [−2.25, 2.25] plus two chamfer
    /// prismatoids of height 0.5 whose mid-section is the 0.25 mm inset (prismatoid rule).
    static var barVolumeMM3: Double {
        let st: [SIMD2<Float>] = [SIMD2(12.558, -0.344), SIMD2(12.558, 0.344), SIMD2(9.47, 3.43),
                                  SIMD2(-9.47, 3.43), SIMD2(-12.558, 0.344), SIMD2(-12.558, -0.344),
                                  SIMD2(-9.47, -3.43), SIMD2(9.47, -3.43)]
        let a0 = shoelace(st), aM = shoelace(inset(st, 0.25)), a1 = shoelace(inset(st, 0.5))
        return a0 * 4.5 + 2 * (0.5 / 6) * (a0 + 4 * aM + a1)
    }

    /// Closed-manifold audit: every undirected edge in exactly two faces, every DIRECTED
    /// edge exactly once (consistent outward winding), and the signed volume.
    static func audit(_ pts: [SIMD3<Float>], _ faces: [(Int, Int, Int)]) -> (badEdges: Int, badDirected: Int, verts: Int, volume: Double, com: SIMD3<Double>) {
        var undirected: [Int: Int] = [:], directed: [Int: Int] = [:]
        var used = Set<Int>()
        var vol = 0.0
        var com = SIMD3<Double>.zero
        for t in faces {
            for (a, b) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
                undirected[min(a, b) * 4096 + max(a, b), default: 0] += 1
                directed[a * 4096 + b, default: 0] += 1
            }
            used.formUnion([t.0, t.1, t.2])
            let a = SIMD3<Double>(pts[t.0]), b = SIMD3<Double>(pts[t.1]), c = SIMD3<Double>(pts[t.2])
            let v = simd_dot(a, simd_cross(b, c)) / 6
            vol += v
            com += (a + b + c) / 4 * v
        }
        return (undirected.values.filter { $0 != 2 }.count, directed.values.filter { $0 != 1 }.count,
                used.count, vol, vol != 0 ? com / vol : .zero)
    }

    /// The same shape at 1 mm scale (the real bar, 25 mm long) and at ×1000 (1 m units)
    /// must give the same hull: all 32 vertices, 60 faces, closed and consistently wound,
    /// volume = the exact prismatoid volume, COM at the symmetric centre — in BOTH input
    /// orders (the ring order dropped 3 vertices and left 11 non-manifold edges before).
    func testHullMathIsScaleInvariantOnTheClockBar() throws {
        let exact = Self.barVolumeMM3
        var prepared: [Float: CoinHullMath.Prepared] = [:]
        for scale in [Float(1), 1000] {
            for midFirst in [false, true] {
                let pts = Self.barPoints(scale: scale, midRingsFirst: midFirst)
                let faces = try XCTUnwrap(CoinHullMath.convexHullFaces(pts), "scale \(scale) midFirst \(midFirst)")
                let a = Self.audit(pts, faces)
                let volMM3 = a.volume / pow(Double(scale) * 1e-3, 3)
                let comMM = simd_length(a.com) / (Double(scale) * 1e-3)
                let prep = try XCTUnwrap(CoinHullMath.prepare(pts))
                prepared[scale] = prep
                print("A1_1b bar scale=\(scale) midFirst=\(midFirst): faces=\(faces.count) verts=\(a.verts) badEdges=\(a.badEdges) badDirected=\(a.badDirected) volume=\(volMM3) mm³ (exact \(exact)) |COM|=\(comMM) mm; prepared verts=\(prep.vertices.count) comOffset=\(prep.comOffset / (scale * 1e-3)) mm principal=\(2 * acos(min(1, abs(prep.principalRotation.real))) * 180 / .pi)° invK=\(prep.invInertiaK)")
                XCTAssertEqual(a.verts, 32, "all 32 bar vertices are hull vertices")
                XCTAssertEqual(faces.count, 60, "V − E + F = 2 with 32 vertices ⇒ 60 triangles")
                XCTAssertEqual(a.badEdges, 0, "closed")
                XCTAssertEqual(a.badDirected, 0, "consistently wound")
                XCTAssertEqual(volMM3, exact, accuracy: exact * 1e-5, "volume")
                XCTAssertLessThan(comMM, 1e-4, "COM at the symmetric centre")
                XCTAssertEqual(prep.vertices.count, 32)
                XCTAssertLessThan(simd_length(prep.comOffset) / (scale * 1e-3), 1e-4, "prepared COM offset ≈ 0")
            }
        }
        // Scale invariance of the prepared frame: inertia (per unit mass) scales as L²,
        // bounding radius as L.
        let p1 = try XCTUnwrap(prepared[1]), p1000 = try XCTUnwrap(prepared[1000])
        for k in 0..<3 {
            XCTAssertEqual(p1.invInertiaK[k] / p1000.invInertiaK[k], 1e6, accuracy: 1e6 * 1e-4, "invInertiaK axis \(k) scales 1/L²")
        }
        XCTAssertEqual(p1000.boundingRadius / p1.boundingRadius, 1000, accuracy: 1000 * 1e-5)
    }

    /// A 1 mm cube with a 0.2 mm bump on each face centre: 14 hull vertices, 24 faces,
    /// volume 1 + 6 · (1 · 0.2 / 3) = 1.4 mm³ — at 1 mm and at 1 m. (Before: 8 of 14.)
    func testHullMathKeepsSmallBumpsOnAMillimetreCube() throws {
        for scale in [Float(1e-3), 1] {
            var pts: [SIMD3<Float>] = []
            for x in [Float(-0.5), 0.5] { for y in [Float(-0.5), 0.5] { for z in [Float(-0.5), 0.5] { pts.append(SIMD3(x, y, z) * scale) } } }
            for axis in 0..<3 {
                for sgn in [Float(-1), 1] {
                    var p = SIMD3<Float>.zero
                    p[axis] = sgn * 0.7
                    pts.append(p * scale)
                }
            }
            let faces = try XCTUnwrap(CoinHullMath.convexHullFaces(pts))
            let a = Self.audit(pts, faces)
            let vol = a.volume / pow(Double(scale), 3)
            print("A1_1b bumpy cube scale=\(scale): faces=\(faces.count) verts=\(a.verts) badEdges=\(a.badEdges) badDirected=\(a.badDirected) volume=\(vol) (exact 1.4) |COM|=\(simd_length(a.com) / Double(scale))")
            XCTAssertEqual(a.verts, 14)
            XCTAssertEqual(faces.count, 24)
            XCTAssertEqual(a.badEdges, 0)
            XCTAssertEqual(a.badDirected, 0)
            XCTAssertEqual(vol, 1.4, accuracy: 1.4 * 1e-5)
            XCTAssertLessThan(simd_length(a.com) / Double(scale), 1e-5)
        }
    }

    /// The principal frame of a hull whose inertia is NOT diagonal in its input frame.
    /// Every other hull test here is axis-aligned and symmetric, so the Jacobi solve
    /// never ran a rotation; this one does. A 24 × 6.8 × 5 mm box (unit mass: I_x =
    /// (b² + c²)/3 …), rotated 37° about (1, 2, 3) and offset from the origin, must come
    /// back with exactly the box's principal moments, its COM, and — in the principal
    /// frame — vertices at ±the half-extents; also at ×1000. Plus a direct eigen check on
    /// a dense symmetric matrix. (Found by the A1 verifier: the Jacobi rotation was the
    /// transpose of the one its θ/t formula zeroes, so it ran all 50 sweeps and left the
    /// off-diagonal at ~0.43 of a 5-unit matrix — any tilted/asymmetric hull got the
    /// inertia diagonal of an unrotated frame.)
    func testHullPrincipalFrameOfARotatedBox() throws {
        let m = simd_double3x3(rows: [SIMD3(2, 1, 0.3), SIMD3(1, 0, 0.2), SIMD3(0.3, 0.2, 5)])
        let (ev, V) = CoinHullMath.jacobiEigen(m)
        var worstResidual = 0.0
        for k in 0..<3 { worstResidual = max(worstResidual, simd_length(m * V[k] - ev[k] * V[k])) }
        let offDiag = V.transpose * m * V
        print("A1_1b jacobi dense 3×3: eigen \(ev) worst |M·v − λ·v| = \(worstResidual), off-diagonal after = \(abs(offDiag[1][0]) + abs(offDiag[2][0]) + abs(offDiag[2][1]))")
        XCTAssertLessThan(worstResidual, 1e-12, "Jacobi converges to true eigenpairs")

        for scale in [Float(1), 1000] {
            let mm = 0.001 * scale
            let he = SIMD3<Float>(12, 3.4, 2.5) * mm
            let rot = simd_quatf(angle: 37 * .pi / 180, axis: simd_normalize(SIMD3<Float>(1, 2, 3)))
            let centre = SIMD3<Float>(5, -3, 7) * mm
            var pts: [SIMD3<Float>] = []
            for sx in [Float(-1), 1] { for sy in [Float(-1), 1] { for sz in [Float(-1), 1] {
                pts.append(centre + simd_act(rot, SIMD3(sx, sy, sz) * he))
            } } }
            let prep = try XCTUnwrap(CoinHullMath.prepare(pts))
            let a = he.x, b = he.y, c = he.z
            let expectK = [3 / (b * b + c * c), 3 / (a * a + c * c), 3 / (a * a + b * b)].sorted()
            let gotK = [prep.invInertiaK.x, prep.invInertiaK.y, prep.invInertiaK.z].sorted()
            // Principal-frame vertices: every |coordinate| is one of the half-extents.
            var worstVertErr: Float = 0
            let heSorted = [a, b, c].sorted()
            for v in prep.vertices {
                let absSorted = [abs(v.x), abs(v.y), abs(v.z)].sorted()
                for k in 0..<3 { worstVertErr = max(worstVertErr, abs(absSorted[k] - heSorted[k])) }
            }
            print("A1_1b rotated box scale=\(scale): invK=\(gotK) expected \(expectK); COM err=\(simd_length(prep.comOffset - centre) / mm) mm; worst principal-frame vertex err=\(worstVertErr / mm) mm")
            for k in 0..<3 {
                XCTAssertEqual(gotK[k], expectK[k], accuracy: expectK[k] * 1e-4, "scale \(scale): principal moment \(k)")
            }
            XCTAssertLessThan(simd_length(prep.comOffset - centre) / mm, 1e-4, "scale \(scale): COM = box centre")
            XCTAssertLessThan(worstVertErr / mm, 1e-3, "scale \(scale): principal frame = the box's own axes")
        }
    }

    /// Deterministic output (the face ARRAY, not just the set) and clean rejection of
    /// degenerate input at any scale; interior / duplicate points are dropped.
    func testHullMathDeterministicAndRejectsDegenerateInput() throws {
        let pts = Self.barPoints(scale: 1, midRingsFirst: false)
        let f1 = try XCTUnwrap(CoinHullMath.convexHullFaces(pts))
        for _ in 0..<3 {
            let f2 = try XCTUnwrap(CoinHullMath.convexHullFaces(pts))
            XCTAssertTrue(f1.elementsEqual(f2, by: { $0 == $1 }), "identical face array on every call")
        }
        // Interior + duplicate points never become vertices.
        let withJunk = pts + [SIMD3<Float>(0, 0, 0), SIMD3<Float>(0.001, 0.0005, 0.0002), pts[3], pts[17]]
        let fj = try XCTUnwrap(CoinHullMath.convexHullFaces(withJunk))
        XCTAssertEqual(Self.audit(withJunk, fj).verts, 32)
        XCTAssertEqual(Self.audit(withJunk, fj).badEdges, 0)
        for scale in [Float(1e-3), 1, 1000] {
            let flat = (0..<12).map { i -> SIMD3<Float> in
                let a = Float(i) * 0.5
                return SIMD3(cos(a), sin(a), 0) * scale
            }
            XCTAssertNil(CoinHullMath.convexHullFaces(flat), "coplanar input at scale \(scale)")
            let line = (0..<6).map { SIMD3<Float>(Float($0), 2 * Float($0), 0) * scale }
            XCTAssertNil(CoinHullMath.convexHullFaces(line), "collinear input at scale \(scale)")
        }
        XCTAssertNil(CoinHullMath.convexHullFaces([SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)]))
    }

    /// On the GPU: the real-size bar hull from the natural ring order, standing on its
    /// 5.5 mm-wide long side (long axis along X, face normal along Z, width up) on a
    /// floor, stays upright for 1 s and rests at its half-width. (Before: the dropped
    /// vertices and an off-centre COM tipped it.)
    func testMillimetreBarHullStandsOnItsSide() throws {
        let (s, q) = try makeSolver(maxCoins: 4, maxRadius: 0.02)
        let h = try XCTUnwrap(s.registerHull(vertices: Self.barPoints(scale: 1, midRingsFirst: false)))
        XCTAssertEqual(h.vertices.count, 32)
        // Stand on the bottom flank: rotate the bar so its plan +Y (width, 3.43 mm) is up.
        let rd = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        let restY: Float = 0.00343 + 1e-4
        let b = try XCTUnwrap(s.spawnHull(at: SIMD3(0, restY, 0) + simd_act(rd, h.comOffset), hull: h,
                                          orient: Self.v4(rd * h.principalRotation), mass: 0.00103,
                                          friction: 0.3, restitution: 0.2))
        let q0 = s.orientation(of: b)!
        var worst: Float = 0
        step(s, q, frames: 60) { _ in
            let dq = s.orientation(of: b)! * q0.inverse
            worst = max(worst, 2 * acos(min(1, abs(dq.real))) * 180 / .pi)
        }
        let y = s.position(of: b)!.y
        print("A1_1b mm bar on its side: worst rotation over 1 s = \(worst)° COM y = \(y * 1000) mm (flank half-width 3.43)")
        XCTAssertLessThan(worst, 2, "the bar stands")
        XCTAssertEqual(y, 0.00343, accuracy: 3e-4, "resting on its flank")
    }
}
