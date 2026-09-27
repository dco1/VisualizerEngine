import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Regression gate for the CoinDEM joint work of the Digital Clock engine plan §6 items
/// 4a (native WELD), 4b (block-solved, warm-started joints), 4c (the per-frame active
/// joint list) and 4e (limits / rotation locks measured from the pose at creation) —
/// VZ-0150, VZ-0156. Same runtime-compiled-library seam and toy-scale §G configuration
/// (1/180 s, 6 velocity iterations, sleep on, sleepLinVel 2e-4) as CoinDEMActuationTests;
/// measured numbers are PRINTed with an `A3_` prefix so a run's log is the record.
@MainActor
final class CoinDEMJointBlockTests: XCTestCase {

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

    private func makeSolver(maxCoins: Int = 16, maxRadius: Float = 0.27, iterations: Int = 6,
                            boundsMin: SIMD3<Float> = SIMD3(-0.4, -0.1, -0.4),
                            boundsMax: SIMD3<Float> = SIMD3(0.4, 1.0, 0.6),
                            sleep: Bool = true, gravity: Float = CoinDEMJointBlockTests.g) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: maxRadius, halfThickness: maxRadius,
                                    boundsMin: boundsMin, boundsMax: boundsMax)
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = gravity
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = iterations
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = false
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.rollingResistance = 0.004
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2e-4
        s.maxSpeed = 3; s.maxHSpeed = 3; s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = sleep
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.setColliders([])
        return (s, queue)
    }

    /// Encode + commit + wait `frames` frames; returns each frame's physics GPU time (ms).
    @discardableResult
    private func step(_ s: CoinDEMSolver, _ q: MTLCommandQueue, frames: Int, wallDt: Float = 1.0 / 60,
                      perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var gpu: [Double] = []
        for f in 0..<frames {
            guard let cb = q.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            gpu.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
            perFrame?(f)
        }
        return gpu
    }

    static func deg(_ d: Float) -> Float { d * .pi / 180 }
    static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }
    /// Rotation angle of a quaternion, degrees — from the vector part, so it resolves
    /// hundredths of a degree (2·acos(|w|) in float cannot resolve below ≈ 0.04°).
    static func angleDeg(_ q: simd_quatf) -> Float {
        let n = q.normalized
        return 2 * atan2(simd_length(n.imag), abs(n.real)) * 180 / .pi
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4b — block-solved, warm-started joints
    // ══════════════════════════════════════════════════════════════════════════

    /// The §A1 jib (520 × 12 × 14 mm, 30 g) on ONE world hinge about +Y at the mast, its
    /// COM 0.17 m off the pivot, gravity on, no contacts — the smallest repro of VZ-0150.
    /// Measured on the A2 engine (rows solved one after another from zero impulse every
    /// substep): residual |v| after 40 frames 69.9 / 25.2 / 11.8 / 6.04 mm/s at 1 / 2 / 3 /
    /// 4 iterations, never asleep below 12. The joint is now one exact block (ball core +
    /// both axis rows, and the brake's motor row) warm-started across substeps: it must be
    /// still and asleep by frame 25 (sleepFrames 20) at EVERY count from 1 to 4, braked or not.
    func testLeverHingeSleepsAtFourIterationsOrFewer() throws {
        for brake in [false, true] {
            for iters in 1...4 {
                let (s, q) = try makeSolver(iterations: iters)
                let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
                let j = try XCTUnwrap(s.addHingeJoint(bodyA: jib, bodyB: nil, worldAnchor: SIMD3(0.17, 0.194, 0),
                                                      worldAxis: SIMD3(0, 1, 0),
                                                      motor: brake ? (targetVelocity: 0, maxTorque: 0.01) : nil))
                let p0 = s.position(of: jib)!
                var sleptAt = -1
                var v40: Float = -1, worstGap: Float = 0
                step(s, q, frames: 60) { f in
                    if f == 39 { v40 = simd_length(s.velocity(of: jib)!) }
                    if sleptAt < 0, s.isAsleep(jib) { sleptAt = f + 1 }
                    let a = s.jointWorldAnchors(j)!
                    worstGap = max(worstGap, simd_length(a.a - a.b))
                }
                let drop = (p0.y - s.position(of: jib)!.y) * 1000
                print("A3_4b lever brake=\(brake) iters=\(iters): |v|@40=\(v40 * 1000) mm/s sleptAt=\(sleptAt) drop=\(drop) mm worst anchor gap=\(worstGap * 1000) mm")
                XCTAssertLessThan(v40, 1e-4, "brake=\(brake) iters=\(iters): no residual velocity")
                XCTAssertGreaterThan(sleptAt, 0, "brake=\(brake) iters=\(iters): the lever hinge sleeps")
                XCTAssertLessThanOrEqual(sleptAt, 25, "brake=\(brake) iters=\(iters): within sleepFrames + 5")
                XCTAssertLessThan(abs(drop), 0.05, "brake=\(brake) iters=\(iters): the jib does not sag")
                XCTAssertLessThan(worstGap, 1e-4, "brake=\(brake) iters=\(iters): the pivot holds")
            }
        }
    }

    /// The Digital Clock crane (§C1, p2-FINAL), every brake on, no contacts, 3 s:
    /// jib on a world hinge (0.17 m lever), trolley on a prismatic along the jib, a 12 g
    /// block on 4 parallel distance "falls", the plunger rail on a prismatic under the block
    /// (limits 0…12 mm, spawned AT its lower stop), the magnet puck on the wrist hinge
    /// (limits −5°…95°), a 1.03 g bar hull welded to the puck — 9 joints, chain order.
    /// Measured on the A2 engine at 6 iterations: rail 25.8°, puck 62.9°, plunger run to its
    /// 12 mm stop, 133 mm/s still moving at 3 s, never asleep (4 iterations: 41° / 82°).
    /// Gate (§6 4b): at 4 AND 6 iterations every body settles < 0.5° from its spawn
    /// orientation, the joints close (anchor gaps, falls, plunger), and the chain sleeps.
    /// The COLD-START transient — the first frames, before the warm-started impulses carry
    /// the load — is printed: at the default jointInnerPasses = 1 the puck swings ≈5° while
    /// the chain loads, at 3 inner passes ≈1°.
    func testBrakedCraneChainHoldsAndSleeps() throws {
        for (iters, passes) in [(4, 1), (6, 1), (6, 3)] {
            let r = try runCrane(iterations: iters, passes: passes)
            let names = ["jib", "trolley", "block", "rail", "puck", "bar"]
            print("A3_4b crane iters=\(iters) passes=\(passes): settled rot (°) \(zip(names, r.settledRot).map { "\($0)=\(String(format: "%.3f", $1))" }) | transient max (°) \(zip(names, r.maxRot).map { "\($0)=\(String(format: "%.3f", $1))" }) | falls err \(r.fallErrMM) mm, plunger \(r.plungerMM) mm, wrist \(r.wristDeg)°, J7 gap \(r.j7GapMM) mm | asleep at frame \(r.sleptAt), tail |v| \(r.tailSpeed * 1000) mm/s")
            for (n, rot) in zip(names, r.settledRot) {
                XCTAssertLessThan(rot, 0.5, "iters=\(iters) passes=\(passes): \(n) holds < 0.5°")
            }
            XCTAssertLessThan(r.fallErrMM, 0.05, "iters=\(iters): the 4 falls hold their length")
            XCTAssertLessThan(abs(r.plungerMM), 0.5, "iters=\(iters): the braked plunger stays at its stop")
            XCTAssertLessThan(r.j7GapMM, 0.01, "iters=\(iters): the wrist pivot is closed")
            XCTAssertGreaterThan(r.sleptAt, 0, "iters=\(iters) passes=\(passes): the braked crane sleeps")
            XCTAssertLessThanOrEqual(r.sleptAt, 60, "iters=\(iters) passes=\(passes): within 1 s")
            XCTAssertEqual(r.tailSpeed, 0, "iters=\(iters): asleep ⇒ still")
            if passes == 3 { XCTAssertLessThan(r.maxRot.max()!, 2.0, "3 inner passes: cold-start transient < 2°") }
        }
    }

    /// The joint solve's threadgroup-cached path (joints and their bodies staged in
    /// threadgroup memory, used while ≤ 40 joints / 80 bodies) and its direct device-memory
    /// path run the same per-joint code (one template) on the same values: the braked crane
    /// and the same crane driven by its motors for 1 s agree to float rounding. Not bit for
    /// bit: the core is inlined into two call sites and fast math may associate the sums
    /// differently in each (measured worst |Δ| 7e-7 over positions ~0.8 m and quaternion
    /// components — about 10 ulps). Each configuration is itself deterministic; the path
    /// only changes when the active-joint count crosses 40.
    func testThreadgroupCachedJointSolveMatchesDirectPath() throws {
        func run(cacheOff: Bool) throws -> [SIMD4<Float>] {
            let (s, q, bodies, rig) = try buildCrane(iterations: 6, passes: 2)
            s.jointThreadgroupCacheDisabledForTesting = cacheOff
            step(s, q, frames: 30)
            s.wake(bodies)
            s.setHingeMotor(rig.j0, targetVelocity: 0.2, maxTorque: 0.01)
            s.setPrismaticMotor(rig.j1, targetVelocity: -0.03, maxForce: 0.02)
            s.setPrismaticMotor(rig.j6, targetVelocity: 0.01, maxForce: 0.003)
            s.setHingeMotor(rig.j7, targetVelocity: 1.0, maxTorque: 2e-4)
            step(s, q, frames: 60)
            return bodies.flatMap { b -> [SIMD4<Float>] in
                let p = s.position(of: b)!, o = s.orientation(of: b)!
                return [SIMD4(p, 0), SIMD4(o.imag, o.real)]
            }
        }
        let cached = try run(cacheOff: false), direct = try run(cacheOff: true), again = try run(cacheOff: false)
        let worst = zip(cached, direct).map { simd_length($0 - $1) }.max()!
        let rerun = zip(cached, again).map { simd_length($0 - $1) }.max()!
        print("A3_4b threadgroup-cached vs direct joint solve, crane braked 0.5 s then driven 1 s: worst |Δ| over poses = \(worst); same path re-run |Δ| = \(rerun)")
        XCTAssertLessThan(worst, 1e-5, "the two paths agree to float rounding")
        XCTAssertEqual(rerun, 0, "each path is deterministic")
    }

    struct CraneRun {
        var settledRot: [Float] = [], maxRot: [Float] = []
        var fallErrMM: Float = 0, plungerMM: Float = 0, wristDeg: Float = 0, j7GapMM: Float = 0
        var sleptAt = -1
        var tailSpeed: Float = 0
    }

    struct CraneRig { var j0 = -1, j1 = -1, j6 = -1, j7 = -1; var falls: [Int] = []; var cable: Float = 0 }

    private func runCrane(iterations: Int, passes: Int) throws -> CraneRun {
        let (s, q, bodies, rig) = try buildCrane(iterations: iterations, passes: passes)
        let (j6, j7, falls, cable) = (rig.j6, rig.j7, rig.falls, rig.cable)
        XCTAssertEqual(s.activeJointCount, 9)

        let q0 = bodies.map { s.orientation(of: $0)! }
        var r = CraneRun()
        r.maxRot = [Float](repeating: 0, count: bodies.count)
        step(s, q, frames: 180) { f in
            for (i, b) in bodies.enumerated() {
                r.maxRot[i] = max(r.maxRot[i], Self.angleDeg(s.orientation(of: b)! * q0[i].inverse))
            }
            if f >= 120 { r.tailSpeed = max(r.tailSpeed, bodies.map { simd_length(s.velocity(of: $0)!) }.max()!) }
            if r.sleptAt < 0, bodies.allSatisfy({ s.isAsleep($0) }) { r.sleptAt = f + 1 }
        }
        r.settledRot = bodies.indices.map { Self.angleDeg(s.orientation(of: bodies[$0])! * q0[$0].inverse) }
        r.fallErrMM = falls.map { abs(s.distanceJointLength($0)! - cable) * 1000 }.max()!
        r.plungerMM = s.prismaticSlide(j6)! * 1000
        r.wristDeg = s.hingeTwist(j7)! * 180 / .pi
        let a7 = s.jointWorldAnchors(j7)!
        r.j7GapMM = simd_length(a7.a - a7.b) * 1000
        return r
    }

    /// The §C1 crane, braked, no contacts: (solver, queue, [jib, trolley, block, rail, puck, bar], rig).
    private func buildCrane(iterations: Int, passes: Int) throws -> (CoinDEMSolver, MTLCommandQueue, [Int], CraneRig) {
        let (s, q) = try makeSolver(maxCoins: 16, iterations: iterations,
                                    boundsMin: SIMD3(-0.3, 0.5, 0.0), boundsMax: SIMD3(0.3, 0.9, 0.46))
        s.jointInnerPasses = passes
        let hull = try XCTUnwrap(s.registerHull(vertices: CoinDEMActuationTests.barHullPoints()))
        let xt: Float = 0.10, cable: Float = 0.09
        let blockTop = 0.8175 - cable
        let railY = blockTop - 0.014 - 0.0005 - 0.0025
        let heldBar = try XCTUnwrap(s.spawnHull(at: SIMD3<Float>(xt, railY, 0.1967 - 0.00275) + hull.comOffset, hull: hull,
                                                orient: Self.v4(hull.principalRotation),
                                                mass: 0.00103, friction: 0.3, restitution: 0.2))
        let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.8315, 0.205), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03))
        let trolley = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, 0.8215, 0.205), halfExtents: SIMD3(0.01, 0.004, 0.008), mass: 0.006))
        let block = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, blockTop - 0.007, 0.205), halfExtents: SIMD3(0.011, 0.007, 0.005), mass: 0.012))
        let rail = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, railY, 0.210), halfExtents: SIMD3(0.0025, 0.0025, 0.010), mass: 0.002))
        let puck = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, railY, 0.1982), halfExtents: SIMD3(0.0025, 0.0025, 0.0015), mass: 0.0015))
        let bodies = [jib, trolley, block, rail, puck, heldBar]
        // Chain order (slot order = the serial Gauss–Seidel order): J0, J1, falls, J6, J7, grip.
        let rig = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: jib))
        let j0 = try XCTUnwrap(rig.takeSlot())
        s.enableHinge(slot: j0, bodyA: jib, bodyB: nil, worldAnchor: SIMD3(0.17, 0.8255, 0.205), worldAxis: SIMD3(0, 1, 0))
        let j1 = try XCTUnwrap(rig.takeSlot())
        s.enablePrismatic(slot: j1, bodyA: trolley, bodyB: jib, worldAnchor: SIMD3(xt, 0.8215, 0.205), worldAxis: SIMD3(1, 0, 0))
        var falls: [Int] = []
        for (dx, dz) in [(Float(-0.006), Float(-0.005)), (0.006, -0.005), (-0.006, 0.005), (0.006, 0.005)] {
            falls.append(try XCTUnwrap(s.addDistanceJoint(bodyA: trolley, bodyB: block,
                                                          worldAnchorA: SIMD3(xt + dx, 0.8175, 0.205 + dz),
                                                          worldAnchorB: SIMD3(xt + dx, blockTop, 0.205 + dz))))
        }
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 3, placeholderBody: jib))
        let j6 = try XCTUnwrap(pool.takeSlot())
        s.enablePrismatic(slot: j6, bodyA: rail, bodyB: block, worldAnchor: SIMD3(xt, railY, 0.210), worldAxis: SIMD3(0, 0, -1))
        s.setJointLimits(j6, 0...0.012)
        let j7 = try XCTUnwrap(pool.takeSlot())
        s.enableHinge(slot: j7, bodyA: rail, bodyB: puck, worldAnchor: SIMD3(xt, railY, 0.1982), worldAxis: SIMD3(0, 0, 1))
        s.setJointLimits(j7, Self.deg(-5)...Self.deg(95))
        let grip = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(grip, bodyA: puck, bodyB: heldBar, worldAnchor: SIMD3(xt, railY, 0.1967))
        s.setHingeMotor(j0, targetVelocity: 0, maxTorque: 0.01)
        s.setPrismaticMotor(j1, targetVelocity: 0, maxForce: 0.02)
        s.setPrismaticMotor(j6, targetVelocity: 0, maxForce: 0.003)
        s.setHingeMotor(j7, targetVelocity: 0, maxTorque: 2e-4)
        return (s, q, bodies, CraneRig(j0: j0, j1: j1, j6: j6, j7: j7, falls: falls, cable: cable))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4a — native WELD
    // ══════════════════════════════════════════════════════════════════════════

    /// A native weld (addWeldJoint, one slot) holds a lever at ANY relative orientation:
    /// a 60 mm, 10 g bar welded to the WORLD 30 mm off its COM at a 37° skew orientation,
    /// and a two-body weld hanging a 6 g box 25 mm off a world-welded plate at 170° (the
    /// relative quaternion's w near 0, where an absolute twist would wrap). Gravity on,
    /// sleep on: both hold (< 0.02 mm / 0.05° from the pose at creation) and sleep with no
    /// residual velocity — the 6-row block is exact and warm-started.
    func testNativeWeldHoldsLeversAtAnyOrientationAndSleeps() throws {
        let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05)
        let skew = simd_quatf(angle: Self.deg(37), axis: simd_normalize(SIMD3<Float>(1, 2, 0.5)))
        let lever = try XCTUnwrap(s.spawnBox(at: SIMD3(0.0, 0.3, 0.0), halfExtents: SIMD3(0.03, 0.004, 0.006),
                                             orient: Self.v4(skew), mass: 0.01))
        let anchor = s.position(of: lever)! + simd_act(skew, SIMD3<Float>(-0.03, 0, 0))
        let jw = try XCTUnwrap(s.addWeldJoint(bodyA: lever, bodyB: nil, worldAnchor: anchor))
        let plate = try XCTUnwrap(s.spawnBox(at: SIMD3(0.2, 0.3, 0.0), halfExtents: SIMD3(0.02, 0.004, 0.02), mass: 0.02))
        let flip = simd_quatf(angle: Self.deg(170), axis: simd_normalize(SIMD3<Float>(0.3, 1, 0.2)))
        let box = try XCTUnwrap(s.spawnBox(at: SIMD3(0.225, 0.3, 0.0), halfExtents: SIMD3(0.005, 0.005, 0.005),
                                           orient: Self.v4(flip), mass: 0.006))
        _ = try XCTUnwrap(s.addWeldJoint(bodyA: plate, bodyB: nil, worldAnchor: SIMD3(0.2, 0.3, 0.0)))
        let jp = try XCTUnwrap(s.addWeldJoint(bodyA: plate, bodyB: box, worldAnchor: SIMD3(0.2125, 0.3, 0.0)))
        XCTAssertEqual(s.activeJointCount, 3, "one slot per weld")
        let refs = [(lever, s.position(of: lever)!, s.orientation(of: lever)!), (box, s.position(of: box)!, s.orientation(of: box)!)]
        var worst = (mm: Float(0), deg: Float(0)), sleptAt = -1
        step(s, q, frames: 90) { f in
            for (b, p0, q0) in refs {
                worst = (max(worst.mm, simd_length(s.position(of: b)! - p0) * 1000),
                         max(worst.deg, Self.angleDeg(s.orientation(of: b)! * q0.inverse)))
            }
            if sleptAt < 0, [lever, plate, box].allSatisfy({ s.isAsleep($0) }) { sleptAt = f + 1 }
        }
        let jr = s.jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: CoinDEMSolver.maxJoints)
        print("A3_4a native weld: world lever (37° skew, 30 mm off COM) + two-body (170°): worst drift \(worst.mm) mm / \(worst.deg)°, all asleep at frame \(sleptAt); ref(lever)=\(jr[jw].ref) ref(pair)=\(jr[jp].ref)")
        XCTAssertLessThan(worst.mm, 0.02)
        XCTAssertLessThan(worst.deg, 0.05)
        XCTAssertGreaterThan(sleptAt, 0, "welded levers sleep")
        XCTAssertLessThanOrEqual(sleptAt, 25)
        XCTAssertEqual(jr[jw].meta.x, 4, "type 4 = WELD")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4e — limits / rotation locks measured from creation (VZ-0156)
    // ══════════════════════════════════════════════════════════════════════════

    /// The prismatic rotation lock holds the relative orientation at CREATION: a slider on
    /// a world prismatic spawned 0.8 rad about its own slide axis (and one at a skew 50°
    /// orientation) keeps that orientation, and a two-body slider between differently
    /// oriented bodies keeps theirs. The old twist lock drove the ABSOLUTE twist to zero
    /// (the world slider snapped 0.8 rad).
    func testPrismaticRotationLockHoldsCreationOrientation() throws {
        let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05, gravity: 0)
        let axis = SIMD3<Float>(1, 0, 0)
        let about = simd_quatf(angle: 0.8, axis: axis)
        let skew = simd_quatf(angle: Self.deg(50), axis: simd_normalize(SIMD3<Float>(0.2, 1, 0.7)))
        let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.3, -0.2), halfExtents: SIMD3(0.01, 0.004, 0.006), orient: Self.v4(about), mass: 0.006))
        let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.3, 0.0), halfExtents: SIMD3(0.01, 0.004, 0.006), orient: Self.v4(skew), mass: 0.006))
        let c = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.3, 0.2), halfExtents: SIMD3(0.02, 0.004, 0.02), mass: 0.02))
        let d = try XCTUnwrap(s.spawnBox(at: SIMD3(0.03, 0.3, 0.2), halfExtents: SIMD3(0.005, 0.005, 0.005), orient: Self.v4(skew), mass: 0.004))
        _ = try XCTUnwrap(s.addPrismaticJoint(bodyA: a, bodyB: nil, worldAnchor: SIMD3(0, 0.3, -0.2), worldAxis: axis))
        _ = try XCTUnwrap(s.addPrismaticJoint(bodyA: b, bodyB: nil, worldAnchor: SIMD3(0, 0.3, 0.0), worldAxis: axis))
        _ = try XCTUnwrap(s.addPrismaticJoint(bodyA: d, bodyB: c, worldAnchor: SIMD3(0.03, 0.3, 0.2), worldAxis: axis))
        let q0 = [a, b, d].map { s.orientation(of: $0)! }
        let rel0 = s.orientation(of: c)!.inverse * s.orientation(of: d)!
        // Push each slider along its axis (and spin c) so the locks are loaded.
        s.setVelocity(ofSlot: a, to: SIMD3(0.05, 0, 0)); s.setVelocity(ofSlot: b, to: SIMD3(-0.05, 0, 0))
        s.setAngularVelocity(ofSlot: c, to: SIMD3(0, 2, 0))
        var worst: Float = 0, worstRel: Float = 0
        step(s, q, frames: 60) { _ in
            for (i, body) in [a, b].enumerated() { worst = max(worst, Self.angleDeg(s.orientation(of: body)! * q0[i].inverse)) }
            worstRel = max(worstRel, Self.angleDeg((s.orientation(of: c)!.inverse * s.orientation(of: d)!) * rel0.inverse))
        }
        print("A3_4e prismatic: world sliders (0.8 rad about the axis, 50° skew) worst rotation \(worst)°; two-body (skew vs identity, carrier spun 2 rad/s) worst relative rotation \(worstRel)°")
        XCTAssertLessThan(worst, 0.05, "world sliders keep their creation orientation (the old lock snapped 0.8 rad)")
        XCTAssertLessThan(worstRel, 0.3, "a two-body slider keeps its creation relative orientation while turning")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4c — the active joint list
    // ══════════════════════════════════════════════════════════════════════════

    /// 1000 reserved-but-disabled pool slots next to one active hinge cost nothing: the
    /// joint kernels, generate's collideConnected scan and the island union read only the
    /// per-frame list of enabled slots. On the A2 engine the same 1000 disabled slots added
    /// +16 ms per frame (the serial joint pass and the generate scan walked every slot).
    /// Compared on min frame GPU time over 40 frames (robust to other GPU work).
    func testDisabledPoolSlotsAreFree() throws {
        func frameMin(reserve: Int) throws -> (Double, Int) {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05, sleep: false)
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.01, 0.3, 0), halfExtents: SIMD3(0.01, 0.003, 0.004), mass: 0.004))
            if reserve > 0 { _ = try XCTUnwrap(CoinJointPool(solver: s, reserve: reserve, placeholderBody: b)) }
            _ = try XCTUnwrap(s.addHingeJoint(bodyA: b, bodyB: nil, worldAnchor: SIMD3(0, 0.3, 0), worldAxis: SIMD3(0, 0, 1)))
            step(s, q, frames: 10, wallDt: 1.0 / 30)
            return (step(s, q, frames: 40, wallDt: 1.0 / 30).min()!, s.jointCount)
        }
        let (bare, n0) = try frameMin(reserve: 0)
        let (pooled, n1) = try frameMin(reserve: 1000)
        print("A3_4c one hinge: \(n0) slot(s) → min frame \(String(format: "%.3f", bare)) ms; + 1000 disabled pool slots (\(n1) slots) → \(String(format: "%.3f", pooled)) ms (Δ \(String(format: "%.3f", pooled - bare)) ms; A2: +16 ms)")
        XCTAssertEqual(n1, n0 + 1000)
        XCTAssertLessThan(pooled - bare, 0.3, "disabled slots are never read by the GPU")
    }

    /// A pooled slot re-enabled between OTHER bodies in the same host tick starts cold. A
    /// weld carrying a 10 g lever 30 mm off its anchor (warm-start impulse ≈ m·g·dt) is
    /// disabled and, in the same tick, its slot is enabled as a weld between a box stacked
    /// on another box resting on a table; the run is repeated with a FRESH pool slot (never
    /// used) for that weld. The pair must move bit-identically in both runs: the recycled
    /// slot carries none of the lever's impulse. The table matters: a stale impulse inside
    /// ONE isolated joint is internal (zero net momentum) and its exact block cancels it in
    /// the first pass, so a free pair shows nothing either way — the old version of this
    /// test passed with the reset flag removed. Here the contact colour runs between the
    /// joint's warm start and its solve and turns the stale impulse into lift: without the
    /// reset the runs differ by 73 mm/s (measured).
    func testReenabledPoolSlotStartsCold() throws {
        func run(recycle: Bool) throws -> [SIMD3<Float>] {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05)
            s.setColliders([.box(center: SIMD3(0.2, -0.01, 0), halfExtents: SIMD3(0.05, 0.01, 0.05), friction: 0.5)])
            let lever = try XCTUnwrap(s.spawnBox(at: SIMD3(0.03, 0.3, 0), halfExtents: SIMD3(0.03, 0.004, 0.006), mass: 0.01))
            let y = try XCTUnwrap(s.spawnBox(at: SIMD3(0.2, 0.005, 0), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.004))
            let x = try XCTUnwrap(s.spawnBox(at: SIMD3(0.2, 0.015, 0), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.004))
            let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: lever))
            let w = try XCTUnwrap(pool.takeWeld()), fresh = try XCTUnwrap(pool.takeWeld())
            s.enableWeld(w, bodyA: lever, bodyB: nil, worldAnchor: SIMD3(0, 0.3, 0))
            s.sleepEnabled = false
            step(s, q, frames: 30)                                // the weld carries the lever
            XCTAssertLessThan(simd_length(s.velocity(of: lever)!), 1e-4, "precondition: the lever is held")
            let px = s.position(of: x)!, py = s.position(of: y)!
            s.disableWeld(w)
            s.enableWeld(recycle ? w : fresh, bodyA: x, bodyB: y, worldAnchor: (px + py) / 2)   // same tick
            var trace: [SIMD3<Float>] = []
            step(s, q, frames: 10) { _ in
                trace += [s.velocity(of: x)!, s.velocity(of: y)!, s.position(of: x)!, s.position(of: y)!]
            }
            return trace
        }
        let recycled = try run(recycle: true), fresh = try run(recycle: false)
        let worst = zip(recycled, fresh).map { simd_length($0 - $1) }.max()!
        print("A3_4c re-enabled slot vs a fresh slot, welded pair on a table: worst |Δ| over velocities and positions = \(worst)")
        XCTAssertEqual(worst, 0, "a recycled slot starts exactly as cold as a fresh one")
    }

    /// The joint table changed AFTER a frame was encoded and BEFORE the GPU ran it (a frame
    /// in flight): `removeJoint` + `addBallJoint` re-use the slot for two other bodies. The
    /// joint solve's threadgroup cache was built from the table at encode (the old pair);
    /// the block was prepared from the table as the GPU read it (the new pair). The pass
    /// must then fall back to the direct path — which is what the pre-A3 kernel amounted
    /// to: the new joint simply takes effect a frame early — and never apply the new
    /// joint's block to the OLD bodies. Measured before the fallback: the old pair's
    /// velocity changed by 0.09 mm/s and the new pair was kicked to 1.85 mm/s by the
    /// uncorrected stale warm start; the direct path leaves both untouched.
    func testSlotReboundWhileFrameInFlightNeverTouchesTheOldPair() throws {
        func run(cacheOff: Bool) throws -> [SIMD3<Float>] {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.05, gravity: 0)
            let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.3, 0), halfExtents: SIMD3(0.01, 0.004, 0.006), mass: 0.01))
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.03, 0.3, 0), halfExtents: SIMD3(0.01, 0.004, 0.006), mass: 0.01))
            let c = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.3, 0.2), halfExtents: SIMD3(0.01, 0.004, 0.006), mass: 0.004))
            let d = try XCTUnwrap(s.spawnBox(at: SIMD3(0.03, 0.3, 0.2), halfExtents: SIMD3(0.01, 0.004, 0.006), mass: 0.004))
            s.jointThreadgroupCacheDisabledForTesting = cacheOff
            let j = try XCTUnwrap(s.addHingeJoint(bodyA: a, bodyB: b, worldAnchor: SIMD3(0.015, 0.3, 0), worldAxis: SIMD3(0, 1, 0)))
            step(s, q, frames: 2)
            s.wake([a, b, c, d])
            s.setAngularVelocity(ofSlot: a, to: SIMD3(0, 5, 0))   // load the hinge (warm impulses)
            step(s, q, frames: 1)
            let before = [a, b, c, d].map { s.velocity(of: $0)! }
            guard let cb = q.makeCommandBuffer() else { throw XCTSkip("no command buffer") }
            s.encode(to: cb, wallDt: 1.0 / 60)                    // encoded with slot j = (a, b) …
            s.removeJoint(j)                                       // … re-bound before it runs
            let j2 = try XCTUnwrap(s.addBallJoint(bodyA: c, bodyB: d, worldAnchor: SIMD3(0.015, 0.3, 0.2)))
            XCTAssertEqual(j2, j, "precondition: the slot is re-used")
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            let after = [a, b, c, d].map { s.velocity(of: $0)! }
            print("A3_4c slot re-bound in flight (cache \(cacheOff ? "off" : "on")): |v| a,b,c,d before \(before.map { simd_length($0) * 1000 }) mm/s → after \(after.map { simd_length($0) * 1000 }) mm/s")
            return after
        }
        let cached = try run(cacheOff: false), direct = try run(cacheOff: true)
        let worst = zip(cached, direct).map { simd_length($0 - $1) }.max()!
        print("A3_4c slot re-bound in flight: threadgroup-cached vs direct worst |Δv| = \(worst * 1000) mm/s")
        XCTAssertLessThan(worst, 1e-6, "the cached pass falls back to the direct one: no impulse on the old pair")
        XCTAssertLessThan(simd_length(cached[2]) + simd_length(cached[3]), 1e-6, "the new, still pair is not kicked")
    }
}
