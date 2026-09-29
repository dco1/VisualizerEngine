import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Engine gate for VZ-0168 (stage B2): Coulomb friction in the sheaves of the crane's rope falls
/// (CoinDEMSolver+SheaveFriction.swift; CoinDEM.metal "Swing friction of a DISTANCE joint").
/// The §C1 crane (p2-FINAL) — jib on a world hinge, trolley on a prismatic, a 12 g block on 4
/// parallel distance falls, plunger rail, wrist puck, a welded 1.03 g bar — at the Digital
/// Clock's §G configuration. After a representative trolley move with the brakes then on, the
/// hook assembly swings as a pendulum of length L on its falls; the expectation is the Coulomb
/// law worked by hand: with friction arm c per fall the swing loses 2c of amplitude per half
/// cycle, comes to rest within ±c of plumb, and the crane sleeps; with c = 0 it keeps swinging.
/// Measured values are PRINTed with a `SHEAVE_` prefix.
@MainActor
final class CoinDEMSheaveFrictionTests: XCTestCase {

    // ── Harness (the joint-block suite's toy-scale solver) ────────────────────

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let url = CoinDEMFootTests.shaderURL
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("CoinDEM.metal not found") }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: url), queue)
        cached = c
        return c
    }

    static let g: Float = 9.81

    private func makeSolver(iterations: Int = 6) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 16,
                                    coinRadius: 0.27, halfThickness: 0.27,
                                    boundsMin: SIMD3(-0.3, 0.5, 0.0), boundsMax: SIMD3(0.3, 0.9, 0.46))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = Self.g
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
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.scaleAwareDeadStop = true          // the clock's setting (a slow slew is motion)
        s.setColliders([])
        return (s, queue)
    }

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

    struct Crane { var bodies: [Int] = []; var j0 = -1, j1 = -1, j6 = -1, j7 = -1; var falls: [Int] = []; var cable: Float = 0 }

    /// The §C1 crane, braked, no contacts — CoinDEMJointBlockTests' chain, falls `cable` long.
    private func buildCrane(_ s: CoinDEMSolver, cable: Float = 0.09) throws -> Crane {
        let hull = try XCTUnwrap(s.registerHull(vertices: CoinDEMActuationTests.barHullPoints()))
        let xt: Float = 0.10
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
        var c = Crane(bodies: [jib, trolley, block, rail, puck, heldBar], cable: cable)
        let rig = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: jib))
        c.j0 = try XCTUnwrap(rig.takeSlot())
        s.enableHinge(slot: c.j0, bodyA: jib, bodyB: nil, worldAnchor: SIMD3(0.17, 0.8255, 0.205), worldAxis: SIMD3(0, 1, 0))
        c.j1 = try XCTUnwrap(rig.takeSlot())
        s.enablePrismatic(slot: c.j1, bodyA: trolley, bodyB: jib, worldAnchor: SIMD3(xt, 0.8215, 0.205), worldAxis: SIMD3(1, 0, 0))
        for (dx, dz) in [(Float(-0.006), Float(-0.005)), (0.006, -0.005), (-0.006, 0.005), (0.006, 0.005)] {
            c.falls.append(try XCTUnwrap(s.addDistanceJoint(bodyA: trolley, bodyB: block,
                                                            worldAnchorA: SIMD3(xt + dx, 0.8175, 0.205 + dz),
                                                            worldAnchorB: SIMD3(xt + dx, blockTop, 0.205 + dz))))
        }
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 3, placeholderBody: jib))
        c.j6 = try XCTUnwrap(pool.takeSlot())
        s.enablePrismatic(slot: c.j6, bodyA: rail, bodyB: block, worldAnchor: SIMD3(xt, railY, 0.210), worldAxis: SIMD3(0, 0, -1))
        s.setJointLimits(c.j6, 0...0.012)
        c.j7 = try XCTUnwrap(pool.takeSlot())
        s.enableHinge(slot: c.j7, bodyA: rail, bodyB: puck, worldAnchor: SIMD3(xt, railY, 0.1982), worldAxis: SIMD3(0, 0, 1))
        s.setJointLimits(c.j7, Self.deg(-5)...Self.deg(95))
        let grip = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(grip, bodyA: puck, bodyB: heldBar, worldAnchor: SIMD3(xt, railY, 0.1967))
        brakes(s, c)
        return c
    }

    private func brakes(_ s: CoinDEMSolver, _ c: Crane) {
        s.setHingeMotor(c.j0, targetVelocity: 0, maxTorque: 0.01)
        s.setPrismaticMotor(c.j1, targetVelocity: 0, maxForce: 0.02)
        s.setPrismaticMotor(c.j6, targetVelocity: 0, maxForce: 0.003)
        s.setHingeMotor(c.j7, targetVelocity: 0, maxTorque: 2e-4)
    }

    struct SwayRun {
        var x0: Float = 0                  // sway amplitude at the brake (m)
        var peaks: [Float] = []            // |sway| at successive turning points after the brake (m)
        var decrementPerHalf: Float = 0    // mean amplitude loss per half cycle (m)
        var restOffset: Float = 0          // |sway| at the end (m)
        var sleptAfter: Float = -1         // s after the brake until every crane body is asleep (−1 never)
        var awakeAtEnd = 0
        var fallErrMM: Float = 0
        var idleGPU: Double = 0            // median physics GPU ms over the last second
    }

    /// Settle braked, then a representative trolley move — 60 mm/s for 0.8 s on the §C4 motor
    /// (0.02 N), then brake, as an operator without input shaping would — and follow the hook
    /// assembly's sway (block COM − trolley COM along the jib, relative to its hanging offset)
    /// for `idle` seconds.
    private func swayAfterMove(arm: Float, idle: Float = 10) throws -> SwayRun {
        let (s, q) = try makeSolver()
        let c = try buildCrane(s)
        for f in c.falls { s.setDistanceSwingFriction(f, arm: arm) }
        XCTAssertEqual(s.distanceSwingFriction(c.falls[0]), arm)
        XCTAssertNil(s.distanceSwingFriction(c.j0), "only distance joints carry swing friction")
        let (trolley, block) = (c.bodies[1], c.bodies[2])
        step(s, q, frames: 60)
        let hang = s.position(of: block)! - s.position(of: trolley)!
        func sway() -> Float { (s.position(of: block)! - s.position(of: trolley)! - hang).x }
        s.wake(c.bodies)
        s.setPrismaticMotor(c.j1, targetVelocity: 0.06, maxForce: 0.02)
        step(s, q, frames: 48) { _ in s.wake(c.bodies) }             // an active command keeps the rig awake (N3)
        brakes(s, c)
        var r = SwayRun()
        var xs: [Float] = [sway()]
        var gpu: [Double] = []
        let frames = Int(idle * 60)
        gpu = step(s, q, frames: frames) { f in
            xs.append(sway())
            if r.sleptAfter < 0, c.bodies.allSatisfy({ s.isAsleep($0) }) { r.sleptAfter = Float(f + 1) / 60 }
        }
        // Turning points: sign changes of the sway's frame-to-frame slope.
        var peaks: [Float] = []
        for i in 1..<(xs.count - 1) {
            let a = xs[i] - xs[i - 1], b = xs[i + 1] - xs[i]
            if a * b < 0 { peaks.append(abs(xs[i])) }
        }
        r.x0 = peaks.first ?? abs(xs[0])
        r.peaks = peaks
        // Mean loss per half cycle over the swinging part (peaks well outside the stick zone).
        let swinging = peaks.prefix { $0 > 1.5 * max(arm, 1e-4) }
        if swinging.count >= 2 { r.decrementPerHalf = (swinging.first! - swinging.last!) / Float(swinging.count - 1) }
        r.restOffset = abs(xs.last!)
        r.awakeAtEnd = c.bodies.filter { !s.isAsleep($0) }.count
        r.fallErrMM = c.falls.map { abs(s.distanceJointLength($0)! - c.cable) * 1000 }.max()!
        r.idleGPU = gpu.suffix(60).sorted()[30]
        return r
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VZ-0168 — the hook block's swing damps by sheave friction, and the crane sleeps.
    // ══════════════════════════════════════════════════════════════════════════

    /// Toy nylon sheaves on 1.5 mm steel pins (c = 2·0.25·0.75 mm = 0.375 mm per fall): after
    /// the move the swing loses ≈ 2c per half cycle (Coulomb: linear, not exponential), ends
    /// within c of plumb, and the whole crane is asleep a few seconds after the brake — a real
    /// toy crane's hook settles in seconds. With c = 0 the falls are the frictionless rods they
    /// were: the swing keeps > 80 % of its amplitude for 10 s and the crane never sleeps.
    func testFallSheaveFrictionDampsTheSwingAndTheCraneSleeps() throws {
        let arm = CoinDEMSolver.toyCraneFallFrictionArm
        XCTAssertEqual(arm, 0.000375, accuracy: 1e-9)
        let damped = try swayAfterMove(arm: arm)
        let free = try swayAfterMove(arm: 0)
        func mm(_ a: [Float]) -> String { a.prefix(12).map { String(format: "%.2f", $0 * 1000) }.joined(separator: " ") }
        print(String(format: "SHEAVE_1 c %.3f mm: sway at brake %.2f mm, peaks (mm) %@ | loss per half cycle %.3f mm (2c = %.3f mm) | rest offset %.3f mm (≤ c) | asleep %.2f s after the brake, awake at end %d/6, falls err %.4f mm, idle GPU p50 %.3f ms",
                     arm * 1000, damped.x0 * 1000, mm(damped.peaks), damped.decrementPerHalf * 1000, 2 * arm * 1000,
                     damped.restOffset * 1000, damped.sleptAfter, damped.awakeAtEnd, damped.fallErrMM, damped.idleGPU))
        print(String(format: "SHEAVE_1 c 0: sway at brake %.2f mm, peaks (mm) %@ … last %.2f mm | asleep %.2f s, awake at end %d/6, idle GPU p50 %.3f ms",
                     free.x0 * 1000, mm(free.peaks), (free.peaks.last ?? 0) * 1000, free.sleptAfter, free.awakeAtEnd, free.idleGPU))
        XCTAssertGreaterThan(damped.x0, 4 * arm, "the move left a real swing to damp (control)")
        XCTAssertEqual(damped.decrementPerHalf, 2 * arm, accuracy: 0.35 * 2 * arm, "Coulomb loss ≈ 2c per half cycle")
        XCTAssertLessThanOrEqual(damped.restOffset, arm * 1.1 + 2e-5, "comes to rest within c of plumb")
        XCTAssertGreaterThan(damped.sleptAfter, 0, "the crane sleeps")
        XCTAssertLessThan(damped.sleptAfter, 5, "…within a few seconds of the brake, like a real toy crane")
        XCTAssertEqual(damped.awakeAtEnd, 0)
        XCTAssertLessThan(damped.fallErrMM, 0.05, "the falls hold their length")
        XCTAssertGreaterThan(free.x0, 4 * arm, "control: the same move swings the frictionless crane")
        XCTAssertGreaterThan(free.peaks.last ?? 0, 0.8 * free.x0, "c = 0: no damping beyond the global 0.99995/substep")
        XCTAssertLessThan(free.sleptAfter, 0, "c = 0: the crane never sleeps (VZ-0168 as filed)")
    }

    /// The price of the swing-friction rows while the crane WORKS (awake, trolley shuttling):
    /// two identical cranes, one with toy-crane sheave friction on its 4 falls and one without,
    /// stepped in interleaved blocks; the difference per substep is the rows' cost (a 2×2
    /// effective mass per fall in the prepare, a disc-clamped 2-D row per fall per iteration).
    func testSheaveFrictionRowCost() throws {
        func crane(arm: Float) throws -> (CoinDEMSolver, MTLCommandQueue, Crane) {
            let (s, q) = try makeSolver()
            s.sleepEnabled = false
            let c = try buildCrane(s)
            for f in c.falls { s.setDistanceSwingFriction(f, arm: arm) }
            step(s, q, frames: 30)
            return (s, q, c)
        }
        let worlds = [try crane(arm: CoinDEMSolver.toyCraneFallFrictionArm), try crane(arm: 0)]
        var ms: [[Double]] = [[], []]
        for block in 0..<8 {
            let v: Float = block % 2 == 0 ? 0.04 : -0.04                   // shuttle the trolley
            for (i, w) in worlds.enumerated() {
                w.0.setPrismaticMotor(w.2.j1, targetVelocity: v, maxForce: 0.02)
                ms[i] += step(w.0, w.1, frames: 30)
            }
        }
        func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
        let substeps = Double(worlds[0].0.lastStepCount)
        let perSub = (median(ms[0]) - median(ms[1])) / substeps * 1000
        print(String(format: "SHEAVE_2 working crane, frame p50: 4 falls with sheave friction %.3f ms, frictionless %.3f ms → %+.1f µs per substep (%d substeps/frame)",
                     median(ms[0]), median(ms[1]), perSub, Int(substeps)))
        XCTAssertLessThan(perSub, 100, "four falls' friction rows cost well under 0.1 ms per substep")
    }
}
