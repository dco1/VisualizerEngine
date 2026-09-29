import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Stage B3 (engine plan 3b / 3c). Two ways of running FEWER dispatches for the same
/// simulation, asserted bit for bit against the one-step kernels they replace:
///
///  • 3b — the multi-dispatch substep's bookkeeping kernels fused (Shaders/CoinDEMFusedKernels.h),
///    always on: every world, however large, steps bit-identically to the one-step kernels
///    (`fusedSubstepKernelsDisabledForTesting`).
///  • 3c — the OPT-IN small-world path (`smallWorldPath`, Shaders/CoinDEMSmallWorld.h): a whole
///    frame of a small world as ONE single-threadgroup dispatch. The clock world (hull bars on
///    welds, the crane chain with motors / limits / sheave-friction falls / a weld grip, torsion-
///    patched Weebles) and the spring-feet world step bit-identically on both paths, frame by
///    frame; a world that outgrows the limits falls back and comes back with no seam.
///
/// The whole CoinDEM suite also runs with the path forced on: `VIZ_COINDEM_SMALLWORLD=1` (the
/// default limits) or `=all` (limits lifted — every constraint-path world through the kernel).
@MainActor
final class CoinDEMSmallWorldTests: XCTestCase {
    typealias A = CoinDEMActuationTests

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader), queue)
        cached = c
        return c
    }

    // ── Harness ──────────────────────────────────────────────────────────────────

    /// One frame; returns the GPU time (ms).
    @discardableResult
    private func frame(_ s: CoinDEMSolver, wall: Float = 1.0 / 30) throws -> Double {
        let (_, _, q) = try Self.shared()
        let cb = try XCTUnwrap(q.makeCommandBuffer())
        s.encode(to: cb, wallDt: wall)
        cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        XCTAssertNotEqual(cb.status, .error, "command buffer error: \(String(describing: cb.error))")
        return max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000
    }

    private static func bodies(_ s: CoinDEMSolver) -> [CoinBody] {
        let p = s.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: s.maxCoins)
        return (0..<s.highWater).map { p[$0] }
    }

    private static func bits(_ v: SIMD4<Float>) -> SIMD4<UInt32> {
        SIMD4(v.x.bitPattern, v.y.bitPattern, v.z.bitPattern, v.w.bitPattern)
    }

    /// Every lane of the dynamic state, bit for bit (a -0 / +0 or NaN difference counts).
    private static func sameBits(_ a: CoinBody, _ b: CoinBody) -> Bool {
        bits(a.posInvMass) == bits(b.posInvMass) && bits(a.orient) == bits(b.orient)
            && bits(a.vel) == bits(b.vel) && bits(a.angVel) == bits(b.angVel)
            && bits(a.prevPos) == bits(b.prevPos) && bits(a.prevOrient) == bits(b.prevOrient)
    }

    /// The live contacts as bit patterns in a canonical order: the two paths append in different
    /// orders (the small-world kernel pairs over a one-cell grid), and nothing downstream depends
    /// on the order — so the SET, with every field, must match.
    private static func contactBits(_ s: CoinDEMSolver) -> [[UInt32]] {
        let n = s.contactCount
        let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(n, 1))
        return (0..<n).map { i -> [UInt32] in
            let c = p[i]
            return [c.meta, bits(c.nrm), bits(c.rA), bits(c.rB), bits(c.tan1), bits(c.tan2), bits(c.aux), bits(c.ext)]
                .flatMap { [$0.x, $0.y, $0.z, $0.w] }
        }.sorted { $0.lexicographicallyPrecedes($1) }
    }

    typealias World = (solver: CoinDEMSolver, stepper: (Int) -> Void)

    /// Step `a` and `b` side by side (the same host script) and require every body bit-identical
    /// after every frame, and the contact sets identical field for field at every `contactEvery`th.
    private func lockstep(_ name: String, _ a: World, _ b: World, frames: Int, wall: Float = 1.0 / 30,
                          contactEvery: Int = 10, afterFrame: ((Int) -> Void)? = nil) throws {
        var compared = 0
        for f in 0..<frames {
            a.stepper(f); b.stepper(f)
            try frame(a.solver, wall: wall); try frame(b.solver, wall: wall)
            afterFrame?(f)
            XCTAssertEqual(a.solver.lastStepCount, b.solver.lastStepCount)
            let sa = Self.bodies(a.solver), sb = Self.bodies(b.solver)
            XCTAssertEqual(sa.count, sb.count)
            if let i = (0..<min(sa.count, sb.count)).first(where: { !Self.sameBits(sa[$0], sb[$0]) }) {
                XCTFail("\(name): body \(i) differs after frame \(f): x \(sa[i].posInvMass) / \(sb[i].posInvMass), v \(sa[i].vel) / \(sb[i].vel)")
                return
            }
            if f % contactEvery == contactEvery - 1 || f == frames - 1 {
                let ca = Self.contactBits(a.solver), cb = Self.contactBits(b.solver)
                if ca != cb {
                    XCTFail("\(name): contact sets differ after frame \(f) (\(ca.count) vs \(cb.count))")
                    return
                }
                compared += ca.count
            }
            XCTAssertEqual(a.solver.asleepCount, b.solver.asleepCount, "\(name) frame \(f)")
        }
        XCTAssertGreaterThan(compared, 0, "\(name): the comparison saw contacts")
    }

    // ── Worlds ───────────────────────────────────────────────────────────────────

    static func clockConfig(_ s: CoinDEMSolver, dt: Float = 1.0 / 180, iterations: Int = 6) {
        s.solverMode = .constraint
        s.gravity = 9.81; s.fixedDt = dt; s.maxSubsteps = 10; s.velocityIterations = iterations
        s.colorRounds = 16; s.islandUnionRounds = 6; s.warmStart = false
        s.linDamping = 0.99995; s.angDamping = 0.9998; s.frictionCoeff = 0.5
        s.rollingResistance = 0.024 / Float(iterations); s.restitution = 0.15; s.restThreshold = 0.14
        s.contactSlop = 1e-4; s.baumgarteBeta = 0.2; s.speculativeMargin = 2e-4
        s.maxSpeed = 3; s.maxHSpeed = 3; s.maxOmega = 60; s.floorY = -0.5
        s.sleepEnabled = true; s.sleepFrames = 20; s.sleepLinVel = 2e-4
    }

    /// The clock world — a compact copy of CoinDEMActuationTests.runClockFence (the §A2 statics
    /// with the §A3 zone culling, 21 latched bars on welds, 16 stocked bars in the cradles and the
    /// rack, 4 hopping Weebles, the crane chain working all axes then parking). `b2` turns on the
    /// stage-B1/B2 opt-ins the scene runs (manifold solve, warm start, speculative colouring, the
    /// workers' torsion patch, sheave friction on the falls).
    func clockWorld(b2: Bool, frames: Int) throws -> World {
        let (engine, lib, _) = try Self.shared()
        let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: 48, coinRadius: 0.27, halfThickness: 0.27,
                                            boundsMin: SIMD3(-0.30, 0.55, 0.00), boundsMax: SIMD3(0.30, 0.90, 0.46)))
        Self.clockConfig(s)
        if b2 { s.manifoldSolve = true; s.warmStart = true; s.coloringScheme = .speculative }
        let statics = A.clockStatics()
        s.setColliders(statics.all)
        let wall: Float = 1.0 / 30
        let hull = try XCTUnwrap(s.registerHull(vertices: A.barHullPoints()))
        func bar(_ rd: simd_quatf, _ td: SIMD3<Float>) -> Int? {
            let qq = rd * hull.principalRotation
            return s.spawnHull(at: td + simd_act(rd, hull.comOffset), hull: hull, orient: SIMD4(qq.imag, qq.real),
                               mass: A.barMass, friction: 0.3, restitution: 0.2)
        }
        var latched: [Int] = []
        let reading: [[Int]] = [[1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]
        for (dIdx, segs) in reading.enumerated() {
            for seg in segs {
                let (dx, dy, vertical) = A.sockets[seg]
                let c = SIMD3(A.digitX[dIdx] + dx / 1000, A.rowD + dy / 1000, A.seatZ)
                latched.append(try XCTUnwrap(bar(vertical ? A.barVertical : A.barHorizontal, c)))
            }
        }
        for x in [Float(-0.2325), -0.2205, 0.081, 0.105] { _ = try XCTUnwrap(bar(A.barVertical, SIMD3(x, 0.6405 + 0.0001, A.seatZ))) }
        for k in 0..<12 { _ = try XCTUnwrap(bar(A.barFacingX, SIMD3(-0.24575, 0.6405 + 0.0001, 0.235 + 0.01 * Float(k)))) }
        var workers: [Int] = []
        for x in [Float(-0.205), -0.175, -0.145, -0.115] {
            workers.append(try XCTUnwrap(s.spawnBallastedEgg(A.worker, fatCenter: SIMD3(x, A.tableTop + 0.0115, 0.30),
                                                             friction: 0.6, restitution: 0.15)))
        }
        if b2 { for w in workers { s.setPatchRadius(w, 0.003) } }
        let xt: Float = 0.10, cable: Float = 0.09
        let blockTop = 0.8175 - cable
        let railY = blockTop - 0.014 - 0.0005 - 0.0025
        let heldBar = try XCTUnwrap(bar(A.barHorizontal, SIMD3(xt, railY, 0.1967 - 0.00275)))
        let jib = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.8315, 0.205), halfExtents: SIMD3(0.26, 0.006, 0.007), mass: 0.03, friction: 0.3, restitution: 0.1))
        let trolley = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, 0.8215, 0.205), halfExtents: SIMD3(0.01, 0.004, 0.008), mass: 0.006, friction: 0.3, restitution: 0.1))
        let block = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, blockTop - 0.007, 0.205), halfExtents: SIMD3(0.011, 0.007, 0.005), mass: 0.012, friction: 0.3, restitution: 0.1))
        let rail = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, railY, 0.210), halfExtents: SIMD3(0.0025, 0.0025, 0.010), mass: 0.002, friction: 0.3, restitution: 0.1))
        let puck = try XCTUnwrap(s.spawnBox(at: SIMD3(xt, railY, 0.1982), halfExtents: SIMD3(0.0025, 0.0025, 0.0015), mass: 0.0015, friction: 0.5, restitution: 0.05))
        let crane = [jib, trolley, block, rail, puck, heldBar]
        let rp = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: jib))
        let j0 = try XCTUnwrap(rp.takeSlot()); s.enableHinge(slot: j0, bodyA: jib, bodyB: nil, worldAnchor: SIMD3(0.17, 0.8255, 0.205), worldAxis: SIMD3(0, 1, 0))
        let j1 = try XCTUnwrap(rp.takeSlot()); s.enablePrismatic(slot: j1, bodyA: trolley, bodyB: jib, worldAnchor: SIMD3(xt, 0.8215, 0.205), worldAxis: SIMD3(1, 0, 0))
        var falls: [Int] = []
        for (dx, dz) in [(Float(-0.006), Float(-0.005)), (0.006, -0.005), (-0.006, 0.005), (0.006, 0.005)] {
            falls.append(try XCTUnwrap(s.addDistanceJoint(bodyA: trolley, bodyB: block,
                                                          worldAnchorA: SIMD3(xt + dx, 0.8175, 0.205 + dz),
                                                          worldAnchorB: SIMD3(xt + dx, blockTop, 0.205 + dz))))
        }
        if b2 { for f in falls { s.setDistanceSwingFriction(f, arm: CoinDEMSolver.toyCraneFallFrictionArm) } }
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 74, placeholderBody: latched[0]))
        let j6 = try XCTUnwrap(pool.takeSlot()); s.enablePrismatic(slot: j6, bodyA: rail, bodyB: block, worldAnchor: SIMD3(xt, railY, 0.210), worldAxis: SIMD3(0, 0, -1))
        s.setJointLimits(j6, 0...0.012)
        let j7 = try XCTUnwrap(pool.takeSlot()); s.enableHinge(slot: j7, bodyA: rail, bodyB: puck, worldAnchor: SIMD3(xt, railY, 0.1982), worldAxis: SIMD3(0, 0, 1))
        s.setJointLimits(j7, A.deg(-5)...A.deg(95))
        s.enableWeld(try XCTUnwrap(pool.takeWeld()), bodyA: puck, bodyB: heldBar, worldAnchor: SIMD3(xt, railY, 0.1967))
        for b in latched { s.enableWeld(try XCTUnwrap(pool.takeWeld()), bodyA: b, bodyB: nil, worldAnchor: s.position(of: b)!) }
        func brakes() {
            s.setHingeMotor(j0, targetVelocity: 0, maxTorque: 0.01)
            s.setPrismaticMotor(j1, targetVelocity: 0, maxForce: 0.02)
            s.setPrismaticMotor(j6, targetVelocity: 0, maxForce: 0.003)
            s.setHingeMotor(j7, targetVelocity: 0, maxTorque: 2e-4)
        }
        brakes()
        var states: [CoinBodyState] = []
        let stepper: (Int) -> Void = { f in
            let t = Float(f) * wall
            if f >= 40 && f < frames - 60 {          // settle, work, then park
                s.wake(crane)
                s.setHingeMotor(j0, targetVelocity: 0.03 * sin(2 * .pi * t / 6), maxTorque: 0.01)
                s.setPrismaticMotor(j1, targetVelocity: -0.03 * sin(2 * .pi * t / 4), maxForce: 0.02)
                let L = cable - 0.015 * (1 - cos(2 * .pi * t / 5))
                for r in falls { s.setDistanceRestLength(r, L) }
                s.setPrismaticMotor(j6, targetVelocity: 0.012 * sin(2 * .pi * t / 2.5), maxForce: 0.003)
                s.setHingeMotor(j7, targetVelocity: 1.2 * sin(2 * .pi * t / 3), maxTorque: 2e-4)
                if f % 20 == 0 {
                    s.readStates(workers, into: &states)
                    for (i, w) in workers.enumerated() where states[i].velocity.y > -0.05 && abs(states[i].velocity.y) < 0.02 {
                        let dir: Float = (f / 20 + i) % 2 == 0 ? 1 : -1
                        s.setRigidVelocity(slots: [w], linear: SIMD3(0, (2 * 9.81 * 0.012).squareRoot(), 0.05 * dir),
                                           angular: SIMD3(0, 2, 0), about: states[i].position)
                    }
                }
            } else if f == frames - 60 {
                brakes()
            }
            s.setColliders(A.culledColliders(s, statics, frameTime: wall))
        }
        return (s, stepper)
    }

    /// Spring feet: 4 footed, patched workers — two hop on pulses (canted), one turns planted on
    /// the table, one stands on a dynamic plate and turns (so its foot iterates every velocity
    /// iteration); speculative colouring, manifold solve and warm start on.
    func feetWorld() throws -> World {
        let (engine, lib, _) = try Self.shared()
        let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: 16, coinRadius: 0.05, halfThickness: 0.05,
                                            boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.4, 0.4)))
        Self.clockConfig(s)
        s.scaleAwareDeadStop = true
        s.manifoldSolve = true; s.warmStart = true; s.coloringScheme = .speculative
        s.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.1),
                        .box(center: SIMD3(0, -0.011, 0), halfExtents: SIMD3(0.3, 0.011, 0.3), friction: 0.5, restitution: 0.1)])
        let props = CoinDEMSolver.ballastedEggProperties(A.worker)
        let hc = A.worker.fatRadius - props.comBelowFatCenter
        _ = try XCTUnwrap(s.spawnBox(at: SIMD3(0.12, 0.003, 0.0), halfExtents: SIMD3(0.03, 0.003, 0.03), mass: 0.05,
                                     friction: 0.5, restitution: 0.1))
        var workers: [Int] = []
        for (i, x) in [Float(-0.12), -0.04, 0.04, 0.12].enumerated() {
            let y: Float = (i == 3) ? 0.006 : 0
            let w = try XCTUnwrap(s.spawnBallastedEgg(A.worker, fatCenter: SIMD3(x, y + A.worker.fatRadius, 0),
                                                      friction: 0.6, restitution: 0.15))
            XCTAssertTrue(s.addFoot(body: w, spec: .weebleWorker(hipHeight: hc)))
            s.setPatchRadius(w, 0.002)
            workers.append(w)
        }
        let stepper: (Int) -> Void = { f in
            if f == 30 || f == 90 {
                for (k, w) in workers.prefix(2).enumerated() {
                    let y0 = s.position(of: w)!.y
                    let cant: Float = (k == 0 ? 10 : -14) * .pi / 180
                    s.setFoot(body: w, rest: .pulse, extendedLength: CoinFootSpec.legLength(hipHeight: y0, cant: cant) + 0.004,
                              cant: cant, yaw: Float(k) * 0.7)
                }
            }
            if f == 40 {
                for w in workers.suffix(2) {
                    s.setFoot(body: w, rest: .extended, extendedLength: s.position(of: w)!.y - (w == workers[3] ? 0.006 : 0) + 0.00015,
                              cant: 0, yaw: 0, yawRate: 2.0)
                }
            }
            if f == 100 { for w in workers.suffix(2) { s.setFoot(body: w, yawRate: 0) } }
            if f == 140 { for w in workers.suffix(2) { s.setFoot(body: w, rest: .retracted) } }
        }
        return (s, stepper)
    }

    /// A pile poured into a walled bin over an oriented slab and a half-cylinder: `kind` picks
    /// the bodies and the solver features (see the call sites). Deterministic spawn script.
    func pileWorld(_ kind: String, bodies: Int = 120, boundsScale: Float = 1) throws -> World {
        let (engine, lib, _) = try Self.shared()
        let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: max(160, bodies), coinRadius: 0.0347, halfThickness: 0.0347,
                                            boundsMin: SIMD3(-0.5, -0.1, -0.5) * boundsScale, boundsMax: SIMD3(0.5, 1.2, 0.5) * boundsScale))
        Self.clockConfig(s, dt: 1.0 / 120, iterations: 4)
        s.frictionCoeff = 0.42; s.rollingResistance = 0.02; s.restitution = 0.22; s.linDamping = 0.999
        s.maxSpeed = 9; s.maxHSpeed = 4; s.maxOmega = 18; s.maxSubsteps = 4
        s.quadraticDrag = 0.18; s.dragRefRadius = 0.02418; s.speculativeMargin = 0.03
        s.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: 0),
            .box(center: SIMD3(0.3, 0.1, 0), halfExtents: SIMD3(0.02, 0.1, 0.3)),
            .box(center: SIMD3(-0.3, 0.1, 0), halfExtents: SIMD3(0.02, 0.1, 0.3)),
            .orientedBox(center: SIMD3(0, 0.05, 0.25), halfExtents: SIMD3(0.3, 0.05, 0.02), orientation: simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0))),
            .orientedBox(center: SIMD3(0.05, 0.02, -0.1), halfExtents: SIMD3(0.08, 0.02, 0.05),
                         orientation: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(1, 0.2, 0)))),
            .cylinder(center: SIMD3(0, 0.03, -0.2), axis: SIMD3(1, 0, 0), radius: 0.03, up: SIMD3(0, 1, 0), halfLength: 0.2, lowerHalfOnly: true)])
        switch kind {
        case "mixedWarm":            // spheres, capsules, discs, eggs; warm start; islands sleep
            s.warmStart = true
        case "boxManifold":          // boxes + spheres; manifold solve + warm start; sleep off
            s.warmStart = true; s.manifoldSolve = true; s.sleepEnabled = false
        case "starved":              // too few colouring rounds: an uncoloured bucket every substep
            s.colorRounds = 3; s.sleepEnabled = false
        default:                     // eggs, the JP colouring, sleep on
            break
        }
        var seed: UInt64 = 0xB1_9A41
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
        func rq() -> SIMD4<Float> {
            let q = simd_quatf(angle: rnd() * 6.283, axis: simd_normalize(SIMD3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) + 1e-3))
            return SIMD4(q.imag, q.real)
        }
        var spawned = 0
        let stepper: (Int) -> Void = { _ in
            for _ in 0..<2 where spawned < bodies {
                spawned += 1
                let p = SIMD3((rnd() - 0.5) * 0.5, 0.6 + rnd() * 0.3, (rnd() - 0.5) * 0.4)
                switch kind {
                case "eggs":
                    _ = s.spawnEgg(at: p, fatRadius: 0.02418, tipRadius: 0.01519, centerDistance: 0.02263, orient: rq(), mass: 0.06)
                case "boxManifold":
                    if Int(rnd() * 2) == 0 { _ = s.spawnBox(at: p, halfExtents: SIMD3(0.012, 0.008, 0.015), orient: rq(), mass: 0.02) }
                    else { _ = s.spawnSphere(at: p, radius: 0.01 + rnd() * 0.01, mass: 0.02) }
                default:
                    switch Int(rnd() * 4) {
                    case 0: _ = s.spawnSphere(at: p, radius: 0.01 + rnd() * 0.01, mass: 0.02)
                    case 1: _ = s.spawnCapsule(at: p, radius: 0.008, halfLength: 0.015, orient: rq(), mass: 0.02)
                    case 2: _ = s.spawnBox(at: p, halfExtents: SIMD3(0.012, 0.008, 0.015), orient: rq(), mass: 0.02)
                    default: _ = s.spawnEgg(at: p, fatRadius: 0.02418, tipRadius: 0.01519, centerDistance: 0.02263, orient: rq(), mass: 0.06)
                    }
                }
            }
        }
        return (s, stepper)
    }

    /// Every kind of pair the polytope narrowphase's separating-axis test sees: compound tables
    /// and L-brackets (child × box, × hull, × compound, × a DISC's prism), hull bars (hull ×
    /// hull, × box, × static box), boxes on a slatted floor of static boxes (box × static box),
    /// plus capsules and eggs (the swept pairs, one thread each) — tumbled onto each other with
    /// the manifold solve + warm start (the clock's configuration) or without.
    func polytopeMixWorld(manifold: Bool) throws -> World {
        let (engine, lib, _) = try Self.shared()
        let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: 48, coinRadius: 0.06, halfThickness: 0.06,
                                            boundsMin: SIMD3(-0.3, -0.05, -0.3), boundsMax: SIMD3(0.3, 0.5, 0.3)))
        Self.clockConfig(s)
        s.sleepEnabled = false
        if manifold { s.manifoldSolve = true; s.warmStart = true; s.coloringScheme = .speculative }
        var cols: [CoinStaticCollider] = [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.1)]
        var x: Float = -0.12
        while x <= 0.12 {
            cols.append(.box(center: SIMD3(x, 0.004, 0), halfExtents: SIMD3(0.004, 0.004, 0.12), friction: 0.5, restitution: 0.1))
            x += 0.02
        }
        cols.append(.orientedBox(center: SIMD3(0, 0.03, 0.1), halfExtents: SIMD3(0.1, 0.004, 0.03),
                                 orientation: simd_quatf(angle: 0.25, axis: SIMD3(1, 0, 0)), friction: 0.5, restitution: 0.1))
        s.setColliders(cols)
        let table = try XCTUnwrap(s.registerCompound(boxes: CoinDEMCompoundTests.table))
        let ell = try XCTUnwrap(s.registerCompound(boxes: CoinDEMCompoundTests.ell))
        let bar = try XCTUnwrap(s.registerHull(vertices: A.barHullPoints()))
        var seed: UInt64 = 0xB3_0198
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
        func rq() -> simd_quatf { simd_quatf(angle: rnd() * 6.283, axis: simd_normalize(SIMD3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) + 1e-3)) }
        var spawned = 0
        let stepper: (Int) -> Void = { f in
            guard f % 4 == 0, spawned < 40 else { return }
            let p = SIMD3((rnd() - 0.5) * 0.16, 0.08 + rnd() * 0.08, (rnd() - 0.5) * 0.12)
            let q = rq()
            switch spawned % 7 {
            case 0: _ = s.spawnCompound(design: q, p, compound: table, mass: 0.02)
            case 1: _ = s.spawnCompound(design: q, p, compound: ell, mass: 0.004)
            case 2: _ = s.spawn(at: p, orient: SIMD4(q.imag, q.real), radius: 0.008, halfThickness: 0.002, mass: 0.002)
            case 3:
                let qq = q * bar.principalRotation
                _ = s.spawnHull(at: p + simd_act(q, bar.comOffset), hull: bar, orient: SIMD4(qq.imag, qq.real), mass: A.barMass,
                                friction: 0.3, restitution: 0.2)
            case 4: _ = s.spawnBox(at: p, halfExtents: SIMD3(0.006, 0.004, 0.008), orient: SIMD4(q.imag, q.real), mass: 0.003)
            case 5: _ = s.spawnCapsule(at: p, radius: 0.003, halfLength: 0.008, orient: SIMD4(q.imag, q.real), mass: 0.002)
            default: _ = s.spawnBallastedEgg(A.worker, fatCenter: p, orient: q, friction: 0.6, restitution: 0.15)
            }
            spawned += 1
        }
        return (s, stepper)
    }

    // ── VZ-0198: the polytope SAT on a SIMD group ────────────────────────────────

    /// The polytope narrowphase runs its separating-axis test on a whole SIMD group
    /// (cdPolySATSG: the lanes share the face axes and the Minkowski edge pairs) — the SAME answer
    /// as the one-thread SAT (cdPolySAT, still run by `polySATSerialForTesting`), so every world
    /// steps bit-identically either way, contacts field for field: the clock (hull bars in their
    /// sockets / cradles / rack, the crane's boxes, both configurations) and a tumbling mix of
    /// compounds, discs, hull bars and boxes on static boxes — on the multi-dispatch path and in
    /// the small-world kernel.
    func testPolytopeSATOnASIMDGroupMatchesOneThread() throws {
        let worlds: [(String, () throws -> World)] = [
            ("clock default", { try self.clockWorld(b2: false, frames: 200) }),
            ("clock B2", { try self.clockWorld(b2: true, frames: 200) }),
            ("polytope mix", { try self.polytopeMixWorld(manifold: false) }),
            ("polytope mix, manifold", { try self.polytopeMixWorld(manifold: true) }),
        ]
        for (name, make) in worlds {
            for small in [false, true] {
                let a = try make(), b = try make()
                a.solver.polySATSerialForTesting = true
                a.solver.smallWorldPath = small; b.solver.smallWorldPath = small
                try lockstep("\(name) (\(small ? "small world" : "multi-dispatch"))", a, b, frames: 200) { _ in
                    if small && b.solver.lastStepCount > 0 { XCTAssertTrue(b.solver.lastFrameUsedSmallWorld) }
                }
                XCTAssertGreaterThan(b.solver.polyPairCount, 0, "\(name): polytope pairs were listed")
            }
        }
    }

    /// Every polytope pair kind through both paths: compound tables and L-brackets (child × box, ×
    /// hull, × compound, × a disc's prism), hull bars, boxes on static boxes, capsules and eggs (the
    /// one-thread swept pairs) — the small-world kernel's two-pass narrowphase (first face per thread,
    /// survivors per SIMD group) against coinPolyNarrow's SIMD group per pair, bit for bit.
    func testSmallWorldStepsThePolytopeMixBitForBit() throws {
        for manifold in [false, true] {
            let a = try polytopeMixWorld(manifold: manifold), b = try polytopeMixWorld(manifold: manifold)
            a.solver.smallWorldPath = false; b.solver.smallWorldPath = true
            try lockstep("polytope mix (manifold \(manifold))", a, b, frames: 200) { _ in
                if b.solver.lastStepCount > 0 { XCTAssertTrue(b.solver.lastFrameUsedSmallWorld) }
            }
        }
    }

    /// The opt-in prepared contact rows (`preparedContactSolve`, Shaders/CoinDEMPreparedSolve.h) run
    /// on both paths — the small-world kernel's prepare phase and prepared tail, the multi-dispatch
    /// path's prepare / per-colour / tail kernels — and the two step bit-identically: the clock with
    /// every B2 opt-in, the tumbling polytope mix (grouped manifolds, torsion-free) and spring feet
    /// (torsion rows, pads).
    func testPreparedContactRowsAreBitIdenticalOnBothPaths() throws {
        let worlds: [(String, () throws -> World)] = [
            ("clock B2", { try self.clockWorld(b2: true, frames: 200) }),
            ("polytope mix, manifold", { try self.polytopeMixWorld(manifold: true) }),
            ("spring feet", { try self.feetWorld() }),
        ]
        for (name, make) in worlds {
            let a = try make(), b = try make()
            a.solver.preparedContactSolve = true; b.solver.preparedContactSolve = true
            a.solver.smallWorldPath = false; b.solver.smallWorldPath = true
            try lockstep("prepared rows: \(name)", a, b, frames: 200) { _ in
                if b.solver.lastStepCount > 0 { XCTAssertTrue(b.solver.lastFrameUsedSmallWorld) }
            }
            XCTAssertTrue(a.solver.preparedContactsLive && b.solver.preparedContactsLive, "\(name): the prepared rows ran")
        }
    }

    // ── 3c: the small-world path ─────────────────────────────────────────────────

    /// The params block the host writes is the one the kernel reads (Shaders/CoinDEMSmallWorld.h
    /// static-asserts the Metal side at 224 bytes).
    func testParamsLayoutMatchesTheKernel() {
        XCTAssertEqual(MemoryLayout<CoinSmallWorldParamsGPU>.size, 224)
        XCTAssertEqual(MemoryLayout<CoinSmallWorldParamsGPU>.stride, 224)
        XCTAssertEqual(MemoryLayout<CoinFootUniformsGPU>.size, 32)
        XCTAssertEqual(MemoryLayout<CoinSmallWorldParamsGPU>.offset(of: \.footLater), 160)
        XCTAssertEqual(MemoryLayout<CoinSmallWorldParamsGPU>.offset(of: \.footLast), 192)
    }

    /// The optional stage-B3 pipelines are compiled only for the path a solver opts into — their cold
    /// backend compiles are main-actor stalls (coinSmallWorldFrame ≈ 6 s, each prepared-row kernel
    /// ≈ 53–97 ms on an M1 Max). The small-world dispatch used to resolve coinPrepareContacts just to
    /// decide whether to bind buffer 27, so every small-world solver compiled it (verifier, stage B3).
    func testOptionalPipelinesResolveOnlyForTheirOptIn() throws {
        let names = ["coinSmallWorldFrame", "coinPrepareContacts", "coinSolveVelocityColorPrep", "coinSolveVelocityTailPrep"]
        func run(small: Bool, prepared: Bool) throws -> [Bool] {
            let w = try pileWorld("eggs")
            w.solver.smallWorldPath = small; w.solver.preparedContactSolve = prepared   // (whatever the env says)
            for f in 0..<20 { w.stepper(f); _ = try frame(w.solver) }
            XCTAssertEqual(w.solver.lastFrameUsedSmallWorld, small)
            XCTAssertGreaterThan(w.solver.contactCount, 0, "precondition: contacts were solved")
            return names.map { w.solver.optionalPipelineResolvedForTesting($0) }
        }
        XCTAssertEqual(try run(small: false, prepared: false), [false, false, false, false], "no opt-in: nothing compiled")
        XCTAssertEqual(try run(small: true, prepared: false), [true, false, false, false], "small world: its own kernel only")
        XCTAssertEqual(try run(small: false, prepared: true), [false, true, true, true], "prepared rows: their three kernels")
        XCTAssertEqual(try run(small: true, prepared: true), [true, true, true, true])
    }

    /// Threadgroup memory: the small-world kernel fits the 32 KB every A11-or-later GPU has (the
    /// joint solve's threadgroup cache already required that, so it excludes no further device),
    /// and the 3b fused kernels fit the 16 KB of the oldest Apple GPUs.
    func testThreadgroupMemoryFitsTheAppleGPUs() throws {
        let (engine, lib, _) = try Self.shared()
        let device = engine.device
        func tgMemory(_ name: String) throws -> (Int, Int) {
            let fn = try XCTUnwrap(lib.makeFunction(name: name), name)
            let pso = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness inspects the pipeline
            return (pso.staticThreadgroupMemoryLength, pso.maxTotalThreadsPerThreadgroup)
        }
        let (sw, swThreads) = try tgMemory("coinSmallWorldFrame")
        let (joint, _) = try tgMemory("coinJointSolveCS")
        print("B3 threadgroup memory: coinSmallWorldFrame \(sw) B (max \(swThreads) threads), coinJointSolveCS \(joint) B; device limit \(device.maxThreadgroupMemoryLength) B")
        XCTAssertLessThanOrEqual(sw, 32 * 1024, "small-world kernel fits A11+ threadgroup memory")
        XCTAssertLessThanOrEqual(joint, 32 * 1024)
        XCTAssertGreaterThanOrEqual(swThreads, 256, "the kernel's default 256-thread group is launchable")
        for name in ["coinSubstepBegin", "coinBroadphaseTG", "coinColorPrepareTG", "coinColorFinishTG", "coinIntegratePositionFinalize"] {
            let (m, _) = try tgMemory(name)
            print("B3 threadgroup memory: \(name) \(m) B")
            XCTAssertLessThanOrEqual(m, 16 * 1024, "\(name) fits every Apple GPU's threadgroup memory")
        }
    }

    /// The clock world, default engine config: the small-world kernel steps it bit-identically
    /// to the multi-dispatch path, frame by frame — contacts field for field — while the crane
    /// works, the Weebles hop and the parked world goes to sleep.
    func testSmallWorldStepsTheClockBitForBit() throws {
        let frames = 240
        let a = try clockWorld(b2: false, frames: frames)
        let b = try clockWorld(b2: false, frames: frames)
        b.solver.smallWorldPath = true
        a.solver.smallWorldPath = false
        try lockstep("clock (default config)", a, b, frames: frames) { f in
            XCTAssertFalse(a.solver.lastFrameUsedSmallWorld)
            if b.solver.lastStepCount > 0 { XCTAssertTrue(b.solver.lastFrameUsedSmallWorld, "frame \(f) took the small-world path") }
        }
        XCTAssertGreaterThan(b.solver.asleepCount, 30, "the parked world sleeps")
    }

    /// The same with every stage-B1/B2 opt-in the scene runs (manifold solve, warm start, the
    /// speculative colouring, the workers' torsion patch, sheave friction on the falls).
    func testSmallWorldStepsTheClockB2ConfigBitForBit() throws {
        let frames = 240
        let a = try clockWorld(b2: true, frames: frames)
        let b = try clockWorld(b2: true, frames: frames)
        b.solver.smallWorldPath = true
        a.solver.smallWorldPath = false
        try lockstep("clock (B2 opt-ins)", a, b, frames: frames) { _ in
            if b.solver.lastStepCount > 0 { XCTAssertTrue(b.solver.lastFrameUsedSmallWorld) }
        }
    }

    /// Spring feet (plant, pulse, turn on a dynamic plate) through the kernel's foot phases.
    func testSmallWorldStepsSpringFeetBitForBit() throws {
        let a = try feetWorld(), b = try feetWorld()
        b.solver.smallWorldPath = true
        a.solver.smallWorldPath = false
        try lockstep("spring feet", a, b, frames: 200) { _ in
            if b.solver.lastStepCount > 0 { XCTAssertTrue(b.solver.lastFrameUsedSmallWorld) }
        }
    }

    /// A world that outgrows the limits falls back to the multi-dispatch path by itself, and a
    /// world that shrinks back under them resumes the kernel — with no seam either way: every
    /// piece of state that outlives a frame (bodies, contacts, the warm-start snapshot and hash,
    /// joint blocks, sleep timers / keys) lives in the solver's own buffers.
    func testFallsBackPastTheLimitsAndResumesWithoutASeam() throws {
        let frames = 240
        let a = try clockWorld(b2: true, frames: frames)
        let b = try clockWorld(b2: true, frames: frames)
        a.solver.smallWorldPath = false
        b.solver.smallWorldPath = true
        b.solver.smallWorldMaxBodies = 64; b.solver.smallWorldMaxContacts = 2048   // (the defaults, whatever the env)
        var used: [Bool] = [], expected: [Bool] = []
        var contactsBefore = 0
        let script: (Int) -> Void = { f in
            switch f {
            case 60:  b.solver.smallWorldMaxBodies = 16          // 47 bodies > 16 → multi-dispatch
            case 100: b.solver.smallWorldMaxBodies = 64          // back under the limit
            case 140: b.solver.smallWorldMaxContacts = 2         // a frame after > 2 contacts → multi-dispatch
            case 180: b.solver.smallWorldMaxContacts = 2048
            default: break
            }
            contactsBefore = b.solver.contactCount              // what the limit is judged on
        }
        try lockstep("fallback + resume", a, (b.solver, { f in script(f); b.stepper(f) }), frames: frames) { f in
            used.append(b.solver.lastFrameUsedSmallWorld)
            expected.append(b.solver.highWater <= b.solver.smallWorldMaxBodies
                            && contactsBefore <= b.solver.smallWorldMaxContacts && b.solver.lastStepCount > 0)
        }
        XCTAssertEqual(used, expected, "the path follows the limits frame by frame")
        XCTAssertTrue(used[40..<60].allSatisfy { $0 }, "small world before the limit drops")
        XCTAssertTrue(used[60..<100].allSatisfy { !$0 }, "past smallWorldMaxBodies: multi-dispatch")
        XCTAssertTrue(used[100..<140].allSatisfy { $0 }, "resumed")
        XCTAssertTrue(used[140..<180].contains(false), "past smallWorldMaxContacts: multi-dispatch")
        XCTAssertTrue(used[181..<200].allSatisfy { $0 }, "resumed")
    }

    // ── 3b: the fused substep kernels ────────────────────────────────────────────

    /// The fused bookkeeping kernels step every kind of world bit-identically to the one-step
    /// kernels they replace: a mixed pile with warm start and island sleep, boxes with the
    /// manifold solve, a starved colouring (3 rounds: an uncoloured bucket the fused colour-finish
    /// kernel orders and sub-colours every substep), and the clock (joints, hull bars, culling).
    func testFusedSubstepKernelsMatchTheOneStepKernels() throws {
        // (Both on the multi-dispatch path whatever VIZ_COINDEM_SMALLWORLD says: this is its test.)
        for kind in ["mixedWarm", "boxManifold", "starved", "eggs"] {
            let a = try pileWorld(kind), b = try pileWorld(kind)
            a.solver.fusedSubstepKernelsDisabledForTesting = true
            a.solver.smallWorldPath = false; b.solver.smallWorldPath = false
            try lockstep("3b \(kind)", a, b, frames: 150, wall: 1.0 / 60)
            if kind == "starved" { XCTAssertGreaterThan(b.solver.colorStats.uncolored, 0, "the starved colouring left contacts for the bucket") }
        }
        let a = try clockWorld(b2: true, frames: 200), b = try clockWorld(b2: true, frames: 200)
        a.solver.fusedSubstepKernelsDisabledForTesting = true
        a.solver.smallWorldPath = false; b.solver.smallWorldPath = false
        try lockstep("3b clock", a, b, frames: 200)
    }

    /// A grid past 1024 scan blocks (151 × 64 × 151 ≈ 1.46 M cells, 1 426 blocks): the fused
    /// broadphase — FORCED past its one-chunk limit (`fusedBroadphaseMaxBlocksForTesting`) — scans it
    /// in two chunks of its 1024 threads, and small piles spread over the whole volume — cells on
    /// both sides of the chunk seam (z-major: z ≳ 1.3 m is the second chunk) — step bit-identically
    /// to the one-step scan. By default such a grid takes the one-step broadphase behind the fused
    /// coinSubstepBegin (whose cell clear it then skips): that combination is held to the one-step
    /// kernels too. (Stage-B3 verifier: past one chunk the one-threadgroup scan cost +3 ms per frame at
    /// 1.28 M cells and +90 ms at 8 M — see `CoinDEMSolver.fusedBroadphaseFits`.)
    func testFusedBroadphaseScansAHouseScaleGrid() throws {
        let (engine, lib, _) = try Self.shared()
        func world() throws -> World {
            let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: 96, coinRadius: 0.02, halfThickness: 0.02,
                                                boundsMin: SIMD3(-3, -0.1, -3), boundsMax: SIMD3(3, 2.4, 3)))
            Self.clockConfig(s, dt: 1.0 / 120, iterations: 4)
            s.maxSubsteps = 4; s.sleepEnabled = false; s.warmStart = true
            var cols: [CoinStaticCollider] = [.plane(normal: SIMD3(0, 1, 0), offset: 0)]
            var spots: [SIMD3<Float>] = []
            for (x, z) in [(Float(-2.6), Float(-2.6)), (2.6, -2.6), (-2.6, 2.6), (2.6, 2.6), (0, 0), (1.3, -1.9)] {
                let h: Float = 0.2 + 0.3 * abs(x + z)
                cols.append(.box(center: SIMD3(x, h / 2, z), halfExtents: SIMD3(0.08, h / 2, 0.08)))
                spots.append(SIMD3(x, h, z))
            }
            s.setColliders(cols)
            var k = 0
            return (s, { f in
                guard f % 3 == 0, k < 90 else { return }
                let p = spots[k % spots.count] + SIMD3(Float(k % 5) * 0.004 - 0.008, 0.05 + Float(k / spots.count) * 0.012, 0)
                _ = s.spawnSphere(at: p, radius: 0.005, mass: 0.01)
                k += 1
            })
        }
        for forceFused in [true, false] {
            let a = try world(), b = try world()
            a.solver.fusedSubstepKernelsDisabledForTesting = true
            a.solver.smallWorldPath = false; b.solver.smallWorldPath = false
            if forceFused { b.solver.fusedBroadphaseMaxBlocksForTesting = Int.max }
            XCTAssertEqual(b.solver.fusedBroadphaseFits, forceFused, "1 426 scan blocks: the fused scan only when forced")
            try lockstep("3b house-scale grid (\(forceFused ? "fused, two chunks" : "fused begin + one-step broadphase"))",
                         a, b, frames: 330, wall: 1.0 / 60)
            XCTAssertGreaterThan(b.solver.contactCount, 60, "the piles are in contact")
        }
        XCTAssertTrue(try pileWorld("eggs").solver.fusedBroadphaseFits, "a one-chunk grid takes the fused broadphase")
    }

    // ── Measurements (env-gated: machine-dependent numbers, printed for the record) ─────────

    /// The crossover the default limits (64 bodies / 2048 contacts) sit under. Two sweeps, each
    /// world stepped on BOTH paths in lockstep (bit-identical, so the same work) with sleep off:
    /// bodies (a mixed pile of 16 … 256), and contacts at the 64-body limit (boxes on a slatted
    /// floor — more slats, more contacts per box). Prints GPU ms p50 for each path.
    ///   VIZ_COINDEM_SMALLWORLD_CROSSOVER=1 ./Scripts/test.sh --filter testSmallWorldCrossover
    func testSmallWorldCrossover() throws {
        guard ProcessInfo.processInfo.environment["VIZ_COINDEM_SMALLWORLD_CROSSOVER"] != nil else {
            throw XCTSkip("measurement; set VIZ_COINDEM_SMALLWORLD_CROSSOVER=1")
        }
        func p50(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        func measure(_ label: String, _ make: () throws -> World, frames: Int, timed: Range<Int>) throws {
            let a = try make(), b = try make()
            a.solver.smallWorldPath = false
            b.solver.smallWorldPath = true
            b.solver.smallWorldMaxBodies = .max; b.solver.smallWorldMaxContacts = .max
            var ta: [Double] = [], tb: [Double] = [], contacts: [Int] = []
            for f in 0..<frames {
                a.stepper(f); b.stepper(f)
                let ma = try frame(a.solver), mb = try frame(b.solver)
                if timed.contains(f) { ta.append(ma); tb.append(mb); contacts.append(b.solver.contactCount) }
            }
            XCTAssertTrue(b.solver.lastFrameUsedSmallWorld)
            let c = contacts.sorted()[contacts.count / 2]
            print(String(format: "CROSSOVER %@ bodies=%d contacts(p50)=%d | multi p50 %.3f ms | small-world p50 %.3f ms | ratio %.2f",
                         label, b.solver.activeCount, c, p50(ta), p50(tb), p50(tb) / p50(ta)))
        }
        for n in [16, 32, 48, 64, 96, 128, 192, 256] {
            try measure("pile n=\(n)", {
                let w = try pileWorld("mixedWarm", bodies: n)
                w.solver.sleepEnabled = false
                return (w.solver, { f in for _ in 0..<4 { w.stepper(f) } })
            }, frames: 150, timed: 90..<150)
        }
        // 64 boxes on a floor of slats: pitch → contacts per box.
        let (engine, lib, _) = try Self.shared()
        for pitch in [Float(0), 0.012, 0.006, 0.004] {
            try measure("64 boxes, slat pitch \(pitch * 1000) mm", {
                let s = try XCTUnwrap(CoinDEMSolver(engine: engine, library: lib, maxCoins: 64, coinRadius: 0.03, halfThickness: 0.03,
                                                    boundsMin: SIMD3(-0.3, -0.05, -0.3), boundsMax: SIMD3(0.3, 0.3, 0.3)))
                Self.clockConfig(s)
                s.sleepEnabled = false; s.manifoldSolve = true; s.warmStart = true; s.coloringScheme = .speculative
                var cols: [CoinStaticCollider] = [.plane(normal: SIMD3(0, 1, 0), offset: 0)]
                if pitch > 0 {
                    var x: Float = -0.2
                    while x <= 0.2 && cols.count < 250 {
                        cols.append(.box(center: SIMD3(x, 0.004, 0), halfExtents: SIMD3(0.0008, 0.004, 0.2)))
                        x += pitch
                    }
                }
                s.setColliders(cols)
                let floorY: Float = pitch > 0 ? 0.008 : 0
                for i in 0..<64 {
                    let p = SIMD3(Float(i % 8) * 0.045 - 0.1575, floorY + 0.0101, Float(i / 8) * 0.045 - 0.1575)
                    _ = s.spawnBox(at: p, halfExtents: SIMD3(0.018, 0.01, 0.018), mass: 0.02)
                }
                return (s, { _ in })
            }, frames: 120, timed: 60..<120)
        }
    }
}
