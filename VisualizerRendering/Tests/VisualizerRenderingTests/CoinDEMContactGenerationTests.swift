import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Gate for the CoinDEM contact-generation + dispatch work of the Digital Clock work
/// plan §6 items 2a–2c and 3a (VZ-0151, VZ-0152, VZ-0153, VZ-0154). Each test is the
/// regression proof of one root cause; measured numbers are PRINTed with an `A2_`
/// prefix. Same runtime-compiled-library seam and §G toy-scale configuration as
/// CoinDEMActuationTests / CoinDEMCorrectnessTests.
@MainActor
final class CoinDEMContactGenerationTests: XCTestCase {

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

    /// §G knobs (1/180 s × 6, sleep on, sleepLinVel 2e-4, 0.2 mm speculative margin).
    private func makeSolver(maxCoins: Int = 64, maxRadius: Float = 0.03,
                            boundsMin: SIMD3<Float> = SIMD3(-0.4, -0.1, -0.4),
                            boundsMax: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4),
                            dt: Float = 1.0 / 180, iterations: Int = 6,
                            colliders: [CoinStaticCollider]? = nil) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: maxRadius, halfThickness: maxRadius,
                                    boundsMin: boundsMin, boundsMax: boundsMax)
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = dt
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
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.setColliders(colliders ?? [.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.15)])
        return (s, queue)
    }

    /// Encode + commit + wait; returns each frame's GPU ms (0 for a skipped frame).
    @discardableResult
    private func step(_ s: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
                      wallDt: Float = 1.0 / 60, perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var out: [Double] = []
        for f in 0..<frames {
            perFrame?(f)
            guard let cb = queue.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            out.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        return out
    }

    static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }

    /// A bitwise, order-independent image of the contact buffer (the append order is
    /// racy; the SET is what the narrowphase decides).
    private func contactSet(_ s: CoinDEMSolver) -> [String] {
        let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
        return (0..<s.contactCount).map { i -> String in
            let c = p[i]
            let f: [Float] = [c.nrm.x, c.nrm.y, c.nrm.z, c.nrm.w, c.rA.x, c.rA.y, c.rA.z, c.rB.x, c.rB.y, c.rB.z]
            return "\(c.meta.x) \(c.meta.y) \(c.meta.z) \(c.meta.w) " + f.map { String($0.bitPattern, radix: 16) }.joined(separator: " ")
        }.sorted()
    }

    /// The Digital Clock segment bar (CoinDEMActuationTests.barHullPoints).
    private func bar(_ s: CoinDEMSolver) throws -> CoinDEMSolver.HullHandle {
        try XCTUnwrap(s.registerHull(vertices: CoinDEMActuationTests.barHullPoints()))
    }

    /// Min wall time of one generateContactsNow (broadphase + generate, committed and
    /// waited). The minimum over many repetitions: contention on a shared GPU only adds.
    private func generateWallMs(_ s: CoinDEMSolver, reps: Int = 41) -> Double {
        for _ in 0..<3 { s.generateContactsNow() }
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<reps {
            let t0 = CFAbsoluteTimeGetCurrent()
            s.generateContactsNow()
            best = min(best, (CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        return best
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2a — VZ-0151: per-collider bounding reject in the static loop
    // ══════════════════════════════════════════════════════════════════════════

    /// The reject must be EXACT: every shape (box, disc, sphere, capsule, egg, hull)
    /// scattered at random poses through every culled collider kind (plane, AABB, OBB,
    /// full tube, half-pipe) emits the bit-identical contact set with the reject on and
    /// off (the test seam), at the §G margin, a margin the size of the bodies (lots of
    /// speculative near-contacts right at the reject's boundary), and no margin.
    func testColliderRejectLeavesContactSetBitIdentical() throws {
        var seed: UInt64 = 0xA2_0151
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
        func rq() -> SIMD4<Float> {
            Self.v4(simd_quatf(angle: rnd() * 2 * .pi, axis: simd_normalize(SIMD3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) + 1e-3)))
        }
        var cols: [CoinStaticCollider] = [.plane(normal: SIMD3(0, 1, 0), offset: 0),
                                          .plane(normal: simd_normalize(SIMD3(1, 0.3, 0)), offset: -0.12)]
        for _ in 0..<10 {
            cols.append(.box(center: SIMD3((rnd() - 0.5) * 0.2, rnd() * 0.08, (rnd() - 0.5) * 0.2),
                             halfExtents: SIMD3(0.004 + rnd() * 0.02, 0.004 + rnd() * 0.02, 0.004 + rnd() * 0.02)))
            let q = simd_quatf(angle: rnd() * 2 * .pi, axis: simd_normalize(SIMD3(rnd(), rnd(), rnd()) + 1e-3))
            cols.append(.orientedBox(center: SIMD3((rnd() - 0.5) * 0.2, rnd() * 0.08, (rnd() - 0.5) * 0.2),
                                     halfExtents: SIMD3(0.004 + rnd() * 0.02, 0.003 + rnd() * 0.01, 0.004 + rnd() * 0.02),
                                     orientation: q))
        }
        cols.append(.cylinder(center: SIMD3(0, 0.05, 0), axis: SIMD3(1, 0, 0.2), radius: 0.035,
                              up: SIMD3(0, 1, 0), halfLength: 0.09, lowerHalfOnly: false))
        cols.append(.cylinder(center: SIMD3(0, 0.03, 0.05), axis: SIMD3(0, 0, 1), radius: 0.03,
                              up: SIMD3(0, 1, 0), halfLength: 0.08, lowerHalfOnly: true))
        // Far colliders interleaved AFTER the near ones (collider indices of the near set,
        // which ride in the contacts, stay the same).
        for i in 0..<40 {
            cols.append(.box(center: SIMD3(0.35, 0.35, -0.35 + 0.017 * Float(i)), halfExtents: SIMD3(0.004, 0.004, 0.004)))
        }

        for spec: Float in [2e-4, 0.006, 0] {
            let (s, _) = try makeSolver(maxCoins: 96, maxRadius: 0.03, colliders: cols)
            s.speculativeMargin = spec
            let h = try bar(s)
            for i in 0..<90 {
                let p = SIMD3((rnd() - 0.5) * 0.24, rnd() * 0.1, (rnd() - 0.5) * 0.24)
                switch i % 6 {
                case 0: _ = s.spawnBox(at: p, halfExtents: SIMD3(0.004 + rnd() * 0.006, 0.003 + rnd() * 0.004, 0.005), orient: rq(), mass: 0.01)
                case 1: _ = s.spawn(at: p, orient: rq(), radius: 0.006 + rnd() * 0.006, halfThickness: 0.0015, mass: 0.005)
                case 2: _ = s.spawnSphere(at: p, radius: 0.004 + rnd() * 0.008, mass: 0.01)
                case 3: _ = s.spawnCapsule(at: p, radius: 0.003 + rnd() * 0.003, halfLength: 0.004 + rnd() * 0.008, orient: rq(), mass: 0.01)
                case 4: _ = s.spawnEgg(at: p, fatRadius: 0.009, tipRadius: 0.006, centerDistance: 0.011, orient: rq(), mass: 0.01)
                default: _ = s.spawnHull(at: p, hull: h, orient: rq(), mass: 0.001)
                }
            }
            s.generateContactsNow()
            let culled = contactSet(s)
            s.colliderCullDisabledForTesting = true
            s.generateContactsNow()
            let full = contactSet(s)
            let statics = full.filter { $0.split(separator: " ")[1] == "4294967295" }.count
            print("A2_2a spec=\(spec): contacts culled=\(culled.count) unculled=\(full.count) (static \(statics)) identical=\(culled == full)")
            XCTAssertGreaterThan(statics, 60, "precondition: the scene has plenty of static contacts at spec \(spec)")
            XCTAssertEqual(culled, full, "the per-collider reject never changes the contact set (spec \(spec))")
        }
    }

    /// The cost the reject removes: one bar hull against 134 far colliders used to cost
    /// ≈18.7 µs per collider per generate pass (+2.5 ms, measured); now ≈ the floor.
    func testFarCollidersCostAboutNothing() throws {
        let far = CoinStaticCollider.box(center: SIMD3(0, 0.3, 0.3), halfExtents: SIMD3(0.001, 0.001, 0.001))
        let (s, _) = try makeSolver(colliders: [])
        s.gravity = 0
        _ = s.spawnHull(at: SIMD3(0, 0.1, 0), hull: try bar(s), mass: 0.001)
        _ = s.spawnBallastedEgg(CoinDEMCorrectnessTests.worker, fatCenter: SIMD3(0.1, 0.1, 0))
        _ = s.spawnBox(at: SIMD3(-0.1, 0.1, 0), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.01)
        let floor = generateWallMs(s)
        s.setColliders(Array(repeating: far, count: 134))
        let loaded = generateWallMs(s)
        print(String(format: "A2_2a generate pass (min wall): 0 colliders %.3f ms, 134 far colliders %.3f ms (Δ %.3f ms; was ≈ +2.5 ms)",
                     floor, loaded, loaded - floor))
        XCTAssertLessThan(loaded - floor, 0.5, "134 far colliders cost about nothing per generate pass")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2b — VZ-0152: asleep bodies generate nothing, islands still wake as one
    // ══════════════════════════════════════════════════════════════════════════

    /// Two rows of 6 cubes lie on the floor, each cube touching only its neighbours
    /// (0.05 mm side gaps, inside the 0.2 mm speculative margin) — a chain island, so a
    /// wake that spread one contact layer per frame would need 5 frames to reach the far
    /// end. They settle and sleep while a host-driven sphere keeps the world stepping.
    /// Asleep, neither row emits a single contact (asleep–asleep and asleep–static pairs
    /// used to run the full narrowphase every substep). Waking the END cube of row 1
    /// wakes all six in that same frame and leaves row 2 asleep; an awake cube dropped
    /// on the end of row 2 wakes all six of it in the first frame any of it wakes.
    func testAsleepRowsGenerateNothingAndWakeAsOneIsland() throws {
        let (s, q) = try makeSolver()
        s.rollingResistance = 0
        let he: Float = 0.005
        func row(_ z: Float) -> [Int] {
            (0..<6).map { k in s.spawnBox(at: SIMD3(-0.03 + Float(k) * (2 * he + 5e-5), he + 5e-5, z),
                                          halfExtents: SIMD3(he, he, he), mass: 0.01)! }
        }
        let r1 = row(-0.05), r2 = row(0.05)
        let sphere = s.spawnSphere(at: SIMD3(-0.3, 0.01, 0.2), radius: 0.01, mass: 0.01)!
        func keepSphereMoving(_: Int) { s.setVelocity(ofSlot: sphere, to: SIMD3(0.02, 0, 0)) }
        func asleep(_ b: [Int]) -> Int { b.filter { s.isAsleep($0) }.count }
        func linked(_ b: [Int]) -> Bool {       // every neighbour pair in contact
            zip(b, b.dropFirst()).allSatisfy { x, y in
                s.contacts(touching: [x]).contains { Int($0.meta.x) == y || Int($0.meta.y) == y }
            }
        }

        var settled = -1, chained1 = false, chained2 = false
        for f in 0..<240 {
            step(s, q, frames: 1, perFrame: keepSphereMoving)
            if asleep(r1) == 6 && asleep(r2) == 6 { settled = f; break }
            if asleep(r1) == 0 { chained1 = linked(r1) }   // the last fully-awake frame's contacts
            if asleep(r2) == 0 { chained2 = linked(r2) }
        }
        let chained = chained1 && chained2
        XCTAssertGreaterThanOrEqual(settled, 0, "both rows fall asleep")
        XCTAssertTrue(chained, "precondition: each row is one chain of neighbour contacts")
        step(s, q, frames: 2, perFrame: keepSphereMoving)
        XCTAssertFalse(s.didSkipLastFrame, "the moving sphere keeps the world stepping")
        let rowContacts = s.contacts(touching: Set(r1 + r2)).count
        print("A2_2b rows asleep at frame \(settled) (chained \(chained)); row contacts while asleep = \(rowContacts); world contacts = \(s.contactCount)")
        XCTAssertEqual(rowContacts, 0, "asleep–asleep and asleep–static pairs emit no contacts")

        // (a) Host-wake the END cube of row 1: the whole row is awake after ONE frame.
        s.wake([r1[5]])
        step(s, q, frames: 1, perFrame: keepSphereMoving)
        print("A2_2b after wake(end of r1) + 1 frame: r1 asleep \(asleep(r1))/6, r2 asleep \(asleep(r2))/6")
        XCTAssertEqual(asleep(r1), 0, "waking one body wakes its whole island in the same frame")
        XCTAssertEqual(asleep(r2), 6, "a separate island stays asleep")
        step(s, q, frames: 1, perFrame: keepSphereMoving)   // the first frame generated with it all awake
        XCTAssertTrue(linked(r1), "the woken row collides again")

        // Re-settle.
        for _ in 0..<240 {
            step(s, q, frames: 1, perFrame: keepSphereMoving)
            if asleep(r1) == 6 { break }
        }
        XCTAssertEqual(asleep(r1), 6, "row 1 re-sleeps")

        // (b) Drop an awake cube on the end of row 2: the frame any of it wakes, all of it does.
        let endX = s.position(of: r2[5])!.x
        let dropped = s.spawnBox(at: SIMD3(endX, 3 * he + 0.01, 0.05), halfExtents: SIMD3(he, he, he), mass: 0.01)!
        var firstWake = -1, awakeAtFirst = -1
        for f in 0..<60 {
            step(s, q, frames: 1, perFrame: keepSphereMoving)
            let a = asleep(r2)
            if a < 6 && firstWake < 0 { firstWake = f; awakeAtFirst = 6 - a }
        }
        print("A2_2b drop on the end of r2: first wake at frame \(firstWake), awake then \(awakeAtFirst)/6; r1 asleep \(asleep(r1))/6; dropped asleep=\(s.isAsleep(dropped))")
        XCTAssertGreaterThanOrEqual(firstWake, 0, "the impact wakes row 2")
        XCTAssertEqual(awakeAtFirst, 6, "the whole row wakes in the first frame any of it does")
        XCTAssertEqual(asleep(r1), 6, "the other row is not woken")
    }

    /// Asleep bars cost ≈ nothing: the 21 seated Digital Clock bars (neighbours inside
    /// bounding reach) with the 134 clock statics used to cost a full hull–hull +
    /// static pass each substep while anything else was awake (+2.3 ms per generate
    /// pass over the floor, measured).
    func testAsleepSeatedBarsCostAboutNothing() throws {
        typealias T = CoinDEMActuationTests
        let (s, _) = try makeSolver(maxCoins: 48, maxRadius: 0.27,
                                    boundsMin: SIMD3(-0.30, 0.55, 0.00), boundsMax: SIMD3(0.30, 0.90, 0.46), colliders: [])
        let h = try bar(s)
        let reading: [[Int]] = [[1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]
        var bars: [Int] = []
        for (dIdx, segs) in reading.enumerated() {
            for seg in segs {
                let (dx, dy, vertical) = T.sockets[seg]
                let rd = vertical ? T.barVertical : T.barHorizontal
                let td = SIMD3(T.digitX[dIdx] + dx / 1000, T.rowD + dy / 1000, T.seatZ)
                bars.append(s.spawnHull(at: td + simd_act(rd, h.comOffset), hull: h, orient: Self.v4(rd * h.principalRotation), mass: 0.00103)!)
            }
        }
        let box = s.spawnBox(at: SIMD3(0.2, 0.8, 0.4), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.01)!
        _ = box
        s.setColliders([])
        let floor = generateWallMs(s)          // 21 awake bars, no statics
        s.setColliders(T.clockStatics().all)
        let awake = generateWallMs(s)
        let a = s.asleepBuffer.contents().bindMemory(to: UInt32.self, capacity: s.maxCoins)
        for b in bars { a[b] = 1 }
        let asleep = generateWallMs(s)
        s.setColliders([])
        let asleepNoStatics = generateWallMs(s)
        print(String(format: "A2_2b 21 seated bars, generate pass (min wall): awake, no statics %.3f | awake + 134 statics %.3f | ASLEEP + 134 statics %.3f | asleep, no statics %.3f ms",
                     floor, awake, asleep, asleepNoStatics))
        XCTAssertLessThan(asleep - asleepNoStatics, 0.15, "asleep bars skip the static loop")
        XCTAssertLessThan(asleep, awake * 0.5, "asleep bars cost a fraction of awake ones")
        XCTAssertEqual(s.contacts(touching: Set(bars)).count, 0)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2c — VZ-0153: the collideConnected scan runs only for jointed pairs in reach
    // ══════════════════════════════════════════════════════════════════════════

    /// Seven overlapping cubes in a row, every neighbour pair 1 mm deep: A–B hinged
    /// (collideConnected off) must emit nothing; C–D likewise; B–C (both jointed, but
    /// not to each other), D–E (E unjointed), a world-jointed F and a collideConnected
    /// G–H pair must all still contact; a DISABLED pooled slot suppresses nothing.
    func testJointedBodyMarkingGatesOnlyJointPairs() throws {
        let (s, _) = try makeSolver(colliders: [])
        s.gravity = 0
        let he = SIMD3<Float>(0.005, 0.005, 0.005)
        let ids = (0..<8).map { i in s.spawnBox(at: SIMD3(Float(i) * 0.009, 0.1, 0), halfExtents: he, mass: 0.01)! }
        let (a, b, c, d, e, f, g, h) = (ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7])
        _ = s.addHingeJoint(bodyA: a, bodyB: b, worldAnchor: SIMD3(0.0045, 0.1, 0), worldAxis: SIMD3(0, 0, 1))
        _ = s.addHingeJoint(bodyA: c, bodyB: d, worldAnchor: SIMD3(0.0225, 0.1, 0), worldAxis: SIMD3(0, 0, 1))
        _ = s.addBallJoint(bodyA: f, bodyB: nil, worldAnchor: SIMD3(0.045, 0.12, 0))
        _ = s.addBallJoint(bodyA: g, bodyB: h, worldAnchor: SIMD3(0.0585, 0.1, 0), collideConnected: true)
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 6, placeholderBody: a))
        let spare = try XCTUnwrap(pool.takeSlot())
        s.enableBall(slot: spare, bodyA: e, bodyB: f, worldAnchor: SIMD3(0.0405, 0.1, 0))
        s.disableJoint(slot: spare)                                // disabled: suppresses nothing
        s.generateContactsNow()
        func n(_ x: Int, _ y: Int) -> Int {
            s.contacts(touching: [x]).filter { Int($0.meta.x) == y || Int($0.meta.y) == y }.count
        }
        let pairs = [("A–B hinge", n(a, b)), ("B–C", n(b, c)), ("C–D hinge", n(c, d)), ("D–E", n(d, e)),
                     ("E–F (disabled slot)", n(e, f)), ("F–G (F world-jointed)", n(f, g)), ("G–H collideConnected", n(g, h))]
        print("A2_2c " + pairs.map { "\($0.0)=\($0.1)" }.joined(separator: " "))
        XCTAssertEqual(n(a, b), 0, "a hinge with collideConnected off suppresses its pair")
        XCTAssertEqual(n(c, d), 0)
        XCTAssertGreaterThan(n(b, c), 0, "two jointed bodies that are not joined to each other still collide")
        XCTAssertGreaterThan(n(d, e), 0)
        XCTAssertGreaterThan(n(e, f), 0, "a disabled slot suppresses nothing")
        XCTAssertGreaterThan(n(f, g), 0, "a world joint never suppresses a body pair")
        XCTAssertGreaterThan(n(g, h), 0, "collideConnected keeps the pair's contacts")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3a — VZ-0154: colour sweep = seen + 1, serial tail for the rest
    // ══════════════════════════════════════════════════════════════════════════

    /// The tail pass solves colours in the same order with the same per-contact code, so
    /// WHERE the host splits the sweep cannot change the result: a tumbling mixed pile
    /// stepped with the normal sweep and with the sweep forced to 0 every frame (every
    /// colour solved by the one-threadgroup tail) ends bit-identical — with and without
    /// warm starting (whose apply pass has its own tail).
    func testColourTailIsExactWhereverTheSweepSplits() throws {
        for warm in [false, true] {
            var poses: [[SIMD4<Float>]] = []
            var beyond: [Int] = [], maxColor = 0, uncolored = 0
            for forceTail in [false, true] {
                let (s, q) = try makeSolver(maxCoins: 48, maxRadius: 0.02,
                                            colliders: RigidPileField.bin(innerHalf: SIMD2(0.05, 0.05), floorY: 0))
                s.warmStart = warm
                s.sleepEnabled = false
                var seed: UInt64 = 0x3A_0154
                func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
                for i in 0..<36 {
                    let p = SIMD3((rnd() - 0.5) * 0.08, 0.01 + rnd() * 0.12, (rnd() - 0.5) * 0.08)
                    let t = SIMD3((rnd() - 0.5) * 20, (rnd() - 0.5) * 20, (rnd() - 0.5) * 20)
                    switch i % 3 {
                    case 0: _ = s.spawnBox(at: p, halfExtents: SIMD3(0.006, 0.006, 0.006), tumble: t, mass: 0.01)
                    case 1: _ = s.spawnSphere(at: p, radius: 0.007, tumble: t, mass: 0.01)
                    default: _ = s.spawn(at: p, tumble: t, radius: 0.009, halfThickness: 0.002, mass: 0.005)
                    }
                }
                step(s, q, frames: 90) { _ in if forceTail { s.resetColorStats() } }
                let p = s.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: s.maxCoins)
                poses.append((0..<s.highWater).flatMap { [p[$0].posInvMass, p[$0].orient] })
                let cs = s.colorStats
                beyond.append(cs.beyondSweep)
                if !forceTail { maxColor = cs.maxColorUsed; uncolored = cs.uncolored }
            }
            let identical = poses[0] == poses[1]
            var worst: Float = 0
            for (x, y) in zip(poses[0], poses[1]) { worst = max(worst, simd_reduce_max(simd_abs(x - y))) }
            print("A2_3a warmStart=\(warm): maxColorUsed=\(maxColor) uncolored=\(uncolored) beyondSweep normal=\(beyond[0]) forced-tail(last frame)=\(beyond[1]) identical=\(identical) worst|Δ|=\(worst)")
            XCTAssertGreaterThan(maxColor, 4, "precondition: a pile that needs several colours")
            XCTAssertEqual(uncolored, 0)
            XCTAssertGreaterThan(beyond[1], 0, "precondition: the forced run really solved in the tail")
            XCTAssertTrue(identical, "tail-solved and sweep-solved piles are bit-identical (warmStart \(warm)); worst |Δ| \(worst)")
        }
    }

    /// Contacts the colouring's fixed round budget leaves UNCOLOURED used to be dropped —
    /// never solved (the dense hull pile in CoinDEMGenericEngineTests had a cube fall
    /// through another, 68 mm deep, on one such contact). The tail now solves them, in
    /// deterministic greedy sub-colours. With colorRounds = 0 EVERY contact is
    /// uncoloured: a row of 4 touching cubes must still rest where it lies (before,
    /// nothing was solved and the row fell through the floor), and two identical runs
    /// must agree bit for bit.
    func testUncolouredContactsAreSolvedSeriallyAndReproducibly() throws {
        var finals: [[SIMD4<Float>]] = []
        for _ in 0..<2 {
            let (s, q) = try makeSolver()
            s.colorRounds = 0
            s.sleepEnabled = false
            let he: Float = 0.005
            let boxes = (0..<4).map { k in s.spawnBox(at: SIMD3(Float(k) * (2 * he + 5e-5), he + 5e-5, 0),
                                                      halfExtents: SIMD3(he, he, he), mass: 0.01)! }
            step(s, q, frames: 90)
            let cs = s.colorStats
            let ys = boxes.map { s.position(of: $0)!.y }
            let sideContacts = s.contacts(touching: Set(boxes)).filter { $0.meta.y != 0xFFFF_FFFF }.count
            print(String(format: "A2_3a colorRounds=0: uncolored=%d (all in the tail bucket); cube–cube contacts %d; cubes y = %.5f %.5f %.5f %.5f (rest %.5f)",
                         cs.uncolored, sideContacts, ys[0], ys[1], ys[2], ys[3], he))
            XCTAssertGreaterThan(cs.uncolored, 0, "precondition: nothing was coloured")
            XCTAssertGreaterThan(sideContacts, 0, "precondition: the cubes touch each other too")
            for y in ys { XCTAssertEqual(y, he, accuracy: 3e-4, "uncoloured contacts hold each cube up") }
            let p = s.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: s.maxCoins)
            finals.append(boxes.flatMap { [p[$0].posInvMass, p[$0].orient] })
        }
        XCTAssertEqual(finals[0], finals[1], "the serial order is deterministic")
    }

    /// The uncoloured bucket's cost is bounded. Separated resting cubes (4 floor contacts
    /// each, no cube–cube contacts) with colorRounds 0 put exactly 4·n contacts in the
    /// bucket every substep. At 64 cubes (bucket 256, the cap) the bucket used to be ordered
    /// and greedy-coloured by ONE thread out of device memory — +28.8 ms per frame at
    /// 1/180 s × 6 (≈ 0.15 µs·n² per substep) — and past the cap it was solved one contact
    /// per pass (a Daydream-Home-scale egg rain: 11–12 → 55–81 ms per frame). Now: the
    /// capped bucket is staged in threadgroup memory (the cubes still rest), and an
    /// oversized one is left unsolved, as before the tail existed, and counted.
    func testUncolouredBucketCostIsBoundedAndOversizeIsCounted() throws {
        func run(cubes k: Int, rounds: Int) throws -> (minMs: Double, bucket: Double, unsolved: Int, worstDy: Float) {
            let (s, q) = try makeSolver(maxCoins: 80)
            s.colorRounds = rounds
            s.sleepEnabled = false
            let he: Float = 0.005
            let ids = (0..<k).map { i in
                s.spawnBox(at: SIMD3(-0.3 + Float(i % 16) * 0.035, he + 5e-5, -0.3 + Float(i / 16) * 0.035),
                           halfExtents: SIMD3(he, he, he), mass: 0.01)!
            }
            step(s, q, frames: 20)
            s.resetColorStats()
            let t = step(s, q, frames: 60)
            let cs = s.colorStats
            let dy = ids.map { abs(s.position(of: $0)!.y - he) }.max() ?? 0
            return (t.min() ?? 0, Double(cs.uncolored) / Double(60 * 3), s.uncoloredUnsolved, dy)
        }
        let coloured = try run(cubes: 64, rounds: 24)
        let capped = try run(cubes: 64, rounds: 0)
        let over = try run(cubes: 72, rounds: 0)
        print(String(format: "A2_3a bucket cost: 64 cubes coloured min %.3f ms | bucket %.0f/substep (at cap) min %.3f ms, unsolved %d, worst |Δy| %.6f | bucket %.0f/substep (over cap) min %.3f ms, unsolved %d",
                     coloured.minMs, capped.bucket, capped.minMs, capped.unsolved, capped.worstDy, over.bucket, over.minMs, over.unsolved))
        XCTAssertEqual(coloured.bucket, 0, "precondition: 24 rounds colour every floor contact")
        XCTAssertEqual(capped.bucket, 256, accuracy: 0.5, "precondition: the bucket sits exactly at the cap")
        XCTAssertEqual(capped.unsolved, 0, "a bucket at the cap is solved")
        XCTAssertLessThan(capped.worstDy, 3e-4, "…and holds every cube up")
        XCTAssertLessThan(capped.minMs - coloured.minMs, 5.0, "ordering + sub-colouring a 256-contact bucket costs little (was +28.8 ms/frame)")
        XCTAssertGreaterThan(over.bucket, 256, "precondition: the bucket is over the cap")
        XCTAssertEqual(Double(over.unsolved), over.bucket * 180, accuracy: 0.5, "an oversized bucket is left unsolved and counted, every substep")
        XCTAssertLessThan(over.minMs - coloured.minMs, 5.0, "an oversized bucket costs no serial detour")
    }

    /// A world that has never had a contact (a crane hanging on joints, a body in
    /// flight) used to sweep all 64 colours: 64 empty indirect dispatches per velocity
    /// iteration per substep, ≈ 8 ms/frame at 1/180 s × 6 for ONE floating box. It now
    /// sweeps none (one tail dispatch that returns at once per iteration).
    func testContactFreeWorldDoesNotSweepColours() throws {
        let (s, q) = try makeSolver(colliders: [])
        s.gravity = 0
        s.sleepEnabled = false
        _ = s.spawnBox(at: SIMD3(0, 0.2, 0), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.01)
        step(s, q, frames: 10, wallDt: 1.0 / 30)
        let t = step(s, q, frames: 40, wallDt: 1.0 / 30)
        let best = t.min() ?? 0, med = t.sorted()[t.count / 2]
        print(String(format: "A2_3a floating box, 6 substeps × 6 iterations: frame GPU min %.3f ms, p50 %.3f ms (was ≈ 8 ms); maxColorUsed=%d",
                     best, med, s.colorStats.maxColorUsed))
        XCTAssertEqual(s.colorStats.maxColorUsed, 0, "precondition: no contact ever")
        XCTAssertLessThan(best, 2.5, "a contact-free frame no longer pays for 64 empty colours")
    }
}
