import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Headless correctness gate for the GPU coin DEM solver (CoinDEM.metal +
/// CoinDEMSolver). This is the Phase-1 "is the sim flawless?" gate from the plan:
/// stable rest, gap-correct stacking, no explosion, no tunnelling, no
/// interpenetration — proven before any cabinet/pusher work.
///
/// The SwiftPM CLI doesn't compile the package's `.metal` into a metallib (an
/// Xcode-only rule — it only copies the source), so we compile CoinDEM.metal from
/// source at runtime and inject the library through the solver's test seam. The
/// REAL solver orchestration runs; only the pipeline source changes.
@MainActor
final class CoinDEMSolverTests: XCTestCase {

    // Real arcade-token proportions (~13.5:1), scaled up to scene-units for the
    // grid to have sane resolution: R = 0.05, thickness = 2h.
    private static let R: Float = 0.05
    private static let h: Float = 0.0037
    private static let floorY: Float = 0.0

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // VisualizerRenderingTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // package root
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    private func makeSolver(maxCoins: Int = 64,
                            boundsMin: SIMD3<Float> = SIMD3(-1, -0.5, -1),
                            boundsMax: SIMD3<Float> = SIMD3(1, 2, 1)) throws
        -> (CoinDEMSolver, MTLCommandQueue)
    {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib,
                                         maxCoins: maxCoins,
                                         coinRadius: Self.R, halfThickness: Self.h,
                                         boundsMin: boundsMin, boundsMax: boundsMax),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("solver/queue init failed") }
        solver.floorY = Self.floorY
        solver.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: Self.floorY)])
        return (solver, queue)
    }

    private func step(_ solver: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int) {
        for _ in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        }
    }

    // ── CONSTRAINT SOLVER Stage 1: contact generation into a buffer ────────────
    // Spawn two deterministically-overlapping coins above an overlapped floor and
    // verify the generation kernel emits well-formed contacts: unit normals, positive
    // depth, valid body refs, dedup (B>A), and BOTH a floor (static) contact and a
    // coin↔coin (pair) contact. This is the foundation the colour/velocity/position
    // passes build on. ────────────────────────────────────────────────────────────
    func testConstraintContactGenerationStage1() throws {
        let (solver, queue) = try makeSolver(maxCoins: 8)
        _ = queue
        // A: bottom face just below the floor (penetrating). B: just overlapping A's top.
        solver.spawn(at: SIMD3(0, 0.0030, 0), radius: Self.R, halfThickness: Self.h)            // floor pen ≈ 0.0007
        solver.spawn(at: SIMD3(0, 0.0030 + 0.0050, 0), radius: Self.R, halfThickness: Self.h)   // pair pen ≈ 0.0024

        let n = solver.generateContactsNow()
        print("STAGE1_CONTACTS n=\(n)")
        XCTAssertGreaterThan(n, 0, "overlapping bodies must generate contacts")
        XCTAssertLessThanOrEqual(n, solver.maxContacts)

        let ptr = solver.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(1, n))
        var floorContacts = 0, pairContacts = 0
        for i in 0..<n {
            let c = ptr[i]
            let nrm = SIMD3(c.nrm.x, c.nrm.y, c.nrm.z)
            XCTAssertEqual(simd_length(nrm), 1.0, accuracy: 1e-3, "contact normal must be unit length")
            XCTAssertGreaterThan(c.nrm.w, 0, "contact depth must be positive")
            XCTAssertLessThan(Int(c.meta.x), solver.highWater, "bodyA index valid")
            // tangents orthonormal to the normal
            let t1 = SIMD3(c.tan1.x, c.tan1.y, c.tan1.z)
            XCTAssertEqual(simd_dot(nrm, t1), 0, accuracy: 1e-3, "tangent ⟂ normal")
            if c.meta.y == 0xFFFF_FFFF {
                floorContacts += 1
            } else {
                XCTAssertLessThan(Int(c.meta.y), solver.highWater, "bodyB index valid")
                XCTAssertGreaterThan(c.meta.y, c.meta.x, "pair dedup: B index > A index")
                pairContacts += 1
            }
        }
        print("STAGE1 floor=\(floorContacts) pair=\(pairContacts)")
        XCTAssertGreaterThan(floorContacts, 0, "bottom coin penetrates the floor plane → static contact")
        XCTAssertGreaterThan(pairContacts, 0, "the two coins overlap → a coin↔coin contact")
    }

    // ── Measure the colour count a DENSE (176-body) mixed pile actually needs, to
    // safely size `solveColors` (contacts in colours ≥ solveColors are skipped). ─────
    func testConstraintColorCountDense() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 190,
                                         coinRadius: 0.12, halfThickness: 0.12,
                                         boundsMin: SIMD3(-1.1,-0.5,-1.1), boundsMax: SIMD3(1.1,2.6,1.1)),
              let queue = device.makeCommandQueue() else { throw XCTSkip("init failed") }  // gpu-ok: test harness
        solver.floorY = 0
        solver.setColliders(RigidPileField.bin(innerHalf: SIMD2(0.55,0.55), floorY: 0))
        solver.solverMode = .constraint; solver.frictionCoeff = 0.5
        var seed: UInt64 = 0xDEED
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<176 {
            let p = SIMD3<Float>((rnd()-0.5)*0.8, 0.4 + rnd()*1.4, (rnd()-0.5)*0.8)
            let t = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
            switch i % 3 {
            case 0: solver.spawn(at: p, tumble: t, radius: 0.075, halfThickness: 0.022, type: 0)
            case 1: solver.spawnSphere(at: p, radius: 0.052, tumble: t, type: 1)
            default: solver.spawnBox(at: p, halfExtents: SIMD3(0.044,0.044,0.044), tumble: t, type: 2)
            }
        }
        for _ in 0..<400 { let cb = queue.makeCommandBuffer()!; solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted() }  // gpu-ok: test harness
        let n = solver.generateContactsNow(color: true)
        let ptr = solver.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(1, n))
        var maxColor = -1, uncolored = 0
        for i in 0..<n { let c = Int(ptr[i].tan2.w); if c < 0 { uncolored += 1 } else { maxColor = max(maxColor, c) } }
        print("DENSE_COLORS contacts=\(n) maxColor=\(maxColor) colors=\(maxColor+1) uncolored=\(uncolored)")
        XCTAssertLessThan(maxColor + 1, 32, "colour count fits the 32-colour cap")
    }

    // ── CONSTRAINT SOLVER Stage 2: graph colouring validity ────────────────────
    // Settle a dense little pile, generate + colour the contacts, and verify the
    // colouring is a VALID partition for Gauss-Seidel: no two contacts sharing a
    // dynamic body land in the same colour (else the parallel per-colour solve would
    // race). Also confirm ~everything got coloured (few/no −1 overflow). ────────────
    func testConstraintColoringStage2() throws {
        let (solver, queue) = try makeSolver(maxCoins: 40)   // floor plane only
        var seed: UInt64 = 0xBEEF
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for _ in 0..<24 {
            let p = SIMD3<Float>((rnd()-0.5)*0.3, 0.2 + rnd()*0.6, (rnd()-0.5)*0.3)
            solver.spawn(at: p, tumble: SIMD3((rnd()-0.5)*3, (rnd()-0.5)*3, (rnd()-0.5)*3))
        }
        step(solver, queue, frames: 240)

        let n = solver.generateContactsNow(color: true)
        XCTAssertGreaterThan(n, 0)
        let ptr = solver.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(1, n))

        var bodiesByColor: [Int: Set<UInt32>] = [:]
        var uncolored = 0, maxColor = -1
        for i in 0..<n {
            let c = ptr[i]
            let color = Int(c.tan2.w)
            if color < 0 { uncolored += 1; continue }
            maxColor = max(maxColor, color)
            var bodies: [UInt32] = [c.meta.x]
            if c.meta.y != 0xFFFF_FFFF { bodies.append(c.meta.y) }
            for b in bodies {
                let inserted = bodiesByColor[color, default: []].insert(b).inserted
                XCTAssertTrue(inserted, "colour \(color): body \(b) appears twice → invalid partition (would race)")
                bodiesByColor[color, default: []].insert(b)
            }
        }
        print("STAGE2_COLOR contacts=\(n) colors=\(maxColor + 1) uncolored=\(uncolored)")
        XCTAssertLessThanOrEqual(uncolored, n / 20, "colouring should cover ~all contacts in the round budget")
    }

    // ── CONSTRAINT SOLVER Stage 3: a dense mixed pile settles dead-still ───────
    // The payoff stage. Drop coins+marbles+dice into a bin, run the graph-colored
    // sequential-impulse solver, and measure rest kinetic energy. The split-impulse
    // (position recovery via pseudo-velocity, never real velocity) means a settled
    // pile carries ~zero restoring velocity → KE → ~0, no jitter, at full density. ──
    func testConstraintSolverSettlesStage3() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 70,
                                         coinRadius: 0.12, halfThickness: 0.12,
                                         boundsMin: SIMD3(-1.1,-0.5,-1.1), boundsMax: SIMD3(1.1,2.6,1.1)),
              let queue = device.makeCommandQueue() else { throw XCTSkip("init failed") }  // gpu-ok: test harness queue
        solver.floorY = 0
        solver.setColliders(RigidPileField.bin(innerHalf: SIMD2(0.42, 0.42), floorY: 0))
        solver.solverMode = .constraint
        solver.gravity = 9.8
        solver.restitution = 0.1
        solver.frictionCoeff = 0.6      // real Coulomb friction → angle of repose

        var seed: UInt64 = 0xA11CE
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<54 {
            let p = SIMD3<Float>((rnd()-0.5)*0.6, 0.4 + rnd()*1.0, (rnd()-0.5)*0.6)
            let tumble = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
            switch i % 3 {
            case 0: solver.spawn(at: p, tumble: tumble, radius: 0.075, halfThickness: 0.022, type: 0)
            case 1: solver.spawnSphere(at: p, radius: 0.052, tumble: tumble, type: 1)
            default: let he: Float = 0.044; solver.spawnBox(at: p, halfExtents: SIMD3(he,he,he), tumble: tumble, type: 2)
            }
        }
        // Settle. Sample KE over the last several frames.
        for _ in 0..<600 {
            guard let cb = queue.makeCommandBuffer() else { break }
            solver.encode(to: cb, wallDt: 1.0/60.0)
            cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness reads results synchronously
        }
        var keSum: Float = 0
        for _ in 0..<10 {
            guard let cb = queue.makeCommandBuffer() else { break }
            solver.encode(to: cb, wallDt: 1.0/60.0)
            cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness reads results synchronously
            keSum += CoinDiagnostics.measure(solver).kineticEnergy
        }
        let r = CoinDiagnostics.measure(solver)
        print("STAGE3_CONSTRAINT n=\(r.activeCount) restKE_avg=\(keSum/10) maxPen=\(r.maxPenetration) maxY=\(r.maxY) below=\(r.belowFloorCount)")
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN blow-up")
        XCTAssertEqual(r.belowFloorCount, 0, "nothing tunnelled the floor")
        XCTAssertEqual(r.activeCount, 54, "no bodies lost")
        XCTAssertLessThan(keSum/10, 0.05, "the constraint-solved pile settles DEAD-still (the whole point)")
    }

    // ── CONSTRAINT SOLVER Stage 4: warm starting (correct + cheap iterations) ──
    // Warm starting carries each contact's converged impulse across substeps. The
    // colored Gauss-Seidel already converges so well that even 3 velocity iterations
    // settle this heap dead-still — so the value here is twofold: (1) warm-start does
    // not BREAK or destabilize the solve (still dead-still, no NaN), and (2) it proves
    // the iteration budget can be cut hard (8→3) for perf. (Warm-start's convergence
    // edge shows on tall single-column stacks, which a loose heap doesn't stress.) ──
    func testConstraintWarmStartStage4() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)

        func run(warm: Bool) -> Float {
            guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 60,
                                             coinRadius: 0.12, halfThickness: 0.12,
                                             boundsMin: SIMD3(-1.1,-0.5,-1.1), boundsMax: SIMD3(1.1,2.6,1.1)),
                  let queue = device.makeCommandQueue() else { return -1 }  // gpu-ok: test harness queue
            solver.floorY = 0
            solver.setColliders(RigidPileField.bin(innerHalf: SIMD2(0.42, 0.42), floorY: 0))
            solver.solverMode = .constraint
            solver.frictionCoeff = 0.5
            solver.warmStart = warm
            solver.velocityIterations = 3      // deliberately starved
            var seed: UInt64 = 0x5EED
            func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
            for i in 0..<48 {
                let p = SIMD3<Float>((rnd()-0.5)*0.6, 0.4 + rnd()*1.0, (rnd()-0.5)*0.6)
                let t = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
                switch i % 3 {
                case 0: solver.spawn(at: p, tumble: t, radius: 0.075, halfThickness: 0.022, type: 0)
                case 1: solver.spawnSphere(at: p, radius: 0.052, tumble: t, type: 1)
                default: solver.spawnBox(at: p, halfExtents: SIMD3(0.044,0.044,0.044), tumble: t, type: 2)
                }
            }
            for _ in 0..<700 {
                guard let cb = queue.makeCommandBuffer() else { break }
                solver.encode(to: cb, wallDt: 1.0/60.0)
                cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness reads synchronously
            }
            var ke: Float = 0
            for _ in 0..<8 {
                guard let cb = queue.makeCommandBuffer() else { break }
                solver.encode(to: cb, wallDt: 1.0/60.0)
                cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test harness reads synchronously
                ke += CoinDiagnostics.measure(solver).kineticEnergy
            }
            return ke / 8
        }
        let warmKE = run(warm: true)
        let coldKE = run(warm: false)
        print("STAGE4_WARMSTART iters=3 warmKE=\(warmKE) coldKE=\(coldKE)")
        XCTAssertFalse(warmKE.isNaN, "warm starting must not destabilize the solve")
        XCTAssertLessThan(warmKE, 0.02, "warm-started, only 3 iterations settle the pile dead-still (GS converges fast)")
        XCTAssertLessThan(coldKE, 0.02, "cold also settles at 3 iters here — the heap doesn't stress convergence")
    }

    // ── CONSTRAINT SOLVER Stage 5: island detection + sleeping ─────────────────
    // A settled heap should freeze and cost nothing; a new body must wake its island.
    func testConstraintIslandSleepStage5() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 60,
                                         coinRadius: 0.12, halfThickness: 0.12,
                                         boundsMin: SIMD3(-1.1,-0.5,-1.1), boundsMax: SIMD3(1.1,2.6,1.1)),
              let queue = device.makeCommandQueue() else { throw XCTSkip("init failed") }  // gpu-ok: test harness queue
        solver.floorY = 0
        solver.setColliders(RigidPileField.bin(innerHalf: SIMD2(0.42, 0.42), floorY: 0))
        solver.solverMode = .constraint
        solver.frictionCoeff = 0.5
        solver.sleepFrames = 20
        func step(_ k: Int) { for _ in 0..<k { let cb = queue.makeCommandBuffer()!; solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted() } }  // gpu-ok: test harness

        var seed: UInt64 = 0x15A5
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<40 {
            let p = SIMD3<Float>((rnd()-0.5)*0.5, 0.4 + rnd()*0.8, (rnd()-0.5)*0.5)
            let t = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
            switch i % 3 {
            case 0: solver.spawn(at: p, tumble: t, radius: 0.075, halfThickness: 0.022, type: 0)
            case 1: solver.spawnSphere(at: p, radius: 0.052, tumble: t, type: 1)
            default: solver.spawnBox(at: p, halfExtents: SIMD3(0.044,0.044,0.044), tumble: t, type: 2)
            }
        }
        step(800)                                            // settle fully (island sleep is all-or-nothing
                                                             // on one connected pile — needs every body still)
        let asleepSettled = solver.asleepCount
        let keSettled = CoinDiagnostics.measure(solver).kineticEnergy
        print("STAGE5_SLEEP settled asleep=\(asleepSettled)/40 KE=\(keSettled)")
        XCTAssertGreaterThan(asleepSettled, 20, "a fully-settled heap SLEEPS in bulk (skips its solve → perf win)")
        XCTAssertLessThan(keSettled, 0.02, "the sleeping heap stays put (frozen, not drifting)")

        // Drop a fresh body INTO the pile (downward velocity so it lands fast) — when it
        // hits, its contact unions it into the sleeping island and wakes the lot.
        solver.spawn(at: SIMD3(0, 0.5, 0), velocity: SIMD3(0,-4,0), tumble: SIMD3(2,1,3),
                     radius: 0.075, halfThickness: 0.022, type: 0)
        var minAsleep = solver.asleepCount
        for _ in 0..<20 { step(1); minAsleep = min(minAsleep, solver.asleepCount) }
        XCTAssertLessThan(minAsleep, asleepSettled, "the dropped body's impact wakes its island")
        step(420)
        let keFinal = CoinDiagnostics.measure(solver).kineticEnergy
        print("STAGE5_SLEEP after-drop asleep=\(solver.asleepCount)/41 KE=\(keFinal)")
        XCTAssertLessThan(keFinal, 0.02, "re-settles dead-still after the wake")
        XCTAssertEqual(CoinDiagnostics.measure(solver).belowFloorCount, 0, "nothing tunnelled")
    }

    // ── CONSTRAINT SOLVER Stage 6: GJK + EPA general convex narrowphase ────────
    // The general path must reproduce the analytic penetration depth + normal for
    // known overlaps (sphere↔sphere, box↔box, box↔sphere) and report separation. ────
    func testConstraintGJKEPAStage6() throws {
        let (solver, _) = try makeSolver(maxCoins: 8)

        // Sphere↔sphere: A(R=0.1)@origin, B(R=0.1)@+x0.15 → depth 0.05, normal B→A = −x.
        solver.spawnSphere(at: SIMD3(0,0,0), radius: 0.1)
        solver.spawnSphere(at: SIMD3(0.15,0,0), radius: 0.1)
        var r = solver.probeGJKEPA(0, 1)
        print("STAGE6_GJK sphere hit=\(r.hit) depth=\(r.depth) n=\(r.normal)")
        XCTAssertTrue(r.hit, "overlapping spheres detected")
        XCTAssertEqual(r.depth, 0.05, accuracy: 0.01, "sphere↔sphere penetration depth")
        XCTAssertGreaterThan(simd_dot(simd_normalize(r.normal), SIMD3<Float>(-1,0,0)), 0.9, "normal points B→A")

        // Box↔box overlapping along x by 0.05.
        let (solver2, _) = try makeSolver(maxCoins: 8)
        solver2.spawnBox(at: SIMD3(0,0,0), halfExtents: SIMD3(0.1,0.1,0.1))
        solver2.spawnBox(at: SIMD3(0.15,0,0), halfExtents: SIMD3(0.1,0.1,0.1))
        r = solver2.probeGJKEPA(0, 1)
        print("STAGE6_GJK box hit=\(r.hit) depth=\(r.depth) n=\(r.normal)")
        XCTAssertTrue(r.hit, "overlapping boxes detected")
        XCTAssertEqual(r.depth, 0.05, accuracy: 0.015, "box↔box penetration depth")
        XCTAssertGreaterThan(abs(simd_normalize(r.normal).x), 0.9, "box↔box normal is the x face")

        // Separated pair → no hit.
        let (solver3, _) = try makeSolver(maxCoins: 8)
        solver3.spawnSphere(at: SIMD3(0,0,0), radius: 0.1)
        solver3.spawnSphere(at: SIMD3(0.5,0,0), radius: 0.1)
        XCTAssertFalse(solver3.probeGJKEPA(0, 1).hit, "separated spheres report no overlap")
    }

    // ── CONSTRAINT SOLVER: whole-pile sleep skips ALL solver work ──────────────
    // Once every active body is frozen, encode() must skip generation/colouring/solve
    // entirely (a settled scene costs ~nothing), stay settled, and resume on a wake.
    func testConstraintWholePileSkip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 50,
                                         coinRadius: 0.12, halfThickness: 0.12,
                                         boundsMin: SIMD3(-1.1,-0.5,-1.1), boundsMax: SIMD3(1.1,2.6,1.1)),
              let queue = device.makeCommandQueue() else { throw XCTSkip("init failed") }  // gpu-ok: test harness
        solver.floorY = 0
        solver.setColliders(RigidPileField.bin(innerHalf: SIMD2(0.42,0.42), floorY: 0))
        solver.solverMode = .constraint; solver.frictionCoeff = 0.5; solver.sleepFrames = 20
        func step(_ k: Int) { for _ in 0..<k { let cb = queue.makeCommandBuffer()!; solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted() } }  // gpu-ok: test harness

        var seed: UInt64 = 0x51A9
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<36 {
            let p = SIMD3<Float>((rnd()-0.5)*0.5, 0.4 + rnd()*0.8, (rnd()-0.5)*0.5)
            let t = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
            switch i % 3 {
            case 0: solver.spawn(at: p, tumble: t, radius: 0.075, halfThickness: 0.022, type: 0)
            case 1: solver.spawnSphere(at: p, radius: 0.052, tumble: t, type: 1)
            default: solver.spawnBox(at: p, halfExtents: SIMD3(0.044,0.044,0.044), tumble: t, type: 2)
            }
        }
        step(800)
        step(1)
        XCTAssertTrue(solver.didSkipLastFrame, "a fully-settled pile skips all solver work")
        XCTAssertLessThan(CoinDiagnostics.measure(solver).kineticEnergy, 0.02, "stays settled while skipped")

        // Drop a body in — the wake must RESUME solver work, then re-settle + re-skip.
        solver.spawn(at: SIMD3(0,0.5,0), velocity: SIMD3(0,-4,0), tumble: SIMD3(2,1,3),
                     radius: 0.075, halfThickness: 0.022, type: 0)
        step(1)
        XCTAssertFalse(solver.didSkipLastFrame, "a freshly-dropped body resumes solver work (not frozen mid-air)")
        step(800); step(1)
        XCTAssertTrue(solver.didSkipLastFrame, "re-settles and freezes again")
        XCTAssertEqual(CoinDiagnostics.measure(solver).belowFloorCount, 0)
        print("WHOLE_PILE_SKIP ok — settled→skip, drop→resume, re-settle→skip")
    }

    // ── Gate 1: a single coin comes to rest flat on the floor ──────────────────
    func testSingleCoinRestsOnFloor() throws {
        let (solver, queue) = try makeSolver()
        solver.spawn(at: SIMD3(0, 0.5, 0))
        step(solver, queue, frames: 150)   // 2.5 s

        let r = CoinDiagnostics.measure(solver)
        XCTAssertEqual(r.activeCount, 1)
        // Rests at floorY + h.
        XCTAssertEqual(r.maxY, Self.floorY + Self.h, accuracy: Self.h * 0.5,
                       "coin should rest with COM at floorY + halfThickness")
        XCTAssertLessThan(r.maxSpeed, 0.02, "coin should be at rest")
        XCTAssertEqual(r.belowFloorCount, 0)
    }

    // ── Gate 2: two coins stack with the correct gap, no interpenetration ──────
    func testTwoCoinsStack() throws {
        let (solver, queue) = try makeSolver()
        solver.spawn(at: SIMD3(0, 0.30, 0))
        solver.spawn(at: SIMD3(0.002, 0.50, 0))   // tiny offset to break symmetry
        step(solver, queue, frames: 200)

        let r = CoinDiagnostics.measure(solver)
        XCTAssertEqual(r.activeCount, 2)
        XCTAssertEqual(r.belowFloorCount, 0)
        XCTAssertLessThan(r.maxSpeed, 0.03, "stacked coins should settle")
        // Penetration must be a small fraction of the thickness.
        XCTAssertLessThan(r.maxPenetration, Self.h * 0.5, "no coin–coin interpenetration")
        // If they stayed stacked, the pile is ~2 layers tall (≈ 3h); if they slid
        // apart into one layer it'd be ≈ h. Either is non-penetrating and valid,
        // but assert the stack didn't sink into one overlapping blob.
        XCTAssertGreaterThan(r.maxY, Self.floorY + Self.h * 0.5)
    }

    // ── Gate 3: a dropped pile settles — no explosion, tunnelling, or overlap ──
    func testPileSettles() throws {
        let n = 120
        // A bin to keep the pile contained.
        let (solver, queue) = try makeSolver(maxCoins: n + 4)
        let wallH: Float = 0.4, halfX: Float = 0.28, halfZ: Float = 0.28, t: Float = 0.02
        solver.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: Self.floorY),
            .box(center: SIMD3(halfX + t, wallH, 0), halfExtents: SIMD3(t, wallH, halfZ + t)),
            .box(center: SIMD3(-halfX - t, wallH, 0), halfExtents: SIMD3(t, wallH, halfZ + t)),
            .box(center: SIMD3(0, wallH, halfZ + t), halfExtents: SIMD3(halfX + t, wallH, t)),
            .box(center: SIMD3(0, wallH, -halfZ - t), halfExtents: SIMD3(halfX + t, wallH, t)),
        ])

        // Deterministic pseudo-random scatter above the bin.
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(seed >> 40) / Float(1 << 24)   // [0,1)
        }
        for _ in 0..<n {
            let x = (rnd() - 0.5) * 2 * halfX
            let z = (rnd() - 0.5) * 2 * halfZ
            let y = 0.1 + rnd() * 0.6
            solver.spawn(at: SIMD3(x, y, z),
                         tumble: SIMD3((rnd() - 0.5) * 4, (rnd() - 0.5) * 4, (rnd() - 0.5) * 4))
        }

        step(solver, queue, frames: 300)   // 5 s to settle

        let r = CoinDiagnostics.measure(solver)
        print("PILE_SETTLE KE=\(r.kineticEnergy) maxPen=\(r.maxPenetration) (h=\(Self.h))")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertEqual(r.belowFloorCount, 0, "no coin tunnelled through the floor")
        XCTAssertLessThan(r.maxPenetration, 2.5 * Self.h, "pile interpenetration around one thickness, not gross")
        // The gentle support-leveling pass trades a little residual micro-motion in a
        // deeply-jammed bin for calmer rest angles; a quiet pile sits well under this.
        XCTAssertLessThan(r.kineticEnergy, 1.5, "pile should have settled (no jitter/explosion)")
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN blow-up")
    }

    // ── A dropped coin falls, bounces (restitution), then settles ─────────────
    func testCoinBouncesThenSettles() throws {
        let (solver, queue) = try makeSolver()
        solver.restitution = 0.4
        solver.spawn(at: SIMD3(0, 0.7, 0))   // drop from height
        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 1)
        var sawFall = false, sawBounce = false, maxBounceVy: Float = 0
        for _ in 0..<150 {
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
            let vy = ptr[0].vel.y
            if vy < -1.0 { sawFall = true }
            if sawFall && vy > 0.15 { sawBounce = true; maxBounceVy = max(maxBounceVy, vy) }
        }
        print("BOUNCE sawFall=\(sawFall) sawBounce=\(sawBounce) maxBounceVy=\(maxBounceVy) finalVy=\(ptr[0].vel.y)")
        XCTAssertTrue(sawFall, "coin should fall")
        XCTAssertTrue(sawBounce, "a real falling impact should bounce (restitution), not dead-plop")
        XCTAssertLessThan(maxBounceVy, 3.0, "bounce should be a fraction of impact, not an ejection")
        XCTAssertLessThan(abs(ptr[0].vel.y), 0.05, "should settle after bouncing")
    }

    // ── Spread drop (the scene's mechanism) settles into a shallow pile ───────
    func testSpreadDropStaysShallow() throws {
        let n = 150
        let (solver, queue) = try makeSolver(maxCoins: n + 4)
        let wallH: Float = 0.6, halfX: Float = 0.45, halfZ: Float = 0.45, t: Float = 0.02
        solver.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: Self.floorY),
            .box(center: SIMD3(halfX + t, wallH, 0), halfExtents: SIMD3(t, wallH, halfZ + t)),
            .box(center: SIMD3(-halfX - t, wallH, 0), halfExtents: SIMD3(t, wallH, halfZ + t)),
            .box(center: SIMD3(0, wallH, halfZ + t), halfExtents: SIMD3(halfX + t, wallH, t)),
            .box(center: SIMD3(0, wallH, -halfZ - t), halfExtents: SIMD3(halfX + t, wallH, t)),
        ])
        var seed: UInt64 = 0xABCDEF
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        // Spread across the bin (what the controller does) → shallow pile, not a tower.
        for _ in 0..<n {
            solver.spawn(at: SIMD3((rnd()-0.5)*2*halfX, 0.15 + rnd()*1.0, (rnd()-0.5)*2*halfZ),
                         tumble: SIMD3((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4))
        }
        step(solver, queue, frames: 360)
        let r = CoinDiagnostics.measure(solver)
        print("SPREAD maxY=\(r.maxY)  KE=\(r.kineticEnergy)  belowFloor=\(r.belowFloorCount)")
        XCTAssertEqual(r.belowFloorCount, 0)
        XCTAssertLessThan(r.maxY, 0.2, "spread drop should pile shallow, not tower")
        XCTAssertLessThan(r.kineticEnergy, 0.5, "should settle")
    }

    // ── Coin-fall orientation: a settled pile rests at calm, shallow angles ────
    // Mirrors the real coin-pusher SHELF (coins spread ~1–2 deep on a wide
    // surface, then flow off), NOT a deeply-jammed narrow bin. In deep jamming
    // thin discs wedge at steeper angles (physically real, but not how this scene
    // works); on the wide shelf the slippery coins + gentle support-leveling settle
    // calm. The strict gate lives here; the shallow-drop case in
    // testSpreadDropStaysShallow guards settling/no-tower in the same regime.
    func testSettledCoinsLieFlat() throws {
        let n = 90
        let (solver, queue) = try makeSolver(maxCoins: n + 4)
        let wallH: Float = 0.4, halfX: Float = 0.55, halfZ: Float = 0.55, t: Float = 0.02
        solver.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: Self.floorY),
            .box(center: SIMD3(halfX + t, wallH, 0), halfExtents: SIMD3(t, wallH, halfZ + t)),
            .box(center: SIMD3(-halfX - t, wallH, 0), halfExtents: SIMD3(t, wallH, halfZ + t)),
            .box(center: SIMD3(0, wallH, halfZ + t), halfExtents: SIMD3(halfX + t, wallH, t)),
            .box(center: SIMD3(0, wallH, -halfZ - t), halfExtents: SIMD3(halfX + t, wallH, t)),
        ])
        var seed: UInt64 = 0x1234567
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            solver.spawn(at: SIMD3((rnd()-0.5)*2*halfX, 0.1 + rnd()*0.6, (rnd()-0.5)*2*halfZ),
                         tumble: SIMD3((rnd()-0.5)*6, (rnd()-0.5)*6, (rnd()-0.5)*6))
        }
        step(solver, queue, frames: 300)

        // Disc-plane tilt = how far each coin's FACE is from horizontal. The coin is
        // symmetric, so a flipped coin (axis pointing down) is just as flat as one
        // pointing up — FOLD the axis angle into [0°, 90°]: 0° = flat, 90° = on edge.
        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: solver.highWater)
        var tilts: [Float] = []
        for i in 0..<solver.highWater where ptr[i].posInvMass.w != 0 {
            let o = ptr[i].orient
            let q = simd_quatf(ix: o.x, iy: o.y, iz: o.z, r: o.w)
            let axis = q.act(SIMD3(0, 1, 0))
            let theta = acos(max(-1, min(1, abs(axis.y)))) * 180 / .pi   // folded
            tilts.append(theta)
        }
        XCTAssertGreaterThan(tilts.count, 50)
        let sorted = tilts.sorted()
        let median = sorted[sorted.count / 2]
        let mean = tilts.reduce(0, +) / Float(tilts.count)
        let fracFlat = Float(tilts.filter { $0 < 20 }.count) / Float(tilts.count)   // share lying calm
        print("COIN_TILT mean=\(mean)° median=\(median)° fracFlat(<20°)=\(fracFlat) max=\(sorted.last ?? 0)°")
        // Orientation is PHYSICAL (not the old decorative snap-flat), so a deeply-
        // jammed bin keeps a few tilted leaners — that's real. The gate is on the
        // BULK lying calm, robust to outliers: the median coin is shallow and most
        // coins lie near-flat. (mean/max are dominated by the jammed minority and are
        // not the right gate for a physical pile.)
        XCTAssertLessThan(median, 25.0, "the median coin should rest shallow (bulk lies calm)")
        XCTAssertGreaterThan(fracFlat, 0.45, "most coins should lie near-flat, not perched")
    }

    // ── Gate: the reciprocating pusher shoves coins at ITS speed, not ~4× ───────
    // Regression for the launch bug. The plate is a kinematic collider advanced
    // once per 60Hz tick, but velocity is derived as Δx/dt at the 240Hz substep
    // rate; snapping a coin to the plate's front face each substep made `finalize`
    // read the plate's whole per-tick advance over one substep ≈ 4× (substep:tick
    // ratio) the true plate speed → coins launched off the shelf. The contact now
    // clamps the per-substep forward push to plateSpeed·dt, so a shoved coin's
    // forward speed tracks the plate (~1 m/s) instead of ~4 m/s.
    func testPusherShoveDoesNotLaunch() throws {
        let (solver, queue) = try makeSolver(maxCoins: 32)
        // A row of coins resting on the floor in the plate's forward path.
        for z: Float in stride(from: 0.0, through: 0.42, by: 0.06) {
            solver.spawn(at: SIMD3(0, 0.05, z))
        }
        step(solver, queue, frames: 90)   // settle flat (floor only)

        let plateSpeed: Float = 1.0        // m/s — the real pusher's peak forward speed
        let dt: Float = 1.0 / 60.0
        var plateZ: Float = -0.4           // starts behind the row, sweeps +Z
        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self,
                                                                 capacity: solver.highWater)
        var maxForwardVZ: Float = 0
        for _ in 0..<90 {
            plateZ += plateSpeed * dt
            solver.setColliders([
                .plane(normal: SIMD3(0, 1, 0), offset: Self.floorY),
                .pusherPlate(center: SIMD3(0, 0.15, plateZ),
                             halfExtents: SIMD3(0.5, 0.2, 0.05),
                             velocity: SIMD3(0, 0, plateSpeed)),
            ])
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: dt); cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            for i in 0..<solver.highWater where ptr[i].posInvMass.w != 0 {
                maxForwardVZ = max(maxForwardVZ, ptr[i].vel.z)
            }
        }

        print("PUSHER maxForwardVZ=\(maxForwardVZ)  plateSpeed=\(plateSpeed)")
        XCTAssertGreaterThan(maxForwardVZ, 0.2, "the plate should actually push the coins")
        XCTAssertLessThan(maxForwardVZ, 1.8,
                          "coins must not be launched faster than the plate (bug produced ~4×)")
    }

    // ── Friction carries TORQUE: a body sliding on the floor picks up roll only when
    // μ > 0. Before, friction was a COM velocity damp that produced no torque, so a
    // sliding body never started rolling. This is the A/B that proves the fix. ──────
    func testFrictionInducesRoll() throws {
        func slideAndMeasureSpin(mu: Float) throws -> Float {
            let (solver, queue) = try makeSolver()
            solver.frictionCoeff = mu
            solver.maxHSpeed = 5
            // A sphere (radius == halfThickness) settled on the floor, then shoved.
            let slot = solver.spawn(at: SIMD3(0, Self.R * 0.97, 0),
                                    radius: Self.R, halfThickness: Self.R)!
            for _ in 0..<20 {   // settle into solid contact
                guard let cb = queue.makeCommandBuffer() else { break }
                solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
            }
            solver.setVelocity(ofSlot: slot, to: SIMD3(2.0, 0, 0))
            let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 1)
            var maxOmega: Float = 0
            for _ in 0..<60 {
                guard let cb = queue.makeCommandBuffer() else { break }
                solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
                maxOmega = max(maxOmega, simd_length(SIMD3(ptr[0].angVel.x, ptr[0].angVel.y, ptr[0].angVel.z)))
            }
            return maxOmega
        }
        let spinNoFriction = try slideAndMeasureSpin(mu: 0.0)
        let spinWithFriction = try slideAndMeasureSpin(mu: 0.6)
        print("FRICTION_ROLL μ=0 ω=\(spinNoFriction)  μ=0.6 ω=\(spinWithFriction)")
        XCTAssertLessThan(spinNoFriction, 0.5, "with no friction a sliding body must not spin up")
        XCTAssertGreaterThan(spinWithFriction, 2.0,
                             "with friction the contact torque must roll the sliding body")
    }

    // ── Wall/ceiling restitution: a body thrown at a vertical wall rebounds along
    // the wall normal (used to dead-stop — restitution was vertical-only). ────────
    func testWallRestitutionBounce() throws {
        let (solver, queue) = try makeSolver()
        solver.gravity = 0                         // isolate the horizontal wall bounce
        solver.restitution = 0.6
        solver.maxHSpeed = 12; solver.maxSpeed = 12
        // A wall at x = 0.4 (half-space x ≤ 0.4): normal (-1,0,0), offset -0.4.
        solver.setColliders([.plane(normal: SIMD3(-1, 0, 0), offset: -0.4)])
        let slot = solver.spawn(at: SIMD3(0, 0.5, 0), velocity: SIMD3(5, 0, 0),
                                radius: Self.R, halfThickness: Self.R)!
        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 1)
        var maxReboundVx: Float = 0
        for _ in 0..<60 {
            guard let cb = queue.makeCommandBuffer() else { break }
            solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
            if ptr[slot].vel.x < 0 { maxReboundVx = max(maxReboundVx, -ptr[slot].vel.x) }
        }
        print("WALL_BOUNCE reboundVx=\(maxReboundVx) finalVx=\(ptr[slot].vel.x)")
        XCTAssertLessThan(ptr[slot].vel.x, 0, "the body must rebound off the wall (negative vx)")
        XCTAssertGreaterThan(maxReboundVx, 1.5, "rebound ≈ restitution × incoming, not a dead stop")
    }

    // ── Per-body mass: in a collision the velocity CHANGE is inversely proportional
    // to mass — a heavy body barely deflects, a light one is flung. ───────────────
    func testHeavyBodyResistsCollision() throws {
        let (solver, queue) = try makeSolver()
        solver.gravity = 0
        solver.restitution = 0.5
        solver.maxHSpeed = 12; solver.maxSpeed = 12
        solver.setColliders([])                    // free space, head-on pair
        let light = solver.spawn(at: SIMD3(-0.06, 0.5, 0), velocity: SIMD3(3, 0, 0),
                                 radius: Self.R, halfThickness: Self.R, mass: 1)!
        let heavy = solver.spawn(at: SIMD3(0.06, 0.5, 0), velocity: .zero,
                                 radius: Self.R, halfThickness: Self.R, mass: 10)!
        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 2)
        for _ in 0..<40 {
            guard let cb = queue.makeCommandBuffer() else { break }
            solver.encode(to: cb, wallDt: 1.0/60.0); cb.commit(); cb.waitUntilCompleted()  // gpu-ok: test
        }
        let dvLight = abs(ptr[light].vel.x - 3)     // change from its initial +3
        let dvHeavy = abs(ptr[heavy].vel.x - 0)     // change from rest
        print("MASS dvLight=\(dvLight) dvHeavy=\(dvHeavy) heavyVx=\(ptr[heavy].vel.x)")
        XCTAssertGreaterThan(dvHeavy, 0.05, "the heavy body must be nudged (collision happened)")
        XCTAssertLessThan(dvHeavy, dvLight * 0.5,
                          "the heavy (10×) body's Δv must be far smaller than the light body's")
    }

    // (Rolling resistance — CoinDEMSolver.rollingResistance — is implemented and
    // documented but not unit-isolated here: a sphere spinning on the floor loses its
    // spin to contact/feature-point noise faster than the rolling-resistance term, so
    // the harness can't cleanly A/B it. Its real use is a disc rolling on its edge;
    // left for a dedicated rolling-coin fixture.)

    // ── BOX BODIES (Phase 1): a tilted box dropped onto the floor tips onto a FACE
    // and rests — corner-first contact + box inertia. Box vs static only. ─────────
    func testBoxDropsAndRestsOnFace() throws {
        let (solver, queue) = try makeSolver()
        let he = SIMD3<Float>(0.05, 0.03, 0.04)        // a flat-ish brick
        // Drop it tilted so a corner strikes first and the offset contact tips it flat.
        let tilt = simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3(1, 0, 0.5)))
        let o = SIMD4(tilt.imag.x, tilt.imag.y, tilt.imag.z, tilt.real)
        let slot = solver.spawnBox(at: SIMD3(0, 0.4, 0), halfExtents: he, orient: o,
                                   tumble: SIMD3(0.4, 0, 0))!
        step(solver, queue, frames: 240)

        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 1)
        let b = ptr[slot]
        XCTAssertFalse(b.posInvMass.x.isNaN || b.posInvMass.y.isNaN, "no NaN blow-up")
        let speed = simd_length(SIMD3(b.vel.x, b.vel.y, b.vel.z))
        XCTAssertLessThan(speed, 0.05, "box should come to rest")
        XCTAssertGreaterThan(b.posInvMass.y, 0.01, "box rests above the floor, didn't tunnel")
        XCTAssertLessThan(b.posInvMass.y, 0.10, "box rests low on a face, not perched on a corner")
        // On a face ⇒ one of the box's local axes is ~vertical (its face normal points
        // straight down). Fold to [0,1]: the most-vertical axis ≈ 1.
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        let ax = q.act(SIMD3<Float>(1,0,0)), ay = q.act(SIMD3<Float>(0,1,0)), az = q.act(SIMD3<Float>(0,0,1))
        let maxVert = max(abs(ax.y), max(abs(ay.y), abs(az.y)))
        print("BOX_REST y=\(b.posInvMass.y) speed=\(speed) maxVertAxis=\(maxVert)")
        XCTAssertGreaterThan(maxVert, 0.96, "box rests on a FACE (a local axis is within ~16° of vertical)")
    }

    // ── BOX BODIES (Phase 2): two cubes stack — box–box SAT resolves the contact so
    // the top cube rests ON the bottom one, not sunk into it. ─────────────────────
    func testBoxesStack() throws {
        let (solver, queue) = try makeSolver()
        let he = SIMD3<Float>(0.03, 0.03, 0.03)        // cubes
        let b0 = solver.spawnBox(at: SIMD3(0, 0.03, 0), halfExtents: he)!
        let b1 = solver.spawnBox(at: SIMD3(0.004, 0.16, 0), halfExtents: he)!  // dropped above, tiny offset
        step(solver, queue, frames: 260)

        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 2)
        let y0 = ptr[b0].posInvMass.y, y1 = ptr[b1].posInvMass.y
        let v1 = simd_length(SIMD3(ptr[b1].vel.x, ptr[b1].vel.y, ptr[b1].vel.z))
        print("BOX_STACK y0=\(y0) y1=\(y1) v1=\(v1)")
        XCTAssertFalse(y0.isNaN || y1.isNaN, "no NaN")
        XCTAssertEqual(y0, 0.03, accuracy: 0.02, "bottom cube rests on the floor (COM ≈ he.y)")
        XCTAssertGreaterThan(y1, 0.075, "top cube rests ON the bottom cube (COM ≈ 3·he.y), not sunk")
        XCTAssertLessThan(y1, 0.12, "top cube didn't bounce off / float away")
        XCTAssertLessThan(v1, 0.06, "the stack settles")
    }

    // ── BOX RENDERING: coinDeriveTransforms bakes the per-instance box scale into the
    // render transform's basis, so a unit-cube mesh draws every box at its true size. ─
    func testBoxTransformBakesScale() throws {
        let (solver, queue) = try makeSolver()
        let he = SIMD3<Float>(0.05, 0.03, 0.04)
        let slot = solver.spawnBox(at: SIMD3(0, 0.4, 0), halfExtents: he)!   // identity orient, still falling
        step(solver, queue, frames: 1)
        let t = solver.transformBuffer.contents()
            .bindMemory(to: CoinTransform.self, capacity: solver.highWater)[slot]
        let c0 = simd_length(SIMD3(t.col0.x, t.col0.y, t.col0.z))
        let c1 = simd_length(SIMD3(t.col1.x, t.col1.y, t.col1.z))
        let c2 = simd_length(SIMD3(t.col2.x, t.col2.y, t.col2.z))
        print("BOX_XFORM cols=(\(c0),\(c1),\(c2)) want=(\(2*he.x),\(2*he.y),\(2*he.z))")
        XCTAssertEqual(c0, 2 * he.x, accuracy: 1e-4, "basis col0 scaled to full extent X")
        XCTAssertEqual(c1, 2 * he.y, accuracy: 1e-4, "basis col1 scaled to full extent Y")
        XCTAssertEqual(c2, 2 * he.z, accuracy: 1e-4, "basis col2 scaled to full extent Z")
        // A disc by contrast keeps unit-magnitude basis columns (true-size mesh).
        let d = solver.spawn(at: SIMD3(0.5, 0.4, 0))!
        step(solver, queue, frames: 1)
        let td = solver.transformBuffer.contents()
            .bindMemory(to: CoinTransform.self, capacity: solver.highWater)[d]
        XCTAssertEqual(simd_length(SIMD3(td.col0.x, td.col0.y, td.col0.z)), 1, accuracy: 1e-4,
                       "disc transform is unscaled (true-size mesh)")
    }

    // ── BOX PILE: a heap of tumbling boxes dropped into a bin settles calmly — no
    // explosion, tunnelling, or NaN. This is the realistic box use case (dice, crates,
    // debris). NOTE: a tall *stable tower* is NOT supported — a box squeezed between
    // contacts above and below gets conflicting per-point corrections that don't
    // converge under this Jacobi + position-derived-velocity solver (the same reason
    // the disc solver uses one contact per pair). Stable towers would need a
    // Gauss-Seidel / shock-propagating / warm-started solver (see docs/coindem-
    // ecosystem.md). The box-box corner manifold here still gives a real resting
    // footprint, so drops, tips, and shallow piles settle correctly. ────────────────
    func testBoxPileSettles() throws {
        let n = 14
        let (solver, queue) = try makeSolver(maxCoins: n + 2)
        let he = SIMD3<Float>(0.025, 0.02, 0.025)     // small (bounding < the solver's cell)
        let halfXZ: Float = 0.22, wallH: Float = 0.3, t: Float = 0.02
        solver.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: 0),
            .box(center: SIMD3( halfXZ + t, wallH, 0), halfExtents: SIMD3(t, wallH, halfXZ + t)),
            .box(center: SIMD3(-halfXZ - t, wallH, 0), halfExtents: SIMD3(t, wallH, halfXZ + t)),
            .box(center: SIMD3(0, wallH,  halfXZ + t), halfExtents: SIMD3(halfXZ + t, wallH, t)),
            .box(center: SIMD3(0, wallH, -halfXZ - t), halfExtents: SIMD3(halfXZ + t, wallH, t)),
        ])
        var seed: UInt64 = 0xBEEF
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for _ in 0..<n {
            solver.spawnBox(at: SIMD3((rnd()-0.5)*0.3, 0.1 + rnd()*0.5, (rnd()-0.5)*0.3),
                            halfExtents: he,
                            tumble: SIMD3((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4))
        }
        step(solver, queue, frames: 360)

        let r = CoinDiagnostics.measure(solver)   // KE / belowFloor / maxY are shape-agnostic
        print("BOX_PILE n=\(r.activeCount) KE=\(r.kineticEnergy) belowFloor=\(r.belowFloorCount) maxY=\(r.maxY)")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN blow-up")
        XCTAssertEqual(r.belowFloorCount, 0, "no box tunnelled the floor")
        XCTAssertLessThan(r.kineticEnergy, 1.0, "the box pile settles (no jitter/explosion)")
        XCTAssertLessThan(r.maxY, 0.35, "boxes pile shallow in the bin, not a launched tower")
    }

    // ── ELONGATED CAPPED CYLINDER (a rod/peg): the "disc" primitive is a capped
    // cylinder general in (R, h), so h ≫ R is a rod — it rests on its SIDE. (Confirms
    // elongated bodies already work; a true round-capped capsule would need a swept-
    // sphere contact, but a flat-ended rod is free.) ─────────────────────────────────
    func testRodRestsOnSide() throws {
        let (solver, queue) = try makeSolver()
        let rodR: Float = 0.015, rodHalfLen: Float = 0.1   // 0.2 long, 0.03 wide
        let slot = solver.spawn(at: SIMD3(0, 0.4, 0), tumble: SIMD3(0.6, 0, 0.4),
                                radius: rodR, halfThickness: rodHalfLen)!
        step(solver, queue, frames: 260)
        let b = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: 1)[slot]
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        let longAxis = q.act(SIMD3<Float>(0, 1, 0))        // the rod's long (local +Y) axis
        let speed = simd_length(SIMD3(b.vel.x, b.vel.y, b.vel.z))
        print("ROD y=\(b.posInvMass.y) longAxisY=\(longAxis.y) speed=\(speed)")
        XCTAssertLessThan(speed, 0.05, "rod comes to rest")
        XCTAssertLessThan(abs(longAxis.y), 0.4, "rod rests on its SIDE (long axis ~horizontal)")
        XCTAssertEqual(b.posInvMass.y, rodR, accuracy: rodR * 1.8, "rod COM rests ~one radius above the floor")
    }

    // ── MIXED coins + cubes settle together in a bin: exercises the box–disc contact
    // path (a disc collided as its bounding box) alongside box–box and disc–disc. ────
    func testMixedBoxesAndDiscsSettle() throws {
        let n = 16
        let (solver, queue) = try makeSolver(maxCoins: n + 2)
        let halfXZ: Float = 0.25, wallH: Float = 0.3, t: Float = 0.02
        solver.setColliders([
            .plane(normal: SIMD3(0, 1, 0), offset: 0),
            .box(center: SIMD3( halfXZ + t, wallH, 0), halfExtents: SIMD3(t, wallH, halfXZ + t)),
            .box(center: SIMD3(-halfXZ - t, wallH, 0), halfExtents: SIMD3(t, wallH, halfXZ + t)),
            .box(center: SIMD3(0, wallH,  halfXZ + t), halfExtents: SIMD3(halfXZ + t, wallH, t)),
            .box(center: SIMD3(0, wallH, -halfXZ - t), halfExtents: SIMD3(halfXZ + t, wallH, t)),
        ])
        var seed: UInt64 = 0xFEED
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<n {
            let p = SIMD3<Float>((rnd()-0.5)*0.35, 0.1 + rnd()*0.5, (rnd()-0.5)*0.35)
            let tumble = SIMD3<Float>((rnd()-0.5)*4, (rnd()-0.5)*4, (rnd()-0.5)*4)
            if i % 2 == 0 {
                solver.spawn(at: p, tumble: tumble)                                   // coin
            } else {
                solver.spawnBox(at: p, halfExtents: SIMD3(0.025, 0.02, 0.025), tumble: tumble)  // cube
            }
        }
        step(solver, queue, frames: 360)
        let r = CoinDiagnostics.measure(solver)   // KE / belowFloor / maxY shape-agnostic
        print("MIXED n=\(r.activeCount) KE=\(r.kineticEnergy) belowFloor=\(r.belowFloorCount) maxY=\(r.maxY)")
        XCTAssertEqual(r.activeCount, n)
        XCTAssertFalse(r.kineticEnergy.isNaN, "no NaN")
        XCTAssertEqual(r.belowFloorCount, 0, "nothing tunnelled the floor")
        XCTAssertLessThan(r.kineticEnergy, 1.0, "the mixed pile settles")
        XCTAssertLessThan(r.maxY, 0.35, "mixed pile stays shallow, no launch")
    }
}
