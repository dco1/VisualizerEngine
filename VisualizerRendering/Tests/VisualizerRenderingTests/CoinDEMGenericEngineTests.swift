import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Headless correctness gate for the GENERIC-engine extensions to CoinDEMSolver
/// (the "fully capable rigid-body engine" pass): per-body materials, exact
/// capsules, convex hulls, and generic joints — all on the constraint path.
/// Mirrors CoinDEMSolverTests' harness: CoinDEM.metal is compiled from source at
/// runtime (SwiftPM doesn't build metallibs) and injected via the test seam.
@MainActor
final class CoinDEMGenericEngineTests: XCTestCase {

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("CoinDEM.metal not found at \(shader.path)")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    /// Constraint-path solver over a flat floor. `maxDim` sizes the broadphase
    /// cell (pass the largest body half-dimension you plan to spawn).
    private func makeSolver(maxCoins: Int = 64,
                            radius: Float = 0.05,
                            maxDim: Float = 0.1,
                            boundsMin: SIMD3<Float> = SIMD3(-2, -0.5, -2),
                            boundsMax: SIMD3<Float> = SIMD3(2, 2, 2)) throws
        -> (CoinDEMSolver, MTLCommandQueue)
    {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib,
                                         maxCoins: maxCoins,
                                         coinRadius: radius, halfThickness: maxDim,
                                         boundsMin: boundsMin, boundsMax: boundsMax),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("solver/queue init failed") }
        solver.floorY = 0
        solver.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0)])
        solver.solverMode = .constraint
        return (solver, queue)
    }

    private func step(_ solver: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
                      perFrame: ((Int) -> Void)? = nil) {
        for f in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { return }
            solver.encode(to: cb, wallDt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            perFrame?(f)
        }
    }

    // ── Capsule: side rest ─────────────────────────────────────────────────────
    // A horizontal capsule dropped on the floor must settle with its COM at
    // exactly the cross-section radius (the 2-endpoint-probe manifold holding it
    // level), stay level, and go still.
    func testCapsuleRestsOnSideAtRadius() throws {
        let (solver, queue) = try makeSolver(radius: 0.03, maxDim: 0.09)
        let r: Float = 0.03, hl: Float = 0.06
        // Axis along world X: rotate local +Y → +X (quarter turn about Z).
        let q = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))
        let slot = solver.spawnCapsule(at: SIMD3(0, 0.25, 0), radius: r, halfLength: hl,
                                       orient: SIMD4(q.imag, q.real))
        XCTAssertNotNil(slot)
        step(solver, queue, frames: 240)

        let p = solver.position(of: slot!)!
        let v = solver.velocity(of: slot!)!
        XCTAssertEqual(p.y, r, accuracy: r * 0.25,
                       "side-lying capsule rests with COM at its cross-section radius")
        XCTAssertLessThan(simd_length(v), 0.02, "capsule at rest")
        // Still lying flat: the body axis stays horizontal.
        let axis = simd_act(solver.orientation(of: slot!)!, SIMD3<Float>(0, 1, 0))
        XCTAssertLessThan(abs(axis.y), 0.25, "capsule stayed on its side (didn't tip or sink)")
    }

    // ── Capsule: crossed stack ────────────────────────────────────────────────
    // A capsule dropped crosswise onto a resting capsule must stack (top COM ≈ 3r)
    // rather than sink through or over-separate — the capsule↔capsule seg-seg
    // contact at work.
    func testCapsulesStackCrossed() throws {
        let (solver, queue) = try makeSolver(radius: 0.03, maxDim: 0.09)
        let r: Float = 0.03, hl: Float = 0.05
        let qx = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))   // axis → X
        let qz = simd_quatf(angle:  .pi / 2, axis: SIMD3(1, 0, 0))   // axis → Z
        let bottom = solver.spawnCapsule(at: SIMD3(0, 0.10, 0), radius: r, halfLength: hl,
                                         orient: SIMD4(qx.imag, qx.real))
        let top = solver.spawnCapsule(at: SIMD3(0.001, 0.30, 0), radius: r, halfLength: hl,
                                      orient: SIMD4(qz.imag, qz.real))
        XCTAssertNotNil(bottom); XCTAssertNotNil(top)
        step(solver, queue, frames: 300)

        let pb = solver.position(of: bottom!)!
        let pt = solver.position(of: top!)!
        XCTAssertEqual(pb.y, r, accuracy: r * 0.3, "bottom capsule on the floor")
        XCTAssertEqual(pt.y, 3 * r, accuracy: r * 0.5,
                       "top capsule rests ON the bottom one (COM ≈ 3r), no sink-through")
        let stats = solver.measurePenetration(threshold: r * 0.15)
        XCTAssertLessThan(stats.maxPenetration, r * 0.4, "no deep capsule↔capsule overlap")
    }

    // ── Capsule ↔ sphere ──────────────────────────────────────────────────────
    // A sphere dropped squarely onto a side-lying capsule contacts its rounded
    // surface (segment + radius): it must not sink in, and it ends up beside the
    // capsule on the floor (spheres roll off round things — that's physics).
    func testSphereRollsOffCapsuleWithoutPenetrating() throws {
        let (solver, queue) = try makeSolver(radius: 0.04, maxDim: 0.1)
        let r: Float = 0.03, hl: Float = 0.06, rs: Float = 0.04
        let q = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))
        solver.spawnCapsule(at: SIMD3(0, 0.15, 0), radius: r, halfLength: hl,
                            orient: SIMD4(q.imag, q.real))
        let ball = solver.spawnSphere(at: SIMD3(0.004, 0.4, 0), radius: rs)
        XCTAssertNotNil(ball)
        var minBallY: Float = .greatestFiniteMagnitude
        step(solver, queue, frames: 300) { _ in
            if let p = self.ballPos(solver, ball!) { minBallY = min(minBallY, p.y) }
        }
        // While interacting, the ball's COM never dipped meaningfully below the
        // capsule-top contact height (2r + rs would be resting exactly on top; a
        // graze during roll-off may ride lower but must clear the sphere's own
        // floor-rest height minus a small slop — i.e. no tunnelling INTO the capsule).
        XCTAssertGreaterThan(minBallY, rs * 0.8, "ball never sank into the capsule or floor")
        let stats = solver.measurePenetration(threshold: r * 0.15)
        XCTAssertLessThan(stats.maxPenetration, r * 0.5, "sphere↔capsule stayed separated")
    }

    private func ballPos(_ solver: CoinDEMSolver, _ slot: Int) -> SIMD3<Float>? {
        solver.position(of: slot)
    }

    // ── Convex hulls ──────────────────────────────────────────────────────────
    // Registration: interior points dropped, exact solid inertia diagonalized.
    func testHullRegistrationDropsInteriorAndFindsInertia() throws {
        let (solver, _) = try makeSolver()
        let h: Float = 0.05
        var pts: [SIMD3<Float>] = []
        for i in 0..<8 {
            pts.append(SIMD3((i & 1) != 0 ? h : -h, (i & 2) != 0 ? h : -h, (i & 4) != 0 ? h : -h))
        }
        pts.append(.zero)                                 // interior — must be dropped
        pts.append(SIMD3(0.01, 0.0, -0.01))               // interior — must be dropped
        guard let hull = solver.registerHull(vertices: pts) else {
            return XCTFail("cube registration failed")
        }
        XCTAssertEqual(hull.vertices.count, 8, "interior points dropped, 8 corners kept")
        XCTAssertEqual(hull.boundingRadius, h * sqrt(3), accuracy: 1e-4)
        // Solid cube inertia per unit mass: I = (2/3)h² each axis → k = 1/I.
        let expected = 1.0 / ((2.0 / 3.0) * h * h)
        let prepK = 1.0 / hull.boundingRadius   // placeholder to keep tuple use simple
        _ = prepK
        // Reconstruct k from a spawned body's hullRef.
        let slot = solver.spawnHull(at: SIMD3(0, 1, 0), hull: hull)!
        let body = solver.coinBuffer.buffer.contents()
            .bindMemory(to: CoinBody.self, capacity: solver.maxCoins)[slot]
        for lane in [body.hullRef.y, body.hullRef.z, body.hullRef.w] {
            XCTAssertEqual(lane, Float(expected), accuracy: Float(expected) * 0.02,
                           "solid-cube inverse inertia (exact boundary integral)")
        }
    }

    // A hull CUBE dropped tilted must land, tip flat onto a face, and rest at
    // exactly its half-extent — behavioural parity with the analytic box path,
    // but through GJK/EPA + the clipped manifold.
    func testHullCubeRestsFlatLikeABox() throws {
        let (solver, queue) = try makeSolver(radius: 0.09, maxDim: 0.09)
        let h: Float = 0.05
        var pts: [SIMD3<Float>] = []
        for i in 0..<8 {
            pts.append(SIMD3((i & 1) != 0 ? h : -h, (i & 2) != 0 ? h : -h, (i & 4) != 0 ? h : -h))
        }
        let hull = solver.registerHull(vertices: pts)!
        let tilt = simd_quatf(angle: 0.35, axis: simd_normalize(SIMD3<Float>(1, 0, 0.4)))
        let slot = solver.spawnHull(at: SIMD3(0, 0.4, 0), hull: hull,
                                    orient: SIMD4(tilt.imag, tilt.real))!
        step(solver, queue, frames: 360)

        let p = solver.position(of: slot)!
        let v = solver.velocity(of: slot)!
        XCTAssertEqual(p.y, h, accuracy: h * 0.25, "hull cube rests at its half-extent")
        XCTAssertLessThan(simd_length(v), 0.03, "hull cube at rest")
        // Flat on a face: some local axis is within ~10° of world-up.
        let q = solver.orientation(of: slot)!
        let ax = [simd_act(q, SIMD3<Float>(1,0,0)), simd_act(q, SIMD3<Float>(0,1,0)), simd_act(q, SIMD3<Float>(0,0,1))]
        let bestUp = ax.map { abs($0.y) }.max()!
        XCTAssertGreaterThan(bestUp, 0.96, "tipped flat onto a face (a local axis ≈ world-up)")
    }

    // Two hull cubes stacked: without the clipped manifold a single EPA contact
    // point can't resist tipping — the top cube would wobble off. It must stay.
    func testHullCubesStack() throws {
        let (solver, queue) = try makeSolver(radius: 0.09, maxDim: 0.09)
        let h: Float = 0.05
        var pts: [SIMD3<Float>] = []
        for i in 0..<8 {
            pts.append(SIMD3((i & 1) != 0 ? h : -h, (i & 2) != 0 ? h : -h, (i & 4) != 0 ? h : -h))
        }
        let hull = solver.registerHull(vertices: pts)!
        solver.frictionCoeff = 0.6      // real cube-on-cube grip — frictionless cubes DON'T stack
        let bottom = solver.spawnHull(at: SIMD3(0, 0.06, 0), hull: hull)!
        let top    = solver.spawnHull(at: SIMD3(0.008, 0.25, 0), hull: hull)!
        step(solver, queue, frames: 360)

        let pb = solver.position(of: bottom)!
        let pt = solver.position(of: top)!
        print("HULL_STACK bottom=\(pb) top=\(pt)")
        XCTAssertEqual(pb.y, h, accuracy: h * 0.3, "bottom cube on the floor")
        XCTAssertEqual(pt.y, 3 * h, accuracy: h * 0.5, "top cube RESTS ON the bottom (COM ≈ 3h)")
        XCTAssertLessThan(abs(pt.x - pb.x), h * 0.8, "top didn't slide/wobble off the stack")
        let stats = solver.measurePenetration(threshold: h * 0.1)
        XCTAssertLessThan(stats.maxPenetration, h * 0.4, "no deep hull↔hull overlap")
    }

    // ── Capsule ↔ hull (roadmap: capsule↔hull 2-point manifold) ────────────────
    // A capsule resting crosswise on a flat hull "table" must settle level with
    // BOTH segment stations supported. Before the 2-point manifold, a hull pair
    // involving a capsule collapsed to the single midpoint EPA contact, which
    // can't resist the roll torque about that one point — the capsule teeters
    // instead of coming to rest.
    func testCapsuleRestsLevelOnHullTable() throws {
        let (solver, queue) = try makeSolver(radius: 0.15, maxDim: 0.15,
                                             boundsMin: SIMD3(-2, -0.5, -2),
                                             boundsMax: SIMD3(2, 2, 2))
        let th: Float = 0.03            // table half-thickness
        let tw: Float = 0.15            // table half-width/depth
        var tablePts: [SIMD3<Float>] = []
        for i in 0..<8 {
            tablePts.append(SIMD3((i & 1) != 0 ? tw : -tw, (i & 2) != 0 ? th : -th, (i & 4) != 0 ? tw : -tw))
        }
        let table = solver.registerHull(vertices: tablePts)!
        let tableSlot = solver.spawnHull(at: SIMD3(0, th, 0), hull: table)!

        let r: Float = 0.03, hl: Float = 0.10   // capsule spans most of the table, axis → X
        let q = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))
        let cap = solver.spawnCapsule(at: SIMD3(0, 2 * th + r + 0.05, 0), radius: r, halfLength: hl,
                                      orient: SIMD4(q.imag, q.real))!

        var tailAxisY: [Float] = []
        step(solver, queue, frames: 420) { f in
            if f > 360, let o = solver.orientation(of: cap) {
                tailAxisY.append(simd_act(o, SIMD3<Float>(0, 1, 0)).y)
            }
        }

        let pTable = solver.position(of: tableSlot)!
        let pCap = solver.position(of: cap)!
        let vCap = solver.velocity(of: cap)!
        let restY = pTable.y + th + r
        XCTAssertEqual(pCap.y, restY, accuracy: r * 0.5,
                       "capsule rests ON the table top (COM ≈ table top + r), no sink-through")
        XCTAssertLessThan(simd_length(vCap), 0.03, "capsule at rest")
        let axis = simd_act(solver.orientation(of: cap)!, SIMD3<Float>(0, 1, 0))
        XCTAssertLessThan(abs(axis.y), 0.25, "capsule stayed level (didn't tip off the table)")
        // No teeter: over the settled tail, the axis tilt barely moves — a
        // single-point manifold rocks the capsule back and forth forever.
        let tiltSpread = (tailAxisY.max() ?? 0) - (tailAxisY.min() ?? 0)
        XCTAssertLessThan(tiltSpread, 0.03, "capsule didn't keep rocking on the hull (2-point manifold holds it still)")
        let stats = solver.measurePenetration(threshold: r * 0.15)
        XCTAssertLessThan(stats.maxPenetration, r * 0.5, "no deep capsule↔hull overlap")
    }

    // ── Hull ↔ hull: per-point manifold depths (roadmap item 2) ────────────────
    // A tilted hull box overlapping a flat hull box must produce a manifold whose
    // contact points carry DIFFERENT depths — one edge digs in further than the
    // other. Before per-point depths, every clipped point shared the single EPA
    // witness depth, which is wrong (and a source of bias-recovery noise) on any
    // contact that isn't perfectly face-parallel. This is a Stage-1-style direct
    // contact-buffer probe (generateContactsNow), not a settle test.
    func testHullHullManifoldHasPerPointDepths() throws {
        let (solver, _) = try makeSolver(maxCoins: 8, radius: 0.3, maxDim: 0.3,
                                         boundsMin: SIMD3(-1, -0.5, -1), boundsMax: SIMD3(1, 1, 1))
        let bh: Float = 0.3   // bottom hull half-extent (flat "floor" slab)
        var bottomPts: [SIMD3<Float>] = []
        for i in 0..<8 {
            bottomPts.append(SIMD3((i & 1) != 0 ? bh : -bh, (i & 2) != 0 ? 0.03 : -0.03, (i & 4) != 0 ? bh : -bh))
        }
        let bottomHull = solver.registerHull(vertices: bottomPts)!
        let bottom = solver.spawnHull(at: SIMD3(0, 0.03, 0), hull: bottomHull)!

        // Top hull tilted ~0.9° about Z — shallow enough that BOTH hulls' support
        // sets still present ≥3 points each (a real face-face contact, not an
        // edge-only one), but tilted enough that the near/far corners penetrate
        // by clearly different amounts.
        let th: Float = 0.1
        var topPts: [SIMD3<Float>] = []
        for i in 0..<8 {
            topPts.append(SIMD3((i & 1) != 0 ? th : -th, (i & 2) != 0 ? th : -th, (i & 4) != 0 ? th : -th))
        }
        let topHull = solver.registerHull(vertices: topPts)!
        let tilt = simd_quatf(angle: 0.016, axis: SIMD3(0, 0, 1))
        let top = solver.spawnHull(at: SIMD3(0, 0.06 + th - 0.01, 0), hull: topHull,
                                   orient: SIMD4(tilt.imag, tilt.real))!

        let n = solver.generateContactsNow()
        XCTAssertGreaterThan(n, 0, "tilted overlapping hulls must generate contacts")
        let ptr = solver.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(1, n))
        var depths: [Float] = []
        let lo = min(bottom, top), hi = max(bottom, top)
        for i in 0..<n {
            let c = ptr[i]
            if Int(c.meta.x) == lo && Int(c.meta.y) == hi { depths.append(c.nrm.w) }
        }
        print("HULLHULL_TILT depths=\(depths)")
        XCTAssertGreaterThanOrEqual(depths.count, 2, "tilted face contact needs a real (not single-point) manifold")
        let spread = (depths.max() ?? 0) - (depths.min() ?? 0)
        XCTAssertGreaterThan(spread, 0.003,
                             "manifold points on a tilted face must carry DIFFERENT depths, not one shared EPA value")
    }

    // An OCTAHEDRON (a hull nothing like a box) dropped point-down must tip onto
    // a face and settle at the face's support height — real generic-shape physics.
    func testHullOctahedronSettlesOnAFace() throws {
        let (solver, queue) = try makeSolver(radius: 0.08, maxDim: 0.08)
        let r: Float = 0.06
        let pts: [SIMD3<Float>] = [
            SIMD3( r, 0, 0), SIMD3(-r, 0, 0),
            SIMD3(0,  r, 0), SIMD3(0, -r, 0),
            SIMD3(0, 0,  r), SIMD3(0, 0, -r),
        ]
        let hull = solver.registerHull(vertices: pts)!
        XCTAssertEqual(hull.vertices.count, 6)
        let slot = solver.spawnHull(at: SIMD3(0, 0.3, 0), hull: hull,
                                    tumble: SIMD3(0.4, 0, 0.2))!
        step(solver, queue, frames: 420)

        let p = solver.position(of: slot)!
        let v = solver.velocity(of: slot)!
        // Face-down rest height = distance from COM to a face plane = r/√3.
        let faceH = r / sqrt(3)
        print("HULL_OCTA rest y=\(p.y) faceH=\(faceH)")
        XCTAssertLessThan(simd_length(v), 0.03, "octahedron settled")
        XCTAssertEqual(p.y, faceH, accuracy: faceH * 0.35,
                       "rests on a FACE (COM at ~r/√3), not balanced on a vertex (r)")
    }

    // ── Perf gate: dense hull + capsule pile (roadmap item 4) ───────────────────
    // "Nobody has measured a 200-hull pile yet" — hull narrowphase is O(verts)
    // per support call, and this is the first real measurement. 100 hull bodies
    // (cubes + tetrahedra, sharing 2 registered hull shapes — maxHulls=64 caps
    // distinct REGISTRATIONS, not instances) + 100 capsules dropped into a bin.
    // Measures real GPU command-buffer time (gpuEndTime−gpuStartTime, the same
    // signal IlluminatoramaRenderer uses) per frame — a fence, not a spec, per
    // frame-time-gate.sh's own convention (numbers are machine-dependent).
    func testDenseHullCapsulePilePerf() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let solver = CoinDEMSolver(engine: engine, library: lib, maxCoins: 200,
                                         coinRadius: 0.06, halfThickness: 0.06,
                                         boundsMin: SIMD3(-1.1, -0.5, -1.1), boundsMax: SIMD3(1.1, 2.6, 1.1)),
              let queue = device.makeCommandQueue()  // gpu-ok: test harness queue
        else { throw XCTSkip("solver/queue init failed") }
        solver.floorY = 0
        solver.setColliders(RigidPileField.bin(innerHalf: SIMD2(0.55, 0.55), floorY: 0))
        solver.solverMode = .constraint
        solver.frictionCoeff = 0.5

        let h: Float = 0.045
        var cubePts: [SIMD3<Float>] = []
        for i in 0..<8 {
            cubePts.append(SIMD3((i & 1) != 0 ? h : -h, (i & 2) != 0 ? h : -h, (i & 4) != 0 ? h : -h))
        }
        let cube = solver.registerHull(vertices: cubePts)!
        let te: Float = 0.06
        let tetraPts: [SIMD3<Float>] = [
            SIMD3(te, te, te), SIMD3(te, -te, -te), SIMD3(-te, te, -te), SIMD3(-te, -te, te),
        ]
        let tetra = solver.registerHull(vertices: tetraPts)!

        var seed: UInt64 = 0xC0FFEE
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<100 {
            let p = SIMD3<Float>((rnd() - 0.5) * 0.9, 0.4 + rnd() * 1.8, (rnd() - 0.5) * 0.9)
            let t = SIMD3<Float>((rnd() - 0.5) * 4, (rnd() - 0.5) * 4, (rnd() - 0.5) * 4)
            _ = solver.spawnHull(at: p, hull: i % 2 == 0 ? cube : tetra, tumble: t)
        }
        for _ in 0..<100 {
            let p = SIMD3<Float>((rnd() - 0.5) * 0.9, 0.4 + rnd() * 1.8, (rnd() - 0.5) * 0.9)
            let t = SIMD3<Float>((rnd() - 0.5) * 4, (rnd() - 0.5) * 4, (rnd() - 0.5) * 4)
            let q = simd_quatf(angle: rnd() * 2 * .pi, axis: simd_normalize(SIMD3<Float>(rnd(), rnd(), rnd()) + 0.01))
            _ = solver.spawnCapsule(at: p, radius: 0.03, halfLength: 0.05,
                                    orient: SIMD4(q.imag, q.real), tumble: t)
        }

        var frameMs: [Double] = []
        for _ in 0..<420 {
            guard let cb = queue.makeCommandBuffer() else { break }
            solver.encode(to: cb, wallDt: 1.0 / 60.0)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            frameMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1000.0)
        }
        frameMs.sort()
        let p50 = frameMs[frameMs.count / 2]
        let p95 = frameMs[Int(Double(frameMs.count) * 0.95)]
        let worst = frameMs.last ?? 0
        let stats = solver.measurePenetration(threshold: 0.01)
        print("HULLCAPSULE_PERF frames=\(frameMs.count) p50ms=\(p50) p95ms=\(p95) worstMs=\(worst) maxPen=\(stats.maxPenetration) awake=\(solver.activeCount - solver.asleepCount)")
        // This is the FIRST measurement of this shape mix (see roadmap item 4)
        // — there's no established SLA to gate against yet, so this is a
        // regression FENCE against the current measured baseline (p95≈37ms,
        // worst≈38ms — the ACTIVE/falling phase, well over a 16.67ms 60fps
        // budget; the whole-pile-sleep skip makes the settled tail ~free,
        // which is why p50 is near zero). Catches it getting WORSE; does not
        // assert it's fast enough yet — that's color-compaction (per-colour
        // contact dispatch, noted as the next lever in this doc's Verification
        // section), tracked as its own follow-up, not silently baked in here.
        XCTAssertLessThan(p95, 60.0, "dense hull+capsule pile p95 frame time regression fence")
        XCTAssertLessThan(stats.maxPenetration, 0.03, "no runaway interpenetration under load")
    }

    // ── Speculative contacts (anti-tunneling) ─────────────────────────────────
    // A small sphere fired at 120 m/s crosses a same-size target's whole overlap
    // window between substeps (0.5 m/substep vs a 0.14 m window, phased to miss),
    // so WITHOUT the margin it sails straight through. With a speculative margin
    // the near-contact caps its approach to gap/dt and it lands ON the target.
    func testSpeculativeMarginStopsTunneling() throws {
        func fire(margin: Float) throws -> (projectileX: Float, targetX: Float) {
            let (solver, queue) = try makeSolver(radius: 0.07, maxDim: 0.2,
                                                 boundsMin: SIMD3(-4, -0.5, -1),
                                                 boundsMax: SIMD3(4, 1, 1))
            solver.maxSpeed = 200
            solver.restitution = 0
            solver.speculativeMargin = margin
            let target = solver.spawnSphere(at: SIMD3(0, 0.05, 0), radius: 0.05)!
            // 120 m/s → 0.5 m/substep; start phased so sampled gaps skip the window.
            let proj = solver.spawnSphere(at: SIMD3(-2.27, 0.05, 0), radius: 0.02,
                                          velocity: SIMD3(120, 0, 0))!
            step(solver, queue, frames: 30)
            return (solver.position(of: proj)!.x, solver.position(of: target)!.x)
        }
        let without = try fire(margin: 0)
        let with    = try fire(margin: 0.3)
        print("SPECULATIVE without=\(without) with=\(with)")
        XCTAssertGreaterThan(without.projectileX, 1.0,
                             "control: at 120 m/s the projectile tunnels straight through")
        XCTAssertLessThan(abs(without.targetX), 0.05, "control: target untouched")
        XCTAssertGreaterThan(with.targetX, 0.1,
                             "speculative: the target was actually HIT and carried forward")
    }

    // ── Speculative margin: box vs static AABB wall (roadmap item 5) ───────────
    // A thin box fired at 120 m/s tunnels a thin static AABB wall WITHOUT the
    // margin (control) — cdOrientedBoxPush is inside-only, and no corner ever
    // lands "inside" the wall between samples — and is stopped WITH it, via
    // cdOrientedBoxPushSpeculative's exterior closest-point near-contact.
    // ── Speculative margin: box vs static AABB wall (roadmap item 5) ───────────
    // A thin box fired at 120 m/s tunnels a thin static AABB wall WITHOUT the
    // margin (control) — cdOrientedBoxPush is inside-only, and no corner ever
    // lands "inside" the wall between samples — and is stopped WITH it, via
    // cdOrientedBoxPushSpeculative's exterior closest-point near-contact. The
    // near face's corners share ONE target depth (the closest approach), not
    // each their own, so the box decelerates cleanly instead of picking up
    // spurious spin from a lone off-center contact.
    func testSpeculativeMarginStopsBoxTunnelingThroughAABBWall() throws {
        func fire(margin: Float) throws -> Float {
            let (solver, queue) = try makeSolver(radius: 0.07, maxDim: 0.5,
                                                 boundsMin: SIMD3(-4, -0.5, -1),
                                                 boundsMax: SIMD3(4, 1, 1))
            solver.maxSpeed = 200
            solver.restitution = 0
            solver.speculativeMargin = margin
            solver.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0),
                                 .box(center: SIMD3(0, 0.05, 0), halfExtents: SIMD3(0.02, 0.5, 0.5))])
            let proj = solver.spawnBox(at: SIMD3(-2.04, 0.05, 0), halfExtents: SIMD3(0.02, 0.02, 0.02),
                                       velocity: SIMD3(120, 0, 0))!
            step(solver, queue, frames: 30)
            return solver.position(of: proj)!.x
        }
        let without = try fire(margin: 0)
        let with    = try fire(margin: 0.6)
        print("SPECULATIVE_BOX without=\(without) with=\(with)")
        XCTAssertGreaterThan(without, 1.0,
                             "control: at 120 m/s the box tunnels straight through the thin wall")
        XCTAssertLessThan(with, 0.5, "speculative: the box was actually STOPPED by the wall")
    }

    // ── Sphere vs AABB static: exact closest point (roadmap item 5) ────────────
    // A sphere positioned diagonally off an AABB static box's CORNER (not a flat
    // face) must report a contact whose depth exactly matches the analytic
    // sphere-radius-minus-distance-to-corner — provable only by a true exterior
    // closest-point computation. The old path routed a sphere through the box's
    // generic feature-point loop (a 14-point DISC surrogate with fixed sample
    // directions unrelated to the true corner diagonal) — a corner approach is
    // exactly the case that approximation gets wrong.
    func testSphereVsAABBCornerIsExact() throws {
        let (solver, _) = try makeSolver(maxCoins: 8, radius: 0.05, maxDim: 0.05,
                                         boundsMin: SIMD3(-1, -0.5, -1), boundsMax: SIMD3(1, 1, 1))
        let he: Float = 0.05
        solver.setColliders([.box(center: SIMD3(0, 0.05, 0), halfExtents: SIMD3(he, he, he))])
        let corner = SIMD3<Float>(he, 0.05 + he, he)
        let diag = simd_normalize(SIMD3<Float>(1, 1, 1))
        let d: Float = 0.02, r: Float = 0.03   // expected penetration = r - d = 0.01
        let ball = solver.spawnSphere(at: corner + diag * d, radius: r)!
        let n = solver.generateContactsNow()
        let ptr = solver.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(1, n))
        var found: CoinContact?
        for i in 0..<n where Int(ptr[i].meta.x) == ball && ptr[i].meta.y == 0xFFFF_FFFF { found = ptr[i] }
        guard let c = found else { return XCTFail("sphere↔AABB-corner must generate a contact") }
        let depth = c.nrm.w, normal = SIMD3(c.nrm.x, c.nrm.y, c.nrm.z)
        print("SPHERE_AABB_CORNER depth=\(depth) normal=\(normal)")
        XCTAssertEqual(depth, r - d, accuracy: 1e-4, "exact closest-point depth, not a disc-surrogate approximation")
        XCTAssertEqual(simd_dot(normal, diag), 1.0, accuracy: 1e-3, "normal points exactly along the true corner diagonal")
    }

    // ── Hull (>14 verts) vs static box with speculative margin ─────────────────
    // The static generic-else loop's per-point candidate cache is sized to
    // CD_MAX_STATIC_FP (32, the hull vertex cap), not CD_NPTS (14, the box/disc
    // feature-point count) — a >14-vertex hull near a static AABB/OBB with
    // speculativeMargin > 0 is exactly the configuration that overflowed a
    // 14-slot cache before that fix. A 24-point near-spherical hull registers
    // ~24 real vertices (all extremal on a convex point cloud), well past 14.
    func testHullOver14VertsVsStaticBoxWithMarginDoesNotOverflow() throws {
        let (solver, _) = try makeSolver(maxCoins: 8, radius: 0.05, maxDim: 0.05,
                                         boundsMin: SIMD3(-1, -0.5, -1), boundsMax: SIMD3(1, 1, 1))
        solver.speculativeMargin = 0.3
        solver.setColliders([.box(center: SIMD3(0, 0.05, 0), halfExtents: SIMD3(0.05, 0.05, 0.05)),
                             .orientedBox(center: SIMD3(0.3, 0.05, 0), halfExtents: SIMD3(0.05, 0.05, 0.05),
                                          orientation: simd_quatf(angle: 0.3, axis: SIMD3(0, 0, 1)))])
        var pts: [SIMD3<Float>] = []
        let r: Float = 0.04
        for i in 0..<24 {
            // Fibonacci sphere distribution — every point sits on the convex
            // hull of the point cloud, so registration keeps ~24 real vertices.
            let y = 1 - (Float(i) / Float(23)) * 2
            let radiusAtY = sqrt(max(0, 1 - y * y))
            let theta = Float(i) * 2.399963  // golden angle
            pts.append(SIMD3(cos(theta) * radiusAtY, y, sin(theta) * radiusAtY) * r)
        }
        let hull = solver.registerHull(vertices: pts)!
        XCTAssertGreaterThan(hull.vertices.count, 14, "test hull must actually exceed the old 14-slot cache")
        // Close enough (gap 0.05, well within the 0.3 margin) that MOST of the
        // hull's ~24 vertices register simultaneously — not just the closest —
        // actually filling the candidate cache past the old 14-slot size.
        // One near each static (AABB kind==1, OBB kind==3) so both fixed loops run.
        _ = solver.spawnHull(at: SIMD3(0, 0.05 + 0.05 + r + 0.05, 0), hull: hull)
        _ = solver.spawnHull(at: SIMD3(0.3, 0.05 + 0.05 + r + 0.05, 0), hull: hull)
        let n = solver.generateContactsNow()   // must not crash / corrupt memory
        print("HULL_OVER14_MARGIN vertexCount=\(hull.vertices.count) contacts=\(n)")
        XCTAssertGreaterThanOrEqual(n, 0)
    }

    // ── Rolling resistance (constraint path) ──────────────────────────────────
    // Two identical balls rolled with the same send-off; the solver with rolling
    // resistance must stop its ball far shorter than the free-rolling control.
    func testRollingResistanceStopsTheBall() throws {
        func roll(muR: Float) throws -> Float {
            let (solver, queue) = try makeSolver(radius: 0.05, maxDim: 0.05,
                                                 boundsMin: SIMD3(-1, -0.5, -1),
                                                 boundsMax: SIMD3(14, 1, 1))
            solver.frictionCoeff = 0.6          // grip so it truly ROLLS
            solver.rollingResistance = muR
            let ball = solver.spawnSphere(at: SIMD3(0, 0.05, 0), radius: 0.05,
                                          velocity: SIMD3(3, 0, 0))!
            step(solver, queue, frames: 420)    // 7 s
            return solver.position(of: ball)!.x
        }
        let free    = try roll(muR: 0)
        let resisted = try roll(muR: 0.4)
        print("ROLLING free=\(free) resisted=\(resisted)")
        XCTAssertGreaterThan(free, resisted + 0.5,
                             "rolling resistance stops the ball well short of the free roller")
    }

    // ── Pusher plate on the constraint path ──────────────────────────────────
    // The kinematic pusher now exists for constraint-mode scenes: a coin sitting
    // in front of an advancing plate is carried forward, and the depth clamp
    // keeps the shove at plate speed (no launching).
    func testConstraintPusherShovesWithoutLaunching() throws {
        let (solver, queue) = try makeSolver(radius: 0.05, maxDim: 0.05,
                                             boundsMin: SIMD3(-1, -0.5, -1),
                                             boundsMax: SIMD3(1, 1, 2))
        let coin = solver.spawn(at: SIMD3(0, 0.006, 0.1), radius: 0.05, halfThickness: 0.0037)!
        let plateSpeed: Float = 0.5
        var plateZ: Float = -0.06
        var maxVz: Float = 0
        step(solver, queue, frames: 180) { _ in
            plateZ += plateSpeed / 60
            solver.setColliders([
                .plane(normal: SIMD3(0, 1, 0), offset: 0),
                .pusherPlate(center: SIMD3(0, 0.05, plateZ), halfExtents: SIMD3(0.3, 0.05, 0.05),
                             velocity: SIMD3(0, 0, plateSpeed)),
            ])
            if let v = solver.velocity(of: coin) { maxVz = max(maxVz, v.z) }
        }
        let p = solver.position(of: coin)!
        print("CS_PUSHER finalZ=\(p.z) maxVz=\(maxVz)")
        XCTAssertGreaterThan(p.z, 0.6, "coin was carried forward by the plate")
        XCTAssertLessThan(maxVz, plateSpeed * 2.5, "shove never launched the coin")
    }

    // ── Joints: distance (pendulum on a tether) ───────────────────────────────
    // A sphere tethered to a world anchor swings under gravity; the tether length
    // must hold through the whole swing and the bob must end up BELOW the anchor.
    func testDistanceJointPendulumHoldsLength() throws {
        let (solver, queue) = try makeSolver(radius: 0.05, maxDim: 0.05)
        let anchor = SIMD3<Float>(0, 1.0, 0)
        let start  = SIMD3<Float>(0.5, 1.0, 0)          // horizontal release
        let bob = solver.spawnSphere(at: start, radius: 0.05)!
        let j = solver.addDistanceJoint(bodyA: bob, bodyB: nil,
                                        worldAnchorA: start, worldAnchorB: anchor)
        XCTAssertNotNil(j)
        var worstStretch: Float = 0
        step(solver, queue, frames: 300) { _ in
            guard let p = solver.position(of: bob) else { return }
            worstStretch = max(worstStretch, abs(simd_length(p - anchor) - 0.5))
        }
        let p = solver.position(of: bob)!
        print("JOINT_DISTANCE worstStretch=\(worstStretch) final=\(p)")
        XCTAssertLessThan(worstStretch, 0.05, "tether length held through the swing (≤10% drift)")
        XCTAssertLessThan(p.y, 0.95, "bob swung below the anchor")
        XCTAssertFalse(p.x.isNaN)
    }

    // ── Joints: ball (point pendulum holds its anchor) ────────────────────────
    func testBallJointHoldsAnchor() throws {
        let (solver, queue) = try makeSolver(radius: 0.06, maxDim: 0.06)
        let anchor = SIMD3<Float>(0, 0.9, 0)
        let box = solver.spawnBox(at: SIMD3(0.1, 0.8, 0), halfExtents: SIMD3(0.04, 0.04, 0.04))!
        XCTAssertNotNil(solver.addBallJoint(bodyA: box, bodyB: nil, worldAnchor: anchor))
        let localR = simd_length(SIMD3<Float>(0.1, 0.8, 0) - anchor)
        var worstAnchorErr: Float = 0
        step(solver, queue, frames: 300) { _ in
            guard let p = solver.position(of: box) else { return }
            // The COM must stay on the sphere of radius |localAnchor| around the pin.
            let dist = simd_length(p - anchor)
            worstAnchorErr = max(worstAnchorErr, abs(dist - localR))
        }
        let p = solver.position(of: box)!
        print("JOINT_BALL worstErr=\(worstAnchorErr) final=\(p)")
        XCTAssertLessThan(worstAnchorErr, 0.03, "COM stayed on the sphere around the ball anchor")
        XCTAssertLessThan(p.y, 0.9, "body hangs below the anchor")
    }

    // ── Joints: hinge limits ──────────────────────────────────────────────────
    // Two identical flaps hinged to the world about Z at their edge; gravity
    // swings them down. The limited one must stop at its stop; the free one
    // swings far past it — proving the limit (not friction) is what held.
    func testHingeLimitStopsTheFlap() throws {
        let (solver, queue) = try makeSolver(radius: 0.16, maxDim: 0.16)
        let he = SIMD3<Float>(0.12, 0.015, 0.05)
        func flap(z: Float, limits: ClosedRange<Float>?) -> Int {
            let com = SIMD3<Float>(0.12, 1.0, z)         // extends +X from the hinge edge
            let b = solver.spawnBox(at: com, halfExtents: he)!
            _ = solver.addHingeJoint(bodyA: b, bodyB: nil,
                                     worldAnchor: SIMD3(0, 1.0, z),
                                     worldAxis: SIMD3(0, 0, 1),
                                     limits: limits)
            return b
        }
        let limited = flap(z: -0.4, limits: -0.25...0.25)
        let free    = flap(z:  0.4, limits: nil)
        var freeDropped: Float = 0
        step(solver, queue, frames: 300) { _ in
            if let q = solver.orientation(of: free) {
                freeDropped = max(freeDropped, abs(simd_act(q, SIMD3<Float>(1, 0, 0)).y))
            }
        }
        let qL = solver.orientation(of: limited)!
        let tiltLimited = abs(simd_act(qL, SIMD3<Float>(1, 0, 0)).y)   // sin(swing angle)
        print("JOINT_HINGE tiltLimited=\(tiltLimited) freeDroppedMax=\(freeDropped)")
        XCTAssertLessThan(tiltLimited, 0.35, "limited flap held near its ±0.25 rad stop")
        XCTAssertGreaterThan(freeDropped, 0.6, "free flap swung far past the stop")
    }

    // ── Joints: collideConnected (roadmap item 3) ───────────────────────────────
    // Two spheres (radius 0.05, combined resting separation 0.1) distance-jointed
    // to a too-short rest length (0.04) — the joint FORCES an overlap that a
    // contact would otherwise fight. This is a direct contact-buffer probe
    // (generateContactsNow, no stepping) of the actual mechanism: collideConnected
    // gates whether coinGenerateContacts's pair loop emits a contact for the
    // joint's own bodyA/bodyB at all.
    func testCollideConnectedGatesContactGeneration() throws {
        func run(_ collideConnected: Bool) throws -> Int {
            let (solver, _) = try makeSolver(maxCoins: 8, radius: 0.05, maxDim: 0.05,
                                             boundsMin: SIMD3(-1, -0.5, -1), boundsMax: SIMD3(1, 3, 1))
            let r: Float = 0.05
            let posA = SIMD3<Float>(0, 2.0, 0)
            let posB = SIMD3<Float>(0.04, 2.0, 0)   // forced overlap: 0.04 < combined radius 0.1
            let a = solver.spawnSphere(at: posA, radius: r)!
            let b = solver.spawnSphere(at: posB, radius: r)!
            _ = solver.addDistanceJoint(bodyA: a, bodyB: b, worldAnchorA: posA, worldAnchorB: posB,
                                        restLength: 0.04, collideConnected: collideConnected)
            let n = solver.generateContactsNow()
            let ptr = solver.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: max(1, n))
            let lo = min(a, b), hi = max(a, b)
            return (0..<n).filter { Int(ptr[$0].meta.x) == lo && Int(ptr[$0].meta.y) == hi }.count
        }
        let offContacts = try run(false)
        let onContacts = try run(true)
        print("JOINT_COLLIDECONNECTED off=\(offContacts) on=\(onContacts)")
        XCTAssertEqual(offContacts, 0,
                       "collideConnected=false: the jointed pair's own overlap must NOT generate a contact")
        XCTAssertGreaterThan(onContacts, 0,
                             "collideConnected=true: the pair still overlaps geometrically, so a contact must be generated")
    }

    // ── Per-body restitution ──────────────────────────────────────────────────
    // Two identical spheres, same drop, different per-body restitution: the
    // bouncy one must rebound visibly higher. Globals stay at their defaults —
    // the override alone drives the difference.
    func testPerBodyRestitutionControlsBounce() throws {
        let (solver, queue) = try makeSolver(radius: 0.05, maxDim: 0.05)
        solver.restitution = 0.05          // global default: nearly dead
        solver.restThreshold = 0.3
        let dead   = solver.spawnSphere(at: SIMD3(-0.5, 0.8, 0), radius: 0.05)
        let bouncy = solver.spawnSphere(at: SIMD3( 0.5, 0.8, 0), radius: 0.05,
                                        restitution: 0.85)
        XCTAssertNotNil(dead); XCTAssertNotNil(bouncy)

        var reboundDead: Float = 0, reboundBouncy: Float = 0
        var impacted = false
        step(solver, queue, frames: 240) { _ in
            guard let pd = solver.position(of: dead!), let pb = solver.position(of: bouncy!) else { return }
            if pd.y < 0.08 { impacted = true }          // first floor hit happened
            if impacted {
                reboundDead   = max(reboundDead, pd.y)
                reboundBouncy = max(reboundBouncy, pb.y)
            }
        }
        print("MATERIAL_RESTITUTION dead=\(reboundDead) bouncy=\(reboundBouncy)")
        XCTAssertGreaterThan(reboundBouncy, reboundDead + 0.1,
                             "e=0.85 sphere rebounds well above the e=0.05 sphere")
        XCTAssertLessThan(reboundDead, 0.25, "dead sphere barely rebounds")
    }

    // ── Per-body friction ─────────────────────────────────────────────────────
    // Two identical boxes slid along the floor with the same initial velocity,
    // different per-body μ: the grippy one must stop far shorter. Global
    // frictionCoeff stays 0 — vs statics a body's own material governs.
    func testPerBodyFrictionControlsSlide() throws {
        let (solver, queue) = try makeSolver(radius: 0.06, maxDim: 0.06)
        let he = SIMD3<Float>(0.04, 0.04, 0.04)
        let slick = solver.spawnBox(at: SIMD3(-0.8, 0.041, -0.5), halfExtents: he,
                                    velocity: SIMD3(2.0, 0, 0), friction: 0.02)
        let grippy = solver.spawnBox(at: SIMD3(-0.8, 0.041,  0.5), halfExtents: he,
                                     velocity: SIMD3(2.0, 0, 0), friction: 0.9)
        XCTAssertNotNil(slick); XCTAssertNotNil(grippy)
        step(solver, queue, frames: 240)

        let xSlick  = solver.position(of: slick!)!.x - (-0.8)
        let xGrippy = solver.position(of: grippy!)!.x - (-0.8)
        print("MATERIAL_FRICTION slick=\(xSlick) grippy=\(xGrippy)")
        XCTAssertGreaterThan(xSlick, xGrippy + 0.2,
                             "μ=0.02 box slides well past the μ=0.9 box")
        XCTAssertLessThan(simd_length(solver.velocity(of: grippy!)!), 0.05,
                          "grippy box has stopped")
    }
}
