import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Gate for the CoinDEM contact work of the Digital Clock work plan stage B1 (items 5d and
/// the millimetre-scale contact bugs): the exact polytope narrowphase (SAT + face clipping
/// for boxes / topology hulls / compounds, CoinDEMNarrowphase.h), the exact swept-sphere
/// contact (the egg's flank, T8), the opt-in manifold solve, warm-start matching by
/// position, and the size-aware broadphase neighbourhood — VZ-0157, VZ-0158, VZ-0161,
/// VZ-0162, VZ-0163. Each test is the regression proof of one root cause; measured numbers
/// are PRINTed with a `B1_` prefix. Same runtime-compiled-library seam and §G toy-scale
/// configuration (1/180 s, 6 velocity iterations) as CoinDEMActuationTests.
@MainActor
final class CoinDEMContactsTests: XCTestCase {

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

    /// §G knobs (CoinDEMActuationTests.applyClockConfig): 1/180 s × 6, μ 0.5, legacy rolling
    /// 0.024 / 6, 0.2 mm speculative margin, contact slop 0.1 mm, warm start OFF.
    func makeSolver(maxCoins: Int = 32, maxRadius: Float = 0.03,
                    boundsMin: SIMD3<Float> = SIMD3(-0.4, -0.1, -0.4),
                    boundsMax: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4),
                    sleep: Bool = true, colliders: [CoinStaticCollider]? = nil) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: maxCoins,
                                    coinRadius: maxRadius, halfThickness: maxRadius,
                                    boundsMin: boundsMin, boundsMax: boundsMax)
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.warmStart = false
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.rollingResistance = 0.024 / 6
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

    @discardableResult
    func step(_ s: CoinDEMSolver, _ queue: MTLCommandQueue, frames: Int,
              wallDt: Float = 1.0 / 60, perFrame: ((Int) -> Void)? = nil) -> [Double] {
        var out: [Double] = []
        for f in 0..<frames {
            guard let cb = queue.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: wallDt)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            out.append(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000)
            perFrame?(f)
        }
        return out
    }

    static func angleDeg(_ q: simd_quatf) -> Float {
        let n = q.normalized
        return 2 * atan2(simd_length(n.imag), abs(n.real)) * 180 / .pi
    }
    static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }

    /// Contacts between two bodies (or a body and statics when `b` is nil).
    func pairContacts(_ s: CoinDEMSolver, _ a: Int, _ b: Int?) -> [CoinContact] {
        s.contacts(touching: [a]).filter { c in
            if let b { return (Int(c.meta.x) == a && Int(c.meta.y) == b) || (Int(c.meta.x) == b && Int(c.meta.y) == a) }
            return c.meta.y == 0xFFFF_FFFF
        }
    }

    /// Orientation that points a cube's body diagonal (1, 1, 1) straight down: one corner low.
    static let cornerDown: simd_quatf = simd_quatf(from: simd_normalize(SIMD3<Float>(-1, -1, -1)), to: SIMD3(0, -1, 0))

    // ══════════════════════════════════════════════════════════════════════════
    // VZ-0163 — a box corner exactly on a face, and exactly at the margin
    // ══════════════════════════════════════════════════════════════════════════

    /// A 10 mm cube balanced on one corner — the corner exactly ON the face below it (gap 0),
    /// then exactly AT the 0.2 mm speculative margin — must produce a contact against a box
    /// body, a static box and a plane. The old box paths tested a corner as `inside` only
    /// when strictly inside (d > 0) and dropped a near-miss with dist < 1e-8 or dist ≥ spec,
    /// so both cases were NO contact (and two cubes stacked square — every corner exactly on
    /// the other's edge — had none either). Square-stacked cubes, face to face, get four.
    func testBoxCornerOnFaceAndAtMarginMakeContacts() throws {
        let he: Float = 0.005
        let tip = he * sqrt(3)                             // centre → corner along the diagonal
        for (label, gap) in [("on the face", Float(0)), ("at the margin", Float(2e-4))] {
            // Box body below.
            do {
                let (s, _) = try makeSolver(colliders: [])
                let a = s.spawnBox(at: SIMD3(0, 0.1, 0), halfExtents: SIMD3(he, he, he), mass: 0.01)!
                let b = s.spawnBox(at: SIMD3(0.001, 0.1 + he + gap + tip, 0.0005), halfExtents: SIMD3(he, he, he),
                                   orient: Self.v4(Self.cornerDown), mass: 0.01)!
                s.generateContactsNow()
                let cs = pairContacts(s, a, b)
                print("B1_0163 box body, corner \(label): \(cs.count) contact(s), depths \(cs.map { $0.nrm.w * 1000 }) mm, n.y \(cs.map { $0.nrm.y })")
                XCTAssertFalse(cs.isEmpty, "a corner \(label) of a box body is a contact")
                for c in cs {
                    XCTAssertEqual(c.nrm.w, -gap, accuracy: 2e-6, "its depth is the gap")
                    XCTAssertEqual(abs(c.nrm.y), 1, accuracy: 1e-4, "its normal is the face normal")
                }
            }
            // Static box and plane below.
            for staticKind in ["static box", "plane"] {
                let col: CoinStaticCollider = staticKind == "plane"
                    ? .plane(normal: SIMD3(0, 1, 0), offset: 0.1)
                    : .box(center: SIMD3(0, 0.09, 0), halfExtents: SIMD3(0.02, 0.01, 0.02))
                let (s, _) = try makeSolver(colliders: [col])
                let b = s.spawnBox(at: SIMD3(0.001, 0.1 + gap + tip, 0.0005), halfExtents: SIMD3(he, he, he),
                                   orient: Self.v4(Self.cornerDown), mass: 0.01)!
                s.generateContactsNow()
                let cs = pairContacts(s, b, nil)
                print("B1_0163 \(staticKind), corner \(label): \(cs.count) contact(s), depths \(cs.map { $0.nrm.w * 1000 }) mm")
                XCTAssertFalse(cs.isEmpty, "a corner \(label) of a \(staticKind) is a contact")
                for c in cs { XCTAssertEqual(c.nrm.w, -gap, accuracy: 2e-6) }
            }
        }
        // Two cubes stacked exactly square, face to face: a four-point manifold.
        let (s, _) = try makeSolver(colliders: [])
        let a = s.spawnBox(at: SIMD3(0, 0.1, 0), halfExtents: SIMD3(he, he, he), mass: 0.01)!
        let b = s.spawnBox(at: SIMD3(0, 0.1 + 2 * he, 0), halfExtents: SIMD3(he, he, he), mass: 0.01)!
        s.generateContactsNow()
        let cs = pairContacts(s, a, b)
        print("B1_0163 square stack, face to face: \(cs.count) contacts, depths \(cs.map { $0.nrm.w }) m")
        XCTAssertEqual(cs.count, 4, "exactly coincident faces give the full four-corner manifold (was 0–1)")
        for c in cs { XCTAssertEqual(c.nrm.w, 0, accuracy: 1e-7) }
    }

    /// The 5-cube, 10 mm tower of VZ-0163. The contact construct above is half of it; the
    /// other half is the solver: a stack is a chain of REDUNDANT 4-point manifolds, and
    /// cold-started Gauss–Seidel at 6 iterations leaves it asymmetric every substep (a cube
    /// on a cube got ω = 0.48 rad/s from its first cold solve), so it needs warm starting —
    /// and warm starting needs identities that survive a clipped corner flickering in and out
    /// (the position fallback of coinWarmStartMatch). With warmStart + manifoldSolve at
    /// 1/180 s × 6 (sleep on, §G otherwise) the tower stands 10 s and sleeps.
    func testFiveCubeTowerStandsTenSeconds() throws {
        let he: Float = 0.005
        struct Run { var worstTilt: Float = 0; var worstLateral: Float = 0; var asleepAt = -1; var ys: [Float] = [] }
        func run(warm: Bool, manifold: Bool, gap: Float) throws -> Run {
            let (s, q) = try makeSolver()
            s.warmStart = warm
            s.manifoldSolve = manifold
            let ids = (0..<5).map { k in
                s.spawnBox(at: SIMD3(-0.05, he + Float(k) * (2 * he + gap) + gap, 0), halfExtents: SIMD3(he, he, he), mass: 0.01)!
            }
            var r = Run()
            step(s, q, frames: 600) { f in
                for id in ids {
                    let p = s.position(of: id)!
                    r.worstLateral = max(r.worstLateral, simd_length(SIMD2(p.x + 0.05, p.z)))
                    r.worstTilt = max(r.worstTilt, Self.angleDeg(s.orientation(of: id)!))
                }
                if r.asleepAt < 0 && ids.allSatisfy({ s.isAsleep($0) }) { r.asleepAt = f }
            }
            r.ys = ids.map { s.position(of: $0)!.y }
            return r
        }
        let cold = try run(warm: false, manifold: false, gap: 0)
        let square = try run(warm: true, manifold: true, gap: 0)
        let gapped = try run(warm: true, manifold: true, gap: 5e-5)
        print("B1_0163 5-cube tower, 10 s: §G cold (warm off, per point) worst tilt \(cold.worstTilt)°, top y \(cold.ys.last! * 1000) mm | warm + manifold, face to face: tilt \(square.worstTilt)°, lateral \(square.worstLateral * 1000) mm, asleep@frame \(square.asleepAt), y \(square.ys.map { $0 * 1000 }) mm | 0.05 mm gaps: tilt \(gapped.worstTilt)°, lateral \(gapped.worstLateral * 1000) mm, asleep@\(gapped.asleepAt), y \(gapped.ys.map { $0 * 1000 }) mm")
        for (name, r) in [("face to face", square), ("0.05 mm gaps", gapped)] {
            XCTAssertLessThan(r.worstTilt, 3, "the tower stands (\(name))")
            XCTAssertLessThan(r.worstLateral, 0.0005, "no cube slides off (\(name))")
            // Cube k rests on k + 1 interfaces (the floor and the cubes below). The split-
            // impulse bias corrects only penetration beyond `contactSlop` (0.1 mm), so each
            // interface may sit up to one slop deep: cube k lies within (k + 1) × slop below
            // its design height — and never above it.
            for (k, y) in r.ys.enumerated() {
                let design = he + Float(k) * 2 * he
                XCTAssertLessThanOrEqual(y, design + 1e-5, "cube \(k) (\(name))")
                XCTAssertGreaterThanOrEqual(y, design - Float(k + 1) * 1e-4 - 1e-5, "cube \(k) is still in the tower (\(name))")
            }
            XCTAssertGreaterThan(r.asleepAt, 0, "the tower falls asleep (\(name))")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VZ-0157 — a resting box creeps; VZ-0158 — the stocked bars topple
    // ══════════════════════════════════════════════════════════════════════════

    /// A 10 mm, 5 g box resting on a μ 0.5 plane with sleep OFF at the §G config (cold start,
    /// 1/180 s × 6) crept 1.81 mm and turned 9.7° in 10 s: every substep its four corner
    /// contacts were solved one per colour, once per iteration, from zero impulse — the
    /// redundant 4-point system never converged symmetrically, and the fixed colour order
    /// repeated the same residual (|v| 1.6e-4 m/s, a 0.015 rad/s yaw) every substep. With
    /// manifoldSolve the manifold is solved as one unit and the box does not move.
    func testRestingBoxDoesNotCreep() throws {
        func run(manifold: Bool) throws -> (mm: Float, deg: Float) {
            let (s, q) = try makeSolver(sleep: false)
            s.manifoldSolve = manifold
            let box = s.spawnBox(at: SIMD3(0, 0.0051, 0), halfExtents: SIMD3(0.005, 0.005, 0.005), mass: 0.005)!
            var p0 = SIMD3<Float>.zero, q0 = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            step(s, q, frames: 630) { f in if f == 29 { p0 = s.position(of: box)!; q0 = s.orientation(of: box)! } }
            return (simd_length(s.position(of: box)! - p0) * 1000, Self.angleDeg(s.orientation(of: box)! * q0.inverse))
        }
        let perPoint = try run(manifold: false)
        let manifold = try run(manifold: true)
        print("B1_0157 resting 10 mm box, sleep off, §G, 10 s: per-point solve \(perPoint.mm) mm / \(perPoint.deg)° | manifoldSolve \(manifold.mm) mm / \(manifold.deg)°")
        XCTAssertLessThan(manifold.mm, 1e-3, "no slide")
        XCTAssertLessThan(manifold.deg, 1e-3, "no yaw")
    }

    /// The fence's stocked bars (4 upright in the V-ridge cradles, 12 standing on their tips
    /// in the rack — CoinDEMActuationTests.clockStatics / runClockFence layout) alone. At the
    /// §G config they did not settle (the fence: 91° worst, 1/16 asleep); the VZ-0148 hull fix
    /// was not the cause. It is VZ-0157's mechanism on a knife edge: each bar rests on two
    /// 4-point manifolds (the two ridge slopes), and the cold per-point residual walks a bar
    /// off its 5.5 mm edge (with the old vertex-only hull statics a rack bar also rested on
    /// ridge EDGES it could not see). With manifoldSolve, still cold, all 16 settle and sleep.
    func testStockedBarsSettleAndSleep() throws {
        typealias T = CoinDEMActuationTests
        func run(manifold: Bool, warm: Bool) throws -> (worst: Float, asleep: Int) {
            let (s, q) = try makeSolver(maxCoins: 48, maxRadius: 0.27,
                                        boundsMin: SIMD3(-0.30, 0.55, 0.00), boundsMax: SIMD3(0.30, 0.90, 0.46),
                                        colliders: T.clockStatics().all)
            s.manifoldSolve = manifold
            s.warmStart = warm
            let hull = try XCTUnwrap(s.registerHull(vertices: T.barHullPoints()))
            func bar(_ rd: simd_quatf, _ td: SIMD3<Float>) -> Int {
                let qq = rd * hull.principalRotation
                return s.spawnHull(at: td + simd_act(rd, hull.comOffset), hull: hull, orient: Self.v4(qq),
                                   mass: T.barMass, friction: 0.3, restitution: 0.2)!
            }
            var ids: [Int] = []
            for x in [Float(-0.2325), -0.2205, 0.081, 0.105] { ids.append(bar(T.barVertical, SIMD3(x, 0.6405 + 0.0001, T.seatZ))) }
            for k in 0..<12 { ids.append(bar(T.barFacingX, SIMD3(-0.24575, 0.6405 + 0.0001, 0.235 + 0.01 * Float(k)))) }
            let q0 = ids.map { s.orientation(of: $0)! }
            var worst: Float = 0
            step(s, q, frames: 180, wallDt: 1.0 / 30) { _ in
                for (i, b) in ids.enumerated() { worst = max(worst, Self.angleDeg(s.orientation(of: b)! * q0[i].inverse)) }
            }
            return (worst, ids.filter { s.isAsleep($0) }.count)
        }
        let perPoint = try run(manifold: false, warm: false)
        let manifold = try run(manifold: true, warm: false)
        let both = try run(manifold: true, warm: true)
        print("B1_0158 16 stocked bars, 6 s at 30 fps: per-point §G worst rotation \(perPoint.worst)°, asleep \(perPoint.asleep)/16 | manifoldSolve \(manifold.worst)°, asleep \(manifold.asleep)/16 | + warmStart \(both.worst)°, asleep \(both.asleep)/16")
        XCTAssertLessThan(manifold.worst, 1.5, "manifoldSolve: every stocked bar stays standing (cold start)")
        // Canonical manifold order (see testStackedHullBarsRest): 0.052° sorted, 0.61° with
        // cdEmitGroupF writing reduction order (verifier mutation run, stage B1).
        XCTAssertLessThan(manifold.worst, 0.25, "canonical manifold order: the cold-started bars barely turn")
        XCTAssertEqual(manifold.asleep, 16, "…and falls asleep")
        XCTAssertLessThan(both.worst, 1.5)
        XCTAssertEqual(both.asleep, 16)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VZ-0161 — bodies bigger than the broadphase cell
    // ══════════════════════════════════════════════════════════════════════════

    /// Two capsules (r 30 mm, half-length 50 mm: 80 mm from COM to tip) in a solver whose cell
    /// is 2 × 0.06 = 0.12 m, overlapping end to end by 30 mm with their COMs 0.13 m apart —
    /// in cells 7 and 9 along x. The broadphase scanned only the 3×3×3 cells around a body,
    /// so it never paired them (the classic scan, kept as a test seam, still misses them —
    /// the witness below): the dense hull/capsule pile (seed 5) had capsules 20–30 mm deep
    /// in each other with no contact. The neighbourhood now covers the largest body
    /// spawned; and a world whose bodies fit the cell keeps the classic ±1 scan —
    /// bit-identical contacts either way.
    func testBodiesLargerThanTheCellStillCollide() throws {
        let (engine, lib, _) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 8, coinRadius: 0.06, halfThickness: 0.06,
                                    boundsMin: SIMD3(-1, -0.5, -1), boundsMax: SIMD3(1, 1, 1))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.setColliders([])
        let axisX = Self.v4(simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1)))
        // Cell boundaries at x = −1 + 0.12k: −0.045 is in cell 7, 0.085 in cell 9.
        let xa: Float = -0.045, xb: Float = 0.085
        let a = s.spawnCapsule(at: SIMD3(xa, 0.2, 0.05), radius: 0.03, halfLength: 0.05, orient: axisX)!
        let b = s.spawnCapsule(at: SIMD3(xb, 0.2, 0.05), radius: 0.03, halfLength: 0.05, orient: axisX)!
        let cellA = Int(((xa + 1) / 0.12).rounded(.down)), cellB = Int(((xb + 1) / 0.12).rounded(.down))
        s.broadphaseClassicScanForTesting = true
        s.generateContactsNow()
        let classicPair = pairContacts(s, a, b)
        s.broadphaseClassicScanForTesting = false
        s.generateContactsNow()
        let cs = pairContacts(s, a, b)
        print("B1_0161 capsules end to end, COMs 0.13 m apart (cells \(cellA) and \(cellB)): classic ±1 scan \(classicPair.count) contact(s); sized scan \(cs.count) contact(s), depth \(cs.map { $0.nrm.w * 1000 }) mm; maxBodyBound \(s.maxBodyBound)")
        XCTAssertEqual(cellB - cellA, 2, "precondition: the COMs are two cells apart")
        XCTAssertTrue(classicPair.isEmpty, "witness: the classic 3×3×3 scan never pairs them")
        XCTAssertFalse(cs.isEmpty, "overlapping capsules two cells apart collide")
        XCTAssertEqual(cs.map { $0.nrm.w }.max() ?? 0, 0.03, accuracy: 1e-4, "at their true 30 mm overlap")

        // A world whose bodies fit the cell: the wider scan changes nothing.
        guard let t = CoinDEMSolver(engine: engine, library: lib, maxCoins: 64, coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.4, -0.1, -0.4), boundsMax: SIMD3(0.4, 0.4, 0.4))
        else { throw XCTSkip("solver init failed") }
        t.solverMode = .constraint
        t.speculativeMargin = 2e-4
        t.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0)])
        var seed: UInt64 = 0xB1_0161
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
        for i in 0..<48 {
            let p = SIMD3((rnd() - 0.5) * 0.12, rnd() * 0.06, (rnd() - 0.5) * 0.12)
            switch i % 3 {
            case 0: _ = t.spawnSphere(at: p, radius: 0.004 + rnd() * 0.01)
            case 1: _ = t.spawnCapsule(at: p, radius: 0.004, halfLength: 0.008, orient: Self.v4(simd_quatf(angle: rnd() * 6, axis: SIMD3(0, 0, 1))))
            default: _ = t.spawnEgg(at: p, fatRadius: 0.009, tipRadius: 0.006, centerDistance: 0.011)
            }
        }
        t.generateContactsNow()
        let wide = contactSet(t)
        t.broadphaseClassicScanForTesting = true
        t.generateContactsNow()
        let classic = contactSet(t)
        print("B1_0161 fitting world (bound \(t.maxBodyBound) ≤ cell/2 0.03): \(wide.count) contacts, identical to the classic ±1 scan: \(wide == classic)")
        XCTAssertGreaterThan(wide.count, 20)
        XCTAssertEqual(wide, classic, "bodies that fit the cell: the classic neighbourhood, bit for bit")
    }

    func contactSet(_ s: CoinDEMSolver) -> [String] {
        let p = s.contactBuffer.contents().bindMemory(to: CoinContact.self, capacity: s.maxContacts)
        return (0..<s.contactCount).map { i -> String in
            let c = p[i]
            let f: [Float] = [c.nrm.x, c.nrm.y, c.nrm.z, c.nrm.w, c.rA.x, c.rA.y, c.rA.z, c.rB.x, c.rB.y, c.rB.z]
            return "\(c.meta.x) \(c.meta.y) \(c.meta.z) \(c.meta.w) " + f.map { String($0.bitPattern, radix: 16) }.joined(separator: " ")
        }.sorted()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 5d — exact hull contacts: static-box edges (T5), the egg's flank (T8), stacks
    // ══════════════════════════════════════════════════════════════════════════

    /// Brute-force outward face planes of a convex point set (n·x ≤ d inside) — an oracle
    /// independent of the engine's own hull code.
    static func hullPlanes(_ pts: [SIMD3<Float>]) -> [(SIMD3<Double>, Double)] {
        let P = pts.map { SIMD3<Double>(Double($0.x), Double($0.y), Double($0.z)) }
        var out: [(SIMD3<Double>, Double)] = []
        let n = P.count
        for i in 0..<n { for j in (i + 1)..<n { for k in (j + 1)..<n {
            var nn = simd_cross(P[j] - P[i], P[k] - P[i])
            let l = simd_length(nn); if l < 1e-14 { continue }
            nn /= l
            var d = simd_dot(nn, P[i])
            var pos = 0, neg = 0
            for m in 0..<n { let s = simd_dot(nn, P[m]) - d; if s > 1e-9 { pos += 1 } else if s < -1e-9 { neg += 1 } }
            if pos > 0 && neg > 0 { continue }
            if pos > 0 { nn = -nn; d = -d }
            if !out.contains(where: { simd_dot($0.0, nn) > 1 - 1e-9 && abs($0.1 - d) < 1e-9 }) { out.append((nn, d)) }
        } } }
        return out
    }

    /// T5. A Digital Clock bar lying face down slides (along its 7 mm width) across a 3 mm
    /// pocket mouth between two static boxes. While it bridges the mouth, a rim — a box
    /// EDGE — carries it on its bottom face, where no bar vertex is: the old hull-vs-static
    /// path probed only the bar's vertices, and the bar sank 3.96 mm through the rim
    /// (measured on the pre-B1 engine). The polytope narrowphase (SAT + face clipping) holds
    /// it: the rim edges never penetrate the bar deeper than the 0.2 mm speculative margin,
    /// per-point or manifold solve. Penetration is measured on the CPU against the bar's
    /// own hull planes (a brute-force oracle), at 162 points along each rim every frame.
    func testBarEdgeSlidingAcrossAPocketMouthStaysWithinTheMargin() throws {
        typealias T = CoinDEMActuationTests
        let half: Float = 0.0015
        let boxes: [CoinStaticCollider] = [
            .box(center: SIMD3(-half - 0.015, -0.005, 0), halfExtents: SIMD3(0.015, 0.005, 0.02), friction: 0, restitution: 0.1),
            .box(center: SIMD3( half + 0.015, -0.005, 0), halfExtents: SIMD3(0.015, 0.005, 0.02), friction: 0, restitution: 0.1)]
        let pts = T.barHullPoints()
        let planes = Self.hullPlanes(pts)
        for manifold in [false, true] {
            let (s, q) = try makeSolver(maxCoins: 8, maxRadius: 0.03, sleep: false, colliders: boxes)
            s.manifoldSolve = manifold
            let hull = try XCTUnwrap(s.registerHull(vertices: pts))
            XCTAssertTrue(s.hullHasTopology(hull.index))
            let rd = simd_quatf(simd_float3x3(columns: (SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0))))   // face down, width along x
            let qq = rd * hull.principalRotation
            let bar = try XCTUnwrap(s.spawnHull(at: SIMD3(-0.008, 0.00276, 0) + simd_act(rd, hull.comOffset), hull: hull,
                                                orient: Self.v4(qq), mass: T.barMass, friction: 0, restitution: 0.1))
            var worst: Float = 0, crossed: Float = -1
            step(s, q, frames: 24) { _ in
                s.setVelocity(ofSlot: bar, to: SIMD3(0.04, s.velocity(of: bar)!.y, 0))
                let qb = s.orientation(of: bar)! * hull.principalRotation.inverse
                let xb = s.position(of: bar)! - simd_act(qb, hull.comOffset)
                for sx in [-half, half] {
                    for iz in -80...80 {
                        let pl = simd_act(qb.inverse, SIMD3<Float>(sx, 0, Float(iz) * 0.000075) - xb)
                        let pD = SIMD3<Double>(Double(pl.x), Double(pl.y), Double(pl.z))
                        var maxS = -1e30
                        for (nn, d) in planes { maxS = max(maxS, simd_dot(nn, pD) - d) }
                        if maxS < 0 { worst = max(worst, Float(-maxS)) }
                    }
                }
                for v in pts {
                    let w = xb + simd_act(qb, v)
                    if w.y < 0 && abs(w.x) > half { worst = max(worst, -w.y) }
                }
                crossed = max(crossed, s.position(of: bar)!.x)
            }
            print("B1_5d_T5 bar across a 3 mm pocket mouth (\(manifold ? "manifold" : "per-point") solve): max penetration \(worst * 1000) mm (margin 0.2 mm; pre-B1 engine 3.96 mm), furthest x \(crossed * 1000) mm")
            XCTAssertLessThanOrEqual(worst, 2e-4, "the rim edges never penetrate the bar beyond the speculative margin")
            XCTAssertGreaterThan(crossed, -0.0017, "precondition: the bar reached the mouth")
        }
    }

    /// T8. The Digital Clock worker (the ballasted Weeble, upright) with a hull EDGE touching
    /// its tapering flank at heights from 14.5 to 29 mm above the table. The flank is the
    /// cone tangent to both spheres (half-angle asin((r1 − r2)/c) = 13.6°), so the analytic
    /// contact is the edge's own height, the depth is the edge's inset × cos 13.6°, and the
    /// normal is the flank normal. The old egg–hull contact tested the two end spheres
    /// against the hull's support PLANE: two ghost points at the spheres' tangency heights
    /// (14.2 and 30.2 mm, measured on the pre-B1 engine) whatever the edge height. The swept
    /// contact finds the exact minimum of sd(c(t)) − r(t) along the axis.
    func testEggFlankContactIsExact() throws {
        typealias T = CoinDEMActuationTests
        let (s, _) = try makeSolver(maxCoins: 8, maxRadius: 0.04, colliders: [])
        s.gravity = 0
        let hx: Float = 0.0035, hy: Float = 0.00275, hz: Float = 0.013
        var pts: [SIMD3<Float>] = []
        for i in 0..<8 { pts.append(SIMD3((i & 1) != 0 ? hx : -hx, (i & 2) != 0 ? hy : -hy, (i & 4) != 0 ? hz : -hz)) }
        let hull = try XCTUnwrap(s.registerHull(vertices: pts))
        let w = T.worker
        let egg = try XCTUnwrap(s.spawnBallastedEgg(w, fatCenter: SIMD3(0, 0.0115, 0)))
        let sinB = (w.fatRadius - w.tipRadius) / w.centerDistance, cosB = (1 - sinB * sinB).squareRoot()
        let inset: Float = 0.00005
        var worstDy: Float = 0, worstDepth: Float = 0, worstAngle: Float = 0
        for ye in [Float(0.0145), 0.017, 0.020, 0.025, 0.029] {
            let rho = (w.fatRadius - (ye - 0.0115) * sinB) / cosB          // flank radius at that height
            // The hull's BOTTOM −x edge at (rho − inset, ye): only that edge reaches the flank.
            let bar = try XCTUnwrap(s.spawnHull(at: SIMD3(rho - inset + hx, ye + hy, 0) + hull.comOffset, hull: hull,
                                                orient: Self.v4(hull.principalRotation), mass: 0.00103))
            s.generateContactsNow()
            let cs = pairContacts(s, egg, bar).filter { $0.nrm.w > 0 }
            XCTAssertEqual(cs.count, 1, "one penetrating contact, on the flank, at y \(ye * 1000) mm")
            for c in cs {
                let cp = SIMD3(c.rA.x, c.rA.y, c.rA.z) + (Int(c.meta.x) == egg ? s.position(of: egg)! : s.position(of: bar)!)
                let n = SIMD3(c.nrm.x, c.nrm.y, c.nrm.z) * (Int(c.meta.x) == egg ? 1 : -1)   // from the hull toward the egg
                let flankOut = SIMD3<Float>(cosB, sinB, 0)                                   // egg's outward normal there
                worstDy = max(worstDy, abs(cp.y - ye))
                worstDepth = max(worstDepth, abs(c.nrm.w - inset * cosB))
                worstAngle = max(worstAngle, acos(min(1, simd_dot(-n, flankOut))) * 180 / .pi)
            }
            s.despawn(bar)
        }
        print("B1_5d_T8 egg flank vs hull edge, 14.5–29 mm: worst |contact height − edge height| \(worstDy * 1000) mm, worst depth error \(worstDepth * 1000) mm, worst normal error \(worstAngle)° (pre-B1: ghost points at 14.2 / 30.2 mm)")
        XCTAssertLessThan(worstDy, 1e-4, "the flank contact height matches the analytic tangency to < 0.1 mm")
        XCTAssertLessThan(worstDepth, 2e-6, "depth = the edge inset along the flank normal")
        XCTAssertLessThan(worstAngle, 0.1, "normal = the flank normal")
    }

    /// Three Digital Clock bars stacked face down. Hull–hull used GJK/EPA: a flat face–face
    /// Minkowski simplex made EPA invent 0.26 mm of depth for bars 0.01 mm apart (the axis
    /// probe that proved them apart was ignored), and its 1 mm + 2 % R support band mixed a
    /// bar's face ring with its chamfer ring — the stack blew apart (the upper bars ended
    /// on their sides). The polytope narrowphase stacks them; they rest (manifold + warm
    /// start, sleep off, 10 s) and fall asleep with sleep on.
    func testStackedHullBarsRest() throws {
        typealias T = CoinDEMActuationTests
        func run(sleep: Bool) throws -> (ys: [Float], drift: Float, rot: Float, asleep: Int) {
            let (s, q) = try makeSolver(maxCoins: 16, sleep: sleep)
            s.manifoldSolve = true
            s.warmStart = true
            let hull = try XCTUnwrap(s.registerHull(vertices: T.barHullPoints()))
            let rd = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))          // face (local z) → world +y
            let qq = rd * hull.principalRotation
            let ids = (0..<3).map { k -> Int in
                let td = SIMD3<Float>(0, 0.00275 + Float(k) * 0.0055 + 0.00001 * Float(k + 1), 0)
                return s.spawnHull(at: td + simd_act(rd, hull.comOffset), hull: hull, orient: Self.v4(qq),
                                   mass: T.barMass, friction: 0.3, restitution: 0.2)!
            }
            var p1: [SIMD3<Float>] = [], q1: [simd_quatf] = []
            step(s, q, frames: 600) { f in if f == 59 { p1 = ids.map { s.position(of: $0)! }; q1 = ids.map { s.orientation(of: $0)! } } }
            let drift = zip(ids, p1).map { simd_length(s.position(of: $0.0)! - $0.1) }.max()!
            let rot = zip(ids, q1).map { Self.angleDeg(s.orientation(of: $0.0)! * $0.1.inverse) }.max()!
            return (ids.map { s.position(of: $0)!.y }, drift, rot, ids.filter { s.isAsleep($0) }.count)
        }
        let awake = try run(sleep: false)
        let slept = try run(sleep: true)
        print("B1_5d_stack 3 bars face down: sleep off, 1→10 s drift \(awake.drift * 1000) mm / \(awake.rot)°, y \(awake.ys.map { $0 * 1000 }) mm (design 2.75 / 8.25 / 13.75) | sleep on: y \(slept.ys.map { $0 * 1000 }), asleep \(slept.asleep)/3")
        // Bar k rests on k + 1 interfaces, each up to one contact slop (0.1 mm) deep (the
        // split-impulse bias corrects only beyond it) — as in the cube tower.
        for ys in [awake.ys, slept.ys] {
            for (k, y) in ys.enumerated() {
                let design = 0.00275 + Float(k) * 0.0055
                XCTAssertLessThanOrEqual(y, design + 1e-5, "bar \(k)")
                XCTAssertGreaterThanOrEqual(y, design - Float(k + 1) * 1e-4 - 1e-5, "bar \(k) is still in the stack")
            }
        }
        XCTAssertLessThan(awake.drift, 0.001, "the awake stack rests (< 1 mm in 9 s)")
        XCTAssertLessThan(awake.rot, 2)
        XCTAssertEqual(slept.asleep, 3, "the stack falls asleep")
        // The CANONICAL manifold order (cdCanonicalManifoldOrder — the manifold solve visits a
        // grouped manifold's points in the order they are written). Verifier mutation runs,
        // stage B1: with cdEmitGroupF (the polytope pairs) writing reduction order instead, this
        // stack drifted 0.074 mm / 0.118° in 9 s; with cdEmitGroup (the plane) doing so,
        // 0.0125 mm / 0.170° — sorted, 0.0003 mm / 0.00005°. The 1 mm / 2° gates above can't
        // see either, so the fix had no failing test; these can.
        XCTAssertLessThan(awake.drift, 5e-6, "canonical manifold order: the awake stack does not creep")
        XCTAssertLessThan(awake.rot, 0.01, "canonical manifold order: nor turn")
    }

    /// A hull that fits the GPU topology caps (faces ≤ 64, edges ≤ 192, loops ≤ 16) but has
    /// MORE than 64 vertices — a 5-ring × 14 barrel: 70 vertices, 58 faces, 126 edges. The
    /// narrowphase read such a hull's vertex count as min(count, 64) (cdHullTopoOf), and that
    /// count bounds cdPolyMinDot, the SAT support: along a static box's face the barrel looked
    /// up to 0.31 mm further away than it is, so a vertex 0.05 mm into the box read as a
    /// 0.26 mm gap — beyond the margin, NO contact (verifier, stage B1). Every vertex counts now.
    func testHullWithMoreThan64VerticesKeepsEveryVertexInTheSAT() throws {
        let floorTop: Float = 0
        let (s, _) = try makeSolver(maxCoins: 8, sleep: false,
                                    colliders: [.box(center: SIMD3(0, floorTop - 0.01, 0), halfExtents: SIMD3(0.2, 0.01, 0.2))])
        var pts: [SIMD3<Float>] = []
        for (y, r) in zip([Float(-2), -1, 0, 1, 2], [Float(0.6), 0.9, 1.0, 0.9, 0.6]) {
            for j in 0..<14 { let a = Float(j) * 2 * .pi / 14; pts.append(SIMD3(r * cos(a), y * 0.5, r * sin(a)) * 0.01) }
        }
        let hull = try XCTUnwrap(s.registerHull(vertices: pts))
        XCTAssertEqual(hull.vertices.count, 70, "precondition: more than 64 vertices")
        XCTAssertTrue(s.hullHasTopology(hull.index), "precondition: the exact polytope narrowphase owns it")
        // A vertex past index 63 that is the unique support in its own direction — pointed down.
        var chosen: SIMD3<Float>?
        for i in stride(from: hull.vertices.count - 1, through: 64, by: -1) {
            let d = simd_normalize(hull.vertices[i])
            let others = hull.vertices.enumerated().filter { $0.offset != i }.map { simd_dot($0.element, d) }.max()!
            if simd_dot(hull.vertices[i], d) - others > 1e-4 { chosen = d; break }
        }
        let d = try XCTUnwrap(chosen, "precondition: an isolated support vertex ≥ 64")
        let q = simd_quatf(from: d, to: SIMD3(0, -1, 0))
        let lowAll = hull.vertices.map { simd_act(q, $0).y }.min()!
        let low64 = hull.vertices.prefix(64).map { simd_act(q, $0).y }.min()!
        for deep: Float in [0.001, 0.00005] {
            let b = try XCTUnwrap(s.spawnHull(at: SIMD3(0.01, floorTop - lowAll - deep, 0.02), hull: hull, orient: Self.v4(q)))
            s.generateContactsNow()
            let maxD = s.contacts(touching: [b]).map { $0.nrm.w }.max() ?? -1
            print("B1v_nV barrel (70 vertices) \(deep * 1000) mm into a static box: deepest contact \(maxD * 1000) mm (depth seen by vertices 0..<64 alone: \((lowAll + deep - low64) * 1000) mm)")
            XCTAssertEqual(maxD, deep, accuracy: 2e-6, "the deepest contact is the true \(deep * 1000) mm")
            s.despawn(b)
        }
    }

    /// The bar hull's GPU topology: quickhull's 60 triangles merge into the prismatoid's 26
    /// faces (2 octagons, 8 sides, 16 chamfer quads) and 56 edges, every loop CCW about its
    /// outward normal and on its plane, V − E + F = 2 — at 1 mm and at 1 m.
    func testClockBarHullTopology() throws {
        for scale: Float in [1, 1000] {
            let pts = CoinDEMActuationTests.barHullPoints().map { $0 * scale }
            let prep = try XCTUnwrap(CoinHullMath.prepare(pts))
            let topo = try XCTUnwrap(prep.topology)
            XCTAssertEqual(topo.faces.count, 26, "scale \(scale)")
            XCTAssertEqual(topo.edges.count, 56, "scale \(scale)")
            XCTAssertTrue(topo.fitsGPU)
            XCTAssertEqual(prep.vertices.count - topo.edges.count + topo.faces.count, 2)
            let L = Double(26 * scale) * 1e-3
            for f in topo.faces {
                var newell = SIMD3<Double>.zero
                for i in 0..<f.loop.count {
                    let a = SIMD3<Double>(prep.vertices[f.loop[i]]), b = SIMD3<Double>(prep.vertices[f.loop[(i + 1) % f.loop.count]])
                    newell += simd_cross(a, b)
                    XCTAssertEqual(simd_dot(f.normal, a), f.offset, accuracy: 1e-5 * L, "loop vertex on its plane")
                }
                XCTAssertGreaterThan(simd_dot(newell, f.normal), 0, "loop CCW about the outward normal")
            }
            print("B1_topo bar ×\(scale): \(topo.faces.count) faces (loop sizes \(topo.faces.map { $0.loop.count })), \(topo.edges.count) edges")
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Swift ↔ Metal layouts
    // ══════════════════════════════════════════════════════════════════════════

    /// The structs the host writes and the kernels read must agree byte for byte — stage B1
    /// grew CoinUniforms by two fields (maxBodyBound, manifoldPasses). Asserted by the Metal
    /// compiler itself: a kernel appended to CoinDEM.metal reports sizeof / offsetof of each
    /// shared struct, compared with Swift's MemoryLayout.
    func testSharedStructLayoutsMatchMetal() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let (_, _, queue) = try Self.shared()
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        let src = try MetalSourceLoader.source(contentsOf: url) + """

        kernel void zzCoinLayouts(device uint* out [[ buffer(0) ]], uint id [[ thread_position_in_grid ]]) {
            if (id != 0u) return;
            out[0] = sizeof(CoinUniforms);  out[1] = sizeof(CoinContact); out[2] = sizeof(CoinBody);
            out[3] = sizeof(CoinStaticCollider); out[4] = sizeof(CoinJoint);
            out[5] = __builtin_offsetof(CoinUniforms, solverFlags);
            out[6] = __builtin_offsetof(CoinUniforms, maxBodyBound);
            out[7] = __builtin_offsetof(CoinUniforms, manifoldPasses);
            out[8] = __builtin_offsetof(CoinContact, aux);
            out[9] = __builtin_offsetof(CoinContact, ext);
        }
        """
        let lib = try device.makeLibrary(source: src, options: nil)
        let fn = try XCTUnwrap(lib.makeFunction(name: "zzCoinLayouts"))
        let pso = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let out = try XCTUnwrap(device.makeBuffer(length: 16 * 4, options: .storageModeShared))
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        let m = out.contents().bindMemory(to: UInt32.self, capacity: 16)
        let swift: [Int] = [MemoryLayout<CoinUniforms>.stride, MemoryLayout<CoinContact>.stride, MemoryLayout<CoinBody>.stride,
                            MemoryLayout<CoinStaticCollider>.stride, MemoryLayout<CoinJoint>.stride,
                            MemoryLayout<CoinUniforms>.offset(of: \.solverFlags)!, MemoryLayout<CoinUniforms>.offset(of: \.maxBodyBound)!,
                            MemoryLayout<CoinUniforms>.offset(of: \.manifoldPasses)!, MemoryLayout<CoinContact>.offset(of: \.aux)!,
                            MemoryLayout<CoinContact>.offset(of: \.ext)!]
        let names = ["CoinUniforms", "CoinContact", "CoinBody", "CoinStaticCollider", "CoinJoint",
                     "CoinUniforms.solverFlags", "CoinUniforms.maxBodyBound", "CoinUniforms.manifoldPasses", "CoinContact.aux",
                     "CoinContact.ext"]
        print("B1_layout Metal \((0..<10).map { m[$0] }) Swift \(swift)")
        for i in 0..<10 { XCTAssertEqual(Int(m[i]), swift[i], names[i]) }
        XCTAssertEqual(MemoryLayout<CoinUniforms>.stride, 148)
        // Stage B2 added the `ext` lane (the accumulated torsional impulse, engine plan 5a): 112 → 128.
        XCTAssertEqual(MemoryLayout<CoinContact>.stride, 128)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VZ-0162 — warm start wrote velocity onto asleep bodies
    // ══════════════════════════════════════════════════════════════════════════

    /// A 10 mm cube resting on an ASLEEP cube (held by the kinematic-hold flag, sleep off):
    /// the warm-start apply used to add each contact's carried impulse to the asleep body's
    /// velocity with its real inverse mass — a velocity nothing integrates but the solve
    /// then read as that body's motion: the held cube "moved away" under the top cube, whose
    /// contact unloaded, so the top cube sank into it. The witness is the TOP cube: with the
    /// guard removed (mutation run, stage B1) it sank 0.94 mm into the held cube in 60 frames
    /// (y 14.06 mm, not 15.0). The held cube's own velocity is only a sanity check — finalize
    /// zeroes an asleep body's velocity before the frame ends, so it reads 0 either way.
    func testWarmStartLeavesAsleepBodiesStill() throws {
        let (s, q) = try makeSolver(sleep: false)
        s.warmStart = true
        s.manifoldSolve = true
        let he: Float = 0.005
        let held = s.spawnBox(at: SIMD3(0, he, 0), halfExtents: SIMD3(he, he, he), mass: 0.01)!
        let top = s.spawnBox(at: SIMD3(0, 3 * he, 0), halfExtents: SIMD3(he, he, he), mass: 0.01)!
        s.setKinematicHold(held, true)
        var worstV: Float = 0
        step(s, q, frames: 60) { _ in worstV = max(worstV, simd_length(s.velocity(of: held)!), simd_length(s.angularVelocity(of: held)!)) }
        print("B1_0162 held cube under a warm-started cube: worst |v|,|ω| of the held cube \(worstV); top y \(s.position(of: top)!.y * 1000) mm")
        XCTAssertEqual(worstV, 0, "an asleep / held body stays still (sanity: finalize zeroes it anyway)")
        // One interface: within the 0.1 mm contact slop (the unguarded warm start: 0.94 mm deep).
        XCTAssertEqual(s.position(of: top)!.y, 3 * he, accuracy: 1e-4 + 1e-5, "the held cube still carries the top cube")
    }
}
