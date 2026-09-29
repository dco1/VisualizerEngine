import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// Two constraint-path opt-ins found building Digital Clock's lumber stock (two 2 × 5 stacks of
/// on-edge 25 mm bars): `keepSleepingContacts` (a sleeping island keeps its warm start —
/// CD_FLAG_KEEP_ASLEEP_CONTACTS, coinCarryDormant) and `separatingSplitImpulse` (the split-impulse
/// bias row only ever pushes — CD_FLAG_SEPARATING_BIAS). Each test proves its mechanism and that the
/// default (flag off) is what every existing world steps. Numbers are PRINTed with a `SLEEP_` prefix.
@MainActor
final class CoinDEMSleepWarmStartTests: XCTestCase {

    // ── Harness (CoinDEMContactsTests' runtime-compiled library + the clock's configuration) ──

    private static var cached: (SimEngine, MTLLibrary, MTLCommandQueue)?
    private static func shared() throws -> (SimEngine, MTLLibrary, MTLCommandQueue) {
        if let c = cached { return c }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness queue
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard FileManager.default.fileExists(atPath: shader.path) else { throw XCTSkip("CoinDEM.metal not found") }
        let c = (SimEngine(device: device), try MetalSourceLoader.makeLibrary(device: device, contentsOf: shader), queue)
        cached = c
        return c
    }

    /// Digital Clock's rigid-world configuration (ClockRigidWorld.configure): 1/180 s × 6, warm
    /// start + manifold solve + speculative colouring, a 2.5 mm speculative margin, 0.1 mm slop.
    private func makeSolver(keep: Bool, separating: Bool, smallWorld: Bool = false,
                            colliders: [CoinStaticCollider]) throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 32, coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.2, -0.05, -0.2), boundsMax: SIMD3(0.2, 0.3, 0.2))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.colorRounds = 16
        s.islandUnionRounds = 6
        s.linDamping = 0.99995
        s.angDamping = 0.9998
        s.frictionCoeff = 0.5
        s.accumulatedRollingResistance = true
        s.rollingResistance = 0.024
        s.scaleAwareDeadStop = true
        s.restitution = 0.15
        s.restThreshold = 0.14
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2.5e-3
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.maxOmega = 60
        s.floorY = -0.5
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.manifoldSolve = true
        s.warmStart = true
        s.coloringScheme = .speculative
        s.smallWorldPath = smallWorld
        s.keepSleepingContacts = keep
        s.separatingSplitImpulse = separating
        s.setColliders(colliders)
        return (s, queue)
    }

    private func step(_ s: CoinDEMSolver, _ q: MTLCommandQueue, frames: Int, keepAwake: [Int] = [],
                      perFrame: ((Int) -> Void)? = nil) {
        for f in 0..<frames {
            if !keepAwake.isEmpty { s.wake(keepAwake) }
            guard let cb = q.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: 1.0 / 60)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            if cb.status == .error { XCTFail("command buffer error: \(String(describing: cb.error))"); break }
            perFrame?(f)
        }
    }

    private static func v4(_ q: simd_quatf) -> SIMD4<Float> { SIMD4(q.imag, q.real) }

    /// A 2 × 5 stock of on-edge clock bars (face +Z, rows 0.5 mm apart) on a static oak slab, each
    /// layer at the solver's resting depth (one slop) on the one below; and a sphere far off that the
    /// test keeps awake — a world that steps every frame while the stock sleeps (the clock's crew
    /// never sleeps).
    private func buildStock(_ s: CoinDEMSolver) throws -> (bars: [Int], busy: Int) {
        let hull = try XCTUnwrap(s.registerHull(vertices: CoinDEMActuationTests.barHullPoints()))
        let rd = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)               // on its long edge, face +Z
        let q = (rd * hull.principalRotation).normalized
        var bars: [Int] = []
        let halfW: Float = 0.00343, pitch: Float = 2 * halfW, slop: Float = 1e-4
        for l in 0..<5 {
            for r in 0..<2 {
                let c = SIMD3<Float>(0, halfW + Float(l) * pitch - Float(l + 1) * slop, Float(r) * 0.006)
                bars.append(try XCTUnwrap(s.spawnHull(at: c + simd_act(rd, hull.comOffset), hull: hull, orient: Self.v4(q),
                                                      mass: CoinDEMActuationTests.barMass, friction: 0.3, restitution: 0.2)))
            }
        }
        let busy = try XCTUnwrap(s.spawnSphere(at: SIMD3(0.12, 0.004, 0.12), radius: 0.004, mass: 0.002))
        return (bars, busy)
    }

    private static var oak: [CoinStaticCollider] {
        [.box(center: SIMD3(0, -0.01, 0), halfExtents: SIMD3(0.18, 0.01, 0.18), friction: 0.5, restitution: 0.2)]
    }

    /// With the flag, the substep after the stock falls asleep carries its contacts on as DORMANT
    /// records (never coloured), and a wake a second later resumes from their impulses: nothing moves.
    /// Without it (the default, bit for bit what VZ-0152 does) the sleeping stock has no contacts in
    /// the list and the same wake starts cold.
    func testASleepingStackKeepsItsWarmStartThroughAWake() throws {
        for smallWorld in [false, true] {
            var worst: [Bool: (move: Float, speed: Float, listed: Int)] = [:]
            for keep in [false, true] {
                let (s, q) = try makeSolver(keep: keep, separating: true, smallWorld: smallWorld, colliders: Self.oak)
                let (bars, busy) = try buildStock(s)
                var sleptAt: Int?
                step(s, q, frames: 360, keepAwake: [busy]) { f in if sleptAt == nil, bars.allSatisfy(s.isAsleep) { sleptAt = f } }
                XCTAssertNotNil(sleptAt, "the stock falls asleep (keep \(keep))")
                step(s, q, frames: 60, keepAwake: [busy])                  // the world steps on; the stock sleeps
                let listed = s.contacts(touching: Set(bars))
                if keep {
                    XCTAssertFalse(listed.isEmpty, "the sleeping stock's contacts are carried")
                    XCTAssertTrue(listed.allSatisfy { $0.ext.z > 0.5 }, "every carried contact is dormant")
                    XCTAssertTrue(listed.allSatisfy { $0.tan2.w < 0 }, "no dormant contact is coloured")
                } else {
                    XCTAssertTrue(listed.isEmpty, "default: a sleeping pair generates nothing (VZ-0152)")
                }
                let before = bars.map { s.position(of: $0)! }
                s.wake(bars)
                var firstSpeed: Float = 0
                step(s, q, frames: 1, keepAwake: [busy]) { _ in
                    firstSpeed = bars.map { simd_length(s.velocity(of: $0) ?? .zero) }.max() ?? 0
                }
                step(s, q, frames: 90, keepAwake: [busy])
                let move = zip(bars, before).map { simd_distance(s.position(of: $0.0)!, $0.1) }.max()!
                worst[keep] = (move, firstSpeed, listed.count)
                XCTAssertTrue(bars.allSatisfy(s.isAsleep), "the woken stock sleeps again (keep \(keep))")
            }
            let off = worst[false]!, on = worst[true]!
            print(String(format: "SLEEP_stock %@: woken after 1 s asleep — cold (default): first-frame speed %.2f mm/s, move %.4f mm | carried (%d dormant): %.2f mm/s, %.4f mm",
                         smallWorld ? "small-world" : "multi-dispatch", off.speed * 1000, off.move * 1000, on.listed, on.speed * 1000, on.move * 1000))
            XCTAssertLessThan(on.speed, 0.002, "a warm wake does not drop the stack (\(smallWorld ? "small-world" : "multi-dispatch"))")
            XCTAssertLessThan(on.move, 0.00002, "a warm wake does not move the stack")
        }
    }

    /// The flag changes nothing while nothing sleeps: an awake world with it on steps exactly as with
    /// it off (the carry only ever copies contacts whose ends are both inert).
    func testKeepSleepingContactsIsInertWhileEverythingIsAwake() throws {
        var runs: [[SIMD3<Float>]] = []
        for keep in [false, true] {
            let (s, q) = try makeSolver(keep: keep, separating: true, colliders: Self.oak)
            s.sleepEnabled = false
            let (bars, _) = try buildStock(s)
            step(s, q, frames: 120)
            runs.append(bars.map { s.position(of: $0)! })
        }
        XCTAssertEqual(runs[0], runs[1], "bit-identical with nothing asleep")
    }

    /// The split-impulse bias row with the flag only ever PUSHES. A tall box A pressed 0.3 mm into a
    /// low static block on its left is recovered to the right, AWAY from a small box B floating beside
    /// A's upper half 0.2 mm off it (inside the 2.5 mm speculative margin, never touching it; 4 mm over
    /// the block): off (the default), the speculative A–B bias row drives their relative bias
    /// velocity to zero both ways and drags B after A through the position channel; on, B stays put.
    func testSeparatingBiasNeverPullsASpeculativeNeighbour() throws {
        var moves: [Bool: Float] = [:]
        for separating in [false, true] {
            let block = CoinStaticCollider.box(center: SIMD3(-0.0015, 0.0015, 0), halfExtents: SIMD3(0.0015, 0.0015, 0.004),
                                               friction: 0, restitution: 0)
            let (s, q) = try makeSolver(keep: false, separating: separating, colliders: [block])
            s.gravity = 0
            s.sleepEnabled = false
            let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0.0025 - 0.0003, 0.005, 0), halfExtents: SIMD3(0.0025, 0.005, 0.0025),
                                             mass: 0.001, friction: 0, restitution: 0))
            let b = try XCTUnwrap(s.spawnBox(at: SIMD3(-0.0003 - 0.0002 - 0.00125, 0.008, 0), halfExtents: SIMD3(0.00125, 0.001, 0.00125),
                                             mass: 0.0002, friction: 0, restitution: 0))
            let a0 = s.position(of: a)!, b0 = s.position(of: b)!
            step(s, q, frames: 30)
            XCTAssertGreaterThan(s.position(of: a)!.x - a0.x, 0.0001, "A is recovered out of the block (towards the slop)")
            moves[separating] = simd_distance(s.position(of: b)!, b0)
        }
        print(String(format: "SLEEP_bias: the untouched neighbour moved %.4f mm (bilateral, default) / %.4f mm (separating)",
                     moves[false]! * 1000, moves[true]! * 1000))
        XCTAssertLessThan(moves[true]!, 0.000001, "a separating bias row never moves a body it does not touch")
        XCTAssertGreaterThan(moves[false]!, 0.00005, "the default's bilateral row drags it (the mechanism this flag removes)")
    }
}
