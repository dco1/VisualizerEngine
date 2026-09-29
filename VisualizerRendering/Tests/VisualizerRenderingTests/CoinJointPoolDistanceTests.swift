import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// `CoinDEMSolver.enableDistance` — a POOLED distance joint (rigid rod) switched on and off without
/// `wakeAll`: it holds its anchors at the rest length exactly like `addDistanceJoint`, wakes only
/// its own bodies, and `CoinJointPool.give` frees the body again. Numbers PRINTed with `PDIST_`.
@MainActor
final class CoinJointPoolDistanceTests: XCTestCase {

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

    func makeSolver() throws -> (CoinDEMSolver, MTLCommandQueue) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 8, coinRadius: 0.02, halfThickness: 0.02,
                                    boundsMin: SIMD3(-0.3, -0.1, -0.3), boundsMax: SIMD3(0.3, 0.3, 0.3))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.maxSubsteps = 10
        s.velocityIterations = 6
        s.warmStart = true
        s.manifoldSolve = true
        s.contactSlop = 1e-4
        s.restitution = 0.1
        s.floorY = -0.5
        s.sleepEnabled = true
        s.sleepFrames = 20
        s.sleepLinVel = 2e-4
        s.maxSpeed = 3
        s.maxHSpeed = 3
        s.setColliders([.plane(normal: SIMD3(0, 1, 0), offset: 0, friction: 0.5, restitution: 0.1)])
        return (s, queue)
    }

    func step(_ s: CoinDEMSolver, _ q: MTLCommandQueue, frames: Int, perFrame: ((Int) -> Void)? = nil) {
        for f in 0..<frames {
            guard let cb = q.makeCommandBuffer() else { return }
            s.encode(to: cb, wallDt: 1.0 / 60)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test reads every frame synchronously
            perFrame?(f)
        }
    }

    /// One pendulum: a bob hung from a world point 60 mm to its side at the same height, released
    /// from horizontal, on a pooled rod (`enableDistance`) or an `addDistanceJoint` rod. Returns the
    /// bob's height per frame, the worst rod-length error, and whether a sleeping neighbour woke.
    func pendulum(pooled: Bool) throws -> (ys: [Float], worst: Float, neighbourWoke: Bool, releasedY: Float) {
        let (s, q) = try makeSolver()
        guard let sleeper = s.spawnSphere(at: SIMD3(0.1, 0.01, 0), radius: 0.01, mass: 0.01) else { throw XCTSkip("spawn") }
        let pool = pooled ? CoinJointPool(solver: s, reserve: 1, placeholderBody: sleeper) : nil
        let slot = pool?.takeSlot()
        step(s, q, frames: 40)                       // the neighbour comes to rest and sleeps
        XCTAssertTrue(s.isAsleep(sleeper), "the neighbour is asleep before the rod")
        let start = SIMD3<Float>(-0.1, 0.08, 0), pivot = start + SIMD3(0.06, 0, 0)
        guard let bob = s.spawnSphere(at: start, radius: 0.01, mass: 0.01) else { throw XCTSkip("spawn") }
        var handle: Int?
        if let slot { s.enableDistance(slot: slot, bodyA: bob, bodyB: nil, worldAnchorA: start, worldAnchorB: pivot) }
        else { handle = s.addDistanceJoint(bodyA: bob, bodyB: nil, worldAnchorA: start, worldAnchorB: pivot) }
        let woke = !s.isAsleep(sleeper)
        var ys: [Float] = [], worst: Float = 0
        var st: [CoinBodyState] = []
        step(s, q, frames: 30) { _ in
            s.readStates([bob], into: &st)
            ys.append(st[0].position.y)
            worst = max(worst, abs(simd_distance(st[0].position, pivot) - 0.06))
        }
        if let slot { pool?.give(slot) } else if let handle { s.removeJoint(handle) }
        step(s, q, frames: 60)
        s.readStates([bob], into: &st)
        return (ys, worst, woke, st[0].position.y)
    }

    /// A pooled rod is the SAME rod as `addDistanceJoint`'s (same trajectory), wakes only its own
    /// body — a sleeping neighbour stays asleep, where add/remove wake the world — and `give`
    /// lets the body fall free.
    func testPooledRodMatchesAddDistanceJointWakesOnlyItsBodyAndReleases() throws {
        let p = try pendulum(pooled: true), a = try pendulum(pooled: false)
        var diff: Float = 0
        for (x, y) in zip(p.ys, a.ys) { diff = max(diff, abs(x - y)) }
        print(String(format: "PDIST_ pooled vs addDistanceJoint: max height difference %.6f mm over 30 frames; rod length error %.3f / %.3f mm; lowest %.2f mm; neighbour woke: pooled %d, add %d; released y %.2f mm",
                     diff * 1000, p.worst * 1000, a.worst * 1000, (p.ys.min() ?? 0) * 1000, p.neighbourWoke ? 1 : 0, a.neighbourWoke ? 1 : 0, p.releasedY * 1000))
        XCTAssertLessThan(diff, 1e-5, "the pooled rod swings exactly as addDistanceJoint's")
        XCTAssertLessThan(p.ys.min() ?? 1, 0.03, "the bob swung down on the rod")
        XCTAssertFalse(p.neighbourWoke, "enableDistance must not wake anything else (no wakeAll)")
        XCTAssertLessThan(p.releasedY, 0.0105, "given back, the rod no longer holds the bob")
    }
}
