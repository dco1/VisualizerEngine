import XCTest
import Foundation
import Metal
import simd
@testable import VisualizerRendering

/// `enableWeld(…, bodyBAnchor:)` — a pooled weld that brings a point of B to A's anchor: B is held
/// that far from where it was welded (Digital Clock's fitter lifting a bar 0.3 mm off the sawhorse
/// it rests on, so it is carried clear instead of dragged across it). Default (nil) = unchanged.
@MainActor
final class CoinDEMWeldOffsetTests: XCTestCase {

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

    private func run(offset: Float?) throws -> (lift: Float, heldGap: Float, aMove: Float) {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 8, coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.2, -0.05, -0.2), boundsMax: SIMD3(0.2, 0.3, 0.2))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 9.81
        s.fixedDt = 1.0 / 180
        s.velocityIterations = 6
        s.warmStart = true
        s.manifoldSolve = true
        s.contactSlop = 1e-4
        s.baumgarteBeta = 0.2
        s.speculativeMargin = 2.5e-3
        s.sleepEnabled = false
        s.jointInnerPasses = 6
        s.setColliders([.box(center: SIMD3(0, -0.01, 0), halfExtents: SIMD3(0.1, 0.01, 0.1), friction: 0.5, restitution: 0.1)])
        // A: an 11 g "worker" block standing on the slab; B: a 1 g "bar" lying on the slab beside it.
        let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.01 - 1e-4, 0), halfExtents: SIMD3(0.01, 0.01, 0.01), mass: 0.011))
        let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0.014, 0.0025 - 1e-4, 0), halfExtents: SIMD3(0.0035, 0.0025, 0.0035), mass: 0.001))
        func frames(_ n: Int) {
            for _ in 0..<n {
                guard let cb = queue.makeCommandBuffer() else { return }
                s.encode(to: cb, wallDt: 1.0 / 60)
                cb.commit()
                cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
            }
        }
        frames(30)
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: a))
        let w = try XCTUnwrap(pool.takeWeld())
        let a0 = s.position(of: a)!, b0 = s.position(of: b)!
        let anchor = b0 + SIMD3(-0.0035, 0.002, 0)                     // "the hands": B's near upper edge
        s.enableWeld(w, bodyA: a, bodyB: b, worldAnchor: anchor, bodyBAnchor: offset.map { anchor - SIMD3(0, $0, 0) })
        frames(60)
        let b1 = s.position(of: b)!, a1 = s.position(of: a)!
        // B's underside over the slab's top (its collider bottom = centre − 2.5 mm).
        let gap = b1.y - 0.0025
        return (b1.y - b0.y, gap, simd_distance(a1, a0))
    }

    func testWeldOffsetLiftsTheBodyAndHoldsIt() throws {
        let plain = try run(offset: nil)
        let lifted = try run(offset: 0.0003)
        print(String(format: "WELD_offset: plain weld moves B %.4f mm (gap %.4f mm) · 0.3 mm offset lifts B %.4f mm (its underside %.4f mm over the slab), A moved %.4f mm",
                     plain.lift * 1000, plain.heldGap * 1000, lifted.lift * 1000, lifted.heldGap * 1000, lifted.aMove * 1000))
        XCTAssertLessThan(abs(plain.lift), 0.00002, "without an offset the weld holds B where it was welded")
        XCTAssertEqual(lifted.lift, 0.0003, accuracy: 0.00003, "the offset lifts B by it")
        XCTAssertGreaterThan(lifted.heldGap, 0.00015, "B hangs clear of the slab it lay on")
        XCTAssertLessThan(lifted.aMove, 0.0001, "the 11 g block barely moves")
    }

    /// `bodyBTurn`: the weld turns B by a world rotation from where it was welded and holds it there —
    /// a magnet pulling a face that touched it 2° off square flush. Floating pair, no gravity.
    func testWeldTurnSquaresTheBodyAndHoldsIt() throws {
        let (engine, lib, queue) = try Self.shared()
        guard let s = CoinDEMSolver(engine: engine, library: lib, maxCoins: 8, coinRadius: 0.03, halfThickness: 0.03,
                                    boundsMin: SIMD3(-0.2, -0.05, -0.2), boundsMax: SIMD3(0.2, 0.3, 0.2))
        else { throw XCTSkip("solver init failed") }
        s.solverMode = .constraint
        s.gravity = 0
        s.fixedDt = 1.0 / 180
        s.velocityIterations = 6
        s.warmStart = true
        s.sleepEnabled = false
        s.jointInnerPasses = 6
        s.setColliders([])
        let tilt = simd_quatf(angle: 2 * .pi / 180, axis: SIMD3(1, 0, 0))
        let a = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.1, 0), halfExtents: SIMD3(0.004, 0.004, 0.004), mass: 0.01))
        let b = try XCTUnwrap(s.spawnBox(at: SIMD3(0, 0.1, 0.0065), halfExtents: SIMD3(0.012, 0.0035, 0.00275),
                                         orient: SIMD4(tilt.imag, tilt.real), mass: 0.001))
        let pool = try XCTUnwrap(CoinJointPool(solver: s, reserve: 2, placeholderBody: a))
        let w = try XCTUnwrap(pool.takeWeld())
        s.enableWeld(w, bodyA: a, bodyB: b, worldAnchor: SIMD3(0, 0.1, 0.004), bodyBTurn: tilt.inverse)
        for _ in 0..<60 {
            guard let cb = queue.makeCommandBuffer() else { break }
            s.encode(to: cb, wallDt: 1.0 / 60)
            cb.commit()
            cb.waitUntilCompleted()  // gpu-ok: test harness must read results synchronously
        }
        let rel = (s.orientation(of: a)!.inverse * s.orientation(of: b)!).normalized
        let off = 2 * atan2(simd_length(rel.imag), abs(rel.real)) * 180 / .pi
        print(String(format: "WELD_turn: B welded 2.00° off square, turned to %.3f° off A (should be 0)", off))
        XCTAssertLessThan(off, 0.1, "the weld turns B square to A and holds it")
    }
}
