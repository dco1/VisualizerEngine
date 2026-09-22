import XCTest
@testable import VisualizerRendering

/// The DDGI settle gate's state machine, as a pure value — no Metal, no renderer
/// (this package's `swift test` produces no metallib and cannot construct one).
///
/// The two properties that pull against each other, exactly like the RT hang
/// guard's:
///   1. a SETTLED field (camera-only orbit — the hash holds still) stops tracing
///      after the convergence window, so an orbit costs nothing; and
///   2. a MOVING field (a wall drag, the sun rotating — the hash changes every
///      frame) keeps tracing, so the picture is never stale.
/// Plus the opt-out that guarantees zero blast radius: budget 0 ⇒ trace always.
final class DDGIConvergenceGateTests: XCTestCase {

    private let budget = 100

    // ── the opt-out: legacy every-frame behaviour ───────────────────────────

    func testZeroBudgetAlwaysTraces() {
        var g = DDGIConvergenceGate()
        for f in 0..<500 {
            // Even with the hash held perfectly still, budget 0 traces every frame.
            XCTAssertTrue(g.shouldTrace(inputHash: 0xABCD, convergenceFrames: 0),
                          "budget 0 must trace unconditionally (frame \(f))")
            XCTAssertTrue(g.didTraceLastQuery)
        }
    }

    func testNegativeBudgetAlsoAlwaysTraces() {
        var g = DDGIConvergenceGate()
        XCTAssertTrue(g.shouldTrace(inputHash: 1, convergenceFrames: -5))
    }

    // ── property 1: a settled field stops tracing ───────────────────────────

    func testSettledFieldTracesExactlyBudgetFramesThenStops() {
        var g = DDGIConvergenceGate()
        for f in 0..<budget {
            XCTAssertTrue(g.shouldTrace(inputHash: 7, convergenceFrames: budget),
                          "still converging at frame \(f)")
        }
        // Budget spent, hash unchanged (a continuing orbit): every further frame skips.
        for f in 0..<1000 {
            XCTAssertFalse(g.shouldTrace(inputHash: 7, convergenceFrames: budget),
                           "converged field must not re-trace on a camera-only frame \(f)")
            XCTAssertFalse(g.didTraceLastQuery)
        }
    }

    func testCountdownIsReadableAndReachesZero() {
        var g = DDGIConvergenceGate()
        _ = g.shouldTrace(inputHash: 1, convergenceFrames: 5)  // re-arm + trace 1 (4 left)
        XCTAssertEqual(g.framesUntilConverged, 4)
        for expected in stride(from: 3, through: 0, by: -1) {
            _ = g.shouldTrace(inputHash: 1, convergenceFrames: 5)
            XCTAssertEqual(g.framesUntilConverged, expected)
        }
        XCTAssertFalse(g.shouldTrace(inputHash: 1, convergenceFrames: 5), "settled")
        XCTAssertEqual(g.framesUntilConverged, 0)
    }

    // ── property 2: a moving field keeps tracing ────────────────────────────

    func testHashChangeReArmsTheWindow() {
        var g = DDGIConvergenceGate()
        for _ in 0..<budget { _ = g.shouldTrace(inputHash: 1, convergenceFrames: budget) }
        XCTAssertFalse(g.shouldTrace(inputHash: 1, convergenceFrames: budget), "settled")
        // The sun rotates — a new hash. Trace resumes for a fresh window.
        XCTAssertTrue(g.shouldTrace(inputHash: 2, convergenceFrames: budget),
                      "an input change must re-arm the trace")
        XCTAssertEqual(g.framesUntilConverged, budget - 1)
    }

    func testEveryFrameHashChangeTracesEveryFrame() {
        var g = DDGIConvergenceGate()
        // A live wall drag: the geometry epoch (folded into the hash) moves each frame.
        for f in 0..<(budget * 5) {
            XCTAssertTrue(g.shouldTrace(inputHash: UInt64(f), convergenceFrames: budget),
                          "a per-frame input change must trace every frame (\(f))")
        }
    }

    func testAlmostSettledThenOneChangeRestartsTheWindow() {
        var g = DDGIConvergenceGate()
        for _ in 0..<(budget - 1) { _ = g.shouldTrace(inputHash: 1, convergenceFrames: budget) }
        // One frame short of converged, the input moves — the window restarts in full.
        XCTAssertTrue(g.shouldTrace(inputHash: 9, convergenceFrames: budget))
        XCTAssertEqual(g.framesUntilConverged, budget - 1)
    }

    // ── reset ───────────────────────────────────────────────────────────────

    func testResetReArms() {
        var g = DDGIConvergenceGate()
        for _ in 0..<budget { _ = g.shouldTrace(inputHash: 1, convergenceFrames: budget) }
        XCTAssertFalse(g.shouldTrace(inputHash: 1, convergenceFrames: budget), "settled")
        g.reset()
        // Same hash as before the reset, yet it re-traces — reset forgets the field.
        XCTAssertTrue(g.shouldTrace(inputHash: 1, convergenceFrames: budget),
                      "reset must force a re-trace even on an unchanged hash")
    }

    func testFreshGateTracesOnFirstQuery() {
        var g = DDGIConvergenceGate()
        XCTAssertTrue(g.shouldTrace(inputHash: 0, convergenceFrames: budget),
                      "a fresh gate (lastHash nil) re-arms and traces")
    }
}
