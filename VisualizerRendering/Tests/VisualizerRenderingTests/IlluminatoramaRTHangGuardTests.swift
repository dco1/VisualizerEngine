import XCTest
@testable import VisualizerRendering

/// The RT hang-guard's recovery semantics.
///
/// The bug these pin down: the guard used to LATCH. In a document editor the
/// model mutates on every tick of a drag (push/pull a wall ⇒ new wall meshes ⇒
/// a new TLAS topology hash every frame), so a few frames into the gesture the
/// thrash counter tripped, RT glass switched off mid-drag, **and never came
/// back** — a perf spike followed by a permanent visual pop with no
/// user-reachable way to restore it short of reopening the document.
///
/// Two properties have to hold at once, and they pull against each other:
///   1. a scene that SETTLES gets RT back (the editor case), and
///   2. a scene that genuinely thrashes every frame (GPU-fed foliage) stays
///      disabled — that is the hang-proofing the guard exists for.
final class IlluminatoramaRTHangGuardTests: XCTestCase {

    private let stable = IlluminatoramaRTHangGuard.recoveryStableFrames

    func testFreshGuardAdmitsEverything() {
        var g = IlluminatoramaRTHangGuard()
        XCTAssertFalse(g.autoDisabled)
        XCTAssertNil(g.trip)
        XCTAssertNil(g.framesUntilRetry)
        XCTAssertTrue(g.admitsRetry(topologyHash: 1))
        XCTAssertTrue(g.admitsRetry(topologyHash: 2), "a fresh guard never gates on topology")
    }

    func testTripDisablesAndRecordsTheReason() {
        var g = IlluminatoramaRTHangGuard()
        XCTAssertTrue(g.recordTrip(.topologyThrashing), "first trip is a state transition — log it")
        XCTAssertTrue(g.autoDisabled)
        XCTAssertEqual(g.trip, .topologyThrashing)
        XCTAssertEqual(g.tripCount, 1)
        XCTAssertEqual(g.framesUntilRetry, stable)
    }

    func testRepeatedTripOfTheSameReasonIsNotANewTransition() {
        // Both trip sites are re-evaluated EVERY frame now. If a repeat counted
        // as a transition the renderer would emit a warning per frame for the
        // whole length of a drag.
        var g = IlluminatoramaRTHangGuard()
        XCTAssertTrue(g.recordTrip(.sceneTooHeavy))
        for _ in 0..<200 {
            XCTAssertFalse(g.recordTrip(.sceneTooHeavy), "same reason ⇒ no new transition, no new log")
        }
        XCTAssertEqual(g.tripCount, 1)
    }

    func testChangedReasonIsANewTransition() {
        var g = IlluminatoramaRTHangGuard()
        XCTAssertTrue(g.recordTrip(.topologyThrashing))
        XCTAssertTrue(g.recordTrip(.sceneTooHeavy), "the reason changed — the host wants to know")
        XCTAssertEqual(g.trip, .sceneTooHeavy)
        XCTAssertEqual(g.tripCount, 2)
    }

    // ── property 1: a settled scene recovers ────────────────────────────────

    func testSettledTopologyReArmsRT() {
        var g = IlluminatoramaRTHangGuard()
        g.recordTrip(.topologyThrashing)
        // The drag ends: the same topology now holds frame after frame.
        for frame in 0..<stable {
            XCTAssertFalse(g.admitsRetry(topologyHash: 0xBEEF),
                           "must not re-arm early (frame \(frame))")
            XCTAssertTrue(g.autoDisabled)
        }
        XCTAssertTrue(g.admitsRetry(topologyHash: 0xBEEF), "settled ⇒ RT comes back")
        XCTAssertFalse(g.autoDisabled)
        XCTAssertNil(g.trip)
        XCTAssertNil(g.framesUntilRetry)
        XCTAssertEqual(g.tripCount, 1, "trip history survives a recovery")
    }

    func testCountdownIsReadableAndReachesZeroExactlyOnRecovery() {
        var g = IlluminatoramaRTHangGuard()
        g.recordTrip(.topologyThrashing)
        XCTAssertEqual(g.framesUntilRetry, stable)
        _ = g.admitsRetry(topologyHash: 7)            // baseline frame
        XCTAssertEqual(g.framesUntilRetry, stable)
        for i in 1..<stable {
            _ = g.admitsRetry(topologyHash: 7)
            XCTAssertEqual(g.framesUntilRetry, stable - i)
        }
        XCTAssertEqual(g.framesUntilRetry, 1)
        XCTAssertTrue(g.admitsRetry(topologyHash: 7))
    }

    // ── property 2: a thrashing scene stays disabled ────────────────────────

    func testTopologyChangingEveryFrameNeverReArms() {
        var g = IlluminatoramaRTHangGuard()
        g.recordTrip(.topologyThrashing)
        for frame in 0..<(stable * 20) {
            XCTAssertFalse(g.admitsRetry(topologyHash: frame),
                           "a per-frame topology change must never re-arm RT")
        }
        XCTAssertTrue(g.autoDisabled)
        XCTAssertEqual(g.framesUntilRetry, stable, "the countdown resets on every churn frame")
    }

    func testAlmostSettledThenOneChangeRestartsTheWindow() {
        // The mid-drag pause: the pointer hovers, the topology holds for a
        // while, then the drag resumes. Re-arming on a partial window would
        // rebuild the TLAS straight into the next churn.
        var g = IlluminatoramaRTHangGuard()
        g.recordTrip(.topologyThrashing)
        for _ in 0..<(stable - 1) { _ = g.admitsRetry(topologyHash: 1) }
        XCTAssertFalse(g.admitsRetry(topologyHash: 2), "topology moved — start over")
        XCTAssertEqual(g.framesUntilRetry, stable)
        for _ in 0..<(stable - 1) { XCTAssertFalse(g.admitsRetry(topologyHash: 2)) }
        XCTAssertTrue(g.admitsRetry(topologyHash: 2))
    }

    func testATripDuringTheSettleWindowRestartsIt() {
        var g = IlluminatoramaRTHangGuard()
        g.recordTrip(.topologyThrashing)
        for _ in 0..<(stable - 2) { _ = g.admitsRetry(topologyHash: 3) }
        g.recordTrip(.sceneTooHeavy)
        XCTAssertEqual(g.framesUntilRetry, stable, "a fresh trip starts its countdown from zero")
    }

    // ── the drag → settle → drag → settle cycle end to end ──────────────────

    func testDragSettleDragSettleCycle() {
        var g = IlluminatoramaRTHangGuard()
        for cycle in 0..<3 {
            // Drag: topology churns, guard trips, RT stays off for the gesture.
            g.recordTrip(.topologyThrashing)
            for f in 0..<40 {
                XCTAssertFalse(g.admitsRetry(topologyHash: cycle * 1000 + f))
            }
            // Settle.
            for _ in 0..<stable { XCTAssertFalse(g.admitsRetry(topologyHash: cycle)) }
            XCTAssertTrue(g.admitsRetry(topologyHash: cycle), "cycle \(cycle) must recover")
            XCTAssertFalse(g.autoDisabled)
        }
        XCTAssertEqual(g.tripCount, 3, "every gesture is counted — a host can see the flapping")
    }

    func testResetClearsEverythingIncludingHistory() {
        var g = IlluminatoramaRTHangGuard()
        g.recordTrip(.sceneTooHeavy)
        _ = g.admitsRetry(topologyHash: 5)
        g.reset()
        XCTAssertEqual(g, IlluminatoramaRTHangGuard())
        XCTAssertEqual(g.tripCount, 0)
        XCTAssertTrue(g.admitsRetry(topologyHash: 99))
    }
}
