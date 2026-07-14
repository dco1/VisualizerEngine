import XCTest
import simd
@testable import VisualizerRendering

/// GPU-free correctness gate for the loss-of-support drop animation. Proves the
/// host-side `DropBounceTracker` (whose scalar offset is fed to the GPU `swayJostle`
/// channel) falls from the old support height, bounces, and — the key requirement —
/// resolves to EXACTLY the grounded pose (offset 0), so a settled lamp matches a
/// fresh grounded placement rather than floating.
final class DropBounceTrackerTests: XCTestCase {

    // Appearing in the scene is not a fall: the first frame an object is seen must
    // never animate (otherwise every placed lamp would drop on creation).
    func testFirstSightingNeverFalls() {
        var t = DropBounceTracker()
        XCTAssertEqual(t.update(id: 1, restY: 0.74), 0, accuracy: 0)
    }

    // A steady support height produces no motion, ever.
    func testStableSupportStaysQuiet() {
        var t = DropBounceTracker()
        _ = t.update(id: 1, restY: 0.74)
        for _ in 0..<120 {
            XCTAssertEqual(t.update(id: 1, restY: 0.74), 0, accuracy: 0)
        }
    }

    // Raising support (placing a lamp ONTO a table) is not a fall — no animation.
    func testRisingSupportDoesNotAnimate() {
        var t = DropBounceTracker()
        _ = t.update(id: 1, restY: 0)
        XCTAssertEqual(t.update(id: 1, restY: 0.74), 0, accuracy: 0)
    }

    // The core gate: a table lamp resting at 0.74 m loses its support (restY
    // collapses to 0). It must fall from near the old height, bounce at least once,
    // and settle EXACTLY at the grounded pose — and stay there.
    func testLossOfSupportFallsBouncesAndSettlesToGround() {
        var t = DropBounceTracker()
        let table: Float = 0.74
        _ = t.update(id: 1, restY: table)            // resting on the table

        // Table deleted: support height is the floor (0) from here on.
        var offsets: [Float] = []
        for _ in 0..<240 {
            let o = t.update(id: 1, restY: 0)
            XCTAssertGreaterThanOrEqual(o, 0, "object must never sink below its rest plane")
            offsets.append(o)
        }

        // Begins the fall near where it was supported.
        XCTAssertGreaterThan(offsets.first ?? 0, table * 0.9,
                             "fall should begin from the old support height")

        // Fully settled and STAYS settled: the resolved pose is the grounded pose.
        XCTAssertEqual(offsets.last!, 0, accuracy: 0, "must resolve to the grounded pose")
        let lastActive = offsets.lastIndex(where: { $0 > 1e-5 }) ?? -1
        XCTAssertGreaterThan(lastActive, 0, "the drop must actually animate")
        XCTAssertLessThan(lastActive, 90, "settle should feel snappy (< ~1.5 s at 60 fps)")

        // It bounced: somewhere before settling it touched down (~0) then rose again.
        var bounced = false
        for i in 1..<lastActive where offsets[i] <= 1e-4 && offsets[i + 1] > 1e-3 {
            bounced = true; break
        }
        XCTAssertTrue(bounced, "a settling bounce should lift the object back up at least once")
    }

    // Objects animate independently, keyed by id: dropping one leaves the other still.
    func testPerObjectIndependence() {
        var t = DropBounceTracker()
        _ = t.update(id: 1, restY: 0.74)
        _ = t.update(id: 2, restY: 0.50)
        let a = t.update(id: 1, restY: 0)      // id 1 loses its support
        let b = t.update(id: 2, restY: 0.50)   // id 2 unchanged
        XCTAssertGreaterThan(a, 0, "the unsupported object falls")
        XCTAssertEqual(b, 0, accuracy: 0, "the still-supported object does not move")
    }
}
