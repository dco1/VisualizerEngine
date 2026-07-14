import XCTest
import simd
@testable import VisualizerRendering

/// Deterministic, GPU-free verification of the drag/impact secondary-motion driver
/// (`DragSwayTracker`) — the host-side counterpart to the `applySway` vertex shader.
/// Pure spring physics, so the behaviour is asserted as numbers, not eyeballed.
final class DragSwayTrackerTests: XCTestCase {

    /// At rest with no input, the tracker reports a flat (zero) sway.
    func testSwayIsZeroAtRest() {
        var t = DragSwayTracker()
        let s = t.update(id: 1, pos: SIMD2<Float>(0, 0), angle: 0)
        XCTAssertEqual(s.leanX, 0, accuracy: 1e-6)
        XCTAssertEqual(s.jostle, 0, accuracy: 1e-6)
    }

    /// Steady acceleration along +X makes the object *trail* (lean −X) — the inertia
    /// lag of a normal drag, asserted so the recoil's opposite sign is meaningful.
    func testSteadyDragLeansAgainstMotion() {
        var t = DragSwayTracker()
        var x: Float = 0
        _ = t.update(id: 1, pos: SIMD2<Float>(x, 0), angle: 0)        // seed prevPos
        for _ in 0 ..< 6 { x += 0.02; _ = t.update(id: 1, pos: SIMD2<Float>(x, 0), angle: 0) }
        let s = t.update(id: 1, pos: SIMD2<Float>(x + 0.02, 0), angle: 0)
        XCTAssertLessThan(s.leanX, -0.02, "object trails the +X motion")
    }

    /// A registered knock lurches the object the way it was heading (+X here),
    /// overshoots back through zero (a real spring, not a decay), then settles to rest —
    /// the distinct collision reaction on top of the steady lean.
    func testImpactRecoilLurchesOvershootsAndSettles() {
        var t = DragSwayTracker()
        _ = t.update(id: 7, pos: SIMD2<Float>(0, 0), angle: 0)        // at rest, no drag velocity
        t.registerImpact(id: 7, worldDir: SIMD2(1, 0), speed: 0.03, angle: 0)

        // Early frames: a clear forward (+X) lurch.
        var peak: Float = 0
        for _ in 0 ..< 12 { peak = max(peak, t.update(id: 7, pos: SIMD2<Float>(0, 0), angle: 0).leanX) }
        XCTAssertGreaterThan(peak, 0.03, "knock produces a visible forward lurch")

        // It must swing back past zero (overshoot), not merely decay toward it.
        var minLean: Float = 0
        for _ in 0 ..< 40 { minLean = min(minLean, t.update(id: 7, pos: SIMD2<Float>(0, 0), angle: 0).leanX) }
        XCTAssertLessThan(minLean, -0.001, "spring overshoots through zero")

        // And it settles: after ~2 s of frames the recoil is gone.
        var last: Float = 1
        for _ in 0 ..< 120 { last = t.update(id: 7, pos: SIMD2<Float>(0, 0), angle: 0).leanX }
        XCTAssertEqual(last, 0, accuracy: 1e-3, "recoil settles to rest")
    }

    /// The knock is projected into the object's local frame: the same world-space impact
    /// drives leanX through the object's rotation, so a turned object recoils about its
    /// own width axis, not the world X.
    func testImpactProjectsIntoLocalFrame() {
        var t = DragSwayTracker()
        // Object rotated 90°: a world +Y push is along its local X.
        _ = t.update(id: 3, pos: SIMD2<Float>(0, 0), angle: .pi / 2)
        t.registerImpact(id: 3, worldDir: SIMD2(0, 1), speed: 0.03, angle: .pi / 2)
        var peak: Float = 0
        for _ in 0 ..< 12 { peak = max(peak, t.update(id: 3, pos: SIMD2<Float>(0, 0), angle: .pi / 2).leanX) }
        XCTAssertGreaterThan(peak, 0.03, "world +Y maps to local-X lean for a 90°-turned object")
    }

    /// Micro-nudges below the speed floor don't fire a recoil (no buzzing on tiny jitter).
    func testTinyImpactIsIgnored() {
        var t = DragSwayTracker()
        _ = t.update(id: 9, pos: SIMD2<Float>(0, 0), angle: 0)
        t.registerImpact(id: 9, worldDir: SIMD2(1, 0), speed: 0.001, angle: 0)
        var peak: Float = 0
        for _ in 0 ..< 12 { peak = max(peak, abs(t.update(id: 9, pos: SIMD2<Float>(0, 0), angle: 0).leanX)) }
        XCTAssertEqual(peak, 0, accuracy: 1e-4, "sub-threshold knock is a no-op")
    }
}
