import simd
import XCTest
@testable import VisualizerVegetation

/// The two invariants the shared margin resolver promises, and that everything downstream assumes.
final class LeafSilhouetteTests: XCTestCase {

    /// `subdivisions <= 1` must return the authored controls UNCHANGED.
    ///
    /// This is not an optimization to be tidied away later: it is the byte-identity escape hatch the
    /// yard-tree bake stands on. `ForestTreeValidityTests` pins that tree's exact baked triangle
    /// totals and asserts a hero leaf card is exactly 20 triangles, and a smoother that quietly
    /// started rounding at `subdivisions: 1` would move both — as a diff in a gate about trees, in a
    /// file about outlines, with nothing naming the cause.
    func testSmoothIsTheIdentityAtOrBelowOneSubdivision() {
        for s in shippedSilhouettes {
            for tier: LeafSilhouette.Tier in [.hero, .mid] {
                let authored = s.controls(tier: tier)
                for subdivisions in [-3, 0, 1] {
                    XCTAssertEqual(LeafSilhouette.smooth(authored, subdivisions: subdivisions),
                                   authored,
                                   "\(s.name)/\(tier) changed at subdivisions \(subdivisions)")
                    XCTAssertEqual(s.margin(tier: tier, subdivisions: subdivisions), authored,
                                   "\(s.name)/\(tier) changed via margin() at \(subdivisions)")
                }
            }
        }
    }

    /// Smoothing must keep `v` non-decreasing and `u` non-negative, for every shipped outline at
    /// every rounding a species might ask for.
    ///
    /// Both failures are silent and both wreck the blade rather than blemish it. A Catmull-Rom
    /// overshoot that reverses `v` walks the midrib parameter backwards for one segment, which
    /// inverts that strip triangle — a single black facet in the middle of an otherwise fine leaf,
    /// under a single-sided material. A negative `u` folds the margin across the midrib and turns
    /// the blade inside out along that run. The resolver spends a linear `v` interpolation and a
    /// `max(0, ·)` to prevent exactly these; this is the test that says so.
    func testSmoothKeepsVMonotonicAndUNonNegative() {
        for s in shippedSilhouettes {
            for tier: LeafSilhouette.Tier in [.hero, .mid] {
                for subdivisions in [2, 3, 4] {
                    let m = s.margin(tier: tier, subdivisions: subdivisions)
                    XCTAssertGreaterThanOrEqual(m.count, s.controls(tier: tier).count,
                                                "\(s.name) lost controls at \(subdivisions)")
                    for i in 0 ..< m.count {
                        XCTAssertGreaterThanOrEqual(m[i].u, 0,
                            "\(s.name)/\(tier)@\(subdivisions): u went negative at \(i) (\(m[i].u))")
                        XCTAssertTrue(m[i].u.isFinite && m[i].v.isFinite,
                            "\(s.name)/\(tier)@\(subdivisions): non-finite control at \(i)")
                        if i > 0 {
                            XCTAssertGreaterThanOrEqual(m[i].v, m[i - 1].v,
                                "\(s.name)/\(tier)@\(subdivisions): v reversed at \(i)")
                        }
                    }
                    // The endpoints are the blade's base and tip and must survive exactly, or the
                    // blade changes length as a side effect of asking for a rounder outline.
                    XCTAssertEqual(m.first!, s.controls(tier: tier).first!)
                    XCTAssertEqual(m.last!, s.controls(tier: tier).last!)
                }
            }
        }
    }

    /// The catalog's own convention: a half-margin starts and ends on the midrib.
    ///
    /// Both constructors rely on it — the blade to close its tip, the pad to pinch its ends into the
    /// fans that make the solid watertight — so an outline authored without it is a hole waiting to
    /// happen rather than a stylistic choice.
    func testEveryShippedOutlinePinchesAtBaseAndTip() {
        for s in shippedSilhouettes {
            for tier: LeafSilhouette.Tier in [.hero, .mid] {
                let c = s.controls(tier: tier)
                XCTAssertGreaterThanOrEqual(c.count, 3, "\(s.name)/\(tier) is too short to be an outline")
                XCTAssertEqual(c.first!.v, 0, accuracy: 1e-12, "\(s.name)/\(tier) does not start at the base")
                XCTAssertEqual(c.first!.u, 0, accuracy: 1e-12, "\(s.name)/\(tier) base is not pinched")
                XCTAssertEqual(c.last!.v, 1, accuracy: 1e-12, "\(s.name)/\(tier) does not end at the tip")
                XCTAssertEqual(c.last!.u, 0, accuracy: 1e-12, "\(s.name)/\(tier) tip is not pinched")
            }
        }
    }
}
