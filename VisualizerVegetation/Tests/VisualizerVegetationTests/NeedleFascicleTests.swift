import simd
import XCTest
@testable import VisualizerVegetation

/// A fascicle is an ARRANGEMENT over the shared blade constructor. These tests check the arranging —
/// count, azimuth, frame, where the bases land — and deliberately do not re-test the stitch, which
/// `LeafConstructorTests` already owns. If a needle's triangles were ever produced by anything other
/// than `emitBlade`, the first assertion here is the one that would catch it.
final class NeedleFascicleTests: XCTestCase {

    private let sheath = SIMD3<Double>(1.2, 2.4, -0.8)
    private let axis = simd_normalize(SIMD3<Double>(0.2, 1.0, 0.35))
    private let reference = SIMD3<Double>(0, 1, 0)

    /// A bundle of N costs exactly N times one needle — no per-fascicle extra geometry, no sheath
    /// collar, no silently duplicated cap.
    ///
    /// The per-needle figure is the shared blade's own: `(controls - 1) × 2` strip quads at 2 or 4
    /// triangles each. Asserting the bundle against a SEPARATELY emitted single blade (rather than
    /// against a literal) is what makes this a test of the arrangement instead of a restatement of
    /// the arithmetic — it fails if the fascicle ever grows its own loop.
    func testFascicleCostsExactlyOneBladePerNeedle() {
        for winding: LeafConstructor.Winding in [.singleSided, .doubleSidedShell] {
            var one = CollectingSink<Double>()
            LeafConstructor.emitBlade(into: &one, placement: flatPlacement(width: 0.003, height: 0.09),
                                      silhouette: .needle, winding: winding)

            for count in 1 ... 5 {
                var bundle = CollectingSink<Double>()
                let reported = NeedleFascicle.emit(into: &bundle, sheath: sheath, axis: axis,
                                                   reference: reference, count: count,
                                                   length: 0.09, winding: winding)
                XCTAssertEqual(bundle.count, count * one.count,
                               "\(count) needles, \(winding)")
                XCTAssertEqual(reported, bundle.count, "reported cost disagrees with what was emitted")
                XCTAssertEqual(reported,
                               NeedleFascicle.triangleCount(count: count, winding: winding),
                               "triangleCount() disagrees with emit()")
                XCTAssertTrue(bundle.allValuesAreFinite)
            }
        }

        // The documented cost, written down so a silhouette or a default that changes it has to
        // change this line and say why. 12 triangles per needle single-sided, 24 double.
        XCTAssertEqual(NeedleFascicle.triangleCount(count: 1, winding: .singleSided), 12)
        XCTAssertEqual(NeedleFascicle.triangleCount(count: 1, winding: .doubleSidedShell), 24)
        XCTAssertEqual(NeedleFascicle.triangleCount(count: 3, winding: .singleSided), 36)
        XCTAssertEqual(NeedleFascicle.triangleCount(count: 5, winding: .doubleSidedShell), 120)
    }

    /// A degenerate bundle emits nothing rather than a NaN or a division by zero.
    func testEmptyOrZeroLengthFasciclesEmitNothing() {
        for (count, length) in [(0, 0.09), (-2, 0.09), (3, 0.0), (3, -0.09)] {
            var sink = CollectingSink<Double>()
            let n = NeedleFascicle.emit(into: &sink, sheath: sheath, axis: axis,
                                        reference: reference, count: count,
                                        length: length, winding: .singleSided)
            XCTAssertEqual(sink.count, 0, "count \(count), length \(length)")
            XCTAssertEqual(n, 0)
        }
    }

    /// Every needle in a bundle points somewhere different, and the whole set is spread evenly
    /// around the shoot.
    ///
    /// Two ways this goes wrong and neither is visible in a triangle count: an azimuth formula that
    /// wraps (index `count` landing on index 0) collapses two needles onto each other, and a
    /// half-step offset dropped from the formula puts needle 0 of every fascicle exactly on the
    /// reference vector — which grows a seam of co-planar needles the length of a shoot that shares
    /// one reference.
    func testNeedleAzimuthsAreDistinctEvenlySpacedAndOffTheReference() {
        for count in 1 ... 6 {
            let azimuths = (0 ..< count).map { NeedleFascicle.azimuth(index: $0, count: count) }
            for a in azimuths {
                XCTAssertTrue(a > 0 && a < 2 * Double.pi, "azimuth \(a) left the circle")
                // The half-step offset: no needle sits on azimuth zero, i.e. on `reference`.
                XCTAssertGreaterThan(min(a, 2 * Double.pi - a), 1e-6,
                                     "a needle landed exactly on the reference direction")
            }
            for i in 1 ..< count {
                XCTAssertEqual(azimuths[i] - azimuths[i - 1], 2 * Double.pi / Double(count),
                               accuracy: 1e-12, "azimuths are not evenly spaced at count \(count)")
            }

            // And the directions those azimuths produce are genuinely distinct in space.
            let f = NeedleFascicle.frame(axis: axis, reference: reference)
            let dirs = (0 ..< count).map {
                NeedleFascicle.needleDirection(index: $0, count: count, frame: f, splayRadians: 0.18)
            }
            for d in dirs {
                XCTAssertEqual(simd_length(d), 1.0, accuracy: 1e-12, "direction was not unit")
                XCTAssertEqual(simd_dot(d, f.axis), cos(0.18), accuracy: 1e-12,
                               "needle is not pitched off the axis by splayRadians")
            }
            for i in 0 ..< count {
                for j in (i + 1) ..< count {
                    XCTAssertLessThan(simd_dot(dirs[i], dirs[j]), 1.0 - 1e-9,
                                      "needles \(i) and \(j) of \(count) point the same way")
                }
            }
        }
    }

    /// The splay frame is orthonormal and right-handed, and survives a reference vector that is
    /// parallel to the axis (or zero) instead of producing a NaN frame that poisons every needle.
    func testFrameIsOrthonormalAndSurvivesADegenerateReference() {
        for reference in [SIMD3<Double>(0, 1, 0),
                          axis,                       // parallel — the Gram-Schmidt residue is zero
                          axis * -3.0,
                          SIMD3<Double>(0, 0, 0)] {   // no reference at all
            let f = NeedleFascicle.frame(axis: axis, reference: reference)
            XCTAssertEqual(simd_length(f.axis), 1.0, accuracy: 1e-12)
            XCTAssertEqual(simd_length(f.e1), 1.0, accuracy: 1e-12)
            XCTAssertEqual(simd_length(f.e2), 1.0, accuracy: 1e-12)
            XCTAssertEqual(simd_dot(f.axis, f.e1), 0, accuracy: 1e-12)
            XCTAssertEqual(simd_dot(f.axis, f.e2), 0, accuracy: 1e-12)
            XCTAssertEqual(simd_dot(f.e1, f.e2), 0, accuracy: 1e-12)
            XCTAssertEqual(simd_length(simd_cross(f.axis, f.e1) - f.e2), 0, accuracy: 1e-12,
                           "frame is left-handed")
        }
    }

    /// Every needle's base lands exactly on the sheath, and its tip a full `length` away.
    ///
    /// `Placement.position` is the blade CENTROID, so a fascicle has to push each needle forward by
    /// `length * 0.46` to put its base at the binding point. Getting that wrong is invisible on a
    /// leaf hanging off a petiole and glaring on a bundle: the needles no longer meet, and the
    /// fascicle reads as an exploded starburst floating above its shoot.
    func testEveryNeedleBaseLandsOnTheSheath() {
        let count = 5, length = 0.11
        var sink = CollectingSink<Double>()
        NeedleFascicle.emit(into: &sink, sheath: sheath, axis: axis, reference: reference,
                            count: count, length: length, curl: 0, winding: .singleSided)

        // With curl 0 the blade is planar, so distance-from-sheath is exact.
        let distances = sink.positions.map { simd_length($0 - sheath) }
        XCTAssertEqual(distances.min()!, 0, accuracy: 1e-12,
                       "no vertex sits at the sheath — the bundle is not bound")
        XCTAssertEqual(distances.max()!, length, accuracy: 1e-9,
                       "the needle tips are not exactly `length` from the sheath")

        // And it is every needle, not just one: each needle contributes at least one vertex at the
        // sheath (its pinched base) and one at full length (its pinched tip).
        let perNeedle = sink.count / count
        for n in 0 ..< count {
            let slice = sink.triangles[(n * perNeedle) ..< ((n + 1) * perNeedle)]
            let d = slice.flatMap { $0.vertices }.map { simd_length($0 - sheath) }
            XCTAssertEqual(d.min()!, 0, accuracy: 1e-12, "needle \(n) is not bound at the sheath")
            XCTAssertEqual(d.max()!, length, accuracy: 1e-9, "needle \(n) is the wrong length")
        }
    }

    /// A fascicle is a pure function of its arguments — the same call twice gives the same bytes, in
    /// the same order.
    ///
    /// This is the property that lets a caller own the randomness. A constructor that drew from its
    /// own RNG would make emission ORDER load-bearing: insert one fascicle upstream and every
    /// downstream needle moves, turning a baked-triangle-count gate into a tripwire that fires on
    /// unrelated edits.
    func testEmissionIsDeterministic() {
        for count in [1, 3, 5] {
            var a = CollectingSink<Double>(), b = CollectingSink<Double>()
            NeedleFascicle.emit(into: &a, sheath: sheath, axis: axis, reference: reference,
                                count: count, length: 0.09, curl: 0.14, subdivisions: 2,
                                winding: .doubleSidedShell)
            NeedleFascicle.emit(into: &b, sheath: sheath, axis: axis, reference: reference,
                                count: count, length: 0.09, curl: 0.14, subdivisions: 2,
                                winding: .doubleSidedShell)
            XCTAssertEqual(a.count, b.count)
            XCTAssertEqual(a.positions, b.positions, "fascicle of \(count) is not deterministic")
            for (x, y) in zip(a.triangles, b.triangles) {
                XCTAssertEqual(x.outward, y.outward)
                XCTAssertEqual(x.geometricNormal, y.geometricNormal)
            }
        }
    }

    /// Each needle presents its cup radially OUTWARD, away from the shoot axis.
    ///
    /// A bundle that cups inward hides its own shading gradient inside itself, where nothing can see
    /// it, and reads as a flat green smear. The presented normal is
    /// `radial·cos(splay) − axis·sin(splay)` in closed form precisely so it stays exactly
    /// perpendicular to the needle without a Gram-Schmidt that goes singular at 90° splay.
    func testNeedlesPresentTheirCupAwayFromTheShootAxis() {
        let count = 4
        let splay = 0.25
        var sink = CollectingSink<Double>()
        NeedleFascicle.emit(into: &sink, sheath: sheath, axis: axis, reference: reference,
                            count: count, length: 0.09, splayRadians: splay, winding: .singleSided)

        let f = NeedleFascicle.frame(axis: axis, reference: reference)
        let perNeedle = sink.count / count
        for n in 0 ..< count {
            let th = NeedleFascicle.azimuth(index: n, count: count)
            let radial = f.e1 * cos(th) + f.e2 * sin(th)
            let dir = NeedleFascicle.needleDirection(index: n, count: count, frame: f,
                                                     splayRadians: splay)
            let slice = sink.triangles[(n * perNeedle) ..< ((n + 1) * perNeedle)]
            for t in slice {
                XCTAssertGreaterThan(simd_dot(t.outward, radial), 0,
                                     "needle \(n) presents its cup toward the shoot axis")
                XCTAssertEqual(simd_dot(t.outward, dir), 0, accuracy: 1e-9,
                               "needle \(n)'s presented normal is not perpendicular to it")
                XCTAssertEqual(simd_length(t.outward), 1.0, accuracy: 1e-9)
            }
        }
    }
}
