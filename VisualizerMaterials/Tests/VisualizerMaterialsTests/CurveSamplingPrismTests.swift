import XCTest
import simd
@testable import VisualizerMaterials

/// CurveSampling spends points where a curve bends and keeps every chord within tolerance; Mesh3.prism makes a closed,
/// outward-wound solid of the right volume from any simple outline.
final class CurveSamplingPrismTests: XCTestCase {
    /// The largest distance from the curve to the chord of each sampled span (checked on a fine grid).
    private func worstStray(_ f: (Double) -> Vec2, _ ts: [Double]) -> Double {
        var worst = 0.0
        for (a, b) in zip(ts, ts.dropFirst()) {
            let pa = f(a), pb = f(b), d = simd_normalize(pb - pa)
            for k in 1..<32 {
                let q = f(a + (b - a) * Double(k) / 32) - pa
                worst = max(worst, abs(cross2(d, q)))
            }
        }
        return worst
    }

    func testArcIsEvenAndWithinTolerance() {
        let r = 2.0, tol = 0.002
        let arc = { (t: Double) in Vec2(r * cos(t), r * sin(t)) }
        let ts = CurveSampling.parameters(arc, from: 0, to: .pi / 2, tolerance: tol)
        // Even spacing on constant curvature, and the count the sagitta formula predicts: L = √(8·tol·r).
        let steps = zip(ts, ts.dropFirst()).map { $1 - $0 }
        XCTAssertLessThan(steps.max()! - steps.min()!, 0.02 * steps.max()!)
        XCTAssertEqual(steps.count, Int((r * .pi / 2 / (8 * tol * r).squareRoot()).rounded(.up)))
        XCTAssertLessThanOrEqual(worstStray(arc, ts), tol * 1.05)
    }

    func testWaveCrowdsTheBendsAndBeatsEvenSpacing() {
        let tol = 0.004
        let wave = { (x: Double) in Vec2(x, -0.1 * cos(4 * .pi * x)) }
        let ts = CurveSampling.parameters(wave, from: -0.5, to: 0.5, tolerance: tol)
        XCTAssertLessThanOrEqual(worstStray(wave, ts), tol * 1.15)
        // Symmetric about x = 0, as the wave is.
        for (a, b) in zip(ts, ts.reversed()) { XCTAssertEqual(a, -b, accuracy: 1e-9) }
        // Tighter at a crest (x = −0.25) than on a flank (x = −0.375).
        func step(near x: Double) -> Double { zip(ts, ts.dropFirst()).first { $0.0 <= x && x <= $0.1 }.map { $0.1 - $0.0 }! }
        XCTAssertLessThan(step(near: -0.25), step(near: -0.375))
        // Fewer points than the even spacing that meets the same tolerance.
        var even = 1
        while worstStray(wave, (0...even).map { -0.5 + Double($0) / Double(even) }) > tol * 1.15 { even += 1 }
        XCTAssertLessThan(ts.count - 1, even)
    }

    func testStraightLineIsJustItsEnds() {
        XCTAssertEqual(CurveSampling.parameters({ Vec2($0, 2 * $0 + 1) }, from: 0, to: 3, tolerance: 1e-4), [0, 3])
    }

    func testPrismIsClosedOutwardAndHasTheRightVolume() {
        // A concave "L" in clockwise order, with a repeated and a collinear point to drop.
        let l: [Vec2] = [Vec2(0, 0), Vec2(0, 2), Vec2(1, 2), Vec2(1, 1), Vec2(2, 1), Vec2(3, 1), Vec2(3, 0), Vec2(3, 0)]
        let m = Mesh3.prism(l, z0: -0.1, z1: 0.2).welded(distance: 1e-9)
        XCTAssertEqual(MeshKit.openEdges(m), 0)
        XCTAssertEqual(MeshKit.volume(m), 4 * 0.3, accuracy: 1e-9)           // positive: wound outward
        XCTAssertEqual(m.indices.count / 3, 2 * (6 - 2) + 2 * 6)            // six points kept
    }

    /// A six-toothed cog (a crest symbol, a few cm across) whose tooth corners line up across the hub: ear clipping
    /// once cut a sliver here that Mesh3.addTriangle drops as degenerate, leaving a hole in the cap.
    func testPrismOfACogHasNoSliverHoles() {
        let cog: [Vec2] = [Vec2(0.1875, 0.095), Vec2(0.20157291281149714, 0.103125), Vec2(0.18282291281149712, 0.13560095264191646),
            Vec2(0.16875, 0.12747595264191644), Vec2(0.16875, 0.14372595264191645), Vec2(0.13125, 0.14372595264191645),
            Vec2(0.13125, 0.12747595264191647), Vec2(0.11717708718850287, 0.13560095264191646), Vec2(0.09842708718850286, 0.10312500000000001),
            Vec2(0.11249999999999999, 0.095), Vec2(0.09842708718850288, 0.08687500000000001), Vec2(0.11717708718850285, 0.05439904735808356),
            Vec2(0.13124999999999998, 0.06252404735808356), Vec2(0.13124999999999998, 0.04627404735808356), Vec2(0.16874999999999998, 0.046274047358083555),
            Vec2(0.16875, 0.06252404735808356), Vec2(0.18282291281149712, 0.05439904735808355), Vec2(0.2015729128114971, 0.086875)]
        for tris in [Triangulation.earClip(cog)] {
            XCTAssertEqual(tris.count, cog.count - 2)
            for (a, b, c) in tris { XCTAssertGreaterThan(orient(cog[a], cog[b], cog[c]), Triangulation.keptOrient) }
        }
        let m = Mesh3.prism(cog, z0: -0.009, z1: 0.009).welded(distance: 1e-9)
        XCTAssertEqual(MeshKit.openEdges(m), 0)
        XCTAssertEqual(MeshKit.volume(m), area(cog) * 0.018, accuracy: 1e-12)
    }

    func testCurvedPrismMatchesTheSampledPolygon() {
        let half = CurveSampling.points({ Vec2(cos($0), sin($0)) }, from: 0, to: .pi, tolerance: 0.001)
        let m = Mesh3.prism(half, z0: 0, z1: 1).welded(distance: 1e-9)
        XCTAssertEqual(MeshKit.openEdges(m), 0)
        XCTAssertEqual(MeshKit.volume(m), signedArea(half), accuracy: 1e-9)
        XCTAssertEqual(MeshKit.volume(m), .pi / 2, accuracy: 0.01)
    }
}
