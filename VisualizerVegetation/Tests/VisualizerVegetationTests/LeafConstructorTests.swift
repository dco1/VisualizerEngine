import simd
import XCTest
@testable import VisualizerVegetation

/// Regression protection for the one strip-stitch every plant in every consuming app now goes
/// through. Nothing here is subtle; all of it has shipped broken at least once.
final class LeafConstructorTests: XCTestCase {

    /// `.doubleSidedShell` costs exactly twice `.singleSided`, for the same margin.
    ///
    /// The two modes are a real difference between the consumers, not an accident: `Mesh3` plants
    /// are registered single-sided and need both faces as geometry, while the yard tree makes its
    /// material two-sided and takes the halving — which is the difference between a ~50k and a ~100k
    /// triangle tree. Anything that changes that ratio changes a canopy's triangle budget by 2×
    /// without touching a number anyone would think to look at.
    func testDoubleSidedShellCostsExactlyTwiceSingleSided() {
        for s in shippedSilhouettes {
            for subdivisions in [1, 3] {
                let margin = s.margin(subdivisions: subdivisions)
                var single = CollectingSink<Double>()
                var double = CollectingSink<Double>()
                LeafConstructor.emitBlade(into: &single, placement: flatPlacement(),
                                          margin: margin, fold: 0.3, curl: 0.2, winding: .singleSided)
                LeafConstructor.emitBlade(into: &double, placement: flatPlacement(),
                                          margin: margin, fold: 0.3, curl: 0.2, winding: .doubleSidedShell)
                XCTAssertEqual(single.count, (margin.count - 1) * 4, "\(s.name)@\(subdivisions)")
                XCTAssertEqual(double.count, 2 * single.count, "\(s.name)@\(subdivisions)")
            }
        }
    }

    /// Every position, outward and geometric normal a blade emits is finite, in both precisions.
    ///
    /// One NaN vertex poisons a whole baked mesh — the bounding box goes infinite, the BVH build
    /// that consumes it produces garbage, and the plant vanishes rather than looking wrong, which is
    /// the hardest failure to trace back to a leaf.
    func testEveryEmittedValueIsFinite() {
        for s in shippedSilhouettes {
            for subdivisions in [1, 2, 4] {
                var d = CollectingSink<Double>()
                LeafConstructor.emitBlade(into: &d, placement: flatPlacement(), silhouette: s,
                                          subdivisions: subdivisions, fold: 0.42, curl: 0.5,
                                          winding: .doubleSidedShell)
                XCTAssertTrue(d.allValuesAreFinite, "\(s.name)@\(subdivisions) (Double)")
                XCTAssertGreaterThan(d.count, 0, "\(s.name)@\(subdivisions) emitted nothing")

                var f = CollectingSink<Float>()
                let p = flatPlacement()
                let fp = LeafConstructor.Placement<Float>(
                    position: SIMD3(p.position), xAxis: SIMD3(p.xAxis), yAxis: SIMD3(p.yAxis),
                    bentNormal: SIMD3(p.bentNormal), width: Float(p.width), height: Float(p.height))
                LeafConstructor.emitBlade(into: &f, placement: fp, silhouette: s,
                                          subdivisions: subdivisions, fold: 0.42, curl: 0.5,
                                          winding: .singleSided)
                XCTAssertTrue(f.allValuesAreFinite, "\(s.name)@\(subdivisions) (Float)")
            }
        }
    }

    /// A margin that is zero-width throughout produces no NaNs.
    ///
    /// This is the degenerate input the guards in `LeafConstructor` exist for: every strip quad
    /// collapses onto the midrib, so every face-normal cross product is zero, and an unguarded
    /// `simd_normalize` there returns NaN for the whole blade. It is not a hypothetical — a species
    /// whose margin pinches (a deep sinus, a decimation that drops the wide controls, an arranger
    /// that scales a blade to zero width at the canopy edge) hits it, and the tree grew its guards
    /// the expensive way before the constructor was shared.
    func testDegenerateZeroWidthMarginProducesNoNaNs() {
        let flat: [LeafSilhouette.Control] = [
            .init(0.0, 0.0), .init(0.25, 0.0), .init(0.5, 0.0), .init(0.75, 0.0), .init(1.0, 0.0),
        ]
        for w: LeafConstructor.Winding in [.singleSided, .doubleSidedShell] {
            var sink = CollectingSink<Double>()
            LeafConstructor.emitBlade(into: &sink, placement: flatPlacement(),
                                      margin: flat, fold: 0.3, curl: 0.3, winding: w)
            XCTAssertTrue(sink.allValuesAreFinite, "degenerate margin produced a non-finite value")
            for t in sink.triangles {
                let n = t.geometricNormal
                XCTAssertEqual(simd_length(n), 1.0, accuracy: 1e-9,
                               "degenerate normal was not replaced by the unit fallback")
            }
        }

        // A one-control margin is not a blade at all and must be refused, not stitched.
        var empty = CollectingSink<Double>()
        LeafConstructor.emitBlade(into: &empty, placement: flatPlacement(),
                                  margin: [.init(0, 0)], fold: 0.3, winding: .singleSided)
        XCTAssertEqual(empty.count, 0)
    }

    /// Emission is a pure function of its arguments — the same call twice gives the same bytes.
    ///
    /// Everything downstream that pins a triangle count or diffs a bake assumes this, and the way it
    /// breaks is a constructor quietly reaching for randomness (or, worse, for `hashValue`, which
    /// Swift re-seeds every process — the "books rearranged on every launch" defect).
    func testEmissionIsDeterministic() {
        for s in shippedSilhouettes {
            var a = CollectingSink<Double>(), b = CollectingSink<Double>()
            LeafConstructor.emitBlade(into: &a, placement: flatPlacement(), silhouette: s,
                                      subdivisions: 3, curl: 0.31, winding: .doubleSidedShell)
            LeafConstructor.emitBlade(into: &b, placement: flatPlacement(), silhouette: s,
                                      subdivisions: 3, curl: 0.31, winding: .doubleSidedShell)
            XCTAssertEqual(a.count, b.count, "\(s.name)")
            for (p, q) in zip(a.positions, b.positions) {
                XCTAssertEqual(p, q, "\(s.name) is not deterministic")
            }
        }
    }

    /// The blade's base lands where the placement says it does.
    ///
    /// `Placement.position` is the CENTROID and the base sits `height * 0.46` back along `-yAxis`,
    /// which is the convention every arranger — and `NeedleFascicle`, below — has to invert to put a
    /// petiole or a sheath in the right place. Pinning it here means an arranger that gets it wrong
    /// fails against a stated contract instead of against a screenshot.
    func testBladeBaseSitsAtTheDocumentedOffsetFromTheCentroid() {
        let p = flatPlacement(width: 0.2, height: 0.5)
        var sink = CollectingSink<Double>()
        LeafConstructor.emitBlade(into: &sink, placement: p, silhouette: .citrus,
                                  fold: 0, curl: 0, winding: .singleSided)
        let expectedBase = p.position - p.yAxis * (p.height * 0.46)
        let alongMidrib = sink.positions.map { simd_dot($0 - expectedBase, p.yAxis) }
        XCTAssertEqual(alongMidrib.min()!, 0, accuracy: 1e-12)
        XCTAssertEqual(alongMidrib.max()!, p.height, accuracy: 1e-12)
    }

    /// A blade split into a base piece and a tip piece — two `emitBlade` calls back to back, in one
    /// function, into one sink — IS the unsplit blade, triangle for triangle, bit for bit.
    ///
    /// DH-0777 recorded the opposite: a two-tone leaf built this way came out wider and flatter than
    /// the whole leaf, as if `fold` and `curl` had been dropped, and it was filed as a possible
    /// compiler miscompile of the generic constructor. Splitting a margin is a legitimate thing to
    /// want (one colour group per band on a `Mesh3`, which has no colour channel), so the question is
    /// pinned here as a number: every station is a pure function of `(placement, v, u, fold, curl)`,
    /// a cut at a shared control evaluates the seam identically from both sides, and nothing a
    /// previous call did can reach the next. Asked of both scalars and both windings, on a tilted,
    /// non-axis-aligned frame where a dropped fold or curl moves every coordinate.
    func testBackToBackSplitBladesAreTheUnsplitBlade() {
        let y = simd_normalize(SIMD3<Double>(0.31, 0.88, -0.36))
        let x = simd_normalize(simd_cross(y, SIMD3<Double>(0.2, 0.1, 1)))
        let bent = simd_cross(x, y)
        let p = LeafConstructor.Placement<Double>(position: SIMD3(0.12, 0.64, -0.05), xAxis: x,
                                                  yAxis: y, bentNormal: bent,
                                                  width: 0.216, height: 0.52)
        let fp = LeafConstructor.Placement<Float>(
            position: SIMD3(p.position), xAxis: SIMD3(p.xAxis), yAxis: SIMD3(p.yAxis),
            bentNormal: SIMD3(p.bentNormal), width: Float(p.width), height: Float(p.height))

        for s in shippedSilhouettes {
            let margin = s.margin(subdivisions: 3)
            guard margin.count >= 4 else { continue }
            // One cut, and two — the base/tip two-tone shape, and a three-band petal.
            let cutSets = [[margin.count / 2], [margin.count / 3, 2 * margin.count / 3]]
            for cuts in cutSets {
                for w: LeafConstructor.Winding in [.singleSided, .doubleSidedShell] {
                    let label = "\(s.name) cuts=\(cuts) \(w)"
                    assertSplitIsWhole(placement: p, margin: margin, cuts: cuts, winding: w,
                                       label: label + " Double")
                    assertSplitIsWhole(placement: fp, margin: margin, cuts: cuts, winding: w,
                                       label: label + " Float")
                }
            }
        }
    }

    private func assertSplitIsWhole<S: LeafScalar>(placement: LeafConstructor.Placement<S>,
                                                   margin: [LeafSilhouette.Control], cuts: [Int],
                                                   winding: LeafConstructor.Winding, label: String) {
        let fold = S(0.30), curl = S(0.34), wave = S(0.06)
        var whole = CollectingSink<S>()
        LeafConstructor.emitBlade(into: &whole, placement: placement, margin: margin, fold: fold,
                                  curl: curl, winding: winding, waveAmplitude: wave, wavePhase: 0.7)

        // The pieces share their cut control, so the seam is one station seen from both sides.
        var split = CollectingSink<S>()
        let bounds = [0] + cuts + [margin.count - 1]
        for (lo, hi) in zip(bounds, bounds.dropFirst()) {
            LeafConstructor.emitBlade(into: &split, placement: placement,
                                      margin: Array(margin[lo ... hi]), fold: fold, curl: curl,
                                      winding: winding, waveAmplitude: wave, wavePhase: 0.7)
        }

        XCTAssertEqual(split.count, whole.count, "\(label): triangle count")
        func key(_ t: CollectingSink<S>.Triangle) -> [SIMD3<S>] { [t.a, t.b, t.c, t.outward] }
        let wholeKeys = Set(whole.triangles.map(key))
        let strays = split.triangles.filter { !wholeKeys.contains(key($0)) }
        XCTAssertTrue(strays.isEmpty,
                      "\(label): \(strays.count) split triangles are not on the unsplit blade")
        XCTAssertEqual(split.extent(across: placement.xAxis), whole.extent(across: placement.xAxis),
                       "\(label): width")
        XCTAssertEqual(split.extent(across: placement.bentNormal),
                       whole.extent(across: placement.bentNormal), "\(label): fold/curl depth")
    }
}

private extension CollectingSink {
    func extent(across axis: SIMD3<Scalar>) -> Scalar {
        let d = positions.map { $0.x * axis.x + $0.y * axis.y + $0.z * axis.z }
        return (d.max() ?? 0) - (d.min() ?? 0)
    }
}
