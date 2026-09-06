import simd
import XCTest
@testable import VisualizerVegetation

/// A pad's whole reason to exist is that it is a SOLID, not a sheet — so its tests are solid tests.
/// Manifoldness, orientation and thickness are the three claims the consuming app's `MeshAudit` and
/// `Mesh3.signedVolume` will check downstream; checking them here means a failure names the
/// constructor rather than a plant that happens to use it.
final class PadConstructorTests: XCTestCase {

    private let padOutlines: [LeafSilhouette] = [.succulentPad, .petal, .fiddleLeafFig, .citrus]

    private func pad(_ s: LeafSilhouette,
                     subdivisions: Int = 3,
                     thickness: Double = 0.022,
                     crown: Double = PadConstructor.defaultCrown,
                     placement: LeafConstructor.Placement<Double>? = nil,
                     winding: LeafConstructor.Winding = .singleSided) -> CollectingSink<Double> {
        var sink = CollectingSink<Double>()
        PadConstructor.emitPad(into: &sink,
                               placement: placement ?? flatPlacement(width: 0.26, height: 0.34),
                               silhouette: s, subdivisions: subdivisions,
                               thickness: thickness, crown: crown, winding: winding)
        return sink
    }

    /// The pad is a closed manifold: every edge is used by exactly two triangles.
    ///
    /// This is the assertion the whole geometry was designed backwards from. Two places make it hard
    /// and both are handled by construction rather than by patching afterwards: the MIDLINE, where
    /// the two mirrored halves must share their column of vertices bit-for-bit (they do — `side * 0`
    /// is ±0 and adds to nothing), and the two PINCHED ENDS, where a whole row of grid points
    /// collapses onto one position and the adjoining strips must degenerate into fans. An open shell
    /// shows up here as edges used once; a doubled surface as edges used four times.
    func testPadIsAClosedManifoldSolid() {
        for s in padOutlines {
            for subdivisions in [1, 2, 3] {
                let sink = pad(s, subdivisions: subdivisions)
                XCTAssertGreaterThan(sink.count, 0, "\(s.name)@\(subdivisions) emitted nothing")
                XCTAssertTrue(sink.allValuesAreFinite, "\(s.name)@\(subdivisions)")

                let counts = sink.undirectedEdgeUseCounts()
                let open = counts.filter { $0.value != 2 }
                XCTAssertTrue(open.isEmpty,
                    "\(s.name)@\(subdivisions): \(open.count) of \(counts.count) edges are not shared "
                    + "by exactly 2 triangles (uses seen: \(Set(open.values).sorted()))")
            }
        }
    }

    /// The solid is wound consistently OUTWARD — its divergence-theorem volume is positive, and in
    /// the right ballpark for a slab of this footprint and thickness.
    ///
    /// Read together with the edge census this is the full soundness pair: closed (every edge twice)
    /// plus consistently oriented (positive volume) is exactly what `MeshAudit.isSound` and
    /// `Mesh3.signedVolume` demand of a plant mesh. An inside-out solid renders INVISIBLE under a
    /// single-sided material — it looks like a missing-asset bug and gets chased for rounds — which
    /// is why winding here is declared through the sink's `outward` and never hand-authored.
    func testPadVolumeIsPositiveAndPhysicallyPlausible() {
        for s in padOutlines {
            let thickness = 0.022
            let sink = pad(s, thickness: thickness)
            let v = sink.signedVolume
            XCTAssertGreaterThan(v, 0, "\(s.name) is inside out")

            // Bracket it against the slab it is carved from: the pad cannot hold more than its
            // bounding box, and a crowned pad of this outline holds well over a twentieth of it.
            let bb = sink.boundingBox
            let boxVolume = (bb.max.x - bb.min.x) * (bb.max.y - bb.min.y) * (bb.max.z - bb.min.z)
            XCTAssertLessThan(v, boxVolume, "\(s.name) claims more volume than its bounding box")
            XCTAssertGreaterThan(v, boxVolume * 0.05, "\(s.name) is suspiciously hollow")
        }
    }

    /// Front and back faces are wound to agree with the direction they were DECLARED to point, and
    /// that direction is away from the pad's centre.
    ///
    /// Two claims in one, because they fail independently. The first is the `LeafCardSink` contract:
    /// a sink that stores triangles verbatim gets an order that already agrees with `outward`, so
    /// the normal implied by the stored vertices must match the declared one. The second is that the
    /// declaration is itself right — a pad whose front face declares `-bentNormal` would pass the
    /// first check perfectly while being inside out.
    func testFaceNormalsAgreeWithTheirDeclarationAndPointAwayFromTheCentre() {
        let placement = flatPlacement(width: 0.26, height: 0.34)
        for s in padOutlines {
            let sink = pad(s, placement: placement)
            var front = 0, back = 0, rim = 0
            for t in sink.triangles {
                XCTAssertEqual(simd_length(t.outward), 1.0, accuracy: 1e-9, "\(s.name): outward is not unit")
                XCTAssertGreaterThan(simd_dot(t.windingNormal, t.outward), 0,
                                     "\(s.name): a stored triangle is wound against its declaration")
                XCTAssertGreaterThan(simd_dot(t.geometricNormal, t.outward), 0,
                                     "\(s.name): the reported geometric normal opposes the declaration")

                // Away from the pad centre — the pad is a convex-ish slab, so every face points out.
                XCTAssertGreaterThan(simd_dot(t.windingNormal, t.centroid - placement.position), 0,
                                     "\(s.name): a face points back toward the pad centre")

                let f = simd_dot(t.outward, placement.bentNormal)
                if f > 0.999 { front += 1 } else if f < -0.999 { back += 1 } else { rim += 1 }
            }
            // Front and back are mirror images of each other; the rim is what joins them.
            XCTAssertEqual(front, back, "\(s.name): the two faces are not mirror images")
            XCTAssertGreaterThan(rim, 0, "\(s.name): there is no rim — this is a sheet, not a solid")
            XCTAssertEqual(rim, (s.margin(subdivisions: 3).count - 1) * 4,
                           "\(s.name): the rim is not one band around the whole margin")
        }
    }

    /// The pad's thickness is really there: the extent along its normal is `thickness` at the rim
    /// and `thickness × (1 + 2·crown)` through the crowned centre.
    ///
    /// The reason to assert this rather than trust it: a pad emitted as a double-sided BLADE — the
    /// tempting shortcut, since `succulentPad` is already a silhouette — has an extent along its
    /// normal of exactly zero from the thickness and only the taco fold otherwise. It looks fine
    /// head-on in a screenshot and disappears at grazing angles.
    func testThicknessAndCrownShowUpAlongTheNormalAxis() {
        let placement = flatPlacement(width: 0.26, height: 0.34)
        let n = placement.bentNormal
        for thickness in [0.008, 0.022, 0.05] {
            // With no crown the pad is a plain slab: the extent IS the thickness.
            let slab = pad(.succulentPad, thickness: thickness, crown: 0, placement: placement)
            XCTAssertEqual(slab.extent(along: n), thickness, accuracy: 1e-9,
                           "an uncrowned pad is not `thickness` thick")

            // With a crown the centre bulges by `crown × thickness × sin(πv)` on EACH face.
            //
            // The `sin(πv)` is evaluated at the margin's OWN stations, and a resolved margin does
            // not generally sample `v = 0.5` — so the realized peak is a hair under the analytic
            // one (0.998 of it for `succulentPad`). Asserting the analytic peak would be asserting
            // a station the pad does not have; asserting the sampled one is exact, and it is also
            // the number that would move if the crown profile or the margin ever changed.
            let peak = LeafSilhouette.succulentPad.margin(subdivisions: 3)
                .map { sin($0.v * Double.pi) }.max()!
            for crown in [0.25, 0.55, 1.0] {
                let fleshy = pad(.succulentPad, thickness: thickness, crown: crown, placement: placement)
                XCTAssertEqual(fleshy.extent(along: n), thickness * (1 + 2 * crown * peak), accuracy: 1e-12,
                               "crown \(crown) at thickness \(thickness) did not bulge as documented")
            }

            // And the crown must not leak into the pad's footprint — a bulge that moved vertices in
            // the plane would change the silhouette as a side effect of changing the fleshiness.
            let a = pad(.succulentPad, thickness: thickness, crown: 0, placement: placement)
            let b = pad(.succulentPad, thickness: thickness, crown: 0.8, placement: placement)
            XCTAssertEqual(a.extent(along: placement.xAxis), b.extent(along: placement.xAxis), accuracy: 1e-12)
            XCTAssertEqual(a.extent(along: placement.yAxis), b.extent(along: placement.yAxis), accuracy: 1e-12)
        }
    }

    /// The pad's footprint is the placement's, exactly — same convention as a blade, so a species can
    /// describe a pad with the same numbers it would describe a leaf with.
    func testPadFillsItsPlacementFootprint() {
        let placement = flatPlacement(width: 0.26, height: 0.34)
        let sink = pad(.succulentPad, placement: placement)
        let base = placement.position - placement.yAxis * (placement.height * 0.46)
        let along = sink.positions.map { simd_dot($0 - base, placement.yAxis) }
        XCTAssertEqual(along.min()!, 0, accuracy: 1e-12)
        XCTAssertEqual(along.max()!, placement.height, accuracy: 1e-12)

        // Width is the widest margin control's `u` fraction of the half-width, doubled.
        let widest = LeafSilhouette.succulentPad.margin(subdivisions: 3).map(\.u).max()!
        XCTAssertEqual(sink.extent(along: placement.xAxis), widest * placement.width, accuracy: 1e-12)
    }

    /// A pad's triangle cost, written down.
    ///
    /// A pad is not a leaf and must not be budgeted like one: it is a large, near-view object placed
    /// in single digits, so 284 triangles at hero rounding is the right register. Pinning it means a
    /// change to the across-column count or the pinch handling has to move this line and say why.
    func testPadTriangleCostIsWhatTheDocumentationClaims() {
        XCTAssertEqual(pad(.succulentPad, subdivisions: 1).count, 84)
        XCTAssertEqual(pad(.succulentPad, subdivisions: 3).count, 284)

        // The general formula: two faces of (n-1) × (acrossStations-1) quads per half, plus a rim
        // band, less the fans that collapse at the two pinched ends.
        for s in padOutlines {
            for subdivisions in [1, 2, 3] {
                let n = s.margin(subdivisions: subdivisions).count
                let faces = 2 * 2 * (n - 1) * (PadConstructor.acrossStations - 1) * 2
                let rim = 2 * (n - 1) * 2
                let collapsed = 2 * 2 * 2 * (PadConstructor.acrossStations - 1)
                XCTAssertEqual(pad(s, subdivisions: subdivisions).count, faces + rim - collapsed,
                               "\(s.name)@\(subdivisions)")
            }
        }
    }

    /// `.doubleSidedShell` emits every face twice — and is therefore NOT a solid.
    ///
    /// It exists for a consumer whose material is single-sided and that wants the pad's interior
    /// visible (a cut-away, a hollow prop). Stating the consequence in a test is the point: the
    /// doubled shell must never be handed to a manifoldness or volume audit, and its edge census
    /// says exactly why.
    func testDoubleSidedShellDoublesEveryFaceAndIsNoLongerAClosedSolid() {
        let solid = pad(.succulentPad, winding: .singleSided)
        let shell = pad(.succulentPad, winding: .doubleSidedShell)
        XCTAssertEqual(shell.count, 2 * solid.count)
        XCTAssertEqual(shell.signedVolume, 0, accuracy: 1e-12, "a doubled shell encloses nothing")
        XCTAssertTrue(shell.undirectedEdgeUseCounts().allSatisfy { $0.value == 4 },
                      "a doubled shell should use every edge four times")
    }

    /// Degenerate inputs are refused rather than stitched into something unaudited.
    func testDegenerateInputsEmitNothing() {
        var sink = CollectingSink<Double>()
        XCTAssertEqual(PadConstructor.emitPad(into: &sink, placement: flatPlacement(),
                                              margin: [.init(0, 0)], thickness: 0.02,
                                              winding: .singleSided), 0)
        XCTAssertEqual(PadConstructor.emitPad(into: &sink, placement: flatPlacement(),
                                              silhouette: .succulentPad, thickness: 0,
                                              winding: .singleSided), 0)
        XCTAssertEqual(PadConstructor.emitPad(into: &sink, placement: flatPlacement(),
                                              silhouette: .succulentPad, thickness: -0.02,
                                              winding: .singleSided), 0)
        XCTAssertEqual(sink.count, 0)
    }
}
