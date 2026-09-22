import Foundation
import simd
import VisualizerMaterials

/// A fleshy flat PAD — an opuntia (prickly-pear) segment, a jade paddle — as a closed solid.
///
/// **This one really is not a blade, and that is the whole justification for a second constructor.**
/// Everything else in this module (leaf, petal, pinnule, needle) is a SHEET: a surface with two
/// faces and no interior, which is why they all reduce to `LeafConstructor.emitBlade` and why a
/// third strip-stitch would be duplication. A cactus pad is a SOLID: 15–25 mm of water-storing
/// tissue between two faces, with a rim you can see the thickness of from the side and a silhouette
/// that reads as a rounded slab rather than a card. Emitting one as a double-sided blade gives a
/// pad with zero thickness — at grazing angles it disappears entirely, and where it meets the
/// ground or the parent pad the two coincident surfaces z-fight.
///
/// So this constructor emits a genuine watertight solid: a front face, a back face, and a rim band
/// joining them. That matters concretely — `MeshAudit` in the consuming app checks manifoldness and
/// `Mesh3.signedVolume` checks the solid is not inside out, and an open shell fails both. Every
/// interior edge here is shared by exactly two triangles by construction (see the pinch note below,
/// which is the one place that is easy to get wrong).
///
/// **What it shares with the blade, deliberately.** The same half-margin convention
/// (`LeafSilhouette.Control`: `v` 0 at the base → 1 at the tip, `u` the half-width fraction,
/// mirrored to both sides) and the same `LeafConstructor.Placement`. A pad outline is therefore an
/// ordinary silhouette from the same catalog — `.succulentPad` is already the right shape for a
/// jade paddle — and a species describes its pad with the same table it would describe a leaf with.
/// It also means the smoothing, the `v`-monotonicity guarantee and the `u >= 0` clamp in
/// `LeafSilhouette.smooth` apply here unchanged.
///
/// **Triangle cost.** With `n` resolved margin controls, the two faces cost
/// `2 faces × 2 sides × (n-1) × (acrossStations-1) × 2` triangles and the rim costs
/// `2 sides × (n-1) × 2`, less the `2 faces × 2 sides × 2 ends × (acrossStations-1)` triangles that
/// collapse to nothing at the two pinched ends. For `.succulentPad` at `subdivisions: 3` (16
/// controls) that is **284 triangles per pad**; at `subdivisions: 1` (6 controls) it is **84**. A
/// pad is a large, near-view object placed in single digits, so this is the right register — it is
/// not a leaf and must not be budgeted like one.
public enum PadConstructor {

    /// How much thicker the pad's centre is than its rim, as a MULTIPLE of `thickness`.
    ///
    /// Expressed as a ratio rather than a length so it survives being scaled: a pad authored at
    /// 0.25 m and one at 0.4 m want the same fleshiness, not the same millimetres of bulge. The
    /// default, 0.55, makes the centre a bit over twice the rim thickness, which is roughly what an
    /// opuntia segment measures.
    public static let defaultCrown: Double = 0.55

    /// How many columns each half-face is divided into across, from the midline (`u = 0`) to the
    /// margin (`u = 1`).
    ///
    /// Three, so the crown has somewhere to be round. With two the bulge profile is a single
    /// straight ramp from centre to margin, which is a TENT with a hard crease down the midline —
    /// and a pad has no midrib, so that crease reads as a fold in a sheet, defeating the whole point
    /// of crowning it. Three columns cost `(n-1) × 4` extra triangles per pad, which at pad counts
    /// is nothing.
    @usableFromInline static let acrossStations = 3

    /// Emit ONE pad as a closed solid.
    ///
    /// - Parameters:
    ///   - sink: where the triangles go.
    ///   - placement: pad centroid, axes, and the pad's overall width × height. `bentNormal` is the
    ///     FRONT face direction; `xAxis` is across the pad, `yAxis` runs base → tip.
    ///   - margin: the half-margin, already resolved (see `LeafSilhouette.margin(tier:subdivisions:)`).
    ///     `v` must be non-decreasing.
    ///   - thickness: rim thickness, in metres — what a caliper on the pad's EDGE would read. The
    ///     centre is thicker by `crown`.
    ///   - crown: centre bulge as a multiple of `thickness`. See `defaultCrown`.
    ///   - winding: `.singleSided` emits the solid — each face once, outward. That is what a closed
    ///     pad wants and the only mode `MeshAudit` will accept. `.doubleSidedShell` additionally
    ///     emits every face reversed, for a consumer whose material is single-sided and that wants
    ///     the interior visible; it is NOT a manifold solid and must never be handed to a volume or
    ///     manifoldness audit.
    /// - Returns: the number of triangles emitted, for budget accounting.
    @discardableResult
    public static func emitPad<Sink: LeafCardSink>(
        into sink: inout Sink,
        placement: LeafConstructor.Placement<Sink.Scalar>,
        margin: [LeafSilhouette.Control],
        thickness: Sink.Scalar,
        crown: Sink.Scalar = Sink.Scalar(PadConstructor.defaultCrown),
        winding: LeafConstructor.Winding
    ) -> Int {
        typealias S = Sink.Scalar
        guard margin.count >= 2, thickness > 0 else { return 0 }

        let bent = safeNormalize(placement.bentNormal, fallback: SIMD3<S>(0, 0, 1))
        let yA = placement.yAxis
        let xA = placement.xAxis
        let hw = placement.width / 2
        let h = placement.height
        let half = thickness / 2
        let centre = placement.position
        let base = centre - yA * (h * S(0.46))
        let cols = Self.acrossStations
        let lastCol = cols - 1

        // The crown profile.
        //
        // `1 - u²` rather than `1 - u`: both vanish at the margin (so the rim stays a clean band of
        // exactly `thickness` and cannot tear away from the faces), but a straight ramp puts a
        // crease down the midline, and a pad is a dome in section, not a tent. `sin(πv)` windows it
        // to zero at BOTH ends, which is what keeps the two pinched tips — where the front, back and
        // margin points all coincide — from being pulled apart into a hole.
        func bulge(u: S, v: S) -> S {
            crown * thickness * (1 - u * u) * S(sin(Double(v) * Double.pi))
        }

        /// A point on one face. `face` is +1 for the front, -1 for the back; `col` runs 0 (midline)
        /// → `lastCol` (margin); `side` is which mirrored half.
        func point(_ i: Int, side: S, col: Int, face: S) -> LeafStripVertex<S> {
            let c = margin[i]
            let v = S(c.v)
            let u = S(Double(col) / Double(lastCol))
            let halfWidthHere = S(c.u) * u
            let p = base
                + yA * (h * v)
                + xA * (side * halfWidthHere * hw)
                + bent * (face * (half + bulge(u: u, v: v)))
            return LeafStripVertex(position: p, v: v, u: u, side: side)
        }

        var emitted = 0

        // Wind to agree with the DECLARED outward direction rather than authoring a vertex order per
        // face and hoping. `LeafCardSink` promises `outward` is the direction the finished face must
        // point, and a sink that stores triangles verbatim (the yard tree's soup) depends on getting
        // an order that already agrees with it — so the swap is derived from `outward`, here, once.
        //
        // The zero-area guard is load-bearing, not defensive. A pad outline pinches to `u = 0` at
        // its base and its tip, which collapses that whole row of grid points onto one position: the
        // strips adjoining it degenerate into fans, and each fan's outer triangle has zero area.
        // Dropping those is exactly what turns the grid into a manifold cap — keeping them would
        // leave zero-length edges that no edge-pairing audit can balance.
        func add(_ a: LeafStripVertex<S>, _ b: LeafStripVertex<S>, _ c: LeafStripVertex<S>,
                 outward: SIMD3<S>) {
            let n = cross3(b.position - a.position, c.position - a.position)
            let m2 = n.x * n.x + n.y * n.y + n.z * n.z
            guard m2 > S(1e-24), m2.isFinite else { return }
            let gn = safeNormalize(n, fallback: outward)
            let facing = gn.x * outward.x + gn.y * outward.y + gn.z * outward.z
            if facing < 0 {
                sink.addLeafTriangle(a, c, b, outward: outward, geometricNormal: -gn)
            } else {
                sink.addLeafTriangle(a, b, c, outward: outward, geometricNormal: gn)
            }
            emitted += 1
            if winding == .doubleSidedShell {
                let back = -outward
                if facing < 0 {
                    sink.addLeafTriangle(a, b, c, outward: back, geometricNormal: gn)
                } else {
                    sink.addLeafTriangle(a, c, b, outward: back, geometricNormal: -gn)
                }
                emitted += 1
            }
        }

        // ---- The two faces.
        //
        // Both halves share the midline column bit-for-bit (`side * 0 * hw` is ±0, which adds to
        // nothing), so the seam down the middle is welded rather than coincident — an edge there is
        // used once by each half, which is what makes it manifold instead of a crack.
        for face in [S(1), S(-1)] {
            let outward = bent * face
            for side in [S(1), S(-1)] {
                for i in 0 ..< (margin.count - 1) {
                    for col in 0 ..< lastCol {
                        let a = point(i, side: side, col: col, face: face)
                        let b = point(i, side: side, col: col + 1, face: face)
                        let c = point(i + 1, side: side, col: col + 1, face: face)
                        let d = point(i + 1, side: side, col: col, face: face)
                        // One fixed diagonal (a–c) so the quad's two triangles share it and nothing
                        // else does.
                        add(a, b, c, outward: outward)
                        add(a, c, d, outward: outward)
                    }
                }
            }
        }

        // ---- The rim.
        //
        // A band around the margin joining front to back. At a pinched end the front and back margin
        // points of BOTH halves collapse onto the same two points, so the band closes there over a
        // single vertical edge — the pad ends in a blunt `thickness`-tall edge rather than a knife
        // point, which is both correct for a fleshy pad and the reason the solid stays closed.
        for side in [S(1), S(-1)] {
            for i in 0 ..< (margin.count - 1) {
                let f0 = point(i, side: side, col: lastCol, face: 1)
                let f1 = point(i + 1, side: side, col: lastCol, face: 1)
                let b0 = point(i, side: side, col: lastCol, face: -1)
                let b1 = point(i + 1, side: side, col: lastCol, face: -1)

                // Radially outward: perpendicular to both the margin's tangent and the pad normal,
                // then faced away from the pad centre. Derived per segment rather than taken as
                // `xAxis * side`, which would be wrong everywhere the margin turns — most of it.
                let mid0 = (f0.position + b0.position) * S(0.5)
                let mid1 = (f1.position + b1.position) * S(0.5)
                let tangent = mid1 - mid0
                let radial = faceToward(safeNormalize(cross3(tangent, bent), fallback: xA * side),
                                        (mid0 + mid1) * S(0.5) - centre)

                add(f0, f1, b1, outward: radial)
                add(f0, b1, b0, outward: radial)
            }
        }

        return emitted
    }

    /// Emit ONE pad by silhouette — the ordinary entry point, mirroring
    /// `LeafConstructor.emitBlade(into:placement:silhouette:…)` so a species reads the same whether
    /// its part is a sheet or a solid.
    @discardableResult
    public static func emitPad<Sink: LeafCardSink>(
        into sink: inout Sink,
        placement: LeafConstructor.Placement<Sink.Scalar>,
        silhouette: LeafSilhouette,
        tier: LeafSilhouette.Tier = .hero,
        subdivisions: Int = 1,
        thickness: Sink.Scalar,
        crown: Sink.Scalar = Sink.Scalar(PadConstructor.defaultCrown),
        winding: LeafConstructor.Winding
    ) -> Int {
        emitPad(into: &sink,
                placement: placement,
                margin: silhouette.margin(tier: tier, subdivisions: subdivisions),
                thickness: thickness,
                crown: crown,
                winding: winding)
    }
}
