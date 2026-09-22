import Foundation
import simd
import VisualizerMaterials

/// A conifer's needle BUNDLE — pine's 2–5 needles bound at a common sheath, or the single needle a
/// cedar/spruce shoot carries at each station.
///
/// **This is an ARRANGEMENT, not a second constructor.** A needle at any distance a house is viewed
/// from is a very narrow blade: same midrib, same margin mirrored to both sides, same cup, same
/// droop. So a fascicle is `count` calls to `LeafConstructor.emitBlade` with a computed frame each,
/// and it inherits the fold, the curl, the NaN guards and the winding contract for free. Writing a
/// second strip-stitch here — even a "simpler" one, because a needle is nearly a quad — is exactly
/// the failure this module exists to prevent: the two historical leaf systems each started as one
/// small honest loop, and the margin fix that landed in one and not the other is what DH-0626 /
/// DH-0628 / DH-0629 / DH-0647 each rediscovered in turn. There is one loop. This file only decides
/// where the blades point.
///
/// **No randomness, deliberately.** Each needle's azimuth is a pure function of its index and the
/// bundle size, and the whole emission is a pure function of its arguments. A conifer shoot carries
/// hundreds of fascicles and the caller is the one that must own the jitter — partly because the
/// per-tree seed belongs to the tree (`DocumentID.seed64`, never `hashValue`, which Swift re-seeds
/// every process), and partly because a constructor that draws from its own RNG makes emission
/// order load-bearing: insert one fascicle upstream and every downstream needle moves, which turns
/// a baked-triangle-count gate into a tripwire. Callers vary `reference`, `splayRadians`, `length`
/// and `curl`; this type stays deterministic.
///
/// **Triangle cost.** One needle is `(controls - 1) × 2` strip quads — the half-margin mirrored —
/// at 2 triangles per quad single-sided, 4 double-sided. With the shipped `LeafSilhouette.needle`
/// (4 controls, `subdivisions: 1`) that is **12 triangles per needle single-sided, 24 double**, so
/// a 3-needle pine fascicle costs **36 / 72**. That number is multiplied by the fascicle count of a
/// whole conifer — a mid-density pine shoot carries ~40 fascicles and a tree ~600 shoots — so the
/// levers that matter are, in order: `count`, `subdivisions` (each step multiplies the segment
/// count), and the silhouette's control count. `triangleCount(...)` returns the figure so a
/// species' budget can be asserted rather than guessed at.
public enum NeedleFascicle {

    /// Needle width as a fraction of its length. A pine needle is ~90 mm long and ~1.2 mm across,
    /// but the silhouette's widest control is `u ≈ 0.92` of the half-width, so the ratio that lands
    /// that width is a little above the naive one. Deliberately not per-species: a species that
    /// wants a broader spruce needle passes its own.
    public static let defaultAspect: Double = 0.030

    /// How far off the shoot axis a needle is pitched, in radians (~10°). A fascicle that splays
    /// too wide reads as a starfish rather than a bundle; one that splays at zero reads as a single
    /// thick needle, because every blade in the bundle occupies the same line.
    public static let defaultSplay: Double = 0.18

    /// The orthonormal frame a fascicle splays inside: the shoot `axis`, plus two perpendiculars
    /// that fix where azimuth zero points.
    ///
    /// Exposed because a caller that wants to hang something else off the same shoot — a cone, a
    /// bud, the next season's growth — must use the SAME frame or its parts will not agree with the
    /// needles, and rebuilding the frame from a re-derived perpendicular is how that drifts.
    public struct Frame<Scalar: LeafScalar>: Sendable {
        /// The shoot direction, normalized. Needles pitch off this.
        public var axis: SIMD3<Scalar>
        /// Azimuth zero, perpendicular to `axis`.
        public var e1: SIMD3<Scalar>
        /// Azimuth 90°, completing a right-handed frame with `axis` and `e1`.
        public var e2: SIMD3<Scalar>

        public init(axis: SIMD3<Scalar>, e1: SIMD3<Scalar>, e2: SIMD3<Scalar>) {
            self.axis = axis
            self.e1 = e1
            self.e2 = e2
        }
    }

    /// Build the splay frame from a shoot direction and a reference vector.
    ///
    /// `reference` only has to be non-parallel to `axis` — it is Gram-Schmidt'd against it, so a
    /// caller can pass world up, the parent branch direction, or a per-shoot jitter vector without
    /// pre-orthogonalizing. A parallel (or zero) reference falls back to an arbitrary perpendicular
    /// rather than producing a NaN frame that poisons every needle downstream.
    public static func frame<S: LeafScalar>(axis: SIMD3<S>, reference: SIMD3<S>) -> Frame<S> {
        let a = safeNormalize(axis, fallback: SIMD3<S>(0, 1, 0))
        let d = reference.x * a.x + reference.y * a.y + reference.z * a.z
        let e1 = safeNormalize(reference - a * d, fallback: anyPerpendicular(a))
        return Frame(axis: a, e1: e1, e2: cross3(a, e1))
    }

    /// The azimuth of needle `index` in a bundle of `count`, in radians.
    ///
    /// Evenly spaced, with a HALF-STEP offset. The offset is not cosmetic: without it needle 0 of
    /// every fascicle sits exactly on `reference`, so a shoot that passes one reference for all its
    /// fascicles grows a visible seam of co-planar needles running its whole length. The half step
    /// also makes a 2-needle pine fascicle straddle the reference plane symmetrically, which is
    /// what a real two-needle bundle does.
    public static func azimuth(index: Int, count: Int) -> Double {
        guard count > 0 else { return 0 }
        return 2 * Double.pi * (Double(index) + 0.5) / Double(count)
    }

    /// The unit direction needle `index` grows in.
    ///
    /// Exposed so a caller can place something at a needle's tip (`sheath + direction * length`)
    /// without re-deriving the frame maths and drifting from what was emitted.
    public static func needleDirection<S: LeafScalar>(index: Int,
                                                      count: Int,
                                                      frame f: Frame<S>,
                                                      splayRadians: S) -> SIMD3<S> {
        let th = azimuth(index: index, count: count)
        let radial = f.e1 * S(cos(th)) + f.e2 * S(sin(th))
        let s = Double(splayRadians)
        return f.axis * S(cos(s)) + radial * S(sin(s))
    }

    /// The exact triangle count `emit` will produce, without emitting it.
    ///
    /// A needle bundle is the one place in a conifer where a small per-piece number is multiplied by
    /// a large arrangement count, so a species' budget wants to be an assertion, not an estimate.
    public static func triangleCount(silhouette: LeafSilhouette = .needle,
                                     tier: LeafSilhouette.Tier = .hero,
                                     subdivisions: Int = 1,
                                     count: Int,
                                     winding: LeafConstructor.Winding) -> Int {
        guard count > 0 else { return 0 }
        let controls = silhouette.margin(tier: tier, subdivisions: subdivisions).count
        guard controls >= 2 else { return 0 }
        let perQuad = (winding == .doubleSidedShell) ? 4 : 2
        return count * (controls - 1) * 2 * perQuad
    }

    /// Emit one fascicle: `count` needles rising from `sheath`, splayed evenly in azimuth around
    /// `axis` and pitched off it by `splayRadians`.
    ///
    /// Every needle's BASE lands exactly on `sheath` — `LeafConstructor.Placement.position` is the
    /// blade centroid and the base sits `height * 0.46` back along `-yAxis`, so the centroid is
    /// pushed forward by that same fraction here. Getting this wrong is invisible on a leaf hanging
    /// off a petiole and glaring on a fascicle, where all `count` bases must coincide inside the
    /// sheath or the bundle reads as an exploded starburst.
    ///
    /// Each needle presents the face pointing radially AWAY from the shoot axis: the cup opens
    /// outward, so a bundle catches its shading gradient from the outside where it is seen, rather
    /// than cupping toward its own centre where nothing can see it. That normal is
    /// `radial·cos(splay) − axis·sin(splay)`, which is the exact unit perpendicular to the needle
    /// direction in the radial plane — worth writing in closed form rather than re-orthogonalizing,
    /// because a Gram-Schmidt here would divide by `cos(splay)` and go singular at 90°.
    ///
    /// - Returns: the number of triangles emitted, for budget accounting.
    @discardableResult
    public static func emit<Sink: LeafCardSink>(
        into sink: inout Sink,
        sheath: SIMD3<Sink.Scalar>,
        axis: SIMD3<Sink.Scalar>,
        reference: SIMD3<Sink.Scalar>,
        count: Int,
        length: Sink.Scalar,
        aspect: Sink.Scalar = Sink.Scalar(NeedleFascicle.defaultAspect),
        splayRadians: Sink.Scalar = Sink.Scalar(NeedleFascicle.defaultSplay),
        curl: Sink.Scalar = 0,
        silhouette: LeafSilhouette = .needle,
        tier: LeafSilhouette.Tier = .hero,
        subdivisions: Int = 1,
        fold: Sink.Scalar = Sink.Scalar(LeafConstructor.defaultFold),
        winding: LeafConstructor.Winding
    ) -> Int {
        typealias S = Sink.Scalar
        guard count > 0, length > 0 else { return 0 }
        let margin = silhouette.margin(tier: tier, subdivisions: subdivisions)
        guard margin.count >= 2 else { return 0 }

        let f = frame(axis: axis, reference: reference)
        let splay = Double(splayRadians)
        let cs = S(cos(splay)), sn = S(sin(splay))

        for i in 0 ..< count {
            let th = azimuth(index: i, count: count)
            let radial = f.e1 * S(cos(th)) + f.e2 * S(sin(th))
            let dir = f.axis * cs + radial * sn
            // Unit, and exactly perpendicular to `dir`, by construction rather than by projection.
            let bent = radial * cs - f.axis * sn
            let across = safeNormalize(cross3(dir, bent), fallback: f.e1)

            let placement = LeafConstructor.Placement(
                position: sheath + dir * (length * S(0.46)),
                xAxis: across,
                yAxis: dir,
                bentNormal: bent,
                width: length * aspect,
                height: length)

            LeafConstructor.emitBlade(into: &sink,
                                      placement: placement,
                                      margin: margin,
                                      fold: fold,
                                      curl: curl,
                                      winding: winding)
        }

        let perQuad = (winding == .doubleSidedShell) ? 4 : 2
        return count * (margin.count - 1) * 2 * perQuad
    }
}

/// Some unit vector perpendicular to `a`.
///
/// Internal and NaN-safe for the same reason `safeNormalize` is: the caller reaching for this is
/// always recovering from a degenerate input (a zero or axis-parallel reference vector), and
/// handing back a NaN there would replace one bad frame with a whole bad plant. The `0.9` pivot
/// picks whichever world axis is furthest from `a`, so the cross product never approaches zero.
@inlinable
func anyPerpendicular<S: LeafScalar>(_ a: SIMD3<S>) -> SIMD3<S> {
    let alt: SIMD3<S> = a.y.magnitude < S(0.9) ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
    return safeNormalize(cross3(a, alt), fallback: SIMD3(1, 0, 0))
}
