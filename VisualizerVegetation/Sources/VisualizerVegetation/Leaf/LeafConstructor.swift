import simd
import VisualizerMaterials

/// THE single blade constructor. One strip-stitch loop, for every plant in every consuming app.
///
/// A "blade" here is the shared idiom behind a leaf, a petal, a fern pinnule and a succulent
/// paddle: a straight midrib SPINE with a half-margin OUTLINE mirrored to both sides, cupped into
/// a shallow taco FOLD so it catches a cross-surface shading gradient instead of reading as a flat
/// facet, and optionally bowed along its length by a gravity CURL.
///
/// **Why this type exists.** That stitch used to be written twice — `LeafCardGeometry.emitCard` in
/// DaydreamCore and `ForestTreeGeometry.emitLeafCard` in the app target — line for line the same
/// spine/margin/droop math, differing only in scalar precision, a hardcoded fold, single- versus
/// double-sided winding, and what each recorded per vertex. Two copies meant a margin fix landed
/// in one and not the other, which is precisely how DH-0626 / DH-0628 / DH-0629 / DH-0647 each
/// independently rediscovered the same shard-edged leaf. The differences that were real are
/// parameters here; the payload difference is the `LeafCardSink`; the duplication is gone.
///
/// Generic over the scalar so each consumer keeps its own arithmetic exactly — see `LeafScalar`.
public enum LeafConstructor {

    /// How the two faces of a blade are emitted.
    ///
    /// This is a genuine difference between the consumers, not an accident to be normalized away.
    public enum Winding: Sendable {
        /// Emit BOTH faces as geometry: every strip quad is wound once toward the cup direction and
        /// once away. The blade is visible from either side under a single-sided material, at 2×
        /// the triangles. What `Mesh3` plant meshes use, because they are registered single-sided.
        case doubleSidedShell
        /// Emit ONE face per quad, wound toward the cup direction. Half the triangles; the consumer
        /// is responsible for making the mesh two-sided at the material level
        /// (`IlluminatoramaMesh.doubleSided`). What the yard tree uses — at canopy scale the
        /// halving is the difference between a ~50k and a ~100k triangle tree.
        case singleSided
    }

    /// The blade's placement and shape, in the caller's own scalar.
    ///
    /// `position` is the blade CENTROID. The base sits `height * 0.46` back along `-yAxis` from it,
    /// so a petiole attaches at the blade base rather than its middle.
    public struct Placement<Scalar: LeafScalar>: Sendable {
        /// Blade centroid.
        public var position: SIMD3<Scalar>
        /// Across-blade axis (the direction the margin mirrors along).
        public var xAxis: SIMD3<Scalar>
        /// Along-midrib axis, base → tip.
        public var yAxis: SIMD3<Scalar>
        /// The outward cup direction — the face the blade presents.
        public var bentNormal: SIMD3<Scalar>
        /// Blade width, tip to tip across the widest point, in metres.
        public var width: Scalar
        /// Blade length, base to tip, in metres.
        public var height: Scalar

        public init(position: SIMD3<Scalar>, xAxis: SIMD3<Scalar>, yAxis: SIMD3<Scalar>,
                    bentNormal: SIMD3<Scalar>, width: Scalar, height: Scalar) {
            self.position = position
            self.xAxis = xAxis
            self.yAxis = yAxis
            self.bentNormal = bentNormal
            self.width = width
            self.height = height
        }
    }

    /// The cross-blade taco cup: the margin lifts toward `bentNormal` in proportion to `u`, so the
    /// midrib sits in a valley. 0.30 is the value both historical call sites used.
    public static let defaultFold: Double = 0.30

    /// Emit ONE blade from an explicit half-margin.
    ///
    /// - Parameters:
    ///   - sink: where the triangles go; also decides what is recorded per vertex.
    ///   - placement: where the blade is and how big.
    ///   - margin: the half-margin, already resolved to the detail wanted (see
    ///     `LeafSilhouette.margin(tier:subdivisions:)`). `v` must be non-decreasing.
    ///   - fold: cross-blade cup depth. See `defaultFold`.
    ///   - curl: longitudinal gravity droop. The midrib bows out of the flat plane along
    ///     `-bentNormal`, accumulating toward the tip as `v²` — the small-deflection curve of a
    ///     self-weighted cantilever, evaluated once at build. `curl == 0` gives a flat blade. A
    ///     drooping leaf reads as a solid 3D form beside the furniture instead of a paper cutout,
    ///     and it costs no extra triangles: it repositions the stations that already exist.
    ///   - winding: see `Winding`.
    ///   - waveAmplitude: a gentle undulating RIPPLE on the margin — the width at each station is
    ///     modulated by `1 + waveAmplitude·sin(v·waveFrequency·2π + wavePhase)`, applied identically
    ///     to both sides so the leaf pinches/bulges symmetrically about the midrib rather than
    ///     scalloping unevenly. `0` (the default) reproduces the old razor-straight margin exactly.
    ///     A big SIMPLE entire leaf (fiddle-leaf fig) reads as die-cut plastic without this; keep it
    ///     small (~0.05-0.10) — this is a ripple, not the lobed/serrated read `margin`'s own control
    ///     points already carry, and a large value re-creates the DH-0628 "shard" look this
    ///     constructor was built to kill.
    public static func emitBlade<Sink: LeafCardSink>(
        into sink: inout Sink,
        placement: Placement<Sink.Scalar>,
        margin: [LeafSilhouette.Control],
        fold: Sink.Scalar,
        curl: Sink.Scalar = 0,
        winding: Winding,
        waveAmplitude: Sink.Scalar = 0,
        waveFrequency: Double = 3.0,
        wavePhase: Double = 0
    ) {
        guard margin.count >= 2 else { return }

        typealias S = Sink.Scalar
        let bent = placement.bentNormal
        let back = -bent
        let hw = placement.width / 2
        let h = placement.height
        let base = placement.position - placement.yAxis * (h * S(0.46))

        // Cantilever droop: the whole blade (spine and both margins) bends along -bentNormal by
        // curl·h·v², zero at the base, strongest at the tip.
        func droop(_ v: S) -> SIMD3<S> { bent * (-curl * h * v * v) }
        func spine(_ v: S) -> LeafStripVertex<S> {
            LeafStripVertex(position: base + placement.yAxis * (h * v) + droop(v),
                            v: v, u: 0, side: 0)
        }
        func edge(_ v: S, _ u: S, side: S) -> LeafStripVertex<S> {
            let ripple = waveAmplitude == 0 ? S(1)
                : S(1) + waveAmplitude * S(sin(Double(v) * waveFrequency * 2 * .pi + wavePhase))
            let wu = u * ripple
            let p = base
                + placement.yAxis * (h * v)
                + placement.xAxis * (side * wu * hw)
                + bent * (wu * hw * fold)
                + droop(v)
            return LeafStripVertex(position: p, v: v, u: wu, side: side)
        }

        for side in [S(1), S(-1)] {
            for i in 0 ..< (margin.count - 1) {
                let a = margin[i], b = margin[i + 1]
                let av = S(a.v), au = S(a.u), bv = S(b.v), bu = S(b.u)
                let sp0 = spine(av), sp1 = spine(bv)
                let m0 = edge(av, au, side: side)
                let m1 = edge(bv, bu, side: side)

                // The strip quad's true face normal, faced toward the cup direction. A sink that
                // blends its shading normal toward `bentNormal` needs this; one that shades flat
                // can ignore it.
                let raw = cross3(m0.position - sp0.position, m1.position - sp0.position)
                let gn = faceToward(safeNormalize(raw, fallback: bent), bent)

                switch winding {
                case .doubleSidedShell:
                    // Both faces, for both halves — the plant-mesh convention. `Mesh3.addTriangle`
                    // rewinds to `outward`, so the vertex order below is presentational.
                    sink.addLeafTriangle(sp0, m0, m1, outward: bent, geometricNormal: gn)
                    sink.addLeafTriangle(sp0, m1, sp1, outward: bent, geometricNormal: gn)
                    sink.addLeafTriangle(sp0, m1, m0, outward: back, geometricNormal: -gn)
                    sink.addLeafTriangle(sp0, sp1, m1, outward: back, geometricNormal: -gn)
                case .singleSided:
                    // One face per quad, wound toward the cup. The mirrored half needs the opposite
                    // vertex order to end up facing the same way — a sink that stores triangles
                    // verbatim depends on getting exactly this order.
                    if side > 0 {
                        sink.addLeafTriangle(sp0, m0, m1, outward: bent, geometricNormal: gn)
                        sink.addLeafTriangle(sp0, m1, sp1, outward: bent, geometricNormal: gn)
                    } else {
                        sink.addLeafTriangle(sp0, m1, m0, outward: bent, geometricNormal: gn)
                        sink.addLeafTriangle(sp0, sp1, m1, outward: bent, geometricNormal: gn)
                    }
                }
            }
        }
    }

    /// Emit ONE blade by silhouette — the ordinary entry point.
    ///
    /// A species names a silhouette and a detail tier; the margin is resolved and rounded here.
    /// `subdivisions <= 1` leaves the authored controls untouched, which is the byte-identity path
    /// the yard tree's pinned triangle counts depend on.
    public static func emitBlade<Sink: LeafCardSink>(
        into sink: inout Sink,
        placement: Placement<Sink.Scalar>,
        silhouette: LeafSilhouette,
        tier: LeafSilhouette.Tier = .hero,
        subdivisions: Int = 1,
        fold: Sink.Scalar = Sink.Scalar(LeafConstructor.defaultFold),
        curl: Sink.Scalar = 0,
        winding: Winding,
        waveAmplitude: Sink.Scalar = 0,
        waveFrequency: Double = 3.0,
        wavePhase: Double = 0
    ) {
        emitBlade(into: &sink,
                  placement: placement,
                  margin: silhouette.margin(tier: tier, subdivisions: subdivisions),
                  fold: fold, curl: curl, winding: winding,
                  waveAmplitude: waveAmplitude, waveFrequency: waveFrequency, wavePhase: wavePhase)
    }
}

// MARK: - Small vector helpers
//
// Local, and deliberately NaN-safe. A blade whose margin briefly pinches to zero width produces a
// degenerate cross product, and an unguarded `simd_normalize` turns that into a NaN vertex that
// poisons a whole baked mesh. The tree learned this the expensive way and grew guards at every
// combination point; sharing the constructor shares the guard too.

/// Generic cross product. `simd_cross` is overloaded only for the concrete float/double vectors, so
/// in a generic context it resolves against `simd_half3` and the call fails to type-check.
@inlinable
func cross3<S: LeafScalar>(_ a: SIMD3<S>, _ b: SIMD3<S>) -> SIMD3<S> {
    SIMD3(a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x)
}

@inlinable
func safeNormalize<S: LeafScalar>(_ v: SIMD3<S>, fallback: SIMD3<S>) -> SIMD3<S> {
    let m2 = v.x * v.x + v.y * v.y + v.z * v.z
    guard m2 > S(1e-24), m2.isFinite else { return fallback }
    let inv = S(1) / m2.squareRoot()
    let n = v * inv
    return (n.x.isFinite && n.y.isFinite && n.z.isFinite) ? n : fallback
}

@inlinable
func faceToward<S: LeafScalar>(_ n: SIMD3<S>, _ target: SIMD3<S>) -> SIMD3<S> {
    let d = n.x * target.x + n.y * target.y + n.z * target.z
    return d < 0 ? -n : n
}
