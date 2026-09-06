import simd
import VisualizerMaterials

/// The scalar a vegetation mesh is built in.
///
/// Both precisions are real consumers and neither can be converted to the other for free: the
/// houseplant path builds `Mesh3`, which is `Double` throughout, while the yard tree builds a
/// `Float` triangle soup whose baked triangle counts and per-vertex positions are regression-gated.
/// Rounding one through the other would move vertices by an ulp for no reason, so the constructor
/// is generic and each consumer keeps its own arithmetic exactly.
public protocol LeafScalar: BinaryFloatingPoint, SIMDScalar, Sendable, Codable where SIMD4Storage: Sendable {}
extension Float: LeafScalar {}
extension Double: LeafScalar {}

/// One vertex of an emitted blade, in the blade's own parametric terms as well as in space.
///
/// This is the whole reason the leaf constructor can be shared. The two historical leaf systems
/// stitched the SAME strip and then disagreed about what to *record* per vertex: the houseplant
/// path wanted a flat `Mesh3` triangle with a face normal and no colour, while the yard tree
/// wanted a blended shading normal plus a midrib→margin→tip colour gradient and a per-vertex
/// foliage/SSS mark. Neither payload belongs in the constructor — but both are pure functions of
/// where the vertex sits ON the blade, which is exactly what `v` / `u` / `side` carry.
///
/// So the constructor computes geometry once and hands each consumer the parametric coordinates it
/// needs to compute its own payload. Nothing is lost by sharing, and the ~25 duplicated lines of
/// strip-stitch stop being duplicated.
public struct LeafStripVertex<Scalar: LeafScalar>: Sendable {
    /// Position, with the taco fold and the cantilever droop already applied.
    public var position: SIMD3<Scalar>
    /// Fraction along the midrib: 0 at the blade base, 1 at the tip.
    public var v: Scalar
    /// Half-width fraction across the blade: 0 on the midrib, →1 at the margin. Never negative —
    /// `side` carries which half, so a consumer colouring by distance-from-midrib uses `u` directly.
    public var u: Scalar
    /// Which half of the mirrored blade this vertex is on: `+1` or `-1`.
    public var side: Scalar

    public init(position: SIMD3<Scalar>, v: Scalar, u: Scalar, side: Scalar) {
        self.position = position
        self.v = v
        self.u = u
        self.side = side
    }
}

/// The receiver a leaf / petal / frond / needle / pad is emitted into.
///
/// Conformances are deliberately thin adapters, one per vertex format in the codebase:
///
///  - `Mesh3` (DaydreamCore) — indexed `Double` mesh, face normal by construction, no colour
///    channel. Ignores the parametric fields and forwards to `Mesh3.addTriangle(outward:)`, so
///    winding stays declared once and `MeshAudit` keeps its contract.
///  - `ForestTreeGeometry.Soup` (the yard tree) — de-indexed `Float` soup with caller-supplied
///    smooth normals, per-vertex colour and several sidecar arrays. Uses `v` / `u` to evaluate its
///    midrib→margin→tip gradient and the sun-facing albedo gate, exactly as it did when it owned
///    the loop.
///
/// `geometricNormal` is the strip quad's true face normal, already faced toward the cup direction.
/// A consumer that shades flat can ignore it (`Mesh3` derives an equivalent itself); a consumer
/// that blends toward the bent normal — as the tree does, `0.62·bent + 0.38·geometric` — needs it
/// and would otherwise recompute the same cross product.
public protocol LeafCardSink {
    associatedtype Scalar: LeafScalar

    /// Append one triangle of a blade.
    ///
    /// `outward` is the direction the finished face must point. A sink whose backing store rewinds
    /// triangles (like `Mesh3`) should honour `outward` and ignore the given vertex order; a sink
    /// that stores triangles verbatim should append in the order given, which the constructor has
    /// already wound to agree with `outward`.
    mutating func addLeafTriangle(_ a: LeafStripVertex<Scalar>,
                                  _ b: LeafStripVertex<Scalar>,
                                  _ c: LeafStripVertex<Scalar>,
                                  outward: SIMD3<Scalar>,
                                  geometricNormal: SIMD3<Scalar>)
}
