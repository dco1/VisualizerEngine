import simd

/// An engine-neutral indexed triangle mesh — the output of the structure→geometry
/// bridge. Flat-shaded by construction (each triangle owns its three vertices and a
/// single face normal), because architecture wants **hard** edges at wall corners,
/// not smoothing across them. The Phase-3 `HouseSceneBuilder` narrows this to
/// `[IlluminatoramaVertex]` (Double→Float) + synthesised tangents and uploads it.
///
/// "The geometry the app builds IS the geometry that renders" (§15): there is no
/// model→JSON→engine translation between this and the renderer, so it is directly
/// auditable — see `MeshAudit`.
public struct Mesh3: Equatable, Sendable {
    public var positions: [Vec3] = []
    public var normals: [Vec3] = []
    public var uvs: [Vec2] = []
    public var indices: [UInt32] = []

    public init() {}

    /// Wrap pre-built parallel arrays (the Phase-3 adapter round-trips through this;
    /// tests use it to construct deliberately-corrupt meshes for the auditor).
    public init(positions: [Vec3], normals: [Vec3], uvs: [Vec2], indices: [UInt32]) {
        self.positions = positions; self.normals = normals
        self.uvs = uvs; self.indices = indices
    }

    public var vertexCount: Int { positions.count }
    public var triangleCount: Int { indices.count / 3 }
    public var isEmpty: Bool { indices.isEmpty }

    /// Axis-aligned bounds of the vertices, or `nil` when empty.
    public var bounds: (min: Vec3, max: Vec3)? {
        guard let first = positions.first else { return nil }
        var lo = first, hi = first
        for p in positions.dropFirst() {
            lo = simd.min(lo, p); hi = simd.max(hi, p)
        }
        return (lo, hi)
    }
}

public extension Mesh3 {
    /// Minimum segment count for a revolved circle of `radius` (metres) so its chord
    /// error stays within `maxChordError` (metres, default 1 mm). Inverts
    /// `radius · (1 − cos(π/n)) ≤ maxChordError`. Use this instead of hand-picking
    /// 8/16/24 so a wide shade gets more sides than a thin pole **by construction** —
    /// the fix lever for `FacetingAudit` findings. Clamped to `[floor, cap]`.
    static func segmentsFor(radius: Double, maxChordError: Double = 0.001,
                            floor: Int = 12, cap: Int = 96) -> Int {
        guard radius > maxChordError, maxChordError > 0 else { return floor }
        let inner = max(-1.0, min(1.0, 1 - maxChordError / radius))   // = max cos(π/n)
        let n = Double.pi / acos(inner)
        return min(cap, Swift.max(floor, Int(n.rounded(.up))))
    }
}
