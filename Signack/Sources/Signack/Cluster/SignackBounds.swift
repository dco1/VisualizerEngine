import simd

/// A bounding sphere packed as (centre.xyz, radius) — the layout S2 uploads to the GPU as-is.
public typealias SignackSphere = SIMD4<Float>

public extension SIMD4 where Scalar == Float {
    /// The sphere's centre (a `SignackSphere` is centre.xyz + radius).
    @inlinable var center: SIMD3<Float> { SIMD3(x, y, z) }
    /// The sphere's radius.
    @inlinable var radius: Float { w }
}

/// Bounding volumes for clusters: spheres (culling + LOD error projection) and normal cones
/// (back-face cluster culling for single-sided content).
public enum SignackBounds {

    /// A tight-ish sphere around `points` (Ritter: farthest-pair seed, then grow). Deterministic.
    public static func sphere(_ points: [SIMD3<Float>]) -> SignackSphere {
        guard let first = points.first else { return .zero }
        var a = first
        var best: Float = -1
        for p in points { let d = simd_distance_squared(p, first); if d > best { best = d; a = p } }
        var b = a
        best = -1
        for p in points { let d = simd_distance_squared(p, a); if d > best { best = d; b = p } }
        var c = (a + b) * 0.5
        var r = simd_distance(a, b) * 0.5
        for p in points {
            let d = simd_distance(p, c)
            if d > r {
                let nr = (r + d) * 0.5
                c += (p - c) * ((nr - r) / d)
                r = nr
            }
        }
        return SignackSphere(c, r * 1.0001 + 1e-6)
    }

    /// A sphere enclosing every sphere in `spheres` (largest first, then grown). Deterministic, and
    /// it CONTAINS each input — the property the DAG's monotone LOD bounds rest on.
    public static func union(_ spheres: [SignackSphere]) -> SignackSphere {
        guard var s = spheres.max(by: { $0.w < $1.w }) else { return .zero }
        for t in spheres {
            let d = simd_distance(s.center, t.center)
            if d + t.w <= s.w { continue }
            if d + s.w <= t.w { s = t; continue }
            let nr = (s.w + d + t.w) * 0.5
            let c = d > 1e-12 ? s.center + (t.center - s.center) * ((nr - s.w) / d) : s.center
            s = SignackSphere(c, nr)
        }
        // Float slack, so `contains` holds for every input after rounding.
        return SignackSphere(s.center, s.w * 1.0001 + 1e-6)
    }

    /// Does `outer` contain `inner` (within `slack`)?
    @inlinable public static func contains(_ outer: SignackSphere, _ inner: SignackSphere,
                                           slack: Float = 1e-4) -> Bool {
        simd_distance(outer.center, inner.center) + inner.w <= outer.w * (1 + slack) + slack
    }
}

/// The spread of a cluster's face normals: every normal lies within `acos(cutoff)`… of `axis`,
/// stored meshoptimizer-style so the back-face test is one dot product (see `isBackfacing`).
public struct SignackNormalCone: Sendable, Equatable {
    public var axis: SIMD3<Float>
    /// sin of the cone's half-angle; 1 means "spans a hemisphere or more — never cull".
    public var cutoff: Float

    public static let open = SignackNormalCone(axis: SIMD3(0, 1, 0), cutoff: 1)

    public init(axis: SIMD3<Float>, cutoff: Float) { self.axis = axis; self.cutoff = cutoff }

    /// The cone of the given unit face normals.
    public init(normals: [SIMD3<Float>]) {
        var sum = SIMD3<Float>(repeating: 0)
        for n in normals { sum += n }
        let len = simd_length(sum)
        guard len > 1e-8 else { self = .open; return }
        let axis = sum / len
        var minDot: Float = 1
        for n in normals { minDot = min(minDot, simd_dot(n, axis)) }
        // Wider than ~84° from the axis: the cone test can never succeed, store "never cull".
        self.axis = axis
        self.cutoff = minDot <= 0.1 ? 1 : (1 - minDot * minDot).squareRoot()
    }

    /// True when every face of a cluster bounded by `sphere` faces away from `camera` — the cluster
    /// can be skipped for SINGLE-sided content. Double-sided content (leaf cards) must not use it.
    @inlinable public func isBackfacing(sphere: SignackSphere, camera: SIMD3<Float>) -> Bool {
        guard cutoff < 1 else { return false }
        let d = sphere.center - camera
        return simd_dot(d, axis) >= cutoff * simd_length(d) + sphere.w
    }
}
