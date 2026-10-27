import Foundation

/// Ray intersection primitives shared by every picker, BVH and occlusion test, generic over
/// Float/Double so a single implementation serves the CPU geometry (Double) and the
/// renderer-facing code (Float).
public enum RayIntersect {
    @inlinable @inline(__always)
    static func dot<S: BinaryFloatingPoint & SIMDScalar>(_ a: SIMD3<S>, _ b: SIMD3<S>) -> S {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    @inlinable @inline(__always)
    static func cross<S: BinaryFloatingPoint & SIMDScalar>(_ a: SIMD3<S>, _ b: SIMD3<S>) -> SIMD3<S> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    /// Möller–Trumbore, no backface cull. Returns the ray parameter, or nil for a miss / a ray
    /// parallel to the triangle's plane. `tolerance` widens the barycentric bounds so a ray
    /// through a shared edge hits at least one of the two triangles.
    @inlinable @inline(__always)
    public static func triangle<S: BinaryFloatingPoint & SIMDScalar>(
        origin o: SIMD3<S>, direction d: SIMD3<S>,
        _ a: SIMD3<S>, _ b: SIMD3<S>, _ c: SIMD3<S>, tolerance: S
    ) -> S? {
        let e1 = b - a, e2 = c - a
        let p = cross(d, e2)
        let det = dot(e1, p)
        guard abs(det) > 1e-12 else { return nil }
        let inv = 1 / det
        let s = o - a
        let u = dot(s, p) * inv
        guard u >= -tolerance, u <= 1 + tolerance else { return nil }
        let q = cross(s, e1)
        let v = dot(d, q) * inv
        guard v >= -tolerance, u + v <= 1 + tolerance else { return nil }
        return dot(e2, q) * inv
    }

    /// Ray vs a box that is rotated about +Y by `angle` radians (slab test in box-local space).
    /// Returns the nearest positive hit distance (the exit distance when the origin is inside),
    /// or nil. The local→world rotation has columns X=(c,0,-s), Z=(s,0,c), so world→local is
    /// its transpose.
    @inlinable
    public static func yRotatedBox<S: BinaryFloatingPoint & SIMDScalar>(
        origin o: SIMD3<S>, direction d: SIMD3<S>,
        center: SIMD3<S>, half: SIMD3<S>, angle: S
    ) -> S? {
        let c = S(cos(Double(angle))), s = S(sin(Double(angle)))
        let p = o - center
        let lo = SIMD3<S>(c * p.x - s * p.z, p.y, s * p.x + c * p.z)
        let ld = SIMD3<S>(c * d.x - s * d.z, d.y, s * d.x + c * d.z)
        var tmin = -S.greatestFiniteMagnitude, tmax = S.greatestFiniteMagnitude
        for i in 0..<3 {
            if abs(ld[i]) < 1e-8 {
                if lo[i] < -half[i] || lo[i] > half[i] { return nil }   // parallel & outside
            } else {
                var t1 = (-half[i] - lo[i]) / ld[i], t2 = (half[i] - lo[i]) / ld[i]
                if t1 > t2 { swap(&t1, &t2) }
                tmin = max(tmin, t1); tmax = min(tmax, t2)
                if tmin > tmax { return nil }
            }
        }
        if tmax < 0 { return nil }
        return tmin > 0 ? tmin : tmax
    }
}
