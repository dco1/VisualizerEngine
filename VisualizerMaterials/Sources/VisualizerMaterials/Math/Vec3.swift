import simd

/// 3D point / vector in **world** space (meters; X=East, Y=up, Z=South — see §12).
/// `SIMD3<Double>` for precision in the core; we narrow to `Float` only at the
/// engine boundary (the Phase-3 `HouseSceneBuilder` → `IlluminatoramaVertex`).
public typealias Vec3 = SIMD3<Double>

/// 3D cross product. Named `cross3` to avoid colliding with `simd.cross`.
@inlinable public func cross3(_ a: Vec3, _ b: Vec3) -> Vec3 {
    Vec3(a.y * b.z - a.z * b.y,
         a.z * b.x - a.x * b.z,
         a.x * b.y - a.y * b.x)
}
@inlinable public func dot3(_ a: Vec3, _ b: Vec3) -> Double { a.x * b.x + a.y * b.y + a.z * b.z }
@inlinable public func len3(_ a: Vec3) -> Double { dot3(a, a).squareRoot() }
@inlinable public func dist3(_ a: Vec3, _ b: Vec3) -> Double { len3(b - a) }

@inlinable public func normalize3(_ a: Vec3) -> Vec3 {
    let l = len3(a)
    return l > 0 ? a / l : a
}
