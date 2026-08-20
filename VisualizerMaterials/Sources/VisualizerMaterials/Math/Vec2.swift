import simd

/// 2D point / vector in model space (meters; X=East, Y=North in plan).
/// We use `SIMD2<Double>` for speed + free Codable/Hashable/Sendable.
public typealias Vec2 = SIMD2<Double>

/// 2D scalar cross product (z-component of the 3D cross). Named `cross2` to avoid
/// colliding with `simd.cross`, which returns a `SIMD3` for 2D inputs.
@inlinable public func cross2(_ a: Vec2, _ b: Vec2) -> Double { a.x * b.y - a.y * b.x }
// NO `dot(Vec2, Vec2)` HERE — use `simd.dot`, which is the same function.
//
// A local `dot` used to live on this line, and it only ever compiled because Swift prefers a
// SAME-MODULE overload to an imported one. The moment these types moved out of DaydreamCore
// into this package that preference was gone and every call site tied with `simd.dot`
// ("ambiguous use of 'dot'") — both take `SIMD2<Double>` and both return `Double`, so no
// argument type can break the tie. Re-adding it here would arm the same trap for every future
// consumer that imports this package alongside `simd`.
@inlinable public func len(_ a: Vec2) -> Double { (a.x * a.x + a.y * a.y).squareRoot() }
@inlinable public func len2(_ a: Vec2) -> Double { a.x * a.x + a.y * a.y }
@inlinable public func dist(_ a: Vec2, _ b: Vec2) -> Double { len(b - a) }

@inlinable public func norm(_ a: Vec2) -> Vec2 {
    let l = len(a)
    return l > 0 ? a / l : a
}

/// Polar angle of a direction vector, normalized to `[0, 2π)`.
/// Used to sort edges around a vertex for face traversal.
@inlinable public func angle(_ a: Vec2) -> Double {
    let t = atan2(a.y, a.x)
    return t < 0 ? t + 2 * Double.pi : t
}

/// Orientation of the ordered triple (a, b, c):
/// `> 0` left turn (CCW), `< 0` right turn (CW), `0` collinear.
@inlinable public func orient(_ a: Vec2, _ b: Vec2, _ c: Vec2) -> Double {
    cross2(b - a, c - a)
}
