//  Robust orientation predicates for the mesh-intersection boolean (`Mesh3+Boolean*.swift`).
//
//  `orient3d` / `orient2d` return the EXACT sign of the determinant of double-precision input. Each tries a
//  floating-point filter first (Shewchuk's static error bound); only when the result is too close to zero to
//  trust does it fall back to exact expansion arithmetic (sums of non-overlapping doubles), so the answer never
//  depends on rounding and is the same whichever triangle asks. That is what keeps two neighbouring triangles
//  agreeing about which side of a plane a shared vertex is on — the property the watertight output rests on.
//
//  Guarantees: the sign is exact for finite inputs (no overflow handling beyond the usual double range).
//  Does not: make computed intersection POINTS exact; those are ordinary doubles, welded by a tolerance.

import Foundation
import simd

enum Robust {
    // MARK: expansion arithmetic (Shewchuk): arrays of non-overlapping doubles, increasing magnitude

    @inline(__always) static func twoSum(_ a: Double, _ b: Double) -> (Double, Double) {
        let x = a + b, bv = x - a, av = x - bv
        return (x, (a - av) + (b - bv))
    }
    @inline(__always) static func twoDiff(_ a: Double, _ b: Double) -> (Double, Double) {
        let x = a - b, bv = a - x, av = x + bv
        return (x, (a - av) + (bv - b))
    }
    @inline(__always) static func twoProd(_ a: Double, _ b: Double) -> (Double, Double) {
        let x = a * b
        return (x, fma(a, b, -x))
    }

    /// e + b, exactly.
    static func grow(_ e: [Double], _ b: Double) -> [Double] {
        var q = b
        var h: [Double] = []
        h.reserveCapacity(e.count + 1)
        for ei in e {
            let (x, y) = twoSum(q, ei)
            q = x
            if y != 0 { h.append(y) }
        }
        if q != 0 || h.isEmpty { h.append(q) }
        return h
    }
    static func add(_ e: [Double], _ f: [Double]) -> [Double] {
        var r = e
        for fi in f { r = grow(r, fi) }
        return r
    }
    static func neg(_ e: [Double]) -> [Double] { e.map { -$0 } }
    static func scale(_ e: [Double], _ b: Double) -> [Double] {
        var r: [Double] = [0]
        for ei in e {
            let (hi, lo) = twoProd(ei, b)
            r = grow(grow(r, lo), hi)
        }
        return r
    }
    static func mul(_ e: [Double], _ f: [Double]) -> [Double] {
        var r: [Double] = [0]
        for fi in f { r = add(r, scale(e, fi)) }
        return r
    }
    static func diff(_ a: Double, _ b: Double) -> [Double] {
        let (x, y) = twoDiff(a, b)
        return y == 0 ? [x] : [y, x]
    }
    /// Sign of the sum an expansion represents (its largest, last non-zero component).
    static func sign(_ e: [Double]) -> Int {
        for v in e.reversed() where v != 0 { return v > 0 ? 1 : -1 }
        return 0
    }

    // MARK: orient3d

    /// Sign of `(b−a)·((c−a)×(d−a))`: +1 when `d` is on the side of the normal of CCW triangle `abc`, −1 on the
    /// other, 0 only when the four points are exactly coplanar.
    static func orient3d(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3) -> Int {
        let u = b - a, v = c - a, w = d - a
        let m0 = v.y * w.z - v.z * w.y
        let m1 = v.z * w.x - v.x * w.z
        let m2 = v.x * w.y - v.y * w.x
        let det = u.x * m0 + u.y * m1 + u.z * m2
        let perm = abs(u.x) * (abs(v.y * w.z) + abs(v.z * w.y))
                 + abs(u.y) * (abs(v.z * w.x) + abs(v.x * w.z))
                 + abs(u.z) * (abs(v.x * w.y) + abs(v.y * w.x))
        let bound = 1.0e-15 * perm
        if det > bound { return 1 }
        if det < -bound { return -1 }
        return orient3dExact(a, b, c, d)
    }

    static func orient3dExact(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3) -> Int {
        let ux = diff(b.x, a.x), uy = diff(b.y, a.y), uz = diff(b.z, a.z)
        let vx = diff(c.x, a.x), vy = diff(c.y, a.y), vz = diff(c.z, a.z)
        let wx = diff(d.x, a.x), wy = diff(d.y, a.y), wz = diff(d.z, a.z)
        let m0 = add(mul(vy, wz), neg(mul(vz, wy)))
        let m1 = add(mul(vz, wx), neg(mul(vx, wz)))
        let m2 = add(mul(vx, wy), neg(mul(vy, wx)))
        let det = add(add(mul(ux, m0), mul(uy, m1)), mul(uz, m2))
        return sign(det)
    }

    // MARK: orient2d

    /// Sign of `(b−a)×(c−a)`: +1 when `a→b→c` turns counter-clockwise.
    static func orient2d(_ a: Vec2, _ b: Vec2, _ c: Vec2) -> Int {
        let l = (b.x - a.x) * (c.y - a.y)
        let r = (b.y - a.y) * (c.x - a.x)
        let det = l - r
        let bound = 3.3306690738754716e-16 * (abs(l) + abs(r))
        if det > bound { return 1 }
        if det < -bound { return -1 }
        let e = add(mul(diff(b.x, a.x), diff(c.y, a.y)), neg(mul(diff(b.y, a.y), diff(c.x, a.x))))
        return sign(e)
    }
}
