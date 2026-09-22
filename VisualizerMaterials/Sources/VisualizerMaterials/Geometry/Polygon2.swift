import simd

/// Signed area of a closed polygon (vertices in order, no repeated closing point).
/// `> 0` ⇒ counter-clockwise, `< 0` ⇒ clockwise. Magnitude = area.
public func signedArea(_ pts: [Vec2]) -> Double {
    guard pts.count >= 3 else { return 0 }
    var s = 0.0
    for i in 0..<pts.count {
        let a = pts[i]
        let b = pts[(i + 1) % pts.count]
        s += cross2(a, b)
    }
    return s * 0.5
}

public func area(_ pts: [Vec2]) -> Double { abs(signedArea(pts)) }

public func centroid(_ pts: [Vec2]) -> Vec2 {
    guard pts.count >= 3 else {
        guard !pts.isEmpty else { return .zero }
        return pts.reduce(.zero, +) / Double(pts.count)
    }
    var a = 0.0
    var c = Vec2.zero
    for i in 0..<pts.count {
        let p = pts[i]
        let q = pts[(i + 1) % pts.count]
        let f = cross2(p, q)
        a += f
        c += (p + q) * f
    }
    a *= 0.5
    guard abs(a) > 1e-12 else { return pts.reduce(.zero, +) / Double(pts.count) }
    return c / (6 * a)
}

/// Point-in-polygon test (ray casting). Boundary points are not guaranteed.
public func contains(polygon pts: [Vec2], _ p: Vec2) -> Bool {
    guard pts.count >= 3 else { return false }
    var inside = false
    var j = pts.count - 1
    for i in 0..<pts.count {
        let a = pts[i], b = pts[j]
        if (a.y > p.y) != (b.y > p.y) {
            let xCross = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x
            if p.x < xCross { inside.toggle() }
        }
        j = i
    }
    return inside
}
