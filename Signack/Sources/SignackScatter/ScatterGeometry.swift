import simd

/// Plan-polygon queries the rules need: inside, area, bounds, distance to the outline.
/// Plan metres, any winding, no self-intersection assumed beyond what even-odd tolerates.
public enum ScatterGeometry {

    /// Even-odd point-in-polygon.
    public static func contains(_ poly: [SIMD2<Double>], _ p: SIMD2<Double>) -> Bool {
        guard poly.count >= 3 else { return false }
        var inside = false
        var j = poly.count - 1
        for i in 0..<poly.count {
            let a = poly[i], b = poly[j]
            if (a.y > p.y) != (b.y > p.y),
               p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x { inside.toggle() }
            j = i
        }
        return inside
    }

    /// Unsigned area (shoelace).
    public static func area(_ poly: [SIMD2<Double>]) -> Double {
        guard poly.count >= 3 else { return 0 }
        var s = 0.0
        for i in 0..<poly.count {
            let a = poly[i], b = poly[(i + 1) % poly.count]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) * 0.5
    }

    /// Distance from `p` to segment `a`–`b`.
    public static func distance(_ p: SIMD2<Double>, segment a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let d = b - a
        let l2 = simd_length_squared(d)
        let t = l2 > 1e-18 ? max(0, min(1, simd_dot(p - a, d) / l2)) : 0
        return simd_distance(p, a + d * t)
    }

    /// Distance from `p` to the polygon's OUTLINE (0 on it; the same inside or out).
    public static func distanceToOutline(_ poly: [SIMD2<Double>], _ p: SIMD2<Double>) -> Double {
        guard poly.count >= 2 else { return .infinity }
        var best = Double.infinity
        for i in 0..<poly.count {
            best = min(best, distance(p, segment: poly[i], poly[(i + 1) % poly.count]))
        }
        return best
    }

    /// Axis-aligned bounds of many polygons, or nil when there is no vertex.
    public static func bounds(_ polys: [[SIMD2<Double>]]) -> (lo: SIMD2<Double>, hi: SIMD2<Double>)? {
        var lo = SIMD2<Double>(repeating: .infinity), hi = -lo
        for poly in polys { for p in poly { lo = simd_min(lo, p); hi = simd_max(hi, p) } }
        return lo.x <= hi.x ? (lo, hi) : nil
    }
}
