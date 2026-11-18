//  A flat outline extruded into a closed solid: badges, signs, crests, inlays, any cut-out shape with thickness.

import simd

/// The polygon without repeated points and without points on a straight line between their neighbours (clipping and
/// sampling leave both; they make zero-area cap triangles and needless wall quads).
public func removingCollinear(_ poly: [Vec2], tolerance: Double = 1e-12) -> [Vec2] {
    var p = poly
    var changed = true
    while changed, p.count > 3 {
        changed = false
        for i in p.indices {
            let a = p[(i + p.count - 1) % p.count], b = p[i], c = p[(i + 1) % p.count]
            if simd_distance(a, b) < 1e-9 || abs(orient(a, b, c)) < tolerance { p.remove(at: i); changed = true; break }
        }
    }
    return p
}

public extension Mesh3 {
    /// A simple polygon (either winding, no holes) in the xy plane as a closed solid from `z0` to `z1`: ear-clipped caps
    /// (`Triangulation.earClip`) and one quad per edge, every face wound outward. Repeated and collinear points are
    /// dropped first. For n kept points it costs 2(n − 2) + 2n triangles.
    static func prism(_ outline: [Vec2], z0: Double, z1: Double) -> Mesh3 {
        var p = removingCollinear(outline)
        if signedArea(p) < 0 { p.reverse() }
        var m = Mesh3()
        guard p.count >= 3, z1 > z0 else { return m }
        func v(_ q: Vec2, _ z: Double) -> Vec3 { Vec3(q.x, q.y, z) }
        for (a, b, c) in Triangulation.earClip(p) {
            m.addTriangle(v(p[a], z1), v(p[b], z1), v(p[c], z1), outward: Vec3(0, 0, 1))
            m.addTriangle(v(p[a], z0), v(p[b], z0), v(p[c], z0), outward: Vec3(0, 0, -1))
        }
        for i in p.indices {
            let a = p[i], b = p[(i + 1) % p.count], e = b - a
            m.addQuad(v(a, z0), v(b, z0), v(b, z1), v(a, z1), outward: Vec3(e.y, -e.x, 0))
        }
        return m
    }

    /// Several disjoint outlines, each its own closed solid, in one mesh.
    static func prism(_ outlines: [[Vec2]], z0: Double, z1: Double) -> Mesh3 {
        var m = Mesh3()
        for o in outlines { m.append(prism(o, z0: z0, z1: z1)) }
        return m
    }
}
