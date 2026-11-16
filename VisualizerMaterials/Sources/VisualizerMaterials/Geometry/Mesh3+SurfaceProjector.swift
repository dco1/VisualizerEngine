//  SurfaceProjector — nearest point on a triangle mesh, via a uniform grid. Used by the remesher to keep
//  relaxed vertices on the original surface.
//
//  Guarantees: returns the exact closest point on the triangles it was built from (searching grid rings outward
//  until no closer triangle can exist). Does not guarantee speed for queries far from the surface (it caps the
//  search and returns the query point). CPU; one query per vertex per iteration, a candidate for a GPU port.

import simd

struct SurfaceProjector {
    private var tris: [(Vec3, Vec3, Vec3)] = []
    private var grid: [SIMD3<Int32>: [Int32]] = [:]
    private let cell: Double

    init(_ t: TriTopology, cell: Double) {
        self.cell = Swift.max(cell, 1e-9)
        for f in 0 ..< t.tri.count where t.alive[f] {
            let a = t.pos[t.tri[f].x], b = t.pos[t.tri[f].y], c = t.pos[t.tri[f].z]
            let id = Int32(tris.count)
            tris.append((a, b, c))
            let lo = simd.min(a, simd.min(b, c)), hi = simd.max(a, simd.max(b, c))
            let k0 = key(lo), k1 = key(hi)
            for x in k0.x ... k1.x { for y in k0.y ... k1.y { for z in k0.z ... k1.z {
                grid[SIMD3(x, y, z), default: []].append(id)
            } } }
        }
    }

    private func key(_ p: Vec3) -> SIMD3<Int32> {
        SIMD3(Int32((p.x / cell).rounded(.down)), Int32((p.y / cell).rounded(.down)), Int32((p.z / cell).rounded(.down)))
    }

    func closest(to p: Vec3, maxRings: Int = 6) -> Vec3 {
        let k = key(p)
        var best = p, bestD = Double.infinity
        var seen = Set<Int32>()
        for r in 0 ... maxRings {
            let ri = Int32(r)
            for x in (k.x - ri) ... (k.x + ri) { for y in (k.y - ri) ... (k.y + ri) { for z in (k.z - ri) ... (k.z + ri) {
                if r > 0, abs(x - k.x) != ri, abs(y - k.y) != ri, abs(z - k.z) != ri { continue }
                guard let ids = grid[SIMD3(x, y, z)] else { continue }
                for id in ids where seen.insert(id).inserted {
                    let tr = tris[Int(id)]
                    let c = SurfaceProjector.closestOnTriangle(p, tr.0, tr.1, tr.2)
                    let d = dot3(c - p, c - p)
                    if d < bestD { bestD = d; best = c }
                }
            } } }
            if bestD.isFinite, bestD.squareRoot() <= Double(r) * cell { break }
        }
        return bestD.isFinite ? best : p
    }

    /// Ericson, Real-Time Collision Detection §5.1.5.
    static func closestOnTriangle(_ p: Vec3, _ a: Vec3, _ b: Vec3, _ c: Vec3) -> Vec3 {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = dot3(ab, ap), d2 = dot3(ac, ap)
        if d1 <= 0, d2 <= 0 { return a }
        let bp = p - b, d3 = dot3(ab, bp), d4 = dot3(ac, bp)
        if d3 >= 0, d4 <= d3 { return b }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0, d1 >= 0, d3 <= 0 { return a + ab * (d1 / (d1 - d3)) }
        let cp = p - c, d5 = dot3(ab, cp), d6 = dot3(ac, cp)
        if d6 >= 0, d5 <= d6 { return c }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0, d2 >= 0, d6 <= 0 { return a + ac * (d2 / (d2 - d6)) }
        let va = d3 * d6 - d5 * d4
        if va <= 0, d4 - d3 >= 0, d5 - d6 >= 0 { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) }
        let denom = 1 / (va + vb + vc)
        return a + ab * (vb * denom) + ac * (vc * denom)
    }
}
