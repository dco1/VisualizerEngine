//  TriangleBVH — a bounding-volume hierarchy over a Mesh3's triangles with nearest-point and ray queries.
//
//  Built top-down by median split on the longest axis of the triangle centroids (leaves of ≤ 4 triangles). Immutable
//  once built, so it is Sendable and safe to query from many threads. Guarantees: `closest` returns the exact
//  nearest point on the mesh (not just on a vertex); `raycast` the exact nearest hit along the ray. Not guaranteed:
//  no handling of degenerate (zero-area) triangles beyond skipping them as hits, and it does not know which side
//  of a surface is "inside".

import simd

struct TriangleBVH: Sendable {
    struct Closest { var point: Vec3; var tri: Int; var bary: Vec3; var distance: Double }
    struct Hit { var point: Vec3; var tri: Int; var bary: Vec3; var t: Double }

    private struct Node: Sendable { var lo: Vec3; var hi: Vec3; var left: Int32; var right: Int32; var start: Int32; var count: Int32 }
    private var nodes: [Node] = []
    private var order: [Int32] = []
    let a: [Vec3], b: [Vec3], c: [Vec3]
    let mesh: Mesh3

    init(_ m: Mesh3) {
        mesh = m
        var A: [Vec3] = [], B: [Vec3] = [], C: [Vec3] = []
        var t = 0
        while t + 2 < m.indices.count {
            A.append(m.positions[Int(m.indices[t])]); B.append(m.positions[Int(m.indices[t + 1])]); C.append(m.positions[Int(m.indices[t + 2])])
            t += 3
        }
        a = A; b = B; c = C
        order = (0..<A.count).map { Int32($0) }
        guard !A.isEmpty else { return }
        let cen = (0..<A.count).map { (A[$0] + B[$0] + C[$0]) / 3 }
        _ = build(0, A.count, cen)
    }

    private mutating func build(_ s: Int, _ e: Int, _ cen: [Vec3]) -> Int32 {
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        var clo = lo, chi = hi
        for k in s..<e {
            let i = Int(order[k])
            lo = simd.min(lo, simd.min(a[i], simd.min(b[i], c[i]))); hi = simd.max(hi, simd.max(a[i], simd.max(b[i], c[i])))
            clo = simd.min(clo, cen[i]); chi = simd.max(chi, cen[i])
        }
        let idx = Int32(nodes.count)
        nodes.append(Node(lo: lo, hi: hi, left: -1, right: -1, start: Int32(s), count: Int32(e - s)))
        if e - s <= 4 { return idx }
        let ext = chi - clo
        let axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2)
        order[s..<e].sort { cen[Int($0)][axis] < cen[Int($1)][axis] }
        let mid = (s + e) / 2
        let l = build(s, mid, cen), r = build(mid, e, cen)
        nodes[Int(idx)].left = l; nodes[Int(idx)].right = r; nodes[Int(idx)].count = 0
        return idx
    }

    /// Tri face normal (unit, winding order), or nil if degenerate.
    func faceNormal(_ i: Int) -> Vec3? {
        let n = cross3(b[i] - a[i], c[i] - a[i])
        return len3(n) > 1e-14 ? normalize3(n) : nil
    }

    /// Interpolated surface normal at a hit: the mesh's own vertex normals when it has them, else the face normal.
    func normal(tri i: Int, bary w: Vec3) -> Vec3 {
        if mesh.normals.count == mesh.positions.count {
            let n = mesh.normals[Int(mesh.indices[i * 3])] * w.x + mesh.normals[Int(mesh.indices[i * 3 + 1])] * w.y
                + mesh.normals[Int(mesh.indices[i * 3 + 2])] * w.z
            if len3(n) > 1e-9 { return normalize3(n) }
        }
        return faceNormal(i) ?? Vec3(0, 1, 0)
    }

    func closest(to p: Vec3, maxDistance: Double = .infinity) -> Closest? {
        guard !nodes.isEmpty else { return nil }
        var best: Closest? = nil
        var bestD2 = maxDistance * maxDistance
        var stack: [Int32] = [0]
        while let ni = stack.popLast() {
            let n = nodes[Int(ni)]
            let d = simd.max(n.lo - p, simd.max(p - n.hi, Vec3(repeating: 0)))
            if dot3(d, d) > bestD2 { continue }
            if n.left < 0 {
                for k in Int(n.start)..<Int(n.start + n.count) {
                    let i = Int(order[k])
                    let q = TriangleBVH.closestPoint(p, a[i], b[i], c[i])
                    let d2 = dot3(q - p, q - p)
                    if d2 <= bestD2 {
                        bestD2 = d2
                        best = Closest(point: q, tri: i, bary: TriangleBVH.barycentric(q, a[i], b[i], c[i]), distance: d2.squareRoot())
                    }
                }
            } else {
                // visit the nearer child last (popped first)
                let l = nodes[Int(n.left)], r = nodes[Int(n.right)]
                let dl = dot3((l.lo + l.hi) / 2 - p, (l.lo + l.hi) / 2 - p), dr = dot3((r.lo + r.hi) / 2 - p, (r.lo + r.hi) / 2 - p)
                if dl < dr { stack.append(n.right); stack.append(n.left) } else { stack.append(n.left); stack.append(n.right) }
            }
        }
        return best
    }

    /// Nearest hit with t in (0, maxT]; the ray is `origin + t·dir`, `dir` unit.
    func raycast(origin o: Vec3, direction d: Vec3, maxT: Double = .infinity) -> Hit? {
        guard !nodes.isEmpty else { return nil }
        var best: Hit? = nil
        var bestT = maxT
        var stack: [Int32] = [0]
        let inv = Vec3(1 / (d.x == 0 ? 1e-30 : d.x), 1 / (d.y == 0 ? 1e-30 : d.y), 1 / (d.z == 0 ? 1e-30 : d.z))
        while let ni = stack.popLast() {
            let n = nodes[Int(ni)]
            // 1 nm of padding: a ray lying in a box face's plane (a seam) must still enter it.
            let pad = Vec3(repeating: 1e-9)
            let t1 = (n.lo - pad - o) * inv, t2 = (n.hi + pad - o) * inv
            let tmin = simd.min(t1, t2), tmax = simd.max(t1, t2)
            let enter = Swift.max(tmin.x, tmin.y, tmin.z), exit = Swift.min(tmax.x, tmax.y, tmax.z)
            if exit < Swift.max(enter, 0) || enter > bestT { continue }
            if n.left < 0 {
                for k in Int(n.start)..<Int(n.start + n.count) {
                    let i = Int(order[k])
                    if let (t, bary) = TriangleBVH.rayTriangle(o, d, a[i], b[i], c[i]), t > 1e-12, t < bestT {
                        bestT = t; best = Hit(point: o + d * t, tri: i, bary: bary, t: t)
                    }
                }
            } else { stack.append(n.left); stack.append(n.right) }
        }
        return best
    }

    // MARK: primitives

    static func barycentric(_ p: Vec3, _ a: Vec3, _ b: Vec3, _ c: Vec3) -> Vec3 {
        let v0 = b - a, v1 = c - a, v2 = p - a
        let d00 = dot3(v0, v0), d01 = dot3(v0, v1), d11 = dot3(v1, v1), d20 = dot3(v2, v0), d21 = dot3(v2, v1)
        let den = d00 * d11 - d01 * d01
        guard abs(den) > 1e-30 else { return Vec3(1, 0, 0) }
        let v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den
        return Vec3(1 - v - w, v, w)
    }

    /// Ericson, Real-Time Collision Detection §5.1.5.
    static func closestPoint(_ p: Vec3, _ a: Vec3, _ b: Vec3, _ c: Vec3) -> Vec3 {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = dot3(ab, ap), d2 = dot3(ac, ap)
        if d1 <= 0 && d2 <= 0 { return a }
        let bp = p - b, d3 = dot3(ab, bp), d4 = dot3(ac, bp)
        if d3 >= 0 && d4 <= d3 { return b }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 { return a + ab * (d1 / (d1 - d3)) }
        let cp = p - c, d5 = dot3(ab, cp), d6 = dot3(ac, cp)
        if d6 >= 0 && d5 <= d6 { return c }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 { return a + ac * (d2 / (d2 - d6)) }
        let va = d3 * d6 - d5 * d4
        if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) }
        let den = 1 / (va + vb + vc)
        return a + ab * (vb * den) + ac * (vc * den)
    }

    /// Möller–Trumbore, double sided. Returns (t, barycentric).
    static func rayTriangle(_ o: Vec3, _ d: Vec3, _ a: Vec3, _ b: Vec3, _ c: Vec3) -> (Double, Vec3)? {
        let e1 = b - a, e2 = c - a, pv = cross3(d, e2), det = dot3(e1, pv)
        guard abs(det) > 1e-14 else { return nil }
        let inv = 1 / det, tv = o - a
        let u = dot3(tv, pv) * inv
        guard u >= -1e-9, u <= 1 + 1e-9 else { return nil }
        let qv = cross3(tv, e1), v = dot3(d, qv) * inv
        guard v >= -1e-9, u + v <= 1 + 1e-9 else { return nil }
        return (dot3(e2, qv) * inv, Vec3(1 - u - v, u, v))
    }
}
