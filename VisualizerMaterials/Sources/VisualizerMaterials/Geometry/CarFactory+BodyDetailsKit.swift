import Foundation
import simd

// plausibility: real — every detail is a closed solid (thin strip / patch) laid on the real shell, found by ray-casting
// the finished `a.body` triangles; no CSG, no textures.

/// A point on the shell and its outward unit normal.
struct DSurf { var p: Vec3; var n: Vec3 }

/// Ray-caster over a finished mesh (the paint shell): nearest hit along an arbitrary ray, with a smooth outward normal.
struct BodyProbe {
    private var A: [Vec3] = [], B: [Vec3] = [], C: [Vec3] = []
    private var NA: [Vec3] = [], NB: [Vec3] = [], NC: [Vec3] = []
    private var lo: [Vec3] = [], hi: [Vec3] = []

    init(_ m: Mesh3) {
        let hasN = m.normals.count == m.positions.count
        var i = 0
        while i + 2 < m.indices.count {
            let ia = Int(m.indices[i]), ib = Int(m.indices[i + 1]), ic = Int(m.indices[i + 2]); i += 3
            let a = m.positions[ia], b = m.positions[ib], c = m.positions[ic]
            A.append(a); B.append(b); C.append(c)
            if hasN { NA.append(m.normals[ia]); NB.append(m.normals[ib]); NC.append(m.normals[ic]) }
            else { let n = normalize3(cross3(b - a, c - a)); NA.append(n); NB.append(n); NC.append(n) }
            lo.append(simd_min(a, simd_min(b, c))); hi.append(simd_max(a, simd_max(b, c)))
        }
    }

    /// Nearest hit. `fromInside` orients the returned normal along the ray (the ray leaves a solid), otherwise against it.
    func cast(_ o: Vec3, _ d: Vec3, fromInside: Bool = false, maxT: Double = 20) -> DSurf? {
        let inv = Vec3(d.x == 0 ? .infinity : 1 / d.x, d.y == 0 ? .infinity : 1 / d.y, d.z == 0 ? .infinity : 1 / d.z)
        var bestT = maxT
        var best: Int = -1
        var bu = 0.0, bv = 0.0
        for k in 0..<A.count {
            // slab test
            var t0 = 0.0, t1 = bestT
            var ok = true
            for ax in 0..<3 {
                if d[ax] == 0 { if o[ax] < lo[k][ax] - 1e-9 || o[ax] > hi[k][ax] + 1e-9 { ok = false; break } }
                else {
                    var ta = (lo[k][ax] - o[ax]) * inv[ax], tb = (hi[k][ax] - o[ax]) * inv[ax]
                    if ta > tb { swap(&ta, &tb) }
                    t0 = max(t0, ta); t1 = min(t1, tb)
                    if t0 > t1 + 1e-9 { ok = false; break }
                }
            }
            if !ok { continue }
            let e1 = B[k] - A[k], e2 = C[k] - A[k]
            let pv = cross3(d, e2), det = dot3(e1, pv)
            if abs(det) < 1e-14 { continue }
            let tv = o - A[k], u = dot3(tv, pv) / det
            if u < -1e-9 || u > 1 + 1e-9 { continue }
            let qv = cross3(tv, e1), v = dot3(d, qv) / det
            if v < -1e-9 || u + v > 1 + 1e-9 { continue }
            let t = dot3(e2, qv) / det
            if t > 1e-6 && t < bestT { bestT = t; best = k; bu = u; bv = v }
        }
        guard best >= 0 else { return nil }
        var n = normalize3(NA[best] * (1 - bu - bv) + NB[best] * bu + NC[best] * bv)
        let fn = normalize3(cross3(B[best] - A[best], C[best] - A[best]))
        if dot3(n, fn) < 0.5 { n = fn }
        if fromInside { if dot3(n, d) < 0 { n = -n } } else if dot3(n, d) > 0 { n = -n }
        return DSurf(p: o + d * bestT, n: n)
    }
}

/// Builders for thin closed solids that lie on a surface.
enum DKit {

    static func dedupe(_ s: [DSurf], eps: Double = 5e-4) -> [DSurf] {
        var out: [DSurf] = []
        for v in s where out.isEmpty || len3(v.p - out.last!.p) > eps { out.append(v) }
        return out
    }

    /// A line of constant `width` (across the surface) laid along `pts`: rectangular section running from `embed` below the
    /// surface to `proud` above it. Closed solid (end caps) unless `closed` (then the ring wraps).
    static func strip(_ pts0: [DSurf], width: Double, embed: Double, proud: Double, closed: Bool = false) -> Mesh3 {
        var pts = dedupe(pts0)
        if closed, pts.count > 2, len3(pts[0].p - pts[pts.count - 1].p) < 5e-4 { pts.removeLast() }
        let n = pts.count
        guard n >= 2, !(closed && n < 3) else { return Mesh3() }
        let hw = width / 2
        var rings: [[Vec3]] = []
        var tans: [Vec3] = []
        for i in 0..<n {
            let a = pts[closed ? (i + n - 1) % n : max(i - 1, 0)].p, b = pts[closed ? (i + 1) % n : min(i + 1, n - 1)].p
            var t = normalize3(b - a)
            let nn = pts[i].n
            t = normalize3(t - nn * dot3(t, nn))
            tans.append(t)
            let bn = normalize3(cross3(t, nn))
            let p = pts[i].p
            rings.append([p + nn * proud - bn * hw, p + nn * proud + bn * hw, p - nn * embed + bn * hw, p - nn * embed - bn * hw])
        }
        var m = Mesh3()
        let segs = closed ? n : n - 1
        for i in 0..<segs {
            let j = (i + 1) % n
            let outs = [pts[i].n + pts[j].n, ringSide(pts, i, j, +1), -(pts[i].n + pts[j].n), ringSide(pts, i, j, -1)]
            for k in 0..<4 {
                let k2 = (k + 1) % 4
                m.addQuad(rings[i][k], rings[i][k2], rings[j][k2], rings[j][k], outward: outs[k])
            }
        }
        if !closed {
            m.addQuad(rings[0][0], rings[0][1], rings[0][2], rings[0][3], outward: -tans[0])
            m.addQuad(rings[n - 1][0], rings[n - 1][1], rings[n - 1][2], rings[n - 1][3], outward: tans[n - 1])
        }
        return m
    }
    private static func ringSide(_ pts: [DSurf], _ i: Int, _ j: Int, _ s: Double) -> Vec3 {
        // +B / -B direction at the segment (B = t × n)
        let t = normalize3(pts[j].p - pts[i].p), nn = normalize3(pts[i].n + pts[j].n)
        return normalize3(cross3(t, nn)) * s
    }

    /// A patch: `rows[r][c]` lattice of surface points → closed slab `embed` below to `proud` above.
    static func grid(_ rows: [[DSurf]], embed: Double, proud: Double) -> Mesh3 {
        let R = rows.count, C = rows.first?.count ?? 0
        guard R >= 2, C >= 2, rows.allSatisfy({ $0.count == C }) else { return Mesh3() }
        func O(_ r: Int, _ c: Int) -> Vec3 { rows[r][c].p + rows[r][c].n * proud }
        func I(_ r: Int, _ c: Int) -> Vec3 { rows[r][c].p - rows[r][c].n * embed }
        func Nn(_ r: Int, _ c: Int) -> Vec3 { rows[r][c].n }
        var m = Mesh3()
        for r in 0..<(R - 1) { for c in 0..<(C - 1) {
            let nn = Nn(r, c) + Nn(r + 1, c + 1)
            m.addQuad(O(r, c), O(r + 1, c), O(r + 1, c + 1), O(r, c + 1), outward: nn)
            m.addQuad(I(r, c), I(r + 1, c), I(r + 1, c + 1), I(r, c + 1), outward: -nn)
        } }
        for c in 0..<(C - 1) {
            m.addQuad(O(0, c), O(0, c + 1), I(0, c + 1), I(0, c), outward: rows[0][c].p - rows[1][c].p)
            m.addQuad(O(R - 1, c), O(R - 1, c + 1), I(R - 1, c + 1), I(R - 1, c), outward: rows[R - 1][c].p - rows[R - 2][c].p)
        }
        for r in 0..<(R - 1) {
            m.addQuad(O(r, 0), O(r + 1, 0), I(r + 1, 0), I(r, 0), outward: rows[r][0].p - rows[r][1].p)
            m.addQuad(O(r, C - 1), O(r + 1, C - 1), I(r + 1, C - 1), I(r, C - 1), outward: rows[r][C - 1].p - rows[r][C - 2].p)
        }
        return m
    }

    /// Rounded rectangle loop (CCW in (u,v)), `per` facets per corner.
    static func roundedRect(cu: Double, cv: Double, hu: Double, hv: Double, r: Double, per: Int = 5) -> [Vec2] {
        let rr = min(r, min(hu, hv) * 0.95)
        var out: [Vec2] = []
        for (cx, cy, a0) in [(hu - rr, hv - rr, 0.0), (-(hu - rr), hv - rr, 90.0), (-(hu - rr), -(hv - rr), 180.0), (hu - rr, -(hv - rr), 270.0)] {
            for k in 0...per {
                let a = (a0 + 90 * Double(k) / Double(per)) * .pi / 180
                out.append(Vec2(cu + cx + rr * cos(a), cv + cy + rr * sin(a)))
            }
        }
        return out
    }

    /// Place a mesh authored with local x = up, y = forward (along the car), z = outward normal at `s`.
    static func onSurface(_ m: Mesh3, at s: DSurf) -> Mesh3 {
        let n = normalize3(s.n)
        var up = Vec3(0, 1, 0)
        if abs(dot3(up, n)) > 0.95 { up = Vec3(0, 0, -1) }
        up = normalize3(up - n * dot3(up, n))
        let fwd = cross3(n, up)
        return m.placed(xAxis: up, yAxis: fwd, zAxis: n, origin: s.p)
    }
}
