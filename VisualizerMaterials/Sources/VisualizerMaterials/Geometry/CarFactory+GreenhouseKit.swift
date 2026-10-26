import Foundation
import simd

// plausibility: real — closed prisms/rings extruded off the cabin's real surface; no CSG, no textures.

/// Small convex-polygon + extrusion kit used by the greenhouse steps. Polygons are `Vec2` loops
/// (x,y are whatever the surface parametrisation says: side = (z, y), windscreen = (x, s)).
enum GHKit {

    // MARK: 2D polygon ops (convex)

    /// Signed area (positive = counter-clockwise in the (x,y) plane).
    static func area(_ p: [Vec2]) -> Double {
        var s = 0.0
        for i in 0..<p.count { let a = p[i], b = p[(i + 1) % p.count]; s += a.x * b.y - b.x * a.y }
        return s / 2
    }

    static func ccw(_ p: [Vec2]) -> [Vec2] { area(p) < 0 ? p.reversed() : p }

    /// Drops consecutive (and wrap-around) near-duplicate points.
    static func dedupe(_ p: [Vec2], eps: Double = 1e-4) -> [Vec2] {
        var out: [Vec2] = []
        for v in p where out.isEmpty || simd_length(v - out.last!) > eps { out.append(v) }
        while out.count > 1, simd_length(out[0] - out[out.count - 1]) <= eps { out.removeLast() }
        return out
    }

    /// Keep the half-plane `a*x + b*y <= c` (Sutherland–Hodgman).
    static func clip(_ poly: [Vec2], a: Double, b: Double, c: Double) -> [Vec2] {
        guard !poly.isEmpty else { return [] }
        var out: [Vec2] = []
        func side(_ p: Vec2) -> Double { a * p.x + b * p.y - c }
        for i in 0..<poly.count {
            let p = poly[i], q = poly[(i + 1) % poly.count]
            let sp = side(p), sq = side(q)
            if sp <= 0 { out.append(p) }
            if (sp < 0 && sq > 0) || (sp > 0 && sq < 0) {
                let t = sp / (sp - sq)
                out.append(p + (q - p) * t)
            }
        }
        return dedupe(out)
    }

    /// Half-plane for the line `x = x0 + k*(y - y0)`: keep x <= line (`front`) or x >= line.
    static func clipLine(_ poly: [Vec2], x0: Double, k: Double, y0: Double, keepLess: Bool) -> [Vec2] {
        // x - k*y <= x0 - k*y0
        if keepLess { return clip(poly, a: 1, b: -k, c: x0 - k * y0) }
        return clip(poly, a: -1, b: k, c: -(x0 - k * y0))
    }

    /// Outward offset of a convex polygon (negative = inset). Miter length is capped.
    static func offset(_ p0: [Vec2], _ d: Double) -> [Vec2] {
        let p = ccw(p0)
        let n = p.count
        guard n >= 3 else { return p0 }
        var out: [Vec2] = []
        for i in 0..<n {
            let a = p[(i + n - 1) % n], b = p[i], c = p[(i + 1) % n]
            let e1 = simd_normalize(b - a), e2 = simd_normalize(c - b)
            let n1 = Vec2(e1.y, -e1.x), n2 = Vec2(e2.y, -e2.x)     // outward for CCW
            let k = 1 + simd_dot(n1, n2)
            let m = (n1 + n2) / max(k, 0.35)
            out.append(b + m * d)
        }
        return out
    }

    /// Rounds each corner with a 3-point quadratic arc whose reach is `d` (limited by the adjacent edge lengths).
    static func round(_ p0: [Vec2], _ d: Double) -> [Vec2] {
        let p = ccw(dedupe(p0))
        let n = p.count
        guard n >= 3 else { return p }
        var out: [Vec2] = []
        for i in 0..<n {
            let a = p[(i + n - 1) % n], b = p[i], c = p[(i + 1) % n]
            let la = simd_length(a - b), lc = simd_length(c - b)
            let u1 = (a - b) / max(la, 1e-9), u2 = (c - b) / max(lc, 1e-9)
            if simd_dot(u1, u2) < -0.985 { out.append(b); continue }       // practically straight
            let r = min(d, la * 0.4, lc * 0.4)
            if r < 0.002 { out.append(b); continue }
            let p0 = b + u1 * r, p2 = b + u2 * r
            out.append(p0); out.append(p0 * 0.25 + b * 0.5 + p2 * 0.25); out.append(p2)
        }
        return dedupe(out)
    }

    /// Adds interior points so no edge is longer than `maxLen` (keeps panes following a curved surface).
    static func densify(_ p: [Vec2], _ maxLen: Double) -> [Vec2] {
        var out: [Vec2] = []
        for i in 0..<p.count {
            let a = p[i], b = p[(i + 1) % p.count]
            let k = max(1, Int((simd_length(b - a) / maxLen).rounded(.up)))
            for j in 0..<k { out.append(a + (b - a) * (Double(j) / Double(k))) }
        }
        return out
    }

    // MARK: Extrusion

    /// Closed prism between two equal-length loops. `top[i]` is `bottom[i]` pushed along the extrusion.
    /// Loops must be convex (fan caps) and are re-ordered so the winding is outward.
    static func prism(bottom b0: [Vec3], top t0: [Vec3]) -> Mesh3 {
        guard b0.count >= 3, b0.count == t0.count else { return Mesh3() }
        var b = b0, t = t0
        var e = Vec3.zero
        for i in 0..<b.count { e += t[i] - b[i] }
        e = normalize3(e)
        var nw = Vec3.zero                                   // Newell normal of the top loop
        for i in 0..<t.count { let p = t[i], q = t[(i + 1) % t.count]
            nw += Vec3((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y)) }
        if dot3(nw, e) < 0 { b.reverse(); t.reverse() }      // CCW seen from +e
        let n = b.count
        var m = Mesh3()
        let cb = b.reduce(Vec3.zero, +) / Double(n), ct = t.reduce(Vec3.zero, +) / Double(n)
        func tri(_ p: Vec3, _ q: Vec3, _ r: Vec3) { m.addTriangle(p, q, r, outward: cross3(q - p, r - p)) }
        for i in 0..<n {
            let j = (i + 1) % n
            tri(ct, t[i], t[j])                               // top cap (+e)
            tri(cb, b[j], b[i])                               // bottom cap (-e)
            tri(b[i], b[j], t[j]); tri(b[i], t[j], t[i])      // outer wall
        }
        return m
    }

    /// Closed annular prism (a frame): outer loop with a hole loop of the SAME vertex count, same orientation.
    static func ring(outerB ob0: [Vec3], outerT ot0: [Vec3], innerB ib0: [Vec3], innerT it0: [Vec3]) -> Mesh3 {
        let n = ob0.count
        guard n >= 3, ot0.count == n, ib0.count == n, it0.count == n else { return Mesh3() }
        var ob = ob0, ot = ot0, ib = ib0, it = it0
        var e = Vec3.zero
        for i in 0..<n { e += ot[i] - ob[i] }
        e = normalize3(e)
        var nw = Vec3.zero
        for i in 0..<n { let p = ot[i], q = ot[(i + 1) % n]
            nw += Vec3((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y)) }
        if dot3(nw, e) < 0 { ob.reverse(); ot.reverse(); ib.reverse(); it.reverse() }
        var m = Mesh3()
        func tri(_ p: Vec3, _ q: Vec3, _ r: Vec3) { m.addTriangle(p, q, r, outward: cross3(q - p, r - p)) }
        for i in 0..<n {
            let j = (i + 1) % n
            // top annulus (+e): O_i, O_j, I_j, I_i
            tri(ot[i], ot[j], it[j]); tri(ot[i], it[j], it[i])
            // bottom annulus (-e)
            tri(ob[j], ob[i], ib[i]); tri(ob[j], ib[i], ib[j])
            // outer wall
            tri(ob[i], ob[j], ot[j]); tri(ob[i], ot[j], ot[i])
            // inner wall (faces the hole)
            tri(ib[j], ib[i], it[i]); tri(ib[j], it[i], it[j])
        }
        return m
    }

    /// A torus (closed, wrap-around) about `centre` with axis `axis`: centre-line radius `big`, tube radius `small`.
    static func torus(centre: Vec3, axis: Vec3, big: Double, small: Double, around: Int = 32, tube: Int = 10) -> Mesh3 {
        let ax = normalize3(axis)
        let seed = abs(ax.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let u = normalize3(cross3(ax, seed)), v = cross3(ax, u)
        func pt(_ i: Int, _ j: Int) -> (p: Vec3, c: Vec3) {
            let a = Double(i % around) / Double(around) * 2 * .pi, b = Double(j % tube) / Double(tube) * 2 * .pi
            let dir = u * cos(a) + v * sin(a)
            let c = centre + dir * big
            return (c + dir * (small * cos(b)) + ax * (small * sin(b)), c)
        }
        var m = Mesh3()
        for i in 0..<around { for j in 0..<tube {
            let p00 = pt(i, j), p10 = pt(i + 1, j), p11 = pt(i + 1, j + 1), p01 = pt(i, j + 1)
            let mid = (p00.p + p10.p + p11.p + p01.p) / 4, cc = (p00.c + p10.c) / 2
            let out = mid - cc
            m.addTriangle(p00.p, p10.p, p11.p, outward: out); m.addTriangle(p00.p, p11.p, p01.p, outward: out)
        } }
        return m.smoothed(creaseDegrees: 60)
    }

    /// A slab over a plan whose half-width varies with z (`hw(z)`), top surface `top(x, z)`, bottom `thickness` lower.
    /// Closed, outward-wound; `nx` cells across, one row per entry of `zs`.
    static func slab(zs: [Double], cx: Double = 0, hw: (Double) -> Double, nx: Int, thickness: Double, bottom: ((Double, Double) -> Double)? = nil, top: (Double, Double) -> Double) -> Mesh3 {
        guard zs.count >= 2, nx >= 1 else { return Mesh3() }
        func X(_ i: Int, _ j: Int) -> Double { cx + (-1 + 2 * Double(i) / Double(nx)) * hw(zs[j]) }
        func T(_ i: Int, _ j: Int) -> Vec3 { Vec3(X(i, j), top(X(i, j), zs[j]), zs[j]) }
        func B(_ i: Int, _ j: Int) -> Vec3 { let t = T(i, j); return Vec3(t.x, bottom?(t.x, t.z) ?? (t.y - thickness), t.z) }
        let nz = zs.count - 1
        var m = Mesh3()
        for i in 0..<nx { for j in 0..<nz {
            m.addQuad(T(i, j), T(i + 1, j), T(i + 1, j + 1), T(i, j + 1), outward: Vec3(0, 1, 0))
            m.addQuad(B(i, j), B(i + 1, j), B(i + 1, j + 1), B(i, j + 1), outward: Vec3(0, -1, 0))
        } }
        for j in 0..<nz {
            m.addQuad(T(0, j), T(0, j + 1), B(0, j + 1), B(0, j), outward: Vec3(-1, 0, 0))
            m.addQuad(T(nx, j), T(nx, j + 1), B(nx, j + 1), B(nx, j), outward: Vec3(1, 0, 0))
        }
        for i in 0..<nx {
            m.addQuad(T(i, 0), T(i + 1, 0), B(i + 1, 0), B(i, 0), outward: Vec3(0, 0, -1))
            m.addQuad(T(i, nz), T(i + 1, nz), B(i + 1, nz), B(i, nz), outward: Vec3(0, 0, 1))
        }
        return m
    }
}

/// A parametrised patch of the cabin surface: maps a 2D coordinate to a point and its outward unit normal.
struct GHSurface {
    let map: (Vec2) -> (p: Vec3, n: Vec3)

    /// Prism of `poly` (in the surface's 2D space) from `inward` below the surface to `outward` above it.
    func prism(_ poly: [Vec2], inward: Double, outward: Double) -> Mesh3 {
        let (b, t) = loops(poly, inward: inward, outward: outward)
        return GHKit.prism(bottom: b, top: t)
    }

    func loops(_ poly: [Vec2], inward: Double, outward: Double) -> ([Vec3], [Vec3]) {
        var b: [Vec3] = [], t: [Vec3] = []
        for q in poly { let s = map(q); b.append(s.p - s.n * inward); t.append(s.p + s.n * outward) }
        return (b, t)
    }

    /// A frame around `poly`: `width` wide outside the polygon, lipped `overlap` over its edge.
    func frame(_ poly: [Vec2], width: Double, overlap: Double, inward: Double, outward: Double) -> Mesh3 {
        let inner = GHKit.offset(poly, -overlap), outer = GHKit.offset(poly, width)
        let (ib, it) = loops(inner, inward: inward, outward: outward)
        let (ob, ot) = loops(outer, inward: inward, outward: outward)
        return GHKit.ring(outerB: ob, outerT: ot, innerB: ib, innerT: it)
    }
}
