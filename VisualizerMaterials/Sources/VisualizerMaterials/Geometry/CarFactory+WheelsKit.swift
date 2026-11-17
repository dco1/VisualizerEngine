import Foundation
import simd

// plausibility: real — every wheel/underbody primitive is an exact closed solid built by declared-outward quads
// (lathe grid with per-vertex displacement, arc slabs, convex prisms); nothing is a texture or a stand-in.

/// Small solid-modelling helpers shared by the wheel / brake / underbody builders (CarFactory+Wheels*.swift).
enum WheelKit {

    typealias RZ = (r: Double, y: Double)

    /// A CLOSED (r, y) loop revolved about +Y into a closed torus-like solid. Unlike `Mesh3.revolve`, the profile never
    /// touches the axis, so a tyre / disc plate / pipe tip can be hollow. The loop must run counter-clockwise in
    /// (r right, y up): outer surface with increasing y, return along the smaller radius. `disp(i, j)` displaces ring
    /// point `j` at column `i` by `(dr, dy)` — the way tread blocks, sidewall lettering and brake slots are real geometry
    /// without a boolean (the vertex columns are shared, so the solid stays watertight).
    static func gridLathe(_ loop: [RZ], n: Int, disp: ((Int, Int) -> (Double, Double))? = nil) -> Mesh3 {
        var m = Mesh3()
        let M = loop.count
        guard M >= 3, n >= 3 else { return m }
        var v = [[Vec3]](repeating: [], count: M)
        for j in 0..<M {
            v[j] = (0..<n).map { i in
                let a = 2 * Double.pi * Double(i) / Double(n)
                let d = disp?(i, j) ?? (0, 0)
                let r = max(1e-4, loop[j].r + d.0)
                return Vec3(r * cos(a), loop[j].y + d.1, r * sin(a))
            }
        }
        for j in 0..<M {
            let k = (j + 1) % M
            let dR = loop[k].r - loop[j].r, dY = loop[k].y - loop[j].y
            if abs(dR) < 1e-12 && abs(dY) < 1e-12 { continue }
            for i in 0..<n {
                let i2 = (i + 1) % n
                let am = 2 * Double.pi * (Double(i) + 0.5) / Double(n)
                m.addQuad(v[j][i], v[j][i2], v[k][i2], v[k][i], outward: Vec3(cos(am) * dY, -dR, sin(am) * dY))
            }
        }
        return m
    }

    /// Lathe frame (axis +Y, outboard +Y) → wheel frame (axle +X, outboard +X).
    static func toWheel(_ m: Mesh3) -> Mesh3 { m.rotatedZ(-.pi / 2) }

    /// An annular sector slab in WORLD coordinates about the axis `(·, cy, cz)` parallel to X. Angle `a` runs from the
    /// vertical (`a = 0` straight up) toward +Z: direction (dy, dz) = (cos a, sin a). x ∈ [x0, x1], radius ∈ [r0, r1].
    static func arcSlab(cy: Double, cz: Double, r0: Double, r1: Double, a0: Double, a1: Double,
                        x0: Double, x1: Double, steps: Int) -> Mesh3 {
        var m = Mesh3()
        func P(_ r: Double, _ a: Double, _ x: Double) -> Vec3 { Vec3(x, cy + r * cos(a), cz + r * sin(a)) }
        func dir(_ a: Double) -> Vec3 { Vec3(0, cos(a), sin(a)) }
        for s in 0..<steps {
            let ta = a0 + (a1 - a0) * Double(s) / Double(steps), tb = a0 + (a1 - a0) * Double(s + 1) / Double(steps)
            let am = (ta + tb) / 2
            m.addQuad(P(r1, ta, x0), P(r1, ta, x1), P(r1, tb, x1), P(r1, tb, x0), outward: dir(am))
            m.addQuad(P(r0, ta, x0), P(r0, ta, x1), P(r0, tb, x1), P(r0, tb, x0), outward: dir(am) * -1)
            m.addQuad(P(r0, ta, x0), P(r1, ta, x0), P(r1, tb, x0), P(r0, tb, x0), outward: Vec3(-1, 0, 0))
            m.addQuad(P(r0, ta, x1), P(r1, ta, x1), P(r1, tb, x1), P(r0, tb, x1), outward: Vec3(1, 0, 0))
        }
        let tg = { (a: Double) in Vec3(0, -sin(a), cos(a)) }
        m.addQuad(P(r0, a0, x0), P(r1, a0, x0), P(r1, a0, x1), P(r0, a0, x1), outward: tg(a0) * -1)
        m.addQuad(P(r0, a1, x0), P(r1, a1, x0), P(r1, a1, x1), P(r0, a1, x1), outward: tg(a1))
        return m
    }

    /// A CONVEX polygon in the (z, y) plane extruded along X ∈ [x0, x1] — fins, ramps, brackets.
    static func prismX(_ poly: [(z: Double, y: Double)], x0: Double, x1: Double) -> Mesh3 {
        var m = Mesh3()
        guard poly.count >= 3 else { return m }
        let cz = poly.reduce(0) { $0 + $1.z } / Double(poly.count), cy = poly.reduce(0) { $0 + $1.y } / Double(poly.count)
        for e in 0..<poly.count {
            let p = poly[e], q = poly[(e + 1) % poly.count]
            m.addQuad(Vec3(x0, p.y, p.z), Vec3(x1, p.y, p.z), Vec3(x1, q.y, q.z), Vec3(x0, q.y, q.z),
                      outward: Vec3(0, (p.y + q.y) / 2 - cy, (p.z + q.z) / 2 - cz))
            m.addTriangle(Vec3(x1, cy, cz), Vec3(x1, p.y, p.z), Vec3(x1, q.y, q.z), outward: Vec3(1, 0, 0))
            m.addTriangle(Vec3(x0, cy, cz), Vec3(x0, p.y, p.z), Vec3(x0, q.y, q.z), outward: Vec3(-1, 0, 0))
        }
        return m
    }

    /// A chamfered hex nut / bolt head standing on y = 0 (lathe frame), `h` tall, across-corners radius `r`.
    static func hexNut(r: Double, h: Double) -> Mesh3 {
        var m = Mesh3()
        m.revolve(profile: [.init(r: 0, y: 0), .init(r: r, y: 0), .init(r: r, y: h * 0.78),
                            .init(r: r * 0.82, y: h), .init(r: 0, y: h)], segments: 6)
        return m
    }

    /// Solid disc-of-revolution from a (r, y) polyline that starts and ends ON the axis.
    static func lathe(_ p: [RZ], segments: Int) -> Mesh3 {
        var m = Mesh3()
        m.revolve(profile: p.map { .init(r: $0.r, y: $0.y) }, segments: segments)
        return m
    }

    /// 3-wide × 4-tall glyphs for the raised sidewall lettering.
    static let glyphs: [Character: [String]] = [
        "T": ["111", "010", "010", "010"], "I": ["111", "010", "010", "111"], "R": ["110", "101", "110", "101"],
        "E": ["111", "110", "100", "111"], "M": ["101", "111", "101", "101"], "A": ["010", "101", "111", "101"],
        "X": ["101", "010", "010", "101"], "O": ["111", "101", "101", "111"], " ": ["000", "000", "000", "000"]]

    /// 3-wide x 5-tall glyphs for the size designation on the sidewall.
    static let glyphs5: [Character: [String]] = [
        "P": ["110", "101", "110", "100", "100"], "R": ["110", "101", "110", "101", "101"], "/": ["001", "001", "010", "100", "100"],
        "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"], "2": ["110", "001", "010", "100", "111"],
        "3": ["110", "001", "010", "001", "110"], "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "110", "001", "110"],
        "6": ["011", "100", "110", "101", "010"], "7": ["111", "001", "010", "010", "010"], "8": ["010", "101", "010", "101", "010"],
        "9": ["010", "101", "011", "001", "110"]]

    /// Is lettering pixel (column `i`, row `row` 0…3) lit? Four columns per character (3 glyph + 1 gap).
    static func letterBit(_ text: [Character], column i: Int, row: Int) -> Bool {
        guard !text.isEmpty else { return false }
        let ch = text[(i / 4) % text.count], c = i % 4
        guard c < 3, let g = glyphs[ch] else { return false }
        return Array(g[row])[c] == "1"
    }
}
