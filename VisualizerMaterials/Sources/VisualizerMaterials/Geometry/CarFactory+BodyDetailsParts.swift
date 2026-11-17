import Foundation
import simd

// plausibility: real — mirrors, wipers and antennas are closed solids mounted on the real shell / glass surfaces.

extension DetailEnv {

    // MARK: 17 — mirrors (right-hand; caller mirrors)

    func mirrorPieces() -> [(String, DPiece)] {
        var out: [(String, DPiece)] = []
        func add(_ n: String, _ m: Mesh3, _ t: DTarget) { if !m.isEmpty { out.append((n, DPiece(name: n, mesh: m, target: t))) } }

        if kind == .bus || kind == .van {
            // large rectangular arm mirror, two-point bracket
            let z0 = kind == .bus ? c.frontZ - 0.30 : m.zF - 0.10
            guard let base = (kind == .bus ? flank(z0, 2.25) : (shoulder(z0, 1.1) ?? flank(z0, m.yB - 0.02))) else { return out }
            let k = kind == .bus ? 1.3 : 1.0
            let hx0 = base.p.x + 0.04, hx1 = base.p.x + 0.04 + (kind == .bus ? 0.17 : 0.145)
            let hh = (kind == .bus ? 0.40 : 0.30), hd = 0.045
            let yc = base.p.y + (kind == .bus ? 0.02 : 0.20)
            add("mirror.housing", carBevelBox(Vec3(hx0, yc - hh / 2, z0 - hd / 2), Vec3(hx1, yc + hh / 2, z0 + hd / 2), 0.012), .trim)
            add("mirror.face", carBevelBox(Vec3(hx0 + 0.010, yc - hh / 2 + 0.010, z0 - hd / 2 - 0.004), Vec3(hx1 - 0.010, yc + hh / 2 - 0.010, z0 - hd / 2 + 0.002), 0.0015), .chrome)
            var arm = Mesh3()
            arm.append(Mesh3.cylinder(from: base.p - base.n * 0.01 + Vec3(0, 0.07 * k, 0), to: Vec3(hx0 + 0.02, yc + hh * 0.30, z0), radius: 0.011, chordError: 0.0003))
            arm.append(Mesh3.cylinder(from: base.p - base.n * 0.01 - Vec3(0, 0.05 * k, 0), to: Vec3(hx0 + 0.02, yc - hh * 0.30, z0 - 0.004), radius: 0.011, chordError: 0.0003))
            add("mirror.arm", arm, .trim)
            return out
        }

        let z0 = m.zF - 0.17
        guard let base = shoulder(z0, 1.1) ?? flank(z0, m.yB - 0.02) else { return out }
        let b = base.p
        let k = big ? 1.12 : 1.0
        if classic {
            // small round chrome mirror on a bent stalk
            let cx = b.x + 0.15, cy = b.y + 0.15
            let stalk = Mesh3.tube(along: [b - base.n * 0.01, b + Vec3(0.045, 0.085, 0.0), Vec3(cx - 0.02, cy - 0.01, z0 - 0.02)], radius: 0.0075, segments: 10)
            add("mirror.stalk", stalk, .chrome)
            add("mirror.base", Mesh3.cylinder(from: b - base.n * 0.012, to: b + base.n * 0.008, radius: 0.022), .chrome)
            add("mirror.housing", Mesh3.cylinder(from: Vec3(cx, cy, z0 - 0.018), to: Vec3(cx, cy, z0 + 0.026), radius: 0.062).translated(by: .zero), .chrome)
            add("mirror.face", Mesh3.cylinder(from: Vec3(cx, cy, z0 - 0.027), to: Vec3(cx, cy, z0 - 0.0175), radius: 0.050), .chrome)
            return out
        }
        // modern / nineties: a teardrop (D-section, tapering to the tip) painted housing with a big recessed glass on
        // its flat rear face, on a flag base + stalk from the door.
        let hx0 = b.x + 0.03, len = 0.20 * k
        let hh = 0.11 * k, hd = 0.09 * k
        let yc = b.y + 0.085
        let (housing, glass) = Self.teardropMirror(x0: hx0, length: len, yc: yc, z0: z0, hh: hh, hd: hd)
        add("mirror.housing", housing, .body)
        add("mirror.face", glass, .chrome)
        // flag base: a plate on the door shoulder + the stalk into the housing
        add("mirror.base", Mesh3.cylinder(from: b - base.n * 0.012, to: b + base.n * 0.012, radius: 0.036 * k), .trim)
        add("mirror.stalk", Mesh3.tube(along: [b + base.n * 0.004, Vec3(hx0 + 0.02, yc - hh * 0.30, z0 + 0.010)], radius: 0.015 * k, segments: 12), .trim)
        return out
    }

    /// Mirror housing lofted along +x from a D-shaped section (flat rear face, rounded front) that swells and then
    /// closes toward the tip; plus the glass: a slab proud of the flat rear face covering most of it.
    static func teardropMirror(x0: Double, length: Double, yc: Double, z0: Double, hh: Double, hd: Double) -> (Mesh3, Mesh3) {
        func scale(_ t: Double) -> Double {      // 0 inner … 1 tip
            let t0 = 0.62
            let base = 0.82 + 0.18 * min(1, t / 0.3)
            if t <= t0 { return base }
            let u = (t - t0) / (1 - t0)
            return max(0.34, base * sqrt(max(0, 1 - u * u)))
        }
        let a = hd / 2
        // D section in (z, v): flat rear at z = −a, half-ellipse front.
        var sec: [(Double, Double)] = []
        let ch = 0.008
        sec += [(-a + ch, hh / 2), (0, hh / 2)]
        for i in 1..<10 { let th = Double.pi / 2 - Double.pi * Double(i) / 10; sec.append((a * cos(th), hh / 2 * sin(th))) }
        sec += [(0, -hh / 2), (-a + ch, -hh / 2), (-a, -hh / 2 + ch), (-a, hh / 2 - ch)]
        let n = 22
        var sections: [[Vec2]] = [], path: [Vec3] = []
        for i in 0...n {
            let t = Double(i) / Double(n), sc = scale(t)
            // loft frame for a +x path: section x → world z, section y → world −y
            sections.append(sec.map { Vec2(-a + ($0.0 + a) * sc, -$0.1 * sc) })   // rear plane stays at z = −a
            path.append(Vec3(x0 + length * t, yc, z0))
        }
        let housing = Mesh3.loft(sections: sections, along: path).smoothed(creaseDegrees: 55)
        // glass: follows the taper, 11 mm inside the rim, proud 3 mm of the flat rear plane
        var top: [Vec3] = [], bot: [Vec3] = []
        let m = 0.011
        let ts = (0...12).map { 0.02 + 0.80 * Double($0) / 12 }
        for t in ts {
            let h = max(0.01, hh * scale(t) / 2 - m)
            top.append(Vec3(x0 + length * t, yc + h, z0 - a - 0.003)); bot.append(Vec3(x0 + length * t, yc - h, z0 - a - 0.003))
        }
        let loopA = top + bot.reversed()
        let loopB = loopA.map { $0 + Vec3(0, 0, 0.004) }
        let glass = GHKit.prism(bottom: loopB, top: loopA)
        return (housing, glass)
    }

    // MARK: 19 — wipers

    func wiperPieces() -> [(String, DPiece)] {
        var out: [(String, DPiece)] = []
        guard kind != .bus else { return out }
        let ws = m.windscreenSurface
        let L = m.wsLen, dz = (m.z1 - m.zF) / L, dy = (m.roofY - m.yB) / L
        let slope = Vec3(0, dy, dz), n = normalize3(Vec3(0, -dz, dy))
        func frame(_ s: Double, x: Double = 0) -> (Mesh3) -> Mesh3 {
            let o = Vec3(x, m.yB + dy * s, m.zF + dz * s)
            return { $0.placed(xAxis: Vec3(1, 0, 0), yAxis: slope, zAxis: n, origin: o) }
        }
        _ = ws
        // windscreen pair, at rest along the base of the glass, blades pointing to the centre
        var blades = Mesh3(), arms = Mesh3()
        let xb = min(0.62, m.sideHW(m.zF + dz * 0.07, m.yB + dy * 0.07) - 0.12)
        let place = frame(0.075)
        for sgn in [-1.0, 1.0] {
            var bl = carBevelBox(Vec3(0.06, -0.011, 0.005), Vec3(xb - 0.12, 0.011, 0.020), 0.004)
            var ar = carBevelBox(Vec3(xb - 0.13, -0.006, 0.010), Vec3(xb + 0.02, 0.006, 0.0165), 0.0025)
            var pv = Mesh3.cylinder(from: Vec3(xb + 0.01, 0, 0.0), to: Vec3(xb + 0.01, 0, 0.026), radius: 0.011)
            if sgn < 0 { bl = bl.mirrored(.x); ar = ar.mirrored(.x); pv = pv.mirrored(.x) }
            blades.append(place(bl)); arms.append(place(ar)); arms.append(place(pv))
        }
        out.append(("wiper.blades", DPiece(name: "wiper.blades", mesh: blades, target: .trim)))
        out.append(("wiper.arms", DPiece(name: "wiper.arms", mesh: arms, target: .trim)))

        // rear wiper: hatch / SUV only (van has no rear glass here; pickup none)
        if [.hatch, .suv].contains(kind), m.hasRearGlass, !m.rearVertical {
            let RL = m.rearLen, rdz = (m.z2 - m.zR) / RL, rdy = (m.roofY - m.yR) / RL
            let rn = normalize3(Vec3(0, rdz, -rdy))
            let down = Vec3(0, -rdy, -rdz)                    // local +y = down the glass, so x × y = outward
            let o = Vec3(0, m.yR + rdy * 0.11, m.zR + rdz * 0.11)
            let bl = carBevelBox(Vec3(-0.15, -0.010, 0.005), Vec3(0.15, 0.010, 0.019), 0.004)
            let ar = carBevelBox(Vec3(-0.006, 0.0, 0.009), Vec3(0.006, 0.075, 0.0155), 0.0025)
            let pv = Mesh3.cylinder(from: Vec3(0, 0.075, 0), to: Vec3(0, 0.075, 0.024), radius: 0.012)
            let pl: (Mesh3) -> Mesh3 = { $0.placed(xAxis: Vec3(1, 0, 0), yAxis: down, zAxis: rn, origin: o) }
            out.append(("wiper.rear.blade", DPiece(name: "wiper.rear.blade", mesh: pl(bl), target: .trim)))
            var armPart = pl(ar); armPart.append(pl(pv))
            out.append(("wiper.rear.arm", DPiece(name: "wiper.rear.arm", mesh: armPart, target: .trim)))
        }
        return out
    }

    // MARK: 19 — antenna

    func antennaPieces() -> [(String, DPiece)] {
        var out: [(String, DPiece)] = []
        switch era {
        case .modern where [.sedan, .hatch, .suv].contains(kind):
            let zf = m.z2 + 0.32
            let y0 = m.roofTop(0, zf) - 0.005
            func lp(_ x: Double) -> [Vec3] {
                [Vec3(x, y0, zf + 0.085), Vec3(x, y0, zf - 0.085), Vec3(x, y0 + 0.068, zf - 0.035), Vec3(x, y0 + 0.066, zf + 0.020)]
            }
            let fin = GHKit.prism(bottom: lp(-0.016), top: lp(0.016)).beveled(radius: 0.007, creaseDegrees: 35, segments: 2)
            out.append(("antenna.fin", DPiece(name: "antenna.fin", mesh: fin, target: .trim)))
        case .classic1950s, .nineties:
            guard kind != .bus else { break }
            let rearFender = !classic && (kind == .sedan || kind == .coupe)   // only where the deck is low and clear of the cabin
            let z = rearFender ? c.rearZ + 0.60 : m.zF + 0.38
            let h = c.shellHalf(z)
            guard let s = down(h[4].x - 0.02, z) else { break }
            let len = classic ? 0.72 : 0.58
            var mast = Mesh3()
            mast.append(Mesh3.cylinder(from: s.p - Vec3(0, 0.006, 0), to: s.p + Vec3(0, 0.022, 0), radius: 0.014, chordError: 0.0003))
            let tip = s.p + Vec3(0, 0.02, 0) + normalize3(Vec3(0.05, 1, classic ? -0.18 : -0.12)) * len
            mast.append(Mesh3.tube(along: [s.p + Vec3(0, 0.018, 0), tip], radius: 0.0035, segments: 8))
            mast.append(Mesh3.cylinder(from: tip - Vec3(0, 0.004, 0), to: tip + Vec3(0, 0.010, 0), radius: 0.0065, chordError: 0.0003))
            out.append(("antenna.mast", DPiece(name: "antenna.mast", mesh: mast, target: .chrome)))
        default: break
        }
        return out
    }
}
