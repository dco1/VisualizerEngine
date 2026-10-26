import Foundation
import simd

// plausibility: real — sills, handles, shut lines and the fuel flap are closed solids laid on the ray-cast shell surface.

extension DetailEnv {

    // MARK: 15 — rocker / sill

    func sillPieces(_ lay: DoorLayout) -> [(String, DPiece)] {
        var out: [(String, DPiece)] = []
        let sill = c.bp.sill
        let z0 = archRearFront + 0.03, z1 = archFrontRear - 0.03
        guard z1 - z0 > 0.5 else { return out }
        let nz = Int(((z1 - z0) / 0.07).rounded(.up))
        func band(_ yLo: Double, _ yHi: Double, rows: Int = 3) -> [[DSurf]] {
            (0..<rows).compactMap { r -> [DSurf]? in
                let y = yLo + (yHi - yLo) * Double(r) / Double(rows - 1)
                let row = (0...nz).compactMap { flank(z0 + (z1 - z0) * Double($0) / Double(nz), y) }
                return row.count == nz + 1 ? row : nil
            }
        }
        // bottom rolled edge: a round-ish bead along the lower lip
        let edgeRow = (0...nz).compactMap { flank(z0 + (z1 - z0) * Double($0) / Double(nz), sill + 0.045) }
        let edge = DKit.strip(edgeRow, width: 0.02, embed: 0.004, proud: 0.010)
        out.append(("rolled", DPiece(name: "rolled", mesh: edge, target: classic ? .body : .trim)))
        if classic {
            // chrome rocker moulding
            let rows = band(sill + 0.075, sill + 0.108, rows: 2)
            out.append(("rocker.chrome", DPiece(name: "rocker.chrome", mesh: DKit.grid(rows, embed: 0.003, proud: 0.009), target: .chrome)))
        } else if !c.shellStyle.cladding {
            let rows = band(sill + 0.066, lay.yBot - 0.006)
            out.append(("rocker", DPiece(name: "rocker", mesh: DKit.grid(rows, embed: 0.004, proud: 0.012), target: .trim)))
        }
        return out
    }

    // MARK: 16 — door handles + pocket

    func handleLength() -> Double {
        switch era { case .classic1950s: return 0.085; case .nineties: return 0.12; case .modern: return 0.13 } }

    func handleZs(_ lay: DoorLayout) -> [(String, Double)] {
        let L = handleLength() * (big ? 1.2 : 1.0)
        switch kind {
        case .coupe: return [("front", lay.zRear + 0.12 + L / 2 + 0.05)]
        case .van:
            let zs = lay.zShut ?? lay.zFront - 1
            return [("front", zs + 0.10 + L / 2), ("slider", zs - 0.10 - L / 2)]
        default:
            let zs = lay.zShut ?? lay.zRear
            return [("front", zs + 0.10 + L / 2), ("rear", lay.zRear + 0.12 + L / 2)]
        }
    }

    func handlePieces(_ lay: DoorLayout) -> [(String, DPiece)] {
        var out: [(String, DPiece)] = []
        let k = big ? 1.2 : 1.0
        let L = handleLength() * k
        for (name, z) in handleZs(lay) {
            let y = creaseY(z) + (classic ? 0.036 : 0)   // 1950s: above the chrome spear
            guard var s = flank(z, y) else { continue }
            // pocket: dark pad around the handle
            let pl = L + 0.035, ph = classic ? 0.058 : 0.052
            var rows: [[DSurf]] = []
            for r in 0..<3 {
                let dy = -ph / 2 + ph * Double(r) / 2
                let row = (0..<7).compactMap { c -> DSurf? in flank(z - pl / 2 + pl * Double(c) / 6, y + dy) }
                if row.count == 7 { rows.append(row) }
            }
            // follow the surface: stand the handle on the highest point of its pocket (bold swages curve away)
            var lift = 0.0
            for row in rows { for q in row { lift = max(lift, dot3(q.p - s.p, s.n)) } }
            s.p += s.n * lift
            if rows.count == 3 {
                out.append(("pocket." + name, DPiece(name: "pocket." + name, mesh: DKit.grid(rows, embed: 0.004, proud: 0.0012), target: .trim)))
            }
            // the handle itself
            var h = Mesh3()
            switch era {
            case .classic1950s:
                h.append(carBevelBox(Vec3(-0.016, -L / 2, -0.004), Vec3(0.016, L / 2, 0.011), 0.005))
                h.append(Mesh3.cylinder(from: Vec3(0, -L / 2 + 0.018, 0.008), to: Vec3(0, -L / 2 + 0.018, 0.019), radius: 0.0095))
            case .nineties:
                h.append(carBevelBox(Vec3(-0.0095, -L / 2, 0.012), Vec3(0.0095, L / 2, 0.022), 0.004))
                for sgn in [-1.0, 1.0] { h.append(carBevelBox(Vec3(-0.008, sgn * (L / 2 - 0.016) - 0.008, -0.004), Vec3(0.008, sgn * (L / 2 - 0.016) + 0.008, 0.014), 0.003)) }
            case .modern:
                h.append(carBevelBox(Vec3(-0.011 * k, -L / 2, -0.004), Vec3(0.011 * k, L / 2, 0.012 * k), 0.004))
            }
            out.append(("handle." + name, DPiece(name: "handle." + name, mesh: DKit.onSurface(h, at: s), target: .chrome)))
        }
        return out
    }

    // MARK: 12 — hood, trunk, tailgate

    private func outline(_ poly: [Vec2], round r: Double, cast: (Vec2) -> DSurf?) -> [DSurf] {
        let loop = GHKit.densify(GHKit.round(poly, r), 0.05)
        return loop.compactMap(cast)
    }

    func hoodShutLine() -> Mesh3 {
        guard kind != .bus else { return Mesh3() }
        let zC = c.shellCowlZ + 0.14, zN = c.frontZ - 0.30
        guard zN - zC > 0.4 else { return Mesh3() }
        func xh(_ z: Double) -> Double { c.shellHalf(z)[3].x }
        let zM = (zN + zC) / 2
        let poly: [Vec2] = [Vec2(xh(zN), zN), Vec2(xh(zM), zM), Vec2(xh(zC), zC), Vec2(-xh(zC), zC), Vec2(-xh(zM), zM), Vec2(-xh(zN), zN)]
        let pts = outline(poly, round: 0.07) { down($0.x, $0.y) }
        return DKit.strip(pts, width: 0.005, embed: 0.002, proud: 0.0012, closed: true)
    }

    func trunkShutLine() -> Mesh3 {
        guard kind == .sedan || kind == .coupe else { return Mesh3() }
        let zL = m.zR - 0.07, zT = c.rearZ + 0.14
        guard zL - zT > 0.3 else { return Mesh3() }
        func xh(_ z: Double) -> Double { c.shellHalf(z)[3].x }
        let zM = (zL + zT) / 2
        let poly: [Vec2] = [Vec2(xh(zL), zL), Vec2(xh(zM), zM), Vec2(xh(zT), zT), Vec2(-xh(zT), zT), Vec2(-xh(zM), zM), Vec2(-xh(zL), zL)]
        let pts = outline(poly, round: 0.07) { down($0.x, $0.y) }
        return DKit.strip(pts, width: 0.005, embed: 0.002, proud: 0.0012, closed: true)
    }

    /// Hatch / SUV / van tailgate: a U on the rear face (sides + lower edge).
    func tailgateShutLine() -> Mesh3 {
        guard [.hatch, .suv, .van].contains(kind) else { return Mesh3() }
        let hw = c.W / 2 * c.planFactor(c.rearZ + 0.05)
        let xg = min(hw - 0.16, (lampInner ?? 9) - 0.03)
        let yLow = c.bp.endBottom + 0.22 + (kind == .van ? 0.1 : 0)
        let yTop = min(m.yR, c.shellDeck(c.rearZ + 0.05)) - 0.07
        guard yTop - yLow > 0.2 else { return Mesh3() }
        let r = 0.05
        var pts2: [Vec2] = []
        func vline(_ x: Double, _ y0: Double, _ y1: Double) { let n = max(1, Int((abs(y1 - y0) / 0.05).rounded(.up))); for i in 0...n { pts2.append(Vec2(x, y0 + (y1 - y0) * Double(i) / Double(n))) } }
        vline(-xg, yTop, yLow + r)
        for k in 1...5 { let a = Double(k) / 5 * .pi / 2; pts2.append(Vec2(-xg + r - r * cos(a), yLow + r - r * sin(a))) }
        let nh = max(2, Int(((2 * xg - 2 * r) / 0.06).rounded(.up)))
        for i in 1..<nh { pts2.append(Vec2(-xg + r + (2 * xg - 2 * r) * Double(i) / Double(nh), yLow)) }
        for k in 0...5 { let a = Double(k) / 5 * .pi / 2; pts2.append(Vec2(xg - r + r * sin(a), yLow + r - r * cos(a))) }
        vline(xg, yLow + r, yTop)
        let pts = pts2.compactMap { rear($0.x, $0.y) }
        return DKit.strip(pts, width: 0.005, embed: 0.002, proud: 0.0012)
    }

    /// 1950s: chrome hood ornament (rocket) on the centreline.
    func hoodOrnament() -> Mesh3 {
        guard kind != .bus, kind != .van else { return Mesh3() }
        let z0 = c.frontZ - 0.30
        guard let s = down(0, z0) else { return Mesh3() }
        var m = Mesh3()
        m.append(Mesh3.cylinder(from: s.p - Vec3(0, 0.004, 0), to: s.p + Vec3(0, 0.012, 0), radius: 0.020))
        let base = s.p + Vec3(0, 0.012, 0)
        let path: [Vec3] = [base + Vec3(0, 0.002, -0.050), base + Vec3(0, 0.020, -0.012), base + Vec3(0, 0.030, 0.040), base + Vec3(0, 0.026, 0.100), base + Vec3(0, 0.021, 0.150)]
        m.append(Mesh3.tube(along: path, radius: 0.020, segments: 12, radiusScale: [0.35, 0.95, 1.0, 0.7, 0.22]))
        for sgn in [-1.0, 1.0] {
            // wing as a flat blade out to the side and back
            var q = Mesh3()
            let a = base + Vec3(sgn * 0.006, 0.026, 0.020), b = base + Vec3(sgn * 0.075, 0.020, -0.045), c2 = base + Vec3(sgn * 0.075, 0.020, -0.030), d = base + Vec3(sgn * 0.006, 0.026, 0.060)
            let th = Vec3(0, 0.004, 0)
            let top = [a + th, b + th, c2 + th, d + th], bot = [a - th, b - th, c2 - th, d - th]
            q.addQuad(top[0], top[1], top[2], top[3], outward: Vec3(0, 1, 0))
            q.addQuad(bot[0], bot[1], bot[2], bot[3], outward: Vec3(0, -1, 0))
            for i in 0..<4 { let j = (i + 1) % 4
                let mid = (top[i] + top[j] + bot[i] + bot[j]) / 4, cen = (top[0] + top[1] + top[2] + top[3] + bot[0] + bot[1] + bot[2] + bot[3]) / 8
                q.addQuad(top[i], top[j], bot[j], bot[i], outward: mid - cen) }
            m.append(q)
        }
        return m
    }

    /// 1950s: chrome side spear along the character line, right side.
    func sideSpear(_ lay: DoorLayout) -> Mesh3 {
        guard kind != .bus else { return Mesh3() }
        var pts: [DSurf] = []
        let z0 = archFrontRear - 0.05, z1 = c.rearZ + 0.28
        let n = Int(((z0 - z1) / 0.08).rounded(.up))
        for i in 0...n {
            let z = z0 - (z0 - z1) * Double(i) / Double(n)
            let h = c.shellHalf(z)
            if let s = flank(z, (h[8].y + h[9].y) / 2 - 0.015) { pts.append(s) }
        }
        return DKit.strip(pts, width: 0.022, embed: 0.002, proud: 0.007)
    }

    // MARK: 13 — fuel flap (right-hand geometry; the caller mirrors it to the left side)

    struct Flap { var groove = Mesh3(), hinge = Mesh3(), cap = Mesh3() }

    func fuelFlap() -> Flap {
        var f = Flap()
        guard kind != .bus else { return f }
        let zc2 = max(c.axles[1] - openingR - 0.10 - 0.085, c.rearZ + 0.38)
        let yc = flankTop(zc2) - 0.15
        guard flank(zc2, yc) != nil else { return f }
        let hu = 0.085, hv = 0.07      // 0.17 × 0.14 outline
        func loop(_ hu: Double, _ hv: Double, _ r: Double) -> [DSurf] {
            DKit.roundedRect(cu: 0, cv: 0, hu: hu, hv: hv, r: r, per: 5).compactMap { flank(zc2 + $0.x, yc + $0.y) }
        }
        f.groove = DKit.strip(loop(hu, hv, 0.028), width: 0.004, embed: 0.002, proud: 0.0011, closed: true)
        f.cap = DKit.strip(loop(hu - 0.026, hv - 0.026, 0.022), width: 0.003, embed: 0.002, proud: 0.0009, closed: true)
        // hinge: two small knuckles on the front (+z) edge
        for dy in [-0.03, 0.03] {
            if let s = flank(zc2 + hu, yc + dy) {
                f.hinge.append(Mesh3.cylinder(from: s.p - s.n * 0.002 - Vec3(0, 0.009, 0), to: s.p - s.n * 0.002 + Vec3(0, 0.009, 0), radius: 0.0045).translated(by: s.n * 0.004))
            }
        }
        // finger notch on the rear edge
        if let s = flank(zc2 - hu + 0.012, yc) {
            f.hinge.append(DKit.onSurface(carBevelBox(Vec3(-0.012, -0.007, -0.003), Vec3(0.012, 0.007, 0.0016), 0.0015), at: s))
        }
        return f
    }
}
