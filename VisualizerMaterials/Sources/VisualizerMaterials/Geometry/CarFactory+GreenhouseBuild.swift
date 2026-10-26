import Foundation
import simd

// plausibility: real — panes, pillars, frames, roof crown and drip rails are authored closed solids off the cabin surface.

extension CarContext {

    func greenhousePieces(_ m: GHModel) -> GHPieces {
        var g = GHPieces()
        let classic = era == .classic1950s
        let side = m.sideSurface
        let yB = m.yB, yPane = m.yB + m.sillH
        let thick = 0.008, fw = m.frameW

        func dens(_ p: [Vec2]) -> [Vec2] { GHKit.densify(GHKit.ccw(p), 0.11) }

        func addSidePane(_ name: String, _ poly0: [Vec2]) {
            guard m.polyArea(poly0) > 0.008 else { return }
            let poly = dens(poly0)
            let glass = side.prism(poly, inward: thick, outward: 0)
            let frame = side.frame(poly, width: fw, overlap: 0.004, inward: 0.014, outward: 0.005)
            g.panes.append(GHNamed(name: name + ".R", mesh: glass)); g.panes.append(GHNamed(name: name + ".L", mesh: glass.mirrored(.x)))
            g.frames.append(GHNamed(name: name + ".R", mesh: frame)); g.frames.append(GHNamed(name: name + ".L", mesh: frame.mirrored(.x)))
            if classic {
                let ch = side.frame(GHKit.offset(poly, fw), width: 0.009, overlap: 0, inward: 0.012, outward: 0.007)
                g.chromeFrames.append(GHNamed(name: name + ".R", mesh: ch)); g.chromeFrames.append(GHNamed(name: name + ".L", mesh: ch.mirrored(.x)))
            }
        }
        func pillarPoly(_ f: GHLine?, _ r: GHLine?) -> [Vec2] { m.region(front: f, rear: r, yLo: yB - 0.01, drip: 0, underCamber: true, round: 0) }
        func addPillar(_ name: String, _ poly0: [Vec2], trim: Bool = false, cap: Bool = false, strip: Bool = false) {
            guard m.polyArea(poly0) > 0.002 else { return }
            let poly = dens(poly0)
            let mesh = side.prism(poly, inward: 0.075, outward: 0.002)
            if trim { g.pillarsTrim.append(GHNamed(name: name + ".R", mesh: mesh)); g.pillarsTrim.append(GHNamed(name: name + ".L", mesh: mesh.mirrored(.x))) }
            else { g.pillarsBody.append(GHNamed(name: name + ".R", mesh: mesh)); g.pillarsBody.append(GHNamed(name: name + ".L", mesh: mesh.mirrored(.x))) }
            if cap || strip {
                let inset = GHKit.offset(poly, -(strip ? 0.014 : min(0.014, 0.28 * m.wB)))
                if m.polyArea(inset) > 0.0005, GHKit.area(inset) > 0 {
                    let cm = side.prism(inset, inward: 0.01, outward: 0.007)
                    if cap { g.pillarCaps.append(GHNamed(name: name + ".R", mesh: cm)); g.pillarCaps.append(GHNamed(name: name + ".L", mesh: cm.mirrored(.x))) }
                    else { g.pillarsTrim.append(GHNamed(name: name + "-strip.R", mesh: cm)); g.pillarsTrim.append(GHNamed(name: name + "-strip.L", mesh: cm.mirrored(.x))) }
                }
            }
        }

        // ── windscreen (centre) ───────────────────────────────────────────────
        do {
            let ws = m.windscreenSurface, L = m.wsLen
            let s0 = 0.02, s1 = L * (1 - 0.02 / max(m.roofY - yB, 0.2))
            let dz = (m.z1 - m.zF) / L, dy = (m.roofY - yB) / L
            let inset = classic ? 0.016 : 0.045
            func xe(_ s: Double) -> Double { m.sideHW(m.zF + dz * s, yB + dy * s) - inset }
            let poly = GHKit.round([Vec2(-xe(s0), s0), Vec2(xe(s0), s0), Vec2(xe(s1), s1), Vec2(-xe(s1), s1)], 0.035)
            let p = GHKit.densify(poly, 0.2)
            g.panes.append(GHNamed(name: "windscreen", mesh: ws.prism(p, inward: thick, outward: 0)))
            g.frames.append(GHNamed(name: "windscreen", mesh: ws.frame(p, width: fw, overlap: 0.004, inward: 0.014, outward: 0.005)))
            if classic { g.chromeFrames.append(GHNamed(name: "windscreen", mesh: ws.frame(GHKit.offset(p, fw), width: 0.009, overlap: 0, inward: 0.012, outward: 0.007))) }
        }

        // ── side glass + pillars ──────────────────────────────────────────────
        let lineAr = m.lineA.shifted(-m.wA)
        addPillar("A", pillarPoly(m.lineA, lineAr), cap: classic)
        let C = m.lineC, Cf = m.lineCf
        switch m.kind {
        case .sedan, .hatch, .suv, .pickup:
            let doorEnd1 = m.lineBf
            addSidePane("door-front", m.region(front: lineAr, rear: doorEnd1, yLo: yPane, drip: 0.04))
            let zRD = m.zRD
            var quarter: [Vec2] = []
            if m.hasQuarterWanted {
                quarter = m.region(front: m.vline(zRD - m.wDiv), rear: Cf, yLo: yPane, drip: 0.04)
                if m.polyArea(quarter) < 0.012 { quarter = [] }
            }
            let rearEnd = m.hasQuarterWanted ? m.vline(zRD) : Cf
            addSidePane("door-rear", m.region(front: m.lineBr, rear: rearEnd, yLo: yPane, drip: 0.04))
            if era == .modern { addPillar("B", pillarPoly(m.lineBf, m.lineBr), trim: true) }
            else if era == .nineties { addPillar("B", pillarPoly(m.lineBf, m.lineBr), strip: true) }
            else { addPillar("B", pillarPoly(m.lineBf, m.lineBr), cap: true) }
            if m.hasQuarterWanted {
                if quarter.isEmpty { addPillar("C", pillarPoly(m.vline(zRD), C), cap: classic) }
                else {
                    addSidePane("quarter", quarter)
                    addPillar("div", pillarPoly(m.vline(zRD), m.vline(zRD - m.wDiv)), cap: classic)
                    addPillar("C", pillarPoly(Cf, C), cap: classic)
                }
            } else { addPillar("C", pillarPoly(Cf, C), cap: classic) }
        case .coupe:
            let end = m.vline(m.zDoorEndCoupe)
            addSidePane("door-front", m.region(front: lineAr, rear: end, yLo: yPane, drip: 0.04))
            let q = m.region(front: m.vline(m.zDoorEndCoupe - m.wDiv), rear: Cf, yLo: yPane, drip: 0.04)
            addPillar("div", pillarPoly(end, m.vline(m.zDoorEndCoupe - m.wDiv)), cap: classic)
            if m.polyArea(q) >= 0.012 { addSidePane("quarter", q); addPillar("C", pillarPoly(Cf, C), cap: classic) }
            else { addPillar("C", pillarPoly(end, C), cap: classic) }
        case .van:
            let end = m.vline(m.zR + 0.10)
            addSidePane("door-front", m.region(front: lineAr, rear: end, yLo: yPane, drip: 0.04))
            addPillar("C", pillarPoly(end, C), cap: classic)
        case .bus: break
        }

        // ── rear / tailgate glass (centre) ───────────────────────────────────
        if m.hasRearGlass && !m.rearVertical {
            let rs = m.rearSurface, L = m.rearLen
            let s0 = 0.02 + (classic ? 0.03 : 0.0), s1 = L * (1 - 0.02 / max(m.roofY - m.yR, 0.2))
            let dz = (m.z2 - m.zR) / L, dy = (m.roofY - m.yR) / L
            let inset = classic ? 0.016 : 0.045
            func xe(_ s: Double) -> Double { m.sideHW(m.zR + dz * s, m.yR + dy * s) - inset }
            let poly = GHKit.round([Vec2(-xe(s0), s0), Vec2(xe(s0), s0), Vec2(xe(s1), s1), Vec2(-xe(s1), s1)], 0.04)
            let p = GHKit.densify(poly, 0.2)
            g.panes.append(GHNamed(name: "rear", mesh: rs.prism(p, inward: thick, outward: 0)))
            g.frames.append(GHNamed(name: "rear", mesh: rs.frame(p, width: fw, overlap: 0.004, inward: 0.014, outward: 0.005)))
            if classic { g.chromeFrames.append(GHNamed(name: "rear", mesh: rs.frame(GHKit.offset(p, fw), width: 0.009, overlap: 0, inward: 0.012, outward: 0.007))) }
        }

        // ── window-sill chrome strip (beltline) ──────────────────────────────
        do {
            let zA = lineAr.z(yB) + 0.02
            let zE = m.kind == .van ? m.zR + 0.10 : max(Cf.z(yB), m.zR + 0.05)
            let poly = GHKit.densify([Vec2(zE, yB), Vec2(zA, yB), Vec2(zA, yB + m.sillH), Vec2(zE, yB + m.sillH)], 0.08)
            let r = side.prism(poly, inward: 0.004, outward: classic ? 0.011 : 0.008)
            g.sill = r; g.sill.append(r.mirrored(.x))
        }

        // ── roof panel (crowned) + drip rail ─────────────────────────────────
        do {
            var (za, zb) = m.roofRange
            if m.kind == .van { za -= 0.03 }   // bury the slab's rear end in the cargo box: no coplanar seam with the step wall
            _ = zb; zb = m.roofRange.1
            var zs: [Double] = [za, zb]
            var z = za + 0.15
            while z < zb - 0.02 { zs.append(z); z += 0.15 }
            for k in [m.z2, m.z1] where k > za + 0.01 && k < zb - 0.01 { zs.append(k) }
            zs.sort()
            var u: [Double] = []
            for v in zs where u.isEmpty || v - u.last! > 0.01 { u.append(v) }
            // The header lips (z beyond the flat roof, over the windscreen / rear glass) thin out to 16 mm so the slab's
            // end face stays ABOVE the glass plane — a clean painted edge, nothing of it seen through the glass.
            let zf1 = m.z1, zf2 = m.z2, rv = m.rearVertical
            g.roof = GHKit.slab(zs: u, hw: { m.hwRoofPanel($0) }, nx: 10, thickness: 0.055,
                                bottom: { x, z in
                let dz = z > zf1 ? z - zf1 : ((z < zf2 && !rv) ? zf2 - z : 0)
                return m.roofTop(x, z) - (0.055 - 0.039 * min(1, dz / 0.03))
            }, top: { m.roofTop($0, $1) }).smoothed(creaseDegrees: 25)

            // drip rail
            func edge(_ z: Double) -> Double {
                var y = Double.infinity
                for s in m.segs { y = min(y, s.y0 + s.m * (z - s.z0) - (s.flat ? m.crown + 0.004 : 0)) }
                return y
            }
            let zlo = m.rearVertical ? m.zR + 0.06 : m.zR + 0.14, zhi = m.z1 + 0.03
            var rz: [Double] = [zlo, zhi]
            var q = zlo + 0.12
            while q < zhi - 0.03 { rz.append(q); q += 0.12 }
            for k in [m.z2, m.z1] where k > zlo + 0.03 && k < zhi - 0.03 { rz.append(k) }
            rz.sort()
            var rzu: [Double] = []
            for v in rz where rzu.isEmpty || v - rzu.last! > 0.02 { rzu.append(v) }
            let path: [Vec3] = rzu.map { z in let y = edge(z) - 0.022; return Vec3(m.sideHW(z, y) + 0.006, y, z) }
            let prof = [Vec2(-0.009, -0.015), Vec2(0.009, -0.015), Vec2(0.009, 0.015), Vec2(-0.009, 0.015)]
            let rail = Mesh3.sweep(profile: prof, along: path)
            g.dripRail = rail; g.dripRail.append(rail.mirrored(.x))

            // sunroof
            if m.hasSunroof {
                let flat = m.z1 - m.z2
                let sz1 = m.z1 - 0.12, sz0 = sz1 - min(0.62, 0.5 * flat)
                let hwS = 0.36, fwid = 0.028
                func zsFor(_ a: Double, _ b: Double) -> [Double] { let n = max(2, Int(((b - a) / 0.15).rounded(.up))); return (0...n).map { a + (b - a) * Double($0) / Double(n) } }
                g.sunroofGlass = GHKit.slab(zs: zsFor(sz0, sz1), hw: { _ in hwS }, nx: 6, thickness: 0.010) { x, z in m.roofTop(x, z) + 0.005 }
                var fr = Mesh3()
                let top: (Double, Double) -> Double = { x, z in m.roofTop(x, z) + 0.009 }
                fr.append(GHKit.slab(zs: zsFor(sz1 + 0.001, sz1 + fwid), hw: { _ in hwS + fwid }, nx: 6, thickness: 0.016, top: top))
                fr.append(GHKit.slab(zs: zsFor(sz0 - fwid, sz0 - 0.001), hw: { _ in hwS + fwid }, nx: 6, thickness: 0.016, top: top))
                for sgn in [-1.0, 1.0] {
                    fr.append(GHKit.slab(zs: zsFor(sz0, sz1), cx: sgn * (hwS + fwid / 2), hw: { _ in fwid / 2 }, nx: 1, thickness: 0.016, top: top))
                }
                g.sunroofFrame = fr
            }
        }
        return g
    }
}
