import Foundation
import simd

// plausibility: real — class-specific features authored at real positions

extension CarContext {
    /// Class-specific cues: pickup bed, roof rails, taxi livery + roof sign, bus doors/sign, van doors, per-class rear.
    /// Items 45–50 of the car-detail brief. Every position is read off the finished shell (`CCProbe`), not guessed.
    func buildClassCues(_ a: inout CarAccumulator) {
        let cls = ccClass
        let bodyProbe = CCProbe(a.body)
        let roofProbe = CCProbe(a.body, a.glass)

        if ccIsTaxi { buildTaxiCues(&a, body: bodyProbe, roof: roofProbe) }
        switch cls {
        case .pickup:
            buildPickupBed(&a)
            buildRearFaceCues(&a, body: CCProbe(a.body), kind: .pickup)
        case .suv:
            buildRoofRails(&a, roof: roofProbe, crossbars: true)
            buildRearFaceCues(&a, body: bodyProbe, kind: .suv)
            buildRoofSpoiler(&a, roof: roofProbe, big: true)
            buildBumperStep(&a)
        case .hatch:
            buildRoofRails(&a, roof: roofProbe, crossbars: false)
            buildRearFaceCues(&a, body: bodyProbe, kind: .hatch)
            buildRoofSpoiler(&a, roof: roofProbe, big: false)
        case .van:
            buildVanCues(&a, body: bodyProbe)
        case .bus:
            buildBusCues(&a)
        case .sedan:
            buildDeckLidCues(&a, body: bodyProbe, coupe: false)
        case .coupe:
            buildDeckLidCues(&a, body: bodyProbe, coupe: true)
        }
    }

    // MARK: - 47 · taxi livery

    /// Roof sign (lofted wedge housing in `roofSign`, lit TAXI panes in `signals`, glyphs and feet in `trim`), the
    /// checker band along both flanks (black squares in `trim`; the yellow is the paint underneath) and the
    /// door-number plaque on both front doors (`plates` + `trim` frame).
    func buildTaxiCues(_ a: inout CarAccumulator, body: CCProbe, roof: CCProbe) {
        let cp = cabinPoly
        guard cp.count >= 4 else { return }
        let zc = (cp[1].0 + cp[2].0) / 2

        // ── roof sign ──
        let hx = classic ? 0.27 : 0.30, hz = classic ? 0.125 : 0.11
        let sh = classic ? 0.165 : (era == .nineties ? 0.150 : 0.135)
        let inset = classic ? 0.015 : 0.040
        var roofMax = 0.0
        for x in stride(from: -hx, through: hx, by: 0.05) { for dz in [-hz, 0, hz] { roofMax = max(roofMax, roof.top(x, zc + dz) ?? 0) } }
        let base = roofMax + 0.022
        let hmid = sh / 2
        let poly: [Vec2] = [Vec2(-hz, -hmid), Vec2(hz, -hmid), Vec2(hz, sh - 0.065 - hmid), Vec2(hz - inset, hmid), Vec2(-hz + inset, hmid), Vec2(-hz, sh - 0.065 - hmid)]
        let ring = GHKit.round(poly, 0.018)
        // sweep along +x: section (px, py) → world (z = px, y = -py)
        let prof = ring.map { Vec2($0.x, -$0.y) }
        var xs: [Double] = []
        for e in [0.0, 0.010, 0.022, 0.036, 0.050] { xs.append(-hx + e) }
        xs.append(0)
        for e in [0.050, 0.036, 0.022, 0.010, 0.0] { xs.append(hx - e) }
        let sc: [Double] = xs.map { x in
            let e = hx - abs(x)
            return e >= 0.05 ? 1 : max(0.55, (1 - (1 - e / 0.05) * (1 - e / 0.05)).squareRoot())
        }
        a.roofSign = Mesh3.sweep(profile: prof, along: xs.map { Vec3($0, base + hmid, zc) }, scale: sc)
        // feet
        for sx in [-0.20, 0.20] {
            let rt = roof.top(sx, zc) ?? roofMax
            a.trim.append(carBevelBox(Vec3(sx - 0.03, rt - 0.012, zc - 0.045), Vec3(sx + 0.03, base + 0.006, zc + 0.045), 0.006))
        }
        // lit TAXI panes + glyphs on both faces (front authored, rear is its mirror in z)
        let py0 = base + 0.024, py1 = py0 + 0.050
        let pw = hx - 0.075
        var front = Mesh3()
        front.append(carBox(Vec3(-pw, py0, zc + hz - 0.005), Vec3(pw, py1, zc + hz + 0.0025)))
        var glyphs = Mesh3()
        let gh = 0.034, gw = 0.026, gt = 0.007, pitch = 0.040
        let nG = 4, total = Double(nG - 1) * pitch + gw
        for (i, ch) in ["T", "A", "X", "I"].enumerated() {
            let u0 = -total / 2 + Double(i) * pitch         // viewer-right coordinate; world x = -u on the +z face
            for s in ccGlyph(ch, w: gw, h: gh, t: gt) {
                let bot = s.map { Vec3((u0 + $0.x), py0 + 0.008 + $0.y, zc + hz + 0.0025) }
                let top = s.map { Vec3((u0 + $0.x), py0 + 0.008 + $0.y, zc + hz + 0.0040) }
                glyphs.append(GHKit.prism(bottom: bot, top: top))
            }
        }
        a.signals.append(front); a.signals.append(front.mirrored(.z, at: zc))
        a.trim.append(glyphs); a.trim.append(glyphs.mirrored(.z, at: zc))

        // ── checker band + door number ──
        let yb = bp.belt * rH
        let sq = 0.07, gapEdge = 0.0005
        let zHi = frontZ - 0.80, zLo = rearZ + 0.70
        let nCols = Int((zHi - zLo) / sq)
        var checks = Mesh3()
        let rowY = [yb - 0.280, yb - 0.210]          // the flat door band, below the shoulder crease and clear of the arches
        for col in 0..<nCols {
            let z = zHi - (Double(col) + 0.5) * sq
            let row = col % 2
            if let m = ccStamp(body, from: Vec3(0, rowY[row], z), dir: Vec3(1, 0, 0), w: sq - gapEdge, h: sq - gapEdge,
                               back: 0.004, proud: 0.0015, maxDev: 0.003) {
                checks.append(m)
            }
        }
        a.trim.append(checks); a.trim.append(checks.mirrored(.x))

        let zp = cp[3].0 - 0.46, yp = yb - 0.36
        if let frame = ccStamp(body, from: Vec3(0, yp, zp), dir: Vec3(1, 0, 0), w: 0.178, h: 0.098, back: 0.004, proud: 0.0012, maxDev: 0.02),
           let plate = ccStamp(body, from: Vec3(0, yp, zp), dir: Vec3(1, 0, 0), w: 0.160, h: 0.080, back: 0.004, proud: 0.0028, maxDev: 0.02) {
            a.trim.append(frame); a.trim.append(frame.mirrored(.x))
            a.plates.append(plate); a.plates.append(plate.mirrored(.x))
        }
    }

    /// Block-letter strokes as convex (u, v) quads for T, A, X, I (cell `w` × `h`, stroke `t`).
    func ccGlyph(_ ch: String, w: Double, h: Double, t: Double) -> [[Vec2]] {
        func r(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> [Vec2] { [Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y1), Vec2(x0, y1)] }
        switch ch {
        case "T": return [r(0, h - t, w, h), r(w / 2 - t / 2, 0, w / 2 + t / 2, h - t - 0.0005)]
        case "I": return [r(w / 2 - t / 2, 0, w / 2 + t / 2, h)]
        case "A":
            let k = t * 1.1
            return [[Vec2(0, 0), Vec2(k, 0), Vec2(w / 2 + k / 2, h), Vec2(w / 2 - k / 2, h)],
                    [Vec2(w - k, 0), Vec2(w, 0), Vec2(w / 2 + k / 2 + 0.0012, h - 0.0010), Vec2(w / 2 - k / 2 + 0.0012, h - 0.0010)],
                    r(w * 0.30, h * 0.26, w * 0.70, h * 0.26 + t * 0.8)]
        default: // X
            let k = t * 1.1
            return [[Vec2(0, 0), Vec2(k, 0), Vec2(w, h), Vec2(w - k, h)],
                    [Vec2(w - k, 0), Vec2(w, 0), Vec2(k, h), Vec2(0, h)]]
        }
    }

    // MARK: - 46 · roof rails

    /// Two longitudinal rails (and, on the SUV, two crossbars) standing on stanchion feet and following the roof crown.
    func buildRoofRails(_ a: inout CarAccumulator, roof: CCProbe, crossbars: Bool) {
        guard !classic else { return }
        let cp = cabinPoly
        guard cp.count >= 4 else { return }
        let zRear = cp[1].0 + 0.10, zFront = cp[2].0 - 0.12
        guard zFront - zRear > 0.6 else { return }
        let zm = (zRear + zFront) / 2
        let y0 = roof.top(0, zm) ?? bp.height
        var xe = 0.3
        while xe < W / 2, (roof.top(xe + 0.01, zm) ?? 0) > y0 - 0.045 { xe += 0.01 }
        let xr = max(0.35, xe - 0.10)
        let gap = 0.030, bh = 0.026, bw = 0.032
        var rail = Mesh3(), feet = Mesh3()
        let n = max(6, Int((zFront - zRear) / 0.08))
        for sx in [1.0, -1.0] {
            var path: [Vec3] = []
            for i in 0...n {
                let z = zRear + (zFront - zRear) * Double(i) / Double(n)
                path.append(Vec3(sx * xr, (roof.top(sx * xr, z) ?? y0) + gap + bh / 2, z))
            }
            rail.append(ccBar(path, w: bw, h: bh, corner: 0.010))
            for f in [0.0, 0.34, 0.68, 1.0] {
                let z = zRear + (zFront - zRear) * f
                let rt = roof.top(sx * xr, z) ?? y0
                feet.append(carBevelBox(Vec3(sx * xr - 0.014, rt - 0.008, z - 0.022), Vec3(sx * xr + 0.014, rt + gap + 0.004, z + 0.022), 0.005))
            }
        }
        if crossbars {
            for f in [0.30, 0.72] {
                let z = zRear + (zFront - zRear) * f
                let railTop = (roof.top(xr, z) ?? y0) + gap + bh
                var path: [Vec3] = []
                for i in 0...6 { let x = -xr + 2 * xr * Double(i) / 6; path.append(Vec3(x, railTop + 0.012, z)) }
                // lifted so the bar clears the crown and rests on the rails' tops
                let top = path.map(\.y).max() ?? railTop
                rail.append(ccBar(path.map { Vec3($0.x, top, $0.z) }, w: 0.040, h: 0.024, corner: 0.010))
                for sx in [1.0, -1.0] {
                    feet.append(carBevelBox(Vec3(sx * xr - 0.020, railTop - 0.004, z - 0.030), Vec3(sx * xr + 0.020, top - 0.010, z + 0.030), 0.005))
                }
            }
        }
        if era == .nineties { a.chrome.append(rail); a.chrome.append(feet) } else { a.trim.append(rail); a.trim.append(feet) }
    }

    // MARK: - 50 · per-class rear

    enum RearKind { case pickup, suv, hatch }

    /// Shut lines (4 mm slots), a handle recess + handle and a badge on the tailgate face, read off the rear surface.
    func buildRearFaceCues(_ a: inout CarAccumulator, body: CCProbe, kind: RearKind) {
        let zIn = rearZ + 0.6
        let xg: Double, yLo: Double, yHi: Double
        switch kind {
        case .pickup: xg = 0.905; yLo = 0.64; yHi = (body.top(0, rearZ + 0.04) ?? 1.1) - 0.03
        case .suv:    xg = 0.80; yLo = 0.64; yHi = (CCProbe(a.body, a.glass).top(0, rearZ + 0.33) ?? 1.6) - 0.02
        case .hatch:  xg = 0.74; yLo = 0.66; yHi = (CCProbe(a.body, a.glass).top(0, rearZ + 0.30) ?? 1.3) - 0.01
        }
        guard yHi > yLo + 0.2 else { return }
        var lines = Mesh3()
        func vline(_ x: Double, _ ya: Double, _ yb: Double, w: Double = 0.004) {
            let n = max(1, Int(((yb - ya) / 0.10).rounded(.up)))
            for i in 0..<n {
                let y0 = ya + (yb - ya) * Double(i) / Double(n), y1 = ya + (yb - ya) * Double(i + 1) / Double(n)
                if let m = ccStamp(body, from: Vec3(x, (y0 + y1) / 2, zIn), dir: Vec3(0, 0, -1), w: w, h: y1 - y0 + 0.002, back: 0.004, proud: 0.0012, maxDev: 0.03) { lines.append(m) }
            }
        }
        func hline(_ y: Double, _ xa: Double, _ xb: Double, w: Double = 0.004) {
            let n = max(1, Int(((xb - xa) / 0.10).rounded(.up)))
            for i in 0..<n {
                let x0 = xa + (xb - xa) * Double(i) / Double(n), x1 = xa + (xb - xa) * Double(i + 1) / Double(n)
                if let m = ccStamp(body, from: Vec3((x0 + x1) / 2, y, zIn), dir: Vec3(0, 0, -1), up: Vec3(1, 0, 0), w: w, h: x1 - x0 + 0.002, back: 0.004, proud: 0.0012, maxDev: 0.03) { lines.append(m) }
            }
        }
        vline(xg, yLo, yHi); vline(-xg, yLo, yHi)
        hline(yLo, -xg, xg)
        // hinge knuckles on the lower corners (pickup: the drop-gate hinges)
        if kind == .pickup {
            for sx in [-1.0, 1.0] {
                if let m = ccStamp(body, from: Vec3(sx * (xg - 0.05), yLo, zIn), dir: Vec3(0, 0, -1), w: 0.07, h: 0.026, back: 0.004, proud: 0.004, maxDev: 0.03) { lines.append(m) }
            }
        }
        a.trim.append(lines)
        // handle recess (dark pocket) + handle bar
        let yh = yHi - (kind == .pickup ? 0.075 : 0.10)
        if kind == .pickup || kind == .suv {
            if let pocket = ccStamp(body, from: Vec3(0, yh, zIn), dir: Vec3(0, 0, -1), w: 0.20, h: 0.058, back: 0.004, proud: 0.0012, maxDev: 0.03) { a.trim.append(pocket) }
            if let bar = ccStamp(body, from: Vec3(0, yh, zIn), dir: Vec3(0, 0, -1), w: 0.15, h: 0.014, back: 0.004, proud: 0.0045, maxDev: 0.03) { a.chrome.append(bar) }
        } else {
            if let pocket = ccStamp(body, from: Vec3(0, yh, zIn), dir: Vec3(0, 0, -1), w: 0.16, h: 0.034, back: 0.004, proud: 0.0012, maxDev: 0.03) { a.trim.append(pocket) }
        }
        // badge
        if let badge = ccStamp(body, from: Vec3(-0.30, yh, zIn), dir: Vec3(0, 0, -1), w: 0.11, h: 0.026, back: 0.004, proud: 0.0030, maxDev: 0.03) { a.chrome.append(badge) }
    }

    /// Roof-edge spoiler (body colour); the third brake light below it comes from the Lamps pass.
    func buildRoofSpoiler(_ a: inout CarAccumulator, roof: CCProbe, big: Bool) {
        let cp = cabinPoly
        guard cp.count >= 4 else { return }
        let zEdge = cp[1].0                      // rear edge of the flat roof
        let hw = big ? 0.64 : 0.60
        var yMin = 10.0
        for x in stride(from: -hw, through: hw, by: 0.1) { yMin = min(yMin, roof.top(x, zEdge + 0.03) ?? 10) }
        guard yMin < 9 else { return }
        let y0 = yMin - 0.006
        let depth = big ? 0.16 : 0.20, hgt = big ? 0.056 : 0.062
        let zf = zEdge + 0.06, zb = zEdge - depth
        let poly: [Vec2] = [Vec2(zf, y0 + 0.006), Vec2(zb + 0.02, y0 - 0.004), Vec2(zb, y0 + 0.004), Vec2(zb, y0 + hgt), Vec2(zf - 0.04, y0 + hgt * 0.75)]
        a.body.append(ccSidePrism(poly, x0: -hw, x1: hw))
        // (no stop-light insert of our own: the Lamps pass already puts the third brake light on the rear glass below the wing)
    }

    /// SUV rear bumper step plate (dark, ribbed).
    func buildBumperStep(_ a: inout CarAccumulator) {
        let probe = CCProbe(a.trim, a.chrome)
        var topY: Double? = nil
        for x in [0.25, 0.5, 0.7] {
            let h = probe.hits(Vec3(x, 0.9, rearZ - 0.03), Vec3(0, -1, 0))
            if let f = h.first { let y = 0.9 - f.t; topY = max(topY ?? 0, y) }
        }
        guard let ty = topY, ty > 0.15, ty < 0.7 else { return }
        let zb = rearZ - 0.075, zf = rearZ + 0.02
        var plate = carBevelBox(Vec3(-0.62, ty - 0.004, zb), Vec3(0.62, ty + 0.014, zf), 0.004)
        for i in 0..<7 {
            let x = -0.54 + 0.18 * Double(i)
            plate.append(carBox(Vec3(x - 0.012, ty + 0.012, zb + 0.012), Vec3(x + 0.012, ty + 0.019, zf - 0.012)))
        }
        a.trim.append(plate)
    }

    /// Sedan / taxi / coupe deck lid: shut lines, handle recess + badge; lip spoiler (ducktail) on the coupe.
    func buildDeckLidCues(_ a: inout CarAccumulator, body: CCProbe, coupe: Bool) {
        let cp = cabinPoly
        guard cp.count >= 4 else { return }
        let zFront = cp[0].0 - 0.12, zBack = rearZ + 0.07
        let xl = coupe ? 0.66 : 0.70
        var lines = Mesh3()
        func seg(_ x: Double, _ z: Double, w: Double, h: Double, up: Vec3) {
            if let m = ccStamp(body, from: Vec3(x, 0.30, z), dir: Vec3(0, 1, 0), up: up, w: w, h: h, back: 0.004, proud: 0.0012, maxDev: 0.02) { lines.append(m) }
        }
        let n = max(2, Int(((zFront - zBack) / 0.10).rounded(.up)))
        for i in 0..<n {
            let z0 = zBack + (zFront - zBack) * Double(i) / Double(n), z1 = zBack + (zFront - zBack) * Double(i + 1) / Double(n)
            for sx in [-1.0, 1.0] { seg(sx * xl, (z0 + z1) / 2, w: 0.004, h: z1 - z0 + 0.002, up: Vec3(0, 0, 1)) }
        }
        let m = max(2, Int(((2 * xl) / 0.10).rounded(.up)))
        for i in 0..<m {
            let x0 = -xl + 2 * xl * Double(i) / Double(m), x1 = -xl + 2 * xl * Double(i + 1) / Double(m)
            seg((x0 + x1) / 2, zFront, w: 0.004, h: x1 - x0 + 0.002, up: Vec3(1, 0, 0))
        }
        a.trim.append(lines)
        // handle recess + badge on the lid's rear face above the plate
        let zIn = rearZ + 0.5
        let yh = (body.top(0, rearZ + 0.05) ?? 0.9) - 0.12
        if let pocket = ccStamp(body, from: Vec3(0, yh, zIn), dir: Vec3(0, 0, -1), w: 0.17, h: 0.034, back: 0.004, proud: 0.0012, maxDev: 0.03) { a.trim.append(pocket) }
        if let bar = ccStamp(body, from: Vec3(0, yh, zIn), dir: Vec3(0, 0, -1), w: 0.12, h: 0.011, back: 0.004, proud: 0.0040, maxDev: 0.03) { a.chrome.append(bar) }
        if let badge = ccStamp(body, from: Vec3(-0.30, yh, zIn), dir: Vec3(0, 0, -1), w: 0.10, h: 0.024, back: 0.004, proud: 0.0030, maxDev: 0.03) { a.chrome.append(badge) }
        if coupe {
            // ducktail lip spoiler: a low wedge on the deck's trailing edge
            let zb = rearZ + 0.035, zf = rearZ + 0.20
            var yMin = 10.0
            for x in stride(from: -0.62, through: 0.62, by: 0.1) { for z in [zb, (zb + zf) / 2, zf] { yMin = min(yMin, body.top(x, z) ?? 10) } }
            guard yMin < 9 else { return }
            let y0 = yMin - 0.014
            let poly: [Vec2] = [Vec2(zf, y0 + 0.012), Vec2(zb + 0.015, y0 + 0.004), Vec2(zb, y0 + 0.018), Vec2(zb, y0 + 0.066), Vec2(zb + 0.05, y0 + 0.058)]
            a.body.append(ccSidePrism(poly, x0: -0.63, x1: 0.63))
        }
    }
}
