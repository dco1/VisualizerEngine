import Foundation
import simd

// plausibility: real — headlamp housings/lenses/projectors, grilles, fogs and repeaters are separate closed solids
// placed on the shell's own plan outline; honeycomb/slats are real small solids, not textures.

enum LampClass { case sedan, hatch, suv, pickup, van, coupe, bus }

enum GrilleKind { case bars(Int), honeycomb, slats(Int), heavy }

struct LampFaceStyle {
    var lampH: Double       // headlamp lens height
    var xIn: Double         // inboard start of the lamp, fraction of the nose half-width
    var wrap: Double        // metres the lens wraps back along the fender
    var bumpH: Double       // bumper height
    var grille: GrilleKind
}

extension CarContext {
    /// Body class recovered from the blueprint (the context carries no `CarStyle`). Reference dimensions are unique
    /// per class: bus boxy; pickup > 5.5 m; van ≈ 1.97 m tall; suv > 1.6 m; hatch < 4.4 m; coupe has the narrow cabin.
    var lampClass: LampClass {
        if bp.boxy { return .bus }
        if bp.length > 5.5 { return .pickup }
        if bp.height > 1.9 { return .van }
        if bp.height > 1.6 { return .suv }
        if bp.length < 4.4 { return .hatch }
        if bp.cabinWidthFrac < 0.835 || (bp.cornerRadius > 0.15 && bp.cabinWidthFrac < 0.87) { return .coupe }
        return .sedan
    }

    var lampFaceStyle: LampFaceStyle {
        var s: LampFaceStyle
        switch lampClass {
        case .sedan:  s = LampFaceStyle(lampH: 0.085, xIn: 0.50, wrap: 0.26, bumpH: 0.20, grille: .bars(2))
        case .hatch:  s = LampFaceStyle(lampH: 0.090, xIn: 0.52, wrap: 0.22, bumpH: 0.20, grille: .bars(1))
        case .suv:    s = LampFaceStyle(lampH: 0.100, xIn: 0.50, wrap: 0.24, bumpH: 0.21, grille: .honeycomb)
        case .pickup: s = LampFaceStyle(lampH: 0.115, xIn: 0.56, wrap: 0.20, bumpH: 0.19, grille: .heavy)
        case .van:    s = LampFaceStyle(lampH: 0.100, xIn: 0.60, wrap: 0.18, bumpH: 0.22, grille: .slats(3))
        case .coupe:  s = LampFaceStyle(lampH: 0.070, xIn: 0.46, wrap: 0.30, bumpH: 0.20, grille: .honeycomb)
        case .bus:    s = LampFaceStyle(lampH: 0.22, xIn: 0.0, wrap: 0.10, bumpH: 0.28, grille: .slats(6))
        }
        if era == .nineties { s.lampH *= 1.2; s.wrap = min(s.wrap, 0.07); s.xIn += 0.0 }
        if classic { s.bumpH = 0.19 }
        return s
    }

    /// Metres of straight fender flank available between the nose and the front wheel arch (minus clearance).
    func flankRoom(_ e: LampEnd) -> Double {
        let ax = e.front ? axles[0] : axles[1]
        let overhang = e.front ? frontZ - ax : ax - rearZ
        return max(0.12, overhang - archR - 0.05)
    }

    /// Front bumper band bottom at the very nose.
    func bumperBottom(_ e: LampEnd) -> Double { bp.endBottom - 0.04 }

    // MARK: - Front end

    func buildFront(_ a: inout CarAccumulator, _ r: inout LampRecorder) {
        let e = lampEnd(true)
        let st = lampFaceStyle
        let hwN = hwAt(e, 0)
        let z0 = e.z0, dir = e.dir
        let room = flankRoom(e)

        if bp.boxy { buildBusFront(&a, &r, e, st); return }
        let band = wallBand(e, 0)
        let bumpTop = bumperBottom(e) + st.bumpH

        if classic {
            buildClassicHeadlamps(&a, &r, e, hwN: hwN, bumpTop: bumpTop)
            buildClassicGrille(&a, &r, e, hwN: hwN, bumpTop: bumpTop)
            let rA = max(0.12, room - 0.16), rB = min(rA + 0.08, room)
            let yR = max(wallBand(e, rB).lo + 0.04, min(bumpTop + 0.16, wallBand(e, rA).hi - 0.04))
            emitPair(&a, &r, .signals, .sideRepeater, lampWrap(lampPath(e, faceX0: nil, flankD0: rA, flankD1: rB), chamfer: 0.005) { _ in
                LampSection(y0: yR - 0.014, y1: yR + 0.014, out: 0.014, inn: 0.03)
            })
            return
        }

        // ── Headlamp unit ──
        let xIn = st.xIn * hwN
        let yTop = band.hi - 0.012
        let wrap = min(st.wrap, room * 0.5)
        let modern = era == .modern
        // Lens never reaches down into the bumper band (leaves room for the DRL line under it).
        let hLens = max(0.05, min(st.lampH, (yTop - (bumpTop + 0.05)) / (modern ? 1.0 : 1.12)))
        func lensBand(_ s: LStation) -> (Double, Double) {
            let t = s.onFace ? 0 : min(1, max(0, (s.u - (hwN - xIn)) / max(wrap, 0.02)))
            if modern {
                let y1 = yTop + 0.02 * t
                return (y1 - hLens * (1 - 0.45 * t), y1)       // swept: rises and slims towards the fender
            }
            let k = s.onFace ? max(0, (s.x - xIn) / max(hwN - xIn, 0.05)) : 1       // wedge: taller outboard
            return (yTop - hLens * (1 + 0.12 * k), yTop)
        }
        let path = lampPath(e, faceX0: xIn, flankD1: wrap)
        let housing = lampWrap(path, chamfer: 0.005) { s in
            let (y0, y1) = lensBand(s)
            return LampSection(y0: y0 - 0.010, y1: y1 + 0.010, out: 0.006, inn: 0.05)
        }
        emitPair(&a, &r, .trim, .headlampHousing, housing)

        func lens(_ p: LampPath) -> Mesh3 {
            lampWrap(p, chamfer: 0.008) { s in
                let (y0, y1) = lensBand(s)
                return LampSection(y0: y0, y1: y1, out: 0.026, inn: -0.016)
            }
        }
        if modern {
            emitPair(&a, &r, .headlights, .headlampLens, lens(path))
        } else {
            // Two halogen units behind separate rectangular lenses, divided by a housing web.
            let mid = xIn + (hwN - xIn) * 0.5
            emitPair(&a, &r, .headlights, .headlampLens, lens(lampPath(e, faceX0: xIn + 0.012, faceX1: mid - 0.012)))
            emitPair(&a, &r, .headlights, .headlampLens, lens(lampPath(e, faceX0: mid + 0.012, flankD1: wrap)))
        }

        // Projector / reflector internals: bowl + ring + lens dome standing on the housing floor, under the cover.
        let yMid = (lensBand(path.s[0]).0 + lensBand(path.s[0]).1) / 2 + (modern ? 0 : -0.012)
        let rim = z0 + dir * 0.0145
        let nUnits = modern ? (hwN - xIn > 0.34 ? 2 : 1) : 2
        for k in 0..<nUnits {
            let R = modern ? min(0.030, hLens * 0.36) : min(0.034, hLens * 0.30)
            let x = modern ? xIn + 0.07 + Double(k) * (2 * R + 0.045)
                           : (k == 0 ? xIn + (hwN - xIn) * 0.25 : xIn + (hwN - xIn) * 0.72) - 0.03 * Double(k)
            let bowl: [(Double, Double)] = [(0, -0.014), (R + 0.004, -0.014), (R + 0.004, 0), (R, 0),
                                            (R * 0.6, -0.006), (R * 0.25, -0.0095), (0, -0.010)]
            emitPair(&a, &r, .chrome, .headlampReflector, lampLathe(bowl, at: Vec3(x, yMid, rim), dirZ: dir, segments: 20))
            if modern {
                emitPair(&a, &r, .chrome, .headlampReflector, lampTorus(at: Vec3(x, yMid, rim - dir * 0.003), major: R + 0.002, minor: 0.0025, segsMajor: 20, segsMinor: 6))
                let d: [(Double, Double)] = [(0, -0.008), (R * 0.62, -0.008), (R * 0.55, -0.003), (R * 0.3, 0.0), (0, 0.001)]
                emitPair(&a, &r, .headlights, .projectorLens, lampLathe(d, at: Vec3(x, yMid, rim), dirZ: dir, segments: 16))
            }
        }

        // ── Daytime-running-light strip under the lamp ──
        let drlPath = lampPath(e, faceX0: xIn + 0.02, flankD1: wrap * 0.9)
        emitPair(&a, &r, .signals, .drl, lampWrap(drlPath, chamfer: 0.003, smooth: 50) { s in
            let (y0, _) = lensBand(s)
            let yy = y0 - 0.032
            return LampSection(y0: yy - 0.006, y1: yy + 0.006, out: 0.014, inn: 0.03)
        })

        // ── Indicator segment beyond the lens + side repeater ──
        let (iy0, iy1) = lensBand(LStation(x: 0, z: 0, n: .zero, u: 1e9, onFace: false))
        let indA = wrap + 0.012, indB = min(wrap + 0.012 + min(0.10, 0.18 * room), room - 0.04)
        if indB > indA + 0.03 {
            emitPair(&a, &r, .signals, .frontIndicator, lampWrap(lampPath(e, faceX0: nil, flankD0: indA, flankD1: indB), chamfer: 0.006) { _ in
                LampSection(y0: iy0, y1: iy1, out: 0.020, inn: 0.03)
            })
        }
        let repA = max(indB + 0.025, room - 0.14), repB = min(repA + 0.075, room)
        let yRep = max(wallBand(e, repA).lo + 0.03, min(iy0 - 0.07, wallBand(e, repB).hi - 0.03))
        emitPair(&a, &r, .signals, .sideRepeater, lampWrap(lampPath(e, faceX0: nil, flankD0: repA, flankD1: repB), chamfer: 0.005) { _ in
            LampSection(y0: yRep - 0.014, y1: yRep + 0.014, out: 0.014, inn: 0.03)
        })

        // ── Bumper, fog lamps, intakes, plate ──
        buildModernBumperFront(&a, &r, e, hwN: hwN, st: st, room: room)

        // ── Grille ──
        let gx = max(0.18, xIn - 0.035)
        let gy0 = bumpTop + 0.018
        // Pickups carry a tall, bold grille that climbs the whole nose face up to the hood lip.
        let gy1 = lampClass == .pickup ? band.hi + 0.07 : max(gy0 + 0.07, yTop + 0.03)
        buildModernGrille(&a, &r, e, kind: st.grille, gx: gx, y0: gy0, y1: gy1)
    }

    // MARK: modern grille

    func buildModernGrille(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, kind: GrilleKind,
                           gx: Double, y0: Double, y1: Double) {
        let z0 = e.z0, dir = e.dir
        func box(_ x0: Double, _ x1: Double, _ ya: Double, _ yb: Double, _ zIn: Double, _ zOut: Double, bevel: Double = 0.003) -> Mesh3 {
            let za = z0 + dir * zIn, zb = z0 + dir * zOut
            return carBevelBox(Vec3(x0, ya, min(za, zb)), Vec3(x1, yb, max(za, zb)), bevel)
        }
        // Bezel plate (black plastic), embedded in the face.
        emit(&a, &r, .trim, .grille, box(-gx, gx, y0, y1, -0.03, 0.020, bevel: 0.008))
        let ix = gx - 0.026, iy0 = y0 + 0.016, iy1 = y1 - 0.016
        let h = iy1 - iy0
        var chrome = Mesh3()
        switch kind {
        case .bars(let n):
            // Bars inset from the plate's ends so no bar shares an edge with the surround.
            for i in 0..<n {
                let yc = iy0 + h * (Double(i) + 0.5) / Double(n)
                chrome.append(box(-ix, ix, yc - 0.009, yc + 0.009, 0.010, 0.032))
            }
        case .honeycomb:
            emit(&a, &r, .chrome, .grille, lampHoneycomb(cx: 0, cy: (iy0 + iy1) / 2, hw: ix, hh: h / 2, z: z0 + dir * 0.012, dir: dir,
                                                         cell: min(0.013, h / 7), height: 0.020, maxCells: 260))
            chrome.append(box(-ix - 0.004, ix + 0.004, iy1 - 0.004, iy1 + 0.010, 0.010, 0.030))   // top chrome lip
        case .slats(let n):
            for i in 0..<n {
                let yc = iy0 + h * (Double(i) + 0.5) / Double(n)
                chrome.append(box(-ix, ix, yc - 0.010, yc + 0.010, 0.008, 0.030))
            }
        case .heavy:
            let n = 3
            for i in 0..<n {
                let yc = iy0 + h * (Double(i) + 0.5) / Double(n)
                chrome.append(box(-ix, ix, yc - 0.020, yc + 0.020, 0.008, 0.044, bevel: 0.008))
            }
            chrome.append(box(-gx - 0.012, gx + 0.012, y1 - 0.010, y1 + 0.012, -0.01, 0.040, bevel: 0.005))
            chrome.append(box(-gx - 0.012, -gx + 0.012, y0 + 0.004, y1 - 0.002, -0.01, 0.040, bevel: 0.005))
            chrome.append(box(gx - 0.012, gx + 0.012, y0 + 0.004, y1 - 0.002, -0.01, 0.040, bevel: 0.005))
        }
        emit(&a, &r, .chrome, .grille, chrome)
        // Badge disc: chrome, on the grille centre.
        let bd = [(0.0, 0.0), (0.040, 0.0), (0.041, 0.004), (0.036, 0.0095), (0.0, 0.011)]
        let by = (y0 + y1) / 2
        emit(&a, &r, .chrome, .badge, lampLathe(bd, at: Vec3(0, by, z0 + dir * 0.050), dirZ: dir, segments: 24))
    }

    // MARK: modern fog lamps / bumper (front)

    func buildModernBumperFront(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, hwN: Double,
                                st: LampFaceStyle, room: Double) {
        let z0 = e.z0, dir = e.dir
        let bumpH = st.bumpH
        let cls = lampClass
        let y0N = bumperBottom(e)
        let dB = room - 0.02
        let out = cls == .pickup ? 0.07 : 0.045
        let path = lampPath(e, faceX0: 0, flankD1: dB).full()
        let body = lampWrap(path, chamfer: 0.02, smooth: 50) { s in
            let d = s.onFace ? 0 : abs(e.z0 - s.z), t = min(1, d / dB)
            return LampSection(y0: y0N, y1: y0N + bumpH, out: out * (1 - 0.7 * t * t), inn: 0.04)
        }
        emit(&a, &r, cls == .pickup ? .chrome : .trim, .bumper, body)
        // Lower valance / lip: a narrower, deeper splitter under the bumper.
        let lipPath = lampPath(e, faceX0: 0, faceX1: hwN * 0.84).full()
        emit(&a, &r, .trim, .bumperLip, lampWrap(lipPath, chamfer: 0.012, smooth: 50) { s in
            let k = 1 - 0.6 * pow(s.x / max(hwN * 0.84, 0.1), 4)
            return LampSection(y0: y0N - 0.035, y1: y0N + 0.045, out: out + 0.014 * k, inn: 0.03)
        })
        if cls == .pickup {
            // Body-colour valance behind the chrome blade, as black plastic.
            emit(&a, &r, .trim, .bumperLip, lampWrap(lampPath(e, faceX0: 0, faceX1: hwN * 0.9).full(), chamfer: 0.01) { _ in
                LampSection(y0: y0N - 0.10, y1: y0N + 0.01, out: 0.030, inn: 0.03)
            })
        }

        // Fog lamps (+ intakes) in the lower bumper.
        let yc = y0N + bumpH * 0.5
        let fogX = hwN * 0.82
        do {
            let rf = cls == .pickup ? 0.042 : 0.036
            let base = z0 + dir * (out - 0.012)
            let housing: [(Double, Double)] = [(0, 0), (rf + 0.014, 0), (rf + 0.014, 0.018), (rf + 0.006, 0.024), (0, 0.024)]
            emitPair(&a, &r, .trim, .fogHousing, lampLathe(housing, at: Vec3(fogX, yc, base), dirZ: dir, segments: 24))
            emitPair(&a, &r, .chrome, .fogHousing, lampTorus(at: Vec3(fogX, yc, base + dir * 0.0225), major: rf + 0.002, minor: 0.0028, segsMajor: 24, segsMinor: 6))
            let lensP: [(Double, Double)] = [(0, 0.004), (rf, 0.004), (rf * 0.96, 0.015), (rf * 0.7, 0.0235), (rf * 0.3, 0.0275), (0, 0.0285)]
            emitPair(&a, &r, .headlights, .fogLamp, lampLathe(lensP, at: Vec3(fogX, yc, base), dirZ: dir, segments: 24))
        }
        // Side intakes: honeycomb studs either side of the plate, outside the plate frame.
        if cls == .sedan || cls == .hatch || cls == .suv || cls == .coupe {
            let ix0 = 0.235, ix1 = fogX - 0.062
            if ix1 - ix0 > 0.10 {
                let m = lampHoneycomb(cx: (ix0 + ix1) / 2, cy: yc, hw: (ix1 - ix0) / 2, hh: bumpH / 2 - 0.036,
                                      z: z0 + dir * (out - 0.004), dir: dir, cell: 0.0105, height: 0.014, maxCells: 110)
                emitPair(&a, &r, .chrome, .grille, m)
            }
        }
        // Licence plate in the middle of the bumper: frame proud of the bumper, plate recessed behind it.
        buildPlate(&a, &r, e, y: yc, zBase: out)
    }

    /// US plate 0.305 × 0.152 m in a four-bar frame. The frame stands 16 mm proud of `zBase`, the plate 6 mm.
    func buildPlate(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, y: Double, zBase: Double) {
        let z0 = e.z0, dir = e.dir
        let w = 0.305, h = 0.152, f = 0.018
        func zb(_ o: Double) -> Double { z0 + dir * o }
        func b(_ x0: Double, _ x1: Double, _ y0: Double, _ y1: Double) -> Mesh3 {
            let za = zb(zBase - 0.012), zc = zb(zBase + 0.016)
            return carBevelBox(Vec3(x0, y0, min(za, zc)), Vec3(x1, y1, max(za, zc)), 0.003)
        }
        var frame = Mesh3()
        frame.append(b(-w / 2 - f - 0.004, w / 2 + f + 0.004, y + h / 2 - 0.004, y + h / 2 + f))      // top (overlaps the sides)
        frame.append(b(-w / 2 - f - 0.004, w / 2 + f + 0.004, y - h / 2 - f, y - h / 2 + 0.004))      // bottom
        frame.append(b(-w / 2 - f, -w / 2 + 0.0, y - h / 2 + 0.001, y + h / 2 - 0.001))               // left
        frame.append(b(w / 2 - 0.0, w / 2 + f, y - h / 2 + 0.001, y + h / 2 - 0.001))                 // right
        emit(&a, &r, .trim, .plateFrame, frame)
        emit(&a, &r, .plates, .plate,
             lampPlatePanel(x: 0, y: y, w: w, h: h, zFront: zb(zBase + 0.006), dir: dir, t: 0.014))
    }

    // MARK: - 1950s

    func buildClassicHeadlamps(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, hwN: Double, bumpTop: Double) {
        let z0 = e.z0, dir = e.dir
        let band = wallBand(e, 0)
        let xh = hwN * 0.74
        let yh = bumpTop + 0.016 + 0.098 + 0.012
        classicRoundLamp(&a, &r, e, base: Vec3(xh, yh, z0), scale: 0.8)
    }

    /// 1950s sealed-beam unit about +z/-z: chrome bezel ring with a recess, glass dome 4 mm below its rim, reflector
    /// bowl behind. `scale` scales the radii only.
    func classicRoundLamp(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, base: Vec3, scale: Double) {
        let dir = e.dir
        let bezel: [(Double, Double)] = [(0, -0.03), (0.122, -0.03), (0.122, 0.020), (0.116, 0.031), (0.106, 0.034),
                                         (0.096, 0.033), (0.092, 0.026), (0.092, 0.004), (0, 0.004)]
        func sc(_ p: [(Double, Double)]) -> [(Double, Double)] { p.map { ($0.0 * scale, $0.1) } }
        emitPair(&a, &r, .chrome, .headlampHousing, lampLathe(sc(bezel), at: base, dirZ: dir, segments: 32))
        let dome: [(Double, Double)] = [(0, 0.002), (0.088, 0.002), (0.087, 0.010), (0.072, 0.021), (0.045, 0.0285), (0.02, 0.0300), (0, 0.0305)]
        emitPair(&a, &r, .headlights, .headlampLens, lampLathe(sc(dome), at: base, dirZ: dir, segments: 32))
        let bowl: [(Double, Double)] = [(0, -0.024), (0.085, -0.024), (0.085, 0.0), (0.075, 0.0), (0.045, -0.012), (0.02, -0.019), (0, -0.0205)]
        emitPair(&a, &r, .chrome, .headlampReflector, lampLathe(sc(bowl), at: base, dirZ: dir, segments: 24))
    }

    func buildClassicGrille(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, hwN: Double, bumpTop: Double) {
        let z0 = e.z0, dir = e.dir
        let band = wallBand(e, 0)
        let xh = hwN * 0.74
        let gx = xh - 0.098 - 0.022
        let gy0 = bumpTop + 0.012, gy1 = max(gy0 + 0.10, band.hi - 0.01)
        func box(_ x0: Double, _ x1: Double, _ ya: Double, _ yb: Double, _ zIn: Double, _ zOut: Double, _ bevel: Double = 0.004) -> Mesh3 {
            let za = z0 + dir * zIn, zb = z0 + dir * zOut
            return carBevelBox(Vec3(x0, ya, min(za, zb)), Vec3(x1, yb, max(za, zb)), bevel)
        }
        emit(&a, &r, .trim, .grille, box(-gx + 0.01, gx - 0.01, gy0 + 0.01, gy1 - 0.01, -0.03, 0.012, 0.005))   // dark opening
        var c = Mesh3()
        c.append(box(-gx - 0.010, gx + 0.010, gy1 - 0.026, gy1 + 0.004, -0.01, 0.050, 0.008))               // top bar
        c.append(box(-gx - 0.010, gx + 0.010, gy0 - 0.004, gy0 + 0.026, -0.01, 0.050, 0.008))               // bottom bar
        c.append(box(-gx - 0.010, -gx + 0.014, gy0 + 0.010, gy1 - 0.010, -0.01, 0.050, 0.006))              // ends
        c.append(box(gx - 0.014, gx + 0.010, gy0 + 0.010, gy1 - 0.010, -0.01, 0.050, 0.006))
        c.append(box(-gx + 0.014, gx - 0.014, (gy0 + gy1) / 2 - 0.007, (gy0 + gy1) / 2 + 0.007, 0.0, 0.040, 0.004))   // centre bar
        let teeth = 11
        for i in 0..<teeth {
            let x = -gx + 0.05 + (2 * gx - 0.10) * Double(i) / Double(teeth - 1)
            c.append(box(x - 0.0085, x + 0.0085, gy0 + 0.020, gy1 - 0.020, 0.0, 0.058, 0.006))             // the "teeth"
        }
        emit(&a, &r, .chrome, .grille, c)
        let bd = [(0.0, 0.0), (0.055, 0.0), (0.057, 0.005), (0.050, 0.012), (0.0, 0.0145)]
        emit(&a, &r, .chrome, .badge, lampLathe(bd, at: Vec3(0, (gy0 + gy1) / 2, z0 + dir * 0.052), dirZ: dir, segments: 28))

        // Chrome blade bumper with bullet overriders, an amber parking/indicator lens and a plate bracket.
        let st = lampFaceStyle
        let y0N = bumperBottom(e), room = flankRoom(e)
        let out = 0.075
        let blade = lampWrap(lampPath(e, faceX0: 0, flankD1: room - 0.02).full(), chamfer: 0.035, smooth: 55) { s in
            let d = s.onFace ? 0 : abs(e.z0 - s.z), t = min(1, d / (room - 0.02))
            return LampSection(y0: y0N, y1: y0N + st.bumpH, out: out * (1 - 0.7 * t * t), inn: 0.02)
        }
        emit(&a, &r, .chrome, .bumper, blade)
        let yb = y0N + st.bumpH / 2
        let bullet: [(Double, Double)] = [(0, 0), (0.046, 0), (0.054, 0.022), (0.045, 0.052), (0.028, 0.075), (0.010, 0.086), (0, 0.089)]
        for fx in [0.36, 0.60] {
            emitPair(&a, &r, .chrome, .overrider, lampLathe(bullet, at: Vec3(hwN * fx, yb, z0 + dir * (out - 0.03)), dirZ: dir, segments: 20))
        }
        let amber: [(Double, Double)] = [(0, 0), (0.034, 0), (0.033, 0.006), (0.022, 0.014), (0.01, 0.018), (0, 0.0185)]
        emitPair(&a, &r, .signals, .frontIndicator, lampLathe(amber, at: Vec3(hwN * 0.86, yb, z0 + dir * (out - 0.004)), dirZ: dir, segments: 20))
        buildClassicPlate(&a, &r, e, y: yb, out: out)
    }

    func buildClassicPlate(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, y: Double, out: Double) {
        // The plate bolts to the blade bumper between the bullets: frame stands off the blade face.
        buildPlate(&a, &r, e, y: y, zBase: out)
    }

    // MARK: - Bus

    func buildBusFront(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, _ st: LampFaceStyle) {
        let hwN = hwAt(e, 0), z0 = e.z0, dir = e.dir
        let room = min(flankRoom(e), 0.5)
        let yb0 = bumperBottom(e)
        // Wide rubber bumper.
        let bumper = lampWrap(lampPath(e, faceX0: 0, flankD1: min(room, 0.4)).full(), chamfer: 0.03, smooth: 50) { _ in
            LampSection(y0: yb0, y1: yb0 + st.bumpH, out: 0.07, inn: 0.04)
        }
        emit(&a, &r, .trim, .bumper, bumper)
        emit(&a, &r, .trim, .bumperLip, lampWrap(lampPath(e, faceX0: 0, faceX1: hwN * 0.9).full(), chamfer: 0.01) { _ in
            LampSection(y0: yb0 - 0.05, y1: yb0 + 0.02, out: 0.10, inn: 0.03)
        })
        let yTop = 1.10, h = st.lampH
        let xIn = hwN - 0.56
        if classic {
            classicRoundLamp(&a, &r, e, base: Vec3(hwN - 0.36, yTop - h / 2, z0), scale: 1.6)
            emitPair(&a, &r, .signals, .frontIndicator, lampWrap(lampPath(e, faceX0: nil, flankD0: 0.03, flankD1: 0.15), chamfer: 0.006) { _ in
                LampSection(y0: yTop - h * 0.6, y1: yTop - h * 0.2, out: 0.020, inn: 0.03)
            })
            emitPair(&a, &r, .signals, .sideRepeater, lampWrap(lampPath(e, faceX0: nil, flankD0: 0.30, flankD1: 0.38), chamfer: 0.005) { _ in
                LampSection(y0: yTop - h * 0.5 - 0.016, y1: yTop - h * 0.5 + 0.016, out: 0.014, inn: 0.03)
            })
            busIntakeAndSign(&a, &r, e, gx: hwN - 0.36 - 0.2 - 0.05, yb0: yb0, st: st, hwN: hwN)
            return
        }
        let path = lampPath(e, faceX0: xIn, flankD1: 0.10)
        emitPair(&a, &r, .trim, .headlampHousing, lampWrap(path, chamfer: 0.006) { _ in
            LampSection(y0: yTop - h - 0.012, y1: yTop + 0.012, out: 0.008, inn: 0.05)
        })
        emitPair(&a, &r, .headlights, .headlampLens, lampWrap(path, chamfer: 0.01) { _ in
            LampSection(y0: yTop - h, y1: yTop, out: 0.030, inn: -0.016)
        })
        let rim = z0 + dir * 0.0145
        for k in 0..<2 {
            let x = xIn + 0.14 + Double(k) * 0.24
            let bowl: [(Double, Double)] = [(0, -0.012), (0.075, -0.012), (0.075, 0), (0.068, 0), (0.04, -0.007), (0.015, -0.0105), (0, -0.011)]
            emitPair(&a, &r, .chrome, .headlampReflector, lampLathe(bowl, at: Vec3(x, yTop - h / 2, rim), dirZ: dir, segments: 20))
        }
        emitPair(&a, &r, .signals, .drl, lampWrap(lampPath(e, faceX0: xIn + 0.02, flankD1: 0.09), chamfer: 0.003, smooth: 50) { _ in
            LampSection(y0: yTop - h - 0.045, y1: yTop - h - 0.030, out: 0.016, inn: 0.03)
        })
        emitPair(&a, &r, .signals, .frontIndicator, lampWrap(lampPath(e, faceX0: nil, flankD0: 0.115, flankD1: 0.23), chamfer: 0.006) { _ in
            LampSection(y0: yTop - h, y1: yTop, out: 0.020, inn: 0.03)
        })
        emitPair(&a, &r, .signals, .sideRepeater, lampWrap(lampPath(e, faceX0: nil, flankD0: 0.30, flankD1: 0.38), chamfer: 0.005) { _ in
            LampSection(y0: yTop - h * 0.5 - 0.016, y1: yTop - h * 0.5 + 0.016, out: 0.014, inn: 0.03)
        })
        busIntakeAndSign(&a, &r, e, gx: xIn - 0.05, yb0: yb0, st: st, hwN: hwN)
    }

    func busIntakeAndSign(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, gx: Double, yb0: Double,
                          st: LampFaceStyle, hwN: Double) {
        let z0 = e.z0, dir = e.dir
        // Low intake: dark panel with horizontal louvres between the lamps.
        emit(&a, &r, .trim, .grille, carBevelBox(Vec3(-gx, 0.84, z0 - 0.03), Vec3(gx, 1.24, z0 + 0.02), 0.01))
        var slats = Mesh3()
        for i in 0..<6 {
            let yc = 0.88 + Double(i) * 0.064
            slats.append(carBevelBox(Vec3(-gx + 0.04, yc - 0.012, z0 + 0.008), Vec3(gx - 0.04, yc + 0.012, z0 + 0.036), 0.004))
        }
        emit(&a, &r, .chrome, .grille, slats)
        let bd = [(0.0, 0.0), (0.060, 0.0), (0.062, 0.005), (0.055, 0.012), (0.0, 0.0145)]
        emit(&a, &r, .chrome, .badge, lampLathe(bd, at: Vec3(0, 1.36, z0 + 0.015), dirZ: 1, segments: 28))
        buildPlate(&a, &r, e, y: yb0 + st.bumpH * 0.5, zBase: 0.07)
        // Destination sign over the windscreen: dark housing + amber LED display.
        let sy = bp.belt * rH + 1.62 * rH
        emit(&a, &r, .trim, .destinationSign, carBevelBox(Vec3(-hwN + 0.5, sy, z0 - 0.012), Vec3(hwN - 0.5, sy + 0.24, z0 + 0.014), 0.01))
        emit(&a, &r, .signals, .destinationSign, carBox(Vec3(-hwN + 0.62, sy + 0.045, z0 + 0.0145), Vec3(hwN - 0.62, sy + 0.195, z0 + 0.019)))
    }
}
