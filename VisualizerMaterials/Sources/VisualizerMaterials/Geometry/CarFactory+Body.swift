import Foundation
import simd

// plausibility: real — the paint shell is a stepped-section loft (32 points per section: hood crown, shut-line
// valley, fender crest, shoulder radius, character ridge, rocker, wheel-well roof and floor) whose per-station
// numbers come from the class and era; the wheel openings are cut into the flank by following the tyre circle
// (+3.5 cm) and are framed by a swept, bevelled lip. No CSG, no textures, no cylinders poking through a slab.

// MARK: - Class / era numbers

/// Thread-safe memo for `CarContext.shellClass`.
private enum ShellClassCache {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var table: [String: CarStyle] = [:]
    static func get(_ k: String) -> CarStyle? { lock.lock(); defer { lock.unlock() }; return table[k] }
    static func set(_ k: String, _ v: CarStyle) { lock.lock(); table[k] = v; lock.unlock() }
}

/// Shape numbers for the shell, by body class. Heights in metres, all measured on the finished car.
struct CarShellStyle {
    var dome = 0.0            // hood crown above the shut line
    var valley = 0.0          // fender crest above the hood panel at the shut line
    var rake = 0.0            // tail-up wedge: tail lines sit this much higher than the nose lines
    var beltRise = 0.0        // door-top (beltline) rise from the windscreen base to the C-pillar
    var kick = 0.0            // deck-lid kick-up / tailgate lip height
    var kickStart = 0.5       // distance from the tail where the kick begins
    var kickSat = 0.18        // ...and where it is fully up
    var inset = 0.03          // rocker tuck-in under the flank
    var cladding = false      // black sill cladding + black fender flares (SUV / pickup / van / bus)
    var flare = 1.0           // wheel-lip size multiplier
    var scoop = 0             // 0 none, 1 centre scoop (pickup), 2 twin extractors (coupe)
    var noseDrop = 0.0        // nose tip height relative to the blueprint tip (negative = lower, wedge)
    var noseRun = 0.3         // length over which the nose profile eases into the hood
    var noseTaper = 0.0       // extra plan narrowing at the very tip (fraction of half-width)
    var noseRound = 0.0       // extra shoulder radius at the nose (rounded corners)
    var finH = 0.0            // 1950s tail-fin height above the deck
}

extension CarContext {

    /// The class behind `bp` (taxi reports as sedan — same shell). Memoised: it is read thousands of times per build.
    var shellClass: CarStyle {
        let key = "\(era)|\(bp.length)|\(bp.height)|\(bp.wheelbase)|\(bp.cabin.count)|\(bp.top.count)|\(bp.cabin.first?.0 ?? 0)|\(bp.top.last?.0 ?? 0)"
        if let hit = ShellClassCache.get(key) { return hit }
        var found = CarStyle.sedan
        for s in CarStyle.allCases {
            let r = CarBlueprint.make(s, era: era)
            if r.length == bp.length, r.height == bp.height, r.wheelbase == bp.wheelbase,
               r.cabin.count == bp.cabin.count, r.top.count == bp.top.count,
               zip(r.cabin, bp.cabin).allSatisfy({ $0.0 == $1.0 && $0.1 == $1.1 }),
               zip(r.top, bp.top).allSatisfy({ $0.0 == $1.0 && $0.1 == $1.1 }) { found = s; break }
        }
        ShellClassCache.set(key, found)
        return found
    }

    var shellStyle: CarShellStyle {
        var s = CarShellStyle()
        switch shellClass {
        case .sedan, .taxi: s.dome = 0.035; s.valley = 0.025; s.rake = 0.055; s.beltRise = 0.020; s.kick = 0.055
                            s.noseDrop = -0.02; s.noseRun = 0.65; s.noseTaper = 0.04
        case .hatchback:    s.dome = 0.030; s.valley = 0.022; s.rake = 0.045; s.beltRise = 0.020; s.kick = 0.020
                            s.kickStart = 0.40; s.kickSat = 0.12
                            s.noseDrop = 0.0; s.noseRun = 0.28; s.noseTaper = 0.015; s.noseRound = 0.02
        case .suv:          s.dome = 0.050; s.valley = 0.020; s.rake = 0.020; s.beltRise = 0.020; s.kick = 0.020
                            s.kickStart = 0.40; s.kickSat = 0.12; s.cladding = true; s.flare = 1.5; s.inset = 0.025
                            s.noseRun = 0.3
        case .pickup:       s.dome = 0.040; s.valley = 0.030; s.rake = 0.025; s.beltRise = 0.020; s.kick = 0.035
                            s.kickStart = 0.16; s.kickSat = 0.07; s.cladding = true; s.flare = 1.6; s.scoop = 1; s.inset = 0.02
                            s.noseRun = 0.3
        case .van:          s.dome = 0.020; s.valley = 0.020; s.rake = 0.010; s.kick = 0.018
                            s.kickStart = 0.35; s.kickSat = 0.10; s.cladding = true; s.flare = 1.2; s.inset = 0.012
                            s.noseRun = 0.16; s.noseTaper = 0.07; s.noseRound = 0.07
        case .coupe:        s.dome = 0.045; s.valley = 0.030; s.rake = 0.060; s.beltRise = 0.025; s.kick = 0.070
                            s.kickStart = 0.55; s.kickSat = 0.16; s.scoop = 2
                            s.noseDrop = -0.07; s.noseRun = 1.1; s.noseTaper = 0.08
        case .bus:          s.cladding = true; s.flare = 1.3; s.inset = 0.0; s.kick = 0.0; s.noseRun = 0.1
        }
        if classic { s.rake *= 0.5; s.beltRise *= 0.4; s.inset = 0.0; s.dome *= 1.1; if [.sedan, .taxi, .coupe].contains(shellClass) { s.finH = 0.12 } }
        if bp.boxy { s.dome = 0; s.valley = 0; s.beltRise = 0; s.noseDrop = 0 }
        return s
    }

    // MARK: scalar helpers

    fileprivate func sStep(_ t: Double) -> Double { let x = min(max(t, 0), 1); return x * x * (3 - 2 * x) }

    /// Radius of the wheel opening in the flank: the tyre radius plus a 3.5 cm running gap.
    var shellOpeningR: Double { R + 0.035 }
    /// Horizontal run of the rounded "toe" that carries the opening down into the sill line.
    var shellToe: Double { max(0.05, 0.6 * (R - bp.sill)) }

    /// Lower edge of the flank at `z`: the wheel opening (circle about the axle, then a quarter-ellipse toe down to
    /// the sill) or the sill height itself.
    func shellArchEdge(_ z: Double) -> Double {
        var e = bp.sill
        let ro = shellOpeningR, dl = shellToe
        for ax in axles {
            let dz = abs(z - ax)
            if dz < ro { e = max(e, R + (ro * ro - dz * dz).squareRoot()) }
            else if dz < ro + dl {
                let t = (dz - ro) / dl
                e = max(e, R - max(R - bp.sill, 0) * (1 - (1 - t) * (1 - t)).squareRoot())
            }
        }
        return e
    }

    /// Height of the belly plate (and the approach/departure ramp that lifts it toward the bumper tips).
    func shellUnderside(_ z: Double) -> Double {
        let st = shellStyle
        var y = bp.sill
        let ro = shellOpeningR + shellToe + 0.02
        let zf0 = axles[0] + ro, zr0 = axles[1] - ro
        if z > zf0 { y += (bp.endBottom - bp.sill) * sStep((z - zf0) / max(frontZ - zf0, 0.1)) }
        if z < zr0 { y += (bp.endBottom + st.rake - bp.sill) * sStep((zr0 - z) / max(zr0 - rearZ, 0.1)) }
        return y
    }

    /// Flank bottom edge: the higher of the opening and the belly plate.
    func shellFlankBottom(_ z: Double) -> Double { max(shellArchEdge(z), shellUnderside(z)) }

    /// 1 inside the wheel-arch zone of either axle, easing to 0 a short way beyond it.
    func shellArchBlend(_ z: Double) -> Double {
        let edge = shellOpeningR + shellToe + 0.05
        var b = 0.0
        for ax in axles { b = max(b, 1 - sStep((abs(z - ax) - edge) / 0.25)) }
        return b
    }

    /// Where the windscreen base sits (front of the cabin); the hood runs from here to the nose.
    var shellCowlZ: Double { bp.cabin.isEmpty ? rearZ : cabinSpan.1 }

    func shellHoodWindow(_ z: Double) -> Double {
        guard !bp.boxy, !bp.cabin.isEmpty else { return 0 }
        let lh = frontZ - shellCowlZ
        guard lh > 0.4 else { return 0 }
        let a = min(0.35, 0.3 * lh), b = min(0.35, 0.3 * lh)
        return sStep((frontZ - z - 0.22) / a) * sStep((z - shellCowlZ) / b)
    }

    /// Door-top rise and wedge: 0 at the windscreen base, full at the C-pillar and over the deck.
    func shellBeltRise(_ z: Double) -> Double {
        guard !bp.cabin.isEmpty else { return 0 }
        let (z0, z1) = cabinSpan
        let t = min(max((z1 - z) / max(z1 - z0, 1e-6), 0), 1)
        return shellStyle.beltRise * t
    }

    func shellRake(_ z: Double) -> Double { shellStyle.rake * (frontZ - z) / L }

    func shellKick(_ z: Double) -> Double {
        let st = shellStyle
        guard st.kick > 0 else { return 0 }
        let dr = z - rearZ
        return st.kick * sStep((st.kickStart - dr) / max(st.kickStart - st.kickSat, 1e-6))
    }

    /// Hood outline: the blueprint, except the nose, which eases from a class-specific tip height into the hood
    /// (sloped sedan, low coupe wedge, short blunt hatch, near-vertical van).
    func shellNoseY(_ z: Double) -> Double {
        let st = shellStyle
        let top = poly(bp.top)
        let d = frontZ - z
        guard d < st.noseRun else { return interp(top, z) }
        let tip = interp(top, frontZ) + st.noseDrop
        let yRun = interp(top, frontZ - st.noseRun)
        let t = d / st.noseRun
        return tip + (yRun - tip) * (1 - pow(1 - t, 1.6))
    }

    /// Outline + wedge + rising belt (no tail lip, no fins): what the crease line follows.
    func shellBase(_ z: Double) -> Double { shellNoseY(z) + shellRake(z) + shellBeltRise(z) }

    /// Deck / hood-panel height: the base plus the tail lip.
    func shellDeck(_ z: Double) -> Double { shellBase(z) + shellKick(z) }

    /// Tail fins (1950s): the fender crests climb toward the tail while the deck between them stays low.
    func shellFin(_ z: Double) -> Double {
        let h = shellStyle.finH
        guard h > 0 else { return 0 }
        return h * sStep((0.95 - (z - rearZ)) / 0.55)
    }

    /// Fender-crest height (deck + fin).
    func shellCrest(_ z: Double) -> Double { shellDeck(z) + shellFin(z) }

    /// Door-handle-height line the character crease follows (low-passed so a pickup's bed step does not kink it).
    fileprivate func shellCreaseNominal(_ z: Double) -> Double {
        func n(_ z: Double) -> Double {
            let belt = bp.belt * rH + shellRake(z) + shellBeltRise(z)
            return min(shellBase(z), belt) - (shellTopRadius(z) + 0.065)
        }
        return (n(z - 0.5) + 2 * n(z) + n(z + 0.5)) / 4
    }

    /// Shoulder radius: full on the fender, tight on the door tops.
    func shellTopRadius(_ z: Double) -> Double {
        let st = shellStyle
        let nr = st.noseRound * (1 - sStep((frontZ - z) / 0.45))
        let hood = min(max(bp.cornerRadius * 0.5, 0.04), 0.08) + nr
        guard !bp.cabin.isEmpty else { return hood }
        let (z0, z1) = cabinSpan
        let w = sStep((z - z0 + 0.1) / 0.3) * sStep((z1 + 0.1 - z) / 0.3)
        return hood + (0.035 - hood) * w
    }
    fileprivate func shellTumble(_ z: Double) -> Double {
        guard !bp.cabin.isEmpty else { return 0.015 }
        let (z0, z1) = cabinSpan
        let w = sStep((z - z0 + 0.1) / 0.3) * sStep((z1 + 0.1 - z) / 0.3)
        return 0.008 + 0.027 * w
    }

    fileprivate var shellRidge: (ridge: Double, recess: Double, up: Double, down: Double) {
        // Only sedan / taxi / coupe have a real 1950s body (CarBlueprint.make); other classes keep the modern crease.
        switch (era == .classic1950s && ![.sedan, .taxi, .coupe].contains(shellClass)) ? CarEra.nineties : era {
        case .classic1950s: return (0.034, 0.026, 0.055, 0.060)    // bold swage
        case .nineties:     return (0.008, 0.004, 0.030, 0.035)
        case .modern:       return (0.013, 0.006, 0.030, 0.035)    // sharp crease
        }
    }

    /// Fraction of the crease present at `z` (0 within 4 cm of either end so the end rings stay clean).
    fileprivate func shellEdgeFade(_ z: Double) -> Double {
        sStep((frontZ - z - 0.04) / 0.30) * sStep((z - rearZ - 0.04) / 0.38)
    }

    // MARK: cabin tub

    /// Where the shell is hollowed between the door tops: rear / front bulkhead z and the floor height.
    /// Cars: floor = `interiorFloorY` (underside of the seat cushions), bulkheads just behind the rear seat zone /
    /// ahead of the dash. Bus: floor at the seat-base height of `buildBusGreenhouse`, open end to end.
    var shellTub: (zRb: Double, zFb: Double, yFl: Double)? {
        if bp.boxy { return (rearZ, frontZ, max(2 * R + 0.03, 0.62) - 0.002) }
        guard bp.cabin.count >= 4 else { return nil }
        let m = GHModel(self)
        let rearWall = (m.kind == .sedan || m.kind == .coupe) ? m.zR + 0.30 : m.zR + 0.09
        return (rearWall, m.zF - 0.07, interiorFloorY(m) - 0.002)
    }

    func shellOpenness(_ z: Double) -> Double {
        guard let t = shellTub else { return 0 }
        if bp.boxy { return 1 }
        return sStep((z - t.zRb) / 0.03) * sStep((t.zFb - z) / 0.03)
    }

    /// Half cross-section (x ≥ 0) at `z`, from the top centre down to the belly centre. 17 points.
    func shellHalf(_ z: Double) -> [(x: Double, y: Double)] { shellHalfRaw(z, wellX: shellWellX) }

    /// Wheel-well wall x (inboard of the tyre's inner face by 4 cm). Keyed to the SAME rule the wheels use
    /// (`wheelTrackHalf`: outer tyre face 12 mm inside the median flank skin at five heights above the arch),
    /// evaluated on this shell's own section data, since the shell is built before the wheels.
    var shellWellX: Double {
        var inner = Double.infinity
        for ax in axles {
            let h = shellHalfRaw(ax, wellX: trackHalf - wheelW / 2 - 0.03)
            var xs: [Double] = []
            for dy in [0.05, 0.10, 0.15, 0.20, 0.25] {
                let y = 2 * R + dy
                for i in 5..<12 where h[i].y >= y && h[i + 1].y <= y {
                    let t = (h[i].y - y) / max(h[i].y - h[i + 1].y, 1e-9)
                    xs.append(h[i].x + (h[i + 1].x - h[i].x) * t); break
                }
            }
            let flank = xs.count >= 3 ? xs.sorted()[xs.count / 2] : W / 2 - 0.02
            inner = min(inner, min(flank - 0.012, W / 2 - 0.005) - wheelW)
        }
        return inner - 0.04
    }

    func shellHalfRaw(_ z: Double, wellX: Double) -> [(x: Double, y: Double)] {
        let st = shellStyle
        let dN = frontZ - z
        let hwTaper = 1 - st.noseTaper * (1 - sin(min(dN / max(st.noseRun, 0.05), 1) * .pi / 2))
        let hw = W / 2 * planFactor(z) * hwTaper
        let w = shellHoodWindow(z)
        let yF = bp.boxy ? bp.belt * rH : shellCrest(z)
        let yP = bp.boxy ? yF : shellDeck(z) - st.valley * w
        let yc = yP + st.dome * w
        let cT = shellTopRadius(z), tum = shellTumble(z)
        let xf = hw - tum - cT
        let fn = st.finH > 0 ? shellFin(z) / st.finH : 0
        let xh = 0.62 * xf * (1 - fn) + max(0.62 * xf, xf - 0.10) * fn
        let sill = shellUnderside(z), b = shellFlankBottom(z)
        let inset = st.inset * (1 - shellArchBlend(z))
        let xa = hw - inset

        var p: [(x: Double, y: Double)] = []
        p.append((0, yc))
        p.append((0.35 * xh, yP + st.dome * w * (1 - 0.1225)))
        p.append((0.70 * xh, yP + st.dome * w * (1 - 0.49)))
        p.append((xh, yP))
        p.append((xf, yF))
        // Hollow cabin: blend the solid crown into a U (floor → inner wall → door-top rim) between the bulkheads.
        let o = shellOpenness(z)
        if o > 0, let t = shellTub {
            let hb = bp.boxy ? hw - 0.10 : GHModel(self).hwBase(z) - 0.03
            var xi = min(hb, xf - 0.02)
            let c = sStep((b - (t.yFl - 0.10)) / 0.10)      // a wheelhouse reaches above the floor: pull the wall in
            xi += (wellX - 0.06 - xi) * c
            xi = max(xi, 0.2)
            let tub: [(Double, Double)] = [(0, t.yFl), (0.5 * xi, t.yFl), (xi, t.yFl), (xi, yF), (xf, yF)]
            for i in 0..<5 { p[i] = (p[i].x + (tub[i].0 - p[i].x) * o, p[i].y + (tub[i].1 - p[i].y) * o) }
        }
        for deg in [30.0, 60.0, 90.0] {
            let t = deg * .pi / 180
            p.append((xf + cT * sin(t), yF - cT * (1 - cos(t))))
        }
        let y7 = yF - cT

        // Character line: ridge at door-handle height, pushed down only as far as the wheel lip allows.
        let rd = shellRidge
        let g = 0.012
        let nom = shellCreaseNominal(z)
        var y8 = min(nom + rd.up, y7 - g)
        var y9 = min(nom, y8 - g)
        var y10 = min(nom - rd.down, y9 - g)
        let yMin = b + 0.044
        y10 = max(y10, yMin); y9 = max(y9, y10 + g); y8 = max(y8, y9 + g)
        if y8 > y7 - g {   // not enough room: spread what there is evenly
            let lo = yMin, hi = y7 - g
            y10 = lo; y9 = lo + (hi - lo) / 2; y8 = hi
        }
        let room = (y8 - y10) / (rd.up + rd.down)
        let amp = shellEdgeFade(z) * min(1, max(0, room * 1.8))
        p.append((hw - 0.35 * tum - 0.002, y8))
        p.append((hw + rd.ridge * amp, y9))
        p.append((hw - rd.recess * amp, y10))

        // Lower flank, rocker, wheel-well roof, well wall, belly.
        let y12 = b + 0.02
        p.append(((hw - rd.recess * amp + xa) / 2 + 0.004, (y10 + y12) / 2))
        p.append((xa, y12))
        p.append((xa - 0.02, b))
        p.append((wellX, b))
        p.append((wellX - 0.03, sill))
        p.append((0, sill))
        return p
    }

    /// Height of the hood surface at lateral offset `x` (0 = crown), read off the same section the shell is built from.
    func shellHoodY(_ z: Double, _ x: Double) -> Double {
        let h = shellHalf(z)
        let ax = abs(x)
        for i in 1...4 where ax <= h[i].x { let a = h[i - 1], c = h[i]; return a.y + (c.y - a.y) * (ax - a.x) / max(c.x - a.x, 1e-9) }
        return h[4].y
    }

    // MARK: stations

    func shellStations() -> [Double] {
        var ex: [Double] = poly(bp.top).map(\.0)
        let ro = shellOpeningR, dl = shellToe
        for ax in axles {
            for k in 0...28 { ex.append(ax + ro * cos(Double.pi * Double(k) / 28)) }
            for k in 1...10 { let u = ro + dl * Double(k) / 10; ex.append(ax + u); ex.append(ax - u) }
            ex.append(ax + ro + dl + 0.3); ex.append(ax - ro - dl - 0.3)
        }
        if !bp.cabin.isEmpty { ex += cabinPoly.map(\.0) }
        for d in [0.12, 0.22, 0.34, 0.5, 0.7] { ex.append(frontZ - d) }
        let zc = shellCowlZ
        for d in [0.1, 0.2, 0.3, 0.4] { ex.append(zc + d) }
        for d in [0.04, 0.08, 0.14, 0.2, 0.3, 0.4, 0.55, 0.8, shellStyle.kickSat, shellStyle.kickStart] { ex.append(rearZ + d) }
        if let t = shellTub, !bp.boxy {
            for d in [0.0, 0.015, 0.03] { ex.append(t.zRb + d) }
            for d in [0.0, 0.015, 0.03] { ex.append(t.zFb - d) }
        }
        let st = shellStyle
        for d in [0.04, 0.08, 0.15, 0.25, 0.4, 0.6, 0.85, 1.1] where d < st.noseRun + 0.3 { ex.append(frontZ - d) }
        var zs = [rearZ, frontZ] + ex.filter { $0 > rearZ && $0 < frontZ }
        var s = rearZ + 0.2
        while s < frontZ - 0.01 { zs.append(s); s += 0.2 }
        zs.sort()
        var out: [Double] = []
        for z in zs where out.isEmpty || z - out.last! >= 0.0012 { out.append(z) }
        if frontZ - out.last! < 0.0012, out.count > 1 { out[out.count - 1] = frontZ } else if out.last! < frontZ { out.append(frontZ) }
        return out
    }

    // MARK: build

    /// Body-in-white: the paint shell, stance, wheel arches, panel structure, mirrors, handles.
    /// Appends to `a.body` (paint), `a.trim` (black plastic/rubber), `a.chrome`, `a.underbody`.
    func buildBody(_ a: inout CarAccumulator) {
        let st = shellStyle
        let zs = shellStations()

        // ── 1. The shell: stepped sections lofted nose to tail (closed by hand-made fan caps; both end rings are convex).
        var rings: [[Vec3]] = []
        let wellX0 = shellWellX
        for z in zs {
            let h = shellHalfRaw(z, wellX: wellX0)
            var pts: [(Double, Double)] = h.map { ($0.x, $0.y) }
            for i in stride(from: h.count - 2, through: 1, by: -1) { pts.append((-h[i].x, h[i].y)) }
            rings.append(pts.map { Vec3($0.0, $0.1, z) })
        }
        let shell = shellLoft(rings)
        a.body = shell.smoothed(creaseDegrees: 40)

        // ── 2. Wheel-opening lips (flared, bevelled): swept along the opening edge on the flank.
        var lipMesh = Mesh3()
        for ax in axles { lipMesh.append(shellWheelLip(axle: ax)) }
        let lipSide = lipMesh.smoothed(creaseDegrees: 35)
        if st.cladding { a.trim.append(lipSide); a.trim.append(lipSide.mirrored(.x)) }
        else { a.body.append(lipSide); a.body.append(lipSide.mirrored(.x)) }

        // ── 3. Lower valances (black plastic) front and rear, 1.5–2 cm proud of the lower fascia.
        for front in [true, false] {
            let len = 0.44
            let z0 = front ? frontZ - len : rearZ, z1 = front ? frontZ + 0.012 : rearZ + len
            let band = loftRings(stations(z0, z1, [])) { z in
                let u = shellUnderside(z)
                let hw = W / 2 * planFactor(z) - st.inset + 0.018
                return Ring(hw: hw, bottom: u, top: u + 0.10, corner: 0.02)
            }
            a.trim.append(band.smoothed(creaseDegrees: 45))
        }

        // ── 4. Sill cladding between the wheels (SUV / pickup / van / bus).
        if st.cladding {
            let ro = shellOpeningR + shellToe + 0.035
            let zf = axles[0] - ro, zr = axles[1] + ro
            if zf - zr > 0.5 {
                let xo = W / 2 - st.inset
                let box = carBevelBox(Vec3(xo - 0.035, bp.sill, zr), Vec3(xo + 0.018, bp.sill + 0.13, zf), 0.012)
                a.trim.append(box); a.trim.append(box.mirrored(.x))
            }
        }

        // ── 5. Hood scoops / extractors (pickup, coupe).
        shellHoodScoops(&a)

        // ── 6. Bus: the roof, window mullions, side walls and end panels over the open tub.
        if bp.boxy { shellBusUpper(&a) }

    }

    /// The lip around one right-hand wheel opening, swept along the flank edge (opening circle + toes).
    fileprivate func shellWheelLip(axle ax: Double) -> Mesh3 {
        let st = shellStyle
        let ro = shellOpeningR, dl = shellToe
        var dzs: [Double] = []
        for k in 0...10 { dzs.append(-(ro + dl) + dl * Double(k) / 10); dzs.append(ro + dl * Double(k) / 10) }
        for k in 1..<28 { dzs.append(ro * cos(Double.pi * Double(k) / 28)) }
        dzs = Array(Set(dzs.map { ($0 * 1e6).rounded() / 1e6 })).sorted()
        let pts: [(z: Double, y: Double, x: Double)] = dzs.map { dz in
            let z = ax + dz
            let hw = W / 2 * planFactor(z) - st.inset * (1 - shellArchBlend(z))
            return (z, shellArchEdge(z), hw)
        }
        let f = st.flare * (classic ? 0.85 : 1.0)
        let prof: [(Double, Double)] = [(-0.012, -0.006), (0.010 * f, -0.006), (0.020 * f, 0.004),
                                        (0.020 * f, 0.020), (0.008 * f, 0.032), (-0.012, 0.032)]
        var rings: [[Vec3]] = []
        for i in 0..<pts.count {
            let a = pts[max(0, i - 1)], c = pts[min(pts.count - 1, i + 1)]
            var tz = c.z - a.z, ty = c.y - a.y
            let l = (tz * tz + ty * ty).squareRoot(); tz /= l; ty /= l
            let nz = -ty, ny = tz     // in-plane normal, +y on the crown, into the metal
            rings.append(prof.map { Vec3(pts[i].x + $0.0, max(bp.sill, pts[i].y + ny * $0.1), pts[i].z + nz * $0.1) })
        }
        var m = Mesh3()
        _ = m.loft(rings: rings)
        return m
    }

    /// Raised scoops (pickup) / twin heat extractors (coupe) with a dark mouth, symmetric about the centreline.
    fileprivate func shellHoodScoops(_ a: inout CarAccumulator) {
        let st = shellStyle
        guard st.scoop != 0, frontZ - shellCowlZ > 1.1 else { return }
        let cx: Double, hwid: Double, hmax: Double, d0: Double, d1: Double
        let facingFront: Bool
        if st.scoop == 1 { cx = 0; hwid = 0.19; hmax = 0.05; d0 = 0.55; d1 = 1.10; facingFront = true }
        else { cx = 0.34; hwid = 0.13; hmax = 0.028; d0 = 0.62; d1 = 1.05; facingFront = false }
        let zFront = frontZ - d0, zRear = frontZ - d1
        let zs = (0...10).map { zRear + (zFront - zRear) * Double($0) / 10 }
        var rings: [[Vec3]] = []
        for z in zs {
            let u = (z - zRear) / (zFront - zRear)
            let h = (facingFront ? pow(u, 0.8) : pow(1 - u, 0.8)) * hmax + 0.002
            let bottom = min(shellHoodY(z, cx - hwid), shellHoodY(z, cx + hwid)) - 0.025
            let top = shellHoodY(z, cx) + h
            let hh = (top - bottom) / 2, cy = (top + bottom) / 2
            let prof = Mesh3.roundedRectProfile(halfWidth: hwid, halfHeight: hh, cornerRadius: min(0.025, hh * 0.9), corner: 4)
            rings.append(prof.map { Vec3(cx + $0.x, cy + $0.y, z) })
        }
        var scoop = Mesh3()
        _ = scoop.loft(rings: rings, capStart: true, capEnd: true)
        scoop = scoop.smoothed(creaseDegrees: 45)
        // Dark mouth on the open face.
        let zm = facingFront ? zFront : zRear
        let yb = shellHoodY(zm, cx) + 0.008, yt = shellHoodY(zm, cx) + hmax - 0.008
        let mouth = facingFront
            ? carBox(Vec3(cx - hwid + 0.03, yb, zm), Vec3(cx + hwid - 0.03, yt, zm + 0.004))
            : carBox(Vec3(cx - hwid + 0.03, yb, zm - 0.004), Vec3(cx + hwid - 0.03, yt, zm))
        if st.scoop == 1 {
            a.body.append(scoop); a.underbody.append(mouth)
        } else {
            a.body.append(scoop); a.body.append(scoop.mirrored(.x))
            a.underbody.append(mouth); a.underbody.append(mouth.mirrored(.x))
        }
    }

    /// Loft of z-stationed rings (all wound the same way, any concavity). Outward comes from the ring's known (x, y)
    /// orientation, NOT from centroids: `Mesh3.loft` derives it from the centroid track, which tilts when the
    /// section changes quickly between stations (the cabin-tub bulkhead ramps) and flips strips inside out.
    fileprivate func shellLoft(_ rings: [[Vec3]]) -> Mesh3 {
        var m = Mesh3()
        guard rings.count >= 2 else { return m }
        let r0 = rings[0], n = r0.count
        var area = 0.0
        for e in 0..<n { let p = r0[e], q = r0[(e + 1) % n]; area += p.x * q.y - q.x * p.y }
        let sgn: Double = area > 0 ? 1 : -1
        for s in 0..<(rings.count - 1) {
            let a = rings[s], b = rings[s + 1]
            for e in 0..<n {
                let f = (e + 1) % n
                m.addQuad(a[e], a[f], b[f], b[e], outward: cross3(a[f] - a[e], Vec3(0, 0, 1)) * sgn)
            }
        }
        func cap(_ r: [Vec3], _ out: Vec3) {
            if let tris = Mesh3.concaveCapTriangles(ring: r, axis: out) {
                for (i, j, k) in tris { m.addTriangle(r[i], r[j], r[k], outward: out) }
            } else {
                let c = r.reduce(Vec3(0, 0, 0), +) / Double(r.count)
                for e in 0..<r.count { m.addTriangle(c, r[e], r[(e + 1) % r.count], outward: out) }
            }
        }
        cap(rings[0], Vec3(0, 0, -1)); cap(rings[rings.count - 1], Vec3(0, 0, 1))
        return m
    }

    // MARK: bus upper structure

    /// Closed convex hexahedron from 8 corners (bottom 0-3 ccw, top 4-7), faces wound outward from the centroid.
    fileprivate func shellHexa(_ p: [Vec3]) -> Mesh3 {
        var m = Mesh3()
        let c = p.reduce(Vec3(0, 0, 0), +) / 8
        let faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        for f in faces {
            let q = f.map { p[$0] }, mid = (q[0] + q[1] + q[2] + q[3]) / 4
            m.addQuad(q[0], q[1], q[2], q[3], outward: mid - c)
        }
        return m
    }

    /// A wall slab along z (both sides), outer face following the plan taper, `th` thick, from y0 to y1.
    fileprivate func shellWall(_ z0: Double, _ z1: Double, _ y0: Double, _ y1: Double, th: Double = 0.10) -> Mesh3 {
        var m = Mesh3()
        let n = max(1, Int(((z1 - z0) / 0.2).rounded(.up)))
        for i in 0..<n {
            let za = z0 + (z1 - z0) * Double(i) / Double(n) - (i > 0 ? 0.01 : 0), zb = z0 + (z1 - z0) * Double(i + 1) / Double(n)
            let ha = W / 2 * planFactor(za), hb = W / 2 * planFactor(zb)
            m.append(shellHexa([Vec3(ha - th, y0, za), Vec3(ha, y0, za), Vec3(hb, y0, zb), Vec3(hb - th, y0, zb),
                                Vec3(ha - th, y1, za), Vec3(ha, y1, za), Vec3(hb, y1, zb), Vec3(hb - th, y1, zb)]))
        }
        return m
    }

    fileprivate func shellBusUpper(_ a: inout CarAccumulator) {
        let (zs, half, _, yHi) = busWindowLayout
        guard let t = shellTub, let zFirst = zs.first, let zLast = zs.last else { return }
        let yb = bp.belt * rH, ySlab = yHi - 0.02
        let top = poly(bp.top)
        let hn = W / 2 * bp.noseTaper, ht = W / 2 * bp.tailTaper

        // Roof: a closed slab from just above the side windows to the roof line; a cross-beam over the windscreen.
        func roof(_ z0: Double, _ z1: Double, _ bottom: Double) -> Mesh3 {
            loftRings(stations(z0, z1, top.map(\.0))) { z in
                Ring(hw: W / 2 * planFactor(z), bottom: bottom, top: self.interp(top, z), corner: 0.07)
            }.smoothed(creaseDegrees: 45)
        }
        a.body.append(roof(rearZ, frontZ - 0.15, ySlab))
        a.body.append(roof(frontZ - 0.30, frontZ, yb + 1.45 * rH + 0.002))

        // Side walls above the belt: rear end, front door zone, and a mullion between each pair of panes.
        let yWall1 = ySlab + 0.04
        var walls = shellWall(rearZ, zFirst - half, yb - 0.01, yWall1)
        walls.append(shellWall(zLast + half, frontZ, yb - 0.01, yWall1))
        for i in 0..<(zs.count - 1) { walls.append(shellWall(zs[i] + half, zs[i + 1] - half, yb - 0.01, yWall1)) }
        // The middle door pair sits over a window bay: give it a solid wall to be stamped on (the class cues need a surface).
        let door = busDoorLayout
        walls.append(shellWall(door.zMiddle - 0.623, door.zMiddle + 0.617, yb - 0.013, door.yTop + 0.13))
        a.body.append(walls); a.body.append(walls.mirrored(.x))

        // End panels below the glass: nose panel, rear panel, and the posts either side of the screens.
        let yLo = t.yFl - 0.05
        // Built as right halves + mirror (a box spanning both signs is not bit-symmetric after bevelling).
        func pair(_ m: Mesh3) { a.body.append(m); a.body.append(m.mirrored(.x)) }
        pair(carBevelBox(Vec3(-0.005, yLo, frontZ - 0.10), Vec3(hn, yb, frontZ), 0.01))
        pair(carBevelBox(Vec3(-0.005, yLo, rearZ), Vec3(ht, yb + 0.40, rearZ + 0.10), 0.01))
        let xf0 = hn - 0.17, xr0 = ht - 0.45
        pair(carBevelBox(Vec3(xf0, yb - 0.01, frontZ - 0.10), Vec3(hn, yWall1, frontZ), 0.01))
        pair(carBevelBox(Vec3(xr0, yb - 0.01, rearZ), Vec3(ht, yWall1, rearZ + 0.10), 0.01))
        pair(carBevelBox(Vec3(-0.008, yb - 0.03, frontZ - 0.09), Vec3(0.06, yWall1, frontZ - 0.01), 0.01))
    }
}
