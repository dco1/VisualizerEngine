import Foundation
import simd

// plausibility: real — tyres, rims, discs, drums, callipers, liners, undertray and exhaust are all closed solids sized from
// the real tyre designation (see CarFactory+Styles.swift); tread blocks, grooves and sidewall lettering are vertex-displaced
// lathe geometry (no textures), spokes are lofted solids.

// MARK: - Wheel style table

enum WheelClass { case sedan, taxi, hatch, coupe, suv, pickup, van, bus, classic }
enum RimDesign { case alloy, steelRibbed, steelBus, classicDog }
enum BrakeKind { case discSlotted, disc, drum }

/// Everything era/class-dependent about a wheel, in one place.
struct WheelStyle {
    var cls: WheelClass
    var rimFrac: Double            // rim radius / tyre radius (from the real designation, e.g. 235/45R18 → 0.66)
    var design: RimDesign
    var spokes = 5, twin = false
    var bwRoot = 0.016, bwTip = 0.020, thick = 0.010, lean = 0.0   // spoke half-width at root/tip, half-thickness, tangential lean
    var lugs = 5, pcd = 0.057      // lug-nut count and pitch-circle radius (m)
    var whitewall = false, letters = true
    var front = BrakeKind.disc, rear = BrakeKind.disc
}

extension CarContext {
    var wheelClass: WheelClass {
        if bp.boxy { return .bus }
        if classic && abs(bp.length - 4.99) < 1e-6 { return .classic }
        if bp.length > 5.5 { return .pickup }
        if abs(bp.length - 4.973) < 0.002 { return .van }
        if abs(bp.length - 4.811) < 0.002 { return .coupe }
        if bp.length < 4.4 { return .hatch }
        if abs(bp.length - 4.6) < 0.002 { return .suv }
        return bp.roofSign ? .taxi : .sedan
    }

    var wheelStyle: WheelStyle {
        let c = wheelClass
        let modern = era == .modern
        var s: WheelStyle
        switch c {
        case .bus:
            s = WheelStyle(cls: c, rimFrac: 0.57, design: .steelBus, lugs: 10, pcd: 0.143, letters: true, front: .drum, rear: .drum)
        case .classic:
            s = WheelStyle(cls: c, rimFrac: 0.53, design: .classicDog, lugs: 5, pcd: 0.060, whitewall: true, letters: false, front: .drum, rear: .drum)
        case .taxi:
            s = WheelStyle(cls: c, rimFrac: 0.60, design: .steelRibbed, lugs: 5, pcd: 0.057)
        case .van:
            s = WheelStyle(cls: c, rimFrac: 0.59, design: .steelRibbed, lugs: 5, pcd: 0.0635)
        case .sedan:
            s = WheelStyle(cls: c, rimFrac: 0.66, design: .alloy, spokes: modern ? 5 : 6, twin: modern,
                           bwRoot: modern ? 0.013 : 0.032, bwTip: modern ? 0.017 : 0.040, thick: modern ? 0.010 : 0.009)
        case .hatch:
            s = WheelStyle(cls: c, rimFrac: 0.64, design: .alloy, spokes: modern ? 7 : 5, bwRoot: modern ? 0.014 : 0.034,
                           bwTip: modern ? 0.021 : 0.042, thick: modern ? 0.010 : 0.009, lean: modern ? 0.03 : 0, pcd: 0.050)
            s.rear = modern ? .disc : .drum
        case .coupe:
            s = WheelStyle(cls: c, rimFrac: 0.70, design: .alloy, spokes: 5, twin: false, bwRoot: modern ? 0.020 : 0.030,
                           bwTip: modern ? 0.031 : 0.040, thick: 0.012, lean: modern ? 0.02 : 0, pcd: 0.0572)
            s.front = .discSlotted; s.rear = .disc
        case .suv:
            s = WheelStyle(cls: c, rimFrac: 0.60, design: .alloy, spokes: modern ? 6 : 8, bwRoot: modern ? 0.022 : 0.026,
                           bwTip: modern ? 0.031 : 0.034, thick: 0.013, pcd: 0.0572)
        case .pickup:
            s = WheelStyle(cls: c, rimFrac: 0.56, design: .alloy, spokes: modern ? 6 : 8, bwRoot: modern ? 0.029 : 0.026,
                           bwTip: modern ? 0.040 : 0.034, thick: 0.015, lugs: 6, pcd: 0.0675)
        }
        if c == .bus { return s }
        if era == .classic1950s && c != .classic {          // any class in a 1950s world rides steel wheels with dog-dish caps
            s.design = .classicDog; s.rimFrac = min(s.rimFrac, 0.56); s.letters = false; s.whitewall = true
            s.front = .drum; s.rear = .drum
        } else if era == .nineties {
            s.rimFrac = max(0.5, s.rimFrac - 0.05)
            if s.rear == .disc && c != .coupe && c != .pickup { s.rear = .drum }
            if s.front == .discSlotted { s.front = .disc }
        }
        return s
    }

    // MARK: tyre shape shared by tyre, whitewall and rim

    struct TyreShape {
        var hw: Double, R: Double, rimR: Double, s: Double, r0: Double, r1: Double
        func wallY(_ r: Double) -> Double {
            let f = min(max((r - r0) / max(r1 - r0, 1e-6), 0), 1)
            return hw * (0.86 + 0.14 * sin(.pi * (0.04 + 0.92 * f)))
        }
    }
    var tyreShape: TyreShape {
        let st = wheelStyle
        let rimR = st.rimFrac * R, H = R - rimR
        let s = min(1, H / 0.105)
        return TyreShape(hw: wheelW / 2, R: R, rimR: rimR, s: s, r0: rimR + 0.046 * s, r1: max(R - 0.016 * s, rimR + 0.046 * s + 0.01))
    }

    // MARK: build step

    /// Tyres, rims, brakes, wheel wells, undertray, exhaust. Fills `a.tyres/rims/tyreUnit/rimUnit/wheelCentres`,
    /// `a.brakes`, `a.underbody`, `a.chrome`.
    ///
    /// Contract kept from the original: `tyreUnit` / `rimUnit` are ONE right-hand wheel at the origin, axle +X, outer face +X,
    /// so a host can instance + spin them about X. Everything that must NOT spin is world-placed: `a.brakes` (discs/callipers/
    /// drums), the centre caps in `a.chrome` (axisymmetric, so a spinning host sees no difference), liners in `a.underbody`.
    /// The whitewall ring (1950s) and chrome beauty ring are merged into the RIM mesh so they spin with the wheel.
    func buildWheels(_ a: inout CarAccumulator) {
        let st = wheelStyle
        a.tyreUnit = tyreUnit(); a.rimUnit = rimUnit()
        let cap = centreCapUnit(), liner = wellLinerUnit()
        let brakeF = brakeUnit(st.front), brakeR = brakeUnit(st.rear)
        let probe = BodyProbe(a.body)
        for (ai, az) in axles.enumerated() {
            let half = wheelTrackHalf(probe, az)
            for s in [1.0, -1.0] {
                let p = Vec3(s * half, R, az)
                func put(_ m: Mesh3) -> Mesh3 { (s > 0 ? m : m.mirrored(.x)).translated(by: p) }
                a.tyres.append(put(a.tyreUnit))
                a.rims.append(put(a.rimUnit))
                a.wheelCentres.append(p)
                a.chrome.append(put(cap))
                a.underbody.append(put(liner))
                a.brakes.append(put(ai == 0 ? brakeF : brakeR))
            }
        }
        buildUnderbody(&a)
    }

    /// Half-track at one axle, measured on the finished paint shell: ray-cast inward at five heights above the arch
    /// (the skin undulates +-2 cm with the arch lip and creases), take the MEDIAN skin x as the fender line, and put the tyre's
    /// outer face 12 mm inside it. Falls back to `trackHalf` if the shell is not hit.
    func wheelTrackHalf(_ probe: BodyProbe, _ z: Double) -> Double {
        var xs: [Double] = []
        for dy in [0.05, 0.10, 0.15, 0.20, 0.25] {
            if let h = probe.cast(Vec3(W, 2 * R + dy, z), Vec3(-1, 0, 0)) { xs.append(h.p.x) }
        }
        guard xs.count >= 3 else { return trackHalf }
        let flank = xs.sorted()[xs.count / 2]
        guard flank > W * 0.3 else { return trackHalf }
        return min(flank - 0.012, W / 2 - 0.005) - wheelW / 2
    }

    // MARK: tyre

    /// Right-hand tyre about the origin, axle +X, outer face +X. A real hollow tyre: bore, bead, rim-protector ridge, sidewall
    /// bulge, raised lettering band (outboard), shoulder, three circumferential grooves and transverse tread notches.
    func tyreUnit() -> Mesh3 {
        let st = wheelStyle, ts = tyreShape
        let hw = ts.hw, rimR = ts.rimR, s = ts.s
        enum Tag { case plain, rib, letter(Int) }
        let letters = st.letters && (ts.r1 - ts.r0) >= 0.035
        var side: [(r: Double, y: Double)] = [
            (rimR - 0.001, 0.86 * hw), (rimR + 0.020 * s, 0.86 * hw), (rimR + 0.026 * s, 0.905 * hw), (rimR + 0.040 * s, 0.905 * hw)]
        for f in [0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0] as [Double] {     // chord follows the bulge to <0.5 mm
            let r = ts.r0 + f * (ts.r1 - ts.r0)
            side.append((r, ts.wallY(r)))
        }
        side.append((R - 0.010 * s, 0.86 * hw))      // shoulder
        // Tread, ascending y: (y/hw, radial drop, notched?). Wide (19 mm) deep (8 mm) grooves, 6 mm transverse notches.
        let half: [(Double, Double, Bool)] = [
            (0.10, 0, true), (0.30, 0, true), (0.38, 0.008, false), (0.46, 0, true), (0.66, 0, true), (0.76, 0.004, true)]
        var full: [(Double, Double, Bool)] = half.reversed().map { (-$0.0, $0.1, $0.2) }
        full.append((0, 0.009, false))
        full += half
        var loop: [WheelKit.RZ] = [], rib: [Bool] = []
        for p in side { loop.append((p.r, -p.y)); rib.append(false) }
        for p in full { loop.append((R - p.1, p.0 * hw)); rib.append(p.2) }
        for p in side.reversed() { loop.append((p.r, p.y)); rib.append(false) }
        let m = WheelKit.gridLathe(loop, n: 60) { i, j in rib[j] && i % 2 == 1 ? (-0.006, 0) : (0, 0) }
        var out = WheelKit.toWheel(m).smoothed(creaseDegrees: 40)
        if letters { out.append(sidewallLettering(ts)) }
        return out
    }

    /// Raised dot-matrix lettering on the OUTBOARD sidewall: the real size designation (e.g. "P235/45R18"), 5 rows tall
    /// (~12-15 mm), each lit pixel a small closed box 2.4 mm proud of the wall, upright at the top of the wheel. Wheel frame.
    func sidewallLettering(_ ts: TyreShape) -> Mesh3 {
        let width = Int((wheelW * 1000 / 5).rounded()) * 5
        let aspect = Int(((R - ts.rimR) / wheelW * 100 / 5).rounded()) * 5
        let inch = Int((ts.rimR * 2 / 0.0254).rounded())
        let text = Array("P\(width)/\(aspect)R\(inch)")
        let span = ts.r1 - ts.r0
        let pitch = min(0.003, span * 0.8 / 5), box = pitch * 0.8
        let rMid = ts.r0 + 0.5 * span
        var m = Mesh3()
        let charPitch = 4 * pitch, total = Double(text.count) * charPitch
        for (ci, ch) in text.enumerated() {
            guard let g = WheelKit.glyphs5[ch] else { continue }
            for row in 0..<5 { for col in 0..<3 where Array(g[row])[col] == "1" {
                let r = rMid + (2 - Double(row)) * pitch                                  // row 0 on top
                let u = (Double(ci) * charPitch + Double(col) * pitch) - total / 2        // arc length along the text
                let phi = u / rMid
                let er = Vec3(0, cos(phi), -sin(phi)), et = Vec3(0, -sin(phi), -cos(phi))
                let c = Vec3(0, r * cos(phi), -r * sin(phi))
                let x0 = ts.wallY(r) - 0.0008, x1 = ts.wallY(r) + 0.0020
                func P(_ a: Double, _ b: Double, _ x: Double) -> Vec3 { c + et * (a * box / 2) + er * (b * box / 2) + Vec3(x, 0, 0) }
                for (n, q) in [(Vec3(1, 0, 0), [(-1.0, -1.0, x1), (1, -1, x1), (1, 1, x1), (-1, 1, x1)]),
                               (Vec3(-1, 0, 0), [(-1, -1, x0), (1, -1, x0), (1, 1, x0), (-1, 1, x0)])] {
                    m.addQuad(P(q[0].0, q[0].1, q[0].2), P(q[1].0, q[1].1, q[1].2), P(q[2].0, q[2].1, q[2].2), P(q[3].0, q[3].1, q[3].2), outward: n)
                }
                for (sa, sb, n) in [(1.0, 0.0, et), (-1.0, 0.0, et * -1), (0.0, 1.0, er), (0.0, -1.0, er * -1)] {
                    let ta = sa != 0 ? (sa, -1.0) : (-1.0, sb), tb = sa != 0 ? (sa, 1.0) : (1.0, sb)
                    m.addQuad(P(ta.0, ta.1, x0), P(tb.0, tb.1, x0), P(tb.0, tb.1, x1), P(ta.0, ta.1, x1), outward: n)
                }
            } }
        }
        return m
    }
}
