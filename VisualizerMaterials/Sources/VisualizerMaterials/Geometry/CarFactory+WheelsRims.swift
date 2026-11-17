import Foundation
import simd

// plausibility: real — rims are a hollow barrel with flanges plus lofted spokes / stamped steel faces / lug nuts; the
// centre cap and badge disc are separate chrome lathes. No textures stand in for any shape.

extension CarContext {

    /// Right-hand rim about the origin, axle +X, outer face +X. Closed solids only (barrel, face, spokes, nuts, rings).
    /// Rim set back from the tyre bulge: flange face at 0.93·halfWidth, spoke faces at ~0.85–0.95·halfWidth, hub nose to the
    /// tyre-face plane where the (non-spinning, chrome) centre cap sits.
    func rimUnit() -> Mesh3 {
        let st = wheelStyle, ts = tyreShape
        let hw = ts.hw, rimR = ts.rimR, lip = 0.020 * ts.s
        var m = Mesh3()
        let barrel: [WheelKit.RZ] = [
            (rimR - 0.006, -0.93 * hw), (rimR + lip, -0.93 * hw), (rimR + lip, -0.88 * hw), (rimR, -0.82 * hw),
            (rimR, 0.82 * hw), (rimR + lip, 0.88 * hw), (rimR + lip, 0.93 * hw), (rimR - 0.006, 0.93 * hw)]
        m.append(WheelKit.gridLathe(barrel, n: 72).smoothed(creaseDegrees: 40))
        switch st.design {
        case .alloy:       m.append(alloyFace(st, ts))
        case .steelRibbed: m.append(steelFace(st, ts, ribs: 10))
        case .steelBus:    m.append(steelFace(st, ts, ribs: 0))
        case .classicDog:  m.append(steelFace(st, ts, ribs: 0))
        }
        if st.whitewall { m.append(whitewallRing(ts)) }
        if st.design == .classicDog {
            // Chrome "beauty ring" on the flange. Lives in the rim mesh (not a.chrome) so it spins with the wheel and stays
            // clear of testWheelsClearOfBodyShell, which forbids chrome vertices inside the tyre cylinder.
            let r1 = rimR - 0.022, r2 = rimR + 0.020, yb = 0.90 * hw, yt = 0.968 * hw
            let ring: [WheelKit.RZ] = [(r1, yb), (r2, yb), (r2, yt - 0.004), (r2 - 0.004, yt), (r1 + 0.004, yt), (r1, yt - 0.004)]
            m.append(WheelKit.gridLathe(ring, n: 72).smoothed(creaseDegrees: 40))
        }
        return WheelKit.toWheel(m)
    }

    // MARK: alloy

    private func alloyFace(_ st: WheelStyle, _ ts: TyreShape) -> Mesh3 {
        let hw = ts.hw, rimR = ts.rimR
        var m = Mesh3()
        let rb = st.pcd + 0.0175
        // Hub boss, rising to the tyre-face plane so the chrome cap sits on it.
        m.append(WheelKit.lathe([(0, 0.78 * hw), (rb, 0.78 * hw), (rb, hw - 0.006), (rb - 0.006, hw + 0.0004), (0, hw + 0.0004)],
                                segments: 48).smoothed(creaseDegrees: 40))
        let u0 = rb * 0.5, u1 = rimR - 0.004, yRoot = 0.84 * hw, yTip = 0.86 * hw
        func spoke(zRoot: Double, zTip: Double) -> Mesh3 {
            let K = 7
            var rings: [[Vec3]] = []
            for k in 0..<K {
                let t = Double(k) / Double(K - 1)
                let bw = st.bwRoot + (st.bwTip - st.bwRoot) * t, th = st.thick * (1 - 0.2 * t)
                let sec = Mesh3.roundedRectProfile(halfWidth: bw, halfHeight: th, cornerRadius: min(0.006, th * 0.8), corner: 2)
                let u = u0 + (u1 - u0) * t, y = yRoot + (yTip - yRoot) * t
                let z = zRoot + (zTip - zRoot) * t + st.lean * t * t
                rings.append(sec.map { Vec3(u, y + $0.y, z + $0.x) })
            }
            var s = Mesh3(); _ = s.loft(rings: rings)
            return s.smoothed(creaseDegrees: 40)
        }
        var slot = Mesh3()
        if st.twin {
            slot.append(spoke(zRoot: 0.020, zTip: 0.031)); slot.append(spoke(zRoot: -0.020, zTip: -0.031))
        } else {
            slot.append(spoke(zRoot: 0, zTip: 0))
        }
        m.append(slot.radialArray(count: st.spokes))
        // Lug nuts on the boss.
        let nut = WheelKit.hexNut(r: 0.0105, h: 0.015)
        for k in 0..<st.lugs {
            let a = 2 * Double.pi * (Double(k) + 0.5) / Double(st.lugs)
            m.append(nut.translated(by: Vec3(st.pcd * cos(a), hw - 0.001, st.pcd * sin(a))).smoothed(creaseDegrees: 50))
        }
        return m
    }

    // MARK: steel (taxi, van, bus, 1950s)

    private func steelFace(_ st: WheelStyle, _ ts: TyreShape, ribs: Int) -> Mesh3 {
        let hw = ts.hw, rimR = ts.rimR
        var m = Mesh3()
        let yc = 0.80 * hw, ye = 0.86 * hw
        func fy(_ r: Double) -> Double {
            let t = min(max(r / (rimR - 0.01), 0), 1)
            return yc + (ye - yc) * (t * t * (3 - 2 * t)) + 0.006 * exp(-pow((r - 0.58 * rimR) / 0.014, 2))
        }
        let rEnd = rimR - 0.004, steps = 14
        let rs = (0...steps).map { rEnd * Double($0) / Double(steps) }
        var prof: [WheelKit.RZ] = rs.map { ($0, fy($0) - 0.004) }
        prof += rs.reversed().map { ($0, fy($0) + 0.004) }
        m.append(WheelKit.lathe(prof, segments: 72).smoothed(creaseDegrees: 40))
        // Neck up to the tyre-face plane under the chrome cap.
        let capR = capRadius(st), rn = capR - 0.004
        m.append(WheelKit.lathe([(0, yc - 0.002), (rn, yc - 0.002), (rn, hw + 0.0004), (0, hw + 0.0004)], segments: 48)
            .smoothed(creaseDegrees: 40))
        if ribs > 0 {
            let rib = carBevelBox(Vec3(0.07, yc - 0.004, -0.006), Vec3(rimR - 0.03, ye + 0.008, 0.006), 0.003)
            m.append(rib.radialArray(count: ribs))
        }
        if st.design == .steelBus {
            // Ten big lug-nut bosses on the 285.75 mm bolt circle + five stamped hand-hole bezels between them.
            let collar = WheelKit.lathe([(0, 0), (0.030, 0), (0.030, 0.008), (0.022, 0.010), (0, 0.010)], segments: 24)
                .smoothed(creaseDegrees: 50)
            let nut = WheelKit.hexNut(r: 0.021, h: 0.030)
            for k in 0..<st.lugs {
                let a = 2 * Double.pi * Double(k) / Double(st.lugs)
                let base = Vec3(st.pcd * cos(a), fy(st.pcd) + 0.003, st.pcd * sin(a))
                m.append(collar.translated(by: base))
                m.append(nut.translated(by: base + Vec3(0, 0.009, 0)).smoothed(creaseDegrees: 50))
            }
            let bez: [WheelKit.RZ] = [(0.026, 0), (0.040, 0), (0.040, 0.012), (0.026, 0.012)]
            let bezel = WheelKit.gridLathe(bez, n: 24).smoothed(creaseDegrees: 50)
            for k in 0..<5 {
                let a = 2 * Double.pi * (Double(k) + 0.5) / 5
                let rr = rimR * 0.76
                m.append(bezel.translated(by: Vec3(rr * cos(a), fy(rr) + 0.003, rr * sin(a))))
            }
        } else {
            let nut = WheelKit.hexNut(r: 0.0105, h: 0.016)
            for k in 0..<st.lugs {
                let a = 2 * Double.pi * (Double(k) + 0.5) / Double(st.lugs)
                m.append(nut.translated(by: Vec3(st.pcd * cos(a), fy(st.pcd) + 0.003, st.pcd * sin(a))).smoothed(creaseDegrees: 50))
            }
        }
        return m
    }

    /// 1950s whitewall: a thin band proud of the outboard sidewall by ~2 mm. Merged into the RIM mesh (spins with the wheel);
    /// a host that wants it white tints `rims`, one that does not simply hides it. Documented trade-off: there is no
    /// per-wheel white material in `CarParts`.
    private func whitewallRing(_ ts: TyreShape) -> Mesh3 {
        let ra = ts.r0 + 0.10 * (ts.r1 - ts.r0), rb = ts.r0 + 0.66 * (ts.r1 - ts.r0)
        let rs = (0...6).map { ra + (rb - ra) * Double($0) / 6 }
        var loop: [WheelKit.RZ] = rs.map { ($0, ts.wallY($0) - 0.003) }
        loop += rs.reversed().map { ($0, ts.wallY($0) + 0.0022) }
        return WheelKit.gridLathe(loop, n: 72).smoothed(creaseDegrees: 40)
    }

    // MARK: centre cap (chrome, axisymmetric, stands wholly outboard of the tyre face plane)

    func capRadius(_ st: WheelStyle) -> Double {
        switch st.design {
        case .alloy: return max(0.035, st.pcd - 0.014)
        case .steelRibbed: return 0.050
        case .steelBus: return 0.070
        case .classicDog: return 0.094
        }
    }

    /// Right-hand hub cap with a raised badge disc, about the wheel origin. Its lowest vertex is at x = wheelW/2 so it never
    /// enters the tyre cylinder (CarFactoryTests.testWheelsClearOfBodyShell checks chrome against it).
    func centreCapUnit() -> Mesh3 {
        let st = wheelStyle, y0 = wheelW / 2 + 0.0002, rc = capRadius(st)
        let prof: [WheelKit.RZ]
        if st.design == .classicDog {
            prof = [(0, y0), (rc, y0), (rc, y0 + 0.004), (rc - 0.006, y0 + 0.014), (0.060, y0 + 0.024), (0.030, y0 + 0.030),
                    (0.022, y0 + 0.0315), (0.022, y0 + 0.035), (0, y0 + 0.035)]
        } else {
            let rb2 = rc * 0.42
            prof = [(0, y0), (rc, y0), (rc, y0 + 0.003), (rc - 0.004, y0 + 0.008), (rc * 0.6, y0 + 0.0115),
                    (rb2, y0 + 0.0115), (rb2, y0 + 0.0145), (0, y0 + 0.0145)]
        }
        return WheelKit.toWheel(WheelKit.lathe(prof, segments: 64).smoothed(creaseDegrees: 40))
    }
}
