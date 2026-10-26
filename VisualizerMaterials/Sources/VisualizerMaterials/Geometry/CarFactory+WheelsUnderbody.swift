import Foundation
import simd

// plausibility: real — undertray is a lofted pan following the sill/approach/departure profile, exhaust is a muffler + swept
// pipe + hollow chrome tip, tow hooks are swept loops on brackets, diffuser fins are extruded wedges. All closed solids.

extension CarContext {

    /// Underside height without the arch cut-outs: sill between the axles, easing to `endBottom` at the nose/tail
    /// (the same law `bottom(_:)` uses before it adds the arches).
    func undersideY(_ z: Double) -> Double {
        let t = min(min(frontZ - z, z - rearZ) / 0.5, 1)
        return bp.endBottom + (bp.sill - bp.endBottom) * sin(t * .pi / 2)
    }

    /// Undertray (item 8/43), exhaust (41), tow hooks and diffuser (43). Appends to `a.underbody` and `a.chrome`.
    func buildUnderbody(_ a: inout CarAccumulator) {
        let st = wheelStyle
        // ── Undertray: a dark pan hung 22 mm under the body's underside between the inboard tyre faces, with the
        //    approach / departure rise toward the ends.
        let hwTray = trackHalf - wheelW / 2 - 0.045
        var extra: [Double] = []
        var d = 0.0
        while d <= 0.5 { extra += [frontZ - 0.12 - d, rearZ + 0.12 + d]; d += 0.06 }
        let zs = stations(rearZ + 0.12, frontZ - 0.12, extra)
        a.underbody.append(loftRings(zs) { z in
            let b = self.undersideY(z)
            return Ring(hw: min(hwTray, W / 2 * self.planFactor(z) - 0.10), bottom: b - 0.022, top: b + 0.004, corner: 0.008)
        }.smoothed(creaseDegrees: 45))

        // ── Exhaust
        if wheelClass != .bus { buildExhaust(&a, st) }

        // ── Front tow hooks on the rugged classes and the classics
        if [.pickup, .suv, .classic].contains(wheelClass) { buildTowHooks(&a) }

        // ── Rear diffuser lines on the coupe
        if wheelClass == .coupe { buildDiffuser(&a) }
    }

    // MARK: exhaust

    private func exhaustLayout() -> (xs: [Double], tipR: Double, groups: [[Double]]) {
        switch wheelClass {
        case .coupe:   return ([0.44, 0.62], 0.042, [[0.44, 0.62]])                 // quad: two tips each side
        case .sedan, .taxi: return era == .modern ? ([0.44], 0.038, [[0.44]]) : ([0], 0.036, [[0]])
        case .classic: return ([0.55], 0.040, [[0.55]])
        case .pickup:  return ([0.62], 0.046, [[0.62]])                             // under-bumper, both sides
        case .suv:     return ([0.50], 0.036, [[0.50]])
        case .van, .hatch: return ([0], 0.034, [[0]])
        case .bus:     return ([], 0, [])
        }
    }

    func buildExhaust(_ a: inout CarAccumulator, _ st: WheelStyle) {
        let lay = exhaustLayout()
        let rearAxle = axles[1]
        let rm = min(0.07, bp.sill * 0.45), ym = bp.sill - 0.6 * rm
        let yTip = bp.endBottom - 0.08
        let zTipBase = rearZ + 0.07, tipLen = 0.10
        // one mirrored half (or the single centred set when x == 0)
        func one(_ x: Double, groupX: Double) -> (under: Mesh3, chrome: Mesh3) {
            var u = Mesh3(), c = Mesh3()
            // muffler can (one per group, drawn with the first tip only)
            let ov = rearAxle - rearZ, mlen = min(0.5, 0.5 * ov), zm0 = rearAxle - 0.18 * ov, zm1 = zm0 - mlen
            if x == lay.xs.first! {
                u.append(Mesh3.cylinder(from: Vec3(groupX, ym, zm0), to: Vec3(groupX, ym, zm1), radius: rm, chordError: 0.0005))
            }
            let sx = groupX + 0.4 * (x - groupX), za = zm1 + 0.02 + (x == lay.xs.first! ? 0 : 0.03), zb = zTipBase + 0.02, d = max(za - zb, 0.05)     // monotonic in z: never loops back
            let path = Curve3.bezier(Vec3(sx, ym, za), Vec3(sx, ym, za - 0.35 * d), Vec3(x, yTip, zb + 0.35 * d),
                                     Vec3(x, yTip, za - d), samples: 16)
            u.append(Mesh3.tube(along: path, radius: 0.024, segments: 12))
            // hollow chrome tip: wall 6 mm, 100 mm long, open at the rear
            let tr = lay.tipR
            let tip = WheelKit.gridLathe([(tr - 0.006, 0), (tr, 0), (tr, tipLen), (tr - 0.006, tipLen)], n: 32).smoothed(creaseDegrees: 45)
            c.append(tip.rotatedX(-.pi / 2).translated(by: Vec3(x, yTip, zTipBase)))
            return (u, c)
        }
        for x in lay.xs {
            if x == 0 {
                let r = one(0, groupX: 0); a.underbody.append(r.under); a.chrome.append(r.chrome)
            } else {
                let gx = lay.groups[0].reduce(0, +) / Double(lay.groups[0].count)
                let r = one(x, groupX: gx)
                a.underbody.append(r.under); a.chrome.append(r.chrome)
                a.underbody.append(r.under.mirrored(.x)); a.chrome.append(r.chrome.mirrored(.x))
            }
        }
        // a pickup's/SUV's pipe is simply under the bumper; the quad coupe additionally gets a mirrored second pair (xs has two entries → handled above)
    }

    // MARK: tow hooks

    func buildTowHooks(_ a: inout CarAccumulator) {
        // A bolted plate flush on the front bumper face with a thick forged eye standing proud of it (eye axis along Z).
        let out = classic ? 0.12 : 0.045
        let yb0 = bp.endBottom - 0.04
        let bumpH = bp.boxy ? 0.28 : (classic ? 0.17 : 0.16)
        let yc = yb0 + bumpH * 0.45, zf = frontZ + out
        let eye = WheelKit.gridLathe([(0.018, 0), (0.040, 0), (0.040, 0.020), (0.018, 0.020)], n: 32).smoothed(creaseDegrees: 50)
            .rotatedX(.pi / 2)
        for sx in [1.0, -1.0] {
            let x = sx * 0.40
            a.underbody.append(carBevelBox(Vec3(x - 0.055, yc - 0.055, zf - 0.004), Vec3(x + 0.055, yc + 0.055, zf + 0.010), 0.004))
            a.underbody.append(eye.translated(by: Vec3(x, yc, zf + 0.010)))
        }
    }

    // MARK: diffuser

    func buildDiffuser(_ a: inout CarAccumulator) {
        let zs = rearZ + 0.03, zf = rearZ + 0.45
        let yRearLow = bp.endBottom - 0.015, yFrontLow = max(0.09, bp.sill - 0.03)
        // ramp panel
        a.underbody.append(WheelKit.prismX([(zs, yRearLow), (zf, yFrontLow), (zf, yFrontLow - 0.012), (zs, yRearLow - 0.012)],
                                           x0: -0.36, x1: 0.36))
        // five fins hanging under the ramp
        for i in -2...2 {
            let x = Double(i) * 0.14
            a.underbody.append(WheelKit.prismX([(zs, yRearLow - 0.012), (zf, yFrontLow - 0.012), (zf, yFrontLow - 0.042),
                                                (zs, yRearLow - 0.012 - 0.030 * 0.5)],
                                               x0: x - 0.005, x1: x + 0.005))
        }
    }
}
