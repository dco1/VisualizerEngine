import Foundation
import simd

// plausibility: real — tail units are separate brake / indicator / reverse shells in a housing on the shell's own
// plan outline; bumpers are lofted along the tail taper; the third brake light is placed from the greenhouse slope.

extension CarContext {
    struct TailStyle { var h: Double, xIn: Double, wrap: Double }

    var tailStyle: TailStyle {
        if classic { return TailStyle(h: 0.15, xIn: 0.62, wrap: 0.16) }
        switch lampClass {
        case .sedan:  return TailStyle(h: 0.105, xIn: 0.44, wrap: 0.22)
        case .hatch:  return TailStyle(h: 0.150, xIn: 0.52, wrap: 0.18)
        case .suv:    return TailStyle(h: 0.230, xIn: 0.62, wrap: 0.14)
        case .pickup: return TailStyle(h: 0.260, xIn: 0.70, wrap: 0.12)
        case .van:    return TailStyle(h: 0.330, xIn: 0.72, wrap: 0.10)
        case .coupe:  return TailStyle(h: 0.090, xIn: 0.42, wrap: 0.26)
        case .bus:    return TailStyle(h: 0.300, xIn: 0.0, wrap: 0.10)
        }
    }

    func buildRear(_ a: inout CarAccumulator, _ r: inout LampRecorder) {
        let e = lampEnd(false)
        let st = lampFaceStyle, ts = tailStyle
        let hwT = hwAt(e, 0), z0 = e.z0, dir = e.dir
        let room = flankRoom(e)
        let cls = lampClass
        let y0N = bumperBottom(e)
        let bumpTop = y0N + st.bumpH
        let band = wallBand(e, 0)

        // ── Bumper ──
        var plateY = bumpTop + 0.012 + 0.018 + 0.076
        var plateZ = 0.0
        if classic {
            buildClassicRearBumper(&a, &r, e, hwT: hwT, st: st, room: room)
        } else if cls == .bus {
            plateY = y0N + st.bumpH * 0.5; plateZ = 0.07
            emit(&a, &r, .trim, .bumper, lampWrap(lampPath(e, faceX0: 0, flankD1: min(room, 0.4)).full(), chamfer: 0.03, smooth: 50) { _ in
                LampSection(y0: y0N, y1: y0N + st.bumpH, out: 0.07, inn: 0.04)
            })
            emit(&a, &r, .trim, .bumperLip, lampWrap(lampPath(e, faceX0: 0, faceX1: hwT * 0.9).full(), chamfer: 0.01) { _ in
                LampSection(y0: y0N - 0.05, y1: y0N + 0.02, out: 0.10, inn: 0.03)
            })
        } else {
            let out: Double = (cls == .pickup) ? 0.075 : (cls == .van ? 0.06 : 0.045)
            let path = lampPath(e, faceX0: 0, flankD1: room - 0.02).full()
            let body = lampWrap(path, chamfer: 0.02, smooth: 50) { s in
                let d = s.onFace ? 0 : abs(e.z0 - s.z), t = min(1, d / (room - 0.02))
                return LampSection(y0: y0N, y1: y0N + st.bumpH, out: out * (1 - 0.7 * t * t), inn: 0.04)
            }
            emit(&a, &r, cls == .pickup ? .chrome : .trim, .bumper, body)
            if cls == .pickup {
                // Step bumper: rubber step pad on the chrome blade's top face, standing proud of it.
                emit(&a, &r, .trim, .bumperLip, lampWrap(lampPath(e, faceX0: 0, faceX1: hwT * 0.82).full(), chamfer: 0.008) { _ in
                    LampSection(y0: bumpTop - 0.006, y1: bumpTop + 0.016, out: out + 0.07, inn: 0.0)
                })
                plateY = y0N + st.bumpH * 0.5; plateZ = out
            } else if cls == .van {
                plateY = y0N + st.bumpH * 0.5; plateZ = out
                emit(&a, &r, .trim, .bumperLip, lampWrap(lampPath(e, faceX0: 0, faceX1: hwT * 0.8).full(), chamfer: 0.008) { _ in
                    LampSection(y0: y0N - 0.03, y1: y0N + 0.02, out: out + 0.03, inn: 0.03)
                })
            } else {
                // Rear valance / diffuser lip.
                emit(&a, &r, .trim, .bumperLip, lampWrap(lampPath(e, faceX0: 0, faceX1: hwT * 0.80).full(), chamfer: 0.012, smooth: 50) { _ in
                    LampSection(y0: y0N - 0.035, y1: y0N + 0.045, out: out + 0.014, inn: 0.03)
                })
            }
        }

        // ── Licence plate ──
        buildPlate(&a, &r, e, y: classic ? bumpTop + 0.10 : plateY, zBase: classic ? 0.0 : plateZ)

        // ── Tail unit ──
        if cls == .bus { buildBusTail(&a, &r, e, hwT: hwT, st: st, ts: ts); return }
        let xIn = ts.xIn * hwT
        let wrap = min(ts.wrap, room * 0.5)
        let yTop = band.hi - 0.012 + (classic ? 0.03 : 0)
        var ly0 = max(yTop - ts.h, bumpTop + 0.02), ly1 = yTop
        // Reverse lamp: a small separate lens under the inboard end of the brake lens (1950s: beside it).
        var revY0 = max(ly0 - 0.050, bumpTop + 0.014), revY1 = revY0 + 0.036
        var revX0 = xIn + 0.004
        var brakeX0 = xIn
        if classic { revY0 = ly0 + 0.01; revY1 = ly1 - 0.01; revX0 = xIn + 0.004; brakeX0 = xIn + 0.095 } else { ly0 = max(ly0, revY1 + 0.008) }
        let faceEnd = hwT - 0.07
        let pad = classic ? 0.016 : 0.010
        let hy0 = min(revY0, ly0) - pad, hy1 = ly1 + pad

        let housingPath = lampPath(e, faceX0: xIn - 0.012, flankD1: wrap + 0.014)
        emitPair(&a, &r, classic ? .chrome : .trim, .tailHousing, lampWrap(housingPath, chamfer: classic ? 0.012 : 0.005) { _ in
            LampSection(y0: hy0, y1: hy1, out: classic ? 0.016 : 0.006, inn: 0.05)
        })
        // Brake / tail lens (red).
        let brakePath = lampPath(e, faceX0: brakeX0, faceX1: faceEnd)
        emitPair(&a, &r, .tailLights, .brakeLamp, lampWrap(brakePath, chamfer: 0.008) { _ in
            LampSection(y0: ly0, y1: ly1, out: 0.022, inn: -0.008)
        })
        // Indicator: amber, continues around the corner into the flank.
        let indPath = lampPath(e, faceX0: hwT - 0.058, flankD1: wrap)
        emitPair(&a, &r, .signals, .rearIndicator, lampWrap(indPath, chamfer: 0.008) { _ in
            LampSection(y0: ly0, y1: ly1, out: 0.022, inn: -0.008)
        })
        // Reverse lamp: white lens.
        let revPath = lampPath(e, faceX0: revX0, faceX1: classic ? xIn + 0.085 : xIn + min(0.15, 0.4 * (faceEnd - xIn)))
        emitPair(&a, &r, .signals, .reverseLamp, lampWrap(revPath, chamfer: 0.006) { _ in
            LampSection(y0: revY0, y1: revY1, out: 0.020, inn: -0.006)
        })

        buildThirdBrake(&a, &r, e, band: band)
    }

    // MARK: third brake light

    func buildThirdBrake(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, band: (lo: Double, hi: Double)) {
        let cp = cabinPoly
        let w = 0.34
        if !bp.boxy, cp.count >= 2, cp[1].0 - cp[0].0 > 0.15, cp[1].1 - cp[0].1 > 0.05 {
            // On the rear glass, high up its slope; box oriented to the glass plane.
            let (za, ya) = cp[0], (zb, yb) = cp[1]
            let t = 0.84
            let z = za + (zb - za) * t, y = ya + (yb - ya) * t - 0.03
            let up = simd_normalize(Vec3(0, yb - ya, zb - za)), x = Vec3(1, 0, 0)
            let n = cross3(x, up)
            let m = carBevelBox(Vec3(-w / 2, -0.014, -0.022), Vec3(w / 2, 0.014, 0.014), 0.004)
                .placed(xAxis: x, yAxis: up, zAxis: n, origin: Vec3(0, y, z))
            emit(&a, &r, .tailLights, .thirdBrake, m)
        } else {
            // Van / bus: along the roof edge of the tail face.
            let y = band.hi - 0.045
            emit(&a, &r, .tailLights, .thirdBrake, carBevelBox(Vec3(-w / 2, y - 0.014, e.z0 - 0.018), Vec3(w / 2, y + 0.014, e.z0 + 0.022), 0.004))
        }
    }

    // MARK: bus tail

    func buildBusTail(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, hwT: Double, st: LampFaceStyle, ts: TailStyle) {
        let yTop = 1.30, h = 0.34
        let xIn = hwT - 0.28
        let housing = lampPath(e, faceX0: xIn - 0.012, flankD1: 0.12)
        emitPair(&a, &r, .trim, .tailHousing, lampWrap(housing, chamfer: 0.006) { _ in
            LampSection(y0: yTop - h - 0.05, y1: yTop + 0.012, out: 0.008, inn: 0.05)
        })
        emitPair(&a, &r, .tailLights, .brakeLamp, lampWrap(lampPath(e, faceX0: xIn, faceX1: hwT - 0.07), chamfer: 0.008) { _ in
            LampSection(y0: yTop - h * 0.62, y1: yTop, out: 0.022, inn: -0.008)
        })
        emitPair(&a, &r, .signals, .rearIndicator, lampWrap(lampPath(e, faceX0: hwT - 0.058, flankD1: 0.10), chamfer: 0.008) { _ in
            LampSection(y0: yTop - h * 0.62, y1: yTop, out: 0.022, inn: -0.008)
        })
        emitPair(&a, &r, .signals, .reverseLamp, lampWrap(lampPath(e, faceX0: xIn + 0.004, faceX1: xIn + 0.15), chamfer: 0.006) { _ in
            LampSection(y0: yTop - h - 0.03, y1: yTop - h * 0.62 - 0.012, out: 0.020, inn: -0.006)
        })
        buildThirdBrake(&a, &r, e, band: wallBand(e, 0))
    }

    // MARK: 1950s rear bumper with exhaust cutouts

    func buildClassicRearBumper(_ a: inout CarAccumulator, _ r: inout LampRecorder, _ e: LampEnd, hwT: Double,
                                st: LampFaceStyle, room: Double) {
        let y0N = bumperBottom(e), z0 = e.z0, dir = e.dir
        let out = 0.075, hBlade = st.bumpH
        let xe = hwT * 0.60, rh = 0.055
        let hole0 = y0N + hBlade * 0.5 - 0.045, hole1 = y0N + hBlade * 0.5 + 0.045
        func sec(_ ya: Double, _ yb: Double, ch: Double = 0.035) -> (LStation) -> LampSection {
            { s in
                let d = s.onFace ? 0 : abs(e.z0 - s.z), t = min(1, d / (room - 0.02))
                return LampSection(y0: ya, y1: yb, out: out * (1 - 0.7 * t * t), inn: 0.02)
            }
        }
        func blade(_ path: LampPath, _ ya: Double, _ yb: Double, _ ch: Double = 0.035) -> Mesh3 {
            lampWrap(path, chamfer: ch, smooth: 55, sec(ya, yb))
        }
        let yb1 = y0N + hBlade
        emit(&a, &r, .chrome, .bumper, blade(lampPath(e, faceX0: 0, faceX1: xe - rh).full(), y0N, yb1))
        emitPair(&a, &r, .chrome, .bumper, blade(lampPath(e, faceX0: xe + rh, flankD1: room - 0.02), y0N, yb1))
        let strip = lampPath(e, faceX0: xe - rh - 0.006, faceX1: xe + rh + 0.006)
        emitPair(&a, &r, .chrome, .bumper, blade(strip, hole1 - 0.004, yb1, 0.012))
        emitPair(&a, &r, .chrome, .bumper, blade(strip, y0N, hole0 + 0.004, 0.012))
        // Exhaust outlet tip standing in each cutout.
        emitPair(&a, &r, .chrome, .exhaustTip,
                 lampTorus(at: Vec3(xe, (hole0 + hole1) / 2, z0 + dir * 0.06), major: 0.034, minor: 0.009, segsMajor: 20, segsMinor: 8))
        // Overriders.
        let bullet: [(Double, Double)] = [(0, 0), (0.046, 0), (0.054, 0.022), (0.045, 0.052), (0.028, 0.075), (0.010, 0.086), (0, 0.089)]
        emitPair(&a, &r, .chrome, .overrider, lampLathe(bullet, at: Vec3(hwT * 0.28, y0N + hBlade / 2, z0 + dir * (out - 0.03)), dirZ: dir, segments: 20))
    }
}
