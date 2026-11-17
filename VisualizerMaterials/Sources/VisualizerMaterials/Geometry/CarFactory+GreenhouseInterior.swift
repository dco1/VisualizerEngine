import Foundation
import simd

// plausibility: real — dash, seats, wheel, door cards are authored solids at measured cabin positions
// (Camry: hip point ~0.55 m, seat back 0.52 m + 0.16 m headrest, wheel Ø0.37 m); nothing is textured.

extension CarContext {

    /// Windscreen-plane height at z (front rake), used to keep the dash below the glass.
    private func wsPlaneY(_ m: GHModel, _ z: Double) -> Double { m.yB + (m.zF - z) * (m.roofY - m.yB) / max(m.zF - m.z1, 1e-6) }

    /// Rear row seat z (back of the cushion) such that the headrest clears the roof / rear glass.
    func rearSeatZ(_ m: GHModel, front zh: Double, hip: Double, gap: Double) -> Double {
        var z = zh - gap
        let forwardLimit = zh - 0.50
        while z < forwardLimit {
            let head = Vec3(0, hip + 0.70 * cos(0.25), z - 0.70 * sin(0.25) - 0.05)
            if head.y + 0.06 <= m.yTop(head.z) { break }
            z += 0.02
        }
        // still too tall: step back toward gap until clear (steep roofs)
        return min(z, forwardLimit)
    }

    func buildInterior(_ a: inout CarAccumulator, _ m: GHModel) {
        let g = GHInteriorLayout(self, m)
        a.interior.append(contentsOf: g.meshes)
    }
}

extension Array where Element == Mesh3 {
    // helper so `a.interior.append(contentsOf:)` reads naturally
}

extension Mesh3 {
    mutating func append(contentsOf ms: [Mesh3]) { for x in ms { append(x) } }
}

/// The interior as a list of closed solids (kept apart so tests can measure seats, wheel etc.).
struct GHInteriorLayout {
    var dash = Mesh3(), binnacle = Mesh3(), stack = Mesh3()
    var wheel = Mesh3(), column = Mesh3()
    var frontSeats: [Mesh3] = [], rearSeats: [Mesh3] = []
    var doorCards: [Mesh3] = [], other: [Mesh3] = []
    var wheelCentre = Vec3.zero, wheelAxis = Vec3(0, 0, 1)
    var hipY = 0.0, zHipFront = 0.0, zHipRear: Double? = nil

    /// Hip drop below the beltline per class (seat cushion top = belt - this).
    static func hipDrop(_ k: GHKind) -> Double { switch k { case .van: return 0.30; case .coupe: return 0.45; default: return 0.40 } }
    /// THE interior floor height: underside of the seat cushions (hip - 0.13). All interior geometry is at/above it.
    static func floorY(_ m: GHModel) -> Double { m.yB - hipDrop(m.kind) - 0.13 }

    var meshes: [Mesh3] { [dash, binnacle, stack, wheel, column] + frontSeats + rearSeats + doorCards + other }

    init(_ c: CarContext, _ m: GHModel) {
        let yB = m.yB
        hipY = yB - GHInteriorLayout.hipDrop(m.kind)
        let wsY: (Double) -> Double = { z in yB + (m.zF - z) * (m.roofY - yB) / max(m.zF - m.z1, 1e-6) }

        // ── dashboard: a side-view convex profile extruded across the cabin ──
        let w = m.hwBase(m.zF - 0.3) - 0.07
        let zf = m.zF - 0.10, ztop1 = m.zF - 0.30, zb = m.zF - 0.46
        let yf = min(yB + 0.035, wsY(zf) - 0.016)
        let prof: [Vec2] = [Vec2(zf, yB - 0.12), Vec2(zf, yf), Vec2(ztop1, min(yB + 0.05, wsY(ztop1) - 0.02)), Vec2(zb, yB + 0.03), Vec2(zb, yB - 0.12)]
        dash = GHKit.prism(bottom: prof.map { Vec3(-w, $0.y, $0.x) }, top: prof.map { Vec3(w, $0.y, $0.x) }).smoothed(creaseDegrees: 35)

        // driver on the left (x < 0)
        let xd = -0.36
        let bt = min(yB + 0.105, wsY(m.zF - 0.27) - 0.02)
        binnacle = carBevelBox(Vec3(xd - 0.17, yB - 0.03, m.zF - 0.47), Vec3(xd + 0.17, bt, m.zF - 0.26), 0.02)
        stack = carBevelBox(Vec3(-0.10, yB - 0.13, m.zF - 0.47), Vec3(0.10, min(yB + 0.04, wsY(m.zF - 0.32) - 0.02), m.zF - 0.32), 0.015)

        // ── steering wheel Ø0.37, column raked ~23° ──
        let tilt = 0.41
        let n = normalize3(Vec3(0, sin(tilt), -cos(tilt)))                // towards the driver
        let wc = Vec3(xd, hipY + 0.36, m.zF - 0.60)
        wheelCentre = wc; wheelAxis = n
        let up = normalize3(cross3(n, Vec3(1, 0, 0)))                       // in-plane up
        let Rc = 0.171, rt = 0.014
        wheel = GHKit.torus(centre: wc, axis: n, big: Rc, small: rt, around: 36, tube: 10)
        wheel.append(Mesh3.cylinder(from: wc - n * 0.02, to: wc + n * 0.035, radius: 0.045))
        for d in [Vec3(1, 0, 0), Vec3(-1, 0, 0), -up] {
            wheel.append(Mesh3.cylinder(from: wc + d * 0.03, to: wc + d * (Rc - 0.004), radius: 0.010))
        }
        let colEnd = Vec3(xd, wc.y - 0.10, m.zF - 0.40)
        column = Mesh3.cylinder(from: wc - n * 0.02, to: colEnd, radius: 0.028)

        // ── seats ──
        let zh = m.zF - 1.02                                               // back of the front cushion
        zHipFront = zh
        let rec = -0.25
        func seat(cx: Double, z: Double, halfW: Double, heads: [Double]) -> [Mesh3] {
            let cushion = carBevelBox(Vec3(cx - halfW, hipY - 0.13, z - 0.02), Vec3(cx + halfW, hipY, z + 0.48), 0.03)
            var back = carBevelBox(Vec3(-halfW, 0, -0.065), Vec3(halfW, 0.52, 0.065), 0.035)
                .rotatedX(rec).translated(by: Vec3(cx, hipY, z - 0.04))
            var out = [cushion]
            out.append(back)
            for hx in heads {
                let hr = carBevelBox(Vec3(hx - 0.11, 0.55, -0.05), Vec3(hx + 0.11, 0.70, 0.045), 0.03)
                    .rotatedX(rec).translated(by: Vec3(cx, hipY, z - 0.04))
                out.append(hr)
            }
            back = Mesh3()
            return out
        }
        let sx: Double = m.kind == .van ? 0.38 : 0.36
        frontSeats = seat(cx: -sx, z: zh, halfW: 0.24, heads: [0]) + seat(cx: sx, z: zh, halfW: 0.24, heads: [0])

        if m.kind != .van {
            let gap = m.kind == .pickup ? 0.60 : 0.85
            let zr = c.rearSeatZ(m, front: zh, hip: hipY, gap: gap)
            zHipRear = zr
            let hwAvail = m.hwBase(zr - 0.1) - 0.09
            let hh = min(0.60, hwAvail)
            let cushion = carBevelBox(Vec3(-hh, hipY - 0.13, zr - 0.02), Vec3(hh, hipY, zr + 0.48), 0.03)
            let back = carBevelBox(Vec3(-hh, 0, -0.065), Vec3(hh, 0.52, 0.065), 0.035).rotatedX(rec).translated(by: Vec3(0, hipY, zr - 0.04))
            rearSeats = [cushion, back]
            for hx in [-(hh - 0.20), hh - 0.20] {
                rearSeats.append(carBevelBox(Vec3(hx - 0.11, 0.55, -0.05), Vec3(hx + 0.11, 0.70, 0.045), 0.03)
                    .rotatedX(rec).translated(by: Vec3(0, hipY, zr - 0.04)))
            }
            // parcel shelf (notch-back boxes only)
            if m.kind == .sedan || m.kind == .coupe {
                let zs1 = zr - 0.18, zs0 = m.zR + 0.30
                if zs1 > zs0 + 0.1 {
                    other.append(carBevelBox(Vec3(-(m.hwBase(zs0) - 0.08), yB - 0.015, zs0), Vec3(m.hwBase(zs0) - 0.08, yB + 0.018, zs1), 0.008))
                }
            }
        } else {
            // cargo bulkhead behind the seats
            let hw = m.sideHW(m.zR + 0.06, m.roofY - 0.11) - 0.03
            other.append(carBevelBox(Vec3(-hw, yB - 0.35, m.zR + 0.03), Vec3(hw, m.roofY - 0.11, m.zR + 0.09), 0.01))
        }

        // ── door cards (inside the glass, bottoms clear of the tyres) ──
        let yBot = max(yB - 0.30, 2 * c.R + 0.03)
        func card(_ z0: Double, _ z1: Double) {
            guard z1 - z0 > 0.2 else { return }
            let zm = (z0 + z1) / 2
            let xo = m.hwBase(zm) - 0.035
            let b = carBevelBox(Vec3(xo - 0.045, yBot, z0), Vec3(xo, yB + 0.018, z1), 0.012)
            doorCards.append(b); doorCards.append(b.mirrored(.x))
        }
        switch m.kind {
        case .sedan, .hatch, .suv, .pickup:
            card(m.zB + 0.06, m.zF - 0.16)
            card(max(m.zRD - 0.15, m.zR + 0.35), m.zB - 0.06)
        case .coupe: card(m.zDoorEndCoupe - 0.25, m.zF - 0.16)
        case .van: card(m.zR + 0.35, m.zF - 0.16)
        case .bus: break
        }

        // ── headliner (dark) under the roof panel ──
        let hz0 = m.z2 + 0.10, hz1 = m.z1 - 0.10
        if hz1 - hz0 > 0.2 {
            var zs = [hz0]; var q = hz0 + 0.3; while q < hz1 - 0.05 { zs.append(q); q += 0.3 }; zs.append(hz1)
            other.append(GHKit.slab(zs: zs, hw: { m.hwRoofPanel($0) - 0.12 }, nx: 4, thickness: 0.012) { _, _ in m.roofY - 0.075 })
        }
    }
}
