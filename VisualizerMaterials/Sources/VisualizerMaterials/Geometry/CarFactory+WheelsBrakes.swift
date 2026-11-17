import Foundation
import simd

// plausibility: real — vented disc = two friction plates + 30 real radial vanes + hat; calliper is a bevelled housing straddling
// the disc; drum = backing plate + shell + hub. Slots are vertex-displaced grooves in the plate face. Not spun (static, world-placed).

extension CarContext {

    /// One right-hand brake about the wheel origin (wheel frame: axle +X, outboard +X), sitting inboard of the spokes.
    /// Discs are modelled static; the slots/vanes are not animated by wheel spin (documented — `a.brakes` does not spin).
    func brakeUnit(_ kind: BrakeKind) -> Mesh3 {
        let ts = tyreShape, st = wheelStyle
        let hw = ts.hw, rimR = ts.rimR
        let ya = -0.35 * hw
        var m = Mesh3()
        switch kind {
        case .disc, .discSlotted:
            let rOut = min(0.68 * rimR, rimR - 0.04), rIn = rOut - max(0.045, 0.24 * rOut)
            let tp = 0.0085, gap = 0.010
            let yb = ya + tp + gap
            // inboard plate
            m.append(WheelKit.gridLathe([(rIn, ya), (rOut, ya), (rOut, ya + tp), (rIn, ya + tp)], n: 72).smoothed(creaseDegrees: 40))
            // outboard plate (slotted face)
            let slot = kind == .discSlotted
            let r1 = rOut - 0.012, r0 = rIn + 0.012
            let loop: [WheelKit.RZ] = [(rIn, yb), (rOut, yb), (rOut, yb + tp), (r1, yb + tp), (r0, yb + tp), (rIn, yb + tp)]
            m.append(WheelKit.gridLathe(loop, n: 72) { i, j in
                (slot && (j == 3 || j == 4) && i % 3 == 0) ? (0, -0.0028) : (0, 0)
            }.smoothed(creaseDegrees: 40))
            // 30 radial cooling vanes bridging the plates (the vented gap shows at the disc rim)
            let vane = carBox(Vec3(rIn + 0.004, ya + tp - 0.0005, -0.0035), Vec3(rOut - 0.003, yb + 0.0005, 0.0035))
            m.append(vane.radialArray(count: 30))
            // hat
            let hatR = min(0.075, rIn - 0.006)
            m.append(WheelKit.lathe([(0, ya), (rIn + 0.004, ya), (rIn + 0.004, ya + 0.020), (hatR, ya + 0.034), (0, ya + 0.034)],
                                    segments: 48).smoothed(creaseDegrees: 40))
            // calliper: housing straddling the disc over the rim, with a bulkier outboard piston body
            var cal = carBevelBox(Vec3(rOut - 0.050, ya - 0.012, -0.056), Vec3(rOut + 0.012, yb + tp + 0.030, 0.056), 0.006)
            cal.append(carBevelBox(Vec3(rOut - 0.044, ya - 0.034, -0.046), Vec3(rOut + 0.004, ya - 0.010, 0.046), 0.005))
            cal.append(carBevelBox(Vec3(rOut - 0.012, yb + tp + 0.008, -0.048), Vec3(rOut + 0.014, yb + tp + 0.034, 0.048), 0.004))
            // Front callipers trail the axle (7 o'clock-ish → mirrored side-agnostic), rear ones lead it.
            let up = 38.0 * .pi / 180, sgn = (kind == .discSlotted || kind == .disc) ? -1.0 : 1.0
            m.append(placeInLatheFrame(cal, elevation: up, towardZ: sgn))
        case .drum:
            let rd = 0.62 * rimR
            m.append(WheelKit.lathe([(0, ya), (rd + 0.014, ya), (rd + 0.014, ya + 0.006), (rd, ya + 0.006), (rd, ya + 0.052),
                                     (rd * 0.70, ya + 0.060), (0.060, ya + 0.060), (0.060, ya + 0.078), (0, ya + 0.078)],
                                    segments: 64).smoothed(creaseDegrees: 40))
            // cooling-fin ring on the drum shell
            let fin = carBox(Vec3(rd - 0.001, ya + 0.012, -0.004), Vec3(rd + 0.010, ya + 0.050, 0.004))
            m.append(fin.radialArray(count: 20))
            _ = st
        }
        return WheelKit.toWheel(m)
    }

    /// Rotate a part authored in the lathe frame (x radial, y axial) about the axle so that its radial axis points at
    /// `elevation` above the horizontal toward ±Z (the wheel-frame direction the calliper hangs at).
    private func placeInLatheFrame(_ part: Mesh3, elevation: Double, towardZ: Double) -> Mesh3 {
        // wheel-frame radial direction wanted: (dy, dz) = (sin e, towardZ·cos e). Lathe radial +x maps to wheel (dy, dz) = (−c, −s)
        // after rotatedY(α) then rotatedZ(−π/2) — solve for α.
        let dy = sin(elevation), dz = towardZ * cos(elevation)
        return part.rotatedY(atan2(-dz, -dy))
    }

    // MARK: wheel-well liner

    /// Dark liner behind one tyre, wheel frame (axle at the origin): an arc-shaped shell hugging the tyre at 35 mm clearance
    /// (arch radius is R + 60 mm) plus an inboard wall panel. Spans ±85° about the vertical so it stays inside the arch hole.
    func wellLinerUnit() -> Mesh3 {
        let hw = wheelW / 2
        let r1 = R + 0.035, r2 = R + 0.047, A = 85.0 * Double.pi / 180
        var m = WheelKit.arcSlab(cy: 0, cz: 0, r0: r1, r1: r2, a0: -A, a1: A, x0: -(hw + 0.035), x1: hw - 0.012, steps: 30)
        m.append(WheelKit.arcSlab(cy: 0, cz: 0, r0: 0.45 * R, r1: r2 - 0.005, a0: -A * 0.97, a1: A * 0.97, x0: -(hw + 0.038), x1: -(hw + 0.027), steps: 30))
        return m
    }
}
