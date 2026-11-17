import Foundation
import simd

// plausibility: real — bus doors (framed leaves with dark panes and a centre seal), destination/route signs and arch
// cladding are authored solids laid on the real shell surface; nothing is a texture.
// NOTE (interior): the shell is NOT hollowed. A localised BSP cut of the passenger cavity + window openings was tried
// (see report): on the welded multi-solid shell it leaves ~10k open edges, 5x the triangles and a 4 s weld, so the body
// would fail the watertight gate. The seats stay hidden behind the opaque shell until the Body step builds a hollow loft.

/// Where the bus cues sit (shared with the tests).
struct BusDoorLayout {
    var zFront: Double, zMiddle: Double      // door-pair centres
    var width: Double = 1.10
    var yBottom: Double, yTop: Double
}

extension CarContext {

    var busDoorLayout: BusDoorLayout {
        let (zs, _, _, yHi) = busWindowLayout
        // middle door: the window pane nearest z = -1.3 (clear of the rear wheel arch)
        let zm = zs.min(by: { abs($0 + 1.3) < abs($1 + 1.3) }) ?? -1.3
        return BusDoorLayout(zFront: frontZ - 0.98, zMiddle: zm, width: 1.10, yBottom: 0.80, yTop: yHi + 0.06)
    }

    func buildBusCues(_ a: inout CarAccumulator) {
        let body = CCProbe(a.body)
        let D = busDoorLayout
        var frames = Mesh3(), glass = Mesh3(), panels = Mesh3(), seals = Mesh3(), handles = Mesh3()

        // ── bi-fold double doors, right side (+x) ──
        for zc in [D.zFront, D.zMiddle] {
            let leafW = (D.width - 0.012) / 2
            for k in 0..<2 {
                let zl = zc + (k == 0 ? 1 : -1) * (0.006 + leafW / 2)         // front leaf, rear leaf
                let yMid = (D.yBottom + D.yTop) / 2
                func stamp(_ dz: Double, _ y: Double, _ w: Double, _ h: Double, proud: Double, maxDev: Double = 0.05) -> Mesh3? {
                    ccStamp(body, from: Vec3(0, y, zl + dz), dir: Vec3(1, 0, 0), w: w, h: h, back: 0.006, proud: proud, maxDev: maxDev)
                }
                let fw = 0.045
                let H = D.yTop - D.yBottom
                // frame: top, bottom, left, right bars + transom
                if let m = stamp(0, D.yTop - fw / 2, leafW, fw, proud: 0.030) { frames.append(m) }
                if let m = stamp(0, D.yBottom + fw / 2, leafW, fw, proud: 0.030) { frames.append(m) }
                for s in [-1.0, 1.0] { if let m = stamp(s * (leafW / 2 - fw / 2), yMid, fw, H - 2 * fw + 0.002, proud: 0.030) { frames.append(m) } }
                let yTr = D.yBottom + H * 0.30
                if let m = stamp(0, yTr, leafW - 2 * fw + 0.002, fw, proud: 0.030) { frames.append(m) }
                // upper dark pane
                let gy0 = yTr + fw / 2, gy1 = D.yTop - fw
                if let m = stamp(0, (gy0 + gy1) / 2, leafW - 2 * fw + 0.004, gy1 - gy0 + 0.004, proud: 0.016) { glass.append(m) }
                // lower solid panel (body colour) with a small window? plain panel
                let py0 = D.yBottom + fw, py1 = yTr - fw / 2
                if let m = stamp(0, (py0 + py1) / 2, leafW - 2 * fw + 0.004, py1 - py0 + 0.004, proud: 0.020) { panels.append(m) }
                // vertical grab handle on the meeting stile (the "handrail" seen from outside)
                let hz = zc + (k == 0 ? -1 : 1) * 0.045
                if let m = ccStamp(body, from: Vec3(0, yMid, hz), dir: Vec3(1, 0, 0), w: 0.024, h: 0.60, back: 0.006, proud: 0.050, maxDev: 0.05) { handles.append(m) }
            }
            // centre rubber seal between the leaves
            if let m = ccStamp(body, from: Vec3(0, (D.yBottom + D.yTop) / 2, zc), dir: Vec3(1, 0, 0), w: 0.016, h: D.yTop - D.yBottom, back: 0.006, proud: 0.034, maxDev: 0.05) { seals.append(m) }
            // door-header frame over both leaves (opening reveal)
            if let m = ccStamp(body, from: Vec3(0, D.yTop + 0.03, zc), dir: Vec3(1, 0, 0), w: D.width + 0.10, h: 0.060, back: 0.006, proud: 0.024, maxDev: 0.05) { frames.append(m) }
        }
        a.trim.append(frames); a.trim.append(seals)
        a.trim.append(glass)           // dark door panes: in `trim` (black gloss), not `glass`, so the greenhouse's pane-count contract (2 + 2n + 1 shells) holds
        a.trim.append(panels)          // dark lower leaf panels (kept out of the paint so the body stays left/right symmetric)
        a.chrome.append(handles)

        // ── destination sign above the windscreen + right-hand route sign ──
        let yb = bp.belt * rH
        let yTopGlass = yb + 1.45 * rH
        let sy = yTopGlass + 0.095
        let hn = (W / 2) * bp.noseTaper
        if let box = ccStamp(body, from: Vec3(0, sy, frontZ - 0.5), dir: Vec3(0, 0, 1), w: 2 * (hn - 0.30), h: 0.165, back: 0.006, proud: 0.045, maxDev: 0.05),
           let disp = ccStamp(body, from: Vec3(0, sy, frontZ - 0.5), dir: Vec3(0, 0, 1), w: 2 * (hn - 0.30) - 0.12, h: 0.105, back: 0.006, proud: 0.052, maxDev: 0.05) {
            a.trim.append(box); a.signals.append(disp)
        }
        let (zs, _, _, yHi) = busWindowLayout
        let zr = (zs.last ?? frontZ - 2.0) + 0.20
        let ry = yHi + 0.17
        if let box = ccStamp(body, from: Vec3(0, ry, zr), dir: Vec3(1, 0, 0), w: 0.95, h: 0.20, back: 0.006, proud: 0.028, maxDev: 0.05),
           let disp = ccStamp(body, from: Vec3(0, ry, zr), dir: Vec3(1, 0, 0), w: 0.83, h: 0.13, back: 0.006, proud: 0.034, maxDev: 0.05) {
            a.trim.append(box); a.signals.append(disp)
        }

        // ── wheel-arch cladding: black annular sector round the upper half of each arch, both sides ──
        var clad = Mesh3()
        for ax in axles {
            let ro = shellOpeningR + shellToe * 0.0
            let yc = R
            let probeY = yc + ro + 0.12
            let fx = body.flank(probeY, ax)?.x ?? W / 2
            clad.append(ccArcBand(zc: ax, yc: yc, rIn: ro + 0.035, rOut: ro + 0.165, a0: 0.05, a1: .pi - 0.05, x0: fx - 0.012, x1: fx + 0.016))
        }
        a.trim.append(clad); a.trim.append(clad.mirrored(.x))
    }

    /// Annular sector solid in the (z, y) plane about (zc, yc), radii rIn…rOut, angles a0…a1 (0 = +z, π/2 = up),
    /// extruded x0…x1. Closed, outward-wound.
    func ccArcBand(zc: Double, yc: Double, rIn: Double, rOut: Double, a0: Double, a1: Double, x0: Double, x1: Double, segs: Int = 20) -> Mesh3 {
        var m = Mesh3()
        func P(_ th: Double, _ r: Double, _ x: Double) -> Vec3 { Vec3(x, yc + r * sin(th), zc + r * cos(th)) }
        for i in 0..<segs {
            let t0 = a0 + (a1 - a0) * Double(i) / Double(segs), t1 = a0 + (a1 - a0) * Double(i + 1) / Double(segs)
            let tm = (t0 + t1) / 2
            let rad = Vec3(0, sin(tm), cos(tm))
            m.addQuad(P(t0, rIn, x1), P(t1, rIn, x1), P(t1, rOut, x1), P(t0, rOut, x1), outward: Vec3(1, 0, 0))
            m.addQuad(P(t0, rIn, x0), P(t1, rIn, x0), P(t1, rOut, x0), P(t0, rOut, x0), outward: Vec3(-1, 0, 0))
            m.addQuad(P(t0, rOut, x0), P(t1, rOut, x0), P(t1, rOut, x1), P(t0, rOut, x1), outward: rad)
            m.addQuad(P(t0, rIn, x0), P(t1, rIn, x0), P(t1, rIn, x1), P(t0, rIn, x1), outward: -rad)
        }
        for (th, sgn) in [(a0, -1.0), (a1, 1.0)] {
            let tan = Vec3(0, cos(th), -sin(th)) * sgn
            m.addQuad(P(th, rIn, x0), P(th, rOut, x0), P(th, rOut, x1), P(th, rIn, x1), outward: tan)
        }
        return m
    }
}
