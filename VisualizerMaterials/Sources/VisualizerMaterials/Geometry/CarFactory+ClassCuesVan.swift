import Foundation
import simd

// plausibility: real — the van's sliding-door outline, track rail, rear barn doors, hinges and handles are authored
// solids laid on the real shell surface (read with ray probes); the bulkhead lives in the greenhouse interior.

/// Where the van cues sit (shared with the tests).
struct VanDoorLayout {
    var zFront: Double, zRear: Double        // sliding door edges (right side)
    var yBottom: Double, yTop: Double
    var railY: Double, railZ0: Double, railZ1: Double
    var barnX: Double, barnYBottom: Double, barnYTop: Double
}

extension CarContext {

    var vanDoorLayout: VanDoorLayout {
        let zcr = cabinSpan.0                        // cabin rear = end of the cab door glass
        let zf = zcr - 0.10, zr = zf - 1.03
        let belt = bp.belt * rH
        return VanDoorLayout(zFront: zf, zRear: zr, yBottom: 0.42, yTop: min(belt + 0.80, H - 0.20),
                             railY: belt - 0.04, railZ0: zr - 0.04, railZ1: max(rearZ + 0.30, zr - 1.45),
                             barnX: W / 2 - 0.07, barnYBottom: 0.50, barnYTop: H - 0.10)
    }

    func buildVanCues(_ a: inout CarAccumulator, body: CCProbe) {
        let D = vanDoorLayout
        var lines = Mesh3(), rail = Mesh3(), chrome = Mesh3(), hinges = Mesh3()
        let xin = 0.0
        // ── sliding door outline, right side (+x) ──
        let dir = Vec3(1, 0, 0)
        lines.append(ccSeam(body, from: Vec3(xin, D.yBottom, D.zFront), to: Vec3(xin, D.yTop, D.zFront), dir: dir, width: 0.005))
        lines.append(ccSeam(body, from: Vec3(xin, D.yBottom, D.zRear), to: Vec3(xin, D.yTop, D.zRear), dir: dir, width: 0.005))
        lines.append(ccSeam(body, from: Vec3(xin, D.yTop, D.zRear), to: Vec3(xin, D.yTop, D.zFront), dir: dir, width: 0.005))
        lines.append(ccSeam(body, from: Vec3(xin, D.yBottom, D.zRear), to: Vec3(xin, D.yBottom, D.zFront), dir: dir, width: 0.005))
        // sliding-door track: a horizontal rail from the door's rear edge toward the tail, with end stops
        rail.append(ccSeam(body, from: Vec3(xin, D.railY, D.railZ0), to: Vec3(xin, D.railY, D.railZ1), dir: dir, width: 0.040, proud: 0.016, back: 0.006, maxDev: 0.05))
        for z in [D.railZ0, D.railZ1] {
            if let m = ccStamp(body, from: Vec3(xin, D.railY, z), dir: dir, w: 0.07, h: 0.07, back: 0.006, proud: 0.022, maxDev: 0.05) { rail.append(m) }
        }
        // door handle (chrome) near the leading edge
        if let h = ccStamp(body, from: Vec3(xin, D.yBottom + 0.62, D.zFront - 0.09), dir: dir, w: 0.030, h: 0.17, back: 0.006, proud: 0.020, maxDev: 0.05) { chrome.append(h) }
        if let p = ccStamp(body, from: Vec3(xin, D.yBottom + 0.62, D.zFront - 0.09), dir: dir, w: 0.060, h: 0.20, back: 0.006, proud: 0.004, maxDev: 0.05) { lines.append(p) }

        // ── rear barn doors ──
        let rz = rearZ + 0.6
        let rd = Vec3(0, 0, -1)
        for sx in [-1.0, 1.0] {
            lines.append(ccSeam(body, from: Vec3(sx * D.barnX, D.barnYBottom, rz), to: Vec3(sx * D.barnX, D.barnYTop, rz), dir: rd, width: 0.005))
        }
        lines.append(ccSeam(body, from: Vec3(-D.barnX, D.barnYTop, rz), to: Vec3(D.barnX, D.barnYTop, rz), dir: rd, width: 0.005))
        lines.append(ccSeam(body, from: Vec3(-D.barnX, D.barnYBottom, rz), to: Vec3(D.barnX, D.barnYBottom, rz), dir: rd, width: 0.005))
        lines.append(ccSeam(body, from: Vec3(0, D.barnYBottom, rz), to: Vec3(0, D.barnYTop, rz), dir: rd, width: 0.008))      // centre seam
        // hinges: two per door on the outer edges
        for sx in [-1.0, 1.0] {
            for f in [0.18, 0.82] {
                let y = D.barnYBottom + (D.barnYTop - D.barnYBottom) * f
                if let m = ccStamp(body, from: Vec3(sx * (D.barnX - 0.03), y, rz), dir: rd, w: 0.060, h: 0.11, back: 0.006, proud: 0.010, maxDev: 0.05) { hinges.append(m) }
            }
        }
        // vertical handle + lock recess on the right-hand door
        if let pocket = ccStamp(body, from: Vec3(0.11, 1.18, rz), dir: rd, w: 0.075, h: 0.19, back: 0.006, proud: 0.0015, maxDev: 0.05) { lines.append(pocket) }
        if let bar = ccStamp(body, from: Vec3(0.11, 1.18, rz), dir: rd, w: 0.022, h: 0.15, back: 0.006, proud: 0.016, maxDev: 0.05) { chrome.append(bar) }
        a.trim.append(lines); a.trim.append(rail); a.trim.append(hinges)
        a.chrome.append(chrome)
    }
}
