import Foundation
import simd

// plausibility: real — bus glazing is individual framed panes between real mullions (the painted body between
// them), split windscreen with a centre post, and seat rows at a 0.8 m pitch; all closed solids, no textures.

extension CarContext {

    /// Side-window layout of the boxy (bus) branch, shared with tests: z centres and half-length of each pane.
    var busWindowLayout: (zs: [Double], half: Double, yLo: Double, yHi: Double) {
        let z0 = rearZ + 0.55, z1 = frontZ - 1.45, pitch = 1.30
        let n = max(1, Int(((z1 - z0) / pitch).rounded()))
        let step = (z1 - z0) / Double(n)
        let yb = bp.belt * rH
        return ((0..<n).map { z0 + step * (Double($0) + 0.5) }, step / 2 - 0.07, yb + 0.02, yb + 1.15 * rH)
    }

    func buildBusGreenhouse(_ a: inout CarAccumulator, _ rec: inout GHGlassRecorder) {
        let hw = W / 2
        let (zs, half, yLo, yHi) = busWindowLayout
        let side = GHSurface { q in (Vec3(hw, q.y, q.x), Vec3(1, 0, 0)) }

        // Side windows: append order = windscreen L,R · side panes (right then left, front to rear) · rear window.
        // Panes are plates proud of the flat body side; the body between them is the mullion.
        func rect(_ x0: Double, _ x1: Double, _ y0: Double, _ y1: Double, _ r: Double) -> [Vec2] {
            GHKit.densify(GHKit.round([Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y1), Vec2(x0, y1)], r), 0.3)
        }
        let yb = bp.belt * rH
        // windscreen: split by a centre post, on the (flat) nose face
        let hn = hw * bp.noseTaper
        let front = GHSurface { q in (Vec3(q.x, q.y, self.frontZ), Vec3(0, 0, 1)) }
        let yw0 = yb + 0.05, yw1 = yb + 1.45 * rH
        for sgn in [-1.0, 1.0] {
            let x0 = sgn > 0 ? 0.06 : -(hn - 0.17), x1 = sgn > 0 ? hn - 0.17 : -0.06
            let p = rect(x0, x1, yw0, yw1, 0.06)
            rec.add(&a, sgn > 0 ? "windscreen.R" : "windscreen.L", .windscreen, front.prism(p, inward: 0.004, outward: 0.008))
            a.trim.append(front.frame(p, width: 0.025, overlap: 0.004, inward: 0.004, outward: 0.012))
        }
        // side windows
        for z in zs {
            let p = rect(z - half, z + half, yLo, yHi, 0.05)
            let g = side.prism(p, inward: 0.004, outward: 0.007)
            let f = side.frame(p, width: 0.022, overlap: 0.004, inward: 0.004, outward: 0.012)
            let grp: GlassGroup = z > (rearZ + frontZ) / 2 ? .front : .rear
            rec.add(&a, "side\(zs.firstIndex(of: z)!).R", grp, g); rec.add(&a, "side\(zs.firstIndex(of: z)!).L", grp, g.mirrored(.x))
            a.trim.append(f); a.trim.append(f.mirrored(.x))
        }
        // rear window
        let ht = hw * bp.tailTaper
        let rear = GHSurface { q in (Vec3(q.x, q.y, self.rearZ), Vec3(0, 0, -1)) }
        let pr = rect(-(ht - 0.45), ht - 0.45, yb + 0.40, yb + 1.15 * rH, 0.06)
        rec.add(&a, "rear", .rear, rear.prism(pr, inward: 0.004, outward: 0.008))
        a.trim.append(rear.frame(pr, width: 0.025, overlap: 0.004, inward: 0.004, outward: 0.012))

        // Interior: driver station + passenger rows (sits inside the body volume — the body shell is opaque,
        // so these are only visible if the body step ever hollows the cabin).
        let yFloor = max(2 * R + 0.03, 0.62)
        let z0 = rearZ + 0.9, z1 = frontZ - 2.2
        let rows = max(1, Int((z1 - z0) / 0.80))
        for r in 0..<rows {
            let z = z0 + Double(r) * 0.80
            for x in [-0.95, -0.50, 0.50, 0.95] {
                a.interior.append(carBox(Vec3(x - 0.20, yFloor, z), Vec3(x + 0.20, yFloor + 0.15, z + 0.42)))
                a.interior.append(carBox(Vec3(x - 0.20, yFloor + 0.12, z - 0.08), Vec3(x + 0.20, yFloor + 0.70, z + 0.02)))
            }
        }
        let zd = frontZ - 1.25, xd = -0.60
        a.interior.append(carBox(Vec3(xd - 0.22, yFloor, zd - 0.45), Vec3(xd + 0.22, yFloor + 0.15, zd)))
        a.interior.append(carBox(Vec3(xd - 0.22, yFloor + 0.12, zd - 0.55), Vec3(xd + 0.22, yFloor + 0.75, zd - 0.43)))
        a.interior.append(carBox(Vec3(-hn + 0.20, yFloor + 0.20, frontZ - 0.55), Vec3(hn - 0.20, yFloor + 0.62, frontZ - 0.10)))
        let tilt = 1.0
        let n = normalize3(Vec3(0, sin(tilt), -cos(tilt)))
        a.interior.append(GHKit.torus(centre: Vec3(xd, yFloor + 0.62, frontZ - 0.70), axis: n, big: 0.215, small: 0.016, around: 36, tube: 10))
    }
}
