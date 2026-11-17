import XCTest
import simd
@testable import VisualizerMaterials

/// Item I, sculpt brushes. Judged by sub-voxel volume, by where the surface sits along a ray, by exact field equality
/// (undo, symmetry, GPU == CPU twin) and by closure of the remeshed result.
final class SculptTests: XCTestCase {
    let cell = 0.005

    func sphereVolume(radius: Double = 0.1, half: Double = 0.16, cell: Double? = nil) -> SculptVolume {
        let c = cell ?? self.cell
        return SculptVolume(sdf: .sphere(radius: radius), lo: Vec3(repeating: -half), hi: Vec3(repeating: half), cell: c)
    }

    /// Distance from `from` along `dir` to the zero crossing (bisection on the trilinear field), nil if none within `maxT`.
    func surface(_ v: SculptVolume, from: Vec3, dir: Vec3, maxT: Double) -> Double? {
        var a = 0.0, b = maxT
        guard (v.value(at: from) < 0) != (v.value(at: from + dir * b) < 0) else { return nil }
        let insideA = v.value(at: from) < 0
        for _ in 0..<40 {
            let m = (a + b) / 2
            if (v.value(at: from + dir * m) < 0) == insideA { a = m } else { b = m }
        }
        return (a + b) / 2
    }

    func topHeight(_ v: SculptVolume, x: Double = 0, z: Double = 0) -> Double {
        0.15 - surface(v, from: Vec3(x, 0.15, z), dir: Vec3(0, -1, 0), maxT: 0.15)!
    }

    func unmatchedEdges(_ m: Mesh3) -> Int {
        var count: [UInt64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for e in 0..<3 { count[UInt64(m.indices[t + e]) << 32 | UInt64(m.indices[t + (e + 1) % 3]), default: 0] += 1 }
            t += 3
        }
        return count.filter { $0.value != 1 || count[($0.key & 0xffff_ffff) << 32 | ($0.key >> 32)] != 1 }.count
    }

    func signedVolume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            v += dot3(m.positions[Int(m.indices[t])], cross3(m.positions[Int(m.indices[t + 1])], m.positions[Int(m.indices[t + 2])])) / 6; t += 3
        }
        return v
    }

    func stamp(_ v: SculptVolume, _ b: SculptBrush, at p: Vec3, normal: Vec3 = Vec3(0, 1, 0), pressure: Double = 1, times: Int = 1) {
        let s = v.beginStroke(b)
        for _ in 0..<times { s.stamp(position: p, normal: normal, pressure: pressure) }
        s.end()
    }

    // MARK: brushes

    func testDrawAddsVolumeAndHeight() {
        let v = sphereVolume()
        let v0 = v.volume(), h0 = topHeight(v)
        XCTAssertEqual(h0, 0.1, accuracy: 0.002)
        stamp(v, SculptBrush(.draw, radius: 0.04, strength: 0.5), at: Vec3(0, 0.1, 0))
        let dv = v.volume() - v0
        // The top rises by strength · radius = 2 cm; the bump is a smooth cap of that height, base radius < 4 cm.
        // The surface rises to where s = a·w(s/r): a bit under strength · radius (2 cm) with the smooth falloff.
        XCTAssertEqual(topHeight(v) - h0, 0.0149, accuracy: 0.002)
        let upper = Double.pi * 0.04 * 0.04 * 0.02          // cylinder bound
        XCTAssertGreaterThan(dv, upper * 0.2)
        XCTAssertLessThan(dv, upper * 1.1)
    }

    func testSubtractCarvesToExpectedDepth() {
        let v = sphereVolume()
        let v0 = v.volume()
        stamp(v, SculptBrush(.draw, radius: 0.04, strength: 0.5, subtract: true), at: Vec3(0, 0.1, 0))
        XCTAssertEqual(0.1 - topHeight(v), 0.0149, accuracy: 0.002)
        XCTAssertLessThan(v.volume(), v0 - 1e-6)
        // Pressure scales the depth.
        let w = sphereVolume()
        stamp(w, SculptBrush(.draw, radius: 0.04, strength: 0.5, subtract: true), at: Vec3(0, 0.1, 0), pressure: 0.5)
        XCTAssertEqual(0.1 - topHeight(w), 0.0085, accuracy: 0.002)
    }

    func testInflateCreaseFlattenPinch() {
        let v = sphereVolume()
        stamp(v, SculptBrush(.inflate, radius: 0.05, strength: 0.2), at: Vec3(0, 0.1, 0))
        XCTAssertEqual(topHeight(v), 0.1 + 0.007, accuracy: 0.003)
        let c = sphereVolume()
        stamp(c, SculptBrush(.crease, radius: 0.04, strength: 0.25), at: Vec3(0, 0.1, 0))
        XCTAssertEqual(topHeight(c), 0.1 - 0.01, accuracy: 0.003)          // a groove
        XCTAssertEqual(topHeight(c, x: 0.03), topHeight(sphereVolume(), x: 0.03), accuracy: 0.002)   // narrow: little effect at 0.75 r
        // Flatten a bumpy top onto the plane through the stamp.
        let f = sphereVolume()
        stamp(f, SculptBrush(.draw, radius: 0.04, strength: 0.5), at: Vec3(0, 0.1, 0))
        let hBump = topHeight(f)
        stamp(f, SculptBrush(.flatten, radius: 0.05, strength: 1), at: Vec3(0, 0.1, 0), times: 4)
        XCTAssertLessThan(topHeight(f), hBump - 0.008)
        // Pinch pulls the surface's features toward the axis: a bump gets narrower.
        let p = sphereVolume()
        stamp(p, SculptBrush(.draw, radius: 0.04, strength: 0.5), at: Vec3(0, 0.1, 0))
        let before = p.volume()
        stamp(p, SculptBrush(.pinch, radius: 0.06, strength: 0.5), at: Vec3(0, 0.1, 0), times: 3)
        XCTAssertNotEqual(p.volume(), before, accuracy: 1e-9)
        XCTAssertGreaterThan(topHeight(p), 0.1 + 0.012)
    }

    func testSmoothReducesRoughness() {
        let rough = SDF.custom { p in
            len3(p) - 0.1 + 0.003 * sin(250 * p.x) * sin(250 * p.y) * sin(250 * p.z)
        }
        let v = SculptVolume(sdf: rough, lo: Vec3(repeating: -0.16), hi: Vec3(repeating: 0.16), cell: cell)
        func roughness(_ m: Mesh3) -> Double {
            let r = m.positions.map { len3($0) }
            let mean = r.reduce(0, +) / Double(r.count)
            return (r.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(r.count)).squareRoot()
        }
        let before = roughness(v.mesh())
        let s = v.beginStroke(SculptBrush(.smooth, radius: 0.2, strength: 1, falloff: .constant))
        for _ in 0..<4 { s.stamp(position: .zero) }
        s.end()
        let after = roughness(v.mesh())
        XCTAssertLessThan(after, before * 0.6, "roughness \(before) → \(after)")
    }

    func testSymmetryMirrors() {
        let v = sphereVolume()
        stamp(v, SculptBrush(.draw, radius: 0.04, strength: 0.5, symmetry: [.x]), at: Vec3(0.07, 0.07, 0.01), normal: Vec3(1, 1, 0))
        let n = v.samples
        var worst: Float = 0
        for k in 0..<n.z { for j in 0..<n.y { for i in 0..<n.x {
            worst = max(worst, abs(v.sample(i, j, k) - v.sample(n.x - 1 - i, j, k)))
        } } }
        XCTAssertLessThan(worst, 1e-5)
        // And it did something on both sides.
        let a = surface(v, from: Vec3(0.2, 0.2, 0.01) * 0.3, dir: normalize3(Vec3(1, 1, 0)), maxT: 0.2)!
        let b = surface(v, from: Vec3(-0.2, 0.2, 0.01) * 0.3, dir: normalize3(Vec3(-1, 1, 0)), maxT: 0.2)!
        XCTAssertEqual(a, b, accuracy: 1e-4)
        // A stamp on the plane is applied once, not twice.
        let on = sphereVolume(), once = sphereVolume()
        stamp(on, SculptBrush(.draw, radius: 0.04, strength: 0.5, symmetry: [.x]), at: Vec3(0, 0.1, 0))
        stamp(once, SculptBrush(.draw, radius: 0.04, strength: 0.5), at: Vec3(0, 0.1, 0))
        XCTAssertEqual(on.fieldCopy(), once.fieldCopy())
    }

    func testMaskProtects() {
        let v = sphereVolume()
        v.setMask { $0.x > 0 ? 1 : 0 }
        let before = v.fieldCopy()
        stamp(v, SculptBrush(.draw, radius: 0.05, strength: 0.5), at: Vec3(0, 0.1, 0))
        let after = v.fieldCopy()
        var left = 0.0, right = 0.0
        let n = v.samples
        for k in 0..<n.z { for j in 0..<n.y { for i in 0..<n.x {
            let idx = (k * n.y + j) * n.x + i, d = Double(abs(after[idx] - before[idx]))
            if v.position(i, j, k).x > 1e-9 { right += d } else { left += d }
        } } }
        XCTAssertEqual(right, 0)
        XCTAssertGreaterThan(left, 0.01)
    }

    func testGrabMovesSurfaceByDrag() {
        let v = sphereVolume()
        let s = v.beginStroke(SculptBrush(.grab, radius: 0.06, strength: 1))
        for i in 0...4 { s.stamp(position: Vec3(0, 0.1 + 0.01 * Double(i), 0)) }
        s.end()
        XCTAssertEqual(topHeight(v), 0.14, accuracy: 0.004)
        // Away from the brush nothing moved.
        XCTAssertEqual(-0.15 + surface(v, from: Vec3(0, -0.15, 0), dir: Vec3(0, 1, 0), maxT: 0.15)!, -0.1, accuracy: 0.002)
    }

    // MARK: undo, locality, remesh

    func testUndoRestoresFieldExactly() {
        let v = sphereVolume()
        let f0 = v.fieldCopy()
        stamp(v, SculptBrush(.draw, radius: 0.04), at: Vec3(0, 0.1, 0), times: 3)
        let f1 = v.fieldCopy()
        XCTAssertNotEqual(f0, f1)
        XCTAssertTrue(v.canUndo)
        let total = v.blocksPerAxis.x * v.blocksPerAxis.y * v.blocksPerAxis.z
        XCTAssertLessThan(v.undoBlockCount, total / 8)           // only the touched blocks were kept
        v.undo()
        XCTAssertEqual(v.fieldCopy(), f0)
        v.redo()
        XCTAssertEqual(v.fieldCopy(), f1)
        v.undo()
        XCTAssertEqual(v.fieldCopy(), f0)
        stamp(v, SculptBrush(.smooth, radius: 0.05), at: Vec3(0.05, 0.05, 0.05))
        XCTAssertFalse(v.canRedo)
    }

    func testStampTouchesOnlyRadiusCubedVoxels() {
        func count(_ half: Double, _ r: Double) -> (Int, Int) {
            let v = sphereVolume(half: half)
            let before = v.fieldCopy()
            stamp(v, SculptBrush(.inflate, radius: r, strength: 0.3), at: Vec3(0, 0.1, 0))
            let after = v.fieldCopy()
            return (v.lastStampVoxels, zip(before, after).filter { $0 != $1 }.count)
        }
        let (a, ca) = count(0.16, 0.02), (b, cb) = count(0.16, 0.04), (c, _) = count(0.3, 0.04)
        XCTAssertLessThanOrEqual(ca, a); XCTAssertLessThanOrEqual(cb, b)
        let side = { (r: Double) in Int((2 * r / self.cell).rounded(.up)) + 5 }
        XCTAssertLessThanOrEqual(a, side(0.02) * side(0.02) * side(0.02))
        XCTAssertLessThanOrEqual(b, side(0.04) * side(0.04) * side(0.04))
        XCTAssertEqual(Double(b), Double(c), accuracy: Double(b) * 0.2, "cost must not depend on the grid's size")
        XCTAssertGreaterThan(Double(b) / Double(a), 4)
        XCTAssertLessThan(Double(b) / Double(a), 9)
    }

    func testRemeshIsWatertight() {
        let v = sphereVolume()
        stamp(v, SculptBrush(.draw, radius: 0.04, strength: 0.6, symmetry: [.x, .z]), at: Vec3(0.05, 0.08, 0.04))
        stamp(v, SculptBrush(.crease, radius: 0.03), at: Vec3(0, 0, 0.1), normal: Vec3(0, 0, 1))
        let m = v.mesh()
        XCTAssertFalse(m.isEmpty)
        XCTAssertEqual(unmatchedEdges(m), 0)
        XCTAssertGreaterThan(signedVolume(m), 0)
        XCTAssertEqual(signedVolume(m), v.volume(), accuracy: v.volume() * 0.03)
    }

    func testAutoRefinement() {
        let v = sphereVolume(half: 0.12, cell: 0.01)
        stamp(v, SculptBrush(.draw, radius: 0.008, strength: 0.5), at: Vec3(0, 0.1, 0))
        XCTAssertLessThan(v.cell, 0.01)
        XCTAssertGreaterThanOrEqual(0.008, v.cell * 2 - 1e-12)
        XCTAssertEqual(topHeight(v), 0.1 + 0.004, accuracy: 0.002)
    }

    func testMeshInputAndEmptyCombine() {
        let m = SDF.sphere(radius: 0.1).mesh(lo: Vec3(repeating: -0.15), hi: Vec3(repeating: 0.15), cell: 0.01)
        let v = SculptVolume(mesh: m, cell: 0.01)
        XCTAssertEqual(v.volume(), 4.0 / 3 * Double.pi * 0.001, accuracy: 0.00008)
        let e = SculptVolume(emptyLo: Vec3(repeating: -0.15), hi: Vec3(repeating: 0.15), cell: 0.01)
        XCTAssertEqual(e.volume(), 0)
        e.combine(.sphere(radius: 0.08), lo: Vec3(repeating: -0.1), hi: Vec3(repeating: 0.1))
        XCTAssertEqual(e.volume(), 4.0 / 3 * Double.pi * 0.08 * 0.08 * 0.08, accuracy: 0.0001)
        e.undo()
        XCTAssertEqual(e.volume(), 0)
    }

    // MARK: GPU == CPU

    func testGPUMatchesCPUTwin() throws {
        try XCTSkipUnless(SculptGPU.available, "no Metal device")
        func run(gpu: Bool) -> (SculptVolume, [Float]) {
            let v = sphereVolume()
            v.useGPU = gpu
            v.setMask { $0.z > 0.08 ? 0.7 : 0 }
            let kinds: [SculptBrushKind] = [.draw, .grab, .smooth, .inflate, .crease, .flatten, .pinch]
            for (n, k) in kinds.enumerated() {
                let s = v.beginStroke(SculptBrush(k, radius: 0.0503, strength: 0.6, falloff: SculptFalloff(rawValue: UInt32(n % 6))!,
                                                  subtract: n % 2 == 1, symmetry: n % 3 == 0 ? [.x] : []))
                for i in 0..<3 {
                    s.stamp(position: Vec3(0.02 * Double(i), 0.1, 0.01 * Double(i)), normal: normalize3(Vec3(0.2, 1, 0.1)), pressure: 0.8)
                }
                s.end()
            }
            return (v, v.fieldCopy())
        }
        let (gv, g) = run(gpu: true), (cv, c) = run(gpu: false)
        var worst: Float = 0
        for (a, b) in zip(g, c) { worst = max(worst, abs(a - b)) }
        XCTAssertLessThan(worst, 1e-4, "GPU vs CPU brush field")
        // Mesh vertex pass.
        let a = gv.mesh(), b = { () -> Mesh3 in cv.useGPU = false; return cv.mesh() }()
        XCTAssertEqual(a.positions.count, b.positions.count)
        var wp = 0.0
        for (p, q) in zip(a.positions, b.positions) { wp = max(wp, len3(p - q)) }
        XCTAssertLessThan(wp, 1e-4)
        // Upsample.
        let u = sphereVolume(half: 0.1, cell: 0.01), w = sphereVolume(half: 0.1, cell: 0.01)
        w.useGPU = false
        XCTAssertTrue(u.refine()); XCTAssertTrue(w.refine())
        var wu: Float = 0
        for (p, q) in zip(u.fieldCopy(), w.fieldCopy()) { wu = max(wu, abs(p - q)) }
        XCTAssertLessThan(wu, 1e-6)
    }

    // MARK: timings

    func testReportTimings() {
        let v = sphereVolume()
        let s = v.beginStroke(SculptBrush(.draw, radius: 0.03))
        var t0 = Date()
        for i in 0..<100 { s.stamp(position: Vec3(0.03 * sin(Double(i) * 0.1), 0.1, 0.03 * cos(Double(i) * 0.1))) }
        v.sync()
        let stampMs = Date().timeIntervalSince(t0) * 10
        s.end()
        t0 = Date()
        let m = v.mesh()
        let meshMs = Date().timeIntervalSince(t0) * 1000
        print("Sculpt timings: \(stampMs) ms/stamp (\(v.useGPU ? "GPU" : "CPU"), \(v.lastStampVoxels) voxels), mesh \(meshMs) ms, \(m.indices.count / 3) tris, grid \(v.samples)")
        XCTAssertFalse(m.isEmpty)
    }
}
