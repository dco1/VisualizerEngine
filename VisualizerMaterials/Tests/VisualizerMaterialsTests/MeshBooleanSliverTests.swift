import XCTest
import simd
@testable import VisualizerMaterials

/// Booleans on inputs with sub-tolerance slivers (e.g. a mesh snapped to 1 nm): the result must stay closed
/// (by index and by position), manifold, the right volume, and free of degenerate and duplicate faces.
final class MeshBooleanSliverTests: XCTestCase {

    // MARK: audits

    struct Defects { var openIdx = 0, nonManIdx = 0, openPos = 0, nonManPos = 0, degenerate = 0, duplicates = 0 }

    static func defects(_ m: Mesh3, posGrid: Double = 1e7) -> Defects {
        var d = Defects()
        var c: [UInt64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0..<3 {
                let a = m.indices[t + k], b = m.indices[t + (k + 1) % 3]
                c[UInt64(min(a, b)) << 32 | UInt64(max(a, b)), default: 0] += 1
            }
            t += 3
        }
        d.openIdx = c.values.filter { $0 == 1 }.count; d.nonManIdx = c.values.filter { $0 > 2 }.count
        struct K: Hashable { let x: Int64, y: Int64, z: Int64 }
        var ids: [K: Int] = [:]
        let vid = m.positions.map { p -> Int in
            let k = K(x: Int64((p.x * posGrid).rounded()), y: Int64((p.y * posGrid).rounded()), z: Int64((p.z * posGrid).rounded()))
            if let i = ids[k] { return i }; ids[k] = ids.count; return ids.count - 1
        }
        var pc: [SIMD2<Int>: Int] = [:]
        var seen = Set<SIMD3<Int>>()
        t = 0
        while t + 2 < m.indices.count {
            let v = (0..<3).map { vid[Int(m.indices[t + $0])] }
            for e in 0..<3 where v[e] != v[(e + 1) % 3] {
                pc[SIMD2(min(v[e], v[(e + 1) % 3]), max(v[e], v[(e + 1) % 3]))] = (pc[SIMD2(min(v[e], v[(e + 1) % 3]), max(v[e], v[(e + 1) % 3]))] ?? 0) + 1
            }
            let s = v.sorted()
            if !seen.insert(SIMD3(s[0], s[1], s[2])).inserted { d.duplicates += 1 }
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], cc = m.positions[Int(m.indices[t + 2])]
            if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] || len3(cross3(b - a, cc - a)) < 1e-18 { d.degenerate += 1 }
            t += 3
        }
        d.openPos = pc.values.filter { $0 == 1 }.count; d.nonManPos = pc.values.filter { $0 > 2 }.count
        return d
    }

    func assertClean(_ m: Mesh3, volume: Double? = nil, tol: Double = 1e-6, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        let d = Self.defects(m)
        XCTAssertEqual(d.openIdx, 0, "\(label): open by index", file: file, line: line)
        XCTAssertEqual(d.nonManIdx, 0, "\(label): non-manifold by index", file: file, line: line)
        XCTAssertEqual(d.openPos, 0, "\(label): open by position", file: file, line: line)
        XCTAssertEqual(d.nonManPos, 0, "\(label): non-manifold by position", file: file, line: line)
        XCTAssertEqual(d.duplicates, 0, "\(label): duplicate faces", file: file, line: line)
        XCTAssertEqual(d.degenerate, 0, "\(label): degenerate faces", file: file, line: line)
        if let volume { XCTAssertEqual(m.signedVolume, volume, accuracy: tol * max(abs(volume), 1e-9), "\(label): volume", file: file, line: line) }
    }

    static func snapped(_ m: Mesh3, _ q: Double = 1e9) -> Mesh3 {
        var o = m
        o.positions = m.positions.map { Vec3(($0.x * q).rounded() / q, ($0.y * q).rounded() / q, ($0.z * q).rounded() / q) }
        return o
    }

    private func box(_ lo: Vec3, _ hi: Vec3) -> Mesh3 { GuitarCarve.boxMesh(lo: lo, hi: hi) }

    /// Builds a closed box-like solid from explicit vertices on a grid: used for sliver faces.
    /// A box [0,1]^3 whose top face (z=1) has an extra vertex `p` on the edge from (0,0,1) to (1,0,1), pushed
    /// off the edge by `off` in y, fanned to the other corners (a sliver triangle against the edge).
    private func boxWithSliver(off: Double) -> Mesh3 {
        var m = Mesh3()
        let c: [Vec3] = (0..<8).map { Vec3(Double($0 & 1), Double(($0 >> 1) & 1), Double(($0 >> 2) & 1)) }
        m.positions = c
        m.positions.append(Vec3(0.37, off, 1))                       // 8: on (or `off` from) the top's y=0 edge
        let quads: [[UInt32]] = [[0, 2, 3, 1], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
        for q in quads { m.indices += [q[0], q[1], q[2], q[0], q[2], q[3]] }
        let top: [UInt32] = [4, 5, 7, 6]
        for i in 0..<4 { m.indices += [top[i], top[(i + 1) % 4], 8] }
        // wind everything outward
        var t = 0
        while t + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], d = m.positions[Int(m.indices[t + 2])]
            if dot3(cross3(b - a, d - a), (a + b + d) / 3 - Vec3(0.5, 0.5, 0.5)) < 0 { m.indices.swapAt(t + 1, t + 2) }
            t += 3
        }
        m.normals = m.positions.map { normalize3($0 - Vec3(0.5, 0.5, 0.5)) }
        m.uvs = m.positions.map { _ in Vec2.zero }
        return m
    }

    /// Runs `body` on the CPU path and again with every boolean sent to the GPU2 pipeline.
    func bothPaths(_ body: (String) -> Void) {
        let saved = BooleanGPU2.minTriangles
        defer { BooleanGPU2.minTriangles = saved }
        BooleanGPU2.minTriangles = Int.max
        body("cpu")
        if BooleanGPU2.shared != nil { let r0 = BooleanGPU2.runs; BooleanGPU2.minTriangles = 0; body("gpu2"); XCTAssertGreaterThan(BooleanGPU2.runs, r0, "the GPU2 pipeline ran") }
    }

    // MARK: hand-built

    func testSliverFaceCutByCutter() {
        bothPaths { path in
            for off in [0.3, 1e-10, 4e-11, 0] {
                let a = boxWithSliver(off: off)
                XCTAssertEqual(a.signedVolume, 1, accuracy: 1e-9, "input volume")
                let cut = box(Vec3(0.2, -0.5, 0.4), Vec3(0.6, 0.6, 1.5))      // crosses the top face and the sliver
                let r = a.boolean(.difference, cut)
                assertClean(r.mesh, volume: 1 - 0.4 * 0.6 * 0.6, tol: 1e-6, "\(path) sliver off=\(off) diff")
                let u = a.boolean(.union, cut)
                assertClean(u.mesh, volume: 1 + 0.4 * 1.1 * 1.1 - 0.4 * 0.6 * 0.6, tol: 1e-6, "\(path) sliver off=\(off) union")
            }
        }
    }

    func testNearCoincidentVertices() {
        bothPaths { path in
            // a cutter whose corner sits 3e-10 from a vertex of the box (inside weld tolerance of 1e-9 * diag)
            let a = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
            let b = box(Vec3(0.5 + 3e-10, 0.5, 0.5), Vec3(1.5, 1.5, 1.5))
            let r = a.boolean(.union, b)
            assertClean(r.mesh, volume: 1 + 1 - 0.5 * 0.5 * 0.5 - 0, tol: 1e-6, "\(path) near-vertex union")
            let c = box(Vec3(1 - 3e-10, 0.2, 0.2), Vec3(1.5, 0.8, 0.8))      // within eps of a's face x=1
            assertClean(a.boolean(.union, c).mesh, "\(path) near-face union")
            assertClean(a.boolean(.difference, c).mesh, "\(path) near-face diff")
        }
    }

    func testCutterEdgeNearVertex() {
        bothPaths { path in
            let a = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
            for e in [1e-10, 5e-10, 2e-9] {
                let b = box(Vec3(0.3, -0.5, 0.3), Vec3(0.7, 1.5, 1 + e))      // top 1e-10 off a's top plane
                let r = a.boolean(.difference, b)
                assertClean(r.mesh, "\(path) cutter near plane e=\(e)")
                let s = a.boolean(.union, Self.snapped(b))
                assertClean(s.mesh, "\(path) union snapped e=\(e)")
            }
        }
    }

    func testTwoVerticesInsideWeldTolerance() {
        bothPaths { path in
            // a box whose x=1 face uses its own copies of the four corners, displaced 3e-10 (inside the weld tolerance)
            var a = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
            a.positions = a.positions.map { $0.x > 0.5 ? $0 + Vec3(3e-10, 2e-10, 0) : $0 }
            let b = box(Vec3(0.5, 0.25, 0.25), Vec3(1.5, 0.75, 0.75))
            assertClean(a.boolean(.union, b).mesh, "\(path) near-weld union")
            assertClean(a.boolean(.difference, b).mesh, "\(path) near-weld diff")
            assertClean(a.boolean(.intersection, b).mesh, volume: 0.5 * 0.5 * 0.5, tol: 1e-6, "\(path) near-weld intersection")
        }
    }

    func testSliverStatsAreReported() {
        let a = boxWithSliver(off: 1e-10)
        let r = a.boolean(.union, box(Vec3(0.2, -0.5, 0.4), Vec3(0.6, 0.6, 1.5)))
        XCTAssertGreaterThan(r.stats.slivers, 0)
        XCTAssertEqual(r.stats.sliverFlips, r.stats.slivers)
        XCTAssertFalse(r.stats.repairIncomplete)
        let clean = box(Vec3(0, 0, 0), Vec3(1, 1, 1)).boolean(.union, box(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 1.5, 1.5)))
        XCTAssertEqual(clean.stats.slivers, 0)
    }

    func testAuditRemovesDuplicatesAndReportsOpenEdges() {
        let m = boxWithSliver(off: 0.3)                                           // indexed, closed
        var r = BooleanResult(mesh: m, stats: BooleanStats())
        r.audit()
        XCTAssertEqual(r.stats.openEdges, 0); XCTAssertFalse(r.stats.repairIncomplete)
        var dup = m
        dup.indices += Array(m.indices[0..<3])                                   // the same triangle twice
        var r2 = BooleanResult(mesh: dup, stats: BooleanStats())
        r2.audit()
        XCTAssertEqual(r2.stats.duplicatesRemoved, 1)
        XCTAssertEqual(r2.mesh.triangleCount, m.triangleCount)
        XCTAssertFalse(r2.stats.repairIncomplete)
        var open = m
        open.indices.removeLast(3)
        var r3 = BooleanResult(mesh: open, stats: BooleanStats())
        r3.audit()
        XCTAssertEqual(r3.stats.openEdges, 3); XCTAssertTrue(r3.stats.repairIncomplete)
        let ok = box(Vec3(0, 0, 0), Vec3(1, 1, 1)).boolean(.union, box(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 1.5, 1.5)), options: { var o = BooleanOptions(); o.audit = true; return o }())
        XCTAssertEqual(ok.stats.openEdges + ok.stats.nonManifoldEdges, 0); XCTAssertFalse(ok.stats.repairIncomplete)
    }

    func testZeroAreaFan() {
        bothPaths { path in
            // a box whose top has a fan with a collinear (zero-area) member: p exactly on the edge
            let a = boxWithSliver(off: 0)
            let b = box(Vec3(0.1, 0.1, 0.5), Vec3(0.9, 0.9, 1.5))
            assertClean(a.boolean(.union, b).mesh, "\(path) zero-area fan union")
            assertClean(a.boolean(.intersection, b).mesh, "\(path) zero-area fan intersect")
        }
    }

    // MARK: guitar step 6 snapped to 1 nm

    func testGuitarStep6Snapped() {
        bothPaths { path in
            let body = GuitarCarve.contour(GuitarCarve.carveNeck(GuitarCarve.roundEdges(GuitarCarve.carveOutline())))
            func s(_ m: Mesh3) -> Mesh3 { Self.snapped(m) }
            let bx = GuitarCarve.bridgeX, px = GuitarCarve.pickupX
            func b(_ lo: Vec3, _ hi: Vec3) -> Mesh3 { s(box(lo, hi)) }
            var m = s(body)
            m = m.booleanDifference(b(Vec3(0.22, -0.105, 0.0215), Vec3(0.47, 0.085, 0.1)))
            var plate = b(Vec3(0.2185, -0.1065, 0.0210), Vec3(0.4715, 0.0865, 0.0275))
            for x in px {
                m = m.booleanDifference(b(Vec3(x - 0.036, -0.021, 0.011), Vec3(x + 0.036, 0.021, 0.1)))
                plate = plate.booleanDifference(b(Vec3(x - 0.038, -0.023, 0.0), Vec3(x + 0.038, 0.023, 0.1)))
            }
            m = m.booleanUnion(plate)
            for x in px { m = m.booleanUnion(b(Vec3(x - 0.033, -0.018, 0.0105), Vec3(x + 0.033, 0.018, 0.0335))) }
            m = m.booleanUnion(b(Vec3(bx - 0.006, -0.05, 0.0235), Vec3(bx + 0.006, 0.05, 0.0345)))
            m = m.booleanUnion(b(Vec3(bx - 0.070, -0.05, 0.0235), Vec3(bx - 0.052, 0.05, 0.0325)))
            m = m.booleanUnion(s(GuitarCarve.knobs()))
            m = m.booleanUnion(s(GuitarCarve.fretboard()))
            assertClean(m, "\(path) guitar step 6 snapped to 1 nm")
        }
    }
}

extension MeshBooleanSliverTests {
}
