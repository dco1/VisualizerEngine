import XCTest
@testable import VisualizerMaterials

/// Robust mesh-intersection booleans (`Mesh3+Boolean*.swift`): judged by exact analytic volumes, by closure
/// (every edge shared by exactly two faces, by index and by position), and by winding.
final class MeshBooleanTests: XCTestCase {

    // MARK: builders (indexed, shared vertices, outward-wound)

    private func fixOutward(_ m: Mesh3) -> Mesh3 {
        guard m.signedVolume < 0 else { return m }
        var o = m
        var t = 0
        while t + 2 < o.indices.count { o.indices.swapAt(t + 1, t + 2); t += 3 }
        o.normals = o.normals.map { -$0 }
        return o
    }

    private func box(_ lo: Vec3, _ hi: Vec3) -> Mesh3 {
        var m = Mesh3()
        let c = (lo + hi) * 0.5
        func q(_ a: Vec3, _ b: Vec3, _ cc: Vec3, _ d: Vec3, _ n: Vec3) {
            m.addQuad(a, b, cc, d, outward: n, uv: (Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)))
        }
        let (x0, y0, z0, x1, y1, z1) = (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
        q(Vec3(x1,y0,z0), Vec3(x1,y1,z0), Vec3(x1,y1,z1), Vec3(x1,y0,z1), Vec3(1,0,0))
        q(Vec3(x0,y0,z0), Vec3(x0,y0,z1), Vec3(x0,y1,z1), Vec3(x0,y1,z0), Vec3(-1,0,0))
        q(Vec3(x0,y1,z0), Vec3(x0,y1,z1), Vec3(x1,y1,z1), Vec3(x1,y1,z0), Vec3(0,1,0))
        q(Vec3(x0,y0,z0), Vec3(x1,y0,z0), Vec3(x1,y0,z1), Vec3(x0,y0,z1), Vec3(0,-1,0))
        q(Vec3(x0,y0,z1), Vec3(x1,y0,z1), Vec3(x1,y1,z1), Vec3(x0,y1,z1), Vec3(0,0,1))
        q(Vec3(x0,y0,z0), Vec3(x0,y1,z0), Vec3(x1,y1,z0), Vec3(x1,y0,z0), Vec3(0,0,-1))
        _ = c
        return m
    }

    private func cylinderZ(radius: Double, z0: Double, z1: Double, n: Int, at c: Vec3 = .zero) -> Mesh3 {
        var m = Mesh3()
        let ring = (0..<n).map { i -> Vec3 in
            let a = 2 * Double.pi * Double(i) / Double(n)
            return Vec3(c.x + radius * cos(a), c.y + radius * sin(a), 0)
        }
        let bot = Vec3(c.x, c.y, z0), top = Vec3(c.x, c.y, z1)
        for i in 0..<n {
            let j = (i + 1) % n
            let a = ring[i], b = ring[j]
            let mid = (a + b) * 0.5 - Vec3(c.x, c.y, 0)
            m.addQuad(Vec3(a.x, a.y, z0), Vec3(b.x, b.y, z0), Vec3(b.x, b.y, z1), Vec3(a.x, a.y, z1), outward: mid)
            m.addTriangle(top, Vec3(a.x, a.y, z1), Vec3(b.x, b.y, z1), outward: Vec3(0, 0, 1))
            m.addTriangle(bot, Vec3(a.x, a.y, z0), Vec3(b.x, b.y, z0), outward: Vec3(0, 0, -1))
        }
        return m
    }

    /// Indexed UV sphere: poles on ±Y, `rings` bands, `sectors` around. UVs = (u: longitude, v: latitude).
    private func sphere(_ r: Double, rings: Int, sectors: Int, at c: Vec3 = .zero) -> Mesh3 {
        var m = Mesh3()
        func vtx(_ p: Vec3, _ uv: Vec2) { m.positions.append(c + p * r); m.normals.append(p); m.uvs.append(uv) }
        vtx(Vec3(0, 1, 0), Vec2(0.5, 0))
        for i in 1..<rings {
            let th = Double.pi * Double(i) / Double(rings)
            for j in 0..<sectors {
                let ph = 2 * Double.pi * Double(j) / Double(sectors)
                vtx(Vec3(sin(th) * cos(ph), cos(th), sin(th) * sin(ph)), Vec2(Double(j) / Double(sectors), Double(i) / Double(rings)))
            }
        }
        vtx(Vec3(0, -1, 0), Vec2(0.5, 1))
        let south = UInt32(m.positions.count - 1)
        func id(_ i: Int, _ j: Int) -> UInt32 { UInt32(1 + (i - 1) * sectors + j % sectors) }
        for j in 0..<sectors { m.indices += [0, id(1, j + 1), id(1, j)] }
        for i in 1..<(rings - 1) { for j in 0..<sectors {
            m.indices += [id(i, j), id(i, j + 1), id(i + 1, j)]
            m.indices += [id(i, j + 1), id(i + 1, j + 1), id(i + 1, j)]
        } }
        for j in 0..<sectors { m.indices += [id(rings - 1, j), id(rings - 1, j + 1), south] }
        return fixOutward(m)
    }

    // MARK: audits

    /// Edges by vertex INDEX used by an odd/other-than-2 number of faces.
    private func indexEdgeDefects(_ m: Mesh3) -> (boundary: Int, nonManifold: Int) {
        var count: [UInt64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0..<3 {
                let a = m.indices[t + k], b = m.indices[t + (k + 1) % 3]
                count[UInt64(min(a, b)) << 32 | UInt64(max(a, b)), default: 0] += 1
            }
            t += 3
        }
        return (count.values.filter { $0 == 1 }.count, count.values.filter { $0 > 2 }.count)
    }

    /// The same by POSITION (1e-7 grid), for outputs that keep per-face vertices.
    private func positionBoundary(_ m: Mesh3) -> Int {
        func key(_ p: Vec3) -> SIMD3<Int64> { SIMD3(Int64((p.x * 1e7).rounded()), Int64((p.y * 1e7).rounded()), Int64((p.z * 1e7).rounded())) }
        struct E: Hashable { let a: SIMD3<Int64>, b: SIMD3<Int64> }
        var count: [E: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            let k = (0..<3).map { key(m.positions[Int(m.indices[t + $0])]) }
            for i in 0..<3 {
                let a = k[i], b = k[(i + 1) % 3]
                let lessThan = a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z)
                count[lessThan ? E(a: a, b: b) : E(a: b, b: a), default: 0] += 1
            }
            t += 3
        }
        return count.values.filter { $0 != 2 }.count
    }

    private func degenerate(_ m: Mesh3) -> Int {
        var n = 0, t = 0
        while t + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], c = m.positions[Int(m.indices[t + 2])]
            if len3(cross3(b - a, c - a)) < 1e-14 { n += 1 }
            t += 3
        }
        return n
    }

    private func assertSolid(_ m: Mesh3, volume: Double, tol: Double = 1e-9, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        let d = indexEdgeDefects(m)
        XCTAssertEqual(d.boundary, 0, "\(label): boundary edges", file: file, line: line)
        XCTAssertEqual(d.nonManifold, 0, "\(label): non-manifold edges", file: file, line: line)
        XCTAssertEqual(m.signedVolume, volume, accuracy: tol * max(1, abs(volume)), "\(label): volume", file: file, line: line)
        XCTAssertEqual(degenerate(m), 0, "\(label): degenerate faces", file: file, line: line)
        XCTAssertEqual(m.normals.count, m.positions.count, "\(label): normals", file: file, line: line)
        XCTAssertEqual(m.uvs.count, m.positions.count, "\(label): uvs", file: file, line: line)
    }

    private func polygonArea(_ r: Double, _ n: Int) -> Double { 0.5 * Double(n) * r * r * sin(2 * Double.pi / Double(n)) }

    // MARK: tests

    func testBoxMinusCylinder() {
        let n = 24
        let a = box(Vec3(-1, -1, -1), Vec3(1, 1, 1)), c = cylinderZ(radius: 0.5, z0: -2, z1: 2, n: n)
        let r = a.boolean(.difference, c)
        assertSolid(r.mesh, volume: 8 - polygonArea(0.5, n) * 2, "box-cyl")
        XCTAssertEqual(r.stats.constraintFailures, 0)
        // keepCreases: closed by position, hard edges kept
        var o = BooleanOptions(); o.weld = .keepCreases
        let k = a.booleanDifference(c, options: o)
        XCTAssertEqual(positionBoundary(k), 0)
        XCTAssertEqual(k.signedVolume, 8 - polygonArea(0.5, n) * 2, accuracy: 1e-9)
    }

    func testBlindHoleAndOffCentre() {
        let n = 32
        let a = box(Vec3(-1, -1, -1), Vec3(1, 1, 1)), c = cylinderZ(radius: 0.3, z0: 0.2, z1: 2, n: n, at: Vec3(0.3, -0.2, 0))
        assertSolid(a.booleanDifference(c), volume: 8 - polygonArea(0.3, n) * 0.8, "blind hole")
    }

    func testSphereIntersectBox() {
        let s = sphere(1, rings: 24, sectors: 32)
        let b = box(Vec3(0, -2, -2), Vec3(2, 2, 2))             // x ≥ 0 half-space, plane through the meridians
        assertSolid(s.booleanIntersection(b), volume: s.signedVolume / 2, "sphere∩box")
    }

    func testUnionOfOverlappingSpheres() {
        let a = sphere(1, rings: 20, sectors: 28), b = sphere(0.8, rings: 18, sectors: 26, at: Vec3(0.9, 0.2, 0.1))
        let u = a.booleanUnion(b), i = a.booleanIntersection(b)
        assertSolid(u, volume: u.signedVolume, "union")
        assertSolid(i, volume: i.signedVolume, "intersection")
        XCTAssertEqual(u.signedVolume + i.signedVolume, a.signedVolume + b.signedVolume, accuracy: 1e-9)
        XCTAssertGreaterThan(u.signedVolume, max(a.signedVolume, b.signedVolume))
        let d = a.booleanDifference(b)
        assertSolid(d, volume: a.signedVolume - i.signedVolume, "difference")
    }

    func testCoplanarFlushCut() {
        let a = box(Vec3(0, 0, 0), Vec3(2, 2, 2))
        assertSolid(a.booleanDifference(box(Vec3(0.5, 0.5, 1), Vec3(1.5, 1.5, 2))), volume: 7, "flush top pocket")
        assertSolid(a.booleanDifference(box(Vec3(0.5, 0.5, -1), Vec3(1.5, 1.5, 3))), volume: 6, "through")
        assertSolid(a.booleanDifference(box(Vec3(0, 0, 0), Vec3(2, 2, 1))), volume: 4, "half flush on four faces")
        assertSolid(a.booleanDifference(box(Vec3(0.5, 0, 0.5), Vec3(1.5, 1, 1.5))), volume: 8 - 1, "pocket in a side face")
        // same-facing coplanar union / intersection
        assertSolid(a.booleanUnion(box(Vec3(0.5, 0.5, 1), Vec3(1.5, 1.5, 2))), volume: 8, "union flush")
        assertSolid(a.booleanIntersection(box(Vec3(0.5, 0.5, 1), Vec3(1.5, 1.5, 2))), volume: 1, "intersect flush")
        // identical meshes
        assertSolid(a.booleanUnion(a), volume: 8, "A∪A")
        assertSolid(a.booleanIntersection(a), volume: 8, "A∩A")
        XCTAssertTrue(a.booleanDifference(a).isEmpty)
    }

    func testTouchingAtAFace() {
        let a = box(Vec3(0, 0, 0), Vec3(1, 1, 1))
        assertSolid(a.booleanUnion(box(Vec3(1, 0, 0), Vec3(2, 1, 1))), volume: 2, "full face")
        assertSolid(a.booleanUnion(box(Vec3(1, 0.25, 0.25), Vec3(2, 0.75, 0.75))), volume: 1.25, "partial face")
        assertSolid(a.booleanUnion(box(Vec3(1, 0.5, 0.5), Vec3(2, 1.5, 1.5))), volume: 1 + 1, "overhanging face")
        XCTAssertTrue(a.booleanIntersection(box(Vec3(1, 0, 0), Vec3(2, 1, 1))).isEmpty)
        assertSolid(a.booleanDifference(box(Vec3(1, 0, 0), Vec3(2, 1, 1))), volume: 1, "diff touching")
    }

    func testNestedAndDisjoint() {
        let a = box(Vec3(0, 0, 0), Vec3(4, 4, 4)), b = box(Vec3(1, 1, 1), Vec3(2, 2, 2))
        assertSolid(a.booleanUnion(b), volume: 64, "nested union")
        assertSolid(a.booleanIntersection(b), volume: 1, "nested intersection")
        assertSolid(a.booleanDifference(b), volume: 63, "cavity")
        XCTAssertTrue(b.booleanDifference(a).isEmpty)
        let far = box(Vec3(10, 0, 0), Vec3(11, 1, 1))
        assertSolid(a.booleanUnion(far), volume: 65, "disjoint union")
        XCTAssertTrue(a.booleanIntersection(far).isEmpty)
        assertSolid(a.booleanDifference(far), volume: 64, "disjoint diff")
    }

    func testSlice() {
        let s = sphere(1, rings: 24, sectors: 32)
        let (up, down) = s.slice(point: Vec3(0, 0.3, 0), normal: Vec3(0, 1, 0))
        assertSolid(up, volume: up.signedVolume, "above")
        assertSolid(down, volume: down.signedVolume, "below")
        XCTAssertEqual(up.signedVolume + down.signedVolume, s.signedVolume, accuracy: 1e-9)
        XCTAssertLessThan(up.signedVolume, down.signedVolume)
        // a tilted plane through a box
        let b = box(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        let (u2, d2) = b.slice(point: .zero, normal: Vec3(1, 1, 0.3))
        XCTAssertEqual(u2.signedVolume, 4, accuracy: 1e-9)
        XCTAssertEqual(d2.signedVolume, 4, accuracy: 1e-9)
        XCTAssertEqual(indexEdgeDefects(u2).boundary, 0)
    }

    func testUVsAreInterpolated() {
        let s = sphere(1, rings: 24, sectors: 32)
        let r = s.booleanIntersection(box(Vec3(0, -2, -2), Vec3(2, 2, 2)))
        XCTAssertTrue(r.uvs.allSatisfy { $0.x >= -1e-9 && $0.x <= 1 + 1e-9 && $0.y >= -1e-9 && $0.y <= 1 + 1e-9 })
        XCTAssertTrue(r.uvs.contains { $0.x > 0.1 && $0.x < 0.9 && $0.y > 0.1 && $0.y < 0.9 })
        XCTAssertTrue(r.normals.allSatisfy { abs(len3($0) - 1) < 1e-6 })
    }

    func testRepeatedAndChainedOperations() {
        let a = box(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        var m = a
        for k in 0..<3 {
            m = m.booleanDifference(cylinderZ(radius: 0.15, z0: -2, z1: 2, n: 16, at: Vec3(-0.5 + 0.5 * Double(k), 0.2, 0)))
        }
        assertSolid(m, volume: 8 - 3 * polygonArea(0.15, 16) * 2, "three holes")
    }

    func testDegenerateAlignments() {
        let s = sphere(1, rings: 16, sectors: 24)
        let ss = s.boolean(.union, s); print("SS stats", ss.stats, ss.mesh.triangleCount, s.triangleCount)
        assertSolid(ss.mesh, volume: s.signedVolume, "S∪S")
        XCTAssertTrue(s.booleanDifference(s).isEmpty)
        // same tessellation shifted by a whole number of cells along the axis: coincident vertices and edges
        let a = box(Vec3(0, 0, 0), Vec3(2, 2, 2))
        assertSolid(a.booleanUnion(box(Vec3(1, 1, 1), Vec3(3, 3, 3))), volume: 15, "corner overlap")
        assertSolid(a.booleanDifference(box(Vec3(1, 1, 1), Vec3(3, 3, 3))), volume: 7, "corner notch")
        assertSolid(a.booleanIntersection(box(Vec3(1, 1, 1), Vec3(3, 3, 3))), volume: 1, "corner common")
        // cylinder axis exactly along a box edge diagonal / through a face diagonal
        let c = cylinderZ(radius: 0.5, z0: -1, z1: 3, n: 16, at: Vec3(1, 1, 0))
        let r = a.boolean(.difference, c)
        assertSolid(r.mesh, volume: 8 - polygonArea(0.5, 16) * 2, "cylinder on the diagonal")
        XCTAssertEqual(r.stats.constraintFailures, 0)
        let e = cylinderZ(radius: 0.5, z0: -1, z1: 3, n: 16, at: Vec3(0, 0, 0))
        assertSolid(a.booleanDifference(e), volume: 8 - polygonArea(0.5, 16) * 2 / 4, "cylinder on an edge")
    }

    // MARK: GPU twins

    func testGPUBroadphaseMatchesBVH() throws {
        guard let g = BooleanGPU.shared else { throw XCTSkip("no Metal device") }
        let a = sphere(1, rings: 40, sectors: 56), b = sphere(0.9, rings: 36, sectors: 48, at: Vec3(0.7, 0.1, 0.2))
        let job = BooleanJob(a, b, options: BooleanOptions())
        job.weld()
        let P = job.positions()
        let ba = job.boxes(0, P), bb = job.boxes(1, P)
        let gpu = try XCTUnwrap(g.broadphase(ba, bb))
        let bvh = BoxBVH(lo: bb.0, hi: bb.1)
        var cpu = Set<SIMD2<Int32>>()
        for i in 0..<ba.0.count { bvh.query(ba.0[i], ba.1[i]) { cpu.insert(SIMD2(Int32(i), Int32($0))) } }
        let gs = Set(gpu)
        print("broadphase gpu=\(gpu.count) cpu=\(cpu.count) missing=\(cpu.subtracting(gs).count) sample=\(Array(cpu.subtracting(gs).prefix(3)))")
        XCTAssertEqual(gs.count, gpu.count, "no duplicate pairs")
        XCTAssertTrue(cpu.isSubset(of: gs), "GPU set must contain every exact pair")
        XCTAssertLessThan(Double(gs.count), Double(cpu.count) * 1.05 + 10, "GPU margin should add almost nothing")
    }

    func testGPUWindingMatchesCPU() throws {
        guard let g = BooleanGPU.shared else { throw XCTSkip("no Metal device") }
        let s = sphere(1, rings: 40, sectors: 56)
        var corners: [Vec3] = []
        for i in s.indices { corners.append(s.positions[Int(i)]) }
        let pts = (0..<500).map { i -> Vec3 in
            let t = Double(i) / 500
            return Vec3(2 * sin(t * 40) * t * 1.2, cos(t * 31) * 1.3 * t, sin(t * 17) * 1.2)
        }.filter { abs(len3($0) - 1) > 0.02 }
        let cpu = BooleanJob.windingCPU(pts, corners)
        let gpu = try XCTUnwrap(g.winding(points: pts, corners: corners))
        for i in 0..<pts.count { XCTAssertEqual(Double(gpu[i]), cpu[i], accuracy: 2e-3) }
        for (p, w) in zip(pts, cpu) { XCTAssertEqual(w, len3(p) < 1 ? 1 : 0, accuracy: 0.05) }
    }

    func testGPUAndCPUPipelinesAgree() {
        let a = sphere(1, rings: 120, sectors: 150), b = sphere(0.8, rings: 100, sectors: 130, at: Vec3(0.7, 0.2, 0.1))
        var cpuOpt = BooleanOptions(); cpuOpt.useGPU = false
        let g = a.boolean(.union, b), c = a.boolean(.union, b, options: cpuOpt)
        XCTAssertEqual(g.mesh.signedVolume, c.mesh.signedVolume, accuracy: 1e-9)
        XCTAssertEqual(indexEdgeDefects(g.mesh).boundary, 0)
        print("GPU flags: broadphase=\(g.stats.broadphaseOnGPU) winding=\(g.stats.windingOnGPU)")
    }

    // MARK: scale

    func testFiftyThousandTriangleTiming() {
        let a = sphere(1, rings: 130, sectors: 200), b = sphere(0.9, rings: 130, sectors: 200, at: Vec3(0.6, 0.15, 0.1))
        XCTAssertGreaterThan(a.triangleCount, 50_000)
        for (name, opt) in [("GPU", BooleanOptions()), ("CPU", { var o = BooleanOptions(); o.useGPU = false; return o }())] {
            let t0 = Date()
            let r = a.boolean(.union, b, options: opt)
            let dt = Date().timeIntervalSince(t0)
            print("[boolean 2×\(a.triangleCount) tris, \(name)] \(String(format: "%.2f", dt)) s  \(r.mesh.triangleCount) tris out  gpuBroad=\(r.stats.broadphaseOnGPU) gpuWind=\(r.stats.windingOnGPU) cut=\(r.stats.cutTriangles) pairs=\(r.stats.candidatePairs)/\(r.stats.intersectingPairs) steps=\(r.stats.seconds.mapValues { String(format: "%.2f", $0) })")
            XCTAssertLessThan(dt, 30)
            XCTAssertEqual(indexEdgeDefects(r.mesh).boundary, 0)
            XCTAssertEqual(indexEdgeDefects(r.mesh).nonManifold, 0)
            XCTAssertEqual(r.stats.constraintFailures, 0)
            let i = a.boolean(.intersection, b, options: opt).mesh
            XCTAssertEqual(r.mesh.signedVolume + i.signedVolume, a.signedVolume + b.signedVolume, accuracy: 1e-8)
        }
    }
}
