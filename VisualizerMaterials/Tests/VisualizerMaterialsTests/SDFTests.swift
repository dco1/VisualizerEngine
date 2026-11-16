import XCTest
import simd
@testable import VisualizerMaterials

/// Item C, the SDF modelling kit. Judged by VOLUME against analytic figures, by field values at named points, by closure
/// (every directed edge has exactly its reverse) and by winding (positive signed volume, normals with the faces).
final class SDFTests: XCTestCase {

    // MARK: helpers

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], c = m.positions[Int(m.indices[t + 2])]
            v += dot3(a, cross3(b, c)) / 6; t += 3
        }
        return v
    }

    /// Directed edges not matched by their reverse, exactly once each (0 = closed, consistently wound, edge-manifold).
    private func unmatchedEdges(_ m: Mesh3) -> Int {
        var count: [UInt64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for e in 0..<3 {
                let a = UInt64(m.indices[t + e]), b = UInt64(m.indices[t + (e + 1) % 3])
                count[a << 32 | b, default: 0] += 1
            }
            t += 3
        }
        return count.filter { $0.value != 1 || count[($0.key & 0xffff_ffff) << 32 | ($0.key >> 32)] != 1 }.count
    }

    /// Triangles whose winding disagrees with their vertex normals.
    private func inverted(_ m: Mesh3) -> Int {
        var bad = 0, t = 0
        while t + 2 < m.indices.count {
            let i = (0..<3).map { Int(m.indices[t + $0]) }
            let n = cross3(m.positions[i[1]] - m.positions[i[0]], m.positions[i[2]] - m.positions[i[0]])
            if dot3(n, m.normals[i[0]] + m.normals[i[1]] + m.normals[i[2]]) <= 0 { bad += 1 }
            t += 3
        }
        return bad
    }

    /// `clipped`: the shape runs out of the box and is capped flat there (the cap's vertex normals follow the field, not the cap).
    private func assertSound(_ m: Mesh3, _ what: String, clipped: Bool = false, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertFalse(m.isEmpty, what, file: file, line: line)
        XCTAssertEqual(unmatchedEdges(m), 0, "\(what): not closed / not manifold", file: file, line: line)
        // A crease (a boolean's edge, a cell-sized feature) can average a vertex normal across it: a handful of triangles.
        if !clipped {
            XCTAssertLessThanOrEqual(inverted(m), m.triangleCount / 200, "\(what): triangles wound against normals", file: file, line: line)
        }
        XCTAssertGreaterThan(volume(m), 0, "\(what): inside-out", file: file, line: line)
        XCTAssertTrue(m.normals.allSatisfy { abs(len3($0) - 1) < 1e-3 }, "\(what): unit normals", file: file, line: line)
    }

    private func mesh(_ s: SDF, _ lo: Vec3, _ hi: Vec3, cell: Double, _ backend: SDFMeshOptions.Backend = .cpu) -> Mesh3 {
        s.mesh(lo: lo, hi: hi, cell: cell, options: SDFMeshOptions(backend: backend))
    }

    private func boxMesh(_ lo: Vec3, _ hi: Vec3) -> Mesh3 {
        var m = Mesh3()
        let (x0, y0, z0, x1, y1, z1) = (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
        m.addQuad(Vec3(x1,y0,z0), Vec3(x1,y1,z0), Vec3(x1,y1,z1), Vec3(x1,y0,z1), outward: Vec3( 1,0,0))
        m.addQuad(Vec3(x0,y0,z0), Vec3(x0,y0,z1), Vec3(x0,y1,z1), Vec3(x0,y1,z0), outward: Vec3(-1,0,0))
        m.addQuad(Vec3(x0,y1,z0), Vec3(x0,y1,z1), Vec3(x1,y1,z1), Vec3(x1,y1,z0), outward: Vec3(0, 1,0))
        m.addQuad(Vec3(x0,y0,z0), Vec3(x1,y0,z0), Vec3(x1,y0,z1), Vec3(x0,y0,z1), outward: Vec3(0,-1,0))
        m.addQuad(Vec3(x0,y0,z1), Vec3(x1,y0,z1), Vec3(x1,y1,z1), Vec3(x0,y1,z1), outward: Vec3(0,0, 1))
        m.addQuad(Vec3(x0,y0,z0), Vec3(x0,y1,z0), Vec3(x1,y1,z0), Vec3(x1,y0,z0), outward: Vec3(0,0,-1))
        return m
    }

    // MARK: primitives and volumes

    func testSphereVolume() {
        let m = mesh(.sphere(radius: 1), Vec3(repeating: -1.2), Vec3(repeating: 1.2), cell: 0.04)
        assertSound(m, "sphere")
        XCTAssertEqual(volume(m), 4.0 / 3 * .pi, accuracy: 4.19 * 0.01)
        // Vertices lie on the surface and normals point out.
        for (p, n) in zip(m.positions, m.normals) {
            XCTAssertEqual(len3(p), 1, accuracy: 0.01)
            XCTAssertGreaterThan(dot3(normalize3(p), n), 0.99)
        }
    }

    func testBoxAndRoundedBoxVolume() {
        let box = SDF.box(size: Vec3(1, 2, 3))
        let m = mesh(box, Vec3(-1, -1.5, -2), Vec3(1, 1.5, 2), cell: 0.07)   // faces fall between grid planes
        assertSound(m, "box")
        XCTAssertEqual(volume(m), 6, accuracy: 6 * 0.02)

        let r = SDF.roundedBox(size: Vec3(1, 1, 1), radius: 0.2)
        let rm = mesh(r, Vec3(repeating: -0.8), Vec3(repeating: 0.8), cell: 0.03)
        assertSound(rm, "rounded box")
        // 1 − (4 − π/… ) : cube with every edge and corner rounded by radius a.
        let a = 0.2, s = 1.0 - 2 * a
        let analytic = s * s * s + 6 * s * s * a + 12 * s * .pi * a * a / 4 + 4.0 / 3 * .pi * a * a * a
        XCTAssertEqual(volume(rm), analytic, accuracy: analytic * 0.015)
    }

    func testOtherPrimitives() {
        let cap = SDF.capsule(from: Vec3(0, -0.5, 0), to: Vec3(0, 0.5, 0), radius: 0.3)
        let cm = mesh(cap, Vec3(-0.6, -1, -0.6), Vec3(0.6, 1, 0.6), cell: 0.03)
        assertSound(cm, "capsule")
        XCTAssertEqual(volume(cm), .pi * 0.09 * 1 + 4.0 / 3 * .pi * 0.027, accuracy: 0.02)

        let tor = SDF.torus(major: 0.5, minor: 0.15)
        let tm = mesh(tor, Vec3(-0.8, -0.3, -0.8), Vec3(0.8, 0.3, 0.8), cell: 0.025)
        assertSound(tm, "torus")
        XCTAssertEqual(volume(tm), 2 * .pi * .pi * 0.5 * 0.0225, accuracy: 0.0222 * 0.05)

        let cone = SDF.cone(height: 1, bottomRadius: 0.5, topRadius: 0)
        let km = mesh(cone, Vec3(-0.7, -0.7, -0.7), Vec3(0.7, 0.7, 0.7), cell: 0.03)
        assertSound(km, "cone")
        XCTAssertEqual(volume(km), .pi * 0.25 / 3, accuracy: 0.262 * 0.03)

        // A plane is a half-space: capped by the box, it fills the box below it.
        let pm = mesh(.plane(normal: Vec3(0, 1, 0), offset: 0.1), Vec3(repeating: -0.5), Vec3(repeating: 0.5), cell: 0.05)
        assertSound(pm, "plane", clipped: true)
        XCTAssertEqual(volume(pm), 0.6, accuracy: 0.6 * 0.12)    // the box's own boundary layer is clipped
    }

    func testTransforms() {
        let s = SDF.box(size: Vec3(1, 1, 1)).scaled(by: 2).rotated(angle: .pi / 4, axis: Vec3(0, 1, 0)).translated(by: Vec3(3, 0, 0))
        XCTAssertEqual(s(Vec3(3, 0, 0)), -1, accuracy: 1e-9)            // centre of a 2 m box
        XCTAssertEqual(s(Vec3(3 + 1.5, 0, 0)), 1.5 * 0.5 * 2.0.squareRoot() * 2.0.squareRoot() * 0 + (1.5 - 1) * 1, accuracy: 0.45)
        // A diagonal of the rotated 2 m square reaches √2 out along x.
        XCTAssertEqual(s(Vec3(3 + 2.0.squareRoot(), 0, 0)), 0, accuracy: 1e-9)
        let t = SDF.box(size: Vec3(0.2, 2, 0.2)).twisted(rate: 1)
        XCTAssertLessThan(t(Vec3(0, 0.5, 0)), 0)
        XCTAssertGreaterThan(t(Vec3(0.5, 0, 0)), 0)
    }

    // MARK: booleans, offset, shell, carve

    func testHardBooleans() {
        let a = SDF.sphere(radius: 1), b = SDF.sphere(radius: 1).translated(by: Vec3(1, 0, 0))
        let lo = Vec3(-1.4, -1.2, -1.2), hi = Vec3(2.4, 1.2, 1.2)
        let sphereV = 4.0 / 3 * Double.pi
        let lens = Double.pi * 1.0 / 12 * (4 + 1) * 1 // lens of two unit spheres at d=1: π(4r+d)(2r−d)²/12 = 5π/12
        let u = mesh(a.union(b), lo, hi, cell: 0.05), i = mesh(a.intersecting(b), lo, hi, cell: 0.04),
            d = mesh(a.subtracting(b), lo, hi, cell: 0.05)
        assertSound(u, "union"); assertSound(i, "intersect"); assertSound(d, "subtract")
        XCTAssertEqual(volume(i), lens, accuracy: lens * 0.03)
        XCTAssertEqual(volume(u), 2 * sphereV - lens, accuracy: sphereV * 0.02)
        XCTAssertEqual(volume(d), sphereV - lens, accuracy: sphereV * 0.02)
    }

    func testSmoothUnionBlends() {
        let a = SDF.sphere(radius: 0.5).translated(by: Vec3(-0.6, 0, 0)), b = SDF.sphere(radius: 0.5).translated(by: Vec3(0.6, 0, 0))
        let hard = a.union(b), soft = a.union(b, smooth: 0.6)
        // Never larger than the hard union, and strictly smaller in the neck between the spheres.
        for p in [Vec3(0, 0, 0), Vec3(0, 0.2, 0), Vec3(0.3, 0.1, 0.2), Vec3(2, 0, 0)] { XCTAssertLessThanOrEqual(soft(p), hard(p) + 1e-12) }
        XCTAssertGreaterThan(hard(Vec3(0, 0.2, 0)), 0)               // the hard union has a gap there…
        XCTAssertLessThan(soft(Vec3(0, 0.2, 0)), 0)                  // …the blend fills it
        let lo = Vec3(-1.5, -1, -1), hi = Vec3(1.5, 1, 1)
        let vh = volume(mesh(hard, lo, hi, cell: 0.04)), vs = mesh(soft, lo, hi, cell: 0.04)
        assertSound(vs, "smooth union")
        XCTAssertGreaterThan(volume(vs), vh + 0.02)
        // Smooth subtract and intersect stay closed and ordered.
        let ss = mesh(a.union(b).subtracting(SDF.sphere(radius: 0.3), smooth: 0.15), lo, hi, cell: 0.04)
        assertSound(ss, "smooth subtract")
        let si = mesh(a.intersecting(b.translated(by: Vec3(-0.4, 0, 0)), smooth: 0.2), lo, hi, cell: 0.04)
        assertSound(si, "smooth intersect")
    }

    func testOffsetAndShellThickness() {
        let sphere = SDF.sphere(radius: 1)
        XCTAssertEqual(sphere.offset(by: 0.25)(Vec3(1.25, 0, 0)), 0, accuracy: 1e-12)
        XCTAssertEqual(sphere.offset(by: -0.25)(Vec3(0.75, 0, 0)), 0, accuracy: 1e-12)

        let lo = Vec3(repeating: -1.2), hi = Vec3(repeating: 1.2)
        let t = 0.1
        let shell = sphere.shell(thickness: t)                       // wall from r = 0.9 to r = 1
        XCTAssertLessThan(shell(Vec3(0.95, 0, 0)), 0)
        XCTAssertGreaterThan(shell(Vec3(0.85, 0, 0)), 0)
        XCTAssertGreaterThan(shell(Vec3(1.05, 0, 0)), 0)
        let m = mesh(shell, lo, hi, cell: 0.02)
        assertSound(m, "shell")                                       // outer and inner skin: a closed solid
        XCTAssertEqual(volume(m), 4.0 / 3 * .pi * (1 - 0.729), accuracy: 0.31 * 0.04)
        // Measured: the wall is `t` thick along a ray from the centre.
        let xs = stride(from: 0.0, to: 1.2, by: 0.001).filter { shell(Vec3($0, 0, 0)) < 0 }
        XCTAssertEqual((xs.last! - xs.first!), t, accuracy: 0.002)
        XCTAssertEqual(sphere.shell(thickness: t, side: .centered)(Vec3(1.04, 0, 0)), -0.01, accuracy: 1e-9)
        XCTAssertLessThan(sphere.shell(thickness: t, side: .outside)(Vec3(1.05, 0, 0)), 0)
    }

    func testCarveDepth() {
        let block = SDF.box(size: Vec3(1, 1, 1))
        // A tall 0.4 m square tool through the top face at the middle; carve it 0.1 m deep only.
        let tool = SDF.box(size: Vec3(0.4, 1, 0.4)).translated(by: Vec3(0, 0.5, 0))   // spans y 0…1: only the top face is cut
        let carved = block.carved(tool: tool, depth: 0.1)
        XCTAssertGreaterThan(carved(Vec3(0, 0.45, 0)), 0)             // inside the dent: empty
        XCTAssertLessThan(carved(Vec3(0, 0.35, 0)), 0)                // below its floor (0.4): solid
        XCTAssertLessThan(carved(Vec3(0.3, 0.45, 0)), 0)              // beside the tool: untouched
        XCTAssertLessThan(carved(Vec3(0, -0.45, 0)), 0)               // the far side is untouched
        let lo = Vec3(repeating: -0.75), hi = Vec3(repeating: 0.75)
        let m = mesh(carved, lo, hi, cell: 0.025)
        assertSound(m, "carve")
        XCTAssertEqual(volume(m), 1 - 0.4 * 0.4 * 0.1, accuracy: 0.02)
        // Falloff eases the dent: the removed volume shrinks and the field stays closed.
        let soft = block.carved(tool: tool, depth: 0.1, falloff: 0.08)
        let sm = mesh(soft, lo, hi, cell: 0.025)
        assertSound(sm, "carve falloff")
        XCTAssertGreaterThan(volume(sm), volume(m) - 1e-3)
        XCTAssertLessThan(soft(Vec3(0, 0.45, 0)), 0.2)
    }

    // MARK: mesher

    func testWatertightAcrossShapes() {
        let model = SDF.roundedBox(size: Vec3(1, 0.6, 0.8), radius: 0.1)
            .union(SDF.capsule(from: Vec3(0, 0, 0), to: Vec3(0, 0.8, 0), radius: 0.15), smooth: 0.1)
            .subtracting(SDF.sphere(radius: 0.2).translated(by: Vec3(0.3, 0, 0.4)))
            .union(SDF.torus(major: 0.35, minor: 0.06).translated(by: Vec3(0, 0.9, 0)))
        let m = mesh(model, Vec3(-0.8, -0.5, -0.7), Vec3(0.8, 1.2, 0.7), cell: 0.03)
        // Two parts (the torus floats): still closed and outward.
        XCTAssertEqual(unmatchedEdges(m), 0)
        XCTAssertLessThanOrEqual(inverted(m), m.triangleCount / 200)
        XCTAssertGreaterThan(volume(m), 0.1)
        // A solid that leaves the box is capped at it, not left open.
        let big = mesh(.sphere(radius: 2), Vec3(repeating: -1), Vec3(repeating: 1), cell: 0.1)
        XCTAssertEqual(unmatchedEdges(big), 0)
        XCTAssertEqual(volume(big), 8, accuracy: 0.8)
    }

    func testBoxProjectedUVs() {
        let opts = SDFMeshOptions(backend: .cpu, uvs: .boxProjected(metersPerUnit: 0.5))
        let m = SDF.box(size: Vec3(1, 1, 1)).mesh(lo: Vec3(repeating: -0.8), hi: Vec3(repeating: 0.8), cell: 0.05, options: opts)
        XCTAssertEqual(m.uvs.count, m.positions.count)
        for (p, (n, uv)) in zip(m.positions, zip(m.normals, m.uvs)) where abs(n.y) > 0.99 {
            XCTAssertEqual(uv.x, p.x / 0.5, accuracy: 1e-9); XCTAssertEqual(uv.y, p.z / 0.5, accuracy: 1e-9)
        }
    }

    // MARK: mesh-to-SDF

    func testMeshToSDFRoundTripOnBox() {
        let box = boxMesh(Vec3(-0.5, -0.25, -0.75), Vec3(0.5, 0.25, 0.75))        // flat-shaded: 3 verts a triangle
        let field = SDF.mesh(box), exact = SDF.box(size: Vec3(1, 0.5, 1.5))
        var rng = SplitMix64(seed: 7)
        var worst = 0.0
        for _ in 0..<2000 {
            let p = Vec3(Double.random(in: -1.5...1.5, using: &rng), Double.random(in: -1.5...1.5, using: &rng),
                         Double.random(in: -1.5...1.5, using: &rng))
            worst = Swift.max(worst, abs(field(p) - exact(p)))
        }
        XCTAssertLessThan(worst, 1e-9, "mesh field vs analytic box, signed (corners, edges and faces all hit)")
        // Re-mesh the mesh field: the volume comes back.
        let back = mesh(SDF.mesh(box), Vec3(-1, -0.6, -1.1), Vec3(1, 0.6, 1.1), cell: 0.05)
        assertSound(back, "remeshed box")
        XCTAssertEqual(volume(back), 0.75, accuracy: 0.75 * 0.03)
        // And a meshed sphere is a sphere to the field.
        let ball = mesh(.sphere(radius: 0.8), Vec3(repeating: -1), Vec3(repeating: 1), cell: 0.05)
        let bf = SDF.mesh(ball)
        XCTAssertEqual(bf(Vec3(0, 0, 0)), -0.8, accuracy: 0.02)
        XCTAssertEqual(bf(Vec3(1.3, 0, 0)), 0.5, accuracy: 0.02)
        XCTAssertTrue(SDF.mesh(box).gpuProgram() == nil)
    }

    // MARK: GPU == CPU

    private func gpuModel() -> SDF {
        SDF.roundedBox(size: Vec3(1, 0.7, 0.8), radius: 0.12)
            .union(SDF.capsule(from: Vec3(-0.2, 0, 0), to: Vec3(0.3, 0.6, 0.1), radius: 0.18), smooth: 0.12)
            .subtracting(SDF.cone(height: 0.6, bottomRadius: 0.25, topRadius: 0.05).rotated(angle: 0.6, axis: Vec3(1, 0, 1)).translated(by: Vec3(0.3, 0, 0.3)))
            .union(SDF.torus(major: 0.3, minor: 0.07).scaled(by: 1.2).translated(by: Vec3(0, 0.5, 0)))
            .carved(tool: SDF.sphere(radius: 0.15).translated(by: Vec3(-0.35, 0.35, 0)), depth: 0.05, falloff: 0.04)
            .intersecting(SDF.plane(normal: Vec3(0.2, 1, 0.1), offset: 0.9))
            .shell(thickness: 0.08)
    }

    func testGPUMatchesCPU() throws {
        guard SDFGPU.shared != nil else { throw XCTSkip("no Metal device") }
        let model = gpuModel()
        let program = try XCTUnwrap(model.gpuProgram())
        let lo = Vec3(-0.9137, -0.6071, -0.8093), cell = 0.03, cells = SIMD3(60, 60, 54)
        let cpu = model.cpuGrid(lo: lo, cell: cell, cells: cells)
        let gpu = try XCTUnwrap(SDFGPU.shared?.grid(program, lo: lo, cell: cell, cells: cells))
        XCTAssertEqual(cpu.count, gpu.count)
        var worst: Float = 0
        for (a, b) in zip(cpu, gpu) { worst = Swift.max(worst, abs(a - b)) }
        XCTAssertLessThan(worst, 2e-4, "grid: worst |CPU − GPU| (Float)")

        let hi = lo + Vec3(Double(cells.x), Double(cells.y), Double(cells.z)) * cell
        let mc = model.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .cpu))
        let mg = model.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .gpu))
        assertSound(mc, "cpu model"); assertSound(mg, "gpu model")
        XCTAssertEqual(Double(mg.positions.count), Double(mc.positions.count), accuracy: Double(mc.positions.count) * 0.001 + 2)
        XCTAssertEqual(volume(mg), volume(mc), accuracy: volume(mc) * 0.002)
        if mg.positions.count == mc.positions.count {
            var wp = 0.0, wn = 1.0
            for i in 0..<mc.positions.count {
                wp = Swift.max(wp, len3(mc.positions[i] - mg.positions[i]))
                wn = Swift.min(wn, dot3(mc.normals[i], mg.normals[i]))
            }
            XCTAssertLessThan(wp, cell * 0.02, "vertex positions")
            XCTAssertGreaterThan(wn, 0.98, "normals")
        }
    }

    func testGPURefusesWhatItCannotRun() {
        XCTAssertNil(SDF.custom { len3($0) - 1 }.gpuProgram())
        XCTAssertNil(SDF.sphere(radius: 1).twisted(rate: 1).gpuProgram())
        XCTAssertNotNil(gpuModel().gpuProgram())
        // `.auto` falls back to the CPU for them, and still meshes.
        let m = SDF.custom { len3($0) - 0.5 }.mesh(lo: Vec3(repeating: -0.7), hi: Vec3(repeating: 0.7), cell: 0.05)
        XCTAssertEqual(unmatchedEdges(m), 0)
    }

    func testTimings() throws {
        let model = gpuModel()
        let lo = Vec3(-0.9, -0.6, -0.8), hi = Vec3(0.9, 0.6, 0.8)
        let cell = 0.012       // ≈ 150 × 100 × 134 cells
        var t = Date()
        let mc = model.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .cpu))
        let cpuT = Date().timeIntervalSince(t)
        var line = String(format: "SDF timing: %d tris, CPU %.0f ms", mc.triangleCount, cpuT * 1000)
        if SDFGPU.shared != nil {
            _ = model.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .gpu))     // pipelines warm
            t = Date()
            let mg = model.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .gpu))
            line += String(format: ", GPU %.0f ms (%d tris)", Date().timeIntervalSince(t) * 1000, mg.triangleCount)
        }
        print(line)
        XCTAssertGreaterThan(mc.triangleCount, 1000)
    }
}
