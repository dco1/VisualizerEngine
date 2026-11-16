import XCTest
import simd
@testable import VisualizerMaterials

/// Modelling-kit item M2: every GPU modifier against its CPU oracle, the resident evaluator, and timings.
final class ModifierGPUTests: XCTestCase {

    override func setUpWithError() throws {
        #if canImport(Metal)
        try XCTSkipIf(ModifierGPU.shared == nil, "no Metal device")
        #else
        throw XCTSkip("no Metal")
        #endif
    }

    // MARK: builders

    /// Flat-shaded bumpy sheet (4 vertices a quad, as `addQuad` makes them), `n` × `n` quads over 1 m × 1 m.
    private func sheet(_ n: Int) -> Mesh3 {
        func h(_ x: Double, _ z: Double) -> Double { 0.08 * sin(7 * x) * cos(5 * z) }
        var m = Mesh3()
        for i in 0..<n { for k in 0..<n {
            let x0 = Double(i) / Double(n) - 0.5, x1 = Double(i + 1) / Double(n) - 0.5
            let z0 = Double(k) / Double(n) - 0.5, z1 = Double(k + 1) / Double(n) - 0.5
            m.addQuad(Vec3(x0, h(x0, z0), z0), Vec3(x0, h(x0, z1), z1), Vec3(x1, h(x1, z1), z1), Vec3(x1, h(x1, z0), z0), outward: Vec3(0, 1, 0))
        } }
        return m
    }

    /// Indexed UV sphere with smooth normals.
    private func sphere(radius: Double, centre: Vec3, seg: Int) -> Mesh3 {
        var m = Mesh3()
        for i in 0...seg { for j in 0...seg {
            let th = Double.pi * Double(i) / Double(seg), ph = 2 * Double.pi * Double(j) / Double(seg)
            let n = Vec3(sin(th) * cos(ph), cos(th), sin(th) * sin(ph))
            m.positions.append(centre + n * radius); m.normals.append(n); m.uvs.append(Vec2(Double(j) / Double(seg), Double(i) / Double(seg)))
        } }
        for i in 0..<seg { for j in 0..<seg {
            let a = UInt32(i * (seg + 1) + j), b = a + 1, c = a + UInt32(seg + 1), d = c + 1
            m.indices += [a, c, b, b, c, d]
        } }
        return m
    }

    // MARK: comparisons

    private func extent(_ m: Mesh3) -> Double {
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for p in m.positions { lo = simd.min(lo, p); hi = simd.max(hi, p) }
        return len3(hi - lo)
    }

    /// Max position error (relative to the mesh's extent) and min normal dot.
    private var extentBase = Mesh3()
    private func compare(_ cpu: Mesh3, _ gpu: Mesh3, tol: Double = 2e-5, outliers frac: Double = 0, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(cpu.positions.count, gpu.positions.count, "vertex count", file: file, line: line)
        XCTAssertEqual(cpu.indices, gpu.indices, "indices", file: file, line: line)
        XCTAssertEqual(cpu.normals.count, gpu.normals.count, file: file, line: line)
        XCTAssertEqual(cpu.uvs.count, gpu.uvs.count, file: file, line: line)
        guard cpu.positions.count == gpu.positions.count else { return }
        let ext = extent(cpu)
        // `frac` of the vertices may differ (ties between equally near triangles, faces collapsed to a sliver whose turn is ill-defined)
        let hasN = cpu.normals.count == gpu.normals.count
        let badList = cpu.positions.indices.filter {
            len3(cpu.positions[$0] - gpu.positions[$0]) / ext > tol || (hasN && dot3(cpu.normals[$0], gpu.normals[$0]) < 0.9995)
        }
        let bad = Set(badList)
        var worst = 0.0, worstN = 1.0
        for i in cpu.positions.indices where !bad.contains(i) || frac == 0 {
            worst = Swift.max(worst, len3(cpu.positions[i] - gpu.positions[i]))
            if hasN { worstN = Swift.min(worstN, dot3(cpu.normals[i], gpu.normals[i])) }
        }
        print(String(format: "    max position error %.2e (%.2e of extent), min normal dot %.7f, %d of %d vertices outside tolerance",
                     worst, worst / ext, worstN, bad.count, cpu.positions.count))
        XCTAssertLessThanOrEqual(Double(bad.count), frac * Double(cpu.positions.count), "vertices outside tolerance", file: file, line: line)
        for i in cpu.uvs.indices { XCTAssertEqual(cpu.uvs[i].x, gpu.uvs[i].x, accuracy: 1e-5, file: file, line: line) }
    }

    private func check(_ name: String, _ m: any Modifier, _ base: Mesh3, tol: Double = 2e-5, outliers: Double = 0, file: StaticString = #filePath, line: UInt = #line) {
        print("  \(name)")
        extentBase = base
        let cpu = m.apply(base, backend: .cpu), gpu = m.apply(base, backend: .gpu)
        compare(cpu, gpu, tol: tol, outliers: outliers, file: file, line: line)
    }

    // MARK: kernels

    func testWeldedNormalsMatchCPU() throws {
        let base = sheet(24)
        let gpu = ModifierGPU.shared!
        let s = GPUMeshState(mesh: base, gpu: gpu)
        let cb = gpu.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        let wN = gpu.weldedNormals(s, enc)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let w = WeldedMesh(base)
        let P = wN.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        var worst = 1.0
        for v in base.positions.indices {
            let g = P[Int(s.topo.cornerID[v])], c = w.normals[w.cornerID[v]]
            worst = Swift.min(worst, dot3(c, Vec3(Double(g.x), Double(g.y), Double(g.z))))
        }
        print("  welded normals: min dot \(worst), \(s.topo.nW) welded of \(base.positions.count)")
        XCTAssertGreaterThan(worst, 0.99999)
        XCTAssertEqual(s.topo.nW, w.points.count)
    }

    func testDisplaceNoiseMatchesCPU() {
        let base = sheet(40)
        check("noise, along normal", NoiseDisplaceModifier(strength: 0.06, frequency: 5, octaves: 4, seed: 12345), base)
        check("noise, fixed direction, 6 octaves, big seed",
              NoiseDisplaceModifier(strength: 0.04, frequency: 9, octaves: 6, seed: 0xDEAD_BEEF_1234_5678, direction: .fixed(Vec3(0.3, 1, 0.2))), base)
        // the noise value itself, through a zero-normal fixed direction trick: displace along +Y of a flat point set
        var pts = Mesh3()
        for i in 0..<64 { pts.addQuad(Vec3(Double(i) * 0.137 - 4, 0, 0.5), Vec3(Double(i) * 0.137 - 4, 0, 0.6), Vec3(Double(i) * 0.137 - 3.9, 0, 0.6),
                                      Vec3(Double(i) * 0.137 - 3.9, 0, 0.5), outward: Vec3(0, 1, 0)) }
        check("noise values over a negative-coordinate line", NoiseDisplaceModifier(strength: 1, frequency: 2.3, octaves: 3, seed: 99, direction: .fixed(Vec3(0, 1, 0))), pts, tol: 1e-5)
    }

    func testSimpleDeformMatchesCPU() {
        let base = sheet(40)
        let modes: [(String, SimpleDeformModifier.Mode)] = [("bend", .bend(angle: 1.2)), ("twist", .twist(angle: 2.0)),
                                                              ("taper", .taper(factor: -0.6)), ("stretch", .stretch(factor: 0.8))]
        for (n, mode) in modes {
            check("simple deform \(n)", SimpleDeformModifier(mode, origin: Vec3(-0.5, 0, 0), axis: Vec3(1, 0.1, 0), lower: 0, upper: 0.8), base)
        }
        check("bend toward a direction", SimpleDeformModifier(.bend(angle: -0.9), origin: Vec3(0, 0, -0.5), axis: Vec3(0, 0, 1), bendDirection: Vec3(0, 1, 0.3), lower: 0.1, upper: 0.9), base)
    }

    func testLatticeMatchesCPU() {
        let base = sheet(40)
        for interp in [LatticeDeformModifier.Interpolation.trilinear, .cubic] {
            var l = LatticeDeformModifier(boxMin: Vec3(-0.5, -0.2, -0.5), boxMax: Vec3(0.5, 0.2, 0.5), divisions: SIMD3(4, 3, 5), interpolation: interp)
            l.displace(1, 1, 1, by: Vec3(0.1, 0.15, -0.05)); l.displace(2, 0, 3, by: Vec3(-0.08, 0.05, 0.12)); l.displace(3, 2, 4, by: Vec3(0, 0.2, 0))
            l.displace(0, 1, 0, by: Vec3(0.05, 0, 0.05))
            check("lattice \(interp)", l, base)
            // a box smaller than the mesh: points outside take the nearest edge's displacement
            var small = LatticeDeformModifier(boxMin: Vec3(-0.2, -0.1, -0.2), boxMax: Vec3(0.2, 0.1, 0.2), divisions: SIMD3(3, 3, 3), interpolation: interp)
            small.displace(1, 1, 1, by: Vec3(0, 0.1, 0))
            check("lattice \(interp), mesh larger than box", small, base)
        }
    }

    func testShrinkwrapMatchesCPU() {
        let base = sheet(40)
        let target = sphere(radius: 0.55, centre: Vec3(0.013, -0.3, 0.0071), seg: 48)
        check("shrinkwrap nearest", ShrinkwrapModifier(target: target, method: .nearestSurface), base, tol: 1e-4, outliers: 0.002)
        check("shrinkwrap nearest, offset + factor", ShrinkwrapModifier(target: target, method: .nearestSurface, offset: 0.03, factor: 0.7), base, tol: 1e-4, outliers: 0.002)
        check("shrinkwrap along normal (both)", ShrinkwrapModifier(target: target, method: .projectAlongNormal(.both)), base, tol: 1e-4)
        check("shrinkwrap along normal (positive)", ShrinkwrapModifier(target: target, method: .projectAlongNormal(.positive), offset: 0.01), base, tol: 1e-4)
        check("shrinkwrap along fixed direction", ShrinkwrapModifier(target: target, method: .projectAlong(Vec3(0, -1, 0))), base, tol: 1e-4)
        // target without normals (face normals)
        var bare = target; bare.normals = []
        check("shrinkwrap nearest, target without normals", ShrinkwrapModifier(target: bare, method: .nearestSurface, offset: 0.02), base, tol: 1e-4, outliers: 0.01)
    }

    func testSolidifyMatchesCPU() {
        let base = sheet(20)
        check("solidify open sheet (rim)", SolidifyModifier(thickness: 0.05), base)
        check("solidify, centred, no rim, uneven", SolidifyModifier(thickness: 0.04, offset: 0, rim: false, evenThickness: false), base)
        check("solidify smooth sphere (closed)", SolidifyModifier(thickness: 0.03, offset: 1), sphere(radius: 0.5, centre: .zero, seg: 24))
        var noNormals = base; noNormals.normals = []
        check("solidify mesh without normals", SolidifyModifier(thickness: 0.05), noNormals)
    }

    // MARK: stack

    func testStackStaysResidentAndMatchesCPU() {
        let base = sheet(40)
        let target = sphere(radius: 0.55, centre: Vec3(0, -0.3, 0), seg: 32)
        var stack = ModifierStack()
        stack.add(NoiseDisplaceModifier(strength: 0.03, frequency: 6, octaves: 3, seed: 7))
        stack.add(SimpleDeformModifier(.bend(angle: 0.8), origin: Vec3(-0.5, 0, 0), axis: Vec3(1, 0, 0), lower: 0, upper: 1))
        stack.add(LatticeDeformModifier(boxMin: Vec3(-0.6, -0.4, -0.6), boxMax: Vec3(0.6, 0.4, 0.6), divisions: SIMD3(3, 3, 3)))
        stack.add(ShrinkwrapModifier(target: target, method: .projectAlongNormal(.both), offset: 0.02, factor: 0.5))
        stack.add(SolidifyModifier(thickness: 0.02))
        stack.add(SimpleDeformModifier(.twist(angle: 0.7), axis: Vec3(0, 1, 0), lower: -0.2, upper: 0.2))     // after Solidify: new topology
        let cpu = stack.evaluate(base), r = stack.evaluateReport(base, backend: .gpu)
        print("  stack: GPU runs \(r.report.gpuRuns), CPU modifiers \(r.report.cpuModifiers)")
        XCTAssertEqual(r.report.gpuRuns.count, 1, "all six in one resident run")
        XCTAssertEqual(r.report.gpuRuns.first?.count, 6)
        XCTAssertTrue(r.report.cpuModifiers.isEmpty)
        compare(cpu, r.mesh, tol: 1e-4)
    }

    func testMixedStackRunsCPUBetweenGPURuns() {
        let base = sheet(30)
        var stack = ModifierStack()
        stack.add(NoiseDisplaceModifier(strength: 0.03, frequency: 6, seed: 3))
        stack.add(DisplaceModifier(label: "closure", strength: 0.02) { p, _ in 0.5 + 0.5 * sin(p.x * 9) })   // closure: CPU
        stack.add(SimpleDeformModifier(.taper(factor: 0.4), lower: -0.5, upper: 0.5))
        stack.add(MirrorModifier(.x))                                                                          // wrapper: CPU
        stack.add(SimpleDeformModifier(.bend(angle: 0.5), origin: Vec3(-1, 0, 0), axis: Vec3(1, 0, 0), lower: 0, upper: 2))
        stack.add(NoiseDisplaceModifier(strength: 0.01, frequency: 3, seed: 4), enabled: false)
        let cpu = stack.evaluate(base), r = stack.evaluateReport(base, backend: .gpu)
        print("  mixed stack: GPU runs \(r.report.gpuRuns), CPU \(r.report.cpuModifiers)")
        XCTAssertEqual(r.report.gpuRuns.map { $0.count }, [1, 1, 1])
        XCTAssertEqual(r.report.cpuModifiers.count, 2)
        compare(cpu, r.mesh, tol: 1e-4)
        // .cpu is the old path; .auto stays CPU for a small mesh
        XCTAssertEqual(stack.evaluate(base, backend: .cpu), cpu)
        XCTAssertEqual(stack.evaluateReport(sheet(10), backend: .auto).report.gpuModifierCount, 0, "small meshes stay on the CPU under .auto")
        XCTAssertGreaterThan(stack.evaluateReport(base, backend: .auto).report.gpuModifierCount, 0)
    }

    func testSessionReEvaluatesAfterAnEdit() {
        let base = sheet(40)
        var stack = ModifierStack()
        let id = stack.add(NoiseDisplaceModifier(strength: 0.03, frequency: 6, seed: 1))
        let session = ModifierGPUSession(base: base, backend: .gpu)
        compare(stack.evaluate(base), session.evaluate(stack), tol: 1e-4)
        stack.replace(id: id, with: NoiseDisplaceModifier(strength: 0.05, frequency: 4, seed: 2))
        compare(stack.evaluate(base), session.evaluate(stack), tol: 1e-4)
    }

    // MARK: timings

    func testTimingsOn200kVertices() {
        let base = sheet(183)               // 33,489 quads x 6 = 200,934 vertices, flat shaded
        let target = sphere(radius: 0.55, centre: Vec3(0.013, -0.3, 0.0071), seg: 70)
        print("  timings: \(base.positions.count) vertices, \(base.triangleCount) triangles; shrinkwrap target \(target.triangleCount) triangles")
        func time(_ f: () -> Mesh3) -> (Double, Mesh3) { let t = Date(); let m = f(); return (Date().timeIntervalSince(t) * 1000, m) }
        let cases: [(String, any Modifier)] = [
            ("Displace noise 4 oct", NoiseDisplaceModifier(strength: 0.03, frequency: 6, octaves: 4, seed: 7)),
            ("SimpleDeform bend", SimpleDeformModifier(.bend(angle: 1), origin: Vec3(-0.5, 0, 0), axis: Vec3(1, 0, 0), lower: 0, upper: 1)),
            ("Lattice cubic 5x5x5", LatticeDeformModifier(boxMin: Vec3(-0.6, -0.4, -0.6), boxMax: Vec3(0.6, 0.4, 0.6), divisions: SIMD3(5, 5, 5))),
            ("Shrinkwrap nearest", ShrinkwrapModifier(target: target, method: .nearestSurface)),
            ("Shrinkwrap along normal", ShrinkwrapModifier(target: target, method: .projectAlongNormal(.both))),
            ("Solidify + rim", SolidifyModifier(thickness: 0.02)),
        ]
        let session = ModifierGPUSession(base: base, backend: .gpu)
        _ = session.evaluate({ var s = ModifierStack(); s.add(cases[0].1); return s }())    // warm: pipelines, topology
        for (name, m) in cases {
            var s = ModifierStack(); s.add(m)
            let (c, cm) = time { s.evaluate(base) }
            let (cold, gm) = time { s.evaluate(base, backend: .gpu) }
            let (warm, _) = time { session.evaluate(s) }
            print(String(format: "    %-24@ CPU %8.1f ms | GPU cold (topology+upload+run+readback) %7.1f ms | GPU warm session %7.1f ms | x%.1f",
                         name as NSString, c, cold, warm, c / Swift.max(warm, 0.001)))
            XCTAssertEqual(cm.positions.count, gm.positions.count)
        }
        var stack = ModifierStack()
        for (n, m) in cases where n != "Shrinkwrap nearest" { stack.add(m) }   // nearest can fold points together: see header
        let (c, cm) = time { stack.evaluate(base) }
        let (cold, gm) = time { stack.evaluate(base, backend: .gpu) }
        let (warm, _) = time { session.evaluate(stack) }
        print(String(format: "    %-24@ CPU %8.1f ms | GPU cold %7.1f ms | GPU warm %7.1f ms | x%.1f (five modifiers, one resident run)",
                     "whole stack" as NSString, c, cold, warm, c / Swift.max(warm, 0.001)))
        // The wrapped sheet has folded, sliver faces; Solidify's even-thickness scale (1 / mean cos) amplifies the
        // Float differences in the few wrapped vertices that sit on a ray's tie, so this tolerance is looser.
        compare(cm, gm, tol: 1e-3, outliers: 0.03)
    }
}
