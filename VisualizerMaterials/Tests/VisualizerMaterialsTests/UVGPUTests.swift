import XCTest
@testable import VisualizerMaterials

final class UVGPUTests: XCTestCase {

    override func setUpWithError() throws {
        #if canImport(Metal)
        try XCTSkipIf(UVGPU.shared == nil, "no Metal device")
        #else
        throw XCTSkip("no Metal")
        #endif
    }

    // MARK: fixtures

    private func sphere(radius: Double = 1, seg: Int, rings: Int) -> Mesh3 {
        var m = Mesh3()
        m.positions.append(Vec3(0, 0, -radius))
        for r in 1..<rings { for s in 0..<seg {
            let lat = -Double.pi / 2 + Double.pi * Double(r) / Double(rings)
            let a = 2 * Double.pi * Double(s) / Double(seg)
            m.positions.append(Vec3(radius * cos(lat) * cos(a), radius * cos(lat) * sin(a), radius * sin(lat)))
        } }
        m.positions.append(Vec3(0, 0, radius))
        let north = m.positions.count - 1
        func ring(_ r: Int, _ s: Int) -> UInt32 { UInt32(1 + (r - 1) * seg + (s % seg)) }
        for s in 0..<seg { m.indices += [0, ring(1, s + 1), ring(1, s)] }
        for r in 1..<(rings - 1) { for s in 0..<seg {
            m.indices += [ring(r, s), ring(r, s + 1), ring(r + 1, s + 1), ring(r, s), ring(r + 1, s + 1), ring(r + 1, s)]
        } }
        for s in 0..<seg { m.indices += [UInt32(north), ring(rings - 1, s), ring(rings - 1, s + 1)] }
        return m
    }

    /// The meridian at s = 0 as seams (pole to pole): the sphere becomes one disc.
    private func meridianSeams(_ m: Mesh3, topo: UVTopology, seg: Int, rings: Int) -> UVSeams {
        func ring(_ r: Int) -> Int { 1 + (r - 1) * seg }
        var chain = [0] + (1..<rings).map { ring($0) } + [m.positions.count - 1]
        _ = chain.count
        var s = Set<UVEdge>()
        for i in 0..<(chain.count - 1) { s.insert(UVEdge(topo.weldedID[chain[i]], topo.weldedID[chain[i + 1]])) }
        chain.removeAll()
        return UVSeams(edges: s)
    }

    /// Open tube around Z.
    private func tube(radius: Double, height: Double, seg: Int, rings: Int) -> Mesh3 {
        var m = Mesh3()
        for r in 0...rings { for s in 0..<seg {
            let a = 2 * Double.pi * Double(s) / Double(seg)
            m.positions.append(Vec3(radius * cos(a), radius * sin(a), height * Double(r) / Double(rings)))
        } }
        for r in 0..<rings { for s in 0..<seg {
            let a = UInt32(r * seg + s), b = UInt32(r * seg + (s + 1) % seg)
            let c = UInt32((r + 1) * seg + (s + 1) % seg), d = UInt32((r + 1) * seg + s)
            m.indices += [a, b, c, a, c, d]
        } }
        return m
    }

    private func time<T>(_ f: () -> T) -> (T, Double) {
        let t = DispatchTime.now().uptimeNanoseconds
        let r = f()
        return (r, Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
    }

    // MARK: projections

    func testProjectionsMatchCPU() {
        let sph = sphere(radius: 0.7, seg: 40, rings: 20)
        let tb = tube(radius: 0.5, height: 1, seg: 48, rings: 10)
        let projections: [(String, UVProjection, Mesh3)] = [
            ("planar", UVProjection(.planar, origin: Vec3(0.1, 0, 0), axis: Vec3(0.2, 0.3, 1), scale: Vec2(2, 3), offset: Vec2(0.5, 0.25), rotationDegrees: 30), sph),
            ("box", UVProjection(.box, origin: Vec3(0, 0.1, 0), scale: Vec2(1.5, 1.5), rotationDegrees: 10), sph),
            ("cylinder", UVProjection(.cylinder, scale: Vec2(1, 2)), tb),
            ("sphere", UVProjection(.sphere, offset: Vec2(0.1, 0)), sph),
            ("sphereTilted", UVProjection(.sphere, axis: Vec3(0.3, 0.2, 1), reference: Vec3(1, 1, 0)), sph),
        ]
        for (name, p, mesh) in projections {
            let cpu = p.cornerUVs(of: mesh), gpu = p.cornerUVs(of: mesh, backend: .gpu)
            XCTAssertEqual(cpu.count, gpu.count, name)
            var worst = 0.0, bad = 0
            for (a, b) in zip(cpu, gpu) {
                // a corner a whole turn apart is a seam-rounding difference, reported separately
                let d = len(a - b)
                if d > 1e-3 { bad += 1 } else { worst = max(worst, d) }
            }
            XCTAssertEqual(bad, 0, "\(name): \(bad) of \(cpu.count) corners differ > 1e-3")
            XCTAssertLessThan(worst, 1e-4, name)
        }
    }

    func testProjectedUVsKeepsMeshShape() {
        let m = sphere(seg: 24, rings: 12)
        let a = m.projectedUVs(UVProjection(.sphere)), b = m.projectedUVs(UVProjection(.sphere), backend: .gpu)
        XCTAssertEqual(a.positions.count, b.positions.count)
        XCTAssertEqual(a.indices.count, b.indices.count)
    }

    // MARK: LSCM

    func testLSCMTubeMatchesCPU() {
        let m = tube(radius: 0.5, height: 1.2, seg: 32, rings: 12)
        let topo = UVTopology(m)
        // seam along the s = 0 line
        let seams = UVSeams(edges: Set((0..<12).map { UVEdge(topo.weldedID[$0 * 32], topo.weldedID[($0 + 1) * 32]) }))
        let isl = topo.islands(seams)
        XCTAssertEqual(isl.count, 1)
        // positions are compared before the min-rectangle alignment: a rectangle has four equal candidates and
        // which one wins is rounding noise in the CPU's Doubles
        let o = UVUnwrapOptions(alignToAxes: false)
        let cpu = UVUnwrap.lscm(m, island: isl[0], options: o)
        let gpu = UVUnwrap.lscm(m, island: isl[0], options: o, backend: .gpu)
        let aligned = UVUnwrap.lscm(m, island: isl[0], backend: .gpu)
        let cpuAligned = UVUnwrap.lscm(m, island: isl[0])
        XCTAssertEqual(aligned.size.x * aligned.size.y, cpuAligned.size.x * cpuAligned.size.y, accuracy: 2e-3 * cpuAligned.size.x * cpuAligned.size.y)
        XCTAssertEqual(cpu.uv.count, gpu.uv.count)
        let size = max(cpu.size.x, cpu.size.y)
        var worst = 0.0
        for (a, b) in zip(cpu.uv, gpu.uv) { worst = max(worst, len(a - b)) }
        print("LSCM tube: worst UV difference \(worst) of size \(size)")
        XCTAssertLessThan(worst, 1e-3 * size)
        // developable: an isometry, so UV area == 3D area and the report's anisotropy is 1
        XCTAssertEqual(gpu.uvArea(m), cpu.uvArea(m), accuracy: 2e-3)   // area3D is a Float sum on the GPU
    }

    func testLSCMSphereDiscMatchesCPU() {
        let seg = 36, rings = 18
        let m = sphere(seg: seg, rings: rings)
        let topo = UVTopology(m)
        let isl = topo.islands(meridianSeams(m, topo: topo, seg: seg, rings: rings))
        XCTAssertEqual(isl.count, 1)
        let o = UVUnwrapOptions(alignToAxes: false)
        let cpu = UVUnwrap.lscm(m, island: isl[0], options: o)
        let gpu = UVUnwrap.lscm(m, island: isl[0], options: o, backend: .gpu)
        let size = max(cpu.size.x, cpu.size.y)
        var worst = 0.0
        for (a, b) in zip(cpu.uv, gpu.uv) { worst = max(worst, len(a - b)) }
        print("LSCM sphere disc: worst UV difference \(worst) of size \(size)")
        XCTAssertLessThan(worst, 2e-3 * size)
    }

    func testUnwrappedUVsEndToEndBackendsAgreeOnReport() {
        let m = tube(radius: 0.5, height: 1.2, seg: 32, rings: 12)
        let topo = UVTopology(m)
        let seams = UVSeams(edges: Set((0..<12).map { UVEdge(topo.weldedID[$0 * 32], topo.weldedID[($0 + 1) * 32]) }))
        let cpu = m.unwrappedUVs(seams: seams), gpu = m.unwrappedUVs(seams: seams, backend: .gpu)
        XCTAssertEqual(cpu.mesh.positions.count, gpu.mesh.positions.count)
        XCTAssertEqual(cpu.report.meanAnisotropy, gpu.report.meanAnisotropy, accuracy: 1e-3)
        XCTAssertEqual(cpu.report.meanDensity, gpu.report.meanDensity, accuracy: 1e-3 * cpu.report.meanDensity)
        XCTAssertEqual(gpu.report.flipped, 0)
        XCTAssertEqual(UVPackGPU.score(gpu.layouts, backend: .gpu).overlapPairs, 0)
    }

    // MARK: report

    func testReportMatchesCPU() {
        let seg = 64, rings = 32
        let m = sphere(seg: seg, rings: rings)
        let topo = UVTopology(m)
        let r = m.unwrappedUVs(seams: meridianSeams(m, topo: topo, seg: seg, rings: rings),
                               options: UVUnwrapOptions(maxIterations: 400), pack: UVPackOptions()).mesh
        let cpu = UVReport(r), gpu = UVReport(r, backend: .gpu)
        XCTAssertEqual(cpu.islands.count, gpu.islands.count)
        func close(_ a: Double, _ b: Double, _ what: String) {
            XCTAssertEqual(a, b, accuracy: 1e-3 * max(1, abs(a)), what)
        }
        close(cpu.meanAnisotropy, gpu.meanAnisotropy, "meanAnisotropy")
        close(cpu.maxAnisotropy, gpu.maxAnisotropy, "maxAnisotropy")
        close(cpu.meanDensity, gpu.meanDensity, "meanDensity")
        close(cpu.densityVariation, gpu.densityVariation, "densityVariation")
        close(cpu.coverage, gpu.coverage, "coverage")
        XCTAssertEqual(cpu.flipped, gpu.flipped)
        for (a, b) in zip(cpu.islands, gpu.islands) {
            XCTAssertEqual(a.triangleCount, b.triangleCount)
            close(a.area3D, b.area3D, "area3D"); close(a.areaUV, b.areaUV, "areaUV")
            close(a.minDensity, b.minDensity, "minDensity"); close(a.maxDensity, b.maxDensity, "maxDensity")
            close(a.bounds.max.x, b.bounds.max.x, "bounds")
        }
        print("report: aniso \(cpu.meanAnisotropy) vs \(gpu.meanAnisotropy), density \(cpu.meanDensity) vs \(gpu.meanDensity)")
    }

    func testReportCountsFlippedLikeCPU() {
        var m = sphere(seg: 16, rings: 8).projectedUVs(UVProjection(.planar))
        m.uvs = m.uvs.map { Vec2(-$0.x, $0.y) }   // mirrored
        let cpu = UVReport(m), gpu = UVReport(m, backend: .gpu)
        XCTAssertEqual(cpu.flipped, gpu.flipped)
        XCTAssertGreaterThan(gpu.flipped, 0)
    }

    // MARK: packing

    private func scattered(_ n: Int) -> [UVIslandLayout] {
        var rng = SystemRandomNumberGenerator()
        return (0..<n).map { i in
            let a = Double.random(in: 0..<Double.pi, using: &rng), w = Double.random(in: 0.2...1), h = Double.random(in: 0.05...0.5)
            var pts: [Vec2] = []
            for k in 0..<40 { pts.append(Vec2(Double.random(in: 0...w), Double.random(in: 0...h))) }
            pts += [Vec2(0, 0), Vec2(w, 0), Vec2(w, h), Vec2(0, h)]
            let rot = pts.map { UVGeometry.rotate($0, a) + Vec2(Double(i), 3) }
            return UVIslandLayout(island: UVIsland(triangles: [], corners: [], vertexCount: rot.count), uv: rot)
        }
    }

    func testMinimumRectangleMatchesCPU() {
        let ls = scattered(60)
        for wide in [false, true] {
            let gpu = UVPackGPU.aligned(ls, wide: wide, UVGPU.shared!)!
            for (l, g) in zip(ls, gpu) {
                // the CPU oracle on the island centred first: far from the origin its Double ties are broken by rounding noise
                let lc = UVGeometry.moved(toOrigin: l)
                let c = UVGeometry.moved(toOrigin: wide ? UVGeometry.wideAlignedToMinimumRectangle(lc) : UVGeometry.alignedToMinimumRectangle(lc))
                XCTAssertEqual(c.size.x * c.size.y, g.size.x * g.size.y, accuracy: 1e-4)
                XCTAssertEqual(g.bounds.min.x, 0, accuracy: 1e-4)
                if wide { XCTAssertGreaterThanOrEqual(g.size.x + 1e-5, g.size.y) }
                // A thin rectangle's hull edges tie exactly, and the CPU breaks the tie on Double rounding noise
                // (≥ its 1e-15 slack), so the winners can differ by a quarter or half turn: compare what a rigid
                // motion keeps (distances from point 0), plus the box and the origin.
                for k in 1..<c.uv.count {
                    XCTAssertEqual(len(c.uv[k] - c.uv[0]), len(g.uv[k] - g.uv[0]), accuracy: 1e-4)
                }
                XCTAssertEqual(max(c.size.x, c.size.y), max(g.size.x, g.size.y), accuracy: 1e-4)
                XCTAssertEqual(g.bounds.min.y, 0, accuracy: 1e-4)
            }
        }
    }

    func testPackScoreAndOverlapMatchCPU() {
        var ls = scattered(300)
        // move them into a grid so some overlap and some don't
        ls = ls.enumerated().map { (i, l) in
            let o = Vec2(Double(i % 20) * 0.045, Double(i / 20) * 0.06) - l.bounds.min
            return l.mapped { $0 + o }
        }
        let cpu = UVPackScore.cpu(ls), gpu = UVPackGPU.score(ls, backend: .gpu)
        XCTAssertEqual(cpu.overlapPairs, gpu.overlapPairs)
        XCTAssertEqual(cpu.boxArea, gpu.boxArea, accuracy: 1e-3 * cpu.boxArea)
        XCTAssertEqual(cpu.insideUnitSquare, gpu.insideUnitSquare)
        XCTAssertGreaterThan(cpu.overlapPairs, 0)
        // a packed set has none
        let m = sphere(seg: 16, rings: 8)
        let packed = UVPackGPU.pack(UVPackGPU.score(ls, backend: .cpu).overlapPairs > 0 ? [] : [], mesh: m, options: UVPackOptions(), backend: .gpu)
        XCTAssertTrue(packed.isEmpty)
    }

    func testGPUPackedBoxProjectionHasNoOverlaps() {
        let m = sphere(seg: 48, rings: 24).projectedUVs(UVProjection(.box))
        let cpu = m.packedUVs(), gpu = m.packedUVs(backend: .gpu)
        XCTAssertEqual(cpu.layouts.count, gpu.layouts.count)
        XCTAssertEqual(UVPackGPU.score(gpu.layouts, backend: .gpu).overlapPairs, 0)
        XCTAssertTrue(UVPackGPU.score(gpu.layouts, backend: .gpu).insideUnitSquare)
        XCTAssertEqual(cpu.report.coverage, gpu.report.coverage, accuracy: 1e-3 * cpu.report.coverage)
    }

    // MARK: timings (200k-triangle sphere)

    func testTimings200kSphere() {
        let seg = 500, rings = 200   // ≈ 2·seg·(rings-1) = 199 k triangles
        let m = sphere(seg: seg, rings: rings)
        print("--- timings: \(m.triangleCount) triangles, \(m.positions.count) vertices ---")
        let proj = UVProjection(.sphere)
        let (cpuUV, tpc) = time { proj.cornerUVs(of: m) }
        _ = proj.cornerUVs(of: m, backend: .gpu)   // warm: compile pipelines
        let (gpuUV, tpg) = time { proj.cornerUVs(of: m, backend: .gpu) }
        var worst = 0.0; for (a, b) in zip(cpuUV, gpuUV) { worst = max(worst, len(a - b)) }
        print(String(format: "projection (sphere): CPU %.1f ms, GPU %.1f ms (incl. float4 upload + readback), worst diff %.2e", tpc, tpg, worst))
        XCTAssertLessThan(worst, 1e-4)

        let (topoMs, topo) = { () -> (Double, UVTopology) in let (t, ms) = time { UVTopology(m) }; return (ms, t) }()
        print(String(format: "UVTopology (CPU in both): %.0f ms", topoMs))
        let seamSet = meridianSeams(m, topo: topo, seg: seg, rings: rings)
        let isl = topo.islands(seamSet)
        XCTAssertEqual(isl.count, 1)

        // report on the projected mesh (with seams from the projection)
        let projected = m.withCornerUVs(cpuUV)
        let t2 = UVTopology(projected)
        let pislands = t2.islands(t2.seams(uvBoundaryOf: projected))
        let (cr, trc) = time { UVReport(projected) }
        _ = UVReportGPU.build(projected, pislands, UVGPU.shared!)
        let (gr, trg) = time { UVReportGPU.build(projected, pislands, UVGPU.shared!)! }
        print(String(format: "report: CPU %.1f ms (incl. its topology), GPU kernels only %.1f ms; meanAniso %.4f vs %.4f, density %.4f vs %.4f",
                     trc, trg, cr.meanAnisotropy, gr.meanAnisotropy, cr.meanDensity, gr.meanDensity))
        XCTAssertEqual(cr.meanDensity, gr.meanDensity, accuracy: 1e-3 * cr.meanDensity)
        XCTAssertEqual(cr.meanAnisotropy, gr.meanAnisotropy, accuracy: 1e-3 * cr.meanAnisotropy)

        // LSCM, a fixed 100 iterations both ways (the disc's CG needs thousands to converge)
        let opts = UVUnwrapOptions(alignToAxes: false, maxIterations: 400)
        _ = UVUnwrap.lscm(m, island: isl[0], options: UVUnwrapOptions(alignToAxes: false, maxIterations: 2), backend: .gpu)
        let (lc, tlc) = time { UVUnwrap.lscm(m, island: isl[0], options: opts) }
        let (lg, tlg) = time { UVUnwrap.lscm(m, island: isl[0], options: opts, backend: .gpu) }
        var lw = 0.0; for (a, b) in zip(lc.uv, lg.uv) { lw = max(lw, len(a - b)) }
        print(String(format: "LSCM 400 CG iterations, 100k vertices: CPU %.0f ms, GPU %.0f ms (%.2f ms/iter); worst UV diff %.2e of size %.2f (float vs double CG drift apart on this ill-conditioned disc)",
                     tlc, tlg, tlg / 400, lw, max(lc.size.x, lc.size.y)))

        // packing: the 200k sphere's single island rotation search
        let (_, tac) = time { UVGeometry.wideAlignedToMinimumRectangle(lc) }
        _ = UVPackGPU.aligned([lg], wide: true, UVGPU.shared!)
        let (_, tag) = time { UVPackGPU.aligned([lg], wide: true, UVGPU.shared!) }
        print(String(format: "min-rectangle alignment of one 100k-point island: CPU %.1f ms, GPU %.1f ms", tac, tag))
    }

    func testTimingsConvergedLSCM20kTriangles() {
        let seg = 160, rings = 64
        let m = sphere(seg: seg, rings: rings)
        let topo = UVTopology(m)
        let isl = topo.islands(meridianSeams(m, topo: topo, seg: seg, rings: rings))
        // a hemisphere-sized patch is better conditioned: keep only the part below the equator
        var keep = [Int](); for t in isl[0].triangles where (0..<3).allSatisfy({ m.positions[Int(m.indices[3 * t + $0])].z < 0.2 }) { keep.append(t) }
        var sub = Mesh3(); sub.positions = m.positions
        for t in keep { sub.indices += [m.indices[3 * t], m.indices[3 * t + 1], m.indices[3 * t + 2]] }
        let st = UVTopology(sub)
        let si = st.islands(UVSeams())
        let big = si.max { $0.triangles.count < $1.triangles.count }!
        print("converged LSCM patch: \(big.triangles.count) triangles")
        let o = UVUnwrapOptions(alignToAxes: false)
        let (c, tc) = time { UVUnwrap.lscm(sub, island: big, options: o) }
        let (g, tg) = time { UVUnwrap.lscm(sub, island: big, options: o, backend: .gpu) }
        var w = 0.0; for (a, b) in zip(c.uv, g.uv) { w = max(w, len(a - b)) }
        print(String(format: "LSCM to convergence: CPU %.0f ms, GPU %.0f ms, worst UV diff %.2e of size %.2f", tc, tg, w, max(c.size.x, c.size.y)))
        XCTAssertLessThan(w, 2e-3 * max(c.size.x, c.size.y))
    }
}
