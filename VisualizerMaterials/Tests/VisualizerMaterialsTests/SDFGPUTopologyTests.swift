import XCTest
import simd
@testable import VisualizerMaterials

/// Item M4: the surface-nets topology stage on the GPU, held to the CPU kernels (SDF.mesh with the GPU grid but the CPU
/// topology is the oracle when no saddle is involved; closure and volume otherwise).
final class SDFGPUTopologyTests: XCTestCase {

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            v += dot3(m.positions[Int(m.indices[t])], cross3(m.positions[Int(m.indices[t + 1])], m.positions[Int(m.indices[t + 2])])) / 6; t += 3
        }
        return v
    }
    private func unmatched(_ m: Mesh3) -> Int {
        var c: [UInt64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for e in 0..<3 { c[UInt64(m.indices[t + e]) << 32 | UInt64(m.indices[t + (e + 1) % 3])] = (c[UInt64(m.indices[t + e]) << 32 | UInt64(m.indices[t + (e + 1) % 3])] ?? 0) + 1 }
            t += 3
        }
        return c.filter { $0.value != 1 || c[($0.key & 0xffff_ffff) << 32 | ($0.key >> 32)] != 1 }.count
    }
    private func model() -> SDF {
        SDF.roundedBox(size: Vec3(1, 0.7, 0.8), radius: 0.12)
            .union(SDF.capsule(from: Vec3(-0.2, 0, 0), to: Vec3(0.3, 0.6, 0.1), radius: 0.18), smooth: 0.12)
            .subtracting(SDF.cone(height: 0.6, bottomRadius: 0.25, topRadius: 0.05).rotated(angle: 0.6, axis: Vec3(1, 0, 1)).translated(by: Vec3(0.3, 0, 0.3)))
            .union(SDF.torus(major: 0.3, minor: 0.07).scaled(by: 1.2).translated(by: Vec3(0, 0.5, 0)))
            .carved(tool: SDF.sphere(radius: 0.15).translated(by: Vec3(-0.35, 0.35, 0)), depth: 0.05, falloff: 0.04)
            .intersecting(SDF.plane(normal: Vec3(0.2, 1, 0.1), offset: 0.9))
            .shell(thickness: 0.08)
    }

    func testSDFGPUTopologyMatchesCPUOnSaddleFreeModels() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        let cases: [(String, SDF, Vec3, Vec3, Double)] = [
            ("sphere", .sphere(radius: 1), Vec3(repeating: -1.2), Vec3(repeating: 1.2), 0.04),
            ("box", .box(size: Vec3(1, 2, 3)), Vec3(-1, -1.5, -2), Vec3(1, 1.5, 2), 0.07),
            ("rounded box", .roundedBox(size: Vec3(1, 1, 1), radius: 0.2), Vec3(repeating: -0.8), Vec3(repeating: 0.8), 0.03),
            ("torus", .torus(major: 0.6, minor: 0.2), Vec3(-1, -0.5, -1), Vec3(1, 0.5, 1), 0.04),
        ]
        for (name, s, lo, hi, cell) in cases {
            let oracle = s.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .gpu))     // GPU grid, CPU topology
            let g = try XCTUnwrap(s.meshGPU(lo: lo, hi: hi, cell: cell), name)
            XCTAssertEqual(g.indices.count, oracle.indices.count, "\(name): triangle count")
            // Same quads and same winding; a symmetric shape's two diagonals tie to within Float noise, so allow those to flip.
            let same = zip(g.indices, oracle.indices).filter { $0 == $1 }.count
            XCTAssertGreaterThan(Double(same) / Double(max(oracle.indices.count, 1)), 0.98, "\(name): triangles identical")
            XCTAssertEqual(g.positions.count, oracle.positions.count, name)
            var worst = 0.0
            for (a, b) in zip(g.positions, oracle.positions) { worst = max(worst, len3(a - b)) }
            XCTAssertLessThan(worst, cell * 1e-3, "\(name): positions")
            XCTAssertEqual(unmatched(g), 0, name)
            XCTAssertGreaterThan(volume(g), 0, name)
        }
    }

    func testSDFGPUTopologyClosedAndSameVolumeIncludingSaddles() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        // Two spheres whose surfaces graze diagonally across cells, thin shells and a coarse grid: saddles galore.
        let kissing = SDF.sphere(radius: 0.5).translated(by: Vec3(-0.5, 0, 0))
            .union(SDF.sphere(radius: 0.5).translated(by: Vec3(0.5, 0.01, 0.0)), smooth: 0)
        let thin = SDF.sphere(radius: 0.7).shell(thickness: 0.12)
        let pairs: [(String, SDF, Vec3, Vec3, Double)] = [
            ("model", model(), Vec3(-0.9137, -0.6071, -0.8093), Vec3(0.9, 0.6, 0.8), 0.03),
            ("kissing", kissing, Vec3(-1.31, -0.71, -0.71), Vec3(1.31, 0.71, 0.71), 0.05),
            ("thin shell", thin, Vec3(repeating: -0.93), Vec3(repeating: 0.93), 0.04),
        ]
        for (name, s, lo, hi, cell) in pairs {
            let cpu = s.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .cpu))
            // How many samples does the CPU's sequential disambiguation flip? (proves the saddle rounds are exercised)
            let cells = SIMD3<Int>(Int(((hi.x - lo.x) / cell).rounded(.up)), Int(((hi.y - lo.y) / cell).rounded(.up)), Int(((hi.z - lo.z) / cell).rounded(.up)))
            var grid = s.cpuGrid(lo: lo, cell: cell, cells: cells), before = grid
            let n = SIMD3(cells.x + 1, cells.y + 1, cells.z + 1)
            for k in 0..<n.z { for j in 0..<n.y { for i in 0..<n.x where i == 0 || j == 0 || k == 0 || i == n.x - 1 || j == n.y - 1 || k == n.z - 1 {
                let x = (k * n.y + j) * n.x + i; if grid[x] <= 0 { grid[x] = Float(cell * 1e-3) } } } }
            before = grid
            SDF.disambiguate(&grid, cells: cells, eps: Float(cell * 1e-3))
            print("saddle samples flipped by the CPU in \(name):", zip(grid, before).filter { $0 != $1 }.count)
            let g = try XCTUnwrap(s.meshGPU(lo: lo, hi: hi, cell: cell), name)
            XCTAssertEqual(unmatched(g), 0, "\(name): GPU mesh not closed / not manifold")
            XCTAssertEqual(unmatched(cpu), 0, "\(name): cpu")
            XCTAssertGreaterThan(volume(g), 0, name)
            XCTAssertEqual(volume(g), volume(cpu), accuracy: abs(volume(cpu)) * 0.01 + 1e-4, "\(name): volume")
            XCTAssertEqual(Double(g.positions.count), Double(cpu.positions.count), accuracy: Double(cpu.positions.count) * 0.01 + 4, name)
            let he = HalfEdgeMesh(g)
            XCTAssertTrue(he.isValid, name); XCTAssertTrue(he.isClosed, "\(name): half-edge closed")
        }
    }

    func testSDFGPUUVsAndRefusal() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        let s = SDF.sphere(radius: 0.5)
        let o = SDFMeshOptions(backend: .gpu, uvs: .boxProjected(metersPerUnit: 0.25))
        let a = try XCTUnwrap(s.meshGPU(lo: Vec3(repeating: -0.7), hi: Vec3(repeating: 0.7), cell: 0.05, options: o))
        let b = s.mesh(lo: Vec3(repeating: -0.7), hi: Vec3(repeating: 0.7), cell: 0.05, options: o)
        XCTAssertEqual(a.uvs.count, b.uvs.count)
        // box-projected UVs flip axis where |nx| ≈ |ny| (Float vs Double): all but those few agree
        let agree = zip(a.uvs, b.uvs).filter { abs($0.x - $1.x) < 1e-2 && abs($0.y - $1.y) < 1e-2 }.count
        XCTAssertGreaterThan(Double(agree) / Double(a.uvs.count), 0.97)
        XCTAssertNil(SDF.custom { len3($0) - 1 }.meshGPU(lo: Vec3(repeating: -1.2), hi: Vec3(repeating: 1.2), cell: 0.1))
    }

    func testSDFGPUTopologyTimings() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        let m = model()
        let lo = Vec3(-0.9, -0.6, -0.8), hi = Vec3(0.9, 0.6, 0.8), cell = 0.012
        var t = Date()
        let cpu = m.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .cpu)); let cpuMS = Date().timeIntervalSince(t) * 1000
        _ = m.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .gpu))
        t = Date(); let hyb = m.mesh(lo: lo, hi: hi, cell: cell, options: .init(backend: .gpu)); let hybMS = Date().timeIntervalSince(t) * 1000
        _ = m.meshGPU(lo: lo, hi: hi, cell: cell)
        var tm = SDFNetTimings()
        let cells = SIMD3<Int>(Int(((hi.x - lo.x) / cell).rounded(.up)), Int(((hi.y - lo.y) / cell).rounded(.up)), Int(((hi.z - lo.z) / cell).rounded(.up)))
        t = Date(); let g = SDFNetGPU.mesh(m, lo: lo, cell: cell, cells: cells, options: .init(), timings: &tm); let gMS = Date().timeIntervalSince(t) * 1000
        let gm = try XCTUnwrap(g)
        print(String(format: "SDF M4 timing %dx%dx%d cells: CPU %.0f ms (%d tris) | GPU grid+CPU topology %.0f ms | all-GPU %.0f ms (%d tris; grid+topology %.0f, emit+refine %.0f, readback→Mesh3 %.0f)",
                     cells.x, cells.y, cells.z, cpuMS, cpu.triangleCount, hybMS, gMS, gm.triangleCount, tm.gridAndTopologyMS, tm.emitAndRefineMS, tm.readbackMS))
        XCTAssertEqual(unmatched(gm), 0)
        XCTAssertEqual(volume(gm), volume(hyb), accuracy: abs(volume(hyb)) * 0.01)
    }
}

final class SDFGPUScanTests: XCTestCase {
    func testSDFGPUPrefixScanIsExact() throws {
        guard let g = ModellingGPU.shared else { throw XCTSkip("no Metal device") }
        for n in [1, 5, 1023, 1024, 1025, 70_000, 1_100_000] {
            let flags = (0..<n).map { UInt32(($0 &* 2654435761) >> 7 & 1) }
            let a = g.buffer(flags), b = g.buffer(length: n * 4)
            XCTAssertTrue(g.run { g.scan($0, input: a, output: b, n: n) })
            let out = g.read(b, UInt32.self, count: n)
            var run: UInt32 = 0, bad = -1
            for i in 0..<n { if out[i] != run && bad < 0 { bad = i }; run += flags[i] }
            XCTAssertEqual(bad, -1, "scan wrong first at \(bad) of \(n)")
        }
    }
}
