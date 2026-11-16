import XCTest
import simd
@testable import VisualizerMaterials

/// M6b: the GPU-heavy boolean pipeline (`Mesh3+BooleanGPU2*.swift`), judged against the CPU path.
final class MeshBooleanGPU2Tests: XCTestCase {
    static func sphere(_ r: Double, rings: Int, sectors: Int, at c: Vec3 = .zero) -> Mesh3 {
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
        if m.signedVolume < 0 {
            var t = 0
            while t + 2 < m.indices.count { m.indices.swapAt(t + 1, t + 2); t += 3 }
            m.normals = m.normals.map { -$0 }
        }
        return m
    }

    private func defects(_ m: Mesh3) -> (boundary: Int, nonManifold: Int) {
        var e: [UInt64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0..<3 {
                let a = m.indices[t + k], b = m.indices[t + (k + 1) % 3]
                e[UInt64(min(a, b)) << 32 | UInt64(max(a, b)), default: 0] += 1
            }
            t += 3
        }
        return (e.values.filter { $0 == 1 }.count, e.values.filter { $0 > 2 }.count)
    }

    func testGPU2MatchesCPUOnFiftyThousand() throws {
        guard BooleanGPU2.shared != nil else { throw XCTSkip("no Metal device") }
        let a = Self.sphere(1, rings: 130, sectors: 200), b = Self.sphere(0.9, rings: 130, sectors: 200, at: Vec3(0.6, 0.15, 0.1))
        var cpu = BooleanOptions(); cpu.useGPU = false
        for op in [BooleanOp.union, .difference, .intersection] {
            let g = a.boolean(op, b), c = a.boolean(op, b, options: cpu)
            print("[gpu2 \(op)] vol gpu=\(g.mesh.signedVolume) cpu=\(c.mesh.signedVolume) tris \(g.mesh.triangleCount)/\(c.mesh.triangleCount) pairs=\(g.stats.candidatePairs) steps=\(g.stats.seconds.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.1f", $0.value * ($0.key.hasPrefix("g.") && $0.value > 5 ? 1 : 1000)))" })")
            XCTAssertEqual(g.mesh.signedVolume, c.mesh.signedVolume, accuracy: 1e-9)
            XCTAssertEqual(g.mesh.triangleCount, c.mesh.triangleCount)
            XCTAssertEqual(defects(g.mesh).boundary, 0)
            XCTAssertEqual(defects(g.mesh).nonManifold, 0)
            XCTAssertEqual(g.stats.constraintFailures, 0)
            XCTAssertTrue(g.stats.windingOnGPU)
        }
    }

    /// Every verdict the GPU narrow phase gives is checked against the exact CPU one: a definite miss must be
    /// `.none`, a definite hit a segment with the same endpoints (as a set).
    func testNarrowPhaseVerdictsAreSound() throws {
        guard let g = BooleanGPU2.shared else { throw XCTSkip("no Metal device") }
        let cases: [(Mesh3, Mesh3)] = [
            (Self.sphere(1, rings: 130, sectors: 200), Self.sphere(0.9, rings: 130, sectors: 200, at: Vec3(0.6, 0.15, 0.1))),
            (Self.sphere(1, rings: 40, sectors: 56), Self.sphere(1, rings: 40, sectors: 56, at: Vec3(0, 0.5, 0))),      // coincident vertices
            (Self.sphere(1, rings: 30, sectors: 40), Self.sphere(0.5, rings: 30, sectors: 40, at: Vec3(0.5, 0, 0))),    // internally tangent
        ]
        for (a, b) in cases {
            let job = BooleanJob(a, b, options: BooleanOptions())
            let run = GPU2Run(job: job, g: g)
            run.keepNarrow = true
            job.weld()
            XCTAssertTrue(run.cutAll())
            let pairs = g.read(try XCTUnwrap(run.lastPairs), SIMD2<UInt32>.self, job.stats.candidatePairs)
            let code = g.read(try XCTUnwrap(run.lastCode), SIMD4<UInt32>.self, pairs.count)
            let P = job.pool.points
            var n = [0, 0, 0], wrongMiss = 0, wrongHit = 0, badPoint = 0
            for (k, pr) in pairs.enumerated() {
                let ta = job.tris[0][Int(pr.x)], tb = job.tris[1][Int(pr.y)]
                let T = (P[Int(ta.x)], P[Int(ta.y)], P[Int(ta.z)]), S = (P[Int(tb.x)], P[Int(tb.y)], P[Int(tb.z)])
                let st = Int(code[k].x)
                n[st] += 1
                let cpu = TriTri.intersect(T, S, eps: job.eps)
                switch (st, cpu) {
                case (0, .none): break
                case (0, _): wrongMiss += 1
                case (1, .segment(let p, let q)):
                    let u = GPU2Run.endpoint(T, S, code[k].y), v = GPU2Run.endpoint(T, S, code[k].z)
                    let d1 = max(len3(u - p), len3(v - q)), d2 = max(len3(u - q), len3(v - p))
                    if min(d1, d2) > 1e-12 { badPoint += 1 }
                case (1, _): wrongHit += 1
                default: break
                }
            }
            print("[narrow soundness] pairs \(pairs.count): miss \(n[0]) hit \(n[1]) uncertain \(n[2]); wrongMiss \(wrongMiss) wrongHit \(wrongHit) badPoint \(badPoint)")
            XCTAssertEqual(wrongMiss, 0); XCTAssertEqual(wrongHit, 0); XCTAssertEqual(badPoint, 0)
        }
    }

    /// The pipeline forced onto small and awkward inputs (nested, disjoint, touching, coincident): agrees with the
    /// CPU path and never falls back.
    func testForcedPipelineOnSmallCases() throws {
        guard BooleanGPU2.shared != nil else { throw XCTSkip("no Metal device") }
        let saved = BooleanGPU2.minTriangles
        BooleanGPU2.minTriangles = 0
        defer { BooleanGPU2.minTriangles = saved }
        let r0 = BooleanGPU2.runs, f0 = BooleanGPU2.fallbacks
        let a = Self.sphere(1, rings: 24, sectors: 32)
        let others: [Mesh3] = [Self.sphere(0.5, rings: 20, sectors: 28), Self.sphere(0.5, rings: 20, sectors: 28, at: Vec3(5, 0, 0)),
                               Self.sphere(1, rings: 24, sectors: 32), Self.sphere(0.8, rings: 24, sectors: 32, at: Vec3(0.5, 0.2, 0.1)),
                               Self.sphere(1, rings: 24, sectors: 32, at: Vec3(2, 0, 0)), Self.sphere(1, rings: 30, sectors: 40, at: Vec3(0, 0.3, 0))]
        var cpu = BooleanOptions(); cpu.useGPU = false
        for (i, b) in others.enumerated() { for op in [BooleanOp.union, .difference, .intersection] {
            let g = a.boolean(op, b), c = a.boolean(op, b, options: cpu)
            XCTAssertEqual(g.mesh.signedVolume, c.mesh.signedVolume, accuracy: 1e-8, "case \(i) \(op)")
            if i != 4 { XCTAssertEqual(defects(g.mesh).boundary, 0, "case \(i) \(op)") }
        } }
        print("[forced pipeline] runs \(BooleanGPU2.runs - r0) fallbacks \(BooleanGPU2.fallbacks - f0)")
        XCTAssertEqual(BooleanGPU2.fallbacks - f0, 0)
    }

    /// Triangle soup (three private vertices per triangle: every vertex has six coincident twins to weld) and a slice
    /// by a plane (boxes far bigger than the grid's cells), both forced onto the pipeline.
    func testSoupAndSliceOnPipeline() throws {
        guard BooleanGPU2.shared != nil else { throw XCTSkip("no Metal device") }
        let saved = BooleanGPU2.minTriangles
        BooleanGPU2.minTriangles = 0
        defer { BooleanGPU2.minTriangles = saved }
        func soup(_ m: Mesh3) -> Mesh3 {
            var o = Mesh3()
            for i in m.indices { o.positions.append(m.positions[Int(i)]); o.normals.append(m.normals[Int(i)]); o.uvs.append(m.uvs[Int(i)]); o.indices.append(UInt32(o.indices.count)) }
            return o
        }
        let a = Self.sphere(1, rings: 30, sectors: 40), b = Self.sphere(0.8, rings: 30, sectors: 40, at: Vec3(0.5, 0.2, 0.1))
        var cpu = BooleanOptions(); cpu.useGPU = false
        let g = soup(a).boolean(.union, soup(b)), c = a.boolean(.union, b, options: cpu)
        XCTAssertEqual(g.mesh.signedVolume, c.mesh.signedVolume, accuracy: 1e-8)
        XCTAssertEqual(defects(g.mesh).boundary, 0)
        XCTAssertGreaterThan(g.stats.seconds["g.weldPairs"] ?? 0, 1000)
        let big = Self.sphere(1, rings: 60, sectors: 80)
        let before = BooleanGPU2.fallbacks
        let (up, down) = big.slice(point: Vec3(0, 0.3, 0), normal: Vec3(0.2, 1, 0.1))
        XCTAssertEqual(BooleanGPU2.fallbacks, before)
        XCTAssertEqual(up.signedVolume + down.signedVolume, big.signedVolume, accuracy: 1e-8)
        XCTAssertEqual(defects(up).boundary + defects(down).boundary, 0)
    }

    /// Prints the end-to-end union time CPU / first GPU path / GPU pipeline (best of 9, after one warm-up).
    func testBenchmarkUnion() throws {
        guard BooleanGPU2.shared != nil else { throw XCTSkip("no Metal device") }
        print("[shader compile (cold start)] \(String(format: "%.0f", BooleanGPU2.shared!.compileSeconds * 1000)) ms")
        let sizes: [(Int, Int)] = [(24, 32), (40, 56), (70, 100), (100, 140), (130, 200), (160, 310), (230, 440)]
        for (rings, sectors) in sizes {
            let a = Self.sphere(1, rings: rings, sectors: sectors), b = Self.sphere(0.9, rings: rings, sectors: sectors, at: Vec3(0.6, 0.15, 0.1))
            var cpu = BooleanOptions(); cpu.useGPU = false
            var gpu1 = BooleanOptions(); gpu1.useGPUPipeline = false
            var line = "[bench union 2x\(a.triangleCount) tris]"
            var meshes: [Mesh3] = []
            for (name, opt) in [("CPU", cpu), ("GPU1", gpu1), ("GPU2", BooleanOptions())] {
                _ = a.boolean(.union, b, options: opt)
                var ts: [Double] = []
                var last: BooleanResult?
                for _ in 0..<9 { let t = Date(); let r = a.boolean(.union, b, options: opt); let dt = Date().timeIntervalSince(t); ts.append(dt); if last == nil || dt <= ts.min()! { last = r } }
                ts.sort()
                line += " \(name) \(String(format: "%.0f", ts[0] * 1000)) ms"
                meshes.append(last!.mesh)
                if name == "GPU2" { line += " steps=" + last!.stats.seconds.filter { !$0.key.hasPrefix("g.why") && !$0.key.hasPrefix("g.cc") }.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.1f", $0.value * ($0.key.hasPrefix("g.") && $0.value > 5 ? 1 : 1000)))" }.joined(separator: ",") }
            }
            print(line)
            XCTAssertEqual(meshes[2].signedVolume, meshes[0].signedVolume, accuracy: 1e-8)
            XCTAssertEqual(defects(meshes[2]).boundary, 0)
        }
    }
}
