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

    /// Prints the end-to-end union time CPU / first GPU path / GPU pipeline (median of 5, after one warm-up).
    func testBenchmarkUnion() throws {
        guard BooleanGPU2.shared != nil else { throw XCTSkip("no Metal device") }
        let sizes: [(Int, Int)] = [(130, 200), (160, 310), (230, 440)]
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
                for _ in 0..<5 { let t = Date(); last = a.boolean(.union, b, options: opt); ts.append(Date().timeIntervalSince(t)) }
                ts.sort()
                line += " \(name) \(String(format: "%.0f", ts[2] * 1000)) ms"
                meshes.append(last!.mesh)
                if name == "GPU2" { line += " steps=" + last!.stats.seconds.filter { !$0.key.hasPrefix("g.") }.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.1f", $0.value * 1000))" }.joined(separator: ",") }
            }
            print(line)
            XCTAssertEqual(meshes[2].signedVolume, meshes[0].signedVolume, accuracy: 1e-8)
            XCTAssertEqual(defects(meshes[2]).boundary, 0)
        }
    }

    func testProfileCPUStages() {
        let a = Self.sphere(1, rings: 130, sectors: 200), b = Self.sphere(0.9, rings: 130, sectors: 200, at: Vec3(0.6, 0.15, 0.1))
        var o = BooleanOptions(); o.useGPU = false
        for _ in 0..<2 {
        let job = BooleanJob(a, b, options: o)
        var clock = Date()
        func lap(_ n: String) { print("  \(n) \(String(format: "%.1f", Date().timeIntervalSince(clock) * 1000)) ms"); clock = Date() }
        job.weld(); lap("weld")
        let P = job.pool.points
        let ba = job.boxes(0, P), bb = job.boxes(1, P); lap("boxes")
        let pairs = job.findPairs(P, ba, bb); lap("broadphase \(pairs.count)")
        let eps = job.eps
        var n = 0
        for pr in pairs {
            let a = job.tris[0][Int(pr.x)], b = job.tris[1][Int(pr.y)]
            if case .none = TriTri.intersect((P[Int(a.x)], P[Int(a.y)], P[Int(a.z)]), (P[Int(b.x)], P[Int(b.y)], P[Int(b.z)]), eps: eps) {} else { n += 1 }
        }
        lap("narrow(serial) \(n)")
        }
        let job = BooleanJob(a, b, options: o)
        var clock = Date()
        func lap(_ n: String) { print("  \(n) \(String(format: "%.1f", Date().timeIntervalSince(clock) * 1000)) ms"); clock = Date() }
        job.weld(); lap("weld")
        let P0 = job.pool.points
        let ba = job.boxes(0, P0), bb = job.boxes(1, P0); lap("boxes")
        let pairs = job.findPairs(P0, ba, bb); lap("pairs")
        var segs: [(SIMD2<Int32>, PairHit)] = []
        for pr in pairs { let a = job.tris[0][Int(pr.x)], b = job.tris[1][Int(pr.y)]
            let h = TriTri.intersect((P0[Int(a.x)], P0[Int(a.y)], P0[Int(a.z)]), (P0[Int(b.x)], P0[Int(b.y)], P0[Int(b.z)]), eps: job.eps)
            if case .none = h {} else { segs.append((pr, h)) } }
        lap("narrow")
        for (pr, h) in segs { if case .segment(let p, let q) = h { let u = job.pool.add(p), v = job.pool.add(q)
            if u != v { let sg = SIMD2(Int32(u), Int32(v)); job.cons[0][pr.x, default: []].append(sg); job.cons[1][pr.y, default: []].append(sg) } } }
        lap("pool+link")
        job.findEdgeSplits(boxA: ba, boxB: bb); lap("edgeSplits")
        for s in 0..<2 { job.retriangulate(s) }; lap("retri")
        _ = job.classify(0); lap("classify0"); _ = job.classify(1); lap("classify1")
    }
}
