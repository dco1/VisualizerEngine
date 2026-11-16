import XCTest
@testable import VisualizerMaterials

final class MeshRemeshGPUTests: XCTestCase {
    override func setUpWithError() throws { try XCTSkipIf(ModelGPU.shared == nil, "no Metal device") }

    private func stats(_ m: Mesh3, _ L: Double) -> (mean: Double, inBand: Double) {
        let e = MeshKit.edgeLengths(m)
        return (e.reduce(0, +) / Double(e.count), Double(e.filter { $0 > 0.7 * L && $0 < 1.3 * L }.count) / Double(e.count))
    }

    func testBVHClosestPointMatchesCPUProjector() throws {
        let g = ModelGPU.shared!
        let sphere = MeshKit.icosphere(levels: 3)
        let topo = TriTopology(sphere)
        let proj = SurfaceProjector(topo, cell: 0.2)
        let tris: [(SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)] = (0 ..< topo.tri.count).map { f in
            func v(_ i: Int) -> SIMD4<Float> { let p = topo.pos[i]; return SIMD4(Float(p.x), Float(p.y), Float(p.z), 0) }
            return (v(topo.tri[f].x), v(topo.tri[f].y), v(topo.tri[f].z))
        }
        let bvh = try XCTUnwrap(SurfaceBVHGPU(g, tris: tris))
        var s: UInt64 = 99
        func r() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 40) / Double(1 << 24) * 4 - 2 }
        let pts = (0 ..< 2000).map { _ in Vec3(r(), r(), r()) }
        let got = try XCTUnwrap(bvh.closest(g, pts.map { SIMD4(Float($0.x), Float($0.y), Float($0.z), 0) }))
        var worst = 0.0
        for (p, q) in zip(pts, got) {
            let c = proj.closest(to: p, maxRings: 40)
            // compare distances: a point near the centre has many equally close surface points
            let d = abs(len3(c - p) - len3(Vec3(Double(q.x), Double(q.y), Double(q.z)) - p))
            worst = Swift.max(worst, d)
        }
        print("BVH closest point vs CPU projector, 2000 random points: worst distance difference \(worst)")
        XCTAssertLessThan(worst, 1e-5)
    }

    func testCoarsenSphereToTargetEdgeLength() {
        let sphere = MeshKit.icosphere(levels: 4)
        let L = 0.15
        let t0 = Date()
        let r = sphere.isotropicRemeshed(targetEdgeLength: L, iterations: 6, backend: .gpu)
        let s = stats(r, L)
        print("GPU remesh coarsen \(sphere.triangleCount) -> \(r.triangleCount) in \(Date().timeIntervalSince(t0)) s, mean \(s.mean) inBand \(s.inBand)")
        XCTAssertEqual(s.mean / L, 1, accuracy: 0.2)
        XCTAssertGreaterThan(s.inBand, 0.8)
        XCTAssertEqual(MeshKit.openEdges(r), 0)
        XCTAssertEqual(MeshKit.volume(r) / MeshKit.volume(sphere), 1, accuracy: 0.03)
        XCTAssertLessThan(r.positions.map { abs(len3($0) - 1) }.max()!, 0.02)
    }

    func testRefineSphereToTargetEdgeLength() {
        let sphere = MeshKit.icosphere(levels: 2)
        let L = 0.1
        let t0 = Date()
        let r = sphere.isotropicRemeshed(targetEdgeLength: L, iterations: 6, backend: .gpu)
        let s = stats(r, L)
        print("GPU remesh refine \(sphere.triangleCount) -> \(r.triangleCount) in \(Date().timeIntervalSince(t0)) s, mean \(s.mean) inBand \(s.inBand)")
        XCTAssertEqual(s.mean / L, 1, accuracy: 0.2)
        XCTAssertGreaterThan(s.inBand, 0.8)
        XCTAssertEqual(MeshKit.openEdges(r), 0)
        XCTAssertGreaterThan(r.positions.map { len3($0) }.min()!, 0.9)
    }

    func testRemeshKeepsOpenBoundary() {
        let plane = MeshKit.plane(n: 10)
        let r = plane.isotropicRemeshed(targetEdgeLength: 0.2, iterations: 4, backend: .gpu)
        XCTAssertEqual(MeshKit.area(r), 1, accuracy: 1e-5)
        for i in MeshKit.boundaryVertices(r) {
            let p = r.positions[i]
            XCTAssertTrue(p.x < 1e-5 || p.x > 1 - 1e-5 || p.z < 1e-5 || p.z > 1 - 1e-5, "\(p)")
        }
        let s = stats(r, 0.2); print("GPU plane mean \(s.mean) inBand \(s.inBand)")
        XCTAssertGreaterThan(s.inBand, 0.6)
    }

    func testRepeatedRunsAreAlwaysClosed() {
        let sphere = MeshKit.icosphere(levels: 3)
        for run in 0 ..< 20 {
            let r = sphere.isotropicRemeshed(targetEdgeLength: 0.12, iterations: 4, backend: .gpu)
            XCTAssertEqual(MeshKit.openEdges(r), 0, "run \(run)")
        }
    }

    func testGPUMatchesCPUAndTimes82k() {
        let sphere = MeshKit.icosphere(levels: 6)
        let L = 0.04
        _ = MeshKit.icosphere(levels: 2).isotropicRemeshed(targetEdgeLength: 0.2, iterations: 1, backend: .gpu)   // warm
        var t = Date()
        let c = sphere.isotropicRemeshed(targetEdgeLength: L, iterations: 3)
        let tc = Date().timeIntervalSince(t)
        t = Date()
        let g = sphere.isotropicRemeshed(targetEdgeLength: L, iterations: 3, backend: .gpu)
        let tg = Date().timeIntervalSince(t)
        let sc = stats(c, L), sg = stats(g, L)
        print("remesh 81920 -> L \(L), 3 iterations: CPU \(String(format: "%.3f", tc)) s (\(c.triangleCount) tris, mean \(sc.mean), inBand \(sc.inBand)); "
              + "GPU \(String(format: "%.3f", tg)) s (\(g.triangleCount) tris, mean \(sg.mean), inBand \(sg.inBand))")
        XCTAssertEqual(MeshKit.openEdges(g), 0)
        XCTAssertEqual(sg.mean / L, 1, accuracy: 0.25)
        XCTAssertGreaterThan(sg.inBand, 0.7)
        XCTAssertEqual(MeshKit.volume(g) / MeshKit.volume(sphere), 1, accuracy: 0.03)
        XCTAssertEqual(Double(g.triangleCount) / Double(c.triangleCount), 1, accuracy: 0.15)
    }
}
