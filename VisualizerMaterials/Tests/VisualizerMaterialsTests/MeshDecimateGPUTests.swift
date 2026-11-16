import XCTest
@testable import VisualizerMaterials

final class MeshDecimateGPUTests: XCTestCase {
    override func setUpWithError() throws { try XCTSkipIf(ModelGPU.shared == nil, "no Metal device") }

    private func hausdorffToSphere(_ m: Mesh3) -> Double { m.positions.map { abs(len3($0) - 1) }.max()! }

    func testSphere20kTo2kKeepsVolumeAndIsWatertight() {
        let sphere = MeshKit.icosphere(levels: 5)
        let v0 = MeshKit.volume(sphere)
        let (d, report) = sphere.decimate(targetTriangles: 2000, backend: .gpu)
        print("GPU decimate 20480 -> \(d.triangleCount) tris, maxError \(report.maxError), collapses \(report.collapses)")
        XCTAssertLessThanOrEqual(d.triangleCount, 2010)
        XCTAssertGreaterThan(d.triangleCount, 1500)
        XCTAssertEqual(MeshKit.volume(d) / v0, 1, accuracy: 0.02)
        XCTAssertEqual(MeshKit.openEdges(d), 0, "watertight and consistently wound")
        XCTAssertLessThan(hausdorffToSphere(d), 0.02)
        XCTAssertGreaterThan(report.maxError, 0)
        XCTAssertLessThan(report.maxError, 0.05)
        XCTAssertEqual(report.triangles, d.triangleCount)
    }

    /// Atomics make the vertex-face order differ run to run; every run must still be a closed, consistently wound mesh.
    func testRepeatedRunsAreAlwaysWatertight() {
        let sphere = MeshKit.icosphere(levels: 5)
        for run in 0 ..< 40 {
            let d = sphere.decimated(targetTriangles: 2000, backend: .gpu)
            XCTAssertEqual(MeshKit.openEdges(d), 0, "run \(run)")
            XCTAssertEqual(d.vertexCount, 1002, "run \(run)")
        }
    }

    func testRatioAPI() {
        let d = MeshKit.icosphere(levels: 3).decimated(ratio: 0.25, backend: .gpu)
        XCTAssertEqual(Double(d.triangleCount), 320, accuracy: 30)
        XCTAssertEqual(MeshKit.openEdges(d), 0)
    }

    func testOpenPlaneKeepsBoundaryExactly() {
        let d = MeshKit.plane(n: 16).decimated(targetTriangles: 40, backend: .gpu)
        XCTAssertLessThan(d.triangleCount, 70)
        XCTAssertEqual(MeshKit.area(d), 1, accuracy: 1e-6)
        let b = d.bounds!
        XCTAssertEqual(b.min, Vec3(0, 0, 0)); XCTAssertEqual(b.max, Vec3(1, 0, 1))
        let bv = MeshKit.boundaryVertices(d)
        XCTAssertFalse(bv.isEmpty)
        for i in bv {
            let p = d.positions[i]
            XCTAssertTrue(p.x < 1e-9 || p.x > 1 - 1e-9 || p.z < 1e-9 || p.z > 1 - 1e-9, "boundary vertex left the boundary: \(p)")
        }
        for corner in [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1), Vec3(1, 0, 1)] {
            XCTAssertTrue(d.positions.contains { len3($0 - corner) < 1e-9 }, "corner \(corner) lost")
        }
        d.normals.forEach { XCTAssertEqual($0.y, 1, accuracy: 1e-5) }
    }

    func testCreaseBoxStaysABox() {
        let box = MeshKit.box(n: 8)
        let (d, report) = box.decimate(targetTriangles: 48, backend: .gpu)
        XCTAssertLessThanOrEqual(d.triangleCount, 80)
        XCTAssertEqual(MeshKit.volume(d), 1, accuracy: 1e-6)
        XCTAssertEqual(MeshKit.area(d), 6, accuracy: 1e-5)
        XCTAssertEqual(MeshKit.openEdges(d.welded(distance: 1e-9)), 0)
        XCTAssertLessThan(report.maxError, 1e-3)
        for c in 0 ..< 8 {
            let corner = Vec3(Double(c & 1), Double((c >> 1) & 1), Double((c >> 2) & 1))
            XCTAssertTrue(d.positions.contains { len3($0 - corner) < 1e-6 }, "corner \(corner) lost")
        }
        for p in d.positions { XCTAssertTrue((0 ..< 3).contains { p[$0] < 1e-6 || p[$0] > 1 - 1e-6 }) }
    }

    func testUVSeamIsKept() {
        var m = MeshKit.plane(n: 8)
        var seam = Mesh3()
        for t in 0 ..< m.triangleCount {
            let i = (0 ..< 3).map { Int(m.indices[t * 3 + $0]) }
            let cx = (m.positions[i[0]].x + m.positions[i[1]].x + m.positions[i[2]].x) / 3
            let off = cx > 0.5 ? 0.25 : 0
            seam.addTriangle(m.positions[i[0]], m.positions[i[1]], m.positions[i[2]], outward: Vec3(0, 1, 0),
                             uv: (m.uvs[i[0]] + Vec2(off, 0), m.uvs[i[1]] + Vec2(off, 0), m.uvs[i[2]] + Vec2(off, 0)))
        }
        m = seam
        let d = m.decimated(targetTriangles: 24, backend: .gpu)
        var onSeamUV: [Double] = []
        for (i, p) in d.positions.enumerated() where abs(p.x - 0.5) < 1e-6 { onSeamUV.append(d.uvs[i].x) }
        XCTAssertTrue(onSeamUV.contains { abs($0 - 0.5) < 1e-4 } && onSeamUV.contains { abs($0 - 0.75) < 1e-4 }, "seam uv split lost: \(onSeamUV)")
        for (i, p) in d.positions.enumerated() {
            let du = d.uvs[i].x - p.x
            XCTAssertTrue(abs(du) < 1e-4 || abs(du - 0.25) < 1e-4, "uv drifted \(du) at \(p)")
            XCTAssertEqual(d.uvs[i].y, p.z, accuracy: 1e-4)
        }
        XCTAssertEqual(MeshKit.area(d), 1, accuracy: 1e-6)
    }

    func testGPUMatchesCPUQualityAndTimes82k() {
        let sphere = MeshKit.icosphere(levels: 6)
        XCTAssertEqual(sphere.triangleCount, 81920)
        let v0 = MeshKit.volume(sphere)
        _ = MeshKit.icosphere(levels: 3).decimated(ratio: 0.5, backend: .gpu)          // warm the pipelines
        var t = Date()
        let (c, rc) = sphere.decimate(targetTriangles: 8192)
        let tc = Date().timeIntervalSince(t)
        t = Date()
        let (g, rg) = sphere.decimate(targetTriangles: 8192, backend: .gpu)
        let tg = Date().timeIntervalSince(t)
        print("decimate 81920 -> 8192: CPU \(String(format: "%.3f", tc)) s (\(c.triangleCount) tris, err \(rc.maxError), hausdorff \(hausdorffToSphere(c))); "
              + "GPU \(String(format: "%.3f", tg)) s (\(g.triangleCount) tris, err \(rg.maxError), hausdorff \(hausdorffToSphere(g)))")
        XCTAssertLessThanOrEqual(g.triangleCount, 8210)
        XCTAssertGreaterThan(g.triangleCount, 7000)
        XCTAssertEqual(MeshKit.volume(g) / v0, 1, accuracy: 0.01)
        XCTAssertEqual(MeshKit.openEdges(g), 0)
        XCTAssertLessThan(hausdorffToSphere(g), 3 * Swift.max(hausdorffToSphere(c), 0.002))
    }
}
