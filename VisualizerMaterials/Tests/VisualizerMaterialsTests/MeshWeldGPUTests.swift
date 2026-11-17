import XCTest
@testable import VisualizerMaterials

final class MeshWeldGPUTests: XCTestCase {
    /// Every triangle owns its three vertices.
    static func soup(_ m: Mesh3) -> Mesh3 {
        var o = Mesh3()
        for i in m.indices {
            o.positions.append(m.positions[Int(i)]); o.normals.append(m.normals[Int(i)]); o.uvs.append(m.uvs[Int(i)])
            o.indices.append(UInt32(o.positions.count - 1))
        }
        return o
    }

    private func jitteredBox() -> Mesh3 {
        var m = MeshKit.box(n: 1)
        var s: UInt64 = 7
        func r() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 40) / Double(1 << 24) - 0.5 }
        m.positions = m.positions.map { $0 + Vec3(r(), r(), r()) * 1e-5 }
        return m
    }

    func testGPUWeldClosesSoupCubeLikeCPU() throws {
        try XCTSkipIf(ModelGPU.shared == nil, "no Metal device")
        let soup = jitteredBox()
        let w = soup.welded(distance: 1e-4, backend: .gpu)
        XCTAssertEqual(w.vertexCount, 8); XCTAssertEqual(w.triangleCount, 12)
        XCTAssertEqual(MeshKit.openEdges(w), 0)
        XCTAssertEqual(MeshKit.volume(w), 1, accuracy: 1e-3)
        w.normals.forEach { XCTAssertEqual(len3($0), 1, accuracy: 1e-5) }
        let hard = soup.welded(distance: 1e-4, normalAngleDegrees: 30, backend: .gpu)
        XCTAssertEqual(hard.vertexCount, 24)
        XCTAssertEqual(hard.vertexCount, soup.welded(distance: 1e-4, normalAngleDegrees: 30).vertexCount)
    }

    func testGPUWeldKeepsUVSeamsAndDropsDegenerates() throws {
        try XCTSkipIf(ModelGPU.shared == nil, "no Metal device")
        var s = Mesh3()
        s.addTriangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1), outward: Vec3(0, 1, 0), uv: (Vec2(0, 0), Vec2(1, 0), Vec2(0, 1)))
        s.addTriangle(Vec3(1, 0, 0), Vec3(1, 0, 1), Vec3(0, 0, 1), outward: Vec3(0, 1, 0), uv: (Vec2(0.5, 0), Vec2(1, 1), Vec2(0.5, 1)))
        XCTAssertEqual(s.welded(distance: 1e-6, backend: .gpu).vertexCount, 6)
        XCTAssertEqual(s.welded(distance: 1e-6, uvTolerance: 1, backend: .gpu).vertexCount, 4)
        var m = Mesh3()
        m.positions = [Vec3(0, 0, 0), Vec3(1e-9, 0, 0), Vec3(0, 0, 1), Vec3(1, 0, 0)]
        m.indices = [0, 1, 2, 0, 3, 2]
        let w = m.welded(distance: 1e-6, backend: .gpu)
        XCTAssertEqual(w.vertexCount, 3); XCTAssertEqual(w.triangleCount, 1)
    }

    func testGPUWeld82kSphereSoupMatchesCPUAndTimes() throws {
        try XCTSkipIf(ModelGPU.shared == nil, "no Metal device")
        let sphere = MeshKit.icosphere(levels: 6)
        let soup = MeshWeldGPUTests.soup(sphere)
        XCTAssertEqual(soup.triangleCount, 81920)
        _ = soup.welded(distance: 1e-5, backend: .gpu)    // warm: shader pipelines
        var t = Date()
        let cpu = soup.welded(distance: 1e-5)
        let tc = Date().timeIntervalSince(t)
        t = Date()
        let gpu = soup.welded(distance: 1e-5, backend: .gpu)
        let tg = Date().timeIntervalSince(t)
        print("weld 81920-tri soup (245760 verts): CPU \(String(format: "%.3f", tc)) s, GPU \(String(format: "%.3f", tg)) s -> \(cpu.vertexCount)/\(gpu.vertexCount) verts")
        XCTAssertEqual(gpu.vertexCount, cpu.vertexCount)
        XCTAssertEqual(gpu.vertexCount, sphere.vertexCount)
        XCTAssertEqual(gpu.triangleCount, cpu.triangleCount)
        XCTAssertEqual(MeshKit.openEdges(gpu), 0)
        XCTAssertEqual(MeshKit.volume(gpu), MeshKit.volume(cpu), accuracy: 1e-9)
        gpu.normals.forEach { XCTAssertEqual(len3($0), 1, accuracy: 1e-5) }
    }
}
