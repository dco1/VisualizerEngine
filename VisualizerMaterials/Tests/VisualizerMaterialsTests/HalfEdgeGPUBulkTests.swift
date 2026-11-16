import XCTest
import simd
@testable import VisualizerMaterials

/// Item M4: half-edge bulk kernels (normals, areas, volume), the directed-edge table and the Mesh3 weld on the GPU.
final class HalfEdgeGPUBulkTests: XCTestCase {

    private func sphere(lon: Int, lat: Int, r: Double = 1.5) -> Mesh3 {
        func p(_ i: Int, _ j: Int) -> Vec3 {
            let th = Double.pi * Double(j) / Double(lat), ph = 2 * Double.pi * Double(i % lon) / Double(lon)
            return Vec3(r * sin(th) * cos(ph), r * cos(th), r * sin(th) * sin(ph))
        }
        var m = Mesh3()
        for j in 0..<lat { for i in 0..<lon {
            let a = p(i, j), b = p(i + 1, j), c = p(i + 1, j + 1), d = p(i, j + 1), out = (a + b + c + d) / 4
            if j == 0 { m.addTriangle(a, c, d, outward: out) }
            else if j == lat - 1 { m.addTriangle(a, b, c, outward: out) }
            else { m.addTriangle(a, b, c, outward: out); m.addTriangle(a, c, d, outward: out) }
        } }
        return m
    }

    func testHalfEdgeBulkKernelsMatchCPU() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        var he = HalfEdgeMesh(sphere(lon: 48, lat: 24))
        he.mergeCoplanarFaces()                                       // some n-gons
        he.deleteFace(he.faces[7])                                    // a hole and a dead slot
        let cpu = he.cpuBulk()
        let gpu = try XCTUnwrap(HalfEdgeBulkGPU.run(he.soa()))
        XCTAssertEqual(cpu.faceNormals.count, gpu.faceNormals.count)
        var wn = 0.0, wa = 0.0, wv = 0.0
        for i in 0..<cpu.faceNormals.count {
            wn = max(wn, len3(cpu.faceNormals[i] - gpu.faceNormals[i])); wa = max(wa, abs(cpu.faceAreas[i] - gpu.faceAreas[i]) / max(cpu.faceAreas[i], 1e-9))
        }
        for i in 0..<cpu.vertexNormals.count { wv = max(wv, len3(cpu.vertexNormals[i] - gpu.vertexNormals[i])) }
        XCTAssertLessThan(wn, 1e-4); XCTAssertLessThan(wa, 1e-4); XCTAssertLessThan(wv, 1e-4)
        XCTAssertEqual(gpu.signedVolume, cpu.signedVolume, accuracy: abs(cpu.signedVolume) * 1e-4)
        XCTAssertGreaterThan(cpu.signedVolume, 0)
        // A closed mesh far from the origin: the volume must not drift (centred sums).
        var far = sphere(lon: 32, lat: 16); far.positions = far.positions.map { $0 + Vec3(300, -200, 150) }
        let hf = HalfEdgeMesh(far)
        XCTAssertEqual(hf.gpuBulk().signedVolume, hf.cpuBulk().signedVolume, accuracy: abs(hf.cpuBulk().signedVolume) * 1e-3)
    }

    func testHalfEdgeGPUEdgeTableMatchesCPU() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        var he = HalfEdgeMesh(sphere(lon: 40, lat: 20))
        he.deleteFace(he.faces[11]); he.deleteFace(he.faces[100])
        let s = he.soa()
        let t = try XCTUnwrap(HalfEdgeMesh.gpuEdgeTable(origin: s.origin, next: s.next, face: s.face))
        XCTAssertEqual(t.duplicates, 0)
        XCTAssertEqual(t.twin, he.soa().twin)
        // CPU edge ids keep tombstones after deletions; the GPU's are contiguous: same pairing, edgeCount edges.
        XCTAssertEqual(t.edgeHalfedge.count, he.edgeCount)
        for h in 0..<he.halfedgeSlots where he.face(of: h) >= 0 {
            XCTAssertEqual(t.edge[h], t.edge[Int(t.twin[h] >= 0 ? t.twin[h] : Int32(h))])
            XCTAssertEqual(t.edgeHalfedge[Int(t.edge[h])] == Int32(h) || t.twin[h] == t.edgeHalfedge[Int(t.edge[h])], true)
        }
        let fresh = HalfEdgeMesh(sphere(lon: 40, lat: 20)), fs = fresh.soa()
        let ft = try XCTUnwrap(HalfEdgeMesh.gpuEdgeTable(origin: fs.origin, next: fs.next, face: fs.face))
        XCTAssertEqual(ft.edge, fresh.halfedgeSlots == 0 ? [] : (0..<fresh.halfedgeSlots).map { Int32(fresh.edge(of: $0)) })
        XCTAssertEqual(ft.edgeHalfedge, (0..<fresh.edgeSlots).map { Int32(fresh.halfedge(ofEdge: $0)) })
        // a duplicate directed edge is reported
        var org = s.origin, nxt = s.next, fc = s.face
        let extra = org.count; org += [org[0], org[1], org[2]]; nxt += [Int32(extra + 1), Int32(extra + 2), Int32(extra)]; fc += [Int32(he.faceSlots), Int32(he.faceSlots), Int32(he.faceSlots)]
        XCTAssertGreaterThan(try XCTUnwrap(HalfEdgeMesh.gpuEdgeTable(origin: org, next: nxt, face: fc)).duplicates, 0)
    }

    func testHalfEdgeGPUImportEqualsCPUImport() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        var soup = sphere(lon: 36, lat: 18)
        // degenerate triangles + a repeated vertex to weld
        let base = UInt32(soup.positions.count)
        for p in [Vec3(5, 5, 5), Vec3(5, 5, 5), Vec3(6, 5, 5), Vec3(7, 5, 5)] { soup.positions.append(p); soup.normals.append(Vec3(0, 1, 0)); soup.uvs.append(.zero) }
        soup.indices += [base, base + 1, base + 2, base, base + 2, base + 3]       // a welded repeat, and a zero-area sliver
        let cpu = HalfEdgeMesh.imported(soup)
        let gpu = try XCTUnwrap(HalfEdgeMesh.importedGPU(soup))
        XCTAssertEqual(gpu.report, cpu.report)
        XCTAssertGreaterThan(cpu.report.degenerate, 0)
        let (a, b) = (gpu.mesh, cpu.mesh)
        XCTAssertEqual(a.heOrigin, b.heOrigin); XCTAssertEqual(a.heNext, b.heNext); XCTAssertEqual(a.hePrev, b.hePrev)
        XCTAssertEqual(a.heTwin, b.heTwin); XCTAssertEqual(a.heFace, b.heFace); XCTAssertEqual(a.heEdge, b.heEdge)
        XCTAssertEqual(a.faceHE, b.faceHE); XCTAssertEqual(a.edgeHE, b.edgeHE); XCTAssertEqual(a.vertexHE, b.vertexHE)
        XCTAssertEqual(a.directed, b.directed); XCTAssertEqual(a.heUV, b.heUV)
        XCTAssertEqual(a.positions.count, b.positions.count)
        for (p, q) in zip(a.positions, b.positions) { XCTAssertEqual(len3(p - q), 0, accuracy: 1e-12) }
        XCTAssertTrue(a.isValid)
        // inconsistently wound input: the GPU declines, `importedAuto` still gives the CPU's mesh
        var flipped = soup
        for t in stride(from: 0, to: 30, by: 3) { flipped.indices.swapAt(t, t + 1) }
        XCTAssertNil(HalfEdgeMesh.importedGPU(flipped))
        XCTAssertEqual(HalfEdgeMesh.importedAuto(flipped).report, HalfEdgeMesh.imported(flipped).report)
    }

    func testHalfEdgeGPUTimings() throws {
        guard ModellingGPU.shared != nil else { throw XCTSkip("no Metal device") }
        let s = SDF.sphere(radius: 0.8).union(SDF.torus(major: 0.7, minor: 0.2), smooth: 0.1)
            .meshGPU(lo: Vec3(repeating: -1.1), hi: Vec3(repeating: 1.1), cell: 0.0075)
        let soup = try XCTUnwrap(s)
        // Mesh3 straight from the mesher is welded and indexed, which is what the weld wants; make it a soup too.
        var tri = Mesh3()
        for t in stride(from: 0, to: soup.indices.count, by: 3) {
            let i = (0..<3).map { Int(soup.indices[t + $0]) }
            let b = UInt32(tri.positions.count)
            for k in i { tri.positions.append(soup.positions[k]); tri.normals.append(soup.normals[k]); tri.uvs.append(soup.uvs[k]) }
            tri.indices += [b, b + 1, b + 2]
        }
        func ms<T>(_ f: () -> T) -> (T, Double) { let t = Date(); let r = f(); return (r, Date().timeIntervalSince(t) * 1000) }
        _ = HalfEdgeMesh.importedGPU(tri)
        let (cpu, cpuMS) = ms { HalfEdgeMesh.imported(tri) }
        let (gpu, gpuMS) = ms { HalfEdgeMesh.importedGPU(tri) }
        let (kern, kernMS) = ms { HalfEdgeTopologyGPU.weldAndBuild(tri, tolerance: 1e-6) }
        XCTAssertNotNil(kern)
        let g = try XCTUnwrap(gpu)
        XCTAssertEqual(g.mesh.heTwin, cpu.mesh.heTwin)
        let he = cpu.mesh
        _ = he.gpuBulk()
        let (_, bulkCPU) = ms { he.cpuBulk() }
        let (_, bulkGPU) = ms { he.gpuBulk() }
        let (_, soaMS) = ms { he.soa() }
        print(String(format: "HalfEdge M4 timing %d faces: import CPU %.0f ms | GPU import total %.0f ms (kernels+readback %.0f, rest = Int arrays, uv/normal gather, directed dict) | bulk CPU %.1f ms vs GPU %.1f ms (of which soa() %.1f)",
                     he.faceCount, cpuMS, gpuMS, kernMS, bulkCPU, bulkGPU, soaMS))
    }
}
