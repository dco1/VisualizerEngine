import XCTest
import simd
@testable import VisualizerMaterials

/// MeshIO's GPU path (item M5) held to the CPU path: STL / PLY decode and encode, STL weld + smooth, and the GPU
/// winding / volume report. Skipped without a Metal device. `testTimings1M` prints CPU vs GPU on a 1M-triangle STL.
final class MeshIOGPUTests: XCTestCase {

    override func setUpWithError() throws {
        guard MeshIO.gpuAvailable else { throw XCTSkip("no Metal device") }
    }

    // MARK: fixtures

    /// Shared-vertex UV sphere without the seam / pole slivers, as a triangle soup builder input.
    private func sphere(r: Double = 1, seg: Int, rings: Int) -> Mesh3 {
        var m = Mesh3()
        for j in 0 ... rings {
            let th = Double.pi * Double(j) / Double(rings)
            for i in 0 ... seg {
                let ph = 2 * Double.pi * Double(i) / Double(seg)
                let n = Vec3(sin(th) * cos(ph), cos(th), sin(th) * sin(ph))
                m.positions.append(n * r); m.normals.append(n)
                m.uvs.append(Vec2(Double(i) / Double(seg), Double(j) / Double(rings)))
            }
        }
        for j in 0 ..< rings { for i in 0 ..< seg {
            let a = UInt32(j * (seg + 1) + i), b = a + 1, c = a + UInt32(seg + 1), d = c + 1
            for t in [(a, b, c), (b, d, c)] {
                let p = [m.positions[Int(t.0)], m.positions[Int(t.1)], m.positions[Int(t.2)]]
                if len3(cross3(p[1] - p[0], p[2] - p[0])) < 1e-12 { continue }
                m.indices += [t.0, t.1, t.2]
            }
        } }
        return m
    }

    private func box(_ lo: Vec3, _ hi: Vec3) -> Mesh3 {
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

    /// A box and a sphere in one STL (creases and smooth parts), via the CPU writer.
    private func mixedSTL() -> (Mesh3, Data) {
        var m = sphere(r: 0.8, seg: 40, rings: 20)
        m.append(box(Vec3(2, 0, 0), Vec3(3, 1, 2)))
        return (m, MeshIO.write(m, format: .stl, backend: .cpu))
    }

    private func maxNormalDiff(_ a: Mesh3, _ b: Mesh3) -> Double {
        zip(a.normals, b.normals).map { len3($0 - $1) }.max() ?? 0
    }

    // MARK: STL

    func testSTLDecodeMatchesCPU() throws {
        var (_, d) = mixedSTL()
        // Stored normals the reader has to judge: zero, against the winding, and agreeing.
        d.withUnsafeMutableBytes { raw in
            for (i, mode) in [(0, 0), (1, 1), (2, 2)] {
                let o = 84 + 50 * i
                for k in 0 ..< 3 {
                    let v = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o + 4 * k, as: UInt32.self)))
                    let w: Float = mode == 0 ? 0 : (mode == 1 ? -v : v)
                    raw.storeBytes(of: w.bitPattern.littleEndian, toByteOffset: o + 4 * k, as: UInt32.self)
                }
            }
        }
        let cpu = try MeshIO.read(data: d, format: .stl, backend: .cpu).merged
        let gpu = try MeshIO.read(data: d, format: .stl, backend: .gpu).merged
        XCTAssertEqual(cpu.positions, gpu.positions, "positions are bit-identical (float32 widened)")
        XCTAssertEqual(cpu.indices, gpu.indices)
        XCTAssertEqual(cpu.uvs, gpu.uvs)
        XCTAssertLessThan(maxNormalDiff(cpu, gpu), 1e-6)
    }

    func testSTLWeldSmoothMatchesCPU() throws {
        let (m, d) = mixedSTL()
        let o = MeshIO.ReadOptions(weldAndSmooth: true, creaseDegrees: 40)
        let cpu = try MeshIO.read(data: d, format: .stl, options: o, backend: .cpu).merged
        let gpu = try MeshIO.read(data: d, format: .stl, options: o, backend: .gpu).merged
        XCTAssertGreaterThan(cpu.positions.count, 100)
        XCTAssertLessThan(cpu.positions.count, m.indices.count, "welded")
        XCTAssertEqual(cpu.positions.count, gpu.positions.count)
        XCTAssertEqual(cpu.indices, gpu.indices, "same vertex order, same triangles")
        XCTAssertEqual(cpu.positions, gpu.positions)
        XCTAssertLessThan(maxNormalDiff(cpu, gpu), 2e-3, "smooth normals agree (float math)")
    }

    func testSTLEncodeMatchesCPU() throws {
        let (m, _) = mixedSTL()
        let cpu = MeshIO.write(m, format: .stl, backend: .cpu)
        let gpu = MeshIO.write([m], format: .stl, backend: .gpu)
        XCTAssertEqual(cpu.count, gpu.count)
        XCTAssertEqual(cpu.prefix(84), gpu.prefix(84), "header and count")
        var worst: Float = 0, attrDiff = 0, posDiff = 0
        cpu.withUnsafeBytes { a in gpu.withUnsafeBytes { b in
            for t in 0 ..< (cpu.count - 84) / 50 {
                let o = 84 + 50 * t
                for k in 0 ..< 3 { worst = max(worst, abs(Float(bitPattern: a.loadUnaligned(fromByteOffset: o + 4 * k, as: UInt32.self))
                                                          - Float(bitPattern: b.loadUnaligned(fromByteOffset: o + 4 * k, as: UInt32.self)))) }
                for k in 12 ..< 48 where a[o + k] != b[o + k] { posDiff += 1 }
                for k in 48 ..< 50 where a[o + k] != b[o + k] { attrDiff += 1 }
            }
        } }
        XCTAssertEqual(posDiff, 0, "vertex bytes identical")
        XCTAssertEqual(attrDiff, 0)
        XCTAssertLessThan(worst, 1e-6, "facet normals (float vs double math)")
        // and it reads back as the same mesh
        let rt = try MeshIO.read(data: gpu, format: .stl, backend: .cpu).merged
        XCTAssertEqual(rt.indices.count, m.indices.count)
    }

    // MARK: PLY

    func testPLYWriteMatchesCPUByteForByte() throws {
        var m = sphere(seg: 24, rings: 12)
        m.append(box(Vec3(2, 0, 0), Vec3(3, 1, 2)))
        var o = MeshIO.WriteOptions()
        o.colors = [(0 ..< m.positions.count).map { SIMD3<Float>(Float($0 % 7) / 7, Float($0 % 5) / 5, Float($0 % 3) / 3) }]
        let cpu = MeshIO.write([m], format: .ply, options: o, backend: .cpu)
        let gpu = MeshIO.write([m], format: .ply, options: o, backend: .gpu)
        XCTAssertEqual(cpu, gpu, "binary PLY with normals, uvs and colours: byte-identical")
        // no colours, and no normals / uvs on the mesh (the CPU invents them from the faces)
        var bare = m; bare.normals = []; bare.uvs = []
        let c2 = MeshIO.write([bare], format: .ply, backend: .cpu), g2 = MeshIO.write([bare], format: .ply, backend: .gpu)
        XCTAssertEqual(c2.count, g2.count)
        let rc = try MeshIO.read(data: c2, backend: .cpu).merged, rg = try MeshIO.read(data: g2, backend: .cpu).merged
        XCTAssertEqual(rc.positions, rg.positions); XCTAssertEqual(rc.indices, rg.indices)
        XCTAssertLessThan(maxNormalDiff(rc, rg), 1e-6)
    }

    func testPLYReadMatchesCPU() throws {
        var m = sphere(seg: 24, rings: 12)
        m.append(box(Vec3(2, 0, 0), Vec3(3, 1, 2)))
        var o = MeshIO.WriteOptions()
        o.colors = [(0 ..< m.positions.count).map { SIMD3<Float>(Float($0 % 7) / 7, Float($0 % 5) / 5, 0.5) }]
        let d = MeshIO.write([m], format: .ply, options: o, backend: .cpu)
        let cpu = try MeshIO.read(data: d, backend: .cpu), gpu = try MeshIO.read(data: d, backend: .gpu)
        XCTAssertEqual(cpu.merged.positions, gpu.merged.positions)
        XCTAssertLessThan(maxNormalDiff(cpu.merged, gpu.merged), 1e-6, "stored normals, renormalised in float")
        XCTAssertEqual(cpu.merged.uvs, gpu.merged.uvs)
        XCTAssertEqual(cpu.merged.indices, gpu.merged.indices)
        XCTAssertEqual(cpu.colors, gpu.colors, "uchar colours -> 0…1 the same way")
    }

    /// A hand-written big-endian PLY with mixed property types and no normals (so the GPU smooths).
    func testPLYBigEndianMixedTypesNoNormals() throws {
        let s = sphere(seg: 16, rings: 8)
        // keep one copy of each position (drop seam duplicates) so the 0.1 mm weld finds no twins
        var seen: [SIMD3<Int64>: UInt32] = [:], pos: [Vec3] = [], remap = [UInt32](repeating: 0, count: s.positions.count)
        for (i, p) in s.positions.enumerated() {
            let k = Mesh3.qpos(p)
            if let j = seen[k] { remap[i] = j } else { remap[i] = UInt32(pos.count); seen[k] = remap[i]; pos.append(p) }
        }
        let tris = stride(from: 0, to: s.indices.count, by: 3).map { (remap[Int(s.indices[$0])], remap[Int(s.indices[$0 + 1])], remap[Int(s.indices[$0 + 2])]) }
            .filter { $0.0 != $0.1 && $0.1 != $0.2 && $0.0 != $0.2 }
        var d = Data("ply\nformat binary_big_endian 1.0\nelement vertex \(pos.count)\nproperty float x\nproperty float y\nproperty float z\nproperty ushort s\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face \(tris.count)\nproperty list uchar uint vertex_indices\nend_header\n".utf8)
        func be<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.bigEndian) { d.append(contentsOf: $0) } }
        for (i, p) in pos.enumerated() {
            be(Float(p.x).bitPattern); be(Float(p.y).bitPattern); be(Float(p.z).bitPattern)
            be(UInt16(i % 1000)); d.append(UInt8(i % 256)); d.append(UInt8(255 - i % 256)); d.append(UInt8(7))
        }
        for t in tris { d.append(3); be(t.0); be(t.1); be(t.2) }
        let cpu = try MeshIO.read(data: d, format: .ply, backend: .cpu), gpu = try MeshIO.read(data: d, format: .ply, backend: .gpu)
        XCTAssertEqual(cpu.merged.positions, gpu.merged.positions)
        XCTAssertEqual(cpu.merged.uvs, gpu.merged.uvs)
        XCTAssertEqual(cpu.merged.indices, gpu.merged.indices)
        XCTAssertEqual(cpu.colors, gpu.colors)
        XCTAssertLessThan(maxNormalDiff(cpu.merged, gpu.merged), 2e-3)
    }

    func testPLYFallsBackOnQuadsAndAscii() throws {
        var quad = Data("ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n".utf8)
        for f in [Float(0), 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0] { withUnsafeBytes(of: f.bitPattern.littleEndian) { quad.append(contentsOf: $0) } }
        quad.append(4)
        for i in [Int32(0), 1, 2, 3] { withUnsafeBytes(of: i.littleEndian) { quad.append(contentsOf: $0) } }
        XCTAssertNil(MeshIO.gpuReadPLY(quad), "n-gons are the CPU's")
        let r = try MeshIO.read(data: quad, format: .ply, backend: .gpu)
        XCTAssertEqual(r.merged.indices.count, 6, "the quad is still triangulated, by the CPU path")
        let ascii = MeshIO.write([sphere(seg: 8, rings: 4)], format: .ply, options: { var o = MeshIO.WriteOptions(); o.binary = false; return o }(), backend: .gpu)
        XCTAssertTrue(ascii.starts(with: Array("ply\nformat ascii".utf8)), "ascii write stays CPU")
    }

    // MARK: report

    func testReportVolumeAndWinding() throws {
        let welded = MeshIO.weldedSmooth(sphere(r: 2, seg: 48, rings: 24), crease: 40)
        let rep = try XCTUnwrap(MeshIO.gpuReport(welded))
        XCTAssertEqual(rep.signedVolume, welded.signedVolume, accuracy: abs(welded.signedVolume) * 1e-4)
        XCTAssertGreaterThan(rep.signedVolume, 0)
        XCTAssertTrue(rep.isClosedConsistent, "\(rep)")
        XCTAssertEqual(rep.flippedAgainstNormals, 0)
        // inside out
        var flipped = welded
        for t in stride(from: 0, to: flipped.indices.count, by: 3) { flipped.indices.swapAt(t + 1, t + 2) }
        let f = try XCTUnwrap(MeshIO.gpuReport(flipped))
        XCTAssertLessThan(f.signedVolume, 0)
        XCTAssertEqual(f.flippedAgainstNormals, flipped.indices.count / 3, "every triangle faces against the outward normals")
        XCTAssertTrue(f.isClosedConsistent, "flipped but consistent")
        // one triangle turned over: two duplicate directed edges... and its own three have no partner
        var bad = welded
        bad.indices.swapAt(31 * 3 + 1, 31 * 3 + 2)
        let b = try XCTUnwrap(MeshIO.gpuReport(bad))
        XCTAssertFalse(b.isClosedConsistent)
        XCTAssertEqual(b.duplicateDirectedEdges, 3, "\(b)")
        // a hole: dropping a triangle opens 3 edges
        var hole = welded; hole.indices.removeSubrange(30 * 3 ..< 31 * 3)
        let h = try XCTUnwrap(MeshIO.gpuReport(hole))
        XCTAssertEqual(h.openEdges, 3, "\(h)")
        XCTAssertEqual(h.duplicateDirectedEdges, 0)
        // CPU directed-edge count agrees
        var directed: [UInt64: Int] = [:]
        for t in stride(from: 0, to: hole.indices.count, by: 3) { for k in 0 ..< 3 {
            directed[UInt64(hole.indices[t + k]) << 32 | UInt64(hole.indices[t + (k + 1) % 3]), default: 0] += 1 } }
        XCTAssertEqual(directed.keys.filter { directed[($0 & 0xFFFF_FFFF) << 32 | ($0 >> 32)] == nil }.count, h.openEdges)
    }

    // MARK: timings

    /// 1M-triangle binary STL (a UV sphere as unshared facets), both ways. Prints; asserts equality, not speed.
    func testTimings1M() throws {
        let seg = 1000, rings = 501
        func pt(_ j: Int, _ i: Int) -> SIMD3<Float> {
            let th = Float.pi * Float(j) / Float(rings), ph = 2 * Float.pi * Float(i % seg) / Float(seg)
            return SIMD3(sin(th) * cos(ph), cos(th), sin(th) * sin(ph))
        }
        var d = Data(count: 84)
        var tris = 0
        d.reserveCapacity(84 + 50 * seg * rings * 2)
        func put(_ v: SIMD3<Float>) { for c in [v.x, v.y, v.z] { withUnsafeBytes(of: c.bitPattern.littleEndian) { d.append(contentsOf: $0) } } }
        for j in 0 ..< rings { for i in 0 ..< seg {
            let a = pt(j, i), b = pt(j, i + 1), c = pt(j + 1, i), e = pt(j + 1, i + 1)
            for t in [(a, b, c), (b, e, c)] {
                let n = simd_normalize(simd_cross(t.1 - t.0, t.2 - t.0))
                if !(simd_length(simd_cross(t.1 - t.0, t.2 - t.0)) > 1e-9) { continue }
                put(n); put(t.0); put(t.1); put(t.2); d.append(contentsOf: [0, 0]); tris += 1
            }
        } }
        d.withUnsafeMutableBytes { $0.storeBytes(of: UInt32(tris).littleEndian, toByteOffset: 80, as: UInt32.self) }
        func time<T>(_ label: String, _ f: () throws -> T) rethrows -> (T, Double) {
            let t0 = CFAbsoluteTimeGetCurrent(); let r = try f(); let dt = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "M5 %-34@ %9.1f ms", label as NSString, dt * 1000)); return (r, dt)
        }
        print("M5 triangles: \(tris), file \(d.count / 1_000_000) MB")
        _ = try MeshIO.read(data: d, format: .stl, backend: .gpu)                       // warm: shader compile happens once
        let (cpuR, _) = try time("STL read  CPU -> Mesh3") { try MeshIO.read(data: d, format: .stl, backend: .cpu) }
        let (gpuR, _) = try time("STL read  GPU -> Mesh3") { try MeshIO.read(data: d, format: .stl, backend: .gpu) }
        _ = time("STL read  GPU -> floats only") { MeshIO.gpuDecodeSTL(d, options: .init()) }
        XCTAssertEqual(cpuR.merged.positions, gpuR.merged.positions)
        let o = MeshIO.ReadOptions(weldAndSmooth: true, creaseDegrees: 40)
        let (cw, _) = try time("STL weld+smooth CPU -> Mesh3") { try MeshIO.read(data: d, format: .stl, options: o, backend: .cpu) }
        let (gw, _) = try time("STL weld+smooth GPU -> Mesh3") { try MeshIO.read(data: d, format: .stl, options: o, backend: .gpu) }
        _ = time("STL weld+smooth GPU -> floats only") { MeshIO.gpuDecodeSTL(d, options: o) }
        XCTAssertEqual(cw.merged.positions.count, gw.merged.positions.count)
        XCTAssertEqual(cw.merged.indices, gw.merged.indices)
        XCTAssertLessThan(maxNormalDiff(cw.merged, gw.merged), 2e-3)
        let welded = gw.merged
        let (cd, _) = time("STL write CPU (welded mesh)") { MeshIO.write([welded], format: .stl, backend: .cpu) }
        let (gd, _) = time("STL write GPU (welded mesh)") { MeshIO.write([welded], format: .stl, backend: .gpu) }
        XCTAssertEqual(cd.count, gd.count)
        let (cp, _) = time("PLY write CPU") { MeshIO.write([welded], format: .ply, backend: .cpu) }
        let (gp, _) = time("PLY write GPU") { MeshIO.write([welded], format: .ply, backend: .gpu) }
        XCTAssertEqual(cp, gp)
        let (pc, _) = try time("PLY read  CPU") { try MeshIO.read(data: cp, backend: .cpu) }
        let (pg, _) = try time("PLY read  GPU") { try MeshIO.read(data: cp, backend: .gpu) }
        XCTAssertEqual(pc.merged.positions, pg.merged.positions)
        let (rep, _) = time("report GPU (volume + edge table)") { MeshIO.gpuReport(welded) }
        _ = time("signedVolume CPU") { welded.signedVolume }
        print("M5 report: \(String(describing: rep))")
    }
}
