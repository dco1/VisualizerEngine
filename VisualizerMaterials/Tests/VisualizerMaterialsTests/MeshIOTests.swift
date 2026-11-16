import XCTest
@testable import VisualizerMaterials

/// MeshIO: OBJ / PLY / STL round trips judged by what a format must keep (triangle winding, volume, UVs, normals),
/// plus the awkward inputs (n-gons, negative indices, binary STL under a "solid" header, garbage).
final class MeshIOTests: XCTestCase {

    // MARK: fixtures

    private func box(_ lo: Vec3 = Vec3(0, 0, 0), _ hi: Vec3 = Vec3(1, 2, 3)) -> Mesh3 {
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

    /// UV sphere with shared vertices, smooth normals, UVs, outward winding.
    private func sphere(r: Double = 0.5, seg: Int = 12, rings: Int = 8) -> Mesh3 {
        var m = Mesh3()
        for j in 0...rings {
            let th = Double.pi * Double(j) / Double(rings)
            for i in 0...seg {
                let ph = 2 * Double.pi * Double(i) / Double(seg)
                let n = Vec3(sin(th) * cos(ph), cos(th), sin(th) * sin(ph))
                m.positions.append(n * r); m.normals.append(n)
                m.uvs.append(Vec2(Double(i) / Double(seg), Double(j) / Double(rings)))
            }
        }
        for j in 0..<rings { for i in 0..<seg {
            let a = UInt32(j * (seg + 1) + i), b = a + 1, c = a + UInt32(seg + 1), d = c + 1
            for t in [(a, b, c), (b, d, c)] {
                let p = [m.positions[Int(t.0)], m.positions[Int(t.1)], m.positions[Int(t.2)]]
                if len3(cross3(p[1] - p[0], p[2] - p[0])) < 1e-12 { continue }            // pole sliver
                m.indices += [t.0, t.1, t.2]
            }
        } }
        return m
    }

    private func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            v += dot3(m.positions[Int(m.indices[t])], cross3(m.positions[Int(m.indices[t + 1])], m.positions[Int(m.indices[t + 2])])) / 6; t += 3
        }
        return v
    }

    /// Per triangle, per corner: (position, uv, normal) — independent of vertex numbering, dependent on winding order.
    private func corners(_ m: Mesh3) -> [[Double]] {
        var out: [[Double]] = []; var t = 0
        while t + 2 < m.indices.count {
            var row: [Double] = []
            for k in 0..<3 {
                let i = Int(m.indices[t + k])
                row += [m.positions[i].x, m.positions[i].y, m.positions[i].z, m.uvs[i].x, m.uvs[i].y,
                        m.normals[i].x, m.normals[i].y, m.normals[i].z]
            }
            out.append(row); t += 3
        }
        return out
    }

    private func assertSame(_ a: Mesh3, _ b: Mesh3, tol: Double, file: StaticString = #filePath, line: UInt = #line) {
        let (ca, cb) = (corners(a), corners(b))
        XCTAssertEqual(ca.count, cb.count, "triangle count", file: file, line: line)
        for (x, y) in zip(ca, cb) { for (p, q) in zip(x, y) { XCTAssertEqual(p, q, accuracy: tol, file: file, line: line) } }
        XCTAssertEqual(b.normals.count, b.positions.count, file: file, line: line)
        XCTAssertEqual(b.uvs.count, b.positions.count, file: file, line: line)
    }

    private func obj(_ s: String) throws -> MeshIO.Result { try MeshIO.read(data: Data(s.utf8), format: .obj) }

    // MARK: round trips

    func testBoxRoundTripAllFormats() throws {
        let b = box()
        for (fmt, bin, tol) in [(MeshIO.Format.obj, true, 1e-12), (.ply, true, 1e-6), (.ply, false, 1e-12), (.stl, true, 1e-6), (.stl, false, 1e-12)] {
            var o = MeshIO.WriteOptions(); o.binary = bin
            let r = try MeshIO.read(data: MeshIO.write(b, format: fmt, options: o), format: fmt)
            XCTAssertEqual(r.meshes.count, 1, "\(fmt) bin=\(bin)")
            let m = r.meshes[0]
            XCTAssertEqual(volume(m), 6, accuracy: 1e-5, "\(fmt) bin=\(bin): volume (and so winding)")
            XCTAssertEqual(m.triangleCount, 12)
            // positions in triangle order are the same; normals are the flat face normals for the box in every format
            // (STL carries no UVs, so compare just positions + normals there).
            if fmt == .stl {
                for (x, y) in zip(corners(b), corners(m)) {
                    for k in 0..<3 { for c in [0, 1, 2, 5, 6, 7] { XCTAssertEqual(x[k * 8 + c], y[k * 8 + c], accuracy: tol) } }
                }
            } else { assertSame(b, m, tol: tol) }
        }
    }

    func testSphereWithUVsRoundTrip() throws {
        let s = sphere()
        XCTAssertGreaterThan(volume(s), 0.4)
        for (fmt, tol) in [(MeshIO.Format.obj, 1e-12), (.ply, 1e-6)] {
            let r = try MeshIO.read(data: MeshIO.write(s, format: fmt), format: fmt).meshes[0]
            assertSame(s, r, tol: tol)
            XCTAssertEqual(volume(r), volume(s), accuracy: 1e-6)
        }
        // shared vertices stay shared in OBJ (key = v/vt/vn)
        let r = try MeshIO.read(data: MeshIO.write(s, format: .obj), format: .obj).meshes[0]
        XCTAssertLessThanOrEqual(r.vertexCount, s.vertexCount)
    }

    func testOBJMultipleMeshesNamesAndMaterials() throws {
        var o = MeshIO.WriteOptions()
        o.names = ["crate", "ball"]; o.materialNames = ["wood", nil]; o.mtlFileName = "scene.mtl"
        let (objData, mtl) = MeshIO.writeOBJ([box(), sphere()], options: o)
        XCTAssertTrue(String(decoding: mtl, as: UTF8.self).contains("newmtl wood"))
        let r = try MeshIO.read(data: objData, format: .obj)
        XCTAssertEqual(r.names, ["crate", "ball"])
        XCTAssertEqual(r.materialNames.map { $0 ?? "-" }, ["wood", "-"])
        XCTAssertEqual(r.materialLibraries, ["scene.mtl"])
        XCTAssertEqual(r.meshes[1].triangleCount, sphere().triangleCount)
    }

    func testPLYColours() throws {
        let b = box()
        var o = MeshIO.WriteOptions()
        o.colors = [(0..<b.vertexCount).map { SIMD3<Float>(Float($0 % 2), 0.5, 1) }]
        for bin in [true, false] {
            o.binary = bin
            let r = try MeshIO.read(data: MeshIO.write(b, format: .ply, options: o), format: .ply)
            XCTAssertEqual(r.colors[0].count, b.vertexCount)
            XCTAssertEqual(r.colors[0][1].x, 1, accuracy: 1e-6)
            XCTAssertEqual(r.colors[0][1].y, 0.5, accuracy: 1 / 255)
        }
    }

    // MARK: OBJ specifics

    func testNgonFacesTriangulatedWindingKept() throws {
        // a pentagon, a concave L (CCW seen from +Z) and a quad face with v/vt/vn corners
        let r = try obj("""
        v 0 0 0
        v 2 0 0
        v 3 1.5 0
        v 1 3 0
        v -1 1.5 0
        v 10 0 0
        v 13 0 0
        v 13 1 0
        v 11 1 0
        v 11 3 0
        v 10 3 0
        f 1 2 3 4 5
        f 6 7 8 9 10 11
        """)
        let m = r.meshes[0]
        XCTAssertEqual(m.triangleCount, 3 + 4)
        var area = 0.0; var t = 0
        while t + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], c = m.positions[Int(m.indices[t + 2])]
            let n = cross3(b - a, c - a)
            XCTAssertGreaterThan(n.z, 0, "every triangle keeps the polygon's CCW winding")
            area += n.z / 2; t += 3
        }
        // pentagon 7.5 (shoelace) + L: 3*1 + 1*2 = 5
        XCTAssertEqual(area, 7.5 + 5, accuracy: 1e-9)
        XCTAssertEqual(m.normals.count, m.positions.count)
        XCTAssertEqual(m.normals[0].z, 1, accuracy: 1e-9, "no vn in the file → made from the faces")
    }

    func testNegativeIndicesAndIndexForms() throws {
        let r = try obj("""
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        vt 0 0
        vt 1 0
        vt 1 1
        vt 0 1
        vn 0 0 1
        f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1
        f 1//1 2//1 3//1
        f 1/1 2/2 3/3
        f 1 3 4
        """)
        let m = r.meshes[0]
        XCTAssertEqual(m.triangleCount, 2 + 1 + 1 + 1)
        XCTAssertEqual(Set(m.positions.map { [$0.x, $0.y] }), [[0, 0], [1, 0], [1, 1], [0, 1]])
        for i in 0..<m.vertexCount {          // the file's uvs equal its positions' xy; every corner kept its own
            XCTAssertTrue(m.uvs[i] == .zero || (m.uvs[i].x == m.positions[i].x && m.uvs[i].y == m.positions[i].y), "uv is the file's or none")
            XCTAssertEqual(m.normals[i], Vec3(0, 0, 1))
        }
    }

    func testGroupsSplitIntoMeshes() throws {
        let r = try obj("""
        mtllib a.mtl b.mtl
        v 0 0 0
        v 1 0 0
        v 0 1 0
        g first
        usemtl red
        f 1 2 3
        g second
        usemtl blue
        f 1 2 3
        usemtl green
        f 3 2 1
        """)
        XCTAssertEqual(r.meshes.count, 3)
        XCTAssertEqual(r.names, ["first", "second", "second"])
        XCTAssertEqual(r.materialNames.compactMap { $0 }, ["red", "blue", "green"])
        XCTAssertEqual(r.materialLibraries, ["a.mtl", "b.mtl"])
    }

    // MARK: STL

    func testBinarySTLWithSolidHeaderIsBinary() throws {
        var d = MeshIO.write(box(), format: .stl)           // binary
        let h = Array("solid exported-by-some-cad-tool".utf8)
        d.replaceSubrange(0..<h.count, with: h)
        XCTAssertTrue(d.starts(with: Array("solid".utf8)))
        let r = try MeshIO.read(data: d)
        XCTAssertEqual(r.meshes[0].triangleCount, 12)
        XCTAssertEqual(volume(r.meshes[0]), 6, accuracy: 1e-5)
    }

    func testSTLWeldAndSmooth() throws {
        let d = MeshIO.write(box(), format: .stl)
        let flat = try MeshIO.read(data: d).meshes[0]
        XCTAssertEqual(flat.vertexCount, 36)
        let w = try MeshIO.read(data: d, options: .init(weldAndSmooth: true, creaseDegrees: 80)).meshes[0]
        XCTAssertEqual(w.vertexCount, 8)
        XCTAssertEqual(volume(w), 6, accuracy: 1e-5)
        // a sphere welds closed and gets outward smooth normals
        let s = try MeshIO.read(data: MeshIO.write(sphere(), format: .stl), options: .init(weldAndSmooth: true)).meshes[0]
        for (p, n) in zip(s.positions, s.normals) { XCTAssertGreaterThan(dot3(normalize3(p), n), 0.95) }
    }

    func testAsciiSTLNamesAndMultipleSolids() throws {
        var o = MeshIO.WriteOptions(); o.binary = false; o.names = ["one", "two"]
        let r = try MeshIO.read(data: MeshIO.write([box(), box(Vec3(5, 5, 5), Vec3(6, 6, 6))], format: .stl, options: o))
        XCTAssertEqual(r.names, ["one", "two"])
        XCTAssertEqual(r.meshes.map(\.triangleCount), [12, 12])
    }

    // MARK: malformed input

    func testMalformedInputThrowsNotCrashes() {
        func fails(_ d: Data, _ f: MeshIO.Format?, _ what: String, line: UInt = #line) {
            XCTAssertThrowsError(try MeshIO.read(data: d, format: f), what, line: line)
        }
        fails(Data(), .obj, "empty obj"); fails(Data(), nil, "empty, undetectable")
        fails(Data("v 0 0\nf 1 2 3".utf8), .obj, "short vertex")
        fails(Data("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4".utf8), .obj, "index past end")
        fails(Data("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -9".utf8), .obj, "negative past start")
        fails(Data("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2".utf8), .obj, "index 0")
        fails(Data("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2".utf8), .obj, "two-corner face")
        fails(Data("v a b c".utf8), .obj, "nan text")
        fails(Data("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nend_header\n1 2\n".utf8), .ply, "ply no y z")
        fails(Data("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n".utf8), .ply, "ply truncated")
        fails(Data("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n3 0 1 2\n".utf8), .ply, "ply face index range")
        fails(Data("ply\nformat ascii 1.0\nelement vertex 1\n".utf8), .ply, "no end_header")
        fails(Data("solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\n".utf8), .stl, "ascii stl cut off")
        fails(Data("garbage garbage".utf8), .stl, "stl garbage")
        var bin = MeshIO.write(box(), format: .stl); bin.removeLast(10)
        fails(bin, .stl, "binary stl truncated")
        var ply = MeshIO.write(box(), format: .ply); ply.removeLast(7)
        fails(ply, .ply, "binary ply truncated")
    }

    func testByteMutationsNeverCrash() {
        var rng = SystemRandomNumberGenerator()
        let seeds = [MeshIO.write(sphere(), format: .obj), MeshIO.write(box(), format: .ply),
                     MeshIO.write(box(), format: .stl), MeshIO.write(box(), format: .stl, options: { var o = MeshIO.WriteOptions(); o.binary = false; return o }())]
        for seed in seeds {
            for _ in 0..<60 {
                var d = seed
                for _ in 0..<5 { d[Int.random(in: 0..<d.count, using: &rng)] = UInt8.random(in: 0...255, using: &rng) }
                if Bool.random(using: &rng) { d = d.prefix(Int.random(in: 0..<d.count, using: &rng)) }
                for f in MeshIO.Format.allCases { _ = try? MeshIO.read(data: d, format: f) }   // result irrelevant; no trap
            }
        }
    }

    func testURLWrapperAndDetect() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("meshio-\(UUID().uuidString).ply")
        defer { try? FileManager.default.removeItem(at: url) }
        try MeshIO.write([box()], to: url)
        XCTAssertEqual(try MeshIO.read(url: url).meshes[0].triangleCount, 12)
        XCTAssertEqual(MeshIO.detect(MeshIO.write(box(), format: .obj)), .obj)
        XCTAssertEqual(MeshIO.detect(MeshIO.write(box(), format: .stl)), .stl)
    }
}
