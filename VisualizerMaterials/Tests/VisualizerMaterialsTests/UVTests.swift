import XCTest
@testable import VisualizerMaterials

final class UVTests: XCTestCase {

    // MARK: fixtures

    private func box(_ w: Double, _ h: Double, _ d: Double, at o: Vec3 = .zero) -> Mesh3 {
        var m = Mesh3()
        let (x, y, z) = (w, h, d)
        func q(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ e: Vec3, _ n: Vec3) {
            m.addQuad(a + o, b + o, c + o, e + o, outward: n)
        }
        q(Vec3(x,0,0), Vec3(x,y,0), Vec3(x,y,z), Vec3(x,0,z), Vec3( 1,0,0))
        q(Vec3(0,0,0), Vec3(0,0,z), Vec3(0,y,z), Vec3(0,y,0), Vec3(-1,0,0))
        q(Vec3(0,y,0), Vec3(0,y,z), Vec3(x,y,z), Vec3(x,y,0), Vec3(0, 1,0))
        q(Vec3(0,0,0), Vec3(x,0,0), Vec3(x,0,z), Vec3(0,0,z), Vec3(0,-1,0))
        q(Vec3(0,0,z), Vec3(x,0,z), Vec3(x,y,z), Vec3(0,y,z), Vec3(0,0, 1))
        q(Vec3(0,0,0), Vec3(0,y,0), Vec3(x,y,0), Vec3(x,0,0), Vec3(0,0,-1))
        return m
    }

    /// Open tube around Z, indexed: ring-major, `seg` vertices per ring (no duplicated seam column).
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

    /// UV sphere with single pole vertices (indexed); returns the mesh, south and north pole indices.
    private func sphere(radius: Double, seg: Int, rings: Int) -> (Mesh3, Int, Int) {
        var m = Mesh3()
        m.positions.append(Vec3(0, 0, -radius))
        for r in 1..<rings { for s in 0..<seg {
            let lat = -Double.pi / 2 + Double.pi * Double(r) / Double(rings)
            let a = 2 * Double.pi * Double(s) / Double(seg)
            m.positions.append(Vec3(radius * cos(lat) * cos(a), radius * cos(lat) * sin(a), radius * sin(lat)))
        } }
        m.positions.append(Vec3(0, 0, radius))
        let south = 0, north = m.positions.count - 1
        func ring(_ r: Int, _ s: Int) -> UInt32 { UInt32(1 + (r - 1) * seg + (s % seg)) }
        for s in 0..<seg { m.indices += [0, ring(1, s + 1), ring(1, s)] }
        for r in 1..<(rings - 1) { for s in 0..<seg {
            m.indices += [ring(r, s), ring(r, s + 1), ring(r + 1, s + 1), ring(r, s), ring(r + 1, s + 1), ring(r + 1, s)]
        } }
        for s in 0..<seg { m.indices += [UInt32(north), ring(rings - 1, s), ring(rings - 1, s + 1)] }
        return (m, south, north)
    }

    private func overlap(_ a: (min: Vec2, max: Vec2), _ b: (min: Vec2, max: Vec2)) -> Bool {
        a.min.x < b.max.x - 1e-9 && b.min.x < a.max.x - 1e-9 && a.min.y < b.max.y - 1e-9 && b.min.y < a.max.y - 1e-9
    }

    // MARK: projections and islands

    func testCubeBoxProjectionHasSixIslands() {
        let cube = box(1, 1, 1, at: Vec3(-0.5, -0.5, -0.5))
        let projected = cube.projectedUVs(UVProjection(.box))
        XCTAssertEqual(projected.triangleCount, 12)
        let isl = projected.uvLayouts()
        XCTAssertEqual(isl.count, 6)
        XCTAssertTrue(isl.allSatisfy { $0.island.triangles.count == 2 })
        let r = UVReport(projected)
        XCTAssertEqual(r.flipped, 0, "box faces must not be mirrored")
        XCTAssertEqual(r.maxAnisotropy, 1, accuracy: 1e-6)
        XCTAssertEqual(r.densityVariation, 0, accuracy: 1e-6)
    }

    func testPlanarScaleOffsetAndCylinderWrap() {
        var m = Mesh3()
        m.addQuad(Vec3(0,0,0), Vec3(2,0,0), Vec3(2,1,0), Vec3(0,1,0), outward: Vec3(0,0,1))
        let uv = UVProjection(.planar, scale: Vec2(0.5, 2), offset: Vec2(0.25, 0)).cornerUVs(of: m)
        XCTAssertEqual(uv.map(\.x).max()!, 1.25, accuracy: 1e-12)
        XCTAssertEqual(uv.map(\.y).max()!, 2, accuracy: 1e-12)

        let cyl = tube(radius: 1, height: 2, seg: 24, rings: 2).projectedUVs(UVProjection(.cylinder, reference: Vec3(1, 0, 0)))
        for t in 0..<cyl.triangleCount {
            let us = (0..<3).map { cyl.uvs[Int(cyl.indices[3 * t + $0])].x }
            XCTAssertLessThan(us.max()! - us.min()!, 0.5, "no triangle may be smeared across the wrap")
        }
    }

    func testSpherePolesDoNotSmear() {
        let (s, _, _) = sphere(radius: 1, seg: 24, rings: 12)
        let p = s.projectedUVs(UVProjection(.sphere))
        for t in 0..<p.triangleCount {
            let us = (0..<3).map { p.uvs[Int(p.indices[3 * t + $0])].x }
            XCTAssertLessThan(us.max()! - us.min()!, 0.5)
        }
        XCTAssertEqual(p.uvs.map(\.y).min()!, 0, accuracy: 1e-9)
        XCTAssertEqual(p.uvs.map(\.y).max()!, 1, accuracy: 1e-9)
    }

    func testSeamsByAngleAndExplicit() {
        let cube = box(1, 1, 1)
        let topo = UVTopology(cube)
        XCTAssertEqual(topo.seams(angleDegrees: 30).count, 12)       // all cube edges
        XCTAssertEqual(topo.islands(topo.seams(angleDegrees: 30)).count, 6)
        XCTAssertEqual(topo.islands(UVSeams()).count, 1)
        XCTAssertEqual(topo.boundaryEdges.count, 0)
    }

    // MARK: unwrap

    func testCylinderSideUnwrapsToARectangle() {
        let (r, h, seg) = (0.5, 2.0, 48)
        let t = tube(radius: r, height: h, seg: seg, rings: 8)
        let topo = UVTopology(t)
        let seam = topo.seams(edges: (0..<8).map { (($0) * seg, ($0 + 1) * seg) })
        let res = t.unwrappedUVs(seams: seam, pack: nil)
        XCTAssertEqual(res.layouts.count, 1)
        XCTAssertEqual(res.report.maxAnisotropy, 1, accuracy: 1e-4)
        XCTAssertEqual(res.report.flipped, 0)
        let size = res.layouts[0].size
        let perimeter = 2 * Double(seg) * r * sin(Double.pi / Double(seg))
        XCTAssertEqual(max(size.x, size.y), perimeter, accuracy: perimeter * 1e-3)
        XCTAssertEqual(min(size.x, size.y), h, accuracy: h * 1e-3)
        // the unwrap is metric, so the UV area is the surface area
        XCTAssertEqual(res.report.islands[0].areaUV, res.report.islands[0].area3D, accuracy: 1e-6)
        // seam vertices are duplicated, nothing else
        XCTAssertEqual(res.mesh.vertexCount, t.vertexCount + 9)
    }

    func testSphereUnwrapHasBoundedStretch() {
        let (s, south, north) = sphere(radius: 1, seg: 32, rings: 16)
        let topo = UVTopology(s)
        let seam = topo.seams(pathFrom: south, to: north, in: s)
        XCTAssertEqual(seam.count, 16)
        let res = s.unwrappedUVs(seams: seam, pack: nil)
        XCTAssertEqual(res.layouts.count, 1)
        XCTAssertEqual(res.report.flipped, 0)
        XCTAssertLessThan(res.report.meanAnisotropy, 1.5)
        XCTAssertLessThan(res.report.maxAnisotropy, 3.0)
    }

    func testCubeUnwrapByAngleIsIsometricAndUniform() {
        let cube = box(1, 2, 3)
        let topo = UVTopology(cube)
        let res = cube.unwrappedUVs(seams: topo.seams(angleDegrees: 30))
        XCTAssertEqual(res.layouts.count, 6)
        XCTAssertEqual(res.report.maxAnisotropy, 1, accuracy: 1e-6)
        XCTAssertLessThan(res.report.densityVariation, 1e-6)
        XCTAssertEqual(res.report.flipped, 0)
    }

    // MARK: packing

    func testPackedIslandsFitAndDoNotOverlap() {
        // a 1×2×3 box and a big 4×4×4 one, welded as one mesh: 12 islands of very different size
        var m = box(1, 2, 3)
        m.append(box(4, 4, 4, at: Vec3(10, 0, 0)))
        let topo = UVTopology(m)
        for rotate in [true, false] {
            let opts = UVPackOptions(margin: 0.01, rotateToFit: rotate)
            let res = m.unwrappedUVs(seams: topo.seams(angleDegrees: 30), pack: opts)
            XCTAssertEqual(res.layouts.count, 12)
            let boxes = res.layouts.map(\.bounds)
            for b in boxes {
                XCTAssertGreaterThanOrEqual(b.min.x, 0.01 - 1e-9); XCTAssertGreaterThanOrEqual(b.min.y, 0.01 - 1e-9)
                XCTAssertLessThanOrEqual(b.max.x, 0.99 + 1e-9); XCTAssertLessThanOrEqual(b.max.y, 0.99 + 1e-9)
            }
            for i in boxes.indices { for j in (i + 1)..<boxes.count {
                XCTAssertFalse(overlap(boxes[i], boxes[j]), "islands \(i) and \(j) overlap (rotate \(rotate))")
            } }
            // one common scale: texel density uniform
            XCTAssertLessThan(res.report.densityVariation, 1e-6)
            XCTAssertEqual(res.report.maxAnisotropy, 1, accuracy: 1e-6)
            XCTAssertGreaterThan(res.report.coverage, 0.2)
            XCTAssertEqual(res.report.texelDensity(textureSize: 1024), res.report.meanDensity * 1024, accuracy: 1e-9)
        }
    }

    func testNormalizedPerIslandGivesEqualAreas() {
        var m = box(1, 1, 1)
        m.append(box(5, 5, 5, at: Vec3(10, 0, 0)))
        let topo = UVTopology(m)
        let res = m.unwrappedUVs(seams: topo.seams(angleDegrees: 30), pack: UVPackOptions(density: .normalizedPerIsland))
        let areas = res.report.islands.map(\.areaUV)
        XCTAssertEqual(areas.min()!, areas.max()!, accuracy: areas.max()! * 1e-6)
        XCTAssertGreaterThan(res.report.densityVariation, 0.1)
    }

    func testPackingProjectedBoxCubeKeepsStretchFree() {
        let cube = box(1, 1, 1)
        let res = cube.projectedUVs(UVProjection(.box, origin: Vec3(0.5, 0.5, 0.5))).packedUVs()
        XCTAssertEqual(res.layouts.count, 6)
        let boxes = res.layouts.map(\.bounds)
        for i in boxes.indices { for j in (i + 1)..<boxes.count { XCTAssertFalse(overlap(boxes[i], boxes[j])) } }
        XCTAssertEqual(res.report.maxAnisotropy, 1, accuracy: 1e-6)
        XCTAssertEqual(res.report.flipped, 0)
        XCTAssertLessThanOrEqual(res.mesh.uvs.map(\.x).max()!, 1)
    }
}
