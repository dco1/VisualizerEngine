import XCTest
import SceneKit
import Metal
import simd
@testable import VisualizerRendering

/// Repro for the `from(scnGeometry:)` primitive mis-transform: large/rotated
/// SCNBox collapses toward a unit cube and flat/rotated SCNCylinder shears.
/// We read the OBJECT-space vertices the converter produces and compare the
/// derived bounding box against the authored primitive dimensions.
@MainActor
final class IlluminatoramaMeshPrimitiveReproTests: XCTestCase {

    private func bounds(_ positions: [SIMD3<Float>]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return (lo, hi)
    }

    func testBoundingBoxRecoversRealDims() throws {
        let box = SCNBox(width: 10, height: 3, length: 6, chamferRadius: 0)
        let bb = box.boundingBox
        print("BOX boundingBox min=\(bb.min) max=\(bb.max)")
        let cyl = SCNCylinder(radius: 4, height: 0.2)
        let cbb = cyl.boundingBox
        print("CYL boundingBox min=\(cbb.min) max=\(cbb.max)")
        // Hand-built mesh: vertex bounds should already equal boundingBox.
        let pts: [SCNVector3] = [SCNVector3(-2, 0, -3), SCNVector3(5, 0, -3),
                                 SCNVector3(5, 1, 7), SCNVector3(-2, 1, 7)]
        let src = SCNGeometrySource(vertices: pts)
        let elem = SCNGeometryElement(indices: [Int32]([0, 1, 2, 0, 2, 3]), primitiveType: .triangles)
        let custom = SCNGeometry(sources: [src], elements: [elem]) // winding-ok: test fixture flat quad, not shippable geometry
        let chb = custom.boundingBox
        if let tri = IlluminatoramaMesh.objectTriangles(scnGeometry: custom, elementIndex: 0) {
            let b = bounds(tri.positions)
            print("CUSTOM boundingBox min=\(chb.min) max=\(chb.max)  vertBounds min=\(b.min) max=\(b.max)")
        }
    }

    func testLargeBoxObjectSpaceExtent() throws {
        let box = SCNBox(width: 10, height: 3, length: 6, chamferRadius: 0)
        guard let tri = IlluminatoramaMesh.objectTriangles(scnGeometry: box, elementIndex: 0) else {
            return XCTFail("objectTriangles returned nil for SCNBox")
        }
        let b = bounds(tri.positions)
        let size = b.max - b.min
        print("BOX authored=(10,3,6)  object-space size=\(size)  min=\(b.min) max=\(b.max)  verts=\(tri.positions.count)")
        XCTAssertEqual(size.x, 10, accuracy: 0.01, "box width collapsed")
        XCTAssertEqual(size.y, 3,  accuracy: 0.01, "box height collapsed")
        XCTAssertEqual(size.z, 6,  accuracy: 0.01, "box length collapsed")
    }

    func testFlatCylinderObjectSpaceExtent() throws {
        // A wide, flat disc: radius 4, height 0.2.
        let cyl = SCNCylinder(radius: 4, height: 0.2)
        guard let tri = IlluminatoramaMesh.objectTriangles(scnGeometry: cyl, elementIndex: 0) else {
            return XCTFail("objectTriangles returned nil for SCNCylinder")
        }
        let b = bounds(tri.positions)
        let size = b.max - b.min
        print("CYL authored r=4 h=0.2 -> expect size~(8,0.2,8)  object-space size=\(size)  min=\(b.min) max=\(b.max)  verts=\(tri.positions.count)")
        XCTAssertEqual(size.x, 8,   accuracy: 0.05, "cylinder diameter wrong")
        XCTAssertEqual(size.y, 0.2, accuracy: 0.01, "cylinder height wrong")
        XCTAssertEqual(size.z, 8,   accuracy: 0.05, "cylinder diameter wrong")
    }

    /// A flat disc rotated 90° about X (nozzle / pump-island disc on its side):
    /// once object-space proportions are correct, the node rotation maps the
    /// disc cleanly. With the old unit-cylinder bug the rotation acted on a
    /// 1×1×1 cube and the result read as a sheared squat blob.
    func testRotatedFlatCylinderWorldExtent() throws {
        let cyl = SCNCylinder(radius: 4, height: 0.2)
        guard let tri = IlluminatoramaMesh.objectTriangles(scnGeometry: cyl, elementIndex: 0) else {
            return XCTFail("objectTriangles returned nil for SCNCylinder")
        }
        // 90° about X turns the disc's flat (Y) axis into Z.
        let rot = simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0)))
        let world = tri.positions.map { p -> SIMD3<Float> in
            let v = rot * SIMD4<Float>(p, 1); return SIMD3(v.x, v.y, v.z)
        }
        let b = bounds(world)
        let size = b.max - b.min
        print("CYL rotated 90°X -> expect size~(8,8,0.2)  world size=\(size)")
        XCTAssertEqual(size.x, 8,   accuracy: 0.05, "disc width lost under rotation")
        XCTAssertEqual(size.y, 8,   accuracy: 0.05, "disc depth lost under rotation")
        XCTAssertEqual(size.z, 0.2, accuracy: 0.01, "disc thinness lost under rotation")
    }

    /// The concrete gasStation primitives that drove the ~9-round chase: the
    /// 7×0.4×5 m canopy roof (collapsed → tiny canopy), the 0.52×1.62×0.34 m
    /// pump housing (collapsed → squat pumps), and the cylinder nozzle spout
    /// (sheared → "fat black blob"). All must now convert at authored size, so
    /// the next forecourt scene can feed real SCNBox/SCNCylinder through the
    /// converter instead of the unit-`.box` workaround.
    func testGasStationPrimitivesAtAuthoredSize() throws {
        func extent(_ g: SCNGeometry) -> SIMD3<Float> {
            guard let tri = IlluminatoramaMesh.objectTriangles(scnGeometry: g, elementIndex: 0)
            else { return .zero }
            let b = bounds(tri.positions); return b.max - b.min
        }
        let roof = SCNBox(width: 7, height: 0.4, length: 5, chamferRadius: 0)
        let rs = extent(roof)
        print("CANOPY ROOF size=\(rs)")
        XCTAssertEqual(rs.x, 7,   accuracy: 0.02)
        XCTAssertEqual(rs.y, 0.4, accuracy: 0.02)
        XCTAssertEqual(rs.z, 5,   accuracy: 0.02)

        let housing = SCNBox(width: 0.52, height: 1.62, length: 0.34, chamferRadius: 0.09)
        let hs = extent(housing)
        print("PUMP HOUSING size=\(hs)")
        XCTAssertEqual(hs.x, 0.52, accuracy: 0.02)
        XCTAssertEqual(hs.y, 1.62, accuracy: 0.02)
        XCTAssertEqual(hs.z, 0.34, accuracy: 0.02)
        // The tall axis must stay tall: this ratio is what "collapsed toward a
        // cube" destroyed (1 : 1 : 1 instead of ~1.5 : 4.8 : 1).
        XCTAssertGreaterThan(hs.y / hs.x, 2.5, "housing no longer reads as a tall pump")

        let nozzle = SCNCylinder(radius: 0.025, height: 0.16)
        let ns = extent(nozzle)
        print("NOZZLE size=\(ns)")
        XCTAssertEqual(ns.x, 0.05, accuracy: 0.005)
        XCTAssertEqual(ns.y, 0.16, accuracy: 0.005)
        XCTAssertEqual(ns.z, 0.05, accuracy: 0.005)
    }

    /// Regression guard: a hand-built mesh (vertex AABB == boundingBox) must be
    /// returned byte-identical — the correction is a strict no-op there.
    func testHandBuiltMeshUntouched() throws {
        let pts: [SCNVector3] = [SCNVector3(-2, 0, -3), SCNVector3(5, 0, -3),
                                 SCNVector3(5, 1, 7), SCNVector3(-2, 1, 7)]
        let src = SCNGeometrySource(vertices: pts)
        let elem = SCNGeometryElement(indices: [Int32]([0, 1, 2, 0, 2, 3]), primitiveType: .triangles)
        let custom = SCNGeometry(sources: [src], elements: [elem]) // winding-ok: test fixture flat quad, not shippable geometry
        guard let tri = IlluminatoramaMesh.objectTriangles(scnGeometry: custom, elementIndex: 0) else {
            return XCTFail("objectTriangles returned nil for hand-built mesh")
        }
        for (i, p) in tri.positions.enumerated() {
            XCTAssertEqual(p.x, Float(pts[i].x), accuracy: 0, "vertex \(i) x mutated")
            XCTAssertEqual(p.y, Float(pts[i].y), accuracy: 0, "vertex \(i) y mutated")
            XCTAssertEqual(p.z, Float(pts[i].z), accuracy: 0, "vertex \(i) z mutated")
        }
    }
}
