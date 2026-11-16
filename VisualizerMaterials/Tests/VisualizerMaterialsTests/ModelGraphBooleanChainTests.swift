import XCTest
import simd
@testable import VisualizerMaterials

/// A boolean chain through `ModelGraph` must be as closed as the same chain called on `Mesh3` directly.
final class ModelGraphBooleanChainTests: XCTestCase {
    struct Defects { var byIndex = 0, byPosition = 0 }

    private func defects(_ m: Mesh3) -> Defects {
        func count(_ id: (Int) -> Int) -> Int {
            var c: [SIMD2<Int>: Int] = [:]
            var i = 0
            while i + 2 < m.indices.count {
                let t = (0 ..< 3).map { id(Int(m.indices[i + $0])) }
                i += 3
                for e in 0 ..< 3 where t[e] != t[(e + 1) % 3] {
                    c[SIMD2(Swift.min(t[e], t[(e + 1) % 3]), Swift.max(t[e], t[(e + 1) % 3]))
, default: 0] += 1
                }
            }
            return c.values.filter { $0 != 2 }.count
        }
        struct K: Hashable { let x: Int64, y: Int64, z: Int64 }
        var ids: [K: Int] = [:]
        let vid = m.positions.map { p -> Int in
            let k = K(x: Int64((p.x * 1e6).rounded()), y: Int64((p.y * 1e6).rounded()), z: Int64((p.z * 1e6).rounded()))
            if let i = ids[k] { return i }; ids[k] = ids.count; return ids.count - 1
        }
        return Defects(byIndex: count { $0 }, byPosition: count { vid[$0] })
    }

    private func box(_ lo: Vec3, _ hi: Vec3) -> Mesh3 { GraphPrimitives.box(size: hi - lo).placed(GraphTransform(translate: (lo + hi) * 0.5)) }

    /// Guitar step 6 in small: a body (cylinder-ish slab), a pocket, pickup slots, then plates unioned.
    private func steps() -> [(BooleanOp, Mesh3)] {
        var s: [(BooleanOp, Mesh3)] = [(.difference, box(Vec3(0.22, -0.105, 0.0215), Vec3(0.47, 0.085, 0.1)))]
        for x in [0.30, 0.38, 0.44] { s.append((.difference, box(Vec3(x - 0.036, -0.021, 0.011), Vec3(x + 0.036, 0.021, 0.1)))) }
        for x in [0.30, 0.38] { s.append((.union, box(Vec3(x - 0.033, -0.018, 0.0105), Vec3(x + 0.033, 0.018, 0.0335)))) }
        return s
    }
    private func body() -> Mesh3 {
        GraphPrimitives.cylinder(radius: 0.2, height: 0.04, segments: 48).placed(GraphTransform(translate: Vec3(0.35, 0, 0.02)))
    }

    func testChainThroughGraphMatchesDirect() throws {
        var direct = body()
        for (op, m) in steps() { direct = op == .union ? direct.booleanUnion(m) : direct.booleanDifference(m) }
        let g = GraphBuilder()
        var cur = g.importMesh(body())
        for (op, m) in steps() { let t = g.importMesh(m); cur = op == .union ? cur.union(t) : cur.subtracting(t) }
        let viaGraph = try g.build(output: cur).evaluateMesh()
        let dd = defects(direct), dg = defects(viaGraph)
        XCTAssertEqual(dd.byIndex, 0); XCTAssertEqual(dd.byPosition, 0)
        XCTAssertEqual(dg.byIndex, 0, "graph result open by index")
        XCTAssertEqual(dg.byPosition, 0, "graph result open by position")
        XCTAssertEqual(viaGraph.signedVolume, direct.signedVolume, accuracy: 1e-9)
    }

    /// The guitar's step 6 as one graph: the body and every tool imported, the booleans as nodes.
    func testGuitarStep6ChainThroughGraph() throws {
        let body = GuitarCarve.contour(GuitarCarve.carveNeck(GuitarCarve.roundEdges(GuitarCarve.carveOutline())))
        let bx = GuitarCarve.bridgeX, px = GuitarCarve.pickupX
        let b = GuitarCarve.boxMesh
        let g = GraphBuilder()
        func imp(_ m: Mesh3) -> MeshRef { g.importMesh(m) }
        var m = imp(body).subtracting(imp(b(Vec3(0.22, -0.105, 0.0215), Vec3(0.47, 0.085, 0.1))))
        var plate = imp(b(Vec3(0.2185, -0.1065, 0.0210), Vec3(0.4715, 0.0865, 0.0275)))
        for x in px {
            m = m.subtracting(imp(b(Vec3(x - 0.036, -0.021, 0.011), Vec3(x + 0.036, 0.021, 0.1))))
            plate = plate.subtracting(imp(b(Vec3(x - 0.038, -0.023, 0.0), Vec3(x + 0.038, 0.023, 0.1))))
        }
        m = m.union(plate)
        for x in px { m = m.union(imp(b(Vec3(x - 0.033, -0.018, 0.0105), Vec3(x + 0.033, 0.018, 0.0335)))) }
        m = m.union(imp(b(Vec3(bx - 0.006, -0.05, 0.0235), Vec3(bx + 0.006, 0.05, 0.0345))))
        m = m.union(imp(b(Vec3(bx - 0.070, -0.05, 0.0235), Vec3(bx - 0.052, 0.05, 0.0325))))
        let viaGraph = try g.build(output: m).evaluateMesh()

        var d = body.booleanDifference(b(Vec3(0.22, -0.105, 0.0215), Vec3(0.47, 0.085, 0.1)))
        var pl = b(Vec3(0.2185, -0.1065, 0.0210), Vec3(0.4715, 0.0865, 0.0275))
        for x in px {
            d = d.booleanDifference(b(Vec3(x - 0.036, -0.021, 0.011), Vec3(x + 0.036, 0.021, 0.1)))
            pl = pl.booleanDifference(b(Vec3(x - 0.038, -0.023, 0.0), Vec3(x + 0.038, 0.023, 0.1)))
        }
        d = d.booleanUnion(pl)
        for x in px { d = d.booleanUnion(b(Vec3(x - 0.033, -0.018, 0.0105), Vec3(x + 0.033, 0.018, 0.0335))) }
        d = d.booleanUnion(b(Vec3(bx - 0.006, -0.05, 0.0235), Vec3(bx + 0.006, 0.05, 0.0345)))
        d = d.booleanUnion(b(Vec3(bx - 0.070, -0.05, 0.0235), Vec3(bx - 0.052, 0.05, 0.0325)))

        XCTAssertEqual(try g.build(output: m).evaluateMesh(), viaGraph, "same graph, same mesh")
        let dd = defects(d), dg = defects(viaGraph)
        XCTAssertEqual(dd.byPosition, 0)
        XCTAssertEqual(dg.byIndex, 0, "graph result open by index")
        XCTAssertEqual(dg.byPosition, 0, "graph result open by position")
        XCTAssertEqual(viaGraph.signedVolume, d.signedVolume, accuracy: 1e-9)
    }
}
