import XCTest
import simd
@testable import VisualizerMaterials

final class ModelGraphTests: XCTestCase {
    /// Edge defects of an index-welded-by-position mesh: (boundary, non-manifold).
    private func edgeDefects(_ m: Mesh3) -> (boundary: Int, nonManifold: Int) {
        struct K: Hashable { let x: Int64, y: Int64, z: Int64 }
        func k(_ p: Vec3) -> K { K(x: Int64((p.x * 1e6).rounded()), y: Int64((p.y * 1e6).rounded()), z: Int64((p.z * 1e6).rounded())) }
        var ids: [K: Int] = [:]
        let vid = m.positions.map { p -> Int in let kk = k(p); if let i = ids[kk] { return i }; ids[kk] = ids.count; return ids.count - 1 }
        var count: [SIMD2<Int>: Int] = [:]
        var i = 0
        while i + 2 < m.indices.count {
            let t = (0 ..< 3).map { vid[Int(m.indices[i + $0])] }
            i += 3
            for e in 0 ..< 3 {
                let a = t[e], b = t[(e + 1) % 3]
                if a != b { count[SIMD2(Swift.min(a, b), Swift.max(a, b)), default: 0] += 1 }
            }
        }
        return (count.values.filter { $0 == 1 }.count, count.values.filter { $0 > 2 }.count)
    }

    private func boxMinusCylinder(radius: Double = 0.5) -> (ModelGraph, Int, Int) {
        let g = GraphBuilder()
        let box = g.box(size: Vec3(2, 2, 2))
        let cyl = g.cylinder(radius: radius, height: 4, segments: 48)
        let out = box.subtracting(cyl)
        return (g.build(output: out), cyl.id, out.id)
    }

    func testBoxMinusCylinderVolume() throws {
        let (graph, _, _) = boxMinusCylinder()
        let m = try graph.evaluateMesh()
        let n = 48.0, r = 0.5
        let expected = 8 - (n / 2 * r * r * sin(2 * .pi / n)) * 2        // polygon cylinder's area × the box's 2 m
        XCTAssertEqual(m.signedVolume, expected, accuracy: 1e-6)
        let d = edgeDefects(m)
        XCTAssertEqual(d.boundary, 0); XCTAssertEqual(d.nonManifold, 0)
    }

    func testBSPEngineSelectable() throws {
        let g = GraphBuilder()
        let out = g.box(size: Vec3(2, 2, 2)).subtracting(g.cylinder(radius: 0.5, height: 4, segments: 24), engine: "bsp")
        let m = try g.build(output: out).evaluateMesh()
        XCTAssertEqual(m.signedVolume, 8 - (24.0 / 2 * 0.25 * sin(2 * .pi / 24)) * 2, accuracy: 1e-3)
    }

    func testCachingRecomputesOnlyDownstream() throws {
        let g = GraphBuilder()
        let box = g.box(size: Vec3(2, 2, 2))
        let cyl = g.cylinder(radius: 0.5, height: 4, segments: 32)
        let cut = box.subtracting(cyl)
        let moved = cut.transformed(translate: Vec3(1, 0, 0))
        let other = g.sphere(radius: 0.3)                 // not connected to the output
        _ = other
        var graph = g.build(output: moved)
        let cache = GraphCache()

        let first = try graph.evaluateMesh(cache: cache)
        XCTAssertEqual(cache.computeCount, 4, "box, cylinder, boolean, transform")
        cache.resetCounters()

        _ = try graph.evaluateMesh(cache: cache)
        XCTAssertEqual(cache.computeCount, 0, "nothing changed: all cache hits")

        graph.setParam(cyl.id, "radius", 0.6)             // cylinder, boolean, transform recompute; the box does not
        let second = try graph.evaluateMesh(cache: cache)
        XCTAssertEqual(cache.computed.map(\.kind), [.cylinder, .boolean, .meshTransform])
        XCTAssertNotEqual(first, second)
        cache.resetCounters()

        graph.setParam(moved.id, "translate", .vec3(Vec3(2, 0, 0)))   // only the last node
        _ = try graph.evaluateMesh(cache: cache)
        XCTAssertEqual(cache.computed.map(\.kind), [.meshTransform])
        cache.resetCounters()

        graph.setParam(cyl.id, "radius", 0.5)             // back to the first state: served from the cache for the boolean? no: key differs from the last run, equals the first
        _ = try graph.evaluateMesh(cache: cache)
        XCTAssertEqual(cache.computed.map(\.kind), [.meshTransform], "cylinder and boolean of the first state are still cached")
    }

    func testCycleIsReported() throws {
        var g = ModelGraph()
        let a = g.add(.meshTransform), b = g.add(.meshTransform), c = g.add(.meshTransform)
        g.connect(b, to: a, input: "mesh"); g.connect(c, to: b, input: "mesh"); g.connect(a, to: c, input: "mesh")
        g.output = a
        XCTAssertThrowsError(try g.evaluate()) { e in
            guard case GraphError.cycle(let path) = e else { return XCTFail("\(e)") }
            XCTAssertEqual(path.first, path.last)
            XCTAssertEqual(Set(path), [a, b, c])
            XCTAssertTrue("\(e)".contains("cycle"))
        }
    }

    func testTypeAndMissingInputErrors() throws {
        var g = ModelGraph()
        let s = g.add(.sdfSphere)
        let bevel = g.add(.bevel, inputs: ["mesh": s])
        XCTAssertThrowsError(try g.validate()) { XCTAssertEqual($0 as? GraphError, .typeMismatch(node: bevel, input: "mesh", expected: .mesh, got: .sdf)) }
        var h = ModelGraph()
        let bx = h.add(.box), b = h.add(.boolean, inputs: ["a": bx])
        XCTAssertThrowsError(try h.validate()) { XCTAssertEqual($0 as? GraphError, .missingInput(node: b, kind: "boolean", input: "b")) }
    }

    func testJSONRoundTrip() throws {
        let g = GraphBuilder()
        let part = g.box(size: Vec3(2, 2, 2)).subtracting(g.cylinder(radius: 0.5, height: 4)).beveled(radius: 0.02, segments: 2)
        let sdf = g.sdfBox(size: Vec3(1, 1, 1)).carved(tool: g.sdfSphere(radius: 0.3).transformed(translate: Vec3(0, 0.5, 0)), depth: 0.1)
            .meshed(cell: 0.05, backend: "cpu")
        let imp = g.importMesh(MeshKit.icosphere(radius: 0.4, levels: 1))
        let joined = g.join([part, sdf.transformed(translate: Vec3(4, 0, 0)), imp.modified("subdivide", ["levels": .number(1)])])
        let graph = g.build(output: joined)

        let data = try graph.jsonData()
        let back = try ModelGraph(json: data)
        XCTAssertEqual(graph, back)
        XCTAssertEqual(try back.jsonData(), data, "stable bytes")
        for n in graph.nodes {
            let x = try graph.evaluate(n.id), y = try graph.evaluate(n.id), z = try back.evaluate(n.id)
            XCTAssertEqual(x.mesh, y.mesh, "re-evaluation of \(n.kind) #\(n.id) is deterministic")
            XCTAssertEqual(x.mesh, z.mesh, "\(n.kind) #\(n.id) after the round trip")
        }
        XCTAssertEqual(try graph.evaluateMesh(), try back.evaluateMesh(), "identical mesh")
    }

    func testSDFCarveMeshIsWatertightAndStaysSDF() throws {
        let g = GraphBuilder()
        let block = g.sdfBox(size: Vec3(1, 1, 1))
        let tool = g.sdfSphere(radius: 0.25).transformed(translate: Vec3(0, 0.5, 0))
        let out = block.carved(tool: tool, depth: 0.12, falloff: 0.02).shell(thickness: 0.05).meshed(cell: 0.025)
        let graph = g.build(output: out)
        let cache = GraphCache()
        let m = try graph.evaluateMesh(cache: cache)
        XCTAssertGreaterThan(m.triangleCount, 1000)
        let d = edgeDefects(m)
        XCTAssertEqual(d.boundary, 0); XCTAssertEqual(d.nonManifold, 0)
        // Every SDF node only built a tree; the mesher ran once (one mesh-producing node).
        let sdfKinds: Set<NodeKind> = [.sdfBox, .sdfSphere, .sdfTransform, .sdfCarve, .sdfShell]
        XCTAssertEqual(Set(cache.computed.map(\.kind)), sdfKinds.union([.sdfToMesh]))
        XCTAssertEqual(cache.computed.filter { $0.kind == .sdfToMesh }.count, 1)
        // The carve took material off the top: less volume than the plain shell.
        let h = GraphBuilder()
        let solid = h.sdfBox(size: Vec3(1, 1, 1))
        let tool2 = h.sdfSphere(radius: 0.25).transformed(translate: Vec3(0, 0.5, 0))
        let plain = try h.build(output: solid.meshed(cell: 0.025)).evaluateMesh()
        let carved = try h.build(output: solid.carved(tool: tool2, depth: 0.12, falloff: 0.02).meshed(cell: 0.025)).evaluateMesh()
        XCTAssertLessThan(carved.signedVolume, plain.signedVolume - 0.01)
    }

    func testSDFCPUAndAutoBackendsAgree() throws {
        let g = GraphBuilder()
        let s = g.sdfSphere(radius: 0.4).union(g.sdfBox(size: Vec3(0.5, 0.5, 0.5)), smooth: 0.1)
        let cpu = try g.build(output: s.meshed(cell: 0.05, backend: "cpu")).evaluateMesh()
        let auto = try g.build(output: s.meshed(cell: 0.05, backend: "auto")).evaluateMesh()
        XCTAssertEqual(cpu.triangleCount, auto.triangleCount)
        XCTAssertEqual(cpu.signedVolume, auto.signedVolume, accuracy: 1e-3)
    }

    func testInstanceOnPointsCountAndSeed() throws {
        func build(seed: Int, count: Int = 50) -> ModelGraph {
            let g = GraphBuilder()
            let ground = g.box(size: Vec3(4, 0.2, 4))
            let pts = g.distribute(on: ground, count: count, seed: seed)
            let peg = g.box(size: Vec3(0.1, 0.1, 0.1))
            return g.build(output: g.instance(peg, on: pts, scale: 0.5 ... 1.5, rotateJitter: Vec3(10, 180, 10), alignToNormal: true, seed: seed))
        }
        let a = try build(seed: 7).evaluateMesh()
        XCTAssertEqual(a.triangleCount, 50 * 12)
        XCTAssertEqual(a, try build(seed: 7).evaluateMesh(), "same seed, same mesh")
        XCTAssertNotEqual(a, try build(seed: 8).evaluateMesh(), "other seed, other mesh")
        // The points sit on the ground box's surface.
        let g = GraphBuilder()
        let pts = g.distribute(on: g.box(size: Vec3(4, 0.2, 4)), count: 200, seed: 3)
        let v = try g.build(output: pts).evaluate()
        let p = try XCTUnwrap(v.points)
        XCTAssertEqual(p.count, 200)
        for q in p {
            let d = simd_abs(q.position) - Vec3(2, 0.1, 2)
            XCTAssertLessThan(abs(Swift.max(d.x, Swift.max(d.y, d.z))), 1e-9)
        }
    }

    func testPrimitivesAreClosedAndOutward() throws {
        let g = GraphBuilder()
        let cases: [(MeshRef, Double)] = [(g.box(size: Vec3(1, 2, 3)), 6), (g.cylinder(radius: 0.5, height: 1), .pi * 0.25),
                                          (g.sphere(radius: 0.5), 4.0 / 3 * .pi * 0.125), (g.cone(radius: 0.5, height: 1), .pi * 0.25 / 3),
                                          (g.torus(major: 1, minor: 0.25), 2 * .pi * .pi * 1 * 0.0625)]
        for (r, vol) in cases {
            let m = try g.build(output: r).evaluateMesh()
            XCTAssertEqual(edgeDefects(m).boundary, 0, "\(r.id)")
            XCTAssertEqual(m.signedVolume, vol, accuracy: vol * 0.04, "\(r.id)")
        }
    }

    func testOperatorsRun() throws {
        let g = GraphBuilder()
        let base = g.box(size: Vec3(1, 1, 1))
        let outs: [MeshRef] = [base.mirrored("x", at: 1), base.arrayed(count: 3, offset: Vec3(2, 0, 0)), base.radialArrayed(count: 4, around: Vec3(3, 0, 0)),
                               base.beveled(radius: 0.05), base.subdivided(levels: 1), g.sphere().decimated(ratio: 0.5),
                               base.modified("solidify", ["thickness": .number(0.05)]), base.modified("simpleDeform", ["mode": .string("twist"), "amount": .number(0.5)]),
                               base.uvs("planar", scale: 2), base.transformed(by: g.transform(translate: Vec3(0, 1, 0)))]
        for o in outs { XCTAssertFalse(try g.build(output: o).evaluateMesh().isEmpty, "\(o.id)") }
        XCTAssertThrowsError(try g.build(output: base.modified("nope")).evaluate())
        // a scalar node drives a parameter
        let r = g.scalar(0.25)
        let cyl = g.cylinder(radius: 9, height: 1).bind("radius", to: r)
        let m = try g.build(output: cyl).evaluateMesh()
        XCTAssertEqual(m.bounds!.max.x, 0.25, accuracy: 1e-9)
    }
}
