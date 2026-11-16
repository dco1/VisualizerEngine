import XCTest
@testable import VisualizerMaterials

/// One object, modelled start to finish with the kit: a drawer pull. Half-edge extrude / inset / push-pull shape the
/// stem and cap, a robust boolean drills the screw bore, an SDF carve engraves a groove round the cap, a modifier stack
/// tapers and grains it, the UVs are unwrapped (LSCM islands, packed), and MeshIO writes the OBJ.
final class ModellingDemoTests: XCTestCase {
    func testModelADrawerPullWithTheWholeKit() throws {
        let closed = GuitarCarve.openEdges
        // 1. half-edge: a 6 × 4 × 2 cm plinth; extrude its top up into a stem, inset + push/pull a cap onto the stem
        var he = HalfEdgeMesh.box(size: Vec3(0.06, 0.04, 0.02))
        let top = he.faces.first { he.faceNormal($0).z > 0.9 }!
        let stem = he.extrude(faces: he.selection(.face, [top]), distance: 0.03)!                   // extrude
        let cap = he.inset(stem, thickness: 0.006, depth: 0)!                                       // inset (cap lip)
        he.pushPull(cap, distance: 0.008)                                                           // push/pull
        let base = he.toMesh3()
        XCTAssertGreaterThan(base.signedVolume, 0.06 * 0.04 * 0.02)
        XCTAssertEqual(closed(base), 0, "half-edge model closed")

        // 2. robust boolean: a screw bore up the middle, from the underside, stopping inside the cap
        let bore = Mesh3.cylinder(from: Vec3(0, 0, -0.011), to: Vec3(0, 0, 0.02), radius: 0.006)
        let drilled = base.booleanDifference(bore)
        XCTAssertEqual(closed(drilled), 0, "boolean result closed")
        XCTAssertLessThan(drilled.signedVolume, base.signedVolume)

        // 3. SDF carve: a groove sunk 1.5 mm into the stem by a torus, then re-meshed (surface nets, GPU if present)
        let groove = SDF.torus(major: 0.017, minor: 0.004).translated(by: Vec3(0, 0, 0.032))
        let carved = SDF.mesh(drilled).carved(tool: groove, depth: 0.0015)
        let b = try XCTUnwrap(drilled.bounds)
        let engraved = carved.mesh(lo: b.min - Vec3(repeating: 0.0103), hi: b.max + Vec3(repeating: 0.0103), cell: 0.0025)
        XCTAssertEqual(closed(engraved), 0, "SDF carve closed")
        XCTAssertFalse(engraved.isEmpty)

        // 4. modifier stack: taper the whole thing a touch, then a fine grain
        var stack = ModifierStack()
        _ = stack.add(SimpleDeformModifier(.taper(factor: -0.15), origin: Vec3(0, 0, -0.01), axis: Vec3(0, 0, 1), lower: 0, upper: 0.07), name: "Taper")
        _ = stack.add(DisplaceModifier.noise(strength: 0.0004, frequency: 150, octaves: 2, seed: 3), name: "Grain")
        let styled = stack.evaluate(engraved)
        XCTAssertEqual(closed(styled), 0, "modifiers keep it closed")

        // 5. UV unwrap: seams at sharp angles, LSCM per island, packed into 0…1
        let small = styled.decimated(targetTriangles: 2500)
        let seams = UVTopology(small).seams(angleDegrees: 60)
        let unwrapped = small.unwrappedUVs(seams: seams)
        XCTAssertEqual(closed(unwrapped.mesh), 0, "unwrapping splits UVs, not positions")
        XCTAssertFalse(unwrapped.layouts.isEmpty)
        XCTAssertTrue(unwrapped.mesh.uvs.allSatisfy { $0.x >= -1e-6 && $0.x <= 1 + 1e-6 && $0.y >= -1e-6 && $0.y <= 1 + 1e-6 })

        // 6. MeshIO: write the OBJ (this is the file Danny can open), read it back
        let url = URL(fileURLWithPath: "/private/tmp/dhf-debug/modelling-kit/demo.obj")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try MeshIO.write([unwrapped.mesh], to: url)
        let back = try MeshIO.read(url: url)
        XCTAssertEqual(back.meshes.first?.triangleCount, unwrapped.mesh.triangleCount)
        XCTAssertEqual(closed(try XCTUnwrap(back.meshes.first)), 0, "the OBJ on disk is closed")
        XCTAssertGreaterThan(try XCTUnwrap(FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int), 10_000)
    }
}
