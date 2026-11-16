import XCTest
@testable import VisualizerMaterials

/// Item L's model: a block carved into a double-cutaway electric guitar, step by step (`GuitarCarve.steps()`).
final class GuitarCarveTests: XCTestCase {
    nonisolated(unsafe) static var cached: [GuitarStep]?
    private func steps() -> [GuitarStep] {
        if let s = Self.cached { return s }
        let s = GuitarCarve.steps(); Self.cached = s; return s
    }

    func testStepsAreNonEmptyNamedAndSolid() {
        let s = steps()
        XCTAssertEqual(s.count, 8)
        for (i, st) in s.enumerated() {
            XCTAssertFalse(st.mesh.isEmpty, "step \(i + 1) empty")
            XCTAssertFalse(st.title.isEmpty); XCTAssertFalse(st.tool.isEmpty)
            if st.isSolid { XCTAssertEqual(GuitarCarve.openEdges(st.mesh), 0, "step \(i + 1) \(st.title): open edges") }
            XCTAssertGreaterThan(st.mesh.signedVolume, 0, "step \(i + 1) wound inward")
        }
    }

    func testVolumeDropsThroughTheCarvingSteps() {
        let s = steps()
        var last = s[0].mesh.signedVolume
        XCTAssertEqual(last, 1.06 * 0.36 * 0.05, accuracy: 1e-9)
        for (i, st) in s.enumerated().dropFirst() where st.carves {
            let v = st.mesh.signedVolume
            XCTAssertLessThan(v, last, "step \(i + 1) \(st.title) did not remove material")
            last = v
        }
        XCTAssertEqual(s.filter(\.carves).count, 4, "outline, chamfer, neck, contour")
        XCTAssertLessThan(last, 0.5 * s[0].mesh.signedVolume, "most of the block is gone")
    }

    func testItIsAGuitar() throws {
        let b = try XCTUnwrap(steps()[7].mesh.bounds)
        XCTAssertGreaterThan(b.max.x - b.min.x, 1.0)          // body + neck + headstock ≈ 1.07 m
        XCTAssertGreaterThan(b.max.y - b.min.y, 0.34)         // body ≈ 36 cm wide (+ tuner buttons)
        XCTAssertLessThan(b.max.z - b.min.z, 0.1)
        // the neck is narrow: nothing but the neck, strings and head sits beyond the horns
        let beyond = steps()[3].mesh.positions.filter { $0.x > 0.7 && $0.x < 0.8 }
        XCTAssertFalse(beyond.isEmpty)
        XCTAssertLessThan(beyond.map { abs($0.y) }.max()!, 0.035)
        // the final mesh carries packed UVs in 0…1
        let m = steps()[7].mesh
        XCTAssertEqual(m.uvs.count, m.positions.count)
        XCTAssertTrue(m.uvs.allSatisfy { $0.x >= -1e-6 && $0.x <= 1 + 1e-6 && $0.y >= -1e-6 && $0.y <= 1 + 1e-6 })
    }

    func testToolsAreRecordedAndWriteExportsEveryStep() throws {
        let s = steps()
        let tools = s.map(\.tool).joined(separator: " | ")
        for needle in ["HalfEdgeMesh", "SDF", "booleanDifference", "SculptVolume", "Mesh3.loft", "ModelGraph", "ModifierStack", "UV"] {
            XCTAssertTrue(tools.contains(needle), "no step names \(needle)")
        }
        let folder = URL(fileURLWithPath: "/private/tmp/dhf-debug/guitar-carve")
        let urls = try GuitarCarve.write(to: folder, steps: s)
        XCTAssertEqual(urls.count, s.count + 1)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("steps.json"))) as? [[String: Any]]
        XCTAssertEqual(json?.count, s.count)
        XCTAssertEqual(json?.first?["tool"] as? String, s[0].tool)
        let back = try MeshIO.read(url: urls[7])
        XCTAssertEqual(back.meshes.first?.triangleCount, s[7].mesh.triangleCount)
    }
}
