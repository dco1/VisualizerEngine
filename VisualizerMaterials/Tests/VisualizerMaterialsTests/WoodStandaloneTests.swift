import XCTest
@testable import VisualizerMaterials

/// **The engine's own gate that a wood bakes standalone.**
///
/// Daydream Home's `WoodMaterialTests` is the detailed one and still runs there. This exists
/// because `Scripts/bump-engine.sh` builds and tests the engine BY ITSELF before pushing, and
/// a push that breaks Visualizer is the failure this repo cannot afford. Without a material
/// test in this target that standalone gate is blind to everything that just moved into it.
final class WoodStandaloneTests: XCTestCase {

    /// The headline capability: ask for a species, get boards.
    func testEverySpeciesBakesPlausibleBoards() {
        for species in WoodSpecies.allCases {
            let ch = MaterialGenerator.wood(size: 128, params: WoodParams(species: species))
            XCTAssertEqual(ch.category, .wood)
            XCTAssertEqual(ch.albedo.count, 128 * 128)
            XCTAssertNotNil(ch.grainTangent, "\(species) must carry a grain tangent — wood is anisotropic")
            let mean = ch.albedoMean
            XCTAssertGreaterThan(mean.x, mean.z, "\(species): wood is warm — red above blue")
            XCTAssertTrue((0.02...0.95).contains(mean.x), "\(species) albedo out of band: \(mean.x)")
        }
    }

    /// Knots are DECLARED to the renderer, never drawn into the tile. If this flips, a sparse
    /// landmark is back inside a repeating texture and every floor grids at the tile period.
    func testKnotsAreDeclaredNotBaked() {
        let knotted = MaterialGenerator.wood(size: 128, params: WoodParams(species: .oak, knots: true))
        let clear = MaterialGenerator.wood(size: 128, params: WoodParams(species: .oak, knots: false))
        XCTAssertNotNil(knotted.knots, "oak ships a knot field for the renderer")
        XCTAssertNil(clear.knots, "the knot toggle must actually clear the field")
        XCTAssertEqual(knotted.albedo, clear.albedo,
                       "the BAKE must be byte-identical either way — knots live in the shader")
    }

    /// The board comb the knot shader clips against. `sampleWoodKnots` needs it to know where
    /// the saw cuts are; a wood that stops declaring it lets knots bleed across plank seams.
    func testPlankedWoodDeclaresItsBoardComb() {
        let ch = MaterialGenerator.wood(size: 128,
                                        params: WoodParams(species: .oak, layout: .boards),
                                        tiling: .baked(2.0))
        XCTAssertGreaterThan(ch.patternCells.x, 0, "boards must declare their plank count")
        XCTAssertEqual(ch.patternCells.y, 0, "boards run unbroken down v — that axis is bonded")
    }
}
