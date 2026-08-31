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

    /// DH-0456 — a wood PANEL on an ADJUSTABLE (wall/floor/ceiling) surface declares a per-panel
    /// tone de-repeat so a metres-wide wall does not stamp one veneer figure across itself; the
    /// same panel PREBAKED on a furniture face stays a single continuous board.
    func testPanelWallGetsPerPanelDeRepeatButFurnitureVeneerStaysContinuous() {
        let wall = MaterialGenerator.wood(size: 128,
                                          params: WoodParams(species: .oak, layout: .panel),
                                          tiling: .baked(2.0))
        XCTAssertEqual(wall.patternCells, Vec2(1, 0),
                       "a panel's UV tile IS one panel — one cell per repeat re-tones each panel")
        XCTAssertGreaterThan(wall.patternJitter, 0, "the panel wall must actually vary")

        let veneer = MaterialGenerator.wood(size: 128,
                                            params: WoodParams(species: .oak, layout: .panel),
                                            tiling: .prebaked(1.0))
        XCTAssertEqual(veneer.patternCells, .zero,
                       "a prebaked furniture veneer face is one continuous board — no cells")
        XCTAssertEqual(veneer.patternJitter, 0)
    }

    /// The repeat census scores the DH-0456 defect as a number: a coherent panel wall wide enough
    /// to tile several times must break its period, and the break must be measurable.
    func testRepeatCensusFlagsAnUnbrokenStampAndCreditsTheDeRepeat() {
        let panel = MaterialGenerator.wood(size: 128,
                                           params: WoodParams(species: .oak, layout: .panel),
                                           tiling: .baked(2.0))
        let broken = MaterialRepeatCensus.audit(panel, runMeters: 1.0, surfaceWidthMeters: 4.0)
        XCTAssertGreaterThan(broken.repeatsAcross, MaterialRepeatCensus.visibleRepeatCount)
        XCTAssertFalse(broken.readsAsStampedRepeat, "the panel wall declares a de-repeat")
        XCTAssertGreaterThan(broken.periodRepeatResidual, 0,
                             "the per-panel tone step must be a real, measurable break")

        // A coherent slice with NO de-repeat declared is the defect, and the census must catch it.
        var flat = panel
        flat.patternCells = .zero
        flat.patternJitter = 0
        let stamped = MaterialRepeatCensus.audit(flat, runMeters: 1.0, surfaceWidthMeters: 4.0)
        XCTAssertTrue(stamped.readsAsStampedRepeat, "an unbroken coherent stamp must fail closed")
        XCTAssertEqual(stamped.periodRepeatResidual, 0, "nothing breaks the period")
    }
}
