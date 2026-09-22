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

/// `WoodParams.figure` — flat-sawn cathedral vs quarter-sawn straight grain. The knob must
/// persist (tolerant decode), change the bake, and leave the figure DEFAULT flat-sawn so a
/// document written before it existed keeps its arches.
final class WoodFigureTests: XCTestCase {
    func testFigureRoundTripsAndDefaultsFlatSawn() throws {
        let q = WoodParams(species: .cherry, layout: .panel, knots: false, figure: .quarterSawn)
        let data = try JSONEncoder().encode(q)
        XCTAssertEqual(try JSONDecoder().decode(WoodParams.self, from: data).figure, .quarterSawn)
        // A pre-figure document: no key → the default.
        let legacy = Data(#"{"species":"cherry","layout":"panel","boardWidthInches":12,"knots":false}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(WoodParams.self, from: legacy).figure, .flatSawn)
    }
    func testFlatSawnAndQuarterSawnBakeDifferentFaces() {
        let flat = MaterialGenerator.wood(size: 128, params: WoodParams(species: .cherry, layout: .panel, knots: false, figure: .flatSawn))
        let quarter = MaterialGenerator.wood(size: 128, params: WoodParams(species: .cherry, layout: .panel, knots: false, figure: .quarterSawn))
        var diff = 0.0
        for i in 0 ..< flat.albedo.count {
            let a = flat.albedo[i], b = quarter.albedo[i]
            diff += abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)
        }
        diff /= Double(flat.albedo.count)
        XCTAssertGreaterThan(diff, 0.01, "the figure knob must move the bake (mean |Δalbedo| \(diff))")
    }
    func testGumFlecksAreNotOnALattice() {
        // Cherry's flecks used to sit one per fbm base cell — the same (u, v) offsets in every
        // cell, a drilled-in-rows door. Measure: the fleck mask's row-profile (fraction of dark
        // texels per row) must not repeat at the old cell period. With scattered flecks the
        // correlation between the profile and itself shifted by a quarter-tile is weak.
        let ch = MaterialGenerator.wood(size: 256, params: WoodParams(species: .cherry, layout: .panel, knots: false))
        let n = 256
        var rowDark = [Double](repeating: 0, count: n)
        var lum = [Double](repeating: 0, count: n * n)
        for i in 0 ..< n * n { let a = ch.albedo[i]; lum[i] = 0.2126 * a.x + 0.7152 * a.y + 0.0722 * a.z }
        let mean = lum.reduce(0, +) / Double(n * n)
        for y in 0 ..< n { for x in 0 ..< n where lum[y * n + x] < mean * 0.72 { rowDark[y] += 1 } }
        let m = rowDark.reduce(0, +) / Double(n)
        func corr(_ shift: Int) -> Double {
            var num = 0.0, den = 0.0
            for y in 0 ..< n { let a = rowDark[y] - m, b = rowDark[(y + shift) % n] - m; num += a * b; den += a * a }
            return den > 0 ? num / den : 0
        }
        // The old lattice had 4 base cells per tile along v → a period of n/4 rows.
        XCTAssertLessThan(corr(n / 4), 0.5, "fleck rows repeat at the old quarter-tile period (corr \(corr(n / 4)))")
    }
}
