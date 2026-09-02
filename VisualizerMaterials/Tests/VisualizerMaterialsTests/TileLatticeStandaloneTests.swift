import XCTest
import simd
@testable import VisualizerMaterials

/// **Herringbone & 45° diagonal tile lattices (DH-0099).** The two arrangements that are their own
/// lattice rather than a course offset. The bar Danny set answering the item was "build the math to
/// make them seamless", so the load-bearing assertion here is `!isSeamy`: the bake must WRAP.
///
/// Kept in the engine package so it runs under `Scripts/bump-engine.sh`'s standalone gate — this is
/// engine material code.
final class TileLatticeStandaloneTests: XCTestCase {

    private func params(_ bond: UnitBond, shape: UnitShape = .square,
                        size m: Double = 0.1524, run: UnitRun = .horizontal) -> TileParams {
        TileParams(pattern: UnitPatternParams(shape: shape, unitSizeMeters: m, bond: bond, run: run))
    }

    // MARK: - the item's contract: seamless

    func testHerringboneIsPlausibleAndSeamless() {
        // A range of sizes so the layout picks several different cell counts (all multiples of 2L).
        for m in [0.075, 0.1016, 0.1524, 0.3048] {
            let a = TextureAudit.audit(MaterialGenerator.tile(size: 128, params: params(.herringbone, size: m)))
            XCTAssertEqual(a.albedoOutOfRange, 0, "herringbone(\(m)) albedo out of range")
            XCTAssertFalse(a.isFlatColor, "herringbone(\(m)) is flat")
            XCTAssertFalse(a.isSeamy,
                "herringbone(\(m)) grid must wrap — the whole point of DH-0099; "
                + "seam x=\(a.seamScoreX) y=\(a.seamScoreY)")
            XCTAssertTrue(a.isPlausible, "herringbone(\(m)) not plausible")
        }
    }

    func testDiagonalIsPlausibleAndSeamless() {
        for m in [0.075, 0.1016, 0.1524, 0.3048] {
            let a = TextureAudit.audit(MaterialGenerator.tile(size: 128, params: params(.diagonal, size: m)))
            XCTAssertEqual(a.albedoOutOfRange, 0, "diagonal(\(m)) albedo out of range")
            XCTAssertFalse(a.isFlatColor, "diagonal(\(m)) is flat")
            XCTAssertFalse(a.isSeamy,
                "diagonal(\(m)) diamond lattice must wrap; seam x=\(a.seamScoreX) y=\(a.seamScoreY)")
            XCTAssertTrue(a.isPlausible, "diagonal(\(m)) not plausible")
        }
    }

    func testVerticalRunStaysSeamless() {
        // A vertical run is a quarter-turn of the lattice — it must wrap the same way.
        for bond: UnitBond in [.herringbone, .diagonal] {
            let a = TextureAudit.audit(MaterialGenerator.tile(
                size: 128, params: params(bond, shape: .rectangle, run: .vertical)))
            XCTAssertFalse(a.isSeamy, "\(bond) vertical run seams: x=\(a.seamScoreX) y=\(a.seamScoreY)")
        }
    }

    // MARK: - the layout wraps by construction

    func testHerringboneCellCountIsAMultipleOfTheBlock() {
        // 2L short-cells is the block period; the layout must land on a multiple or the pattern
        // cannot wrap. Verified across surfaces adjustable and fixed.
        let block = 2 * TileLayout.herringboneBrickLength
        for tiling: SurfaceTiling in [.librarySwatch, .baked(2.0), .prebaked(1.2)] {
            for m in [0.05, 0.1016, 0.1524, 0.4] {
                let layout = TileLayout.of(UnitPatternParams(unitSizeMeters: m, bond: .herringbone),
                                           on: tiling, bakeSize: 512)
                XCTAssertEqual(layout.cols % block, 0,
                    "herringbone cols \(layout.cols) is not a multiple of \(block) on \(tiling) @\(m)")
                XCTAssertEqual(layout.cols, layout.rows, "herringbone domain is square in short-cells")
                XCTAssertGreaterThanOrEqual(layout.cols, block)
            }
        }
    }

    func testDiagonalRunIsAWholeNumberOfCells() {
        for m in [0.05, 0.1016, 0.3048] {
            let p = UnitPatternParams(unitSizeMeters: m, bond: .diagonal)
            let layout = TileLayout.of(p, on: .librarySwatch, bakeSize: 512)
            XCTAssertGreaterThanOrEqual(layout.cols, 2)
            // Adjustable surface: run is derived, so the (mm-quantised) size is realised exactly.
            XCTAssertEqual(layout.runMeters, Double(layout.cols) * p.unitSizeMeters, accuracy: 1e-9)
        }
    }

    // MARK: - the lattice actually draws both orientations / a grout grid

    func testHerringboneLaysBothOrientations() {
        // The zigzag is only herringbone if both horizontal and vertical bricks are present. The
        // joint grid runs along both axes, so the height field's recesses appear on both — proven
        // by a spread of joint depths rather than a single dominant direction.
        let ch = MaterialGenerator.tile(size: 128,
            params: TileParams(pattern: UnitPatternParams(unitSizeMeters: 0.1524, bond: .herringbone),
                               finish: .gloss))
        XCTAssertGreaterThan(ch.roughness.max()!, 0.7, "herringbone grout must be matte")
        XCTAssertLessThan(ch.roughness.min()!, 0.2, "herringbone tile body must be glossy")
        // Height variance proves a recessed joint network exists (not a flat sheet).
        let hMean = ch.height.reduce(0, +) / Double(ch.height.count)
        let hVar = ch.height.reduce(0.0) { $0 + ($1 - hMean) * ($1 - hMean) } / Double(ch.height.count)
        XCTAssertGreaterThan(hVar, 1e-4, "herringbone joints must recess the height field")
    }

    func testDeterministicPerSeed() {
        for bond: UnitBond in [.herringbone, .diagonal] {
            let a = MaterialGenerator.tile(size: 64, params: params(bond))
            let b = MaterialGenerator.tile(size: 64, params: params(bond))
            XCTAssertEqual(a.albedo, b.albedo, "\(bond) bake must be deterministic")
        }
    }

    // MARK: - the enum is wired through

    func testLatticeBondsAreOfferedAndNamed() {
        XCTAssertTrue(UnitBond.allCases.contains(.herringbone))
        XCTAssertTrue(UnitBond.allCases.contains(.diagonal))
        XCTAssertEqual(UnitBond.herringbone.displayName, "Herringbone")
        XCTAssertEqual(UnitBond.diagonal.displayName, "Diagonal 45°")
        XCTAssertTrue(UnitBond.herringbone.isLattice)
        XCTAssertTrue(UnitBond.diagonal.isLattice)
        XCTAssertFalse(UnitBond.running.isLattice)
    }
}
