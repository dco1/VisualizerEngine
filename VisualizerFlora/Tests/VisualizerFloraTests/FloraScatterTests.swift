import XCTest
@testable import VisualizerFlora

final class FloraScatterTests: XCTestCase {
    private let flat: @Sendable (Double, Double) -> Double = { _, _ in 1.5 }
    private let full: @Sendable (Double, Double) -> Float = { _, _ in 1 }
    private let key = FloraCellKey(ix: 3, iz: -2)

    func testSameCellGrowsTheSamePlantsAndSitsOnTheGround() {
        let a = FloraScatter.scatter(key, density: 1, seed: 7, ground: flat, coverage: full)
        let b = FloraScatter.scatter(key, density: 1, seed: 7, ground: flat, coverage: full)
        XCTAssertGreaterThan(a.count, 1000, "a 16 m cell at natural density holds ~1800 plants")
        XCTAssertEqual(a.count, b.count)
        XCTAssertEqual(a.map(\.position), b.map(\.position))
        XCTAssertTrue(a.allSatisfy { abs($0.position.y - (1.5 - 0.035)) < 1e-9 }, "seated on the host's ground")
        let s = Double(FloraScatter.cellSize)
        XCTAssertTrue(a.allSatisfy { $0.position.x >= 3 * s && $0.position.x < 4 * s && $0.position.z >= -2 * s && $0.position.z < -1 * s },
                      "every plant belongs to its own cell")
    }

    func testCoverageClearsTheHostsSlabAndThinsItsEdge() {
        let none = FloraScatter.scatter(key, density: 1, seed: 7, ground: flat, coverage: { _, _ in 0 })
        XCTAssertTrue(none.isEmpty)
        let all = FloraScatter.scatter(key, density: 1, seed: 7, ground: flat, coverage: full).count
        let half = FloraScatter.scatter(key, density: 1, seed: 7, ground: flat, coverage: { _, _ in 0.5 }).count
        XCTAssertEqual(Double(half) / Double(all), 0.5, accuracy: 0.08)
    }

    func testMidCellKeepsFewerPlantsWithDistanceAndBuildsCards() {
        let ps = FloraScatter.scatter(key, density: 1, seed: 7, ground: flat, coverage: full)
        let near = FloraScatter.buildMid(cell: key, level: 0, placements: ps)
        let far = FloraScatter.buildMid(cell: key, level: 8, placements: ps)
        XCTAssertGreaterThan(near.triangleCount, far.triangleCount * 4, "farther levels keep far fewer plants")
        XCTAssertGreaterThan(far.triangleCount, 0)
        XCTAssertEqual(FloraScatter.buildMid(cell: key, level: 8, placements: ps).indices, far.indices, "deterministic")
        let lvlNear = FloraScatter.midLevel(30, radius: 240), lvlFar = FloraScatter.midLevel(200, radius: 240)
        XCTAssertLessThan(lvlNear ?? 99, lvlFar ?? -1)
        XCTAssertNil(FloraScatter.midLevel(250, radius: 240))
    }

    func testEveryLibraryVariantBakesBothLODs() {
        XCTAssertFalse(FloraLibrary.plants.isEmpty)
        for (_, p) in FloraLibrary.plants {
            XCTAssertFalse(p.full.vertices.isEmpty)
            XCTAssertLessThanOrEqual(p.lite.triangleCount, p.full.triangleCount + 1, "lite never costs more than full")
        }
    }
}
