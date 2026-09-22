import XCTest
import simd
import VisualizerMaterials
@testable import VisualizerVegetation

/// The sown wildflower planting. The generator moved here from Daydream Home (405605f); its garden
/// contract is ported verbatim from `DaydreamCoreTests/WildflowerPlantingMeshTests` (iterating the
/// GARDEN mix's kinds, which is what that suite's `Kind.allCases` meant before the foothill-meadow
/// kinds existed), and the meadow mix + the uncapped instancing scatter are pinned beside it.
final class WildflowerPlantingMeshTests: XCTestCase {
    typealias W = WildflowerPlantingMesh
    /// The 4000 Sunset parkway — the plot this was built for (≈ 218 m²).
    static let parkway = [Vec2(-9, 7.3), Vec2(13, 7.3), Vec2(13, 17.2), Vec2(-9, 17.2)]
    static let bed = [Vec2(0, 0), Vec2(4, 0), Vec2(4, 3), Vec2(0, 3)]

    private func triangles(_ layers: [W.Layer]) -> Int { layers.reduce(0) { $0 + $1.mesh.triangleCount } }

    // MARK: garden (Daydream's contract)

    func testNoPlantingIsNoGeometry() {
        XCTAssertTrue(W.layers(polygon: [Vec2(0, 0), Vec2(1, 0)], groundHeight: { _ in 0 }).isEmpty)
    }

    func testEveryVariantIsRealGeometry() {
        for kind in W.Kind.allCases {
            let variants = W.library[kind] ?? []
            XCTAssertEqual(variants.count, W.variantCount(kind))
            for v in variants {
                XCTAssertGreaterThan(v.triangleCount, 0, "\(kind)")
                XCTAssertLessThan(v.triangleCount, 700, "\(kind): a sown plant is a few hundred triangles")
                for layer in v.layers {
                    XCTAssertEqual(layer.mesh.normals.count, layer.mesh.positions.count)
                    for p in layer.mesh.positions { XCTAssertTrue(p.x.isFinite && p.y.isFinite && p.z.isFinite) }
                    for n in layer.mesh.normals { XCTAssertEqual(len3(n), 1, accuracy: 1e-3) }
                    let b = layer.mesh.bounds!
                    XCTAssertGreaterThan(b.min.y, -0.05); XCTAssertLessThan(b.max.y, 0.9)
                    XCTAssertLessThan(max(abs(b.min.x), abs(b.max.x), abs(b.min.z), abs(b.max.z)), 0.6)
                }
            }
        }
        for kind in [W.Kind.poppy, .lupine, .goldfields, .popcorn, .blueDicks] {
            for v in W.library[kind] ?? [] {
                XCTAssertTrue(v.layers.contains { !$0.translucent && !W.foliageGreens.contains($0.color)
                                                  && $0.color != W.stemGreen }, "\(kind) has petals")
                XCTAssertTrue(v.layers.contains { $0.translucent }, "\(kind) has leaves")
            }
        }
    }

    func testSowingIsDeterministicAndInsideThePlot() {
        let a = W.plants(polygon: Self.parkway, density: 1, seed: 1)
        XCTAssertEqual(a, W.plants(polygon: Self.parkway, density: 1, seed: 1))
        XCTAssertGreaterThan(a.count, 500)
        for p in a { XCTAssertTrue(contains(polygon: Self.parkway, p.position)) }
        let counts = Dictionary(grouping: a, by: \.kind).mapValues(\.count)
        for kind in W.Mix.garden.kinds { XCTAssertGreaterThan(counts[kind] ?? 0, 0, "\(kind)") }
        for other in [W.Kind.lupine, .goldfields, .popcorn] { XCTAssertGreaterThan(counts[.poppy]!, counts[other]!) }
        XCTAssertNotEqual(a, W.plants(polygon: Self.parkway, density: 1, seed: 2), "the seed is read")
    }

    func testTheGardenNeverSowsAMeadowKind() {
        let a = W.plants(polygon: Self.parkway, density: 2, seed: 9)
        XCTAssertFalse(a.contains { $0.kind == .blueDicks || $0.kind == .wildOat })
        // The generalised entry at the garden defaults IS the legacy scatter, plant for plant.
        XCTAssertEqual(a, W.plants(polygon: Self.parkway, density: 2, seed: 9, mix: .garden,
                                   triangleCeiling: W.maxTriangles))
    }

    func testReshapingAnEdgeLeavesTheMiddleAlone() {
        let wider = [Vec2(0, 0), Vec2(6, 0), Vec2(6, 3), Vec2(0, 3)]
        let before = W.plants(polygon: Self.bed, density: 1, seed: 3)
        let after = W.plants(polygon: wider, density: 1, seed: 3)
        XCTAssertFalse(before.isEmpty)
        for p in before { XCTAssertTrue(after.contains(p), "a plant moved when the far edge did") }
    }

    func testDensityIsReadAndTheCeilingHolds() {
        let thin = W.plants(polygon: Self.bed, density: 0.5, seed: 1).count
        let thick = W.plants(polygon: Self.bed, density: 2, seed: 1).count
        XCTAssertGreaterThan(Double(thick), Double(thin) * 2.5)
        let field = [Vec2(0, 0), Vec2(70, 0), Vec2(70, 70), Vec2(0, 70)]
        let tris = triangles(W.layers(polygon: field, density: 2, groundHeight: { _ in 0 }))
        XCTAssertGreaterThan(tris, W.maxTriangles / 2)
        XCTAssertLessThan(Double(tris), Double(W.maxTriangles) * 1.15)
    }

    func testThePlantingStandsOnTheGroundItIsGiven() {
        let ground: (Vec2) -> Double = { 2.0 + $0.x * 0.15 }
        let layers = W.layers(polygon: Self.bed, groundHeight: ground)
        XCTAssertFalse(layers.isEmpty)
        for layer in layers {
            for p in layer.mesh.positions {
                let g = ground(Vec2(p.x, -p.z))
                XCTAssertGreaterThan(p.y, g - 0.20, "buried")
                XCTAssertLessThan(p.y, g + 1.0, "floating")
            }
        }
    }

    // MARK: foothill meadow + instancing

    /// Uncapped, the same world cell grows the same plant whichever window asks — two abutting
    /// tiles sown separately are exactly the one big tile sown at once. That is what lets a
    /// streamed field regenerate a tile it dropped and get back the plants it had.
    func testUncappedScatterIsTileIndependent() {
        func rect(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> [Vec2] {
            [Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y1), Vec2(x0, y1)]
        }
        let whole = W.plants(polygon: rect(-16, 32, 16, 48), density: 1.6, seed: 5, mix: .foothillMeadow, triangleCeiling: nil)
        let left = W.plants(polygon: rect(-16, 32, 0, 48), density: 1.6, seed: 5, mix: .foothillMeadow, triangleCeiling: nil)
        let right = W.plants(polygon: rect(0, 32, 16, 48), density: 1.6, seed: 5, mix: .foothillMeadow, triangleCeiling: nil)
        XCTAssertEqual(Set(whole.map(\.position)), Set((left + right).map(\.position)))
        XCTAssertEqual(whole.count, left.count + right.count, "no plant on the seam twice")
        // Uncapped means the ceiling is not read: a field-sized window is not thinned.
        let big = rect(0, 0, 60, 60)
        let n = W.plants(polygon: big, density: 2, seed: 1, mix: .foothillMeadow, triangleCeiling: nil).count
        XCTAssertEqual(Double(n), 3600 * W.plantsPerSquareMeter * 2, accuracy: 3600 * W.plantsPerSquareMeter * 2 * 0.05)
    }

    /// The meadow gathers into SWATHS: across a hillside there are stretches that are mostly poppy
    /// and stretches that are mostly blue dicks — not one even blend.
    func testMeadowGathersIntoOrangeAndPurpleSwaths() {
        var orangeCells = 0, purpleCells = 0
        for gy in 0 ..< 20 {
            for gx in 0 ..< 20 {
                let x0 = Double(gx) * 12, y0 = Double(gy) * 12
                let cell = [Vec2(x0, y0), Vec2(x0 + 4, y0), Vec2(x0 + 4, y0 + 4), Vec2(x0, y0 + 4)]
                let ps = W.plants(polygon: cell, density: 1, seed: 3, mix: .foothillMeadow, triangleCeiling: nil)
                let poppy = ps.filter { $0.kind == .poppy }.count, dicks = ps.filter { $0.kind == .blueDicks }.count
                if poppy > ps.count / 2 { orangeCells += 1 }
                if dicks > ps.count / 3 { purpleCells += 1 }
            }
        }
        XCTAssertGreaterThan(orangeCells, 20, "solid orange drifts exist")
        XCTAssertGreaterThan(purpleCells, 15, "solid purple drifts exist")
        let all = W.plants(polygon: [Vec2(0, 0), Vec2(80, 0), Vec2(80, 80), Vec2(0, 80)], density: 1, seed: 3,
                           mix: .foothillMeadow, triangleCeiling: nil)
        for kind in W.Mix.foothillMeadow.kinds {
            XCTAssertGreaterThan(all.filter { $0.kind == kind }.count, 0, "\(kind) is sown")
        }
    }

    /// The far-field colour comes from the same drift the plants do: where the sown plants are
    /// mostly poppy the mean albedo is orange (red ≫ blue), where mostly blue dicks it is purple.
    func testMeanAlbedoFollowsTheSownDrift() {
        var sawOrange = false, sawPurple = false
        for i in 0 ..< 400 {
            let p = Vec2(Double(i % 20) * 13, Double(i / 20) * 13)
            let w = Dictionary(uniqueKeysWithValues: W.weights(at: p, seed: 3, mix: .foothillMeadow))
            let total = w.values.reduce(0, +)
            let c = W.meanAlbedo(at: p, seed: 3, mix: .foothillMeadow)
            XCTAssertTrue(c.x.isFinite && c.x >= 0 && c.x <= 1 && c.z >= 0 && c.z <= 1)
            if w[.poppy]! / total > 0.5 { sawOrange = true; XCTAssertGreaterThan(c.x, c.z * 2) }
            // Purple reads as purple: blue over green where blue dicks are the majority, strongly where they dominate.
            if w[.blueDicks]! / total > 0.5 { sawPurple = true; XCTAssertGreaterThan(c.z, c.y) }
            if w[.blueDicks]! / total > 0.6 { XCTAssertGreaterThan(c.z, c.y * 1.4) }
        }
        XCTAssertTrue(sawOrange && sawPurple)
    }

    func testMeadowKindsCarryTheirSpeciesColour() {
        for v in W.library[.blueDicks]! {
            XCTAssertTrue(v.layers.contains { $0.color == W.blueDicksViolet && !$0.translucent }, "purple florets")
            // The heads ride on tall bare stems, clear above the poppy cups (≈0.45 m).
            XCTAssertGreaterThan(v.layers.compactMap { $0.mesh.bounds?.max.y }.max()!, 0.40)
        }
        for v in W.library[.wildOat]! {
            XCTAssertTrue(v.layers.contains { $0.color == W.oatStraw }, "straw spikelets")
            XCTAssertGreaterThan(v.layers.compactMap { $0.mesh.bounds?.max.y }.max()!, 0.55)
        }
    }
}
