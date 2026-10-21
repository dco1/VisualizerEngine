import XCTest
import simd
@testable import VisualizerRendering

/// The default product studio is data: these pin the properties each card exists for, so an edit that quietly breaks the
/// look (a card in frame, a floor that no longer touches y = 0, the gradient losing its ramp) fails as a number.
final class IlluminatoramaStudioTests: XCTestCase {
    private let rig = IlluminatoramaStudio.productShot

    func testFloorTopIsExactlyTheOrigin() {
        XCTAssertEqual(rig.floor.center.y + rig.floor.size.y / 2, 0, accuracy: 1e-6,
                       "a piece on y = 0 must rest ON the floor, not hover above it")
    }

    func testEveryCardStaysOutOfFrameOfAPieceAtTheOrigin() {
        for c in rig.cards {
            let nearest = c.center - c.size / 2
            let gap = max(abs(c.center.x) - c.size.x / 2, abs(c.center.z) - c.size.z / 2, c.center.y - c.size.y / 2)
            XCTAssertGreaterThan(gap, 0.8, "\(c.name) is \(gap) m from the origin: it would crowd a normal piece (\(nearest))")
        }
    }

    func testTheGradientBankIsARampAndTheFlagsAreDark() {
        let ramp = rig.cards.filter { $0.name.hasPrefix("gradient-") }.map(\.emission)
        XCTAssertEqual(ramp.count, 5)
        XCTAssertEqual(ramp, ramp.sorted(by: >), "bright → dark, so a flat face shows form as a gradient")
        XCTAssertGreaterThan(ramp[0] / ramp[4], 30)
        for flag in rig.cards.filter({ $0.name.hasSuffix("-flag") }) { XCTAssertEqual(flag.emission, 0) }
        XCTAssertGreaterThan(rig.cards.first { $0.name == "front-strip" }!.emission, 1, "the front strip must light")
    }

    func testInstancesAreTheFloorPlusEveryCardAndCarryTheirEmission() {
        let inst = rig.instances()
        XCTAssertEqual(inst.count, 1 + rig.cards.count)
        XCTAssertEqual(inst[0].data.emission, .zero, "the floor does not emit")
        for (c, i) in zip(rig.cards, inst.dropFirst()) {
            XCTAssertEqual(i.data.emission.x, c.emission, accuracy: 1e-6, c.name)
            XCTAssertEqual(i.data.modelMatrix.columns.3.x, c.center.x, accuracy: 1e-6)
            XCTAssertEqual(i.data.modelMatrix.columns.0.x, c.size.x, accuracy: 1e-6, "scale is the card's size")
        }
    }

    func testScaledKeepsTheAngularSizeAndTheRadiance() {
        let big = rig.scaled(by: 3)
        for (a, b) in zip(rig.cards, big.cards) {
            XCTAssertEqual(b.center, a.center * 3)
            XCTAssertEqual(b.size, a.size * 3)
            XCTAssertEqual(b.emission, a.emission, "emission is radiance: it does not change with scale")
        }
        XCTAssertEqual(big.floor.center.y + big.floor.size.y / 2, 0, accuracy: 1e-5, "still resting on y = 0")
    }

    func testTranslatedMovesTheWholeRigTogether() {
        let moved = rig.translated(by: SIMD3(3, 0, -3))
        for (a, b) in zip(rig.cards, moved.cards) { XCTAssertEqual(b.center, a.center + SIMD3(3, 0, -3)) }
        XCTAssertEqual(moved.floor.center.x, 3, accuracy: 1e-6)
        XCTAssertEqual(moved.floor.size, rig.floor.size)
    }

    func testTheRecommendedLookIsABlackStudioWithReflections() {
        XCTAssertTrue(rig.raytracedReflections, "metal is flat cream without them (DH-1444)")
        XCTAssertFalse(rig.sunEnabled)
        XCTAssertEqual(rig.whiteBalanceK, 6500)
    }
}
