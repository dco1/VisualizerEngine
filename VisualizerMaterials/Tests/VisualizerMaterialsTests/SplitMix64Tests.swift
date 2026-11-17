import XCTest
@testable import VisualizerMaterials

final class SplitMix64Tests: XCTestCase {
    func testKnownSequence() {
        // Reference SplitMix64 outputs for seed 0.
        var r = SplitMix64(seed: 0)
        XCTAssertEqual(r.next(), 0xE220_A839_7B1D_CDAF)
        XCTAssertEqual(r.next(), 0x6E78_9E6A_A1B9_65F4)
    }
    func testUnitRangeAndSameStreamAsNoiseMix() {
        var r = SplitMix64(seed: 42)
        for i in 1...100 {
            let u = r.nextUnit(); XCTAssertTrue(u >= 0 && u < 1)
            XCTAssertEqual(r.state, 42 &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15)
        }
        var a = SplitMix64(seed: 7)
        XCTAssertEqual(a.next(), Noise.mix(7))
    }
    func testFloatSmoothstep() {
        XCTAssertEqual(smoothstep(0 as Float, 1, 0.5), 0.5, accuracy: 1e-6)
        XCTAssertEqual(smoothstep(0 as Float, 1, 2), 1)
    }
}
