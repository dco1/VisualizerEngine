import XCTest
@testable import VisualizerMaterials

final class FNV1aTests: XCTestCase {
    /// Published FNV-1a 64 test vectors — every migrated call site must keep producing these.
    func testKnownVectors() {
        XCTAssertEqual(FNV1a.hash(""), 0xcbf2_9ce4_8422_2325)
        XCTAssertEqual(FNV1a.hash("a"), 0xaf63_dc4c_8601_ec8c)
        XCTAssertEqual(FNV1a.hash("foobar"), 0x8594_4171_f739_67e8)
        XCTAssertEqual(FNV1a.hash(bytes: Array("foobar".utf8)), FNV1a.hash("foobar"))
    }

    func testWordFoldMatchesInlineLoop() {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v: UInt64 in [1, 2, 3] { h = (h ^ v) &* 0x100000001b3 }
        XCTAssertEqual([1, 2, 3].reduce(FNV1a.offsetBasis) { FNV1a.fold($1, into: $0) }, h)
    }
}
