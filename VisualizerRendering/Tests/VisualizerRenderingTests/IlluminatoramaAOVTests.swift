import XCTest
@testable import VisualizerRendering

/// The AOV set is defined ONCE (Daydream DH-1304): the slice indices the kernels write live in
/// `Shaders/IlluminatoramaAOV.h`, and `IlluminatoramaAOV` mirrors them. This reads the header's
/// `#define kAOV<Name>` lines and fails if either side drifts — a slice renumbered in one place
/// would otherwise read one term's picture under another's name, silently.
final class IlluminatoramaAOVTests: XCTestCase {

    private func headerDefines() throws -> [String: Int] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaAOV.h")
        let text = try String(contentsOf: url, encoding: .utf8)
        var out: [String: Int] = [:]
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count >= 3, parts[0] == "#define", parts[1].hasPrefix("kAOV") else { continue }
            let value = parts[2].trimmingCharacters(in: CharacterSet(charactersIn: "u"))
            if let v = Int(value) { out[String(parts[1])] = v }
        }
        return out
    }

    func testSwiftSlicesMatchTheHeader() throws {
        let defs = try headerDefines()
        XCTAssertFalse(defs.isEmpty, "no kAOV defines found in IlluminatoramaAOV.h")
        for aov in IlluminatoramaAOV.allCases {
            guard let slice = aov.slice else { continue }
            let key = "kAOV" + aov.rawValue.prefix(1).uppercased() + aov.rawValue.dropFirst()
            XCTAssertEqual(defs[key], slice, "\(key): header \(String(describing: defs[key])) vs Swift \(slice)")
        }
        XCTAssertEqual(defs["kAOVCount"], IlluminatoramaAOV.sliceCount)
        XCTAssertEqual(defs["kAOVLightingCount"], IlluminatoramaAOV.lightingTerms.count)
        XCTAssertEqual(defs["kAOVBeauty"], IlluminatoramaAOV.beauty.slice)
    }

    func testLightingTermsAreTheSlicesBelowBeauty() {
        let slices = IlluminatoramaAOV.lightingTerms.compactMap(\.slice)
        XCTAssertEqual(slices, Array(0..<IlluminatoramaAOV.lightingTerms.count), "lighting terms must be slices 0..n-1 in order")
        XCTAssertEqual(IlluminatoramaAOV.beauty.slice, IlluminatoramaAOV.lightingTerms.count)
        XCTAssertEqual(IlluminatoramaAOV.sliceCount, IlluminatoramaAOV.lightingTerms.count + 1)
        let allSlices = IlluminatoramaAOV.allCases.compactMap(\.slice)
        XCTAssertEqual(Set(allSlices).count, allSlices.count, "two terms share a slice")
    }
}
