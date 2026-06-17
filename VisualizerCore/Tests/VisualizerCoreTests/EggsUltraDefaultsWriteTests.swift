import XCTest
@testable import VisualizerCore

/// Verifies `SettingsDefaultsWriter` correctly rewrites the literal forms used
/// by `EggsUltraDefaults` — the `static var` mirror that backs Egg Roller
/// Ultra's "Save current values as new defaults". The fixture reproduces every
/// literal shape that file uses: enum-with-type-annotation (`.case`), plain
/// Double, negative Double, Int, and Bool. A miss here is the exact failure
/// that made Ultra's save look broken.
@MainActor
final class EggsUltraDefaultsWriteTests: XCTestCase {

    private let fixture = """
    @MainActor
    public enum EggsUltraDefaults {
        static var trackLayout: EggTrackLayout = .megalopolis
        static var railLightIntensity = 0.1200
        static var cameraTilt = -0.3319
        static var rimSpotlightCount = 20
        static var bulbsEnabled = true
        static func apply() {}
    }
    """

    func testRoundTripsEveryLiteralShape() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("EggsUltraDefaults-\(UUID().uuidString).swift")
        try fixture.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let export = """
        # Eggs settings
        trackLayout = chase
        railLightIntensity = 0.6400
        cameraTilt = 0.5000
        rimSpotlightCount = 42
        bulbsEnabled = false
        """

        try SettingsDefaultsWriter.write(filePath: url.path, exportText: export)
        let out = try String(contentsOf: url, encoding: .utf8)

        // Enum keeps `.case` form with the new rawValue.
        XCTAssertTrue(out.contains("static var trackLayout: EggTrackLayout = .chase"), out)
        // Plain + negative Double, Int, Bool all rewrite.
        XCTAssertTrue(out.contains("static var railLightIntensity = 0.6400"), out)
        XCTAssertTrue(out.contains("static var cameraTilt = 0.5000"), out)
        XCTAssertTrue(out.contains("static var rimSpotlightCount = 42"), out)
        XCTAssertTrue(out.contains("static var bulbsEnabled = false"), out)
        // No stray invalid literal (e.g. `.42` or bare `chase`).
        XCTAssertFalse(out.contains("= chase\n"), out)
    }

    /// A key with no matching `static var` must throw, not silently no-op —
    /// this is the guard that keeps `apply(to:)`/`exportText()` in sync.
    func testUnmatchedKeyThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("EggsUltraDefaults-\(UUID().uuidString).swift")
        try "enum X { static var a = 1.0 }".write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(
            try SettingsDefaultsWriter.write(filePath: url.path, exportText: "missingKey = 2.0")
        )
    }
}
