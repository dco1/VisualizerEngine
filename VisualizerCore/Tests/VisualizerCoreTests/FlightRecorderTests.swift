import XCTest
@testable import VisualizerCore

/// The recorder is only worth having if it catches a real stall and survives to be read, so
/// these block the REAL main thread and read the file back.
@MainActor
final class FlightRecorderTests: XCTestCase {
    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("flight-\(UUID().uuidString)/flight.log")
    }

    func testStalledMainThreadIsLoggedWithItsStackAndRecovery() throws {
        let url = tempURL()
        var cfg = FlightRecorder.Configuration(url: url)
        cfg.interval = 0.1; cfg.stallThreshold = 0.3; cfg.resampleEvery = 0.5
        let rec = FlightRecorder.start(cfg)
        rec.setContext("unit-test stall")
        rec.addProbe("probe") { "42" }

        busyWaitOnMain(seconds: 1.6)                       // main cannot answer pings
        let settle = expectation(description: "recover"); settle.isInverted = true
        wait(for: [settle], timeout: 0.6)                  // run loop turns → ping answered
        rec.stop()

        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("main=STALLED"), text)
        XCTAssertTrue(text.contains("main-stack"), text)
        XCTAssertTrue(text.contains("busyWaitOnMain"), "stack should name the blocking frame:\n\(text)")
        XCTAssertTrue(text.contains("main recovered"), text)
        XCTAssertTrue(text.contains("ctx=\"unit-test stall\""), text)
        XCTAssertTrue(text.contains("probe=42"), text)
        XCTAssertTrue(text.hasSuffix("SESSION END\n"), text)
    }

    func testPreviousUncleanSessionIsReported() throws {
        let url = tempURL()
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "2026-01-01 00:00:00.000 hb dt=1.00s mem=10MB\n".write(to: url, atomically: true, encoding: .utf8)
        // `shared` is a process singleton; exercise the file-level check directly.
        XCTAssertEqual(FlightRecorderProbe.endedCleanly(url), false)
        try "2026-01-01 00:00:00.000 SESSION END\n".write(to: url, atomically: true, encoding: .utf8)
        XCTAssertEqual(FlightRecorderProbe.endedCleanly(url), true)
    }

    @inline(never)
    private func busyWaitOnMain(seconds: Double) {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end { _ = sqrt(Double.random(in: 1...2)) }
    }
}
