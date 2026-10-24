import XCTest
import simd
import Metal
@testable import VisualizerRendering

/// The built-in headless path: no window, render → read back → PNG.
@MainActor
final class IlluminatoramaHeadlessTests: XCTestCase {

    func testHeadlessFrameReadsBackAndWritesPNG() throws {
        let engine = SimEngine.shared
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: engine, width: 128, height: 96, camera: cam)
        let frame = try XCTUnwrap(r.renderHeadless(frames: 2), "headless readback returned nil")
        XCTAssertEqual(frame.width, r.outputTexture.width)
        XCTAssertEqual(frame.height, r.outputTexture.height)
        XCTAssertEqual(frame.bytes.count, frame.width * frame.height * 4)

        let path = NSTemporaryDirectory() + "illuminatorama-headless-\(UUID().uuidString).png"
        defer { try? FileManager.default.removeItem(atPath: path) }
        XCTAssertNotNil(r.renderHeadlessPNG(to: path))
        XCTAssertGreaterThan((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0, 0)
    }
}
