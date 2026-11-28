import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Bloom off skips the pyramid (`bloomSkipped`): the frame must be the one the pyramid would have drawn at a vanishing
/// intensity, and turning bloom back on in the same renderer must bring it back.
@MainActor
final class IlluminatoramaBloomSkipTests: XCTestCase {

    private func renderer() throws -> IlluminatoramaRenderer {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 128, height: 96, camera: cam)
        let rig = IlluminatoramaStudio.productShot
        rig.apply(to: r)
        r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
        r.instances = rig.instances()
        r.pointLights = [IlluminatoramaPointLight(position: SIMD3(0.6, 0.9, 0.9), radius: 6, color: SIMD3(40, 30, 20))]   // hot: something to bloom
        r.taaJitterPixels = 0
        r.blockingLightingCompile = true
        r.bloomThreshold = 0.5
        return r
    }

    private func worst(_ a: IlluminatoramaFrame, _ b: IlluminatoramaFrame) -> Int {
        var w = 0
        for i in 0..<min(a.bytes.count, b.bytes.count) { w = max(w, abs(Int(a.bytes[i]) - Int(b.bytes[i]))) }
        return w
    }

    func testBloomOffSkipsThePyramidAndDrawsTheSameFrame() throws {
        let off = try renderer(); off.bloomIntensity = 0
        let offFrame = try XCTUnwrap(off.renderHeadless(frames: 3))
        XCTAssertTrue(off.bloomSkipped)
        XCTAssertNotNil(off.pfx.bloomOffTexture, "the tonemap read the black texel")

        let faint = try renderer(); faint.bloomIntensity = 1e-7
        let faintFrame = try XCTUnwrap(faint.renderHeadless(frames: 3))
        XCTAssertFalse(faint.bloomSkipped)
        XCTAssertLessThanOrEqual(worst(offFrame, faintFrame), 1, "skipping bloom changed the frame")

        off.bloomIntensity = 0.6                       // back on, same renderer: the pyramid runs again and shows
        let onFrame = try XCTUnwrap(off.renderHeadless(frames: 3))
        XCTAssertFalse(off.bloomSkipped)
        XCTAssertGreaterThan(worst(onFrame, offFrame), 4, "bloom back on must glow")
    }
}
