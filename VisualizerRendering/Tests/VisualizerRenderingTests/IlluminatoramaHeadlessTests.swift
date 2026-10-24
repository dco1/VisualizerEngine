import XCTest
import simd
import Metal
import AVFoundation
@testable import VisualizerRendering


/// A scene with something in it — an EMPTY scene takes the blank-sky early-out and never produces a real frame.
@MainActor private func stageScene(_ r: IlluminatoramaRenderer) {
    let rig = IlluminatoramaStudio.productShot
    rig.apply(to: r)
    r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
    r.instances = rig.instances()
}

/// The built-in headless path: no window, render → read back → PNG.
@MainActor
final class IlluminatoramaHeadlessTests: XCTestCase {

    func testHeadlessFrameReadsBackAndWritesPNG() throws {
        let engine = SimEngine.shared
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: engine, width: 128, height: 96, camera: cam)
        stageScene(r)
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

/// `renderStill`: one call, settle + readback, and the renderer is left exactly as found.
@MainActor
final class IlluminatoramaStillTests: XCTestCase {

    func testStillSettlesReadsBackAndRestoresTheRenderer() throws {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 128, height: 96, camera: cam)
        stageScene(r)
        let before = (r.outputWidth, r.outputHeight, r.internalRenderScale, r.taaJitterPixels,
                      r.taaHistoryBlend, r.ssaoTemporalBlend, r.autoExposureEnabled)

        var kinds: [IlluminatoramaStillFrame] = []
        let still = try XCTUnwrap(r.renderStill(
            .init(frames: 4, jitterPixels: 1, outputSize: (160, 120), internalScale: 1.5, lockExposure: true),
            step: { kind in kinds.append(kind); return r.render(blocking: true) }))

        XCTAssertEqual(still.frame.width, 160)
        XCTAssertEqual(still.frame.height, 120)
        XCTAssertGreaterThan(Set(stride(from: 0, to: still.frame.bytes.count, by: 4).map { still.frame.bytes[$0] }).count, 2,
                             "the still is a flat colour — nothing was rendered")
        XCTAssertEqual(still.accumulatedFrames, 4)
        XCTAssertEqual(kinds.filter(\.accumulates).count, 4)
        XCTAssertEqual(kinds.last, .accumulate(4))
        XCTAssertEqual(still.internalScale, 1.5)

        let after = (r.outputWidth, r.outputHeight, r.internalRenderScale, r.taaJitterPixels,
                     r.taaHistoryBlend, r.ssaoTemporalBlend, r.autoExposureEnabled)
        XCTAssertEqual("\(before)", "\(after)", "renderStill left the renderer changed: ")
    }
}

/// `renderVideo`: one call writes a playable file with the requested frames, and the host's pose ran per frame.
@MainActor
final class IlluminatoramaVideoTests: XCTestCase {

    func testRenderVideoWritesAPlayableMovWithEveryFrame() async throws {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 128, height: 96, camera: cam)
        stageScene(r)
        let url = URL(fileURLWithPath: NSTemporaryDirectory() + "illuminatorama-video-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }

        var poses: [Double] = []
        let result = try r.renderVideo(.init(url: url, size: (129, 97), fps: 10, frameCount: 5, still: .init(frames: 2))) { _, t in
            poses.append(t)
        }
        XCTAssertEqual(poses, [0, 0.1, 0.2, 0.3, 0.4])
        XCTAssertEqual(result.width, 128, "odd sizes round down to even")
        XCTAssertEqual(result.height, 96)

        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let size = try await track.load(.naturalSize)
        XCTAssertEqual(Int(size.width), 128)
        let seconds = try await asset.load(.duration).seconds
        XCTAssertEqual(seconds, 0.5, accuracy: 0.05)
        XCTAssertEqual(r.outputWidth, 128, "the renderer is restored to its own size")
    }
}
