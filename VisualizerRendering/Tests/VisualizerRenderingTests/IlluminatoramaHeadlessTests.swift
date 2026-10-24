import XCTest
import simd
import Metal
import VisualizerStill
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

@MainActor
final class IlluminatoramaStillStreamingTests: XCTestCase {

    func testStreamingStillReportsProgressAndMatchesTheBlockingSequence() async throws {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 128, height: 96, camera: cam)
        stageScene(r)
        var seen: [Double] = []
        let result = await r.renderStillStreaming(.init(frames: 3, jitterPixels: 1, outputSize: (160, 120)),
                                                  progress: { seen.append($0) })
        let still = try XCTUnwrap(result)
        XCTAssertEqual(still.frame.width, 160)
        XCTAssertEqual(seen.count, 3)
        XCTAssertEqual(seen.last ?? 0, 1, accuracy: 1e-9)
        XCTAssertEqual(seen, seen.sorted())
        XCTAssertEqual(r.outputWidth, 128, "restored")
    }
}

@MainActor
final class IlluminatoramaSweepTests: XCTestCase {

    func testSweepPosesEachShotAndRestoresTheRenderer() throws {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1, 4), target: .zero, up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 3, aspect: 4.0 / 3.0, zNear: 0.05, zFar: 100)
        let r = try IlluminatoramaRenderer(engine: SimEngine.shared, width: 128, height: 96, camera: cam)
        stageScene(r)
        let halfLife = r.autoExposureHalfLife
        var posed: [Int] = []
        var accumulate: [Int] = []
        let shots = r.renderStillSweep(shots: 3, .init(frames: 2, outputSize: (160, 120)), leadInFrames: 2,
                                       autoExposureHalfLife: 0.02,
                                       step: { kind in if case .accumulate(let k) = kind { accumulate.append(k) }
                                                       return r.render(blocking: true) },
                                       pose: { posed.append($0) })
        XCTAssertEqual(posed, [0, 1, 2])
        XCTAssertEqual(shots.count, 3)
        XCTAssertTrue(shots.allSatisfy { $0.width == 160 && $0.height == 120 })
        // lead-in (2 frames) then 3 shots × accumulate 1,2 — each shot restarts at 1
        XCTAssertEqual(Array(accumulate.suffix(6)), [1, 2, 1, 2, 1, 2])
        XCTAssertEqual(r.autoExposureHalfLife, halfLife)
        XCTAssertEqual(r.outputWidth, 128)
    }
}

final class IlluminatoramaPhotoQualityOptionsTests: XCTestCase {
    func testPhotoQualityOptionsAreTheMeasuredTable() {
        let o = IlluminatoramaStillOptions.photoQuality(canvas: (1280, 720))
        XCTAssertEqual(o.frames, PhotoQuality.settleFrames)
        XCTAssertEqual(o.outputSize?.width, PhotoQuality.targetLongEdge)
        XCTAssertEqual(o.internalScale, PhotoQuality.scale(outputPixels: 3840 * 2160), "scale follows the OUTPUT size")
        XCTAssertEqual(StillAccumulation.blend(frame: 4, floor: 0.01), 0.25)
        XCTAssertEqual(StillAccumulation.blend(frame: 1000, floor: 0.01), 0.01)
    }
}
