import Foundation
import AVFoundation
import CoreVideo
import Metal
import VisualizerCore

// One call to make a video: `renderVideo` walks a timeline, asks the host to pose the scene at each
// frame, takes a still of it (`renderStill`) and appends it to an H.264/HEVC file.
//
//     try renderer.renderVideo(.init(url: url, size: (1920, 1080), fps: 30, frameCount: 90,
//                                    still: .init(frames: 16))) { frame, t in
//         camera.orbit(at: t); renderer.camera = camera     // pose the scene for time `t`
//     }
//
// It is OFFLINE: every frame is settled like a photograph, so video is as clean as a still and the
// result does not depend on how fast the machine renders. A host with per-frame scene state passes
// its own `step` (see `renderStill`). Frame time is `frame / fps` seconds — the host owns the clock.

public struct IlluminatoramaVideoOptions {
    public enum Codec: Sendable { case h264, hevc }

    public var url: URL
    /// Output size in pixels; rounded DOWN to even (the codecs need it). nil ⇒ the renderer's current size.
    public var size: (width: Int, height: Int)?
    public var fps: Int
    public var frameCount: Int
    public var codec: Codec
    /// Average bitrate in bits/s; nil ⇒ ~0.2 bits per pixel per frame (visually clean for stills-quality frames).
    public var bitrate: Int?
    /// Per-frame still settings. `size` here is ignored — the video's size governs. Default: 8 frames, light jitter.
    public var still: IlluminatoramaStillOptions

    public init(url: URL, size: (width: Int, height: Int)? = nil, fps: Int = 30, frameCount: Int,
                codec: Codec = .h264, bitrate: Int? = nil,
                still: IlluminatoramaStillOptions = .init(frames: 8)) {
        self.url = url; self.size = size; self.fps = fps; self.frameCount = frameCount
        self.codec = codec; self.bitrate = bitrate; self.still = still
    }
}

public struct IlluminatoramaVideoResult {
    public let url: URL
    public let width: Int
    public let height: Int
    public let frames: Int
    public let seconds: Double
    public let milliseconds: Double
}

public enum IlluminatoramaVideoError: Error {
    case badOptions(String)
    case writerFailed(String)
    case frameFailed(Int)
}

public extension IlluminatoramaRenderer {

    /// Render `options.frameCount` frames and write them to `options.url` (replacing any file there).
    /// `pose(frameIndex, timeSeconds)` runs before each frame: move the camera, advance the host's
    /// clock, change the scene. Throws if the file can't be written or a frame can't be read back;
    /// a partly written file is removed.
    @MainActor
    @discardableResult
    func renderVideo(_ o: IlluminatoramaVideoOptions,
                     step: ((IlluminatoramaStillFrame) -> Bool)? = nil,
                     pose: (_ frame: Int, _ seconds: Double) -> Void) throws -> IlluminatoramaVideoResult {
        guard o.frameCount > 0, o.fps > 0 else { throw IlluminatoramaVideoError.badOptions("frameCount and fps must be > 0") }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let want = o.size ?? (outputWidth, outputHeight)
        let w = want.width - want.width % 2, h = want.height - want.height % 2
        guard w >= 2, h >= 2 else { throw IlluminatoramaVideoError.badOptions("size too small") }

        try? FileManager.default.removeItem(at: o.url)
        let writer = try AVAssetWriter(outputURL: o.url, fileType: .mov)
        let bits = o.bitrate ?? max(1_000_000, Int(0.2 * Double(w * h * o.fps)))
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: o.codec == .hevc ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: bits],
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        guard writer.canAdd(input) else { throw IlluminatoramaVideoError.writerFailed("cannot add video input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw IlluminatoramaVideoError.writerFailed(writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)

        func fail(_ e: Error) -> Error {
            input.markAsFinished(); writer.cancelWriting(); try? FileManager.default.removeItem(at: o.url)
            return e
        }

        var still = o.still
        still.outputSize = (w, h)
        for i in 0..<o.frameCount {
            pose(i, Double(i) / Double(o.fps))
            guard let shot = renderStill(still, step: step)?.frame,
                  shot.width == w, shot.height == h else { throw fail(IlluminatoramaVideoError.frameFailed(i)) }
            // Backpressure: the writer drains on its own queue; an offline loop just waits for room.
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            guard let pool = adaptor.pixelBufferPool else { throw fail(IlluminatoramaVideoError.writerFailed("no pixel buffer pool")) }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
            guard let pb else { throw fail(IlluminatoramaVideoError.writerFailed("pixel buffer allocation failed")) }
            CVPixelBufferLockBaseAddress(pb, [])
            let dst = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(pb)
            shot.bytes.withUnsafeBufferPointer { src in
                for y in 0..<h {
                    let s = src.baseAddress! + y * w * 4, d = dst + y * stride
                    if shot.isBGRA { memcpy(d, s, w * 4) }
                    else { for x in 0..<w { d[x*4] = s[x*4+2]; d[x*4+1] = s[x*4+1]; d[x*4+2] = s[x*4]; d[x*4+3] = s[x*4+3] } }
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, [])
            guard adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: CMTimeScale(o.fps))) else {
                throw fail(IlluminatoramaVideoError.writerFailed(writer.error?.localizedDescription ?? "append failed"))
            }
        }
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        guard writer.status == .completed else {
            try? FileManager.default.removeItem(at: o.url)
            throw IlluminatoramaVideoError.writerFailed(writer.error?.localizedDescription ?? "finish failed")
        }
        return IlluminatoramaVideoResult(
            url: o.url, width: w, height: h, frames: o.frameCount, seconds: Double(o.frameCount) / Double(o.fps),
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
    }
}
