import Foundation
import Metal
import VisualizerCore

// One call to take a STILL: settle the scene to a converged picture and hand back the pixels.
//
//     let still = renderer.renderStill(.init(frames: 32, jitterPixels: 1, outputSize: (3840, 2400),
//                                            internalScale: 2))
//
// The mechanism lives here so every host (Daydream Home's export, Visualizer, a script) gets the
// same sequence; the NUMBERS are the host's — what a still is worth for a given product is a
// measured table (Daydream's `PhotoQuality`), passed in as options. The engine never decides them.
//
// The sequence, and why each step is shaped the way it is:
//   1. enter the still scope — output size, internal (supersample) scale, SVGF, reconstruction
//      filter, optional exposure lock — remembering every value so the scope restores them on
//      EVERY exit path;
//   2. warm-up frames (RT-sun TLAS build) and radiance-cache build frames, run with accumulation
//      OFF (jitter 0, blend 1) so they are held out of every running mean;
//   3. `frames` accumulated frames: sub-pixel jitter, and a CUMULATIVE AVERAGE `1/k` TAA blend
//      (not a fixed EMA — all k frames weigh equally), AO opened to the same floor;
//   4. read the output back.

/// What kind of frame the still is about to render. A host whose own frame step maintains
/// per-frame state (a scene bridge syncing sky and lights) reads this to hold build frames out.
public enum IlluminatoramaStillFrame: Sendable, Equatable {
    /// RT-sun soft-shadow TLAS warm-up — shaded against a structure still being built.
    case warmup
    /// Radiance-cache build frame — a full-walk build that must not be averaged into the picture.
    case cacheBuild
    /// The k-th (1-based) accumulated frame.
    case accumulate(Int)

    public var accumulates: Bool { if case .accumulate = self { return true } else { return false } }
}

public struct IlluminatoramaStillOptions {
    /// Accumulated frames (≥ 1).
    public var frames: Int
    /// Sub-pixel jitter width in internal pixels. 0 ⇒ accumulate identical frames (no supersampling gain).
    public var jitterPixels: Float
    /// Smallest per-frame TAA / AO weight — the resolve kernel's own floor (`clamp(alpha, 0.01, 1)`).
    public var accumulationFloor: Float
    /// Output size to resize to for the still; nil ⇒ keep the renderer's current size.
    public var outputSize: (width: Int, height: Int)?
    /// Internal render scale (supersample); nil ⇒ keep the renderer's.
    public var internalScale: Float?
    /// SVGF à-trous levels for the RT-GI denoise; nil ⇒ leave the denoiser alone.
    public var svgfLevels: Int?
    /// Reconstruction filter for the still; nil ⇒ leave as is.
    public var reconstructionFilter: IlluminatoramaReconstructionFilter?
    /// Pin auto-exposure for the still (a grade with a fixed exposure).
    public var lockExposure: Bool
    /// Frames to warm the RT-sun TLAS with when soft sun shadows are on.
    public var sunWarmupFrames: Int
    /// Cache build frames; nil ⇒ the renderer's own (`rtPathRadianceCacheBuildFrames` when the
    /// path-traced radiance cache is live, else 0).
    public var cacheBuildFrames: Int?

    public init(frames: Int = 32, jitterPixels: Float = 1, accumulationFloor: Float = 0.01,
                outputSize: (width: Int, height: Int)? = nil, internalScale: Float? = nil,
                svgfLevels: Int? = nil, reconstructionFilter: IlluminatoramaReconstructionFilter? = nil,
                lockExposure: Bool = false, sunWarmupFrames: Int = 3, cacheBuildFrames: Int? = nil) {
        self.frames = frames; self.jitterPixels = jitterPixels; self.accumulationFloor = accumulationFloor
        self.outputSize = outputSize; self.internalScale = internalScale; self.svgfLevels = svgfLevels
        self.reconstructionFilter = reconstructionFilter; self.lockExposure = lockExposure
        self.sunWarmupFrames = sunWarmupFrames; self.cacheBuildFrames = cacheBuildFrames
    }
}

/// What a settle did (no pixels). `IlluminatoramaStillResult` adds the readback.
public struct IlluminatoramaStillReport {
    public let warmupFrames: Int
    public let cacheBuildFrames: Int
    public let accumulatedFrames: Int
    public let internalScale: Float
    public let milliseconds: Double
}

public struct IlluminatoramaStillResult {
    public let frame: IlluminatoramaFrame
    public let warmupFrames: Int
    public let cacheBuildFrames: Int
    public let accumulatedFrames: Int
    public let internalScale: Float
    public let milliseconds: Double
}

public extension IlluminatoramaRenderer {

    /// Take a still. `step` renders ONE frame and returns whether it was committed; the default
    /// drives the renderer directly (`render(blocking: true)`), a host with per-frame scene
    /// state passes its own (and reads the `IlluminatoramaStillFrame` to hold build frames out of
    /// its accumulators). Everything the still changes on the renderer is restored on return.
    @MainActor
    func renderStill(_ o: IlluminatoramaStillOptions,
                     step: ((IlluminatoramaStillFrame) -> Bool)? = nil) -> IlluminatoramaStillResult? {
        var frame: IlluminatoramaFrame?
        let report = settleStill(o, step: step) { frame = $0.readFrame() }
        guard let frame else { return nil }
        return IlluminatoramaStillResult(
            frame: frame, warmupFrames: report.warmupFrames, cacheBuildFrames: report.cacheBuildFrames,
            accumulatedFrames: report.accumulatedFrames, internalScale: report.internalScale,
            milliseconds: report.milliseconds)
    }

    /// The settle without the readback, for a host that reads the output itself (it may want a
    /// `CGImage`, or to keep the renderer's own texture). `atEnd` runs INSIDE the still scope,
    /// after the last frame and before anything is restored — the only moment `outputTexture` is
    /// the still's size, so a readback taken after return would be of the restored (canvas) size.
    @MainActor
    @discardableResult
    func settleStill(_ o: IlluminatoramaStillOptions,
                     step: ((IlluminatoramaStillFrame) -> Bool)? = nil,
                     atEnd: (IlluminatoramaRenderer) -> Void = { _ in }) -> IlluminatoramaStillReport {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let step = step ?? { [unowned self] _ in self.render(blocking: true) }

        // ── enter ────────────────────────────────────────────────────────────
        let savedSize = (width: outputWidth, height: outputHeight)
        let savedScale = internalRenderScale
        let savedJitter = taaJitterPixels, savedBlend = taaHistoryBlend, savedAO = ssaoTemporalBlend
        let savedExposure = autoExposureEnabled
        let savedSVGF = (shared: IlluminatoramaSharedSettings.shared.svgfEnabled, levels: svgfLevels)
        let savedFilter = reconstructionFilter
        defer {
            taaJitterPixels = savedJitter; taaHistoryBlend = savedBlend; ssaoTemporalBlend = savedAO
            autoExposureEnabled = savedExposure
            IlluminatoramaSharedSettings.shared.svgfEnabled = savedSVGF.shared
            svgfLevels = savedSVGF.levels
            reconstructionFilter = savedFilter
            if internalRenderScale != savedScale { internalRenderScale = savedScale }
            if outputWidth != savedSize.width || outputHeight != savedSize.height {
                resize(width: savedSize.width, height: savedSize.height)
            }
        }
        if let s = o.outputSize, s.width != outputWidth || s.height != outputHeight {
            resize(width: s.width, height: s.height)
        }
        if let scale = o.internalScale, scale != internalRenderScale { internalRenderScale = scale }
        resetAOAccumulation()
        if o.lockExposure { autoExposureEnabled = false }
        if let levels = o.svgfLevels {
            IlluminatoramaSharedSettings.shared.svgfEnabled = true
            svgfLevels = levels
        }
        if let f = o.reconstructionFilter { reconstructionFilter = f }

        // Frames held out of every accumulator: no jitter, and a blend of 1 so the history is
        // REPLACED by them rather than averaged with them.
        func heldOut(_ kind: IlluminatoramaStillFrame, _ n: Int) {
            for _ in 0..<n {
                taaJitterPixels = 0; taaHistoryBlend = 1
                _ = step(kind)
            }
        }

        // ── warm-up + cache build ────────────────────────────────────────────
        let warmup = rtSunSoftShadowsEnabled ? max(0, o.sunWarmupFrames) : 0
        if warmup > 0 {
            heldOut(.warmup, warmup)
            resetRTGIAccumulation()
        }
        let build = o.cacheBuildFrames
            ?? (rtPathTracedGI && rtPathRadianceCacheCell > 0 ? max(0, rtPathRadianceCacheBuildFrames) : 0)
        if build > 0 {
            resetRTGIAccumulation()                 // a FRESH cache: never the last still's
            heldOut(.cacheBuild, build)
            resetRTGIAccumulation(keepingRadianceCache: true)
        }

        // ── accumulate ───────────────────────────────────────────────────────
        let n = max(o.frames, 1)
        ssaoTemporalBlend = max(savedAO, 1 - o.accumulationFloor)   // 1/N governs, not the dial's EMA
        for k in 1...n {
            taaJitterPixels = o.jitterPixels
            taaHistoryBlend = max(1 / Float(k), o.accumulationFloor)
            _ = step(.accumulate(k))
        }

        atEnd(self)
        return IlluminatoramaStillReport(
            warmupFrames: warmup, cacheBuildFrames: build, accumulatedFrames: n,
            internalScale: internalRenderScale,
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
    }
}
