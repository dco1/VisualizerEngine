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

/// A still in progress. The sequence lives HERE once; `renderStill` (blocking), `renderStillStreaming`
/// (one suspension per frame, with progress) and any host that must interleave its own work between
/// frames (a thumbnail batch, an adaptive driver) are all a three-line loop over `next()`:
///
///     let s = IlluminatoramaStillSession(renderer, options); defer { s.finish() }
///     while let kind = s.next() { /* render ONE frame */ }
///
/// `init` enters the still scope (size, scale, SVGF, filter, exposure lock) and `finish()` restores
/// every value it changed. `next()` applies the per-frame knobs for the frame the caller is about to
/// render and returns what kind it is, or nil when the sequence is complete.
@MainActor
public final class IlluminatoramaStillSession {
    private let r: IlluminatoramaRenderer
    private let o: IlluminatoramaStillOptions
    private let kinds: [IlluminatoramaStillFrame]
    private var index = 0
    private let t0 = DispatchTime.now().uptimeNanoseconds
    private var finished = false

    private let savedSize: (width: Int, height: Int)
    private let savedScale: Float
    private let savedJitter: Float, savedBlend: Float, savedAO: Float
    private let savedExposure: Bool
    private let savedSVGF: (shared: Bool, levels: Int)
    private let savedFilter: IlluminatoramaReconstructionFilter

    public let warmupFrames: Int
    public let cacheBuildFrames: Int
    public let accumulatedFrames: Int
    /// Frames the sequence will render in all — for a progress fraction.
    public var totalFrames: Int { kinds.count }
    /// Frames rendered so far.
    public var framesDone: Int { index }

    public init(_ renderer: IlluminatoramaRenderer, _ options: IlluminatoramaStillOptions) {
        r = renderer; o = options
        savedSize = (renderer.outputWidth, renderer.outputHeight)
        savedScale = renderer.internalRenderScale
        savedJitter = renderer.taaJitterPixels; savedBlend = renderer.taaHistoryBlend
        savedAO = renderer.ssaoTemporalBlend
        savedExposure = renderer.autoExposureEnabled
        savedSVGF = (IlluminatoramaSharedSettings.shared.svgfEnabled, renderer.svgfLevels)
        savedFilter = renderer.reconstructionFilter

        if let s = o.outputSize, s.width != r.outputWidth || s.height != r.outputHeight {
            r.resize(width: s.width, height: s.height)
        }
        if let scale = o.internalScale, scale != r.internalRenderScale { r.internalRenderScale = scale }
        r.resetAOAccumulation()
        if o.lockExposure { r.autoExposureEnabled = false }
        if let levels = o.svgfLevels {
            IlluminatoramaSharedSettings.shared.svgfEnabled = true
            r.svgfLevels = levels
        }
        if let f = o.reconstructionFilter { r.reconstructionFilter = f }

        warmupFrames = r.rtSunSoftShadowsEnabled ? max(0, o.sunWarmupFrames) : 0
        cacheBuildFrames = o.cacheBuildFrames
            ?? (r.rtPathTracedGI && r.rtPathRadianceCacheCell > 0 ? max(0, r.rtPathRadianceCacheBuildFrames) : 0)
        accumulatedFrames = max(o.frames, 1)
        kinds = Array(repeating: .warmup, count: warmupFrames)
            + Array(repeating: .cacheBuild, count: cacheBuildFrames)
            + (1...accumulatedFrames).map { .accumulate($0) }
    }

    /// Prepare the renderer for the next frame and say what it is; nil once the sequence is done.
    public func next() -> IlluminatoramaStillFrame? {
        guard index < kinds.count else { return nil }
        let kind = kinds[index]
        let previous = index > 0 ? kinds[index - 1] : nil
        switch kind {
        case .warmup, .cacheBuild:
            // Held out of every accumulator: no jitter, and a blend of 1 so the history is
            // REPLACED by them rather than averaged with them.
            if kind == .cacheBuild, previous != .cacheBuild {
                if previous == .warmup { r.resetRTGIAccumulation() }   // out of the GI running mean
                r.resetRTGIAccumulation()                              // a FRESH cache: never the last still's
            }
            r.taaJitterPixels = 0; r.taaHistoryBlend = 1
        case .accumulate(let k):
            if k == 1 {
                if previous == .cacheBuild { r.resetRTGIAccumulation(keepingRadianceCache: true) }
                else if previous == .warmup { r.resetRTGIAccumulation() }
                r.ssaoTemporalBlend = max(savedAO, 1 - o.accumulationFloor)   // 1/N governs, not the dial's EMA
            }
            r.taaJitterPixels = o.jitterPixels
            r.taaHistoryBlend = max(1 / Float(k), o.accumulationFloor)
        }
        index += 1
        return kind
    }

    /// Restore everything the still changed. Safe to call twice.
    @discardableResult
    public func finish() -> IlluminatoramaStillReport {
        let report = IlluminatoramaStillReport(
            warmupFrames: warmupFrames, cacheBuildFrames: cacheBuildFrames, accumulatedFrames: accumulatedFrames,
            internalScale: r.internalRenderScale,
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000)
        guard !finished else { return report }
        finished = true
        r.taaJitterPixels = savedJitter; r.taaHistoryBlend = savedBlend; r.ssaoTemporalBlend = savedAO
        r.autoExposureEnabled = savedExposure
        IlluminatoramaSharedSettings.shared.svgfEnabled = savedSVGF.shared
        r.svgfLevels = savedSVGF.levels
        r.reconstructionFilter = savedFilter
        if r.internalRenderScale != savedScale { r.internalRenderScale = savedScale }
        if r.outputWidth != savedSize.width || r.outputHeight != savedSize.height {
            r.resize(width: savedSize.width, height: savedSize.height)
        }
        return report
    }
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
        let step = step ?? { [unowned self] _ in self.render(blocking: true) }
        let s = IlluminatoramaStillSession(self, o)
        defer { s.finish() }
        while let kind = s.next() { _ = step(kind) }
        atEnd(self)
        return s.finish()
    }

    /// `renderStill` for an app that must stay responsive: ONE awaited frame per suspension (the
    /// run loop turns between frames, so a progress ring repaints and the cursor moves), and
    /// `progress` is called with 0…1 after each. The host MUST stop its own draw loop for the
    /// duration — an interactive frame rendered mid-still would be folded into the accumulator.
    /// `step` defaults to the renderer's own awaited frame; a host with per-frame scene state passes
    /// its own and awaits whatever it must (`renderAwaited()` + `awaitCommittedFrame()`).
    @MainActor
    func renderStillStreaming(_ o: IlluminatoramaStillOptions,
                              step: ((IlluminatoramaStillFrame) async -> Bool)? = nil,
                              progress: (Double) -> Void = { _ in }) async -> IlluminatoramaStillResult? {
        let step = step ?? { [unowned self] _ in
            let ok = self.renderAwaited()
            await self.awaitCommittedFrame()
            return ok
        }
        let s = IlluminatoramaStillSession(self, o)
        defer { s.finish() }
        while let kind = s.next() {
            _ = await step(kind)
            progress(Double(s.framesDone) / Double(max(s.totalFrames, 1)))
            try? await Task.sleep(nanoseconds: 1_000_000)   // a run-loop turn between frames
        }
        guard let frame = readFrame() else { return nil }
        let report = s.finish()
        return IlluminatoramaStillResult(
            frame: frame, warmupFrames: report.warmupFrames, cacheBuildFrames: report.cacheBuildFrames,
            accumulatedFrames: report.accumulatedFrames, internalScale: report.internalScale,
            milliseconds: report.milliseconds)
    }
}
