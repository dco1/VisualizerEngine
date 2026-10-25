import Foundation
import QuartzCore
import VisualizerStill

/// One adaptive still (Export Quality ▸ Maximum, "render until clean"), frame by frame: `next()` says what
/// the next frame is for (and sets the renderer up for it), the caller renders it, `rendered(_:)` books it.
/// The same driver serves the blocking settle and the awaited export, so the two cannot drift apart.
/// The statistics (`AdaptiveStill.Run`) live in VisualizerStill; the per-batch image folding is
/// `IlluminatoramaStillBatches`. What a HOST must do around the frames — hold build frames out of its own
/// accumulators, restart its colour accumulator at each batch, print the held mean — comes in as `Hooks`.
@MainActor
public final class IlluminatoramaAdaptiveStillDriver {
    public enum Step: Equatable, Sendable { case build, accumulate, print }

    /// The host's side of an adaptive still. All optional: a bare renderer needs none of it.
    public struct Hooks {
        /// A build frame is about to render / has finished: hold it out of (release it back into) the host's
        /// own accumulators.
        public var holdOutBuildFrames: (Bool) -> Void = { _ in }
        /// The colour accumulator restarts at a batch: the next blend is 1/1.
        public var restartColourAccumulator: () -> Void = {}
        /// The held mean is being printed (true) / the still is over (false): TAA blend 0 while held.
        public var holdHistory: (Bool) -> Void = { _ in }
        /// A batch has just been folded (index from 0) — the moment its own picture is in the output.
        public var batchFolded: (Int) -> Void = { _ in }
        /// The still is over and `report` is set.
        public var finished: (IlluminatoramaAdaptiveStillDriver) -> Void = { _ in }
        public var log: (String) -> Void = { _ in }
        public init() {}
    }

    public unowned let renderer: IlluminatoramaRenderer
    public var hooks: Hooks
    public private(set) var run: AdaptiveStill.Run
    private let batches: IlluminatoramaStillBatches
    public let buildFrames: Int
    public let baseCell: Float
    private let savedScramble: UInt32
    /// `DAYDREAM_MAXIMUM_ADAPTIVE_SHARED_CACHE`-style arm: build the cache ONCE and read it in every batch.
    public let sharedCache: Bool
    private var phase: Phase = .startBatch
    private var lastFrameSeconds: Double = 0
    private var frameStart: Double = 0
    public private(set) var report: AdaptiveStill.Report?
    /// Per batch, every tile's mean display code — kept only for the measurement dump.
    public private(set) var batchDisplayCodes: [[Float]] = []

    private enum Phase: Equatable { case startBatch, building(Int), accumulating(Int), printing, done }

    private static func now() -> Double { CACurrentMediaTime() }

    /// nil if the engine could not set up the batch passes — the caller settles the fixed way. `buildFrames`
    /// is the radiance-cache build length (0 ⇒ no cache in play).
    public init?(renderer r: IlluminatoramaRenderer, budget: AdaptiveStill.Budget, buildFrames: Int,
                 sharedCache: Bool = false, hooks: Hooks = Hooks()) {
        guard let b = IlluminatoramaStillBatches(renderer: r, tileSize: budget.tileSize) else {
            hooks.log("adaptive still — batch passes unavailable; settling the fixed way")
            return nil
        }
        renderer = r
        self.hooks = hooks
        self.sharedCache = sharedCache
        batches = b
        run = AdaptiveStill.Run(budget: budget, tilesX: b.tilesX, tilesY: b.tilesY)
        self.buildFrames = buildFrames
        baseCell = r.rtPathRadianceCacheCell
        savedScramble = r.aoAccumulationScramble
    }

    /// How the light cache took part — recorded so the report says what it cannot vouch for.
    public var cacheMode: AdaptiveStill.CacheMode {
        guard baseCell > 0, buildFrames > 0 else { return .off }
        if sharedCache { return .sharedAcrossBatches }
        return run.budget.cacheGridJitter ? .perBatchRealigned : .perBatch
    }

    /// 0…1 by whichever cap is nearer.
    public var progress: Double {
        let b = run.budget
        return max(Double(run.batches) / Double(b.maxBatches), b.maxSeconds > 0 ? elapsed / b.maxSeconds : 0)
    }
    private var t0: Double?
    private var elapsed: Double { t0.map { Self.now() - $0 } ?? 0 }

    /// This batch's cache cell: the shipped one, or (the realignment arm) the batch's step of the sequence.
    /// A shared cache keeps the cell it was built with.
    private var batchCell: Float {
        AdaptiveStillCacheGrid.cell(base: baseCell, batch: sharedCache ? 0 : run.batches, jitter: run.budget.cacheGridJitter)
    }

    /// What the next frame is for, with the bridge set up for it; nil when the still is finished.
    public func next() -> Step? {
        let r = renderer
        if phase == .startBatch {
            if t0 == nil { t0 = Self.now() }
            if run.beginBatch(at: Self.now(), buildFrames: buildFrames, finishing: lastFrameSeconds) {
                r.rtPathRadianceCacheCell = batchCell
                if sharedCache && run.batches > 0 && buildFrames > 0 {
                    // The experiment arm: keep the first batch's cache, restart only the means.
                    r.resetRTGIAccumulation(keepingRadianceCache: true)
                    beginAccumulating()
                } else if buildFrames > 0 {
                    // A fresh cache (cleared with the GI history), built in held-out frames.
                    r.resetRTGIAccumulation()
                    hooks.holdOutBuildFrames(true)   // held out of every accumulator (DH-1344)
                    phase = .building(0)
                } else {
                    r.resetRTGIAccumulation()               // a fresh GI mean
                    beginAccumulating()
                }
            } else {
                // A cap: print the mean of what was rendered.
                batches.writeMeanToHistory()
                hooks.holdHistory(true)
                phase = .printing
            }
        }
        // Re-assert this batch's cell every frame: a policy re-derivation mid-batch would otherwise
        // hand the paths a cell size the cache was not built with.
        if baseCell > 0, phase != .printing { r.rtPathRadianceCacheCell = batchCell }
        frameStart = Self.now()
        switch phase {
        case .building: return .build
        case .accumulating: return .accumulate
        case .printing: return .print
        case .startBatch, .done: return nil
        }
    }

    private func beginAccumulating() {
        let r = renderer
        // The AO mean restarts on its own sample stream (a scramble change re-keys it); with the
        // production scramble every batch would repeat batch 0's AO rays exactly.
        r.aoAccumulationScramble = savedScramble &+ UInt32(run.batches) &* 0x9E37_79B9
        hooks.restartColourAccumulator()   // the colour accumulator restarts: next blend is 1/1
        phase = .accumulating(0)
    }

    public func rendered(_ step: Step) {
        lastFrameSeconds = max(lastFrameSeconds, Self.now() - frameStart)
        switch phase {
        case .building(let k):
            if k + 1 < buildFrames { phase = .building(k + 1); return }
            hooks.holdOutBuildFrames(false)
            renderer.resetRTGIAccumulation(keepingRadianceCache: true)
            beginAccumulating()
        case .accumulating(let k):
            if k + 1 < run.budget.framesPerBatch { phase = .accumulating(k + 1); return }
            let tiles = batches.foldBatch()
            guard tiles.count == run.tileCount else {
                hooks.log("adaptive still — the frame changed size mid-still; stopping")
                finish(); return
            }
            batchDisplayCodes.append(tiles.map(\.displayCode))
            hooks.batchFolded(run.batches)
            let decision = run.endBatch(tileMeans: tiles.map(\.meanLuminance), features: tiles.map(Self.features),
                                        at: Self.now())
            if case .stop = decision {
                batches.writeMeanToHistory()
                hooks.holdHistory(true)
                phase = .printing
            } else {
                phase = .startBatch
            }
        case .printing:
            finish()
        case .startBatch, .done:
            break
        }
    }

    private func finish() {
        hooks.holdHistory(false)
        renderer.aoAccumulationScramble = savedScramble
        renderer.rtPathRadianceCacheCell = baseCell
        if run.stopReason == nil { _ = run.beginBatch(at: .infinity) }   // a size change: book it as a cap
        report = run.report(cacheMode: cacheMode)   // an export has no reference: G2 unverified, and why
        if let report {
            hooks.log(String(format: "adaptive still — %@ after %d batches (%d frames), %.0f s; noise clean on %.1f %%; verdict %@",
                                report.stopReason.rawValue, report.batches, report.accumulatedFrames,
                                report.seconds, report.g1ConvergedArea * 100, report.verdict.rawValue))
        }
        phase = .done
        hooks.finished(self)
    }

    /// The tile reduction as the estimator's features.
    public static func features(_ t: IlluminatoramaStillBatches.Tile) -> AdaptiveStill.TileFeatures {
        AdaptiveStill.TileFeatures(areaFraction: Double(t.areaFraction), displayCode: Double(t.displayCode),
                                   clippedFraction: Double(t.clippedFraction), skyFraction: Double(t.skyFraction),
                                   emissiveFraction: Double(t.emissiveFraction), normalSpread: Double(t.normalSpread),
                                   subBlockRatio: Double(t.subBlockRatio), peakToMean: Double(t.peakToMean))
    }

}
