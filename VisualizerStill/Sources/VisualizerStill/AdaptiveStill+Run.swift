import Foundation

extension AdaptiveStill {

    // MARK: - The run: the stop rule and both caps (contract § 8.2 – 8.3)

    public enum Decision: Equatable, Sendable {
        case renderAnotherBatch
        case stop(StopReason)
    }

    /// The stop rule as a state machine the renderer drives: `beginBatch` before rendering a batch,
    /// `endBatch` with its tile statistics after; stop when either says so. Time is passed in (seconds on
    /// any monotonic clock), so the caps are testable without a GPU.
    public struct Run: Sendable {
        public let budget: Budget
        public let tolerances: Tolerances
        public let classifier: Classifier
        public let tilesX: Int, tilesY: Int
        public var tileCount: Int { tilesX * tilesY }
        public private(set) var batches = 0
        public private(set) var stopReason: StopReason?
        public private(set) var buildFramesPerBatch = 0
        /// Per tile, its batches' means: `means[tile * budget.maxBatches + b]`.
        private var means: [Float]
        private var featureSums: [TileFeatures]
        private var startTime: Double?
        private var lastTime: Double?
        private var batchStart: Double?
        private var slowestBatch: Double = 0

        public init(budget: Budget, tilesX: Int, tilesY: Int,
                    tolerances: Tolerances = .maximum, classifier: Classifier = Classifier()) {
            self.budget = budget
            self.tolerances = tolerances
            self.classifier = classifier
            self.tilesX = max(1, tilesX)
            self.tilesY = max(1, tilesY)
            means = [Float](repeating: 0, count: self.tilesX * self.tilesY * budget.maxBatches)
            featureSums = [TileFeatures](repeating: .zero, count: self.tilesX * self.tilesY)
        }

        /// The tile grid for a picture of this size.
        public static func tiles(outputWidth: Int, outputHeight: Int, tileSize: Int) -> (x: Int, y: Int) {
            ((outputWidth + tileSize - 1) / tileSize, (outputHeight + tileSize - 1) / tileSize)
        }

        public var isFinished: Bool { stopReason != nil }

        /// May another batch start at `now`? `finishing` is the time the export still needs after the
        /// last batch (the frame that prints the mean). False — with the reason recorded — at either cap.
        /// The first batch always starts: an export needs a picture.
        public mutating func beginBatch(at now: Double, buildFrames: Int = 0, finishing: Double = 0) -> Bool {
            if stopReason != nil { return false }
            if startTime == nil { startTime = now }
            buildFramesPerBatch = buildFrames
            if batches > 0 {
                if batches >= budget.maxBatches { stopReason = .sampleCap; return false }
                if now - startTime! + slowestBatch + finishing > budget.maxSeconds { stopReason = .timeCap; return false }
            }
            batchStart = now
            return true
        }

        /// Fold one finished batch: its tile means (linear luminance of the batch's mean image) and
        /// features, in tile order (row-major, `tilesX` wide).
        public mutating func endBatch(tileMeans: [Float], features: [TileFeatures], at now: Double) -> Decision {
            precondition(tileMeans.count == tileCount && features.count == tileCount, "tile count mismatch")
            precondition(batches < budget.maxBatches, "endBatch past the sample cap")
            for t in 0..<tileCount {
                means[t * budget.maxBatches + batches] = tileMeans[t]
                featureSums[t] = featureSums[t] + features[t]
            }
            batches += 1
            if let s = batchStart { slowestBatch = max(slowestBatch, now - s) }
            lastTime = now
            if batches >= budget.minBatches, evaluate().allSatisfy({ $0.g1 == .converged || $0.g1 == .unconvergedExempt }) {
                stopReason = .converged
                return .stop(.converged)
            }
            if batches >= budget.maxBatches { stopReason = .sampleCap; return .stop(.sampleCap) }
            return .renderAnotherBatch
        }

        /// Tile `t`'s batch means so far.
        public func batchMeans(tile t: Int) -> ArraySlice<Float> {
            means[(t * budget.maxBatches)..<(t * budget.maxBatches + batches)]
        }

        /// The tile's features, averaged over batches (classification is stable as batches accrue).
        public func features(tile t: Int) -> TileFeatures {
            batches > 0 ? featureSums[t].scaled(1 / Double(batches)) : TileFeatures()
        }

        /// Every tile's class, estimate and outcome NOW. A tile not yet converged takes the run's stop
        /// reason once it has one (and `stoppedAtSampleCap`'s placeholder while it is still running).
        public func evaluate() -> [TileResult] {
            (0..<tileCount).map { t in
                let f = features(tile: t)
                let est = AdaptiveStill.estimate(batchMeans: batchMeans(tile: t), confidence: budget.confidence)
                let (cls, plain) = classifier.classify(f, dominantShare: est?.dominantShare, batches: batches)
                let limit = tolerances.limit(for: cls, plain: plain)
                let predicted: Double? = est.map { cls.isAbsolute ? $0.predictedDeltaCodes(displayCode: f.displayCode) : $0.predictedDelta }
                let ok = batches >= budget.minBatches && (predicted.map { $0 <= limit } ?? false)
                let g1: G1Outcome
                if ok { g1 = .converged }
                else if budget.exemptClasses.contains(cls) { g1 = .unconvergedExempt }
                else { g1 = stopReason == .timeCap ? .stoppedAtTimeCap : .stoppedAtSampleCap }
                // G2 is filled by `applyingG2` — never by the noise estimate.
                return TileResult(x: t % tilesX, y: t / tilesX, regionClass: cls, plain: plain, limit: limit,
                                  predicted: predicted, estimate: est, g1: g1,
                                  g2: .unverified(.noReference), areaFraction: f.areaFraction,
                                  displayCode: f.displayCode)
            }
        }

        /// Gate 2 for every tile. Without a reference every tile is unverified — and a cached render says
        /// its cache is why. With one, a cached render is judged only against a CACHE-OFF reference, and
        /// only where the reference's own floor is under half the tolerance.
        public func applyingG2(_ results: [TileResult], reference: ReferenceCheck?, cacheMode: CacheMode,
                               g2Tolerances: G2Tolerances = .maximum) -> [TileResult] {
            guard let ref = reference else {
                let why: UnverifiedReason = cacheMode == .off ? .noReference : .cacheBiasUnmeasured
                return results.map { var r = $0; r.g2 = .unverified(why); return r }
            }
            precondition(ref.candidate.count == results.count && ref.reference.count == results.count,
                         "reference tiling mismatch")
            return results.enumerated().map { t, r0 in
                var r = r0
                if cacheMode != .off && !ref.referenceIsCacheOff { r.g2 = .unverified(.referenceUsesTheCache); return r }
                let c = ref.candidate[t], g = ref.reference[t]
                let limit = g2Tolerances.limit(for: r.regionClass)
                let ratio: Double = (c <= 0 && g <= 0) ? 1 : (g > 0 ? c / g : .infinity)
                let error: Double = r.regionClass.isAbsolute
                    ? AdaptiveStill.darkCodeGain * r.displayCode * abs(ratio - 1)
                    : abs(log2(max(ratio, 1e-12)))
                if let floors = ref.referenceFloor {
                    let f = floors[t]
                    let floorInUnits = r.regionClass.isAbsolute ? AdaptiveStill.darkCodeGain * r.displayCode * (exp2(f) - 1) : f
                    if floorInUnits >= limit / 2 { r.g2 = .unverified(.referenceTooNoisy); return r }
                }
                r.g2Error = error; r.g2Limit = limit
                r.g2 = error <= limit ? .correct : .wrong
                return r
            }
        }

        /// The export's account of itself, both gates per region. `reference` is the harness's; an export
        /// passes nil and every region reads "correctness unverified".
        public func report(reference: ReferenceCheck? = nil, cacheMode: CacheMode = .off,
                           g2Tolerances: G2Tolerances = .maximum) -> Report {
            let results = applyingG2(evaluate(), reference: reference, cacheMode: cacheMode, g2Tolerances: g2Tolerances)
            let seconds = (lastTime ?? 0) - (startTime ?? 0)
            var classes: [ClassSummary] = []
            for c in RegionClass.allCases {
                let r = results.filter { $0.regionClass == c }
                guard !r.isEmpty else { continue }
                classes.append(ClassSummary(
                    regionClass: c, tiles: r.count, areaFraction: r.reduce(0) { $0 + $1.areaFraction },
                    g1Converged: r.filter { $0.g1 == .converged }.count,
                    g2Correct: r.filter { $0.g2.isCorrect }.count, g2Wrong: r.filter { $0.g2.isWrong }.count,
                    g2Unverified: r.filter { if case .unverified = $0.g2 { true } else { false } }.count,
                    worstNoise: r.compactMap(\.predicted).max(), noiseLimit: r.map(\.limit).max() ?? 0))
            }
            func area(_ p: (TileResult) -> Bool) -> Double { results.filter(p).reduce(0) { $0 + $1.areaFraction } }
            var reasons: [UnverifiedReason: Double] = [:]
            for r in results { if case .unverified(let why) = r.g2 { reasons[why, default: 0] += r.areaFraction } }
            return Report(verdict: Verdict.of(results), stopReason: stopReason ?? .sampleCap, batches: batches,
                          framesPerBatch: budget.framesPerBatch, buildFramesPerBatch: buildFramesPerBatch,
                          seconds: seconds, budget: budget, cacheMode: cacheMode, tilesX: tilesX, tilesY: tilesY,
                          classes: classes, regions: Self.regions(results, tilesX: tilesX, tilesY: tilesY),
                          g1ConvergedArea: area { $0.g1 == .converged },
                          g2CorrectArea: area { $0.g2.isCorrect }, g2WrongArea: area { $0.g2.isWrong },
                          g2UnverifiedArea: area { if case .unverified = $0.g2 { true } else { false } },
                          unverifiedReasons: reasons)
        }

        /// 4-connected components of tiles with an unresolved or failed gate (G1 short of converged, or G2
        /// wrong) sharing a class and both gate results, largest first. A G2-unverified tile whose noise
        /// converged is not a REGION (that is every export's state) — the report's area line carries it.
        static func regions(_ results: [TileResult], tilesX: Int, tilesY: Int) -> [Region] {
            func problem(_ r: TileResult) -> Bool { r.g1 != .converged || r.g2.isWrong }
            func same(_ a: TileResult, _ b: TileResult) -> Bool {
                a.regionClass == b.regionClass && a.g1 == b.g1 && a.g2.kind == b.g2.kind
            }
            var seen = [Bool](repeating: false, count: results.count)
            var out: [Region] = []
            for start in results.indices where !seen[start] && problem(results[start]) {
                let s = results[start]
                var stack = [start]; seen[start] = true
                var reg = Region(regionClass: s.regionClass, g1: s.g1, g2: s.g2, tiles: 0, areaFraction: 0,
                                 worstNoise: nil, noiseLimit: 0, worstError: nil, errorLimit: nil,
                                 minX: .max, minY: .max, maxX: .min, maxY: .min)
                while let t = stack.popLast() {
                    let r = results[t]
                    reg.tiles += 1; reg.areaFraction += r.areaFraction
                    reg.noiseLimit = max(reg.noiseLimit, r.limit)
                    if let p = r.predicted { reg.worstNoise = max(reg.worstNoise ?? p, p) }
                    if let e = r.g2Error { reg.worstError = max(reg.worstError ?? e, e) }
                    if let l = r.g2Limit { reg.errorLimit = max(reg.errorLimit ?? l, l) }
                    reg.minX = min(reg.minX, r.x); reg.maxX = max(reg.maxX, r.x)
                    reg.minY = min(reg.minY, r.y); reg.maxY = max(reg.maxY, r.y)
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nx = r.x + dx, ny = r.y + dy
                        guard nx >= 0, ny >= 0, nx < tilesX, ny < tilesY else { continue }
                        let n = ny * tilesX + nx
                        if !seen[n], problem(results[n]), same(results[n], s) { seen[n] = true; stack.append(n) }
                    }
                }
                out.append(reg)
            }
            return out.sorted { $0.areaFraction > $1.areaFraction }
        }
    }

    // MARK: - Instrument: the class / gate map

    /// One RGBA8 pixel per tile: the class's hue — full where both gates hold, a third where noise is
    /// unresolved, and pure red where G2 is WRONG — the picture to look at before trusting a classification
    /// threshold. Row-major, `tilesX` wide.
    public static func classMap(_ results: [TileResult], tilesX: Int, tilesY: Int) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: tilesX * tilesY * 4)
        for r in results {
            var rgb: (Double, Double, Double)
            switch r.regionClass {
            case .surface:   rgb = r.plain ? (0.20, 0.55, 1.00) : (0.10, 0.85, 0.95)   // blue / cyan
            case .shadow:    rgb = (1.00, 0.55, 0.05)                                   // orange
            case .highlight: rgb = (1.00, 0.10, 0.85)                                   // magenta
            case .emitter:   rgb = (1.00, 1.00, 0.20)                                   // yellow
            case .dark:      rgb = (0.55, 0.30, 1.00)                                   // violet
            case .air:       rgb = (0.30, 1.00, 0.30)                                   // green
            }
            if r.g2.isWrong { rgb = (1, 0, 0) }
            let k = r.g1 == .converged ? 1.0 : 0.33
            let i = (r.y * tilesX + r.x) * 4
            px[i] = UInt8(rgb.0 * k * 255); px[i + 1] = UInt8(rgb.1 * k * 255)
            px[i + 2] = UInt8(rgb.2 * k * 255); px[i + 3] = 255
        }
        return px
    }
}
