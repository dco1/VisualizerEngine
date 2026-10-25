import Foundation

/// **Export Quality ▸ Maximum — render until clean** (DH-1360, contract `docs/MAXIMUM_QUALITY_CONTRACT.md`
/// § 2 and § 8). The GPU-free half: what a region's noise target is, how its error is estimated, when the
/// render stops, and what the export is allowed to say about each region afterwards. The app half
/// (`HouseRenderBridge+AdaptiveStill`) renders the batches and hands each one's tile statistics here.
///
/// ## The estimator — batch means over INDEPENDENT renders
///
/// A Maximum still is rendered as `G` BATCHES. Each batch is a complete still of its own: a freshly built
/// radiance cache (its own held-out build frames), a fresh GI running mean, a fresh AO mean, and
/// `framesPerBatch` accumulated frames. The batches are therefore independent estimates of the same
/// picture, and the export is their plain mean. For every tile (`tileSize` output pixels square) the
/// spread of the batches' tile means gives the standard error of the export's tile mean with `G − 1`
/// degrees of freedom — measured on the frames the pipeline actually produced (filter, cache, haze and
/// all), not on a model of them.
///
/// Why not a per-frame twin: inside one still, every frame reads the SAME radiance cache and the colour
/// accumulator averages a GI mean that is itself cumulative, so frames are not independent and an
/// even/odd split cannot see the error they share (contract § 8.8).
///
/// ## What it cannot see — and the measured case of it
///
/// Error COMMON to every batch. On Oak Credenza Hall (GoalBaseline `hallway2`, 1566 × 1920) the ladder of
/// independent High stills — each with its OWN freshly built cache — differs from rung to rung by
/// 0.48 % (left wall, blotch band, 32 vs 64 frames) while every rung is ~3.0 % off the 4000-path reference
/// (whose own floor is ~0.4 %). So ~3 % of the left wall's blotch is common to independently built caches:
/// it is not the cache's sampling noise, and neither twin caches nor more frames can reduce it — nor can
/// this estimator SEE it. Hence the two gates are kept apart in every result (`TileResult.g1` / `.g2`):
/// G1 (`converged` / stopped at a cap) is what the batch spread measures — stochastic variation only;
/// G2 (`correct` / `wrong` / `unverified`) needs a deep CACHE-OFF reference, which an export never has, so an
/// export's regions read "correctness unverified" — for a cached render, because the cache's error has no
/// completion criterion of its own (`UnverifiedReason.cacheBiasUnmeasured`). The stop rule is G1's alone
/// and is never tuned to make a biased picture pass. `Budget.cacheGridJitter` (off by default) is an A/B
/// arm for the measurement that decides what the common error IS; it never makes a region verified.
///
/// Host-relative budgets (DH-1257) change `Budget` — the time it may take — never `Tolerances`.
public enum AdaptiveStill {

    // MARK: - Region classes and their G1 tolerances (contract § 2)

    /// The estimation problem a tile poses (contract § 2). `air` is part of the contract but has no
    /// per-pixel signal in the G-buffer (the haze is not a surface), so tiles are never classed as air:
    /// a haze-dominated tile is judged as the surface behind it — the stricter tolerance.
    public enum RegionClass: String, CaseIterable, Sendable, Codable, Comparable {
        case surface, shadow, highlight, emitter, dark, air
        public static func < (a: Self, b: Self) -> Bool {
            allCases.firstIndex(of: a)! < allCases.firstIndex(of: b)!
        }
        /// Whether this class's tolerance is in display codes rather than a relative error.
        public var isAbsolute: Bool { self == .dark }
    }

    /// § 2's G1 column — Maximum's Δ(N→2N) tolerances, at TILE scale. A constant table: a host's speed
    /// may change how long a region takes to meet it, never the number (DH-1257).
    public struct Tolerances: Equatable, Sendable {
        /// Relative Δ for a matte surface tile with visible texture.
        public let surface: Double
        /// Relative Δ for a PLAIN surface tile — the blotch band on a plain wall (§ 2: ≤ 0.5 % on Oak's
        /// walls). Tile-scale error IS that band: a tile mean is the blotch-scale average.
        public let plainSurface: Double
        public let shadow: Double
        public let highlight: Double
        public let emitter: Double
        /// Δ in display codes (0–255) for near-black tiles — never in stops.
        public let darkCodes: Double
        public let air: Double

        public static let maximum = Tolerances(surface: 0.010, plainSurface: 0.005, shadow: 0.020,
                                               highlight: 0.05, emitter: 0.010, darkCodes: 1.0, air: 0.03)

        public func limit(for regionClass: RegionClass, plain: Bool) -> Double {
            switch regionClass {
            case .surface:   return plain ? plainSurface : surface
            case .shadow:    return shadow
            case .highlight: return highlight
            case .emitter:   return emitter
            case .dark:      return darkCodes
            case .air:       return air
            }
        }
    }

    /// G1's Δ(N→2N) between two INDEPENDENT renders at N and 2N samples is √(σ² + σ²/2) = √1.5 · σ_N
    /// (contract § 1). The stop rule predicts Δ from the estimated σ_N of the export.
    public static let deltaPerSigma = 1.5.squareRoot()

    // MARK: - Budget (the caps, declared up front)

    public struct Budget: Equatable, Sendable {
        /// Tile edge in OUTPUT pixels (contract § 8.1: ≈ 16–32).
        public var tileSize: Int
        /// Accumulated frames per batch. The app rounds it up to a multiple of the haze's pixel stride
        /// (each pixel's haze is traced on every K-th frame; a partial cycle would weight it wrongly).
        public var framesPerBatch: Int
        /// Batches before the stop rule may call anything converged — the estimate's degrees of freedom
        /// are `batches − 1`. At least 2 (one dof: the confidence factor is then ~8× at 90 %, so in practice
        /// only an experiment uses it); the default 4.
        public var minBatches: Int
        /// The SAMPLE cap: at most this many batches (`maxBatches × framesPerBatch` accumulated frames).
        public var maxBatches: Int
        /// The WALL-CLOCK cap, seconds, for the whole settle. A batch is never started when the slowest
        /// batch so far would cross it; the first batch always runs (an export needs a picture).
        public var maxSeconds: Double
        /// One-sided confidence with which a tile's σ is bounded from its `G − 1` dof before it is
        /// compared with its tolerance (0 ⇒ the point estimate).
        public var confidence: Double
        /// Classes whose unconverged tiles do not hold the render. EMPTY by default — contract § 8.5: a
        /// known non-converger stops at the cap and is reported, it does not get a pass.
        public var exemptClasses: Set<RegionClass>
        /// Vary the radiance cache's grid alignment per batch — an A/B ARM for the measurement of what the
        /// cache's common error is (see the type's docs), OFF by default. It moves that error into batch
        /// spread; it does not make a cached region verified (G2 stays `unverified` without a reference).
        public var cacheGridJitter: Bool

        public init(tileSize: Int = 16, framesPerBatch: Int = 16, minBatches: Int = 4, maxBatches: Int = 64,
                    maxSeconds: Double = 20 * 60, confidence: Double = 0.9,
                    exemptClasses: Set<RegionClass> = [], cacheGridJitter: Bool = false) {
            self.tileSize = max(4, tileSize)
            self.framesPerBatch = max(1, framesPerBatch)
            self.minBatches = max(2, minBatches)
            self.maxBatches = max(self.minBatches, maxBatches)
            self.maxSeconds = max(0, maxSeconds)
            self.confidence = min(max(confidence, 0), 0.999)
            self.exemptClasses = exemptClasses
            self.cacheGridJitter = cacheGridJitter
        }

        /// The sample cap in accumulated frames.
        public var maxFrames: Int { maxBatches * framesPerBatch }

        /// A host-relative budget (DH-1257): the same targets, a different amount of time to reach
        /// them. `speed` is this host's throughput relative to the reference machine (0.5 = half as
        /// fast). Only the caps scale; `Tolerances` is not a parameter of anything here.
        public func scaled(forHostSpeed speed: Double) -> Budget {
            var b = self
            let s = min(max(speed, 0.1), 10)
            b.maxSeconds = maxSeconds / s
            return b
        }
    }

    // MARK: - Per-tile features (from the GPU) and classification

    /// What the GPU reduces per tile at the end of a batch, from the batch's mean image (linear HDR), its
    /// tonemapped output and the G-buffer.
    public struct TileFeatures: Equatable, Sendable {
        /// The tile's share of the frame's area (edge tiles are partial).
        public var areaFraction: Double
        /// Mean display luma code (0–255) of the tonemapped picture.
        public var displayCode: Double
        /// Share of output pixels with a channel at ≥ 250 (clipped: a window, a lamp's lens).
        public var clippedFraction: Double
        /// Share of G-buffer pixels that see the sky (no geometry).
        public var skyFraction: Double
        /// Share of G-buffer pixels with emission.
        public var emissiveFraction: Double
        /// 1 − |Σ n| / count over the tile's G-buffer normals: 0 on one plane, ~0.3 half wall / half floor.
        public var normalSpread: Double
        /// max / min of the tile's 4 × 4 sub-block mean luminances (1 = flat).
        public var subBlockRatio: Double
        /// Brightest pixel / tile mean (linear): a glint or caustic is a few pixels far above the rest.
        public var peakToMean: Double

        public init(areaFraction: Double = 0, displayCode: Double = 128, clippedFraction: Double = 0,
                    skyFraction: Double = 0, emissiveFraction: Double = 0, normalSpread: Double = 0,
                    subBlockRatio: Double = 1, peakToMean: Double = 1) {
            self.areaFraction = areaFraction; self.displayCode = displayCode
            self.clippedFraction = clippedFraction; self.skyFraction = skyFraction
            self.emissiveFraction = emissiveFraction; self.normalSpread = normalSpread
            self.subBlockRatio = subBlockRatio; self.peakToMean = peakToMean
        }

        static let zero = TileFeatures(displayCode: 0, subBlockRatio: 0, peakToMean: 0)
        static func + (a: Self, b: Self) -> Self {
            Self(areaFraction: a.areaFraction + b.areaFraction, displayCode: a.displayCode + b.displayCode,
                 clippedFraction: a.clippedFraction + b.clippedFraction, skyFraction: a.skyFraction + b.skyFraction,
                 emissiveFraction: a.emissiveFraction + b.emissiveFraction, normalSpread: a.normalSpread + b.normalSpread,
                 subBlockRatio: a.subBlockRatio + b.subBlockRatio, peakToMean: a.peakToMean + b.peakToMean)
        }
        func scaled(_ k: Double) -> Self {
            Self(areaFraction: areaFraction * k, displayCode: displayCode * k, clippedFraction: clippedFraction * k,
                 skyFraction: skyFraction * k, emissiveFraction: emissiveFraction * k, normalSpread: normalSpread * k,
                 subBlockRatio: subBlockRatio * k, peakToMean: peakToMean * k)
        }
    }

    /// Contract § 8.1's classing rules, as declared thresholds. Precedence: emitter → dark → highlight →
    /// shadow → surface (plain or textured). Every looser class (shadow, highlight, dark) must be EARNED by
    /// a measured feature; anything unclassifiable is a surface — the strictest tolerance.
    public struct Classifier: Equatable, Sendable {
        /// A tile is an emitter when at least this share of it is emissive, sky, or clipped.
        public var emitterShare = 0.5
        /// Near-black: mean display code at or below this (§ 8.1: "darker than ~2 display codes").
        public var darkCode = 2.0
        /// A glint / caustic: one pixel this many times the tile mean.
        public var highlightPeak = 30.0
        /// A geometric edge (junction, contact): normal spread at or above this — ~5 % of the tile on a
        /// perpendicular face. A normal-mapped plane stays near 0.004.
        public var edgeNormalSpread = 0.04
        /// A penumbra: a 4 × 4 sub-block at this ratio to another (a shadow edge across the tile).
        public var penumbraRatio = 2.0
        /// PLAIN surface: sub-blocks within this ratio (no texture, no gradient worth the name).
        public var plainRatio = 1.15

        public init() {}

        public func classify(_ f: TileFeatures, dominantShare: Double?, batches: Int) -> (RegionClass, plain: Bool) {
            if max(f.emissiveFraction, f.skyFraction, f.clippedFraction) >= emitterShare { return (.emitter, false) }
            if f.displayCode <= darkCode { return (.dark, false) }
            if f.peakToMean >= highlightPeak { return (.highlight, false) }
            if let share = dominantShare, let bound = AdaptiveStill.dominanceBound(batches: batches), share > bound {
                return (.highlight, false)
            }
            if f.normalSpread >= edgeNormalSpread || f.subBlockRatio >= penumbraRatio { return (.shadow, false) }
            return (.surface, f.subBlockRatio <= plainRatio)
        }
    }

    /// "A tile whose variance is dominated by a few samples" (§ 8.1), made a test: the largest single
    /// batch's share of the batches' squared deviations, against its 99.9th percentile for Gaussian
    /// batches (Monte Carlo, 200 000 trials per G). Below 8 batches one batch can carry (G−1)/G of a
    /// Gaussian spread by chance, so the test is off (nil) and only `highlightPeak` can class a highlight.
    public static func dominanceBound(batches g: Int) -> Double? {
        let table: [(Double, Double)] = [(8, 0.813), (10, 0.777), (12, 0.734), (16, 0.644), (24, 0.518),
                                         (32, 0.430), (48, 0.316), (64, 0.258), (96, 0.184), (128, 0.147)]
        guard g >= 8 else { return nil }
        let x = log(Double(g))
        if Double(g) >= table.last!.0 {   // beyond the table: its last decade's power law (≈ G^−0.81)
            let (g0, s0) = table[table.count - 2], (g1, s1) = table[table.count - 1]
            let slope = log(s1 / s0) / log(g1 / g0)
            return s1 * exp(slope * (x - log(g1)))
        }
        for i in 1..<table.count where Double(g) <= table[i].0 {
            let (g0, s0) = table[i - 1], (g1, s1) = table[i]
            let t = (x - log(g0)) / (log(g1) - log(g0))
            return s0 + t * (s1 - s0)
        }
        return table.last!.1
    }

    // MARK: - The estimate of one tile

    public struct Estimate: Equatable, Sendable {
        public var batches: Int
        /// Mean over batches of the tile's mean luminance (linear, the export's tile mean).
        public var mean: Double
        /// Standard error of that mean (linear units), from the batches' spread.
        public var standardError: Double
        /// `standardError / mean` (0 when the tile is black).
        public var relativeError: Double
        /// The factor the point estimate is raised by to bound σ at `confidence` from G − 1 dof.
        public var upperFactor: Double
        /// Largest single batch's share of the squared deviations (see `dominanceBound`).
        public var dominantShare: Double
        /// Predicted G1 Δ(N→2N) of the export, relative: √1.5 · upperFactor · relativeError.
        public var predictedDelta: Double { AdaptiveStill.deltaPerSigma * upperFactor * relativeError }
        /// The same in display codes, for a dark tile of mean code `displayCode`: Δcode ≈ g · code · Δrel,
        /// with g = `darkCodeGain` (the display transform's log-slope in its toe).
        public func predictedDeltaCodes(displayCode: Double) -> Double {
            guard predictedDelta.isFinite else { return .infinity }
            return AdaptiveStill.darkCodeGain * max(0, displayCode) * predictedDelta
        }
    }

    /// Log-slope of display code vs scene luminance in the toe, for converting a dark tile's relative
    /// error into codes. sRGB's linear toe is 1; AgX's toe behaves like a ~2.3 power (engine
    /// `hueStableToe` docs), which AMPLIFIES relative error — so the larger, rounded up. Conservative: it
    /// can only make a dark tile look noisier than it is. Calibrate on the GPU (see the DH-1360 report).
    public static let darkCodeGain = 2.5

    /// Batch-means estimate from a tile's per-batch means. nil below 2 batches (no spread to measure).
    public static func estimate<C: Collection>(batchMeans: C, confidence: Double) -> Estimate?
    where C.Element == Float {
        let g = batchMeans.count
        guard g >= 2 else { return nil }
        var mean = 0.0
        for v in batchMeans { mean += Double(v) }
        mean /= Double(g)
        var ss = 0.0, maxDev = 0.0
        for v in batchMeans {
            let d = (Double(v) - mean) * (Double(v) - mean)
            ss += d; maxDev = max(maxDev, d)
        }
        let variance = ss / Double(g - 1)
        let se = (variance / Double(g)).squareRoot()
        let rel = mean > 0 ? se / mean : (se > 0 ? Double.infinity : 0)   // luminance ≥ 0: mean 0 ⇒ all 0
        return Estimate(batches: g, mean: mean, standardError: se, relativeError: rel,
                        upperFactor: sigmaUpperFactor(dof: g - 1, confidence: confidence),
                        dominantShare: ss > 0 ? maxDev / ss : 0)
    }

    /// The factor by which a sample standard deviation with `dof` degrees of freedom must be raised to
    /// bound the true σ at one-sided `confidence`: √(ν / χ²_{1−c}(ν)), χ² by Wilson–Hilferty (within
    /// 2 % of the exact quantile for ν ≥ 2). 1 when `confidence` is 0.
    public static func sigmaUpperFactor(dof: Int, confidence: Double) -> Double {
        guard confidence > 0, dof >= 1 else { return 1 }
        let nu = Double(dof)
        let z = inverseNormal(1 - confidence)          // the lower tail's quantile (negative)
        let h = 2 / (9 * nu)
        let base = max(1e-3, 1 - h + z * h.squareRoot())
        let chi2 = nu * base * base * base
        return (nu / chi2).squareRoot()
    }

    /// Φ⁻¹(p) — Acklam's rational approximation (relative error < 1.2e-9).
    public static func inverseNormal(_ p: Double) -> Double {
        let p = min(max(p, 1e-12), 1 - 1e-12)
        let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                 6.680131188771972e+01, -1.328068155288572e+01]
        let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                 -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
        let lo = 0.02425
        if p < lo {
            let q = (-2 * log(p)).squareRoot()
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        }
        if p > 1 - lo {
            let q = (-2 * log(1 - p)).squareRoot()
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        }
        let q = p - 0.5, r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    }

    // MARK: - Two gates per region, kept apart (contract § 1, § 8.4 – 8.6)
    //
    // Every tile carries BOTH results and nothing folds them into one green/red:
    //   G1 (noise)       converged / stopped at the sample cap / stopped at the time cap — what the batch
    //                    spread can see: STOCHASTIC variation only, never total error.
    //   G2 (correctness) correct / wrong / UNVERIFIED — only a deep, cache-off path reference can say. An
    //                    export has none, so its every region is unverified, and says why.
    // G1 may say converged while G2 fails; a G2 failure is reported as a correctness failure, never as noise;
    // and the stop rule is G1's alone — it is never tuned to make a biased picture pass.

    public enum StopReason: String, Equatable, Sendable, Codable {
        /// Every non-exempt tile is within its class's NOISE tolerance (G1 only).
        case converged
        /// The sample cap (`Budget.maxBatches`) was reached first.
        case sampleCap
        /// The wall-clock cap (`Budget.maxSeconds`) was reached first.
        case timeCap
    }

    /// Gate 1, noise. `converged` means more samples would not change the tile beyond its tolerance —
    /// NOT that the tile is right (G2) or looks like the photograph (G3).
    public enum G1Outcome: String, CaseIterable, Equatable, Sendable, Codable {
        case converged
        case stoppedAtSampleCap
        case stoppedAtTimeCap
        /// Over its tolerance, in a class the budget exempted from holding the render (never by default).
        case unconvergedExempt
    }

    /// Why a tile's correctness could not be judged.
    public enum UnverifiedReason: String, CaseIterable, Equatable, Sendable, Codable {
        /// No reference was compared (every export).
        case noReference
        /// The render read the radiance cache, whose error is common to every batch (a fixed grid) and has
        /// no completion criterion of its own — no estimator here can bound it. Only a cache-off reference
        /// can, and none was compared.
        case cacheBiasUnmeasured
        /// The reference itself read the radiance cache: it shares the approximation it would judge.
        case referenceUsesTheCache
        /// The reference's own noise is at least half the tolerance: it cannot tell right from wrong here.
        case referenceTooNoisy
    }

    /// Gate 2, correctness, at tile scale: the tile's level against the reference's.
    public enum G2Status: Equatable, Sendable {
        case correct
        case wrong
        case unverified(UnverifiedReason)
        public var isCorrect: Bool { self == .correct }
        public var isWrong: Bool { self == .wrong }
        var kind: Int { switch self { case .correct: 0; case .wrong: 1; case .unverified: 2 } }
    }

    /// § 2's G2 column at tile scale: |level| in stops (relative classes) or |ΔI| in display codes (dark).
    public struct G2Tolerances: Equatable, Sendable {
        public let surface: Double, shadow: Double, highlight: Double, emitter: Double, air: Double
        public let darkCodes: Double
        public static let maximum = G2Tolerances(surface: 0.05, shadow: 0.10, highlight: 0.15, emitter: 0.05,
                                                 air: 0.10, darkCodes: 1.0)
        public func limit(for c: RegionClass) -> Double {
            switch c {
            case .surface: surface
            case .shadow: shadow
            case .highlight: highlight
            case .emitter: emitter
            case .air: air
            case .dark: darkCodes
            }
        }
    }

    /// How the radiance cache took part in the still — recorded so a report can say what it cannot vouch for.
    public enum CacheMode: String, Equatable, Sendable, Codable {
        case off
        /// One freshly built cache per batch, on the fixed world grid.
        case perBatch
        /// One per batch, the grid realigned per batch (an A/B arm: it turns the grid's error into batch
        /// spread, which G1 then sees — it does not make the cache verified).
        case perBatchRealigned
        /// ONE cache built before the first batch and read by every batch — an experiment arm only: the
        /// batch spread then excludes the cache's own sampling error (within-cache accumulation noise).
        case sharedAcrossBatches
    }

    /// A comparison against a deep path-traced reference (the harness's, never an export's). All three
    /// arrays are TILE means in the same tiling and the same units — `tileMeans(luminance:…)` of the
    /// linearised export and of the reference.
    public struct ReferenceCheck: Equatable, Sendable {
        public var candidate: [Double]
        public var reference: [Double]
        /// Per tile, the reference's own noise in stops (|log2(halfA / halfB)| / 2) — nil when unknown,
        /// which leaves the floor test out (and says so: see `referenceTooNoisy`).
        public var referenceFloor: [Double]?
        /// The reference was rendered with the radiance cache OFF (contract § 1: a reference never uses
        /// the approximation it judges).
        public var referenceIsCacheOff: Bool
        public init(candidate: [Double], reference: [Double], referenceFloor: [Double]? = nil, referenceIsCacheOff: Bool) {
            self.candidate = candidate; self.reference = reference
            self.referenceFloor = referenceFloor; self.referenceIsCacheOff = referenceIsCacheOff
        }
    }

    /// Mean of `luminance` (row-major, `width` × `height`) over each `tileSize` tile, in the run's tiling.
    public static func tileMeans(luminance: [Double], width: Int, height: Int, tileSize: Int) -> [Double] {
        let tx = (width + tileSize - 1) / tileSize, ty = (height + tileSize - 1) / tileSize
        var sum = [Double](repeating: 0, count: tx * ty), n = [Double](repeating: 0, count: tx * ty)
        for y in 0..<height {
            for x in 0..<width {
                let t = (y / tileSize) * tx + x / tileSize
                sum[t] += luminance[y * width + x]; n[t] += 1
            }
        }
        return zip(sum, n).map { $1 > 0 ? $0 / $1 : 0 }
    }

    public struct TileResult: Equatable, Sendable {
        public var x: Int, y: Int
        public var regionClass: RegionClass
        public var plain: Bool
        /// G1: the noise tolerance (relative Δ, or codes for `dark`), the predicted Δ(N→2N) in the same
        /// units (nil below 2 batches), and the outcome.
        public var limit: Double
        public var predicted: Double?
        public var estimate: Estimate?
        public var g1: G1Outcome
        /// G2, and — when a reference was compared — the measured error in its units (stops, or codes for
        /// `dark`) against `g2Limit`.
        public var g2: G2Status
        public var g2Error: Double? = nil
        public var g2Limit: Double? = nil
        public var areaFraction: Double
        public var displayCode: Double
    }

    /// A 4-connected run of tiles with one class and the same two gate results — reported only when a gate
    /// is unresolved or failed (G1 not converged, or G2 wrong).
    public struct Region: Equatable, Sendable {
        public var regionClass: RegionClass
        public var g1: G1Outcome
        public var g2: G2Status
        public var tiles: Int
        public var areaFraction: Double
        /// Worst predicted noise Δ / measured G2 error, in the class's units.
        public var worstNoise: Double?
        public var noiseLimit: Double
        public var worstError: Double?
        public var errorLimit: Double?
        /// Tile-space bounds, inclusive.
        public var minX: Int, minY: Int, maxX: Int, maxY: Int
    }

    /// The whole export, from its tiles — and only from them, so no summary can be greener than its worst
    /// region: any G2 `wrong` ⇒ `failed`; else any G1 short of converged or any G2 unverified ⇒ `incomplete`;
    /// `verified` only when EVERY tile is G1 converged AND G2 correct.
    public enum Verdict: String, Equatable, Sendable, Codable {
        case verified, incomplete, failed

        public static func of(_ tiles: [TileResult]) -> Verdict {
            if tiles.contains(where: { $0.g2.isWrong }) { return .failed }
            if tiles.isEmpty || tiles.contains(where: { $0.g1 != .converged || !$0.g2.isCorrect }) { return .incomplete }
            return .verified
        }
    }

    public struct ClassSummary: Equatable, Sendable {
        public var regionClass: RegionClass
        public var tiles: Int
        public var areaFraction: Double
        public var g1Converged: Int
        public var g2Correct: Int, g2Wrong: Int, g2Unverified: Int
        public var worstNoise: Double?
        public var noiseLimit: Double
    }

    public struct Report: Equatable, Sendable {
        public var verdict: Verdict
        public var stopReason: StopReason
        public var batches: Int
        public var framesPerBatch: Int
        public var buildFramesPerBatch: Int
        public var seconds: Double
        public var budget: Budget
        public var cacheMode: CacheMode
        public var tilesX: Int, tilesY: Int
        public var classes: [ClassSummary]
        /// Regions with an unresolved or failed gate, largest first.
        public var regions: [Region]
        /// Area shares by gate result.
        public var g1ConvergedArea: Double
        public var g2CorrectArea: Double, g2WrongArea: Double, g2UnverifiedArea: Double
        /// Area unverified, by reason.
        public var unverifiedReasons: [UnverifiedReason: Double]

        public var accumulatedFrames: Int { batches * framesPerBatch }

        /// The Render Details rows: (label, detail, ok). Labels are unique (the panel keys rows by them).
        /// `ok` is true only for a gate that PASSED — an unverified or stopped gate is never drawn as a pass.
        public func lines(maxRegions: Int = 8) -> [(label: String, detail: String, ok: Bool)] {
            var out: [(String, String, Bool)] = []
            let minutes = seconds / 60
            let time = minutes >= 1 ? String(format: "%.1f min", minutes) : String(format: "%.0f s", seconds)
            switch verdict {
            case .verified:
                out.append(("Maximum", "Verified — every region clean and checked against a reference", true))
            case .incomplete:
                out.append(("Maximum", "Not verified — see below", false))
            case .failed:
                out.append(("Maximum", "Failed — part of the picture is clean but WRONG against the reference", false))
            }
            let g1Detail: String
            switch stopReason {
            case .converged: g1Detail = "Clean — every region within its noise target · \(accumulatedFrames) frames, \(time)"
            case .sampleCap: g1Detail = "Stopped at the frame limit — \(Self.percent(1 - g1ConvergedArea)) not clean · \(accumulatedFrames) frames, \(time)"
            case .timeCap:   g1Detail = "Stopped at the time limit — \(Self.percent(1 - g1ConvergedArea)) not clean · \(accumulatedFrames) frames, \(time)"
            }
            out.append(("Noise", g1Detail, g1ConvergedArea >= 1 - 1e-9 && stopReason == .converged))
            var g2: [String] = []
            if g2CorrectArea > 0 { g2.append("correct \(Self.percent(g2CorrectArea))") }
            if g2WrongArea > 0 { g2.append("WRONG \(Self.percent(g2WrongArea))") }
            for r in UnverifiedReason.allCases {
                if let a = unverifiedReasons[r], a > 0 { g2.append("unverified \(Self.percent(a)) (\(Self.reasonText(r)))") }
            }
            out.append(("Correctness", g2.joined(separator: " · "), g2CorrectArea >= 1 - 1e-9))
            if cacheMode != .off {
                out.append(("Light Cache", cacheMode == .perBatch
                            ? "A fresh cache per batch — its own error is not measured by the noise check"
                            : "A fresh, realigned cache per batch — its own error is not measured by the noise check", false))
            }
            for (i, r) in regions.prefix(maxRegions).enumerated() {
                var gates: [String] = []
                switch r.g1 {
                case .converged: break
                case .stoppedAtSampleCap, .stoppedAtTimeCap, .unconvergedExempt:
                    let why = r.g1 == .stoppedAtTimeCap ? "time limit" : (r.g1 == .stoppedAtSampleCap ? "frame limit" : "exempt")
                    gates.append("noise unresolved at the \(why) (\(r.worstNoise.map { Self.format($0, r.regionClass) } ?? "unmeasured"), target \(Self.format(r.noiseLimit, r.regionClass)))")
                }
                switch r.g2 {
                case .wrong:
                    gates.append("WRONG vs reference (\(r.worstError.map { Self.formatError($0, r.regionClass) } ?? "?"), target \(r.errorLimit.map { Self.formatError($0, r.regionClass) } ?? "?"))")
                case .unverified(let why): gates.append("correctness unverified (\(Self.reasonText(why)))")
                case .correct: gates.append("correct")
                }
                out.append(("Region \(i + 1) · \(r.regionClass.rawValue.capitalized)",
                            "\(Self.percent(r.areaFraction)) at tiles \(r.minX)–\(r.maxX) × \(r.minY)–\(r.maxY): " + gates.joined(separator: "; "),
                            false))
            }
            if regions.count > maxRegions {
                out.append(("More Regions", "\(regions.count - maxRegions) more with an unresolved gate", false))
            }
            return out
        }

        static func reasonText(_ r: UnverifiedReason) -> String {
            switch r {
            case .noReference: "no reference"
            case .cacheBiasUnmeasured: "light-cache error not measured"
            case .referenceUsesTheCache: "reference used the light cache"
            case .referenceTooNoisy: "reference too noisy"
            }
        }
        static func percent(_ f: Double) -> String { f < 0.001 && f > 0 ? "<0.1 %" : String(format: "%.1f %%", f * 100) }
        static func format(_ v: Double, _ c: RegionClass) -> String {
            c.isAbsolute ? String(format: "±%.1f codes", v) : String(format: "±%.2f %%", v * 100)
        }
        static func formatError(_ v: Double, _ c: RegionClass) -> String {
            c.isAbsolute ? String(format: "%.1f codes", v) : String(format: "%.2f st", v)
        }
    }
}
