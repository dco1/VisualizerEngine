// plausibility: real — local adaptation, not added light. The operator only re-EXPOSES the
// rendered, scene-referred image per neighbourhood (a hue-preserving gain from an edge-aware
// base layer — Durand & Dorsey 2002 on a Chen–Paris–Durand bilateral grid), the way a retina
// or a camera's local tone mapping does; every light in the scene keeps its physical level.
import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

/// **Local tone mapping** — eye-like local adaptation before the display transform.
///
/// A single global exposure has one answer for the whole frame, so a scene whose regions sit
/// many stops apart (red LED digits at 200 cd/m² next to a moonlit window at 0.01 cd/m²)
/// prints everything more than ~7 stops under the key as black — the display transform's toe
/// has no more latitude than that. A local operator adapts each region to its own
/// neighbourhood: dim regions are lifted toward `anchor`, partially and saturating (within
/// `knee` stops of the anchor contrast is kept as shot; farther down, the slope falls to
/// `1 − strength`), and the adaptation never crosses a strong edge (a bilateral grid, so no
/// halos). See `IlluminatoramaLocalToneMap.metal` for the maths.
///
/// **Opt-in.** `strength == 0` (the default) ⇒ the renderer does not encode the pass at all and
/// every existing scene — and Daydream Home — renders byte-identically.
public struct IlluminatoramaLocalToneMapping: Equatable, Sendable {
    /// 0 = OFF (default). The far-tail compression: how much of the range more than `knee`
    /// stops under `anchor` is adapted away (0.5 halves it). Keep < 1 (at 1 the far tail flattens
    /// to one level).
    public var strength: Float = 0
    /// Size of the adaptation neighbourhood — the bilateral grid's spatial scale — as a fraction
    /// of the frame height (resolution-independent). Larger = more global (less local contrast
    /// change, fewer, softer gradients); smaller = more local.
    public var radius: Float = 0.04
    /// Range scale in stops (the grid's bin spacing): two regions more than ~2× this apart never
    /// share an adaptation level, which is what keeps a bright object from casting a dark halo
    /// onto its dim surround. Smaller = sharper edge protection; larger = smoother gain.
    public var edgeStops: Float = 1
    /// Local detail kept: 1 = the adaptation is a function of the smooth base only, so texture and
    /// fine contrast come out as rendered (measured: a 1-stop checker — texture as strong as
    /// `edgeStops` — keeps its swing to 0.04 stops; weaker texture closer still); > 1 boosts
    /// local contrast (the "HDR look" — avoid).
    public var detail: Float = 1
    /// The EXPOSED linear brightness held fixed (after this frame's exposure; same metric as the
    /// meter: max(luma, ½·max channel)). Neighbourhoods at or above it are untouched.
    public var anchor: Float = 0.18
    /// Stops under `anchor` over which contrast is kept nearly as rendered before the compression
    /// sets in (the curve's slope is 1 at the anchor and `1 − strength` far below it).
    public var knee: Float = 2
    /// Largest lift any neighbourhood may receive, in stops, approached smoothly (a 1-stop-wide
    /// smooth minimum, so no tone-curve kink): a floor on how dark a region may be and still be
    /// adapted to — the eye's own limit — and a bound on how far the frame's fp16 quantisation is
    /// amplified (a night room at 2e-7 scene units holds only a handful of fp16 subnormal steps).
    public var maxLift: Float = 8

    // ── Second segment (opt-in) ──────────────────────────────────────────
    /// A SCENE-referred brightness (the lit frame's own units, same metric as the meter) below
    /// which the curve's far-tail slope changes from `1 − strength` to `floorSlope`. 0 = off
    /// (one far-tail slope, the original curve).
    ///
    /// Why: a single far-tail slope has to serve two jobs that want different answers. Between
    /// the key and a dim-but-bright-enough region (a twilight window beside red LED digits) the
    /// range must be compressed hard, or the window never reads; but a region further down (the
    /// room lit only by that window, 5–7 stops under it) compressed at the SAME slope collapses
    /// onto the window — measured on Digital Clock at dusk, a 5.2-stop window/wall ratio printed
    /// as ~1 stop, and the window stopped glowing. With the floor at the window's level, contrast
    /// below it comes back at `floorSlope` (0.5 prints a 6-stop drop as 3). The host usually
    /// knows that level each frame (a radiance probe); the shader converts it with the frame's
    /// exposure, so it tracks auto-exposure. Still under `maxLift`, still strictly monotone.
    public var floorLevel: Float = 0
    /// The curve's slope below `floorLevel` (1 = contrast kept as rendered; clamped to at
    /// least `1 − strength`, so it never compresses harder than the tail above it).
    public var floorSlope: Float = 0.5

    // ── Mesopic vision (opt-in) ──────────────────────────────────────────
    /// 0 = off. The fraction of the CIE 191 mesopic shift applied (1 = the full model): a
    /// dark-adapted eye loses colour as the light falls — cones hand over to the colour-blind
    /// rods between ~5 and 0.005 cd/m² — and weights the spectrum toward the blue (rods peak
    /// at 507 nm: a red light far out goes dark, a blue-grey wall keeps its brightness). The
    /// adaptation coefficient `m` is CIE 191:2010's (MES2: m = 0.767 + 0.3334·log10 L_mes,
    /// iterated) on the neighbourhood's ABSOLUTE luminance — the operator's own edge-aware base,
    /// in cd/m² through `mesopicNitsPerUnit` — and each pixel blends from its colour toward its
    /// rod response (scotopic luminance per Larson et al. 1997, normalised so a D65 grey keeps
    /// its luminance, toward `mesopicTint`) by 1 − m. Needs `mesopicNitsPerUnit` > 0 and the
    /// operator on (`isEnabled`).
    public var mesopic: Float = 0
    /// cd/m² per unit of the lit frame (divide any HDR pre-exposure in).
    public var mesopicNitsPerUnit: Float = 0
    /// Hue of rod vision (luminance-normalised): (1,1,1) = neutral grey; a Purkinje blue such
    /// as (0.78, 0.90, 1.18) reads as moonlight.
    public var mesopicTint: SIMD3<Float> = SIMD3(1, 1, 1)

    // ── Adaptation from the frame (opt-in) ───────────────────────────────
    /// OFF (default) = the lift is bounded by `maxLift` alone. ON = also by the frame's own
    /// EXPOSURE DEFICIT: the bilateral grid is a whole-frame log-brightness histogram at the
    /// applied exposure, so its mean is a meter reading of what the camera actually printed;
    /// the lift ceiling is `adaptationGain · max(0, adaptationTargetEV − mean − adaptationTolerance)`
    /// (computed on the GPU, no readback). A frame exposed within `adaptationTolerance` of the key
    /// is left EXACTLY as rendered (no lift, no mesopic shift — the pass is an exact copy); as the
    /// light falls and a capped / partial meter leaves the field under-exposed, the eye makes the
    /// deficit up stop for stop. One adaptation state, keyed on the frame — not a second ramp
    /// fighting the exposure's.
    public var adaptationFromFrame: Bool = false
    /// The meter key the deficit is measured against (log2 of the exposed brightness a "correct"
    /// field averages to — the auto-exposure target EV).
    public var adaptationTargetEV: Float = -2.2
    /// Stops of underexposure a print keeps before the eye starts adapting.
    public var adaptationTolerance: Float = 1
    /// Stops of lift per stop of deficit past the tolerance (1 = the eye makes it all up).
    public var adaptationGain: Float = 1
    /// Photopic field luminance (cd/m², through `mesopicNitsPerUnit`; 0 = off): a field whose mean
    /// luminance is above it gets one more stop of tolerance per stop above — in cone vision the
    /// eye's lightness constancy keeps pace with a camera's print (a sunlit room reads as the
    /// photograph of it), so the operator only re-adapts where the light has fallen toward mesopic.
    public var adaptationPhotopicNits: Float = 0

    // ── Histogram-adjusted tail (opt-in) ─────────────────────────────────
    /// OFF (default) = the fixed far-tail slope `1 − strength` (and the `floorLevel` segment).
    /// ON = the far-tail slope at each brightness is set from the frame's own histogram (Ward
    /// Larson, Rushmeier & Piatko 1997, histogram adjustment): populated brightness ranges keep
    /// contrast, empty ones (the gap between bright emitters and a dim room) are compressed to
    /// `gapSlope`; the populated slope is then scaled (≥ `minPopulatedScale`) so the absolute
    /// floor luminance `printFloorNits` lands exactly on `printFloorLevel` — the range the eye
    /// adapts across is FITTED into what the display can still print. `strength` still has to be
    /// > 0 (it enables the pass); `floorLevel` / `floorSlope` are ignored.
    public var histogramTail: Bool = false
    /// Slope (output stops per input stop) across an empty brightness range.
    public var gapSlope: Float = 0.05
    /// Fraction of the frame a 1-stop bin must hold to count as fully populated (slope 1 before
    /// the fit's scale).
    public var populatedFraction: Float = 0.01
    /// Lowest the fit may scale the populated slope to.
    public var minPopulatedScale: Float = 0.1
    /// Absolute luminance (cd/m², through `mesopicNitsPerUnit`) the fit lands on
    /// `printFloorLevel`: ~1e-5, where a dark-adapted eye still makes out a room. 0 = no fit.
    public var printFloorNits: Float = 0
    /// EXPOSED brightness (after the lift; the meter's metric) the display prints at the lowest
    /// code the host wants a dark-adapted room to reach — solve it through the display
    /// transform's inverse (`IlluminatoramaDisplayInverse.exposedLevel`).
    public var printFloorLevel: Float = 0

    // ── Mesopic model (opt-in, with `mesopic` > 0) ───────────────────────
    public enum MesopicModel: Int, Sendable {
        /// CIE 191 m as the colour weight (the original — m is a LUMINANCE weighting).
        case cie191 = 0
        /// Jensen et al. (2000)'s blue shift: colour fades with LOG luminance across
        /// `mesopicLogRange` toward rod vision in `mesopicTint` (Jensen's CIE xy (0.25, 0.25) ≈
        /// linear sRGB (0.71, 0.99, 1.97)).
        case blueShift = 1
    }
    public var mesopicModel: MesopicModel = .cie191
    /// log10 cd/m²: fully rod-coloured at or below `.x`, fully photopic at or above `.y`
    /// (Jensen 2000: −2 … 0.6).
    public var mesopicLogRange: SIMD2<Float> = SIMD2(-2, 0.6)
    /// Take the mesopic luminance as max(the pixel's own, its neighbourhood's): the base comes
    /// from the dark side of an edge, so a lit patch's rim would otherwise be rod-tinted.
    public var mesopicPixelFloor: Bool = false

    /// How a pixel's ROD response is computed from its linear sRGB (opt-in; default `.larson`).
    public enum MesopicRodModel: Int, Sendable {
        /// Larson, Rushmeier & Piatko (1997): V = Y·[1.33·(1 + (Y+Z)/X) − 1.68] — a fit that is
        /// NOT additive in radiance: red light added to dim moonlight raises X and the mixture's
        /// rod response falls below the moonlight's own, so the edge of a red light printed a
        /// near-black seam where the mixture was red-dominated but still mesopic (VZ-0194).
        case larson = 0
        /// The sum of per-primary rod responses (Larson's own value at each sRGB primary; D65 grey
        /// within 0.3 %) — additive, as a photoreceptor's response to light is: adding light never
        /// lowers it, and the mesopic print rises monotonically with radiance.
        case additive = 1
    }
    public var mesopicRodModel: MesopicRodModel = .larson
    /// With `mesopicPixelFloor`: floor the mesopic luminance with the pixel's CONE signal (the
    /// operator's brightness metric, max(luma, ½·max channel)) instead of its luma. A small red-lit
    /// feature beside a dark field (a toy's ridge) has a dark neighbourhood base AND a low luma — a
    /// saturated red is ~¼ luma — so it was rod-mapped to near-black while the same red on a larger
    /// patch beside it stayed red: a dark seam printed from radiance no darker than its
    /// neighbours' (VZ-0194). Opt-in.
    public var mesopicConeFloor: Bool = false
    /// Take the mesopic WEIGHT from the neighbourhood's plain (not edge-aware) adaptation level —
    /// the blurred grid's mean log brightness per cell, bilinear — instead of the edge-aware base
    /// sliced at the pixel's own brightness. The eye adapts to its surround, not to each pixel:
    /// the edge-aware base follows a small lit feature's own level, so across a red light's
    /// terminator the weight flipped from rod to cone within a few pixels and the rod-mapped band
    /// (red has almost no rod response) printed darker than both sides (VZ-0194). With a smooth
    /// weight the print mixes cone colour and rod response by the same amount on both sides of an
    /// edge — linear in radiance with non-negative weights, so a brighter input never prints darker.
    /// (`mesopicPixelFloor` still floors it with the pixel's own level.) Opt-in.
    public var mesopicSpatialAdaptation: Bool = false

    /// Rod luminance (photopic-equivalent: D65 grey keeps its luminance) — Larson et al. 1997.
    /// Mirrors the shader's `ltmScotopicRatio` path.
    public static func larsonRodLuminance(_ c: SIMD3<Float>) -> Float {
        let X = simd_dot(c, SIMD3(0.4124, 0.3576, 0.1805))
        let Y = simd_dot(c, SIMD3(0.2126, 0.7152, 0.0722))
        let Z = simd_dot(c, SIMD3(0.0193, 0.1192, 0.9505))
        guard X > 1e-20, Y > 1e-20 else { return Y }
        let vy = max(1.33 * (1 + (Y + Z) / X) - 1.68, 0)
        return min(Y * vy / 2.573, 4 * Y)
    }
    /// Per-primary rod responses (Larson's V/2.573 at R, G and B) — the additive model's weights.
    public static let additiveRodWeights = SIMD3<Float>(0.032876, 0.765326, 0.201635)
    public static func additiveRodLuminance(_ c: SIMD3<Float>) -> Float {
        simd_dot(simd_max(c, .zero), additiveRodWeights)
    }

    /// Jensen et al. (2000)'s scotopic blue-shift chromaticity (CIE xy 0.25, 0.25) in linear
    /// sRGB, unit luma.
    public static let jensenBlueShiftTint = SIMD3<Float>(0.706, 0.990, 1.966)

    public init(strength: Float = 0, radius: Float = 0.04, edgeStops: Float = 1, detail: Float = 1,
                anchor: Float = 0.18, knee: Float = 2, maxLift: Float = 8,
                floorLevel: Float = 0, floorSlope: Float = 0.5,
                mesopic: Float = 0, mesopicNitsPerUnit: Float = 0, mesopicTint: SIMD3<Float> = SIMD3(1, 1, 1)) {
        self.strength = strength
        self.radius = radius
        self.edgeStops = edgeStops
        self.detail = detail
        self.anchor = anchor
        self.knee = knee
        self.maxLift = maxLift
        self.floorLevel = floorLevel
        self.floorSlope = floorSlope
        self.mesopic = mesopic
        self.mesopicNitsPerUnit = mesopicNitsPerUnit
        self.mesopicTint = mesopicTint
    }

    /// True when the renderer encodes the pass.
    public var isEnabled: Bool { strength > 0 && maxLift > 0 }

    /// The lift (stops) a flat neighbourhood `stopsUnderAnchor` under the anchor receives — the
    /// shader's curve, for hosts calibrating against measured levels (and the unit tests).
    /// `floorStopsUnderAnchor` is where `floorLevel` sits under the anchor after this frame's
    /// exposure (nil = the second segment off, whatever `floorLevel` says).
    public func lift(stopsUnderAnchor d: Float, floorStopsUnderAnchor dF: Float? = nil) -> Float {
        guard isEnabled, d > 0 else { return 0 }
        let s = max(0, min(0.98, strength))
        let c = 1 - s
        let f = knee > 1e-4 ? c * d + (1 - c) * knee * (1 - exp(-d / knee)) : c * d
        var lift = d - f
        if let dF, floorLevel > 0 {
            lift -= (Self.effectiveFloorSlope(floorSlope, strength: s) - c) * Self.softplus(d - dF)
        }
        lift = max(0, lift)
        // The shader's soft ceiling (a smooth min with `maxLift`, `ceilingSoftness` stops wide).
        let w = Self.ceilingSoftness
        return max(0, lift - w * log(1 + exp((lift - maxLift) / w)))
    }

    /// Width (stops) of the smooth ceiling at `maxLift` (the shader's `kLTMCeilingSoftness`).
    public static let ceilingSoftness: Float = 1
    /// Width (stops) of the smooth corner at `floorLevel` (the shader's `kLTMFloorSoftness`).
    public static let floorSoftness: Float = 1

    /// `floorSlope` as the shader uses it: never flatter than the tail above it, at most 1.
    static func effectiveFloorSlope(_ slope: Float, strength s: Float) -> Float {
        max(1 - s, min(1, slope))
    }
    /// w·ln(1 + e^(x/w)) — the smooth corner the shader's `ltmSoftplus` draws.
    static func softplus(_ x: Float) -> Float {
        let w = floorSoftness
        return x / w > 20 ? x : w * log(1 + exp(x / w))
    }
}

/// The GPU side of `IlluminatoramaLocalToneMapping`: four compute passes on the host's command
/// buffer (log-brightness pyramid level → bilateral-grid gather → separable 3-axis blur → slice +
/// apply), pipelines from the shared `SimPipelineCache`, no queue of its own. Owns its textures
/// (reallocated only when the frame size or grid shape changes).
@MainActor
final class IlluminatoramaLocalToneMapPass {

    // ── Grid constants ───────────────────────────────────────────────────
    /// Low-res texels per grid cell along x / y (the pyramid level is chosen so a cell is this
    /// many texels across: the gather reads 64 texels per cell and bin).
    static let texelsPerCell = 8
    /// Exposed log2 range the grid spans. −24 is ~21 stops under mid-grey (anything darker is
    /// adapted as if it were this dark — `maxLift` then bounds it); +8 is far past any display
    /// white (above the anchor the operator does nothing anyway).
    static let lMin: Float = -24
    static let lMax: Float = 8
    /// Slice fallback weight (in texel units): where the grid holds less than about this much
    /// weight near a pixel's level, the base leans toward the pixel's own level.
    static let sliceEpsilon: Float = 0.25

    /// Swift mirror of the shader's `LTMParams` (float4 / uint4 fields only — no padding traps).
    struct Params {
        var tone: SIMD4<Float>
        var grid: SIMD4<Float>
        var expo: SIMD4<Float>
        var sizes: SIMD4<UInt32>
        var cells: SIMD4<UInt32>
        var cells2: SIMD4<UInt32>
        /// x = floor level (scene-referred; 0 = off), y = floor slope, zw unused.
        var tail: SIMD4<Float>
        /// x = mesopic amount (0 = off), y = cd/m² per frame unit, zw unused.
        var meso: SIMD4<Float>
        /// xyz = rod-vision tint (luminance-normalised in the shader), w unused.
        var mesoTint: SIMD4<Float>
        /// x = adaptation from the frame ON, y = meter target EV, z = tolerance, w = gain.
        var adapt: SIMD4<Float> = .zero
        /// x = histogram tail ON, y = gap slope, z = populated fraction, w = min populated scale.
        var hist: SIMD4<Float> = .zero
        /// x = print-floor luminance (cd/m²), y = print-floor level (exposed), zw unused.
        var pfloor: SIMD4<Float> = .zero
        /// x = mesopic model, y/z = log10 range, w = pixel floor ON.
        var meso2: SIMD4<Float> = .zero
        /// x = photopic field luminance (cd/m²; 0 = off), yzw unused.
        var adapt2: SIMD4<Float> = .zero
    }

    /// Floats in the stats buffer (`illumiLTMStats`): 4 header + the f(d) table.
    static let statsFloatCount = 4 + 161
    /// The f(d) table's step (stops) — the shader's `kLTMCurveStep`.
    static let curveStep: Float = 0.25

    /// The grid layout for one frame size and radius.
    struct Layout: Equatable {
        var fullW: Int, fullH: Int
        var factor: Int          // full-res px per low texel
        var cellLow: Int         // low texels per cell
        var lowW: Int, lowH: Int
        var gridW: Int, gridH: Int, gridD: Int
        var cellPixels: Int { factor * cellLow }
    }

    static func layout(width: Int, height: Int, settings s: IlluminatoramaLocalToneMapping) -> Layout {
        let cellPx = max(2, Double(s.radius) * Double(height))
        let factor = max(1, Int((cellPx / Double(texelsPerCell)).rounded()))
        let cellLow = max(1, Int((cellPx / Double(factor)).rounded()))
        let lowW = (width + factor - 1) / factor, lowH = (height + factor - 1) / factor
        let bin = max(0.25, s.edgeStops)
        let gridD = Int(((lMax - lMin) / bin).rounded(.up)) + 1
        return Layout(fullW: width, fullH: height, factor: factor, cellLow: cellLow,
                      lowW: lowW, lowH: lowH,
                      gridW: (lowW + cellLow - 1) / cellLow, gridH: (lowH + cellLow - 1) / cellLow,
                      gridD: gridD)
    }

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatorama.ltm")

    let device: MTLDevice
    private let lumPipeline: MTLComputePipelineState?
    private let buildPipeline: MTLComputePipelineState?
    private let blurPipeline: MTLComputePipelineState?
    private let applyPipeline: MTLComputePipelineState?
    /// The neighbourhood's (not edge-aware) mean log brightness per grid cell — `mesopicSpatialAdaptation`.
    private let adaptPipeline: MTLComputePipelineState?
    private var adaptation: MTLTexture?
    private let statsPipeline: MTLComputePipelineState?
    /// Frame statistics + the fitted curve (`illumiLTMStats` → apply). Shared storage so a host
    /// can read the adaptation state the GPU settled on (`lastLiftCeiling` — a plain load of the
    /// latest completed write, never a wait; the same idiom as the renderer's `lastAutoExposure`).
    /// Unified memory: shared costs nothing over private for 165 floats.
    private(set) var statsBuffer: MTLBuffer?
    /// The lift ceiling (stops) `illumiLTMStats` derived from the frame's exposure deficit on the
    /// latest completed frame — `adaptationGain · max(0, deficit − tolerance)` — or −1 when
    /// `adaptationFromFrame` is off or the pass has not run. Lets a host key a print-side grade on
    /// the SAME adaptation state as the lift instead of a second ramp of its own.
    var lastLiftCeiling: Float {
        guard let b = statsBuffer, b.storageMode == .shared else { return -1 }
        return b.contents().advanced(by: MemoryLayout<Float>.stride).assumingMemoryBound(to: Float.self).pointee
    }

    private var layoutCache: Layout?
    private var lowLog: MTLTexture?
    private var gridA: MTLTexture?
    private var gridB: MTLTexture?
    private(set) var output: MTLTexture?

    /// TEST-OBSERVABLE: the layout and params of the last encode.
    private(set) var lastLayout: Layout?
    private(set) var lastParams: Params?

    /// The kernels' names in `IlluminatoramaLocalToneMap.metal`.
    static let kernelNames = ["illumiLTMLogLuminance", "illumiLTMGridBuild", "illumiLTMGridBlur", "illumiLTMApply",
                              "illumiLTMStats", "illumiLTMAdaptation"]

    /// Pipelines from the shared cache (the renderer's own library; no new queue or library).
    convenience init(engine: SimEngine) {
        self.init(device: engine.device, pipeline: { engine.pipeline($0) })
    }

    /// `pipeline` resolves a kernel name — the cache in the app; a runtime-compiled library in
    /// the unit tests (SwiftPM's `swift test` builds no metallib).
    init(device: MTLDevice, pipeline: (String) -> MTLComputePipelineState?) {
        self.device = device
        lumPipeline = pipeline(Self.kernelNames[0])
        buildPipeline = pipeline(Self.kernelNames[1])
        blurPipeline = pipeline(Self.kernelNames[2])
        applyPipeline = pipeline(Self.kernelNames[3])
        statsPipeline = pipeline(Self.kernelNames[4])
        adaptPipeline = pipeline(Self.kernelNames[5])
        statsBuffer = device.makeBuffer(length: Self.statsFloatCount * MemoryLayout<Float>.stride,
                                        options: .storageModeShared)
        statsBuffer?.label = "Illuminatorama.ltm.stats"
    }

    var isAvailable: Bool {
        lumPipeline != nil && buildPipeline != nil && blurPipeline != nil && applyPipeline != nil
            && statsPipeline != nil && statsBuffer != nil && adaptPipeline != nil
    }

    /// Encode the operator on `source` (scene-referred HDR, rgba16Float) and return the adapted
    /// frame — or nil (nothing encoded) when the settings are off or a pipeline is missing.
    /// `exposureBuffer` is the renderer's ExposureState (smoothedExposure at offset 4, written by
    /// the estimate kernel earlier in this command buffer); `hostExposure` / `autoExposure` are
    /// the same inputs the tonemap combines, so the operator's anchor is in the tonemap's units.
    func encode(_ cb: MTLCommandBuffer, source: MTLTexture, exposureBuffer: MTLBuffer,
                hostExposure: Float, autoExposure: Bool,
                settings s: IlluminatoramaLocalToneMapping,
                makeEncoder: (String) -> MTLComputeCommandEncoder?) -> MTLTexture? {
        guard s.isEnabled, isAvailable,
              let lumP = lumPipeline, let buildP = buildPipeline,
              let blurP = blurPipeline, let applyP = applyPipeline,
              let statsP = statsPipeline, let stats = statsBuffer, let adaptP = adaptPipeline else { return nil }
        let lay = Self.layout(width: source.width, height: source.height, settings: s)
        guard ensureTextures(lay) else { return nil }
        guard let lowLog, let gridA, let gridB, let output else { return nil }

        let bin = max(0.25, s.edgeStops)
        var p = Params(
            tone: SIMD4(log2(max(s.anchor, 1e-6)), max(0, min(0.98, s.strength)), max(0, s.knee), max(0, s.maxLift)),
            grid: SIMD4(Self.lMin, bin, max(0, s.detail), Self.sliceEpsilon),
            expo: SIMD4(hostExposure, autoExposure ? 1 : 0, Self.lMax, 0),
            sizes: SIMD4(UInt32(lay.fullW), UInt32(lay.fullH), UInt32(lay.lowW), UInt32(lay.lowH)),
            cells: SIMD4(UInt32(lay.factor), UInt32(lay.cellLow), UInt32(lay.gridW), UInt32(lay.gridH)),
            cells2: SIMD4(UInt32(lay.gridD), 0, 0, 0),
            tail: SIMD4(s.floorLevel > 0 && s.floorLevel.isFinite ? s.floorLevel : 0,
                        IlluminatoramaLocalToneMapping.effectiveFloorSlope(s.floorSlope, strength: max(0, min(0.98, s.strength))),
                        0, 0),
            meso: SIMD4(s.mesopicNitsPerUnit > 0 ? max(0, min(1, s.mesopic)) : 0, max(0, s.mesopicNitsPerUnit), 0, 0),
            mesoTint: SIMD4(simd_max(s.mesopicTint, SIMD3(repeating: 1e-4)), Float(s.mesopicRodModel.rawValue)),
            adapt: s.adaptationFromFrame
                ? SIMD4(1, s.adaptationTargetEV, max(0, s.adaptationTolerance), max(0, s.adaptationGain)) : .zero,
            hist: s.histogramTail
                ? SIMD4(1, max(0, min(1, s.gapSlope)), max(1e-4, s.populatedFraction), max(0, min(1, s.minPopulatedScale))) : .zero,
            pfloor: SIMD4(s.printFloorNits.isFinite ? max(0, s.printFloorNits) : 0,
                          s.printFloorLevel.isFinite ? max(0, s.printFloorLevel) : 0, 0, 0),
            meso2: SIMD4(Float(s.mesopicModel.rawValue), s.mesopicLogRange.x,
                         max(s.mesopicLogRange.y, s.mesopicLogRange.x + 1e-3), s.mesopicPixelFloor ? 1 : 0),
            adapt2: SIMD4(s.adaptationFromFrame && s.adaptationPhotopicNits.isFinite ? max(0, s.adaptationPhotopicNits) : 0,
                          s.mesopicConeFloor ? 1 : 0, s.mesopicSpatialAdaptation ? 1 : 0, 0))
        lastLayout = lay
        lastParams = p

        guard let enc = makeEncoder("localToneMap") else { return nil }
        enc.label = "Illuminatorama.localToneMap"
        // 1. pyramid level
        enc.setComputePipelineState(lumP)
        enc.setTexture(source, index: 0)
        enc.setTexture(lowLog, index: 1)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        enc.setBuffer(exposureBuffer, offset: 0, index: 1)
        Self.dispatch2D(enc, lumP, lay.lowW, lay.lowH)
        // 2. grid gather
        enc.setComputePipelineState(buildP)
        enc.setTexture(lowLog, index: 0)
        enc.setTexture(gridA, index: 1)
        Self.dispatch3D(enc, buildP, lay.gridW, lay.gridH, lay.gridD)
        // 2b. frame statistics + fitted curve, on the UNBLURRED grid (one threadgroup). Always
        // encoded — cheap (a few hundred reads per bin) — so the apply's stats binding is valid.
        enc.setComputePipelineState(statsP)
        // Slot 4: slots 0 / 1 are re-bound by the blurs next, and binding gridA at 0 here would
        // make the first blur's identical binding a "redundant setting" (API validation abort).
        enc.setTexture(gridA, index: 4)
        enc.setBuffer(stats, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(64, statsP.maxTotalThreadsPerThreadgroup),
                                                                height: 1, depth: 1))
        // 3. blur x (A→B), y (B→A), range (A→B)
        enc.setComputePipelineState(blurP)
        for (axis, (src, dst)) in [(gridA, gridB), (gridB, gridA), (gridA, gridB)].enumerated() {
            p.cells2.y = UInt32(axis)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
            enc.setTexture(src, index: 0)
            enc.setTexture(dst, index: 1)
            Self.dispatch3D(enc, blurP, lay.gridW, lay.gridH, lay.gridD)
        }
        // 3b. the neighbourhood's plain (not edge-aware) adaptation level per cell — read by the
        // apply only with `mesopicSpatialAdaptation`; always encoded (tiny) so its slot is bound.
        enc.setComputePipelineState(adaptP)
        enc.setTexture(gridB, index: 3)
        enc.setTexture(adaptation, index: 5)
        Self.dispatch2D(enc, adaptP, lay.gridW, lay.gridH)
        // 4. slice + apply. The grid (index 3) and the adaptation (5) stay bound from 3b and the
        // exposure buffer at 1 from step 1: re-binding an object at the slot it already occupies
        // is a "redundant setting" that API validation (assert mode) aborts on.
        enc.setComputePipelineState(applyP)
        enc.setTexture(source, index: 0)
        enc.setTexture(output, index: 2)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        Self.dispatch2D(enc, applyP, lay.fullW, lay.fullH)
        enc.endEncoding()
        return output
    }

    private func ensureTextures(_ lay: Layout) -> Bool {
        if layoutCache == lay, lowLog != nil, gridA != nil, gridB != nil, output != nil, adaptation != nil { return true }
        func tex2D(_ w: Int, _ h: Int, _ fmt: MTLPixelFormat, _ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: max(1, w),
                                                             height: max(1, h), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }
        func tex3D(_ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor()
            d.textureType = .type3D
            d.pixelFormat = .rg32Float
            d.width = max(1, lay.gridW); d.height = max(1, lay.gridH); d.depth = max(1, lay.gridD)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }
        lowLog = tex2D(lay.lowW, lay.lowH, .r32Float, "Illuminatorama.ltm.lowLog")
        gridA = tex3D("Illuminatorama.ltm.gridA")
        gridB = tex3D("Illuminatorama.ltm.gridB")
        output = tex2D(lay.fullW, lay.fullH, .rgba16Float, "Illuminatorama.ltm.out")
        adaptation = tex2D(lay.gridW, lay.gridH, .r32Float, "Illuminatorama.ltm.adaptation")
        layoutCache = lay
        let ok = lowLog != nil && gridA != nil && gridB != nil && output != nil && adaptation != nil
        if !ok { Self.log.error("local tone map: texture allocation failed") }
        return ok
    }

    private static func dispatch2D(_ enc: MTLComputeCommandEncoder, _ p: MTLComputePipelineState,
                                   _ w: Int, _ h: Int) {
        let tw = p.threadExecutionWidth
        let th = max(1, p.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreadgroups(MTLSize(width: (w + tw - 1) / tw, height: (h + th - 1) / th, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }

    private static func dispatch3D(_ enc: MTLComputeCommandEncoder, _ p: MTLComputePipelineState,
                                   _ w: Int, _ h: Int, _ d: Int) {
        let tw = min(8, p.threadExecutionWidth)
        let th = 8
        let td = max(1, min(4, p.maxTotalThreadsPerThreadgroup / (tw * th)))
        enc.dispatchThreadgroups(MTLSize(width: (w + tw - 1) / tw, height: (h + th - 1) / th,
                                         depth: (d + td - 1) / td),
                                 threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: td))
    }
}

// ── HDR frame dump (diagnostic) ──────────────────────────────────────────────

/// `VIZ_ILLUMI_HDR_DUMP_PATH=<file>`: writes the scene-referred frame the display transform is
/// about to receive (before local tone mapping, after DOF), plus this frame's exposure, so a
/// host can MEASURE its dynamic range (region levels in stops) instead of guessing from the
/// 8-bit print. Layout: "IHDR" · UInt32 width · UInt32 height · Float32 exposure · Float32 host
/// exposure · width × height × RGBA Float16. Env-gated (no cost otherwise); at most one dump in
/// flight, the latest completed frame wins.
@MainActor
final class IlluminatoramaHDRDump {
    static let path: String? = ProcessInfo.processInfo.environment["VIZ_ILLUMI_HDR_DUMP_PATH"]

    private final class InFlight: @unchecked Sendable {
        private let lock = NSLock()
        private var busy = false
        func tryAcquire() -> Bool { lock.lock(); defer { lock.unlock() }; if busy { return false }; busy = true; return true }
        func release() { lock.lock(); busy = false; lock.unlock() }
    }
    private let inFlight = InFlight()

    func encode(_ cb: MTLCommandBuffer, source: MTLTexture, exposureBuffer: MTLBuffer,
                hostExposure: Float, autoExposure: Bool) {
        guard let path = Self.path, source.pixelFormat == .rgba16Float, inFlight.tryAcquire() else { return }
        let w = source.width, h = source.height
        let rowBytes = w * 8
        guard let pixels = source.device.makeBuffer(length: rowBytes * h, options: .storageModeShared),
              let expo = source.device.makeBuffer(length: 16, options: .storageModeShared),
              let blit = cb.makeBlitCommandEncoder() else { inFlight.release(); return }
        blit.label = "Illuminatorama.hdrDump"
        blit.copy(from: source, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1), to: pixels, destinationOffset: 0,
                  destinationBytesPerRow: rowBytes, destinationBytesPerImage: rowBytes * h)
        blit.copy(from: exposureBuffer, sourceOffset: 0, to: expo, destinationOffset: 0, size: 16)
        blit.endEncoding()
        let job = Job(pixels: pixels, expo: expo, width: w, height: h, hostExposure: hostExposure,
                      autoExposure: autoExposure, path: path, gate: inFlight)
        cb.addCompletedHandler { _ in job.write() }
    }

    /// One dump's buffers, handed to the completion handler (Metal buffers are not `Sendable`;
    /// the handler is the only reader once the GPU has finished).
    private final class Job: @unchecked Sendable {
        let pixels: MTLBuffer, expo: MTLBuffer
        let width: Int, height: Int
        let hostExposure: Float, autoExposure: Bool
        let path: String
        let gate: InFlight
        init(pixels: MTLBuffer, expo: MTLBuffer, width: Int, height: Int, hostExposure: Float,
             autoExposure: Bool, path: String, gate: InFlight) {
            self.pixels = pixels; self.expo = expo; self.width = width; self.height = height
            self.hostExposure = hostExposure; self.autoExposure = autoExposure; self.path = path
            self.gate = gate
        }
        func write() {
            defer { gate.release() }
            let smoothed = expo.contents().advanced(by: 4).load(as: Float.self)
            let exposure = (autoExposure ? smoothed : 1) * hostExposure
            var header = Data("IHDR".utf8)
            for v in [UInt32(width), UInt32(height)] { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
            for v in [exposure, hostExposure] { withUnsafeBytes(of: v.bitPattern.littleEndian) { header.append(contentsOf: $0) } }
            header.append(Data(bytes: pixels.contents(), count: width * height * 8))
            try? header.write(to: URL(fileURLWithPath: path), options: .atomic)
        }
    }
}
