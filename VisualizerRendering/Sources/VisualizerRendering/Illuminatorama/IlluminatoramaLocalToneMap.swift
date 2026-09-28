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
    }

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

    private var layoutCache: Layout?
    private var lowLog: MTLTexture?
    private var gridA: MTLTexture?
    private var gridB: MTLTexture?
    private(set) var output: MTLTexture?

    /// TEST-OBSERVABLE: the layout and params of the last encode.
    private(set) var lastLayout: Layout?
    private(set) var lastParams: Params?

    /// The kernels' names in `IlluminatoramaLocalToneMap.metal`.
    static let kernelNames = ["illumiLTMLogLuminance", "illumiLTMGridBuild", "illumiLTMGridBlur", "illumiLTMApply"]

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
    }

    var isAvailable: Bool {
        lumPipeline != nil && buildPipeline != nil && blurPipeline != nil && applyPipeline != nil
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
              let blurP = blurPipeline, let applyP = applyPipeline else { return nil }
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
            mesoTint: SIMD4(simd_max(s.mesopicTint, SIMD3(repeating: 1e-4)), 0))
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
        // 3. blur x (A→B), y (B→A), range (A→B)
        enc.setComputePipelineState(blurP)
        for (axis, (src, dst)) in [(gridA, gridB), (gridB, gridA), (gridA, gridB)].enumerated() {
            p.cells2.y = UInt32(axis)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
            enc.setTexture(src, index: 0)
            enc.setTexture(dst, index: 1)
            Self.dispatch3D(enc, blurP, lay.gridW, lay.gridH, lay.gridD)
        }
        // 4. slice + apply. The grid goes in at index 3 and the exposure buffer stays bound at 1
        // from step 1: re-binding an object at the slot it already occupies is a "redundant
        // setting" that API validation (assert mode) aborts on.
        enc.setComputePipelineState(applyP)
        enc.setTexture(source, index: 0)
        enc.setTexture(output, index: 2)
        enc.setTexture(gridB, index: 3)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        Self.dispatch2D(enc, applyP, lay.fullW, lay.fullH)
        enc.endEncoding()
        return output
    }

    private func ensureTextures(_ lay: Layout) -> Bool {
        if layoutCache == lay, lowLog != nil, gridA != nil, gridB != nil, output != nil { return true }
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
        layoutCache = lay
        let ok = lowLog != nil && gridA != nil && gridB != nil && output != nil
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
