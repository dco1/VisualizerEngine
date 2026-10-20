import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - IBLState

/// Image-based lighting: irradiance/prefilter cubes, DFG + LTC tables, sky handoff textures.
/// Owned by the renderer as `ibls`; every member keeps the name it had on the renderer.
@MainActor
final class IBLState {
    /// **DH-0896 — the deferred composite's specular-IBL share, handed to the RT reflection
    /// pass.** `outHDR` already carries the sky cube reflected along R; a reflection ray that
    /// HITS the scene must replace that sky, not be added on top of it (which is what made a
    /// wall mirror read as a sunset with the room ghosted faintly through it). Allocated
    /// lazily at the HDR size the first frame RT reflections run, dropped on resize.
    var specIBLTexture: MTLTexture?

    /// Whether THIS frame's lighting pass wrote `specIBLTexture` — the RT kernel's read is
    /// gated on it, so a frame that skipped the write can never subtract a stale sky.
    var specIBLWrittenThisFrame = false

    /// The deferred composite's diffuse SKY share, handed to the traced GI pass (see
    /// `rtGIReplacesDiffuseSky`). Lazily allocated at the HDR size, dropped on resize.
    var diffSkyTexture: MTLTexture?

    /// Whether THIS frame's lighting pass wrote `diffSkyTexture` — the traced pass only
    /// subtracts a share that was written this frame.
    var diffSkyWrittenThisFrame = false

    // Phase 3 — sky-probe IBL. The irradiance + prefiltered cubes are baked
    // from `equirectSky` each frame (or whenever the sky texture changes).
    var irradianceCube: MTLTexture!
    var prefilteredCube: MTLTexture!
    var prefilteredMipViews: [MTLTexture]!
    /// 1×1 RGBA16F texture used in place of `equirectSky` when the host
    /// hasn't supplied one. Means the bake kernels and lighting pass can
    /// always bind a real texture and never branch on nil.
    var dummySkyTexture: MTLTexture!
    // Phase 3.2 — split-sum DFG LUT. RG16F keyed on (NdotV, roughness),
    // baked once on init via the `illumi_dfg_bake` kernel. View / material /
    // scene-independent so it never rebakes. Texture is bound at slot 10 of
    // the lighting kernel; the `dfgLUTEnabled` knob above gates sampling so
    // we can A/B against the Lagarde fallback without recompiling.
    var dfgLUT: MTLTexture!

    // #60 task 5 increment 2 — LTC area-light specular LUTs (Minv + magnitude),
    // baked once on init. `ltcValidated` is true only when the bake matched
    // brute-force ground truth; the lighting kernel falls back to MRP otherwise.
    // Bound at slots 16/17 of the lighting kernel.
    var ltcMatTexture: MTLTexture!
    var ltcMagTexture: MTLTexture!
    var ltcValidated: Bool!

    /// Wall-clock time (in renderer `time` seconds) of the last successful
    /// IBL bake — used for the `iblRebakeInterval` fallback.
    var iblLastBakeTime: Float = -.infinity

    var irradianceBakePipeline: MTLComputePipelineState!
    var prefilterBakePipeline: MTLComputePipelineState!
    var dfgBakePipeline: MTLComputePipelineState!
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    public func markIBLDirty() { iblNeedsRebake = true }

    /// **S1.6 — which Smith-GGX `k` remap `illumi_dfg_bake` integrates the split
    /// sum with.** `true` = Karis's IBL form `k = α/2` (physically the right one for
    /// this integral); `false` = the analytic-light form `k = (roughness+1)²/8` the
    /// LUT shipped with from Phase 3.2 until S1.6, which under-reports the lobe's
    /// directional albedo and so darkens specular IBL on every polished surface —
    /// worst at low roughness and grazing angles (roughness 0.12 / NdotV 0.3:
    /// G ≈ 0.54 against the correct ≈ 1.0).
    ///
    /// The LUT is baked ONCE in `init`, so this is read at construction time and a
    /// later change has no effect on an existing renderer — build a new one (which
    /// is exactly what makes an in-process A/B possible: flip, construct, measure).
    /// Env override `VIZ_ILLUMI_DFG_IBL_K=0|1` for a no-recompile A/B in a host app.
    ///
    /// This changes the LOOK of both host apps, so it is deliberately one switch in
    /// one place rather than a silent constant edit.
    public nonisolated(unsafe) static var dfgUseIBLGeometryRemap: Bool = {
        if let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_DFG_IBL_K"] {
            return v != "0"
        }
        return dfgIBLGeometryRemapDefault
    }()

    /// The shipping default for ``dfgUseIBLGeometryRemap``. Named so the decision is
    /// greppable and reversible in one line.
    ///
    /// **`true` — the CORRECTED split-sum bake** (`k = α/2`, the geometry remap the IBL
    /// integral actually needs), adopted 2026-08-28 on Danny's ruling (DH-0366, answered
    /// 2026-08-25: *"Yes, switch to the corrected formula"*). Before this it was `false` —
    /// the historical `k = (roughness+1)²/8` bake that under-reported the lobe's directional
    /// albedo and darkened specular IBL on every polished surface, worst at low roughness /
    /// grazing angles (grazing `Ess` 0.536 → 0.967). The gain lands on dielectrics — marble
    /// +53.9 %, concrete +55.4 %, matte-black +25.4 % in isolated specular IBL — not metals
    /// (chrome +2.4 %, invisible; S1.3b multi-scatter already saturates bright metals).
    ///
    /// **This is a LOOK CHANGE across BOTH host apps at once**, no per-app opt-out — set
    /// `false` (or `VIZ_ILLUMI_DFG_IBL_K=0`) to render the historical arm for an A/B. The
    /// exterior measurements below are the record Danny ruled on.
    ///
    /// **Both host apps have now measured an exterior, and they agree on the verdict while
    /// disagreeing on the mechanism — which is worth knowing before anyone re-opens this.**
    /// Visualizer's `meadow` (2026-08-09): **+9.74 % whole-frame**, visibly worse, read as
    /// veiling glare, and attributed to GRASS as the maximum-exposure case. Daydream Home's
    /// curb hero with 57 636 XPBD blades and four trees (2026-08-11, arms in separate
    /// processes because the LUT is baked once in `init`; each arm re-shot the same rig for
    /// its own drift floor):
    ///
    /// | region | legacy | IBL `k` | Δ | that arm's drift |
    /// |---|---|---|---|---|
    /// | whole frame | 140.111 | 141.454 | **+0.96 %** | 0.017 |
    /// | lawn, sunlit | 146.599 | 146.957 | +0.24 % | **±0.57 — inside it** |
    /// | stucco wall | 142.581 | 142.945 | +0.26 % | ±0.16 |
    /// | driveway | 139.376 | 139.600 | +0.16 % | ±0.03 |
    /// | **tree canopy** | 99.289 | **104.267** | **+5.01 %** | ±0.01 |
    /// | sky | 198.964 | 198.964 | **+0.000** | ±0.01 |
    ///
    /// So the whole-frame cost is an order of magnitude smaller here (0.96 % vs 9.74 %), the
    /// LAWN does not move beyond its own noise, and the sky — correctly — does not move at
    /// all. **The mover is foliage.** A leaf card is a big rough dielectric, which is exactly
    /// what this remap lifts, and looked at 2× the crown picks up a cool blue-grey sky sheen
    /// and loses its green. That IS Visualizer's veiling glare, arriving through the trees
    /// instead of the grass. The S1.6 recommendation was "do not flip it"; Danny overruled
    /// it 2026-08-28 (DH-0366) in favour of the physically-correct bake. Published
    /// `s16-dfg-exterior`, `s16-dfg-canopy`.
    public nonisolated static let dfgIBLGeometryRemapDefault = true

    /// The full-size hand-off target when RT reflections will composite this frame, else nil.
    func specIBLHandoffTarget() -> MTLTexture? {
        // A debug-term frame isolates one term; the kernel skips the write then, so no target.
        guard rtReflectionsEnabled, rtEnabled, rtTLASSupported, debugTerm == .normal else { return nil }
        let w = gbs.hdrTexture.width, h = gbs.hdrTexture.height
        if let t = ibls.specIBLTexture, t.width == w, t.height == h { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        ibls.specIBLTexture = device.makeTexture(descriptor: d)
        ibls.specIBLTexture?.label = "Illuminatorama.rt.specIBL"
        return ibls.specIBLTexture
    }

    /// The full-size hand-off target when the traced GI will replace the diffuse sky this frame.
    func diffSkyHandoffTarget() -> MTLTexture? {
        guard diffSkyHandoffActive else { return nil }
        let w = gbs.hdrTexture.width, h = gbs.hdrTexture.height
        if let t = ibls.diffSkyTexture, t.width == w, t.height == h { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        ibls.diffSkyTexture = device.makeTexture(descriptor: d)
        ibls.diffSkyTexture?.label = "Illuminatorama.rt.diffSky"
        return ibls.diffSkyTexture
    }

    /// Bake the irradiance + GGX-prefiltered cubemaps from `equirectSky` (or
    /// Conditionally bakes the IBL cubes (irradiance + GGX-prefiltered specular)
    /// from `equirectSky`. Skips the bake when the sky hasn't changed and the
    /// fallback interval hasn't elapsed, cutting ~7 GPU dispatches per frame down
    /// to only-when-needed. The host signals a change via `iblNeedsRebake = true`
    /// or `markIBLDirty()`. A `iblRebakeInterval`-second fallback catches slow
    /// cloud drift even when the host forgets to mark dirty.
    ///
    /// Prefilter sample count: reduced from 24 → 12 now that TAA (Phase 2.7)
    /// accumulates frames. With historyBlend=0.1 the effective sample budget over
    /// ~10 frames is ≈ 120 per IBL pixel — well above the aliasing threshold.
    func encodeIBLBakeIfNeeded(_ cb: MTLCommandBuffer) {
        guard iblEnabled else { return }

        // Skip if nothing has changed and the interval fallback hasn't fired.
        let timeSinceLastBake = time - ibls.iblLastBakeTime
        guard iblNeedsRebake || timeSinceLastBake >= iblRebakeInterval else { return }
        iblNeedsRebake = false
        ibls.iblLastBakeTime = time

        let sky = equirectSky ?? ibls.dummySkyTexture
        guard let enc = timedComputeEncoder(cb, "iblBake") else { return }
        enc.label = "Illuminatorama.iblBake"

        // ── Diffuse irradiance ─────────────────────────────────────
        enc.setComputePipelineState(ibls.irradianceBakePipeline)
        enc.setTexture(sky, index: 0)
        enc.setTexture(ibls.irradianceCube, index: 1)
        var bakeDesat = max(0.0, min(1.0, iblBakeDesaturation))
        enc.setBytes(&bakeDesat, length: MemoryLayout<Float>.stride, index: 0)
        dispatchCubeFaces(
            enc, pipeline: ibls.irradianceBakePipeline,
            faceSize: Self.irradianceFaceSize
        )

        // ── GGX-prefiltered specular, one mip at a time ────────────
        // 12 samples: TAA temporal accumulation makes this indistinguishable
        // from 24 at 60fps. Keep 1 sample for the perfect-mirror mip 0.
        enc.setComputePipelineState(ibls.prefilterBakePipeline)
        enc.setTexture(sky, index: 0)
        let mipCount = ibls.prefilteredMipViews.count
        for (mip, view) in ibls.prefilteredMipViews.enumerated() {
            let mipSize = max(1, Self.prefilteredFaceSize >> mip)
            let roughness = mipCount > 1
                ? Float(mip) / Float(mipCount - 1)
                : 0.0
            var params = PrefilterBakeParams(
                roughness: roughness,
                faceWidth: UInt32(mipSize),
                sampleCount: roughness <= 0.0001 ? 1 : 12,
                bakeDesat: max(0.0, min(1.0, iblBakeDesaturation))
            )
            enc.setTexture(view, index: 1)
            enc.setBytes(&params,
                         length: MemoryLayout<PrefilterBakeParams>.stride,
                         index: 0)
            dispatchCubeFaces(
                enc, pipeline: ibls.prefilterBakePipeline, faceSize: mipSize
            )
        }
        enc.endEncoding()
    }

    // ── Phase 3 IBL cube resolutions ──────────────────────────────────────────
    // 16² per face → tiny, since diffuse irradiance is a very low-frequency
    // signal. 128² per face for prefiltered specular with 6 mip levels (128,
    // 64, 32, 16, 8, 4) covering roughness 0 → 1 in equal steps.
    static let irradianceFaceSize: Int = 16

    static let prefilteredFaceSize: Int = 128

    static let prefilteredMipCount: Int = 6

    // Phase 3.2 — 128² is standard; the signal is smooth so larger is wasteful.
    static let dfgLUTSize: Int = 128

    /// Mirror of `PrefilterBakeParams` in `Illuminatorama.metal`.
    struct PrefilterBakeParams {
        var roughness: Float
        var faceWidth: UInt32
        var sampleCount: UInt32
        var bakeDesat: Float    // was _pad — desaturate equirect samples (0=off, 1=grey)
    }

    static func makeIBLCubes(
        device: MTLDevice
    ) throws -> (MTLTexture, MTLTexture, [MTLTexture]) {
        // Irradiance cube — single mip, no `pixelFormatView` needed.
        let irrDesc = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba16Float,
            size: irradianceFaceSize,
            mipmapped: false
        )
        irrDesc.usage = [.shaderRead, .shaderWrite]
        irrDesc.storageMode = .private
        guard let irr = device.makeTexture(descriptor: irrDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.irradianceCube")
        }
        irr.label = "Illuminatorama.irradianceCube"

        // Prefiltered cube — mipped, with per-mip texture views so the bake
        // kernel can write to one mip per dispatch.
        let preDesc = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba16Float,
            size: prefilteredFaceSize,
            mipmapped: true
        )
        preDesc.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
        preDesc.storageMode = .private
        preDesc.mipmapLevelCount = prefilteredMipCount
        guard let pre = device.makeTexture(descriptor: preDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.prefilteredCube")
        }
        pre.label = "Illuminatorama.prefilteredCube"

        var views: [MTLTexture] = []
        views.reserveCapacity(prefilteredMipCount)
        for mip in 0..<prefilteredMipCount {
            guard let view = pre.makeTextureView(
                pixelFormat: .rgba16Float,
                textureType: .typeCube,
                levels: mip..<(mip + 1),
                slices: 0..<6
            ) else {
                throw IlluminatoramaError.bufferAllocationFailed(
                    "Illuminatorama.prefilteredCube.mip\(mip)"
                )
            }
            view.label = "Illuminatorama.prefilteredCube.mip\(mip)"
            views.append(view)
        }
        return (irr, pre, views)
    }

    /// Procedural fallback equirect sky used when the host hasn't supplied a
    /// real one. Previously a 1×1 BLACK texel — which made every overlay
    /// scene render as nearly black because the IBL bake had no light to
    /// integrate and SCN scenes typically rely on ambient + IBL alongside
    /// their explicit lights.
    ///
    /// 64×32 is plenty of resolution for the bake kernels (16² irradiance,
    /// 128² prefilter at mip 0) to extract a smooth gradient. Colors are
    /// HDR linear, format `rgba16Float` so values >1 are preserved through
    /// the bake. The profile is a generic "open-sky" lookup table:
    ///
    ///   • Above horizon: cool pale blue, ~2× brighter at zenith.
    ///   • At horizon: warm near-white (golden-hour-ish tint without the
    ///     blown-out sun disc — keeps the lookup neutral for indoor scenes).
    ///   • Below horizon: dim warm brown, simulating ground reflection.
    ///
    /// Surfaces facing up pick up sky-blue diffuse; surfaces facing down
    /// pick up ground-warmth. Specular reflections at high roughness get a
    /// smooth gradient that's hard to mistake for a real environment but
    /// reads as ambient correctly — exactly the role the lab fills with
    /// VolumetricCloudRenderer's output for scenes that didn't supply one.
    static func makeDummySky(device: MTLDevice) -> MTLTexture {
        let w = 64, h = 32
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        // Float16 is unavailable on x86_64; use rgba32Float so both slices
        // of the universal binary compile. Metal shaders read both formats
        // identically as float4 — no shader change required.
        d.pixelFormat = .rgba32Float
        d.width = w
        d.height = h
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            preconditionFailure("Illuminatorama: failed to allocate dummy sky texture")
        }
        t.label = "Illuminatorama.dummySky"
        // HDR linear sky colours. Values above 1 are intentional — the bake
        // accumulates them with cosine weights, so a "2.0" zenith ends up
        // as a more modest contribution after irradiance integration.
        let zenith   = SIMD3<Float>(0.55, 0.72, 1.20)  // bright cool sky
        let horizon  = SIMD3<Float>(0.95, 0.92, 0.85)  // warm pale neutral
        let nadir    = SIMD3<Float>(0.22, 0.20, 0.16)  // dim warm ground
        var bytes = [Float](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            // Equirect row → latitude. y=0 (top of texture) is +π/2; y=h-1 is −π/2.
            let v = (Float(y) + 0.5) / Float(h)
            let theta = (0.5 - v) * .pi  // +π/2 at top, −π/2 at bottom
            let s = sin(theta)           // +1 top, 0 horizon, −1 bottom
            // Smooth blend between zenith/horizon (above) or nadir/horizon (below).
            // `1 − (1 − s)²` keeps the horizon band visibly distinct from zenith
            // without a hard transition; symmetric blend on the ground side.
            let colour: SIMD3<Float>
            if s >= 0 {
                let k = 1 - (1 - s) * (1 - s)
                colour = horizon + (zenith - horizon) * k
            } else {
                let k = 1 - (1 + s) * (1 + s)
                colour = horizon + (nadir - horizon) * k
            }
            for x in 0..<w {
                let base = (y * w + x) * 4
                bytes[base + 0] = colour.x
                bytes[base + 1] = colour.y
                bytes[base + 2] = colour.z
                bytes[base + 3] = 1.0
            }
        }
        bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            t.replace(
                region: MTLRegionMake2D(0, 0, w, h),
                mipmapLevel: 0,
                withBytes: base,
                bytesPerRow: w * 4 * MemoryLayout<Float>.stride
            )
        }
        return t
    }
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Irradiance / prefilter / DFG bake compute pipelines.
    @MainActor static func buildBakePipelines(into s: IBLState, cache: SimPipelineCache, device: MTLDevice) throws {
        guard let irradianceBakePipelineBuilt = cache.pipelineState(name: "illumi_irradiance_bake", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_irradiance_bake")
        }
        s.irradianceBakePipeline = irradianceBakePipelineBuilt
        guard let prefilterBakePipelineBuilt = cache.pipelineState(name: "illumi_prefilter_bake", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_prefilter_bake")
        }
        s.prefilterBakePipeline = prefilterBakePipelineBuilt
        guard let dfgBakePipelineBuilt = cache.pipelineState(name: "illumi_dfg_bake", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_dfg_bake")
        }
        s.dfgBakePipeline = dfgBakePipelineBuilt
    }

    /// Irradiance / prefilter cubemaps and the 1x1 dummy sky.
    @MainActor static func buildCubes(into s: IBLState, device: MTLDevice) throws {
        let (irrCube, preCube, preViews) = try Self.makeIBLCubes(device: device)
        s.irradianceCube = irrCube
        s.prefilteredCube = preCube
        s.prefilteredMipViews = preViews
        s.dummySkyTexture = Self.makeDummySky(device: device)
    }

    /// The DFG split-sum LUT texture (baked later by `encodeDFGBake`).
    @MainActor static func makeDFGTexture(into s: IBLState, device: MTLDevice) throws -> MTLTexture {
        // ── Phase 3.2 DFG LUT ───────────────────────────────────────────
        // Allocate and bake immediately — the LUT is view-independent so it
        // never needs to be rebuilt. Bake CB is enqueued on the shared queue;
        // subsequent render CBs on the same queue are sequenced after it
        // automatically (no waitUntilCompleted needed).
        let dfgDesc = MTLTextureDescriptor()
        dfgDesc.textureType = .type2D
        // DH-1014 — four channels: (A, B) the env split sum, `.b` the direct lobe's albedo,
        // `.a` the cloth-sheen albedo (IlluminatoramaBSDF.h documents the layout).
        dfgDesc.pixelFormat = .rgba16Float
        dfgDesc.width  = Self.dfgLUTSize
        dfgDesc.height = Self.dfgLUTSize
        dfgDesc.usage  = [.shaderRead, .shaderWrite]
        dfgDesc.storageMode = .private
        guard let dfg = device.makeTexture(descriptor: dfgDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.dfgLUT")
        }
        dfg.label = "Illuminatorama.dfgLUT"
        s.dfgLUT = dfg

        return dfg
    }

    /// LTC area-light LUTs (self-validated); falls back to the DFG LUT as a never-sampled binding.
    @MainActor static func buildLTC(into s: IBLState, device: MTLDevice, dfg: MTLTexture) {
        if let ltc = IlluminatoramaLTC.makeLUTs(device: device) {
            s.ltcMatTexture = ltc.mat
            s.ltcMagTexture = ltc.mag
            s.ltcValidated = ltc.validated
        } else {
            s.ltcMatTexture = dfg
            s.ltcMagTexture = dfg
            s.ltcValidated = false
        }
    }

    /// Enqueue the one-time DFG LUT bake on the shared queue (later CBs on the queue sequence after it).
    @MainActor static func encodeDFGBake(dfg: MTLTexture, dfgBake: MTLComputePipelineState, commandQueue: MTLCommandQueue) {
        if let cb  = commandQueue.makeCommandBuffer(),
           let enc = cb.makeComputeCommandEncoder() {
            cb.label  = "Illuminatorama.dfgBake"
            enc.label = "Illuminatorama.dfgBake"
            enc.setComputePipelineState(dfgBake)
            enc.setTexture(dfg, index: 0)
            // S1.6 — pick the `k` remap the split-sum integral is baked with. Read
            // ONCE, here, because the LUT is never rebaked.
            var dfgUseIBLRemap: UInt32 = Self.dfgUseIBLGeometryRemap ? 1 : 0
            enc.setBytes(&dfgUseIBLRemap, length: MemoryLayout<UInt32>.stride, index: 0)
            let sz  = Self.dfgLUTSize
            let tgw = dfgBake.threadExecutionWidth
            let tgh = max(1, dfgBake.maxTotalThreadsPerThreadgroup / tgw)
            let tg  = MTLSize(width: tgw, height: tgh, depth: 1)
            let groups = MTLSize(
                width:  (sz + tgw - 1) / tgw,
                height: (sz + tgh - 1) / tgh,
                depth:  1
            )
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            enc.endEncoding()
            cb.commit()
        }
    }
}
