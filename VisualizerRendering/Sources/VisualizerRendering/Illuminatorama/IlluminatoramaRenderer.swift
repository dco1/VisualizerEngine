import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// ── ILLUMINATORAMA RENDERER ──────────────────────────────────────────────────
//
// Phase 1: G-buffer → deferred PBR → bloom → tonemap, into a single MTLTexture
// the host SceneKit scene displays via a camera-facing quad.
// Phase 2: SSAO + SSR.
// Phase 2.5: Cascaded shadow maps for the directional sun light. Three
//          cascades are fitted to camera-frustum sub-slices each frame and
//          rendered into a depth2d_array; the lighting pass picks a cascade
//          per pixel and PCF-samples it.
// Phase 2.7: Motion vectors + TAA. G-buffer writes a velocity attachment,
//          the resolve kernel reprojects the previous frame's HDR with a
//          neighborhood-AABB clamp, and a Halton(2,3) sub-pixel projection
//          jitter gives anti-aliasing via temporal accumulation.
// Phase 3: Sky-probe IBL. Host hands us an equirect HDR sky (typically
//          VolumetricCloudRenderer's output); we bake an irradiance cube +
//          GGX-prefiltered specular cube from it each frame and the lighting
//          pass consumes both. Set `equirectSky` and (optionally) toggle
//          `iblEnabled` / `iblIntensity`.
// Phase 3.2: Split-sum DFG LUT. RG16Float LUT baked once at init from the
//          pre-integrated GGX BRDF. The lighting pass samples it at
//          (NdotV, roughness) and uses Karis's `F0 * scale + bias` for the
//          F0-weighted specular IBL — replacing the Lagarde roughness-Schlick
//          that Phase 3.0 shipped. Toggle via `dfgLUTEnabled` for A/B.
//
// ── BEFORE STARTING A NEW PHASE ──────────────────────────────────────────────
// Read docs/known-issues/illuminatorama-renderer-process.md. It captures:
//   * the verification gap (the agent has no Mac in its container, so every
//     PR here ships unverified — expect Swift↔Metal alignment / sampler /
//     cubemap-face / first-frame bugs to surface post-merge), and
//   * accumulating GPU perf debt — the ~1 Hz hiccup that prompted the perf
//     audit is no longer observed, but the underlying budget tightness is
//     still real. Profile before piling on a 3.3+ pass.
//
// Lifecycle per scene:
//
//   let renderer = try IlluminatoramaRenderer(engine: .shared, size: (1280, 720))
//   renderer.addMesh(.box, .unitBox(device: ...))
//   renderer.instances.append(...)              // host populates each frame
//
//   // each tick:
//   renderer.frame.update(camera: cam, time: t, ...)
//   renderer.render()                            // commits a command buffer
//   plane.geometry?.firstMaterial?.diffuse.contents = renderer.outputTexture
//
// See docs/illuminatorama/README.md for the full architecture rationale.

@MainActor
public final class IlluminatoramaRenderer {

    static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatorama")

    // ── Pass state ────────────────────────────────────────────────────────────
    //
    // Swift extensions cannot hold stored properties, so a pass family that lives in an
    // extension file used to leave its state here. Each family now owns a `final class`
    // defined in its own file (reference semantics, one allocation at init) and the
    // renderer holds ONE `let` per family. The holder names end in "s" for "state":
    /// Surface cache (Lumen-style card atlas): cards, atlases, charts, GPU diff buffers.
    let scs = SurfaceCacheState()
    /// Hardware ray tracing: acceleration structures, TLAS/curve pools, GI history + denoise, radiance cache, RT uniform buffers.
    let rts = RTState()
    /// Shadow maps: sun cascades, spot atlas, point cubes, static-scene reuse bookkeeping.
    let shd = ShadowState()
    /// Ambient occlusion: SSAO / RTAO targets, temporal history, pipelines.
    let aos = AOState()
    /// Screen-space reflections and screen-space subsurface scattering.
    let ssrs = SSRSSSState()
    /// Post chain: TAA history, bloom, halation, glare, tonemap, exposure, colour LUT, present ring.
    let pfx = PostFXState()
    /// Depth of field: tile/dilate/prefilter/half/quarter pipelines and targets.
    let dofs = DOFState()
    /// Volumetric shafts, spot-beam scattering and in-view clouds.
    let vols = VolumetricState()
    /// Glass (forward RT / fallback) and caustics.
    let gls = GlassState()
    /// Per-frame light and frame-uniform ring buffers.
    let lts = LightBufferState()
    /// Image-based lighting: irradiance/prefilter cubes, DFG + LTC tables, sky handoff textures.
    let ibls = IBLState()
    /// Selection outline pass: mask, dilation, composite.
    let outl = SelectionState()
    /// Particles, streaks, and GPU mesh repack / normal-synthesis pipelines.
    let fxs = ParticleState()
    /// G-buffer targets and pipelines, instance buffers, mesh draw groups, scene-stability counters, frame pacing.
    let gbs = GBufferState()
    /// Signack S3 visibility buffer: pipelines, the memoryless id target, the A/B lever.
    let vbs = VisBufferState()
    /// Per-term AOV capture.
    let aovs = AOVState()

    // ── Public knobs ──────────────────────────────────────────────────────────
    //
    // Stored in `*Knobs` classes (extensions cannot hold stored properties); the names forward from
    // `IlluminatoramaRenderer+*Knobs.swift`.
    let kLighting = LightingKnobs()
    let kShadow = ShadowKnobs()
    let kPost = PostKnobs()
    let kFilm = FilmKnobs()
    let kSky = SkyKnobs()
    let kRT = RTKnobs()
    let kMisc = MiscKnobs()
    let kGI = GIKnobs()
    let kLens = LensKnobs()
    let kGlass = GlassKnobs()

    // ── Mesh kinds ────────────────────────────────────────────────────────────
    //
    // Phase 1 / 2 hard-coded three procedural primitives that both the G-buffer
    // pass knew how to draw. Phase 2.6 adds `.custom(String)` for host-registered
    // SCNGeometry conversions; the renderer treats them identically to the
    // built-ins for G-buffer / lighting / IBL / SSR purposes.
    public enum MeshKind: Hashable, Sendable {
        case box
        case sphere
        case ground
        case custom(String)

        /// GPU-side meshKind ID written into the per-instance struct. All
        /// `.custom(_)` meshes share the same GPU ID (3); the Swift-side string
        /// identity is what disambiguates them in the mesh table.
        var gpuMeshKind: UInt32 {
            switch self {
            case .box:        return 0
            case .sphere:     return 1
            case .ground:     return 2
            case .custom:     return 3
            }
        }
    }

    public struct InstanceRef: Equatable {
        public var meshKind: MeshKind
        public var data: IlluminatoramaInstance
        /// CPU-only: effective emissive radiance used by the extractor's
        /// emissive-as-light synthesis (Phase 4.27/4.27a). Distinct from
        /// `data.emission` (the rendered G-buffer term) so a texture-driven
        /// glow can drive a light without washing the G-buffer toward white.
        /// Not uploaded to the GPU.
        public var lightEmission: SIMD3<Float>
        /// CPU-only: when non-nil, this instance is a PERFECT ANALYTIC
        /// SUPERQUADRIC IMPOSTOR. `(e1, e2)` are the roundness exponents; the
        /// extents live in `data.modelMatrix`'s scale (object space is the unit
        /// superquadric). The renderer draws it with the impostor pipeline (ray-
        /// traced in-fragment) instead of rasterizing its bounding-box mesh, and
        /// pushes the matching `SuperquadricParam` into the parallel param buffer.
        /// Not uploaded into the GPU instance struct. See IlluminatoramaSuperquadric.swift.
        public var superquadricShape: SIMD2<Float>? = nil
        public init(meshKind: MeshKind,
                    data: IlluminatoramaInstance,
                    lightEmission: SIMD3<Float> = .zero,
                    superquadricShape: SIMD2<Float>? = nil) {
            self.meshKind = meshKind
            self.data = data
            self.lightEmission = lightEmission
            self.superquadricShape = superquadricShape
        }
    }

    // ── Inputs the host updates each frame ────────────────────────────────────


    // ── Scene-content stability gate (perf) ──────────────────────────────────
    //
    // The single biggest per-frame waste in a host like Daydream Home is that a
    // CAMERA-ONLY frame (orbit/pan — by far the common case) still paid every
    // scene-content cost: the instance buffer was regrouped + rewritten, the RT
    // TLAS descriptors rewritten + refit, and every spot/point shadow map
    // re-rasterised the whole scene — all producing byte-identical results.
    // These counters detect "content unchanged since last frame" by comparing
    // the host-set `instances` (and flattened glass) against last frame's; the
    // consumers each gate on the stability they need:
    //   • `uploadInstances`   skips at ≥2 stable frames (both ping-pong buffers
    //     then already hold this exact content — motion vectors stay correct).
    //   • `updateRTAccel`     skips the refit at ≥1 stable frame (the TLAS was
    //     refit from this same content last frame; world-space, camera-free).
    //   • spot/point shadows  skip at ≥1 stable frame IF their light matrices
    //     also held and no GPU-fed geometry is live (see `sceneStaticForShadows`).
    // `lastUploadedInstances` holds a COW reference to the host's array — when
    // the host assigns a fresh array the compare is elementwise (SIMD-tight,
    // sub-ms for thousands); when it mutates in place, COW gives the mutated
    // array new storage so the compare still sees the change.
    /// Dev A/B override: VIZ_ILLUMI_NO_STATIC_SKIP=1 disables every static-scene
    /// skip (instance upload, TLAS refit, spot/point shadow reuse) so the
    /// with/without cost is directly measurable and any suspected staleness
    /// artifact can be ruled in/out in one run. Default: skips on.
    static let staticSkipDisabled =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_STATIC_SKIP"] == "1"
    /// Test-observable skip census — hosts gate perf DETERMINISTICALLY on these
    /// (N static frames must skip N-k uploads), not on wall-clock timings. Note
    /// any per-frame-animated instance (e.g. a flickering TV screen's emission)
    /// keeps the whole array unstable, so these staying 0 in such a scene is
    /// expected, not a bug.
    public struct StaticSkipStats: Equatable, Sendable {
        public var uploadsSkipped = 0
        public var tlasRefitsSkipped = 0
        public var spotShadowPassesSkipped = 0
        public var pointShadowPassesSkipped = 0
    }
    /// Debug window for tests/instruments: the current stability streak lengths
    /// (frames since full-content / shape-only / glass content last changed).
    public var stabilityDebug: (full: Int, shape: Int, glass: Int) {
        (gbs.instanceStableFrames, gbs.instanceShapeStableFrames, gls.glassStableFrames)
    }

    /// Read-only per-frame DRAW CENSUS for host-side profiling (additive; nothing
    /// in the renderer reads it). Answers "what does one frame of this document
    /// actually cost in draw work?" without the host having to re-derive the
    /// grouped layout or keep its own mesh table:
    ///
    ///   • `instances`         — CPU instance refs the host set this frame.
    ///   • `meshGroups`        — distinct draw groups (≈ draw calls in the raster passes).
    ///   • `expandedTriangles` — instance-EXPANDED triangle count (Σ group.count ×
    ///                           mesh tris), i.e. what the G-buffer + shadow passes
    ///                           actually rasterise, not the per-mesh sum. An indirect
    ///                           mesh counts the GPU's own camera-pass count from the last
    ///                           completed frame (`drawnTriangleEstimate`), not its capacity.
    ///   • `rtActive` / `rtAutoDisabled` — whether the ray-traced TLAS path is live,
    ///                           and whether the renderer self-disabled it (heavy /
    ///                           thrashing scene). A silently auto-disabled RT changes
    ///                           what a measured frame means, so profilers must see it.
    ///   • `gpuRepackTasks`    — count of compute-fed geometry tasks (e.g. the grass
    ///                           field). NON-ZERO DEFEATS THE SHADOW STATIC-SKIP
    ///                           (`sceneStaticForShadows`), so this is the number that
    ///                           explains "why is my camera-only frame still re-rendering
    ///                           every spot/point shadow map?".
    public var frameDrawCensus: (instances: Int, meshGroups: Int, expandedTriangles: Int,
                                 rtActive: Bool, rtAutoDisabled: Bool, gpuRepackTasks: Int) {
        var tris = 0
        for g in gbs.meshGroups {
            // A `tracedOnly` proxy is never rasterised — it is the RT side of an object whose
            // raster side is counted under its own (raster-only) kind.
            if let m = meshes[g.kind], !m.tracedOnly { tris += g.count * m.drawnTriangleEstimate }
        }
        return (instances.count, gbs.meshGroups.count, tris,
                rts.rtTLASActive, rtAutoDisabled, fxs.gpuRepackTasks.count)
    }
    /// The triangle count the RT size guard checks against its ceiling — every UNIQUE mesh
    /// once (a mesh placed ten times is one BLAS), glass included — beside the ceiling itself.
    /// `frameDrawCensus.expandedTriangles` counts per instance and so cannot answer "how close
    /// is this still to losing ray tracing?"; over the ceiling the renderer drops the whole TLAS
    /// (GI, RTAO, traced sun, glass) with nothing but a log line to say so.
    public var rtTriangleEstimate: (triangles: Int, ceiling: Int) {
        var tris = 0
        for g in gbs.meshGroups { if let m = meshes[g.kind], !rtGroupIsRayExcluded(g) { tris += m.indexCount / 3 } }
        for g in (rtGlassEnabled ? flattenGlass() : []) { if let m = meshes[g.kind] { tris += m.indexCount / 3 } }
        let strictLive = buildRTFromExtractedScene && rtEnabled && !rtStillCapsRelaxed
        return (tris, strictLive ? Self.rtMaxTrianglesForLiveRT : rtGlassOnlyTriangleCap)
    }
    /// Shadow maps may be reused from last frame only when the scene content is
    /// provably identical AND fully CPU-visible: no vertex-shader sway (time-
    /// animated silhouettes), no compute-fed geometry (`gpuRepackTasks` — e.g.
    /// the grass field), and no GPU-written instance slots (`onEncodeGPUInstances`).
    var sceneStaticForShadows: Bool {
        gbs.instanceShapeStableFrames >= 1 && !gbs.contentHasSway
            && fxs.gpuRepackTasks.isEmpty && onEncodeGPUInstances == nil
    }
    /// The same, minus the GPU-fed-geometry terms (DH-0631): every CPU-visible instance held and
    /// nothing sways. A spot slice whose owner carries
    /// `IlluminatoramaSpotLight.castsShadowIgnoresGPUGeometry` may be reused under this weaker
    /// condition — the host has asserted that the live GPU geometry cannot enter that cone.
    var sceneStaticExceptGPUGeometry: Bool {
        gbs.instanceShapeStableFrames >= 1 && !gbs.contentHasSway
    }

    /// Opt-in GPU-resident instance hook (additive — nil for every scene that
    /// doesn't set it, so behaviour is byte-for-byte unchanged elsewhere).
    ///
    /// Invoked once per frame inside `render()`, immediately AFTER
    /// `uploadInstances()` has packed the CPU `instances` into the current
    /// instance buffer and built the grouped layout, but BEFORE any pass reads
    /// it. The closure receives the frame command buffer, the live instance
    /// `MTLBuffer`, and a `range(for:)` lookup giving a `MeshKind`'s
    /// `[start, count)` slot range in that buffer. A scene can encode a compute
    /// kernel that OVERWRITES those slots with GPU-computed per-instance data
    /// (transform / material / emission) — keeping the data fully GPU-resident
    /// with no CPU readback or rebuild. The CPU still supplies placeholder
    /// `instances` entries for the group (so `instances.count` stays correct for
    /// RT / cull / capacity), but their *data* is never computed on the CPU.
    ///
    /// Writing the same logical instance into the same slot every frame
    /// preserves the TAA motion-vector invariant (prev-buffer holds last frame's
    /// kernel output via the existing instance-buffer ping-pong).
    ///
    /// ⚠️ RT caveat: the hardware-RT TLAS build reads `currentInstanceBuffer`
    /// CONTENTS on the CPU at encode time (refit/rebuildRTAccel), BEFORE this
    /// hook's kernel has run on the GPU — so with `rtEnabled = true` the TLAS
    /// would lag the GPU-written transforms by one CPU/GPU step. Safe for scenes
    /// with RT off (e.g. EggsUltra). A scene needing both must either keep its
    /// transforms CPU-visible for the TLAS or move the AS build after the kernel.
    public var onEncodeGPUInstances: ((_ cb: MTLCommandBuffer,
                                       _ instanceBuffer: MTLBuffer,
                                       _ range: (MeshKind) -> Range<Int>?) -> Void)?

    /// Opt-in GPU-resident POINT-LIGHT hook (additive, nil elsewhere). Same
    /// contract as `onEncodeGPUInstances` but for the point-light buffer:
    /// invoked each frame after `uploadPointLights()` has packed the CPU
    /// `pointLights` (typically zeroed placeholders) into `pointLightBuffer`,
    /// before the lighting pass. A scene kernel can overwrite the `count` slots
    /// with GPU-selected lights (e.g. a camera-near cull). Slots the kernel
    /// leaves as zeroed placeholders cost nothing — the deferred loop skips them
    /// on the `dist > radius` cutoff. `count` is `pointLights.count`.
    public var onEncodeGPUPointLights: ((_ cb: MTLCommandBuffer,
                                         _ pointLightBuffer: MTLBuffer,
                                         _ count: Int) -> Void)?

    /// True when every CPU instance kept its mesh, matrix and sway this frame and nothing sways
    /// (DH-1466: last frame's depth still stands for this frame's occluders). Valid from
    /// `onEncodeFrameCompute` on.
    public var sceneShapeHeld: Bool { gbs.instanceShapeStableFrames >= 1 && !gbs.contentHasSway }

    public var camera: IlluminatoramaCamera
    lazy var bloomGlareTailPipeline: MTLComputePipelineState? =
        engine.pipelineCache.pipelineState(name: "illumi_bloom_glare_tail", device: device)







    /// The captured means: `texture2d_array`, rgba32Float, internal size, `IlluminatoramaAOV.sliceCount`
    /// slices (`IlluminatoramaAOV.slice`). Nil until a frame has rendered with the capture on. Private
    /// storage — `readAOV` blits it.
    public var aovLightingTexture: MTLTexture? { aovs.aovTexture }
    /// Function-constant indices: `kLightingAOVEnabled` (illumi_lighting) / `kRTAOVEnabled` (TLAS kernel).
    static let perfAB = IlluminatoramaPerfAB()
    static let lightingAOVFunctionConstantIndex = 8

    /// **Daydream DH-1308 — glossy path vertices.** True: a path vertex past the primary samples the
    /// SAME layered stack the deferred kernel shades — coat (clearcoat), GGX specular (F0 = mix(0.04,
    /// albedo, metallic), VNDF-sampled, multiple scattering put back) and Lambert under them — one lobe
    /// per vertex picked ∝ its albedo, each lobe's NEE (sun, lamps, windows) MIS'd against its own
    /// density. A polished floor then throws a window's light onto the wall and a metal bounces its
    /// coloured F0, where every vertex used to be Lambert × albedo. The radiance cache keeps only
    /// the diffuse picks (a glossy lobe's light is view-dependent; a cell is not). Needs
    /// `rtPathLightSampling` and `dfgLUTEnabled` (the lobe's albedo is the DFG LUT's); off with
    /// either. Independent of `layeredMultiscatter` (the deferred pass's switch — Daydream ships it
    /// off, DH-1331): the primary keeps whatever weights that switch gives it. Default FALSE: byte-identical until the look is approved. The material comes
    /// from the per-instance `metallic` / `roughness` / `clearcoat` / `clearcoatRoughness` SCALARS
    /// (a textured roughness or metalness is not tapped at a path vertex — the albedo is).
    public var rtPathGlossyVertices: Bool = false {
        didSet { if rtPathGlossyVertices != oldValue { rts.rcNeedsClear = true } }   // the cache changes meaning
    }
    /// Signack S5 (Daydream DH-1461 / DH-1361) — a path vertex on an instance with `thinTransmission`
    /// > 0 (and no clearcoat) is a THIN SHEET: with probability ∝ its transmittance the path crosses
    /// it (cosine about −N, τ from `IlluminatoramaLeafOptics.h` `thinSheetTransmittance`) and the
    /// vertex's NEE sees the light behind it. Light then passes through a canopy, a sheer or a paper
    /// shade in an export. Default FALSE (every sheet is an opaque Lambert vertex, as before).
    public var rtPathThinTransmission: Bool = false {
        didSet { if rtPathThinTransmission != oldValue { rts.rcNeedsClear = true } }
    }
    /// DH-1308 — the roughness FLOOR of a glossy path vertex's lobes (perceptual; ≤ 0.045 ⇒ the
    /// material's own). Path regularisation: every glossy vertex here is reached through the primary's
    /// diffuse integral, so a near-mirror aimed at the sun or a bulb is a caustic path the estimator
    /// finds only by luck — fireflies. A floor trades that for a slightly broader reflection. Biased.
    public var rtPathGlossyMinRoughness: Float = 0 {
        didSet { if rtPathGlossyMinRoughness != oldValue { rts.rcNeedsClear = true } }
    }
    /// DH-1395 — how large a glossy path vertex's lobe sees a lamp: the scale on the lamp's angular
    /// radius (its physical size — `bulbRadius` / `softRadius` / `rtPathLampSourceRadius`, over distance).
    /// 1 ⇒ physical: the lobe is evaluated toward the lamp's SPHERE (a lobe of width √(α² + (ρ/2)²), same
    /// energy). 0 ⇒ a point, the DH-1308 original: toward a point a near-mirror lobe is 8e4 at its peak
    /// and a diffuse path that lands in the patch of surface in the mirror direction is a firefly (a
    /// polished counter reflecting a can onto a plain door). Unbiased relative to a true sphere lamp.
    public var rtPathGlossyLampSize: Float = 1 {
        didSet { if rtPathGlossyLampSize != oldValue { rts.rcNeedsClear = true } }
    }
    /// DH-1395 — a roughness FLOOR (perceptual; α = r²) on a glossy path vertex's lobe, applied ONLY where it is
    /// evaluated toward a LAMP (the NEE term), not to lobe sampling nor to the sun / window estimators (MIS
    /// bounds those). Path regularisation of the caustic camera → diffuse → glossy → lamp, which the
    /// estimator can only find by luck: the lamp's image in a polished surface becomes a little broader, the
    /// light's total is unchanged. 0 ⇒ the material's own roughness.
    public var rtPathGlossyLampMinRoughness: Float = 0 {
        didSet { if rtPathGlossyLampMinRoughness != oldValue { rts.rcNeedsClear = true } }
    }


    // ── AO v2 — accumulated RTAO for a frozen camera (Daydream DH-0887) ─────────────────────
    /// Host opt-in for a still / settled canvas: while RTAO runs, trace it with progressive
    /// blue-noise sampling from the geometric surface with a `1 − (t/R)²` falloff, average it in
    /// its OWN fp32 running mean (no clamp), and denoise the mean with a filter that fades as it
    /// converges — instead of blurring every frame and leaving the average to the colour TAA.
    /// See `IlluminatoramaAOAccumulator`. Default `false`: nothing is allocated or encoded and
    /// every host is byte-identical. Turning it off resets the mean.
    public var aoAccumulationEnabled: Bool = false {
        didSet { if !aoAccumulationEnabled { aos.aoAccumulator?.reset() } }
    }
    /// Frames in the current accumulated AO mean (0 = none; the per-frame path is in use).
    public var aoAccumulatedFrames: Int { aos.aoAccumulator?.frames ?? 0 }
    var rtaoActive: Bool {
        rtaoEnabled && rtaoIntensity > 0 && rtTLASSupported && rts.rtTLASActive && rts.rtTLAS != nil
    }


    /// `rtSunOwnership` resolved against the current lighting state. Read by the
    /// frame-uniform fill (to suppress the deferred sun) and by both RT lighting
    /// encoders (to suppress the RT sun) — one answer, two consumers, so they
    /// cannot disagree.
    var rtOwnsDirectSun: Bool {
        switch rtSunOwnership {
        case .deferred:  return false
        case .rayTraced: return true
        case .automatic: return simd_length_squared(directionalLightColor) <= 0
        }
    }



    // ── Surface cache (lit-radiance atlas feeding RT hits) ───────────
    //
    // When enabled AND cards have been supplied via `setSurfaceCacheCards`, an
    // `illumi_surfcache_update` pass runs before the RT pass and maintains a
    // ping-pong on-surface radiance atlas. The RT GI + reflection rays then
    // read this CACHED radiance at their hit (multi-bounce, accumulated over
    // frames) instead of re-shading sun-only — richer indirect + cheaper hits.
    // Requires `rtEnabled` (shares the AS). No-op without cards or RT support.
    public var surfaceCacheEnabled: Bool = false {
        // DH-0622 — turning the cache on over an existing atlas must not read light from whenever it
        // was last updated (a second export after the sun moved): warm it fresh, like a new bake.
        didSet { if surfaceCacheEnabled && !oldValue { scs.surfCacheWarmPending = max(0, surfaceCacheWarmIterations) } }
    }



    /// Built on first use (a diagnostics path: no host that never turns it on pays the compile).
    lazy var dofReferencePipeline: MTLComputePipelineState? = {
        guard let fn = engine.library?.makeFunction(name: "illumi_dof_reference_accumulate") else { return nil }
        return try? device.makeComputePipelineState(function: fn) // gpu-ok: lazy one-time build, opt-in diagnostics pipeline
    }()
    /// The G-buffer depth this frame rendered (NDC z, Metal [0, 1]) — for diagnostics that need the
    /// per-pixel depth the DOF read (e.g. a circle-of-confusion map). Private storage: blit it.
    public var gbufferDepthTexture: MTLTexture { gbs.depthTexture }
    /// The DOF pass's output this frame — linear HDR at the internal size, before bloom, halation,
    /// local adaptation and the tonemap (under `dofReferenceAccumulation`, the running mean) — for
    /// diagnostics that compare lenses in scene-linear light. Nil until the pass has run; its
    /// contents are this frame's only if `dofApplied`. Private storage: blit it.
    public var dofOutputTextureForDiagnostics: MTLTexture? { dofs.dofOutputTexture }
    /// DH-1293 — the RT lighting pass's raw per-frame diffuse (sun shadow + GI, before any denoise
    /// or temporal blend), for per-sample statistics: with the path lane at one path per pixel per
    /// frame, each texel IS one path sample. Read-only diagnostics; internal resolution, rgba16Float.
    public var rtDiffuseTextureForDiagnostics: MTLTexture { rts.rtDiffuseTexture }
    /// Daydream DH-1360 — the TAA ACCUMULATOR the last rendered frame wrote (linear HDR, internal size,
    /// before bloom and the tonemap; rgba16Float, read + write): the history the NEXT resolve reads. Only
    /// meaningful between frames, and only after a frame that ran TAA. `IlluminatoramaStillBatches` folds
    /// it per batch and writes a batch mean back into it.
    public var latestTAAHistoryTexture: MTLTexture { previousHistoryTexture }
    /// Daydream DH-1360 — the G-buffer's octahedral normal (xy) + roughness (z), and its emission (rgb),
    /// for per-tile classification. Internal size, rgba16Float, read-only.
    public var gbufferNormalRoughnessTexture: MTLTexture { gbs.gbufferNormalRgh }
    public var gbufferEmissionTexture: MTLTexture { gbs.gbufferEmission }



    /// The direct-sun admittance A the last COMPLETED frame's pre-pass measured (0 when the dust
    /// is off or nothing roofed is in view). An instrument read — blocking renders only.
    public var volRTMeasuredAdmittance: Float {
        guard let b = volRTAdmittanceBuffer else { return 0 }
        let c = b.contents().bindMemory(to: UInt32.self, capacity: 2)
        return c[0] > 0 ? Float(c[1]) / Float(c[0]) : 0
    }
    lazy var volRTPipeline: MTLComputePipelineState? = {
        guard device.supportsRaytracing,
              let fn = engine.library?.makeFunction(name: "illumi_volumetric_rt") else { return nil }
        return try? device.makeComputePipelineState(function: fn) // gpu-ok: lazy one-time build, still-lane opt-in pipeline
    }()
    lazy var volRTAdmittancePipeline: MTLComputePipelineState? = {
        guard device.supportsRaytracing,
              let fn = engine.library?.makeFunction(name: "illumi_volumetric_rt_admittance") else { return nil }
        return try? device.makeComputePipelineState(function: fn) // gpu-ok: lazy one-time build, still-lane opt-in pipeline
    }()
    /// Two counters (roofed, sunlit), cleared and filled by the pre-pass each haze frame.
    lazy var volRTAdmittanceBuffer: MTLBuffer? =
        device.makeBuffer(length: 2 * MemoryLayout<UInt32>.stride, options: .storageModeShared) // gpu-ok: lazy one-time 8-byte counter buffer


    /// A **self-luminous column** — a truncated cone of glowing medium (a sci-fi tractor beam,
    /// a plasma jet) integrated along every view ray in the spot-beam pass. It EMITS rather
    /// than scattering a light, so it has no phase function and no distance falloff: each
    /// sample adds `radiance · dt`, which is exactly the emission integral of a thin,
    /// non-absorbing medium. Brightest through the core (longest path), soft at the rim, and
    /// depth-terminated like the beams (geometry in front hides it, a body inside it cuts it).
    ///
    /// It lights NOTHING by itself — pair it with real lights (a spot for the pool it throws,
    /// points along its axis) so what it touches is lit by the colour it glows. Independent of
    /// `volSpotBeamsEnabled` and of the haze knobs. `nil` (default) ⇒ the pass is exactly what
    /// it was.
    public struct EmissiveColumn: Equatable, Sendable {
        /// Centre of the column's top disc (world).
        public var top: SIMD3<Float>
        /// Unit axis pointing from the top toward the bottom.
        public var axis: SIMD3<Float>
        /// Full length along `axis` (m) — the band coordinate spans this.
        public var length: Float
        public var topRadius: Float
        public var bottomRadius: Float
        /// How far down from `top` the medium currently reaches (m). `≥ length` = the whole
        /// column; less = a column growing/receding from the top, with a soft leading edge.
        public var front: Float
        /// Soft rim width as a fraction of the local radius.
        public var edgeFeather: Float = 0.3
        /// Linear emitted radiance per metre of path for the solid colour.
        public var solidRadiance: SIMD3<Float>
        /// 0 = solid colour, 1 = hue bands along the axis.
        public var rainbowMix: Float = 0
        /// Linear radiance per metre of a full-saturation hue band.
        public var rainbowRadiance: Float = 0
        /// Hue cycles over `length`.
        public var rainbowCycles: Float = 1
        /// Hue offset: band hue = frac(up · cycles + scroll), up = 0 at the bottom, 1 at the top
        /// (so a DEcreasing scroll carries the bands upward).
        public var rainbowScroll: Float = 0
        /// 0 = smooth ramp; > 0 = that many discrete swatches.
        public var rainbowSteps: Int = 0
        /// Samples per ray through the column (TAA accumulates the jitter).
        public var steps: Int = 24

        public init(top: SIMD3<Float>, axis: SIMD3<Float>, length: Float,
                    topRadius: Float, bottomRadius: Float, front: Float,
                    solidRadiance: SIMD3<Float>) {
            self.top = top
            self.axis = axis
            self.length = length
            self.topRadius = topRadius
            self.bottomRadius = bottomRadius
            self.front = front
            self.solidRadiance = solidRadiance
        }
    }


    // ── Exterior tone levers — the blown-sunlit-ground family ───────────────
    //
    // Auto-exposure meters the frame's GEOMETRIC MEAN, which answers "how bright
    // is this scene" and not "will anything clip". A frame that is half shaded
    // interior and half sunlit exterior — an architectural cutaway, a room shot
    // through an open wall — meters low, gets pumped to lift the shaded half, and
    // pushes the sunlit half past the ACES shoulder into textureless white. The
    // mean cannot see that: the pixels it blows are the ones it averaged away.
    //
    // Every knob here defaults to an EXACT no-op, so no existing scene moves.




    /// The last histogram the estimator published, or nil if it has never built one.
    /// Valid after the command buffer that encoded it has completed.
    public var lastExposureHistogram: ExposureHistogram? {
        let n = ExposureHistogram.bins
        let p = pfx.exposureHistogramBuffer.contents().assumingMemoryBound(to: Float.self)
        let total = p[n]
        guard total > 0 else { return nil }
        return ExposureHistogram(counts: Array(UnsafeBufferPointer(start: p, count: n)), total: total,
                                 p50: p[n + 1], p95: p[n + 2], p99: p[n + 3],
                                 meanLogLum: p[n + 4], keyLogLum: p[n + 5])
    }











    /// The profile in force: derived from the convention unless a host overrode it.
    var effectiveShadowRasterBias: ShadowRasterBias {
        shadowRasterBiasOverride ?? (shadowStoresFrontFaces ? .frontFace : .off)
    }

    /// The cull mode a shadow pass starts in, so the first group does not have to correct it.
    var shadowInitialCullMode: MTLCullMode { shadowStoresFrontFaces ? .back : .front }

    // MARK: - The substrate (a surface other surfaces are laid flush ON)



    /// The snap reach the traced passes receive, in depth ULPs (0 ⇒ no probe).
    var effectiveRTSurfaceSnapULPs: Float {
        rtSurfaceSnapEnabled && !substrateMeshKinds.isEmpty ? max(0, substrateDepthBias) * 2 : 0
    }


    /// Up to this many shadow-CASTING spots in the `spotLights` array get a shadow map each
    /// frame; the rest render as direct-only (no shadow modulation). This is the host's
    /// `spotShadowCapacityRequest` — the atlas grows to meet it lazily (`ensureSpotShadowAtlas`),
    /// so it is the CAP, not the allocation; the allocation is `spotShadowAtlasSliceCount`.
    public var spotShadowMaxCount: Int { spotShadowCapacityRequest }
    /// **How many spot cones may carry a shadow map (DH-0134).** Default 8 — the shipped live
    /// footprint (8 × 512² × 4 B = 8 MiB), byte-identical for every host that never sets it.
    /// A still has no vsync budget, so a host's photo/export lane asks for
    /// `spotShadowCapacityHardMax` and every casting cone in the scene is shadowed; the live lane
    /// asks for the default again on the way out. The atlas reallocates only when a caster
    /// actually needs a slice it does not have, and shrinks back when the request drops, so
    /// nothing is paid up front and nothing lingers (~1 MiB per slice). Clamped to
    /// `1...spotShadowCapacityHardMax`.
    public var spotShadowCapacityRequest: Int = IlluminatoramaRenderer.spotShadowCapacityDefault {
        didSet {
            let clamped = min(max(1, spotShadowCapacityRequest), Self.spotShadowCapacityHardMax)
            if clamped != spotShadowCapacityRequest { spotShadowCapacityRequest = clamped }
        }
    }
    /// Slices currently ALLOCATED in the spot-shadow atlas — `spotShadowCapacityDefault` until a
    /// host asks for more and a caster needs it. Diagnostics / gates; the cap is `spotShadowMaxCount`.
    public var spotShadowAtlasSliceCount: Int { shd.spotShadowAtlasCapacity }


    /// Up to this many shadow-casting point lights get a depth cube each frame
    /// (the N nearest to the camera; the rest fall back to unshadowed). Bounded by
    /// the cube atlas capacity. 4 cubes × 6 faces × 512² × 4 B ≈ 24 MB (allocated
    /// lazily only when `pointShadowsEnabled` first turns on).
    public var pointShadowMaxCount: Int { shd.pointShadowCubeCapacity }



    /// **What the look-match's AO share actually mixes toward THIS frame (Daydream DH-0785).**
    /// `liveLookAODarken` is a STAND-IN for ray-traced AO on a lane that never traces it, and it
    /// reads the same AO buffer RTAO writes into. When RTAO is live for the frame the buffer
    /// already carries real occlusion, so the stand-in would darken every corner a second time —
    /// on exactly the settled frames a host shows while it converges. 1.0 makes the shader's
    /// `mix(1, darken, aoMask)` a no-op. The GI share is untouched: GI is a separate term, and a
    /// host that traces AO without GI still wants it.
    public var effectiveLiveLookAODarken: Float { rtaoActive ? 1.0 : liveLookAODarken }


    // ── Phase 4.39 denoiser knobs — SPATIAL and TEMPORAL are separate ───
    //
    // These were one flag each (`ssaoDenoiseEnabled` / `ssrDenoiseEnabled`), which
    // made the only way to stop the temporal half's ghosting a fallback to RAW AO.
    // They fail differently and are now gated independently. The compatibility
    // shims below keep pre-split hosts compiling and behaving as they did.

    /// True when the AO spatial pass must be ENCODED — either because it was asked
    /// for, or because the temporal pass needs `aoFilteredTexture` as its input.
    var ssaoSpatialRuns: Bool { ssaoSpatialEnabled || ssaoTemporalEnabled }
    /// Compatibility shim for the pre-split single AO knob. Reads true if either
    /// half is on; writing sets BOTH, preserving old all-or-nothing behaviour.
    @available(*, deprecated, message: "Split — use ssaoSpatialEnabled (default on) and ssaoTemporalEnabled (default off).")
    public var ssaoDenoiseEnabled: Bool {
        get { ssaoSpatialEnabled || ssaoTemporalEnabled }
        set { ssaoSpatialEnabled = newValue; ssaoTemporalEnabled = newValue }
    }
    /// Compatibility shim for the pre-split SSR knob. SSR's chain is temporal-only,
    /// so this is a straight alias of `ssrTemporalEnabled`.
    @available(*, deprecated, message: "Renamed — SSR's denoise is temporal-only; use ssrTemporalEnabled.")
    public var ssrDenoiseEnabled: Bool {
        get { ssrTemporalEnabled }
        set { ssrTemporalEnabled = newValue }
    }




    // ── Spatial supersample (SSAA) ───────────────────────────────────
    /// Internal rendering scale. Internal targets (G-buffer, depth, HDR,
    /// SSAO, SSR composite, TAA history) are sized `outputW * scale ×
    /// outputH * scale`; the tonemap kernel downsamples to the
    /// output-sized `outputTexture` with a 4-tap bilinear filter.
    ///
    /// 1.0 = no SSAA (the image is whatever the lighting kernel writes).
    /// 1.5 = 2.25× the per-frame GPU cost in exchange for proper
    /// geometric anti-aliasing without temporal accumulation. Apple
    /// Silicon's tile memory absorbs much of the bandwidth hit on the
    /// G-buffer pass; the lighting kernel is the dominant cost.
    ///
    /// **Values BELOW 1.0 render sub-native and upscale on present** — the
    /// draft/navigation lever. A deferred frame at this quality tier is almost
    /// purely fill-bound (measured on an architectural document: ≈4 ms fixed +
    /// 2.3 ms per internal megapixel), so the pixel count is the one knob that
    /// moves the whole frame. 1.5 → 1.0 on a 2560×1440 canvas measured
    /// 22.9 ms → 14.2 ms (−38 %). The tonemap is the only pass that bridges
    /// internal → output and it samples with a `filter::linear`,
    /// `clamp_to_edge` sampler in NORMALISED coordinates, so it magnifies as
    /// correctly as it minifies; nothing downstream assumes internal ≥ output.
    /// Clamped to `internalRenderScaleRange` (0.5 … 4.0) — see
    /// `minInternalRenderScale` for why 0.5 is the floor.
    ///
    /// Assigning this **reallocates the internal render targets immediately**
    /// (it re-runs `resize` at the current output size) and resets every
    /// temporal accumulator, exactly as an output resize does. That is a real
    /// cost — see `resize` for the measured numbers — so hosts that flip
    /// between a draft and a photoreal tier must debounce the transition
    /// (drag-start / settle), never drive it per frame. `resize` preserves
    /// `outputTexture`'s identity across an SSAA-only change, so no host
    /// rebinding is needed.
    public var internalRenderScale: Float = IlluminatoramaRenderer.defaultInternalRenderScale {
        didSet {
            // Write the clamped value back so the property always READS as what
            // the pipeline is actually running at. Assigning a property inside
            // its own `didSet` does not re-enter the observer, so this cannot
            // recurse — and the resize below still runs, using `clamped`.
            let ceiling = min(Self.maxInternalRenderScale,
                              Self.canvasBudget.maxInternalRenderScale ?? Self.maxInternalRenderScale)
            let clamped = min(ceiling,
                              max(Self.minInternalRenderScale, internalRenderScale))
            if clamped != internalRenderScale { internalRenderScale = clamped }
            // `oldValue` is always already clamped (this observer maintains that
            // invariant), so this comparison is apples-to-apples.
            guard clamped != oldValue, outputWidth > 0, outputHeight > 0 else { return }
            // Re-derive the internal targets at the new scale. `resize` is a
            // no-op when the rounded internal dims land on the same numbers.
            resize(width: outputWidth, height: outputHeight)
        }
    }

    public nonisolated static var internalRenderScaleRange: ClosedRange<Float> {
        minInternalRenderScale...maxInternalRenderScale
    }

    /// Process-wide canvas ceiling. See `CanvasBudget`.
    public static var canvasBudget: CanvasBudget = .unlimited

    /// Set to `true` when the sky content has changed meaningfully so the
    /// IBL cubes (irradiance + prefiltered specular) are rebaked next frame.
    /// The renderer also auto-rebakes after `iblRebakeInterval` seconds have
    /// elapsed, regardless of this flag, to catch slow cloud drift.
    ///
    /// The host is responsible for setting this whenever sun direction,
    /// cloud coverage, or any other IBL-visible sky parameter changes.
    /// Calling `markIBLDirty()` is equivalent to `iblNeedsRebake = true`.
    /// Desaturate the sky equirect when baking the irradiance cube (0 = off,
    /// 1 = greyscale). Set per-scene via `IlluminatoramaRTOptions.iblBakeDesaturation`
    /// for scenes whose sky is near-monochromatic (broilers, vivid sunsets).
    /// Marks IBL dirty on change so the next bake uses the new value.
    public var iblBakeDesaturation: Float = 0.0 {
        didSet { if oldValue != iblBakeDesaturation { iblNeedsRebake = true } }
    }



    /// TEST-OBSERVABLE (DH-0869): the specialisation flags of the lighting variant the LAST
    /// frame actually dispatched — bit 1 ibl · 2 shadow · 4 dfg · 32 rtSun.
    /// `0xFF` until a specialised variant has run. On the non-blocking path a not-yet-compiled
    /// combination runs the init-time variant and leaves this at the PREVIOUS value, so a test
    /// that flips a flag can assert the frame really ran it.
    public var activeLightingVariantFlags: UInt8 { gbs.lastLightingFlags }



    /// The baked split-sum DFG LUT (RG16F, `x` = F0 scale, `y` = bias, keyed on
    /// (NdotV, roughness) over (0,1]). Exposed read-only so a gate can measure the
    /// integral the GPU actually produced rather than re-deriving it on the CPU —
    /// storage is `.private`, so a reader must blit it to a shared texture.
    public var dfgLUTTexture: MTLTexture { ibls.dfgLUT }

    // ── Output ────────────────────────────────────────────────────────────────

    /// Final LDR texture, ready to bind to `SCNMaterial.diffuse.contents`.
    /// This is the *presented* buffer — the one a consumer (SceneKit's
    /// `background.contents`) may sample. It is NEVER the buffer the current
    /// frame is tonemapping into; the render loop only promotes a pool buffer
    /// to `outputTexture` after that buffer's command buffer has *completed*.
    /// The host should NOT mutate this texture.
    public internal(set) var outputTexture: MTLTexture


    /// **Latency-tolerant pipelining.** The adaptive frames-in-flight guard drops to ONE frame
    /// in flight once the GPU frame passes ~36 ms, because a second queued frame on a heavy scene
    /// only adds a frame of INPUT latency to a drag. A host whose camera is not driven by input —
    /// an autonomous flythrough, a recorder, a screensaver — has no such latency to protect, and at
    /// 30–45 ms GPU frames the single-frame depth quantises delivery to the 60 Hz tick (a 37 ms
    /// frame shows every 50 ms). `true` keeps two frames in flight regardless of GPU time.
    /// Default `false` = the adaptive guard, byte-identical for every existing host.
    public var pipelineLatencyTolerant: Bool = false {
        didSet { pfx.presentSync.setLatencyTolerant(pipelineLatencyTolerant) }
    }
    /// GPU time (ms) of the most recently COMPLETED frame — a host-side hitch hunt without the
    /// pass profiler (per-frame, thread-safe; 0 before the first frame completes).
    public var lastGPUFrameMs: Double { pfx.presentSync.lastGpuMs }
    /// #60 task 4 — occupancy-tuning override for the RT lighting pass's
    /// threadgroup total thread count (`VIZ_ILLUMI_RT_TG`, e.g. 64/128/256). nil =
    /// the full threadgroup (default). Reproducible tuning hook for the
    /// `IlluminatoramaPassTimer` sweep.
    ///
    /// MEASURED (surfaceCacheProbe, per-pass timer, all VERIFIED): the full
    /// threadgroup is already optimal — rtLightingTLAS ≈ 184 ms flat for
    /// 256/512/1024, and WORSE at 128 (186) / 64 (192). The kernel is bound by
    /// ray-tracing work/latency, not occupancy, so it wants max threads in flight
    /// to hide latency. Conclusion: leave the default. The override stays so the
    /// sweep can be re-run on other hardware (the answer can differ per GPU).
    lazy var rtThreadgroupMax: Int? = {
        if let s = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RT_TG"],
           let v = Int(s), v > 0 { return v }
        return nil
    }()

    deinit {
        // Balance the in-flight semaphore before it can be deallocated: the
        // adaptive latency guard may be HOLDING a permit (depth 2→1 on heavy
        // frames), and a DispatchSemaphore deallocated below its creation value
        // traps ("Semaphore object deallocated while in use") — a crash that
        // surfaces at a random later point in the host process. See
        // `IlluminatoramaPresentSync.retire`.
        pfx.presentSync.retire(semaphore: gbs.inFlightSemaphore)
    }

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // ── Internal state ────────────────────────────────────────────────────────

    let engine: SimEngine






    // ── Secondary-ray shading parity: three ABLATION switches ───────────────
    // A SECONDARY ray's opaque hit — refraction or reflection out of the glass pass,
    // a glossy reflection or GI bounce out of the deferred TLAS kernel — is shaded
    // with the same inputs the deferred pass uses for a directly-visible surface:
    // the surface's texture, the local point/spot lights, and the interior fill
    // separation, instead of a per-instance mean colour under exterior-strength sky.
    // Each is separately suppressible so a harness can attribute an image-quality
    // number to ONE of them rather than to all three at once. Production leaves all
    // three on; turning one off restores exactly the pre-parity behaviour for that term.
    //
    // The `glassHit` prefix is historical — glass was simply where the divergence was
    // FOUND. Since the shading body was unified into `shadeSecondarySurface`
    // (IlluminatoramaSecondary.h), these gate every secondary path, which is the only
    // way an ablation number stays meaningful.




    /// Grows on demand in `encodeGlassPass` to hold the flattened glass instances.
    lazy var glassInstanceBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaGlassInstance>.stride,
                          options: .storageModeShared)!
    }()
    lazy var glassRTUniformBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaGlassRTUniforms>.stride,
                          options: .storageModeShared)!
    }()

    lazy var causticUniformBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaCausticUniforms>.stride,
                          options: .storageModeShared)!
    }()

    /// 1×1 stand-in bound to both kernels whenever the hand-off is off — the lighting kernel
    /// skips its write when the target is not full-size, the RT kernel skips its read on the flag.
    lazy var specIBLDummyTexture: MTLTexture? = {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        let t = device.makeTexture(descriptor: d); t?.label = "Illuminatorama.rt.specIBL.dummy"
        return t
    }()

    lazy var diffSkyDummyTexture: MTLTexture? = {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        let t = device.makeTexture(descriptor: d); t?.label = "Illuminatorama.rt.diffSky.dummy"
        return t
    }()
    /// Whether the traced GI will replace a deferred diffuse share this frame (the sky, or with
    /// `rtPathTracedGI` the whole diffuse indirect). The frame uniforms read it to set
    /// `kFrameFlagPathOwnsIndirect`; `diffSkyHandoffTarget` allocates on it.
    var diffSkyHandoffActive: Bool {
        (rtGIReplacesDiffuseSky || rtPathTracedGI) && rtGIStrength > 0 && rtOpaqueLightingEnabled
            && rtEnabled && rtTLASSupported && debugTerm == .normal
    }


    // Phase 4.0 — diffuse-albedo texture atlas, bound at G-buffer fragment
    // shader `texture(0)`. Always allocated; instances opt-in per draw by
    // setting `instance.albedoTextureSlice >= 0`. The extractor populates
    // it during `extractFrame`, falling back to per-instance solid albedo
    // when registration fails (atlas full, image conversion failed, etc.).
    public let albedoAtlas: IlluminatoramaTextureAtlas

    // Phase 4.1 — non-colour material atlas (metallic / roughness / normal),
    // bound at G-buffer fragment shader `texture(1)`. Same shape as the
    // albedo atlas but `.bgra8Unorm` (no sRGB), so values are linear and
    // the shader reads them as direct material parameters.
    public let nonColorAtlas: IlluminatoramaTextureAtlas

    /// Read-only instrument (Daydream DH-0964): the sun cascades' CURRENT light view-projections
    /// (sub-texel jitter included) and the raw depth array they rasterise into, so a host can dump
    /// the shadow map itself instead of inferring it from the lit frame (the DH-0862 method).
    public var debugCascadeViewProjections: [simd_float4x4] { shd.cascadeVPs }
    public var debugShadowMapTexture: MTLTexture { shd.shadowMap }

    /// Per-slice resolution of the spot/area shadow atlas.
    ///
    /// **This is the number that decides whether a chair leg has a shadow.** A slice covers a
    /// whole cone's footprint, so at 512² a pendant lighting ~4 m of floor gets ~7.8 mm per texel
    /// — and a 20 mm furniture leg is 2.5 texels wide, which the 3×3 PCF then averages down to
    /// nothing. The seat above it (~400 mm, 50 texels) casts cleanly, so the object appears to
    /// hover: a shadow blob with no legs reaching it. Daydream Home reported this for months as
    /// "there are no contact shadows" (DH-0601/0441/0640/0709).
    ///
    /// Settable so the host can spend memory where its scenes need it: cost is
    /// `resolution² × 4 bytes × capacity` (512/8 = 8 MB, 1024/8 = 34 MB, 2048/8 = 134 MB), plus
    /// a depth pass per slice — which the static-scene skip already elides on a settled camera.
    /// Clamped to a power of two in [256, 4096]; reallocates the atlas on change.
    public var spotShadowMapResolution: Int = 512 {
        didSet {
            let want = min(4096, max(256, spotShadowMapResolution))
            if want != spotShadowMapResolution { spotShadowMapResolution = want; return }
            guard want != oldValue else { return }
            reallocateSpotShadowAtlas(resolution: want, capacity: shd.spotShadowAtlasCapacity)
        }
    }

    var pointShadowFaceBuffer: MTLBuffer? {
        gbs.useBufferA ? shd.pointShadowFaceBufferA : shd.pointShadowFaceBufferB
    }

    var currentAOHistoryTexture: MTLTexture { aos.aoHistoryToggle ? aos.aoHistoryB : aos.aoHistoryA }
    var previousAOHistoryTexture: MTLTexture { aos.aoHistoryToggle ? aos.aoHistoryA : aos.aoHistoryB }

    /// Does SSAO contribute anything this frame? At `ssaoIntensity <= 0` all
    /// three SSAO kernels are per-pixel early-outs that write the constant 1.0
    /// (see `illumi_ssao` / `illumi_ssao_spatial` in Illuminatorama.metal) — the
    /// dispatches still ran, still bound the depth + normal G-buffer targets and
    /// still paid their encoder overhead, to produce a value known on the CPU.
    /// Measured 1.7 ms of a 22.9 ms frame at 3840×2160 for output nobody uses.
    /// Mirrors the identical `ssrIntensity > 0` gate the SSR chain already has.
    var ssaoActive: Bool { ssaoIntensity > 0 }

    /// Do the SSAO dispatches actually get encoded this frame? Normally the
    /// negation of "SSAO is off", but if the neutral texture could not be
    /// allocated the chain runs anyway so the AO the lighting pass reads is
    /// still correct (the kernels' own early-outs write 1.0). Read by all three
    /// SSAO encoders AND by the end-of-frame history toggle, so the skip can
    /// never leave the ping-pong half-advanced.
    var ssaoChainEncodes: Bool { ssaoActive || aos.aoNeutralTexture == nil }

    var currentSSRHistoryTexture: MTLTexture { ssrs.ssrHistoryToggle ? ssrs.ssrHistoryB : ssrs.ssrHistoryA }
    var previousSSRHistoryTexture: MTLTexture { ssrs.ssrHistoryToggle ? ssrs.ssrHistoryA : ssrs.ssrHistoryB }

    var rtGINeedsFirstFrame: Bool = true {
        // DH-1344 — every GI-history reset also clears the radiance cache. Its own flag, because the
        // history flag is never lowered while the temporal pass is off (an unfiltered still), which
        // would wipe the cache every frame and it would never be read.
        didSet { if rtGINeedsFirstFrame { rts.rcNeedsClear = true } }
    }
    var currentRTGIHistoryTexture: MTLTexture { rts.rtGIHistoryToggle ? rts.rtGIHistoryB : rts.rtGIHistoryA }
    var previousRTGIHistoryTexture: MTLTexture { rts.rtGIHistoryToggle ? rts.rtGIHistoryA : rts.rtGIHistoryB }
    /// The buffer the RT denoise reads: the temporally-accumulated GI when the
    /// pass is on, else the raw RT diffuse (original behaviour).
    var rtDiffuseDenoiseSource: MTLTexture {
        (rtGITemporalEnabled && rtEnabled && rtSupported)
            ? currentRTGIHistoryTexture : rts.rtDiffuseTexture
    }

    /// The SVGF toggle after applying the env A/B override. Use this in the
    /// render path; `svgfEnabled` stays the plain stored property the panel sets.
    var effectiveSVGFEnabled: Bool { Self.svgfEnvOverride ?? (svgfEnabled || svgfStillForced) }
    /// The GI history is irradiance this frame: only where the SVGF cascade is what consumes it (the
    /// bilateral fallback would composite it un-remodulated) and the history is a static mean.
    var giIrradianceMode: Bool {
        rtGIStaticAccumulation && svgfDemodulateAlbedo && effectiveSVGFEnabled && rtGITemporalEnabled
    }
    var effectiveSVGFLumFloor: Float { Self.svgfLumFloorEnv ?? svgfLumFloor }
    var effectiveSVGFLevels: Int { Self.svgfLevelsEnv ?? (svgfStillForced ? svgfStillLevels : svgfLevels) }

    // ── Phase 4.39: denoiser source selectors ────────────────────────
    // These let the lighting and composite passes bind the right texture
    // depending on whether the per-signal denoiser is on.
    var aoSourceTexture: MTLTexture {
        // SSAO skipped this frame ⇒ read the neutral 1.0 source, never a stale
        // history. Falls back to the normal path if the neutral texture could
        // not be allocated (correctness over the saving — `encodeSSAOPass` runs
        // the real passes in that case too).
        if !ssaoActive, let neutral = aos.aoNeutralTexture { return neutral }
        // AO v2 — the accumulated, denoised traced mean, when it produced this frame's AO.
        if aoAccumulatorOwnsAO, let accumulated = aos.aoAccumulator?.outputTexture { return accumulated }
        // Three-way, not two: before the split, "denoise off" fell all the way back
        // to the RAW AO, so escaping the temporal half also cost the spatial filter.
        // `aoFilteredTexture` is the spatial-only result and is the right source
        // whenever the spatial pass ran and the temporal one did not.
        if ssaoTemporalEnabled { return currentAOHistoryTexture }
        if ssaoSpatialRuns     { return aos.aoFilteredTexture }
        return aos.aoTexture
    }
    var ssrDenoisedTexture: MTLTexture {
        ssrTemporalEnabled ? currentSSRHistoryTexture : ssrs.ssrRawTexture
    }


    /// `superquadricParamBuffer` resolves to the CURRENT frame's ring slot, so the
    /// ~40 bind sites that name it need no change.
    var superquadricParamBuffer: MTLBuffer { gbs.superquadricParamRing[lts.frameRingIndex] }

    /// TEST-OBSERVABLE: the auto-exposure multiplier the tonemap applied on the last completed
    /// frame (slot 1 of the shared `ExposureState` buffer). It is the scale a host has to divide
    /// out to express anything composited *before* the tonemap in displayed brightness — and the
    /// control any such claim needs, since a test that never checks this number cannot tell an
    /// exposure-invariant result from two scenes that happened to be exposed the same.
    public var lastAutoExposure: Float {
        pfx.exposureBuffer.contents().advanced(by: 4).assumingMemoryBound(to: Float.self).pointee
    }
    /// The local adaptation's lift ceiling (stops) on the latest completed frame — the eye's
    /// adaptation state when `localToneMapping.adaptationFromFrame` is on (the frame's exposure
    /// deficit past the tolerance, × gain) — or −1 when that is off or the pass has not run. A
    /// shared-buffer load, no wait. For hosts that grade the print (e.g. a night shadow lift) on
    /// the same state the lift uses, instead of on an independent ramp.
    public var lastLocalAdaptationCeiling: Float {
        guard localToneMapping.isEnabled, localToneMapping.adaptationFromFrame else { return -1 }
        return localToneMapPass.lastLiftCeiling
    }
    /// TEST-OBSERVABLE: the log2 luminance the meter TARGETED on the last completed frame
    /// (slot 2 of `ExposureState`) — before the EMA and the boost clamp, so two meters can be
    /// compared even when both are pinned at `autoExposureMaxBoost` (DH-0655).
    public var lastAutoExposureTargetLogLum: Float {
        pfx.exposureBuffer.contents().advanced(by: 8).assumingMemoryBound(to: Float.self).pointee
    }

    /// Placeholder bound to the `surfCards` slot (buffer 9) when the surface cache
    /// is OFF. The lighting kernel never reads it then, but Metal argument
    /// validation still requires the bound buffer to be ≥ one `SurfCard` — and the
    /// generic dummies (`instData` / `triA`) can be smaller than a 192-B card
    /// (e.g. a single-instance RT buffer), which asserts at dispatch. Sized to
    /// exactly one card so the cache-off path validates regardless of struct size.
    lazy var surfCardDummyBuffer: MTLBuffer? =
        device.makeBuffer(length: MemoryLayout<SurfCard>.stride, options: .storageModePrivate)

    var surfGPUDiffStatsBuffer: MTLBuffer? {
        scs.surfGPUDiffStatsRing.isEmpty ? nil : scs.surfGPUDiffStatsRing[lts.frameRingIndex % scs.surfGPUDiffStatsRing.count]
    }
    /// `triFlags` bit layout — mirrors the Metal SC_FLAG_* defines.
    static let scFlagFrameB: UInt32 = 1
    static let scFlagDeforming: UInt32 = 2
    /// The atlas the lighting/RT pass should READ this frame (current). Set by
    /// `encodeSurfaceCacheUpdate` after the ping-pong swap.
    var surfCacheCurrentAtlas: MTLTexture? { scs.surfUseAtlasA ? scs.surfAtlasA : scs.surfAtlasB }
    var surfCachePreviousAtlas: MTLTexture? { scs.surfUseAtlasA ? scs.surfAtlasB : scs.surfAtlasA }
    var surfCacheCurrentIndirectAtlas: MTLTexture? { scs.surfUseAtlasA ? scs.surfIndirectAtlasA : scs.surfIndirectAtlasB }
    var surfCachePreviousIndirectAtlas: MTLTexture? { scs.surfUseAtlasA ? scs.surfIndirectAtlasB : scs.surfIndirectAtlasA }
    /// Phase 5 / B1 — the atlas the GI/reflection CONSUMERS read: the denoised
    /// copy when the cache-domain filter is on (and allocated), else the raw
    /// current INDIRECT atlas (DH-0622). The update kernel's feedback always reads
    /// the raw full atlas.
    var surfConsumerAtlas: MTLTexture? {
        (surfaceCacheDenoise ? scs.surfAtlasDenoised : nil) ?? surfCacheCurrentIndirectAtlas
    }

    /// Whether the device + instanced-RT pipeline are available.
    public var rtTLASSupported: Bool { rts.rtTLASPipeline != nil }
    /// TEST-OBSERVABLE: live RT cache entries. Can never exceed `registeredMeshCount`.
    public var rtBLASCacheCount: Int { rts.rtMeshCache.count }
    /// TEST-OBSERVABLE: how many meshes are currently registered with the renderer.
    public var registeredMeshCount: Int { meshes.count }

    /// TEST-OBSERVABLE: the live registry's KEYS, rendered as strings.
    ///
    /// `registeredMeshCount` can say a host is leaking meshes but not WHICH ones, and a count is
    /// where that investigation stalls: a mesh is evicted only by `IlluminatoramaMeshHandle.deinit`,
    /// so "the count grew" leaves every registration path a suspect. Diffing the key set across one
    /// rebuild names the survivors outright — and the key format itself distinguishes the two ways
    /// a mesh can outlive its scene (a `gpuMesh#UUID` key means a handle is still retained
    /// somewhere; any other key means the mesh was inserted outside the handle contract and nothing
    /// can ever evict it).
    ///
    /// Added 2026-08-11 for the ~231 MB-per-scene-rebuild leak (docs/MEMORY_FOOTPRINT.md).
    public var registeredMeshKeyDescriptions: [String] { meshes.keys.map { String(describing: $0) } }

    /// TEST-OBSERVABLE: how many per-frame GPU repack tasks are live.
    ///
    /// A repack task is the heavier half of a compute-fed mesh — it retains the caller's position,
    /// normal, UV and colour buffers plus the renderer's own interleaved vertex and previous-position
    /// buffers. `registeredMeshCount` can look healthy while these accumulate, because they are
    /// evicted on a separate line of `removeMesh`. Exposed alongside the mesh count so a host can
    /// tell "the mesh table is balanced" from "the memory behind it is".
    public var gpuRepackTaskCount: Int { fxs.gpuRepackTasks.count }

    /// Dev A/B override (VIZ_ILLUMI_NO_BLAS_REFIT=1) — freeze deforming BLASes at
    /// build pose to compare RT-traces-stale vs current. Default off (refit on).
    static let noBLASRefit =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_BLAS_REFIT"] == "1"
    /// Dev A/B override (VIZ_ILLUMI_NO_PRIM_UV=1) — skip baking per-mesh card UVs
    /// into BLAS `primitive_data` (#60 item 6), so every surface-cache hit falls
    /// back to the `triUVa`/`triUVc` side buffers (the pre-fold "before"). Lets the
    /// AS-baked-UV win be measured/compared without a stash. Default off (fold on).
    static let noPrimUVFold =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_PRIM_UV"] == "1"
    /// Whether to bake per-mesh card UVs into BLAS `primitive_data` (#60 item 6):
    /// per-triangle surface-cache path, cache on, fold not disabled. Evaluated at
    /// BLAS-build time; if the cache turns on only AFTER a mesh's BLAS was built,
    /// that BLAS keeps no primitive data and the shader's null-check falls back to
    /// the side buffers (correct, just no perf win for that mesh).
    var bakePrimUVFold: Bool {
        surfaceCacheEnabled && !Self.surfCacheCharts && !Self.noPrimUVFold
    }

    var rtAutoDisabled: Bool { rts.rtGuard.autoDisabled }

    /// Read-only view of the RT hang-guard for hosts and profilers. A silently
    /// auto-disabled TLAS changes what a measured frame MEANS (no RT glass, no
    /// RT GI), so this must be inspectable rather than inferred from pixels.
    public var rtGuardState: (autoDisabled: Bool, trip: IlluminatoramaRTGuardTrip?,
                              tripCount: Int, framesUntilRetry: Int?) {
        (rts.rtGuard.autoDisabled, rts.rtGuard.trip, rts.rtGuard.tripCount, rts.rtGuard.framesUntilRetry)
    }
    /// Diagnostic override for the mesh-group cap (`VIZ_ILLUMI_RT_GROUPCAP`, e.g. 512):
    /// lets a harness MEASURE a scene past the shipped cap — what the first TLAS build
    /// actually costs at N groups, and the steady per-frame tax — before anyone decides
    /// whether the cap itself should move. Mirrors `VIZ_ILLUMI_RT_TG`: an instrument,
    /// not a shipped setting; nil (unset/invalid) = the shipped cap. Read per-renderer
    /// (not a static latch) so one test process can measure both arms.
    lazy var rtMeshGroupCapOverride: Int? = {
        if let s = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RT_GROUPCAP"],
           let v = Int(s), v > 0 { return v }
        return nil
    }()
    /// Diagnostic override for the glass-only TRIANGLE ceiling (`VIZ_ILLUMI_RT_TRICAP`, e.g.
    /// 4000000) — the triangle-axis sibling of `VIZ_ILLUMI_RT_GROUPCAP`: lets a harness measure
    /// what the TLAS rebuild and the steady frame cost past the shipped ceiling. An instrument,
    /// not a shipped setting; nil (unset/invalid) = the shipped ceiling. The live-RT 150 k cap is
    /// never overridden.
    lazy var rtGlassOnlyTriangleCap: Int = {
        if let s = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RT_TRICAP"],
           let v = Int(s), v > 0 { return v }
        return Self.rtMaxTrianglesForGlassOnlyRT
    }()


    /// What the surface cache would card on the current instance set (DH-1314): the
    /// instance-EXPANDED triangles of meshes that take cards — the number the bake compares with
    /// `surfaceCacheMaxTriangles` — and those of meshes that opted out (`surfaceCacheCards =
    /// false`). `carded + excluded` = `frameDrawCensus.expandedTriangles`.
    public var surfaceCacheTriangleCensus: (carded: Int, excluded: Int) {
        var carded = 0, excluded = 0
        for g in gbs.meshGroups {
            guard let m = meshes[g.kind] else { continue }
            let t = g.count * (m.indexCount / 3)
            if m.surfaceCacheCards { carded += t } else { excluded += t }
        }
        return (carded, excluded)
    }

    let cloudShadow = CloudShadowState()
    let cumulusInView = CumulusInViewState()
    lazy var cloudUpsamplePipeline: MTLComputePipelineState? =
        engine.pipelineCache.pipelineState(name: "illumi_cloud_upsample", device: device)
    /// The in-view kernel with the multiple-scattering sky (built on first use).
    lazy var cloudInViewPipelineMS: MTLComputePipelineState? =
        engine.pipelineCache.pipelineState(name: "illumi_cloud_inview_ms", device: device)
    /// The two in-view kernels with the host's aerosol as the Mie medium
    /// (`VolumetricCloudRenderer.Params.atmosphereAerosol` ≠ `.builtin`; built on first use).
    lazy var cloudInViewPipelineAerosol: MTLComputePipelineState? =
        engine.pipelineCache.pipelineState(name: "illumi_cloud_inview_aerosol", device: device)
    lazy var cloudInViewPipelineMSAerosol: MTLComputePipelineState? =
        engine.pipelineCache.pipelineState(name: "illumi_cloud_inview_ms_aerosol", device: device)

    var pointLightBuffer: MTLBuffer { shd.pointLightRing[lts.frameRingIndex] }
    var spotLightBuffer: MTLBuffer { shd.spotLightRing[lts.frameRingIndex] }
    var areaLightBuffer: MTLBuffer { lts.areaLightRing[lts.frameRingIndex] }
    /// DH-0718 — how many area lights a secondary hit reads this frame (0 ⇒ none bound).
    /// Gated by the local-lights switch too: that is the ablation lever every secondary
    /// light term already answers to.
    var secondaryAreaLightCount: Int {
        (secondaryAreaLightsEnabled && glassHitLocalLightsEnabled) ? areaLights.count : 0
    }
    var extraDirectionalBuffer: MTLBuffer { lts.extraDirectionalRing[lts.frameRingIndex] }

    var interiorApertureBuffer: MTLBuffer { lts.interiorApertureRing[lts.frameRingIndex] }
    var frameUniformBuffer: MTLBuffer { lts.frameUniformRing[lts.frameRingIndex] }

    var currentInstanceBuffer: MTLBuffer { gbs.useBufferA ? gbs.instanceBufferA : gbs.instanceBufferB }
    var previousInstanceBuffer: MTLBuffer { gbs.useBufferA ? gbs.instanceBufferB : gbs.instanceBufferA }

    /// Internal rendering size. All compute kernels (G-buffer rasterisation,
    /// SSAO, lighting, SSR, TAA, bloom mid-textures) dispatch over this
    /// resolution. Equals `outputWidth × internalRenderScale` (rounded).
    public internal(set) var width: Int
    public internal(set) var height: Int

    /// Final output size — drives the `outputTexture` allocation and the
    /// tonemap kernel's dispatch. The tonemap is the ONE kernel that
    /// bridges from internal to output: it samples the internal HDR /
    /// bloom textures with a downsample filter and writes one output
    /// pixel per thread.
    public internal(set) var outputWidth: Int
    public internal(set) var outputHeight: Int


    // ── Init ──────────────────────────────────────────────────────────────────

    /// - Parameter shadowMapResolution: Side length (px) of each directional
    ///   (sun) cascade shadow map. Default `2048` (behaviour-preserving for
    ///   existing hosts). Doubling to `4096` halves the outer-cascade ground
    ///   texel — finer, less-aliased sun-shadow edges — at 4× shadow-map memory
    ///   per doubling. Safe for PCF: the lighting shader reads its texel size
    ///   from `shadowMap.get_width()`, so no shader change is needed.
    public init(engine: SimEngine, width: Int, height: Int, camera: IlluminatoramaCamera,
                shadowMapResolution: Int = 2048) throws {
        // Stopwatch for the init breakdown. Kept in a LOCAL and assigned to
        // `self.initTiming` at the very end — mutating a stored property while the
        // rest of `self` is still uninitialised isn't allowed, and a local also keeps
        // the marks out of every early-throw path.
        var timing = IlluminatoramaInitTiming()
        self.engine = engine
        self.device = engine.device
        self.commandQueue = engine.commandQueue
        self.shd.shadowMapResolution = shadowMapResolution
        // `width` / `height` are the OUTPUT (host-visible) target size; the
        // internal pipeline runs at `output × initialScale`. We can't read
        // `self.internalRenderScale` before its memberwise default applies, so
        // read the SAME static the property's default expression reads — one
        // source, no literal to keep in sync.
        //
        // The process-wide `canvasBudget` is applied HERE, before a single
        // target is allocated — a host that can't afford the scene's authored
        // canvas must never allocate it, not shrink it afterwards. Unlimited by
        // default, so this is a no-op for hosts that haven't opted in.
        let budgeted = Self.budgeted(outputW: width, outputH: height,
                                     scale: Self.defaultInternalRenderScale)
        let initialScale = budgeted.scale
        self.outputWidth  = budgeted.w
        self.outputHeight = budgeted.h
        // Must stay HERE, while `self` is still being initialised (`width` /
        // `height` / `camera` are not assigned yet): in that phase Swift does
        // not run property observers, so this records the budgeted scale
        // WITHOUT the `didSet` calling `resize` on a half-built renderer. Moving
        // it below the last stored-property assignment would do exactly that.
        self.internalRenderScale = initialScale
        let (iw, ih) = Self.internalDims(outputW: self.outputWidth,
                                          outputH: self.outputHeight,
                                          scale: initialScale)
        self.width = iw
        self.height = ih
        self.camera = camera

        guard let library = engine.library else {
            throw IlluminatoramaError.libraryMissing
        }

        // ── Pipelines + resources ───────────────────────────────────────
        // Each family builds its own pipelines/resources in its own file (`build…` below, all
        // in the extension files). The ORDER here is the order `init` always ran them in.
        try Self.buildGBufferPipelines(into: gbs, device: device, library: library)
        try Self.buildShadowPipeline(into: shd, device: device, library: library)
        Self.buildGlassPipelines(into: gls, device: device, library: library)
        timing.mark("render pipelines (gbuffer/shadow/glass)")

        // Compute pipelines via the shared cache so we don't recompile across
        // scene rebuilds.
        let cache = engine.pipelineCache
        Self.precompileInitPipelines(cache: cache, device: device)
        timing.mark("compute pipeline precompile (32, concurrent)")

        try Self.buildLightingPipeline(into: gbs, cache: cache, device: device,
                                       ibl: kLighting.iblEnabled, shadow: kShadow.shadowsEnabled, dfg: kLighting.dfgLUTEnabled)
        try Self.buildAOPipelines(into: aos, cache: cache, device: device)
        try Self.buildSSRSSSPipelines(into: ssrs, cache: cache, device: device)
        try Self.buildGIDenoisePipelines(into: rts, cache: cache, device: device)
        try Self.buildBakePipelines(into: ibls, cache: cache, device: device)
        try Self.buildPostPipelines(into: pfx, cache: cache, device: device, library: library)
        timing.mark("compute pipeline fetch (lighting/ssao/ssr/svgf/bloom)")

        self.kRT.rtSupported = try Self.buildRTPipelines(into: rts, cache: cache, device: device, library: library)
        try Self.buildSurfaceCachePipelines(into: scs, cache: cache, device: device, library: library)
        timing.mark("RT + surfcache + TLAS pipelines (specialized)")

        Self.buildDOFPipelines(into: dofs, device: device, library: library)
        try Self.buildVolumetricPipelines(into: vols, device: device, library: library)
        try Self.buildExposureBuffers(into: pfx, device: device)
        try Self.buildParticlePipelines(into: fxs, cache: cache, device: device, library: library)
        Self.buildSelectionPipelines(into: outl, cache: cache, device: device, library: library)
        timing.mark("dof/vol/cloud/particle/streak/selection pipelines")

        // ── Phase 3 cubemaps + dummy sky, DFG LUT, colour LUT, LTC ──────
        try Self.buildCubes(into: ibls, device: device)
        timing.mark("IBL cubes + dummy sky")
        let dfg = try Self.makeDFGTexture(into: ibls, device: device)
        try Self.buildIdentityColorLUT(into: pfx, device: device)
        timing.mark("DFG texture + identity colour LUT")
        Self.buildLTC(into: ibls, device: device, dfg: dfg)
        timing.mark("LTC area-light LUTs")

        self.gbs.passTimer = IlluminatoramaPassTimer(device: device)

        Self.encodeDFGBake(dfg: dfg, dfgBake: ibls.dfgBakePipeline, commandQueue: commandQueue)
        timing.mark("DFG bake dispatch")

        // ── Phase 2.5 shadow map array ──────────────────────────────────
        try Self.buildShadowMaps(into: shd, device: device, sunResolution: shadowMapResolution,
                                 spotResolution: spotShadowMapResolution)
        timing.mark("shadow maps (sun cascades + spot atlas)")

        // ── Phase 4.0/4.1 texture atlases ───────────────────────────
        //
        // Phase 4.25 — sliceSize 256 → 512. The original 256² slice
        // collapsed high-aspect or high-frequency textures (wood grain,
        // bark, fine fabric weaves) into near-uniform averages because
        // the upload pass squishes a 1024×4096 source into a 256²
        // destination, losing 16× of vertical detail. 512² preserves
        // recognisable bark grain + tile patterns; VRAM goes from 16 MB
        // to 64 MB per atlas at default 64 capacity — still well under
        // the per-app texture budget on every Apple Silicon GPU.
        let atlasSliceSize = 512
        self.albedoAtlas = try IlluminatoramaTextureAtlas(
            device: device, pixelFormat: .bgra8Unorm_srgb,
            sliceSize: atlasSliceSize)
        self.nonColorAtlas = try IlluminatoramaTextureAtlas(
            device: device, pixelFormat: .bgra8Unorm,
            sliceSize: atlasSliceSize)
        timing.mark("texture atlases (albedo + non-colour)")

        // ── Targets ──────────────────────────────────────────────────────
        let t = try Self.makeTargets(device: device,
                                      internalW: self.width, internalH: self.height,
                                      outputW: self.outputWidth, outputH: self.outputHeight)
        self.outputTexture = t.ldr
        try installTargets(t)
        timing.mark("render targets (internal \(self.width)×\(self.height) + LDR pool)")

        // ── Buffers ──────────────────────────────────────────────────────
        try buildBuffersAndRings()
        timing.mark("buffers, rings + primitive meshes")

        buildCoverBlitPipeline(library: library)
        timing.mark("cover-blit pipeline")
        timing.finish()
        self.initTiming = timing

        Self.log.info("Illuminatorama renderer ready (internal \(self.width)x\(self.height), output \(self.outputWidth)x\(self.outputHeight))")
    }

    /// Byte stride of a `packed_float3` — the layout of both the host position
    /// buffer and the issue-#65 `prevPositionBuffer` side copy. NOT SIMD3<Float>'s
    /// 16-byte stride; the GPU repack kernel + motion-vector vs read `packed_float3`.
    static let gpuMeshPrevStride = MemoryLayout<Float>.size * 3

    lazy var localToneMapPass = IlluminatoramaLocalToneMapPass(engine: engine)
    lazy var hdrDump = IlluminatoramaHDRDump()


    var currentHistoryTexture: MTLTexture { pfx.historyToggle ? pfx.historyB : pfx.historyA }
    var previousHistoryTexture: MTLTexture { pfx.historyToggle ? pfx.historyA : pfx.historyB }

}


/// Env-gated pass bisector for GPU-cost audits: `VIZ_PERF_AB=nodof,nogi,nort,nosc,novol,noshadow,noglass,noref,notaa`
/// force-disables the named pass every frame (after the scene's own per-frame setup), so a
/// `VIZ_ILLUMI_GPUMS_PATH` p50 delta attributes cost to that pass. Inert when unset.
struct IlluminatoramaPerfAB: Sendable {
    let flags: Set<String>
    init() {
        let v = ProcessInfo.processInfo.environment["VIZ_PERF_AB"] ?? ""
        flags = Set(v.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
    }
    @MainActor func apply(to r: IlluminatoramaRenderer) {
        if flags.isEmpty { return }
        if flags.contains("nodof") { r.dofEnabled = false }
        if flags.contains("nogi") { r.rtGIStrength = 0 }
        if flags.contains("nort") { r.rtEnabled = false; r.rtReflectionsEnabled = false; r.rtGlassEnabled = false }
        if flags.contains("nosc") { r.surfaceCacheEnabled = false }
        if flags.contains("novol") { r.volumetricEnabled = false; r.volumetricRTEnabled = false }
        if flags.contains("noshadow") { r.shadowsEnabled = false }
        if flags.contains("noglass") { r.rtGlassEnabled = false }
        if flags.contains("noref") { r.rtReflectionsEnabled = false }
        if flags.contains("notaa") { r.taaEnabled = false }
    }
}
