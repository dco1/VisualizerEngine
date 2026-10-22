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
    /// Per-term AOV capture.
    let aovs = AOVState()

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

    public var instances: [InstanceRef] = []

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
    public internal(set) var staticSkipStats = StaticSkipStats()
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
    ///                           actually rasterise, not the per-mesh sum.
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
            if let m = meshes[g.kind], !m.tracedOnly { tris += g.count * (m.indexCount / 3) }
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
    /// A/B lever for the per-slice reuse above (default on). Off = every slice re-renders whenever
    /// GPU-fed geometry is live, the pre-DH-0631 behaviour.
    public var spotShadowGPUGeometryOptOutEnabled: Bool = true

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

    /// Opt-in FRAME-COMPUTE hook (additive, nil elsewhere — every other host is byte-identical).
    /// Called once per frame that actually renders — never for a tick `render()` drops because
    /// `maxFramesInFlight` frames are already queued — with the frame's own command buffer, before
    /// any pass (and before the RT acceleration-structure refit) reads a mesh. A host that writes
    /// its own GPU geometry each frame (skinning a `LinearBlendSkinMesh`, an `ArcTubeBatch`) encodes
    /// it here instead of committing a command buffer of its own every tick: at a 30 fps GPU frame
    /// and a 60 Hz tick a per-tick side buffer is encoded twice for every frame drawn, queued behind
    /// the frames in flight, and holds command-buffer slots on the renderer's queue (Digital Clock,
    /// VZ-0233). Same ordering on the GPU as a side buffer committed just before the frame.
    public var onEncodeFrameCompute: ((_ cb: MTLCommandBuffer) -> Void)?

    public var pointLights: [IlluminatoramaPointLight] = []
    public var spotLights: [IlluminatoramaSpotLight] = []
    /// Rectangular LTC area lights (#60 task 5). Populated by the extractor from
    /// SCN `.area` lights; consumed by `illumi_lighting` (buffer 4).
    public var areaLights: [IlluminatoramaAreaLight] = []
    /// Secondary directional lights (#60 task 5 — retires the 4.20 ambient fold).
    /// The PRIMARY directional is `directionalLightDirection`/`directionalLightColor`
    /// (cascade-rigged); these are the fill/back directionals a SCN 3-point rig
    /// ships alongside it. Populated by the extractor; consumed by `illumi_lighting`
    /// (buffer 5) with a real BRDF (no shadow).
    public var extraDirectionals: [IlluminatoramaDirectionalLight] = []
    public var camera: IlluminatoramaCamera
    public var directionalLightDirection: SIMD3<Float> = simd_normalize(SIMD3(0.5, 1.0, 0.3))
    public var directionalLightColor: SIMD3<Float> = SIMD3(1.0, 0.97, 0.92) * 3.0
    public var ambientColor: SIMD3<Float> = SIMD3(0.10, 0.13, 0.18)
    /// Leaf thin-sheet transmission strength (issue #58). 0 = OFF (default) — the
    /// foliage back-light term is skipped entirely unless a scene opts in. Drives
    /// both the deferred lighting kernel (`FrameUniforms.leafTransmission`) and the
    /// RT sun pass (`RTUniforms.leafTransmission`, which owns the sun under RT).
    /// Foliage fragments are flagged by tagging leaf vertices with colour alpha 0.
    public var leafTransmission: Float = 0
    /// Plush material strengths (Teddy Bear Press). `plushSheen` = fur retro-
    /// reflective edge sheen; `plushTransmission` = backlit thin-fabric SSS. Both
    /// 0 = OFF (default) → exact no-op for every scene. Only the plush-flagged
    /// teddy mesh (`normalRoughness.w` ≈ 0.55h) reads them. Mirror the deferred
    /// `FrameUniforms.plushSheen/plushTransmission`.
    public var plushSheen: Float = 0
    public var plushTransmission: Float = 0
    /// Lamp-shade fabric translucency strength (DH-0458). 0 = OFF (default) → the
    /// shade thin-sheet transmission branch is skipped entirely. A paper/linen drum
    /// shade is a thin translucent sheet; this drives the back-lit transmission that
    /// lets external light (a sunset through the glazing, the room behind the shade)
    /// scatter through the fabric rather than landing flat on an opaque cone. Only
    /// the shade-flagged mesh (`normalRoughness.w` ≈ 0.62) reads it. Mirrors the
    /// deferred `FrameUniforms.shadeTransmission`.
    public var shadeTransmission: Float = 0
    public var exposure: Float = 1.0
    public var bloomThreshold: Float = 1.0
    public var bloomIntensity: Float = 0.6
    /// Bloom soft knee, as a FRACTION of `bloomThreshold` (S1.2). 0 restores the
    /// hard `max(0, lum - T)` cut byte-for-byte; the curve above T·(1+knee) is
    /// identical either way, so this only softens the pop as a light crosses the
    /// bar. Default 0.5 — half a stop of ramp, the Unity/Karis convention.
    public var bloomSoftKnee: Float = 0.5
    /// Upsample scatter: the CONVEX blend weight toward the blurred low mip on
    /// the way up the pyramid. 0 leaves only the finest level (a tight rim), 1
    /// only the coarsest (a formless wash). Level i ends up weighted
    /// (1-s)·s^i — a geometric series that sums to exactly 1 at ANY level count,
    /// so changing this changes the halo's SHAPE and never its total energy.
    /// Default 0.7.
    public var bloomScatter: Float = 0.7
    /// **Lens point-spread bloom (Daydream DH-0992).** 0 (default) ⇒ the geometric up-chain above,
    /// byte-identical. > 0 ⇒ the pyramid's per-level weights are set so the halo's surface
    /// brightness falls as a POWER LAW, I(r) ∝ r^−α with α = this value — the long-tailed glare
    /// spread of a real lens (a tight core over a faint skirt that keeps going), where the
    /// geometric chain is an exponential that dies. Each level reaches 2× the radius of the one
    /// before, so an octave's share of the energy is ∝ r^(2−α); the weights are normalised, so the
    /// total (and `bloomIntensity`'s meaning) is unchanged. Real lenses sit near α ≈ 2.5–3.
    public var bloomPSFExponent: Float = 0
    /// **Physical glare (Daydream DH-1001).** false (default) ⇒ the bloom above, byte-identical.
    /// true ⇒ the pyramid IS a physical glare spread function (`IlluminatoramaGlareSpread`; which one
    /// is `bloomGlareModel` — a camera lens by default since DH-1009, the CIE 146:2002 eye before):
    /// no threshold, no Karis weighting, level weights solved against the GSF in this frame's own
    /// angular units (the camera projection's pixel angle — so the halo scales with focal length and
    /// resolution), and the scattered fraction η taken OUT of the direct image (the tonemap forms
    /// `hdr·(1−η) + η·bloom`). Overrides `bloomThreshold`, `bloomSoftKnee`, `bloomIntensity`,
    /// `bloomScatter` and `bloomPSFExponent` while on. The tent radius must stay 1.0 (the measured
    /// kernels are for it). DH-0998: past ~0.8 of the pyramid's reach the glare is the far-field
    /// tail (`illumi_bloom_glare_tail`, an exact angular convolution over a coarse down level), so
    /// it follows the standard to the frame corner instead of stopping a few degrees out.
    public var bloomPhysicalGlare: Bool = false
    /// Whose glare it is (Daydream DH-1009): a camera lens's surface scatter (`.cameraLens`, the
    /// default — Ritt et al. 2020's generic double-Gauss, η ≈ 0.025) or the CIE 146:2002 eye
    /// (`.standardObserver` or `.eye(age:pigment:)`, η ≈ 0.30). Both run the same fit + far tail.
    public var bloomGlareModel: IlluminatoramaGlareSpread.Model = .cameraLens
    /// TEST-OBSERVABLE: the glare fit the last frame used (nil ⇒ physical glare was off).
    public internal(set) var lastBloomGlareFit: IlluminatoramaGlareSpread.Fit?
    lazy var bloomGlareTailPipeline: MTLComputePipelineState? =
        engine.pipelineCache.pipelineState(name: "illumi_bloom_glare_tail", device: device)

    // ── Digital sensor noise (Daydream DH-0992) ───────────────────────
    /// Shot + read noise of a photo-electron count, in the exposed scene signal before the
    /// display transform (see the tonemap's sensor block). 0 (default) ⇒ off, byte-identical;
    /// 1 ⇒ the physical amount for `sensorElectronsPerUnit` and `sensorReadNoise`.
    public var sensorNoiseStrength: Float = 0
    /// Photo-electrons per unit of exposed scene radiance at the camera's own ISO, BEFORE the
    /// meter's push (which the shader counts as extra gain). A full-frame sensor at ISO 100 puts
    /// mid-grey (0.18) near 5 000 e⁻ — ~28 000 per unit; divide by ISO/100 for a faster setting.
    public var sensorElectronsPerUnit: Float = 28_000
    /// Read noise, electrons (a modern CMOS sensor: ~2–5).
    public var sensorReadNoise: Float = 3
    /// How much of the noise the three channels share (demosaicing correlates neighbours' noise
    /// across channels); 0 = independent RGB speckle, 1 = pure luminance noise.
    public var sensorChannelCorrelation: Float = 0.5
    /// The capture medium's EMULSION grain (Daydream DH-1000): a relative σ of the exposed signal
    /// — constant in log exposure, which is what "constant in density" means for a negative — added
    /// in the sensor block as a second variance term beside the shot + read noise, on ONE draw, so
    /// the two are a single noise with a single σ (they cannot stack). 0 (default) ⇒ the sensor
    /// block is exactly DH-0992's; with `sensorNoiseStrength` also 0 the block never runs. The
    /// grain is drawn on the `filmGrainSize` cell grid while this is on. Independent of the legacy
    /// encoded-domain `filmGrainStrength`, which a host using this should leave at 0.
    public var filmDensityGrain: Float = 0
    /// How much of the emulsion grain the three dye layers share (0.80: the DH-0880 figure).
    /// Per pixel, the draw's correlation is the variance-weighted mix of this and
    /// `sensorChannelCorrelation`, so each mechanism keeps its own colour character.
    public var filmDensityGrainCorrelation: Float = 0.80
    /// 3×3 tent radius on the up-chain, in LOW-mip texels. 1.0 is the plain tent;
    /// larger values widen each level's contribution without adding taps.
    public var bloomTentRadius: Float = 1.0
    /// Screen-space SUN lens flare strength in the tonemap pass (ghost train +
    /// halo + anamorphic streak, optically occluded by sampling the HDR frame at
    /// the sun's screen position). 0 = OFF (the default for every scene) → an
    /// exact shader no-op. 1.0 is the intended "on" look; values between scale it.
    public var lensFlareIntensity: Float = 0
    /// TEST-OBSERVABLE: the lens-flare uniform cluster as computed for the last frame
    /// (x = strength, yz = sun screen uv, w = on-screen weight). Lets a host's GPU test
    /// verify the sun projection without reading the uniform buffer back.
    public internal(set) var lastLensFlareParams: SIMD4<Float> = .zero
    /// Lens-style transverse chromatic aberration strength in the tonemap pass.
    /// 0 = OFF (the default for every scene) → an exact shader no-op. Maps into
    /// the trailing `_padPlush0` slot of `IlluminatoramaFrameUniforms`, so the
    /// struct stride is unchanged.
    public var chromaticAberration: Float = 0
    /// Longitudinal / axial chromatic aberration ("purple fringing") strength in
    /// the tonemap pass. 0 = OFF → an exact shader no-op. `fringeTint` is the
    /// dark-side halo colour (linear-ish sRGB components); the bright side gets
    /// its complement. New 16-byte uniform cluster, mirrored byte-for-byte.
    public var fringe: Float = 0
    public var fringeTint: SIMD3<Float> = SIMD3(0.62, 0.12, 0.92)
    /// Spherical-aberration radial blur strength in the tonemap pass.
    /// 0 = OFF (default) → exact shader no-op. Maps into the former
    /// `_padPlush1` slot of `IlluminatoramaFrameUniforms` (stride unchanged).
    public var sphericalAberration: Float = 0
    /// Lens vignette (issue #65): `vignetteStrength` is the corner-darkening amount
    /// (0 = OFF → exact shader no-op) and `vignetteExtent` the normalised radius
    /// kept fully bright before the falloff begins. Film-stock grain:
    /// `filmGrainStrength` is the amplitude (0 = OFF → exact no-op) and
    /// `filmGrainSize` the grain cell size in output px. NEW 16-byte uniform
    /// cluster, mirrored byte-for-byte. Applied directly (not eased) — strength
    /// ramps are gentle enough not to need the glide the blur knobs use.
    public var vignetteStrength: Float = 0
    public var vignetteExtent: Float = 0.55
    public var filmGrainStrength: Float = 0
    public var filmGrainSize: Float = 1.5

    // ── Diagram look ─────────────────────────────────────────────────────────
    /// Cross-fade into the architectural-drawing look, 0…1. **0 is OFF and an exact
    /// no-op**: the tonemap branch never runs, so every existing scene renders
    /// byte-identically. 1 replaces the shaded composite with flat, hemisphere-wrapped
    /// G-buffer albedo plus screen-space outlines.
    ///
    /// This is a POST pass over the G-buffer, not a second forward path — it costs one
    /// extra branch in a fragment shader that already has albedo, normal and depth
    /// bound. The passes it makes pointless (bloom, DoF, grain, film LUT) are not
    /// disabled here: a host that wants the cheap version turns them off itself, and
    /// leaving them independent keeps this one knob honest about what it does.
    ///
    /// Pair it with `camera.projection = .orthographic(...)` for a true axonometric.
    /// The two are deliberately separate — a flat-shaded perspective and a photoreal
    /// axonometric are both legitimate, and neither implies the other.
    public var diagramMix: Float = 0
    /// How much the flat fills vary with surface orientation, 0…1. 0 = perfectly flat
    /// (a white box is one white silhouette and only the outlines separate its faces);
    /// 1 = a full sky/ground hemisphere term. ~0.35 keeps planes legible without the
    /// image drifting back toward a render.
    public var diagramWrap: Float = 0.35
    /// Outline strength, 0…1. 0 = fills only, no line work.
    public var diagramOutlineStrength: Float = 1.0
    /// Outline colour (linear RGB). Near-black by default, as an ink line would be.
    public var diagramOutlineColor: SIMD3<Float> = SIMD3(0.10, 0.11, 0.13)
    /// How readily a DEPTH step becomes a line. Higher = more lines (silhouettes and
    /// near-coplanar overlaps both draw); lower = silhouettes only.
    public var diagramEdgeDepthSensitivity: Float = 1.0
    /// How readily a NORMAL change becomes a line. This is what draws the crease where
    /// two faces of the same object meet — a depth-only detector cannot see it.
    public var diagramEdgeNormalSensitivity: Float = 1.0
    /// Line thickness in OUTPUT pixels. The detector samples a cross at this radius, so
    /// non-integer values are legitimate; much above ~2 reads as a smudge, not a pen.
    public var diagramEdgeThickness: Float = 1.0
    /// Film **halation**: the warm halo real film wears around blown highlights.
    /// Light punches through the emulsion, reflects off the back of the base and
    /// re-exposes it from behind; what reaches the base is mostly red, hence the tint.
    /// A separate half-res threshold → dense four-scale isotropic gaussian chain from
    /// bloom (see `IlluminatoramaPost.metal` § Halation). 0 = OFF (the default for every scene): the halation
    /// passes are not even encoded and the tonemap branch is skipped, so non-opting
    /// scenes are byte-for-byte unchanged. ~0.4 is a tasteful film amount; 1.0 is strong.
    public var halationIntensity: Float = 0
    /// HDR luminance above which a highlight scatters into the halation halo. Only
    /// the excess above this threshold contributes, so the halo grows continuously
    /// out of the highlight rather than popping in.
    ///
    /// Deliberately LOWER than the usual `bloomThreshold` of 1.0. Exposure and the ACES
    /// curve are applied after this, and they put the visual "blown" point of a normally-
    /// exposed frame near HDR 0.6 — keying halation at 1.0 confines it to pixels that are
    /// already clipping, where adding a red halo only pushes them to white and the effect
    /// reads as a neutral second bloom. Measured on the Daydream fixture house: at 1.0 the
    /// tint barely survives (mean R−B moves +0.10 while luma moves +4.6); at 0.6 it reads
    /// as halation (R−B +1.56, and the added light lands in a ΔR:ΔG:ΔB ≈ 1 : 0.23 : 0.27
    /// ratio that matches `halationTint`).
    public var halationThreshold: Float = 0.6
    /// Halation halo radius as a FRACTION OF FRAME HEIGHT (2σ of the widest scale). Halation is a fixed size on the negative — set by
    /// the base thickness — so it must not depend on the render resolution: the old
    /// radius was in internal texels, and a 2× supersampled still got half the halo.
    /// The PSF is four gaussians an octave apart (σ = R/16 … R/2), a long soft tail.
    public var halationRadius: Float = 0.025
    /// Halation TAIL tint (linear RGB) — the deep red where only red survives the round
    /// trip through the base. The core is derived from it in the shader and reads orange.
    public var halationTint: SIMD3<Float> = SIMD3(1.0, 0.20, 0.07)
    /// Post-FX easing time constant (seconds) from the panel's "Easing" picker.
    /// The post-FX knobs above (exposure, bloom, chromatic aberration, fringe) are
    /// treated as TARGETS; each frame `uploadFrameUniforms` eases an internal
    /// shadow toward the target with `k = 1 − e^(−dt/postFXEasingTau)`, so dragging
    /// a slider glides instead of snapping. 0 = instant (no easing).
    public var postFXEasingTau: Double = 0.40
    /// **A host-owned frame clock for the renderer's SELF-TIMED integrators.** `nil` (the default,
    /// and what every live canvas wants) = each integrator measures its own `dt` off the wall clock
    /// (`CACurrentMediaTime()`), which keeps its half-life in SECONDS at any frame rate. Non-nil =
    /// every one of them advances by exactly this many seconds per `render()`, regardless of how long
    /// the frame actually took:
    ///
    ///  - the auto-exposure EMA (`encodeExposureEstimate` → `ExposureState.deltaTime`),
    ///  - the post-FX easing glide (`advancePostFXEasing`),
    ///  - the particle integration step (`encodeParticleStep`).
    ///
    /// Why it exists: a headless "settle N frames" is a frame COUNT, not a duration. With the wall
    /// clock, N frames land wherever machine load puts them along each glide — the auto-exposure
    /// meter at the shipped 0.25 s half-life is ~37 % converged after 10 frames at 60 fps and ~93 %
    /// at 10 fps — so a frame's GAIN was a function of how busy the machine was. `renderer.time`
    /// (the shader oscillators) is already host-owned; this is the other half. Set it to `1/60` for a
    /// deterministic capture and the same N frames produce the same frame on an idle or a loaded
    /// machine (Daydream Home DH-0043).
    public var fixedFrameDelta: Double? = nil
    /// The scene animation clock, in **seconds since the session started** — reaches the GPU as
    /// `FrameUniforms.time` and drives every shader oscillator (`applyTreeWind`, `applySway`
    /// modes 2/3 (pendant, wind flap), the shadow passes' `shadowClock.x`, `applyCurveWind`,
    /// the film-grain reseed) plus the `iblRebakeInterval` fallback.
    ///
    /// **It must be SESSION-relative, and it must stay small.** This is a `Float`, so its
    /// resolution is `time.ulp` — proportional to its own magnitude. A host that assigns a
    /// *boot*-relative clock (`CACurrentMediaTime()`, which is awake-uptime since boot) hands
    /// the GPU a clock whose ulp is 31.25 ms after four days of uptime: consecutive 60 fps
    /// frames narrow to the SAME float, so every oscillator judders in ~32 Hz steps, and the
    /// resolution halves again every time uptime doubles. Use `RenderClock` (which documents the
    /// full precision table and the rejected wrap-the-clock alternative) or an accumulated
    /// `elapsed` starting at 0; assign a pinned literal only for deterministic captures.
    ///
    /// `render()` warns ONCE on stderr if the value it is given is too coarse to animate.
    public var time: Float = 0
    /// Vertex-shader vegetation-wind knobs (#58 #1). `treeWindStrength` is the max
    /// canopy sway in ~metres (0 = no wind, an exact shader no-op); `treeWindHeading`
    /// is the wind/gust travel direction in radians. Per-vertex sway weights ride
    /// in the geometry's tangent channel — packed by tree bakes (ForestGeometry
    /// and derivatives) AND by `GrassFillerMesh` static blade cards (2026-08-16),
    /// so one strength drives canopies and the sward's filler together. Set by
    /// Forest, IlluminatoramaRoom, and Daydream Home's yard.
    ///
    /// **This is a global, and the tangent channel is not a vegetation marker** — every
    /// normal-mapped surface carries a tangent, so a nonzero strength alone would swing
    /// any mesh whose tangents point along the sway axis (Daydream Home's whole house did,
    /// while its ±X faces stayed planted, tearing meshes apart). The draw-side opt-in
    /// `IlluminatoramaInstance.windScale` is therefore mandatory: leave it 0 on every
    /// instance and this knob is inert no matter what you set it to. **Raising the strength
    /// without setting `windScale` on the foliage instances is the "wind does nothing"
    /// symptom** — check the instance, not this value.
    public var treeWindStrength: Float = 0
    public var treeWindHeading: Float = 0
    /// Foliage-wind motion-vector delta, in **seconds** — the simulated time elapsed between
    /// this frame and the previous one (DH-0492). The G-buffer samples the previous frame's
    /// `applyTreeWind` at `time − windPreviousDelta`, so a swaying canopy/blade writes a real
    /// screen-space velocity TAA can reproject instead of crawling between pixels. Leave it at
    /// 0 (the default) and the previous position is sampled at the SAME time as the current —
    /// the pre-DH-0492 no-op, so every scene that never sets it, and every settled capture
    /// (whose clock does not advance between accumulation frames), is byte-identical. The host
    /// feeds it the real per-frame clock delta ONLY on the live, advancing canvas; a frozen
    /// capture must leave it 0 (its previous == current time) to stay still.
    public var windPreviousDelta: Float = 0

    // ── Highlight outline (feathered halo around selected / hovered objects) ──
    /// When true, every box in `highlightMaskInstances` receives a feathered halo
    /// glow traced around its screen-space silhouette (mask → separable dilation →
    /// HDR composite before bloom). Two modes composite in separate passes:
    /// `highlight == 1` in `selectionOutlineColor`, `highlight == 2` in
    /// `hoverOutlineColor`.
    public var selectionOutlineEnabled: Bool = false

    /// Halo style for `highlight == 1` (selected) and `highlight == 2` (hovered).
    public var selectionOutline: HighlightOutlineStyle = .selection
    public var hoverOutline: HighlightOutlineStyle = .hover

    /// Bounding-box proxies drawn only in the halo mask pass (never the G-buffer),
    /// so highlighted elements keep their real material while gaining a halo. One
    /// solid box per selected/hovered element, its `highlight` set to 1 or 2.
    /// Boxes (not detailed meshes) guarantee a hole-free mask → a clean outer ring.
    /// The bridge clears and re-fills this each frame before `render()`.
    public var highlightMaskInstances: [IlluminatoramaInstance] = []

    // ── Hardware ray tracing (soft sun shadows + 1-bounce GI) ────────
    //
    // When `rtEnabled` is true AND a static RT geometry has been supplied via
    // `setRTGeometry`, an `illumi_rt_lighting` compute pass runs after the SSR
    // composite and ADDS hardware-ray-traced sun lighting into the HDR buffer.
    // The host is expected to turn the deferred directional light + cascaded
    // shadows OFF in that case (set `directionalLightColor = 0`,
    // `shadowsEnabled = false`) so the sun isn't double-counted — RT owns it.
    // IBL / SSAO still run for ambient. SSR also runs unless RT reflections
    // are on (see `rtReflectionsEnabled` below — that path zeroes SSR so RT
    // owns scene-geometry reflections). `rtSupported` reflects whether the
    // device + pipeline are available; when false the pass is skipped
    // regardless of `rtEnabled`.
    public var rtEnabled: Bool = false
    /// Decouples the expensive opaque-geometry RT lighting pass (`rtLightingTLAS`
    /// ≈ 18–50 ms — RT GI + soft shadows on the scene's *opaque* meshes) from
    /// `rtEnabled`. Default `true` keeps the historical behaviour: turning
    /// `rtEnabled` on runs that pass. Set `false` to keep the rest of the RT
    /// machinery live — the **surface cache** (`surfaceCacheEnabled`, which
    /// requires `rtEnabled`) and **glass caustics** that splat into it — while
    /// skipping the opaque RT-lighting pass. The opaque geometry then renders
    /// with the cheaper deferred lighting (same fallback the `rtEnabled = false`
    /// path already uses), and glass still refracts/reflects it via the TLAS.
    /// This is the "I want caustics, not RT GI on my floor" case (Glass Lab):
    /// caustics need the cache (hence `rtEnabled`), but were dragging the
    /// ~18 ms opaque pass along with them for no visible gain.
    public var rtOpaqueLightingEnabled: Bool = true
    /// When true, the scene extractor bakes a world-space triangle soup from
    /// the scene it walks and feeds it to `setRTGeometry` (dirty-rebuilt), so
    /// RT GI + soft shadows + reflections work on ANY extracted overlay scene —
    /// not just the hand-authored room that calls `setRTGeometry` directly.
    /// The extractor also points the RT sun at the scene's extracted key light.
    /// (Surface cache stays opt-in/manual — it needs per-card UVs the soup
    /// can't synthesise for arbitrary meshes.)
    public var buildRTFromExtractedScene: Bool = false
    /// Direction TOWARD the sun (normalised), used by the RT pass.
    public var rtSunDirection: SIMD3<Float> = simd_normalize(SIMD3(0.5, 1.0, 0.3))
    /// Sun radiance (pre-multiplied intensity) for the RT direct term.
    public var rtSunColor: SIMD3<Float> = SIMD3(1.0, 0.97, 0.92) * 6.0
    /// Sun angular radius in radians — penumbra softness. ~0.0047 rad is the
    /// real sun; larger values give softer contact shadows.
    public var rtSunSoftnessRad: Float = 0.02
    /// **S4.1 — ray-traced soft sun shadows in the DEFERRED lighting kernel.**
    /// When true (and a TLAS is live), `illumi_lighting`'s `kLightingRTSunShadow`
    /// pipeline variant REPLACES the cascaded-shadow-map sun visibility with
    /// `rtShadowRays` cone-sampled shadow rays across the sun's angular disc
    /// (`rtSunSoftnessRad`) — a penumbra that widens with occluder distance, which
    /// a fixed PCF footprint cannot produce. Everything else about the deferred
    /// sun (the full BRDF, contact shadows, transmission) is untouched: only the
    /// visibility estimate changes. Distinct from `rtSunOwnership = .rayTraced`,
    /// which moves the whole sun term into the cruder RT-instanced kernel.
    ///
    /// Default `false`: the pipeline bit stays 0, the non-RT variant is selected,
    /// and every existing host (all of Visualizer) renders byte-identically.
    /// Frame-graph note: `updateRTAccel` runs AFTER the lighting pass, so the
    /// kernel always traces the PREVIOUS frame's TLAS — fine for the settled
    /// photo lane this exists for; on frame 1 (no TLAS yet) the bit derivation
    /// (`rtSunShadowLightingActive`) falls back to the shadow-map variant.
    public var rtSunSoftShadowsEnabled: Bool = false
    /// Strength of the one-bounce indirect contribution.
    public var rtGIStrength: Float = 1.0
    /// **The traced GI REPLACES the deferred diffuse sky instead of adding a second one.**
    /// A GI ray that misses returns the sky, so the traced estimate already integrates the
    /// whole hemisphere; the deferred pass has ALSO lit the pixel with the sky's irradiance
    /// cube. Added together the sky is counted twice — measured on a sunlit exterior wall
    /// (Daydream Home, north side of 4000 Sunset): (96,101,64) → (129,145,116), a pale blue
    /// lift on every surface that sees sky. When true the deferred pass hands its diffuse
    /// OUTDOOR-cube share (never an interior band) to the traced pass, which subtracts it
    /// where its GI rays ran and scales its misses by `iblIntensity` so the sky-fill dial keeps
    /// its meaning. The diffuse sibling of DH-0896's `reflReplacesIBL`. Default `false`:
    /// byte-identical for every host that does not opt in.
    public var rtGIReplacesDiffuseSky: Bool = false
    public var rtGIOwnsRoomFill: GIRoomFillHandoff = []
    /// **Daydream DH-1320 — a one-bounce GI hit is lit by the host's room fill.** The one-bounce
    /// estimator shades a hit with its sun, lamps, apertures and emission but NO indirect of its
    /// own (`giBounceParams`), because the receiving pixel used to carry the room fill itself.
    /// Once that fill is handed to the traced GI (`rtGIOwnsRoomFill`), the hit needs its own
    /// incoming indirect or every second-and-later bounce goes missing. True: an uncached hit
    /// takes the full secondary fill estimate (`secondaryIndirectFill` on the full-radiance
    /// params — the same stand-in a non-resident surface-cache card falls back to), so the
    /// estimate is one traced bounce onto surfaces lit the way the deferred pass lights them
    /// (a final gather), and a cached hit keeps the cache's traced multi-bounce. Default false ⇒
    /// byte-identical. Ignored on the path lane.
    public var rtGIHitFill: Bool = false
    /// **The path-traced GI lane (Daydream DH-0989; DH-0642's reference is the same lane at a deep
    /// budget).** When true each of `rtGIRays` is a whole multi-bounce path — next-event estimation
    /// of the sun, the local lights and ONE window aperture per vertex, a cosine continuation,
    /// Russian roulette — and it owns the pixel's ENTIRE diffuse indirect: the deferred pass hands
    /// over its diffuse IBL (interior bands and room gains included), its ambient supplement and
    /// its area lights (this host's area lights are window portals, which the path samples as
    /// apertures onto the real sky), and the traced pass subtracts them where it ran. Implies the
    /// `rtGIReplacesDiffuseSky` hand-off. Default `false`: byte-identical for every host that does
    /// not opt in. A host that opts in should run `rtGIStrength` at 1 — the estimate is physical.
    public var rtPathTracedGI: Bool = false
    /// Continuation bounces per path (Russian roulette also ends paths from the third vertex).
    public var rtPathBounces: Int = 6
    /// Luminance cap on one path's estimate (scene-linear, before exposure); 0 ⇒ no clamp. A
    /// path through a sunlit patch seen past a doorway is the classic firefly; the cap trades a
    /// little energy in those paths for a still that converges in the frames it is given.
    public var rtPathClamp: Float = 0
    /// **DH-1011 — how a path vertex samples its lights.** True: the aperture sample and the
    /// cosine continuation are combined by multiple importance sampling (power heuristic) instead
    /// of partitioning the hemisphere — a continuation that leaves through a portal is kept,
    /// weighted, not dropped — which in turn lets the aperture PICK leave out the portals of other
    /// rooms (light-layer mask; the continuation carries their light at full weight); and the area
    /// emitters are ONE pick ∝ their unshadowed contribution with ONE Sobol shadow ray instead of
    /// `secondaryAreaShadowRays` white-noise rays each. Same integrand, so the converged picture
    /// does not move; only its variance does. False: the DH-0989 estimator, byte-identical (the
    /// A/B lever).
    public var rtPathLightSampling: Bool = true
    /// **DH-1011 — shadowed lamps at path vertices.** True: every point / spot light that casts a
    /// shadow in the deferred pass (`castsShadow` — the trapped cones of lamps, cans, sconces,
    /// pendants; candle cubes) gets a traced shadow ray at a path vertex too, and the vertex's
    /// lamps join its emitters in the ONE pick (so one light ray per vertex however many fixtures
    /// the room holds). False: lamps are the unshadowed deterministic fill at a path vertex, as
    /// DH-0989 shipped — which lights the floor under a table from the can above it. Needs
    /// `rtPathLightSampling`. Unshadowed lights (bounce fills, string lights) stay unshadowed.
    /// The ray stops `kPathLampShadowGap` (25 cm) short of the light so the fixture's own housing
    /// is not its occluder. Default TRUE since DH-1011: measured on Daydream's many-lights station
    /// at night it removes a −0.077-stop leak (frame level; 0.085 stops mean tile error; −0.021 by
    /// day) at +38–43 % path-pass GPU time, flat in the number of fixtures (Daydream DH-1292).
    public var rtPathLampShadows: Bool = true
    /// DH-1011 INSTRUMENTS, OR'd into the path flags — never set by a host. 128: the frame is
    /// replaced by the primary surface's traced local-light estimate alone (traced lamp visibility,
    /// as a ratio of two frames); 64: a continuation that hits an emissive surface adds nothing
    /// (the glowing shades' share of the noise, by its absence); 32 / 16 / 8: no local lights /
    /// no sun / no window apertures at the path's vertices (the noise budget by source).
    public var rtPathDebugFlags: UInt32 = 0

    // ── Daydream DH-1304 — the per-term AOV capture ──────────────────────────────────────────
    /// Capture every lighting term of every frame into its own running mean (`IlluminatoramaAOV`,
    /// `Shaders/IlluminatoramaAOV.h` for the set and the contract). DEBUG / MEASURE ONLY: while
    /// false — every host's default — no variant, no texture and no binding exist, and the lighting
    /// kernels are byte-for-byte the pre-DH-1304 code (every tag is behind a function constant).
    /// While true the frame runs the `-aov` variants of `illumi_lighting` and
    /// `illumi_rt_lighting_tlas` (compiled blocking on first use) and accumulates
    /// `aovLightingTexture` as a per-pixel mean over the frames since the last reset. The mean
    /// restarts whenever the GI running mean does (`resetRTGIAccumulation`, the still's held-out
    /// warm-up / cache-build frames) and on `resetAOVAccumulation()`. Σ(lighting slices) == beauty
    /// per pixel on every frame — `HouseRenderBridgeGPUTests_AOV` is the gate.
    /// Only the TLAS lane is tagged: the soup kernel (`illumi_rt_lighting`) and the isolation
    /// `debugTerm` views write no AOVs.
    public var aovCaptureEnabled: Bool = false
    /// The captured means: `texture2d_array`, rgba32Float, internal size, `IlluminatoramaAOV.sliceCount`
    /// slices (`IlluminatoramaAOV.slice`). Nil until a frame has rendered with the capture on. Private
    /// storage — `readAOV` blits it.
    public var aovLightingTexture: MTLTexture? { aovs.aovTexture }
    /// Frames in the current mean (0 ⇒ nothing captured since the last reset).
    public internal(set) var aovAccumulatedFrames: Int = 0
    /// Function-constant indices: `kLightingAOVEnabled` (illumi_lighting) / `kRTAOVEnabled` (TLAS kernel).
    static let lightingAOVFunctionConstantIndex = 8

    /// **DH-1293 INSTRUMENT — floor the lamps' `1/d²` at path vertices** (metres; 0 ⇒ off,
    /// byte-identical). A lamp / can / sconce reaching a path vertex closer than this lights it as
    /// if it were this far away. BIASED by construction — it removes the near-field energy it
    /// floors — so it answers "how much of a still's noise is vertices landing next to a bulb",
    /// and is never set by a host. Path vertices only: the primary's lamps are the deferred pass's.
    /// NEGATIVE ⇒ the exclusion form: a lamp adds nothing to a vertex closer than |value|.
    public var rtPathLampMinDistance: Float = 0
    /// **DH-1293 — a finite SOURCE size for the lamps at path vertices** (metres; 0 ⇒ off,
    /// byte-identical). The falloff becomes `1/(d² + r²)` — the exact on-axis irradiance of a
    /// luminous disc of radius r — combined with each light's own `softRadius` by max. A physical
    /// lever (a can's aperture, a shade's opening has size), default off until it earns its place.
    public var rtPathLampSourceRadius: Float = 0
    /// **DH-1295 — lamps have a physical size at path vertices.** True: each point / spot light's
    /// own `bulbRadius` (host-authored, metres) floors its `1/d²` at path vertices —
    /// `1/max(d² + softRadius², bulbRadius²)`, the law of a luminous sphere of that radius, exact
    /// outside the glass, and no surface is inside it. A light with `bulbRadius` 0 keeps the point
    /// law exactly, so a host that never authors one (Visualizer) is unaffected. False: every
    /// light is a point at path vertices — the shipped-before A/B lever. Path lane only (the
    /// primary's lamps are the deferred pass's; `pathOn` gates the flag), so the real-time lane and
    /// every non-Maximum still are byte-identical either way. Default TRUE: measured on Daydream's
    /// many-lights station (5 cm lamps), −40 % of a 16-spp still's variance at night (−39 % by day)
    /// for a converged-mean shift ≤ 0.001 stops (Daydream DH-1293 → DH-1295).
    public var rtPathLampBulbs: Bool = true
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
    /// Cone-sampled shadow rays per pixel per frame for `rtSunSoftShadowsEnabled`'s traced
    /// sun visibility, host-clamped (`rtSunShadowRayCount`'s write, below) and shader-clamped
    /// (`rtSunSoftVisibility`, IlluminatoramaLighting.metal) to 1…32. DH-0856 measured this
    /// alone does not close the gap to the cascade shadow map's edge quality at any practical
    /// budget — the Monte-Carlo estimator's noise floor drops sharply from 1 to 4 rays, then
    /// only slowly (single-digit percent per doubling) from 4 to 32, so raising the shipped
    /// default costs real per-frame GPU time for little further gain. Default 4, matching the
    /// budget every existing host already renders at.
    public var rtShadowRays: Int = 4
    /// **S4.3 — ray-traced PORTAL (area-light) visibility, in the same deferred-kernel variant
    /// as the soft sun shadows.** When > 0 and `rtSunSoftShadowsEnabled` is live, every
    /// shadow-casting area light (`shadowSliceIndex >= 0` — a host's window portals) has its
    /// single-perspective 512² PCF map REPLACED by this many shadow rays per pixel toward
    /// jittered points on the emitting rectangle, traced against the TLAS. Exact in every
    /// direction (a 150° perspective from the portal centre cannot see a niche beside the
    /// window), and a penumbra that scales with the portal's real size — the daytime interior
    /// shadow a room's dominant source has to cast. Per-frame Monte-Carlo like the sun rays,
    /// so it belongs to a still that accumulates. 0 (default) keeps the PCF path byte-identical.
    public var rtAreaShadowRays: Int = 0
    /// **S4.4 — a STILL may build the extracted-scene TLAS under the glass-arm caps.** The
    /// live-RT caps (128 mesh groups / 150 k triangles) exist because a live loop's first
    /// rebuild of a heavy scene never returns; a still that accumulates pays that rebuild
    /// once, off the vsync budget, so it can carry the same 512-group / 2.5 M-triangle
    /// ceiling the glass-only arm already trusts. A furnished document (200+ groups) is
    /// exactly what the live caps auto-disable RT GI on — which is why the shipped photo
    /// export never had bounce. Default `false`: every live host is byte-identical.
    public var rtStillCapsRelaxed: Bool = false

    // ── RTAO — ray-traced ambient occlusion on the near field (DH-0528) ────────────────
    /// When true AND a TLAS is live, the AO pass traces `rtaoRays` cosine-weighted hemisphere
    /// rays per half-res texel against the real geometry (`illumi_rtao_tlas`) instead of the
    /// screen-space GTAO march, writing the same raw AO texture into the same bilateral +
    /// temporal chain. The reach is a TRUE world-space distance, so one metre grounds a sofa
    /// foot and stays tight at a drywall junction. Per-frame Monte-Carlo: meant for a still
    /// that accumulates. Falls back to GTAO on any frame without a TLAS (`rtaoDidRunLastFrame`
    /// is the record). Default `false`: byte-identical for every host that never opts in.
    public var rtaoEnabled: Bool = false
    /// 0…1 AO strength, the `ssaoIntensity` contract.
    public var rtaoIntensity: Float = 1.0
    /// Occlusion reach in world metres (the ray `max_distance`).
    public var rtaoRadius: Float = 0.5
    /// Multi-bounce AO (Jimenez 2016 fit) on the lighting kernel's AO term. OFF by default => byte-identical.
    /// Rides `filmMedium.z`. DH-0953.
    public var aoMultiBounce: Bool = false
    /// Cosine-hemisphere rays per half-res texel per frame (clamped 1…32).
    public var rtaoRays: Int = 8
    /// Whether the LAST encoded AO pass was the traced one — a gate's proof that RTAO fired
    /// rather than silently falling back to the screen-space march.
    public var rtaoDidRunLastFrame: Bool = false

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
    /// Test-only: offsets the progressive sequence's streams so two otherwise identical settles
    /// sample independently (production sampling is deterministic). 0 = production.
    public var aoAccumulationScramble: UInt32 = 0
    /// Daydream DH-1410 — with `aoAccumulationEnabled` AND `taaStillAccumulation` (the averager), do not keep
    /// the AO's own running mean: each frame's occlusion is filtered (at the converged ½ strength) and handed
    /// to the lighting, and the colour resolve's frozen-camera average is the one mean — the AO twin of
    /// `rtGIStillPerFrame`. Default `false`: byte-identical for every host.
    public var aoStillPerFrame: Bool = false
    /// Frames in the current accumulated AO mean (0 = none; the per-frame path is in use).
    public var aoAccumulatedFrames: Int { aos.aoAccumulator?.frames ?? 0 }
    /// Whether THIS frame's AO came from the accumulator (the selector's switch).
    public var aoAccumulatorOwnsAO: Bool = false
    var rtaoActive: Bool {
        rtaoEnabled && rtaoIntensity > 0 && rtTLASSupported && rts.rtTLASActive && rts.rtTLAS != nil
    }
    public var rtGIRays: Int = 4
    /// Soft sun specular strength on the RT direct term.
    public var rtSpecStrength: Float = 0.25
    public internal(set) var rtSupported: Bool = false

    /// Who computes the sun's direct term while an RT lighting pass is running.
    /// See `RTSunOwnership`. Default `.automatic`.
    public var rtSunOwnership: RTSunOwnership = .automatic

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

    // ── RT diffuse denoiser ──────────────────────────────────────────
    // The RT pass writes its noisy diffuse (soft shadow + 1-bounce GI) term to
    // its own buffer; this depth+normal-guided bilateral cleans it BEFORE it
    // reaches the HDR composite + TAA, so temporal accumulation gets a far less
    // noisy per-frame input (the visible "dithered" grain under orbit is the
    // raw ~4-spp estimate showing through when TAA history is rejected by
    // motion). The sharp reflection term is composited un-filtered. Off → the
    // raw diffuse is composited straight through (pre-split additive behaviour).
    public var rtDenoiseEnabled: Bool = true
    /// Bilateral half-window in pixels (full-res). 2 ⇒ 5×5; 3 ⇒ 7×7.
    public var rtDenoiseRadius: Int = 2
    /// Depth edge-stop sensitivity (larger = tighter boundary preservation).
    public var rtDenoiseDepthSensitivity: Float = 3.0
    /// Normal edge-stop exponent (larger = more sensitive to curvature).
    public var rtDenoiseNormalSharpness: Float = 16.0
    /// Temporal accumulation of the RT diffuse (1-bounce GI + soft shadow) term
    /// BEFORE the spatial denoise. Velocity-reprojected exponential history that
    /// keeps the GI converging under camera motion — targets the wall "crawl"
    /// the main TAA only fixes when static. Low-frequency term, so it accumulates
    /// hard. Needs `rtEnabled`.
    ///
    /// Driven by `IlluminatoramaSharedSettings.rtGITemporalEnabled` (edited from
    /// the Illuminatorama settings window, ⌘⇧L) — host controllers copy it in
    /// each tick. This renderer-level default is just the fallback for hosts
    /// that never set it. Tune via `rtGITemporalBlend` / the kernel clamp+ramp.
    public var rtGITemporalEnabled: Bool = false
    /// Current-frame weight in steady state. 0.06 ≈ a 16-frame exponential
    /// window. Smaller = cleaner/slower to react; larger = noisier/faster.
    public var rtGITemporalBlend: Float = 0.06
    /// Daydream DH-0887 — STILL mode for the GI accumulator: with the camera frozen (a photo, the
    /// settled canvas's refine) average every frame's GI as an unbiased running mean (α = 1/N,
    /// no clamp) instead of the moving-camera EMA, which forgets samples and clamps toward each
    /// noisy frame. The host must reset (`resetTemporalHistory`, or toggle `rtGITemporalEnabled`)
    /// when the picture changes. Default `false`: byte-identical for every other host.
    public var rtGIStaticAccumulation: Bool = false
    /// Daydream DH-1388 / DH-1386 — with `rtGIStaticAccumulation`, do NOT keep the GI's own running mean:
    /// hand every frame's own (albedo-demodulated) sample to the à-trous cascade and let the colour
    /// resolve's frozen-camera average be the ONE mean. Otherwise the composite of frame k carries
    /// F(G_k), G_k = mean(g_1..g_k), and the colour average of those composites weights GI sample i by
    /// (H_K − H_{i−1})/K (harmonic) — ~K/2 effective GI samples in a K-frame still instead of K. The
    /// cascade is linear in its input (still lanes have no luminance edge-stop), so filter-then-average
    /// equals average-then-filter; only the weights change. Needs `taaStillAccumulation` to be the
    /// averager. Default `false`: byte-identical for every host.
    public var rtGIStillPerFrame: Bool = false
    /// Daydream DH-0887 — STILL mode for the colour TAA resolve: the camera is frozen, so average
    /// (current weight `taaHistoryBlend`) with NO neighbourhood clip, read the history at the same
    /// pixel instead of re-sampling it, and take the display sharpen from the accumulation rather
    /// than the current noisy frame. See `illumi_taa_resolve`. Only valid while the camera and the
    /// scene are frozen — the host's contract. Default `false`: byte-identical for every host.
    public var taaStillAccumulation: Bool = false
    /// With `taaStillAccumulation`: keep the Karis 1/(1+Y) sample weights (damps fireflies at the
    /// cost of a slight darkening bias in noisy highlights). Default `false`.
    public var taaStillKarisWeights: Bool = false
    /// With `taaStillAccumulation`: apply the still branch's display sharpen (+0.25 luma unsharp over the
    /// accumulation's 3×3, clamped to its range) to the frame handed downstream. Default `true` (the
    /// shipped still); Daydream's Maximum turns it off — its 2× render + declared reconstruction filter
    /// carry the detail (DH-1365). Rides `TAAStillParams.noSharpen`.
    public var taaStillSharpen: Bool = true
    /// Neighborhood clamp width (sigmas). WIDE on purpose (~4): a tight clamp
    /// re-injects low-frequency crawl by yanking clean history toward the noisy
    /// current 3×3 mean. Only meant to catch gross on-screen disocclusions.
    public var rtGITemporalClamp: Float = 4.0
    /// Daydream DH-1344 — the path-traced still's RADIANCE CACHE cell size (m); 0 ⇒ off (byte-identical).
    /// Build frames' paths write the light they find leaving each surface into a hashed world grid; later
    /// frames' paths end at the grid instead of continuing a long random walk — in a pale room, where most
    /// light has bounced four or more times, that walk is where the per-pixel noise (the mottle) came from.
    /// The build frames must be held out of the still's accumulation (their full walks are the noisy ones):
    /// `resetRTGIAccumulation(keepingRadianceCache: true)` after them. Only with `rtPathTracedGI`.
    public var rtPathRadianceCacheCell: Float = 0
    /// Frames the cache BUILDS (writes only) after it is cleared, before paths start reading it. 4 measured
    /// the same as 8 and 16 for wall mottle against a 4000-path reference (Oak Credenza Hall 1.52 % at all
    /// three; Sunset Gallery within 0.06 points), and each build frame is a full-walk frame (~3.5 s at hero size).
    public var rtPathRadianceCacheBuildFrames: Int = 4
    /// Paths per pixel on a build frame.
    public var rtPathRadianceCacheBuildRays: Int = 1
    /// Daydream DH-1344 — whether a STILL's GI filter (`rtGIStaticAccumulation`) keeps the SVGF
    /// luminance edge-stop. The stop weighs neighbours by their NOISY luminance, so it rejects the rare
    /// bright samples a path tracer's mean depends on: against an unfiltered 256-path reference it
    /// darkened every surface by 0.09–0.24 st and grouped the leftover noise into blotches, while
    /// protecting no in-plane indirect shadow edge (Oak Credenza Hall, Sunset Gallery, 2026-10-14).
    /// Depth and normal stops keep geometric edges either way. Default `true`: byte-identical.
    public var svgfStillLuminanceStop: Bool = true

    // ── Hardware-RT reflections ──────────────────────────────────────
    //
    // Runs inside the same `illumi_rt_lighting` pass (reuses the AS,
    // intersector and per-triangle albedo/normal buffers), so it only fires
    // when `rtEnabled` is also true. Traces the glossy/mirror direction and
    // adds ONLY scene-geometry hits — reflections of the room's own surfaces
    // that the sky-cube IBL can't contain and that SSR loses off-screen. On a
    // ray MISS the sky is left to the deferred IBL specular (no environment
    // double-count). The host should set `ssrIntensity = 0` when this is on so
    // RT owns scene-geometry reflections — same "RT owns it" contract the sun
    // already uses.
    public var rtReflectionsEnabled: Bool = false
    /// **Reflections as their own TLAS trigger (S3.4).** Default `false`: the TLAS is
    /// built only for extracted-scene RT or AAA glass, exactly as before, and every
    /// existing host is byte-identical. When `true` (and `rtEnabled` +
    /// `rtReflectionsEnabled`), the extracted-instance TLAS is built even for a scene
    /// with no glass and no full-scene RT — under the **glass-arm caps**, not the strict
    /// live-RT ones, and for the same reason glass gets the higher ceiling: the
    /// per-pixel cost is bounded by the fraction of the frame that *qualifies*
    /// (`roughness ≤ rtReflRoughnessCutoff`), not by the whole frame, and hosts run
    /// reflections in a settled/photo lane. The pass this feeds is the SAME TLAS
    /// lighting kernel; a host that wants reflections WITHOUT paying the sun/GI ray
    /// terms sets `rtSunOwnership = .deferred` and `rtGIStrength = 0` — each term is
    /// gated independently in the kernel, so those rays are never traced.
    ///
    /// This exists because the alternative — enabling `buildRTFromExtractedScene` —
    /// swaps the caps to the strict live-RT pair (128 groups / 150 k tris), which a
    /// real architectural document busts (4000 Sunset measures 176 / 490 k), tripping
    /// `.sceneTooHeavy` and taking RT GLASS down with it. Reflections must not be able
    /// to regress glass; on this trigger they share its caps instead.
    public var rtTLASForReflections: Bool = false
    /// Master scale on the reflection contribution. 0 = off.
    public var rtReflStrength: Float = 1.0
    /// Max reflection ray length, metres.
    public var rtReflMaxDistance: Float = 40.0
    /// Surfaces rougher than this skip reflections (glossy fades out).
    public var rtReflRoughnessCutoff: Float = 0.6
    /// Glossy samples per pixel (1 = mirror-sharp; TAA denoises the rest).
    public var rtReflRays: Int = 1

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
    /// Indirect rays per atlas texel in the update pass (multi-bounce gather).
    public var surfCacheIndirectRays: Int = 4
    /// Temporal EMA blend for the cache update — fraction of the new radiance
    /// folded in each frame. Lower = smoother/slower convergence.
    public var surfCacheAlpha: Float = 0.1
    /// Extra α = 1 cache updates run on the frame after cards bake or the cache turns on (DH-0622),
    /// so a still that switches the cache on starts from a filled atlas instead of an empty one —
    /// each update carries the light one bounce further. 0 (default) = none.
    public var surfaceCacheWarmIterations: Int = 0
    /// DH-0849 — an indirect ray in the cache update that strikes the BACK of a card (its triangle's
    /// winding normal faces along the ray) brings back no light instead of that card's front-side
    /// irradiance. Without it, wall bands hidden inside a closed plenum (above a ceiling deck, below a
    /// floor) read the deck's room-side light off its back and glow, and GI hits at the crease pick
    /// that up as a bright line. Changing it takes effect on the next update (the EMA re-converges).
    public var surfaceCacheBackFaceGuard: Bool = true

    // ── Depth of field ───────────────────────────────────────────────
    /// Lens-accurate DOF on the resolved HDR before bloom. Off = sharp everywhere.
    public var dofEnabled: Bool = false
    /// Focus plane distance from the camera, in metres.
    public var dofFocusDistance: Float = 3.0
    /// **The whole lens, as one number.** The circle-of-confusion DIAMETER in pixels
    /// per unit of `|z − z_f| / z`, which is the thin-lens relation with everything
    /// that does not vary per pixel — focal length, f-number, focus distance, sensor
    /// size, buffer height — already folded in. The host computes it (in Daydream
    /// Home, `ThinLens.cocCoefficientPixels`); the renderer never re-derives optics.
    ///
    /// Because the host scales it by the buffer height, the defocus is automatically
    /// the same fraction of the frame at every render scale and export size.
    public var dofCoCCoefficient: Float = 0
    /// Clamp on the CoC radius in pixels. A performance and sanity bound on the
    /// gather's reach, not part of the optics — leave it generous.
    public var dofMaxRadius: Float = 32.0
    /// Iris blade count. `0` (or < 3) is a perfectly circular aperture; 5–9 are the
    /// common real values, and give bokeh balls their polygonal edge.
    public var dofBlades: Int = 0
    /// Orientation of the blade polygon, in radians.
    public var dofBladeRotation: Float = 0
    /// How wide each gather tap's colour is prefiltered, as a multiple of that tap's own share
    /// of the confusion disc. 1 = exactly its share (the intended value); 0 = point taps, i.e.
    /// the pre-DH-0728 behaviour, kept as an A/B lever rather than as a default.
    ///
    /// This was `DOFParams._pad0` — a field named as a pad on the Swift side and as a control
    /// on the Metal side — so the prefilter pyramid DH-0728 added to the shader was fed 0 and
    /// the gather point-sampled every tap. See DH-0883.
    public var dofPrefilterScale: Float = 1.0
    /// **Diffraction**, as the Airy disc's CoC RADIUS in pixels of the internal buffer — a
    /// floor under the circle of confusion at every distance, the focus plane included, since
    /// a diffraction-limited image is soft everywhere rather than soft off the focus plane.
    /// The host derives it from the aperture (`ThinLens.diffractionCoCPixels`); 0 = off.
    ///
    /// Stopped down this is not a subtlety: at 550 nm the Airy diameter is 2.44·λ·N, so f/22
    /// puts it at 0.029 mm — exactly the acceptable CoC on full frame, which is why f/22 is
    /// where a 35 mm lens stops resolving. On 4 × 5, f/22 is a normal working aperture.
    public var dofDiffractionCoCPixels: Float = 0
    /// **Natural (cos⁴) vignetting** — `(halfFrameDiagonalMM / focalLengthMM)²`, scaled by how
    /// much of the physical falloff the host wants (1 = all of it). Multiplies the SCENE
    /// before the exposure meter and the tonemap, because it is the lens losing light, not a
    /// grade on the finished image (that is `vignetteStrength`). 0 = off. See DH-0882.
    public var naturalVignetteK: Float = 0

    public var displayTransform: DisplayTransform = .acesRec709
    /// Optical vignetting, 0…1 — how hard the lens barrel clips the aperture toward
    /// the frame corners, turning bokeh discs into cat's-eye lemons. 0 = none.
    public var dofCatsEye: Float = 0
    /// **Sharp subjects in front of a defocused background** (opt-in; off = byte-identical).
    /// Two gather fixes for an in-focus silhouette against a far, blurred field: the in-focus
    /// pixel's OWN confusion disc is counted explicitly (a spiral whose reach a defocused
    /// neighbour pushed out to 8+ px never reliably hits a ≲ 1 px disc, so the subject's edge went
    /// see-through, a band as wide as the background's CoC), and a background tap reads the
    /// prefilter pyramid only where its footprint holds no sharper layer (an HDR-bright subject
    /// swamps the pyramid's CoC floor weight and was carried out a whole background-CoC wide as a
    /// halo). Measured on the Digital Clock's toys (VZ-0188).
    public var dofSharpSubjects: Bool = false
    /// **Partial occlusion — the LAYERED COMPOSITE** (opt-in, with `dofSharpSubjects`; off =
    /// byte-identical). What the thin lens does at a pixel whose own surface has CoC c: the share of
    /// its rays that a NEARER defocused surface stops is that surface's absolute coverage, composited
    /// over the rest; the rest cross the pixel's depth inside its own disc (radius |c|), where the frame
    /// shows either its own layer or — through a HOLE, past an edge or a gap — whatever lies farther.
    /// So the pixel is `α_F·front + (1 − α_F)·(its own disc's average, holes included)`.
    /// Replaces the sharp-subject gather's behind-tap rule (a farther tap at |c_p|/|c_s| of its weight,
    /// within |c_p| + 1) and its normalised averaging of nearer taps, which — measured against an
    /// aperture-sampled reference of the Digital Clock (`dofReferenceAccumulation`) — kept a blurred
    /// wall's edge 97 % wall where the lens has 50 % (a hard step at every defocused silhouette),
    /// left a defocused FOREGROUND member opaque to its own edge (a crane mast's lattice cut out sharp
    /// inside its soft halo; its falls dark and crisp), and gave a foreground over a sharp subject a
    /// third of the pixel at the silhouette where the lens gives half. Keeps VZ-0188's invariant (an
    /// in-focus subject stays opaque, the wall behind it clean). Taps: the own disc and the near-field
    /// disc round the pixel, one spiral or two; prefiltered reads only where a tap's footprint belongs
    /// to its term (the min-CoC pyramid carries the signed-CoC range for that, rgba16Float while on).
    /// The fronts' coverage is sampled finely enough not to print as a dotted line along every
    /// defocused edge: a NEAR-FIELD coverage pyramid (two more half-res mip chains while on) lets a
    /// near-field front count as the coverage of each tap's footprint instead of a hit-or-miss texel,
    /// and a front behind the focus plane much smaller than the pixel's own disc gets a spiral fitted
    /// to it. Held to the exact layered thin-lens integral in `IlluminatoramaDOFPartialOcclusionTests`.
    public var dofPartialOcclusion: Bool = false
    /// **Fast gather** (opt-in, with `dofSharpSubjects`; off = byte-identical). The gather's cost is
    /// taps × pixels and the taps followed the dilated tile max, so every in-focus pixel within a
    /// max-radius of a blurred wall paid the wall's 100+ taps (Digital Clock: 65–73 ms of a 145 ms
    /// frame at 2880×1620). ON: (1) the exact reach — in sharp-subject mode a tap outside the pixel's
    /// own blur and beyond the near field's widest disc can only weigh 0, so it is never taken;
    /// (2) 3 prefiltered taps per px of reach (8…64) instead of 6 (24…128); (3) pixels at least
    /// `dofHalfResMinCoC` px out of focus take a half-resolution gather where the four half-res
    /// texels they read are one layer. Measured against the full gather in
    /// `IlluminatoramaDOFSubjectTests` (VZ-0191).
    public var dofFastGather: Bool = false
    /// Fast gather: the |CoC| (px, radius) from which a pixel may take the half-res result.
    public var dofHalfResMinCoC: Float = 3
    /// **Quarter-resolution background** (opt-in, with `dofFastGather`; 0 = off = byte-identical).
    /// Pixels at least this many px out of focus (radius, internal pixels; clamped to ≥ 6) take a
    /// bilinear read of a gather made once per 4 × 4 block, where that block is one layer, and the
    /// half-res pass skips every texel whose pixels all do — a macro shot's far wall at the CoC
    /// clamp then costs a quarter of the half-res gathers. A ≥ 12 px disc averages > 450 px; its
    /// 4-px bilinear reconstruction (tent σ 1.63 px) widens it by ≤ 3.6 % in σ at 12 px, ~1 % at 32.
    /// Measured against the full gather in `IlluminatoramaDOFSubjectTests` (VZ-0197).
    public var dofQuarterResMinCoC: Float = 0
    /// **Defocus-aware shading rate for the lighting pass** (opt-in, VZ-0197; 0 = off = byte-identical
    /// — the default lighting variant never reads the gate and no extra pass is encoded).
    ///
    /// With depth of field on, a 16×16 tile whose geometry is ALL at least this far out of focus
    /// (CoC radius, internal px, under the DOF's own lens and clamps; floored at 4) has each
    /// one-surface 2×2 quad (four geometry pixels whose view depths agree within 2 %) lit ONCE, at
    /// its top-left pixel, and that radiance written to all four; every other pixel is lit exactly as
    /// before. The lens then spreads each of those pixels over a disc ≥ π·gate² px, so the 2×2
    /// replication (a 2-px box, σ 0.5 px, and a ½-px shift) widens the blur against the disc's σ of
    /// ≥ gate/2 px by < 1 % in σ at an 8 px gate, 1.4 % at 6. It point-samples the quad, so detail
    /// within ~0.05 cycles/px of Nyquist can alias below the disc's cut-off — texture-filtered
    /// G-buffers carry little there; the look diff is the check. Two dispatches of the same kernel (a coarse variant, a thread
    /// per quad, and a skip variant) whose pixel sets partition the frame, plus a depth-only tile
    /// reduction (`illumi_dof_shading_tiles`). Off whenever `dofEnabled` is off. Lighting-pass
    /// consumers that must see every pixel lit individually (SSR, RT reflections, SSS) see the
    /// replicated values in those tiles — acceptable only because the DOF blurs them past a disc.
    public var lightingDefocusShadingMinCoC: Float = 0

    // ── Depth of field: the GROUND TRUTH (opt-in diagnostics; off = exact no-op) ─────────
    /// **Depth of field by aperture sampling** — the reference the gather is measured against.
    /// An offline renderer makes defocus by integrating over the lens: the host renders N pinhole
    /// frames of a frozen instant, frame k with the eye moved to pupil point k and the frustum
    /// sheared by `projectionShiftNDC` so the focus plane stays on the same pixels, and this
    /// averages them (`illumi_dof_reference_accumulate`). While on, the DOF pass does not gather:
    /// it adds this frame's HDR to a float accumulator (per-pixel cat's-eye weight for
    /// `dofReferenceApertureSample`) and hands the running MEAN to local adaptation, bloom,
    /// halation and the tonemap — so the finished frame is the lens's own integral through the same
    /// post chain the gather's output takes. The host turns `dofEnabled` off for it (every pixel
    /// shaded and traced at full rate — no defocus shading rate, no reduced glass trace) and keeps
    /// `dofCatsEye` / `naturalVignetteK` as the lens has them. See the kernel for the optics.
    public var dofReferenceAccumulation: Bool = false
    /// This frame's pupil point, in units of the iris's circumradius, oriented like the gather's
    /// footprint (x right, y DOWN in pixels). Only its cat's-eye clip is read here — the host has
    /// already moved the eye and sheared the frustum for it.
    public var dofReferenceApertureSample: SIMD2<Float> = .zero
    /// false ⇒ this frame is not added: the stored mean is re-presented (settle frames, so the
    /// temporal post state — local adaptation — converges on the finished integral).
    public var dofReferenceAddsFrame: Bool = true
    /// Frames added since the last reset (counted when encoded).
    public var dofReferenceFrameCount: Int = 0
    /// Built on first use (a diagnostics path: no host that never turns it on pays the compile).
    lazy var dofReferencePipeline: MTLComputePipelineState? = {
        guard let fn = engine.library?.makeFunction(name: "illumi_dof_reference_accumulate") else { return nil }
        return try? device.makeComputePipelineState(function: fn) // gpu-ok: lazy one-time build, opt-in diagnostics pipeline
    }()
    /// **Off-axis lens shift**, in NDC (clip.xy += shift · clip.w): slides the image under a fixed
    /// eye without turning the view — a view camera's rise/shift, and the shear that keeps a focus
    /// plane on the same pixels while the eye moves across the pupil (`dofReferenceAccumulation`).
    /// Folded into the projection with the TAA jitter and, like it, kept out of the motion vectors.
    /// Zero (default) = exact no-op.
    public var projectionShiftNDC: SIMD2<Float> = .zero
    /// **Hold the exposure meter** (opt-in; false = exact no-op): skip the estimate and keep
    /// applying the last smoothed exposure, so offline A/B renders of one instant (and the frames of
    /// an aperture-sampled reference, whose eye moves) print through exactly the same exposure.
    public var autoExposureHold: Bool = false
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

    /// **Pin the auto-exposure meter's sample grid** (opt-in, VZ-0197; nil = the internal render size,
    /// as always). `illumi_exposure_estimate` samples 256 × 32 cells by LINEAR index over an
    /// (w/8 × h/8) grid in normalised UV — so which part of the frame it meters depends on the
    /// internal size (VZ-0172: at 4320 × 2430 the top ~5 % of the frame, at 2880 × 1620 the top
    /// ~11 %), and a host that renders one canvas at two internal scales gets two exposures. Set this
    /// to the internal size the host's look was metered at and every scale samples the same UV
    /// positions (the kernel reads them with a normalised sampler).
    public var exposureMeterGridSize: SIMD2<Int>? = nil

    // ── Volumetric light shaft (god-rays) ────────────────────────────
    /// Single-scatter ray-march of the sun through hazy air, making the beam
    /// visible in the air. Uses `rtSunDirection` / `rtSunColor` for the sun.
    public var volumetricEnabled: Bool = false
    public var volFogDensity: Float = 0.05
    public var volScatterStrength: Float = 1.0
    public var volAnisotropy: Float = 0.6
    public var volSteps: Int = 48
    public var volMaxDistance: Float = 24.0
    public var volWindowX: Float = 4.0
    /// (y0, y1, z0, z1) of the window opening for the in-air shadow test.
    public var volWindowRect: SIMD4<Float> = SIMD4(0.9, 2.5, -1.6, 1.6)
    public var volFeather: Float = 0.12
    /// Outdoor mode: skips the analytic window shadow test so all air samples
    /// scatter sun light. Use for exterior scenes (forests, fields, open sky).
    /// The Henyey-Greenstein phase still peaks toward the sun, and the depth
    /// buffer still terminates the march at surfaces, so beams glow brightest
    /// when the camera looks toward the low sun through canopy gaps.
    public var volOutdoorMode: Bool = false

    // ── Traced volumetric light (Daydream DH-0990) ────────────────────
    /// Single scattering in a faint haze, marched along each view ray against the REAL geometry:
    /// the sun through a shadow ray per sample (so the shafts are exactly what the window openings
    /// cut) plus the sky through one window APERTURE per sample (`IlluminatoramaAreaLight
    /// .isAperture`), confined to ROOFED air (a short ray straight up must meet the house), so an
    /// exterior view never fogs. Monte-Carlo per frame and progressive across frames: a STILL
    /// lane feature — it needs a live TLAS and an accumulating resolve. Default off: no cost.
    public var volumetricRTEnabled: Bool = false
    /// Haze extinction = scattering coefficient σ, per metre. A real room's air is ~1e-3; a
    /// photograph's visible shafts need dust, ~1e-2.
    public var volRTDensity: Float = 0.01
    /// Henyey–Greenstein g for the haze (> 0 forward: the beams glow toward the sun).
    public var volRTAnisotropy: Float = 0.5
    /// March samples per pixel per frame (stratified; a still's frames fill in the rest).
    public var volRTSteps: Int = 6
    /// Trace the haze on a random 1/`volRTPixelStride` of the pixels each frame, at that many times the
    /// weight — for a still, which accumulates its frames (1 ⇒ every pixel, as before).
    public var volRTPixelStride: Int = 1
    /// The march's reach along a view ray (metres) — an interior is a few metres deep.
    public var volRTMaxDistance: Float = 20
    /// How far above an air sample a roof may be for the sample to count as indoors (metres).
    public var volRTRoofReach: Float = 8
    /// Separate gains on the sun's and the sky's in-scatter (1 = physical for the density).
    public var volRTSunScatter: Float = 1
    public var volRTSkyScatter: Float = 1
    /// DH-0999 — SUNLIT DUST: a second, sun-only scatterer (per metre at full strength) whose
    /// amount follows the direct sun measured entering the view's air. A coarse pre-pass marches
    /// the view's own rays and counts the roofed samples that see the sun; with A = sunlit /
    /// roofed, the dust is `volRTDustDensity · min(1, A / volRTDustAdmittanceRef)` — a sunbeam
    /// crossing the room reads as a shaft, a room no sun enters gets none (no grey fog). It adds
    /// only the collimated sun's in-scatter: as a conservative scatterer in the room's
    /// near-uniform diffuse field its extinction and diffuse in-scatter cancel, so it never dims
    /// the room. 0 = off: no pre-pass, the air term alone.
    public var volRTDustDensity: Float = 0
    /// HG g of the dust (large particles: strongly forward — the beam glows toward the sun).
    public var volRTDustAnisotropy: Float = 0.6
    /// The sunlit fraction of the roofed air in view at which the dust reaches full density.
    public var volRTDustAdmittanceRef: Float = 0.1
    /// The admittance lattice's width in rays (height follows the aspect).
    public var volRTAdmittanceGrid: Int = 64
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

    // ── Spot-light beam scattering (issue: CK5 concert beams) ───────
    /// Single-scatter march of every `spotLights` cone through hazy air, so
    /// spots read as BEAMS in the air, not just pools on surfaces. Runs after
    /// the sun volumetric pass, independent of `volumetricEnabled` — a stage
    /// scene wants haze-lit beams with no sun window. Per-ray cost is bounded
    /// per spot by analytic range-sphere clipping, so dead air is ~free; keep
    /// `spotLights.count` in the low tens. Beams are depth-terminated but not
    /// shadow-traced inside the interval (geometry in front hides a beam;
    /// geometry inside one doesn't carve it).
    public var volSpotBeamsEnabled: Bool = false
    /// Haze density for the beam march (separate from `volFogDensity` so the
    /// sun shaft and stage haze are tunable independently).
    public var volSpotBeamDensity: Float = 0.06
    public var volSpotBeamStrength: Float = 1.0
    /// Forward-scatter anisotropy: higher = beams flare when the camera looks
    /// into the fixture, like real stage haze.
    public var volSpotBeamAnisotropy: Float = 0.55
    /// Jittered samples per spot interval (TAA accumulates the noise out).
    public var volSpotBeamSteps: Int = 20
    public var volSpotBeamMaxDistance: Float = 60.0
    /// Carve the beams with the spot shadow maps: each march sample is compared against its
    /// spot's shadow slice, so a beam stops where geometry blocks its light (a cone from inside
    /// a mouth ends at the lips instead of glowing on through the face). Only spots that hold a
    /// shadow slice are carved (`spotShadowsEnabled` + capacity). Default false ⇒ the pass is
    /// byte-identical to before for every existing host.
    public var volSpotBeamShadows: Bool = false
    /// Only the first `volSpotBeamSpotLimit` entries of `spotLights` march as haze beams; the
    /// rest light (and shadow) surfaces only. Lets a host carry studio fill / rim cones in
    /// `spotLights` — for their shadow maps — without each one drawing a beam across the frame
    /// (put the beam-worthy spots first). Default `Int.max` ⇒ every spot beams, byte-identical
    /// for every existing host.
    public var volSpotBeamSpotLimit: Int = .max

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
    public var volEmissiveColumn: EmissiveColumn? = nil

    // ── Phase 4.21 — auto-exposure ──────────────────────────────────
    /// Toggle the per-frame log-luminance estimator + EMA. When off,
    /// the tonemap falls back to the static `exposure` scalar above.
    /// On is the default — matches SCN's `wantsExposureAdaptation` so
    /// scenes don't blow out on first frame.
    public var autoExposureEnabled: Bool = true
    /// Target log2-luminance the exposure adapts toward. Calibrated
    /// across the in-tree scene library: dark-sky scenes (Eggs,
    /// FloatingFlowers+) become muddy at -2.0 because the geometric
    /// mean of a mostly-dark frame is very low, and `2^(target -
    /// mean)` then pumps exposure 4-6× into mid-grey. -4.0 lets the
    /// dark scenes stay dark while still pulling bright skies
    /// (HotdogWaterslide) back from full clip. Scene controllers
    /// that want a brighter look can override per-scene.
    public var autoExposureTargetEV: Float = -4.0
    /// Clamp on the final exposure multiplier so a deeply-dark scene
    /// can't pump the look into mid-grey. Capped at 3× by default
    /// (≈ 1.5 stops) — matches the SCN auto-exposure feel of
    /// "compensate, don't fabricate light."
    public var autoExposureMaxBoost: Float = 3.0
    /// Lower clamp on the final exposure multiplier — how far auto-exposure
    /// may DARKEN a bright scene. Phase 4.28: previously hard-coupled to
    /// `1 / maxBoost` (≈ 0.33×), which floored darkening at ~1.5 stops down.
    /// Scenes that use a bright `environmentImage` as both IBL and background
    /// (VintageDiner, Pizza+) want ~0.03–0.08× to come back from full white;
    /// 0.33× left them blown. Decoupled so dark scenes keep their ≤3× boost
    /// cap while bright scenes get ~4.3 stops of darkening headroom. Only
    /// affects scenes bright enough to hit the floor — dark scenes never do.
    public var autoExposureMinExposure: Float = 0.05
    /// EMA half-life in seconds for the exposure smoothing. ~0.25 s
    /// matches the SCN feel — fast enough that a scene cut converges
    /// in ~1 s, slow enough that one bright pixel doesn't pump.
    public var autoExposureHalfLife: Float = 0.25

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

    /// Cap the exposure so the frame's bright half lands at `autoExposureHighlightEV`
    /// rather than wherever the mean-based answer puts it. 0 (default) = OFF, an exact
    /// no-op; 1 = the cap fully governs. The cap is a `min`, never a boost — it can
    /// only pull a frame back from clipping, so a scene with no bright half is
    /// untouched at any strength.
    ///
    /// The statistic is the mean log-luminance of the samples ABOVE the frame mean
    /// (the upper half), not a true percentile: the reduction is already in the
    /// kernel, it costs no extra texture reads, and it is far more stable frame to
    /// frame than a p99 that chases single specular pixels.
    public var autoExposureHighlightProtection: Float = 0
    /// Where the bright half is asked to land, in log2 linear units, pre-tonemap.
    /// 0 ⇒ 1.0 linear ⇒ ~0.80 through the ACES shoulder — bright, still textured.
    /// Negative pulls the highlights further down. Only read when protection > 0.
    public var autoExposureHighlightEV: Float = 0

    public var autoExposureMetering: AutoExposureMetering = .mean
    /// The metered percentile under `.percentile` (0…1). 0.5 = the median.
    public var autoExposureKeyPercentile: Float = 0.5
    /// The highlight-guard percentile (0…1; 0 = no guard), e.g. 0.95 or 0.99.
    public var autoExposureGuardPercentile: Float = 0
    /// Where the guard percentile is asked to land, log2 linear, pre-tonemap.
    public var autoExposureGuardEV: Float = 0
    /// Build and publish the histogram even while metering the mean — the INSTRUMENT.
    /// Read it back with `lastExposureHistogram`. Changes no pixel.
    public var exposureHistogramEnabled: Bool = false

    /// DH-0889 — jitter each window portal's shadow-map projection by a sub-texel Halton offset
    /// per TAA frame, so an accumulating still supersamples the occluder's edge instead of
    /// printing one sawtooth per texel. Default OFF: it re-renders the portal slices every frame,
    /// which a live canvas's static-slice reuse would otherwise skip. Only the shadow-MAP path
    /// uses it; traced portal visibility (`rtAreaShadowRays > 0`) has no texels to jitter.
    public var areaShadowJitterEnabled: Bool = false

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

    public var debugTerm: DebugTerm = .normal

    // ── Phase 2 post-FX knobs ────────────────────────────────────────
    /// Screen-space ambient occlusion intensity (0 = disabled).
    public var ssaoIntensity: Float = 0.85
    /// World-space hemisphere radius for SSAO samples, in metres.
    public var ssaoRadius: Float = 0.4
    /// DH-0441 — outer radius (metres) of the wide AO ring, a second GTAO march whose
    /// occluders are weighted to the annulus beyond `ssaoRadius`. See IlluminatoramaSSAO.metal.
    public var ssaoFarRadius: Float = 0.5
    /// DH-0441 — strength of the wide AO ring. 0 (default) = OFF, the ring is never marched
    /// and the AO field is byte-identical.
    public var ssaoFarIntensity: Float = 0
    /// Screen-space reflection intensity (0 = disabled).
    public var ssrIntensity: Float = 0.7
    /// Max view-space ray length for SSR marches, in metres.
    public var ssrMaxDistance: Float = 18.0
    /// View-space "thickness" tolerance for the depth-crossing hit test.
    public var ssrThickness: Float = 0.6
    /// Cap on march steps per SSR ray.
    public var ssrMaxSteps: UInt32 = 48

    // ── Phase 2.5 shadow knobs ───────────────────────────────────────
    /// Toggle the cascaded-shadow pass + sampling. Off = direct sun lights
    /// every pixel, regardless of occlusion (Phase 2 behaviour).
    public var shadowsEnabled: Bool = true

    /// Mesh kinds excluded from the DIRECTIONAL (cascade/sun) shadow pass: their
    /// geometry is not drawn as a sun-shadow occluder, so it casts no shadow from
    /// the sun. Empty by default → no behaviour change. The host opts a kind in
    /// when a self-illuminated object shouldn't blot its own light pool with a
    /// hard sun shadow (e.g. a lamp: its body casting a crisp shadow across the
    /// circular pool it throws reads as a lopsided pool). Spot-cone shadows
    /// (`encodeSpotShadowPasses`) are unaffected — the object still traps its own
    /// light; only the sun stops shadowing it.
    public var directionalShadowExcludedKinds: Set<MeshKind> = []

    /// Mesh kinds drawn ONLY into the shadow maps (sun cascades, spot atlas, point
    /// cubes) and the RT TLAS — never the G-buffer. Empty by default → no behaviour
    /// change. A "lighting-only" occluder: the host registers geometry that must
    /// BLOCK light without being visible — e.g. a dollhouse ceiling that keeps the
    /// noon sun out of interior rooms while the camera still looks freely down into
    /// them. The instances are uploaded normally (they need slots in the instance
    /// buffer for the shadow passes); only the G-buffer raster skips them, so they
    /// write no albedo/depth/velocity/layer and are invisible + unpickable on screen.
    public var shadowOnlyMeshKinds: Set<MeshKind> = []
    /// Opt-in INTERIOR MAPPING (van Dongen 2008) per mesh kind: every instance of a listed kind is
    /// drawn with a G-buffer variant whose emission is the radiance of a lamp-lit room box behind
    /// the pane, ray-traced analytically per fragment along the camera ray (parallax-correct room
    /// depth, one draw, no extra geometry) — `inst.emission` is the lamp's colour × the room's
    /// luminance. The mesh must carry UVs across the pane's face (u along its width from
    /// `pane.z`, v up from 0) and a tangent along u. Empty (default) ⇒ the variant is never built
    /// or bound ⇒ byte-identical.
    public var interiorMappedMeshKinds: [MeshKind: IlluminatoramaInteriorMapping] = [:]

    /// Interior day-light separation (opt-in; pairs with light-layer masking).
    /// `interiorLayerMask` = OR of every instance-layer bit the host stamped on
    /// INTERIOR rooms; 0 (default) disables the feature entirely (byte-identical).
    /// Fragments carrying an intersecting room stamp get their sky-IBL indirect
    /// scaled by mix(`interiorIBLSide`, `interiorIBLUp`, saturate(N.y)) — up-facing
    /// floors lose the open-top skylight hardest — and their ambient supplement
    /// scaled by `interiorAmbient` (>1 = a warm indoor fill). Rationale: an open-top
    /// dollhouse room is otherwise lit like a patio (full sky irradiance), several
    /// stops over a believable interior.
    public var interiorLayerMask: UInt32 = 0
    public var interiorIBLUp: Float = 1
    public var interiorIBLSide: Float = 1
    public var interiorAmbient: Float = 1
    /// Interior irradiance BANDS — the room's own three-band diffuse environment
    /// (the lawn-green-ceiling fix). The irradiance cube is baked from the outdoor
    /// sky dome, so an interior ceiling (N.y = −1) otherwise samples the GROUND
    /// half and renders in the lawn's colour — the `interiorIBL*` scalars can dim
    /// that, never recolour it. Interior fragments blend their diffuse irradiance
    /// toward these host-authored values by `interiorIrradianceWeight`
    /// (0 = the exact cube sample, the default — byte-identical for every scene
    /// that never opts in; hosts ramp the weight with daylight so night keeps the
    /// cube). As the weight → 1 the DIFFUSE interior scalar folds to 1.0 (the
    /// bands already carry the up/side weighting it faked); specular keeps the
    /// scalar and the prefiltered cube. Units match the cube sample they replace
    /// (pre-`iblIntensity` irradiance). Applied by the deferred kernel AND by
    /// secondary-hit shading (glass refraction / RT reflections / GI), so the room
    /// seen through a pane agrees with the room beside it.
    public var interiorIrradianceUp: SIMD3<Float> = .zero
    public var interiorIrradianceSide: SIMD3<Float> = .zero
    public var interiorIrradianceDown: SIMD3<Float> = .zero
    public var interiorIrradianceWeight: Float = 0
    /// Daylight apertures for the band GRADATION (S3.5 Stage D): the host's glazed
    /// openings, one rect each. With `interiorApertureGradation` > 0, interior fragments
    /// scale their BAND irradiance by an approximate summed solid angle of these — a wall
    /// point near its window reads brighter than the same wall 8 m away, which no
    /// normal-only term can produce. Empty list or strength 0 (the defaults) ⇒ factor is
    /// exactly 1.0 ⇒ byte-identical. Primary (deferred) pass only: the secondary paths
    /// carry the flat bands — a reflection of a wall does not re-derive its gradation.
    public var interiorApertures: [IlluminatoramaInteriorAperture] = []
    /// 0 = off (default); 1 = the full gradation. Fractional = partial (a look dial).
    public var interiorApertureGradation: Float = 0
    /// Per-room BAND LEVEL (S3.5 Stage E): one multiplier per light-layer bit, applied
    /// to a fragment's interior irradiance bands (diffuse and specular alike) according
    /// to the room its `layer` bits name.
    ///
    /// The bands are otherwise ONE environment for the whole frame, so their level knows
    /// the time of day and nothing about the room: a hall of glass and an interior closet
    /// rendered the same ceiling. The host measures how much daylight each room's own
    /// glazing admits and hands the levels here. A fragment on a shared wall carries both
    /// rooms' bits and takes their mean (there is no per-face signal at this point in the
    /// pipeline). Bits beyond `InteriorRoomGains.capacity`, an empty table, or
    /// `interiorRoomGainsEnabled == false` (all defaults) ⇒ every resolve is exactly 1.0
    /// ⇒ byte-identical. Carried by the deferred kernel AND both secondary paths, so a
    /// room through a pane or in a reflection is scaled like the room seen directly.
    public var interiorRoomGains: [Float] = []
    /// Master enable for `interiorRoomGains`. False (default) ⇒ exact no-op.
    public var interiorRoomGainsEnabled: Bool = false
    /// A/B DEBUG: overrides the LTC-specular arm of the area-light eval (nil = the
    /// baked LUT's own validation decides, the shipped behaviour). `false` forces the
    /// most-representative-point fallback — the bisect lever for separating "the LTC
    /// transform misbehaves" from "the polygon integral misbehaves" on a live scene.
    public var areaLTCOverride: Bool?

    /// ── Analytic night sky (stars + moon at SCREEN resolution) ──────────────
    /// The 2048×1024 equirect dome is far coarser than the frame, so celestials
    /// baked into it (VolumetricCloudRenderer's star field / moon) reach the
    /// screen bilinearly magnified into soft blobs. A host that wants pixel-sharp
    /// stars and a crisp, phase-correct moon sets the dome's
    /// `Params.celestialsInDome = false` and drives these instead — the lighting
    /// kernel evaluates them analytically per sky pixel (TAA supersamples the
    /// sub-pixel cores). All zeros (defaults) = feature off, byte-identical.
    ///
    /// Fade both brightness values with `VolumetricCloudRenderer.nightBlend(sunDir:)`
    /// so the analytic sky and the dome share ONE day/night ramp.
    public var nightSkyStarBrightness: Float = 0
    /// Moon disk brightness (HDR). 0 = no moon. ~1.5–3 reads as a luminous moon
    /// that blooms gently at night exposure.
    public var nightSkyMoonIntensity: Float = 0
    /// Unit vector from the scene toward the moon (e.g. a real ephemeris).
    public var nightSkyMoonDirection: SIMD3<Float> = SIMD3(0, 1, 0)
    /// Unit vector from the scene toward the TRUE sun — below the horizon at
    /// night. Lights the moon sphere per-pixel, so the phase (crescent→gibbous→
    /// full) and its orientation come straight from the geometry.
    public var nightSkySunDirection: SIMD3<Float> = SIMD3(0, -1, 0)
    /// Moon angular RADIUS in radians. The real moon is ≈0.0047 (0.27°); the
    /// default is ~2× that for a photographic read. Taste knob.
    public var nightSkyMoonAngularRadius: Float = 0.0095
    /// Which analytic night sky (IlluminatoramaNightSky.h). `.legacy` (default) is the
    /// byte-identical hashed grid + sphere-lit disk; `.physical` is the equal-area,
    /// magnitude-distributed, extinguished, twinkling star field with a Milky Way, the
    /// real-sized maria-mapped moon with earthshine and an aureole. In the physical model
    /// the brightnesses are NOT pre-faded by nightBlend: stars wash out against the sky
    /// behind them on their own, so pass `nightSkyStarBrightness` unscaled. Set
    /// `nightSkyMoonAngularRadius` to 0 for the real 0.26°.
    public var nightSkyModel: VolumetricCloudRenderer.Params.NightSkyModel = .legacy
    /// Physical model only — gains (1 = physical) on the lunar aureole (raise for haze /
    /// thin cloud), earthshine, the Milky Way and stellar scintillation.
    public var nightSkyMoonHalo: Float = 1
    public var nightSkyEarthshine: Float = 1
    public var nightSkyMilkyWay: Float = 1
    public var nightSkyTwinkle: Float = 1
    /// Physical model only — scene radiance of a magnitude-0 flux spread over one steradian:
    /// the ONE scale that puts stars, Milky Way and aureole in their real proportions. Match
    /// the sky dome's `Params.nightRadiance` so the moonlit sky and the stars agree.
    public var nightSkyRadiance: Float = 1.5e-5
    /// Physical model only — world → J2000-equatorial rotation as a quaternion
    /// (ix, iy, iz, r); see `NightSkyEphemeris.celestialOrientation`. Identity by default.
    public var nightSkyCelestialOrientation: SIMD4<Float> = SIMD4(0, 0, 0, 1)
    /// Physical model only — the faintest star drawn (visual magnitude); 0 (default) = no limit
    /// beyond the per-pixel washout. See `VolumetricCloudRenderer.Params.starLimitingMagnitude`.
    public var nightSkyStarLimitingMagnitude: Float = 0
    /// Scotopic tonemap shaping (see `scotopicDesaturation`): the display luma by which the
    /// night desaturation has faded out (0 = the legacy 0.08) and the Purkinje tint rod
    /// vision is pulled toward (.zero = the legacy neutral grey).
    public var scotopicKnee: Float = 0
    public var scotopicTint: SIMD3<Float> = .zero

    // DEFAULTS LESSON (PR #33 fix): the initial Phase 2.5 defaults were
    // shadowBias = 0.0008 and shadowSlopeBias = 0.005 — both too aggressive.
    // The shadow pass uses front-face culling, which already shifts stored
    // depths onto the back of casters and naturally separates lit receivers
    // from self-shadowing. Stacking a generous bias on top lifted shadows
    // off their contact line ("peter-panning") and the user caught it
    // immediately. When adding similar bias/tolerance defaults elsewhere
    // (e.g. SDF GI thickness, voxel cone tracing self-occlusion), default
    // to "off / minimum that does something" — let the user dial up.
    /// Constant depth bias subtracted from light-NDC z before the PCF compare.
    /// Tiny by default because the shadow pass uses front-face culling, which
    /// stores the back-face depth of casters and already separates lit
    /// receivers from self-shadowing — so we only need a hair of slop to
    /// soak up floating-point noise. Raise this only if you see acne.
    public var shadowBias: Float = 0.0003
    /// Bias added in proportion to `(1 - NdotL_sun)`. Defaults to zero
    /// because the front-face-culling bias above handles grazing surfaces
    /// too. Raise this only if grazing-angle acne shows up that the constant
    /// bias can't fix — too much causes "peter-panning" (shadows visibly
    /// detached from their caster's contact line with the receiver).
    public var shadowSlopeBias: Float = 0.0
    /// **TEST LEVER — production must leave this false.** Makes the three shadow passes
    /// pick their cull mode from `IlluminatoramaMesh.doubleSided` instead of the narrow
    /// `shadowCastsBothFaces` opt-in. This is the "obvious" version of the open-shell
    /// shadow fix, and it exists only so an A/B can MEASURE what it costs: `doubleSided`
    /// is a shading property that closed solids set freely (Daydream's walls, floors,
    /// millwork and stairs all do), so routing the shadow cull through it removes the
    /// second-depth acne defence from the entire structural shell. Keep it reachable so
    /// the acne is a number in a gate rather than a claim in a comment.
    public var shadowCullFollowsDoubleSidedForTest: Bool = false

    /// **Which face of a closed solid writes the sun's shadow depth.**
    ///
    /// `false` (default) culls FRONT faces, so the map stores each occluder's BACK face —
    /// second-depth shadow mapping, which is a strong acne defence because the stored depth sits
    /// a whole solid's thickness behind any lit surface. Its price is at CONTACTS: where a
    /// receiver touches the occluder's back face, stored and receiver depth are equal, the
    /// comparison ties, and the tie resolves to LIT. A floor meeting a wall is exactly that
    /// geometry, so the sun draws a bright line along the junction and the PCF kernel spreads it
    /// into a band (Danny, 2026-08-18/19, the Pantry wall-foot leak).
    ///
    /// `true` stores the FRONT face instead. At a wall foot the stored depth is then the wall's
    /// OUTER face — a full wall thickness in front of the receiver — so the junction is
    /// unambiguously shadowed and no PCF tap inside the wall's footprint can escape. Acne is
    /// then carried by `ShadowRasterBias` — the slope-scaled raster bias below, which this
    /// property selects automatically — plus the normal-offset bias in `cascadeVisibility`.
    public var shadowStoresFrontFaces: Bool = false

    /// Overrides the profile `shadowStoresFrontFaces` would otherwise pick. `nil` derives it, so
    /// a host that flips the convention gets the matching acne defence without a second call —
    /// the two are one decision. Set it to sweep values in a gate.
    public var shadowRasterBiasOverride: ShadowRasterBias?

    /// The profile in force: derived from the convention unless a host overrode it.
    var effectiveShadowRasterBias: ShadowRasterBias {
        shadowRasterBiasOverride ?? (shadowStoresFrontFaces ? .frontFace : .off)
    }

    /// The cull mode a shadow pass starts in, so the first group does not have to correct it.
    var shadowInitialCullMode: MTLCullMode { shadowStoresFrontFaces ? .back : .front }

    // MARK: - The substrate (a surface other surfaces are laid flush ON)

    /// Mesh kinds that are **substrate** — the ground a street, a driveway or a court line is
    /// laid flush on, whose separation from them is millimetres of deliberate lift rather than
    /// real space. These are biased AWAY from the camera so that anything resting on them wins
    /// the depth test at any distance.
    ///
    /// **Why the substrate recedes instead of the decal advancing.** Both close the same gap,
    /// and the gap is real: the camera projects into a NON-reversed `depth32Float` buffer where
    /// the smallest resolvable separation grows with the square of the distance —
    /// `≈ ulp(1.0) · z² / zNear`, which at `zNear = 0.05` is 0.5 mm at 20 m but **7.6 mm at
    /// 80 m and 17 mm at 120 m**. Any fixed lift therefore has a range past which the surface
    /// underneath starts winning pixels and the decal dithers into it. Measured in Daydream
    /// Home, 2026-08-28: a street with a 10 mm lift is clean to ~90 m and eaten by dirt beyond.
    ///
    /// Pushing the DECAL forward closes it too, but it spends the whole bias in the direction
    /// where something legitimately sits above the decal — a pavement stands on a 150 mm kerb,
    /// and a bias big enough to fix the road at 150 m is big enough to lift the road over that
    /// kerb. Pushing the SUBSTRATE back spends it in the direction where nothing is: there is,
    /// by definition, nothing under the ground. So the bias can be generous without inverting
    /// anything, and every surface laid on the ground is fixed at once rather than one by one.
    ///
    /// Empty by default — a host that declares no substrate gets byte-identical frames.
    public var substrateMeshKinds: Set<MeshKind> = []

    /// How far the substrate is pushed away from the camera, in depth-buffer ULPs (positive =
    /// further, since this buffer is not reversed).
    ///
    /// **Why 32.** Swept on Daydream Home's flat-lot street at a fixed camera, measuring the
    /// share of two windows lying wholly ON the roadway — one a few tens of metres out, one
    /// past where a 10 mm lift stops resolving — that read as bare ground rather than asphalt:
    ///
    /// | bias | near roadway eaten | far roadway eaten |
    /// |---|---|---|
    /// | 0 | 0.06 | **0.77** |
    /// | 16 | 0.00 | 0.00 |
    /// | 64 | 0.00 | 0.00 |
    /// | 128 | 0.00 | 0.00 |
    ///
    /// 16 already clears it completely; 32 is one doubling of margin for scenes whose substrate
    /// sits at a different depth range. It is not pushed higher for free: the G-buffer depth is
    /// what SSAO and contact shadows sample, and a bias is a real error in that texture — 32
    /// ULPs is ~1.5 cm of world-equivalent depth error on ground 20 m away and ~38 cm at 100 m,
    /// which is under the AO radius near the camera and irrelevant at the far end, where ground
    /// is a handful of pixels. Doubling again would start eating into the near figure.
    public var substrateDepthBias: Float = 32

    /// **Traced rays leave from the surface the camera sees** (Daydream DH-0715). The substrate's
    /// depth bias above buries its depth-rebuilt world position BEHIND the surface — ≈ 10 cm down
    /// the view ray at 50 m — so every traced sun, portal, bounce and AO ray from the ground started
    /// underground and hit it from below: Daydream's settled canvas and exports put a sunlit noon
    /// lawn in full shadow (sRGB luma 40 vs 153 on the shadow-map path). With this on, each traced
    /// pass first probes from the rebuilt point back toward the camera (`illumiSnapToVisibleSurface`)
    /// and moves the ray origin onto the surface it meets; a point not buried meets nothing and is
    /// unchanged. Effective only while `substrateMeshKinds` is non-empty (reach: twice
    /// `substrateDepthBias`), so a host with no substrate is byte-identical.
    public var rtSurfaceSnapEnabled: Bool = true
    /// The snap reach the traced passes receive, in depth ULPs (0 ⇒ no probe).
    var effectiveRTSurfaceSnapULPs: Float {
        rtSurfaceSnapEnabled && !substrateMeshKinds.isEmpty ? max(0, substrateDepthBias) * 2 : 0
    }

    /// PCF kernel radius in shadow-map texels: 0 = single tap, 1 = 3×3, 2 = 5×5.
    public var shadowPcfRadius: UInt32 = 1
    /// How far from the camera the outermost cascade extends, in metres. Past
    /// this distance, surfaces are treated as lit (no shadows). Cascades are
    /// distributed in [zNear, shadowMaxDistance] using a practical
    /// log/uniform split.
    public var shadowMaxDistance: Float = 50.0

    // ── Phase 4.10 — spot light shadow maps ──────────────────────────
    /// Toggle per-spot shadow casting. Off by default — scenes with
    /// dozens of spot lights (Eggs' rail spotlights) would otherwise pay
    /// N additional depth render passes per frame. Flip on per-scene
    /// when the visual gain justifies the cost.
    public var spotShadowsEnabled: Bool = false
    /// Constant depth bias subtracted from the spot light-space depth
    /// before the PCF compare. Combats acne on lit surfaces.
    public var spotShadowBias: Float = 0.0005
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

    /// **The photo-lane G-buffer (DH-0140).** When true the G-buffer pass runs the six-target
    /// pipeline variant and writes `GBufferOut.material` — a per-pixel material payload the
    /// five-target live pass has no room for (its one spare scalar is a discrete class tag), and
    /// the lighting kernel reads it. The sixth target and its two pipeline variants are
    /// allocated / compiled on FIRST use and dropped on resize, so a host that never sets this —
    /// every Visualizer scene, Daydream's live canvas — is byte-identical and pays nothing.
    /// Payloads so far: per-material clearcoat roughness (slice 1). Reserved lanes:
    /// transmission/alpha, per-texel grain tangent (DH-0478). A host's photo / export lane sets
    /// this for the still and clears it on the way out.
    public var extendedGBufferEnabled: Bool = false
    /// True while THIS frame's G-buffer pass wrote the sixth target — the flag AND the target
    /// allocation both succeeded. Diagnostics / gates.
    public internal(set) var extendedGBufferActive: Bool = false

    // ── Point-light cubemap shadows (opt-in, additive) ────────────────
    /// Master enable for the point-light depth-cubemap shadow pass. Default
    /// OFF — no shadow textures are allocated, no depth is rendered, and the
    /// lighting kernel takes the exact pre-change point-light path, so every
    /// scene that never flips this (Visualizer) is byte-identical. Turn on
    /// per-scene (Daydream) and additionally set `castsShadow` on the point
    /// lights you want shadowed (default off per light too).
    ///
    /// Mechanism: for each shadow-casting point light the renderer renders SIX
    /// 90°-perspective depth maps (the faces of a cube centred on the bulb) into
    /// a shared depth atlas, then the deferred lighting kernel projects each
    /// fragment through the dominant cube face and PCF-compares — so the bulb is
    /// blocked by geometry, PASSES through openings (doorways), and casts
    /// intra-room object shadows. Because it's a REAL depth occlusion, a shadowed
    /// light no longer needs (and must not use) the cheap position-based room
    /// layer mask — see `pointShadowSupersedesLayerMask`.
    public var pointShadowsEnabled: Bool = false
    /// Constant depth bias subtracted from the point light-space depth before the
    /// PCF compare (same role as `spotShadowBias`). Combats acne on lit surfaces.
    public var pointShadowBias: Float = 0.0006
    /// Up to this many shadow-casting point lights get a depth cube each frame
    /// (the N nearest to the camera; the rest fall back to unshadowed). Bounded by
    /// the cube atlas capacity. 4 cubes × 6 faces × 512² × 4 B ≈ 24 MB (allocated
    /// lazily only when `pointShadowsEnabled` first turns on).
    public var pointShadowMaxCount: Int { shd.pointShadowCubeCapacity }

    // ── Phase 4.15: Tonemap + IBL saturation parity ──────────────────
    /// Post-tonemap saturation multiplier. Narkowicz's fitted ACES curve
    /// compresses midtone chroma a little harder than SCN's HDR chain, so a
    /// mild boost keeps parity with the SCN baseline. Was 1.30 while the
    /// tonemap double-encoded sRGB into a non-sRGB background texture (see
    /// [[scenekit-background-texture-double-srgb]]); that washout — the main
    /// cause of the flatness — is now fixed, leaving only genuine ACES chroma
    /// compression to compensate, so 1.10 is enough. 1.0 = no change. Tunable
    /// per-scene if a scene's intentional pastel palette doesn't want the boost.
    public var tonemapSaturation: Float = 1.10
    /// **Local tone mapping** — eye-like local adaptation of the scene-referred frame before the
    /// display transform (edge-aware, bilateral-grid base layer; see
    /// `IlluminatoramaLocalToneMapping`). Runs after DOF, before bloom / halation / tonemap, and
    /// after the exposure meter (the meter never sees its own adaptation). `strength == 0`
    /// (default) ⇒ not encoded ⇒ byte-identical frames for every scene that doesn't opt in.
    public var localToneMapping = IlluminatoramaLocalToneMapping()
    // ── Tonemap colour-grade (white-balance / tint / contrast / shadows / highlights) ──
    /// White-balance colour temperature in Kelvin (~2000–10000). 6500 = neutral
    /// (channel gain (1,1,1) → exact no-op). Photo-tool convention (DH-0453):
    /// higher K = warmer, lower K = cooler — the inverse of the physical blackbody
    /// sign, matching Lightroom/in-camera WB dials. Applied as a linear-HDR channel
    /// multiply BEFORE exposure + ACES.
    public var whiteBalanceK: Float = 6500
    /// Green↔magenta tint on [-1, 1]. 0 = no-op; >0 magenta, <0 green. Luma-
    /// preserving channel gain applied pre-tonemap alongside white-balance.
    public var tint: Float = 0
    /// Shadow lift/crush, ~0.5–1.5. 1.0 = no-op; >1 lifts blacks, <1 crushes.
    /// Luma-weighted tone-curve term in the tonemapped (0..1) domain.
    public var shadows: Float = 1.0
    /// Highlight lift/roll-off, ~0.5–1.5. 1.0 = no-op. Luma-weighted, post-tonemap.
    public var highlights: Float = 1.0
    /// Contrast about mid-grey 0.18, ~0.7–1.3. 1.0 = no-op. Post-tonemap curve.
    public var contrast: Float = 1.0

    // ── Photographic finish (the two things a flat saturation multiply undoes) ──
    /// **Highlight chroma roll-off** — how much a pixel's chroma fades as it
    /// approaches white, 0…1. 0 = OFF (default) → an exact shader no-op.
    ///
    /// ACES already desaturates toward white; `tonemapSaturation` then pushes that
    /// chroma back out with a `mix(luma, colour, S)` that is FLAT in luminance, so
    /// the two fight and the bright end keeps full saturation. That is what makes a
    /// lamp-lit wooden wall or a sunlit render clip to a saturated block of colour
    /// rather than to the pale warm white a camera gives. This restores the roll-off
    /// WITHOUT giving up the midtone chroma the saturation multiplier is there for.
    public var highlightChromaRolloff: Float = 0
    /// **Split tone, shadow end** — colour temperature (Kelvin) applied to the low-luma
    /// end through the same `whiteBalanceGain` curve as the global white balance
    /// (luma-normalized: it tints, it does not dim). 6500 = neutral (default) → an
    /// exact no-op. BELOW 6500 cools the shadows (photo convention, DH-0453), which
    /// is what gives warm practical lights something to contrast against instead of
    /// every tone sitting in one warm universe.
    public var shadowTemperatureK: Float = 6500
    /// **Split tone, highlight end** — the same, masked to the high-luma end.
    /// 6500 = neutral (default) → an exact no-op. Above 6500 keeps highlights warm;
    /// pair it with `highlightChromaRolloff` for the photographic combination —
    /// the scene's own chroma rolls off toward white, and a controlled amount of
    /// warmth goes back in.
    public var highlightTemperatureK: Float = 6500

    // ── DH-0715: live-lane look-match ─────────────────────────────────────────
    /// Master enable/blend [0,1] for the live-lane approximation of the still's
    /// RT bounce-GI + RTAO passes (`illumi_tonemap_fs`, applied in the HDR
    /// domain BEFORE white-balance/exposure/ACES/grade — see that file for why
    /// that ordering is what makes this compose with whatever Visuals settings
    /// a scene has, rather than fighting them). 0 (default) = exact no-op. Hosts
    /// gate this live-lane-only — a still keeps the real RT terms.
    public var liveLookMatchStrength: Float = 0
    /// Multiplier the fake-bounce term mixes toward in shadow/mid tones, <1.
    /// The still's real GI reads DARKER at the same exposure (ACES's shoulder
    /// compresses the wider dynamic range bounce adds), not brighter.
    ///
    /// **Calibrated 2026-09-11, TWICE.** First against one media-console frame
    /// alone (0.44) — validated against 3 real Views × 4 `RenderStyle` presets
    /// (12 combos) and found to close anywhere from 101% to 1668% of the
    /// measured gap depending on scene/preset, because a flat constant has no
    /// way to know how big the real gap actually is. `illumi_tonemap_fs` then
    /// gained two adaptive normalizers (`gradeNorm` off `tonemapSaturation`,
    /// `sceneNorm` off the auto-exposure EMA) so the term scales with the
    /// scene and the Visuals dials instead of applying a fixed push, and this
    /// is the REFIT against that shader (`HouseRenderBridgeGPUTests
    /// _LiveLookMatchMatrix.testLiveLookMatchMultiSceneCalibration`, a 4-scene
    /// joint fit spanning both saturation extremes and two very different
    /// brightness profiles): total squared error 15.08 (no correction at all)
    /// → 6.15 at this value, a real interior minimum (error rises again past
    /// it in both directions), not a boundary artifact.
    public var liveLookGIDarken: Float = 0.15
    /// Strength of the fake-bounce warm gain spread (push red up / blue down),
    /// clamped [0, 4] — NOT a 0..1 blend fraction: it scales the tilt's own
    /// magnitude continuously. **0 after the 2026-09-11 multi-scene refit**: the
    /// joint fit across 4 scenes never preferred a nonzero value — the darken
    /// term's own interaction with the existing post-ACES saturation lerp
    /// already carries the chroma match, and adding an explicit R/B tilt on
    /// top only overshot it. Left as a live lever (not deleted) in case a
    /// future fit over more scenes finds it earns its keep after all.
    public var liveLookGIWarmth: Float = 0.0
    /// Multiplier the fake-AO term mixes toward, keyed by the EXISTING AO
    /// visibility buffer (`inAO`) rather than a flat screen tint — extra
    /// contact darkening approximating RTAO, riding on real occlusion
    /// geometry. Refit 2026-09-11 alongside `liveLookGIDarken` (see there) —
    /// the error surface was nearly flat over [0.40, 0.70] here, so this
    /// sits in the middle of that range rather than at the exact minimum.
    public var liveLookAODarken: Float = 0.50
    /// **What the look-match's AO share actually mixes toward THIS frame (Daydream DH-0785).**
    /// `liveLookAODarken` is a STAND-IN for ray-traced AO on a lane that never traces it, and it
    /// reads the same AO buffer RTAO writes into. When RTAO is live for the frame the buffer
    /// already carries real occlusion, so the stand-in would darken every corner a second time —
    /// on exactly the settled frames a host shows while it converges. 1.0 makes the shader's
    /// `mix(1, darken, aoMask)` a no-op. The GI share is untouched: GI is a separate term, and a
    /// host that traces AO without GI still wants it.
    public var effectiveLiveLookAODarken: Float { rtaoActive ? 1.0 : liveLookAODarken }
    /// Opt-in hex-stochastic anti-tiling strength [0,1]. **Default 0 = OFF**, which
    /// is a hard no-op: the G-buffer shader short-circuits every textured sample to a
    /// single plain read, so scenes that leave this at 0 render byte-for-byte
    /// identical to the pre-anti-tiling engine. Raise it (e.g. Daydream Home's large
    /// tiled floors/walls) to break the visible tile-repeat on big planar surfaces —
    /// three stochastically-offset lattice taps blended with variance-preserving
    /// cubic weights, applied consistently across albedo / roughness / normal so the
    /// channels stay registered. 1.0 = full de-repeat; intermediate values crossfade.
    public var antiTilingStrength: Float = 0
    /// Saturation boost on the IBL diffuse term in the lighting kernel.
    /// Procedural-gradient backdrops (HotdogDrop+'s peach→tan, every
    /// `+`-tier sky) integrate to a near-grey irradiance, and the
    /// `diffuseIBL = kD · irradiance · albedo` term then multiplies grey
    /// by every albedo → muted indirect lighting. Boosting the
    /// irradiance probe's chroma vs. its luminance preserves the
    /// surface tint through the indirect path too. 1.0 = no change.
    public var iblDiffuseSaturation: Float = 1.30
    /// Phase 4.32 — hue-balance DESATURATION of the indirect fill (diffuse-IBL
    /// irradiance + ambient supplement) for highly-saturated probes. 0 = off
    /// (default). Counterpart to `iblDiffuseSaturation`'s boost: a scene whose
    /// IBL is a saturated colored environment (Pizza's red broiler) washes
    /// neutral surfaces toward that hue through the indirect path. This pulls
    /// the monochromatic fill back toward luminance, ramped in only as the
    /// probe's own saturation rises (neutral/pastel IBL is untouched). The
    /// lever the #46 magnitude knobs (intensity/exposure) couldn't move.
    public var iblDiffuseDesaturation: Float = 0

    /// Scotopic (Purkinje) desaturation strength, applied in the tonemap. Pulls
    /// only the DIMMEST tonemapped pixels toward their own luminance so dark night
    /// surfaces read neutral moonlit-dark instead of holding their (green grass)
    /// albedo tint; bright pixels (lamps/moon/stars) keep full colour. 0 = OFF
    /// (default) → an exact no-op (Visualizer byte-identical). Hosts typically
    /// fade this in with their night blend.
    public var scotopicDesaturation: Float = 0

    // ── Phase 4.39 denoiser knobs — SPATIAL and TEMPORAL are separate ───
    //
    // These were one flag each (`ssaoDenoiseEnabled` / `ssrDenoiseEnabled`), which
    // made the only way to stop the temporal half's ghosting a fallback to RAW AO.
    // They fail differently and are now gated independently. The compatibility
    // shims below keep pre-split hosts compiling and behaving as they did.

    /// **Edge-aware SPATIAL filter on the half-res AO** (depth/normal-weighted
    /// blur). Intra-frame, so it cannot ghost — this is the half that makes raw
    /// 16-sample SSAO stop reading as grain. Default ON.
    ///
    /// Note the pass is still ENCODED when `ssaoTemporalEnabled` is on, because
    /// the temporal kernel reads `aoFilteredTexture` as its input; with this flag
    /// off the kernel passes the raw AO through instead of filtering it. See
    /// `ssaoSpatialRuns`.
    public var ssaoSpatialEnabled: Bool = true
    /// **Velocity-reprojected TEMPORAL accumulation of the AO** (up to 32 frames).
    /// Default OFF: the reprojection uses the velocity of the pixel it is AT, but
    /// AO is a function of that pixel's NEIGHBOURHOOD — a mover's contact darkening
    /// lands on a static receiver whose velocity is zero, so the history reprojects
    /// onto itself and trails behind the mover. Neither guard catches it (the
    /// disocclusion term reads the same zero velocity; the 3×3 ±1.5σ clamp has a
    /// σ floor of 0.03). Fine for a static or slow camera; opt in per scene.
    public var ssaoTemporalEnabled: Bool = false
    /// History retention weight for SSAO temporal (0 = all current, 1 = frozen).
    /// 0.90 gives ~10-frame effective sample count with fast transient recovery.
    public var ssaoTemporalBlend: Float = 0.90
    /// **Velocity-reprojected TEMPORAL accumulation of the SSR gather.** Kills
    /// shimmer on rough/medium surfaces; smears a reflection of anything that moves
    /// across a static reflector, for the same structural reason as the AO twin.
    /// Default OFF. Off = raw single-sample gather goes straight to composite.
    ///
    /// SSR has NO spatial stage (gather → temporal → composite), so there is no
    /// `ssrSpatialEnabled` to pair with this.
    public var ssrTemporalEnabled: Bool = false
    /// History retention weight for SSR temporal. 0.85 balances lag vs noise.
    public var ssrTemporalBlend: Float = 0.85
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
    /// When true (default), `render()` applies the panel's denoiser flags every
    /// frame, so every Illuminatorama scene — including future ones — inherits them
    /// with zero per-scene wiring (the same policy as `appliesSharedLensFX`, and
    /// the thing that stops 11 controllers each hand-copying the two flags). A
    /// scene that must art-direct its own denoise sets this false.
    public var appliesSharedDenoiser: Bool = true
    /// Apply the triangular-PDF debanding dither in the tonemap before the
    /// 8-bit store. On = smooth gradients; off = raw 8-bit quantisation
    /// (exposes contour banding). Exposed for live A/B of gradient banding.
    public var debandDitherEnabled: Bool = true
    /// Opt-in: dither in the TRUE sRGB encoding the 8-bit store applies (its linear toe
    /// included) instead of the legacy `pow(1/2.2)` approximation. Near black the two diverge
    /// — the approximation's slope is up to ~5× the real OETF's below code 4, so its "±1 LSB"
    /// lands as ±0.2–0.5 of a real code and a dark gradient's lowest steps are barely
    /// dithered. A night frame whose whole wall sits in codes 0–11 (Digital Clock) needs the
    /// real ±1 LSB there. False (default) ⇒ the legacy dither, byte-identical.
    public var debandDitherExactSRGB: Bool = false

    // ── Display chain for a physically-scaled night (VZ-0170) ────────────
    /// **HDR pre-exposure** (Frostbite / UE style). The lit frame is stored in rgba16Float, whose
    /// normal range ends at 6.1e-5 and whose smallest subnormal step is 6e-8: a scene kept on a
    /// real photometric scale anchored on a bright emitter (Digital Clock: 1 unit ≈ 1050 cd/m²)
    /// puts a moonlit room at ~1e-7 — a handful of subnormal steps — and a moonless one at
    /// exactly 0, before any exposure or local adaptation can see it. A host that sets this to
    /// K promises it has scaled EVERY light, emission and sky radiance it feeds the frame by K
    /// (the same scene, K times brighter in the buffer); the renderer then divides K back out
    /// wherever an absolute level matters — the tonemap's and the local tone mapping's exposure,
    /// the auto-exposure meter's samples, the bloom / halation thresholds, and the TAA / bloom
    /// anti-firefly `1/(1+luma)` weights — so the printed frame is the unscaled one, with the
    /// dim end kept in fp16's normal range. The HDR dump (`VIZ_ILLUMI_HDR_DUMP_PATH`) records the
    /// K-scaled pixels with the matching exposure, so exposed levels read the same.
    /// 1 (default) ⇒ every divide is by exactly 1 ⇒ byte-identical.
    public var hdrPreExposure: Float = 1
    /// **Hue-stable toe** — opt-in knee [lo, hi] in EXPOSED brightness (the meter's metric,
    /// max(luma, ½·max channel), after exposure and local tone mapping). AgX's per-channel sigmoid
    /// (and its 'punchy' power) behaves like a ~2.3-power in its toe, so every channel RATIO in
    /// the dark end is raised to roughly that power — more with the post-tonemap saturation push
    /// on top (measured on Digital Clock's 19:40 frame: a lavander sky, linear 1 : 0.66 : 1.63,
    /// printed (14, 2, 53)). Below `lo` the display transform runs on the pixel's max-RGB norm
    /// and rescales its colour (the scene's own linear ratios, no per-channel toe) and the
    /// saturation push is off; above `hi` both are exactly the shipped ones; between, a
    /// smoothstep blend. `.zero` (default, `hi == 0`) ⇒ off, byte-identical.
    public var hueStableToe: SIMD2<Float> = .zero
    /// Opt-in, with `hueStableToe` and the AgX display transform: the toe's LEVEL comes from AgX's
    /// bare sigmoid instead of its 'punchy' look. The look's power 1.35 acts as a ~2.3-power toe —
    /// 4 stops under mid-grey prints at sRGB 4, 5 stops at 1 — so a locally-adapted night room is
    /// fitted into ~4 printable stops and re-crushed; the bare sigmoid prints 4 stops under at 21,
    /// 5 under at 7, 6 under at 2 (Digital Clock round 3). The toe keeps the scene's colour ratios
    /// either way; above the knee the shipped (punchy) transform is untouched. False (default) ⇒
    /// displayParams.w = 0 ⇒ byte-identical.
    public var hueStableToeBareLevel: Bool = false

    // ── Phase 9 — film-stock LUT (post-tonemap colour grade) ─────────
    /// The film stock's cube as an `MTLTexture3D`, N×N×N. `nil` = stock bypassed
    /// (identity pass-through). Assign from the host via `setFilmLUT(_:strength:)` —
    /// do NOT set this, `filmLUTStrength` or `filmLUTSize` directly.
    public internal(set) var filmLUTTexture: MTLTexture? = nil
    /// Blend weight [0, 1]: 0 = stock fully bypassed, 1 = the stock's own rendering.
    public internal(set) var filmLUTStrength: Float = 1.0
    /// Cells per axis of `filmLUTTexture`, taken from the texture itself rather than
    /// assumed — the shader's half-texel inset is derived from it (DH-0879).
    public internal(set) var filmLUTSize: Float = 16

    // ── Phase 2.7 TAA knobs ──────────────────────────────────────────
    /// Toggle temporal anti-aliasing + history denoise. Off = bloom/tonemap
    /// read the raw post-SSR HDR (Phase 2.5 behaviour).
    ///
    /// AUTHORITY: this is NOT a per-scene knob. `render()` overwrites it every
    /// frame from `IlluminatoramaSharedSettings.shared.taaEnabled` (unless a
    /// scene sets `sharedTAAOverride`), so the Illuminatorama Settings panel is
    /// the single source of truth. Scenes used to each set this themselves and
    /// drifted — some hardcoded on, some off, some read a scene-local flag — so
    /// the panel toggle was silently ignored. Default matches the shared default
    /// (off) so any path that skips the per-frame pull also starts off.
    public var taaEnabled: Bool = false
    /// Per-scene escape hatch from the panel's TAA authority. `nil` (default) =
    /// follow the panel. Set to a concrete value ONLY when a scene must force
    /// TAA on/off regardless of the panel (e.g. a perf governor disabling it
    /// under load). Prefer leaving it `nil` — the panel should win.
    public var sharedTAAOverride: Bool? = nil
    /// When true (default), `render()` applies the panel's lens-aberration knobs
    /// (chromatic + spherical aberration, fringe, post-FX easing) every frame, so
    /// every Illuminatorama scene — including new ones — inherits them with zero
    /// per-scene wiring. (Bloom is NOT included: scenes art-direct it, so they opt
    /// in via `applySharedPostFX`.) A context that must not be driven by the app
    /// panel (e.g. an isolated AssetLab harness) can set this false.
    public var appliesSharedLensFX: Bool = true
    /// Weight of the current-frame sample in the resolve blend. Smaller =
    /// smoother and more denoised but slower convergence on disocclusion.
    /// 0.1 is the standard "good default" — visible AA from camera jitter
    /// within ~10 frames, and the SSAO/SSR noise drops out almost
    /// immediately on a still camera.
    public var taaHistoryBlend: Float = 0.1
    /// Magnitude of the sub-pixel projection jitter, in pixels. Standard
    /// TAA uses a Halton sequence within ±0.5 px so successive frames
    /// supersample the same output pixel; values >0.5 push samples beyond
    /// the reconstruction filter footprint, making geometry visibly walk
    /// between frames. With SSAA on (`internalRenderScale > 1`) the AA
    /// role is filled by spatial oversampling — leaving jitter at 0 so
    /// TAA contributes denoise of SSAO/SSR noise without adding any
    /// motion. Set back to 0.5 to re-enable temporal supersample AA.
    public var taaJitterPixels: Float = 0.0
    /// The internal → output reduction (Daydream DH-1365): `.legacy` (default, byte-identical) is the
    /// tonemap's 4-tap bilinear; a declared kernel gathers every internal texel within its radius
    /// in linear HDR. See `IlluminatoramaReconstructionFilter`. Rides `FrameUniforms.taaJitterDelta.zw`.
    public var reconstructionFilter: IlluminatoramaReconstructionFilter = .legacy

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

    // ── Phase 3 IBL knobs ────────────────────────────────────────────
    /// Master scale on diffuse + specular IBL contribution. 1 = unscaled.
    public var iblIntensity: Float = 1.0
    /// Toggle the IBL path. When `false`, the lighting kernel falls back to
    /// the legacy hemispheric ambient; sky pixels still come from
    /// `equirectSky` when one is supplied.
    public var iblEnabled: Bool = true
    /// External equirect HDR sky source — typically
    /// `VolumetricCloudRenderer.outputTexture`. When `nil`, the renderer
    /// uses a 1×1 black fallback (sky pixels render black, IBL contributes
    /// nothing). The texture is sampled, so it must be `.shaderRead`-capable.
    public var equirectSky: MTLTexture?
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

    public var iblNeedsRebake: Bool = true
    /// How many seconds may elapse between IBL bakes even when `iblNeedsRebake`
    /// stays false. Default 5 s — enough to keep slow cloud drift from going
    /// completely stale without paying full 60fps bake cost.
    public var iblRebakeInterval: Float = 5.0

    /// **Compile the `illumi_lighting` feature variant BLOCKING instead of async.**
    ///
    /// The per-frame path (`currentLightingPipeline`) normally compiles a
    /// newly-needed function-constant variant in the BACKGROUND and runs the
    /// init-time uber-variant meanwhile (WWDC23 #10127 lever 2), so a live host
    /// never hitches when a feature flag first toggles. The completion handler
    /// lands the compiled variant via `Task { @MainActor }`, which needs the main
    /// runloop to spin.
    ///
    /// A headless `@MainActor` capture harness that renders in a tight synchronous
    /// `render(blocking:)` loop STARVES that hop — the main actor never yields, so
    /// the specialized variant never lands and every frame silently runs the
    /// init-time uber-variant. The uber-variant is built with the flags present at
    /// init, so a harness that flips a flag and settles a still measures a lane
    /// whose flag never ran (DH-0552). Same trap for any flag that differs from init.
    ///
    /// Set this in such a lane to compile the needed variant BLOCKING (the same
    /// choice the RT-sun variant makes for its settled/photo lane): the frame then
    /// actually runs the specialized path. Leave OFF for any live/vsync host —
    /// blocking a frame on a fresh compile is exactly the hitch the async path
    /// exists to avoid. Default OFF.
    public var blockingLightingCompile: Bool = false

    /// TEST-OBSERVABLE (DH-0869): the specialisation flags of the lighting variant the LAST
    /// frame actually dispatched — bit 1 ibl · 2 shadow · 4 dfg · 32 rtSun.
    /// `0xFF` until a specialised variant has run. On the non-blocking path a not-yet-compiled
    /// combination runs the init-time variant and leaves this at the PREVIOUS value, so a test
    /// that flips a flag can assert the frame really ran it.
    public var activeLightingVariantFlags: UInt8 { gbs.lastLightingFlags }

    // ── Phase 3.2 split-sum DFG LUT ──────────────────────────────────
    /// Toggle the pre-integrated DFG LUT for the F0-weighted IBL specular
    /// response. Off = Lagarde's roughness-Schlick approximation (Phase 3.0
    /// behaviour). The LUT itself is baked once at startup and is
    /// view/material/scene-independent, so there is no rebake cost.
    public var dfgLUTEnabled: Bool = true

    /// **DH-1014 — the layered BSDF's energy bookkeeping** (IlluminatoramaBSDF.h). `true`:
    /// multiple-scattering GGX compensation on EVERY specular lobe (sun, lamps, fills, area
    /// portals — the IBL arm has had it since S1.3b) and the diffuse base weighted by what the
    /// specular layer did not reflect, in the deferred pass AND in the ray-traced lighting
    /// kernel's GI / reflection weights, so a white surface under a uniform environment returns
    /// 1.0 at every roughness on the live lane and on Maximum alike. Needs `dfgLUTEnabled`.
    /// `false` (the default) is every lobe's historical expression, byte for byte — a look
    /// change is the host's call (Daydream Home opts in).
    public var layeredMultiscatter: Bool = false
    /// **DH-1301 — diffuse roughness (Oren-Nayar).** `true`: the DIRECT diffuse lobe of rough
    /// surfaces (sun, points, spots, fill directionals, the traced primary sun) is Fujii's
    /// qualitative Oren-Nayar instead of Lambert, so plaster / carpet / matte paint backscatter
    /// toward grazing instead of reading flat. sigma = saturate((roughness − 0.4)/0.6)·0.6 rad
    /// (roughness ≤ 0.4 is exactly Lambert). LTC area lights and the IBL diffuse stay Lambert.
    /// `false` (the default) is byte-identical to before — a look change is the host's call.
    public var diffuseRoughnessEnabled: Bool = false
    /// **DH-1014 / DH-0597 — cloth sheen as a layer.** `true`: the sheen's environment arms read
    /// the fitted Charlie albedo from the DFG LUT (`.a`) instead of the hand-fitted curve, and
    /// the base under the sheen loses what the sheen reflected (albedo scaling), in the deferred
    /// pass and the RT weights. `false` (the default) is byte-identical.
    public var layeredClothSheen: Bool = false

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

    /// Per-frame tap, fired at the end of `present(to:)` with the just-presented
    /// `outputTexture` and this renderer's `commandQueue`. The video recorder
    /// installs it to capture native (MTKView) scenes — there's no SceneKit
    /// render delegate to hook on this path. Copying on the renderer's OWN queue
    /// (not a fresh one) keeps the capture ordered behind the frame's render, so
    /// it reads a fully-written texture. `nil` (default) = no capture overhead.
    /// Set/cleared on the main thread; `present(to:)` runs on main.
    public var onFramePresented: ((_ texture: MTLTexture, _ queue: MTLCommandQueue) -> Void)?

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
    var meshes: [MeshKind: IlluminatoramaMesh] = [:]

    // ── AAA ray-traced glass (#60) ────────────────────────────────────
    /// Host sets these to render glass. `glassPaneKind` is a registered mesh and
    /// `glassPaneInstance` is one glass instance (transform + material). Nil → no
    /// glass. When ray tracing is available the glass is rendered by
    /// `illumi_glass_rt_fs` — true entry+exit refraction, reflection, Beer–Lambert
    /// absorption, and dispersion traced against the scene TLAS (the glass meshes
    /// are added to that TLAS so glass refracts/reflects real geometry and other
    /// glass). Without RT it falls back to `illumi_glass_fallback_fs` (single-
    /// surface Fresnel + sky). Drawn into the HDR composite after lighting,
    /// depth-tested but not written, so opaque geometry in front occludes it.
    public var glassPaneKind: MeshKind? = nil
    public var glassPaneInstance: IlluminatoramaGlassInstance? = nil
    /// Multiple glass instances drawn in ONE instanced draw, all sharing
    /// `glassPaneKind`. Takes precedence over `glassPaneInstance`. Used by Floating
    /// Lenses for an orbit of tumbling concave lenses. Sort back-to-front (far →
    /// near): the pass doesn't write depth, so draw order resolves overlaps (the RT
    /// path returns opaque alpha, so the nearest pane — drawn last — wins, and its
    /// trace already composited everything behind it).
    public var glassInstances: [IlluminatoramaGlassInstance] = []

    public var glassMeshGroups: [GlassMeshGroup] = []

    /// Master switch for the ray-traced glass path. OPT-IN (default off): a scene
    /// that wants AAA glass sets this true, which makes the renderer add its glass
    /// meshes to the TLAS and render them through `illumi_glass_rt_fs` (true
    /// refraction/reflection). Off → the lightweight Fresnel+sky fallback
    /// (`illumi_glass_fallback_fs`), with NO TLAS cost — so existing glass scenes
    /// (incidental tube/wall glass) are unaffected until they opt in. The extractor
    /// sets this automatically when it finds a glass-flagged `SCNMaterial`.
    public var rtGlassEnabled: Bool = false
    /// Dielectric bounce cap for the refraction path (entry+exit ≈ 2; raise for
    /// lens-through-lens stacks). Clamped to 10 in-shader.
    public var rtGlassMaxBounces: UInt32 = 6
    /// Sun shadow rays cast when an opaque hit inside the glass trace has to be
    /// re-shaded (cache off). 1 is usually enough — TAA cleans the penumbra.
    public var rtGlassShadowRays: UInt32 = 1
    /// Global gate for chromatic dispersion. Per-instance `dispersion` still has
    /// to be > 0; this lets a host disable the 3× cost wholesale.
    public var rtGlassDispersionEnabled: Bool = true

    /// **Take the transmitted lobe from the frame we already rendered, near normal.**
    ///
    /// The RT glass pass re-derives the world behind a pane with `shadeSecondarySurface`,
    /// which is a poorer model than the deferred kernel that shaded the same surfaces one
    /// pixel to the side. Measured on a real document, the transmitted world came out
    /// **1.20× the same view with the glazing removed** — a pane brighter than its own hole,
    /// which is what stops a settled window reading as glass. Real float glass transmits ~0.92.
    ///
    /// With this on, at near-normal incidence — where a 6 mm pane's ray displacement is
    /// sub-pixel — the transmitted radiance is sampled from the pre-glass composite instead of
    /// re-shaded. That is parity with the neighbouring pixels by construction, and one texture
    /// tap is cheaper than a ray cast plus a re-shade. It fades back to the traced path over
    /// cosθ 0.95 → 0.80, so curved or strongly-refracting glass (jewels, bubbles) keeps the RT
    /// path across essentially its whole silhouette and is unaffected.
    ///
    /// Costs one full-frame blit of the composite per frame when the RT glass pass runs.
    public var rtGlassScreenSpaceTransmission: Bool = true

    /// **Trace RT glass at the resolution the lens resolves** (VZ-0197). OPT-IN: 1 (the
    /// default) traces every glass fragment, exactly as before.
    ///
    /// With depth of field on, a pane BEHIND the focus plane reaches the frame through the DOF
    /// gather, which spreads each pixel over its circle of confusion — a window behind a macro
    /// subject sits at the CoC clamp, tens of pixels. Everything the per-pixel glass trace
    /// resolves finer than that disc is integrated away, and it is the frame's single most
    /// expensive pass (≈ 7 ray traversals per glass pixel: the front-surface test, entry, exit,
    /// the world behind, the reflection, and a shadow ray at each shaded hit).
    ///
    /// k = 2…4 (clamped) traces the glass once per k × k pixels into a small target, then the
    /// ordinary full-resolution glass pass takes that result — reconstructed bilinearly, each
    /// low-res texel weighted by its own coverage — wherever the pane point is behind the focus
    /// plane and defocused by at least `rtGlassDefocusMinCoCTexels × k` px. Everywhere else
    /// (in focus; in front of the focus plane, where the DOF follows the opaque depth BEHIND the
    /// glass and may keep it sharp; where no low-res texel covers the pixel) it traces in full.
    /// Off whenever `dofEnabled` is off. The low-res pass skips the depth test, so the trace is
    /// continuous behind an occluder's silhouette and the reconstruction never mixes in anything
    /// that is not glass.
    public var rtGlassDefocusTraceFactor: Int = 1
    /// The defocus trace's gate, in LOW-RES texels of CoC radius: a pixel takes the reduced trace
    /// when its |CoC| ≥ this × k px. At 4 a k-px bilinear reconstruction sits under a ≥ 4k-px
    /// disc, which widens the blur by ~2 % (σ² adds: (0.41k)² against (2k)²).
    public var rtGlassDefocusMinCoCTexels: Float = 4

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

    /// Sample the albedo atlas at the hit's barycentric UV. Off ⇒ per-instance mean albedo.
    public var glassHitTexturingEnabled: Bool = true
    /// Accumulate point + spot lights at the hit. Off ⇒ sun + sky only, so a lamp-lit
    /// room goes dark through a window.
    public var glassHitLocalLightsEnabled: Bool = true
    /// Apply the host's interior IBL/ambient separation at the hit. Off ⇒ an interior
    /// surface gets exterior-strength fill, several times short.
    public var glassHitInteriorFillEnabled: Bool = true
    /// DH-0718 — light every secondary hit (a reflection, a GI bounce, the world through a
    /// pane) with the frame's rectangular AREA lights too, by the same form factor the
    /// deferred kernel uses. Off ⇒ the pre-DH-0718 behaviour: a secondary hit saw sun + sky
    /// + point/spot lights but no area light — which in Daydream Home is every window
    /// portal, i.e. a room's dominant daytime source, so a reflected room was lit by a
    /// different light than the room itself and cast none of that light's shadows.
    public var secondaryAreaLightsEnabled: Bool = false   // DH-0718 WIP: default flips to true when it lands
    /// DH-0718 — visibility rays per shadow-casting area light at a secondary hit (toward
    /// jittered points on the rectangle, like `rtAreaShadowRays` on the primary). 0 ⇒ lit
    /// but unoccluded. 1 by default: an accumulating still converges it like the sun cone.
    public var secondaryAreaShadowRays: Int = 1

    /// CHEAP semi-transparent glass (no TLAS, no per-frank BLAS). Off (0) keeps the
    /// plain Fresnel+sky fallback unchanged for every other scene. A host opts in
    /// when it wants "semi-transparent glass" rather than "hardware-RT refraction"
    /// and can't afford a BLAS-per-deforming-body TLAS (Hot Dog Press: #64). Drawn
    /// through `illumi_glass_fallback_fs`, gated on `u.cheapGlassMode`:
    ///   1 = FRESNEL  — synthetic tinted translucent sheet + Fresnel edge + a key
    ///                  specular glint, alpha-blended over the lit scene. Cheapest;
    ///                  the panes read as a coloured glass case but don't lens/tint
    ///                  the bodies behind multiplicatively (no scene read).
    ///   2 = SCREEN-SPACE — samples the pre-glass composite behind the pane
    ///                  (`glassBackdropTexture`), Beer–Lambert-tints + screen-space-
    ///                  refracts it, adds Fresnel + key glint. True "look through
    ///                  tinted glass" read; one extra full-frame blit + one texture
    ///                  sample per glass fragment, still no ray tracing.
    /// Only consulted on the NON-RT path (`rtGlassEnabled == false`); RT glass wins.
    /// Mode 2 reads the per-pane screen-space refraction strength (pixels at grazing)
    /// from each glass instance's `rdrf.w` (the reserved "fresnelPower" slot).
    public var glassCheapMode: Int = 0

    /// THIN-FILM IRIDESCENCE for cheap-glass mode 2 (soap bubbles — Bubble Lab).
    /// Master strength of the interference coat added on top of the screen-space
    /// refraction. 0 (default) → an EXACT no-op, so every other cheap-glass scene
    /// is unaffected. The coat colour is computed in-shader from the optical path
    /// difference `2·n·d·cosθ` at R/G/B wavelengths — real two-beam interference,
    /// not a scrolled texture. Per-bubble thickness variation rides in each glass
    /// instance's `dispersionPad.y`.
    public var thinFilmStrength: Float = 0
    /// Base soap-film thickness in nanometres at the equator (interference order).
    public var thinFilmThicknessNm: Float = 320
    /// Refractive index of the film itself (water + surfactant ≈ 1.33).
    public var thinFilmIOR: Float = 1.33
    /// OSCILLATION-MODE surface undulation for the bubble shells (the glass VS
    /// displaces each vertex radially by a sum of beating angular modes — real
    /// per-vertex lobing an affine transform can't produce). 0 (default) → no
    /// displacement, so every other glass scene is unaffected. Radial amplitude
    /// in object-space units (the unit sphere has r = 1).
    public var bubbleWobbleAmp: Float = 0
    /// Global rate multiplier for the beating oscillation modes.
    public var bubbleWobbleFreq: Float = 1

    /// Grows on demand in `encodeGlassPass` to hold the flattened glass instances.
    lazy var glassInstanceBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaGlassInstance>.stride,
                          options: .storageModeShared)!
    }()
    lazy var glassRTUniformBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaGlassRTUniforms>.stride,
                          options: .storageModeShared)!
    }()

    // ── Glass caustics (#60) ───────────────────────────────────────────
    /// Photon-traced caustics for the RT glass, splatted into the surface-cache
    /// card atlas (so they land on ANY receiver) and read back per primary pixel.
    /// Opt-in AND requires the surface cache on (that's the card store). Off → no
    /// cost. See `docs/illuminatorama/glass.md`.
    public var causticsEnabled: Bool = false
    /// Photons traced per frame. Energy accumulates across frames via `causticDecay`,
    /// so fewer photons/frame just converge a touch slower — 400k is a good
    /// perf/quality balance (1M was ~3× the cost for marginal gain).
    public var causticPhotonCount: Int = 200_000
    /// Composite strength of the caustic term added to the lit receiver.
    public var causticStrength: Float = 1.0
    /// Per-frame EMA decay of the caustic atlas. Higher = more frames accumulated
    /// = less photon speckle (at the cost of slower response to motion). 0.94 ≈ a
    /// ~17-frame running average — smooth pools on a static-ish scene.
    public var causticDecay: Float = 0.94
    /// Per-photon flux scale (normalisation). Tuned for ~400k photons over a small
    /// scene; scale inversely if you change `causticPhotonCount` a lot (brightness
    /// ≈ energy × count). The receiver store is the surface-cache card atlas, so
    /// caustics land at GI-card resolution — focused pools, not razor filaments.
    public var causticPhotonEnergy: Float = 0.06
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

    /// 0 = OFF (default) → the tonemap LUT branch is an exact no-op. 1 = full grade.
    public var colorLUTAmount: Float = 0
    /// Velocity-buffer motion blur (issue #65). `motionBlurStrength` scales the
    /// per-frame screen velocity (≈ shutter fraction; ~0.5 is a natural 180° look);
    /// 0 = OFF → the tonemap gather is skipped (exact no-op). `motionBlurMaxPx` caps
    /// the streak length in output px so a fast pan can't smear the whole frame.
    /// Driven by camera AND per-object motion (the renderer ping-pongs the instance
    /// buffer, so spinning/translating instances streak correctly).
    public var motionBlurStrength: Float = 0
    public var motionBlurMaxPx: Float = 48
    /// Screen-space contact shadows (issue #65). A short screen-space ray march
    /// toward the PRIMARY directional sun in the deferred lighting pass, catching
    /// the fine contact occlusion the cascaded shadow maps + RT soft shadows miss
    /// at object-base scale (a chip on felt, an egg on a floor). 0 strength =
    /// OFF → an exact shader no-op (the march is skipped). `contactShadowLength`
    /// is the march reach in WORLD units; `contactShadowSteps` the sample count;
    /// `contactShadowThickness` the occluder-depth window (acne guard). Driven
    /// from the panel via `applySharedLensFX` (universal off-by-default knob).
    public var contactShadowStrength: Float = 0
    public var contactShadowLength: Float = 0.05
    public var contactShadowSteps: UInt32 = 12
    public var contactShadowThickness: Float = 0.02
    /// Aerial perspective — extinction coefficient σ (1/m) for the distance-haze blend
    /// applied to the deferred lighting pass's FULL composite (never the isolated debug
    /// terms): `mix(airlight, color, exp(-σ·viewDist))`, with the airlight colour sampled
    /// from the prefiltered sky cube at the view azimuth's horizon, so it follows time of
    /// day and sky state with no host plumbing. 0 (the default, and every scene that never
    /// opts in) = an EXACT shader no-op. Physically plausible clear-day Rayleigh is ~1e-4
    /// (≈4 % haze at 400 m) — this is the far-field cue that keeps a distant ground plane
    /// from arriving at the horizon fully saturated, not a fog effect.
    public var aerialPerspectiveDensity: Float = 0
    /// Multiplier on the aerial-perspective AIRLIGHT (default 1 = unchanged). The airlight is a
    /// coarse-mip sample of the prefiltered sky cube just above the horizon, so it averages in
    /// the brighter sky (and any moon aureole) above — by day that is the right broad inscatter,
    /// but under a physical night sky, whose horizon the renderer draws per pixel, it measured
    /// ~2.5× the horizon the camera actually sees (UFO Ultra: haze on the far land glowed as a
    /// lilac band under a black horizon). A host whose haze reads brighter than its own horizon
    /// scales it down here until the airlight is sampled from the model the sky pass draws.
    public var aerialPerspectiveAirlightScale: Float = 1
    /// Take the airlight from the sky cube's SHARPEST mip just above the horizon — the colour the
    /// camera sees at the horizon — instead of the coarse broad-inscatter mip. A scalar can't fix
    /// a hue mismatch: under a moonlit physical sky the coarse mip is the blue zenith's average
    /// while the horizon band is olive, so far land hazed to blue under an olive horizon.
    public var aerialPerspectiveAirlightFromHorizon: Bool = false
    /// Screen-space subsurface scattering (issue #65). Jimenez-style separable SSS
    /// for skin / wax / marble / food. `sssStrength` 0 = OFF → the lighting pass
    /// skips the side-buffer write and the blur/composite passes aren't encoded →
    /// an EXACT no-op (the off-by-default universal pattern). `sssRadius` is the
    /// diffusion mean-free-path in MILLIMETRES; `sssTint` scales the per-channel
    /// scatter distance (default reddish skin profile → warm bleed; marble ≈ 1,1,1;
    /// wax ≈ 1,0.7,0.45). A mesh opts a surface in by tagging its vertices with
    /// colour alpha ∈ [0.90,0.98] (→ `normalRoughness.w` ≈ 0.95h). Driven from the
    /// panel via `applySharedLensFX` (universal off-by-default knob).
    public var sssStrength: Float = 0
    public var sssRadius: Float = 8.0   // diffusion radius, millimetres
    public var sssTint: SIMD3<Float> = SIMD3(1.0, 0.4, 0.25)

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

    /// Enable the SVGF à-trous variance-guided cascade for the RT GI diffuse term.
    /// When off, falls back to the fixed-radius bilateral denoise (still reading the
    /// temporally-accumulated GI, so motion stability is preserved).
    ///
    /// The issue-#65 masonry banding is FIXED (see `encodeSVGFCascade`). Root cause
    /// was NOT the dilated sampling (the earlier hypothesis) but a texture-aliasing
    /// race in the cascade ping-pong: with only three scratch buffers, colour and
    /// variance shared `giAtrousB`, so each level≥1 read one signal at dilated
    /// neighbour taps while writing the other to the same texel at self — an in-place
    /// read/write race that corrupted colour in a dilated-stride pattern (the
    /// masonry). Per-frame grain hid it; temporal averaging exposed it. The fix is a
    /// fourth buffer (`giVarianceB`) so colour `{A,B}` and variance `{Var,VarB}` stay
    /// disjoint. MEASURED on illuminatoramaRoom (high-pass wall residual, lower =
    /// cleaner): broken 0.96% RMS → fixed 0.12%, vs the bilateral's 0.17% — i.e. the
    /// cascade now denoises BETTER than the bilateral, as intended.
    ///
    /// DEFAULT still OFF, now purely on PERF: the cascade (variance + N à-trous +
    /// composite) is ~+12% GPU frame time vs the single-pass bilateral on the Room
    /// (measured), so it stays opt-in pending a quality/perf call. Re-enable per
    /// scene (`renderer.svgfEnabled = true`) where the extra denoise is worth the
    /// cost. HotdogPress and CoinPusher explicitly keep it off.
    public var svgfEnabled: Bool = false
    /// The SVGF toggle after applying the env A/B override. Use this in the
    /// render path; `svgfEnabled` stays the plain stored property the panel sets.
    var effectiveSVGFEnabled: Bool { Self.svgfEnvOverride ?? (svgfEnabled || svgfStillForced) }
    /// Daydream DH-0887 — SVGF for a frozen-camera accumulation (a still, the settled canvas's
    /// refine) regardless of the process-wide `IlluminatoramaSharedSettings.svgfEnabled`, which
    /// is re-read every frame and shared by every window. Runs `svgfStillLevels` levels.
    /// Default `false`: byte-identical for every other host.
    public var svgfStillForced: Bool = false
    /// À-trous levels while `svgfStillForced` (2: the still lane's measured trade between
    /// residual noise and edge ringing, DH-0858).
    public var svgfStillLevels: Int = 2
    /// Daydream DH-0966 — while the GI is a static running mean, keep it as IRRADIANCE: the temporal pass
    /// divides each frame's G-buffer albedo out of that frame's own samples, and the last à-trous level
    /// multiplies the composing frame's albedo back in. The RT pass bakes the albedo into the GI, so a
    /// wide cascade over the modulated term blurs wood grain and fabric weave along with the noise.
    /// Default `true`; moving-camera frames never enter static accumulation and are untouched.
    /// `VIZ_ILLUMI_SVGF_DEMOD=0` A/Bs it.
    public var svgfDemodulateAlbedo: Bool = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF_DEMOD"] != "0"
    /// The GI history is irradiance this frame: only where the SVGF cascade is what consumes it (the
    /// bilateral fallback would composite it un-remodulated) and the history is a static mean.
    var giIrradianceMode: Bool {
        rtGIStaticAccumulation && svgfDemodulateAlbedo && effectiveSVGFEnabled && rtGITemporalEnabled
    }
    /// Number of à-trous cascade levels (1–5). Three levels cover a spatial
    /// reach of 1+2+4 = 7px radius; five levels cover 1+2+4+8+16 = 31px.
    public var svgfLevels: Int = 3
    /// Luminance variance weight (σ_L): smaller values produce tighter edge
    /// preservation; larger values allow more spatial averaging across edges.
    public var svgfSigmaL: Float = 4.0
    /// Depth edge-stopping weight (σ_Z). Default 1.0.
    public var svgfSigmaZ: Float = 1.0
    /// Normal edge-stopping exponent (σ_N). Higher = tighter. The clean bilateral
    /// fallback uses 16; at 128 the `dot(n,nN)^σN` weight amplifies the fp16
    /// oct-encoded normal g-buffer's quantization steps into structured wNorm
    /// jumps — banding contours on a smooth wall (issue #65). 16 matches the
    /// bilateral and keeps the weight smooth across coplanar fp16 normals.
    public var svgfSigmaN: Float = 16.0
    /// Luminance edge-stop floor, RELATIVE to local luminance (issue #65). The
    /// canonical SVGF luminance bandwidth `σ = σ_L·√Var + ε` assumes a temporally
    /// tracked, MC-noisy variance that stays positive on converged surfaces. This
    /// RT-GI is deterministic per frame, so Var genuinely → 0 once converged and a
    /// tiny fixed ε turns the edge-stop into a near-binary contour gate that bands
    /// the smooth wall into horizontal masonry. This floor sets ε =
    /// `svgfLumFloor · max(lumCenter, 0.02)`: large enough that a smooth
    /// low-amplitude gradient (Δlum ≪ floor) blurs freely, small enough that a
    /// genuine high-contrast GI discontinuity (shadow edge, Δlum ≫ floor) is still
    /// preserved. 0.6 ≈ gate onset around a ~30–40% luminance step.
    public var svgfLumFloor: Float = 0.6
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

    // ── DH-0534 — VISIBILITY CULLING, the renderer-side wiring ──────────────────
    // The logic and its correctness argument live in IlluminatoramaVisibilityCull.swift: a group
    // whose world bounds lie wholly outside one clip half-space of the matrix a pass rasterises
    // with cannot produce a fragment there, so skipping its draw changes no pixel. This is the
    // part DH-0871 lost to an accidental `git checkout` and nothing had called since — the logic
    // was committed (0718f54) with zero call sites, so every pass still drew every group.
    /// Master switch. A pure skip, so ON by default; the lever is for A/B and for a host that
    /// wants the old draw-everything behaviour back.
    public var visibilityCullingEnabled = true
    /// Mesh kinds the CAMERA does not see: the raster passes (the G-buffer and the cascade / spot / point shadow maps) skip their
    /// groups, but the ray-tracing structure is built from the same groups and keeps them — so an instance of a hidden kind
    /// still lights the scene (path-traced GI, emission) and still appears in RAY-TRACED reflections and refractions.
    /// This is how a studio's light cards work (`IlluminatoramaStudio`): a polished piece mirrors a softbox that is nowhere in
    /// frame. A kind needs a mesh of its own (`setMesh`), since a group is the unit that is skipped.
    public var cameraHiddenKinds: Set<MeshKind> = []
    /// Draw calls each raster pass issued and skipped, cumulative since init. Take deltas.
    public internal(set) var visibilityCullStats = IlluminatoramaVisibilityCullStats()

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
    // ── Incremental invalidation (default-on, per-triangle TLAS path) ──
    // When `surfaceCacheIncremental` is on and the topology is stable, each frame
    // the moved instances' cards are re-framed in place + flagged dirty so the
    // update kernel re-lights only them (α=1) while stationary cards keep their
    // accumulated multi-bounce. See docs/illuminatorama/surface-cache-incremental-
    // invalidation.md. DEFAULT ON: it self-gates to the per-triangle TLAS path
    // (`surfIncrementalReady`) and is a no-op on chart/soup/static scenes (a fully
    // static frame diffs to zero moved instances → zero re-framing), so enabling it
    // by default only helps animated TLAS scenes. `VIZ_SURFCACHE_INCREMENTAL=0` is
    // the kill-switch. The default lane re-frames on the GPU, on this frame's command
    // buffer (hazard-tracked); the CPU fallback rewrites `surfCardBuffer` in place and
    // first waits for the in-flight frame (`awaitInFlightFrameBeforeCPUCacheWrite`,
    // DH-0537 #1) — the pool is 2, so an unguarded in-place write would race.
    public var surfaceCacheIncremental: Bool = true
    // Phase 5 / B1 — cache-domain à-trous denoiser. Spatially filters the radiance
    // atlas (same-card-guarded, variance-guided) before the GI/reflection consumers
    // read it, so a freshly-reset card resolves its Monte-Carlo noise in ~1 frame
    // instead of ~20. Display-only (the EMA feedback reads the raw atlas), so it
    // accelerates convergence without changing the fixed point. DEFAULT OFF while
    // it's staged in; `VIZ_ILLUMI_SURFCACHE_DENOISE=1` turns it on.
    // See docs/illuminatorama/phase5-radiance-cache-streaming.md.
    public var surfaceCacheDenoise: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_DENOISE"] == "1"
    // Phase 5 / A (streaming) — residency feedback. When on, the GI/reflection
    // cache hits mark `cardRequested[card]=1`; the host drains the buffer the NEXT
    // frame (after waiting for the in-flight one) to log the per-frame WORKING SET size — the data
    // that sizes the atlas budget and that A1's residency pass will key off. Plain
    // instrumentation for now (no behavior change). `VIZ_ILLUMI_SURFCACHE_FEEDBACK=1`.
    public var surfaceCacheFeedback: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_FEEDBACK"] == "1"
    // Phase 5 / A0 (streaming) — card budget. >0 caps the number of RESIDENT cards
    // (atlas sized to the budget, not the scene); cards beyond it get no atlas slot,
    // and a GI/reflection hit on a non-resident card falls back to emission + ambient
    // (never black). Static bake-order residency for A0 (feedback-driven priority is
    // A1). 0/unset ⇒ disabled (every card resident — today's behaviour).
    // `VIZ_ILLUMI_SURFCACHE_CARD_BUDGET`.
    public var surfCardBudget: Int =
        Int(ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_CARD_BUDGET"] ?? "") ?? 0
    // Phase 5 / A1 (streaming) — DYNAMIC feedback-driven residency. When on (with a
    // budget), a per-frame host pass reassigns which cards occupy the fixed atlas
    // SLOTS (= A0's bake-time resident-card rects) based on the `cardRequested`
    // working set: a sampled-but-non-resident card is promoted into a slot freed by
    // an unsampled resident card. Mutates `texelCard` + `cardRect` + `dirty` in place
    // (after `awaitInFlightFrameBeforeCPUCacheWrite`, same class as the incremental writes) — NO
    // kernel addressing change (the kernels still read texel→card / card→rect, and
    // A0's zero-rect-⇒-fallback path already handles a non-resident card).
    // `VIZ_ILLUMI_SURFCACHE_STREAM=1` (implies feedback). Default off.
    public var surfaceCacheStreaming: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_STREAM"] == "1"
    // Phase 5 / A4 — importance-weighted residency (the tractable core of per-card
    // LOD). Under eviction pressure (budget < working set), keep the NEAR cards (large
    // screen footprint) over far ones: promote the most-important requested cards
    // first, demote the least-important cold slots first. Importance = camera proximity
    // from the card's bake-time centroid. (Full variable-slot-size LOD — smaller atlas
    // tiles for far cards — is the deferred bin-packing follow-on; this prioritises
    // WHICH cards hold the fixed slots, not slot size.) `VIZ_ILLUMI_SURFCACHE_LOD=1`.
    public var surfaceCacheLOD: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_LOD"] == "1"

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
    /// Times the BLAS cache served an entry built from a DIFFERENT mesh object than the one
    /// asked for — i.e. a recycled address collided with a dead entry. Structurally
    /// impossible while entries own their meshes, and gated at 0 by
    /// `RTMeshCacheIdentityTests`, which is what proved this was happening.
    public internal(set) var rtStaleBLASReuseCount = 0
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
    /// Instances the last card bake left cardless (DH-1314) — read-only diagnostic.
    public internal(set) var surfaceCacheCardlessInstances: Int = 0

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

    /// Surface-cache (P1c) caps for the TLAS path. The cache allocates one
    /// per-triangle micro-card over the INSTANCE-EXPANDED soup (count =
    /// Σ group.count · meshTris), which can dwarf the per-mesh `estTriangles`
    /// used to gate live RT — so it gets its own cap. Past it the TLAS still
    /// runs (RT lighting intact); only the cache read is skipped.
    ///
    /// A host that turns the cache on for a real scene raises it: a furnished house is well past
    /// 60 000 (4000 Sunset ~154 k), where the cache is skipped and the frame is identical to cache
    /// off. Read at each TLAS topology build, so set it before the build that should bake cards.
    /// `VIZ_ILLUMI_SURFCACHE_MAXTRIS` overrides the default — an instrument, like
    /// `VIZ_ILLUMI_RT_GROUPCAP` (DH-0622 measured 0 cached hits on the ~90 k Debug Tester without it).
    public var surfaceCacheMaxTriangles: Int =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_MAXTRIS"].flatMap { Int($0) } ?? 60_000
    /// Atlas texels per side of a per-triangle card tile, read at each card bake. A large triangle's
    /// texels span tens of cm at 6, which is what DH-0849 (light along junctions) is being A/B'd against.
    public var surfaceCachePerTriangleTileSize: Int = 6
    /// DH-1314 — how the cache update treats a bounce ray that lands on a CARDLESS mesh
    /// (`IlluminatoramaMesh.surfaceCacheCards = false`): true (default) re-shades it once — its
    /// own sun plus the gathering texel's indirect, times its albedo; false brings back nothing
    /// (a black occluder — the A/B arm). Inert in a scene with no cardless mesh.
    public var surfaceCacheCardlessReshade: Bool = true

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

    /// When true AND the three handles below are set, the in-view cloud pass
    /// composites crisp camera-ray clouds into the HDR target (issue #61). OFF
    /// by default → every scene + the dome/IBL path is byte-identical.
    public var inViewCloudsEnabled = false
    /// DH-0578 — the ACCUMULATED screen-resolution cumulus pass (`encodeCumulusInViewPass`): a
    /// `.cumulus` deck marched along each pixel's own ray with detail erosion and a fine light
    /// march, one interleave cell per frame while camera and sky are still, swapped in for the
    /// dome's sample once every pixel has one. Needs the same three handles as the legacy pass
    /// (`cloudNoiseTexture` / `cloudSkyUniforms`; `cloudBurstLights` unused) plus `equirectSky`.
    /// OFF by default → byte-identical.
    public var cumulusInViewEnabled = false
    /// Accumulator resolution divisor (1 = canvas resolution, 2 = half — a quarter of the rays,
    /// bilinear-resolved; at the photo lane's 2× supersample, 2 is still OUTPUT resolution).
    public var cumulusInViewScale = 2
    /// n of the n×n interleave (1, 2 or 4): each frame marches 1/n² of the accumulator.
    public var cumulusInViewInterleave = 4
    /// Samples per pixel after which marching stops (the converged sky costs only the resolve).
    public var cumulusInViewMaxSamples = 4
    /// Camera-ray steps through the slab, and sun light-march steps per dense sample.
    public var cumulusInViewSteps = 128
    public var cumulusInViewLightSteps = 7
    /// High-frequency detail erosion (0 = the dome's field, 1 = the ~4 m octave, 2 = + ~1.5 m).
    public var cumulusInViewDetail: Float = 2
    let cumulusInView = CumulusInViewState()
    /// The clock the in-view sky animates by (the lava lamp's wax, the cirrus threads' flow) —
    /// `nil` (the default) = `time`, byte-identical. A host that lets those speeds change live
    /// passes its own INTEGRATED phase (Σ speed·dt) and sets the sky's speed factors to 1: the
    /// shader multiplies clock × speed, so a speed step on the frame clock jumps the pattern by
    /// time·Δspeed — hundreds of seconds of animation after a few minutes.
    public var inViewSkyTime: Float? = nil
    /// Per-FRAME lava-lamp weight (0…1) for the in-view sky, overriding the cloud renderer's
    /// `lavaFade` (which rides the host-throttled sky uniforms — a fade driven through those
    /// would step at the sky's update rate). `nil` (the default) = the uniform's value.
    public var inViewLavaFade: Float? = nil
    /// The cirrus threads' own clock (≥ 0), for a host whose lava lamp and cirrus are on screen
    /// at once (mid-fade) and animate at independent speeds. `nil` = `inViewSkyTime ?? time`.
    public var inViewCirrusTime: Float? = nil
    /// Per-FRAME cirrus threads weight (0 veil … 1 threads) for the in-view sky, overriding the
    /// cloud renderer's `cirrusThreadsWeight` (throttled sky uniforms). `nil` = the uniform.
    public var inViewCirrusThreads: Float? = nil
    /// Frame-level gain on every instance's `tagGlow.x` weight (petal glow). 1 by default, so a
    /// host that only sets per-instance weights still glows; instances with weight 0 — every
    /// host that never opts in — are untouched by any value.
    public var tagGlowGain: Float = 1
    /// Field sink, 0…1: instances with a sink depth (`tagGlow.y`) slide that far below ground at 1
    /// — a host fades a whole field of objects into / out of the ground with one float per frame
    /// (see `fieldSinkOffset` in IlluminatoramaGBuffer.metal). 0 (the default) is an exact no-op.
    /// Applied in the G-buffer and shadow vertex stages only: RT scene data, glass, selection and
    /// impostor paths still see the UN-sunk pose (as with wind sway) — an RT host would reflect
    /// a sunk field. Give a sunk object's parts one `tagGlow.z` key built from WORLD coordinates:
    /// the fallback hash reads the (render-space) origin, which a floating origin moves.
    public var fieldSinkProgress: Float = 0
    /// Resolution the in-view cloud + atmosphere march runs at, as a fraction of the canvas
    /// (1 = full res, the default — byte-identical). 0.5 marches a quarter of the rays into an
    /// off-screen target and depth-aware-upsamples it into the sky pixels: clouds and the
    /// nishita sky are smooth at that scale, and the march is the most expensive pass in an
    /// outdoor scene (measured 30 ms of a 2880×1620 frame in Superbloom Hills).
    public var inViewCloudsResolutionScale: Float = 1
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
    /// Baked noise volume + packed SkyUniforms + burst-light buffer, supplied by
    /// the scene's `VolumetricCloudRenderer` (reused, not re-packed).
    public var cloudNoiseTexture: MTLTexture?
    public var cloudSkyUniforms: MTLBuffer?
    public var cloudBurstLights: MTLBuffer?

    var pointLightBuffer: MTLBuffer { shd.pointLightRing[lts.frameRingIndex] }
    var spotLightBuffer: MTLBuffer { shd.spotLightRing[lts.frameRingIndex] }
    public internal(set) var spotLightProfileRowCount: Int = 0
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

    /// Phase-by-phase breakdown of how long `init` took (see
    /// `IlluminatoramaInitTiming`). Recorded unconditionally so a host can fold it
    /// into its own launch timeline without a special build; `VIZ_ILLUMI_INIT_TIMING`
    /// additionally emits signposts / a sidecar file.
    public internal(set) var initTiming = IlluminatoramaInitTiming()

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
                                       ibl: iblEnabled, shadow: shadowsEnabled, dfg: dfgLUTEnabled)
        try Self.buildAOPipelines(into: aos, cache: cache, device: device)
        try Self.buildSSRSSSPipelines(into: ssrs, cache: cache, device: device)
        try Self.buildGIDenoisePipelines(into: rts, cache: cache, device: device)
        try Self.buildBakePipelines(into: ibls, cache: cache, device: device)
        try Self.buildPostPipelines(into: pfx, cache: cache, device: device, library: library)
        timing.mark("compute pipeline fetch (lighting/ssao/ssr/svgf/bloom)")

        self.rtSupported = try Self.buildRTPipelines(into: rts, cache: cache, device: device, library: library)
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

    // ── Render ────────────────────────────────────────────────────────────────

    /// - Parameter blocking: the CAPTURE contract — `true` is the headless snapshot /
    ///   export path, `false` the interactive tick.
    ///
    ///   `false` pipelines. A back-pressured GPU DROPS the tick rather than stalling the
    ///   main thread, and the frame this call commits is not the one `outputTexture`
    ///   holds on return — the pool promotes it on the NEXT tick. That is correct for a
    ///   display loop, which only ever wants the freshest fully-rendered buffer.
    ///
    ///   `true` (a) waits for an in-flight slot instead of dropping, because a dropped
    ///   export render leaves `outputTexture` on a stale / never-rendered pool buffer →
    ///   captured as magenta (no `cb.error`); and (b) waits for THIS frame on the GPU and
    ///   promotes it before returning, so `outputTexture` IS the frame you just asked for.
    ///   Without (b) a readback taken straight after a scene mutation samples the buffer
    ///   promoted at the TOP of this call — a frame rendered before the mutation — which
    ///   reads exactly like "the instance upload landed a frame late". It doesn't:
    ///   `uploadInstances` runs the same frame the change is seen. The latency was always
    ///   the presented-buffer pool, never the upload.
    ///
    /// Returns `true` if this call actually produced a frame, `false` if it
    /// dropped (GPU still draining the previous frames — the non-blocking
    /// `inFlightSemaphore` path) or bailed on a transient failure. Callers
    /// use the result to count FPS against *rendered* frames rather than the
    /// fixed Timer cadence, and to skip re-presenting a stale `outputTexture`.
    @discardableResult
    public func render(blocking: Bool = false) -> Bool {
        // Read-and-CLEAR the awaited-frame arming here, before any early return, so no exit
        // path can leave it set for the next caller (see `renderAwaited()`). A nil
        // `pendingFrameCompletion` is what an early return leaves behind, and
        // `awaitCommittedFrame()` reads that as "nothing to wait for".
        let awaited = gbs.armAwaitedFrame
        gbs.armAwaitedFrame = false
        if awaited { gbs.pendingFrameCompletion = nil }
        // Fail LOUD on a boot-relative animation clock. `time` is a Float used as a phase by
        // every GPU oscillator, so a large absolute value quantises all of them (see `time`'s
        // doc and `RenderClock`). This is the check that would have caught the defect on the
        // first frame instead of after it silently killed the film grain; it costs one compare
        // per frame and never fires for a host that sends a session-relative clock.
        if !pfx.warnedCoarseAnimationClock && RenderClock.isTooCoarse(CFTimeInterval(time)) {
            pfx.warnedCoarseAnimationClock = true
            let m = RenderClock.coarseClockDiagnosis(CFTimeInterval(time)) + "\n"
            FileHandle.standardError.write(Data(m.utf8))
        }
        // Promote any pool buffer whose render completed since last tick to
        // the presented `outputTexture`. Done first so consumers binding
        // `outputTexture` this tick see the freshest fully-rendered frame.
        promoteCompletedBuffer()
        // A scene with NO opaque instances is still renderable when it has glass
        // panes (Floating Lenses): the deferred pipeline runs with an empty
        // G-buffer (every pixel resolves to sky in `illumi_lighting`), then the
        // forward glass pass draws the lenses over that sky. Only a truly empty
        // frame (no opaque AND no glass) falls back to the flat sky clear.
        guard !instances.isEmpty || !glassInstances.isEmpty || !glassMeshGroups.isEmpty else {
            // Nothing to draw — still clear the output to the sky color so the
            // host's display quad isn't showing last frame's garbage. This IS a
            // produced frame (the sky), so it counts toward FPS.
            blankSky()
            return true
        }
        // Throttle: drop this tick if the GPU hasn't drained the previous
        // frame yet (non-blocking wait). Prevents queue saturation from
        // freezing the main thread / stalling presentation on heavy scenes.
        // Balanced by `inFlightSemaphore.signal()` in the completion handler.
        // The snapshot export path waits instead (blocking) so it never drops.
        if blocking {
            gbs.inFlightSemaphore.wait()
        } else {
            // Dropped frame — GPU hasn't drained the previous frames. Report
            // `false` so the caller doesn't count it as a rendered frame.
            guard gbs.inFlightSemaphore.wait(timeout: .now()) == .success else { return false }
        }
        // ── Panel is the single source of truth for TAA on/off ──────────────
        // Authority lives HERE, once, instead of in every scene controller. The
        // panel toggle (IlluminatoramaSharedSettings.shared.taaEnabled) wins for
        // every Illuminatorama scene unless that scene explicitly opts out via
        // `sharedTAAOverride`. This is why a scene's own `renderer.taaEnabled = …`
        // write no longer leaks the toggle out of sync — it's overwritten here
        // each frame before any TAA gating reads it.
        taaEnabled = Self.taaEnvOverride ?? sharedTAAOverride ?? IlluminatoramaSharedSettings.shared.taaEnabled
        // Same authority model for the SVGF GI denoiser: the panel toggle drives it
        // for every Illuminatorama scene. (`effectiveSVGFEnabled` still layers the
        // VIZ_ILLUMI_SVGF env A/B override on top.) Safe to apply universally — the
        // cascade self-gates on `rtGITemporalEnabled`, so scenes that force temporal
        // off (CoinPusher, HotdogPress) stay on the bilateral path regardless.
        svgfEnabled = IlluminatoramaSharedSettings.shared.svgfEnabled
        primeTemporalHistoriesOnTransitions()
        // ── Panel is the single source of truth for lens aberration too ─────
        // CA / spherical aberration / fringe (+ post-FX easing) have NO per-scene
        // overrides anywhere, so apply them HERE once for every Illuminatorama
        // scene — including the ones that never wired post-FX and every future
        // scene, which inherit them with zero per-scene code. (Bloom is NOT here:
        // scenes art-direct it and `render()` runs after their tick, so it opts in
        // via `applySharedPostFX`.) Targets are set before `uploadFrameUniforms`
        // below, which eases toward them.
        if appliesSharedLensFX { applySharedLensFX(); applySharedColorGrade() }
        if appliesSharedDenoiser { applySharedDenoiser() }
        // Ping-pong the instance buffer FIRST: after this swap,
        // currentInstanceBuffer is what was previousInstanceBuffer last frame
        // (so uploadInstances overwrites it with this frame's data), and
        // previousInstanceBuffer now points to last frame's data.
        gbs.useBufferA.toggle()
        // Advance the per-frame input-buffer ring in lockstep. From here on,
        // `frameUniformBuffer` / the light buffers / `superquadricParamBuffer`
        // resolve to THIS frame's slot — distinct from the slot the previously
        // pipelined frame (still on the GPU) is reading. Done before the uploads
        // below write into the new slot, and before any encode binds it.
        lts.frameRingIndex = (lts.frameRingIndex + 1) % lts.frameUniformRing.count
        if Self.raceProbeEnabled {
            if Self.raceProbePin0 { lts.frameRingIndex = 0 }   // negative control: pre-fix single-buffer
            pfx.presentSync.raceProbeBeginFrame(slot: lts.frameRingIndex)
        }
        ensureBufferCapacity()  // may set taaNeedsFirstFrame on growth
        updateSceneStability()
        updateCascades()
        // Phase 4.10 — compute per-spot shadow matrices + slice indices
        // BEFORE `uploadSpotLights` so the GPU buffer sees the current
        // frame's values. Done unconditionally even when shadows are
        // disabled (cheap CPU math); the shadow PASS gates on
        // `spotShadowsEnabled` to skip the expensive depth render.
        updateSpotShadows()
        // Point-light cube shadows: assign cube pages + compute the 6 face VPs.
        // No-op (clears indices to -1) when the feature is off, so the uploaded
        // buffer never carries a stale page. Cheap CPU math; the depth PASS gates.
        updatePointShadows()
        uploadFrameUniforms()
        uploadInstances()
        computeCullBounds()   // DH-0534 — reads the CPU-written instance copy, before any GPU hook
        uploadPointLights()
        uploadSpotLights()
        uploadAreaLights()
        uploadExtraDirectionals()
        uploadInteriorApertures()

        // Pick a free pool buffer for this frame's tonemap output (never the
        // presented one SceneKit may sample, nor the one still in flight).
        let writeIdx = acquireWriteTarget()
        pfx.tonemapWriteTarget = pfx.ldrPool[writeIdx]

        guard let cb = commandQueue.makeCommandBuffer() else {
            Self.log.error("makeCommandBuffer() returned nil")
            gbs.inFlightSemaphore.signal()  // release the slot acquired above
            return false
        }
        // Reserve this pool slot until the GPU finishes writing it, so a second
        // pipelined frame (maxFramesInFlight = 2) picks a different buffer. Cleared
        // by `markCompleted` in the completion handler below. Marked after the
        // makeCommandBuffer guard so a failed allocation doesn't leak the slot.
        pfx.presentSync.markInFlight(writeIdx)
        cb.label = "Illuminatorama.frame"
        gbs.passTimer.beginFrame()   // reset per-pass GPU-timer sample assignment (env-gated)
        // Opt-in host GPU work for THIS frame (see `onEncodeFrameCompute`); nil → nothing.
        onEncodeFrameCompute?(cb)

        // Opt-in GPU-resident instance write (additive — see `onEncodeGPUInstances`).
        // `uploadInstances()` above already reserved the slots + built `meshGroups`;
        // here, before any pass reads the instance buffer, a scene kernel can
        // overwrite its group's slots with GPU-computed data. No-op (closure nil)
        // for every scene that doesn't set the hook.
        if let hook = onEncodeGPUInstances {
            let ranges = gbs.meshGroupRange
            hook(cb, currentInstanceBuffer) { kind in ranges[kind] }
        }
        // GPU-resident point lights (additive — see `onEncodeGPUPointLights`).
        // `uploadPointLights()` above packed the CPU placeholders into the buffer;
        // the scene kernel overwrites the reserved slots before the lighting pass.
        if let plHook = onEncodeGPUPointLights, !pointLights.isEmpty {
            plHook(cb, pointLightBuffer, pointLights.count)
        }

        // Phase 4.11 — tick particle simulations BEFORE rendering them.
        // The integration is per-frame; the additive draw lands later in
        // the frame graph (after SSR composite but before TAA history /
        // bloom / tonemap so the additive accumulation participates in
        // those passes naturally).
        encodeParticleStep(cb)

        // Phase 4.13a — repack any compute-fed `DynamicMesh` geometry
        // from separate pos/norm into Illuminatorama's interleaved
        // vertex layout. Must run before shadow / G-buffer reads.
        encodeGPURepacks(cb)

        // Issue #65 — on a GPU mesh's FIRST frame, seed prev = current so its
        // motion vector starts at zero (no first-frame streak). Subsequent frames
        // are seeded by the post-G-buffer capture below.
        encodeGPUMeshPrevCapture(cb, prime: true)

        encodeShadowPasses(cb)
        encodeSpotShadowPasses(cb)
        encodePointShadowPasses(cb)
        encodeGBufferPass(cb)
        // Issue #65 — now that the G-buffer draw has read THIS frame's prev
        // positions, copy the current positions forward to be next frame's prev.
        encodeGPUMeshPrevCapture(cb, prime: false)
        encodeIBLBakeIfNeeded(cb)
        // Phase 4.39: SSAO → bilateral spatial → temporal → lighting (reads denoised AO).
        encodeSSAOPass(cb)
        encodeSSAOSpatialFilter(cb)
        encodeSSAOTemporalPass(cb)
        prepareAOVFrame()   // DH-1304 — this frame's capture (no-op unless `aovCaptureEnabled`)
        encodeLightingPass(cb)
        // Issue #65 — screen-space SSS: diffuse the lit irradiance of SSS-flagged
        // pixels and blend it back into hdrTexture BEFORE SSR/RT/TAA/bloom read it,
        // so the soft light-bleed propagates through the rest of the pipeline. No-op
        // (and not encoded) unless a scene set sssStrength > 0.
        encodeSSSPasses(cb)
        // Phase 4.43 — additive particle / external-point draws (emitter
        // sparks + dust motes) MOVED to AFTER the TAA resolve (see
        // `encodePostResolveFX`). They have no motion vectors of their own,
        // so flowing them through TAA history reprojected each moving mote as
        // static and retained ~90% of the previous frame → visible trails.
        // Trade-off vs the old Phase 4.42 placement: these FX no longer feed
        // the SSR/RT reflection chain (they reflected in nearby metals). For
        // dust/sparks that's an acceptable loss; trail-free wins.
        // Phase 4.39: SSR gather → temporal → composite into hdrComposite.
        // PERF: with SSR off (`ssrIntensity <= 0` — Daydream's default config, and
        // the documented setting for RT-reflection hosts) the ray-march gather, the
        // temporal denoise, and the full-res composite are all pure waste — the
        // shader would early-out per pixel but the dispatches, depth reads, and the
        // hdr→hdrComposite copy still ran. Skip all three and seed `hdrComposite`
        // with one blit instead (downstream additive passes — RT sun/GI, glass
        // backdrop, bloom/tonemap, TAA input — all read `hdrCompositeTexture`).
        if ssrIntensity > 0 {
            encodeSSRGather(cb)
            encodeSSRTemporalPass(cb)
            encodeSSRComposite(cb)
        } else {
            encodeSSRBypassCopy(cb)
        }
        // Incremental invalidation (default-on): re-frame moved instances' cards
        // in place + flag them dirty so the update below re-lights only them,
        // preserving stationary cards' accumulation. Rigid movers AND deforming
        // meshes run as GPU kernels on this command buffer (#60 item 1 + Phase D);
        // the VIZ_SURFCACHE_GPU_DIFF=0 kill-switch takes the CPU loop. No-op
        // unless on.
        applyIncrementalSurfaceCacheUpdate(cb)
        // Curve primitives on the SOUP path (#60 item 7 incr. 2): adopt registry
        // changes (rebuild the soup AS with the curve geometry — topology), then
        // re-displace the control points for wind sway + refit the AS on this cb,
        // BEFORE the cache update + RT pass trace it. No-op for TLAS-path renderers
        // (guarded inside) and curve-free soup scenes.
        syncSoupCurveRegistry()
        encodeCurveWindDisplaceAndRefit(cb)
        // Phase 5 / A1 — dynamic residency: reassign atlas slots to this frame's
        // working set (from the previous frame's feedback) BEFORE the update relights
        // them. After the re-frame above so a promoted card's dirty flag survives.
        // No-op unless streaming is on with a budget.
        runStreamingResidency()
        // Re-light the on-surface radiance atlas (ping-pong) so the RT pass can
        // read cached multi-bounce radiance at its hits. No-op unless enabled.
        encodeSurfaceCacheUpdate(cb)
        // Phase 5 / B1 — spatially denoise the freshly-updated atlas into the
        // consumer atlas BEFORE the RT lighting pass reads it. No-op unless
        // `surfaceCacheDenoise` is on.
        encodeSurfaceCacheDenoise(cb)
        // Instanced-RT for extracted scenes: build/refit the TLAS this frame.
        // Sets `rtTLASActive` → trace the instance AS; otherwise fall back to
        // the primitive-AS soup / room path. No-op unless extraction RT is on.
        updateRTAccel(cb)
        // Hardware RT sun lighting (soft shadows + 1-bounce GI), added into the
        // HDR composite before particles / TAA / bloom. No-op unless enabled.
        //
        // #60 perf — gate the TLAS lighting pass on `rtEnabled`, not just on a live
        // TLAS. AAA glass builds the TLAS for refraction even when a scene only
        // wants glass (not RT GI/shadows on its opaque geometry); without this gate
        // such a scene paid the full RT-opaque-lighting pass (~25–50 ms at the lab's
        // ray counts) for nothing. The opaque floor then renders with the cheaper
        // deferred lighting; glass still refracts/reflects it via the TLAS. The soup
        // `else` branch already self-gates on `rtEnabled`.
        //
        // `rtOpaqueLightingEnabled` (default true) is a SECOND gate so a scene that
        // needs the surface cache / caustics (which require `rtEnabled`) can still
        // skip THIS opaque pass without losing the cache. With it false here, the
        // opaque geometry falls back to deferred lighting (exactly as in the
        // `rtEnabled == false` case), the surface-cache update + caustic splat still
        // run, and the glass still refracts via the live TLAS. See the flag's doc.
        //
        // C2 — the denoise chain is now shared. It used to live INSIDE the soup
        // `else` branch, so the TLAS path composited 4 GI rays raw; in an app that
        // also runs with TAA off, nothing cleaned them at all ("dark noisy
        // blotches"). Both lighting kernels now write the same `rtDiffuseTexture`,
        // so the same chain reads it whichever one ran.
        if rts.rtTLASActive && rtEnabled && rtOpaqueLightingEnabled {
            encodeRTLightingTLASPass(cb)
            encodeRTDiffuseDenoiseChain(cb)
        } else if !rts.rtTLASActive {
            encodeRTLightingPass(cb)
            encodeRTDiffuseDenoiseChain(cb)
        }
        // Volumetric god-ray shaft in the air. No-op unless enabled.
        encodeVolumetricPass(cb)
        encodeVolumetricRTPass(cb)
        encodeSpotBeamPass(cb)
        // In-view perspective cloud composite (issue #61). No-op unless a scene
        // opted in. Lands before TAA (so clouds anti-alias) and before the
        // additive firework particles in encodePostResolveFX (so clouds sit
        // behind them).
        encodeCloudPass(cb)
        // Glass caustics (#60): photon-trace sun→glass→receiver, splat into the
        // surface-cache cards, add per primary pixel. After RT lighting (HDR lit)
        // and before the glass draw. No-op unless caustics + cache + glass are on.
        encodeCausticPass(cb)
        // Forward reflective glass pane, blended over the lit composite (before
        // TAA so its reflections bloom/tonemap). No-op unless a glass pane is set.
        encodeGlassPass(cb)
        encodeTAAResolve(cb)
        // Stash this frame's depth as next frame's "previous" for the resolve's
        // disocclusion test. The resolve above already consumed last frame's
        // copy. The in-place copy is a GPU blit on this command buffer, and buffers on
        // one queue execute in order, so it lands before the next frame's G-buffer pass
        // overwrites `depthTexture` whatever the in-flight pool depth. Skipped when TAA is off (nothing reads it); the
        // enable-transition's taaNeedsFirstFrame covers the first re-enabled frame.
        if taaEnabled, let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: gbs.depthTexture, to: gbs.previousDepthTexture)
            blit.endEncoding()
        }
        // Phase 4.43 — additive overlay FX (dust + emitter sparks) composited
        // AFTER the resolve, into a throwaway display buffer, so the temporal
        // history never sees them and they can't trail. No-op when there are
        // no emitters/fields.
        encodePostResolveFX(cb)
        // Phase 4.21 — auto-exposure estimate. Runs AFTER TAA resolve so
        // we sample the same texture the tonemap will read, and BEFORE
        // bloom + tonemap so the smoothed exposure is fresh by the time
        // the tonemap binds the exposure buffer.
        encodeExposureEstimate(cb)
        pfx.localToneMapOutput = nil
        // Depth of field on the resolved HDR before bloom, so out-of-focus
        // bright dust motes bloom into soft bokeh. No-op unless enabled.
        encodeDOFPass(cb)
        // Feathered halo glow around selected objects. After DOF (outline stays
        // sharp even on out-of-focus geometry) and before bloom (halo blooms soft).
        encodeSelectionOutlinePass(cb)
        // Local adaptation (opt-in) of the finished optical image; bloom / halation /
        // tonemap read its output through `bloomTonemapSource`.
        encodeLocalToneMap(cb)
        encodeBloomPasses(cb)
        // Film halation reads the same HDR source bloom does (post-TAA, post-DOF), so
        // a blown out-of-focus highlight halates as softly as it blooms. No-op — not
        // even encoded — unless halationIntensity > 0.
        encodeHalationPasses(cb)
        encodeTonemapPass(cb)

        // Signal — on GPU completion (background thread) — that this pool
        // buffer is fully rendered and safe to present. `promoteCompletedBuffer`
        // (next tick, main actor) reads this and swaps `outputTexture`. Capturing
        // a plain `Int` keeps the handler `Sendable`-clean.
        commitFrame(cb, writeIdx: writeIdx, awaited: awaited)

        // CAPTURE CONTRACT (see `blocking` on this method). A snapshot caller must be
        // able to render and then immediately read `outputTexture` and get THIS frame.
        // The pool never promotes the in-flight buffer — `promoteCompletedBuffer()` at
        // the top of the NEXT tick does — so without this wait every capture reads a
        // frame at least one behind, and any test that mutates the scene and renders
        // once is asserting against the pre-mutation image. `waitUntilCompleted()`
        // returns only after the completion handler above has run, so `markCompleted`
        // has already published `writeIdx` and the promote below always finds it.
        // Live rendering (`blocking == false`) is untouched and stays pipelined.
        // `awaited` keeps the never-drop half of the contract (the semaphore wait above) and
        // moves ONLY the CPU stall: `awaitCommittedFrame()` performs the same wait-then-promote
        // from a suspension instead of from `__psynch_cvwait` on the caller's thread.
        if blocking && !awaited {
            cb.waitUntilCompleted()
            promoteCompletedBuffer()
        }

        advanceTemporalHistories()
        return true
    }

    /// **A `blocking` frame whose CPU wait is a suspension instead of a stall.** Commit the
    /// frame under the full capture contract — never dropped, promoted before the caller
    /// reads `outputTexture` — but return as soon as it is *committed*, leaving the
    /// wait-and-promote to `awaitCommittedFrame()`.
    ///
    /// WHY. `render(blocking: true)` ends in `cb.waitUntilCompleted()`, which parks the
    /// CALLING thread in `__psynch_cvwait` until the GPU is done. For a headless harness that
    /// is exactly right — there is nothing else for that thread to do. For an interactive
    /// host settling a still on the main thread it is not: a 4-second ray-traced frame is
    /// 4 seconds in which the run loop does not iterate, so the window does not repaint, the
    /// cursor does not track and a progress indicator does not move. Rendering one frame per
    /// `await` does not help by itself — the thread is still inside the wait for all but the
    /// microseconds between frames. The wait itself has to move.
    ///
    /// Pair it strictly:
    /// ```swift
    /// _ = renderer.renderAwaited()
    /// await renderer.awaitCommittedFrame()   // outputTexture is now THIS frame
    /// ```
    /// Between the two calls the frame is in flight and `outputTexture` still holds the
    /// PREVIOUS one, so a reader that skips the await gets a frame behind — the same trap
    /// `blocking: false` has always had, and the reason the blocking path exists.
    ///
    /// Not a replacement for `render(blocking:)`: a synchronous caller (every `swift test`
    /// capture, every GPU gate) has no suspension point to spend and should keep using it.
    @discardableResult
    public func renderAwaited() -> Bool {
        gbs.armAwaitedFrame = true
        return render(blocking: true)
    }

    /// Wait for the frame `renderAwaited()` committed, then promote it — the second half of
    /// that call's capture contract, performed from a suspension so the caller's thread stays
    /// in its run loop while the GPU works.
    ///
    /// Safe to call when nothing is pending (no armed render, or one that returned early on an
    /// empty scene): it promotes and returns, which is precisely what the blocking path leaves
    /// behind in those cases. Awaiting the same frame twice is likewise a no-op rather than a
    /// hang — the signal is consumed on the first await.
    public func awaitCommittedFrame() async {
        if let signal = gbs.pendingFrameCompletion {
            gbs.pendingFrameCompletion = nil
            await signal.wait()
        }
        promoteCompletedBuffer()
    }

    func blankSky() {
        guard let cb = commandQueue.makeCommandBuffer() else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = outputTexture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.85, green: 0.86, blue: 0.90, alpha: 1)
        if let enc = timedRenderEncoder(cb, pass, "blankSky") {
            enc.label = "Illuminatorama.blankSky"
            enc.endEncoding()
        }
        cb.commit()
    }

    lazy var localToneMapPass = IlluminatoramaLocalToneMapPass(engine: engine)
    lazy var hdrDump = IlluminatoramaHDRDump()

    /// TEST-OBSERVABLE: the GPU time of the last `measureBloomImpulseResponse` command buffer
    /// (upload, the whole chain, readback blit) — the A/B a gate times the far-field tail with.
    public internal(set) var lastBloomImpulseGPUMilliseconds: Double = 0

    var currentHistoryTexture: MTLTexture { pfx.historyToggle ? pfx.historyB : pfx.historyA }
    var previousHistoryTexture: MTLTexture { pfx.historyToggle ? pfx.historyA : pfx.historyB }

}

/// Thread-safe hand-off of "which LDR pool buffer finished rendering" from a
/// background `MTLCommandBuffer` completion handler to the `@MainActor` render
/// loop. Command buffers on one queue complete in commit order, so the last
/// `markCompleted` before a `takeCompleted` is the freshest finished frame.
/// Env-gated GPU-time meter. The frame-delivery metric in PerformanceMeasurement
/// is SceneKit's present cadence (vsync-capped, decoupled from this renderer's
/// GPU tick), so it can't see the per-config cost of RT GI rays / surface cache.
/// This reads `buf.gpuEndTime - buf.gpuStartTime` on each completion (background
/// thread) and rewrites the rolling average to `VIZ_ILLUMI_GPUMS_PATH`. Inert
/// when the env var is unset.
final class IlluminatoramaGPUMeter: @unchecked Sendable {
    let lock = NSLock()
    let path: String?
    var sumMs: Double = 0
    var count: Int = 0
    // Per-frame samples for the percentile distribution. On a shared / contended
    // machine the MEAN is dominated by occasional contention spikes (WindowServer,
    // screen-share, a sibling render) and reads ~the same at every SSAA scale, so
    // it can't show whether a render-cost change actually moved the STEADY frame.
    // The p50 rejects that tail and isolates the clean steady-state GPU frame —
    // the number that decides whether vsync can hold 60 (is it < 16.6 ms?). p95/p99
    // describe the drop-frame tail. Ring-bounded so a long run can't grow unbounded.
    var samples: [Double] = []
    static let maxSamples = 8192
    // Last-computed percentiles, refreshed every 30 frames and ALWAYS emitted, so
    // the sidecar file holds the full distribution no matter which frame the
    // headless snapshot reads it on (the recompute is throttled, the write is not).
    var p50: Double = 0, p95: Double = 0, p99: Double = 0
    var minMs: Double = 0, maxMs: Double = 0
    init() { path = ProcessInfo.processInfo.environment["VIZ_ILLUMI_GPUMS_PATH"] }
    func record(_ buf: MTLCommandBuffer) {
        guard let path else { return }
        let ms = (buf.gpuEndTime - buf.gpuStartTime) * 1000.0
        guard ms > 0 else { return }
        lock.lock()
        sumMs += ms; count += 1
        let avg = sumMs / Double(count); let n = count
        if samples.count >= Self.maxSamples { samples.removeFirst(Self.maxSamples / 4) }
        samples.append(ms)
        // Recompute percentiles every 30 frames (sort is O(n log n); throttle so
        // the diagnostic doesn't dominate the very thing it measures).
        if n % 30 == 0 || p50 == 0 {
            let s = samples.sorted()
            func pct(_ q: Double) -> Double { s[min(s.count - 1, max(0, Int(Double(s.count) * q)))] }
            p50 = pct(0.50); p95 = pct(0.95); p99 = pct(0.99)
            minMs = s.first ?? 0; maxMs = s.last ?? 0
        }
        let line = String(format: "gpuMsAvg=%.3f frames=%d p50=%.3f p95=%.3f p99=%.3f min=%.3f max=%.3f\n",
                          avg, n, p50, p95, p99, minMs, maxMs)
        lock.unlock()
        if let d = line.data(using: .utf8) { try? d.write(to: URL(fileURLWithPath: path)) }
    }
}

/// A one-shot "the GPU finished this frame" latch that an `async` caller can await.
///
/// Exists because the two ends of the handshake live on different threads and neither can be
/// moved: `MTLCommandBuffer.addCompletedHandler` fires on a Metal-owned background thread, and
/// the awaiter is main-actor code inside `awaitCommittedFrame()`. A `DispatchSemaphore` would
/// bridge them only by blocking a cooperative-pool thread, which is the stall this whole path
/// exists to remove.
///
/// Latched rather than gated: `signal()` may land BEFORE anyone awaits (a fast frame finishing
/// while the caller is still unwinding the commit), so `wait()` on an already-signalled latch
/// must return immediately instead of parking forever. Both calls are idempotent — a second
/// `signal()` is dropped and the continuation is resumed exactly once.
final class FrameCompletionSignal: @unchecked Sendable {
    let lock = NSLock()
    var completed = false
    var continuation: CheckedContinuation<Void, Never>?

    /// Called from the GPU completion handler. Resumes the awaiter OUTSIDE the lock —
    /// a continuation resume can run arbitrary code, and doing that under the lock would
    /// re-enter it if that code touched this latch again.
    func signal() {
        lock.lock()
        if completed { lock.unlock(); return }
        completed = true
        let waiter = continuation
        continuation = nil
        lock.unlock()
        waiter?.resume()
    }

    func wait() async {
        await withCheckedContinuation { (k: CheckedContinuation<Void, Never>) in
            lock.lock()
            if completed {
                lock.unlock()
                k.resume()          // already finished — never park
            } else {
                continuation = k
                lock.unlock()
            }
        }
    }
}

final class IlluminatoramaPresentSync: @unchecked Sendable {
    let lock = NSLock()
    var completedIdx: Int = -1
    /// Pool slots whose render is queued/running on the GPU but not yet complete.
    /// Inserted on the main thread (after commit) and removed on the GPU
    /// completion thread, so `acquireWriteTarget` never picks a buffer a second
    /// pipelined frame is still writing (`maxFramesInFlight = 2`).
    var inFlight: Set<Int> = []

    // ── Adaptive in-flight depth (latency guard) ───────────────────────────────
    // Pipelining a 2nd frame (`maxFramesInFlight = 2`) recovers the sub-tick GPU
    // idle on light/medium scenes (big fps win). But when a frame is much heavier
    // than the 60 Hz tick (e.g. RT glass + caustics at ~200–600 ms), the 2nd frame
    // adds ~one whole frame of INPUT LATENCY with no throughput gain, so the scene
    // feels sluggish to drag. So we shrink the effective depth back to 1 when the
    // recent GPU frame time is heavy — done WITHOUT touching the render() gate, by
    // simply HOLDING a semaphore permit in the completion handler instead of
    // signalling it (and signalling it back when frames get light again).
    /// EMA of recent GPU frame time (ms). Drives the depth decision.
    var gpuMsEMA: Double = 0
    /// Permits currently held back (0 ⇒ depth 2, 1 ⇒ depth 1). Never exceeds
    /// `maxFramesInFlight - 1`, so at least one permit always circulates (no deadlock).
    var heldPermits: Int = 0
    /// Set by `retire(semaphore:)` when the owning renderer tears down — no
    /// further permits may be held after that (see the deinit-balance contract).
    var retired = false
    /// Current effective depth target, with hysteresis so it doesn't flap.
    var depth: Int = IlluminatoramaRenderer.maxFramesInFlight
    /// Host opted out of the latency guard (`pipelineLatencyTolerant`): always full depth.
    var latencyTolerant = false
    func setLatencyTolerant(_ on: Bool) { lock.lock(); latencyTolerant = on; lock.unlock() }
    /// Hysteresis band (~2 ticks): pipeline only while the GPU frame is light.
    static let pipelineEnterMs = 30.0   // drop to 1→2 when EMA below this
    static let pipelineExitMs  = 36.0   // rise to 2→1 when EMA above this

    /// The most recent completed frame's GPU time (ms), for hosts' own diagnostics. Written
    /// under `lock` by `frameCompletedAdaptive` (completion-handler thread).
    var lastGpuMs: Double { lock.lock(); defer { lock.unlock() }; return _lastGpuMs }
    var _lastGpuMs: Double = 0

    /// Called from the GPU completion handler in place of `semaphore.signal()`.
    /// Updates the GPU-time EMA, recomputes the target depth, and either releases
    /// this frame's permit normally or HOLDS it to converge the effective in-flight
    /// capacity to that depth. Deadlock-free: `heldPermits` is capped at
    /// `maxFramesInFlight - 1`, so a permit always remains in circulation.
    func frameCompletedAdaptive(gpuMs: Double, semaphore: DispatchSemaphore) {
        lock.lock()
        _lastGpuMs = gpuMs
        if gpuMs > 0 {
            gpuMsEMA = gpuMsEMA == 0 ? gpuMs : gpuMsEMA * 0.9 + gpuMs * 0.1
        }
        if latencyTolerant {
            depth = IlluminatoramaRenderer.maxFramesInFlight
        } else if depth >= IlluminatoramaRenderer.maxFramesInFlight {
            if gpuMsEMA > Self.pipelineExitMs { depth = 1 }
        } else {
            if gpuMsEMA < Self.pipelineEnterMs { depth = IlluminatoramaRenderer.maxFramesInFlight }
        }
        let targetHeld = max(0, IlluminatoramaRenderer.maxFramesInFlight - depth)
        if heldPermits < targetHeld, !retired {
            heldPermits += 1            // hold this frame's permit (shrink capacity)
            lock.unlock()
        } else if heldPermits > targetHeld {
            heldPermits -= 1
            lock.unlock()
            semaphore.signal()          // this frame's permit …
            semaphore.signal()          // … plus one previously-held, to grow capacity
        } else {
            lock.unlock()
            semaphore.signal()          // normal release
        }
    }

    // ── Per-frame buffer-race probe (env-gated; see maxFramesInFlight comment) ──
    // `slotInFlight[s]` = a frame that wrote ring slot `s` is still reading it on
    // the GPU. `raceProbeBeginFrame` is called (main thread) when a new frame
    // claims its slot: if that slot is STILL in flight from an earlier frame, the
    // CPU is about to overwrite a buffer the GPU is reading — the exact race —
    // so it's counted. `raceProbeCompleteFrame` clears the flag on GPU completion.
    // Sized to maxFramesInFlight+1 (the ring is maxFramesInFlight; +1 is slack).
    var slotInFlight = [Bool](repeating: false, count: IlluminatoramaRenderer.maxFramesInFlight + 1)
    var raceEvents = 0
    var probeFrames = 0
    let racePath = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PATH"]

    func raceProbeBeginFrame(slot: Int) {
        lock.lock()
        probeFrames += 1
        if slot >= 0, slot < slotInFlight.count {
            if slotInFlight[slot] { raceEvents += 1 }   // overwriting a slot still read by the GPU
            slotInFlight[slot] = true
        }
        let r = raceEvents, f = probeFrames
        lock.unlock()
        if let p = racePath {
            let line = "raceEvents=\(r) frames=\(f)\n"
            if let d = line.data(using: .utf8) { try? d.write(to: URL(fileURLWithPath: p)) }
        }
    }

    func raceProbeCompleteFrame(slot: Int) {
        lock.lock()
        if slot >= 0, slot < slotInFlight.count { slotInFlight[slot] = false }
        lock.unlock()
    }

    /// The owning renderer is tearing down: release every HELD permit and refuse to
    /// hold any more, so `inFlightSemaphore` can never be deallocated under-valued.
    /// A `DispatchSemaphore` whose value at dealloc is below its creation value hits
    /// the libdispatch trap "Semaphore object deallocated while in use" — which is
    /// exactly what happened when a heavy-frame scene (RT glass) ended with the
    /// adaptive guard holding a permit: the renderer + semaphore died one frame
    /// later and the SIGTRAP detonated in whatever unrelated code ran next (it
    /// presented as heap-corruption-looking crashes in the NEXT test's mesh build).
    /// The lock serializes against a completion handler mid-decision: either it
    /// holds first (drained here) or it sees `retired` and signals normally.
    /// Over-signalling past the creation value is safe — only UNDER-value traps.
    func retire(semaphore: DispatchSemaphore) {
        lock.lock()
        retired = true
        let toRelease = heldPermits
        heldPermits = 0
        lock.unlock()
        for _ in 0..<toRelease { semaphore.signal() }
    }

    /// Record (from the main thread, at commit) that pool buffer `idx` is now in flight.
    func markInFlight(_ idx: Int) {
        lock.lock()
        inFlight.insert(idx)
        lock.unlock()
    }

    /// True if `idx`'s render is still in flight (used by `acquireWriteTarget`).
    func isInFlight(_ idx: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return inFlight.contains(idx)
    }

    /// Bumped by `reset()`. A frame captures it when it claims a pool slot; its completion is
    /// ignored if the pool was rebuilt in between (see `markCompleted`).
    var generation = 0
    var currentGeneration: Int { lock.lock(); defer { lock.unlock() }; return generation }

    /// Record (from the GPU completion thread) that pool buffer `idx` is done.
    ///
    /// **A completion from a pool that no longer exists is dropped.** A resize rebuilds the LDR
    /// pool at the new size and resets this object, but a frame committed just before it is still
    /// on the GPU, writing a slot of the OLD pool. Accepting its index would promote the NEW
    /// pool's texture at that index — one nothing has ever written — and present it. A fresh
    /// private texture reads back as solid 255,0,255 on macOS, which is the fuchsia flash a
    /// window resize showed. It must not clear `inFlight[idx]` either: after the reset that entry
    /// belongs to a new-pool frame that may still be writing it.
    func markCompleted(_ idx: Int, generation frameGeneration: Int) {
        lock.lock()
        if frameGeneration == generation {
            completedIdx = idx
            inFlight.remove(idx)
        }
        lock.unlock()
    }

    /// Return the most recently completed pool index and clear it, or nil if
    /// nothing new finished since the last call.
    func takeCompleted() -> Int? {
        lock.lock()
        defer { lock.unlock() }
        let v = completedIdx
        completedIdx = -1
        return v >= 0 ? v : nil
    }

    func reset() {
        lock.lock()
        generation &+= 1          // in-flight frames of the old pool must not promote (markCompleted)
        completedIdx = -1
        inFlight.removeAll()
        // Reset the adaptive-depth EMA/target; leave `heldPermits` alone so no
        // held permit is lost — the next completion self-heals it (targetHeld 0).
        gpuMsEMA = 0
        depth = IlluminatoramaRenderer.maxFramesInFlight
        lock.unlock()
    }
}

public enum IlluminatoramaError: Error, LocalizedError {
    case libraryMissing
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .libraryMissing:
            return "VisualizerRendering Metal library could not be loaded — was the package compiled with Shaders/?"
        case .pipelineCreationFailed(let name):
            return "Illuminatorama pipeline failed to create: \(name)"
        case .bufferAllocationFailed(let what):
            return "Illuminatorama failed to allocate Metal resource: \(what)"
        }
    }
}
