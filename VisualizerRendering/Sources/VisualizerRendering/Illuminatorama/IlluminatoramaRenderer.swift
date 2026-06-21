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
// Phase 3.1: DDGI probe grid. Dynamic-diffuse global-illumination probes sit
//          on a regular grid; each frame we trace `raysPerProbe` rays per
//          probe, hash-blend the resulting irradiance + depth into octahedral
//          atlases with hysteresis, and the lighting kernel replaces the
//          irradiance-cube diffuse term with the probe-grid lookup.
// Phase 3.2: Split-sum DFG LUT. RG16Float LUT baked once at init from the
//          pre-integrated GGX BRDF. The lighting pass samples it at
//          (NdotV, roughness) and uses Karis's `F0 * scale + bias` for the
//          F0-weighted specular IBL — replacing the Lagarde roughness-Schlick
//          that Phase 3.0 shipped. Toggle via `dfgLUTEnabled` for A/B.
// Phase 3.3: DDGI two-bounce. Irradiance + depth atlases ping-pong (A/B);
//          the trace kernel reads PREVIOUS-frame atlases and adds a
//          Lambertian indirect bounce to the recorded radiance. Toggle via
//          `ddgiTwoBounceEnabled`. Atlases ping-pong unconditionally so the
//          accessors stay simple, but the second-bounce contribution is
//          gated by the uniform — toggling off mid-session reverts to
//          Phase 3.1's one-bounce result within one hysteresis time-constant.
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

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatorama")

    // ── Mesh kinds ────────────────────────────────────────────────────────────
    //
    // Phase 1 / 2 hard-coded three procedural primitives that both the G-buffer
    // pass and the DDGI trace kernel knew how to draw and intersect. Phase 2.6
    // adds `.custom(String)` for host-registered SCNGeometry conversions; the
    // renderer treats them identically to the built-ins for G-buffer / lighting
    // / IBL / SSR purposes, but the DDGI trace kernel can't analytically
    // intersect them (no BVH yet — that's Phase 4). Hits on `.custom` meshes
    // pass through DDGI silently as if the geometry weren't there; direct
    // lighting + IBL + SSR all work fine.
    public enum MeshKind: Hashable, Sendable {
        case box
        case sphere
        case ground
        case custom(String)

        /// GPU-side meshKind ID written into the per-instance struct. 0/1/2
        /// keep the DDGI trace kernel's analytic intersection paths;
        /// anything ≥ 3 is treated as "no analytic intersection available"
        /// — see `illumi_ddgi_trace` in Illuminatorama.metal. All
        /// `.custom(_)` meshes share the same GPU ID (3) because the trace
        /// kernel skips them uniformly; the Swift-side string identity is
        /// what disambiguates them in the mesh table.
        var gpuMeshKind: UInt32 {
            switch self {
            case .box:        return 0
            case .sphere:     return 1
            case .ground:     return 2
            case .custom:     return 3
            }
        }
    }

    public struct InstanceRef {
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
    public var exposure: Float = 1.0
    public var bloomThreshold: Float = 1.0
    public var bloomIntensity: Float = 0.6
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
    /// Post-FX easing time constant (seconds) from the panel's "Easing" picker.
    /// The post-FX knobs above (exposure, bloom, chromatic aberration, fringe) are
    /// treated as TARGETS; each frame `uploadFrameUniforms` eases an internal
    /// shadow toward the target with `k = 1 − e^(−dt/postFXEasingTau)`, so dragging
    /// a slider glides instead of snapping. 0 = instant (no easing).
    public var postFXEasingTau: Double = 0.40
    private var easedExposure: Float = 1.0
    private var easedBloomThreshold: Float = 1.0
    private var easedBloomIntensity: Float = 0.6
    private var easedChromaticAberration: Float = 0
    private var easedFringe: Float = 0
    private var easedFringeTint: SIMD3<Float> = SIMD3(0.62, 0.12, 0.92)
    private var easedSphericalAberration: Float = 0
    private var lastPostFXEaseTime: CFTimeInterval = 0
    public var time: Float = 0
    /// Vertex-shader tree-wind knobs (#58 #1). `treeWindStrength` is the max
    /// canopy sway in ~metres (0 = no wind, an exact shader no-op); `treeWindHeading`
    /// is the wind/gust travel direction in radians. Per-vertex sway weights ride
    /// in the geometry's tangent channel. Only the Forest scene sets these.
    public var treeWindStrength: Float = 0
    public var treeWindHeading: Float = 0

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
    /// Strength of the one-bounce indirect contribution.
    public var rtGIStrength: Float = 1.0
    public var rtShadowRays: Int = 4
    public var rtGIRays: Int = 4
    /// Soft sun specular strength on the RT direct term.
    public var rtSpecStrength: Float = 0.25
    public private(set) var rtSupported: Bool = false

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
    /// Neighborhood clamp width (sigmas). WIDE on purpose (~4): a tight clamp
    /// re-injects low-frequency crawl by yanking clean history toward the noisy
    /// current 3×3 mean. Only meant to catch gross on-screen disocclusions.
    public var rtGITemporalClamp: Float = 4.0

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
    public var surfaceCacheEnabled: Bool = false
    /// Indirect rays per atlas texel in the update pass (multi-bounce gather).
    public var surfCacheIndirectRays: Int = 4
    /// Temporal EMA blend for the cache update — fraction of the new radiance
    /// folded in each frame. Lower = smoother/slower convergence.
    public var surfCacheAlpha: Float = 0.1

    // ── Depth of field ───────────────────────────────────────────────
    /// Gather DOF on the resolved HDR before bloom. Off = sharp everywhere.
    public var dofEnabled: Bool = false
    /// Focus plane distance from the camera, in metres.
    public var dofFocusDistance: Float = 3.0
    /// Aperture multiplier on the circle of confusion (bigger = shallower).
    public var dofAperture: Float = 1.0
    /// Maximum blur radius in pixels (internal resolution) at full CoC.
    public var dofMaxRadius: Float = 14.0
    /// View-space distance from the focus plane at which the CoC saturates.
    public var dofFocusRange: Float = 4.0

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

    // ── Per-term split-render diagnostic ─────────────────────────────
    /// Isolates ONE lighting term in the deferred kernel so a flooded /
    /// flat scene can be decomposed (which term dominates the colour).
    /// `.normal` is the production composite. NOTE: the SSR and any planar-RT
    /// reflections composite in *separate* passes after this kernel, so to
    /// isolate them, A/B `ssrIntensity` rather than this enum.
    public enum DebugTerm: UInt32 {
        case normal = 0, directSun = 1, pointLights = 2, spotLights = 3
        case diffuseIBL = 4, specularIBL = 5, emission = 6, ambient = 7
        /// Isolates the RT surface-cache contribution (GI + reflection atlas
        /// reads) in the instanced/TLAS path — REPLACES the composite with just
        /// that term so a moving object's stale-vs-fresh cache is pixel-obvious.
        /// Unlike cases 1–7 (deferred-shading terms in `illumi_lighting`) this is
        /// wired in `illumi_rt_lighting_tlas`; it only shows with RT + surface
        /// cache on. See docs/illuminatorama/surface-cache-incremental-invalidation.md.
        case surfaceCacheGI = 8
        /// Isolates the surface cache's per-texel VARIANCE (E[L²] − μ², tracked in
        /// the atlas .w channel) at GI/reflection hits — a convergence heatmap:
        /// freshly-reset/cold cards bright, long-static cards near-black. Phase 5 /
        /// B0 instrumentation for the cache-domain denoiser. TLAS path only, like
        /// `surfaceCacheGI`. See docs/illuminatorama/phase5-radiance-cache-streaming.md.
        case surfaceCacheVariance = 9
    }
    public var debugTerm: DebugTerm = .normal
    /// Tracked by the renderer each frame (set from the host's
    /// `render()` argument). Used as the auto-exposure EMA's `dt`.
    private var lastFrameDuration: Double = 1.0 / 60.0

    // ── Phase 2 post-FX knobs ────────────────────────────────────────
    /// Screen-space ambient occlusion intensity (0 = disabled).
    public var ssaoIntensity: Float = 0.85
    /// World-space hemisphere radius for SSAO samples, in metres.
    public var ssaoRadius: Float = 0.4
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
    /// Up to this many spots in the `spotLights` array get a shadow map
    /// each frame. Bounded by `spotShadowAtlas`'s slice count. Spots
    /// beyond this index render as direct-only (no shadow modulation).
    public var spotShadowMaxCount: Int { spotShadowAtlasCapacity }

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

    // ── Phase 4.39 spatiotemporal denoiser knobs ────────────────────
    /// Enable the SSAO bilateral spatial filter + temporal accumulation.
    /// When on, AO converges from 16 samples to hundreds over time;
    /// turning off falls back to raw single-tap SSAO (noisier).
    public var ssaoDenoiseEnabled: Bool = true
    /// History retention weight for SSAO temporal (0 = all current, 1 = frozen).
    /// 0.90 gives ~10-frame effective sample count with fast transient recovery.
    public var ssaoTemporalBlend: Float = 0.90
    /// Enable the SSR temporal accumulation pass. Dramatically reduces shimmer
    /// on rough/medium surfaces. Off = raw single-sample gather goes straight
    /// to composite.
    public var ssrDenoiseEnabled: Bool = true
    /// History retention weight for SSR temporal. 0.85 balances lag vs noise.
    public var ssrTemporalBlend: Float = 0.85
    /// Apply the triangular-PDF debanding dither in the tonemap before the
    /// 8-bit store. On = smooth gradients; off = raw 8-bit quantisation
    /// (exposes contour banding). Exposed for live A/B of gradient banding.
    public var debandDitherEnabled: Bool = true

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
    /// Diagnostic A/B override: `VIZ_ILLUMI_TAA=1`/`0` forces the TAA toggle for
    /// every Illuminatorama renderer in the process, bypassing both the panel and
    /// any scene `sharedTAAOverride`. Lets a headless render exercise the TAA path
    /// without UI. `nil` when the env var is unset. Read once.
    private static let taaEnvOverride: Bool? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_TAA"] else { return nil }
        return v == "1" || v.lowercased() == "true"
    }()
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
    /// TAA contributes denoise of SSAO/SSR/DDGI noise without adding any
    /// motion. Set back to 0.5 to re-enable temporal supersample AA.
    public var taaJitterPixels: Float = 0.0

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
    /// Changes take effect on the next `resize(width:height:)` call.
    public var internalRenderScale: Float = 1.5

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
    public func markIBLDirty() { iblNeedsRebake = true }

    // ── Phase 3.1 DDGI knobs ─────────────────────────────────────────
    /// Toggle DDGI probe GI. When `true`, the lighting kernel samples the
    /// probe irradiance atlas for diffuse indirect lighting instead of the
    /// sky-probe irradiance cube. IBL specular is unaffected.
    public var ddgiEnabled: Bool = false
    /// Probe grid dimensions (x × y × z). Default 4×2×4 = 32 probes.
    public var ddgiGridDims: SIMD3<Int> = SIMD3(4, 2, 4)
    /// World-space position of the (0,0,0) corner probe.
    public var ddgiGridOrigin: SIMD3<Float> = SIMD3(-3, 0, -3)
    /// Distance between adjacent probes in metres.
    public var ddgiProbeSpacing: Float = 2.0
    /// Rays fired from each probe per frame. 64 is a good baseline.
    public var ddgiRaysPerProbe: Int = 64
    /// EMA weight kept from the previous irradiance frame (0.9–0.98).
    /// Higher = smoother but slower to react to lighting changes.
    public var ddgiHysteresis: Float = 0.97
    /// Scale on the final DDGI irradiance contribution.
    public var ddgiIrradianceScale: Float = 1.0
    /// Phase 3.3 — toggle one-bounce vs two-bounce DDGI. When on, the trace
    /// kernel reads the previous frame's irradiance atlas at every ray hit
    /// and adds the Lambertian-re-emitted indirect bounce to the recorded
    /// radiance, so the next probe update sees light that's been scattered
    /// once more around the room. First frame after enable still shows
    /// only direct light (the previous atlas is empty); subsequent frames
    /// converge as hysteresis builds up the indirect term.
    public var ddgiTwoBounceEnabled: Bool = false

    // ── Phase 3.2 split-sum DFG LUT ──────────────────────────────────
    /// Toggle the pre-integrated DFG LUT for the F0-weighted IBL specular
    /// response. Off = Lagarde's roughness-Schlick approximation (Phase 3.0
    /// behaviour). The LUT itself is baked once at startup and is
    /// view/material/scene-independent, so there is no rebake cost.
    public var dfgLUTEnabled: Bool = true

    // ── Output ────────────────────────────────────────────────────────────────

    /// Final LDR texture, ready to bind to `SCNMaterial.diffuse.contents`.
    /// This is the *presented* buffer — the one a consumer (SceneKit's
    /// `background.contents`) may sample. It is NEVER the buffer the current
    /// frame is tonemapping into; the render loop only promotes a pool buffer
    /// to `outputTexture` after that buffer's command buffer has *completed*.
    /// The host should NOT mutate this texture.
    public private(set) var outputTexture: MTLTexture

    /// Per-frame tap, fired at the end of `present(to:)` with the just-presented
    /// `outputTexture` and this renderer's `commandQueue`. The video recorder
    /// installs it to capture native (MTKView) scenes — there's no SceneKit
    /// render delegate to hook on this path. Copying on the renderer's OWN queue
    /// (not a fresh one) keeps the capture ordered behind the frame's render, so
    /// it reads a fully-written texture. `nil` (default) = no capture overhead.
    /// Set/cleared on the main thread; `present(to:)` runs on main.
    public var onFramePresented: ((_ texture: MTLTexture, _ queue: MTLCommandQueue) -> Void)?

    // ── Presented-buffer pool (overlay↔SceneKit race fix) ──────────────────────
    //
    // `outputTexture` is sampled by SceneKit on its OWN command queue (the
    // overlay binds it as `displayScene.background.contents`), with no
    // cross-queue sync to this renderer's tonemap write. Writing the presented
    // texture in place therefore races SceneKit's read: a heavy scene whose
    // tonemap is still in flight when SceneKit composites is sampled mid-write
    // and reads as zeros (a transparent / white frame). A light scene wins the
    // race by finishing in the idle gap, which is why the symptom looked
    // scene-specific.
    //
    // Fix: triple-buffer. The tonemap writes into a *free* pool buffer; only
    // after that buffer's command buffer COMPLETES (signalled on a background
    // thread via `presentSync`) is it promoted to `outputTexture` on the next
    // tick. SceneKit thus always samples a fully-rendered buffer, and we never
    // pay a per-frame `waitUntilCompleted` stall. FOUR buffers (`ldrPoolCount`)
    // cover: the presented one, the one SceneKit may still be reading from its
    // previous composite, and the TWO that may be in flight at once now that
    // `maxFramesInFlight = 2` pipelines CPU encode against GPU execution.
    private var ldrPool: [MTLTexture] = []
    private var presentedIdx: Int = 0
    /// The buffer presented on the *previous* tick. SceneKit's compositor can
    /// lag the rebind by ~1 frame, so we must not write into it either.
    private var prevPresentedIdx: Int = -1
    /// The pool buffer the current frame's tonemap writes into. Bound by
    /// `encodeTonemapPass`; chosen by `acquireWriteTarget()` each frame.
    private var tonemapWriteTarget: MTLTexture
    private let presentSync = IlluminatoramaPresentSync()
    private let gpuMeter = IlluminatoramaGPUMeter()
    // #60 task 4 / #6 — per-compute-pass GPU timer (env-gated). Assigned in init
    // once `device` exists. Off by default; no-ops without VIZ_ILLUMI_PASS_PROFILE.
    private let passTimer: IlluminatoramaPassTimer
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
    private lazy var rtThreadgroupMax: Int? = {
        if let s = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RT_TG"],
           let v = Int(s), v > 0 { return v }
        return nil
    }()

    // ── Frame throttle / pipelining (command-queue saturation guard) ────────────
    //
    // The overlay ticks `render()` at 60 Hz and commits a command buffer every
    // tick. A heavy scene whose GPU frame costs more than the tick interval
    // would otherwise back the queue up until `makeCommandBuffer()` blocks the
    // main thread on Metal's slot semaphore — multi-second freezes, and (the
    // symptom that masqueraded as a white frame) the presented buffer never
    // settling on a completed frame for SceneKit to sample. Cap frames in
    // flight and DROP a tick when the GPU hasn't caught up, rather than block.
    //
    // `maxFramesInFlight = 2` (was 1) is the "live-frame gap" fix. With ONE
    // frame in flight, the next frame can't begin encoding until the previous
    // one fully completes on the GPU; the 60 Hz timer then only starts it on the
    // following tick edge, so a >16.6 ms GPU frame leaves the GPU IDLE from its
    // completion until the next tick fires (e.g. a 24 ms frame finishes at 24 ms
    // but the next render() doesn't start until t=33 ms → ~9 ms wasted → the
    // new-frame rate quantises to 60/ceil(gpuMs/16.6) ≈ 30 fps even though the
    // GPU only needs ~24 ms). DOUBLE-buffering lets tick N+1 encode + queue the
    // next frame while frame N is still on the GPU, so the GPU runs back-to-back
    // and the new-frame rate rises to the GPU-bound ceiling (~1000/gpuMs). This
    // is the standard CPU/GPU pipelining every realtime renderer uses; it is NOT
    // the same as removing the throttle (we still DROP when 2 are already in
    // flight, so the queue stays bounded — no unbounded-growth freeze).
    //
    // Pairs with the `ldrPoolCount = 4` present pool: presented + previously-
    // presented (both possibly still read by SceneKit / the cover blit) + the
    // TWO buffers that may be in flight at once. `acquireWriteTarget()` excludes
    // all four so a frame never writes a buffer another consumer is reading.
    nonisolated static let maxFramesInFlight = 2
    private let inFlightSemaphore = DispatchSemaphore(value: IlluminatoramaRenderer.maxFramesInFlight)

    // ── Per-frame buffer-race probe (env-gated diagnostic) ──────────────────────
    // The blocky-lighting / jitter race (see docs/known-issues/illuminatorama-
    // inflight-buffer-race.md) is invisible to the headless harness because the
    // snapshot reads a SETTLED frame — but the race CONDITION still occurs in the
    // headless tick loop. This probe counts it quantitatively, so the bug is now
    // headless-detectable. `VIZ_ILLUMI_RACE_PROBE=1` turns it on; it writes a
    // running `raceEvents=N frames=M` to `VIZ_ILLUMI_RACE_PATH`. With the ring fix
    // the count is 0. `VIZ_ILLUMI_RACE_PIN0=1` pins the ring to slot 0 (reproduces
    // the pre-fix single-buffer behaviour) so the count goes positive — the
    // negative control that proves both the bug and the fix without a live window.
    nonisolated static let raceProbeEnabled = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PROBE"] == "1"
    nonisolated static let raceProbePin0    = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PIN0"] == "1"

    // ── Swap-count jitter probe (env-gated; issue #65 "swap-count jitter probe") ──
    // Measures REAL frame-delivery jitter, not the timer cadence the in-tick
    // VIZ_EGG_TRACE measures. Every time `promoteCompletedBuffer` actually swaps
    // `outputTexture` a new fully-rendered frame has been delivered; we timestamp
    // those swaps and histogram the inter-swap intervals into p50/p95/p99. A tight
    // p99-near-p50 = smooth delivery; a long tail = the jitter. `VIZ_ILLUMI_SWAP_PROBE=1`
    // turns it on, writing `swaps=… fps=… p50=… p95=… p99=… max=… (ms)` to
    // `VIZ_ILLUMI_SWAP_PATH`. Touched only on the main actor (in promoteCompletedBuffer).
    nonisolated static let swapProbeEnabled = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SWAP_PROBE"] == "1"
    private let swapProbePath = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SWAP_PATH"]
    private var swapLastTime: CFTimeInterval = 0
    private var swapIntervalsMs: [Double] = []

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // ── Internal state ────────────────────────────────────────────────────────

    private let engine: SimEngine
    private var meshes: [MeshKind: IlluminatoramaMesh] = [:]

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

    /// Glass that uses DIFFERENT meshes in the same pass. Each group binds its own
    /// mesh and draws its instances; takes precedence over `glassInstances`. Lets
    /// one scene mix glass shapes (a lab row of spheres + prisms + a slab). List
    /// far/background groups first.
    public struct GlassMeshGroup {
        public var kind: MeshKind
        public var instances: [IlluminatoramaGlassInstance]
        public init(kind: MeshKind, instances: [IlluminatoramaGlassInstance]) {
            self.kind = kind
            self.instances = instances
        }
    }
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

    private var glassRTPipeline: MTLRenderPipelineState?       // illumi_glass_rt_fs (traces TLAS)
    private var glassFallbackPipeline: MTLRenderPipelineState? // illumi_glass_fallback_fs (Fresnel+sky)
    private var glassDepthState: MTLDepthStencilState?
    /// Grows on demand in `encodeGlassPass` to hold the flattened glass instances.
    private lazy var glassInstanceBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaGlassInstance>.stride,
                          options: .storageModeShared)!
    }()
    private lazy var glassRTUniformBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaGlassRTUniforms>.stride,
                          options: .storageModeShared)!
    }()
    /// Per-glass-TLAS-instance material (RTGlassData[], grouped == TLAS glass
    /// order), rebuilt with the TLAS. Indexed by `instance_id - glassInstanceBase`.
    private var rtGlassDataBuffer: MTLBuffer?
    /// First glass instance_id in the TLAS (= opaque count + curve count). The RT
    /// glass shader uses this to tell a glass hit from an opaque one.
    private var rtGlassInstanceBase: Int = 0
    private var rtGlassInstanceCount: Int = 0

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
    private var causticDecayPipeline: MTLComputePipelineState?
    private var causticPhotonsPipeline: MTLComputePipelineState?
    private var causticPrimaryPipeline: MTLComputePipelineState?
    /// `atomic_float` RGB atlas (3 floats/texel), sized to the surface-cache atlas.
    private var causticAtlasBuffer: MTLBuffer?
    /// Per-glass-instance emitter discs (float4 centre+radius), rebuilt per frame.
    private var causticDiscBuffer: MTLBuffer?
    private var causticAtlasTexels: Int = 0
    private lazy var causticUniformBuffer: MTLBuffer = {
        device.makeBuffer(length: MemoryLayout<IlluminatoramaCausticUniforms>.stride,
                          options: .storageModeShared)!
    }()
    private var causticFrameSeed: UInt32 = 0

    private var gbufferAlbedoMet: MTLTexture
    private var gbufferNormalRgh: MTLTexture
    private var gbufferEmission: MTLTexture
    private var depthTexture: MTLTexture
    /// Last frame's G-buffer depth, blit-copied from `depthTexture` at the end of
    /// each frame and read by the TAA resolve for disocclusion rejection (so a
    /// fast mover that vacated a pixel onto a similar-luma surface doesn't trail —
    /// the case neighbourhood-clamp alone can't catch). Standalone, not pooled;
    /// safe to read-then-overwrite in place under `maxFramesInFlight == 1`.
    private var previousDepthTexture: MTLTexture
    private var hdrTexture: MTLTexture
    // Phase 2 — half-res visibility map from SSAO, fed to the lighting pass.
    private var aoTexture: MTLTexture
    // RT diffuse (soft shadow + 1-bounce GI) buffer — written by the RT pass,
    // bilateral-filtered into the composite by `encodeRTDenoiseComposite`.
    private var rtDiffuseTexture: MTLTexture
    // Phase 2 — full-res HDR with SSR composited on top of direct lighting.
    // Bloom and tonemap read from this so reflections feed both effects.
    private var hdrCompositeTexture: MTLTexture
    /// Pre-glass copy of `hdrCompositeTexture`, allocated lazily only when cheap
    /// screen-space glass (mode 2) is in use. The glass pass samples this for the
    /// scene behind the pane (you can't sample a render target you're also writing).
    private var glassBackdropTexture: MTLTexture?
    private var bloomBrightHalf: MTLTexture
    private var bloomBlurHHalf: MTLTexture
    private var bloomBlurVHalf: MTLTexture

    // Phase 3 — sky-probe IBL. The irradiance + prefiltered cubes are baked
    // from `equirectSky` each frame (or whenever the sky texture changes).
    private let irradianceCube: MTLTexture
    private let prefilteredCube: MTLTexture
    private let prefilteredMipViews: [MTLTexture]
    /// 1×1 RGBA16F texture used in place of `equirectSky` when the host
    /// hasn't supplied one. Means the bake kernels and lighting pass can
    /// always bind a real texture and never branch on nil.
    private let dummySkyTexture: MTLTexture
    // Phase 3.2 — split-sum DFG LUT. RG16F keyed on (NdotV, roughness),
    // baked once on init via the `illumi_dfg_bake` kernel. View / material /
    // scene-independent so it never rebakes. Texture is bound at slot 10 of
    // the lighting kernel; the `dfgLUTEnabled` knob above gates sampling so
    // we can A/B against the Lagarde fallback without recompiling.
    private let dfgLUT: MTLTexture
    // Issue #65 — 3D colour-grading LUT slot. A `type3D` cube sampled in the
    // tonemap pass AFTER all other post (display-space grade). Baked to identity
    // on init so the default is an EXACT no-op; a scene replaces it via
    // `setColorLUT(...)` (or `loadCubeLUT(...)`). `colorLUTAmount` (0 = off) blends
    // the graded result over the ungraded one. `colorLUTSize` is the per-axis
    // resolution used for the half-texel sampling correction.
    private var colorLUT: MTLTexture
    private var colorLUTSize: Int = IlluminatoramaRenderer.colorLUTDefaultSize
    /// 0 = OFF (default) → the tonemap LUT branch is an exact no-op. 1 = full grade.
    public var colorLUTAmount: Float = 0
    /// Last built-in look baked into `colorLUT`, so `applySharedColorGrade` only
    /// rebakes the (CPU-built) LUT texture when the chosen look actually changes.
    private var appliedColorGradeLook: IlluminatoramaColorGradeLook = .none
    // #60 task 5 increment 2 — LTC area-light specular LUTs (Minv + magnitude),
    // baked once on init. `ltcValidated` is true only when the bake matched
    // brute-force ground truth; the lighting kernel falls back to MRP otherwise.
    // Bound at slots 16/17 of the lighting kernel.
    private let ltcMatTexture: MTLTexture
    private let ltcMagTexture: MTLTexture
    private let ltcValidated: Bool

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

    // Phase 2.5 — cascaded shadow maps. One depth array (3 slices) shared
    // across cascades; per-slice views aren't required because we attach a
    // single slice at a time via `pass.depthAttachment.slice`. The cascade
    // VP matrices are recomputed per frame and stuffed into FrameUniforms.
    private let shadowMap: MTLTexture
    private var cascadeVPs: [simd_float4x4] = Array(repeating: matrix_identity_float4x4, count: 3)

    // Phase 4.10 — depth atlas for spot light shadow maps. One slice per
    // shadowed spot. 8 slices × 512² × 4-byte depth = 8 MB. Fixed at
    // allocation time; `spotShadowMaxCount` reads from this capacity.
    private let spotShadowAtlas: MTLTexture
    private let spotShadowAtlasCapacity: Int = 8
    private let spotShadowMapResolution: Int = 512
    private var cascadeSplitsView: SIMD4<Float> = .zero

    // Phase 2.7 — motion vectors + TAA.
    // Velocity is the 4th G-buffer color attachment; we ping-pong two HDR
    // history textures so the TAA pass can read from one and write the
    // resolved current frame into the other.
    private var velocityTexture: MTLTexture
    private var historyA: MTLTexture
    private var historyB: MTLTexture
    /// When true, historyA holds the previous frame's TAA output and we
    /// write the current frame's resolve into historyB. Toggled each frame.
    private var historyToggle: Bool = false
    /// Previous frame's jittered view-projection matrix — drives the
    /// per-vertex prevClip computation in the G-buffer vertex shader.
    private var previousViewProjection: simd_float4x4 = matrix_identity_float4x4
    /// The jittered VP computed this frame; copied into
    /// `previousViewProjection` at the end of render() so next frame's
    /// vertex shader sees the right matrix.
    private var lastFrameViewProjection: simd_float4x4 = matrix_identity_float4x4
    /// Incremented every TAA-enabled render so the Halton(2,3) jitter walks
    /// through its sequence and accumulates as proper supersampling.
    private var taaFrameIndex: UInt32 = 0
    /// True until the renderer has produced at least one valid history
    /// frame; the TAA kernel uses this to write current straight through.
    /// Reset on resize, on `taaEnabled` going false→true, and on instance
    /// buffer growth (which destroys previous-frame instance data).
    private var taaNeedsFirstFrame: Bool = true
    /// Tracks the previous value of `taaEnabled` so a false→true transition
    /// re-primes the history.
    private var previousTaaEnabled: Bool = false
    /// Wall-clock time (in renderer `time` seconds) of the last successful
    /// IBL bake — used for the `iblRebakeInterval` fallback.
    private var iblLastBakeTime: Float = -.infinity

    // Phase 3.1 — DDGI probe grid. Atlases are allocated lazily on first use
    // (or on grid-dim change) via `ensureDDGIResources()`. They stay nil
    // until `ddgiEnabled` is true and at least one `render()` call has run.
    // Phase 3.3 — ping-pong the irradiance + depth atlases so the trace
    // kernel can sample the PREVIOUS frame's atlases (for the second bounce)
    // while the update kernels write the CURRENT frame's atlases, all
    // within a single compute encoder. Both atlases must ping-pong because
    // both are sampled by `sampleDDGIIrradiance` for the Chebyshev
    // visibility test.
    //
    // After all three DDGI kernels finish, the lighting kernel reads the
    // CURRENT atlases (which now hold this frame's fresh writes). At the
    // end of `encodeDDGIFrame` we toggle `ddgiUseAtlasA` so next frame the
    // labels swap: what was "current" becomes "previous", and the other
    // texture gets overwritten with new data.
    //
    // Two-bounce off: first frame after enable sees an empty previous
    // atlas (one-bounce result); convergence is fast because the trace
    // kernel pumps fresh radiance into the current atlas every frame and
    // the hysteresis EMA bakes it into the steady state over ~20-30 frames.
    private var ddgiIrradianceAtlasA: MTLTexture?
    private var ddgiIrradianceAtlasB: MTLTexture?
    private var ddgiDepthAtlasA: MTLTexture?
    private var ddgiDepthAtlasB: MTLTexture?
    /// True ⇒ A is the current frame's write target / lighting-read source,
    /// B is the previous-frame read source for the trace kernel. Toggles
    /// at end of `encodeDDGIFrame`.
    private var ddgiUseAtlasA: Bool = true

    private var ddgiIrradianceCurrent: MTLTexture? {
        ddgiUseAtlasA ? ddgiIrradianceAtlasA : ddgiIrradianceAtlasB
    }
    private var ddgiIrradiancePrevious: MTLTexture? {
        ddgiUseAtlasA ? ddgiIrradianceAtlasB : ddgiIrradianceAtlasA
    }
    private var ddgiDepthCurrent: MTLTexture? {
        ddgiUseAtlasA ? ddgiDepthAtlasA : ddgiDepthAtlasB
    }
    private var ddgiDepthPrevious: MTLTexture? {
        ddgiUseAtlasA ? ddgiDepthAtlasB : ddgiDepthAtlasA
    }
    private var ddgiRayBuffer: MTLBuffer?
    private var ddgiInstanceDataBuffer: MTLBuffer?
    private var ddgiInstanceDataCapacity: Int = 0
    private var ddgiEmitterBuffer: MTLBuffer?
    private var ddgiEmitterCapacity: Int = 0
    private var activeEmitterLightCount: Int {
        particleEmitters.filter(\.enabled).filter { $0.ddgiLight != nil }.count
        + particleFields.filter { $0.ddgiLight != nil }.count
    }
    private var ddgiCurrentGridDims: SIMD3<Int> = .zero
    // Uniform buffer for DDGI — allocated once, written each frame.
    private let ddgiUniformBuffer: MTLBuffer
    // 1×1 rg16Float placeholder bound to the depth atlas slot when DDGI is off.
    private let ddgiDummyDepthAtlas: MTLTexture

    // ── Phase 4.39: denoiser textures ────────────────────────────────
    // AO spatiotemporal: raw AO → spatial filter → temporal → lighting.
    private var aoFilteredTexture: MTLTexture    // half-res, after bilateral
    private var aoHistoryA: MTLTexture           // half-res, r16Float ping-pong
    private var aoHistoryB: MTLTexture
    private var aoHistoryToggle: Bool = false
    private var aoNeedsFirstFrame: Bool = true

    // Phase 3.4 — per-pixel DDGI irradiance EMA cache (full-res, RGBA16F).
    private var irrCacheA: MTLTexture
    private var irrCacheB: MTLTexture
    private var irrCacheToggle: Bool = false
    private var irrCacheCurrent: MTLTexture { irrCacheToggle ? irrCacheB : irrCacheA }
    private var irrCachePrevious: MTLTexture { irrCacheToggle ? irrCacheA : irrCacheB }
    /// When true, the lighting kernel blends the fresh DDGI probe lookup with
    /// the previous-frame per-pixel cache (EMA), cutting the 8-probe trilinear
    /// blend to 1 texture read in steady state. Only effective when DDGI is on.
    public var ddgiIrrCacheEnabled: Bool = true
    /// EMA blend alpha: 0 = freeze history forever, 1 = always use fresh probe
    /// lookup (same as disabled). Default 0.05 converges in ~20 frames.
    public var ddgiIrrCacheBlend: Float = 0.05
    private var previousSsaoEnabled: Bool = false

    private var currentAOHistoryTexture: MTLTexture { aoHistoryToggle ? aoHistoryB : aoHistoryA }
    private var previousAOHistoryTexture: MTLTexture { aoHistoryToggle ? aoHistoryA : aoHistoryB }

    // SSR spatiotemporal: raw gather → temporal → composite into hdrComposite.
    private var ssrRawTexture: MTLTexture        // full-res, rgba16Float
    private var ssrHistoryA: MTLTexture          // full-res, rgba16Float ping-pong
    private var ssrHistoryB: MTLTexture
    private var ssrHistoryToggle: Bool = false
    private var ssrNeedsFirstFrame: Bool = true
    private var previousSsrEnabled: Bool = false

    private var currentSSRHistoryTexture: MTLTexture { ssrHistoryToggle ? ssrHistoryB : ssrHistoryA }
    private var previousSSRHistoryTexture: MTLTexture { ssrHistoryToggle ? ssrHistoryA : ssrHistoryB }

    // RT-GI temporal accumulation: the rtDiffuse term gets its own
    // velocity-reprojected exponential history so the 1-bounce GI keeps
    // converging under camera motion (the main TAA can't). Ping-pong, full-res.
    private var rtGIHistoryA: MTLTexture          // full-res, rgba16Float
    private var rtGIHistoryB: MTLTexture
    private var rtGIHistoryToggle: Bool = false
    private var rtGINeedsFirstFrame: Bool = true
    private var previousRTGITemporalEnabled: Bool = false
    private var currentRTGIHistoryTexture: MTLTexture { rtGIHistoryToggle ? rtGIHistoryB : rtGIHistoryA }
    private var previousRTGIHistoryTexture: MTLTexture { rtGIHistoryToggle ? rtGIHistoryA : rtGIHistoryB }
    /// The buffer the RT denoise reads: the temporally-accumulated GI when the
    /// pass is on, else the raw RT diffuse (original behaviour).
    private var rtDiffuseDenoiseSource: MTLTexture {
        (rtGITemporalEnabled && rtEnabled && rtSupported)
            ? currentRTGIHistoryTexture : rtDiffuseTexture
    }

    // ── SVGF denoiser (Phase 4.44) ───────────────────────────────────
    // Adaptive sample-count textures (r16Float) — one per temporal signal.
    // Each kernel reads the previous frame's count and writes the updated
    // count, driving adaptive alpha = max(minAlpha, 1/N) for faster
    // convergence in early frames and post-disocclusion recovery.
    private var aoSampleCount: MTLTexture       // half-res
    private var ssrSampleCount: MTLTexture      // full-res
    private var giSampleCount: MTLTexture       // full-res
    // SVGF à-trous cascade textures for RT GI (when svgfEnabled).
    private var giVariance: MTLTexture          // full-res r16Float
    private var giAtrousA: MTLTexture           // full-res rgba16Float
    private var giAtrousB: MTLTexture           // full-res rgba16Float
    /// Enable the SVGF à-trous variance-guided cascade for the RT GI diffuse term.
    /// When off, falls back to the fixed-radius bilateral denoise. Default on.
    public var svgfEnabled: Bool = true
    /// Number of à-trous cascade levels (1–5). Three levels cover a spatial
    /// reach of 1+2+4 = 7px radius; five levels cover 1+2+4+8+16 = 31px.
    public var svgfLevels: Int = 3
    /// Luminance variance weight (σ_L): smaller values produce tighter edge
    /// preservation; larger values allow more spatial averaging across edges.
    public var svgfSigmaL: Float = 4.0
    /// Depth edge-stopping weight (σ_Z). Default 1.0.
    public var svgfSigmaZ: Float = 1.0
    /// Normal edge-stopping exponent (σ_N). Higher = tighter. Default 128.
    public var svgfSigmaN: Float = 128.0

    // ── Phase 4.39: denoiser source selectors ────────────────────────
    // These let the lighting and composite passes bind the right texture
    // depending on whether the per-signal denoiser is on.
    private var aoSourceTexture: MTLTexture {
        ssaoDenoiseEnabled ? currentAOHistoryTexture : aoTexture
    }
    private var ssrDenoisedTexture: MTLTexture {
        ssrDenoiseEnabled ? currentSSRHistoryTexture : ssrRawTexture
    }

    private let gbufferPipeline: MTLRenderPipelineState
    private let shadowPipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let lightingPipeline: MTLComputePipelineState
    private let ssaoPipeline: MTLComputePipelineState
    private let ssrGatherPipeline: MTLComputePipelineState   // was ssrPipeline; now writes ssrRaw
    private let irradianceBakePipeline: MTLComputePipelineState
    private let prefilterBakePipeline: MTLComputePipelineState
    private let dfgBakePipeline: MTLComputePipelineState
    private let ssaoSpatialPipeline: MTLComputePipelineState
    private let ssaoTemporalPipeline: MTLComputePipelineState
    private let ssrTemporalPipeline: MTLComputePipelineState
    private let ssrCompositePipeline: MTLComputePipelineState
    private let rtDenoisePipeline: MTLComputePipelineState
    private let rtGITemporalPipeline: MTLComputePipelineState
    private let svgfVariancePipeline: MTLComputePipelineState  // Phase 4.44 SVGF
    private let svgfAtrousPipeline: MTLComputePipelineState    // Phase 4.44 SVGF
    private let taaResolvePipeline: MTLComputePipelineState
    private let bloomThresholdPipeline: MTLComputePipelineState
    private let bloomBlurHPipeline: MTLComputePipelineState
    private let bloomBlurVPipeline: MTLComputePipelineState
    // Phase 4.28 — tonemap is now a fullscreen RENDER pass (vertex+fragment)
    // rather than a compute kernel, so its write to `outputTexture` is visible
    // to SceneKit's cross-queue background sample without a CPU wait. See the
    // shader comment on `illumi_tonemap_vs`.
    private let tonemapPipeline: MTLRenderPipelineState
    // Phase 4.21 — auto-exposure estimator. Runs once per frame before
    // bloom; reduces HDR luminance and EMAs into a tiny GPU-side state
    // buffer the tonemap then reads `exposure` from.
    private let exposureEstimatePipeline: MTLComputePipelineState
    /// 16-byte GPU-only `ExposureState` (prevTargetLogLum + smoothed
    /// exposure + new target + dt). Persistent across frames because
    /// the EMA needs its own history. Storage mode is `.shared` so the
    /// host can seed it once at init and inspect for diagnostics; the
    /// kernel writes back via a `device&` binding.
    private let exposureBuffer: MTLBuffer

    // Phase 4.11 — particle pipelines. The compute step integrates
    // positions/velocities/life; the render pipeline draws survivors as
    // additive HDR point sprites into the composite.
    private let particleStepPipeline: MTLComputePipelineState
    private let particleDrawPipeline: MTLRenderPipelineState
    /// Phase 4.23 — host-buffer point-sprite renderer. Stride-aware so
    /// FireworksUltra's `SIMD4<Float>` star buffers + `packed_float3`
    /// burst buffers both bind without a copy.
    private let extParticleDrawPipeline: MTLRenderPipelineState
    // Phase 4.13a — repack kernel for the DynamicMesh bridge.
    private let repackPosNormPipeline: MTLComputePipelineState

    // Phase 4.21 — one-shot GPU mesh normal/tangent synthesis (replaces the
    // CPU passes that ran on the main thread inside IlluminatoramaMesh.from).
    private let meshBuildAdjacencyPipeline: MTLComputePipelineState
    private let meshSynthPipeline: MTLComputePipelineState
    /// Cover-blit pipeline — fullscreen textured quad with UV crop, used by
    /// `present(to:)` to route the output texture into an MTKView drawable
    /// without a SceneKit pass. `nil` when the library doesn't include the
    /// shader (shouldn't happen; logged once at init).
    private var coverBlitPipeline: MTLRenderPipelineState?
    /// Depth-stencil that READS the G-buffer depth without writing.
    /// Opaque geometry occludes particles behind it; particles don't
    /// occlude each other (additive HDR needs every drawn fragment to
    /// add to the composite).
    private let particleDepthState: MTLDepthStencilState

    /// Phase 4.12 — GPU-instanced draw recipe built once per frame from
    /// the host's `instances` array. Each entry is one `instanceCount`-
    /// wide draw call: `start` is the offset (in `IlluminatoramaInstance`
    /// units) into both the current and previous instance buffers, and
    /// `count` is the number of instances of `kind` to render in that
    /// single draw call. Built in `uploadInstances` after grouping by
    /// mesh kind; consumed by every pass that iterates the scene's
    /// per-instance data (G-buffer, cascaded shadows, spot shadows).
    private struct MeshDrawGroup {
        let kind: MeshKind
        let start: Int
        let count: Int
    }
    private var meshGroups: [MeshDrawGroup] = []

    // ── Perfect analytic superquadric (hero primitive) ─────────────────────────
    // The impostor pipeline (ray-traces the analytic surface in-fragment, writes
    // the G-buffer + analytic depth + motion vectors) and a parallel param buffer.
    // `impostorMeshKinds` are drawn by the impostor pipeline and SKIPPED by the
    // normal raster loop; `rtProxyMeshKinds` are skipped by BOTH raster loops and
    // exist only in the TLAS (RT shadows/GI/reflections). Both sets are populated
    // by `addSuperquadric` (IlluminatoramaSuperquadric.swift) — internal so that
    // extension can mutate them. The instance struct is untouched: raster-exclusion
    // is host-side via these sets, so there is no stride change.
    private var superquadricImpostorPipeline: MTLRenderPipelineState?
    /// Ring of `maxFramesInFlight` superquadric-param buffers (see `frameRingIndex`).
    /// Was single-buffered; with `maxFramesInFlight == 2` the previous frame's
    /// impostor draw can still be reading slot N while the CPU writes frame N+1,
    /// so this is now cycled per accepted frame exactly like the instance ping-pong.
    private var superquadricParamRing: [MTLBuffer]
    /// `superquadricParamBuffer` resolves to the CURRENT frame's ring slot, so the
    /// ~40 bind sites that name it need no change.
    private var superquadricParamBuffer: MTLBuffer { superquadricParamRing[frameRingIndex] }
    private var superquadricParamCapacity: Int
    var impostorMeshKinds: Set<MeshKind> = []
    var rtProxyMeshKinds: Set<MeshKind> = []

    /// Active particle emitters. Hosts register via `addEmitter` and the
    /// renderer ticks + draws each enabled one per frame.
    private var particleEmitters: [IlluminatoramaParticleEmitter] = []
    /// Particle fields published through `SimEngine.particleFields` for the
    /// active scene, refreshed each frame by the extractor via
    /// `setParticleFields`. Each is drawn as additive HDR point sprites into
    /// the same composite the lighting pass writes; bloom + tonemap process
    /// them naturally. (Replaces the Phase 4.23 SCNGeometry-associated-object
    /// bridge with the shared SimEngine registry — see `ParticleFieldSource`.)
    private var particleFields: [ParticleFieldSource] = []
    /// Velocity-aligned streak sources refreshed each frame by the host.
    /// Rendered after point-sprite fields in `encodePostResolveFX` so both
    /// participate in the same HDR bloom + tonemap pass.
    private var particleStreaks: [StreakSource] = []
    /// Pipeline for velocity-aligned billboard streak quads (`particleStreakVS/FS`).
    /// `nil` when the shaders are absent from the library — streak draws are
    /// silently skipped in that case (logged once at init).
    private var streakPipeline: MTLRenderPipelineState?

    /// Matches the Metal `ExtParticleParams` struct byte-for-byte.
    /// Set via `setVertexBytes` per emitter draw.
    private struct ExternalPointParams {
        var positionStrideFloats: UInt32
        var colorStrideFloats: UInt32
        var positionOffsetFloats: UInt32
        var colorOffsetFloats: UInt32
        var pointSize: Float
        var colorScale: Float
    }
    /// Wall-clock of the previous `render()` call, used to derive a real
    /// `dt` for particle integration even when the host's frame rate
    /// varies.
    private var lastParticleTickTime: CFTimeInterval = CACurrentMediaTime()
    // Phase 3.1 — DDGI compute kernels.
    private let ddgiTracePipeline: MTLComputePipelineState
    private let ddgiUpdateIrrPipeline: MTLComputePipelineState
    private let ddgiUpdateDepthPipeline: MTLComputePipelineState

    // ── Hardware ray tracing state ───────────────────────────────────
    /// Mirror of `RTUniforms` in IlluminatoramaRT.metal.
    private struct RTUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var _pad0: Float = 0
        var sunDir: SIMD3<Float>;         var sunSoftnessRad: Float
        var sunColor: SIMD3<Float>;       var giStrength: Float
        var skyAmbient: SIMD3<Float>;     var specStrength: Float
        var width: UInt32; var height: UInt32
        var shadowRays: UInt32; var giRays: UInt32
        var frameSeed: UInt32; var rayTMin: Float; var maxGIDist: Float; var triangleCount: UInt32
        var reflStrength: Float; var reflMaxDist: Float; var reflRoughnessCutoff: Float
        var reflRays: UInt32; var reflEnabled: UInt32
        // Surface cache: read cached radiance at GI/reflection hits when enabled.
        var surfCacheEnabled: UInt32; var surfTileSize: UInt32; var surfTilesPerRow: UInt32
        var surfAtlasW: UInt32; var surfAtlasH: UInt32
        var emitterCount: UInt32 = 0
        var leafTransmission: Float = 0      // mirrors _padRT0 in the Metal struct
        // Curve primitives (#60 item 7, incr. 2): 1 ⇒ soup AS holds a curve
        // geometry descriptor; mirrors `curvesEnabled` (was `_padRT1`) in Metal.
        var curvesEnabled: UInt32 = 0; var _padRT2: UInt32 = 0
    }
    /// Mirror of `RTDenoiseUniforms` in Illuminatorama.metal (stride 32, 16-aligned).
    private struct RTDenoiseUniforms {
        var width: UInt32; var height: UInt32; var enabled: UInt32; var radius: UInt32
        var kDepth: Float; var kNorm: Float; var _pad0: Float = 0; var _pad1: Float = 0
    }
    /// Mirror of `RTGITemporalUniforms` in Illuminatorama.metal (stride 32, 16-aligned).
    private struct RTGITemporalUniforms {
        var width: UInt32; var height: UInt32; var enabled: UInt32; var isFirstFrame: UInt32
        var blend: Float; var gammaClamp: Float; var _pad0: Float = 0; var _pad1: Float = 0
    }
    private let rtPipeline: MTLComputePipelineState?
    private var rtAccel: MTLAccelerationStructure?
    private var rtScratch: MTLBuffer?
    private var rtVertexBuffer: MTLBuffer?
    private var rtIndexBuffer: MTLBuffer?
    private var rtTriAlbedoBuffer: MTLBuffer?
    private var rtTriNormalBuffer: MTLBuffer?
    private var rtTriangleCount: Int = 0
    private let rtUniformBuffer: MTLBuffer
    private let rtDenoiseUniformBuffer: MTLBuffer
    private let rtGITemporalUniformBuffer: MTLBuffer
    private var rtFrameSeed: UInt32 = 0

    // ── Surface cache (Lumen-style on-surface lit-radiance atlas) ────
    /// Mirror of `SurfCard` in IlluminatoramaSurfaceCache.metal — one atlas tile.
    /// A tile carries up to TWO planar triangle frames (P3 — 2-triangles-per-tile
    /// packing): frame A occupies the lower-left half (texel `u+v ≤ 1`), frame B
    /// the upper-right half (`u+v > 1`), split on the diagonal. `normal.w` is the
    /// PACKED flag: 1 ⇒ the tile is split between frame A and frame B; 0 ⇒ frame A
    /// fills the WHOLE tile (the room's hand-authored quad cards, where both of a
    /// quad's triangles map across the full tile, and a lone odd per-triangle
    /// card). All `SIMD4` (no SIMD3/packed_float3 stride trap).
    public struct SurfCard {
        // Frame A — lower-left half when packed, whole tile when unpacked.
        public var origin: SIMD4<Float>, uAxis: SIMD4<Float>, vAxis: SIMD4<Float>
        public var normal: SIMD4<Float>, albedo: SIMD4<Float>, emission: SIMD4<Float>
        // Frame B — upper-right half; meaningful only when `normal.w == 1`.
        public var originB: SIMD4<Float>, uAxisB: SIMD4<Float>, vAxisB: SIMD4<Float>
        public var normalB: SIMD4<Float>, albedoB: SIMD4<Float>, emissionB: SIMD4<Float>
        /// Unpacked single-frame card — frame A fills the whole tile (room quads,
        /// lone odd per-triangle card). Frame B left zero, packed flag (normal.w) 0.
        public init(origin: SIMD3<Float>, uAxis: SIMD3<Float>, vAxis: SIMD3<Float>,
                    normal: SIMD3<Float>, albedo: SIMD3<Float>, emission: SIMD3<Float>) {
            self.origin = SIMD4(origin, 0); self.uAxis = SIMD4(uAxis, 0)
            self.vAxis = SIMD4(vAxis, 0); self.normal = SIMD4(simd_normalize(normal), 0)
            self.albedo = SIMD4(albedo, 0); self.emission = SIMD4(emission, 0)
            self.originB = .zero; self.uAxisB = .zero; self.vAxisB = .zero
            self.normalB = .zero; self.albedoB = .zero; self.emissionB = .zero
        }
        /// Packed two-triangle card (P3): frame A in the lower-left half, frame B
        /// in the upper-right half. `normal.w = 1` flags the diagonal split.
        public init(aOrigin: SIMD3<Float>, aUAxis: SIMD3<Float>, aVAxis: SIMD3<Float>,
                    aNormal: SIMD3<Float>, aAlbedo: SIMD3<Float>, aEmission: SIMD3<Float>,
                    bOrigin: SIMD3<Float>, bUAxis: SIMD3<Float>, bVAxis: SIMD3<Float>,
                    bNormal: SIMD3<Float>, bAlbedo: SIMD3<Float>, bEmission: SIMD3<Float>) {
            self.origin = SIMD4(aOrigin, 0); self.uAxis = SIMD4(aUAxis, 0)
            self.vAxis = SIMD4(aVAxis, 0); self.normal = SIMD4(simd_normalize(aNormal), 1)
            self.albedo = SIMD4(aAlbedo, 0); self.emission = SIMD4(aEmission, 0)
            self.originB = SIMD4(bOrigin, 0); self.uAxisB = SIMD4(bUAxis, 0)
            self.vAxisB = SIMD4(bVAxis, 0); self.normalB = SIMD4(simd_normalize(bNormal), 0)
            self.albedoB = SIMD4(bAlbedo, 0); self.emissionB = SIMD4(bEmission, 0)
        }
    }
    /// Mirror of `SurfCacheUniforms` in IlluminatoramaSurfaceCache.metal.
    private struct SurfCacheUniforms {
        var sunDir: SIMD4<Float>; var sunColor: SIMD4<Float>; var skyAmbient: SIMD4<Float>
        var atlasW: UInt32; var atlasH: UInt32; var tileSize: UInt32; var tilesPerRow: UInt32
        var cardCount: UInt32; var triangleCount: UInt32; var indirectRays: UInt32; var frameSeed: UInt32
        var alpha: Float; var rayTMin: Float; var maxDist: Float; var incrementalEnabled: UInt32 = 0
    }
    private let surfCachePipeline: MTLComputePipelineState?
    /// §3 endpoint — TLAS-traced cache-update (traces the per-frame-refit instance
    /// AS so a moved object's shadow/bounce-occlusion tracks). Used in place of
    /// `surfCachePipeline` when a TLAS is live (`rtTLASActive`); the soup path keeps
    /// the primitive-AS `surfCachePipeline`.
    private let surfCacheTLASPipeline: MTLComputePipelineState?
    private var surfCardBuffer: MTLBuffer?
    /// Placeholder bound to the `surfCards` slot (buffer 9) when the surface cache
    /// is OFF. The lighting kernel never reads it then, but Metal argument
    /// validation still requires the bound buffer to be ≥ one `SurfCard` — and the
    /// generic dummies (`instData` / `triA`) can be smaller than a 192-B card
    /// (e.g. a single-instance RT buffer), which asserts at dispatch. Sized to
    /// exactly one card so the cache-off path validates regardless of struct size.
    private lazy var surfCardDummyBuffer: MTLBuffer? =
        device.makeBuffer(length: MemoryLayout<SurfCard>.stride, options: .storageModePrivate)
    private var surfTriCardBuffer: MTLBuffer?   // uint per triangle → card index
    private var surfTriUVaBuffer: MTLBuffer?    // float4 (uvA.xy, uvB.xy)
    private var surfTriUVcBuffer: MTLBuffer?    // float4 (uvC.xy, 0, 0)
    // P2 — per-card atlas RECT addressing (replaces the uniform tilesPerRow×tileSize
    // grid). `cardRect[card]` = (atlasX, atlasY, tileW, tileH) in atlas pixels;
    // `texelCard[y·atlasW + x]` = the card owning that texel (0xFFFFFFFF = gap).
    // Lets charts of different world extent get different-sized tiles, shelf-packed
    // into one atlas — the uniform grid is just the special case (all tiles equal).
    private var surfCardRectBuffer: MTLBuffer?  // float4 per card (x,y,w,h px)
    private var surfTexelCardBuffer: MTLBuffer? // uint per atlas texel → card index
    // ── Incremental invalidation (default-on, per-triangle TLAS path) ──
    // When `surfaceCacheIncremental` is on and the topology is stable, each frame
    // the moved instances' cards are re-framed in place + flagged dirty so the
    // update kernel re-lights only them (α=1) while stationary cards keep their
    // accumulated multi-bounce. See docs/illuminatorama/surface-cache-incremental-
    // invalidation.md. DEFAULT ON: it self-gates to the per-triangle TLAS path
    // (`surfIncrementalReady`) and is a no-op on chart/soup/static scenes (a fully
    // static frame diffs to zero moved instances → zero re-framing), so enabling it
    // by default only helps animated TLAS scenes. `VIZ_SURFCACHE_INCREMENTAL=0` is
    // the kill-switch. SAFE only while `maxFramesInFlight == 1` (the in-place
    // `surfCardBuffer` rewrite races a ≥2 pool — ping-pong the card buffer first).
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
    private let surfAtrousPipeline: MTLComputePipelineState?
    // Phase 5 / A (streaming) — residency feedback. When on, the GI/reflection
    // cache hits mark `cardRequested[card]=1`; the host drains the buffer the NEXT
    // frame (maxFramesInFlight==1) to log the per-frame WORKING SET size — the data
    // that sizes the atlas budget and that A1's residency pass will key off. Plain
    // instrumentation for now (no behavior change). `VIZ_ILLUMI_SURFCACHE_FEEDBACK=1`.
    public var surfaceCacheFeedback: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_FEEDBACK"] == "1"
    private var surfCardRequestedBuffer: MTLBuffer?   // uint per card (1 = sampled this frame)
    // Phase 5 / A0 (streaming) — card budget. >0 caps the number of RESIDENT cards
    // (atlas sized to the budget, not the scene); cards beyond it get no atlas slot,
    // and a GI/reflection hit on a non-resident card falls back to emission + ambient
    // (never black). Static bake-order residency for A0 (feedback-driven priority is
    // A1). 0/unset ⇒ disabled (every card resident — today's behaviour).
    // `VIZ_ILLUMI_SURFCACHE_CARD_BUDGET`.
    public var surfCardBudget: Int =
        Int(ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_CARD_BUDGET"] ?? "") ?? 0
    private var surfResidentCardCount: Int = 0   // ≤ surfCardCount; == it when budget disabled
    // Phase 5 / A1 (streaming) — DYNAMIC feedback-driven residency. When on (with a
    // budget), a per-frame host pass reassigns which cards occupy the fixed atlas
    // SLOTS (= A0's bake-time resident-card rects) based on the `cardRequested`
    // working set: a sampled-but-non-resident card is promoted into a slot freed by
    // an unsampled resident card. Mutates `texelCard` + `cardRect` + `dirty` in place
    // (safe under maxFramesInFlight==1, same class as the incremental writes) — NO
    // kernel addressing change (the kernels still read texel→card / card→rect, and
    // A0's zero-rect-⇒-fallback path already handles a non-resident card).
    // `VIZ_ILLUMI_SURFCACHE_STREAM=1` (implies feedback). Default off.
    public var surfaceCacheStreaming: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_STREAM"] == "1"
    private var surfSlotRect: [SIMD4<Float>] = []   // per slot: atlas rect (bake-time resident-card rects)
    private var surfSlotCard: [Int] = []            // slot → occupant card index
    private var surfCardSlot: [Int] = []            // card → slot (−1 = non-resident)
    private let surfMaxSwapsPerFrame = 256          // bound the per-frame texelCard rewrite + anti-thrash
    // Phase 5 / A2 — eviction hysteresis. Per-card "heat": set to `surfStreamHeatFull`
    // when sampled, decremented each frame otherwise. A resident card is demotable only
    // once its heat hits 0 (unsampled for HeatFull consecutive frames), so a card
    // sampled intermittently (grazing rays, the moving box's trail) stays resident
    // instead of flipping in/out every frame — cuts the steady-state swap churn.
    private var surfCardHeat: [Int] = []
    private let surfStreamHeatFull =
        Int(ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_HEAT"] ?? "") ?? 4
    // Phase 5 / A4 — importance-weighted residency (the tractable core of per-card
    // LOD). Under eviction pressure (budget < working set), keep the NEAR cards (large
    // screen footprint) over far ones: promote the most-important requested cards
    // first, demote the least-important cold slots first. Importance = camera proximity
    // from the card's bake-time centroid. (Full variable-slot-size LOD — smaller atlas
    // tiles for far cards — is the deferred bin-packing follow-on; this prioritises
    // WHICH cards hold the fixed slots, not slot size.) `VIZ_ILLUMI_SURFCACHE_LOD=1`.
    public var surfaceCacheLOD: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_LOD"] == "1"
    private var surfCardCentroid: [SIMD3<Float>] = []   // bake-time world centroid per card
    private var surfCardDirtyBuffer: MTLBuffer?     // uint per card (1 = re-light fresh)
    private var surfTriCardCPU: [UInt32] = []       // triCard mirror (per soup triangle)
    private var surfTriIsFrameB: [Bool] = []        // per triangle: B-half of its packed card?
    private var surfSoupBaseCPU: [UInt32] = []       // per instance: first soup-triangle index
    private var surfInstanceKind: [MeshKind] = []    // per instance: mesh kind (for object soup)
    private var surfObjectSoupByKind: [MeshKind: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
    private var surfIncrementalReady: Bool = false   // CPU maps valid for this topology
    // ── Chart-path incremental invalidation (#60 task 1) ──
    // The per-triangle path (above) re-frames each moved triangle from its object
    // soup × model. A coplanar CHART card (`VIZ_SURFCACHE_CHARTS=1`) instead spans
    // many triangles with ONE bbox-derived frame + fixed per-triangle UVs, so the
    // per-triangle re-frame would clobber the chart's parameterisation. But a card
    // frame is AFFINE-COVARIANT: under a rigid/affine model matrix `M`, the world
    // frame is exactly `origin' = M·origin` (point), `uAxis'/vAxis' = M·axis`
    // (vector), and the per-triangle UVs (`uv = (proj(v)−bboxMin)/bboxSize`) are
    // invariant because verts and frame transform together. So a moved chart card
    // re-frames with a single affine transform of its BAKE-pose object-space frame
    // (`M_bake⁻¹ · worldFrame`), with no per-triangle work and no UV change. Built
    // only on the chart bake path; nil ⇒ per-triangle path (or no incremental).
    private struct SurfCardFrame {
        var origin = SIMD3<Float>(0, 0, 0), uAxis = SIMD3<Float>(0, 0, 0)
        var vAxis = SIMD3<Float>(0, 0, 0), normal = SIMD3<Float>(0, 1, 0)
    }
    private struct ChartIncremental {
        var objectFrame: [SurfCardFrame]   // per card: bake-pose frame in owner's object space
        var instCardLo: [Int32]            // per instance: first owned card (−1 if none)
        var instCardHi: [Int32]            // per instance: one-past-last owned card
    }
    private var surfChartIncremental: ChartIncremental?
    // Reused per-frame scratch for the chart incremental path's "which instances
    // moved" pass — sized once at bake time (when instance count is known) and
    // reset in place each frame, so the steady-state loop allocates nothing.
    private var surfChartMovedScratch: [Bool] = []
    // ── GPU-side transform diff (#60 item 1) ──
    // Moves the per-frame moved-set detection + card re-frame off the CPU: a
    // compute kernel on the frame's command buffer diffs each card's owner
    // instance (current vs previous model matrix) and re-frames moved cards in
    // place, replacing the CPU loops in `applyIncrementalSurfaceCacheUpdate` /
    // `applyIncrementalChartSurfaceCacheUpdate` for RIGID movers. The CPU path
    // stays as the fallback for Phase-C deforming scenes (their re-frame reads
    // the live soup — GPU-ifying that is the separate Phase D item) and as the
    // `VIZ_SURFCACHE_GPU_DIFF=0` kill-switch / A/B reference. The maps below are
    // baked alongside the CPU ones and cleared with them on topology change.
    private static let surfCacheGPUDiffDisabled =
        ProcessInfo.processInfo.environment["VIZ_SURFCACHE_GPU_DIFF"] == "0"
    private let surfReframeTriPipeline: MTLComputePipelineState?
    private let surfReframeChartPipeline: MTLComputePipelineState?
    private let surfReframeDeformPipeline: MTLComputePipelineState?
    private var surfGPUDiffTriOwnerBuffer: MTLBuffer?    // uint per soup tri → instance
    private var surfGPUDiffTriFlagsBuffer: MTLBuffer?    // uint per soup tri: SC_FLAG_* bits
    private var surfGPUDiffObjVertsBuffer: MTLBuffer?    // float4 ×3 per soup tri (object space)
    private var surfGPUDiffTriCount: Int = 0
    private var surfGPUDiffChartOwnerBuffer: MTLBuffer?  // int32 per card (−1 = static)
    private var surfGPUDiffChartFrameBuffer: MTLBuffer?  // float4 ×4 per card (object frame)
    // ── Phase D — deforming meshes, GPU-resident ──
    /// Vertex-index triple per GLOBAL soup triangle (topology constant per bake);
    /// the deform kernel reads these + the LIVE packed_float3 position buffer.
    private var surfGPUDiffDeformIdxBuffer: MTLBuffer?
    /// One entry per deforming INSTANCE: the per-dispatch parameters. The
    /// position buffer is looked up per encode from `gpuRepackTasks` by kind
    /// (the solver may re-register; the kind survives the topology bake).
    private struct SurfDeformDispatch {
        var kind: MeshKind; var instanceIndex: Int; var triBase: Int; var triCount: Int
    }
    private var surfGPUDiffDeformDispatches: [SurfDeformDispatch] = []
    /// 4×uint written by the kernels via relaxed atomics ([0] moved instances,
    /// [1] dirtied cards, [2] deformed instances, [3] spare), read + zeroed by
    /// the CPU the NEXT frame (safe under `maxFramesInFlight == 1`) for the
    /// `recordSurfCacheStats` sidecar.
    private var surfGPUDiffStatsBuffer: MTLBuffer?
    /// Swift mirror of the Metal `SCReframeUniforms` (32 B, passed via setBytes).
    private struct SCReframeUniforms {
        var count: UInt32; var cardCount: UInt32; var instanceCount: UInt32; var epsilon: Float
        var rigidEnabled: UInt32; var _pad0: UInt32 = 0; var _pad1: UInt32 = 0; var _pad2: UInt32 = 0
    }
    /// Swift mirror of the Metal `SCDeformUniforms` (16 B, per deform dispatch).
    private struct SCDeformUniforms {
        var triBase: UInt32; var triCount: UInt32; var instanceIndex: UInt32; var vertexCount: UInt32
    }
    /// `triFlags` bit layout — mirrors the Metal SC_FLAG_* defines.
    private static let scFlagFrameB: UInt32 = 1
    private static let scFlagDeforming: UInt32 = 2
    // Phase C — a deforming (DynamicMesh / GPU-fed) mesh contributed surface cards
    // this topology. Its cards are re-framed from LIVE vertices every frame
    // (`objectTriangleSoup()` is empty for its private vertex buffer, so it's
    // otherwise excluded from the cache), and the update kernel must honour the
    // per-card dirty flags (α=1) just as for rigid incremental invalidation.
    private var surfHasDeformingCards: Bool = false
    /// Dev A/B override (`VIZ_ILLUMI_NO_DEFORM_CARDS=1`) — freeze deforming cards
    /// at their build pose (pre-Phase-C behaviour) so the catch/cast win is
    /// directly comparable. Off by default; the re-frame is on whenever a
    /// deforming mesh drives the cache.
    private let noDeformCardsOverride =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_DEFORM_CARDS"] == "1"
    private var surfAtlasA: MTLTexture?
    private var surfAtlasB: MTLTexture?
    // Phase 5 / B1 — cache-domain denoiser output. NOT ping-ponged: recomputed
    // each frame from the just-updated current atlas (display-only; the EMA
    // feedback keeps reading the raw atlas). Bound to the GI/reflection consumers
    // in place of the current atlas when `surfaceCacheDenoise` is on.
    private var surfAtlasDenoised: MTLTexture?
    private var surfUseAtlasA: Bool = true
    private let surfCacheUniformBuffer: MTLBuffer
    private var surfCardCount: Int = 0
    /// Default per-card atlas tile size — sized for the room's big planar cards.
    private var surfTileSize: Int = 64
    /// Tile size of the CURRENTLY-registered card set. Per-triangle micro-cards
    /// (Phase 4.38) pass a much smaller tile (a triangle doesn't need 64²), so
    /// the atlas stays bounded; the room keeps 64. Set by `setSurfaceCacheCards`.
    private var surfActiveTileSize: Int = 64
    private var surfTilesPerRow: Int = 1
    private var surfAtlasW: Int = 1
    private var surfAtlasH: Int = 1
    private var surfFrameSeed: UInt32 = 0
    /// The atlas the lighting/RT pass should READ this frame (current). Set by
    /// `encodeSurfaceCacheUpdate` after the ping-pong swap.
    private var surfCacheCurrentAtlas: MTLTexture? { surfUseAtlasA ? surfAtlasA : surfAtlasB }
    private var surfCachePreviousAtlas: MTLTexture? { surfUseAtlasA ? surfAtlasB : surfAtlasA }
    /// Phase 5 / B1 — the atlas the GI/reflection CONSUMERS read: the denoised
    /// copy when the cache-domain filter is on (and allocated), else the raw
    /// current atlas. The update kernel's feedback always reads the raw atlas.
    private var surfConsumerAtlas: MTLTexture? {
        (surfaceCacheDenoise ? surfAtlasDenoised : nil) ?? surfCacheCurrentAtlas
    }

    // ── TLAS — instanced RT for animated extracted scenes ────────────
    // Per-mesh BLAS built once (object space); a TLAS of per-instance
    // transforms refit each frame. Replaces the room-only / soup primitive-AS
    // path for extracted overlay scenes so RT generalises to ANIMATION without
    // a per-frame vertex transform + geometry rebuild.
    /// Mirror of `RTInstanceData` in IlluminatoramaRTInstanced.metal.
    private struct RTInstanceData {
        var nrm0: SIMD4<Float>; var nrm1: SIMD4<Float>; var nrm2: SIMD4<Float>
        var albedoTriBase: SIMD4<Float>   // xyz = albedo, w = objNormal base (as Float)
    }
    /// Mirror of `RTInstUniforms`.
    private struct RTInstUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var _pad0: Float = 0
        var sunDir: SIMD3<Float>; var sunSoftnessRad: Float
        var sunColor: SIMD3<Float>; var giStrength: Float
        var skyAmbient: SIMD3<Float>; var specStrength: Float
        var width: UInt32; var height: UInt32
        var shadowRays: UInt32; var giRays: UInt32
        var frameSeed: UInt32; var rayTMin: Float; var maxGIDist: Float; var _pad1: UInt32 = 0
        var reflStrength: Float; var reflMaxDist: Float; var reflRoughnessCutoff: Float
        var reflRays: UInt32; var reflEnabled: UInt32
        // Surface cache (P1c) — mirror of the Metal RTInstUniforms tail.
        var surfCacheEnabled: UInt32 = 0
        var surfTileSize: UInt32 = 1; var surfTilesPerRow: UInt32 = 1
        var surfAtlasW: UInt32 = 1; var surfAtlasH: UInt32 = 1
        var surfTriCount: UInt32 = 0
        var debugSurfCacheGI: UInt32 = 0
        // Curve primitives (#60 item 7): TLAS instance ids ≥ curveInstanceBase
        // are curve sets (id − base = set index). 0 sets ⇒ fields unread.
        var curveInstanceBase: UInt32 = 0
        var curveSetCount: UInt32 = 0
        // Phase 5 / B0 — DebugTerm.surfaceCacheVariance isolation (was _padc0).
        var debugSurfCacheVar: UInt32 = 0
        // Phase 5 / A — residency feedback gate (was _padc1).
        var surfFeedbackEnabled: UInt32 = 0
    }
    private let rtTLASPipeline: MTLComputePipelineState?
    /// Whether the device + instanced-RT pipeline are available.
    public var rtTLASSupported: Bool { rtTLASPipeline != nil }
    private var rtBLASByMesh: [ObjectIdentifier: MTLAccelerationStructure] = [:]
    private var rtNormalsByMesh: [ObjectIdentifier: [SIMD4<Float>]] = [:]
    // Phase B — deforming-mesh BLAS refit. A DynamicMesh / GPU-fed mesh keeps a
    // STABLE topology (vertex+index count fixed; only positions move each frame),
    // so its BLAS is built once with `.refit` usage and re-fit per frame from the
    // live (repacked) vertex buffer — instead of being frozen at build-time verts,
    // which made the RT trace see stale geometry (shadows/GI lit for the old pose).
    // The refit is encoded on the FRAME command buffer AFTER `encodeGPURepacks`, so
    // Metal sequences repack-write → refit-read; the BLAS references `mesh.vertexBuffer`,
    // the same buffer the repack writes in place.
    private var rtBLASRefitDesc: [ObjectIdentifier: MTLPrimitiveAccelerationStructureDescriptor] = [:]
    private var rtBLASRefitScratch: [ObjectIdentifier: MTLBuffer] = [:]
    /// Dev A/B override (VIZ_ILLUMI_NO_BLAS_REFIT=1) — freeze deforming BLASes at
    /// build pose to compare RT-traces-stale vs current. Default off (refit on).
    private static let noBLASRefit =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_BLAS_REFIT"] == "1"
    /// Dev A/B override (VIZ_ILLUMI_NO_PRIM_UV=1) — skip baking per-mesh card UVs
    /// into BLAS `primitive_data` (#60 item 6), so every surface-cache hit falls
    /// back to the `triUVa`/`triUVc` side buffers (the pre-fold "before"). Lets the
    /// AS-baked-UV win be measured/compared without a stash. Default off (fold on).
    private static let noPrimUVFold =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_PRIM_UV"] == "1"
    /// Coplanar charts (`VIZ_SURFCACHE_CHARTS=1`) use a different, non-mesh-invariant
    /// UV parameterisation, so the BLAS UV fold is per-triangle-path only.
    private static let surfCacheCharts =
        ProcessInfo.processInfo.environment["VIZ_SURFCACHE_CHARTS"] == "1"
    /// Whether to bake per-mesh card UVs into BLAS `primitive_data` (#60 item 6):
    /// per-triangle surface-cache path, cache on, fold not disabled. Evaluated at
    /// BLAS-build time; if the cache turns on only AFTER a mesh's BLAS was built,
    /// that BLAS keeps no primitive data and the shader's null-check falls back to
    /// the side buffers (correct, just no perf win for that mesh).
    private var bakePrimUVFold: Bool {
        surfaceCacheEnabled && !Self.surfCacheCharts && !Self.noPrimUVFold
    }
    private var rtResidentBuffers: [MTLBuffer] = []   // mesh vb/ib the BLASes reference
    private var rtBLASList: [MTLAccelerationStructure] = []   // index = accelerationStructureIndex
    private var rtObjNormalBuffer: MTLBuffer?    // concatenated object-space per-tri normals (float4)
    private var rtObjNormalCount: Int = 0
    // Surface cache (P1c): per-TLAS-instance base offset into the grouped soup
    // triangle list (== the `triCard`/`triUVa`/`triUVc` index space). Built
    // alongside the world-space soup in `rebuildRTAccel` when the cache is on,
    // so a TLAS hit resolves `soupTriBase[instance_id] + primitive_id` → card.
    private var rtSoupTriBaseBuffer: MTLBuffer?
    private var rtSoupTriBaseCount: Int = 0
    private var rtInstanceDataBuffer: MTLBuffer? // RTInstanceData[], grouped order
    private var rtInstanceDescBuffer: MTLBuffer? // MTLAccelerationStructureInstanceDescriptor[]
    private var rtTLAS: MTLAccelerationStructure?
    private var rtTLASScratch: MTLBuffer?
    private var rtTLASCapacity: Int = 0
    private var rtTLASInstanceCount: Int = 0
    private var rtTLASTopologyHash: Int = .min
    // ── Curve primitives (#60 item 7, increment 1) ──
    // Sets adopted from `IlluminatoramaCurveRegistry` (synced per frame by
    // version). Control points / radii / segment indices for ALL sets are
    // pooled into three shared buffers (segment index values are ABSOLUTE into
    // the pool, so every curve BLAS and the shading kernels read the same
    // pool with only a per-set segment base). One curve BLAS per set; one TLAS
    // instance per set, appended AFTER the triangle instances — a curve hit's
    // `instance_id - curveInstanceBase` is its set index. Kernels that trace
    // the TLAS switch to a `kRTCurvesEnabled` function-constant variant when
    // curves are present (the intersector's geometry-type contract must match
    // the AS content), so curve-free scenes keep the exact pipelines they had.
    private var rtCurveSets: [IlluminatoramaCurveSet] = []
    private var rtCurveSyncVersion: Int = 0
    private var rtCurveBLASList: [MTLAccelerationStructure] = []
    private var rtCurvePoolPoints: MTLBuffer?     // packed float3 (stride 12)
    private var rtCurvePoolRadii: MTLBuffer?      // float
    private var rtCurvePoolSegments: MTLBuffer?   // uint32, absolute into the pool
    private var rtCurvePoolPointCount: Int = 0
    private var rtCurveSetSegmentBase: [Int] = [] // per set: first segment in the pool
    private var rtCurveSetDataBuffer: MTLBuffer?  // RTCurveSetData per set
    private var rtCurveInstanceCount: Int = 0     // sets in the CURRENT TLAS topology
    private var rtTLASCurvePipeline: MTLComputePipelineState?
    private var surfCacheTLASCurvePipeline: MTLComputePipelineState?
    /// Mirror of `RTCurveSetData` in IlluminatoramaRTInstanced.metal (112 B,
    /// SIMD4-aligned — no SIMD3 trap). m0..m3 = the set's object→world matrix
    /// columns; meta.x = the set's first segment index in the pooled buffers.
    private struct RTCurveSetData {
        var m0: SIMD4<Float>; var m1: SIMD4<Float>; var m2: SIMD4<Float>; var m3: SIMD4<Float>
        var albedoRoughness: SIMD4<Float>
        var emissionPad: SIMD4<Float>
        var meta: SIMD4<UInt32>
    }
    // ── Curve primitives on the SOUP path (#60 item 7, increment 2) ──
    // Forest et al. are native soup scenes (`setRTGeometry` → primitive AS,
    // `encodeRTLightingPass`), not the TLAS path. The soup AS gains a SECOND
    // geometry descriptor (a curve geometry) alongside the triangle soup; a hit
    // is a curve iff `res.type == curve` (single set, identity-placed, so no
    // instance_id). The pooled rest points are re-displaced per frame by the
    // wind kernel and the AS is refit, so the RT wood sways with the raster tubes.
    private var rtSoupCurveSet: IlluminatoramaCurveSet?
    private var rtSoupCurveSyncVersion: Int = 0
    private var rtSoupCurvesActive: Bool = false
    private var rtSoupCurveRestPoints: MTLBuffer?   // packed float3, un-swayed (kernel input)
    private var rtSoupCurvePoolPoints: MTLBuffer?   // packed float3, AS-referenced (kernel output / rest copy)
    private var rtSoupCurveRadii: MTLBuffer?        // float
    private var rtSoupCurveSegments: MTLBuffer?     // uint32 (relative — single geometry descriptor)
    private var rtSoupCurveWindAttr: MTLBuffer?     // float4 per point (empty set ⇒ nil ⇒ rigid)
    private var rtSoupCurveSetDataBuffer: MTLBuffer? // single RTCurveSetData
    private var rtSoupCurvePointCount: Int = 0
    private var rtSoupCurveSegmentCount: Int = 0
    /// SEPARATE curve-only primitive AS (#60 item 7 incr. 2). Metal forbids mixing
    /// triangle + curve geometry descriptors in ONE primitive AS ("must be the
    /// same type"), so curves live in their own AS and the kernel traces BOTH
    /// (any-hit occludes; nearest-hit compares distance). `.refit` usage for sway.
    private var rtSoupCurveAccel: MTLAccelerationStructure?
    private var rtSoupRefitScratch: MTLBuffer?
    private var rtSoupCurvePipeline: MTLComputePipelineState?
    private var surfCacheSoupCurvePipeline: MTLComputePipelineState?
    private var curveWindDisplacePipeline: MTLComputePipelineState?
    /// Mirror of `CurveWindUniforms` in IlluminatoramaRT.metal (16 B).
    private struct CurveWindUniforms {
        var time: Float; var windStrength: Float; var windHeading: Float; var pointCount: UInt32
    }
    /// Freeze curves at rest pose (A/B for the sway refit). `VIZ_ILLUMI_NO_CURVE_REFIT=1`.
    private let noCurveRefit = ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_CURVE_REFIT"] == "1"
    private var curveDbgPrinted = false   // VIZ_CURVE_DEBUG one-shot
    private let rtInstUniformBuffer: MTLBuffer
    private var rtInstFrameSeed: UInt32 = 0
    /// True after a successful TLAS (re)build/refit this frame — the RT pass
    /// then traces the instance AS instead of the primitive-AS soup.
    private var rtTLASActive: Bool = false

    // ── Phase 4.37a — RT hang-proof guard ────────────────────────────────
    // A scene whose geometry is GPU-fed (regenerated `IlluminatoramaMesh`
    // identities each frame) or whose mesh set/topology otherwise changes
    // every frame makes `rtTopologyHash()` differ every frame → `updateRTAccel`
    // takes the REBUILD branch each frame (new per-mesh BLAS builds, each with
    // a `waitUntilCompleted`, + a TLAS rebuild) → the main thread hangs and a
    // faulted AS traces to magenta. The descriptor-driven per-scene RT opt-in
    // (Phase 4.36/4.37) runs RT in the live tick loop, so any flagged scene
    // could trip this. Two gates make RT fail safe (fall back to non-RT, never
    // hang): an UP-FRONT size cap, and a rebuild-THRASH counter. Once tripped,
    // `rtAutoDisabled` latches until the next scene attach (`resetRTGuard()`).
    private var rtAutoDisabled: Bool = false
    private var rtConsecutiveRebuilds: Int = 0
    /// Rebuild this many frames running ⇒ topology is thrashing ⇒ disable RT.
    /// Allows a few legitimate settle-frames at scene load before giving up.
    /// (Backup gate — the size cap below is the primary defence, because the
    /// thrash counter can't catch a SINGLE catastrophic rebuild that never
    /// returns.)
    private static let rtRebuildThrashLimit: Int = 6
    /// Scenes past ANY of these caps are too heavy for per-frame RT in the live
    /// tick loop (a single rebuild — thousands of BLAS + CPU normal readbacks +
    /// `waitUntilCompleted` — would stall for seconds-to-minutes). Checked UP
    /// FRONT, before the first build, so a heavy scene auto-disables instead of
    /// hanging. The triangle cap is the real cost predictor (catches both
    /// many-small-mesh and few-huge-mesh scenes); the instance / mesh-group
    /// caps catch BLAS/TLAS build fan-out. Tuned for "light enough that a
    /// per-frame rebuild stays sub-frame" — the room / a box scene pass; Forest
    /// (thousands of leaf cards) does not.
    private static let rtMaxInstancesForLiveRT: Int = 2048
    private static let rtMaxMeshGroupsForLiveRT: Int = 128
    private static let rtMaxTrianglesForLiveRT: Int = 150_000

    /// Surface-cache (P1c) caps for the TLAS path. The cache allocates one
    /// per-triangle micro-card over the INSTANCE-EXPANDED soup (count =
    /// Σ group.count · meshTris), which can dwarf the per-mesh `estTriangles`
    /// used to gate live RT — so it gets its own cap. Past it the TLAS still
    /// runs (RT lighting intact); only the cache read is skipped.
    private static let surfaceCacheMaxTrianglesTLAS: Int = 60_000
    private static let surfaceCachePerTriTileSize: Int = 6

    /// Reset the RT auto-disable guard. Called on every scene attach (via
    /// `IlluminatoramaOverlay.setExtractedSceneRT`) so a freshly-shown scene
    /// gets a clean chance at RT regardless of what the previous scene did.
    public func resetRTGuard() {
        rtAutoDisabled = false
        rtConsecutiveRebuilds = 0
    }

    // ── Depth-of-field state ─────────────────────────────────────────
    private struct DOFParams {
        var invProjection: simd_float4x4
        var focusDist: Float; var aperture: Float; var maxRadius: Float; var focusRange: Float
        var width: UInt32; var height: UInt32; var _p0: Float = 0; var _p1: Float = 0
    }
    private let dofPipeline: MTLComputePipelineState?
    private var dofOutputTexture: MTLTexture?

    // ── Volumetric shaft state ───────────────────────────────────────
    private struct VolUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var fogDensity: Float
        var sunDir: SIMD3<Float>; var scatterStrength: Float
        var sunColor: SIMD3<Float>; var anisotropy: Float
        var windowX: Float; var winY0: Float; var winY1: Float; var winZ0: Float
        var winZ1: Float; var maxDist: Float; var steps: UInt32; var frameSeed: UInt32
        var width: UInt32; var height: UInt32; var feather: Float; var isOutdoor: UInt32 = 0
    }
    private let volPipeline: MTLComputePipelineState?
    private let volUniformBuffer: MTLBuffer
    private var volFrameSeed: UInt32 = 0

    // ── In-view (perspective) cloud pass — OPT-IN ────────────────────
    // Mirror of the Metal `CloudInViewUniforms`. The cloud density/lighting/
    // burst params ride in the cloud renderer's own SkyUniforms buffer
    // (`cloudSkyUniforms`); this only adds the camera inverse-VP + eye so the
    // kernel can reconstruct world rays per pixel.
    private struct CloudInViewUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD4<Float>
    }
    private let cloudInViewPipeline: MTLComputePipelineState?
    private let cloudInViewUniformBuffer: MTLBuffer
    /// When true AND the three handles below are set, the in-view cloud pass
    /// composites crisp camera-ray clouds into the HDR target (issue #61). OFF
    /// by default → every scene + the dome/IBL path is byte-identical.
    public var inViewCloudsEnabled = false
    /// Baked noise volume + packed SkyUniforms + burst-light buffer, supplied by
    /// the scene's `VolumetricCloudRenderer` (reused, not re-packed).
    public var cloudNoiseTexture: MTLTexture?
    public var cloudSkyUniforms: MTLBuffer?
    public var cloudBurstLights: MTLBuffer?
    /// True for the current frame when the DOF pass actually ran (so bloom +
    /// tonemap read its output instead of the raw post-FX source).
    private var dofApplied: Bool = false

    // Phase 2.7 — ping-ponged instance buffers so the G-buffer vertex shader
    // can read both this frame's and last frame's per-instance model matrix
    // at the same `[[instance_id]]`, computing the prev clip-space position
    // for motion-vector reconstruction. `currentInstanceBuffer` /
    // `previousInstanceBuffer` resolve through `useBufferA`.
    //
    // ASSUMPTION (load-bearing): `instances[i]` this frame refers to the
    // same logical instance as `instances[i]` last frame. The vertex shader
    // reads `prevInstances[iid]` to build the previous-frame world position;
    // if the host shuffles the array (or inserts in the middle) between
    // frames, motion vectors will be wrong and TAA will smear wildly.
    //
    // Today this is fine: IlluminatoramaLab pushes a stable [ground, box,
    // sphere] every frame, and any forthcoming scene extractor should
    // similarly produce a stable index per logical SCNNode. The agent's
    // safety-net is `ensureBufferCapacity` setting `taaNeedsFirstFrame =
    // true` on growth — that catches count CHANGES but not REORDERS.
    //
    // When DDGI / dynamic scene extraction lands, the right fix is a
    // stable per-instance ID (UUID in the host-side struct, hashed into a
    // uint32 in the GPU struct); the renderer can then sort `prevInstances`
    // by ID at the start of each frame so `prevInstances[iid]` lines up
    // with `instances[iid].id`. Don't ship that scene extractor without
    // also wiring this through.
    private var instanceBufferA: MTLBuffer
    private var instanceBufferB: MTLBuffer
    private var instanceCapacity: Int
    private var useBufferA: Bool = true
    // ── Per-frame input buffers — ring-buffered (maxFramesInFlight) ─────────────
    // These hold the frame's uniforms + lights + impostor params. They are
    // `memcpy`'d in place at the top of every `render()` and bound (read-only)
    // by that frame's GPU passes. At `maxFramesInFlight == 1` an in-place
    // single buffer was safe. At 2 the CPU encodes frame N+1 (overwriting the
    // buffer) while the GPU still reads it for frame N → torn lighting (the
    // blocky floor-reflection band on the eggs) and uneven cadence (POV
    // jitter, because the camera matrices in the frame uniforms get half-stomped
    // mid-read). Fix: cycle one slot per accepted frame via `frameRingIndex`,
    // the same scheme the instance buffer already uses (`useBufferA`) and the
    // LDR present pool uses (`acquireWriteTarget`). Each property below resolves
    // to the current ring slot so the bind sites are unchanged.
    private var pointLightRing: [MTLBuffer]
    private var pointLightBuffer: MTLBuffer { pointLightRing[frameRingIndex] }
    private var pointLightCapacity: Int
    private var spotLightRing: [MTLBuffer]
    private var spotLightBuffer: MTLBuffer { spotLightRing[frameRingIndex] }
    private var spotLightCapacity: Int
    private var areaLightRing: [MTLBuffer]
    private var areaLightBuffer: MTLBuffer { areaLightRing[frameRingIndex] }
    private var areaLightCapacity: Int
    private var extraDirectionalRing: [MTLBuffer]
    private var extraDirectionalBuffer: MTLBuffer { extraDirectionalRing[frameRingIndex] }
    private var extraDirectionalCapacity: Int
    private var frameUniformRing: [MTLBuffer]
    private var frameUniformBuffer: MTLBuffer { frameUniformRing[frameRingIndex] }

    /// Which ring slot the CURRENT frame writes + binds. Advanced by exactly one
    /// per ACCEPTED `render()` (after the in-flight semaphore admits the frame),
    /// so two consecutive in-flight frames always use different slots. Sized to
    /// `maxFramesInFlight`; the GPU's max concurrency is bounded by the same
    /// semaphore, so a slot is never both written (CPU, frame N+1) and read
    /// (GPU, frame N) at once.
    private var frameRingIndex: Int = 0

    private var currentInstanceBuffer: MTLBuffer { useBufferA ? instanceBufferA : instanceBufferB }
    private var previousInstanceBuffer: MTLBuffer { useBufferA ? instanceBufferB : instanceBufferA }

    /// Internal rendering size. All compute kernels (G-buffer rasterisation,
    /// SSAO, lighting, SSR, TAA, bloom mid-textures) dispatch over this
    /// resolution. Equals `outputWidth × internalRenderScale` (rounded).
    private(set) public var width: Int
    private(set) public var height: Int

    /// Final output size — drives the `outputTexture` allocation and the
    /// tonemap kernel's dispatch. The tonemap is the ONE kernel that
    /// bridges from internal to output: it samples the internal HDR /
    /// bloom textures with a downsample filter and writes one output
    /// pixel per thread.
    private(set) public var outputWidth: Int
    private(set) public var outputHeight: Int

    // ── Init ──────────────────────────────────────────────────────────────────

    public init(engine: SimEngine, width: Int, height: Int, camera: IlluminatoramaCamera) throws {
        self.engine = engine
        self.device = engine.device
        self.commandQueue = engine.commandQueue
        // `width` / `height` are the OUTPUT (host-visible) target size; the
        // internal pipeline runs at `output × initialScale`. The init scale
        // mirrors the default `internalRenderScale` field above — we can't
        // read `self.internalRenderScale` before its memberwise default
        // applies, so duplicate the literal here.
        let initialScale: Float = 1.5
        self.outputWidth  = max(1, width)
        self.outputHeight = max(1, height)
        let (iw, ih) = Self.internalDims(outputW: self.outputWidth,
                                          outputH: self.outputHeight,
                                          scale: initialScale)
        self.width = iw
        self.height = ih
        self.camera = camera

        guard let library = engine.library else {
            throw IlluminatoramaError.libraryMissing
        }

        // ── Pipelines ────────────────────────────────────────────────────
        // G-buffer render pipeline (vertex + fragment).
        let pdesc = MTLRenderPipelineDescriptor()
        pdesc.label = "Illuminatorama.gbuffer"
        pdesc.vertexFunction = library.makeFunction(name: "illumi_vs")
        pdesc.fragmentFunction = library.makeFunction(name: "illumi_fs")
        pdesc.colorAttachments[0].pixelFormat = .rgba16Float
        pdesc.colorAttachments[1].pixelFormat = .rgba16Float
        pdesc.colorAttachments[2].pixelFormat = .rgba16Float
        // Phase 2.7 — velocity attachment (screen-space motion in UV space).
        pdesc.colorAttachments[3].pixelFormat = .rg16Float
        pdesc.depthAttachmentPixelFormat = .depth32Float
        // No vertex descriptor — the shader reads from a device-buffer of
        // Vertex via [[vertex_id]]. Simpler than wiring a vertex descriptor
        // and matches the Swift/Metal struct mirror more directly.
        self.gbufferPipeline = try device.makeRenderPipelineState(descriptor: pdesc)

        // Perfect analytic superquadric impostor — same G-buffer attachment
        // formats so it renders into the SAME G-buffer pass/encoder (no extra
        // pass, no extra clear). The fragment writes analytic depth via
        // `[[depth(less)]]`. Optional: if the functions are missing the impostor
        // simply doesn't draw (the proxy still covers RT), so don't fail init.
        if let sqVS = library.makeFunction(name: "illumi_superquadric_impostor_vs"),
           let sqFS = library.makeFunction(name: "illumi_superquadric_impostor_fs") {
            let sqdesc = MTLRenderPipelineDescriptor()
            sqdesc.label = "Illuminatorama.superquadricImpostor"
            sqdesc.vertexFunction = sqVS
            sqdesc.fragmentFunction = sqFS
            sqdesc.colorAttachments[0].pixelFormat = .rgba16Float
            sqdesc.colorAttachments[1].pixelFormat = .rgba16Float
            sqdesc.colorAttachments[2].pixelFormat = .rgba16Float
            sqdesc.colorAttachments[3].pixelFormat = .rg16Float
            sqdesc.depthAttachmentPixelFormat = .depth32Float
            self.superquadricImpostorPipeline =
                try? device.makeRenderPipelineState(descriptor: sqdesc)
        }

        // Phase 2.5 — depth-only shadow pipeline. Same vertex layout as the
        // G-buffer pass but no fragment shader and no colour attachments;
        // depth is written into one slice of the cascade array per pass.
        let sdesc = MTLRenderPipelineDescriptor()
        sdesc.label = "Illuminatorama.shadow"
        sdesc.vertexFunction = library.makeFunction(name: "illumi_shadow_vs")
        sdesc.fragmentFunction = nil
        sdesc.depthAttachmentPixelFormat = .depth32Float
        self.shadowPipeline = try device.makeRenderPipelineState(descriptor: sdesc)

        let ddesc = MTLDepthStencilDescriptor()
        ddesc.depthCompareFunction = .less
        ddesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: ddesc) else {
            throw IlluminatoramaError.pipelineCreationFailed("depth-stencil state")
        }
        self.depthState = ds

        // ── AAA glass pipelines (#60, forward, src-alpha blended) ──────────
        // Two fragment variants share one vertex stage (`illumi_glass_rt_vs`) and
        // one blend config: `illumi_glass_rt_fs` traces the scene TLAS for true
        // refraction/reflection (returns opaque alpha 1 → nearest pane replaces),
        // `illumi_glass_fallback_fs` is the Fresnel+sky fallback (alpha < 1 →
        // blends). Drawn into the HDR composite after lighting, depth-tested LE but
        // no depth write so opaque geometry in front occludes the glass.
        // Local-capture closure (a nested `func` would implicitly capture `self`
        // before full init); `dev`/`library` are locals so this is init-safe.
        let dev = device
        let makeGlassPipeline: (String, String) -> MTLRenderPipelineState? = { fragName, label in
            guard let gvs = library.makeFunction(name: "illumi_glass_rt_vs"),
                  let gfs = library.makeFunction(name: fragName) else { return nil }
            let gdesc = MTLRenderPipelineDescriptor()
            gdesc.label = label
            gdesc.vertexFunction = gvs
            gdesc.fragmentFunction = gfs
            gdesc.colorAttachments[0].pixelFormat = .rgba16Float   // hdrCompositeTexture
            gdesc.colorAttachments[0].isBlendingEnabled = true
            gdesc.colorAttachments[0].rgbBlendOperation = .add
            gdesc.colorAttachments[0].alphaBlendOperation = .add
            gdesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            gdesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            gdesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            gdesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            gdesc.depthAttachmentPixelFormat = .depth32Float
            return try? dev.makeRenderPipelineState(descriptor: gdesc)
        }
        // The RT fragment shader needs a device that supports ray tracing; build
        // it only there (the function references `instance_acceleration_structure`).
        self.glassRTPipeline = device.supportsRaytracing
            ? makeGlassPipeline("illumi_glass_rt_fs", "Illuminatorama.glass.rt") : nil
        self.glassFallbackPipeline = makeGlassPipeline("illumi_glass_fallback_fs",
                                                       "Illuminatorama.glass.fallback")
        let gds = MTLDepthStencilDescriptor()
        gds.depthCompareFunction = .lessEqual
        gds.isDepthWriteEnabled = false
        self.glassDepthState = device.makeDepthStencilState(descriptor: gds)

        // Compute pipelines via the shared cache so we don't recompile across
        // scene rebuilds.
        let cache = engine.pipelineCache
        // illumi_lighting is function-constant–specialized (WWDC23 #10127). Build the
        // variant matching the renderer's current feature flags; other variants are
        // compiled lazily by `currentLightingPipeline()` when a flag toggles, and this
        // init-time variant doubles as the fallback if a later build ever fails.
        // #60 task 4 — warm the init compute pipelines concurrently (WWDC23
        // #10127 async compilation applied to startup) so a cold pipeline cache
        // compiles in parallel instead of 26 serial blocking calls. The guards
        // below then hit the warm cache. (`illumi_lighting` uses the specialized
        // variant path, so it's compiled by the guard, not here.)
        cache.precompile([
            "illumi_bloom_blur_h", "illumi_bloom_blur_v", "illumi_bloom_threshold",
            "illumi_ddgi_trace", "illumi_ddgi_update_depth", "illumi_ddgi_update_irradiance",
            "illumi_dfg_bake", "illumi_exposure_estimate", "illumi_irradiance_bake",
            "illumi_mesh_build_adjacency", "illumi_mesh_synth", "illumi_particles_step",
            "illumi_prefilter_bake", "illumi_repack_pos_norm", "illumi_rt_denoise",
            "illumi_rt_gi_temporal", "illumi_ssao", "illumi_ssao_spatial",
            "illumi_ssao_temporal", "illumi_ssr_composite", "illumi_ssr_gather",
            "illumi_ssr_temporal", "illumi_surfcache_atrous",
            "illumi_surfcache_reframe_chart",
            "illumi_surfcache_reframe_deform", "illumi_surfcache_reframe_tri",
            "illumi_svgf_atrous", "illumi_svgf_variance", "illumi_taa_resolve",
        ], device: device)

        let (initLightingKey, initLightingConstants) = Self.lightingFeatureConstants(
            ibl: iblEnabled, shadow: shadowsEnabled, dfg: dfgLUTEnabled,
            ddgi: ddgiEnabled, ddgiIrrCache: ddgiIrrCacheEnabled)
        guard let lighting = cache.pipelineState(name: "illumi_lighting", device: device,
                                                 constants: initLightingConstants,
                                                 variantKey: initLightingKey) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_lighting")
        }
        guard let ssao = cache.pipelineState(name: "illumi_ssao", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssao")
        }
        guard let ssaoSpatial = cache.pipelineState(name: "illumi_ssao_spatial", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssao_spatial")
        }
        guard let ssaoTemporal = cache.pipelineState(name: "illumi_ssao_temporal", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssao_temporal")
        }
        guard let ssrGather = cache.pipelineState(name: "illumi_ssr_gather", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssr_gather")
        }
        guard let ssrTemporal = cache.pipelineState(name: "illumi_ssr_temporal", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssr_temporal")
        }
        guard let ssrComposite = cache.pipelineState(name: "illumi_ssr_composite", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssr_composite")
        }
        guard let rtDenoise = cache.pipelineState(name: "illumi_rt_denoise", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_rt_denoise")
        }
        guard let rtGITemporal = cache.pipelineState(name: "illumi_rt_gi_temporal", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_rt_gi_temporal")
        }
        guard let svgfVar = cache.pipelineState(name: "illumi_svgf_variance", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_svgf_variance")
        }
        guard let svgfAtrous = cache.pipelineState(name: "illumi_svgf_atrous", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_svgf_atrous")
        }
        guard let irrBake = cache.pipelineState(name: "illumi_irradiance_bake", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_irradiance_bake")
        }
        guard let prefBake = cache.pipelineState(name: "illumi_prefilter_bake", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_prefilter_bake")
        }
        guard let dfgBake = cache.pipelineState(name: "illumi_dfg_bake", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_dfg_bake")
        }
        guard let taa = cache.pipelineState(name: "illumi_taa_resolve", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_taa_resolve")
        }
        guard let threshold = cache.pipelineState(name: "illumi_bloom_threshold", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_bloom_threshold")
        }
        guard let blurH = cache.pipelineState(name: "illumi_bloom_blur_h", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_bloom_blur_h")
        }
        guard let blurV = cache.pipelineState(name: "illumi_bloom_blur_v", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_bloom_blur_v")
        }
        // Phase 4.28 — tonemap render pipeline (fullscreen triangle → fragment).
        // Outputs into the bgra8Unorm `outputTexture` via a render pass so the
        // write is visible to SceneKit's cross-queue background sample.
        let tmDesc = MTLRenderPipelineDescriptor()
        tmDesc.label = "Illuminatorama.tonemap"
        tmDesc.vertexFunction = library.makeFunction(name: "illumi_tonemap_vs")
        tmDesc.fragmentFunction = library.makeFunction(name: "illumi_tonemap_fs")
        tmDesc.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        let tonemap = try device.makeRenderPipelineState(descriptor: tmDesc)
        // Phase 4.21 — auto-exposure estimator.
        guard let expoEst = cache.pipelineState(name: "illumi_exposure_estimate", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_exposure_estimate")
        }
        // Phase 3.1 — DDGI kernels.
        guard let ddgiTrace = cache.pipelineState(name: "illumi_ddgi_trace", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ddgi_trace")
        }
        guard let ddgiUpdateIrr = cache.pipelineState(name: "illumi_ddgi_update_irradiance", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ddgi_update_irradiance")
        }
        guard let ddgiUpdateDepth = cache.pipelineState(name: "illumi_ddgi_update_depth", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ddgi_update_depth")
        }
        self.lightingPipeline = lighting
        self.ssaoPipeline = ssao
        self.ssaoSpatialPipeline  = ssaoSpatial
        self.ssaoTemporalPipeline = ssaoTemporal
        self.ssrGatherPipeline    = ssrGather
        self.ssrTemporalPipeline  = ssrTemporal
        self.ssrCompositePipeline = ssrComposite
        self.rtDenoisePipeline    = rtDenoise
        self.rtGITemporalPipeline = rtGITemporal
        self.svgfVariancePipeline = svgfVar
        self.svgfAtrousPipeline   = svgfAtrous
        self.irradianceBakePipeline = irrBake
        self.prefilterBakePipeline = prefBake
        self.dfgBakePipeline = dfgBake
        self.taaResolvePipeline = taa
        self.bloomThresholdPipeline = threshold
        self.bloomBlurHPipeline = blurH
        self.bloomBlurVPipeline = blurV
        self.tonemapPipeline = tonemap
        self.exposureEstimatePipeline = expoEst

        // ── Hardware RT pipeline (optional) ──────────────────────────
        // Only build on RT-capable hardware; the intersector kernel can't be
        // compiled to a pipeline on a device without ray-tracing support. A
        // nil pipeline leaves the RT pass disabled — the renderer falls back
        // to the deferred sun + cascaded shadows.
        // The soup `illumi_rt_lighting` + `illumi_surfcache_update` kernels now
        // DECLARE the kRTCurvesEnabled / kSCCurvesEnabled function constant (#60
        // item 7 incr. 2), so — like the TLAS kernels — their base (false) variant
        // MUST be built via makeFunction(name:constantValues:) (a plain
        // makeFunction(name:) aborts at pipeline validation for any constant-
        // declaring function). `curvesOff` is reused by the surfcache/TLAS builds
        // below.
        let curvesOff = MTLFunctionConstantValues()
        var curvesOffValue = false
        curvesOff.setConstantValue(&curvesOffValue, type: .bool,
                                   index: Self.curveFunctionConstantIndex)
        var builtRTPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let rtFn = try? library.makeFunction(name: "illumi_rt_lighting",
                                                constantValues: curvesOff) {
            builtRTPipeline = try? device.makeComputePipelineState(function: rtFn) // gpu-ok: one-time init, RT-capability-gated optional pipeline
        }
        self.rtPipeline = builtRTPipeline
        self.rtSupported = (builtRTPipeline != nil)
        guard let rtUB = device.makeBuffer(length: MemoryLayout<RTUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtUniforms")
        }
        rtUB.label = "Illuminatorama.rt.uniforms"
        self.rtUniformBuffer = rtUB
        guard let rtDnUB = device.makeBuffer(length: MemoryLayout<RTDenoiseUniforms>.stride,
                                             options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtDenoiseUniforms")
        }
        rtDnUB.label = "Illuminatorama.rt.denoiseUniforms"
        self.rtDenoiseUniformBuffer = rtDnUB
        guard let rtGiUB = device.makeBuffer(length: MemoryLayout<RTGITemporalUniforms>.stride,
                                             options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtGITemporalUniforms")
        }
        rtGiUB.label = "Illuminatorama.rt.giTemporalUniforms"
        self.rtGITemporalUniformBuffer = rtGiUB

        // ── Surface-cache update pipeline (RT-gated) ─────────────────
        // Both kernels declare the kSCCurvesEnabled function constant (#60 item 7),
        // so both MUST be specialized via constantValues even for the base (false)
        // variant — Metal refuses a plain makeFunction(name:) pipeline for any
        // constant-declaring function. `curvesOff` was built above.
        var builtSurfCachePipeline: MTLComputePipelineState? = nil
        var builtSurfCacheTLASPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let scFn = try? library.makeFunction(name: "illumi_surfcache_update",
                                                constantValues: curvesOff) {
            builtSurfCachePipeline = try? device.makeComputePipelineState(function: scFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        if device.supportsRaytracing,
           let scTLASFn = try? library.makeFunction(name: "illumi_surfcache_update_tlas",
                                                    constantValues: curvesOff) {
            builtSurfCacheTLASPipeline = try? device.makeComputePipelineState(function: scTLASFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        self.surfCachePipeline = builtSurfCachePipeline
        self.surfCacheTLASPipeline = builtSurfCacheTLASPipeline
        guard let scUB = device.makeBuffer(length: MemoryLayout<SurfCacheUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("surfCacheUniforms")
        }
        scUB.label = "Illuminatorama.surfcache.uniforms"
        self.surfCacheUniformBuffer = scUB
        // #60 item 1 + Phase D — GPU-side incremental-invalidation kernels (plain
        // compute, no RT dependency; nil leaves the CPU re-frame path in charge).
        self.surfReframeTriPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_tri", device: device)
        self.surfReframeChartPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_chart", device: device)
        self.surfReframeDeformPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_deform", device: device)
        // Phase 5 / B1 — cache-domain denoiser (plain compute, no RT dependency;
        // nil leaves consumers reading the raw atlas).
        self.surfAtrousPipeline =
            cache.pipelineState(name: "illumi_surfcache_atrous", device: device)
        // Curve sway displacement (#60 item 7 incr. 2) — plain compute, no RT
        // dependency; nil leaves curves rigid (no per-frame re-displace).
        self.curveWindDisplacePipeline =
            cache.pipelineState(name: "illumi_curve_wind_displace", device: device)
        surfGPUDiffStatsBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * 4, options: .storageModeShared)
        surfGPUDiffStatsBuffer?.label = "Illuminatorama.surfcache.gpuDiffStats"
        if let sb = surfGPUDiffStatsBuffer { memset(sb.contents(), 0, sb.length) }

        // ── Instanced-RT (TLAS) pipeline ─────────────────────────────
        // Same constant-declaring caveat as the surfcache TLAS kernel above:
        // build the explicit kRTCurvesEnabled=false base variant.
        var builtTLASPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let tlasFn = try? library.makeFunction(name: "illumi_rt_lighting_tlas",
                                                  constantValues: curvesOff) {
            builtTLASPipeline = try? device.makeComputePipelineState(function: tlasFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        self.rtTLASPipeline = builtTLASPipeline
        guard let riUB = device.makeBuffer(length: MemoryLayout<RTInstUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtInstUniforms")
        }
        riUB.label = "Illuminatorama.rt.instUniforms"
        self.rtInstUniformBuffer = riUB

        // ── Depth-of-field pipeline ──────────────────────────────────
        if let dofFn = library.makeFunction(name: "illumi_dof") {
            self.dofPipeline = try? device.makeComputePipelineState(function: dofFn) // gpu-ok: one-time init, optional DoF pipeline
        } else {
            self.dofPipeline = nil
        }

        // ── Volumetric shaft pipeline ────────────────────────────────
        if let volFn = library.makeFunction(name: "illumi_volumetric") {
            self.volPipeline = try? device.makeComputePipelineState(function: volFn) // gpu-ok: one-time init, optional volumetric pipeline
        } else {
            self.volPipeline = nil
        }
        guard let volUB = device.makeBuffer(length: MemoryLayout<VolUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("volUniforms")
        }
        volUB.label = "Illuminatorama.vol.uniforms"
        self.volUniformBuffer = volUB

        // ── In-view cloud pipeline (opt-in; nil-safe → pass no-ops) ──
        if let cloudFn = library.makeFunction(name: "illumi_cloud_inview") {
            self.cloudInViewPipeline = try? device.makeComputePipelineState(function: cloudFn) // gpu-ok: one-time init, optional in-view cloud pipeline
        } else {
            self.cloudInViewPipeline = nil
        }
        guard let cloudUB = device.makeBuffer(length: MemoryLayout<CloudInViewUniforms>.stride,
                                              options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("cloudInViewUniforms")
        }
        cloudUB.label = "Illuminatorama.cloudInView.uniforms"
        self.cloudInViewUniformBuffer = cloudUB
        // ExposureState is 16 bytes (4 floats). Storage `.shared` so
        // the renderer can seed it once and (if needed) read it for
        // diagnostics. The kernel updates it in place each frame.
        guard let expoBuf = device.makeBuffer(
            length: 16, options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("exposureState") }
        expoBuf.label = "Illuminatorama.exposureState"
        // Initial state: previous-frame target log lum = -2.0 (mid-grey-ish);
        // smoothed exposure = 1.0 (the static fallback); new target same;
        // dt = 1/60 s (host will overwrite per frame).
        let initialState: [Float] = [-2.0, 1.0, -2.0, 1.0 / 60.0]
        memcpy(expoBuf.contents(), initialState,
               MemoryLayout<Float>.stride * initialState.count)
        self.exposureBuffer = expoBuf
        self.ddgiTracePipeline = ddgiTrace
        self.ddgiUpdateIrrPipeline = ddgiUpdateIrr
        self.ddgiUpdateDepthPipeline = ddgiUpdateDepth

        // ── Phase 4.11 — particle pipelines ─────────────────────────────
        guard let partStep = cache.pipelineState(name: "illumi_particles_step",
                                                  device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_particles_step")
        }
        self.particleStepPipeline = partStep

        // ── Phase 4.13a — DynamicMesh repack kernel ──────────────────
        guard let repack = cache.pipelineState(name: "illumi_repack_pos_norm",
                                                device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_repack_pos_norm")
        }
        self.repackPosNormPipeline = repack

        // ── Phase 4.21 — GPU mesh normal/tangent synthesis ───────────
        guard let meshAdj = cache.pipelineState(name: "illumi_mesh_build_adjacency",
                                                 device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_mesh_build_adjacency")
        }
        self.meshBuildAdjacencyPipeline = meshAdj
        guard let meshSynth = cache.pipelineState(name: "illumi_mesh_synth",
                                                  device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_mesh_synth")
        }
        self.meshSynthPipeline = meshSynth

        let partDrawDesc = MTLRenderPipelineDescriptor()
        partDrawDesc.label = "Illuminatorama.particles.draw"
        partDrawDesc.vertexFunction   = library.makeFunction(name: "illumi_particles_vs")
        partDrawDesc.fragmentFunction = library.makeFunction(name: "illumi_particles_fs")
        partDrawDesc.colorAttachments[0].pixelFormat = .rgba16Float
        // HDR additive blend. `one + one` lets bright particles accumulate
        // into the composite; bloom & tonemap then process them with the
        // same pipeline as direct + indirect lighting.
        partDrawDesc.colorAttachments[0].isBlendingEnabled = true
        partDrawDesc.colorAttachments[0].rgbBlendOperation = .add
        partDrawDesc.colorAttachments[0].alphaBlendOperation = .add
        partDrawDesc.colorAttachments[0].sourceRGBBlendFactor = .one
        partDrawDesc.colorAttachments[0].destinationRGBBlendFactor = .one
        partDrawDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        partDrawDesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        // Depth attachment must match the G-buffer's format so we can
        // bind that texture as a read-only depth target and have opaque
        // scene geometry occlude particles behind it.
        partDrawDesc.depthAttachmentPixelFormat = .depth32Float
        self.particleDrawPipeline = try device.makeRenderPipelineState(descriptor: partDrawDesc)

        // Phase 4.23 — host-owned point-sprite pipeline. Mirrors the
        // particles pipeline's HDR additive blend but uses MTLPrimitiveType
        // .point + a stride-parameterised vertex shader, so callers can
        // bind their own MTLBuffer of (position, color) without first
        // copying into Illuminatorama's particle layout.
        let extPartDesc = MTLRenderPipelineDescriptor()
        extPartDesc.label = "Illuminatorama.extParticles.draw"
        extPartDesc.vertexFunction   = library.makeFunction(name: "illumi_extparticle_vs")
        extPartDesc.fragmentFunction = library.makeFunction(name: "illumi_extparticle_fs")
        extPartDesc.colorAttachments[0].pixelFormat = .rgba16Float
        extPartDesc.colorAttachments[0].isBlendingEnabled = true
        extPartDesc.colorAttachments[0].rgbBlendOperation = .add
        extPartDesc.colorAttachments[0].alphaBlendOperation = .add
        extPartDesc.colorAttachments[0].sourceRGBBlendFactor = .one
        extPartDesc.colorAttachments[0].destinationRGBBlendFactor = .one
        extPartDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        extPartDesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        extPartDesc.depthAttachmentPixelFormat = .depth32Float
        self.extParticleDrawPipeline = try device.makeRenderPipelineState(descriptor: extPartDesc)

        // Phase 4.44b — velocity-aligned streak pipeline. Uses the same
        // `particleStreakVS/FS` shaders as `ParticleStreakPass` but targets
        // the HDR composite directly so streaks participate in bloom + tonemap.
        // Optional: if the shaders are absent the pipeline stays nil and streak
        // draws are silently skipped — logged once here, not per-frame.
        if let svs = library.makeFunction(name: "particleStreakVS"),
           let sfs = library.makeFunction(name: "particleStreakFS") {
            let streakVD = MTLVertexDescriptor()
            // Buffer 0: FWStreakVertex (stride 32) — float3 position @ offset 0,
            //           float4 color (w = stretchFactor) @ offset 16.
            streakVD.attributes[0].format      = .float3
            streakVD.attributes[0].offset      = 0
            streakVD.attributes[0].bufferIndex = 0
            streakVD.attributes[1].format      = .float4
            streakVD.attributes[1].offset      = 16
            streakVD.attributes[1].bufferIndex = 0
            streakVD.layouts[0].stride         = 32
            streakVD.layouts[0].stepFunction   = .perVertex
            // Buffer 1: float2 UV (stride 8).
            streakVD.attributes[2].format      = .float2
            streakVD.attributes[2].offset      = 0
            streakVD.attributes[2].bufferIndex = 1
            streakVD.layouts[1].stride         = 8
            streakVD.layouts[1].stepFunction   = .perVertex
            let sd = MTLRenderPipelineDescriptor()
            sd.label                                                  = "Illuminatorama.streaks"
            sd.vertexFunction                                         = svs
            sd.fragmentFunction                                       = sfs
            sd.vertexDescriptor                                       = streakVD
            sd.colorAttachments[0].pixelFormat                       = .rgba16Float
            sd.colorAttachments[0].isBlendingEnabled                 = true
            sd.colorAttachments[0].rgbBlendOperation                 = .add
            sd.colorAttachments[0].sourceRGBBlendFactor              = .one
            sd.colorAttachments[0].destinationRGBBlendFactor         = .one
            sd.colorAttachments[0].alphaBlendOperation               = .add
            sd.colorAttachments[0].sourceAlphaBlendFactor            = .one
            sd.colorAttachments[0].destinationAlphaBlendFactor       = .one
            sd.depthAttachmentPixelFormat                            = .depth32Float
            self.streakPipeline = try? device.makeRenderPipelineState(descriptor: sd)
            if self.streakPipeline == nil {
                Self.log.error("Illuminatorama: streakPipeline compilation failed — streak draws disabled")
            }
        } else {
            Self.log.warning("Illuminatorama: particleStreakVS/FS not found in library — streak draws disabled")
        }

        // Depth state: less-equal compare, NO write. Lets each particle
        // get tested against G-buffer depth (so the cube/sphere/ground
        // hide particles behind them) while leaving the depth buffer
        // pristine for any subsequent passes.
        let partDepthDesc = MTLDepthStencilDescriptor()
        partDepthDesc.depthCompareFunction = .lessEqual
        partDepthDesc.isDepthWriteEnabled = false
        guard let pds = device.makeDepthStencilState(descriptor: partDepthDesc) else {
            throw IlluminatoramaError.pipelineCreationFailed("particleDepthState")
        }
        self.particleDepthState = pds

        // ── Phase 3 cubemaps + dummy sky ────────────────────────────────
        let (irrCube, preCube, preViews) = try Self.makeIBLCubes(device: device)
        self.irradianceCube = irrCube
        self.prefilteredCube = preCube
        self.prefilteredMipViews = preViews
        self.dummySkyTexture = Self.makeDummySky(device: device)

        // ── Phase 3.2 DFG LUT ───────────────────────────────────────────
        // Allocate and bake immediately — the LUT is view-independent so it
        // never needs to be rebuilt. Bake CB is enqueued on the shared queue;
        // subsequent render CBs on the same queue are sequenced after it
        // automatically (no waitUntilCompleted needed).
        let dfgDesc = MTLTextureDescriptor()
        dfgDesc.textureType = .type2D
        dfgDesc.pixelFormat = .rg16Float
        dfgDesc.width  = Self.dfgLUTSize
        dfgDesc.height = Self.dfgLUTSize
        dfgDesc.usage  = [.shaderRead, .shaderWrite]
        dfgDesc.storageMode = .private
        guard let dfg = device.makeTexture(descriptor: dfgDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.dfgLUT")
        }
        dfg.label = "Illuminatorama.dfgLUT"
        self.dfgLUT = dfg

        // Issue #65 — identity colour-grade LUT (3D). Built on CPU and uploaded;
        // identity means the tonemap LUT sample returns the input unchanged, so
        // the default is an exact no-op until a scene calls `setColorLUT`.
        guard let lut = Self.makeIdentityColorLUT(device: device,
                                                  size: Self.colorLUTDefaultSize) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.colorLUT")
        }
        self.colorLUT = lut

        // #60 task 5 increment 2 — bake the LTC area-light specular LUTs once,
        // self-validated against brute-force MC of the GGX BRDF. Falls back to the
        // dfg LUT as a dummy binding (never sampled) + MRP specular if the bake
        // failed or didn't validate.
        if let ltc = IlluminatoramaLTC.makeLUTs(device: device) {
            self.ltcMatTexture = ltc.mat
            self.ltcMagTexture = ltc.mag
            self.ltcValidated = ltc.validated
        } else {
            self.ltcMatTexture = dfg
            self.ltcMagTexture = dfg
            self.ltcValidated = false
        }

        self.passTimer = IlluminatoramaPassTimer(device: device)

        if let cb  = commandQueue.makeCommandBuffer(),
           let enc = cb.makeComputeCommandEncoder() {
            cb.label  = "Illuminatorama.dfgBake"
            enc.label = "Illuminatorama.dfgBake"
            enc.setComputePipelineState(dfgBake)
            enc.setTexture(dfg, index: 0)
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

        // ── Phase 3.1 DDGI shared resources ─────────────────────────────
        // Uniform buffer — tiny, allocated once, written each frame.
        guard let ddgiUB = device.makeBuffer(
            length: MemoryLayout<IlluminatoramaDDGIUniforms>.stride,
            options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("ddgiUniforms") }
        ddgiUB.label = "Illuminatorama.ddgi.uniforms"
        self.ddgiUniformBuffer = ddgiUB
        // 1×1 rg16Float dummy bound to the depth atlas slot when DDGI is off
        // (always binding a real texture avoids Metal validation warnings).
        let dddDesc = MTLTextureDescriptor()
        dddDesc.textureType = .type2D
        dddDesc.pixelFormat = .rg16Float
        dddDesc.width = 1; dddDesc.height = 1
        dddDesc.usage = [.shaderRead, .shaderWrite]
        dddDesc.storageMode = .private
        guard let ddd = device.makeTexture(descriptor: dddDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("ddgiDummyDepthAtlas")
        }
        ddd.label = "Illuminatorama.ddgi.dummyDepth"
        self.ddgiDummyDepthAtlas = ddd

        // ── Phase 2.5 shadow map array ──────────────────────────────────
        self.shadowMap = try Self.makeShadowMap(device: device)
        self.spotShadowAtlas = try Self.makeSpotShadowAtlas(
            device: device,
            resolution: spotShadowMapResolution,
            capacity: spotShadowAtlasCapacity
        )

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

        // ── Targets ──────────────────────────────────────────────────────
        let t = try Self.makeTargets(device: device,
                                      internalW: self.width, internalH: self.height,
                                      outputW: self.outputWidth, outputH: self.outputHeight)
        self.gbufferAlbedoMet    = t.albedoMet
        self.gbufferNormalRgh    = t.normalRgh
        self.gbufferEmission     = t.emission
        self.depthTexture        = t.depth
        self.previousDepthTexture = try Self.makeDepthLike(device: device, t.depth)
        self.hdrTexture          = t.hdr
        self.aoTexture           = t.ao
        self.hdrCompositeTexture = t.hdrComposite
        self.rtDiffuseTexture    = t.rtDiffuse
        self.velocityTexture     = t.velocity
        self.historyA            = t.historyA
        self.historyB            = t.historyB
        self.bloomBrightHalf     = t.bloomBright
        self.bloomBlurHHalf      = t.bloomBlurH
        self.bloomBlurVHalf      = t.bloomBlurV
        self.outputTexture       = t.ldr
        self.tonemapWriteTarget  = t.ldr
        // Phase 4.39 denoiser textures
        self.aoFilteredTexture   = t.aoFiltered
        self.aoHistoryA          = t.aoHistoryA
        self.aoHistoryB          = t.aoHistoryB
        self.ssrRawTexture       = t.ssrRaw
        self.ssrHistoryA         = t.ssrHistoryA
        self.ssrHistoryB         = t.ssrHistoryB
        self.rtGIHistoryA        = t.rtGIHistoryA
        self.rtGIHistoryB        = t.rtGIHistoryB
        self.irrCacheA           = t.irrCacheA
        self.irrCacheB           = t.irrCacheB
        self.aoSampleCount       = t.aoSampleCount
        self.ssrSampleCount      = t.ssrSampleCount
        self.giSampleCount       = t.giSampleCount
        self.giVariance          = t.giVariance
        self.giAtrousA           = t.giAtrousA
        self.giAtrousB           = t.giAtrousB

        // Triple-buffer the presented LDR texture (see `ldrPool` docs above).
        // `t.ldr` becomes pool slot 0 (== outputTexture); allocate two more.
        var pool: [MTLTexture] = [t.ldr]
        for i in 1..<Self.ldrPoolCount {
            guard let extra = Self.makeLDRTexture(
                device: device, w: self.outputWidth, h: self.outputHeight,
                label: "Illuminatorama.ldr.\(i)") else {
                throw IlluminatoramaError.bufferAllocationFailed("ldrPool.\(i)")
            }
            pool.append(extra)
        }
        self.ldrPool = pool

        // ── Buffers ──────────────────────────────────────────────────────
        let initInstCap = 64
        let initLightCap = 16
        guard let iba = device.makeBuffer(
            length: MemoryLayout<IlluminatoramaInstance>.stride * initInstCap,
            options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("instancesA") }
        guard let ibb = device.makeBuffer(
            length: MemoryLayout<IlluminatoramaInstance>.stride * initInstCap,
            options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("instancesB") }
        iba.label = "Illuminatorama.instancesA"
        ibb.label = "Illuminatorama.instancesB"
        self.instanceBufferA = iba
        self.instanceBufferB = ibb
        self.instanceCapacity = initInstCap

        // Parallel superquadric-impostor param buffer (one entry per instance,
        // grouped-order-aligned with the instance buffer; see uploadInstances).
        // Ring-buffered (maxFramesInFlight) — params are read by the in-flight
        // frame's impostor draw, which can overlap the next frame's CPU write
        // when 2 frames pipeline. Motion vectors still come from current +
        // previous instance modelMatrix (the instance buffer's own ping-pong).
        let framesInFlight = IlluminatoramaRenderer.maxFramesInFlight
        let ringDevice = device   // local copy so the helper doesn't touch a still-initializing `self`
        func makeRing(_ length: Int, _ label: String, _ err: String) throws -> [MTLBuffer] {
            try (0..<framesInFlight).map { i in
                guard let b = ringDevice.makeBuffer(length: length, options: .storageModeShared)
                else { throw IlluminatoramaError.bufferAllocationFailed(err) }
                b.label = "\(label).\(i)"
                return b
            }
        }
        self.superquadricParamRing = try makeRing(
            MemoryLayout<IlluminatoramaInstance.SuperquadricParam>.stride * initInstCap,
            "Illuminatorama.superquadricParams", "superquadricParams")
        self.superquadricParamCapacity = initInstCap

        self.pointLightRing = try makeRing(
            MemoryLayout<IlluminatoramaPointLight>.stride * initLightCap,
            "Illuminatorama.pointLights", "pointLights")
        self.pointLightCapacity = initLightCap

        // Spot lights are bound at lighting kernel buffer(3). Same default
        // capacity as point lights — Eggs's rail-spotlight layout pushes
        // into the 20+ range so grow-on-demand will kick in early there.
        self.spotLightRing = try makeRing(
            MemoryLayout<IlluminatoramaSpotLight>.stride * initLightCap,
            "Illuminatorama.spotLights", "spotLights")
        self.spotLightCapacity = initLightCap

        // Area lights are bound at lighting kernel buffer(4). Few per scene
        // (one softbox / window pane is typical); grow-on-demand covers more.
        self.areaLightRing = try makeRing(
            MemoryLayout<IlluminatoramaAreaLight>.stride * initLightCap,
            "Illuminatorama.areaLights", "areaLights")
        self.areaLightCapacity = initLightCap

        // Secondary directional fills are bound at lighting kernel buffer(5).
        // A SCN 3-point rig ships 1–2 (fill + back); grow-on-demand covers more.
        self.extraDirectionalRing = try makeRing(
            MemoryLayout<IlluminatoramaDirectionalLight>.stride * initLightCap,
            "Illuminatorama.extraDirectionals", "extraDirectionals")
        self.extraDirectionalCapacity = initLightCap

        self.frameUniformRing = try makeRing(
            MemoryLayout<IlluminatoramaFrameUniforms>.stride,
            "Illuminatorama.frame", "frameUniforms")

        // Phase 1 ships three procedural primitives. Hosts can also call
        // `setMesh(_:_:)` to swap in their own.
        self.meshes[.box]    = IlluminatoramaMesh.unitBox(device: device)
        self.meshes[.sphere] = IlluminatoramaMesh.unitSphere(device: device)
        self.meshes[.ground] = IlluminatoramaMesh.unitGround(device: device)

        // ── Cover-blit pipeline (for MTKView direct-present path) ────────
        if let vs = library.makeFunction(name: "illumi_coverblit_vs"),
           let fs = library.makeFunction(name: "illumi_coverblit_fs") {
            let bd = MTLRenderPipelineDescriptor()
            bd.label = "Illuminatorama.coverBlit"
            bd.vertexFunction = vs
            bd.fragmentFunction = fs
            bd.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
            self.coverBlitPipeline = try? device.makeRenderPipelineState(descriptor: bd)
        }
        if coverBlitPipeline == nil {
            Self.log.warning("illumi_coverblit shaders not found — present(to:) will no-op")
        }

        Self.log.info("Illuminatorama renderer ready (internal \(self.width)x\(self.height), output \(self.outputWidth)x\(self.outputHeight))")
    }

    /// Present the current `outputTexture` directly into an `MTKView`
    /// drawable, bypassing SceneKit entirely. Applies a cover-fill UV crop
    /// (same math as the SceneKit `contentsTransform` path) so the fixed
    /// render canvas fills the drawable without stretching or letterboxing.
    ///
    /// Call this from the scene's tick loop INSTEAD of assigning
    /// `scene.background.contents = outputTexture`. The two paths are
    /// mutually exclusive — calling both wastes a full GPU pass.
    ///
    /// Safe to call when `drawable` is nil (no-ops). The blit is
    /// fire-and-forget on the renderer's own command queue; no CPU wait.
    public func present(to drawable: (any CAMetalDrawable)?) {
        guard let drawable,
              let pipeline = coverBlitPipeline,
              let cb = commandQueue.makeCommandBuffer() else { return }
        cb.label = "Illuminatorama.coverBlit"
        let w = drawable.texture.width
        let h = drawable.texture.height
        // A zero-size drawable means the MTKView has no window yet (tick loop
        // fires before MetalViewRenderer.makeNSView). Presenting it produces a
        // nil `priv` in the system's IOSurface callback → EXC_BAD_ACCESS in
        // layer_private_present_impl. Skip until the view is live.
        guard w > 0, h > 0 else { return }
        var uvRect = coverUVRect(viewportWidth: w, viewportHeight: h)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture    = drawable.texture
        pass.colorAttachments[0].loadAction  = .dontCare
        pass.colorAttachments[0].storeAction = .store

        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentTexture(outputTexture, index: 0)
        enc.setFragmentBytes(&uvRect,
                             length: MemoryLayout<SIMD4<Float>>.stride,
                             index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()

        // Feed the recorder (if any). Fired after the present commit so the
        // tap's copy enqueues behind it on the same queue — the texture is
        // fully written by then.
        onFramePresented?(outputTexture, commandQueue)
    }

    public func setMesh(_ kind: MeshKind, _ mesh: IlluminatoramaMesh) {
        meshes[kind] = mesh
    }

    // ── Phase 4.21 — one-shot GPU mesh normal/tangent synthesis ────────
    //
    // Replaces the CPU `synthesiseNormals` / `synthesiseTangents` passes that
    // used to run on the MAIN THREAD inside `IlluminatoramaMesh.from`. On a
    // cold-cache scene the extractor converts every geometry in one synchronous
    // `extractFrame` tick; the CPU tangent scatter-add parked the run loop for
    // seconds (observed ~9 s hang in `accumulateTangents`). This runs the same
    // math on the GPU instead.
    //
    // Builds a bounded per-vertex triangle-adjacency list with INTEGER atomics
    // (the counting-sort idiom; the project avoids float-atomic scatter — see
    // MLSMPM.metal), then gathers per vertex. Three GPU steps on one command
    // buffer: blit-clear the count buffer, `illumi_mesh_build_adjacency` (per
    // triangle), `illumi_mesh_synth` (per vertex). The adjacency and synth
    // dispatches live in SEPARATE compute encoders so the second observes the
    // first's writes; the blit precedes both.
    //
    // The command buffer is committed but NOT waited on: it shares the
    // renderer's single command queue with `render()`, whose command buffer is
    // committed later the same frame, so FIFO queue ordering guarantees the
    // synthesis completes before the G-buffer pass reads the vertex buffer — no
    // main-thread stall, no fence. Metal keeps the bound buffers (vertex/index +
    // the transient scratch) alive until the command buffer completes.
    private static let meshSynthMaxValence: UInt32 = 24

    public func synthesiseMeshGeometry(_ mesh: IlluminatoramaMesh) {
        guard mesh.pendingNormalSynth || mesh.pendingTangentSynth else { return }
        let vertCount = mesh.vertexCount
        let triCount = mesh.indexCount / 3
        guard vertCount > 0, triCount > 0 else {
            mesh.pendingNormalSynth = false
            mesh.pendingTangentSynth = false
            return
        }
        let maxValence = Self.meshSynthMaxValence

        // Transient scratch: per-vertex atomic counter + bounded adjacency list.
        // Freed when the command buffer completes (Metal retains them until then).
        let countBytes = MemoryLayout<UInt32>.stride * vertCount
        let listBytes = MemoryLayout<UInt32>.stride * vertCount * Int(maxValence)
        guard let vertTriCount = device.makeBuffer(length: countBytes,
                                                   options: .storageModePrivate),
              let vertTriList = device.makeBuffer(length: listBytes,
                                                  options: .storageModePrivate),
              let cb = commandQueue.makeCommandBuffer() else {
            // Couldn't allocate / encode — leave the placeholder normals and
            // zero tangents rather than crashing; clear the flags so we don't
            // retry every frame.
            Self.log.warning("synthesiseMeshGeometry: scratch alloc / cb failed; mesh ships unsynthesised")
            mesh.pendingNormalSynth = false
            mesh.pendingTangentSynth = false
            return
        }
        vertTriCount.label = "Illuminatorama.meshSynth.count"
        vertTriList.label = "Illuminatorama.meshSynth.list"
        cb.label = "Illuminatorama.meshSynth"

        var isU32: UInt32 = (mesh.indexType == .uint32) ? 1 : 0
        var triCountU = UInt32(triCount)
        var vertCountU = UInt32(vertCount)
        var maxValenceU = maxValence
        var doNormals: UInt32 = mesh.pendingNormalSynth ? 1 : 0
        var doTangents: UInt32 = mesh.pendingTangentSynth ? 1 : 0

        // 1. Zero the per-vertex counter (private buffers are not zero-inited).
        if let blit = cb.makeBlitCommandEncoder() {
            blit.fill(buffer: vertTriCount, range: 0..<countBytes, value: 0)
            blit.endEncoding()
        }

        // 2. Build adjacency — one thread per triangle.
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Illuminatorama.meshSynth.adjacency"
            enc.setComputePipelineState(meshBuildAdjacencyPipeline)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 0)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 1)
            enc.setBytes(&isU32, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&triCountU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&maxValenceU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBuffer(vertTriCount, offset: 0, index: 5)
            enc.setBuffer(vertTriList, offset: 0, index: 6)
            enc.setBytes(&vertCountU, length: MemoryLayout<UInt32>.stride, index: 7)
            dispatch1D(enc, pipeline: meshBuildAdjacencyPipeline, count: triCount)
            enc.endEncoding()
        }

        // 3. Synthesise normals/tangents — one thread per vertex. Separate
        //    encoder so it observes the adjacency writes from step 2.
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Illuminatorama.meshSynth.gather"
            enc.setComputePipelineState(meshSynthPipeline)
            enc.setBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 1)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 2)
            enc.setBytes(&isU32, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&vertCountU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&maxValenceU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBuffer(vertTriCount, offset: 0, index: 6)
            enc.setBuffer(vertTriList, offset: 0, index: 7)
            enc.setBytes(&doNormals, length: MemoryLayout<UInt32>.stride, index: 8)
            enc.setBytes(&doTangents, length: MemoryLayout<UInt32>.stride, index: 9)
            dispatch1D(enc, pipeline: meshSynthPipeline, count: vertCount)
            enc.endEncoding()
        }

        cb.commit()   // gpu-ok: committed, not waited — render()'s later cb on
                      // the same queue runs after this by FIFO ordering.
        mesh.pendingNormalSynth = false
        mesh.pendingTangentSynth = false
    }

    /// Dispatch a 1-D compute grid covering `count` threads, threadgroup sized
    /// to the pipeline's execution width.
    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            count: Int) {
        let tgWidth = max(1, min(pipeline.maxTotalThreadsPerThreadgroup,
                                 pipeline.threadExecutionWidth * 8))
        let groups = (count + tgWidth - 1) / tgWidth
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    }

    // ── Phase 4.8 — GPU-direct mesh registration ───────────────────────
    //
    // Compute-fed callers (`DynamicMesh`, `BarkRenderer`, MLS-MPM solvers)
    // hand the renderer a ready-made `IlluminatoramaMesh` whose vertex /
    // index buffers they own. The renderer stores it in `meshes` under a
    // freshly-generated `MeshKind.custom(UUID)` so each registration is
    // unique; the returned `IlluminatoramaMeshHandle` carries the kind
    // back to the host, who attaches it to the relevant `SCNGeometry` via
    // `SCNGeometry.illuminatoramaMeshHandle`. The scene extractor checks
    // the associated object before falling through to the CPU-readback
    // path, so a single attach is all the wiring needed per geometry.

    @discardableResult
    public func registerMesh(_ mesh: IlluminatoramaMesh) -> IlluminatoramaMeshHandle {
        let kind = MeshKind.custom("gpuMesh#\(UUID().uuidString)")
        meshes[kind] = mesh
        return IlluminatoramaMeshHandle(kind: kind, mesh: mesh, renderer: self)
    }

    /// Called by `IlluminatoramaMeshHandle.deinit` to evict the entry
    /// when the host drops its last strong reference. Internal because
    /// hosts should not call this directly — letting the handle's
    /// lifetime drive cleanup avoids accidental dangling MeshKind keys
    /// while an SCNNode still references the geometry.
    internal func removeMesh(_ kind: MeshKind) {
        meshes.removeValue(forKey: kind)
        gpuRepackTasks.removeAll { $0.kind == kind }
    }

    // ── Phase 4.13a — DynamicMesh bridge registration ────────────────

    /// A single per-frame GPU repack: take separate
    /// position + normal buffers (compute-fed by `DynamicMesh`) and
    /// interleave into an `IlluminatoramaVertex` buffer the G-buffer
    /// pass reads. Lives on the renderer for the lifetime of the
    /// corresponding `IlluminatoramaMeshHandle`.
    private struct GPURepackTask {
        let kind: MeshKind
        let positionBuffer: MTLBuffer
        let normalBuffer: MTLBuffer
        let vertexBuffer: MTLBuffer       // the output Illuminatorama-format buffer
        let vertexCount: Int
        /// Phase 4.17 — optional UV buffer (packed_float2, stride 8).
        /// `nil` → kernel writes synthetic `(0,0)` UV (the prior
        /// behaviour, fine for non-textured surfaces). Non-nil →
        /// per-vertex UV read from this buffer, enabling textured
        /// compute-fed geometry like LeafField sprite cards.
        let uvBuffer: MTLBuffer?
        /// Optional per-vertex RGBA color stream (stride-16 float4) → albedo.
        let colorBuffer: MTLBuffer?
    }
    private var gpuRepackTasks: [GPURepackTask] = []

    /// Register a compute-fed geometry described by an
    /// `IlluminatoramaGPUMeshDescriptor`. Allocates the Illuminatorama-
    /// format vertex buffer + combined index buffer once, schedules a
    /// per-frame repack on the renderer's queue, and returns a regular
    /// `IlluminatoramaMeshHandle`. Subsequent frames just dispatch the
    /// repack kernel — the host doesn't have to touch this again.
    @discardableResult
    public func registerGPUMesh(_ descriptor: IlluminatoramaGPUMeshDescriptor)
        -> IlluminatoramaMeshHandle?
    {
        let vertexBytes = MemoryLayout<IlluminatoramaVertex>.stride * descriptor.vertexCount
        guard let vertexBuffer = device.makeBuffer(length: vertexBytes,
                                                    options: .storageModePrivate) else {
            return nil
        }
        vertexBuffer.label = "Illuminatorama.gpuMesh.vertices"

        // Build a combined index buffer when the descriptor has both
        // body + caps. Each element is a triangle list with shared
        // vertices — concatenation is correct as long as indices are
        // already absolute (they are, in DynamicMesh's cache). The
        // resulting mesh has one element that draws both halves in a
        // single call.
        let combined = buildCombinedIndexBuffer(descriptor: descriptor)
        guard let combined = combined else { return nil }

        let mesh = IlluminatoramaMesh(
            vertexBuffer: vertexBuffer,
            indexBuffer: combined.buffer,
            indexCount: combined.count,
            indexType: combined.type,
            label: "Illuminatorama.gpuMesh"
        )
        mesh.doubleSided = descriptor.doubleSided
        let kind = MeshKind.custom("gpuMesh#\(UUID().uuidString)")
        meshes[kind] = mesh
        gpuRepackTasks.append(GPURepackTask(
            kind: kind,
            positionBuffer: descriptor.positionBuffer,
            normalBuffer: descriptor.normalBuffer,
            vertexBuffer: vertexBuffer,
            vertexCount: descriptor.vertexCount,
            uvBuffer: descriptor.uvBuffer,
            colorBuffer: descriptor.colorBuffer
        ))
        return IlluminatoramaMeshHandle(kind: kind, mesh: mesh, renderer: self)
    }

    /// Concatenate body + (optional) caps index buffers into one buffer
    /// the IlluminatoramaMesh can hand to a single draw call. Both
    /// elements share the same vertex buffer, so absolute indices stay
    /// valid across the merge. Returns nil if buffer allocation fails.
    private func buildCombinedIndexBuffer(descriptor: IlluminatoramaGPUMeshDescriptor)
        -> (buffer: MTLBuffer, count: Int, type: MTLIndexType)?
    {
        let stride = descriptor.bodyIndexType == .uint16 ? 2 : 4
        let bodyBytes = descriptor.bodyIndexCount * stride
        let capBytes  = descriptor.capIndexCount  * stride
        let totalBytes = bodyBytes + capBytes
        guard totalBytes > 0 else { return nil }
        guard let combined = device.makeBuffer(length: totalBytes,
                                                options: .storageModeShared) else {
            return nil
        }
        combined.label = "Illuminatorama.gpuMesh.indices"
        // Copy via blit so we can handle both .private and .shared source
        // buffers uniformly without `memcpy(...buffer.contents()...)`.
        // Reuse the renderer's shared SimEngine queue — no per-operation queue.
        guard let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: descriptor.bodyIndexBuffer, sourceOffset: 0,
                  to: combined, destinationOffset: 0, size: bodyBytes)
        if let capBuf = descriptor.capIndexBuffer, descriptor.capIndexCount > 0 {
            blit.copy(from: capBuf, sourceOffset: 0,
                      to: combined, destinationOffset: bodyBytes,
                      size: capBytes)
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted() // gpu-ok: one-shot setup, not per-frame
        let totalCount = descriptor.bodyIndexCount + descriptor.capIndexCount
        return (buffer: combined, count: totalCount, type: descriptor.bodyIndexType)
    }

    /// Per-frame repack pass — runs the `illumi_repack_pos_norm` kernel
    /// for every registered GPU mesh. Encodes one compute pass with all
    /// tasks dispatched back-to-back. Lives BEFORE the shadow / G-buffer
    /// passes so downstream reads see fresh interleaved vertices.
    private func encodeGPURepacks(_ cb: MTLCommandBuffer) {
        guard !gpuRepackTasks.isEmpty else { return }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.gpuMesh.repack"
        enc.setComputePipelineState(repackPosNormPipeline)
        for task in gpuRepackTasks {
            var count = UInt32(task.vertexCount)
            enc.setBuffer(task.positionBuffer, offset: 0, index: 0)
            enc.setBuffer(task.normalBuffer,   offset: 0, index: 1)
            enc.setBuffer(task.vertexBuffer,   offset: 0, index: 2)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
            // Phase 4.17 — UV stream + flag. When the caller supplied a
            // UV buffer, bind it and set `hasUV = 1`; otherwise bind the
            // position buffer as a one-byte dummy (Metal won't fault on
            // an unread bound buffer) and set `hasUV = 0`. The kernel
            // gates on the flag and writes a synthetic `(0,0)` UV in
            // the fallback path.
            var hasUV: UInt32 = task.uvBuffer != nil ? 1 : 0
            enc.setBuffer(task.uvBuffer ?? task.positionBuffer,
                          offset: 0, index: 4)
            enc.setBytes(&hasUV, length: MemoryLayout<UInt32>.stride, index: 5)
            // Optional per-vertex color stream → albedo (coin DEBUG tints).
            var hasColor: UInt32 = task.colorBuffer != nil ? 1 : 0
            enc.setBuffer(task.colorBuffer ?? task.positionBuffer,
                          offset: 0, index: 6)
            enc.setBytes(&hasColor, length: MemoryLayout<UInt32>.stride, index: 7)
            let tg = min(repackPosNormPipeline.maxTotalThreadsPerThreadgroup,
                         repackPosNormPipeline.threadExecutionWidth * 8)
            let groups = MTLSize(width: (task.vertexCount + tg - 1) / tg,
                                  height: 1, depth: 1)
            let threadgroup = MTLSize(width: tg, height: 1, depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
        }
        enc.endEncoding()
    }

    // ── Phase 4.11 — particle emitter registration ───────────────────
    //
    // Scenes that want Metal-native particles (per the "always Metal"
    // rule) create an `IlluminatoramaParticleEmitter` and register it.
    // Per-frame: the renderer ticks every enabled emitter's compute step
    // and draws survivors as additive HDR point sprites into the
    // composite. The same MTLBuffer the host writes into with `emit(...)`
    // is what the GPU reads.

    public func addEmitter(_ emitter: IlluminatoramaParticleEmitter) {
        if !particleEmitters.contains(where: { $0 === emitter }) {
            particleEmitters.append(emitter)
        }
    }

    public func removeEmitter(_ emitter: IlluminatoramaParticleEmitter) {
        particleEmitters.removeAll { $0 === emitter }
    }

    // ── Hardware RT geometry ──────────────────────────────────────────
    //
    // Supply the static world-space triangle soup the RT pass traces against
    // (the room's walls / floor / ceiling, with the window left as a hole so
    // sun rays pass through). Per-triangle albedo + geometric normal travel in
    // parallel arrays so the GI bounce can shade hit points. Builds the
    // primitive acceleration structure once; call again only if the geometry
    // changes. No-op on hardware without RT support.
    /// Diagnostic: append an AS-build command buffer's error to the sidecar file
    /// named by `VIZ_ILLUMI_CBERROR_PATH` (sandbox eats os_log). No-op without it.
    nonisolated static func recordASBuildError(_ cmd: MTLCommandBuffer, _ label: String) {
        guard cmd.status == .error || cmd.error != nil,
              let p = ProcessInfo.processInfo.environment["VIZ_ILLUMI_CBERROR_PATH"] else { return }
        let nsErr = cmd.error as NSError?
        let line = "\(label) status=\(cmd.status.rawValue) domain=\(nsErr?.domain ?? "?") code=\(nsErr?.code ?? 0) \(nsErr?.localizedDescription ?? "")\n"
        guard let data = line.data(using: .utf8) else { return }
        if let h = FileHandle(forWritingAtPath: p) { h.seekToEndOfFile(); h.write(data); try? h.close() }
        else { try? data.write(to: URL(fileURLWithPath: p)) }
    }

    /// Diagnostic: append surface-cache atlas sizing to the sidecar file named by
    /// `VIZ_ILLUMI_SURFCACHE_STATS_PATH` (sandbox eats os_log). Used to quantify
    /// the P2/P3 atlas-efficiency wins before/after. No-op without the env var.
    nonisolated static func recordSurfCacheStats(_ line: String) {
        guard let p = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_STATS_PATH"] else { return }
        guard let data = (line + "\n").data(using: .utf8) else { return }
        if let h = FileHandle(forWritingAtPath: p) { h.seekToEndOfFile(); h.write(data); try? h.close() }
        else { try? data.write(to: URL(fileURLWithPath: p)) }
    }

    public func setRTGeometry(positions: [SIMD3<Float>],
                              indices: [UInt32],
                              triangleAlbedo: [SIMD3<Float>],
                              triangleNormal: [SIMD3<Float>]) {
        guard rtSupported, rtPipeline != nil else { return }
        guard !positions.isEmpty, indices.count % 3 == 0, !indices.isEmpty else { return }
        let triCount = indices.count / 3
        guard triangleAlbedo.count == triCount, triangleNormal.count == triCount else {
            Self.log.error("setRTGeometry: per-triangle arrays must match triangle count")
            return
        }
        // Enforce the index↔vertex invariant at this registration boundary too
        // (symmetric with IlluminatoramaMesh.sanitiseIndices on the from() path):
        // an index >= positions.count would make the GPU acceleration-structure
        // BUILD read vertices past the buffer — a GPU fault that's far harder to
        // diagnose than this early return (the app sandbox eats os_log, so a fault
        // shows up only as a dropped frame / device loss). Callers are controlled
        // today, so this is defense-in-depth, not a known bug.
        // See docs/known-issues/illuminatorama-index-exceeds-vertexcount.md.
        if let mx = indices.max(), Int(mx) >= positions.count {
            Self.log.error("setRTGeometry: index \(mx) >= vertexCount \(positions.count); rejecting (would fault the AS build).")
            return
        }
        // Vertex positions as packed float3 (12-byte) for the AS triangle
        // descriptor; index buffer as uint32; per-tri albedo/normal as float4.
        var packedPos = [SIMD3<Float>]()  // 16-byte stride; AS reads .float3 with stride 16
        packedPos.reserveCapacity(positions.count)
        packedPos.append(contentsOf: positions)
        let albedo4 = triangleAlbedo.map { SIMD4<Float>($0, 1) }
        let normal4 = triangleNormal.map { SIMD4<Float>(simd_normalize($0), 0) }

        guard
            let vb = device.makeBuffer(bytes: packedPos,
                length: MemoryLayout<SIMD3<Float>>.stride * packedPos.count,
                options: .storageModeShared),
            let ib = device.makeBuffer(bytes: indices,
                length: MemoryLayout<UInt32>.stride * indices.count,
                options: .storageModeShared),
            let ab = device.makeBuffer(bytes: albedo4,
                length: MemoryLayout<SIMD4<Float>>.stride * albedo4.count,
                options: .storageModeShared),
            let nb = device.makeBuffer(bytes: normal4,
                length: MemoryLayout<SIMD4<Float>>.stride * normal4.count,
                options: .storageModeShared)
        else {
            Self.log.error("setRTGeometry: buffer allocation failed")
            return
        }
        vb.label = "Illuminatorama.rt.positions"
        ib.label = "Illuminatorama.rt.indices"
        ab.label = "Illuminatorama.rt.triAlbedo"
        nb.label = "Illuminatorama.rt.triNormal"
        rtVertexBuffer = vb
        rtIndexBuffer = ib
        rtTriAlbedoBuffer = ab
        rtTriNormalBuffer = nb
        rtTriangleCount = triCount

        buildSoupAccel()   // builds the primitive AS (triangle soup + curve geom when active)
        Self.log.notice("RT geometry built: \(triCount) triangles")
    }

    /// Build the soup primitive acceleration structure from the stored triangle
    /// buffers, plus — when a soup curve set is active (#60 item 7 incr. 2) — a
    /// SECOND curve geometry descriptor (round Catmull-Rom wood). Rebuilt on a
    /// topology change (new triangle geometry OR a curve registry version bump).
    /// When curves are active the AS uses `.refit` usage so the per-frame sway
    /// kernel + `refitRTSoupAccel` can update the control points in place.
    private func buildSoupAccel() {
        guard let vb = rtVertexBuffer, let ib = rtIndexBuffer, rtTriangleCount > 0 else { return }
        // Triangle soup AS — its OWN primitive AS (curves can't share it). Static,
        // so default usage (the room/extractor scenes never refit this).
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = vb
        geom.vertexBufferOffset = 0
        geom.vertexStride = MemoryLayout<SIMD3<Float>>.stride   // 16
        geom.vertexFormat = .float3
        geom.indexBuffer = ib
        geom.indexBufferOffset = 0
        geom.indexType = .uint32
        geom.triangleCount = rtTriangleCount

        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [geom]
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                            options: .storageModePrivate),
            let cmd = commandQueue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("buildSoupAccel: AS build setup failed")
            return
        }
        storage.label = "Illuminatorama.rt.accel"
        enc.build(accelerationStructure: storage, descriptor: asDesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: one-time / topology-change AS build (not per-frame)
        Self.recordASBuildError(cmd, "buildSoupAccel.primAS")
        rtAccel = storage
        rtScratch = scratch
        // Curves live in a SEPARATE primitive AS (Metal forbids mixed-type
        // descriptors in one AS). Built with .refit so the per-frame sway kernel
        // can update the control points + refit in place.
        buildSoupCurveAccel()
    }

    /// Build the curve-only primitive AS (#60 item 7 incr. 2) for the active soup
    /// curve set. Separate from the triangle AS (Metal: one geometry type per
    /// primitive AS). `.refit` usage so `encodeCurveWindDisplaceAndRefit` can sway.
    private func buildSoupCurveAccel() {
        rtSoupCurveAccel = nil; rtSoupRefitScratch = nil
        guard rtSoupCurvesActive, let cgeom = makeSoupCurveGeom() else { return }
        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [cgeom]
        asDesc.usage = .refit
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                            options: .storageModePrivate),
            let cmd = commandQueue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("buildSoupCurveAccel: AS build setup failed; dropping curves.")
            rtSoupCurvesActive = false; return
        }
        storage.label = "Illuminatorama.rt.curveAccel"
        enc.build(accelerationStructure: storage, descriptor: asDesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: one-time / topology-change curve AS build
        Self.recordASBuildError(cmd, "buildSoupCurveAccel.curveAS")
        rtSoupCurveAccel = storage
        // Scratch sized for BOTH refit and a full in-place rebuild — the per-frame
        // sway path REBUILDS the curve AS (cheap for the few-segment curve set),
        // confirmed to update the traced control points (the displace→rebuild→trace
        // +2m-offset probe moved the curves' shadows; whole-frame RMSE 0.0064).
        rtSoupRefitScratch = device.makeBuffer(
            length: max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize, 16),
            options: .storageModePrivate)
    }

    /// Build the curve geometry descriptor for the active soup curve set from the
    /// pooled buffers. The control-point buffer is the AS-referenced (swayed) pool;
    /// the kernel writes this each frame and the AS refits to it.
    private func makeSoupCurveGeom() -> MTLAccelerationStructureCurveGeometryDescriptor? {
        guard let cpts = rtSoupCurvePoolPoints, let crad = rtSoupCurveRadii,
              let cseg = rtSoupCurveSegments, rtSoupCurvePointCount > 0,
              rtSoupCurveSegmentCount > 0 else { return nil }
        let g = MTLAccelerationStructureCurveGeometryDescriptor()
        g.controlPointBuffer = cpts
        g.controlPointCount = rtSoupCurvePointCount
        g.controlPointStride = 12       // packed float3
        g.controlPointFormat = .float3
        g.radiusBuffer = crad
        g.radiusFormat = .float
        g.radiusStride = MemoryLayout<Float>.stride
        g.indexBuffer = cseg
        g.indexType = .uint32
        g.segmentCount = rtSoupCurveSegmentCount
        g.segmentControlPointCount = 4
        g.curveType = .round            // exact 3D normals (branches)
        g.curveBasis = .catmullRom
        g.curveEndCaps = .sphere
        g.opaque = true
        return g
    }

    /// Adopt the process-wide curve registry for the SOUP path (#60 item 7 incr. 2).
    /// Called from `render()`'s non-TLAS branch before the RT passes. A registry
    /// version bump is a TOPOLOGY change here (the soup AS gains/replaces its curve
    /// geometry descriptor), so it rebuilds `rtAccel`. Requires BOTH curve-variant
    /// pipelines — if either fails to compile the set is dropped (a curve-holding
    /// AS traced by a triangle-contract intersector is UB, never shippable).
    private func syncSoupCurveRegistry() {
        // TLAS-path renderers (extracted scenes) handle curves via syncCurveRegistry
        // in updateRTAccel; this is the soup/native path only.
        guard !buildRTFromExtractedScene else { return }
        let reg = IlluminatoramaCurveRegistry.shared
        guard reg.version != rtSoupCurveSyncVersion else { return }
        rtSoupCurveSyncVersion = reg.version
        // The soup path supports ONE curve set (single geometry descriptor). A
        // scene that needs many concatenates them into one set before registering.
        rtSoupCurveSet = device.supportsRaytracing ? reg.sets.first : nil
        if rtSoupCurveSet != nil {
            let cache = engine.pipelineCache
            var on = true
            if rtSoupCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                rtSoupCurvePipeline = cache.pipelineState(
                    name: "illumi_rt_lighting", device: device, constants: cv, variantKey: "soupCurves")
            }
            if surfCacheSoupCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                surfCacheSoupCurvePipeline = cache.pipelineState(
                    name: "illumi_surfcache_update", device: device, constants: cv, variantKey: "soupCurves")
            }
            if rtSoupCurvePipeline == nil || surfCacheSoupCurvePipeline == nil {
                Self.log.error("Soup curve RT: kRTCurvesEnabled pipeline variant failed; dropping curve set.")
                rtSoupCurveSet = nil
            }
        }
        buildSoupCurvePool()
        rtSoupCurvesActive = (rtSoupCurveSet != nil)
            && rtSoupCurvePoolPoints != nil && rtSoupCurveRadii != nil
            && rtSoupCurveSegments != nil && rtSoupCurveSetDataBuffer != nil
        buildSoupAccel()   // rebuild the AS with / without the curve geometry descriptor
    }

    /// Pool the active soup curve set's control points (flat `[Float]` → read as
    /// `packed_float3`, stride 12), radii, segment indices (RELATIVE — one geometry
    /// descriptor), per-point wind attribute, and the single `RTCurveSetData`
    /// (identity transform — the soup set is world-space). Keeps a REST copy of the
    /// points so the per-frame wind kernel can re-displace from rest each frame.
    private func buildSoupCurvePool() {
        rtSoupCurveRestPoints = nil; rtSoupCurvePoolPoints = nil; rtSoupCurveRadii = nil
        rtSoupCurveSegments = nil; rtSoupCurveWindAttr = nil; rtSoupCurveSetDataBuffer = nil
        rtSoupCurvePointCount = 0; rtSoupCurveSegmentCount = 0
        guard let set = rtSoupCurveSet else { return }
        var pts: [Float] = []; pts.reserveCapacity(set.controlPoints.count * 3)
        for p in set.controlPoints { pts.append(p.x); pts.append(p.y); pts.append(p.z) }
        let data = [RTCurveSetData(
            m0: set.transform.columns.0, m1: set.transform.columns.1,
            m2: set.transform.columns.2, m3: set.transform.columns.3,
            albedoRoughness: SIMD4(set.material.albedo, set.material.roughness),
            emissionPad: SIMD4(set.material.emission, 0),
            meta: SIMD4<UInt32>(0, 0, 0, 0))]   // single set ⇒ segment base 0
        let ptLen = MemoryLayout<Float>.stride * pts.count
        guard ptLen > 0,
              let rest = device.makeBuffer(bytes: pts, length: ptLen, options: .storageModeShared),
              let pool = device.makeBuffer(bytes: pts, length: ptLen, options: .storageModeShared),
              let rb = device.makeBuffer(bytes: set.radii,
                  length: MemoryLayout<Float>.stride * set.radii.count, options: .storageModeShared),
              let sb = device.makeBuffer(bytes: set.segmentIndices,
                  length: MemoryLayout<UInt32>.stride * set.segmentIndices.count, options: .storageModeShared),
              let db = device.makeBuffer(bytes: data,
                  length: MemoryLayout<RTCurveSetData>.stride, options: .storageModeShared)
        else {
            Self.log.error("Soup curve RT: pool allocation failed; dropping curve set.")
            rtSoupCurveSet = nil; return
        }
        rest.label = "Illuminatorama.rt.soupCurve.restPoints"
        pool.label = "Illuminatorama.rt.soupCurve.points"
        rb.label = "Illuminatorama.rt.soupCurve.radii"
        sb.label = "Illuminatorama.rt.soupCurve.segments"
        db.label = "Illuminatorama.rt.soupCurve.setData"
        rtSoupCurveRestPoints = rest; rtSoupCurvePoolPoints = pool
        rtSoupCurveRadii = rb; rtSoupCurveSegments = sb; rtSoupCurveSetDataBuffer = db
        rtSoupCurvePointCount = set.controlPoints.count
        rtSoupCurveSegmentCount = set.segmentIndices.count
        // Per-point wind attribute (sway/phase/flutter) — only when the set opts in.
        if set.windAttr.count == set.controlPoints.count {
            rtSoupCurveWindAttr = device.makeBuffer(bytes: set.windAttr,
                length: MemoryLayout<SIMD4<Float>>.stride * set.windAttr.count,
                options: .storageModeShared)
            rtSoupCurveWindAttr?.label = "Illuminatorama.rt.soupCurve.windAttr"
        }
    }

    /// Per-frame curve sway (#60 item 7 incr. 2): re-displace the rest control
    /// points into the AS-referenced pool with the SAME `applyTreeWind` the vertex
    /// shader uses, then refit the soup AS. No-op for a rigid set (no windAttr) or
    /// under `VIZ_ILLUMI_NO_CURVE_REFIT=1`.
    private func encodeCurveWindDisplaceAndRefit(_ cb: MTLCommandBuffer) {
        guard rtSoupCurvesActive, !noCurveRefit,
              let pipe = curveWindDisplacePipeline,
              let rest = rtSoupCurveRestPoints, let pool = rtSoupCurvePoolPoints,
              let wind = rtSoupCurveWindAttr, rtSoupCurvePointCount > 0 else {
            if ProcessInfo.processInfo.environment["VIZ_CURVE_DEBUG"] == "1", !curveDbgPrinted {
                curveDbgPrinted = true
                let m = "CURVE displace SKIPPED: active=\(rtSoupCurvesActive) noRefit=\(noCurveRefit) pipe=\(curveWindDisplacePipeline != nil) rest=\(rtSoupCurveRestPoints != nil) pool=\(rtSoupCurvePoolPoints != nil) wind=\(rtSoupCurveWindAttr != nil) pts=\(rtSoupCurvePointCount)\n"
                FileHandle.standardError.write(m.data(using: .utf8)!)
            }
            return
        }
        if ProcessInfo.processInfo.environment["VIZ_CURVE_DEBUG"] == "1", !curveDbgPrinted {
            curveDbgPrinted = true
            let m = "CURVE displace RAN: pts=\(rtSoupCurvePointCount) time=\(time) windStrength=\(treeWindStrength) heading=\(treeWindHeading)\n"
            FileHandle.standardError.write(m.data(using: .utf8)!)
        }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(rest, offset: 0, index: 0)
        enc.setBuffer(wind, offset: 0, index: 1)
        enc.setBuffer(pool, offset: 0, index: 2)
        var u = CurveWindUniforms(time: time, windStrength: treeWindStrength,
                                  windHeading: treeWindHeading,
                                  pointCount: UInt32(rtSoupCurvePointCount))
        enc.setBytes(&u, length: MemoryLayout<CurveWindUniforms>.stride, index: 3)
        let w = min(pipe.threadExecutionWidth, rtSoupCurvePointCount)
        enc.dispatchThreads(MTLSize(width: rtSoupCurvePointCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
        enc.endEncoding()
        // Rebuild the CURVE-only AS in place from the displaced control points on
        // the SAME command buffer (Metal sequences the compute write → AS-build
        // read). A full rebuild is cheap for the few-segment curve set and
        // unambiguously re-reads the swayed pool (confirmed: the +2m-offset probe
        // moved the curves' shadows). gpu-ok: per-frame rebuild encoded on the
        // frame cb, no waitUntilCompleted.
        guard let accel = rtSoupCurveAccel, let scratch = rtSoupRefitScratch,
              let cgeom = makeSoupCurveGeom() else { return }
        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [cgeom]
        asDesc.usage = .refit
        guard let renc = cb.makeAccelerationStructureCommandEncoder() else { return }
        renc.build(accelerationStructure: accel, descriptor: asDesc,
                   scratchBuffer: scratch, scratchBufferOffset: 0)
        renc.endEncoding()
    }

    // ── Surface-cache cards ───────────────────────────────────────────
    //
    // Supply the planar cards (one per floor / wall / ceiling / window jamb)
    // and the per-triangle → card-UV mapping the surface cache needs. `triCard`
    // / `triUVa` / `triUVc` are indexed by the SAME `primitive_id` the RT AS
    // uses (so a ray hit maps to a card texel). Allocates the ping-pong
    // radiance atlas sized to fit `cards.count` tiles. Call once after
    // `setRTGeometry`; no-op without RT support.
    //
    // `triUVa[i] = (uvA.xy, uvB.xy)`, `triUVc[i] = (uvC.xy, 0, 0)` for triangle
    // i's three vertices in its card's [0,1]² parameterisation.
    /// Phase 4.38 — surface-cache generalisation (#17 item 3, P1). Synthesise a
    /// per-triangle "micro-card" parameterisation for an ARBITRARY triangle soup,
    /// so any extracted mesh — not just hand-authored planar quads like the room
    /// — can drive the surface cache. The insight: a single triangle IS planar,
    /// so it fits `SurfCard` exactly — `origin = v0`, `uAxis = v1−v0`,
    /// `vAxis = v2−v0`, and the reference UVs `A(0,0) B(1,0) C(0,1)` make the
    /// update kernel's `origin + u·uAxis + v·vAxis` reconstruct points in the
    /// triangle's plane (the triangle occupies the `u+v ≤ 1` half of its square
    /// tile). One card per triangle; `triCard[t] = t`. No new GPU code — the
    /// existing atlas / ping-pong / EMA / `sampleSurfCache` machinery is reused
    /// verbatim. Atlas size scales with triangle count (bounded by the live-RT
    /// triangle cap). **P3 (done):** two triangles share each square tile (frame
    /// A in the `u+v ≤ 1` lower-left half, frame B in the `u+v > 1` upper-right
    /// half) — since a single triangle already only used ~half its tile's texels,
    /// packing fills the wasted half with a second triangle for ~2× atlas/relight
    /// efficiency at the SAME per-triangle resolution. Coplanar-chart merge (P2)
    /// for better texel coherence remains the follow-up.
    ///
    /// Pair with `setRTGeometry(positions:indices:triangleAlbedo:triangleNormal:)`
    /// using the SAME `positions`/`indices`, then feed the result straight to
    /// `setSurfaceCacheCards`.
    ///
    /// `blockStarts` — triangle indices where each instance's block begins in the
    /// global soup. The 2-tri pairing NEVER crosses one, so every instance block
    /// of a given mesh produces an IDENTICAL frame-A/B layout. That mesh-invariance
    /// is what lets the per-triangle card UVs be baked into the shared BLAS
    /// `primitive_data` (#60 item 6) — without it, an odd-triangle-count mesh would
    /// flip A/B parity on each subsequent instance (or a pair would straddle two
    /// instances), and a single BLAS-baked UV would sample the wrong tile half for
    /// some instances. Empty `blockStarts` keeps the legacy global pairing.
    public static func makePerTriangleSurfaceCards(
        positions: [SIMD3<Float>],
        indices: [UInt32],
        triangleAlbedo: [SIMD3<Float>],
        triangleNormal: [SIMD3<Float>],
        triangleEmission: [SIMD3<Float>]? = nil,
        blockStarts: [Int] = []
    ) -> (cards: [SurfCard], triCard: [UInt32],
          triUVa: [SIMD4<Float>], triUVc: [SIMD4<Float>]) {
        let triCount = indices.count / 3
        // Instance-block boundaries the pairing must not cross (interior only).
        let boundary = Set(blockStarts.filter { $0 > 0 && $0 < triCount })
        var cards: [SurfCard] = [];   cards.reserveCapacity((triCount + 1) / 2)
        // triCard / triUVa / triUVc are indexed by PRIMITIVE (triangle), so they
        // stay length `triCount`. Pre-size with placeholders so a `continue`d
        // (degenerate-index) triangle keeps the array aligned with primitive ids.
        var triCard = [UInt32](repeating: 0, count: triCount)
        var triUVa  = [SIMD4<Float>](repeating: .zero, count: triCount)
        var triUVc  = [SIMD4<Float>](repeating: .zero, count: triCount)

        // P3 reference UVs. Frame A (lower-left half, u+v ≤ 1): A(0,0) B(1,0) C(0,1).
        // Frame B (upper-right half, u+v ≥ 1): A(1,1) B(0,1) C(1,0). The update
        // kernel mirrors B's texels back into a lower-left frame, so a hit's
        // barycentric→UV lands in the correct half of the SHARED tile.
        let uvAa = SIMD4<Float>(0, 0, 1, 0), uvAc = SIMD4<Float>(0, 1, 0, 0)
        let uvBa = SIMD4<Float>(1, 1, 0, 1), uvBc = SIMD4<Float>(1, 0, 0, 0)

        func vertsOK(_ t: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)? {
            let i0 = Int(indices[3 * t]), i1 = Int(indices[3 * t + 1]), i2 = Int(indices[3 * t + 2])
            guard i0 < positions.count, i1 < positions.count, i2 < positions.count,
                  t < triangleAlbedo.count, t < triangleNormal.count else { return nil }
            return (positions[i0], positions[i1], positions[i2])
        }

        // Pair consecutive triangles two-per-tile. A lone trailing triangle (odd
        // count, or when its partner has degenerate indices) becomes an unpacked
        // single-frame card occupying the tile's lower-left half.
        var t = 0
        while t < triCount {
            guard let (a0, a1, a2) = vertsOK(t) else { t += 1; continue }
            let aE = triangleEmission?[t] ?? .zero
            // Look for a valid partner triangle to share this tile — but never
            // pair across an instance-block boundary (keeps the layout mesh-invariant).
            if t + 1 < triCount, !boundary.contains(t + 1), let (b0, b1, b2) = vertsOK(t + 1) {
                let bE = triangleEmission?[t + 1] ?? .zero
                let card = UInt32(cards.count)
                cards.append(SurfCard(
                    aOrigin: a0, aUAxis: a1 - a0, aVAxis: a2 - a0,
                    aNormal: triangleNormal[t], aAlbedo: triangleAlbedo[t], aEmission: aE,
                    bOrigin: b0, bUAxis: b1 - b0, bVAxis: b2 - b0,
                    bNormal: triangleNormal[t + 1], bAlbedo: triangleAlbedo[t + 1], bEmission: bE))
                triCard[t] = card;     triUVa[t] = uvAa;     triUVc[t] = uvAc
                triCard[t + 1] = card; triUVa[t + 1] = uvBa; triUVc[t + 1] = uvBc
                t += 2
            } else {
                let card = UInt32(cards.count)
                cards.append(SurfCard(
                    origin: a0, uAxis: a1 - a0, vAxis: a2 - a0,
                    normal: triangleNormal[t], albedo: triangleAlbedo[t], emission: aE))
                triCard[t] = card; triUVa[t] = uvAa; triUVc[t] = uvAc
                t += 1
            }
        }
        return (cards, triCard, triUVa, triUVc)
    }

    /// Per-triangle card-frame UVs baked into a mesh's BLAS `primitive_data`
    /// (#60 item 6). One element per BLAS primitive (== mesh-local triangle),
    /// 24 B, `SIMD2<Float>` stride 8 ↔ Metal `float2` (the safe layout — NOT the
    /// `SIMD3`/`packed_float3` trap). A TLAS hit reads its card UVs straight from
    /// the AS payload (`res.primitive_data`) instead of the per-instance global
    /// `triUVa`/`triUVc` side buffers, dropping 32 B/hit of dependent loads on the
    /// surface-cache GI/reflection path.
    struct IlluminatoramaPrimUV { var uvA: SIMD2<Float>; var uvB: SIMD2<Float>; var uvC: SIMD2<Float> }

    /// The canonical, mesh-invariant frame-A/B UV table for a mesh's local
    /// triangles — the exact pattern `makePerTriangleSurfaceCards` produces for a
    /// single instance block (pairing reset at the block boundary). Triangle 2k
    /// gets frame A, 2k+1 gets frame B; a lone trailing triangle gets frame A.
    /// Depends ONLY on the triangle count, so it's identical for every instance of
    /// the mesh — that's the property the BLAS fold relies on. (In the global pass
    /// out-of-range triangles are pre-dropped before pairing, so the BLAS primitive
    /// count == the per-instance soup triangle count == this length — the same
    /// invariant the existing `soupTriBase[iid]+primitive_id` math already assumes.)
    static func perMeshCardUVs(triangleCount: Int) -> [IlluminatoramaPrimUV] {
        // Frame A (u+v ≤ 1): A(0,0) B(1,0) C(0,1). Frame B (u+v ≥ 1): A(1,1) B(0,1) C(1,0).
        let frameA = IlluminatoramaPrimUV(uvA: SIMD2(0, 0), uvB: SIMD2(1, 0), uvC: SIMD2(0, 1))
        let frameB = IlluminatoramaPrimUV(uvA: SIMD2(1, 1), uvB: SIMD2(0, 1), uvC: SIMD2(1, 0))
        var out = [IlluminatoramaPrimUV](repeating: frameA, count: max(0, triangleCount))
        var t = 0
        while t < triangleCount {
            out[t] = frameA
            if t + 1 < triangleCount { out[t + 1] = frameB; t += 2 } else { t += 1 }
        }
        return out
    }

    /// DEBUG guard for the #60-item-6 fold: the BLAS bakes `perMeshCardUVs`, so the
    /// global pass's per-instance UV layout MUST match it for every instance (not
    /// just the first). Spot-checks the first instance block of each mesh kind and
    /// logs the first mismatch. A failure means the packing isn't mesh-invariant —
    /// the fold would sample the wrong tile half — so it must never ship silent.
    static func assertPerMeshUVInvariance(triUVa: [SIMD4<Float>], triUVc: [SIMD4<Float>],
                                          soupBase: [UInt32], instanceKinds: [MeshKind],
                                          deformingKinds: Set<MeshKind>, total: Int) {
        #if DEBUG
        var checked = Set<MeshKind>()
        for i in 0..<total where i < instanceKinds.count {
            let kind = instanceKinds[i]
            if checked.contains(kind) { continue }
            // Deforming kinds aren't folded (their BLAS bakes no primitive data), so
            // a parity shift from their un-sanitised soup is harmless — skip them.
            if deformingKinds.contains(kind) { continue }
            let lo = Int(soupBase[i])
            let hi = (i + 1 < total) ? Int(soupBase[i + 1]) : triUVa.count
            let count = hi - lo
            guard count > 0, hi <= triUVa.count, hi <= triUVc.count else { continue }
            checked.insert(kind)
            let canon = perMeshCardUVs(triangleCount: count)
            for t in 0..<count {
                let a = triUVa[lo + t], c = triUVc[lo + t], p = canon[t]
                let okA = a.x == p.uvA.x && a.y == p.uvA.y && a.z == p.uvB.x && a.w == p.uvB.y
                let okC = c.x == p.uvC.x && c.y == p.uvC.y
                if !(okA && okC) {
                    Self.log.error("perMeshCardUVs invariance FAIL: kind=\(String(describing: kind)) localTri=\(t) global=\(lo + t) — fold would corrupt this mesh's cache UVs. (#60 item 6)")
                    break
                }
            }
        }
        #endif
    }

    /// Phase 4.38c — surface-cache generalisation (#17 item 3, P2). **Coplanar-
    /// chart merging.** Instead of one micro-card per triangle, merge adjacent
    /// COPLANAR same-material triangles into a single larger `SurfCard` "chart"
    /// with a shared 2D parameterisation. Wins over the per-triangle path:
    ///   • Far fewer cards — a triangulated wall / road / building face re-emerges
    ///     as one chart (the room's hand-authored quads come back automatically).
    ///   • Better texel coherence — a flat surface is one contiguous tile, not N
    ///     disjoint micro-tiles with seams.
    ///   • Larger, more STABLE cards — the unit incremental invalidation tracks.
    ///
    /// A chart is an UNPACKED single-frame `SurfCard` (frame A fills the whole
    /// tile) — exactly the room's hand-authored-quad mode — so the update kernel
    /// + the consumer samplers are reused VERBATIM (zero GPU change vs P1/P3).
    ///
    /// Grouping (flood fill over edge adjacency):
    ///   • coplanar — two edge-adjacent triangles with parallel normals
    ///     (`dot > normalCosThresh`) are coplanar by construction (a shared edge
    ///     lies in both planes; parallel normals ⇒ same plane).
    ///   • same material — albedo + emission within tolerance (keeps lit windows
    ///     out of wall charts; the card carries ONE albedo/emission).
    ///   • bounded extent — the chart's 2D bbox side is capped (`maxExtent`) so a
    ///     uniform atlas tile keeps a consistent texel density (a huge wall splits
    ///     into a few medium charts, not one giant blurry one — still far fewer
    ///     than per-triangle).
    ///
    /// Parameterisation: fix a 2D basis (tangent/bitangent) from the seed normal,
    /// project all chart verts, take the 2D bbox → frame `origin = P0 + umin·T +
    /// vmin·B`, `uAxis = (umax−umin)·T`, `vAxis = (vmax−vmin)·B` (so the kernel's
    /// `origin + s·uAxis + t·vAxis` reconstructs the plane). Each triangle's verts
    /// map to `(coord2D − bboxMin)/bboxSize ∈ [0,1]²`.
    public static func makeCoplanarChartCards(
        positions: [SIMD3<Float>],
        indices: [UInt32],
        triangleAlbedo: [SIMD3<Float>],
        triangleNormal: [SIMD3<Float>],
        triangleEmission: [SIMD3<Float>]? = nil,
        normalCosThresh: Float = 0.996,   // ≈5°
        albedoTol: Float = 0.05,
        emissionTol: Float = 0.15,
        maxExtent: Float = 4.0            // metres, chart 2D bbox side cap
    ) -> (cards: [SurfCard], triCard: [UInt32],
          triUVa: [SIMD4<Float>], triUVc: [SIMD4<Float>]) {
        let triCount = indices.count / 3
        var triCard = [UInt32](repeating: 0, count: triCount)
        var triUVa  = [SIMD4<Float>](repeating: .zero, count: triCount)
        var triUVc  = [SIMD4<Float>](repeating: .zero, count: triCount)
        var cards: [SurfCard] = []; cards.reserveCapacity(triCount / 4 + 1)

        func tri(_ t: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)? {
            let i0 = Int(indices[3*t]), i1 = Int(indices[3*t+1]), i2 = Int(indices[3*t+2])
            guard i0 < positions.count, i1 < positions.count, i2 < positions.count,
                  t < triangleAlbedo.count, t < triangleNormal.count else { return nil }
            return (positions[i0], positions[i1], positions[i2])
        }
        // Orthonormal 2D basis for a plane normal (Duff et al. branchless frame).
        func basis(_ n: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
            let s: Float = n.z >= 0 ? 1 : -1
            let a = -1 / (s + n.z)
            let b = n.x * n.y * a
            let tangent = SIMD3<Float>(1 + s * n.x * n.x * a, s * b, -s * n.x)
            let bitan = SIMD3<Float>(b, s + n.y * n.y * a, -n.y)
            return (tangent, bitan)
        }
        func matClose(_ t0: Int, _ t1: Int) -> Bool {
            let da = simd_abs(triangleAlbedo[t0] - triangleAlbedo[t1])
            if max(da.x, max(da.y, da.z)) > albedoTol { return false }
            let e0 = triangleEmission?[t0] ?? .zero, e1 = triangleEmission?[t1] ?? .zero
            let de = simd_abs(e0 - e1)
            return max(de.x, max(de.y, de.z)) <= emissionTol
        }

        // Edge → adjacent triangles. Key = sorted vertex-index pair packed in u64.
        var edgeMap: [UInt64: [Int]] = [:]; edgeMap.reserveCapacity(triCount * 3)
        func edgeKey(_ a: UInt32, _ b: UInt32) -> UInt64 {
            let lo = min(a, b), hi = max(a, b)
            return (UInt64(lo) << 32) | UInt64(hi)
        }
        for t in 0..<triCount {
            guard tri(t) != nil else { continue }
            let i0 = indices[3*t], i1 = indices[3*t+1], i2 = indices[3*t+2]
            edgeMap[edgeKey(i0, i1), default: []].append(t)
            edgeMap[edgeKey(i1, i2), default: []].append(t)
            edgeMap[edgeKey(i2, i0), default: []].append(t)
        }
        func neighbours(_ t: Int) -> [Int] {
            let i0 = indices[3*t], i1 = indices[3*t+1], i2 = indices[3*t+2]
            var out: [Int] = []
            for k in [edgeKey(i0, i1), edgeKey(i1, i2), edgeKey(i2, i0)] {
                if let ts = edgeMap[k] { for o in ts where o != t { out.append(o) } }
            }
            return out
        }

        var visited = [Bool](repeating: false, count: triCount)
        for seed in 0..<triCount {
            if visited[seed] { continue }
            guard let (s0, s1, s2) = tri(seed) else {
                // Degenerate triangle — lone zero card to keep primitive alignment.
                visited[seed] = true
                let card = UInt32(cards.count)
                cards.append(SurfCard(origin: .zero, uAxis: .zero, vAxis: .zero,
                                      normal: SIMD3<Float>(0, 1, 0), albedo: .zero, emission: .zero))
                triCard[seed] = card
                continue
            }
            let N = triangleNormal[seed]
            let (T, B) = basis(N)
            let P0 = s0
            func proj(_ v: SIMD3<Float>) -> SIMD2<Float> {
                let d = v - P0; return SIMD2(simd_dot(d, T), simd_dot(d, B))
            }
            // BFS, growing the chart while the 2D bbox stays within maxExtent.
            var members: [Int] = [seed]
            visited[seed] = true
            var uMin = Float.greatestFiniteMagnitude, uMax = -Float.greatestFiniteMagnitude
            var vMin = Float.greatestFiniteMagnitude, vMax = -Float.greatestFiniteMagnitude
            func extend(_ verts: [SIMD3<Float>]) {
                for v in verts { let p = proj(v)
                    uMin = min(uMin, p.x); uMax = max(uMax, p.x)
                    vMin = min(vMin, p.y); vMax = max(vMax, p.y) }
            }
            extend([s0, s1, s2])
            var queue = [seed]; var qi = 0
            while qi < queue.count {
                let cur = queue[qi]; qi += 1
                for nb in neighbours(cur) where !visited[nb] {
                    guard let (n0, n1, n2) = tri(nb) else { continue }
                    if simd_dot(triangleNormal[nb], N) < normalCosThresh { continue }
                    if !matClose(seed, nb) { continue }
                    // Tentative bbox if we add nb — reject if it busts the extent cap.
                    var tuMin = uMin, tuMax = uMax, tvMin = vMin, tvMax = vMax
                    for v in [n0, n1, n2] { let p = proj(v)
                        tuMin = min(tuMin, p.x); tuMax = max(tuMax, p.x)
                        tvMin = min(tvMin, p.y); tvMax = max(tvMax, p.y) }
                    if (tuMax - tuMin) > maxExtent || (tvMax - tvMin) > maxExtent { continue }
                    visited[nb] = true
                    members.append(nb)
                    uMin = tuMin; uMax = tuMax; vMin = tvMin; vMax = tvMax
                    queue.append(nb)
                }
            }
            // Build the chart card from the final bbox.
            let du = max(uMax - uMin, 1e-4), dv = max(vMax - vMin, 1e-4)
            let origin = P0 + uMin * T + vMin * B
            let card = UInt32(cards.count)
            cards.append(SurfCard(
                origin: origin, uAxis: du * T, vAxis: dv * B,
                normal: N, albedo: triangleAlbedo[seed],
                emission: triangleEmission?[seed] ?? .zero))
            // Map each member triangle's verts into the chart's [0,1]² space.
            for m in members {
                guard let (m0, m1, m2) = tri(m) else { continue }
                func uv(_ v: SIMD3<Float>) -> SIMD2<Float> {
                    let p = proj(v)
                    return SIMD2((p.x - uMin) / du, (p.y - vMin) / dv)
                }
                let a = uv(m0), b = uv(m1), c = uv(m2)
                triCard[m] = card
                triUVa[m] = SIMD4(a.x, a.y, b.x, b.y)
                triUVc[m] = SIMD4(c.x, c.y, 0, 0)
            }
        }
        return (cards, triCard, triUVa, triUVc)
    }

    /// Register surface-cache cards + their per-triangle UV map and allocate the
    /// ping-pong radiance atlas. Addressing is per-card RECT (`cardRect` +
    /// `texelCard`), shelf-packed — the uniform tile grid is the special case
    /// where every card requests the same tile size.
    ///
    /// `adaptiveTiles == false` (room quads, P3 per-triangle): every card gets a
    /// `tileSize²` tile. `adaptiveTiles == true` (P2 charts): each card's tile is
    /// sized to its world extent (`|uAxis| × |vAxis| · texelsPerMeter`, clamped),
    /// so a big wall chart gets more texels than a small facet and the atlas stays
    /// efficient regardless of chart-size spread.
    public func setSurfaceCacheCards(cards: [SurfCard],
                                     triCard: [UInt32],
                                     triUVa: [SIMD4<Float>],
                                     triUVc: [SIMD4<Float>],
                                     tileSize overrideTileSize: Int? = nil,
                                     adaptiveTiles: Bool = false,
                                     texelsPerMeter: Float = 6.0) {
        guard rtSupported, surfCachePipeline != nil else { return }
        guard !cards.isEmpty, triCard.count == rtTriangleCount,
              triUVa.count == rtTriangleCount, triUVc.count == rtTriangleCount else {
            Self.log.error("setSurfaceCacheCards: counts must match triangle/card sizes")
            return
        }
        let baseTile = max(2, overrideTileSize ?? surfTileSize)
        surfActiveTileSize = baseTile

        // (1) Per-card desired tile size (w,h) in texels.
        let minTile = 4, maxTile = 48
        var sizes = [SIMD2<Int>](repeating: SIMD2(baseTile, baseTile), count: cards.count)
        if adaptiveTiles {
            for c in 0..<cards.count {
                let wm = simd_length(SIMD3(cards[c].uAxis.x, cards[c].uAxis.y, cards[c].uAxis.z))
                let hm = simd_length(SIMD3(cards[c].vAxis.x, cards[c].vAxis.y, cards[c].vAxis.z))
                let tw = min(maxTile, max(minTile, Int((wm * texelsPerMeter).rounded())))
                let th = min(maxTile, max(minTile, Int((hm * texelsPerMeter).rounded())))
                sizes[c] = SIMD2(tw, th)
            }
        }

        // Phase 5 / A0 — residency budget. RESIDENT cards (the first `budget` in
        // bake order) get an atlas slot + are packed; the rest get a zero rect (no
        // slot), so the atlas is sized to the BUDGET, not the scene. A GI/reflection
        // hit on a non-resident card detects the zero rect and falls back. Disabled
        // (budget 0 / ≥ count) ⇒ every card resident, identical to before.
        let residentCount = (surfCardBudget > 0) ? min(surfCardBudget, cards.count) : cards.count
        let isResident: (Int) -> Bool = { $0 < residentCount }

        // (2) Shelf-pack the rects. Sort by height desc for tight shelves; place
        // left→right, wrap to a new shelf when the row exceeds the target width.
        // Only resident cards contribute area / get packed.
        let totalArea = (0..<cards.count).reduce(0) { isResident($1) ? $0 + sizes[$1].x * sizes[$1].y : $0 }
        let maxTileW = (0..<cards.count).reduce(baseTile) { isResident($1) ? max($0, sizes[$1].x) : $0 }
        let atlasW = max(maxTileW, Int((Double(totalArea).squareRoot() * 1.08).rounded(.up)))
        let order = (0..<cards.count).filter(isResident).sorted { sizes[$0].y > sizes[$1].y }
        var rects = [SIMD4<Float>](repeating: .zero, count: cards.count)  // x,y,w,h (.zero = non-resident)
        var penX = 0, penY = 0, shelfH = 0
        for c in order {
            let (w, h) = (sizes[c].x, sizes[c].y)
            if penX + w > atlasW { penX = 0; penY += shelfH; shelfH = 0 }
            rects[c] = SIMD4(Float(penX), Float(penY), Float(w), Float(h))
            penX += w
            shelfH = max(shelfH, h)
        }
        let atlasH = penY + shelfH
        surfResidentCardCount = residentCount
        // Phase 5 / A1 — slot maps (only when a budget is active). A "slot" is a
        // bake-time resident card's atlas rect; A1 reassigns occupants per frame.
        if residentCount < cards.count {
            surfSlotRect = (0..<residentCount).map { rects[$0] }
            surfSlotCard = Array(0..<residentCount)
            surfCardSlot = (0..<cards.count).map { $0 < residentCount ? $0 : -1 }
            surfCardHeat = [Int](repeating: 0, count: cards.count)   // A2 — all cold at bake
            // A4 — bake-time world centroid per card (origin + ½(uAxis + vAxis)).
            surfCardCentroid = cards.map {
                let o = SIMD3($0.origin.x, $0.origin.y, $0.origin.z)
                let u = SIMD3($0.uAxis.x, $0.uAxis.y, $0.uAxis.z)
                let v = SIMD3($0.vAxis.x, $0.vAxis.y, $0.vAxis.z)
                return o + 0.5 * (u + v)
            }
        } else {
            surfSlotRect = []; surfSlotCard = []; surfCardSlot = []; surfCardHeat = []
            surfCardCentroid = []
        }
        guard atlasW > 0, atlasH > 0 else { Self.log.error("setSurfaceCacheCards: empty atlas"); return }

        // (3) texelCard map (uint per atlas texel; 0xFFFFFFFF = unused gap).
        var texelCard = [UInt32](repeating: 0xFFFF_FFFF, count: atlasW * atlasH)
        for c in 0..<cards.count {
            let r = rects[c]
            let x0 = Int(r.x), y0 = Int(r.y), w = Int(r.z), h = Int(r.w)
            for yy in y0..<(y0 + h) {
                let rowBase = yy * atlasW
                for xx in x0..<(x0 + w) { texelCard[rowBase + xx] = UInt32(c) }
            }
        }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: atlasW, height: atlasH, mipmapped: false)
        texDesc.usage = [.shaderRead, .shaderWrite]
        texDesc.storageMode = .private
        guard
            let cb = device.makeBuffer(bytes: cards,
                length: MemoryLayout<SurfCard>.stride * cards.count, options: .storageModeShared),
            let tcb = device.makeBuffer(bytes: triCard,
                length: MemoryLayout<UInt32>.stride * triCard.count, options: .storageModeShared),
            let tab = device.makeBuffer(bytes: triUVa,
                length: MemoryLayout<SIMD4<Float>>.stride * triUVa.count, options: .storageModeShared),
            let tcc = device.makeBuffer(bytes: triUVc,
                length: MemoryLayout<SIMD4<Float>>.stride * triUVc.count, options: .storageModeShared),
            let crb = device.makeBuffer(bytes: rects,
                length: MemoryLayout<SIMD4<Float>>.stride * rects.count, options: .storageModeShared),
            let txb = device.makeBuffer(bytes: texelCard,
                length: MemoryLayout<UInt32>.stride * texelCard.count, options: .storageModeShared),
            let atlasA = device.makeTexture(descriptor: texDesc),
            let atlasB = device.makeTexture(descriptor: texDesc)
        else {
            Self.log.error("setSurfaceCacheCards: allocation failed")
            return
        }
        cb.label = "Illuminatorama.surfcache.cards"
        crb.label = "Illuminatorama.surfcache.cardRect"
        txb.label = "Illuminatorama.surfcache.texelCard"
        atlasA.label = "Illuminatorama.surfcache.atlasA"
        atlasB.label = "Illuminatorama.surfcache.atlasB"
        surfCardBuffer = cb
        surfTriCardBuffer = tcb
        surfTriUVaBuffer = tab
        surfTriUVcBuffer = tcc
        surfCardRectBuffer = crb
        surfTexelCardBuffer = txb
        // Incremental-invalidation per-card dirty buffer (all clean initially).
        surfCardDirtyBuffer = device.makeBuffer(
            length: max(1, MemoryLayout<UInt32>.stride * cards.count), options: .storageModeShared)
        surfCardDirtyBuffer?.label = "Illuminatorama.surfcache.cardDirty"
        if let d = surfCardDirtyBuffer { memset(d.contents(), 0, d.length) }
        surfIncrementalReady = false   // bake site repopulates the CPU maps
        surfChartIncremental = nil     // ditto (chart path repopulates if it bakes)
        surfChartMovedScratch = []     // re-sized at the chart bake site
        // GPU-diff maps are topology-keyed too; the bake site rebuilds them.
        surfGPUDiffTriOwnerBuffer = nil; surfGPUDiffTriFlagsBuffer = nil
        surfGPUDiffObjVertsBuffer = nil; surfGPUDiffTriCount = 0
        surfGPUDiffChartOwnerBuffer = nil; surfGPUDiffChartFrameBuffer = nil
        surfGPUDiffDeformIdxBuffer = nil; surfGPUDiffDeformDispatches = []
        surfAtlasA = atlasA
        surfAtlasB = atlasB
        // Phase 5 / B1 — denoiser output (same descriptor; not ping-ponged). nil on
        // alloc failure just leaves consumers on the raw atlas (the filter no-ops).
        surfAtlasDenoised = device.makeTexture(descriptor: texDesc)
        surfAtlasDenoised?.label = "Illuminatorama.surfcache.atlasDenoised"
        // Phase 5 / A — residency feedback buffer (uint per card; 1 = sampled).
        // Shared so the host can drain + zero it per frame. nil ⇒ feedback no-ops.
        surfCardRequestedBuffer = device.makeBuffer(
            length: max(1, MemoryLayout<UInt32>.stride * cards.count), options: .storageModeShared)
        surfCardRequestedBuffer?.label = "Illuminatorama.surfcache.cardRequested"
        if let r = surfCardRequestedBuffer { memset(r.contents(), 0, r.length) }
        surfCardCount = cards.count
        surfTilesPerRow = 1
        surfAtlasW = atlasW
        surfAtlasH = atlasH
        surfUseAtlasA = true
        let fillPct = Int((Double(totalArea) / Double(atlasW * atlasH) * 100).rounded())
        let budgetNote = residentCount < cards.count ? " [budget: \(residentCount)/\(cards.count) resident]" : ""
        Self.log.notice("Surface cache: \(cards.count) cards, \(atlasW)×\(atlasH) atlas (\(fillPct)% packed)\(budgetNote)")
        Self.recordSurfCacheStats("cards=\(cards.count) resident=\(residentCount) tris=\(rtTriangleCount) adaptive=\(adaptiveTiles) atlas=\(atlasW)x\(atlasH) texels=\(atlasW * atlasH) used=\(totalArea) fill=\(fillPct)% bytesRGBA16F=\(atlasW * atlasH * 8)")
    }

    /// Whether the surface-cache update should run this frame.
    private var surfCacheActive: Bool {
        surfaceCacheEnabled && rtEnabled && rtSupported
            && surfCachePipeline != nil && surfCardCount > 0 && rtAccel != nil
    }

    /// Read a deforming (DynamicMesh / GPU-fed) mesh's CURRENT object-space
    /// triangle soup directly from the shared buffers the GPU traces: live
    /// positions from the repack task's `packed_float3` source buffer (the host
    /// or a kernel rewrites it every frame), indices from the mesh's combined
    /// index buffer. `IlluminatoramaMesh.objectTriangleSoup()` can't do this — the
    /// interleaved vertex buffer is `.private`, so it returns empty. Returns nil
    /// for a non-deforming kind, a private source buffer, or empty topology.
    private func liveDeformingObjectSoup(kind: MeshKind)
        -> (positions: [SIMD3<Float>], indices: [UInt32])?
    {
        guard let mesh = meshes[kind],
              let task = gpuRepackTasks.first(where: { $0.kind == kind }),
              task.positionBuffer.storageMode == .shared,
              mesh.indexBuffer.storageMode == .shared else { return nil }
        let n = task.vertexCount
        guard n > 0, mesh.indexCount >= 3 else { return nil }
        // Source positions are packed_float3 (stride 12) — the layout the repack
        // kernel + BLAS read. Reading them as SIMD3<Float> (stride 16) would shear
        // every vertex past the first (the SIMD3 stride trap).
        let pbase = task.positionBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        var positions = [SIMD3<Float>](); positions.reserveCapacity(n)
        for v in 0..<n { positions.append(SIMD3<Float>(pbase[3*v], pbase[3*v+1], pbase[3*v+2])) }
        let ibase = mesh.indexBuffer.contents()
        var indices = [UInt32](); indices.reserveCapacity(mesh.indexCount)
        if mesh.indexType == .uint16 {
            for i in 0..<mesh.indexCount { indices.append(UInt32(ibase.load(fromByteOffset: i * 2, as: UInt16.self))) }
        } else {
            for i in 0..<mesh.indexCount { indices.append(ibase.load(fromByteOffset: i * 4, as: UInt32.self)) }
        }
        return (positions, indices)
    }

    /// Re-frame dynamic surface cards in place each frame, BEFORE
    /// `encodeSurfaceCacheUpdate`, so the update kernel re-lights moved/deformed
    /// surfaces at their CURRENT pose. Two dynamic sources, one shared mechanism
    /// (write the card's plane + flag it dirty → kernel uses α=1 to discard stale
    /// history while stationary cards keep their accumulated multi-bounce):
    ///
    ///  • **Rigid incremental invalidation** (`surfaceCacheIncremental`, default
    ///    on): an instance whose model matrix changed re-frames from its STATIC
    ///    object soup × the new transform. See
    ///    docs/illuminatorama/surface-cache-incremental-invalidation.md.
    ///  • **Phase C — deforming meshes** (DynamicMesh / GPU-fed): a mesh whose
    ///    *shape* changes in place (the transform may never move, so the rigid
    ///    diff misses it). Re-frames from this frame's LIVE object soup ×
    ///    transform, every frame. This is what makes a rippling sheet / sloshing
    ///    fluid / waving flag both CATCH the room's bounced colour (its cards sit
    ///    on the real surface, sampling neighbours' caches) and CAST it back
    ///    (its now-correctly-placed cards re-light and are sampled by others'
    ///    GI/reflection rays — and, on the TLAS update path, with current-pose
    ///    occlusion). On by default whenever a deforming mesh drives the cache;
    ///    `VIZ_ILLUMI_NO_DEFORM_CARDS=1` freezes it at build pose for A/B.
    ///
    /// PERF (Phase C → D): the deforming re-frame was a per-frame CPU readback +
    /// transform of the live soup; Phase D moved it to
    /// `illumi_surfcache_reframe_deform` (a compute kernel writing
    /// `surfCardBuffer` straight from the solver's live position buffer). The
    /// CPU loop below remains only as the `VIZ_SURFCACHE_GPU_DIFF=0` fallback.
    ///
    /// `surfCardBuffer` is written in place (not ping-ponged) — SAFE only because
    /// `maxFramesInFlight == 1` (the previous frame's GPU read has completed
    /// before this CPU write begins). If that pool is ever raised to ≥2, ping-pong
    /// the card buffer or this becomes a read-during-write hazard.
    ///
    /// **#60 item 1 + Phase D — GPU-side transform diff.** Both dynamic sources
    /// now run as compute kernels on the frame's command buffer: RIGID movers
    /// via `illumi_surfcache_reframe_tri` / `_chart` (each thread diffs its
    /// owner instance's current-vs-previous model matrix inline and re-frames
    /// its card (half) in place), DEFORMING meshes via
    /// `illumi_surfcache_reframe_deform` (one dispatch per deforming instance,
    /// reading the solver's live position buffer — no CPU readback). All write
    /// the same `surfCardBuffer` / dirty flags the update kernel consumes later
    /// in the SAME command buffer (hazard tracking orders them). The CPU loops
    /// below remain solely as the `VIZ_SURFCACHE_GPU_DIFF=0` kill-switch / A/B
    /// reference (and the safety fallback when a map or live buffer is
    /// unresolvable mid-topology).
    private func applyIncrementalSurfaceCacheUpdate(_ cb: MTLCommandBuffer) {
        guard surfCacheActive, surfIncrementalReady,
              let dirtyBuf = surfCardDirtyBuffer, let cardBuf = surfCardBuffer else { return }
        // Chart path (`VIZ_SURFCACHE_CHARTS=1`): re-frame each moved rigid instance's
        // cards by a single affine transform of their baked object-space frame.
        if let chart = surfChartIncremental {
            if surfaceCacheIncremental, !Self.surfCacheGPUDiffDisabled,
               let pipe = surfReframeChartPipeline,
               let ownerBuf = surfGPUDiffChartOwnerBuffer,
               let frameBuf = surfGPUDiffChartFrameBuffer,
               let statsBuf = surfGPUDiffStatsBuffer,
               chart.instCardLo.count == instances.count {
                encodeGPUDiffReframe(cb, pipeline: pipe, threadCount: surfCardCount,
                                     dirtyBuf: dirtyBuf, cardBuf: cardBuf,
                                     ownerBuf: ownerBuf, statsBuf: statsBuf,
                                     chartFrameBuf: frameBuf, tag: "[chart gpu]")
                return
            }
            applyIncrementalChartSurfaceCacheUpdate(chart, dirtyBuf: dirtyBuf, cardBuf: cardBuf)
            return
        }
        // Per-triangle path: GPU kernels (#60 item 1 + Phase D). The rigid
        // kernel owns every non-deforming card's diff + re-frame + dirty flag;
        // one deform dispatch per deforming instance re-frames its cards from
        // the LIVE position buffer (no readback). Falls back to the CPU loop
        // below when the kill-switch is set or any map / pipeline / live
        // buffer is missing.
        if !Self.surfCacheGPUDiffDisabled,
           encodeGPUDiffPerTriPath(cb, dirtyBuf: dirtyBuf, cardBuf: cardBuf) {
            return
        }
        guard surfSoupBaseCPU.count == instances.count,        // topology matches the baked maps
              !surfTriCardCPU.isEmpty else { return }

        // Which kinds deform in place this frame? (Phase C re-frames them always.)
        let deformingKinds: Set<MeshKind> =
            (surfHasDeformingCards && !noDeformCardsOverride) ? Set(gpuRepackTasks.map(\.kind)) : []
        let rigidIncremental = surfaceCacheIncremental
        // Nothing dynamic → leave every card fully accumulating (cheapest path).
        guard rigidIncremental || !deformingKinds.isEmpty else { return }

        // Snapshot each deforming kind's live object soup ONCE (shared by all its
        // instances). Indices are constant topology; positions are this frame's.
        var liveSoupByKind: [MeshKind: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
        for kind in deformingKinds {
            if let soup = liveDeformingObjectSoup(kind: kind) { liveSoupByKind[kind] = soup }
        }

        let total = instances.count
        let curPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let prevPtr = previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)

        // Clear last frame's dirty flags; cards that stop moving revert to EMA.
        memset(dirtyBuf.contents(), 0, dirtyBuf.length)
        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: surfCardCount)
        let cards = cardBuf.contents().bindMemory(to: SurfCard.self, capacity: surfCardCount)
        let soupTriTotal = surfTriCardCPU.count

        func moved(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
            let e: Float = 1e-5
            for c in 0..<4 {
                let d = a[c] - b[c]
                if abs(d.x) > e || abs(d.y) > e || abs(d.z) > e || abs(d.w) > e { return true }
            }
            return false
        }

        var movedInstances = 0
        var deformedInstances = 0
        var dirtiedCards = 0
        for i in 0..<total {
            let kind = surfInstanceKind[i]
            let isDeforming = deformingKinds.contains(kind)
            let model = curPtr[i].modelMatrix
            // Re-frame when: a deforming instance (its verts changed — always), OR
            // a rigid instance whose transform moved (incremental invalidation).
            let soup: (positions: [SIMD3<Float>], indices: [UInt32])?
            if isDeforming {
                soup = liveSoupByKind[kind]
                deformedInstances += 1
            } else {
                guard rigidIncremental, moved(model, prevPtr[i].modelMatrix) else { continue }
                soup = surfObjectSoupByKind[kind]
                movedInstances += 1
            }
            guard let soup, !soup.indices.isEmpty else { continue }
            let firstTri = Int(surfSoupBaseCPU[i])
            let nextTri = (i + 1 < total) ? Int(surfSoupBaseCPU[i + 1]) : soupTriTotal
            let localCount = min(nextTri - firstTri, soup.indices.count / 3)
            for j in 0..<max(0, localCount) {
                let gt = firstTri + j
                guard gt < soupTriTotal else { break }
                let i0 = Int(soup.indices[3*j]), i1 = Int(soup.indices[3*j+1]), i2 = Int(soup.indices[3*j+2])
                guard i0 < soup.positions.count, i1 < soup.positions.count, i2 < soup.positions.count else { continue }
                func world(_ p: SIMD3<Float>) -> SIMD3<Float> { let w = model * SIMD4<Float>(p, 1); return SIMD3(w.x, w.y, w.z) }
                let v0 = world(soup.positions[i0]), v1 = world(soup.positions[i1]), v2 = world(soup.positions[i2])
                var n = simd_cross(v1 - v0, v2 - v0)
                let len = simd_length(n); n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
                let card = Int(surfTriCardCPU[gt])
                guard card < surfCardCount else { continue }
                // Re-frame only this triangle's half of its (possibly packed) card;
                // preserve the packed flag (normal.w) + albedo/emission.
                if surfTriIsFrameB[gt] {
                    cards[card].originB = SIMD4(v0, 0); cards[card].uAxisB = SIMD4(v1 - v0, 0)
                    cards[card].vAxisB = SIMD4(v2 - v0, 0); cards[card].normalB = SIMD4(n, cards[card].normalB.w)
                } else {
                    cards[card].origin = SIMD4(v0, 0); cards[card].uAxis = SIMD4(v1 - v0, 0)
                    cards[card].vAxis = SIMD4(v2 - v0, 0); cards[card].normal = SIMD4(n, cards[card].normal.w)
                }
                if dirty[card] == 0 { dirtiedCards += 1 }
                dirty[card] = 1
            }
        }
        // Diagnostic (env-gated, sandbox-safe sidecar): proves the dynamic path
        // fired this frame and how much it touched. No-op without
        // VIZ_ILLUMI_SURFCACHE_STATS_PATH.
        if movedInstances > 0 || deformedInstances > 0 {
            Self.recordSurfCacheStats("dynamic: moved=\(movedInstances) deformed=\(deformedInstances) dirtiedCards=\(dirtiedCards) of \(surfCardCount)")
        }
    }

    /// Chart-path counterpart of `applyIncrementalSurfaceCacheUpdate` (#60 task 1).
    ///
    /// A coplanar chart card spans many triangles under ONE bbox-derived frame, so
    /// the per-triangle re-frame would clobber its parameterisation. Instead we use
    /// the card frame's **affine covariance**: for a rigid/affine model matrix `M`,
    /// the world frame is exactly `M · objectFrame` — `origin` as a point, `uAxis`
    /// / `vAxis` as vectors — and the fixed per-triangle UVs are preserved (verts
    /// and frame transform together). So each moved instance's cards re-frame with
    /// one matrix multiply per axis, no per-triangle work, no UV touch. Cards owned
    /// by a moved instance get α=1 (dirty) so they re-light from the new pose;
    /// every stationary card keeps its converged radiance.
    private func applyIncrementalChartSurfaceCacheUpdate(_ chart: ChartIncremental,
                                                         dirtyBuf: MTLBuffer, cardBuf: MTLBuffer) {
        // Charts are rigid-mover only — the deforming per-card path doesn't apply.
        guard surfaceCacheIncremental else { return }
        let total = instances.count
        guard chart.instCardLo.count == total, chart.instCardHi.count == total,
              chart.objectFrame.count == surfCardCount else { return }
        let curPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let prevPtr = previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)

        func moved(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
            let e: Float = 1e-5
            for c in 0..<4 {
                let d = a[c] - b[c]
                if abs(d.x) > e || abs(d.y) > e || abs(d.z) > e || abs(d.w) > e { return true }
            }
            return false
        }

        // Which instances moved this frame? (Decide before clearing dirty flags.)
        // Reuse the bake-time scratch buffer (no per-frame allocation); reset in
        // place. Fall back to a fresh buffer only if it's somehow mis-sized.
        if surfChartMovedScratch.count != total {
            surfChartMovedScratch = [Bool](repeating: false, count: total)
        } else {
            for i in 0..<total { surfChartMovedScratch[i] = false }
        }
        var movedInstances = 0
        for i in 0..<total where moved(curPtr[i].modelMatrix, prevPtr[i].modelMatrix) {
            surfChartMovedScratch[i] = true; movedInstances += 1
        }
        // Clear last frame's dirty flags so cards that stopped moving revert to EMA
        // (same reset semantics as the per-triangle path); skip the rest if static.
        memset(dirtyBuf.contents(), 0, dirtyBuf.length)
        guard movedInstances > 0 else { return }

        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: surfCardCount)
        let cards = cardBuf.contents().bindMemory(to: SurfCard.self, capacity: surfCardCount)
        var dirtiedCards = 0
        for i in 0..<total where surfChartMovedScratch[i] {
            let lo = Int(chart.instCardLo[i]), hi = Int(chart.instCardHi[i])
            guard lo >= 0, hi > lo, hi <= surfCardCount else { continue }
            let m = curPtr[i].modelMatrix
            for c in lo..<hi {
                let f = chart.objectFrame[c]
                let o = m * SIMD4<Float>(f.origin, 1)
                let u = m * SIMD4<Float>(f.uAxis, 0)
                let v = m * SIMD4<Float>(f.vAxis, 0)
                let nn = m * SIMD4<Float>(f.normal, 0)
                var n = SIMD3<Float>(nn.x, nn.y, nn.z)
                let nl = simd_length(n); n = nl > 1e-8 ? n / nl : SIMD3<Float>(0, 1, 0)
                cards[c].origin = SIMD4(o.x, o.y, o.z, 0)
                cards[c].uAxis  = SIMD4(u.x, u.y, u.z, 0)
                cards[c].vAxis  = SIMD4(v.x, v.y, v.z, 0)
                cards[c].normal = SIMD4(n, cards[c].normal.w)   // preserve packed flag (0 for charts)
                if dirty[c] == 0 { dirtiedCards += 1 }
                dirty[c] = 1
            }
        }
        Self.recordSurfCacheStats("dynamic: moved=\(movedInstances) deformed=0 dirtiedCards=\(dirtiedCards) of \(surfCardCount) [chart]")
    }

    /// Bake the per-triangle GPU-diff maps (#60 item 1 + Phase D): owner
    /// instance, flag bits (frame-B half / deforming), OBJECT-space verts (rigid
    /// re-frame), and vertex-index triples (deform re-frame) per grouped-soup
    /// triangle, accumulated in lockstep with the soup build so global triangle
    /// order matches exactly. Any mismatch leaves the buffers nil → the CPU
    /// re-frame path stays in charge.
    private func buildGPUDiffPerTriBuffers(triOwner: [UInt32],
                                           objVerts: [SIMD4<Float>],
                                           isFrameB: [Bool],
                                           triIdx: [SIMD4<UInt32>],
                                           deforming: [Bool]) {
        surfGPUDiffChartOwnerBuffer = nil; surfGPUDiffChartFrameBuffer = nil
        surfGPUDiffTriOwnerBuffer = nil; surfGPUDiffTriFlagsBuffer = nil
        surfGPUDiffObjVertsBuffer = nil; surfGPUDiffTriCount = 0
        surfGPUDiffDeformIdxBuffer = nil; surfGPUDiffDeformDispatches = []
        let triCount = triOwner.count
        guard triCount > 0, triCount == rtTriangleCount,
              objVerts.count == triCount * 3, isFrameB.count == triCount,
              triIdx.count == triCount, deforming.count == triCount,
              surfReframeTriPipeline != nil else { return }
        var flags = [UInt32](repeating: 0, count: triCount)
        for t in 0..<triCount {
            flags[t] = (isFrameB[t] ? Self.scFlagFrameB : 0)
                     | (deforming[t] ? Self.scFlagDeforming : 0)
        }
        guard let ob = device.makeBuffer(bytes: triOwner,
                  length: MemoryLayout<UInt32>.stride * triCount, options: .storageModeShared),
              let fb = device.makeBuffer(bytes: flags,
                  length: MemoryLayout<UInt32>.stride * triCount, options: .storageModeShared),
              let vb = device.makeBuffer(bytes: objVerts,
                  length: MemoryLayout<SIMD4<Float>>.stride * objVerts.count, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: triIdx,
                  length: MemoryLayout<SIMD4<UInt32>>.stride * triCount, options: .storageModeShared)
        else { return }
        ob.label = "Illuminatorama.surfcache.gpuDiff.triOwner"
        fb.label = "Illuminatorama.surfcache.gpuDiff.triFlags"
        vb.label = "Illuminatorama.surfcache.gpuDiff.objVerts"
        ib.label = "Illuminatorama.surfcache.gpuDiff.triIdx"
        surfGPUDiffTriOwnerBuffer = ob
        surfGPUDiffTriFlagsBuffer = fb
        surfGPUDiffObjVertsBuffer = vb
        surfGPUDiffDeformIdxBuffer = ib
        surfGPUDiffTriCount = triCount
        if let sb = surfGPUDiffStatsBuffer { memset(sb.contents(), 0, sb.length) }
    }

    /// Chart-path GPU-diff maps (#60 item 1): per-card owner instance (−1 =
    /// static / degenerate, never re-framed) + bake-pose object-space frame,
    /// flattened to 4×float4 per card for the kernel.
    private func buildGPUDiffChartBuffers(objFrames: [SurfCardFrame], cardOwner: [Int32]) {
        surfGPUDiffTriOwnerBuffer = nil; surfGPUDiffTriFlagsBuffer = nil
        surfGPUDiffObjVertsBuffer = nil; surfGPUDiffTriCount = 0
        surfGPUDiffChartOwnerBuffer = nil; surfGPUDiffChartFrameBuffer = nil
        guard surfCardCount > 0, objFrames.count == surfCardCount,
              cardOwner.count == surfCardCount,
              surfReframeChartPipeline != nil else { return }
        var frames = [SIMD4<Float>](); frames.reserveCapacity(surfCardCount * 4)
        for f in objFrames {
            frames.append(SIMD4(f.origin, 0)); frames.append(SIMD4(f.uAxis, 0))
            frames.append(SIMD4(f.vAxis, 0));  frames.append(SIMD4(f.normal, 0))
        }
        guard let ob = device.makeBuffer(bytes: cardOwner,
                  length: MemoryLayout<Int32>.stride * cardOwner.count, options: .storageModeShared),
              let fb = device.makeBuffer(bytes: frames,
                  length: MemoryLayout<SIMD4<Float>>.stride * frames.count, options: .storageModeShared)
        else { return }
        ob.label = "Illuminatorama.surfcache.gpuDiff.cardOwner"
        fb.label = "Illuminatorama.surfcache.gpuDiff.cardFrame"
        surfGPUDiffChartOwnerBuffer = ob
        surfGPUDiffChartFrameBuffer = fb
        if let sb = surfGPUDiffStatsBuffer { memset(sb.contents(), 0, sb.length) }
    }

    /// Phase D + #60 item 1 — the full GPU per-triangle dynamic-card pass.
    /// Returns false (CPU fallback) unless every map, pipeline, and deforming
    /// instance's live position buffer is resolvable; on success encodes the
    /// rigid kernel (diff + re-frame + dirty ownership for every non-deforming
    /// card) followed by one deform dispatch per deforming instance.
    private func encodeGPUDiffPerTriPath(_ cb: MTLCommandBuffer,
                                         dirtyBuf: MTLBuffer, cardBuf: MTLBuffer) -> Bool {
        let deformingActive = surfHasDeformingCards && !noDeformCardsOverride
        guard surfaceCacheIncremental || deformingActive,
              let pipe = surfReframeTriPipeline,
              let ownerBuf = surfGPUDiffTriOwnerBuffer,
              let flagsBuf = surfGPUDiffTriFlagsBuffer,
              let vertsBuf = surfGPUDiffObjVertsBuffer,
              let statsBuf = surfGPUDiffStatsBuffer,
              let triCardBuf = surfTriCardBuffer,
              surfGPUDiffTriCount == rtTriangleCount,
              surfSoupBaseCPU.count == instances.count else { return false }
        // Resolve every deform dispatch's live position buffer up front — if any
        // is missing (solver unregistered ahead of the re-bake), take the CPU
        // path this frame rather than half-updating the dirty flags.
        var deformBindings: [(d: SurfDeformDispatch, positions: MTLBuffer, vertexCount: Int)] = []
        if deformingActive {
            guard surfReframeDeformPipeline != nil, surfGPUDiffDeformIdxBuffer != nil
            else { return false }
            for d in surfGPUDiffDeformDispatches {
                guard d.instanceIndex < instances.count,
                      let task = gpuRepackTasks.first(where: { $0.kind == d.kind })
                else { return false }
                deformBindings.append((d, task.positionBuffer, task.vertexCount))
            }
        }
        encodeGPUDiffReframe(cb, pipeline: pipe, threadCount: surfGPUDiffTriCount,
                             dirtyBuf: dirtyBuf, cardBuf: cardBuf,
                             ownerBuf: ownerBuf, statsBuf: statsBuf,
                             rigidEnabled: surfaceCacheIncremental,
                             triCardBuf: triCardBuf, triFlagsBuf: flagsBuf,
                             triVertsBuf: vertsBuf, tag: "[gpu]")
        if !deformBindings.isEmpty {
            encodeGPUDiffDeform(cb, bindings: deformBindings,
                                dirtyBuf: dirtyBuf, cardBuf: cardBuf, statsBuf: statsBuf)
        }
        return true
    }

    /// Phase D — encode the deforming-instance re-frames: one dispatch per
    /// deforming instance, threads = its local soup triangles, reading the
    /// solver's LIVE packed_float3 position buffer through the baked
    /// vertex-index triples. Runs after the rigid kernel (which skips
    /// deform-flagged triangles, so this pass solely owns these cards).
    private func encodeGPUDiffDeform(_ cb: MTLCommandBuffer,
                                     bindings: [(d: SurfDeformDispatch, positions: MTLBuffer, vertexCount: Int)],
                                     dirtyBuf: MTLBuffer, cardBuf: MTLBuffer,
                                     statsBuf: MTLBuffer) {
        guard let pipeline = surfReframeDeformPipeline,
              let idxBuf = surfGPUDiffDeformIdxBuffer,
              let triCardBuf = surfTriCardBuffer,
              let flagsBuf = surfGPUDiffTriFlagsBuffer else { return }
        guard let enc = timedComputeEncoder(cb, "surfcacheReframeDeform") else { return }
        enc.label = "Illuminatorama.surfcache.gpuDiff.deform"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(cardBuf, offset: 0, index: 0)
        enc.setBuffer(dirtyBuf, offset: 0, index: 1)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setBuffer(triCardBuf, offset: 0, index: 3)
        enc.setBuffer(flagsBuf, offset: 0, index: 4)
        enc.setBuffer(idxBuf, offset: 0, index: 5)
        enc.setBuffer(statsBuf, offset: 0, index: 8)
        for b in bindings {
            var u = SCDeformUniforms(triBase: UInt32(b.d.triBase),
                                     triCount: UInt32(b.d.triCount),
                                     instanceIndex: UInt32(b.d.instanceIndex),
                                     vertexCount: UInt32(b.vertexCount))
            enc.setBuffer(b.positions, offset: 0, index: 6)
            enc.setBytes(&u, length: MemoryLayout<SCDeformUniforms>.stride, index: 7)
            dispatch1D(enc, pipeline: pipeline, count: b.d.triCount)
        }
        enc.endEncoding()
    }

    /// Encode the GPU-side moved-set detection + re-frame (#60 item 1). One
    /// dispatch; the surface-cache update kernel later in the same command
    /// buffer reads the re-framed cards + dirty flags (hazard tracking orders
    /// the two). Also drains LAST frame's atomic stats into the diagnostic
    /// sidecar — safe to read on the CPU here because `maxFramesInFlight == 1`
    /// guarantees that command buffer has completed.
    private func encodeGPUDiffReframe(_ cb: MTLCommandBuffer,
                                      pipeline: MTLComputePipelineState,
                                      threadCount: Int,
                                      dirtyBuf: MTLBuffer, cardBuf: MTLBuffer,
                                      ownerBuf: MTLBuffer, statsBuf: MTLBuffer,
                                      rigidEnabled: Bool = true,
                                      chartFrameBuf: MTLBuffer? = nil,
                                      triCardBuf: MTLBuffer? = nil,
                                      triFlagsBuf: MTLBuffer? = nil,
                                      triVertsBuf: MTLBuffer? = nil,
                                      tag: String) {
        guard threadCount > 0 else { return }
        let stats = statsBuf.contents().bindMemory(to: UInt32.self, capacity: 4)
        if stats[0] > 0 || stats[1] > 0 || stats[2] > 0 {
            Self.recordSurfCacheStats("dynamic: moved=\(stats[0]) deformed=\(stats[2]) dirtiedCards=\(stats[1]) of \(surfCardCount) \(tag)")
        }
        stats[0] = 0; stats[1] = 0; stats[2] = 0; stats[3] = 0
        var u = SCReframeUniforms(count: UInt32(threadCount),
                                  cardCount: UInt32(surfCardCount),
                                  instanceCount: UInt32(instances.count),
                                  epsilon: 1e-5,   // matrix-diff threshold, mirrors the CPU loop
                                  rigidEnabled: rigidEnabled ? 1 : 0)
        guard let enc = timedComputeEncoder(cb, "surfcacheReframe") else { return }
        enc.label = "Illuminatorama.surfcache.gpuDiff"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(cardBuf, offset: 0, index: 0)
        enc.setBuffer(dirtyBuf, offset: 0, index: 1)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setBuffer(previousInstanceBuffer, offset: 0, index: 3)
        enc.setBuffer(ownerBuf, offset: 0, index: 4)
        if let triCardBuf, let triFlagsBuf, let triVertsBuf {
            enc.setBuffer(triCardBuf, offset: 0, index: 5)
            enc.setBuffer(triFlagsBuf, offset: 0, index: 6)
            enc.setBuffer(triVertsBuf, offset: 0, index: 7)
            enc.setBytes(&u, length: MemoryLayout<SCReframeUniforms>.stride, index: 8)
            enc.setBuffer(statsBuf, offset: 0, index: 9)
        } else if let chartFrameBuf {
            enc.setBuffer(chartFrameBuf, offset: 0, index: 5)
            enc.setBytes(&u, length: MemoryLayout<SCReframeUniforms>.stride, index: 6)
            enc.setBuffer(statsBuf, offset: 0, index: 7)
        }
        dispatch1D(enc, pipeline: pipeline, count: threadCount)
        enc.endEncoding()
    }

    /// Re-light the on-surface radiance atlas (ping-pong) before the RT pass.
    /// Reads PREVIOUS atlas for the indirect bounce (→ multi-bounce over frames);
    /// writes CURRENT. The RT pass then samples CURRENT at its hits.
    private func encodeSurfaceCacheUpdate(_ cb: MTLCommandBuffer) {
        guard surfCacheActive,
              let cards = surfCardBuffer, let tc = surfTriCardBuffer,
              let ta = surfTriUVaBuffer, let tcc = surfTriUVcBuffer,
              let crect = surfCardRectBuffer, let txcard = surfTexelCardBuffer,
              let cdirty = surfCardDirtyBuffer else { return }
        // §3 endpoint — trace the per-frame-refit TLAS when one is live (a moved
        // object's shadow/bounce-occlusion then tracks its motion); otherwise fall
        // back to the static-soup primitive AS (the soup-path scenes, no TLAS).
        let useTLAS = rtTLASActive && surfCacheTLASPipeline != nil
            && rtTLAS != nil && rtSoupTriBaseBuffer != nil
            && rtSoupTriBaseCount == instances.count
        let pipeline: MTLComputePipelineState
        let accel: MTLAccelerationStructure
        if useTLAS {
            // Curve primitives (#60 item 7): a TLAS holding curve instances
            // must be traced with the curve-contract variant (curve hits then
            // occlude sun rays and terminate indirect rays — see the kernel).
            if rtCurveInstanceCount > 0 {
                guard let cp = surfCacheTLASCurvePipeline else { return }
                pipeline = cp
            } else {
                pipeline = surfCacheTLASPipeline!
            }
            accel = rtTLAS!
        } else {
            guard let soupAccel = rtAccel else { return }
            // Curve primitives (#60 item 7 incr. 2): a soup AS holding a curve
            // geometry descriptor must be traced with the curve-contract variant
            // (curve hits occlude the sun ray + terminate indirect rays).
            if rtSoupCurvesActive, let cp = surfCacheSoupCurvePipeline {
                pipeline = cp
            } else if let soupPipe = surfCachePipeline {
                pipeline = soupPipe
            } else { return }
            accel = soupAccel
        }
        // Honour per-card dirty flags (α=1 → re-light fresh at the re-framed pose)
        // for either dynamic source: rigid incremental invalidation OR Phase-C
        // deforming meshes (whose cards `applyIncrementalSurfaceCacheUpdate`
        // re-frames every frame). Both require the per-triangle CPU maps.
        let incremental = (surfaceCacheIncremental || surfHasDeformingCards) && surfIncrementalReady

        // Ping-pong swap at the START (mirrors DDGI): Current is what we write
        // AND what the RT pass reads after we return; Previous is the bounce
        // source. First frame after (re)alloc both are empty → indirect = 0.
        surfUseAtlasA.toggle()
        guard let curAtlas = surfCacheCurrentAtlas,
              let prevAtlas = surfCachePreviousAtlas else { return }

        surfFrameSeed &+= 1
        var u = SurfCacheUniforms(
            sunDir: SIMD4(simd_normalize(rtSunDirection), 0),
            sunColor: SIMD4(rtSunColor, 0),
            skyAmbient: SIMD4(ambientColor, 0),
            atlasW: UInt32(surfAtlasW), atlasH: UInt32(surfAtlasH),
            tileSize: UInt32(surfActiveTileSize), tilesPerRow: UInt32(surfTilesPerRow),
            cardCount: UInt32(surfCardCount), triangleCount: UInt32(rtTriangleCount),
            indirectRays: UInt32(max(1, min(16, surfCacheIndirectRays))),
            frameSeed: surfFrameSeed,
            alpha: max(0.02, min(1.0, surfCacheAlpha)), rayTMin: 0.004, maxDist: 60.0,
            incrementalEnabled: incremental ? 1 : 0)
        memcpy(surfCacheUniformBuffer.contents(), &u, MemoryLayout<SurfCacheUniforms>.stride)

        guard let enc = timedComputeEncoder(cb, "surfcacheUpdate") else { return }
        enc.label = "Illuminatorama.surfcache"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(curAtlas, index: 0)
        enc.setTexture(prevAtlas, index: 1)   // read access
        enc.setTexture(prevAtlas, index: 2)   // sample access
        enc.setTexture(equirectSky ?? dummySkyTexture, index: 3)
        enc.setAccelerationStructure(accel, bufferIndex: 0)
        enc.setBuffer(cards, offset: 0, index: 1)
        enc.setBuffer(tc, offset: 0, index: 2)
        enc.setBuffer(ta, offset: 0, index: 3)
        enc.setBuffer(tcc, offset: 0, index: 4)
        enc.setBuffer(surfCacheUniformBuffer, offset: 0, index: 5)
        enc.setBuffer(crect, offset: 0, index: 6)
        enc.setBuffer(txcard, offset: 0, index: 7)
        enc.setBuffer(cdirty, offset: 0, index: 8)
        if useTLAS {
            // TLAS hit → global soup tri via soupTriBase[instance_id]+primitive_id.
            // The TLAS references the BLASes, which reference the mesh vertex/index
            // buffers (and the curve BLASes the pooled curve buffers) — all must
            // be resident for the intersector.
            enc.setBuffer(rtSoupTriBaseBuffer, offset: 0, index: 9)
            for blas in rtBLASList { enc.useResource(blas, usage: .read) }
            for blas in rtCurveBLASList { enc.useResource(blas, usage: .read) }
            for buf in rtResidentBuffers { enc.useResource(buf, usage: .read) }
        } else if let vb = rtVertexBuffer, let ib = rtIndexBuffer {
            // Curve-only primitive AS at buffer(9) — the SOUP kernel's slot (the
            // TLAS variant uses 9 for soupTriBase; different pipeline). Dummy =
            // the triangle `accel` for the base (curve-free) variant.
            enc.setAccelerationStructure((rtSoupCurvesActive ? rtSoupCurveAccel : nil) ?? accel, bufferIndex: 9)
            enc.useResource(vb, usage: .read)
            enc.useResource(ib, usage: .read)
            // Curve AS + pool buffers the soup curve trace references (#60 item 7 incr. 2).
            if rtSoupCurvesActive {
                if let ca = rtSoupCurveAccel { enc.useResource(ca, usage: .read) }
                if let p = rtSoupCurvePoolPoints { enc.useResource(p, usage: .read) }
                if let r = rtSoupCurveRadii { enc.useResource(r, usage: .read) }
                if let s = rtSoupCurveSegments { enc.useResource(s, usage: .read) }
            }
        }
        dispatch(enc, pipeline: pipeline, width: surfAtlasW, height: surfAtlasH)
        enc.endEncoding()
    }

    // Phase 5 / B1 — cache-domain à-trous denoiser. Mirror of the Metal
    // `SurfAtrousUniforms` (stride 32).
    private struct SurfAtrousUniforms {
        var atlasW: UInt32; var atlasH: UInt32; var stepSize: UInt32; var radius: UInt32
        var phi: Float; var _ap0: Float = 0; var _ap1: Float = 0; var _ap2: Float = 0
    }

    /// Phase 5 / B1 — spatially denoise the just-updated radiance atlas into
    /// `surfAtlasDenoised`, which the GI/reflection consumers then read. Encoded
    /// AFTER `encodeSurfaceCacheUpdate` (current atlas written) and BEFORE the RT
    /// lighting pass (same command buffer ⇒ compute passes serialize). No-op when
    /// the denoiser is off, the pipeline is unavailable, or the textures aren't
    /// allocated — consumers fall back to the raw atlas via `surfConsumerAtlas`.
    private func encodeSurfaceCacheDenoise(_ cb: MTLCommandBuffer) {
        guard surfaceCacheDenoise,
              let pipeline = surfAtrousPipeline,
              let src = surfCacheCurrentAtlas,
              let dst = surfAtlasDenoised,
              let txcard = surfTexelCardBuffer,
              surfCardCount > 0 else { return }
        guard let enc = timedComputeEncoder(cb, "surfcacheDenoise") else { return }
        enc.label = "Illuminatorama.surfcache.atrous"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(dst, index: 0)   // write
        enc.setTexture(src, index: 1)   // read (current atlas)
        enc.setBuffer(txcard, offset: 0, index: 0)
        var u = SurfAtrousUniforms(
            atlasW: UInt32(surfAtlasW), atlasH: UInt32(surfAtlasH),
            stepSize: 1, radius: 2, phi: 4.0)
        enc.setBytes(&u, length: MemoryLayout<SurfAtrousUniforms>.stride, index: 1)
        dispatch(enc, pipeline: pipeline, width: surfAtlasW, height: surfAtlasH)
        enc.endEncoding()
    }

    /// Phase 5 / A1 — dynamic feedback-driven residency. Run once per frame BEFORE
    /// the cache update (so the update relights this frame's resident set) and AFTER
    /// the incremental re-frame (so a promoted card's dirty flag isn't clobbered by
    /// the rigid-mover pass). Reads the PREVIOUS frame's `cardRequested` working set
    /// (safe under maxFramesInFlight==1), promotes sampled-but-non-resident cards into
    /// slots freed by unsampled residents, and rewrites `texelCard` / `cardRect` /
    /// `dirty` in place. No kernel change — A0's zero-rect fallback already covers a
    /// card the instant it's demoted. Logs the structural success metric
    /// (working-set cards already resident / total) for verification.
    private func runStreamingResidency() {
        guard surfaceCacheStreaming, surfCardBudget > 0,
              surfResidentCardCount < surfCardCount,
              !surfSlotRect.isEmpty,
              surfCardSlot.count == surfCardCount,
              let reqBuf = surfCardRequestedBuffer,
              let rectBuf = surfCardRectBuffer,
              let texelBuf = surfTexelCardBuffer,
              let dirtyBuf = surfCardDirtyBuffer else { return }
        let req = reqBuf.contents().bindMemory(to: UInt32.self, capacity: surfCardCount)
        // A2 — refresh per-card heat from this frame's working set (sampled → full,
        // else cool by one). Done before demotability so a just-sampled card is hot.
        let heatOn = surfCardHeat.count == surfCardCount
        if heatOn {
            for c in 0..<surfCardCount {
                if req[c] != 0 { surfCardHeat[c] = surfStreamHeatFull }
                else if surfCardHeat[c] > 0 { surfCardHeat[c] -= 1 }
            }
        }
        // Working set (requested last frame) + how much of it was already resident
        // (the structural metric: → 100% as residency converges to the working set).
        var reqTotal = 0, reqResident = 0
        var promote: [Int] = []
        for c in 0..<surfCardCount where req[c] != 0 {
            reqTotal += 1
            if surfCardSlot[c] >= 0 { reqResident += 1 } else { promote.append(c) }
        }
        defer { memset(reqBuf.contents(), 0, surfCardCount * MemoryLayout<UInt32>.stride) }
        guard !promote.isEmpty else {
            if reqTotal > 0 {
                Self.recordSurfCacheStats("stream: resident=\(surfResidentCardCount) workingSet=\(reqTotal) residentHit=\(reqResident) promoted=0")
            }
            return
        }
        // Slots free to reassign: empty, OR (A2) occupant gone cold (heat 0). With
        // hysteresis off, falls back to "unsampled this frame" — the A1 behaviour.
        var demotable: [Int] = []
        for s in 0..<surfSlotCard.count {
            let occ = surfSlotCard[s]
            let cold = heatOn ? (occ >= 0 && surfCardHeat[occ] <= 0) : (occ >= 0 && req[occ] == 0)
            if occ < 0 || cold { demotable.append(s) }
        }
        // A4 — importance order: under contention (promote.count > demotable.count or
        // the swap cap), the most-important requested cards take the least-important
        // freed slots, so the resident set converges to the NEAR working set.
        if surfaceCacheLOD, surfCardCentroid.count == surfCardCount {
            let cam = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self).cameraWorldPos
            let d0sq: Float = 12 * 12
            func imp(_ c: Int) -> Float {
                let d = surfCardCentroid[c] - cam
                return 1.0 / (1.0 + simd_length_squared(d) / d0sq)
            }
            promote.sort { imp($0) > imp($1) }                 // nearest requested first
            demotable.sort {                                   // farthest / empty slot first
                let a = surfSlotCard[$0], b = surfSlotCard[$1]
                return (a >= 0 ? imp(a) : -1) < (b >= 0 ? imp(b) : -1)
            }
        }
        let rects = rectBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: surfCardCount)
        let texel = texelBuf.contents().bindMemory(to: UInt32.self, capacity: surfAtlasW * surfAtlasH)
        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: surfCardCount)
        let nSwap = min(promote.count, demotable.count, surfMaxSwapsPerFrame)
        for i in 0..<nSwap {
            let slot = demotable[i]
            let oldC = surfSlotCard[slot]
            let newC = promote[i]
            if oldC >= 0 { surfCardSlot[oldC] = -1; rects[oldC] = .zero      // demote → fallback
                if heatOn { surfCardHeat[oldC] = 0 } }
            surfSlotCard[slot] = newC
            surfCardSlot[newC] = slot
            if heatOn { surfCardHeat[newC] = surfStreamHeatFull }           // promoted ⇒ hot
            let r = surfSlotRect[slot]
            rects[newC] = r
            // Repoint the slot's atlas texels at the new occupant.
            let x0 = Int(r.x), y0 = Int(r.y), w = Int(r.z), h = Int(r.w)
            for yy in y0..<(y0 + h) {
                let rowBase = yy * surfAtlasW
                for xx in x0..<(x0 + w) { texel[rowBase + xx] = UInt32(newC) }
            }
            dirty[newC] = 1   // cold slot (held the demoted card's radiance) → relight fresh
        }
        // A4 verification metric: mean camera-distance of resident cards (lower with
        // LOD on under pressure ⇒ nearer cards held).
        var distNote = ""
        if surfCardCentroid.count == surfCardCount {
            let cam = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self).cameraWorldPos
            var sum: Float = 0; var n = 0
            for occ in surfSlotCard where occ >= 0 { sum += simd_length(surfCardCentroid[occ] - cam); n += 1 }
            if n > 0 { distNote = String(format: " meanResidentDist=%.2f lod=\(surfaceCacheLOD ? 1 : 0)", sum / Float(n)) }
        }
        Self.recordSurfCacheStats("stream: resident=\(surfResidentCardCount) workingSet=\(reqTotal) residentHit=\(reqResident) promoted=\(nSwap) demoted=\(nSwap)\(distNote)")
    }

    // ── TLAS update (instanced RT for animated extracted scenes) ──────
    //
    // Called from `render()` after `uploadInstances()` (so `meshGroups` +
    // `currentInstanceBuffer` are this frame's, grouped). On a topology change
    // (new mesh set / instance count) does a full BLAS+TLAS rebuild; otherwise
    // refits the TLAS in place on the frame's command buffer — cheap, no
    // per-frame vertex transform, no geometry rebuild. `rtTLASActive` tells the
    // RT pass to trace the instance AS instead of the primitive-AS soup.

    private func rtTopologyHash() -> Int {
        var h = instances.count
        for g in meshGroups { h = h &* 31 &+ g.kind.hashValue; h = h &* 31 &+ g.count }
        // Curve sets are TLAS topology too (#60 item 7). The registry version
        // covers identity changes (re-register ⇒ rebuild even at equal counts).
        h = h &* 31 &+ rtCurveSyncVersion
        h = h &* 31 &+ rtCurveSets.count
        for s in rtCurveSets {
            h = h &* 31 &+ s.segmentIndices.count
            h = h &* 31 &+ s.controlPoints.count
        }
        // AAA glass (#60): glass count + per-group kind/count are TLAS topology too
        // (a changed glass count must trigger a rebuild, not a refit).
        if rtGlassEnabled {
            for (kind, insts) in flattenGlass() {
                h = h &* 31 &+ kind.hashValue
                h = h &* 31 &+ insts.count
            }
        }
        return h
    }

    private func updateRTAccel(_ cb: MTLCommandBuffer) {
        // P1c — surface-cache scenes now ride the TLAS too: the lighting pass
        // reads cards via `soupTriBase[instance_id] + primitive_id`, while the
        // cache-UPDATE pass keeps tracing the grouped soup `rtAccel` that
        // `rebuildRTAccel`→`buildGroupedSurfaceSoup` bakes in TLAS-instance order.
        // Two triggers for a live TLAS: the extracted-scene RT path (opaque
        // lighting), OR AAA glass opted in (#60) — a native glass scene with no
        // opaque instances (Floating Lenses) still needs the TLAS for the glass
        // pass to trace. Both ride the same build below.
        let glassGroups = rtGlassEnabled ? flattenGlass() : []
        let hasGlass = !glassGroups.isEmpty
        let extractedRT = buildRTFromExtractedScene && rtEnabled
            && !meshGroups.isEmpty && !instances.isEmpty
        guard rtTLASSupported, !rtAutoDisabled, (extractedRT || hasGlass) else {
            rtTLASActive = false; return }
        // Curve primitives (#60 item 7): adopt registry changes BEFORE the
        // topology hash, so a registration/unregistration lands as a rebuild.
        syncCurveRegistry()
        // Hang-proof gate 1 (up-front size cap — the PRIMARY defence): a scene
        // this heavy would stall on its very first rebuild (which never returns
        // for minutes), so the thrash counter below could never catch it. Reject
        // BEFORE any build, based on a cheap cost estimate. Latches until the
        // next scene attach.
        var estTriangles = 0
        for g in meshGroups {
            if let m = meshes[g.kind] { estTriangles += m.indexCount / 3 }
        }
        var glassInstCount = 0
        for (kind, insts) in glassGroups {
            glassInstCount += insts.count
            if let m = meshes[kind] { estTriangles += m.indexCount / 3 }
        }
        if instances.count + glassInstCount > Self.rtMaxInstancesForLiveRT
            || meshGroups.count > Self.rtMaxMeshGroupsForLiveRT
            || estTriangles > Self.rtMaxTrianglesForLiveRT {
            rtAutoDisabled = true
            rtTLASActive = false
            Self.log.warning("""
                Illuminatorama RT auto-disabled — scene too heavy for live-loop \
                RT (\(self.instances.count) instances, \(self.meshGroups.count) mesh \
                groups, ~\(estTriangles) tris; caps \
                \(Self.rtMaxInstancesForLiveRT)/\(Self.rtMaxMeshGroupsForLiveRT)/\(Self.rtMaxTrianglesForLiveRT)). \
                Falling back to non-RT.
                """)
            return
        }
        let topo = rtTopologyHash()
        if topo != rtTLASTopologyHash || rtTLAS == nil {
            // Hang-proof gate 2 (rebuild-thrash): topology changing every frame
            // (GPU-fed / regenerated geometry — e.g. Forest's LeafField + bark +
            // swaying branches) ⇒ a full rebuild every frame ⇒ hang + faulted-AS
            // magenta. Allow a few legitimate settle frames, then disable.
            rtConsecutiveRebuilds += 1
            if rtConsecutiveRebuilds > Self.rtRebuildThrashLimit {
                rtAutoDisabled = true
                rtTLASActive = false
                Self.log.warning("""
                    Illuminatorama RT auto-disabled — TLAS rebuilt \
                    \(self.rtConsecutiveRebuilds) frames running (topology \
                    thrashing: GPU-fed / animated-topology scene). Falling back to non-RT.
                    """)
                return
            }
            rebuildRTAccel()
            rtTLASTopologyHash = topo
        } else {
            rtConsecutiveRebuilds = 0   // stable topology this frame → reset
            // Phase B — re-fit deforming BLASes from this frame's live vertices
            // BEFORE the TLAS refit so the RT passes trace current geometry.
            refitDeformingBLAS(cb)
            refitRTAccel(cb)
        }
    }

    /// Curve primitives (#60 item 7) — adopt the process-wide registry when its
    /// version moved: copy the sets, rebuild the pooled GPU buffers, and (on
    /// first use) compile the `kRTCurvesEnabled` kernel variants. The topology
    /// hash includes the sync version, so the caller's rebuild branch fires
    /// next. Curve support requires BOTH variant pipelines — if either fails
    /// to compile, the sets are dropped (a TLAS containing curves traced by a
    /// triangle-contract intersector is undefined behaviour, never shippable).
    private func syncCurveRegistry() {
        let reg = IlluminatoramaCurveRegistry.shared
        guard reg.version != rtCurveSyncVersion else { return }
        rtCurveSyncVersion = reg.version
        rtCurveSets = device.supportsRaytracing ? reg.sets : []
        if !rtCurveSets.isEmpty {
            let cache = engine.pipelineCache
            var on = true
            if rtTLASCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                rtTLASCurvePipeline = cache.pipelineState(
                    name: "illumi_rt_lighting_tlas", device: device,
                    constants: cv, variantKey: "curves")
            }
            if surfCacheTLASCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                surfCacheTLASCurvePipeline = cache.pipelineState(
                    name: "illumi_surfcache_update_tlas", device: device,
                    constants: cv, variantKey: "curves")
            }
            if rtTLASCurvePipeline == nil || surfCacheTLASCurvePipeline == nil {
                Self.log.error("Curve RT: kRTCurvesEnabled pipeline variant failed; dropping \(self.rtCurveSets.count) curve set(s).")
                rtCurveSets = []
            }
        }
        buildCurvePools()
    }

    /// Function-constant index for `kRTCurvesEnabled` in
    /// IlluminatoramaRTInstanced.metal + IlluminatoramaSurfaceCache.metal.
    private static let curveFunctionConstantIndex = 30

    /// Pool every adopted set's control points (packed float3 — stride 12, the
    /// layout both the curve BLAS descriptor and the shading kernels read),
    /// radii, and segment indices (made ABSOLUTE into the pool) into shared
    /// buffers, plus the per-set `RTCurveSetData` records. Bake-time only.
    private func buildCurvePools() {
        rtCurvePoolPoints = nil; rtCurvePoolRadii = nil; rtCurvePoolSegments = nil
        rtCurveSetDataBuffer = nil; rtCurveSetSegmentBase = []
        rtCurvePoolPointCount = 0
        guard !rtCurveSets.isEmpty else { return }
        var pts: [Float] = []          // 3 floats per control point (packed)
        var radii: [Float] = []
        var segs: [UInt32] = []
        var data: [RTCurveSetData] = []
        for set in rtCurveSets {
            let pointBase = UInt32(radii.count)
            rtCurveSetSegmentBase.append(segs.count)
            data.append(RTCurveSetData(
                m0: set.transform.columns.0, m1: set.transform.columns.1,
                m2: set.transform.columns.2, m3: set.transform.columns.3,
                albedoRoughness: SIMD4(set.material.albedo, set.material.roughness),
                emissionPad: SIMD4(set.material.emission, 0),
                meta: SIMD4<UInt32>(UInt32(segs.count), 0, 0, 0)))
            for p in set.controlPoints { pts.append(p.x); pts.append(p.y); pts.append(p.z) }
            radii.append(contentsOf: set.radii)
            segs.append(contentsOf: set.segmentIndices.map { $0 + pointBase })
        }
        guard let pb = device.makeBuffer(bytes: pts,
                  length: MemoryLayout<Float>.stride * pts.count, options: .storageModeShared),
              let rb = device.makeBuffer(bytes: radii,
                  length: MemoryLayout<Float>.stride * radii.count, options: .storageModeShared),
              let sb = device.makeBuffer(bytes: segs,
                  length: MemoryLayout<UInt32>.stride * segs.count, options: .storageModeShared),
              let db = device.makeBuffer(bytes: data,
                  length: MemoryLayout<RTCurveSetData>.stride * data.count, options: .storageModeShared)
        else {
            Self.log.error("Curve RT: pool allocation failed; dropping curve sets.")
            rtCurveSets = []; rtCurveSetSegmentBase = []
            return
        }
        pb.label = "Illuminatorama.rt.curve.points"
        rb.label = "Illuminatorama.rt.curve.radii"
        sb.label = "Illuminatorama.rt.curve.segments"
        db.label = "Illuminatorama.rt.curve.setData"
        rtCurvePoolPoints = pb; rtCurvePoolRadii = rb
        rtCurvePoolSegments = sb; rtCurveSetDataBuffer = db
        rtCurvePoolPointCount = radii.count
    }

    /// Phase B — re-fit every deforming mesh's BLAS in place from its live
    /// (repacked) vertex buffer, on the FRAME command buffer. Runs only in the
    /// refit branch (stable topology); the repack at the top of the frame already
    /// wrote this frame's positions into `mesh.vertexBuffer` (the same buffer the
    /// BLAS references), and Metal sequences the compute-write → AS-refit-read on
    /// the shared command buffer. No `waitUntilCompleted` — refit is cheap + async.
    private func refitDeformingBLAS(_ cb: MTLCommandBuffer) {
        guard !rtBLASRefitDesc.isEmpty else { return }
        // Dev A/B override: VIZ_ILLUMI_NO_BLAS_REFIT=1 freezes deforming BLASes at
        // their build-time pose (the pre-Phase-B behaviour) so the RT-traces-stale
        // -vs-current difference is directly checkable. Default: refit (on).
        if Self.noBLASRefit { return }
        guard let enc = cb.makeAccelerationStructureCommandEncoder() else { return }
        enc.label = "Illuminatorama.rt.blas.refit"
        for (mid, desc) in rtBLASRefitDesc {
            guard let blas = rtBLASByMesh[mid],
                  let scratch = rtBLASRefitScratch[mid] else { continue }
            enc.refit(sourceAccelerationStructure: blas, descriptor: desc,
                      destinationAccelerationStructure: blas,
                      scratchBuffer: scratch, scratchBufferOffset: 0)
        }
        enc.endEncoding()
    }

    private func packed4x3(_ m: simd_float4x4) -> MTLPackedFloat4x3 {
        func col(_ c: SIMD4<Float>) -> MTLPackedFloat3 {
            var p = MTLPackedFloat3(); p.x = c.x; p.y = c.y; p.z = c.z; return p
        }
        var p = MTLPackedFloat4x3()
        p.columns.0 = col(m.columns.0); p.columns.1 = col(m.columns.1)
        p.columns.2 = col(m.columns.2); p.columns.3 = col(m.columns.3)
        return p
    }

    /// Full rebuild: build a BLAS + object-normal slot for every not-yet-cached
    /// mesh (one setup-time command buffer), assemble the concatenated normal
    /// buffer + per-instance descriptors/data in grouped order, and build the
    /// TLAS. Runs on a topology change.
    private func rebuildRTAccel() {
        rtTLASActive = false
        // AAA glass (#60): the canonical flattened glass list in TLAS-append order.
        // Glass instances ride the SAME TLAS as the opaque/curve instances (so glass
        // refracts/reflects real geometry and other glass) but carry mask 0x02, so
        // the deferred RT lighting / surface-cache passes (which trace with mask
        // 0x01) never see them. They're appended AFTER opaque + curve instances;
        // `rtGlassInstanceBase` is the first glass instance_id.
        let glassFlat: [(kind: MeshKind, inst: IlluminatoramaGlassInstance)] =
            (rtGlassEnabled ? flattenGlass() : []).flatMap { g in g.insts.map { (g.kind, $0) } }
        // (a) Build BLAS + cache object normals for any newly-seen mesh.
        // Deforming (GPU-fed / DynamicMesh) kinds get a refit-capable BLAS so the
        // per-frame `refitDeformingBLAS` can track their moving vertices (Phase B).
        let deformingKinds = Set(gpuRepackTasks.map(\.kind))
        var pending: [(MTLAccelerationStructure, MTLPrimitiveAccelerationStructureDescriptor, MTLBuffer)] = []
        for group in meshGroups {
            guard let mesh = meshes[group.kind], mesh.indexCount >= 3 else { continue }
            let mid = ObjectIdentifier(mesh)
            if rtBLASByMesh[mid] != nil { continue }
            let deforming = deformingKinds.contains(group.kind)
            let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
            geom.vertexBuffer = mesh.vertexBuffer
            geom.vertexStride = MemoryLayout<IlluminatoramaVertex>.stride
            geom.vertexFormat = .float3
            geom.indexBuffer = mesh.indexBuffer
            geom.indexType = mesh.indexType
            geom.triangleCount = mesh.indexCount / 3
            // #60 item 6 — bake this mesh's per-triangle card-frame UVs into the
            // BLAS `primitive_data` (one `IlluminatoramaPrimUV` per primitive,
            // mesh-invariant). A TLAS hit then reads its surface-cache UVs from the
            // AS payload (`res.primitive_data`) instead of the per-instance global
            // `triUVa`/`triUVc` side buffers. Per-triangle path only (charts use a
            // non-invariant parameterisation) and only when the cache is on. Metal
            // COPIES this buffer into the AS at build time, so it only needs to live
            // through the build below — `pending`→`d`→`geom` retains it until then.
            //
            // DEFORMING kinds are excluded: their soup comes from the UN-sanitised
            // `liveDeformingObjectSoup`, so a degenerate/out-of-range triangle would
            // drop from the global soup (shifting its post-drop frame-A/B parity)
            // while the raw-count BLAS table would not — a silent wrong-tile-half.
            // Their BLAS keeps no primitive data → the shader null-check falls back
            // to the (correct, post-drop) side-buffer UVs. They also re-light every
            // frame (α=1), so the cache-read saving matters least for them anyway.
            if bakePrimUVFold && !deforming {
                let uvs = Self.perMeshCardUVs(triangleCount: mesh.indexCount / 3)
                let stride = MemoryLayout<IlluminatoramaPrimUV>.stride   // 24 B (3×float2)
                if !uvs.isEmpty,
                   let pdb = device.makeBuffer(bytes: uvs, length: stride * uvs.count,
                                               options: .storageModeShared) {
                    pdb.label = "Illuminatorama.rt.blas.primUV"
                    geom.primitiveDataBuffer = pdb
                    geom.primitiveDataBufferOffset = 0
                    geom.primitiveDataStride = stride
                    geom.primitiveDataElementSize = stride
                }
            }
            let d = MTLPrimitiveAccelerationStructureDescriptor()
            d.geometryDescriptors = [geom]
            // `.refit` lets the BLAS be re-fit in place each frame from the live
            // vertex buffer (fixed topology). Only deforming meshes need it; static
            // meshes keep the default build (smaller, no refit scratch retained).
            if deforming { d.usage = .refit }
            let sizes = device.accelerationStructureSizes(descriptor: d)
            guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                                   options: .storageModePrivate) else { continue }
            blas.label = deforming ? "Illuminatorama.rt.blas.deforming" : "Illuminatorama.rt.blas"
            rtBLASByMesh[mid] = blas
            rtNormalsByMesh[mid] = mesh.objectFaceNormals()
            pending.append((blas, d, scratch))
            // Retain the descriptor + a refit-sized scratch so the per-frame refit
            // can re-encode without re-deriving sizes. Refit scratch can differ from
            // build scratch; size it explicitly.
            if deforming {
                rtBLASRefitDesc[mid] = d
                if let rscratch = device.makeBuffer(
                        length: max(sizes.refitScratchBufferSize, 16),
                        options: .storageModePrivate) {
                    rtBLASRefitScratch[mid] = rscratch
                }
            }
        }
        // (a2) Curve BLASes (#60 item 7) — one per registered set, rebuilt on
        // every topology change (cheap: that's the whole point of curve
        // primitives vs triangle tubes). Each BLAS reads its segment slice of
        // the pooled buffers via `indexBufferOffset`; segment index values are
        // absolute into the pool, so `controlPointBuffer` is the whole pool.
        rtCurveBLASList.removeAll(keepingCapacity: true)
        if !rtCurveSets.isEmpty,
           let cpts = rtCurvePoolPoints, let crad = rtCurvePoolRadii,
           let cseg = rtCurvePoolSegments {
            for (s, set) in rtCurveSets.enumerated() where s < rtCurveSetSegmentBase.count {
                let geom = MTLAccelerationStructureCurveGeometryDescriptor()
                geom.controlPointBuffer = cpts
                geom.controlPointCount = rtCurvePoolPointCount
                geom.controlPointStride = 12       // packed float3
                geom.controlPointFormat = .float3
                geom.radiusBuffer = crad
                geom.radiusFormat = .float
                geom.radiusStride = MemoryLayout<Float>.stride
                geom.indexBuffer = cseg
                geom.indexBufferOffset = rtCurveSetSegmentBase[s] * MemoryLayout<UInt32>.stride
                geom.indexType = .uint32
                geom.segmentCount = set.segmentIndices.count
                geom.segmentControlPointCount = 4
                geom.curveType = .round            // exact 3D normals (branches)
                geom.curveBasis = .catmullRom
                geom.curveEndCaps = .sphere
                geom.opaque = true
                let d = MTLPrimitiveAccelerationStructureDescriptor()
                d.geometryDescriptors = [geom]
                let sizes = device.accelerationStructureSizes(descriptor: d)
                guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                      let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                                       options: .storageModePrivate) else { continue }
                blas.label = "Illuminatorama.rt.blas.curve.\(set.label)"
                rtCurveBLASList.append(blas)
                pending.append((blas, d, scratch))
            }
        }
        // (a3) Glass BLASes (#60 AAA glass) — one per distinct glass mesh kind not
        // already cached (a glass kind may be shared with an opaque mesh, e.g.
        // `.box`; `rtBLASByMesh` dedupes). Static (no refit / no primUV bake): glass
        // doesn't participate in the surface cache as a hit surface.
        for (kind, _) in glassFlat {
            guard let mesh = meshes[kind], mesh.indexCount >= 3 else { continue }
            let mid = ObjectIdentifier(mesh)
            if rtBLASByMesh[mid] != nil { continue }
            let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
            geom.vertexBuffer = mesh.vertexBuffer
            geom.vertexStride = MemoryLayout<IlluminatoramaVertex>.stride
            geom.vertexFormat = .float3
            geom.indexBuffer = mesh.indexBuffer
            geom.indexType = mesh.indexType
            geom.triangleCount = mesh.indexCount / 3
            let d = MTLPrimitiveAccelerationStructureDescriptor()
            d.geometryDescriptors = [geom]
            let sizes = device.accelerationStructureSizes(descriptor: d)
            guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                                   options: .storageModePrivate) else { continue }
            blas.label = "Illuminatorama.rt.blas.glass"
            rtBLASByMesh[mid] = blas
            rtNormalsByMesh[mid] = mesh.objectFaceNormals()
            pending.append((blas, d, scratch))
        }
        if !pending.isEmpty {
            guard let cmd = commandQueue.makeCommandBuffer(),
                  let enc = cmd.makeAccelerationStructureCommandEncoder() else { return }
            for (blas, d, scratch) in pending {
                enc.build(accelerationStructure: blas, descriptor: d,
                          scratchBuffer: scratch, scratchBufferOffset: 0)
            }
            enc.endEncoding(); cmd.commit()
            cmd.waitUntilCompleted()   // gpu-ok: one-time BLAS build (per newly-seen mesh / per-topology curve set)
            Self.recordASBuildError(cmd, "rebuildRTAccel.blas")
        }

        // (b) Assemble BLAS list + concat normals + per-mesh slot for this topology.
        rtBLASList.removeAll(keepingCapacity: true)
        rtResidentBuffers.removeAll(keepingCapacity: true)
        var concat: [SIMD4<Float>] = []
        var slot: [ObjectIdentifier: (asIndex: Int, normalBase: Int)] = [:]
        for group in meshGroups {
            guard let mesh = meshes[group.kind] else { continue }
            let mid = ObjectIdentifier(mesh)
            if slot[mid] != nil { continue }
            guard let blas = rtBLASByMesh[mid], let nrm = rtNormalsByMesh[mid] else { continue }
            slot[mid] = (rtBLASList.count, concat.count)
            rtBLASList.append(blas)
            rtResidentBuffers.append(mesh.vertexBuffer); rtResidentBuffers.append(mesh.indexBuffer)
            concat.append(contentsOf: nrm)
        }
        // Glass meshes join the same BLAS list + concat normals (a glass hit's
        // normal resolves through `objNormal[normalBase + prim]`, same as opaque).
        for (kind, _) in glassFlat {
            guard let mesh = meshes[kind] else { continue }
            let mid = ObjectIdentifier(mesh)
            if slot[mid] != nil { continue }
            guard let blas = rtBLASByMesh[mid], let nrm = rtNormalsByMesh[mid] else { continue }
            slot[mid] = (rtBLASList.count, concat.count)
            rtBLASList.append(blas)
            rtResidentBuffers.append(mesh.vertexBuffer); rtResidentBuffers.append(mesh.indexBuffer)
            concat.append(contentsOf: nrm)
        }
        guard !rtBLASList.isEmpty, !concat.isEmpty else { return }
        rtObjNormalBuffer = device.makeBuffer(
            bytes: concat, length: MemoryLayout<SIMD4<Float>>.stride * concat.count,
            options: .storageModeShared)
        rtObjNormalCount = concat.count
        // Curve pool buffers must stay resident for the intersector (the curve
        // BLASes reference them), and the shading kernels read them at hits.
        if !rtCurveBLASList.isEmpty,
           let cpts = rtCurvePoolPoints, let crad = rtCurvePoolRadii,
           let cseg = rtCurvePoolSegments {
            rtResidentBuffers.append(cpts)
            rtResidentBuffers.append(crad)
            rtResidentBuffers.append(cseg)
        }

        // (c) Per-instance descriptors + data in GROUPED order (== currentInstanceBuffer).
        // Curve-set instances (#60 item 7) append after the triangle instances.
        let total = instances.count
        let curveCount = rtCurveBLASList.count
        let glassCount = glassFlat.count
        let totalInstances = total + curveCount + glassCount
        ensureRTInstanceBuffers(capacity: totalInstances)
        guard let descBuf = rtInstanceDescBuffer, let dataBuf = rtInstanceDataBuffer else { return }
        // Zero first so any unslotted instance has mask=0 (never intersected). The
        // data buffer is zeroed across ALL instances (incl. glass) so a glass hit's
        // RTInstanceData (filled below) and the curve gap are well-defined.
        memset(descBuf.contents(), 0, MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * totalInstances)
        memset(dataBuf.contents(), 0, MemoryLayout<RTInstanceData>.stride * totalInstances)
        let instPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let descPtr = descBuf.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: totalInstances)
        let dataPtr = dataBuf.contents().bindMemory(to: RTInstanceData.self, capacity: totalInstances)
        for group in meshGroups {
            guard let mesh = meshes[group.kind], let s = slot[ObjectIdentifier(mesh)] else { continue }
            for k in 0..<group.count {
                let i = group.start + k
                guard i < total else { break }
                let inst = instPtr[i]
                var desc = MTLAccelerationStructureInstanceDescriptor()
                desc.accelerationStructureIndex = UInt32(s.asIndex)
                // Raster-only instances (#60 item 7) get mask 0: present in the
                // TLAS layout but never intersected. Opaque instances get mask 0x01
                // (AAA glass uses 0x02), so opaque/cache rays — which trace with
                // 0x01 — match exactly what they hit before glass joined the TLAS.
                desc.mask = inst.rtExclude != 0 ? 0x00 : 0x01
                desc.options = .opaque
                desc.intersectionFunctionTableOffset = 0
                desc.transformationMatrix = packed4x3(inst.modelMatrix)
                descPtr[i] = desc
                let nm = inst.normalMatrix
                dataPtr[i] = RTInstanceData(
                    nrm0: nm.columns.0, nrm1: nm.columns.1, nrm2: nm.columns.2,
                    albedoTriBase: SIMD4(inst.albedo, Float(s.normalBase)))
            }
        }
        // (c2) Curve-set instance descriptors (#60 item 7) — one per set,
        // static rigid placement captured at registration; a hit's
        // `instance_id - total` is the set index into `rtCurveSetDataBuffer`.
        for (c, set) in rtCurveSets.enumerated() where c < curveCount {
            var desc = MTLAccelerationStructureInstanceDescriptor()
            desc.accelerationStructureIndex = UInt32(rtBLASList.count + c)
            desc.mask = 0x01   // opaque/curve mask (glass = 0x02)
            desc.options = .opaque
            desc.intersectionFunctionTableOffset = 0
            desc.transformationMatrix = packed4x3(set.transform)
            descPtr[total + c] = desc
        }
        rtCurveInstanceCount = curveCount

        // (c3) Glass instance descriptors + data + the per-glass material buffer
        // (#60 AAA glass). Appended after opaque + curve; `rtGlassInstanceBase` is
        // the first glass instance_id. Mask 0x02 so only glass-pass rays (which
        // trace 0x03 = opaque+glass) intersect them. The bounce loop reads the hit's
        // material from `rtGlassDataBuffer[instance_id - base]` and its normal from
        // the shared `objNormal` via this RTInstanceData (albedo carries the tint).
        rtGlassInstanceBase = total + curveCount
        rtGlassInstanceCount = glassCount
        if glassCount > 0 {
            var glassData = [IlluminatoramaRTGlassData](
                repeating: IlluminatoramaRTGlassData(tintIor: .zero, rdrf: .zero, dispersionPad: .zero),
                count: glassCount)
            for (gi, entry) in glassFlat.enumerated() {
                let i = rtGlassInstanceBase + gi
                guard let mesh = meshes[entry.kind], let s = slot[ObjectIdentifier(mesh)] else { continue }
                let inst = entry.inst
                var desc = MTLAccelerationStructureInstanceDescriptor()
                desc.accelerationStructureIndex = UInt32(s.asIndex)
                desc.mask = 0x02
                desc.options = .opaque   // opaque AS hit (the dielectric BSDF is shaded in the glass fs)
                desc.intersectionFunctionTableOffset = 0
                desc.transformationMatrix = packed4x3(inst.modelMatrix)
                descPtr[i] = desc
                let nm = inst.normalMatrix
                dataPtr[i] = RTInstanceData(
                    nrm0: nm.columns.0, nrm1: nm.columns.1, nrm2: nm.columns.2,
                    albedoTriBase: SIMD4(inst.tintIor.x, inst.tintIor.y, inst.tintIor.z, Float(s.normalBase)))
                glassData[gi] = IlluminatoramaRTGlassData(
                    tintIor: inst.tintIor, rdrf: inst.rdrf, dispersionPad: inst.dispersionPad)
            }
            let glen = MemoryLayout<IlluminatoramaRTGlassData>.stride * glassCount
            if (rtGlassDataBuffer?.length ?? 0) < glen {
                rtGlassDataBuffer = device.makeBuffer(length: glen, options: .storageModeShared)
                rtGlassDataBuffer?.label = "Illuminatorama.rt.glassData"
            }
            glassData.withUnsafeBytes { src in
                _ = memcpy(rtGlassDataBuffer!.contents(), src.baseAddress!, glen)
            }
        }

        // (d) Build the TLAS (allocate / grow as needed).
        let tdesc = MTLInstanceAccelerationStructureDescriptor()
        tdesc.instanceDescriptorBuffer = descBuf
        tdesc.instanceCount = totalInstances
        tdesc.instancedAccelerationStructures = rtBLASList + rtCurveBLASList
        tdesc.usage = .refit
        let sizes = device.accelerationStructureSizes(descriptor: tdesc)
        if rtTLAS == nil || rtTLASCapacity < totalInstances {
            rtTLAS = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
            rtTLASScratch = device.makeBuffer(
                length: max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize, 16),
                options: .storageModePrivate)
            rtTLAS?.label = "Illuminatorama.rt.tlas"
            rtTLASCapacity = totalInstances
        }
        guard let tlas = rtTLAS, let scratch = rtTLASScratch,
              let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeAccelerationStructureCommandEncoder() else { return }
        enc.build(accelerationStructure: tlas, descriptor: tdesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: topology-change TLAS rebuild (not per-frame for stable scenes)
        Self.recordASBuildError(cmd, "rebuildRTAccel.tlas")
        rtTLASInstanceCount = totalInstances
        rtTLASActive = true

        // P1c — surface cache + TLAS coexistence. Bake a per-instance world-space
        // soup in the SAME grouped order this TLAS enumerates instances, so a
        // hit's `soupTriBase[instance_id] + primitive_id` indexes the soup-ordered
        // per-triangle cards. The soup also feeds the primitive `rtAccel` the
        // cache-UPDATE pass traces. Runs only on topology change (not per refit),
        // so an animated scene's moved geometry reads bounded-stale cache — lighting
        // still animates via the TLAS refit. Skipped (TLAS unaffected) past the cap.
        if surfaceCacheEnabled { buildGroupedSurfaceSoup(total: total) }
    }

    /// Build the grouped-order world-space soup + per-triangle cards + the
    /// `soupTriBase` map for the surface-cache TLAS path. Mesh object triangles
    /// are read once per mesh and transformed by each instance's model matrix in
    /// `meshGroups`/`instance` order — identical to the TLAS instance ordering,
    /// so `soupTriBase[i]` (set just before instance i's triangles are appended)
    /// + the hit's local `primitive_id` resolves to the global card index.
    private func buildGroupedSurfaceSoup(total: Int) {
        surfHasDeformingCards = false   // recomputed below; cleared so an early return can't leave it stale
        // Cost gate: the instance-expanded triangle count, not the per-mesh count.
        var expandedTris = 0
        for g in meshGroups { if let m = meshes[g.kind] { expandedTris += g.count * (m.indexCount / 3) } }
        guard expandedTris > 0, expandedTris <= Self.surfaceCacheMaxTrianglesTLAS else {
            if expandedTris > Self.surfaceCacheMaxTrianglesTLAS {
                Self.log.notice("Surface cache (TLAS) skipped: \(expandedTris) tris > cap \(Self.surfaceCacheMaxTrianglesTLAS)")
            }
            // Leave the cache off for this scene; the kernel gates on the uniform.
            surfCardCount = 0; rtSoupTriBaseBuffer = nil; rtSoupTriBaseCount = 0
            return
        }

        var positions: [SIMD3<Float>] = []; positions.reserveCapacity(expandedTris * 3)
        var indices: [UInt32] = [];        indices.reserveCapacity(expandedTris * 3)
        var triAlbedo: [SIMD3<Float>] = []; triAlbedo.reserveCapacity(expandedTris)
        var triNormal: [SIMD3<Float>] = []; triNormal.reserveCapacity(expandedTris)
        var soupBase = [UInt32](repeating: 0, count: total)
        // #60 item 1 + Phase D — GPU-diff maps, accumulated in lockstep with the
        // appended soup triangles (so global triangle order matches exactly,
        // including dropped out-of-range triangles): owner instance, OBJECT-space
        // verts (rigid re-frame), vertex-index triples (deform re-frame reads the
        // LIVE position buffer through these), and the per-tri deforming flag.
        var gpuTriOwner: [UInt32] = [];     gpuTriOwner.reserveCapacity(expandedTris)
        var gpuObjVerts: [SIMD4<Float>] = []; gpuObjVerts.reserveCapacity(expandedTris * 3)
        var gpuTriIdx: [SIMD4<UInt32>] = []; gpuTriIdx.reserveCapacity(expandedTris)
        var gpuTriDeform: [Bool] = [];       gpuTriDeform.reserveCapacity(expandedTris)

        let instPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        // Phase C — deforming (DynamicMesh / GPU-fed) kinds have a `.private`
        // interleaved vertex buffer, so `objectTriangleSoup()` returns empty and
        // they'd be excluded from the cache. Source their build-pose soup from the
        // live shared buffers instead (the per-frame re-frame then keeps the cards
        // on the current surface). `surfHasDeformingCards` flips the update kernel
        // into honouring per-card dirty flags (α=1) so they re-light each frame.
        let deformingKinds = Set(gpuRepackTasks.map(\.kind))
        var soupByMesh: [ObjectIdentifier: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
        // A group that contributes NO triangles (nil mesh / empty soup) must still
        // leave its instances an empty, correctly-positioned soupBase range — else
        // they keep the init value 0, which makes the chart incremental path read
        // `[0, soupBase[i+1])` and mis-assign instance 0's cards to the empty
        // instance (last-writer-wins). Skipped instances also get mask=0 in the
        // TLAS (never hit), so this only makes soupBase monotonic — no consumer
        // reads the empty range.
        func markEmptyRange(_ group: MeshDrawGroup) {
            let base = UInt32(indices.count / 3)
            for k in 0..<group.count where group.start + k < total { soupBase[group.start + k] = base }
        }
        for group in meshGroups {
            guard let mesh = meshes[group.kind] else { markEmptyRange(group); continue }
            let isDeforming = deformingKinds.contains(group.kind)
            let soup: (positions: [SIMD3<Float>], indices: [UInt32])
            if isDeforming {
                guard let live = liveDeformingObjectSoup(kind: group.kind) else { markEmptyRange(group); continue }
                soup = live
            } else {
                let mid = ObjectIdentifier(mesh)
                soup = soupByMesh[mid] ?? {
                    let s = mesh.objectTriangleSoup(); soupByMesh[mid] = s; return s
                }()
            }
            // Both must be non-empty: a soup with indices but no positions
            // (degenerate read) would make every `positions[base + index]`
            // below reach past the block.
            guard !soup.indices.isEmpty, !soup.positions.isEmpty else { markEmptyRange(group); continue }
            if isDeforming { surfHasDeformingCards = true }
            for k in 0..<group.count {
                let i = group.start + k
                guard i < total else { break }
                soupBase[i] = UInt32(indices.count / 3)
                let model = instPtr[i].modelMatrix
                let albedo = instPtr[i].albedo
                let base = UInt32(positions.count)
                for p in soup.positions {
                    let w = model * SIMD4<Float>(p, 1)
                    positions.append(SIMD3(w.x, w.y, w.z))
                }
                // `soup.indices` must all be < soup.positions.count, or
                // `positions[Int(i0)]` below reads out of bounds and SIGTRAPs
                // the whole render loop. `from(scnGeometry:)` sanitises
                // registered meshes, but the live deforming path
                // (`liveDeformingObjectSoup`) reads positions and indices from
                // independent buffers, so guard here too rather than trust it.
                let vcount = UInt32(soup.positions.count)
                // One-time alarm if a mesh's indices reach past its own vertex
                // block — names the offending mesh in the unified log so a
                // recurrence (e.g. a new scene's hand-built soup) is diagnosable
                // without a debugger. The per-triangle guard below keeps it safe
                // regardless.
                if k == 0, let mx = soup.indices.max(), mx >= vcount {
                    Self.log.error("Surface soup: mesh \(String(describing: group.kind)) (deforming=\(isDeforming)) has index \(mx) >= vertexCount \(vcount); out-of-range triangles dropped.")
                }
                var t = 0
                while t + 2 < soup.indices.count {
                    let l0 = soup.indices[t], l1 = soup.indices[t + 1], l2 = soup.indices[t + 2]
                    t += 3
                    guard l0 < vcount, l1 < vcount, l2 < vcount else { continue }
                    let i0 = base + l0, i1 = base + l1, i2 = base + l2
                    indices.append(i0); indices.append(i1); indices.append(i2)
                    let w0 = positions[Int(i0)], w1 = positions[Int(i1)], w2 = positions[Int(i2)]
                    var n = simd_cross(w1 - w0, w2 - w0)
                    let len = simd_length(n)
                    n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
                    triNormal.append(n); triAlbedo.append(albedo)
                    // Phase D parity: with the freeze override the CPU path never
                    // re-frames a deforming kind (its object soup is empty there),
                    // so mark its triangles sentinel — never re-framed, dirty=0.
                    if isDeforming && noDeformCardsOverride {
                        gpuTriOwner.append(UInt32.max)
                    } else {
                        gpuTriOwner.append(UInt32(i))
                    }
                    gpuTriDeform.append(isDeforming && !noDeformCardsOverride)
                    gpuTriIdx.append(SIMD4<UInt32>(l0, l1, l2, 0))
                    gpuObjVerts.append(SIMD4(soup.positions[Int(l0)], 0))
                    gpuObjVerts.append(SIMD4(soup.positions[Int(l1)], 0))
                    gpuObjVerts.append(SIMD4(soup.positions[Int(l2)], 0))
                }
            }
        }
        guard !indices.isEmpty else { surfCardCount = 0; return }

        // Build the primitive AS (cache-update trace target) + surface-cache cards.
        setRTGeometry(positions: positions, indices: indices,
                      triangleAlbedo: triAlbedo, triangleNormal: triNormal)
        // P2 — coplanar-chart merging is OPT-IN (`VIZ_SURFCACHE_CHARTS=1`).
        // Default is the P3 per-triangle path: on low/medium-tessellation geometry
        // (e.g. cityStreet — quad walls) P3's 2-triangles-per-tile packing already
        // hits ~2:1 tri:tile, which BEATS chart merging's ~1.7:1 there (measured:
        // charts 12.4k cards / 10.5 MB atlas vs per-tri 10.6k / 3.1 MB). Chart
        // merging wins only on HEAVILY-tessellated coplanar surfaces (many
        // triangles per flat face → N:1), and it gives larger/stabler cards for
        // incremental invalidation. See docs/illuminatorama README phase 4.38c.
        // Per-instance mesh kind (used by both paths to relate instances ↔ soup).
        var instanceKinds = [MeshKind](repeating: meshGroups.first?.kind ?? .box, count: total)
        for group in meshGroups {
            for k in 0..<group.count where group.start + k < total { instanceKinds[group.start + k] = group.kind }
        }
        let useCharts = Self.surfCacheCharts
        if useCharts {
            let g = Self.makeCoplanarChartCards(
                positions: positions, indices: indices,
                triangleAlbedo: triAlbedo, triangleNormal: triNormal)
            // Adaptive per-chart tile size: a chart's tile scales with its world
            // extent, shelf-packed, so big walls get more texels than small facets
            // and the atlas stays efficient across the chart-size spread.
            setSurfaceCacheCards(cards: g.cards, triCard: g.triCard,
                                 triUVa: g.triUVa, triUVc: g.triUVc,
                                 adaptiveTiles: true, texelsPerMeter: 5.0)
            // Chart-path incremental invalidation (#60 task 1). A chart is built in
            // WORLD space at the bake pose; record each card's frame back in its
            // owning instance's OBJECT space (M_bake⁻¹ · worldFrame) so the per-frame
            // re-frame is one affine transform (`M_now · objectFrame`) — exact for
            // rigid/affine motion, UVs untouched. Charts never cross an instance
            // (separate vertex blocks ⇒ no cross-instance edge adjacency), so each
            // instance owns a CONTIGUOUS card range. Deforming kinds are excluded
            // (their per-vertex shape change invalidates a planar chart frame; the
            // chart path is rigid-mover only — documented in the design note).
            if surfCardCount == g.cards.count, !g.triCard.isEmpty {
                var cardInstance = [Int32](repeating: -1, count: g.cards.count)
                for i in 0..<total {
                    if deformingKinds.contains(instanceKinds[i]) { continue }
                    let firstTri = Int(soupBase[i])
                    let nextTri = (i + 1 < total) ? Int(soupBase[i + 1]) : g.triCard.count
                    for gt in firstTri..<nextTri where gt >= 0 && gt < g.triCard.count {
                        cardInstance[Int(g.triCard[gt])] = Int32(i)
                    }
                }
                // Per-instance bake-pose inverse — ONLY for invertible matrices. A
                // degenerate model matrix (e.g. a flattened/zero-scale node) would
                // make simd_inverse return NaN/inf and poison every per-frame
                // re-frame; skip it so that instance's cards stay at their baked
                // world frame and are simply never re-framed (they won't track if it
                // moves, but they won't corrupt — the acceptable degenerate fallback).
                var invByInstance = [Int32: simd_float4x4]()
                for i in 0..<total {
                    let m = instPtr[i].modelMatrix
                    if abs(simd_determinant(m)) > 1e-12 { invByInstance[Int32(i)] = simd_inverse(m) }
                }
                var objFrames = [SurfCardFrame](repeating: SurfCardFrame(), count: g.cards.count)
                var instCardLo = [Int32](repeating: -1, count: total)
                var instCardHi = [Int32](repeating: -1, count: total)
                // #60 item 1 — GPU-diff owner per card (−1 = static / degenerate
                // bake pose, never re-framed; mirrors the CPU skip below).
                var cardOwnerGPU = [Int32](repeating: -1, count: g.cards.count)
                for c in 0..<g.cards.count {
                    let card = g.cards[c]
                    let wO = SIMD3<Float>(card.origin.x, card.origin.y, card.origin.z)
                    let wU = SIMD3<Float>(card.uAxis.x, card.uAxis.y, card.uAxis.z)
                    let wV = SIMD3<Float>(card.vAxis.x, card.vAxis.y, card.vAxis.z)
                    let wN = SIMD3<Float>(card.normal.x, card.normal.y, card.normal.z)
                    let inst = cardInstance[c]
                    if inst >= 0, let invM = invByInstance[inst] {
                        let ii = Int(inst)
                        let o = invM * SIMD4<Float>(wO, 1)
                        let u = invM * SIMD4<Float>(wU, 0)
                        let v = invM * SIMD4<Float>(wV, 0)
                        let n = invM * SIMD4<Float>(wN, 0)
                        objFrames[c] = SurfCardFrame(origin: SIMD3(o.x, o.y, o.z), uAxis: SIMD3(u.x, u.y, u.z),
                                                     vAxis: SIMD3(v.x, v.y, v.z), normal: SIMD3(n.x, n.y, n.z))
                        if instCardLo[ii] < 0 || Int32(c) < instCardLo[ii] { instCardLo[ii] = Int32(c) }
                        if Int32(c + 1) > instCardHi[ii] { instCardHi[ii] = Int32(c + 1) }
                        cardOwnerGPU[c] = inst
                    } else {
                        objFrames[c] = SurfCardFrame(origin: wO, uAxis: wU, vAxis: wV, normal: wN)
                    }
                }
                surfChartIncremental = ChartIncremental(objectFrame: objFrames,
                                                        instCardLo: instCardLo, instCardHi: instCardHi)
                surfChartMovedScratch = [Bool](repeating: false, count: instCardLo.count)
                surfTriCardCPU  = g.triCard      // guard parity with the per-triangle path
                surfSoupBaseCPU = soupBase
                surfIncrementalReady = true
                buildGPUDiffChartBuffers(objFrames: objFrames, cardOwner: cardOwnerGPU)
            }
        } else {
            let g = Self.makePerTriangleSurfaceCards(
                positions: positions, indices: indices,
                triangleAlbedo: triAlbedo, triangleNormal: triNormal,
                blockStarts: soupBase.map { Int($0) })
            setSurfaceCacheCards(cards: g.cards, triCard: g.triCard,
                                 triUVa: g.triUVa, triUVc: g.triUVc,
                                 tileSize: Self.surfaceCachePerTriTileSize)
            // #60 item 6 — invariance guard (DEBUG). The fold bakes ONE per-mesh UV
            // table into the shared BLAS; that's only correct if every instance block
            // of a mesh got the same frame-A/B layout above. Verify the first block of
            // each kind matches the canonical `perMeshCardUVs` the BLAS bakes.
            Self.assertPerMeshUVInvariance(triUVa: g.triUVa, triUVc: g.triUVc,
                                           soupBase: soupBase, instanceKinds: instanceKinds,
                                           deformingKinds: deformingKinds, total: total)
            // Capture the CPU maps the incremental-invalidation prototype needs to
            // re-frame a moved instance's cards in place (per-triangle path only).
            // Frame-B membership is read back from the packed reference UVs:
            // frame A's uvA = (0,0,…); frame B's uvA = (1,1,…).
            surfTriCardCPU   = g.triCard
            surfTriIsFrameB  = g.triUVa.map { $0.x > 0.5 }
            surfSoupBaseCPU  = soupBase
            surfObjectSoupByKind.removeAll(keepingCapacity: true)
            for group in meshGroups where surfObjectSoupByKind[group.kind] == nil {
                if let m = meshes[group.kind] { surfObjectSoupByKind[group.kind] = m.objectTriangleSoup() }
            }
            surfInstanceKind = instanceKinds
            surfChartIncremental = nil   // per-triangle path owns the re-frame
            surfIncrementalReady = true
            buildGPUDiffPerTriBuffers(triOwner: gpuTriOwner, objVerts: gpuObjVerts,
                                      isFrameB: surfTriIsFrameB,
                                      triIdx: gpuTriIdx, deforming: gpuTriDeform)
            // Phase D — one deform dispatch per deforming instance. soupBase is
            // monotonic and each instance's appended triangles exactly fill
            // [soupBase[i], soupBase[i+1]), so base/count come straight from it.
            if surfGPUDiffTriCount > 0, !noDeformCardsOverride {
                for group in meshGroups where deformingKinds.contains(group.kind) {
                    for k in 0..<group.count {
                        let i = group.start + k
                        guard i < total else { break }
                        let base = Int(soupBase[i])
                        let next = (i + 1 < total) ? Int(soupBase[i + 1]) : surfGPUDiffTriCount
                        guard next > base else { continue }
                        surfGPUDiffDeformDispatches.append(SurfDeformDispatch(
                            kind: group.kind, instanceIndex: i,
                            triBase: base, triCount: next - base))
                    }
                }
            }
        }
        rtSoupTriBaseBuffer = device.makeBuffer(
            bytes: soupBase, length: MemoryLayout<UInt32>.stride * soupBase.count,
            options: .storageModeShared)
        rtSoupTriBaseBuffer?.label = "Illuminatorama.rt.soupTriBase"
        rtSoupTriBaseCount = soupBase.count
    }

    /// In-place TLAS refit on the frame's command buffer — same topology, only
    /// transforms/materials changed. No vertex transform, no rebuild, no wait.
    private func refitRTAccel(_ cb: MTLCommandBuffer) {
        // AAA glass (#60): glass instances animate too (e.g. orbiting lenses), so
        // the refit must include them — and the instance-count check has to count
        // glass, or a glass scene would rebuild every frame (thrash → auto-disable).
        let glassFlat: [(kind: MeshKind, inst: IlluminatoramaGlassInstance)] =
            (rtGlassEnabled ? flattenGlass() : []).flatMap { g in g.insts.map { (g.kind, $0) } }
        guard let tlas = rtTLAS, let scratch = rtTLASScratch,
              let descBuf = rtInstanceDescBuffer, let dataBuf = rtInstanceDataBuffer,
              rtTLASInstanceCount == instances.count + rtCurveInstanceCount + glassFlat.count,
              glassFlat.count == rtGlassInstanceCount,
              !rtBLASList.isEmpty else {
            rebuildRTAccel(); return   // topology drifted — full rebuild
        }
        // Curve instances keep their rebuild-time descriptors (static rigid
        // placement); the triangle (opaque) + glass instances' transforms refresh.
        let total = instances.count
        let cap = rtTLASInstanceCount
        let instPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let descPtr = descBuf.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: cap)
        let dataPtr = dataBuf.contents().bindMemory(to: RTInstanceData.self, capacity: cap)
        // Update transforms + normal matrices in place; asIndex + normalBase
        // (set during rebuild) are unchanged for a stable topology.
        for group in meshGroups {
            guard meshes[group.kind] != nil else { continue }
            for k in 0..<group.count {
                let i = group.start + k
                guard i < total else { break }
                let inst = instPtr[i]
                descPtr[i].transformationMatrix = packed4x3(inst.modelMatrix)
                let nm = inst.normalMatrix
                dataPtr[i].nrm0 = nm.columns.0; dataPtr[i].nrm1 = nm.columns.1; dataPtr[i].nrm2 = nm.columns.2
                dataPtr[i].albedoTriBase = SIMD4(inst.albedo, dataPtr[i].albedoTriBase.w)
            }
        }
        // Glass instances [rtGlassInstanceBase, …): refresh transform + normal +
        // per-glass material (tint/ior may animate). Same flatten order as rebuild.
        if rtGlassInstanceCount > 0, let gdata = rtGlassDataBuffer {
            let gPtr = gdata.contents().bindMemory(to: IlluminatoramaRTGlassData.self, capacity: rtGlassInstanceCount)
            for (gi, entry) in glassFlat.enumerated() {
                let i = rtGlassInstanceBase + gi
                guard i < cap else { break }
                let inst = entry.inst
                descPtr[i].transformationMatrix = packed4x3(inst.modelMatrix)
                let nm = inst.normalMatrix
                dataPtr[i].nrm0 = nm.columns.0; dataPtr[i].nrm1 = nm.columns.1; dataPtr[i].nrm2 = nm.columns.2
                dataPtr[i].albedoTriBase = SIMD4(inst.tintIor.x, inst.tintIor.y, inst.tintIor.z, dataPtr[i].albedoTriBase.w)
                gPtr[gi] = IlluminatoramaRTGlassData(
                    tintIor: inst.tintIor, rdrf: inst.rdrf, dispersionPad: inst.dispersionPad)
            }
        }
        let tdesc = MTLInstanceAccelerationStructureDescriptor()
        tdesc.instanceDescriptorBuffer = descBuf
        tdesc.instanceCount = rtTLASInstanceCount
        tdesc.instancedAccelerationStructures = rtBLASList + rtCurveBLASList
        tdesc.usage = .refit
        guard let enc = cb.makeAccelerationStructureCommandEncoder() else { return }
        enc.refit(sourceAccelerationStructure: tlas, descriptor: tdesc,
                  destinationAccelerationStructure: tlas, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        rtTLASActive = true
    }

    private func ensureRTInstanceBuffers(capacity: Int) {
        let descLen = MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * max(1, capacity)
        if (rtInstanceDescBuffer?.length ?? 0) < descLen {
            rtInstanceDescBuffer = device.makeBuffer(length: descLen, options: .storageModeShared)
            rtInstanceDescBuffer?.label = "Illuminatorama.rt.instanceDesc"
        }
        let dataLen = MemoryLayout<RTInstanceData>.stride * max(1, capacity)
        if (rtInstanceDataBuffer?.length ?? 0) < dataLen {
            rtInstanceDataBuffer = device.makeBuffer(length: dataLen, options: .storageModeShared)
            rtInstanceDataBuffer?.label = "Illuminatorama.rt.instanceData"
        }
    }

    /// Instanced-RT lighting pass — traces the TLAS. Mirrors `encodeRTLightingPass`
    /// but with per-instance albedo/normal lookups. No-op unless a TLAS is live.
    private func encodeRTLightingTLASPass(_ cb: MTLCommandBuffer) {
        guard rtTLASActive, let basePipeline = rtTLASPipeline, let tlas = rtTLAS,
              let instData = rtInstanceDataBuffer, let objN = rtObjNormalBuffer else { return }
        // Curve primitives (#60 item 7): when the TLAS contains curve
        // instances the intersector's geometry-type contract must include
        // curves, so the kRTCurvesEnabled variant is REQUIRED (rebuild only
        // appends curve instances when the variant compiled).
        let curvesOn = rtCurveInstanceCount > 0
            && rtCurveSetDataBuffer != nil && rtCurvePoolPoints != nil
            && rtCurvePoolRadii != nil && rtCurvePoolSegments != nil
        let pipeline: MTLComputePipelineState
        if curvesOn {
            guard let cp = rtTLASCurvePipeline else { return }
            pipeline = cp
        } else {
            pipeline = basePipeline
        }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        rtInstFrameSeed &+= 1
        var u = RTInstUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos,
            sunDir: simd_normalize(rtSunDirection), sunSoftnessRad: max(0.0005, rtSunSoftnessRad),
            sunColor: rtSunColor, giStrength: rtGIStrength,
            skyAmbient: ambientColor, specStrength: rtSpecStrength,
            width: UInt32(width), height: UInt32(height),
            shadowRays: UInt32(max(1, min(16, rtShadowRays))),
            giRays: UInt32(max(1, min(16, rtGIRays))),
            frameSeed: rtInstFrameSeed, rayTMin: 0.004, maxGIDist: 60.0,
            reflStrength: max(0, rtReflStrength), reflMaxDist: max(0.1, rtReflMaxDistance),
            reflRoughnessCutoff: max(0, rtReflRoughnessCutoff),
            reflRays: UInt32(max(1, min(8, rtReflRays))),
            reflEnabled: rtReflectionsEnabled ? 1 : 0)
        // Surface cache read (P1c): on only when the grouped soup + cards + base
        // are all live this topology. The kernel gates every atlas read on this.
        let cacheOn = surfCacheActive && surfCardCount > 0
            && rtSoupTriBaseBuffer != nil && surfTriCardBuffer != nil
            && surfCacheCurrentAtlas != nil
            && rtSoupTriBaseCount == instances.count   // base buffer matches this topology
        if cacheOn {
            u.surfCacheEnabled = 1
            u.surfTileSize = UInt32(surfActiveTileSize)
            u.surfTilesPerRow = UInt32(surfTilesPerRow)
            u.surfAtlasW = UInt32(surfAtlasW); u.surfAtlasH = UInt32(surfAtlasH)
            u.surfTriCount = UInt32(rtTriangleCount)
        }
        // Debug isolation of the surface-cache term (see DebugTerm.surfaceCacheGI).
        u.debugSurfCacheGI = (debugTerm == .surfaceCacheGI) ? 1 : 0
        // Phase 5 / B0 — surface-cache variance heatmap (see DebugTerm.surfaceCacheVariance).
        u.debugSurfCacheVar = (debugTerm == .surfaceCacheVariance) ? 1 : 0
        // Phase 5 / A — residency feedback. The kernel marks `cardRequested[card]=1`
        // at each cache hit when enabled. Streaming (A1) NEEDS those marks, so it
        // implies feedback. The DRAIN (read + log + zero of the previous frame's
        // working set) is owned by exactly one site: `runStreamingResidency` (frame
        // start) when streaming is on, else the pure-feedback logger here. Draining in
        // both would zero the buffer before the residency pass could read it.
        let needFeedback = (surfaceCacheFeedback || surfaceCacheStreaming)
            && cacheOn && surfCardRequestedBuffer != nil
        u.surfFeedbackEnabled = needFeedback ? 1 : 0
        if surfaceCacheFeedback, !surfaceCacheStreaming,
           let rb = surfCardRequestedBuffer, surfCardCount > 0 {
            let p = rb.contents().bindMemory(to: UInt32.self, capacity: surfCardCount)
            var requested = 0
            for i in 0..<surfCardCount where p[i] != 0 { requested += 1 }
            if requested > 0 {
                let pct = Int((Double(requested) / Double(surfCardCount) * 100).rounded())
                Self.recordSurfCacheStats("feedback: requestedCards=\(requested) of \(surfCardCount) (\(pct)% working set)")
            }
            memset(rb.contents(), 0, surfCardCount * MemoryLayout<UInt32>.stride)
        }
        if curvesOn {
            u.curveInstanceBase = UInt32(instances.count)
            u.curveSetCount = UInt32(rtCurveInstanceCount)
        }
        memcpy(rtInstUniformBuffer.contents(), &u, MemoryLayout<RTInstUniforms>.stride)

        guard let enc = timedComputeEncoder(cb, "rtLightingTLAS") else { return }
        enc.label = "Illuminatorama.rt.tlas"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(depthTexture, index: 0)
        enc.setTexture(gbufferNormalRgh, index: 1)
        enc.setTexture(gbufferAlbedoMet, index: 2)
        enc.setTexture(hdrCompositeTexture, index: 3)
        enc.setTexture(equirectSky ?? dummySkyTexture, index: 4)
        // Current radiance atlas (read at hits); dummy when the cache is off.
        enc.setTexture(cacheOn ? surfConsumerAtlas : (equirectSky ?? dummySkyTexture), index: 5)
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        enc.setBuffer(instData, offset: 0, index: 1)
        enc.setBuffer(objN, offset: 0, index: 2)
        enc.setBuffer(rtInstUniformBuffer, offset: 0, index: 3)
        // Card buffers + per-instance soup base; dummies keep the binding valid
        // when the cache is off (the kernel never reads them then).
        enc.setBuffer(cacheOn ? surfTriCardBuffer : instData, offset: 0, index: 4)
        enc.setBuffer(cacheOn ? surfTriUVaBuffer : instData, offset: 0, index: 5)
        enc.setBuffer(cacheOn ? surfTriUVcBuffer : instData, offset: 0, index: 6)
        enc.setBuffer(cacheOn ? rtSoupTriBaseBuffer : instData, offset: 0, index: 7)
        enc.setBuffer(cacheOn ? surfCardRectBuffer : instData, offset: 0, index: 8)  // per-card atlas rect
        enc.setBuffer(cacheOn ? surfCardBuffer : (surfCardDummyBuffer ?? instData), offset: 0, index: 9)  // per-card material (L_out reconstruction); dummy must be ≥ 1 SurfCard for arg validation
        // Curve-set data + pooled geometry (#60 item 7); dummies keep the
        // bindings valid for the base (curve-free) pipeline variant.
        enc.setBuffer(curvesOn ? rtCurveSetDataBuffer : instData, offset: 0, index: 10)
        enc.setBuffer(curvesOn ? rtCurvePoolPoints : instData, offset: 0, index: 11)
        enc.setBuffer(curvesOn ? rtCurvePoolRadii : instData, offset: 0, index: 12)
        enc.setBuffer(curvesOn ? rtCurvePoolSegments : instData, offset: 0, index: 13)
        // Phase 5 / A — residency feedback (buffer 14). Bound always (kernel
        // signature requires it); written only when `surfFeedbackEnabled`. Dummy =
        // instData when the cache/feedback buffer isn't live (never written then).
        enc.setBuffer(surfCardRequestedBuffer ?? instData, offset: 0, index: 14)
        // The TLAS references the BLASes, which reference the mesh vertex/index
        // buffers (and the curve BLASes the pooled curve buffers) — all must be
        // resident for the intersector.
        for blas in rtBLASList { enc.useResource(blas, usage: .read) }
        for blas in rtCurveBLASList { enc.useResource(blas, usage: .read) }
        for buf in rtResidentBuffers { enc.useResource(buf, usage: .read) }
        dispatch(enc, pipeline: pipeline, width: width, height: height,
                 maxThreadsOverride: rtThreadgroupMax)
        enc.endEncoding()
    }

    // ── Particle fields (SimEngine registry) ──────────────────────────
    //
    // The extractor refreshes this each frame with the active scene's
    // `ParticleFieldSource`s (`SimEngine.particleFields.sources(forScene:)`).
    // Scenes that drive their own GPU particle simulation register a source
    // from their renderer-shim — keeps simulation ownership with the host
    // and renders through the overlay for free, with no SceneKit round-trip.
    public func setParticleFields(_ fields: [ParticleFieldSource]) {
        particleFields = fields
    }

    /// Additive HDR draw of every active-scene particle field (e.g. the
    /// Illuminatorama Room dust) into `hdrTexture`. Called only from
    /// `encodePostResolveFX` (Phase 4.43), AFTER the TAA resolve has seeded
    /// `hdrTexture` with the resolved opaque image — keeping these motion-
    /// vectorless moving points out of the temporal history so they can't
    /// trail. (Pre-4.43 this ran before the SSR gather so sparks reflected in
    /// nearby metals; that placement is what produced the dust trails.)
    private func encodeExternalPointDraw(_ cb: MTLCommandBuffer) {
        guard !particleFields.isEmpty else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = hdrTexture
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.extParticles.draw"
        enc.setRenderPipelineState(extParticleDrawPipeline)
        enc.setDepthStencilState(particleDepthState)
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 2)
        for field in particleFields {
            enc.setVertexBuffer(field.positionBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(field.colorBuffer,    offset: 0, index: 1)
            // Matches the Metal-side `ExtParticleParams` struct: four
            // `uint` strides/offsets + two `float` scalars = 24-byte cluster.
            var params = ExternalPointParams(
                positionStrideFloats: field.positionStrideFloats,
                colorStrideFloats:    field.colorStrideFloats,
                positionOffsetFloats: field.positionOffsetFloats,
                colorOffsetFloats:    field.colorOffsetFloats,
                pointSize:            field.pointSize,
                colorScale:           field.colorScale
            )
            enc.setVertexBytes(&params,
                                length: MemoryLayout<ExternalPointParams>.stride,
                                index: 3)
            enc.drawPrimitives(type: .point,
                               vertexStart: 0,
                               vertexCount: field.vertexCount)
        }
        enc.endEncoding()
    }

    // ── Particle streak path (Phase 4.44b) ──────────────────────────────────
    //
    // Velocity-aligned billboard quad renderer for fireworks / spark streaks.
    // Hosts build FWStreakVertex buffers via their solver's `encodeBuildStreaks`
    // kernel and pass the pre-built Metal buffers here so the HDR additive draw
    // lands AFTER the TAA resolve — same timing as the point-sprite pass above,
    // so streak particles have no temporal trailing.
    //
    // API: call `setParticleStreaks([StreakSource])` from the host's tick();
    // the renderer calls `encodeStreakDraw(cb)` inside `encodePostResolveFX`.

    /// Host-supplied velocity-aligned streak source for `setParticleStreaks`.
    /// One source → one draw call per frame. Build the vertex / UV / index
    /// buffers via the solver's `encodeBuildStreaks` kernel (4 vertices + 6
    /// indices per particle) and pass them here already on the GPU.
    public struct StreakSource {
        /// `FWStreakVertex` buffer (stride 32): float4 position + float4 color,
        /// where `color.a` carries the per-particle stretch factor.
        public var positionColorBuffer: MTLBuffer
        /// Parallel `SIMD2<Float>` UV buffer (stride 8).
        public var uvBuffer: MTLBuffer
        /// UInt32 triangle index buffer — 6 indices per particle (2 triangles).
        public var indexBuffer: MTLBuffer
        /// Number of indices to draw. Typically `particleCapacity × 6`.
        public var indexCount: Int
        /// Soft Gaussian sparkle sprite (128×128, mipmapped). Sampled by the
        /// streak fragment shader for the sphere-imposter surface shading.
        public var spriteTexture: MTLTexture
        /// Per-draw brightness (`intensity`) and saturation (`saturationPow`)
        /// knobs forwarded to `ParticleStreakParams` in the shader.
        public var params: ParticleStreakParams

        public init(positionColorBuffer: MTLBuffer,
                    uvBuffer:            MTLBuffer,
                    indexBuffer:         MTLBuffer,
                    indexCount:          Int,
                    spriteTexture:       MTLTexture,
                    params:              ParticleStreakParams = ParticleStreakParams()) {
            self.positionColorBuffer = positionColorBuffer
            self.uvBuffer            = uvBuffer
            self.indexBuffer         = indexBuffer
            self.indexCount          = indexCount
            self.spriteTexture       = spriteTexture
            self.params              = params
        }
    }

    /// Replace the active streak-source list for this frame. Call from the
    /// host's tick loop after `encodeBuildStreaks` has updated the vertex buffer.
    public func setParticleStreaks(_ sources: [StreakSource]) {
        particleStreaks = sources
    }

    /// Render every `StreakSource` into `hdrTexture` using `particleStreakVS/FS`
    /// (velocity-aligned capsule impostors, additive HDR blend, depth-test no-write).
    /// One render encoder for all sources; buffers / textures rebind per draw.
    private func encodeStreakDraw(_ cb: MTLCommandBuffer) {
        guard !particleStreaks.isEmpty, let pipeline = streakPipeline else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture     = hdrTexture
        pass.colorAttachments[0].loadAction  = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture     = depthTexture
        pass.depthAttachment.loadAction  = .load
        pass.depthAttachment.storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.streaks"
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(particleDepthState)   // depth-test, no depth-write
        enc.setCullMode(.none)
        // `frameUniformBuffer` starts with `viewProjection` (simd_float4x4,
        // offset 0) — matches `constant float4x4& viewProjection [[ buffer(2) ]]`
        // in `particleStreakVS`. Bound once; shared across all draws in this pass.
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 2)
        for source in particleStreaks where source.indexCount > 0 {
            enc.setVertexBuffer(source.positionColorBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(source.uvBuffer,            offset: 0, index: 1)
            var p = source.params
            enc.setFragmentBytes(&p,
                                 length: MemoryLayout<ParticleStreakParams>.stride,
                                 index: 0)
            enc.setFragmentTexture(source.spriteTexture, index: 0)
            enc.drawIndexedPrimitives(type: .triangle,
                                      indexCount:        source.indexCount,
                                      indexType:         .uint32,
                                      indexBuffer:       source.indexBuffer,
                                      indexBufferOffset: 0)
        }
        enc.endEncoding()
    }

    /// "object-fit: cover" UV sub-rect for presenting the FIXED output canvas
    /// (`outputWidth × outputHeight`) into a display viewport of a possibly
    /// different aspect. Returns `(scaleX, scaleY, offsetX, offsetY)` such that
    /// `texCoord = uv * scale + offset` samples a centered crop of the canvas:
    /// the canvas is scaled to FILL the viewport and the overflow on the longer
    /// axis is cropped, centered — no stretch, no letterbox. Returns identity
    /// `(1, 1, 0, 0)` when the aspects already match (the steady state, so the
    /// present is a 1:1 blit and costs nothing). Build a SceneKit
    /// `background.contentsTransform` from the result; see the host controllers.
    ///
    /// The canvas should be ≥ the viewport on the cropped axis or this upscales
    /// (accepted cost of a fixed canvas — see the controller's tick docstring).
    /// Reallocate render targets at a new size. Targets are dropped first so
    /// the device doesn't hold two copies at the peak; if allocation fails we
    /// log and keep the old size — better a stale image than a crash.
    /// Resize the renderer to a new OUTPUT size. The internal pipeline is
    /// sized to `outputW × internalRenderScale`; pass the same dimensions
    /// you would have passed when there was no SSAA (the SCNView's
    /// drawable size). No-op when neither output nor internal size needs
    /// to change.
    ///
    /// `outputTexture` is preserved across resizes when the OUTPUT
    /// dimensions don't change — this matters for consumers that bound
    /// the texture into a SceneKit material at init time. An SSAA-only
    /// change (internal scale slider) keeps `outputTexture`'s identity
    /// stable so the host's `scene.background.contents = renderer.outputTexture`
    /// binding keeps working without any per-frame rebind. Hosts that
    /// also handle viewport resizes (where output dims DO change) still
    /// need to rebind — the `IlluminatoramaOverlay` does this in its tick
    /// by ObjectIdentifier-comparing the texture before and after.
    /// UV transform (scaleU, scaleV, offsetU, offsetV) that COVER-crops the fixed
    /// render canvas (aspect `outputWidth:outputHeight`) to a live viewport of
    /// `viewportWidth:viewportHeight`, centred — fills the viewport, never
    /// stretches, never letterboxes. Bind it to `background.contentsTransform`
    /// (`m11=scaleU, m22=scaleV, m41=offsetU, m42=offsetV`). This avoids resizing
    /// the renderer to the drawable (which reallocates every GPU target and
    /// resets temporal accumulation, flashing noise on every window drag).
    public func coverUVRect(viewportWidth: Int, viewportHeight: Int) -> SIMD4<Float> {
        guard viewportWidth > 0, viewportHeight > 0,
              outputWidth > 0, outputHeight > 0 else { return SIMD4(1, 1, 0, 0) }
        let canvasAR = Float(outputWidth) / Float(outputHeight)
        let viewAR   = Float(viewportWidth) / Float(viewportHeight)
        if viewAR > canvasAR {
            // Viewport wider than canvas → crop top/bottom (sample full width).
            let scaleV = canvasAR / viewAR
            return SIMD4(1, scaleV, 0, (1 - scaleV) * 0.5)
        } else {
            // Viewport taller/narrower → crop sides (sample full height).
            let scaleU = viewAR / canvasAR
            return SIMD4(scaleU, 1, (1 - scaleU) * 0.5, 0)
        }
    }

    public func resize(width: Int, height: Int) {
        let outW = max(1, width)
        let outH = max(1, height)
        let (inW, inH) = Self.internalDims(outputW: outW, outputH: outH,
                                            scale: internalRenderScale)
        guard outW != self.outputWidth || outH != self.outputHeight ||
              inW  != self.width       || inH  != self.height else { return }
        let outputDimsChanged = (outW != self.outputWidth || outH != self.outputHeight)
        do {
            let t = try Self.makeTargets(device: device,
                                          internalW: inW, internalH: inH,
                                          outputW: outW, outputH: outH)
            self.gbufferAlbedoMet    = t.albedoMet
            self.gbufferNormalRgh    = t.normalRgh
            self.gbufferEmission     = t.emission
            self.depthTexture        = t.depth
            self.previousDepthTexture = try Self.makeDepthLike(device: device, t.depth)
            self.hdrTexture          = t.hdr
            self.aoTexture           = t.ao
            self.hdrCompositeTexture = t.hdrComposite
            self.rtDiffuseTexture    = t.rtDiffuse
            self.velocityTexture     = t.velocity
            self.historyA            = t.historyA
            self.historyB            = t.historyB
            self.bloomBrightHalf     = t.bloomBright
            self.bloomBlurHHalf      = t.bloomBlurH
            self.bloomBlurVHalf      = t.bloomBlurV
            self.aoFilteredTexture   = t.aoFiltered
            self.aoHistoryA          = t.aoHistoryA
            self.aoHistoryB          = t.aoHistoryB
            self.ssrRawTexture       = t.ssrRaw
            self.ssrHistoryA         = t.ssrHistoryA
            self.ssrHistoryB         = t.ssrHistoryB
            self.rtGIHistoryA        = t.rtGIHistoryA
            self.rtGIHistoryB        = t.rtGIHistoryB
            self.irrCacheA           = t.irrCacheA
            self.irrCacheB           = t.irrCacheB
            self.aoSampleCount       = t.aoSampleCount
            self.ssrSampleCount      = t.ssrSampleCount
            self.giSampleCount       = t.giSampleCount
            self.giVariance          = t.giVariance
            self.giAtrousA           = t.giAtrousA
            self.giAtrousB           = t.giAtrousB
            // Preserve outputTexture identity for SSAA-only changes — see
            // the docstring. On an output-size change, rebuild the whole
            // presented-buffer pool at the new size (reusing `t.ldr` as
            // slot 0) and reset the present bookkeeping. `t.ldr` would
            // otherwise go out of scope and be freed with the local `t`.
            if outputDimsChanged {
                var pool: [MTLTexture] = [t.ldr]
                for i in 1..<Self.ldrPoolCount {
                    if let extra = Self.makeLDRTexture(
                        device: device, w: outW, h: outH,
                        label: "Illuminatorama.ldr.\(i)") {
                        pool.append(extra)
                    }
                }
                self.ldrPool = pool
                self.presentedIdx = 0
                self.prevPresentedIdx = -1
                self.presentSync.reset()
                self.outputTexture = t.ldr
                self.tonemapWriteTarget = t.ldr
            }
            self.width        = inW
            self.height       = inH
            self.outputWidth  = outW
            self.outputHeight = outH
            // Internal and output share the same multiplier, so aspect is
            // identical for both. Drive camera off the output size as the
            // canonical user-facing dimension.
            self.camera.aspect = Float(outW) / Float(outH)
            // Resize destroys all history textures — next frame has nothing
            // valid to reproject against for any temporal pass.
            self.taaNeedsFirstFrame  = true
            self.aoNeedsFirstFrame   = true
            self.ssrNeedsFirstFrame  = true
            self.rtGINeedsFirstFrame = true
        } catch {
            Self.log.error("Illuminatorama resize(out \(outW)x\(outH) / int \(inW)x\(inH)) failed: \(error.localizedDescription) — keeping out \(self.outputWidth)x\(self.outputHeight) int \(self.width)x\(self.height)")
        }
    }

    /// Map an output size + scale to the internal render-target size,
    /// rounding to multiples of 2 so the half-res bloom / AO textures
    /// don't end up off-by-one on odd output dimensions.
    private static func internalDims(outputW: Int, outputH: Int,
                                      scale: Float) -> (Int, Int) {
        let s = max(1.0, scale)
        let iw = max(2, Int((Float(outputW) * s).rounded()) & ~1)
        let ih = max(2, Int((Float(outputH) * s).rounded()) & ~1)
        return (iw, ih)
    }

    // ── Render ────────────────────────────────────────────────────────────────

    /// - Parameter blocking: when true (headless snapshot export path only),
    ///   WAIT for an in-flight slot instead of dropping the frame. The
    ///   interactive tick passes `false` so a back-pressured GPU drops the tick
    ///   rather than stalling the main thread; the snapshot path passes `true`
    ///   because a dropped export render leaves `outputTexture` on a stale /
    ///   never-rendered pool buffer → captured as magenta (no `cb.error`).
    /// Returns `true` if this call actually produced a frame, `false` if it
    /// dropped (GPU still draining the previous frames — the non-blocking
    /// `inFlightSemaphore` path) or bailed on a transient failure. Callers
    /// use the result to count FPS against *rendered* frames rather than the
    /// fixed Timer cadence, and to skip re-presenting a stale `outputTexture`.
    @discardableResult
    public func render(blocking: Bool = false) -> Bool {
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
            inFlightSemaphore.wait()
        } else {
            // Dropped frame — GPU hasn't drained the previous frames. Report
            // `false` so the caller doesn't count it as a rendered frame.
            guard inFlightSemaphore.wait(timeout: .now()) == .success else { return false }
        }
        // ── Panel is the single source of truth for TAA on/off ──────────────
        // Authority lives HERE, once, instead of in every scene controller. The
        // panel toggle (IlluminatoramaSharedSettings.shared.taaEnabled) wins for
        // every Illuminatorama scene unless that scene explicitly opts out via
        // `sharedTAAOverride`. This is why a scene's own `renderer.taaEnabled = …`
        // write no longer leaks the toggle out of sync — it's overwritten here
        // each frame before any TAA gating reads it.
        taaEnabled = Self.taaEnvOverride ?? sharedTAAOverride ?? IlluminatoramaSharedSettings.shared.taaEnabled
        // Track enable-transitions so a re-enable re-primes each temporal history.
        if taaEnabled && !previousTaaEnabled { taaNeedsFirstFrame = true }
        previousTaaEnabled = taaEnabled
        if ssaoDenoiseEnabled && !previousSsaoEnabled { aoNeedsFirstFrame  = true }
        previousSsaoEnabled = ssaoDenoiseEnabled
        if ssrDenoiseEnabled  && !previousSsrEnabled  { ssrNeedsFirstFrame = true }
        previousSsrEnabled  = ssrDenoiseEnabled
        if rtGITemporalEnabled && !previousRTGITemporalEnabled { rtGINeedsFirstFrame = true }
        previousRTGITemporalEnabled = rtGITemporalEnabled
        // ── Panel is the single source of truth for lens aberration too ─────
        // CA / spherical aberration / fringe (+ post-FX easing) have NO per-scene
        // overrides anywhere, so apply them HERE once for every Illuminatorama
        // scene — including the ones that never wired post-FX and every future
        // scene, which inherit them with zero per-scene code. (Bloom is NOT here:
        // scenes art-direct it and `render()` runs after their tick, so it opts in
        // via `applySharedPostFX`.) Targets are set before `uploadFrameUniforms`
        // below, which eases toward them.
        if appliesSharedLensFX { applySharedLensFX(); applySharedColorGrade() }
        // Ping-pong the instance buffer FIRST: after this swap,
        // currentInstanceBuffer is what was previousInstanceBuffer last frame
        // (so uploadInstances overwrites it with this frame's data), and
        // previousInstanceBuffer now points to last frame's data.
        useBufferA.toggle()
        // Advance the per-frame input-buffer ring in lockstep. From here on,
        // `frameUniformBuffer` / the light buffers / `superquadricParamBuffer`
        // resolve to THIS frame's slot — distinct from the slot the previously
        // pipelined frame (still on the GPU) is reading. Done before the uploads
        // below write into the new slot, and before any encode binds it.
        frameRingIndex = (frameRingIndex + 1) % frameUniformRing.count
        if Self.raceProbeEnabled {
            if Self.raceProbePin0 { frameRingIndex = 0 }   // negative control: pre-fix single-buffer
            presentSync.raceProbeBeginFrame(slot: frameRingIndex)
        }
        ensureBufferCapacity()  // may set taaNeedsFirstFrame on growth
        updateCascades()
        // Phase 4.10 — compute per-spot shadow matrices + slice indices
        // BEFORE `uploadSpotLights` so the GPU buffer sees the current
        // frame's values. Done unconditionally even when shadows are
        // disabled (cheap CPU math); the shadow PASS gates on
        // `spotShadowsEnabled` to skip the expensive depth render.
        updateSpotShadows()
        uploadFrameUniforms()
        uploadInstances()
        uploadPointLights()
        uploadSpotLights()
        uploadAreaLights()
        uploadExtraDirectionals()

        // Pick a free pool buffer for this frame's tonemap output (never the
        // presented one SceneKit may sample, nor the one still in flight).
        let writeIdx = acquireWriteTarget()
        tonemapWriteTarget = ldrPool[writeIdx]

        guard let cb = commandQueue.makeCommandBuffer() else {
            Self.log.error("makeCommandBuffer() returned nil")
            inFlightSemaphore.signal()  // release the slot acquired above
            return false
        }
        // Reserve this pool slot until the GPU finishes writing it, so a second
        // pipelined frame (maxFramesInFlight = 2) picks a different buffer. Cleared
        // by `markCompleted` in the completion handler below. Marked after the
        // makeCommandBuffer guard so a failed allocation doesn't leak the slot.
        presentSync.markInFlight(writeIdx)
        cb.label = "Illuminatorama.frame"
        passTimer.beginFrame()   // reset per-pass GPU-timer sample assignment (env-gated)

        // Opt-in GPU-resident instance write (additive — see `onEncodeGPUInstances`).
        // `uploadInstances()` above already reserved the slots + built `meshGroups`;
        // here, before any pass reads the instance buffer, a scene kernel can
        // overwrite its group's slots with GPU-computed data. No-op (closure nil)
        // for every scene that doesn't set the hook.
        if let hook = onEncodeGPUInstances {
            let groups = meshGroups
            hook(cb, currentInstanceBuffer) { kind in
                guard let g = groups.first(where: { $0.kind == kind }) else { return nil }
                return g.start ..< (g.start + g.count)
            }
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

        encodeShadowPasses(cb)
        encodeSpotShadowPasses(cb)
        encodeGBufferPass(cb)
        encodeIBLBakeIfNeeded(cb)
        encodeDDGIFrame(cb)
        // Phase 4.39: SSAO → bilateral spatial → temporal → lighting (reads denoised AO).
        encodeSSAOPass(cb)
        encodeSSAOSpatialFilter(cb)
        encodeSSAOTemporalPass(cb)
        encodeLightingPass(cb)
        // Phase 4.43 — additive particle / external-point draws (emitter
        // sparks + dust motes) MOVED to AFTER the TAA resolve (see
        // `encodePostResolveFX`). They have no motion vectors of their own,
        // so flowing them through TAA history reprojected each moving mote as
        // static and retained ~90% of the previous frame → visible trails.
        // Trade-off vs the old Phase 4.42 placement: these FX no longer feed
        // the SSR/RT reflection chain (they reflected in nearby metals). For
        // dust/sparks that's an acceptable loss; trail-free wins.
        // Phase 4.39: SSR gather → temporal → composite into hdrComposite.
        encodeSSRGather(cb)
        encodeSSRTemporalPass(cb)
        encodeSSRComposite(cb)
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
        if rtTLASActive && rtEnabled && rtOpaqueLightingEnabled {
            encodeRTLightingTLASPass(cb)
        } else if !rtTLASActive {
            encodeRTLightingPass(cb)
            // Temporally accumulate the RT diffuse (GI) so it keeps converging
            // under camera motion (kills the wall crawl) BEFORE the spatial
            // denoise reads it. No-op unless rtGITemporalEnabled.
            encodeRTGITemporalAccum(cb)
            // Phase 4.44 — SVGF à-trous cascade replaces the fixed-radius
            // bilateral when enabled. Falls back to the original bilateral path.
            if svgfEnabled && rtGITemporalEnabled && rtEnabled && rtSupported {
                encodeSVGFCascade(cb)
            } else {
                // Bilateral-clean the RT diffuse (shadow+GI) buffer into the
                // composite before TAA. The TLAS path still composites diffuse
                // inline, so this only runs for the direct-AS (room) path.
                encodeRTDenoiseComposite(cb)
            }
        }
        // Volumetric god-ray shaft in the air. No-op unless enabled.
        encodeVolumetricPass(cb)
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
        // copy. In-place copy is safe under maxFramesInFlight==1 — this command
        // buffer fully completes before the next frame's G-buffer overwrites
        // `depthTexture`. Skipped when TAA is off (nothing reads it); the
        // enable-transition's taaNeedsFirstFrame covers the first re-enabled frame.
        if taaEnabled, let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: depthTexture, to: previousDepthTexture)
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
        // Depth of field on the resolved HDR before bloom, so out-of-focus
        // bright dust motes bloom into soft bokeh. No-op unless enabled.
        encodeDOFPass(cb)
        encodeBloomPasses(cb)
        encodeTonemapPass(cb)

        // Signal — on GPU completion (background thread) — that this pool
        // buffer is fully rendered and safe to present. `promoteCompletedBuffer`
        // (next tick, main actor) reads this and swaps `outputTexture`. Capturing
        // a plain `Int` keeps the handler `Sendable`-clean.
        let completedIdx = writeIdx
        let raceSlot = Self.raceProbeEnabled ? frameRingIndex : -1
        let sync = presentSync
        let sem = inFlightSemaphore
        let meter = gpuMeter
        let pt = passTimer
        let errPath = ProcessInfo.processInfo.environment["VIZ_ILLUMI_CBERROR_PATH"]
        cb.addCompletedHandler { buf in
            meter.record(buf)
            let gpuMs = (buf.gpuEndTime - buf.gpuStartTime) * 1000.0
            pt.resolve(frameGpuMs: gpuMs)
            // Diagnostic (sandbox eats os_log): on a GPU fault, write the error +
            // status to a sidecar file so the headless harness can read WHY the
            // frame faulted (vs the swallowed magenta). No-op without the env var.
            if buf.status == .error || buf.error != nil, let p = errPath {
                let nsErr = buf.error as NSError?
                let line = "frame status=\(buf.status.rawValue) domain=\(nsErr?.domain ?? "?") code=\(nsErr?.code ?? 0) \(nsErr?.localizedDescription ?? "")\n"
                if let data = line.data(using: .utf8) {
                    if let h = FileHandle(forWritingAtPath: p) {
                        h.seekToEndOfFile(); h.write(data); try? h.close()
                    } else { try? data.write(to: URL(fileURLWithPath: p)) }
                }
            }
            sync.markCompleted(completedIdx)
            if raceSlot >= 0 { sync.raceProbeCompleteFrame(slot: raceSlot) }
            // Adaptive latency guard (replaces a plain `sem.signal()`): update the
            // GPU-time EMA and release THIS frame's permit — UNLESS frames are heavy
            // enough that a 2nd pipelined frame only adds ~one frame of input latency
            // without raising throughput (GPU frame >> tick). In that case HOLD the
            // permit, shrinking the effective in-flight depth from 2 to 1 so heavy RT
            // scenes stay responsive. Converges back to 2 when frames get light again.
            sync.frameCompletedAdaptive(gpuMs: gpuMs, semaphore: sem)
        }
        cb.commit()

        // Bookkeeping for next frame:
        // - Save this frame's jittered VP as the previous VP next time.
        // - Advance the jitter index so the Halton sequence walks forward.
        // - We've now written a valid history frame.
        previousViewProjection = lastFrameViewProjection
        if taaEnabled {
            taaFrameIndex &+= 1
            taaNeedsFirstFrame = false
            historyToggle.toggle()
        }
        if ssaoDenoiseEnabled {
            aoNeedsFirstFrame = false
            aoHistoryToggle.toggle()
        }
        if ssrDenoiseEnabled {
            ssrNeedsFirstFrame = false
            ssrHistoryToggle.toggle()
        }
        if rtGITemporalEnabled && rtEnabled && rtSupported {
            rtGINeedsFirstFrame = false
            rtGIHistoryToggle.toggle()
        }
        if ddgiIrrCacheEnabled {
            irrCacheToggle.toggle()
        }
        return true
    }

    // ── Presented-buffer pool helpers ─────────────────────────────────────────

    // 4 = presented + previously-presented (both still readable by consumers) +
    // the two frames that may be in flight at once (`maxFramesInFlight = 2`).
    static let ldrPoolCount = 4

    /// Allocate one LDR (`bgra8Unorm_srgb`) output texture for the present pool.
    /// sRGB format: the tonemap writes LINEAR and the GPU store applies the sRGB
    /// OETF. SceneKit samples this as `background.contents`, decodes it back to
    /// linear, then re-encodes for its drawable — one correct round-trip. (A
    /// non-sRGB format + a manual `pow(1/2.2)` in the shader double-encoded
    /// against SceneKit's pass and washed the colours out.) sRGB targets don't
    /// support `.shaderWrite`; the tonemap is a render pass, so we don't need it.
    private static func makeLDRTexture(device: MTLDevice, w: Int, h: Int,
                                        label: String) -> MTLTexture? {
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .bgra8Unorm_srgb
        d.width = max(1, w)
        d.height = max(1, h)
        d.usage = [.shaderRead, .renderTarget]
        d.storageMode = .private
        let t = device.makeTexture(descriptor: d)
        t?.label = label
        return t
    }

    /// Promote the most-recently-completed pool buffer to `outputTexture`.
    /// Called at the top of each `render()` (and after the snapshot barrier)
    /// so consumers always bind a fully-rendered buffer. No-op until a buffer
    /// finishes. Internal (not private) so `IlluminatoramaOverlay.renderForSnapshot`
    /// can promote the just-rendered frame after its completion barrier.
    func promoteCompletedBuffer() {
        guard let idx = presentSync.takeCompleted(),
              idx >= 0, idx < ldrPool.count, idx != presentedIdx else { return }
        prevPresentedIdx = presentedIdx
        presentedIdx = idx
        outputTexture = ldrPool[idx]
        if Self.swapProbeEnabled { recordSwapInterval() }   // issue #65 jitter probe
    }

    /// Real frame-delivery jitter probe (issue #65). Records the interval since
    /// the last `outputTexture` swap and writes a rolling p50/p95/p99 to
    /// `VIZ_ILLUMI_SWAP_PATH`. Main-actor only (called from promoteCompletedBuffer),
    /// so no locking. No-op cost when the probe env is off (guarded at the call).
    private func recordSwapInterval() {
        let now = CACurrentMediaTime()
        defer { swapLastTime = now }
        guard swapLastTime > 0 else { return }   // first swap: no interval yet
        let ms = (now - swapLastTime) * 1000.0
        swapIntervalsMs.append(ms)
        if swapIntervalsMs.count > 4000 { swapIntervalsMs.removeFirst(swapIntervalsMs.count - 4000) }
        guard let p = swapProbePath, swapIntervalsMs.count >= 2 else { return }
        let sorted = swapIntervalsMs.sorted()
        func pct(_ q: Double) -> Double { sorted[min(sorted.count - 1, max(0, Int(q * Double(sorted.count - 1)))) ] }
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        let p50 = pct(0.50), p95 = pct(0.95), p99 = pct(0.99)
        let fps = mean > 0 ? 1000.0 / mean : 0
        let line = String(format: "swaps=%d fps=%.1f meanMs=%.2f p50=%.2f p95=%.2f p99=%.2f maxMs=%.2f\n",
                          sorted.count + 1, fps, mean, p50, p95, p99, sorted.last ?? 0)
        if let d = line.data(using: .utf8) { try? d.write(to: URL(fileURLWithPath: p)) }
    }

    /// Choose a pool index to render into this frame: anything that is neither
    /// the currently presented buffer, NOR the previously presented buffer
    /// (SceneKit / the cover blit may still be sampling either), NOR a buffer
    /// whose own render is STILL IN FLIGHT. The in-flight exclusion matters now
    /// that `maxFramesInFlight = 2`: two consecutive `render()` calls can both be
    /// queued before either completes, and they must target DIFFERENT buffers or
    /// the second would clobber the first before it is ever presented. With the
    /// 4-buffer pool the worst case is presented + prev + one already in flight =
    /// 3 excluded, leaving exactly one free slot for this frame.
    private func acquireWriteTarget() -> Int {
        for i in 0..<ldrPool.count
        where i != presentedIdx && i != prevPresentedIdx && !presentSync.isInFlight(i) {
            return i
        }
        // Degenerate fallback (pool too small or all excluded): avoid the presented.
        return (presentedIdx + 1) % max(1, ldrPool.count)
    }

    private func blankSky() {
        guard let cb = commandQueue.makeCommandBuffer() else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = outputTexture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.85, green: 0.86, blue: 0.90, alpha: 1)
        if let enc = cb.makeRenderCommandEncoder(descriptor: pass) {
            enc.label = "Illuminatorama.blankSky"
            enc.endEncoding()
        }
        cb.commit()
    }

    // ── Phase 2.5: Cascaded shadow maps ──────────────────────────────────────
    //
    // Per frame we fit three orthographic light frustums to camera-frustum
    // sub-slices (PSSM, λ=0.5 between log and uniform splits) and rasterise
    // scene depth into a depth2d_array. Disabled cascades still get a depth
    // pass clear, but no draws are issued — keeps the array consistent so
    // the lighting kernel can read from any slice without UB.

    private func encodeShadowPasses(_ cb: MTLCommandBuffer) {
        guard shadowsEnabled else { return }
        for cascade in 0..<Self.cascadeCount {
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = shadowMap
            pass.depthAttachment.slice = cascade
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1.0

            guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { continue }
            enc.label = "Illuminatorama.shadow.c\(cascade)"
            enc.setRenderPipelineState(shadowPipeline)
            enc.setDepthStencilState(depthState)
            // Front-face cull keeps the shadow projection biased on the
            // back faces of geometry, which combats acne on lit surfaces
            // without the peter-panning a constant bias alone would cause.
            enc.setCullMode(.front)
            enc.setFrontFacing(.counterClockwise)

            var lightVP = cascadeVPs[cascade]
            enc.setVertexBytes(&lightVP,
                               length: MemoryLayout<simd_float4x4>.stride,
                               index: 3)

            // Phase 4.12 — instanced draw per mesh kind via the recipe
            // built in `uploadInstances`. Same win as the G-buffer pass:
            // one draw call replaces N per-instance draws per cascade /
            // per spot.
            let instStride = MemoryLayout<IlluminatoramaInstance>.stride
            for group in meshGroups {
                guard let mesh = meshes[group.kind] else { continue }
                let off = instStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: mesh.indexCount,
                    indexType: mesh.indexType,
                    indexBuffer: mesh.indexBuffer,
                    indexBufferOffset: 0,
                    instanceCount: group.count,
                    baseVertex: 0,
                    baseInstance: 0
                )
            }
            enc.endEncoding()
        }
    }

    /// Refit the three cascade VPs to the current camera + sun. Result lands
    /// in `cascadeVPs` and `cascadeSplitsView`; `uploadFrameUniforms` then
    /// stuffs them into the frame uniform buffer.
    private func updateCascades() {
        let near = max(camera.zNear, 0.05)
        let far = min(shadowMaxDistance, camera.zFar)
        let count = Self.cascadeCount
        let lambda: Float = 0.5

        // PSSM split distances (in view-space positive distance) — log + uniform.
        var splitsPos: [Float] = [near]
        for i in 1...count {
            let f = Float(i) / Float(count)
            let logSplit = near * pow(far / near, f)
            let uniSplit = near + (far - near) * f
            splitsPos.append(lambda * logSplit + (1 - lambda) * uniSplit)
        }
        // Expose far-Z of each cascade to the lighting kernel.
        cascadeSplitsView = SIMD4(splitsPos[1], splitsPos[2], splitsPos[3], 0)

        let invView = camera.viewMatrix.inverse
        let halfFovY = camera.fovYRadians * 0.5
        let tanHalfFovY = tan(halfFovY)
        let aspect = camera.aspect

        let toLight = simd_normalize(directionalLightDirection)

        for c in 0..<count {
            let n = splitsPos[c]
            let f = splitsPos[c + 1]
            let hHN = n * tanHalfFovY, hWN = hHN * aspect
            let hHF = f * tanHalfFovY, hWF = hHF * aspect
            // View-space corners (Metal/right-handed: looking down -Z).
            let viewCorners: [SIMD4<Float>] = [
                SIMD4(-hWN, -hHN, -n, 1), SIMD4( hWN, -hHN, -n, 1),
                SIMD4(-hWN,  hHN, -n, 1), SIMD4( hWN,  hHN, -n, 1),
                SIMD4(-hWF, -hHF, -f, 1), SIMD4( hWF, -hHF, -f, 1),
                SIMD4(-hWF,  hHF, -f, 1), SIMD4( hWF,  hHF, -f, 1),
            ]
            var worldCorners: [SIMD3<Float>] = []
            worldCorners.reserveCapacity(8)
            for v in viewCorners {
                let w = invView * v
                worldCorners.append(SIMD3(w.x, w.y, w.z))
            }

            // Bounding sphere of the sub-frustum — rotation-invariant, so
            // the cascade extents don't pulse as the camera spins.
            var center = SIMD3<Float>.zero
            for p in worldCorners { center += p }
            center /= 8
            var radius: Float = 0
            for p in worldCorners {
                radius = max(radius, simd_length(p - center))
            }
            // Round the radius up slightly so PCF taps near the sphere edge
            // still find geometry.
            radius = ceil(radius * 16) / 16

            // Build a stable light-space frame. The "up" pick avoids the
            // Y-aligned degeneracy where cross(up, forward) would collapse.
            let lightUpHint: SIMD3<Float> = abs(toLight.y) < 0.95 ? SIMD3(0, 1, 0)
                                                                  : SIMD3(0, 0, 1)
            // Push the eye back beyond the sphere's near side and stretch the
            // far plane to match, so occluders sitting between the light and
            // the visible sub-frustum are still included in the shadow map.
            // A `2 * radius` slack catches buildings / trees / etc. directly
            // above the camera frustum without exploding depth precision.
            let casterSlack: Float = radius * 2
            let lightEye = center + toLight * (radius + casterSlack)
            let lightView = Self.lookAtRH(eye: lightEye, target: center, up: lightUpHint)
            let lightProj = Self.orthoRH(
                left: -radius, right: radius,
                bottom: -radius, top: radius,
                near: 0, far: 2 * radius + casterSlack
            )
            cascadeVPs[c] = lightProj * lightView
        }
    }

    /// Phase 4.10 — per-spot shadow setup.
    ///
    /// Two things happen here, both on the CPU:
    ///   1. Assign each spot a slice index into `spotShadowAtlas` (first
    ///      `spotShadowAtlasCapacity` spots get slots 0..N-1; the rest
    ///      get -1 = no shadow data).
    ///   2. Compute each shadowed spot's light-space view-projection
    ///      matrix from `(position, direction, outerCone, radius)` and
    ///      stuff it into the spot struct so the uploaded GPU buffer
    ///      carries it through to the lighting kernel.
    ///
    /// The matrix is a standard perspective with FOV = full outer cone
    /// (so the cone fully fits into NDC), near = small fixed bias, far =
    /// the spot's `radius`. Aspect 1.0 (square shadow map).
    private func updateSpotShadows() {
        let capacity = spotShadowAtlasCapacity
        let enabled = spotShadowsEnabled
        for i in 0..<spotLights.count {
            if !enabled || i >= capacity {
                // Force the lighting kernel into the no-shadow code path
                // so it doesn't sample stale slices when the toggle is off.
                spotLights[i].shadowSliceIndex = -1
                continue
            }
            spotLights[i].shadowSliceIndex = Int32(i)
            // Light view: apex at spot.position, looking down the spot's
            // direction (which already points the way light travels).
            let position = spotLights[i].position
            let dir = simd_normalize(spotLights[i].direction)
            // Pick a stable up that isn't degenerate with the cone axis.
            let upHint: SIMD3<Float> = abs(dir.y) < 0.95 ? SIMD3(0, 1, 0)
                                                          : SIMD3(0, 0, 1)
            let lightView = Self.lookAtRH(eye: position,
                                           target: position + dir,
                                           up: upHint)
            // Cone half-angle from cosine. `outerCone` is `cos(half-angle)`
            // and the perspective takes the FULL field-of-view (= 2 ×
            // half-angle). Add a small margin (10%) so the cone edge
            // doesn't graze the edge of the shadow map.
            let halfAngle = acos(min(0.9999, max(0.01, spotLights[i].outerCone)))
            let fovY = min(2.95, halfAngle * 2 * 1.1) // clamp <170° to avoid divergence
            let near: Float = 0.1
            let far  = max(0.5, spotLights[i].radius)
            let proj = Self.perspectiveRH(fovY: fovY, aspect: 1.0, near: near, far: far)
            spotLights[i].shadowMatrix = proj * lightView
        }
    }

    /// Render the depth-only pass for each shadowed spot into its slice
    /// of the spot shadow atlas. Skips when the toggle is off, when no
    /// spots are present, or when the host hasn't pushed any instances
    /// (no occluders → no shadow data needed).
    private func encodeSpotShadowPasses(_ cb: MTLCommandBuffer) {
        guard spotShadowsEnabled, !spotLights.isEmpty, !instances.isEmpty else { return }
        let count = min(spotLights.count, spotShadowAtlasCapacity)
        for slice in 0..<count {
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = spotShadowAtlas
            pass.depthAttachment.slice = slice
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1.0

            guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { continue }
            enc.label = "Illuminatorama.spotShadow.s\(slice)"
            enc.setRenderPipelineState(shadowPipeline)
            enc.setDepthStencilState(depthState)
            // Same back-face-cast trick as the cascaded path.
            enc.setCullMode(.front)
            enc.setFrontFacing(.counterClockwise)

            var lightVP = spotLights[slice].shadowMatrix
            enc.setVertexBytes(&lightVP,
                               length: MemoryLayout<simd_float4x4>.stride,
                               index: 3)

            // Phase 4.12 — instanced draw per mesh kind via the recipe
            // built in `uploadInstances`. Same win as the G-buffer pass:
            // one draw call replaces N per-instance draws per cascade /
            // per spot.
            let instStride = MemoryLayout<IlluminatoramaInstance>.stride
            for group in meshGroups {
                guard let mesh = meshes[group.kind] else { continue }
                let off = instStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: mesh.indexCount,
                    indexType: mesh.indexType,
                    indexBuffer: mesh.indexBuffer,
                    indexBufferOffset: 0,
                    instanceCount: group.count,
                    baseVertex: 0,
                    baseInstance: 0
                )
            }
            enc.endEncoding()
        }
    }

    /// Phase 4.11 — per-emitter integration step. One compute dispatch per
    /// emitter, covering its full capacity (dead particles early-return
    /// inside the kernel so the cost is bounded by capacity, not by
    /// alive count).
    private func encodeParticleStep(_ cb: MTLCommandBuffer) {
        guard !particleEmitters.isEmpty else { return }
        let now = CACurrentMediaTime()
        let dt = Float(max(0, min(0.05, now - lastParticleTickTime)))
        lastParticleTickTime = now
        guard dt > 0 else { return }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.particles.step"
        enc.setComputePipelineState(particleStepPipeline)
        for emitter in particleEmitters where emitter.enabled {
            var pf = IlluminatoramaParticleFrameUniforms(
                dt: dt,
                gravity: emitter.gravity,
                drag: emitter.drag,
                capacity: UInt32(emitter.capacity)
            )
            enc.setBuffer(emitter.particleBuffer, offset: 0, index: 0)
            enc.setBytes(&pf, length: MemoryLayout<IlluminatoramaParticleFrameUniforms>.stride,
                         index: 1)
            // 1D dispatch — the kernel declares `uint gid
            // [[thread_position_in_grid]]` (scalar), so threadgroup MUST
            // also be 1D (height & depth = 1). The shared `dispatch()`
            // helper builds a 2D threadgroup matching the lighting
            // kernel's signature, which crashes Metal validation here
            // with `validateBuiltinArguments` (the scalar gid can't
            // accept a 2D dispatch). Build a dedicated 1D dispatch.
            let tg = min(particleStepPipeline.maxTotalThreadsPerThreadgroup,
                         particleStepPipeline.threadExecutionWidth * 8)
            let groups = MTLSize(width: (emitter.capacity + tg - 1) / tg,
                                  height: 1, depth: 1)
            let threadgroup = MTLSize(width: tg, height: 1, depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
        }
        enc.endEncoding()
    }

    /// Phase 4.11 — additive HDR forward draw of surviving emitter sprites
    /// into `hdrTexture`; the texture carries `.renderTarget` usage (see
    /// `makeTargets`) so this pass can target it directly. Called only from
    /// `encodePostResolveFX` (Phase 4.43), AFTER the TAA resolve, so these
    /// additive sparks don't enter the temporal history. The velocity-aligned
    /// billboard shape gives them their own motion blur, so unlike the dust
    /// they self-masked most history smear — but they belong in the same
    /// post-resolve overlay for consistency and to keep history strictly
    /// opaque-only. (Pre-4.43 this ran before the SSR gather so sparks could
    /// reflect in nearby metals via SSR.)
    private func encodeParticleDraw(_ cb: MTLCommandBuffer) {
        guard !particleEmitters.isEmpty else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = hdrTexture
        pass.colorAttachments[0].loadAction = .load    // additive: keep prior lighting
        pass.colorAttachments[0].storeAction = .store
        // Bind the G-buffer depth read-only so opaque geometry occludes
        // particles behind it. `.load + .store` keeps the existing
        // depth values intact for any later passes (TAA, etc.).
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.particles.draw"
        enc.setRenderPipelineState(particleDrawPipeline)
        enc.setDepthStencilState(particleDepthState)
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        for emitter in particleEmitters where emitter.enabled {
            enc.setVertexBuffer(emitter.particleBuffer, offset: 0, index: 0)
            // Phase 4.11 round 4 — velocity-aligned billboard quads. Six
            // vertices per particle = two triangles sharing the
            // diagonal; the vertex shader picks the corner via
            // `vid % 6` and the particle via `vid / 6`.
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: emitter.capacity * 6)
        }
        enc.endEncoding()
    }

    /// Forward reflective glass pass — one sheet, src-alpha blended over the lit
    /// HDR composite (so its reflections bloom/tonemap), before TAA. No-op unless
    /// the host set a glass pane. Depth-tested against the scene depth (coins in
    /// front occlude) with no depth write (the pile behind shows through).
    /// World-space AABB enclosing all glass instances (unit ±1 object box per
    /// instance transformed to world). Used to aim the caustic photon emitter.
    private func glassWorldAABB() -> (SIMD3<Float>, SIMD3<Float>)? {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var any = false
        for (_, insts) in flattenGlass() {
            for inst in insts {
                let m = inst.modelMatrix
                for cx in [-1.0 as Float, 1] { for cy in [-1.0 as Float, 1] { for cz in [-1.0 as Float, 1] {
                    let p = m * SIMD4<Float>(cx, cy, cz, 1)
                    lo = simd_min(lo, SIMD3(p.x, p.y, p.z))
                    hi = simd_max(hi, SIMD3(p.x, p.y, p.z))
                    any = true
                } } }
            }
        }
        return any ? (lo, hi) : nil
    }

    /// Per-glass-instance sun-facing emitter discs (xyz centre, w bounding radius)
    /// — the caustic photon kernel aims a photon at each so ~every photon hits
    /// glass instead of sampling the whole AABB volume (where most miss). Bounding
    /// sphere of the unit ±1 object box under the instance transform.
    private func glassEmitterDiscs() -> [SIMD4<Float>] {
        var discs: [SIMD4<Float>] = []
        for (_, insts) in flattenGlass() {
            for inst in insts {
                let m = inst.modelMatrix
                let c = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
                // Half-diagonal of the transformed ±1 box ≈ the bounding radius.
                let ex = simd_length(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z))
                let ey = simd_length(SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z))
                let ez = simd_length(SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
                let radius = simd_length(SIMD3(ex, ey, ez))   // ‖(|col0|,|col1|,|col2|)‖
                discs.append(SIMD4(c, radius))
            }
        }
        return discs
    }

    /// Glass caustics (#60): photon-trace sun → refract through the glass → splat
    /// focused energy into the surface-cache card atlas (any receiver), then read
    /// it back per primary opaque pixel and add to the lit composite. No-op unless
    /// `causticsEnabled` AND the surface cache + glass TLAS are live.
    private func encodeCausticPass(_ cb: MTLCommandBuffer) {
        guard causticsEnabled, rtTLASActive, rtGlassInstanceCount > 0,
              let tlas = rtTLAS, let instData = rtInstanceDataBuffer,
              let objN = rtObjNormalBuffer, let gData = rtGlassDataBuffer else { return }
        // Needs the surface-cache card store (this topology) to address receivers.
        let cacheOn = surfCacheActive && surfCardCount > 0
            && rtSoupTriBaseBuffer != nil && surfTriCardBuffer != nil
            && surfTriUVaBuffer != nil && surfTriUVcBuffer != nil && surfCardRectBuffer != nil
            && rtSoupTriBaseCount == instances.count
        guard cacheOn, let aabb = glassWorldAABB() else { return }
        let cache = engine.pipelineCache
        if causticDecayPipeline == nil { causticDecayPipeline = cache.pipelineState(name: "illumi_caustic_decay", device: device) }
        if causticPhotonsPipeline == nil { causticPhotonsPipeline = cache.pipelineState(name: "illumi_caustic_photons", device: device) }
        if causticPrimaryPipeline == nil { causticPrimaryPipeline = cache.pipelineState(name: "illumi_caustic_primary", device: device) }
        guard let decayP = causticDecayPipeline, let photP = causticPhotonsPipeline,
              let primP = causticPrimaryPipeline else { return }
        // Atlas sized to the surface-cache atlas; (re)allocate + clear on growth.
        let texels = surfAtlasW * surfAtlasH
        let needLen = max(1, texels) * 3 * MemoryLayout<Float>.stride
        var freshAtlas = false
        if causticAtlasBuffer == nil || causticAtlasTexels < texels {
            causticAtlasBuffer = device.makeBuffer(length: needLen, options: .storageModePrivate)
            causticAtlasBuffer?.label = "Illuminatorama.caustic.atlas"
            causticAtlasTexels = texels
            freshAtlas = true
        }
        guard let atlas = causticAtlasBuffer else { return }
        if freshAtlas, let blit = cb.makeBlitCommandEncoder() {
            blit.fill(buffer: atlas, range: 0..<needLen, value: 0)   // private buffer can't be memset from CPU
            blit.endEncoding()
        }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        causticFrameSeed &+= 1
        var u = IlluminatoramaCausticUniforms()
        u.invViewProjection = fu.invViewProjection
        u.cameraWorldPos = fu.cameraWorldPos
        u.rayTMin = 0.004
        u.sunDir = simd_normalize(rtSunDirection)
        u.photonEnergy = max(0, causticPhotonEnergy)
        u.sunColor = rtSunColor
        u.decay = max(0, min(1, causticDecay))
        u.aabbMin = aabb.0; u.aabbMax = aabb.1
        u.strength = max(0, causticStrength)
        u.width = UInt32(width); u.height = UInt32(height)
        u.glassInstanceBase = UInt32(rtGlassInstanceBase)
        u.photonCount = UInt32(max(1, causticPhotonCount))
        u.surfTriCount = UInt32(rtTriangleCount)
        u.surfAtlasW = UInt32(surfAtlasW); u.surfAtlasH = UInt32(surfAtlasH)
        u.frameSeed = causticFrameSeed
        u.maxGlassBounces = UInt32(max(2, min(8, rtGlassMaxBounces)))
        // Per-instance emitter discs — aim photons at glass directly (near-100% hit).
        let discs = glassEmitterDiscs()
        let discLen = max(1, discs.count) * MemoryLayout<SIMD4<Float>>.stride
        if (causticDiscBuffer?.length ?? 0) < discLen {
            causticDiscBuffer = device.makeBuffer(length: discLen, options: .storageModeShared)
            causticDiscBuffer?.label = "Illuminatorama.caustic.discs"
        }
        if let db = causticDiscBuffer, !discs.isEmpty {
            discs.withUnsafeBytes { _ = memcpy(db.contents(), $0.baseAddress!, discs.count * MemoryLayout<SIMD4<Float>>.stride) }
        }
        u.glassDiscCount = UInt32(discs.count)
        memcpy(causticUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaCausticUniforms>.stride)

        // Pass 1 — temporal decay.
        if let enc = timedComputeEncoder(cb, "caustic.decay") {
            enc.label = "Illuminatorama.caustic.decay"
            enc.setComputePipelineState(decayP)
            enc.setBuffer(atlas, offset: 0, index: 0)
            enc.setBuffer(causticUniformBuffer, offset: 0, index: 1)
            dispatch1D(enc, pipeline: decayP, count: texels * 3)
            enc.endEncoding()
        }
        // Pass 2 — emit photons, refract through glass, splat into receiver cards.
        if let enc = timedComputeEncoder(cb, "caustic.photons") {
            enc.label = "Illuminatorama.caustic.photons"
            enc.setComputePipelineState(photP)
            enc.setBuffer(atlas, offset: 0, index: 0)
            enc.setBuffer(causticUniformBuffer, offset: 0, index: 1)
            enc.setAccelerationStructure(tlas, bufferIndex: 2)
            enc.setBuffer(instData, offset: 0, index: 3)
            enc.setBuffer(objN, offset: 0, index: 4)
            enc.setBuffer(gData, offset: 0, index: 5)
            enc.setBuffer(rtSoupTriBaseBuffer, offset: 0, index: 6)
            enc.setBuffer(surfTriCardBuffer, offset: 0, index: 7)
            enc.setBuffer(surfTriUVaBuffer, offset: 0, index: 8)
            enc.setBuffer(surfTriUVcBuffer, offset: 0, index: 9)
            enc.setBuffer(surfCardRectBuffer, offset: 0, index: 10)
            enc.setBuffer(causticDiscBuffer ?? causticUniformBuffer, offset: 0, index: 11)
            for blas in rtBLASList { enc.useResource(blas, usage: .read) }
            for buf in rtResidentBuffers { enc.useResource(buf, usage: .read) }
            dispatch1D(enc, pipeline: photP, count: max(1, causticPhotonCount))
            enc.endEncoding()
        }
        // Pass 3 — per primary opaque pixel, read its own card's caustic + add.
        if let enc = timedComputeEncoder(cb, "caustic.primary") {
            enc.label = "Illuminatorama.caustic.primary"
            enc.setComputePipelineState(primP)
            enc.setTexture(depthTexture, index: 0)
            enc.setTexture(gbufferAlbedoMet, index: 1)
            enc.setTexture(hdrCompositeTexture, index: 2)
            enc.setBuffer(causticUniformBuffer, offset: 0, index: 0)
            enc.setBuffer(atlas, offset: 0, index: 1)
            enc.setAccelerationStructure(tlas, bufferIndex: 2)
            enc.setBuffer(rtSoupTriBaseBuffer, offset: 0, index: 3)
            enc.setBuffer(surfTriCardBuffer, offset: 0, index: 4)
            enc.setBuffer(surfTriUVaBuffer, offset: 0, index: 5)
            enc.setBuffer(surfTriUVcBuffer, offset: 0, index: 6)
            enc.setBuffer(surfCardRectBuffer, offset: 0, index: 7)
            for blas in rtBLASList { enc.useResource(blas, usage: .read) }
            for buf in rtResidentBuffers { enc.useResource(buf, usage: .read) }
            dispatch(enc, pipeline: primP, width: width, height: height,
                     maxThreadsOverride: rtThreadgroupMax)
            enc.endEncoding()
        }
    }

    /// Canonical glass draw list — the SINGLE flattening both the raster pass and
    /// the TLAS build use, so a glass hit's material (looked up by TLAS instance
    /// order) matches what the raster fragment drew. Precedence: multi-mesh
    /// groups → single-mesh array → single pane.
    func flattenGlass() -> [(kind: MeshKind, insts: [IlluminatoramaGlassInstance])] {
        if !glassMeshGroups.isEmpty {
            return glassMeshGroups.compactMap { $0.instances.isEmpty ? nil : ($0.kind, $0.instances) }
        }
        if let kind = glassPaneKind, !glassInstances.isEmpty { return [(kind, glassInstances)] }
        if let kind = glassPaneKind, let single = glassPaneInstance { return [(kind, [single])] }
        return []
    }

    private func encodeGlassPass(_ cb: MTLCommandBuffer) {
        guard let glassDepth = glassDepthState else { return }
        let groups = flattenGlass()
        guard !groups.isEmpty else { return }

        // RT path when the device + scene support a live TLAS that holds this
        // frame's glass; otherwise the Fresnel-only fallback. `rtGlassInstanceCount`
        // is set by `rebuildRTAccel` when glass was appended to the TLAS.
        let useRT = rtGlassEnabled && rtTLASActive && rtGlassInstanceCount > 0
            && glassRTPipeline != nil && rtTLAS != nil
            && rtInstanceDataBuffer != nil && rtObjNormalBuffer != nil
            && rtGlassDataBuffer != nil
        guard let pipeline = useRT ? glassRTPipeline : glassFallbackPipeline else { return }

        // Pack every group's instances contiguously; each group draws from its own
        // byte offset (one CPU upload feeds all draws).
        let stride = MemoryLayout<IlluminatoramaGlassInstance>.stride
        var flat: [IlluminatoramaGlassInstance] = []
        var offsets: [Int] = []
        for g in groups { offsets.append(flat.count); flat.append(contentsOf: g.insts) }
        let needed = stride * flat.count
        if glassInstanceBuffer.length < needed {
            guard let grown = device.makeBuffer(length: needed,
                                                options: .storageModeShared) else { return }
            glassInstanceBuffer = grown
        }
        flat.withUnsafeBytes { src in
            _ = memcpy(glassInstanceBuffer.contents(), src.baseAddress!, needed)
        }

        // Per-frame RT-glass uniforms (camera + RT sun/ambient + bases + flags).
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        rtInstFrameSeed &+= 1
        // Surface-cache availability (mirror of the TLAS lighting pass): a glass
        // refraction/reflection hit reads the cached multi-bounce radiance when on.
        let cacheOn = useRT && surfCacheActive && surfCardCount > 0
            && rtSoupTriBaseBuffer != nil && surfTriCardBuffer != nil
            && surfCacheCurrentAtlas != nil
            && rtSoupTriBaseCount == instances.count
        var u = IlluminatoramaGlassRTUniforms()
        u.cameraWorldPos = fu.cameraWorldPos
        u.rayTMin = 0.004
        u.sunDir = simd_normalize(rtSunDirection)
        u.sunSoftnessRad = max(0.0005, rtSunSoftnessRad)
        u.sunColor = rtSunColor
        u.reflStrength = max(0.0001, rtReflStrength)
        u.skyAmbient = ambientColor
        u.skyIntensity = max(0, iblIntensity)
        u.glassInstanceBase = UInt32(rtGlassInstanceBase)
        u.maxBounces = max(2, min(10, rtGlassMaxBounces))
        u.shadowRays = max(0, min(8, rtGlassShadowRays))
        u.frameSeed = rtInstFrameSeed
        u.surfCacheEnabled = cacheOn ? 1 : 0
        u.surfTriCount = cacheOn ? UInt32(rtTriangleCount) : 0
        u.surfAtlasW = cacheOn ? UInt32(surfAtlasW) : 0
        u.surfAtlasH = cacheOn ? UInt32(surfAtlasH) : 0
        u.dispersionEnabled = rtGlassDispersionEnabled ? 1 : 0
        // Cheap glass: only on the non-RT path, and only if a host opted in.
        let cheap = !useRT && glassCheapMode != 0
        u.cheapGlassMode = cheap ? UInt32(glassCheapMode) : 0
        u.viewW = Float(hdrCompositeTexture.width)
        u.viewH = Float(hdrCompositeTexture.height)
        // Thin-film iridescence (cheap mode 2 only; default strength 0 = no-op).
        u.time = time
        u.thinFilmStrength = (cheap && glassCheapMode == 2) ? max(0, thinFilmStrength) : 0
        u.filmThicknessNm = max(1, thinFilmThicknessNm)
        u.filmIOR = max(1.01, thinFilmIOR)
        // Oscillation-mode undulation — only on the cheap path (bubble shells).
        u.wobbleAmp = cheap ? max(0, bubbleWobbleAmp) : 0
        u.wobbleFreq = max(0.01, bubbleWobbleFreq)
        memcpy(glassRTUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaGlassRTUniforms>.stride)

        // Screen-space cheap glass (mode 2) samples the scene BEHIND the pane: copy
        // the pre-glass composite into a backdrop texture (can't sample the render
        // target we're about to write). One full-frame GPU blit; no ray tracing.
        let useBackdrop = cheap && glassCheapMode == 2
        if useBackdrop {
            if glassBackdropTexture?.width != hdrCompositeTexture.width
                || glassBackdropTexture?.height != hdrCompositeTexture.height {
                let d = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: hdrCompositeTexture.pixelFormat,
                    width: hdrCompositeTexture.width, height: hdrCompositeTexture.height,
                    mipmapped: false)
                d.usage = [.shaderRead]; d.storageMode = .private
                glassBackdropTexture = device.makeTexture(descriptor: d)
                glassBackdropTexture?.label = "Illuminatorama.glassBackdrop"
            }
            if let bd = glassBackdropTexture, let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: hdrCompositeTexture, to: bd)
                blit.endEncoding()
            }
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = hdrCompositeTexture
        pass.colorAttachments[0].loadAction = .load     // keep the lit scene
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .load          // test vs scene depth
        pass.depthAttachment.storeAction = .dontCare     // no depth write
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = useRT ? "Illuminatorama.glass.rt"
            : (cheap ? "Illuminatorama.glass.cheap\(glassCheapMode)" : "Illuminatorama.glass.fallback")
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(glassDepth)
        // Perf: the RT path's bounce loop traces the WHOLE glass volume from the
        // front (entry) surface, so back faces are wasted invocations (overwritten
        // by the nearer front face under depth-LE + no-write). Cull them — ~half
        // the glass fragments on convex glass, identical output. (CCW-front matches
        // the G-buffer winding.) The Fresnel+sky FALLBACK still wants both surfaces,
        // so it stays two-sided.
        if useRT || (cheap && glassCheapMode == 2) {
            // Screen-space cheap glass traces from the FRONT pane (samples the scene
            // behind it), so back faces are wasted/double-darkening — cull them.
            enc.setCullMode(.back)
            enc.setFrontFacing(.counterClockwise)
        } else {
            enc.setCullMode(.none)                        // two-sided glass (fallback / synthetic)
        }
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)   // viewProjection is first
        enc.setVertexBuffer(glassRTUniformBuffer, offset: 0, index: 3) // VS reads time/wobble for oscillation-mode undulation
        enc.setFragmentBuffer(glassRTUniformBuffer, offset: 0, index: 1)
        enc.setFragmentTexture(equirectSky ?? dummySkyTexture, index: 0)
        // Backdrop (cheap mode 2) — bind a dummy otherwise so the slot is valid.
        enc.setFragmentTexture((useBackdrop ? glassBackdropTexture : nil) ?? dummySkyTexture, index: 2)
        if useRT, let tlas = rtTLAS, let instData = rtInstanceDataBuffer,
           let objN = rtObjNormalBuffer, let gData = rtGlassDataBuffer {
            enc.setFragmentAccelerationStructure(tlas, bufferIndex: 0)
            enc.setFragmentBuffer(instData, offset: 0, index: 3)
            enc.setFragmentBuffer(objN, offset: 0, index: 4)
            enc.setFragmentBuffer(gData, offset: 0, index: 5)
            // Surface-cache buffers (dummies keep the bindings valid when off);
            // optional-ternary mirrors the TLAS lighting pass — no force-unwrap.
            let dummy: MTLBuffer = instData
            enc.setFragmentBuffer(cacheOn ? surfTriCardBuffer : dummy, offset: 0, index: 6)
            enc.setFragmentBuffer(cacheOn ? surfTriUVaBuffer : dummy, offset: 0, index: 7)
            enc.setFragmentBuffer(cacheOn ? surfTriUVcBuffer : dummy, offset: 0, index: 8)
            enc.setFragmentBuffer(cacheOn ? rtSoupTriBaseBuffer : dummy, offset: 0, index: 9)
            enc.setFragmentBuffer(cacheOn ? surfCardRectBuffer : dummy, offset: 0, index: 10)
            enc.setFragmentBuffer(cacheOn ? surfCardBuffer : (surfCardDummyBuffer ?? dummy), offset: 0, index: 11)
            enc.setFragmentTexture(cacheOn ? surfConsumerAtlas : (equirectSky ?? dummySkyTexture), index: 1)
            // The TLAS references the BLASes which reference mesh buffers — all
            // must be resident for the fragment-stage intersector.
            for blas in rtBLASList { enc.useResource(blas, usage: .read) }
            for buf in rtResidentBuffers { enc.useResource(buf, usage: .read) }
        }
        for (i, g) in groups.enumerated() {
            guard let mesh = meshes[g.kind] else { continue }
            let off = stride * offsets[i]
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(glassInstanceBuffer, offset: off, index: 2)
            enc.setFragmentBuffer(glassInstanceBuffer, offset: off, index: 2)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: mesh.indexType, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0, instanceCount: g.insts.count)
        }
        enc.endEncoding()
    }

    private func encodeGBufferPass(_ cb: MTLCommandBuffer) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = gbufferAlbedoMet
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[1].texture = gbufferNormalRgh
        pass.colorAttachments[1].loadAction = .clear
        pass.colorAttachments[1].storeAction = .store
        pass.colorAttachments[1].clearColor = MTLClearColor(red: 0.5, green: 0.5, blue: 0, alpha: 0)
        pass.colorAttachments[2].texture = gbufferEmission
        pass.colorAttachments[2].loadAction = .clear
        pass.colorAttachments[2].storeAction = .store
        pass.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        // Phase 2.7 — velocity. Sky pixels (no geometry) stay at zero, which
        // means the TAA resolve treats them as stationary — fine for static
        // skies and visibly slightly smeary for moving clouds.
        pass.colorAttachments[3].texture = velocityTexture
        pass.colorAttachments[3].loadAction = .clear
        pass.colorAttachments[3].storeAction = .store
        pass.colorAttachments[3].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1.0

        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.gbuffer"
        enc.setRenderPipelineState(gbufferPipeline)
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.back)
        enc.setFrontFacing(.counterClockwise)

        // Phase 4.0/4.1 — bind the per-material texture atlases once for the
        // whole G-buffer pass. Per-instance slice indices select the slice
        // within each atlas (`< 0` → no sampling, fall back to the per-
        // instance scalar/colour). albedoAtlas is sRGB; nonColorAtlas is
        // linear so metallic/roughness round-trip exactly.
        enc.setFragmentTexture(albedoAtlas.texture, index: 0)
        enc.setFragmentTexture(nonColorAtlas.texture, index: 1)
        // Phase 4.25 (issue #60 item 5) — per-slice UV-scale tables that let the
        // shader tile letterboxed (aspect-preserved) non-square textures at
        // their true aspect. `float2` per slice; indexed by the instance's slice
        // id. Fragment buffer(2) is the instance buffer (set per group below),
        // (4) is unused by the fragment stage — these live at (3)/(5) to match
        // `illumi_fs`'s signature.
        enc.setFragmentBuffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 3)
        enc.setFragmentBuffer(nonColorAtlas.uvScaleBuffer, offset: 0, index: 5)
        // Phase 4.12 — instanced draws via the recipe built in
        // `uploadInstances`. One draw call per mesh kind, with the
        // current/previous instance buffer offsets pointed at the start
        // of that group's contiguous slice. The vertex shader's
        // `instances[iid]` indexing then walks the slice naturally.
        // Killed the prior per-instance inner loop — FloatingFlowers+
        // drops from ~1500 draw calls per frame to ~10.
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride
        for group in meshGroups {
            // Superquadric impostor box kinds are drawn by the impostor pipeline
            // below; RT-proxy kinds exist only in the TLAS. Both are skipped here.
            if impostorMeshKinds.contains(group.kind) ||
               rtProxyMeshKinds.contains(group.kind) { continue }
            guard let mesh = meshes[group.kind] else { continue }
            // Two-sided meshes (open / dynamic MC fluid surfaces) render cull
            // `.none` so they don't go hollow when their back side faces the
            // camera; the fragment shader flips the normal for back faces.
            enc.setCullMode(mesh.doubleSided ? .none : .back)
            let off = instStride * group.start
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
            enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
            enc.setFragmentBuffer(currentInstanceBuffer, offset: off, index: 2)
            // Phase 2.7 — previous-frame instances at buffer(4) for the
            // motion-vector reconstruction. Same per-group offset as
            // current — the ping-pong buffers share the grouped layout
            // because the host's `instances` order is stable across
            // frames for a static scene topology.
            enc.setVertexBuffer(previousInstanceBuffer, offset: off, index: 4)
            enc.drawIndexedPrimitives(
                type: .triangle,
                indexCount: mesh.indexCount,
                indexType: mesh.indexType,
                indexBuffer: mesh.indexBuffer,
                indexBufferOffset: 0,
                instanceCount: group.count,
                baseVertex: 0,
                baseInstance: 0
            )
        }

        // ── Perfect analytic superquadric impostors ───────────────────────────
        // Drawn into the SAME G-buffer pass with the impostor pipeline: each
        // instance's [-1,1] bounding-box mesh is rasterized (BACK faces — cull
        // `.front` — so the primitive survives the camera entering the box), and
        // the fragment ray-traces the analytic surface, writing the G-buffer +
        // analytic depth + analytic motion vectors. The param buffer is bound at
        // the same per-group offset as the instance buffer (grouped-order aligned).
        if let sqPipe = superquadricImpostorPipeline, !impostorMeshKinds.isEmpty {
            enc.setRenderPipelineState(sqPipe)
            enc.setCullMode(.front)
            let pStride = MemoryLayout<IlluminatoramaInstance.SuperquadricParam>.stride
            for group in meshGroups where impostorMeshKinds.contains(group.kind) {
                guard let mesh = meshes[group.kind] else { continue }
                let off  = instStride * group.start
                let poff = pStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 1)
                enc.setFragmentBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(previousInstanceBuffer, offset: off, index: 4)
                enc.setFragmentBuffer(superquadricParamBuffer, offset: poff, index: 6)
                enc.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: mesh.indexCount,
                    indexType: mesh.indexType,
                    indexBuffer: mesh.indexBuffer,
                    indexBufferOffset: 0,
                    instanceCount: group.count,
                    baseVertex: 0,
                    baseInstance: 0
                )
            }
        }
        enc.endEncoding()
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
    private func encodeIBLBakeIfNeeded(_ cb: MTLCommandBuffer) {
        guard iblEnabled else { return }

        // Skip if nothing has changed and the interval fallback hasn't fired.
        let timeSinceLastBake = time - iblLastBakeTime
        guard iblNeedsRebake || timeSinceLastBake >= iblRebakeInterval else { return }
        iblNeedsRebake = false
        iblLastBakeTime = time

        let sky = equirectSky ?? dummySkyTexture
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.iblBake"

        // ── Diffuse irradiance ─────────────────────────────────────
        enc.setComputePipelineState(irradianceBakePipeline)
        enc.setTexture(sky, index: 0)
        enc.setTexture(irradianceCube, index: 1)
        var bakeDesat = max(0.0, min(1.0, iblBakeDesaturation))
        enc.setBytes(&bakeDesat, length: MemoryLayout<Float>.stride, index: 0)
        dispatchCubeFaces(
            enc, pipeline: irradianceBakePipeline,
            faceSize: Self.irradianceFaceSize
        )

        // ── GGX-prefiltered specular, one mip at a time ────────────
        // 12 samples: TAA temporal accumulation makes this indistinguishable
        // from 24 at 60fps. Keep 1 sample for the perfect-mirror mip 0.
        enc.setComputePipelineState(prefilterBakePipeline)
        enc.setTexture(sky, index: 0)
        let mipCount = prefilteredMipViews.count
        for (mip, view) in prefilteredMipViews.enumerated() {
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
                enc, pipeline: prefilterBakePipeline, faceSize: mipSize
            )
        }
        enc.endEncoding()
    }

    // ── Phase 3.1: DDGI ──────────────────────────────────────────────────────

    // Swift mirrors of DDGIUniforms and DDGIInstanceData in Illuminatorama.metal.
    // ALIGNMENT: SIMD3<Float>(size=12, align=16) + UInt32(size=4) = 16 bytes
    // per group, matching Metal's float3+uint layout. See IlluminatoramaTypes.swift
    // for the same pattern applied to FrameUniforms.
    private struct IlluminatoramaDDGIUniforms {
        var gridOrigin: SIMD3<Float>;          var gridDimsX: UInt32        // 0
        var directionalLightDir: SIMD3<Float>; var gridDimsY: UInt32        // 16
        var directionalLightColor: SIMD3<Float>; var gridDimsZ: UInt32      // 32
        var probeSpacing: Float;               var raysPerProbe: UInt32     // 48
        var hysteresis: Float;                 var depthHysteresis: Float   // 56
        var irradianceScale: Float;            var enabled: UInt32          // 64
        var irrTileSize: UInt32;               var depthTileSize: UInt32    // 72
        var instanceCount: UInt32;             var twoBounceEnabled: UInt32 // 80
        var emitterCount: UInt32 = 0; var _pad2: UInt32 = 0                 // 88..92
        // stride: 96 bytes
    }

    private struct DDGIGPUInstanceData {
        var invModelMatrix: simd_float4x4  // 0
        var normalMatrix:   simd_float4x4  // 64
        var albedo:   SIMD3<Float>; var metallic:   Float  // 128
        var emission: SIMD3<Float>; var roughness:  Float  // 144
        var meshKind: UInt32                               // 160
        var _pad0: UInt32 = 0; var _pad1: UInt32 = 0; var _pad2: UInt32 = 0 // 164..172
        // stride: 176 bytes
    }

    // Mirror of DDGIRayRecord — only used to compute buffer size in Swift.
    private struct DDGIRayRecord {
        var dirAndDist: SIMD4<Float>  // 16
        var radiance:   SIMD4<Float>  // 16
        // stride: 32 bytes
    }

    // Swift mirror of DDGIPointEmitter in Illuminatorama.metal.
    private struct DDGIGPUEmitter {
        var position: SIMD3<Float>; var radius: Float   // 0..15
        var color: SIMD3<Float>;    var _pad: Float = 0 // 16..31
        // stride: 32 bytes
    }

    /// Lazily allocate (or reallocate on grid-dim change) the irradiance atlas,
    /// depth atlas, and ray buffer. No-op when dims are already current.
    private func ensureDDGIResources() {
        let dims = ddgiGridDims
        guard dims != ddgiCurrentGridDims else { return }

        let irrTile = Self.ddgiIrrTileSize
        let depTile = Self.ddgiDepthTileSize
        let irrW = (irrTile + 2) * dims.x * dims.z
        let irrH = (irrTile + 2) * dims.y
        let depW = (depTile + 2) * dims.x * dims.z
        let depH = (depTile + 2) * dims.y
        let probeCount  = dims.x * dims.y * dims.z
        let rayBufLen   = MemoryLayout<DDGIRayRecord>.stride * probeCount * max(1, ddgiRaysPerProbe)

        let irrDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: irrW, height: irrH, mipmapped: false)
        irrDesc.usage       = [.shaderRead, .shaderWrite]
        irrDesc.storageMode = .private

        let depDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float, width: depW, height: depH, mipmapped: false)
        depDesc.usage       = [.shaderRead, .shaderWrite]
        depDesc.storageMode = .private

        // Phase 3.3 — allocate ping-pong atlases (A + B for both irradiance
        // and depth). Both start zero-initialised by Metal; the first frame
        // after a (re)allocation will read zeros for the "previous" atlas,
        // which correctly degenerates to one-bounce GI until the hysteresis
        // EMA builds up the indirect term.
        guard let irrA = device.makeTexture(descriptor: irrDesc),
              let irrB = device.makeTexture(descriptor: irrDesc),
              let depA = device.makeTexture(descriptor: depDesc),
              let depB = device.makeTexture(descriptor: depDesc),
              let rays = device.makeBuffer(length: rayBufLen, options: .storageModePrivate)
        else {
            Self.log.error("DDGI: atlas allocation failed for grid \(dims.x)×\(dims.y)×\(dims.z)")
            return
        }
        irrA.label = "Illuminatorama.ddgi.irrAtlas.A"
        irrB.label = "Illuminatorama.ddgi.irrAtlas.B"
        depA.label = "Illuminatorama.ddgi.depthAtlas.A"
        depB.label = "Illuminatorama.ddgi.depthAtlas.B"
        rays.label = "Illuminatorama.ddgi.rays"

        ddgiIrradianceAtlasA = irrA
        ddgiIrradianceAtlasB = irrB
        ddgiDepthAtlasA      = depA
        ddgiDepthAtlasB      = depB
        ddgiRayBuffer        = rays
        ddgiCurrentGridDims  = dims
        // Reset the ping-pong so the first frame after (re)allocation reads
        // from a definitely-zeroed B atlas, not a partially-filled one.
        ddgiUseAtlasA        = true
    }

    /// Grow (or create) the per-frame DDGI instance-data buffer.
    private func ensureDDGIInstanceDataBuffer(count: Int) {
        guard count > ddgiInstanceDataCapacity else { return }
        let newCap = max(count, ddgiInstanceDataCapacity * 2, 8)
        let len = MemoryLayout<DDGIGPUInstanceData>.stride * newCap
        guard let buf = device.makeBuffer(length: len, options: .storageModeShared) else { return }
        buf.label = "Illuminatorama.ddgi.instanceData"
        ddgiInstanceDataBuffer = buf
        ddgiInstanceDataCapacity = newCap
    }

    private func encodeDDGIFrame(_ cb: MTLCommandBuffer) {
        // Phase 3.3 — toggle the ping-pong at the *start* of the DDGI frame
        // (not the end) so accessor semantics stay obvious through the
        // function: `ddgiIrradianceCurrent` is the atlas we write to in
        // this frame AND the atlas the lighting kernel reads after we
        // return. `ddgiIrradiancePrevious` is the atlas the trace kernel
        // reads for the second bounce.
        //
        // First frame after enable / resource (re)allocation:
        //   useAtlasA starts at true → toggles to false → Current = B,
        //   Previous = A (both still empty). Trace reads zero, so two-
        //   bounce contributes zero this frame; one-bounce direct works
        //   normally. Subsequent frames carry real data.
        ddgiUseAtlasA.toggle()

        // Write the DDGIUniforms buffer regardless of the enabled flag —
        // the lighting kernel always binds it (enabled=0 makes the lookup a no-op).
        let dimsNow = ddgiEnabled ? ddgiGridDims : SIMD3(1, 1, 1)
        var u = IlluminatoramaDDGIUniforms(
            gridOrigin:           ddgiGridOrigin,
            gridDimsX:            UInt32(dimsNow.x),
            directionalLightDir:  simd_normalize(directionalLightDirection),
            gridDimsY:            UInt32(dimsNow.y),
            directionalLightColor: directionalLightColor,
            gridDimsZ:            UInt32(dimsNow.z),
            probeSpacing:         ddgiProbeSpacing,
            raysPerProbe:         UInt32(max(1, ddgiRaysPerProbe)),
            hysteresis:           ddgiHysteresis,
            depthHysteresis:      min(ddgiHysteresis + 0.01, 0.99),
            irradianceScale:      ddgiIrradianceScale,
            enabled:              ddgiEnabled ? 1 : 0,
            irrTileSize:          UInt32(Self.ddgiIrrTileSize),
            depthTileSize:        UInt32(Self.ddgiDepthTileSize),
            instanceCount:        UInt32(instances.count),
            twoBounceEnabled:     (ddgiEnabled && ddgiTwoBounceEnabled) ? 1 : 0
        )

        // Collect analytic emitters from registered particle emitters + SimEngine fields.
        var emitterLights: [(position: SIMD3<Float>, color: SIMD3<Float>, radius: Float)] = []
        for e in particleEmitters where e.enabled {
            if let light = e.ddgiLight { emitterLights.append(light) }
        }
        for f in particleFields {
            if let light = f.ddgiLight { emitterLights.append(light) }
        }
        u.emitterCount = UInt32(emitterLights.count)

        memcpy(ddgiUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaDDGIUniforms>.stride)

        guard ddgiEnabled, !instances.isEmpty else { return }
        ensureDDGIResources()
        guard let irrAtlas    = ddgiIrradianceCurrent,
              let depAtlas    = ddgiDepthCurrent,
              let irrAtlasPrv = ddgiIrradiancePrevious,
              let depAtlasPrv = ddgiDepthPrevious,
              let rayBuf      = ddgiRayBuffer else { return }

        // Build the emitter buffer (may be empty; the trace kernel reads emitterCount).
        let emitCount = emitterLights.count
        if emitCount > 0 {
            if emitCount > ddgiEmitterCapacity {
                let newCap = max(emitCount, ddgiEmitterCapacity * 2, 4)
                let len = MemoryLayout<DDGIGPUEmitter>.stride * newCap
                ddgiEmitterBuffer = device.makeBuffer(length: len, options: .storageModeShared)
                ddgiEmitterBuffer?.label = "Illuminatorama.ddgi.emitters"
                ddgiEmitterCapacity = newCap
            }
            if let buf = ddgiEmitterBuffer {
                let ptr = buf.contents().bindMemory(to: DDGIGPUEmitter.self, capacity: emitCount)
                for (i, light) in emitterLights.enumerated() {
                    ptr[i] = DDGIGPUEmitter(position: light.position, radius: light.radius,
                                            color: light.color)
                }
            }
        }

        // Build per-instance DDGI data (invModelMatrix + material) for the trace kernel.
        ensureDDGIInstanceDataBuffer(count: instances.count)
        guard let instBuf = ddgiInstanceDataBuffer else { return }
        let instPtr = instBuf.contents().bindMemory(
            to: DDGIGPUInstanceData.self, capacity: instances.count)
        for (i, ref) in instances.enumerated() {
            // Phase 2.6 — meshKind=3 is "custom mesh, no analytic intersection";
            // the trace kernel skips intersection on anything not in {0,1,2}.
            let kind: UInt32 = ref.meshKind.gpuMeshKind
            instPtr[i] = DDGIGPUInstanceData(
                invModelMatrix: ref.data.modelMatrix.inverse,
                normalMatrix:   ref.data.normalMatrix,
                albedo:         ref.data.albedo,
                metallic:       ref.data.metallic,
                emission:       ref.data.emission,
                roughness:      ref.data.roughness,
                meshKind:       kind
            )
        }

        let probeCount   = ddgiGridDims.x * ddgiGridDims.y * ddgiGridDims.z
        let raysPerProbe = max(1, ddgiRaysPerProbe)
        let sky = equirectSky ?? dummySkyTexture

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.ddgi"

        // ── Trace: (raysPerProbe, probeCount) 2D dispatch ───────────────
        // Phase 3.3 — trace reads the PREVIOUS-frame atlases at texture(1)/(2)
        // for the second-bounce lookup; the update kernels below write to
        // the CURRENT-frame atlases. With ping-pong these are distinct
        // textures so the read-then-write within one encoder doesn't alias.
        enc.setComputePipelineState(ddgiTracePipeline)
        enc.setBuffer(rayBuf,              offset: 0, index: 0)
        enc.setBuffer(ddgiUniformBuffer,   offset: 0, index: 1)
        enc.setBuffer(instBuf,             offset: 0, index: 2)
        enc.setBuffer(ddgiEmitterBuffer ?? instBuf, offset: 0, index: 3)  // dummy when no emitters
        enc.setTexture(sky,         index: 0)
        enc.setTexture(irrAtlasPrv, index: 1)
        enc.setTexture(depAtlasPrv, index: 2)
        let tgwT = ddgiTracePipeline.threadExecutionWidth
        enc.dispatchThreadgroups(
            MTLSize(width: (raysPerProbe + tgwT - 1) / tgwT, height: probeCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgwT, height: 1, depth: 1))

        // ── Update irradiance atlas: (irrTileSize × irrTileSize × probeCount) ─
        enc.setComputePipelineState(ddgiUpdateIrrPipeline)
        enc.setBuffer(rayBuf,             offset: 0, index: 0)
        enc.setBuffer(ddgiUniformBuffer,  offset: 0, index: 1)
        enc.setTexture(irrAtlas, index: 0)
        let irrTile = Self.ddgiIrrTileSize
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: irrTile, depth: probeCount),
            threadsPerThreadgroup: MTLSize(width: irrTile, height: 1, depth: 1))

        // ── Update depth atlas: (depthTileSize × depthTileSize × probeCount) ─
        enc.setComputePipelineState(ddgiUpdateDepthPipeline)
        enc.setBuffer(rayBuf,             offset: 0, index: 0)
        enc.setBuffer(ddgiUniformBuffer,  offset: 0, index: 1)
        enc.setTexture(depAtlas, index: 0)
        let depTile = Self.ddgiDepthTileSize
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: depTile, depth: probeCount),
            threadsPerThreadgroup: MTLSize(width: depTile, height: 1, depth: 1))

        enc.endEncoding()
    }

    private func encodeSSAOPass(_ cb: MTLCommandBuffer) {
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        guard let enc = timedComputeEncoder(cb, "ssao") else { return }
        enc.label = "Illuminatorama.ssao"
        enc.setComputePipelineState(ssaoPipeline)
        enc.setTexture(depthTexture, index: 0)
        enc.setTexture(gbufferNormalRgh, index: 1)
        enc.setTexture(aoTexture, index: 2)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssaoPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }

    private func encodeSSAOSpatialFilter(_ cb: MTLCommandBuffer) {
        guard ssaoDenoiseEnabled else { return }
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        guard let enc = timedComputeEncoder(cb, "ssao.spatial") else { return }
        enc.label = "Illuminatorama.ssao.spatial"
        enc.setComputePipelineState(ssaoSpatialPipeline)
        enc.setTexture(aoTexture,        index: 0)   // raw AO input
        enc.setTexture(depthTexture,     index: 1)
        enc.setTexture(gbufferNormalRgh, index: 2)
        enc.setTexture(aoFilteredTexture, index: 3)  // output
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssaoSpatialPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }

    private func encodeSSAOTemporalPass(_ cb: MTLCommandBuffer) {
        guard ssaoDenoiseEnabled else { return }
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        guard let enc = timedComputeEncoder(cb, "ssao.temporal") else { return }
        enc.label = "Illuminatorama.ssao.temporal"
        enc.setComputePipelineState(ssaoTemporalPipeline)
        enc.setTexture(aoFilteredTexture,        index: 0)  // filtered current
        enc.setTexture(previousAOHistoryTexture,  index: 1)  // history read
        enc.setTexture(velocityTexture,           index: 2)
        enc.setTexture(currentAOHistoryTexture,   index: 3)  // history write
        enc.setTexture(aoSampleCount,             index: 4)  // adaptive count (read_write)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssaoTemporalPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }

    // MARK: - Lighting kernel specialization (WWDC23 #10127)

    /// Builds the `(variantKey, constants)` pair for `illumi_lighting`'s
    /// function-constant specialization from the host feature flags. The booleans
    /// bind at indices 0–4 matching the `kLighting*` declarations in
    /// Illuminatorama.metal; the key is the 5-bit string that disambiguates the
    /// variant in the pipeline cache. These MUST agree with the `frame.*Enabled`
    /// uniforms written in `updateFrameUniforms` — both derive from the same Bools.
    static func lightingFeatureConstants(ibl: Bool, shadow: Bool, dfg: Bool,
                                         ddgi: Bool, ddgiIrrCache: Bool)
        -> (key: String, constants: MTLFunctionConstantValues) {
        // The irradiance cache is only meaningful while DDGI is on — mirror the
        // exact derivation in updateFrameUniforms (`ddgiIrrCacheEnabled && ddgiEnabled`).
        let irrCache = ddgiIrrCache && ddgi
        var flags = [ibl, shadow, dfg, ddgi, irrCache]
        let cv = MTLFunctionConstantValues()
        flags.withUnsafeMutableBufferPointer { buf in
            for i in 0..<buf.count {
                cv.setConstantValue(buf.baseAddress! + i, type: .bool, index: i)
            }
        }
        let key = flags.map { $0 ? "1" : "0" }.joined()
        return (key, cv)
    }

    // One-entry memo of the last resolved lighting variant, so the steady-state
    // path (flags unchanged frame-to-frame, the overwhelmingly common case) is a
    // single UInt8 compare with zero allocation — no MTLFunctionConstantValues or
    // key string built. 0xFF is an impossible 5-bit value, so the first call misses.
    private var lastLightingFlags: UInt8 = 0xFF
    private var lastLightingPipeline: MTLComputePipelineState?

    /// The specialized `illumi_lighting` pipeline for the current feature flags.
    /// The shared cache compiles each flag combination once and returns the
    /// memoised state thereafter; this method additionally short-circuits the
    /// common no-change frame before touching the cache.
    ///
    /// WWDC23 #10127 lever 2 (async compilation): when a flag toggle first needs
    /// a not-yet-compiled variant, we compile it in the BACKGROUND (no frame
    /// hitch) and run the init-time uber-variant (`lightingPipeline`) meanwhile —
    /// it renders identical pixels, just without that combination's dead-code
    /// elimination. The fallback is deliberately NOT memoised, so once the
    /// background compile lands the next frame picks up the specialized variant.
    private func currentLightingPipeline() -> MTLComputePipelineState {
        let irrCache = ddgiIrrCacheEnabled && ddgiEnabled
        let bits: UInt8 = (iblEnabled     ? 1  : 0)
                        | (shadowsEnabled ? 2  : 0)
                        | (dfgLUTEnabled  ? 4  : 0)
                        | (ddgiEnabled    ? 8  : 0)
                        | (irrCache       ? 16 : 0)
        if bits == lastLightingFlags, let p = lastLightingPipeline { return p }

        let (key, cv) = Self.lightingFeatureConstants(
            ibl: iblEnabled, shadow: shadowsEnabled, dfg: dfgLUTEnabled,
            ddgi: ddgiEnabled, ddgiIrrCache: ddgiIrrCacheEnabled)
        // Non-blocking: ready variant → memoise + use it; still compiling → run
        // the uber-variant this frame and re-check next frame (no memo).
        if let ready = engine.pipelineCache.pipelineStateAsync(
            name: "illumi_lighting", device: device, constants: cv, variantKey: key) {
            lastLightingFlags = bits
            lastLightingPipeline = ready
            return ready
        }
        return lightingPipeline
    }

    /// Compute encoder that records this pass's GPU time when the pass timer is
    /// enabled (env-gated; #60 task 4 / #6), else a plain encoder. The `label`
    /// keys the per-pass breakdown.
    private func timedComputeEncoder(_ cb: MTLCommandBuffer, _ label: String) -> MTLComputeCommandEncoder? {
        if let pd = passTimer.descriptor(label: label) {
            return cb.makeComputeCommandEncoder(descriptor: pd)
        }
        return cb.makeComputeCommandEncoder()
    }

    private func encodeLightingPass(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "lighting") else { return }
        enc.label = "Illuminatorama.lighting"
        let pipe = currentLightingPipeline()
        enc.setComputePipelineState(pipe)
        enc.setTexture(gbufferAlbedoMet, index: 0)
        enc.setTexture(gbufferNormalRgh, index: 1)
        enc.setTexture(gbufferEmission, index: 2)
        enc.setTexture(depthTexture, index: 3)
        enc.setTexture(hdrTexture, index: 4)
        enc.setTexture(aoSourceTexture, index: 5)  // denoised AO (or raw when off)
        enc.setTexture(equirectSky ?? dummySkyTexture, index: 6)
        enc.setTexture(irradianceCube, index: 7)
        enc.setTexture(prefilteredCube, index: 8)
        enc.setTexture(shadowMap, index: 9)
        enc.setTexture(dfgLUT, index: 10)
        // Phase 3.1 — DDGI atlas slots (always bind; enabled flag gates sampling).
        // Phase 3.3 — read the ping-pong `Current` atlases, which hold the
        // freshest writes from `encodeDDGIFrame` above. `Previous` is what
        // the trace kernel sampled for the second bounce.
        enc.setTexture(ddgiIrradianceCurrent ?? dummySkyTexture, index: 11)
        enc.setTexture(ddgiDepthCurrent ?? ddgiDummyDepthAtlas, index: 12)
        // Phase 4.10 — per-spot shadow atlas. Always bound (the kernel
        // gates per-spot on `spotShadowSliceIndex >= 0`, which is set by
        // `updateSpotShadows` only when `spotShadowsEnabled` is on AND
        // the spot is within atlas capacity).
        enc.setTexture(spotShadowAtlas, index: 13)
        // Phase 3.4 — irradiance EMA cache. Previous = read, current = write.
        // When cache is disabled the kernel still writes current (fresh lookup),
        // so the toggle and bind are always active regardless of the flag.
        enc.setTexture(irrCachePrevious, index: 14)
        enc.setTexture(irrCacheCurrent,  index: 15)
        enc.setTexture(ltcMatTexture, index: 16)
        enc.setTexture(ltcMagTexture, index: 17)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        enc.setBuffer(pointLightBuffer, offset: 0, index: 1)
        enc.setBuffer(ddgiUniformBuffer, offset: 0, index: 2)
        enc.setBuffer(spotLightBuffer, offset: 0, index: 3)
        enc.setBuffer(areaLightBuffer, offset: 0, index: 4)
        enc.setBuffer(extraDirectionalBuffer, offset: 0, index: 5)
        dispatch(enc, pipeline: pipe, width: width, height: height)
        enc.endEncoding()
    }

    // Phase 4.39: SSR is now split into three passes.
    // 1) Gather  — ray march → write pre-weighted SSR delta to ssrRawTexture.
    // 2) Temporal — accumulate history with YCoCg variance clamp.
    // 3) Composite — hdrComposite = hdrTexture + denoisedSSR.
    private func encodeSSRGather(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "ssr.gather") else { return }
        enc.label = "Illuminatorama.ssr.gather"
        enc.setComputePipelineState(ssrGatherPipeline)
        enc.setTexture(hdrTexture,       index: 0)
        enc.setTexture(gbufferAlbedoMet, index: 1)
        enc.setTexture(gbufferNormalRgh, index: 2)
        enc.setTexture(depthTexture,     index: 3)
        enc.setTexture(ssrRawTexture,    index: 4)  // output: SSR delta
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrGatherPipeline, width: width, height: height)
        enc.endEncoding()
    }

    private func encodeSSRTemporalPass(_ cb: MTLCommandBuffer) {
        guard ssrDenoiseEnabled else { return }
        guard let enc = timedComputeEncoder(cb, "ssr.temporal") else { return }
        enc.label = "Illuminatorama.ssr.temporal"
        enc.setComputePipelineState(ssrTemporalPipeline)
        enc.setTexture(ssrRawTexture,             index: 0)  // current gather
        enc.setTexture(previousSSRHistoryTexture,  index: 1)  // history read
        enc.setTexture(velocityTexture,            index: 2)
        enc.setTexture(currentSSRHistoryTexture,   index: 3)  // history write
        enc.setTexture(ssrSampleCount,             index: 4)  // adaptive count (read_write)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrTemporalPipeline, width: width, height: height)
        enc.endEncoding()
    }

    private func encodeSSRComposite(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "ssr.composite") else { return }
        enc.label = "Illuminatorama.ssr.composite"
        enc.setComputePipelineState(ssrCompositePipeline)
        enc.setTexture(hdrTexture,        index: 0)  // base lighting
        enc.setTexture(ssrDenoisedTexture, index: 1)  // denoised SSR (or raw)
        enc.setTexture(hdrCompositeTexture, index: 2) // output
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrCompositePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Hardware ray-traced sun lighting (soft shadows + 1-bounce GI) added into
    /// the HDR composite. Runs after SSR so the additive result feeds TAA /
    /// bloom / tonemap on the same path as everything else. No-op unless
    /// `rtEnabled`, the device supports RT, and a geometry/AS has been built.
    private func encodeRTLightingPass(_ cb: MTLCommandBuffer) {
        guard rtEnabled, rtSupported,
              let basePipeline = rtPipeline, let accel = rtAccel,
              let triA = rtTriAlbedoBuffer, let triN = rtTriNormalBuffer,
              let vb = rtVertexBuffer, let ib = rtIndexBuffer,
              rtTriangleCount > 0 else { return }
        // Curve primitives (#60 item 7 incr. 2): when the soup AS holds a curve
        // geometry descriptor, the intersector's geometry-type contract must
        // include curves, so the kRTCurvesEnabled variant is REQUIRED.
        let curvesOn = rtSoupCurvesActive && rtSoupCurvePipeline != nil
            && rtSoupCurveSetDataBuffer != nil && rtSoupCurvePoolPoints != nil
            && rtSoupCurveRadii != nil && rtSoupCurveSegments != nil
        let pipeline = curvesOn ? rtSoupCurvePipeline! : basePipeline

        // Pull the (jittered) invVP + camera position the G-buffer used so the
        // world reconstruction matches the deferred passes exactly.
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        rtFrameSeed &+= 1
        var u = RTUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos,
            sunDir: simd_normalize(rtSunDirection), sunSoftnessRad: max(0.0005, rtSunSoftnessRad),
            sunColor: rtSunColor, giStrength: rtGIStrength,
            skyAmbient: ambientColor, specStrength: rtSpecStrength,
            width: UInt32(width), height: UInt32(height),
            shadowRays: UInt32(max(1, min(16, rtShadowRays))),
            giRays: UInt32(max(1, min(16, rtGIRays))),
            frameSeed: rtFrameSeed, rayTMin: 0.004, maxGIDist: 60.0,
            triangleCount: UInt32(rtTriangleCount),
            reflStrength: max(0, rtReflStrength),
            reflMaxDist: max(0.1, rtReflMaxDistance),
            reflRoughnessCutoff: max(0, rtReflRoughnessCutoff),
            reflRays: UInt32(max(1, min(8, rtReflRays))),
            reflEnabled: rtReflectionsEnabled ? 1 : 0,
            surfCacheEnabled: surfCacheActive ? 1 : 0,
            // Must be the ACTIVE tile size the atlas + update pass use (6 for
            // per-triangle cards, 64 for the room), NOT the 64 default — else the
            // read samples at the wrong scale and bleeds across tiles.
            surfTileSize: UInt32(surfActiveTileSize), surfTilesPerRow: UInt32(surfTilesPerRow),
            surfAtlasW: UInt32(surfAtlasW), surfAtlasH: UInt32(surfAtlasH),
            emitterCount: UInt32(activeEmitterLightCount),
            leafTransmission: max(0, leafTransmission))
        u.curvesEnabled = curvesOn ? 1 : 0
        memcpy(rtUniformBuffer.contents(), &u, MemoryLayout<RTUniforms>.stride)

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.rt"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(depthTexture, index: 0)
        enc.setTexture(gbufferNormalRgh, index: 1)
        enc.setTexture(gbufferAlbedoMet, index: 2)
        enc.setTexture(hdrCompositeTexture, index: 3)
        enc.setTexture(equirectSky ?? dummySkyTexture, index: 4)
        // Surface-cache atlas (current) + per-triangle card-UV maps. Always
        // bound; the kernel gates reads on `surfCacheEnabled`. Dummies keep
        // Metal happy when the cache is off / unbuilt.
        enc.setTexture(surfConsumerAtlas ?? dummySkyTexture, index: 5)
        enc.setTexture(rtDiffuseTexture, index: 6)   // diffuse (shadow+GI) → denoise pass
        enc.setAccelerationStructure(accel, bufferIndex: 0)
        enc.setBuffer(triA, offset: 0, index: 1)
        enc.setBuffer(triN, offset: 0, index: 2)
        enc.setBuffer(rtUniformBuffer, offset: 0, index: 3)
        enc.setBuffer(surfTriCardBuffer ?? ib, offset: 0, index: 4)
        enc.setBuffer(surfTriUVaBuffer ?? triA, offset: 0, index: 5)
        enc.setBuffer(surfTriUVcBuffer ?? triA, offset: 0, index: 6)
        enc.setBuffer(ddgiEmitterBuffer ?? triA, offset: 0, index: 7) // dummy when no emitters
        enc.setBuffer(surfCardRectBuffer ?? triA, offset: 0, index: 8) // per-card atlas rect (dummy off)
        enc.setBuffer(surfCardBuffer ?? surfCardDummyBuffer ?? triA, offset: 0, index: 9)     // per-card material (L_out reconstruction; dummy off, ≥ 1 SurfCard for arg validation)
        // Curve set data + pooled geometry (#60 item 7 incr. 2); dummies keep the
        // bindings valid for the base (curve-free) pipeline variant.
        enc.setBuffer(curvesOn ? rtSoupCurveSetDataBuffer : triA, offset: 0, index: 10)
        enc.setBuffer(curvesOn ? rtSoupCurvePoolPoints : triA, offset: 0, index: 11)
        enc.setBuffer(curvesOn ? rtSoupCurveRadii : triA, offset: 0, index: 12)
        enc.setBuffer(curvesOn ? rtSoupCurveSegments : triA, offset: 0, index: 13)
        // Curve-only primitive AS (#60 item 7 incr. 2) — a SECOND acceleration
        // structure the kernel traces alongside the triangle soup. Bind the
        // triangle `accel` as a harmless dummy for the base (curve-free) variant.
        enc.setAccelerationStructure((curvesOn ? rtSoupCurveAccel : nil) ?? accel, bufferIndex: 14)
        // The intersector reads triangle vertex positions (and, when curves are
        // active, the curve control points) through the AS; Metal needs explicit
        // residency hints for the buffers it can't statically prove are referenced.
        enc.useResource(vb, usage: .read)
        enc.useResource(ib, usage: .read)
        if curvesOn {
            if let ca = rtSoupCurveAccel { enc.useResource(ca, usage: .read) }
            if let p = rtSoupCurvePoolPoints { enc.useResource(p, usage: .read) }
            if let r = rtSoupCurveRadii { enc.useResource(r, usage: .read) }
            if let s = rtSoupCurveSegments { enc.useResource(s, usage: .read) }
        }
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Temporal accumulation of the RT diffuse (1-bounce GI + soft shadow)
    /// term: velocity-reprojected exponential history so the GI keeps
    /// converging under camera motion. Reads the raw RT diffuse + previous
    /// accumulated history, writes the new accumulated history (which the
    /// denoise then reads via `rtDiffuseDenoiseSource`, and which becomes next
    /// frame's history after the per-frame toggle). No-op unless enabled.
    private func encodeRTGITemporalAccum(_ cb: MTLCommandBuffer) {
        guard rtGITemporalEnabled, rtEnabled, rtSupported,
              rtPipeline != nil, rtAccel != nil, rtTriangleCount > 0 else { return }
        var u = RTGITemporalUniforms(
            width: UInt32(width), height: UInt32(height),
            enabled: 1,
            isFirstFrame: rtGINeedsFirstFrame ? 1 : 0,
            blend: max(0.01, min(1.0, rtGITemporalBlend)),
            gammaClamp: max(1.0, rtGITemporalClamp))
        memcpy(rtGITemporalUniformBuffer.contents(), &u, MemoryLayout<RTGITemporalUniforms>.stride)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.rt.giTemporal"
        enc.setComputePipelineState(rtGITemporalPipeline)
        enc.setTexture(rtDiffuseTexture,           index: 0)  // current raw GI
        enc.setTexture(previousRTGIHistoryTexture, index: 1)  // history read
        enc.setTexture(velocityTexture,            index: 2)
        enc.setTexture(currentRTGIHistoryTexture,  index: 3)  // accumulated write
        enc.setTexture(giSampleCount,              index: 4)  // adaptive count (read_write)
        enc.setBuffer(rtGITemporalUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: rtGITemporalPipeline, width: width, height: height)
        enc.endEncoding()
    }

    // ── Phase 4.44: SVGF à-trous cascade ─────────────────────────────────────

    private struct SVGFAtrousUniforms {
        var width: UInt32; var height: UInt32; var stepSize: UInt32; var _pad0: UInt32 = 0
        var sigmaL: Float; var sigmaZ: Float; var sigmaN: Float; var _pad1: Float = 0
    }

    /// SVGF denoiser for the RT GI term: variance estimate + N à-trous cascade
    /// levels. Reads `currentRTGIHistoryTexture` (the temporally-accumulated GI)
    /// and writes the final spatially-filtered result into `giAtrousA/B`, then
    /// adds it additively into `hdrCompositeTexture`.
    ///
    /// Only runs when `svgfEnabled && rtGITemporalEnabled && rtEnabled`.
    /// Replaces `encodeRTDenoiseComposite` for the GI term on this path.
    private func encodeSVGFCascade(_ cb: MTLCommandBuffer) {
        guard svgfEnabled,
              rtGITemporalEnabled, rtEnabled, rtSupported,
              rtPipeline != nil, rtAccel != nil, rtTriangleCount > 0 else { return }

        let levels = max(1, min(5, svgfLevels))
        let W = UInt32(width), H = UInt32(height)

        // Pass 1: variance estimate from the accumulated GI.
        do {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "Illuminatorama.svgf.variance"
            enc.setComputePipelineState(svgfVariancePipeline)
            enc.setTexture(currentRTGIHistoryTexture, index: 0)  // accumulated GI
            enc.setTexture(giVariance,                index: 1)  // variance output
            var u = SVGFAtrousUniforms(width: W, height: H, stepSize: 1,
                                        sigmaL: svgfSigmaL, sigmaZ: svgfSigmaZ, sigmaN: svgfSigmaN)
            enc.setBytes(&u, length: MemoryLayout<SVGFAtrousUniforms>.stride, index: 0)
            dispatch(enc, pipeline: svgfVariancePipeline, width: width, height: height)
            enc.endEncoding()
        }

        // À-trous cascade: ping-pong color between giAtrousA/B.
        // Variance ping-pongs between giVariance and giAtrousB (variance never
        // aliases the color output since they're separate textures).
        //
        // Trace:
        //   level 0: colorIn=giHistory  varIn=giVariance  → colorOut=giAtrousA  varOut=giAtrousB
        //   level 1: colorIn=giAtrousA  varIn=giAtrousB   → colorOut=giAtrousB  varOut=giVariance
        //   level 2: colorIn=giAtrousB  varIn=giVariance  → colorOut=giAtrousA  varOut=giAtrousB
        //   … and so on. Final result always lives in `lastColorOut`.
        var lastColorOut = giAtrousA
        for level in 0..<levels {
            let stepSize = UInt32(1 << level)
            let cIn  = level == 0 ? currentRTGIHistoryTexture : (level % 2 == 1 ? giAtrousA : giAtrousB)
            let cOut = level % 2 == 0 ? giAtrousA : giAtrousB
            let vIn  = level % 2 == 0 ? giVariance : giAtrousB
            let vOut = level % 2 == 0 ? giAtrousB  : giVariance
            lastColorOut = cOut

            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "Illuminatorama.svgf.atrous.\(level)"
            enc.setComputePipelineState(svgfAtrousPipeline)
            enc.setTexture(cIn,              index: 0)
            enc.setTexture(vIn,              index: 1)
            enc.setTexture(depthTexture,     index: 2)
            enc.setTexture(gbufferNormalRgh, index: 3)
            enc.setTexture(cOut,             index: 4)
            enc.setTexture(vOut,             index: 5)
            var u = SVGFAtrousUniforms(width: W, height: H, stepSize: stepSize,
                                        sigmaL: svgfSigmaL, sigmaZ: svgfSigmaZ, sigmaN: svgfSigmaN)
            enc.setBytes(&u, length: MemoryLayout<SVGFAtrousUniforms>.stride, index: 0)
            dispatch(enc, pipeline: svgfAtrousPipeline, width: width, height: height)
            enc.endEncoding()
        }

        addGIIntoComposite(cb, source: lastColorOut)
    }

    /// Additively composites the denoised GI buffer into `hdrCompositeTexture`.
    /// Reuses `illumi_rt_denoise` with radius=0 (no bilateral — SVGF already filtered).
    private func addGIIntoComposite(_ cb: MTLCommandBuffer, source: MTLTexture) {
        var du = RTDenoiseUniforms(width: UInt32(width), height: UInt32(height),
                                   enabled: 0, radius: 0,
                                   kDepth: 0, kNorm: 0)
        memcpy(rtDenoiseUniformBuffer.contents(), &du, MemoryLayout<RTDenoiseUniforms>.stride)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.svgf.composite"
        enc.setComputePipelineState(rtDenoisePipeline)
        enc.setTexture(source,              index: 0)
        enc.setTexture(depthTexture,        index: 1)
        enc.setTexture(gbufferNormalRgh,    index: 2)
        enc.setTexture(hdrCompositeTexture, index: 3)
        enc.setBuffer(frameUniformBuffer,       offset: 0, index: 0)
        enc.setBuffer(rtDenoiseUniformBuffer,   offset: 0, index: 1)
        dispatch(enc, pipeline: rtDenoisePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Depth + normal guided bilateral over the RT diffuse buffer, adding the
    /// cleaned soft-shadow + 1-bounce-GI term into the HDR composite (additive,
    /// own-pixel read-modify-write) before the volumetric / TAA passes. Gated on
    /// the same conditions as `encodeRTLightingPass` so it only composites a
    /// buffer that was actually written this frame; with `rtDenoiseEnabled` off
    /// it passes the raw diffuse straight through (the old additive behaviour).
    private func encodeRTDenoiseComposite(_ cb: MTLCommandBuffer) {
        guard rtEnabled, rtSupported,
              rtPipeline != nil, rtAccel != nil, rtTriangleCount > 0 else { return }
        var u = RTDenoiseUniforms(
            width: UInt32(width), height: UInt32(height),
            enabled: rtDenoiseEnabled ? 1 : 0,
            radius: UInt32(max(0, min(4, rtDenoiseRadius))),
            kDepth: max(0, rtDenoiseDepthSensitivity),
            kNorm: max(0, rtDenoiseNormalSharpness))
        memcpy(rtDenoiseUniformBuffer.contents(), &u, MemoryLayout<RTDenoiseUniforms>.stride)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.rt.denoise"
        enc.setComputePipelineState(rtDenoisePipeline)
        enc.setTexture(rtDiffuseDenoiseSource, index: 0)
        enc.setTexture(depthTexture,        index: 1)
        enc.setTexture(gbufferNormalRgh,    index: 2)
        enc.setTexture(hdrCompositeTexture, index: 3)
        enc.setBuffer(frameUniformBuffer,     offset: 0, index: 0)
        enc.setBuffer(rtDenoiseUniformBuffer, offset: 0, index: 1)
        dispatch(enc, pipeline: rtDenoisePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Single-scatter volumetric light shaft, added into the HDR composite
    /// after the RT lighting (so the beam glows in the air in front of lit
    /// surfaces) and before particles / TAA / bloom. No-op unless enabled.
    private func encodeVolumetricPass(_ cb: MTLCommandBuffer) {
        guard volumetricEnabled, let pipeline = volPipeline else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        volFrameSeed &+= 1
        var u = VolUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos, fogDensity: max(0, volFogDensity),
            sunDir: simd_normalize(rtSunDirection), scatterStrength: max(0, volScatterStrength),
            sunColor: rtSunColor, anisotropy: volAnisotropy,
            windowX: volWindowX, winY0: volWindowRect.x, winY1: volWindowRect.y,
            winZ0: volWindowRect.z, winZ1: volWindowRect.w, maxDist: volMaxDistance,
            steps: UInt32(max(1, min(96, volSteps))), frameSeed: volFrameSeed,
            width: UInt32(width), height: UInt32(height), feather: volFeather,
            isOutdoor: volOutdoorMode ? 1 : 0)
        memcpy(volUniformBuffer.contents(), &u, MemoryLayout<VolUniforms>.stride)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.volumetric"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(depthTexture, index: 0)
        enc.setTexture(hdrCompositeTexture, index: 1)
        enc.setBuffer(volUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// In-view (perspective) cloud composite — issue #61. Raymarches the scene's
    /// cloud volume along per-pixel camera rays (crisp at output res, vs the
    /// equirect dome's upscaled-crop blur) and writes the result into
    /// `hdrCompositeTexture`, before bloom/tonemap and before the additive
    /// firework particles (which draw in `encodePostResolveFX`). Opt-in:
    /// no-ops unless the host scene enabled it and wired the reuse handles.
    /// v1 REPLACES sky pixels (the only opt-in scene has no opaque geometry); a
    /// depth-clipped composite-over-geometry generalisation is future work.
    private func encodeCloudPass(_ cb: MTLCommandBuffer) {
        guard inViewCloudsEnabled,
              let pipeline = cloudInViewPipeline,
              let noise = cloudNoiseTexture,
              let skyU = cloudSkyUniforms,
              let lights = cloudBurstLights else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        var u = CloudInViewUniforms(invViewProjection: fu.invViewProjection,
                                    cameraWorldPos: SIMD4<Float>(fu.cameraWorldPos, 0))
        memcpy(cloudInViewUniformBuffer.contents(), &u, MemoryLayout<CloudInViewUniforms>.stride)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.cloudInView"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(hdrCompositeTexture, index: 0)
        enc.setTexture(noise, index: 1)
        enc.setBuffer(skyU, offset: 0, index: 0)
        enc.setBuffer(cloudInViewUniformBuffer, offset: 0, index: 1)
        enc.setBuffer(lights, offset: 0, index: 2)
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Phase 2.7 — TAA resolve. Reprojects last frame's accumulated HDR with
    /// the velocity buffer, neighborhood-clamps it, and blends a small
    /// contribution of the current frame in. Skipped entirely when TAA is
    /// disabled — bloom + tonemap then read directly from hdrCompositeTexture.
    private func encodeTAAResolve(_ cb: MTLCommandBuffer) {
        guard taaEnabled else { return }
        guard let enc = timedComputeEncoder(cb, "taa") else { return }
        enc.label = "Illuminatorama.taa"
        enc.setComputePipelineState(taaResolvePipeline)
        enc.setTexture(hdrCompositeTexture,     index: 0)
        enc.setTexture(previousHistoryTexture,  index: 1)
        enc.setTexture(velocityTexture,         index: 2)
        enc.setTexture(currentHistoryTexture,   index: 3)
        // Phase 4.44 — depth lets the resolve synthesise a camera-only motion
        // vector for sky pixels (which the G-buffer leaves at zero velocity).
        enc.setTexture(depthTexture,            index: 4)
        // Previous frame's depth — lets the resolve reject history at a
        // disocclusion (a fast mover vacated this pixel) even when the vacated
        // surface and the mover share luma, which neighbourhood clamping can't.
        enc.setTexture(previousDepthTexture,    index: 5)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: taaResolvePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// What bloom + tonemap should read from this frame. With TAA on, that's
    /// the history texture we just wrote to; otherwise the SSR-composited HDR.
    private var postFXSource: MTLTexture {
        taaEnabled ? currentHistoryTexture : hdrCompositeTexture
    }

    /// True for the remainder of a frame once `encodePostResolveFX` has drawn
    /// the additive overlay (dust + sparks) into the display buffer. When set,
    /// downstream post-FX read `hdrTexture` (resolved opaque + overlay) instead
    /// of the bare `postFXSource`.
    private var overlayApplied = false

    /// The HDR colour post-FX (exposure, DOF, bloom, tonemap) should sample.
    /// After the post-resolve overlay it's the display buffer; otherwise the
    /// plain resolved source.
    private var displaySource: MTLTexture {
        overlayApplied ? hdrTexture : postFXSource
    }

    /// Phase 4.43 — composite the additive overlay FX (dust motes + emitter
    /// sparks) AFTER the TAA resolve so they never enter the temporal history.
    ///
    /// WHY HERE, NOT EARLIER: these FX are sub-pixel-bright points that move on
    /// their own (curl-noise drift, particle velocity) but write NO motion
    /// vectors into the velocity attachment. Drawn before the resolve (the old
    /// Phase 4.42 placement) the TAA kernel reprojected each mote as if it were
    /// static and blended in ~90% of the previous frame, smearing every mote
    /// into a trail along its path — most visible where the window-shadow test
    /// spikes a mote's brightness in the sunbeam. Compositing past the resolve
    /// keeps the history ping-pong trail-free.
    ///
    /// The display buffer is `hdrTexture`, which is idle once the SSR composite
    /// has consumed it — reused here instead of allocating a fresh full-res
    /// target. We seed it with the resolved opaque image (a blit), then run the
    /// existing additive draws (which already target `hdrTexture` with the
    /// G-buffer depth bound, so opaque geometry still occludes the FX).
    private func encodePostResolveFX(_ cb: MTLCommandBuffer) {
        overlayApplied = false
        let hasEmitters = particleEmitters.contains { $0.enabled }
        guard !particleFields.isEmpty || hasEmitters || !particleStreaks.isEmpty else { return }

        // Seed the display buffer with the resolved opaque image.
        let resolved = postFXSource
        if resolved !== hdrTexture {
            guard let blit = cb.makeBlitCommandEncoder() else { return }
            blit.label = "Illuminatorama.overlay.seed"
            blit.copy(from: resolved,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: width, height: height, depth: 1),
                      to: hdrTexture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }

        // Additive draws into the display buffer (all target `hdrTexture`).
        encodeParticleDraw(cb)
        encodeExternalPointDraw(cb)
        encodeStreakDraw(cb)           // Phase 4.44b — velocity-aligned streak quads
        overlayApplied = true
    }

    /// Source for bloom + tonemap: the DOF output when DOF ran this frame,
    /// otherwise the raw post-FX source. (Exposure estimate still reads the
    /// pre-DOF source so the blur doesn't perturb auto-exposure.)
    private var bloomTonemapSource: MTLTexture {
        (dofApplied ? dofOutputTexture : nil) ?? displaySource
    }

    /// Gather depth-of-field on the resolved HDR. Runs after the exposure
    /// estimate and before bloom; writes a private DOF texture that bloom +
    /// tonemap then read. No-op unless `dofEnabled` and the pipeline exists.
    private func encodeDOFPass(_ cb: MTLCommandBuffer) {
        dofApplied = false
        guard dofEnabled, let pipeline = dofPipeline else { return }
        // Lazily (re)allocate the DOF target to the current internal size.
        if dofOutputTexture == nil
            || dofOutputTexture?.width != width || dofOutputTexture?.height != height {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = max(1, width); d.height = max(1, height)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofOutputTexture = device.makeTexture(descriptor: d)
            dofOutputTexture?.label = "Illuminatorama.dof"
        }
        guard let out = dofOutputTexture else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        var p = DOFParams(
            invProjection: fu.invProjection,
            focusDist: max(0.05, dofFocusDistance), aperture: max(0, dofAperture),
            maxRadius: max(0, dofMaxRadius), focusRange: max(0.05, dofFocusRange),
            width: UInt32(width), height: UInt32(height))
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.dof"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(displaySource, index: 0)
        enc.setTexture(depthTexture, index: 1)
        enc.setTexture(out, index: 2)
        enc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
        dofApplied = true
    }

    private var currentHistoryTexture: MTLTexture { historyToggle ? historyB : historyA }
    private var previousHistoryTexture: MTLTexture { historyToggle ? historyA : historyB }

    /// Phase 4.21 — estimate the scene's log-luminance once per frame and
    /// EMA-smooth into `exposureBuffer.smoothedExposure`. Lives between
    /// the TAA resolve and the bloom passes so it reads the post-TAA
    /// HDR composite (the same texture the tonemap will read) — the
    /// resulting exposure is consistent with what the tonemap then
    /// applies. Single threadgroup, 256 threads, shared-mem reduction.
    /// Wall-clock of the previous `encodeExposureEstimate` call. Used to
    /// derive a real per-frame `dt` for the EMA so the smoothing keeps
    /// the same half-life regardless of the host's frame rate.
    private var lastExposureTickTime: CFTimeInterval = CACurrentMediaTime()

    private func encodeExposureEstimate(_ cb: MTLCommandBuffer) {
        guard autoExposureEnabled else { return }
        // Per-frame dt, clamped to a sane window so a stalled frame
        // (debugger pause, scene reload) doesn't pump the EMA.
        let now = CACurrentMediaTime()
        lastFrameDuration = max(0.001, min(0.5, now - lastExposureTickTime))
        lastExposureTickTime = now
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.exposureEstimate"
        enc.setComputePipelineState(exposureEstimatePipeline)
        enc.setTexture(displaySource, index: 0)
        enc.setBuffer(exposureBuffer, offset: 0, index: 0)
        // imgSize for stride computation inside the kernel.
        var imgSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        enc.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        // params: x = targetEV, y = halfLife, z = maxBoost, w = minBoost.
        // Phase 4.28 — minBoost is now an independent floor (not `1/maxBoost`)
        // so bright `environmentImage` scenes can darken far enough to escape
        // a blown-white frame while dim scenes keep their ≤maxBoost cap.
        var params = SIMD4<Float>(autoExposureTargetEV,
                                   autoExposureHalfLife,
                                   autoExposureMaxBoost,
                                   max(autoExposureMinExposure, 0.01))
        enc.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        // Update the host-driven `dt` slot in the buffer — the EMA step
        // inside the kernel needs to know how much wall-time elapsed
        // since the last estimate so the half-life math is correct.
        let dt = Float(max(0.001, lastFrameDuration))
        // The slot is 4 floats in; ExposureState.deltaTime is offset 12.
        let dtPtr = exposureBuffer.contents().advanced(by: 12)
                                              .bindMemory(to: Float.self, capacity: 1)
        dtPtr.pointee = dt
        // Single threadgroup of 256 threads with shared memory for sum +
        // count. `threadgroupMemoryLength(at:)` is bytes for one Float
        // per thread; two arrays = 2 × 256 × 4 = 2048 bytes for floats
        // and 1024 for uints. Set both indices.
        enc.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.stride, index: 0)
        enc.setThreadgroupMemoryLength(256 * MemoryLayout<UInt32>.stride, index: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    private func encodeBloomPasses(_ cb: MTLCommandBuffer) {
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        guard let enc = timedComputeEncoder(cb, "bloom") else { return }
        enc.label = "Illuminatorama.bloom"

        enc.setComputePipelineState(bloomThresholdPipeline)
        enc.setTexture(bloomTonemapSource, index: 0)
        enc.setTexture(bloomBrightHalf, index: 1)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: bloomThresholdPipeline, width: halfW, height: halfH)

        enc.setComputePipelineState(bloomBlurHPipeline)
        enc.setTexture(bloomBrightHalf, index: 0)
        enc.setTexture(bloomBlurHHalf, index: 1)
        dispatch(enc, pipeline: bloomBlurHPipeline, width: halfW, height: halfH)

        enc.setComputePipelineState(bloomBlurVPipeline)
        enc.setTexture(bloomBlurHHalf, index: 0)
        enc.setTexture(bloomBlurVHalf, index: 1)
        dispatch(enc, pipeline: bloomBlurVPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }

    // ── Issue #65: colour-grading LUT ─────────────────────────────────────────

    /// IEEE-754 binary32 → binary16 (half) bit pattern for `rgba16Float` uploads.
    @inline(__always)
    private static func float32to16(_ v: Float) -> UInt16 { Float16(v).bitPattern }

    /// Build an identity 3D LUT (`rgba16Float`, `size³`) where the texel at
    /// (r,g,b) holds the colour (r,g,b)/(size-1). Sampling it with a colour in
    /// [0,1]³ (half-texel-corrected) returns that colour unchanged — an exact
    /// no-op grade. Uploaded once; `.shared` storage so the CPU bake is visible.
    private static func makeIdentityColorLUT(device: MTLDevice, size: Int) -> MTLTexture? {
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.label = "Illuminatorama.colorLUT"
        var data = [UInt16](repeating: 0, count: size * size * size * 4)
        let inv = 1.0 / Float(size - 1)
        var i = 0
        for b in 0..<size {
            for g in 0..<size {
                for r in 0..<size {
                    data[i+0] = float32to16(Float(r) * inv)
                    data[i+1] = float32to16(Float(g) * inv)
                    data[i+2] = float32to16(Float(b) * inv)
                    data[i+3] = float32to16(1.0)
                    i += 4
                }
            }
        }
        let bpr = size * 4 * MemoryLayout<UInt16>.size
        let bpi = bpr * size
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size),
                        mipmapLevel: 0, slice: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: bpr, bytesPerImage: bpi)
        }
        return tex
    }

    /// Replace the colour-grade LUT from packed RGBA float data laid out as the
    /// standard .cube ordering (red fastest, then green, then blue). `rgba` must
    /// hold `size³` SIMD4 entries. Pass `nil` to reset to identity. The grade only
    /// shows when `colorLUTAmount > 0`. Display-space (the LUT is sampled after
    /// tonemapping, on the sRGB-encoded colour).
    @discardableResult
    public func setColorLUT(size: Int, rgba: [SIMD4<Float>]?) -> Bool {
        guard let rgba else {
            if let id = Self.makeIdentityColorLUT(device: device, size: Self.colorLUTDefaultSize) {
                colorLUT = id; colorLUTSize = Self.colorLUTDefaultSize; return true
            }
            return false
        }
        guard size >= 2, rgba.count == size * size * size else { return false }
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return false }
        tex.label = "Illuminatorama.colorLUT(custom)"
        var data = [UInt16](repeating: 0, count: rgba.count * 4)
        for (j, c) in rgba.enumerated() {
            data[j*4+0] = Self.float32to16(c.x)
            data[j*4+1] = Self.float32to16(c.y)
            data[j*4+2] = Self.float32to16(c.z)
            data[j*4+3] = Self.float32to16(1.0)
        }
        let bpr = size * 4 * MemoryLayout<UInt16>.size
        let bpi = bpr * size
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size),
                        mipmapLevel: 0, slice: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: bpr, bytesPerImage: bpi)
        }
        colorLUT = tex; colorLUTSize = size
        return true
    }

    /// Parse an Adobe/Resolve `.cube` 3D LUT and install it. Returns false on a
    /// malformed file. Supports `LUT_3D_SIZE` + `r g b` rows (the common subset).
    @discardableResult
    public func loadCubeLUT(contentsOf url: URL) -> Bool {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return false }
        var size = 0
        var entries: [SIMD4<Float>] = []
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            if line.uppercased().hasPrefix("LUT_3D_SIZE") {
                size = Int(line.split(separator: " ").last.map(String.init) ?? "") ?? 0
                if size > 1 { entries.reserveCapacity(size*size*size) }
                continue
            }
            let comps = line.split(separator: " ").compactMap { Float($0) }
            if comps.count == 3 { entries.append(SIMD4(comps[0], comps[1], comps[2], 1)) }
        }
        guard size > 1, entries.count == size*size*size else { return false }
        return setColorLUT(size: size, rgba: entries)
    }

    /// Generate one of the built-in procedural grades as packed RGBA LUT data
    /// (.cube order, red fastest). Returns nil for `.none` (caller installs the
    /// identity LUT). The grade is computed in DISPLAY (~sRGB) space, matching how
    /// the tonemap samples the LUT.
    static func generateColorGradeLUT(look: IlluminatoramaColorGradeLook,
                                      size: Int) -> [SIMD4<Float>]? {
        guard look != .none, size >= 2 else { return nil }
        func luma(_ c: SIMD3<Float>) -> Float { c.x*0.2126 + c.y*0.7152 + c.z*0.0722 }
        func contrast(_ c: SIMD3<Float>, _ k: Float) -> SIMD3<Float> { (c - 0.5) * k + 0.5 }
        func smooth(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
            let t = max(0, min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t)
        }
        var out = [SIMD4<Float>](); out.reserveCapacity(size*size*size)
        let inv = 1.0 / Float(size - 1)
        for b in 0..<size { for g in 0..<size { for r in 0..<size {
            var c = SIMD3(Float(r), Float(g), Float(b)) * inv
            switch look {
            case .none: break
            case .tealOrange:
                let t = smooth(0.2, 0.85, luma(c))
                let shadow = SIMD3<Float>(-0.02, 0.03, 0.08)   // push shadows teal
                let high   = SIMD3<Float>( 0.09, 0.02, -0.07)  // push highlights orange
                c += shadow * (1 - t) + high * t
                c = contrast(c, 1.10)
            case .warm:
                c *= SIMD3<Float>(1.07, 1.01, 0.90); c = contrast(c, 1.05)
            case .cool:
                c *= SIMD3<Float>(0.92, 1.00, 1.09); c = contrast(c, 1.05)
            case .noir:
                let l = luma(c); c = contrast(SIMD3(l, l, l), 1.22)
            case .vibrant:
                let l = luma(c); c = l + (c - l) * 1.45; c = contrast(c, 1.06)
            }
            c = simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
            out.append(SIMD4(c, 1))
        }}}
        return out
    }

    /// Apply the shared panel's colour grade (issue #65). Rebakes the LUT only when
    /// the chosen look changes (the CPU bake is a per-frame no-op in steady state),
    /// and sets `colorLUTAmount` each frame. Called from `render()` alongside the
    /// other shared-FX appliers, so every Illuminatorama scene inherits it.
    func applySharedColorGrade(_ s: IlluminatoramaSharedSettings = .shared) {
        if s.colorGradeLook != appliedColorGradeLook {
            appliedColorGradeLook = s.colorGradeLook
            let data = Self.generateColorGradeLUT(look: s.colorGradeLook,
                                                  size: Self.colorLUTDefaultSize)
            setColorLUT(size: Self.colorLUTDefaultSize, rgba: data)
        }
        colorLUTAmount = (s.colorGradeLook == .none) ? 0 : Float(s.colorLUTAmount)
    }

    private func encodeTonemapPass(_ cb: MTLCommandBuffer) {
        // Phase 4.28 — RENDER pass (not compute). Draws a fullscreen triangle
        // into `tonemapWriteTarget`; the end-of-pass `.store` is what makes the
        // result visible to SceneKit's cross-queue background sample without a
        // CPU completion wait (the bug that made heavy scenes show a blank /
        // white overlay frame). `.dontCare` load is fine — the fragment writes
        // every output pixel.
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = tonemapWriteTarget
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.tonemap"
        enc.setRenderPipelineState(tonemapPipeline)
        enc.setFragmentTexture(bloomTonemapSource, index: 0)
        enc.setFragmentTexture(bloomBlurVHalf, index: 1)
        // Issue #65 — 3D colour-grade LUT at texture(2). Always bound (identity by
        // default), so the shader never branches on nil; the grade is gated by
        // `colorLUTAmount` in the uniforms.
        enc.setFragmentTexture(colorLUT, index: 2)
        enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        // Phase 4.21 — exposure buffer at buffer(1). The fragment reads
        // `expoState.smoothedExposure` when `frame.autoExposureEnabled`
        // is set, falls back to `frame.exposure` otherwise.
        enc.setFragmentBuffer(exposureBuffer, offset: 0, index: 1)
        // Fullscreen triangle — 3 vertices, positions synthesised in the
        // vertex shader from `[[vertex_id]]`; no vertex buffer needed.
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          width: Int, height: Int,
                          maxThreadsOverride: Int? = nil) {
        let tgw = pipeline.threadExecutionWidth
        // Default = the full threadgroup (max occupancy by thread count). For
        // register-heavy kernels (RT) a SMALLER group can run faster — more
        // threadgroups resident per core. `maxThreadsOverride` lets a specific
        // pass cap the total threads (#60 task 4 occupancy tuning).
        let cap = pipeline.maxTotalThreadsPerThreadgroup
        let total = max(tgw, min(cap, maxThreadsOverride ?? cap))
        let tgh = max(1, total / tgw)
        let threadgroup = MTLSize(width: tgw, height: tgh, depth: 1)
        let groups = MTLSize(
            width:  (width  + tgw - 1) / tgw,
            height: (height + tgh - 1) / tgh,
            depth: 1
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
    }

    /// 3D dispatch covering (faceSize × faceSize × 6) for cubemap-baking
    /// kernels that take `[[thread_position_in_grid]]` as `uint3` and use
    /// `gid.z` as the face index. Threadgroups are 8×8×1 — depth=1 because
    /// faces don't share state and that keeps the threadgroup tile small
    /// enough to fit on every device.
    private func dispatchCubeFaces(_ enc: MTLComputeCommandEncoder,
                                   pipeline: MTLComputePipelineState,
                                   faceSize: Int) {
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(
            width:  (faceSize + tg.width - 1)  / tg.width,
            height: (faceSize + tg.height - 1) / tg.height,
            depth:  6
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    // ── Upload helpers ────────────────────────────────────────────────────────

    private func ensureBufferCapacity() {
        if instances.count > instanceCapacity {
            let newCap = max(instances.count, instanceCapacity * 2)
            let stride = MemoryLayout<IlluminatoramaInstance>.stride * newCap
            // Grow BOTH ping-pong buffers in lockstep — the previous-frame
            // buffer is now uninitialized for motion-vector purposes, so we
            // also force TAA back into first-frame mode.
            if let a = device.makeBuffer(length: stride, options: .storageModeShared),
               let b = device.makeBuffer(length: stride, options: .storageModeShared) {
                a.label = "Illuminatorama.instancesA"
                b.label = "Illuminatorama.instancesB"
                instanceBufferA = a
                instanceBufferB = b
                instanceCapacity = newCap
                taaNeedsFirstFrame = true
            }
        }
        // Grow-on-demand for the ring-buffered per-frame inputs. Reallocate the
        // WHOLE ring (all maxFramesInFlight slots) so every in-flight frame keeps
        // a same-size buffer — growing only the current slot would let an
        // in-flight frame bind the old (too-small) slot next time round the ring.
        // Slots are uploaded fresh each frame, so there is nothing to copy over.
        func growRing<T>(_ ring: inout [MTLBuffer], _ capacity: inout Int,
                         needed: Int, type: T.Type, label: String) {
            guard needed > capacity else { return }
            let newCap = max(needed, capacity * 2)
            var nr: [MTLBuffer] = []
            nr.reserveCapacity(ring.count)
            for i in 0..<ring.count {
                guard let nb = device.makeBuffer(
                    length: MemoryLayout<T>.stride * newCap, options: .storageModeShared
                ) else { return }   // alloc failed — keep the old ring, retry next frame
                nb.label = "\(label).\(i)"
                nr.append(nb)
            }
            ring = nr
            capacity = newCap
        }
        // Keep the parallel superquadric param ring at least as large as the
        // instance buffer (indexed by the same grouped instance id).
        growRing(&superquadricParamRing, &superquadricParamCapacity,
                 needed: instances.count, type: IlluminatoramaInstance.SuperquadricParam.self,
                 label: "Illuminatorama.superquadricParams")
        growRing(&pointLightRing, &pointLightCapacity,
                 needed: pointLights.count, type: IlluminatoramaPointLight.self,
                 label: "Illuminatorama.pointLights")
        growRing(&spotLightRing, &spotLightCapacity,
                 needed: spotLights.count, type: IlluminatoramaSpotLight.self,
                 label: "Illuminatorama.spotLights")
        growRing(&areaLightRing, &areaLightCapacity,
                 needed: areaLights.count, type: IlluminatoramaAreaLight.self,
                 label: "Illuminatorama.areaLights")
        growRing(&extraDirectionalRing, &extraDirectionalCapacity,
                 needed: extraDirectionals.count, type: IlluminatoramaDirectionalLight.self,
                 label: "Illuminatorama.extraDirectionals")
    }

    /// Advance the eased Post-FX shadows toward their slider targets. `Instant`
    /// (tau = 0) snaps; otherwise exponential smoothing `k = 1 − e^(−dt/tau)`.
    /// Self-timed off `CACurrentMediaTime` so it needs no dt from the caller, and
    /// the first call (lastEaseTime == 0 → dt 0 → k 1) seeds the shadows at the
    /// current targets (no startup glide). Mirrors AspectTest's `easeCamera`.
    private func advancePostFXEasing() {
        let now = CACurrentMediaTime()
        let dt = lastPostFXEaseTime == 0 ? 0 : max(0, min(0.1, now - lastPostFXEaseTime))
        lastPostFXEaseTime = now
        let k = (postFXEasingTau <= 0 || dt == 0) ? 1.0 : 1 - exp(-dt / postFXEasingTau)
        let kf = Float(k)
        easedExposure          += (exposure              - easedExposure)          * kf
        easedBloomThreshold    += (bloomThreshold        - easedBloomThreshold)    * kf
        easedBloomIntensity    += (bloomIntensity        - easedBloomIntensity)    * kf
        easedChromaticAberration += (max(0, chromaticAberration) - easedChromaticAberration) * kf
        easedFringe            += (max(0, fringe)        - easedFringe)            * kf
        easedFringeTint        += (fringeTint            - easedFringeTint)        * kf
        easedSphericalAberration += (max(0, sphericalAberration) - easedSphericalAberration) * kf
    }

    private func uploadFrameUniforms() {
        advancePostFXEasing()
        let view = camera.viewMatrix
        // Apply Phase 2.7 sub-pixel projection jitter — translates the
        // projected clip-space x/y by a Halton(2,3) offset of up to
        // ±0.5 × taaJitterPixels pixels (in NDC units). The same matrix is
        // used everywhere (G-buffer rasterization, SSAO/SSR reconstruction,
        // lighting unprojection) so the internal consistency holds and TAA
        // averages out the sub-pixel offsets as supersampling.
        let unjitteredProj = camera.projectionMatrix
        let proj: simd_float4x4
        if taaEnabled && taaJitterPixels > 0 {
            let h2 = Self.halton(taaFrameIndex &+ 1, base: 2) - 0.5
            let h3 = Self.halton(taaFrameIndex &+ 1, base: 3) - 0.5
            let jx = h2 * taaJitterPixels * 2.0 / Float(width)
            // NDC.y is up; pixel.y is down — but jitter is in NDC space here
            // and we want it expressed consistently with how the motion
            // vector flips Y, so leave +Y up here.
            let jy = h3 * taaJitterPixels * 2.0 / Float(height)
            var J = matrix_identity_float4x4
            J.columns.3 = SIMD4(jx, jy, 0, 1)
            proj = J * unjitteredProj
        } else {
            proj = unjitteredProj
        }
        let vp = proj * view
        let invVP = vp.inverse
        let invProj = proj.inverse
        let invView = view.inverse
        lastFrameViewProjection = vp
        var u = IlluminatoramaFrameUniforms(
            viewProjection: vp,
            view: view,
            projection: proj,
            invViewProjection: invVP,
            invProjection: invProj,
            invView: invView,
            cameraWorldPos: camera.position,
            directionalLightDir: simd_normalize(directionalLightDirection),
            directionalLightColor: directionalLightColor,
            ambientColor: ambientColor,
            exposure: easedExposure,
            bloomThreshold: easedBloomThreshold,
            bloomIntensity: easedBloomIntensity,
            pointLightCount: UInt32(pointLights.count),
            time: time,
            ssaoIntensity: ssaoIntensity,
            ssaoRadius: ssaoRadius,
            ssrIntensity: ssrIntensity,
            ssrMaxDistance: ssrMaxDistance,
            ssrThickness: ssrThickness,
            ssrMaxSteps: ssrMaxSteps,
            iblIntensity: iblIntensity,
            iblPrefilteredMipCount: UInt32(prefilteredMipViews.count),
            iblEnabled: iblEnabled ? 1 : 0,
            shadowVP0: cascadeVPs[0],
            shadowVP1: cascadeVPs[1],
            shadowVP2: cascadeVPs[2],
            cascadeSplitsView: cascadeSplitsView,
            shadowBias: shadowBias,
            shadowSlopeBias: shadowSlopeBias,
            shadowEnabled: shadowsEnabled ? 1 : 0,
            shadowPcfRadius: shadowPcfRadius,
            previousViewProjection: previousViewProjection,
            taaHistoryBlend: taaHistoryBlend,
            taaEnabled: taaEnabled ? 1 : 0,
            taaIsFirstFrame: taaNeedsFirstFrame ? 1 : 0,
            dfgLUTEnabled: dfgLUTEnabled ? 1 : 0,
            spotLightCount: UInt32(spotLights.count),
            spotShadowBias: spotShadowBias,
            tonemapSaturation: tonemapSaturation,
            iblDiffuseSaturation: iblDiffuseSaturation,
            autoExposureEnabled: autoExposureEnabled ? 1 : 0,
            autoExposureTargetEV: autoExposureTargetEV,
            autoExposureHalfLife: autoExposureHalfLife,
            debugTerm: debugTerm.rawValue,
            iblDiffuseDesaturation: iblDiffuseDesaturation,
            ssaoDenoiseEnabled: ssaoDenoiseEnabled ? 1 : 0,
            ssaoTemporalBlend: ssaoTemporalBlend,
            ssrDenoiseEnabled: ssrDenoiseEnabled ? 1 : 0,
            ssrTemporalBlend: ssrTemporalBlend,
            ssaoIsFirstFrame: aoNeedsFirstFrame ? 1 : 0,
            ssrIsFirstFrame: ssrNeedsFirstFrame ? 1 : 0,
            debandDitherEnabled: debandDitherEnabled ? 1 : 0,
            ddgiIrrCacheEnabled: (ddgiIrrCacheEnabled && ddgiEnabled) ? 1 : 0,
            ddgiIrrCacheBlend: max(0.0, min(1.0, ddgiIrrCacheBlend)),
            leafTransmission: max(0, leafTransmission),
            areaLightCount: UInt32(areaLights.count),
            areaLTCEnabled: ltcValidated ? 1 : 0,
            directionalLightCount: UInt32(extraDirectionals.count),
            plushSheen: max(0, plushSheen),
            plushTransmission: max(0, plushTransmission)
        )
        // Tree wind (#58 #1): repurpose the two free pad floats as the vertex-
        // shader vegetation-wind knobs. _padPhase2A = strength (max canopy sway,
        // ~m), _padPhase2B = heading (radians). 0 strength → exact no-op (the
        // shader's applyTreeWind early-returns), so every other scene is unaffected.
        u._padPhase2A = treeWindStrength
        u._padPhase2B = treeWindHeading
        // Chromatic aberration: tonemap CA strength (repurposes the former
        // _padPlush0 slot). 0 → exact no-op (the tonemap branch is gated on it).
        // Eased value (see advancePostFXEasing) so slider drags glide.
        u.chromaticAberration = max(0, easedChromaticAberration)
        // Spherical aberration: radial blur, 0 → exact no-op.
        u.sphericalAberration = max(0, easedSphericalAberration)
        // Axial chromatic aberration ("purple fringing"): strength + dark-side
        // tint. 0 strength → exact no-op (the tonemap branch is gated on it).
        u.fringe = max(0, easedFringe)
        u.fringeTintR = easedFringeTint.x
        u.fringeTintG = easedFringeTint.y
        u.fringeTintB = easedFringeTint.z
        // Vignette + film grain (issue #65). Applied directly (no easing) — 0 → no-op.
        u.vignetteStrength  = max(0, vignetteStrength)
        u.vignetteExtent    = max(0, min(1, vignetteExtent))
        u.filmGrainStrength = max(0, filmGrainStrength)
        u.filmGrainSize     = max(1, filmGrainSize)
        // Colour-grade LUT (issue #65). 0 → no-op; size feeds the half-texel inset.
        u.colorLUTAmount    = max(0, min(1, colorLUTAmount))
        u.colorLUTSize      = Float(colorLUTSize)
        memcpy(frameUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaFrameUniforms>.stride)
    }

    /// Halton low-discrepancy sequence, one dimension. Index 0 is excluded
    /// (the algorithm degenerates there) — callers pass `frameIndex + 1`.
    private static func halton(_ index: UInt32, base: UInt32) -> Float {
        var f: Float = 1
        var r: Float = 0
        var i = index
        while i > 0 {
            f /= Float(base)
            r += f * Float(i % base)
            i /= base
        }
        return r
    }

    private func uploadInstances() {
        // After the ping-pong toggle at the top of render(), the buffer that
        // WAS previous last frame is now "current" — we overwrite it with
        // this frame's data. The other buffer (now "previous") still holds
        // last frame's data, which the G-buffer vertex shader reads for
        // motion-vector reconstruction.
        //
        // Phase 4.12 — write in GROUPED order so all instances of the same
        // `MeshKind` land in a contiguous run. That lets every pass that
        // iterates the scene's geometry (G-buffer, cascaded + spot
        // shadows) replace its per-instance loop with one
        // `drawIndexedPrimitives(..., instanceCount: group.count)` per
        // mesh kind. FloatingFlowers+ goes from ~1500 draw calls to ~10
        // without any shader changes — the vertex shader's
        // `instances[iid]` lookup walks the contiguous slice naturally
        // because we set the vertex/fragment buffer offsets per group.
        //
        // Grouping stability across frames: the host (`extractor`) walks
        // the scene graph in DFS order, so the same scene topology
        // produces the same first-seen ordering of mesh kinds + the same
        // intra-group instance order. That's what motion-vector TAA
        // relies on — `prevInstances[iid]` reads the same logical
        // instance as `instances[iid]` because both buffers used the
        // same grouping last → this frame.
        meshGroups.removeAll(keepingCapacity: true)
        guard !instances.isEmpty else { return }

        // First pass — bucket by mesh kind, preserving first-seen order
        // so a stable scene gives a stable buffer layout.
        var byKind: [MeshKind: [Int]] = [:]
        byKind.reserveCapacity(8)
        var kindOrder: [MeshKind] = []
        kindOrder.reserveCapacity(8)
        for (srcIdx, ref) in instances.enumerated() {
            if byKind[ref.meshKind] == nil {
                byKind[ref.meshKind] = []
                kindOrder.append(ref.meshKind)
            }
            byKind[ref.meshKind, default: []].append(srcIdx)
        }

        // Second pass — write into the buffer in grouped order, building
        // the per-group draw recipe alongside.
        let dst = currentInstanceBuffer.contents().bindMemory(
            to: IlluminatoramaInstance.self, capacity: instances.count
        )
        // Parallel superquadric param buffer, written in the SAME grouped order so
        // `params[iid]` lines up with `instances[iid]` in the impostor draw. Most
        // entries are unused (only impostor instances carry a shape); cheap to fill.
        let sqDst = superquadricParamBuffer.contents().bindMemory(
            to: IlluminatoramaInstance.SuperquadricParam.self, capacity: instances.count
        )
        var dstIdx = 0
        for kind in kindOrder {
            guard let srcIndices = byKind[kind] else { continue }
            let groupStart = dstIdx
            for srcIdx in srcIndices {
                let ref = instances[srcIdx]
                dst[dstIdx] = ref.data
                if let s = ref.superquadricShape {
                    let isEllipsoid = (abs(s.x - 1) < 1e-3 && abs(s.y - 1) < 1e-3)
                    sqDst[dstIdx] = IlluminatoramaInstance.SuperquadricParam(
                        invModel: simd_inverse(ref.data.modelMatrix),
                        shape: SIMD4(s.x, s.y, isEllipsoid ? 1 : 0, 0))
                } else {
                    sqDst[dstIdx] = IlluminatoramaInstance.SuperquadricParam(
                        invModel: matrix_identity_float4x4, shape: .zero)
                }
                dstIdx += 1
            }
            meshGroups.append(MeshDrawGroup(
                kind: kind, start: groupStart, count: srcIndices.count))
        }
    }

    private func uploadPointLights() {
        guard !pointLights.isEmpty else { return }
        pointLights.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                pointLightBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaPointLight>.stride * pointLights.count
            )
        }
    }

    private func uploadSpotLights() {
        guard !spotLights.isEmpty else { return }
        spotLights.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                spotLightBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaSpotLight>.stride * spotLights.count
            )
        }
    }

    private func uploadAreaLights() {
        guard !areaLights.isEmpty else { return }
        areaLights.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                areaLightBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaAreaLight>.stride * areaLights.count
            )
        }
    }

    private func uploadExtraDirectionals() {
        guard !extraDirectionals.isEmpty else { return }
        extraDirectionals.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                extraDirectionalBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaDirectionalLight>.stride * extraDirectionals.count
            )
        }
    }

    // ── Target allocation ─────────────────────────────────────────────────────

    private struct Targets {
        var albedoMet: MTLTexture
        var normalRgh: MTLTexture
        var emission: MTLTexture
        var depth: MTLTexture
        var hdr: MTLTexture
        var ao: MTLTexture
        var hdrComposite: MTLTexture
        var rtDiffuse: MTLTexture    // full-res RT diffuse (shadow+GI) pre-denoise
        // Phase 2.7
        var velocity: MTLTexture
        var historyA: MTLTexture
        var historyB: MTLTexture
        var bloomBright: MTLTexture
        var bloomBlurH: MTLTexture
        var bloomBlurV: MTLTexture
        var ldr: MTLTexture
        // Phase 4.39 denoiser
        var aoFiltered: MTLTexture   // half-res, after bilateral
        var aoHistoryA: MTLTexture   // half-res r16Float ping-pong
        var aoHistoryB: MTLTexture
        var ssrRaw: MTLTexture       // full-res SSR gather delta
        var ssrHistoryA: MTLTexture  // full-res rgba16Float ping-pong
        var ssrHistoryB: MTLTexture
        var rtGIHistoryA: MTLTexture // full-res rgba16Float ping-pong (RT GI temporal)
        var rtGIHistoryB: MTLTexture
        var irrCacheA: MTLTexture    // Phase 3.4 — DDGI irradiance EMA cache (full-res rgba16Float)
        var irrCacheB: MTLTexture
        // SVGF denoiser (Phase 4.44)
        var aoSampleCount: MTLTexture    // half-res r16Float — accumulated frame count per SSAO pixel
        var ssrSampleCount: MTLTexture   // full-res r16Float — accumulated frame count per SSR pixel
        var giSampleCount: MTLTexture    // full-res r16Float — accumulated frame count per RT GI pixel
        var giVariance: MTLTexture       // full-res r16Float — spatial variance of accumulated GI
        var giAtrousA: MTLTexture        // full-res rgba16Float — à-trous ping-pong A
        var giAtrousB: MTLTexture        // full-res rgba16Float — à-trous ping-pong B
    }

    // ── Phase 2.5 cascade constants ───────────────────────────────────────────
    // Three cascades is the sweet spot — four is more "correct" but the extra
    // pass cost (one render pass per cascade) isn't visible at typical scene
    // scale. 2048² per cascade matches what UE / Frostbite use for their inner
    // cascade and gives clean PCF at 3×3.
    private static let cascadeCount: Int = 3
    private static let shadowMapResolution: Int = 2048

    // ── Phase 3 IBL cube resolutions ──────────────────────────────────────────
    // 16² per face → tiny, since diffuse irradiance is a very low-frequency
    // signal. 128² per face for prefiltered specular with 6 mip levels (128,
    // 64, 32, 16, 8, 4) covering roughness 0 → 1 in equal steps.
    private static let irradianceFaceSize: Int = 16
    private static let prefilteredFaceSize: Int = 128
    private static let prefilteredMipCount: Int = 6
    // Phase 3.2 — 128² is standard; the signal is smooth so larger is wasteful.
    private static let dfgLUTSize: Int = 128
    // Issue #65 — colour-grade LUT default per-axis resolution. 33 is the .cube
    // de-facto standard (matches DaVinci / most authoring tools).
    static let colorLUTDefaultSize: Int = 33
    // Phase 3.1 — DDGI tile sizes. irrTileSize=6 is the standard minimum for
    // diffuse irradiance (the signal is low-frequency). depthTileSize=14 is
    // larger to reduce mean-distance bias from undersampled directions.
    private static let ddgiIrrTileSize: Int = 6
    private static let ddgiDepthTileSize: Int = 14

    /// Mirror of `PrefilterBakeParams` in `Illuminatorama.metal`.
    private struct PrefilterBakeParams {
        var roughness: Float
        var faceWidth: UInt32
        var sampleCount: UInt32
        var bakeDesat: Float    // was _pad — desaturate equirect samples (0=off, 1=grey)
    }

    private static func makeIBLCubes(
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

    /// Standalone sibling of the G-buffer depth, sized to match it, used as the
    /// PREVIOUS frame's depth in the TAA resolve's disocclusion test. Not part of
    /// the texture pool — it's blit-filled from `depthTexture` at the end of each
    /// frame. `.shaderRead` so the resolve kernel can `read()` it; `.renderTarget`
    /// only to mirror the source depth's descriptor.
    private static func makeDepthLike(device: MTLDevice, _ tex: MTLTexture) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: tex.width, height: tex.height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.depth.previous")
        }
        t.label = "Illuminatorama.depth.previous"
        return t
    }

    /// Allocate the cascaded shadow depth array. depth32Float gives us plenty
    /// of precision for the comparison test; `.shaderRead` is required so the
    /// lighting kernel can sample it, and `.renderTarget` lets us write into
    /// each slice via a render pass.
    private static func makeShadowMap(device: MTLDevice) throws -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .depth32Float
        d.width = shadowMapResolution
        d.height = shadowMapResolution
        d.arrayLength = cascadeCount
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.shadowMap")
        }
        t.label = "Illuminatorama.shadowMap"
        return t
    }

    /// Phase 4.10 — depth atlas for spot light shadows. Same format /
    /// usage as the cascaded shadow map; smaller per-slice resolution
    /// and N slices (one per shadowed spot).
    private static func makeSpotShadowAtlas(device: MTLDevice,
                                             resolution: Int,
                                             capacity: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .depth32Float
        d.width = resolution
        d.height = resolution
        d.arrayLength = capacity
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.spotShadowAtlas")
        }
        t.label = "Illuminatorama.spotShadowAtlas"
        return t
    }

    // ── Cascade math helpers ──────────────────────────────────────────────────

    /// Right-handed look-at, depth conventions matching the rest of
    /// IlluminatoramaCamera's matrices.
    private static func lookAtRH(eye: SIMD3<Float>,
                                 target: SIMD3<Float>,
                                 up: SIMD3<Float>) -> simd_float4x4 {
        let f = simd_normalize(target - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return simd_float4x4(
            SIMD4( s.x,  u.x, -f.x, 0),
            SIMD4( s.y,  u.y, -f.y, 0),
            SIMD4( s.z,  u.z, -f.z, 0),
            SIMD4(-simd_dot(s, eye),
                  -simd_dot(u, eye),
                   simd_dot(f, eye), 1)
        )
    }

    /// Right-handed orthographic projection with depth range [0, 1] —
    /// matches `IlluminatoramaCamera.projectionMatrix`'s convention.
    private static func orthoRH(left l: Float, right r: Float,
                                bottom b: Float, top t: Float,
                                near n: Float, far f: Float) -> simd_float4x4 {
        let zRange = f - n
        return simd_float4x4(
            SIMD4(2 / (r - l), 0, 0, 0),
            SIMD4(0, 2 / (t - b), 0, 0),
            SIMD4(0, 0, -1 / zRange, 0),
            SIMD4(-(r + l) / (r - l),
                  -(t + b) / (t - b),
                  -n / zRange, 1)
        )
    }

    /// Right-handed perspective projection with depth range [0, 1] —
    /// matches `IlluminatoramaCamera.projectionMatrix`'s convention. Used
    /// by `updateSpotShadows` to build per-spot light-space VPs.
    private static func perspectiveRH(fovY: Float, aspect: Float,
                                       near: Float, far: Float) -> simd_float4x4 {
        let yScale = 1 / tan(fovY * 0.5)
        let xScale = yScale / aspect
        let zRange = far - near
        return simd_float4x4(
            SIMD4(xScale, 0, 0, 0),
            SIMD4(0, yScale, 0, 0),
            SIMD4(0, 0, -far / zRange, -1),
            SIMD4(0, 0, -(far * near) / zRange, 0)
        )
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
    private static func makeDummySky(device: MTLDevice) -> MTLTexture {
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

    private static func makeTargets(
        device: MTLDevice,
        internalW: Int, internalH: Int,
        outputW: Int, outputH: Int
    ) throws -> Targets {
        func make(label: String, format: MTLPixelFormat, w: Int, h: Int,
                  usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = format
            d.width = w
            d.height = h
            d.usage = usage
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else {
                throw IlluminatoramaError.bufferAllocationFailed(label)
            }
            t.label = label
            return t
        }
        // All internal-pipeline targets are sized to internalW × internalH —
        // these are the textures every compute kernel except the tonemap
        // reads from and writes into. The tonemap's `outputTexture` is the
        // only thing at the smaller output size.
        let am = try make(label: "Illuminatorama.gbuffer.albedoMet",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let nr = try make(label: "Illuminatorama.gbuffer.normalRgh",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let em = try make(label: "Illuminatorama.gbuffer.emission",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let depth = try make(label: "Illuminatorama.depth",
                             format: .depth32Float, w: internalW, h: internalH,
                             usage: [.renderTarget, .shaderRead])
        // HDR composite and history need .sample because the tonemap kernel
        // bilinearly downsamples them into the output texture.
        let hdr = try make(label: "Illuminatorama.hdr",
                           format: .rgba16Float, w: internalW, h: internalH,
                           usage: [.shaderRead, .shaderWrite, .renderTarget])
        // Phase 4.11 — `.renderTarget` enables the additive forward
        // particles pass to draw directly into the composite. Compute
        // passes elsewhere still bind it via shaderRead/Write.
        let hdrComposite = try make(label: "Illuminatorama.hdrComposite",
                                    format: .rgba16Float, w: internalW, h: internalH,
                                    usage: [.renderTarget, .shaderRead, .shaderWrite])
        // RT diffuse (soft shadow + 1-bounce GI) buffer, full-res. The RT pass
        // writes here; `illumi_rt_denoise` bilateral-filters it into the composite.
        let rtDiffuse = try make(label: "Illuminatorama.rt.diffuse",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let velocity = try make(label: "Illuminatorama.velocity",
                                format: .rg16Float, w: internalW, h: internalH,
                                usage: [.renderTarget, .shaderRead])
        let historyA = try make(label: "Illuminatorama.historyA",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let historyB = try make(label: "Illuminatorama.historyB",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        // AO and bloom run at half of the INTERNAL resolution (still oversampled
        // vs the output when SSAA is on).
        let halfW = max(1, internalW / 2)
        let halfH = max(1, internalH / 2)
        let ao = try make(label: "Illuminatorama.ao",
                          format: .r16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        // Phase 4.39 — SSAO denoiser intermediates.
        let aoFiltered = try make(label: "Illuminatorama.ao.filtered",
                                  format: .r16Float, w: halfW, h: halfH,
                                  usage: [.shaderRead, .shaderWrite])
        let aoHistA = try make(label: "Illuminatorama.ao.historyA",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        let aoHistB = try make(label: "Illuminatorama.ao.historyB",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        // Phase 4.39 — SSR denoiser intermediates.
        let ssrRaw = try make(label: "Illuminatorama.ssr.raw",
                              format: .rgba16Float, w: internalW, h: internalH,
                              usage: [.shaderRead, .shaderWrite])
        let ssrHistA = try make(label: "Illuminatorama.ssr.historyA",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let ssrHistB = try make(label: "Illuminatorama.ssr.historyB",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        // RT-GI temporal accumulation history (ping-pong, full-res).
        let rtGiHistA = try make(label: "Illuminatorama.rt.giHistoryA",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let rtGiHistB = try make(label: "Illuminatorama.rt.giHistoryB",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        // Phase 4.44 — SVGF denoiser: adaptive sample counts + à-trous buffers.
        let aoCount = try make(label: "Illuminatorama.svgf.aoCount",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        let ssrCount = try make(label: "Illuminatorama.svgf.ssrCount",
                                format: .r16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let giCount = try make(label: "Illuminatorama.svgf.giCount",
                               format: .r16Float, w: internalW, h: internalH,
                               usage: [.shaderRead, .shaderWrite])
        let giVar = try make(label: "Illuminatorama.svgf.giVariance",
                             format: .r16Float, w: internalW, h: internalH,
                             usage: [.shaderRead, .shaderWrite])
        let giAtrousA = try make(label: "Illuminatorama.svgf.giAtrousA",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let giAtrousB = try make(label: "Illuminatorama.svgf.giAtrousB",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        // Phase 3.4 — DDGI irradiance EMA cache (full-res, ping-pong).
        let irrCA = try make(label: "Illuminatorama.ddgi.irrCacheA",
                             format: .rgba16Float, w: internalW, h: internalH,
                             usage: [.shaderRead, .shaderWrite, .renderTarget])
        let irrCB = try make(label: "Illuminatorama.ddgi.irrCacheB",
                             format: .rgba16Float, w: internalW, h: internalH,
                             usage: [.shaderRead, .shaderWrite, .renderTarget])
        let bb = try make(label: "Illuminatorama.bloom.bright",
                          format: .rgba16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        let bh = try make(label: "Illuminatorama.bloom.blurH",
                          format: .rgba16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        let bv = try make(label: "Illuminatorama.bloom.blurV",
                          format: .rgba16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        // Final LDR output at OUTPUT resolution — the only texture sized to
        // what the SCNView shows. Tonemap kernel downsamples from internal
        // HDR + bloom into this each frame.
        let ldrDesc = MTLTextureDescriptor()
        ldrDesc.textureType = .type2D
        // sRGB so the tonemap can write linear + the GPU encodes on store; see
        // `makeLDRTexture`. sRGB targets disallow `.shaderWrite` (render-pass only).
        ldrDesc.pixelFormat = .bgra8Unorm_srgb
        ldrDesc.width = outputW
        ldrDesc.height = outputH
        ldrDesc.usage = [.shaderRead, .renderTarget]
        ldrDesc.storageMode = .private
        guard let ldr = device.makeTexture(descriptor: ldrDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("ldr")
        }
        ldr.label = "Illuminatorama.ldr"
        return Targets(
            albedoMet: am, normalRgh: nr, emission: em, depth: depth,
            hdr: hdr, ao: ao, hdrComposite: hdrComposite, rtDiffuse: rtDiffuse,
            velocity: velocity, historyA: historyA, historyB: historyB,
            bloomBright: bb, bloomBlurH: bh, bloomBlurV: bv,
            ldr: ldr,
            aoFiltered: aoFiltered, aoHistoryA: aoHistA, aoHistoryB: aoHistB,
            ssrRaw: ssrRaw, ssrHistoryA: ssrHistA, ssrHistoryB: ssrHistB,
            rtGIHistoryA: rtGiHistA, rtGIHistoryB: rtGiHistB,
            irrCacheA: irrCA, irrCacheB: irrCB,
            aoSampleCount: aoCount, ssrSampleCount: ssrCount, giSampleCount: giCount,
            giVariance: giVar, giAtrousA: giAtrousA, giAtrousB: giAtrousB
        )
    }
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
private final class IlluminatoramaGPUMeter: @unchecked Sendable {
    private let lock = NSLock()
    private let path: String?
    private var sumMs: Double = 0
    private var count: Int = 0
    // Per-frame samples for the percentile distribution. On a shared / contended
    // machine the MEAN is dominated by occasional contention spikes (WindowServer,
    // screen-share, a sibling render) and reads ~the same at every SSAA scale, so
    // it can't show whether a render-cost change actually moved the STEADY frame.
    // The p50 rejects that tail and isolates the clean steady-state GPU frame —
    // the number that decides whether vsync can hold 60 (is it < 16.6 ms?). p95/p99
    // describe the drop-frame tail. Ring-bounded so a long run can't grow unbounded.
    private var samples: [Double] = []
    private static let maxSamples = 8192
    // Last-computed percentiles, refreshed every 30 frames and ALWAYS emitted, so
    // the sidecar file holds the full distribution no matter which frame the
    // headless snapshot reads it on (the recompute is throttled, the write is not).
    private var p50: Double = 0, p95: Double = 0, p99: Double = 0
    private var minMs: Double = 0, maxMs: Double = 0
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

private final class IlluminatoramaPresentSync: @unchecked Sendable {
    private let lock = NSLock()
    private var completedIdx: Int = -1
    /// Pool slots whose render is queued/running on the GPU but not yet complete.
    /// Inserted on the main thread (after commit) and removed on the GPU
    /// completion thread, so `acquireWriteTarget` never picks a buffer a second
    /// pipelined frame is still writing (`maxFramesInFlight = 2`).
    private var inFlight: Set<Int> = []

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
    private var gpuMsEMA: Double = 0
    /// Permits currently held back (0 ⇒ depth 2, 1 ⇒ depth 1). Never exceeds
    /// `maxFramesInFlight - 1`, so at least one permit always circulates (no deadlock).
    private var heldPermits: Int = 0
    /// Current effective depth target, with hysteresis so it doesn't flap.
    private var depth: Int = IlluminatoramaRenderer.maxFramesInFlight
    /// Hysteresis band (~2 ticks): pipeline only while the GPU frame is light.
    private static let pipelineEnterMs = 30.0   // drop to 1→2 when EMA below this
    private static let pipelineExitMs  = 36.0   // rise to 2→1 when EMA above this

    /// Called from the GPU completion handler in place of `semaphore.signal()`.
    /// Updates the GPU-time EMA, recomputes the target depth, and either releases
    /// this frame's permit normally or HOLDS it to converge the effective in-flight
    /// capacity to that depth. Deadlock-free: `heldPermits` is capped at
    /// `maxFramesInFlight - 1`, so a permit always remains in circulation.
    func frameCompletedAdaptive(gpuMs: Double, semaphore: DispatchSemaphore) {
        lock.lock()
        if gpuMs > 0 {
            gpuMsEMA = gpuMsEMA == 0 ? gpuMs : gpuMsEMA * 0.9 + gpuMs * 0.1
        }
        if depth >= IlluminatoramaRenderer.maxFramesInFlight {
            if gpuMsEMA > Self.pipelineExitMs { depth = 1 }
        } else {
            if gpuMsEMA < Self.pipelineEnterMs { depth = IlluminatoramaRenderer.maxFramesInFlight }
        }
        let targetHeld = max(0, IlluminatoramaRenderer.maxFramesInFlight - depth)
        if heldPermits < targetHeld {
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
    private var slotInFlight = [Bool](repeating: false, count: IlluminatoramaRenderer.maxFramesInFlight + 1)
    private var raceEvents = 0
    private var probeFrames = 0
    private let racePath = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PATH"]

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

    /// Record (from the GPU completion thread) that pool buffer `idx` is done.
    func markCompleted(_ idx: Int) {
        lock.lock()
        completedIdx = idx
        inFlight.remove(idx)
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
