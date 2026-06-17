import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── VOLUMETRIC CLOUD RENDERER ────────────────────────────────────────────────
//
// Renders a Rayleigh-ish atmosphere + a volumetric cumulus deck into TWO
// equirect HDR `MTLTexture`s that SceneKit consumes separately:
//
//   • `lightingEnvironmentTexture` (default 256×128) ← bind to
//     `scene.lightingEnvironment.contents`. Tiny, ~1 ms to render, dispatched
//     EVERY `render(params:)` call. PBR lighting tracks the sun smoothly
//     even when the dome is throttled, so cinematic time-of-day scenes
//     (UFO dawn/dusk, NapaValley balloon traversal) don't feel sluggish.
//
//   • `outputTexture` (default 2048×1024, opt-in 4K) ← visible sky dome.
//     Wrap it via `makeSkyDome(followingCamera:)`. Expensive (~30–50 ms
//     with the baked noise volume), throttled by `minRenderInterval`. Do
//     NOT bind it to `scene.background.contents`: that path flat-maps the
//     equirect across the camera viewport so yawing the camera drags the
//     sky with it. The dome wraps the texture onto an inside-facing
//     SCNSphere whose position follows the camera, giving the equirect a
//     true panoramic projection.
//
// Two compute dispatches per `render(params:)`:
//   1. IBL pass into `lightingEnvironmentTexture` — every call, gated by a
//      separate 2-slot semaphore (back-pressure safety, never expected to
//      trip in practice — kernel is ~1 ms here).
//   2. Dome pass into `outputTexture` — gated by `minRenderInterval` and
//      a 2-slot in-flight semaphore. This is the one that protects against
//      the Metal command-queue saturation freeze documented in commit
//      history; it stays in even with the baked-noise speedup.
//
// Both dispatches use the same `volSkyRender` kernel + same uniforms; only
// the destination texture differs. The kernel reads the baked 3D noise
// volume (see `NoiseVolume`) bound at `[[texture(1)]]` — one trilinear
// sample per march step replaces ~60 hash ops, the dominant speedup over
// the inline-noise V1.
//
// USAGE
// ─────
//
//     let cloud = VolumetricCloudRenderer(engine: .shared)
//     scene.lightingEnvironment.contents = cloud.lightingEnvironmentTexture
//     scene.rootNode.addChildNode(cloud.makeSkyDome(followingCamera: cameraNode))
//     // every tick:
//     cloud.render(params: .init(time: now, coverage: 0.55, ...))
//
// Textures are reference-stable across calls — bind once.

@MainActor
public final class VolumetricCloudRenderer {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "VolumetricCloudRenderer")

    // ── Public API ──────────────────────────────────────────────────

    /// High-resolution equirect HDR texture used by the visible sky dome
    /// (see `makeSkyDome(followingCamera:)`). Updated on the
    /// `minRenderInterval` throttle — this is the expensive dispatch.
    /// Reference-stable across `render(params:)` calls — bind once.
    public private(set) var outputTexture: MTLTexture

    /// Low-resolution equirect HDR texture for SceneKit IBL. Bind to
    /// `scene.lightingEnvironment.contents`. Updated EVERY
    /// `render(params:)` call (kernel is ~1 ms at the default 256×128),
    /// so PBR lighting tracks the sun smoothly regardless of the dome
    /// throttle. Reference-stable — bind once.
    public private(set) var lightingEnvironmentTexture: MTLTexture

    public let device: MTLDevice
    public let resolution: SIMD2<Int>
    public let iblResolution: SIMD2<Int>

    // ── Burst lights (fireworks in the clouds) ──────────────────────
    //
    // Optional point-light list marched as single-scattered in-scatter inside
    // the cloud volume — a burst below the deck glows the cloud base from
    // underneath; a light inside the slab illuminates the interior around
    // itself. The buffer holds `BurstLight` entries (the shared 48-byte
    // struct from BurstLightField); hosts typically pass either
    // `BurstLightField.buffer.buffer` directly or a per-tick packed copy
    // (bursts + rising shell heads). When `burstLights` is nil or count is
    // 0 the kernel skips the loop — zero cost for non-fireworks scenes.

    /// Light list consumed by the next `render(params:)`. Layout-locked to
    /// the shared `BurstLight` struct. Reference is held, not copied — the
    /// host may rewrite contents between renders.
    public var burstLights: MTLBuffer?
    /// Number of live entries at the head of `burstLights`.
    public var burstLightCount: Int = 0
    /// Scalar on the in-scatter term so hosts balance the fireworks glow
    /// against sun/ambient. ~0.2–0.5 for typical burst intensities (≈8).
    public var burstLightGain: Float = 0.3

    /// Renderer parameters. Marked `Sendable` so callers can compute it off
    /// the main actor and pass it in; the renderer itself is `@MainActor`.
    public struct Params: Sendable {
        // ── Camera + time ─────────────────────────────────────
        public var time: Float = 0
        public var cameraPos: SIMD3<Float> = .zero

        // ── Sun ───────────────────────────────────────────────
        /// Direction sunlight TRAVELS (i.e. *from* the sun *to* the scene).
        /// Matches SceneKit's directional-light convention. Will be normalised.
        /// Default corresponds to a sun at azimuth 0.82 rad, elevation
        /// 0.72 rad — upper-right of the frame at typical Flying Cats
        /// FOVs, lining up with the SCN directional fill.
        public var sunDir: SIMD3<Float> = normalize(SIMD3<Float>(-0.513, -0.659, -0.550))
        /// HDR colour of direct sunlight. >1 components are fine — the kernel
        /// applies a soft Reinhard before writing.
        public var sunColor: SIMD3<Float> = SIMD3<Float>(1.30, 1.20, 1.00)
        /// Sun disk intensity scalar (relative to `sunColor`). 1 = normal,
        /// 5+ = bright HDR disk that drives SceneKit's exposure adaptation.
        /// 10.3 = 6 × 1.72 (Cloud Test's "1.72" slider × the 6× host scale).
        public var sunIntensity: Float = 10.3

        // ── Sky gradient ──────────────────────────────────────
        public var skyZenith:   SIMD3<Float> = SIMD3<Float>(0.18, 0.36, 0.78)
        public var skyHorizon:  SIMD3<Float> = SIMD3<Float>(0.74, 0.82, 0.95)
        public var groundColor: SIMD3<Float> = SIMD3<Float>(0.22, 0.20, 0.18)
        /// Power applied to the elevation curve. 1 = linear; >1 = horizon
        /// band sits lower (sky stays deep blue across more of the dome).
        public var hazePower: Float = 1.6
        /// How fast rays below the horizon fade to ground colour. 0..1; 0.5
        /// is a soft transition, 1 is hard.
        public var groundBlend: Float = 1.0

        // ── Cloud slab + shape ────────────────────────────────
        // Defaults are tuned for "scene-scale" Visualizer scenes where the
        // camera sits ~5 units up and the visible world extends ~30–50
        // units in front of it (Flying Cats, etc.). Real-world cloud
        // altitudes (cumulus base 600–1500 m) would put the deck so far
        // overhead that the slab would barely move under any noise period
        // the kernel could sample — the deck would read as a featureless
        // grey wash. Treat these as artistic numbers, not metres.
        // These defaults are the values we settled on in Cloud Test as the
        // baseline "good cumulus" recipe — a thin slab at altitude 28..37,
        // mid-coverage with low erosion and a strong anvil weight, lit
        // with a bright sun + full powder. Going further than ±10–20% on
        // any single one should still read as recognisable cumulus; if a
        // scene wants something different (storm clouds, overcast, etc.)
        // it should pass overrides rather than redefine the baseline here.
        /// Altitude (world Y) the cloud bases sit at.
        public var cloudBaseY: Float = 28
        /// Altitude (world Y) the cloud tops reach.
        public var cloudTopY: Float = 37
        /// Multiplier on world XZ when looking up cloud noise. Larger =
        /// smaller-looking clouds. 0.039 puts roughly four-to-six discrete
        /// cumulus mounds across the visible sky at Flying Cats's camera
        /// scale (cats ~1–4 units, camera ~22 units back).
        public var horizontalScale: Float = 0.039
        /// 0 = no anvil shoulder, 1 = pronounced anvil-shelf cumulonimbus.
        /// 0.94 sounds like cumulonimbus but combined with the thin slab
        /// (9-unit-thick band) actually just gives cumulus a flat, wide
        /// crown — the anvil shoulder doesn't have headroom to flare.
        public var anvil: Float = 0.94

        // ── Cloud field ──────────────────────────────────────
        /// Sky fraction covered. 0 = clear, 1 = overcast.
        public var coverage: Float = 0.51
        /// Mass multiplier on raw FBM density. 1 is normal; 2 = pillowy
        /// dense cumulus, 0.5 = thin haze.
        public var density: Float = 1.0
        /// Worley erosion weight (0..1). Higher chews more gaps into the
        /// dense FBM body — broken cumulus. Low (0.23) keeps each cumulus
        /// mound coherent rather than shredding it into wispy fragments,
        /// which was the "Spanish surrealist blobs" failure mode.
        public var erosion: Float = 0.23
        /// Sky-light fill on the unlit side. 0..1; 0.95 reads as a bright
        /// daytime cumulus where even the shadow side is well-lit by sky
        /// bounce. Drop to ~0.55 for moody / pre-storm clouds.
        public var ambient: Float = 0.95

        // ── Wind ─────────────────────────────────────────────
        /// XZ unit direction the wind blows from -> to. Will be normalised.
        public var windDirection: SIMD2<Float> = SIMD2<Float>(cos(0.2), sin(0.2))
        public var windSpeed: Float = 0.7

        // ── Lighting tuning ──────────────────────────────────
        /// HG forward-scatter lobe. ~0.44 gives a believable silver-lining
        /// brightness around the sun without the inner cloud going opaque.
        public var phaseForward: Float = 0.44
        /// HG back-scatter lobe (negative). Near-zero (-0.05) keeps the
        /// dark side of the cloud from being too dim relative to ambient.
        public var phaseBack: Float = -0.05
        /// Light absorption coefficient. 1..6. Bigger = darker interiors.
        public var absorption: Float = 2.0
        /// Powder term — darkens the lit side of dense regions so the
        /// silhouette pops. Full powder (1.0) crisps the edges nicely
        /// at low coverage; back off toward 0.4 for overcast skies where
        /// you don't want the dense-side dimming.
        public var powder: Float = 1.0

        // ── March budgets ────────────────────────────────────
        public var cloudSteps: Int = 64
        /// 5 light samples (vs. earlier 4) tightens self-shadow tone at
        /// the silhouette edge without measurable cost.
        public var lightSteps: Int = 5

        // ── Night sky ─────────────────────────────────────────────────
        /// Unit vector pointing FROM the scene TOWARD the moon. Ignored when
        /// `moonIntensity == 0`. Default is an arbitrary position; callers
        /// should set this from a time-of-day orbit (e.g. sun's antipodal
        /// position offset by the moon's current orbital phase).
        public var moonDir: SIMD3<Float> = SIMD3<Float>(0.4, 0.6, -0.7)
        /// Moon phase: 0 = new moon (dark side facing us), 0.5 = half,
        /// 1 = full moon. Drives the terminator shading on the disk.
        public var moonPhase: Float = 0.75
        /// Moon disk brightness. 0 = no moon rendered (daytime default).
        /// 0.8–1.2 = clearly visible moon. The kernel multiplies this by
        /// `nightBlend`, so it still fades out if the sun is above the horizon.
        public var moonIntensity: Float = 0.0
        /// Star-field brightness ceiling. 0 = off (daytime default).
        /// 1 = clear dark-sky star field. The kernel multiplies by `nightBlend`
        /// so stars always fade in as the sun sets regardless of this value.
        public var starBrightness: Float = 0.0

        // ── Atmosphere model ───────────────────────────────────────────
        /// Which atmosphere to bake under the clouds.
        ///
        /// `.nishita` (DEFAULT as of 2026-06-12) is a physically-based single-scattering
        /// atmosphere (Nishita 1993) marched in the kernel: correct λ⁻⁴
        /// horizon reddening, real twilight as the sun crosses the horizon,
        /// and consistent aerial perspective, all derived from the sun
        /// position rather than dialed in per scene. `skyZenith`/`skyHorizon`
        /// are ignored in this mode (but `skyZenith` still feeds cloud
        /// ambient, and `groundColor` still fills the lower hemisphere).
        /// Known gap: twilight zenith skews slightly green (no ozone term).
        ///
        /// `.proceduralGradient` is the original art-directed three-band
        /// gradient driven by `skyZenith` / `skyHorizon` / `hazePower` —
        /// cheap, fully tunable, but a fake. Scenes whose look was tuned
        /// around the gradient can opt back in explicitly.
        public enum AtmosphereModel: Int, Sendable {
            case proceduralGradient = 0
            case nishita = 1
        }
        public var atmosphere: AtmosphereModel = .nishita

        // ── Night directional moonlight (cloud form) ───────────────────
        /// Strength of the moon as the cloud deck's DIRECTIONAL light. 0 =
        /// clouds lit only by flat sky-ambient (no lit/shadow face → reads as
        /// a flat blob); ~0.5–1.5 gives a real lit near-face / shadowed
        /// far-face so a night deck reads as cumulus. Uses `moonDir`; cheap
        /// when `sunColor` is zero (the sun light-march is skipped). Tinted
        /// cool inside the kernel.
        public var moonLightGain: Float = 0.0

        // ── Density-contrast curve ─────────────────────────────────────
        /// γ applied to post-threshold cloud density (`d → pow(d, γ)`). 1 =
        /// legacy linear ramp — a wide mid-gray density that integrates to a
        /// fog/gradient wash. >1 (try 2–4) crushes the fringe toward sky while
        /// leaving cores near 1, the "dense turret OR empty gap" distribution
        /// that gives cumulus its broken silhouette and a surface for light to
        /// terminate on.
        public var densityContrast: Float = 1.0

        // ── Debug ──────────────────────────────────────────────────────
        /// Paint a tiled NEON background behind the clouds instead of the sky,
        /// so the deck's silhouette + density read directly (sky gaps glow
        /// neon, cloud bodies occlude it). For dialing in cloud shape in
        /// isolation — never ship enabled.
        public var debugNeonBackground: Bool = false

        // ── Weather map (where clusters form) ──────────────────────────
        /// Frequency multiplier on the large-scale weather field that decides
        /// WHERE cloud clusters form vs clear sky. The legacy value (used when
        /// this is left at its 0.125 default) suits far-away decks; a close,
        /// small-scale deck (clouds a few tens of metres away) needs a much
        /// higher value (try 3–10) or the weather field is constant across the
        /// scene → one continuous bed instead of scattered clouds.
        public var weatherScale: Float = 0.125
        /// Extra contrast on the weather field (0..1) so cloud clusters have
        /// crisp edges with clear sky between, rather than a soft everywhere
        /// haze. 0 = legacy linear.
        public var weatherSharpness: Float = 0.0

        // ── Curl-noise domain warp ──────────────────────────────────────
        /// Amplitude (WORLD units) of a divergence-free curl-noise warp applied
        /// to the cloud-field sample position before the body/erosion lookup.
        /// Shears straight FBM mounds into the wispy, torn-edged silhouette real
        /// cumulus has (issue #18's "curl distortion"). 0 = off (field is
        /// byte-identical to the pre-curl path — the default for every existing
        /// scene). Try a few world units; too high smears the deck into streaks.
        public var curlStrength: Float = 0.0
        /// Frequency of the curl flow field (cycles per world unit). 0 derives
        /// a scale from `horizontalScale` (half the body frequency — a swirl
        /// larger than the cumulus detail it distorts). Higher = tighter curls.
        public var curlScale: Float = 0.0

        // ── Two-scale body (large mound shape + fine detail) ────────────
        /// Weight (0..1) of a second, lower-frequency body noise sample blended
        /// over the detail sample. The baked volume only resolves ~1/hScl m per
        /// texel, so a single detail-frequency sample reads as high-frequency
        /// SPECKLE from above; mixing in a low-frequency sample restores the
        /// large coherent cauliflower MOUND shape (issue #18 "low-freq shape").
        /// 0 = single-scale body (byte-identical to the legacy path). ~0.5–0.6
        /// reads as big turrets carrying fine erosion.
        public var moundMix: Float = 0.0
        /// Frequency ratio of the low-frequency mound sample relative to the
        /// detail sample (`< 1` = larger features). 0 → 0.35.
        public var moundRatio: Float = 0.0

        // ── Multiple scattering ─────────────────────────────────────────
        /// Strength of the multi-octave multiple-scattering approximation on the
        /// directional + burst lighting. 0 = single-scatter only (byte-identical
        /// to the legacy path — the default for every scene). >0 (try ~1) makes
        /// thick banks self-glow from a bright internal/behind source (moonlit
        /// crowns bleeding into the bank, a burst lighting a bank's interior)
        /// instead of reading as flat-lit shells.
        public var multiScatter: Float = 0.0

        // ── Camera-distance fade ───────────────────────────────────────
        /// Clouds within this many metres of the camera fade out (clears the
        /// foreground clutter an infinite slab puts right overhead). 0 = off.
        public var fadeNear: Float = 0.0
        /// Clouds beyond this many metres fade out (kills the horizon pile-up
        /// a flat infinite slab makes at grazing camera angles, leaving only
        /// the mid-distance distinct clouds). 0 = off.
        public var fadeFar: Float = 0.0

        // ── Cloud field (world-anchored finite region) ─────────────────
        /// World XZ centre of a finite cloud region. With `fieldOuter > 0` the
        /// otherwise-infinite slab is bounded to a disc around this point, so
        /// the scattered deck reads the SAME from every camera (unlike the
        /// camera-distance fade, which thins the deck from far cameras).
        public var fieldCenter: SIMD2<Float> = .zero
        /// Inner radius (full density) of the cloud region.
        public var fieldInner: Float = 0.0
        /// Outer radius (density faded to 0) of the cloud region. 0 = off
        /// (infinite slab).
        public var fieldOuter: Float = 0.0
        /// Gain on the Nishita scattering integral. Tuned so a clear midday
        /// sky lands in roughly the same display range as the procedural
        /// gradient before the kernel's tonemap. Ignored for
        /// `.proceduralGradient`. Raise for a punchier sky, lower if the
        /// zenith clips.
        public var atmosphereIntensity: Float = 20.0

        public init() {}
    }

    /// Build a renderer.
    ///
    /// `resolution` is the visible dome equirect (2048×1024 default). Pass
    /// `SIMD2<Int>(4096, 2048)` explicitly for **night scenes** where
    /// sub-pixel point features (stars, distant lights) would otherwise
    /// smear into blobs — see the equirect-stars memory note. UFO and
    /// NapaValley do this.
    ///
    /// `iblResolution` is the small equirect bound to
    /// `scene.lightingEnvironment.contents`. 256×128 is plenty — SceneKit
    /// convolves this internally for diffuse / specular IBL, and visually
    /// indistinguishable past about 128×64. Bumping to 512×256 is the
    /// next sensible step if PBR specular highlights start to look chunky.
    ///
    /// `minRenderInterval` throttles ONLY the dome dispatch. The IBL pass
    /// runs every `render(params:)` call regardless, so lighting tracks
    /// the sun smoothly even at a low dome cadence.
    ///
    /// Default 0.25 s = 4 Hz dome refresh — adequate for cumulus drift at
    /// typical wind speeds, and the baked noise volume (`NoiseVolume`)
    /// makes the kernel cheap enough that 4 Hz no longer pins the GPU.
    /// Pass a larger value (e.g. 1.0) for 4K scenes where the dome
    /// dispatch is ~4× more expensive — UFO and NapaValley do this.
    public init(engine: SimEngine = .shared,
                resolution: SIMD2<Int> = SIMD2<Int>(2048, 1024),
                iblResolution: SIMD2<Int> = SIMD2<Int>(256, 128),
                minRenderInterval: CFTimeInterval = 0.25) {
        precondition(resolution.x > 0 && resolution.y > 0, "Bad sky resolution")
        precondition(iblResolution.x > 0 && iblResolution.y > 0, "Bad IBL resolution")
        self.device = engine.device
        self.resolution = resolution
        self.iblResolution = iblResolution
        self.minRenderInterval = minRenderInterval
        self.commandQueue = engine.commandQueue
        self.outputTexture = Self.makeTexture(device: engine.device,
                                              width: resolution.x,
                                              height: resolution.y,
                                              label: "VolumetricCloudRenderer.dome")
        self.lightingEnvironmentTexture = Self.makeTexture(
            device: engine.device,
            width: iblResolution.x,
            height: iblResolution.y,
            label: "VolumetricCloudRenderer.ibl"
        )
        self.noiseVolume = NoiseVolume(engine: engine)
        self.pipeline = engine.pipelineCache.pipelineState(
            name: "volSkyRender",
            device: engine.device
        )
        guard let buf = engine.device.makeBuffer(length: MemoryLayout<SkyUniforms>.stride,
                                                 options: .storageModeShared) else {
            preconditionFailure("VolumetricCloudRenderer: failed to allocate uniforms buffer")
        }
        buf.label = "VolumetricCloudRenderer.uniforms"
        self.uniformsBuffer = buf
        // Zeroed single-entry fallback so buffer(1) is always bound even for
        // scenes that never set `burstLights` (count 0 → kernel skips the loop,
        // but Metal still requires a valid binding).
        guard let fallback = engine.device.makeBuffer(
            length: MemoryLayout<BurstLight>.stride,
            options: .storageModeShared
        ) else {
            preconditionFailure("VolumetricCloudRenderer: failed to allocate fallback light buffer")
        }
        fallback.label = "VolumetricCloudRenderer.noLights"
        memset(fallback.contents(), 0, MemoryLayout<BurstLight>.stride)
        self.fallbackLightBuffer = fallback
        if pipeline == nil {
            Self.log.error("volSkyRender pipeline missing — check Shaders/VolumetricSky.metal compiles")
        }
    }

    /// Encode one render. Two dispatches:
    ///   1. IBL into `lightingEnvironmentTexture` — small, always runs.
    ///   2. Dome into `outputTexture` — throttled by `minRenderInterval`
    ///      and gated by a 2-slot semaphore. The throttle prevents
    ///      saturating the GPU; the semaphore prevents main-thread
    ///      blocking on `makeCommandBuffer()` if back-pressure ever
    ///      builds anyway.
    ///
    /// Both dispatches use the same uniforms — they're packed once.
    public func render(params: Params) {
        guard let pipeline else { return }

        // Pack uniforms once; both dispatches read the same buffer. Burst-light
        // count + gain ride in marchBudgets.zw (they're renderer properties, not
        // Params, because the light BUFFER can't be Sendable).
        var uniforms = SkyUniforms(params: params, time: params.time)
        let liveLightCount = burstLights == nil ? 0 : max(0, burstLightCount)
        uniforms.marchBudgets.z = Float(liveLightCount)
        uniforms.marchBudgets.w = burstLightGain
        uniformsBuffer.contents()
            .bindMemory(to: SkyUniforms.self, capacity: 1)
            .pointee = uniforms

        // ── IBL pass: cheap, every call ─────────────────────────────
        if iblSem.wait(timeout: .now()) == .success {
            if !dispatchKernel(target: lightingEnvironmentTexture,
                               resolution: iblResolution,
                               label: "volSkyRender.ibl",
                               pipeline: pipeline,
                               sem: iblSem) {
                iblSem.signal()
            }
        }

        // ── Dome pass: expensive, throttled ────────────────────────
        let now = CACurrentMediaTime()
        if now - lastDomeRenderTime < minRenderInterval { return }
        guard domeSem.wait(timeout: .now()) == .success else { return }
        lastDomeRenderTime = now
        if !dispatchKernel(target: outputTexture,
                           resolution: resolution,
                           label: "volSkyRender.dome",
                           pipeline: pipeline,
                           sem: domeSem) {
            domeSem.signal()
        }
    }

    /// Encode and commit one volSkyRender dispatch into `target`. Returns
    /// `true` if the buffer was committed (caller does NOT signal `sem` —
    /// the completion handler will), or `false` if we bailed before
    /// commit (caller MUST signal `sem` to release the slot it acquired).
    private func dispatchKernel(target: MTLTexture,
                                 resolution: SIMD2<Int>,
                                 label: String,
                                 pipeline: MTLComputePipelineState,
                                 sem: DispatchSemaphore) -> Bool {
        guard let cmd = commandQueue.makeCommandBuffer() else { return false }
        cmd.label = label
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        enc.setTexture(target, index: 0)
        enc.setTexture(noiseVolume.texture, index: 1)
        enc.setBuffer(uniformsBuffer, offset: 0, index: 0)
        // Burst lights — fallback zero buffer when the host set none, so the
        // binding is always valid (count 0 → kernel never reads it).
        enc.setBuffer(burstLights ?? fallbackLightBuffer, offset: 0, index: 1)

        let tgw = min(pipeline.threadExecutionWidth, resolution.x)
        let tgh = max(1, pipeline.maxTotalThreadsPerThreadgroup / max(1, tgw))
        let tgSize = MTLSize(width: tgw, height: min(tgh, resolution.y), depth: 1)
        let grid = MTLSize(width: resolution.x, height: resolution.y, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgSize)
        enc.endEncoding()

        // Capture the semaphore (not self) so the completion handler can
        // run on any thread without crossing the @MainActor boundary.
        let capturedSem = sem
        cmd.addCompletedHandler { _ in capturedSem.signal() }
        cmd.commit()
        return true
    }

    // ── Sky dome ────────────────────────────────────────────────────

    /// Build an inside-facing SCNSphere whose material samples
    /// `outputTexture` as a true equirect panorama, and follow `cameraNode`'s
    /// world position each frame via an `SCNReplicatorConstraint`. The dome
    /// stays world-aligned in rotation, so panning the camera reveals a
    /// stationary sky (the bug that motivated this helper: binding the
    /// equirect to `scene.background.contents` flat-maps it across the
    /// camera viewport, so the entire 360° image follows the camera as a
    /// backdrop).
    ///
    /// Caller wires it in like this:
    ///
    ///     scene.lightingEnvironment.contents = cloud.lightingEnvironmentTexture
    ///     scene.rootNode.addChildNode(cloud.makeSkyDome(followingCamera: camera))
    ///     // do NOT set scene.background.contents = cloud.outputTexture
    ///
    /// `radius` defaults to 50 world units — comfortably inside any scene's
    /// zFar while still being well outside the typical scene's geometry.
    /// The material disables depth read/write and sets renderingOrder to
    /// `-1_000` so the dome paints first and everything else draws over it.
    public func makeSkyDome(followingCamera cameraNode: SCNNode,
                            radius: CGFloat = 50) -> SCNNode {
        let sphere = SCNSphere(radius: radius)
        // Higher tessellation than the default 24 so the equirect's polar
        // pinching doesn't show as triangulation when looking near zenith.
        sphere.segmentCount = 96
        sphere.isGeodesic = false
        sphere.firstMaterial = Self.makeDomeMaterial(texture: outputTexture)

        let node = SCNNode(geometry: sphere)
        node.name = "VolumetricCloudRenderer.skyDome"
        node.castsShadow = false
        node.renderingOrder = -1_000

        // Follow the camera's world position (so the dome is always centred
        // on the viewer and we never run into a far-clip) but leave
        // orientation alone — pan/yaw of the camera reveals a stationary sky.
        let follow = SCNReplicatorConstraint(target: cameraNode)
        follow.replicatesPosition = true
        follow.replicatesOrientation = false
        follow.replicatesScale = false
        node.constraints = [follow]
        return node
    }

    private static func makeDomeMaterial(texture: MTLTexture) -> SCNMaterial {
        let mat = SCNMaterial()
        mat.lightingModel = .constant      // the sky IS the light source; no shading
        mat.diffuse.contents = texture
        mat.diffuse.wrapS = .repeat        // u wraps around the equator
        mat.diffuse.wrapT = .clamp         // v clamps at the poles
        mat.diffuse.minificationFilter = .linear
        mat.diffuse.magnificationFilter = .linear
        mat.diffuse.mipFilter = .none
        // SCNSphere's default UV layout puts the u = 0 / u = 1 seam along
        // the -X equator. The kernel writes u = 0 along the +X direction
        // (see VolumetricSky.metal). Translate by +0.5 in U to bring the
        // texture's +X column under the sphere's +X facet so sun position
        // and wind direction line up with the slider conventions.
        mat.diffuse.contentsTransform = SCNMatrix4MakeTranslation(0.5, 0, 0)
        mat.cullMode = .front              // viewer is inside the sphere
        mat.writesToDepthBuffer = false
        mat.readsFromDepthBuffer = false
        return mat
    }

    /// Synchronous, throttle-bypassing dome render for HEADLESS CAPTURE paths
    /// only — re-renders `outputTexture` with the current params + burst
    /// lights and (optionally) blocks until the GPU finishes, so a staged
    /// review frame samples a sky that reflects the staged lights rather than
    /// a stale throttled one. Never call per-frame in a live tick: the
    /// blocking wait defeats the semaphore/throttle machinery that exists to
    /// keep the live path off the CPU.
    public func renderNow(params: Params, blocking: Bool) {
        guard let pipeline else { return }
        var uniforms = SkyUniforms(params: params, time: params.time)
        let liveLightCount = burstLights == nil ? 0 : max(0, burstLightCount)
        uniforms.marchBudgets.z = Float(liveLightCount)
        uniforms.marchBudgets.w = burstLightGain
        uniformsBuffer.contents()
            .bindMemory(to: SkyUniforms.self, capacity: 1)
            .pointee = uniforms

        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return }
        cmd.label = "volSkyRender.captureNow"
        enc.label = "volSkyRender.captureNow"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(outputTexture, index: 0)
        enc.setTexture(noiseVolume.texture, index: 1)
        enc.setBuffer(uniformsBuffer, offset: 0, index: 0)
        enc.setBuffer(burstLights ?? fallbackLightBuffer, offset: 0, index: 1)
        let tgw = min(pipeline.threadExecutionWidth, resolution.x)
        let tgh = max(1, pipeline.maxTotalThreadsPerThreadgroup / max(1, tgw))
        enc.dispatchThreads(
            MTLSize(width: resolution.x, height: resolution.y, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgw, height: min(tgh, resolution.y), depth: 1)
        )
        enc.endEncoding()
        cmd.commit()
        if blocking {
            cmd.waitUntilCompleted()  // gpu-ok: one-shot off-tick capture path; not per-frame
        }
        lastDomeRenderTime = CACurrentMediaTime()
    }

    // ── Reuse surface for an in-view (perspective) cloud pass ────────
    //
    // A host renderer (IlluminatoramaRenderer) can raymarch the SAME cloud
    // volume per output pixel along camera rays — crisp at output resolution,
    // vs the equirect dome's upscaled-crop blur. It reuses these three already-
    // populated GPU resources rather than re-packing anything: the baked noise
    // volume, the per-`render(params:)`-packed SkyUniforms, and the burst-light
    // buffer (or the zeroed fallback). Read-only handles; the cloud renderer
    // still owns their lifetime and writes them on its own `render` cadence.

    /// Baked 3D noise volume the kernel samples (R=fbm, G=erosion).
    public var noiseTexture: MTLTexture { noiseVolume.texture }
    /// The `SkyUniforms` buffer packed by the most recent `render(params:)` /
    /// `renderNow(params:)` — slab, sun, moon, shaping, lighting, burst gain.
    public var skyUniformsBuffer: MTLBuffer { uniformsBuffer }
    /// Bind when `burstLights` is nil so the host's buffer(burst) is valid.
    public var fallbackBurstLightBuffer: MTLBuffer { fallbackLightBuffer }

    // ── Internals ───────────────────────────────────────────────────

    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState?
    private let uniformsBuffer: MTLBuffer

    /// Zeroed single-entry BurstLight buffer bound at index 1 whenever the
    /// host hasn't supplied `burstLights` — keeps the binding valid for the
    /// (majority) scenes that don't use fireworks cloud lighting.
    private let fallbackLightBuffer: MTLBuffer

    /// Baked 3D noise volume sampled by the kernel at every march step.
    /// Replaces the inline fbm4 + worley3 calls of the V1 kernel — the
    /// dominant speedup over the original implementation.
    private let noiseVolume: NoiseVolume

    /// In-flight cap for the dome dispatch. Without this, a saturated
    /// MTLCommandQueue blocks the main thread on `makeCommandBuffer()`
    /// (Metal's internal slot semaphore = a multi-second freeze). With
    /// a 2-slot non-blocking wait, the dome dispatch organically skips
    /// any tick where two are already in flight. Stays as a belt even
    /// though the baked-noise kernel + throttle make over-saturation
    /// unlikely.
    private let domeSem = DispatchSemaphore(value: 2)

    /// Same idea for the IBL dispatch. The IBL kernel is ~1 ms, so this
    /// effectively never trips — it exists so a runtime change (e.g.
    /// bumping `iblResolution` 4×) can't introduce the same freeze
    /// failure mode.
    private let iblSem = DispatchSemaphore(value: 2)

    /// Time-based throttle for the DOME dispatch only. IBL is dispatched
    /// every `render(params:)` call.
    private let minRenderInterval: CFTimeInterval
    private var lastDomeRenderTime: CFTimeInterval = 0

    private static func makeTexture(device: MTLDevice,
                                    width: Int,
                                    height: Int,
                                    label: String) -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.pixelFormat = .rgba16Float
        d.width = width
        d.height = height
        d.usage = [.shaderWrite, .shaderRead]
        // SceneKit reads this on its own renderer thread. `.private` would
        // require a blit; `.shared` is fine and avoids the extra blit.
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            preconditionFailure("VolumetricCloudRenderer: makeTexture failed")
        }
        t.label = label
        return t
    }
}

// ── Uniforms mirror ──────────────────────────────────────────────────────────
//
// Byte-for-byte mirror of `SkyUniforms` in VolumetricSky.metal. Each field is
// a float4 / float2 (no bare float3 — keeps alignment trivial on both sides).
// See the ALIGNMENT RULE in PBDSolver.swift.

private struct SkyUniforms {
    var cameraPos: SIMD4<Float>
    var sunDir: SIMD4<Float>
    var sunColor: SIMD4<Float>
    var skyZenith: SIMD4<Float>
    var skyHorizon: SIMD4<Float>
    var groundColor: SIMD4<Float>
    var cloudSlab: SIMD4<Float>
    var cloudShape: SIMD4<Float>
    var windTime: SIMD4<Float>
    var lighting: SIMD4<Float>
    var marchBudgets: SIMD4<Float>
    var moonParams: SIMD4<Float>
    var nightParams: SIMD4<Float>
    var atmosphereParams: SIMD4<Float>
    var cloudDetail: SIMD4<Float>
    var cloudField: SIMD4<Float>
    var cloudExtra: SIMD4<Float>
    var cloudExtra2: SIMD4<Float>

    init(params: VolumetricCloudRenderer.Params, time: Float) {
        let sun = normalize(params.sunDir)
        let wind = simd_normalize(params.windDirection.x == 0 && params.windDirection.y == 0
                                  ? SIMD2<Float>(1, 0)
                                  : params.windDirection)

        self.cameraPos   = SIMD4<Float>(params.cameraPos, 0)
        self.sunDir      = SIMD4<Float>(sun, 0)
        self.sunColor    = SIMD4<Float>(params.sunColor, params.sunIntensity)
        self.skyZenith   = SIMD4<Float>(params.skyZenith, 0)
        self.skyHorizon  = SIMD4<Float>(params.skyHorizon, params.hazePower)
        self.groundColor = SIMD4<Float>(params.groundColor, params.groundBlend)
        self.cloudSlab   = SIMD4<Float>(params.cloudBaseY,
                                        params.cloudTopY,
                                        params.horizontalScale,
                                        params.anvil)
        self.cloudShape  = SIMD4<Float>(params.coverage,
                                        params.density,
                                        params.erosion,
                                        params.ambient)
        self.windTime    = SIMD4<Float>(wind.x, wind.y, params.windSpeed, time)
        self.lighting    = SIMD4<Float>(params.phaseForward,
                                        params.phaseBack,
                                        params.absorption,
                                        params.powder)
        self.marchBudgets = SIMD4<Float>(Float(params.cloudSteps),
                                         Float(params.lightSteps),
                                         0, 0)

        // Night sky: auto-derive nightBlend from sun elevation so stars
        // and moon always fade in as the sun crosses the horizon without
        // requiring callers to compute it explicitly.
        // sunDir.y > 0 means the light is traveling upward → sun is below
        // the horizon. We ramp from 0 at the horizon to 1 at ~15° below.
        let nightBlend = min(max(sun.y * 4.0, 0.0), 1.0)
        let moonN = params.moonDir == .zero
                    ? SIMD3<Float>(0.4, 0.6, -0.7)
                    : simd_normalize(params.moonDir)
        self.moonParams  = SIMD4<Float>(moonN, params.moonPhase)
        // nightParams.z carries the density-contrast γ (was unused pad).
        self.nightParams = SIMD4<Float>(params.starBrightness,
                                        params.moonIntensity,
                                        params.densityContrast, nightBlend)
        // atmosphereParams.z = debug-neon flag, .w = directional moonlight gain
        // (both were unused pad).
        self.atmosphereParams = SIMD4<Float>(Float(params.atmosphere.rawValue),
                                             params.atmosphereIntensity,
                                             params.debugNeonBackground ? 1 : 0,
                                             params.moonLightGain)
        self.cloudDetail = SIMD4<Float>(params.weatherScale,
                                        params.weatherSharpness,
                                        params.fadeNear,
                                        params.fadeFar)
        self.cloudField = SIMD4<Float>(params.fieldCenter.x,
                                       params.fieldCenter.y,
                                       params.fieldInner,
                                       params.fieldOuter)
        // cloudExtra: curl warp (x = strength, y = scale) + two-scale body
        // (z = mound mix, w = mound frequency ratio).
        self.cloudExtra = SIMD4<Float>(max(0, params.curlStrength),
                                       max(0, params.curlScale),
                                       max(0, min(1, params.moundMix)),
                                       max(0, params.moundRatio))
        // cloudExtra2.x = multiple-scattering strength.
        self.cloudExtra2 = SIMD4<Float>(max(0, params.multiScatter), 0, 0, 0)
    }
}
