import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - PostKnobs

/// Stored state for the renderer's `post` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `PostKnobs` (`kPost`) and forwards each name to it from
/// `IlluminatoramaRenderer+PostKnobs.swift`; callers see no difference.
@MainActor
final class PostKnobs {
    typealias AutoExposureMetering = IlluminatoramaRenderer.AutoExposureMetering
    typealias DisplayTransform = IlluminatoramaRenderer.DisplayTransform
    typealias HighlightOutlineStyle = IlluminatoramaRenderer.HighlightOutlineStyle
    typealias TAAStillParams = IlluminatoramaRenderer.TAAStillParams

    var exposure: Float = 1.0
    /// Post-FX easing time constant (seconds) from the panel's "Easing" picker.
    /// The post-FX knobs above (exposure, bloom, chromatic aberration, fringe) are
    /// treated as TARGETS; each frame `uploadFrameUniforms` eases an internal
    /// shadow toward the target with `k = 1 − e^(−dt/postFXEasingTau)`, so dragging
    /// a slider glides instead of snapping. 0 = instant (no easing).
    var postFXEasingTau: Double = 0.40
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
    var fixedFrameDelta: Double? = nil
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
    var time: Float = 0
    // ── Highlight outline (feathered halo around selected / hovered objects) ──
    /// When true, every box in `highlightMaskInstances` receives a feathered halo
    /// glow traced around its screen-space silhouette (mask → separable dilation →
    /// HDR composite before bloom). Two modes composite in separate passes:
    /// `highlight == 1` in `selectionOutlineColor`, `highlight == 2` in
    /// `hoverOutlineColor`.
    var selectionOutlineEnabled: Bool = false
    /// Halo style for `highlight == 1` (selected) and `highlight == 2` (hovered).
    var selectionOutline: HighlightOutlineStyle = .selection
    var hoverOutline: HighlightOutlineStyle = .hover
    /// Bounding-box proxies drawn only in the halo mask pass (never the G-buffer),
    /// so highlighted elements keep their real material while gaining a halo. One
    /// solid box per selected/hovered element, its `highlight` set to 1 or 2.
    /// Boxes (not detailed meshes) guarantee a hole-free mask → a clean outer ring.
    /// The bridge clears and re-fills this each frame before `render()`.
    var highlightMaskInstances: [IlluminatoramaInstance] = []
    /// Daydream DH-0887 — STILL mode for the colour TAA resolve: the camera is frozen, so average
    /// (current weight `taaHistoryBlend`) with NO neighbourhood clip, read the history at the same
    /// pixel instead of re-sampling it, and take the display sharpen from the accumulation rather
    /// than the current noisy frame. See `illumi_taa_resolve`. Only valid while the camera and the
    /// scene are frozen — the host's contract. Default `false`: byte-identical for every host.
    var taaStillAccumulation: Bool = false
    /// With `taaStillAccumulation`: keep the Karis 1/(1+Y) sample weights (damps fireflies at the
    /// cost of a slight darkening bias in noisy highlights). Default `false`.
    var taaStillKarisWeights: Bool = false
    /// With `taaStillAccumulation`: apply the still branch's display sharpen (+0.25 luma unsharp over the
    /// accumulation's 3×3, clamped to its range) to the frame handed downstream. Default `true` (the
    /// shipped still); Daydream's Maximum turns it off — its 2× render + declared reconstruction filter
    /// carry the detail (DH-1365). Rides `TAAStillParams.noSharpen`.
    var taaStillSharpen: Bool = true
    var displayTransform: DisplayTransform = .acesRec709
    /// **Off-axis lens shift**, in NDC (clip.xy += shift · clip.w): slides the image under a fixed
    /// eye without turning the view — a view camera's rise/shift, and the shear that keeps a focus
    /// plane on the same pixels while the eye moves across the pupil (`dofReferenceAccumulation`).
    /// Folded into the projection with the TAA jitter and, like it, kept out of the motion vectors.
    /// Zero (default) = exact no-op.
    var projectionShiftNDC: SIMD2<Float> = .zero
    /// **Hold the exposure meter** (opt-in; false = exact no-op): skip the estimate and keep
    /// applying the last smoothed exposure, so offline A/B renders of one instant (and the frames of
    /// an aperture-sampled reference, whose eye moves) print through exactly the same exposure.
    var autoExposureHold: Bool = false
    /// **Pin the auto-exposure meter's sample grid** (opt-in, VZ-0197; nil = the internal render size,
    /// as always). `illumi_exposure_estimate` samples 256 × 32 cells by LINEAR index over an
    /// (w/8 × h/8) grid in normalised UV — so which part of the frame it meters depends on the
    /// internal size (VZ-0172: at 4320 × 2430 the top ~5 % of the frame, at 2880 × 1620 the top
    /// ~11 %), and a host that renders one canvas at two internal scales gets two exposures. Set this
    /// to the internal size the host's look was metered at and every scale samples the same UV
    /// positions (the kernel reads them with a normalised sampler).
    var exposureMeterGridSize: SIMD2<Int>? = nil
    // ── Phase 4.21 — auto-exposure ──────────────────────────────────
    /// Toggle the per-frame log-luminance estimator + EMA. When off,
    /// the tonemap falls back to the static `exposure` scalar above.
    /// On is the default — matches SCN's `wantsExposureAdaptation` so
    /// scenes don't blow out on first frame.
    var autoExposureEnabled: Bool = true
    /// Target log2-luminance the exposure adapts toward. Calibrated
    /// across the in-tree scene library: dark-sky scenes (Eggs,
    /// FloatingFlowers+) become muddy at -2.0 because the geometric
    /// mean of a mostly-dark frame is very low, and `2^(target -
    /// mean)` then pumps exposure 4-6× into mid-grey. -4.0 lets the
    /// dark scenes stay dark while still pulling bright skies
    /// (HotdogWaterslide) back from full clip. Scene controllers
    /// that want a brighter look can override per-scene.
    var autoExposureTargetEV: Float = -4.0
    /// Clamp on the final exposure multiplier so a deeply-dark scene
    /// can't pump the look into mid-grey. Capped at 3× by default
    /// (≈ 1.5 stops) — matches the SCN auto-exposure feel of
    /// "compensate, don't fabricate light."
    var autoExposureMaxBoost: Float = 3.0
    /// Lower clamp on the final exposure multiplier — how far auto-exposure
    /// may DARKEN a bright scene. Phase 4.28: previously hard-coupled to
    /// `1 / maxBoost` (≈ 0.33×), which floored darkening at ~1.5 stops down.
    /// Scenes that use a bright `environmentImage` as both IBL and background
    /// (VintageDiner, Pizza+) want ~0.03–0.08× to come back from full white;
    /// 0.33× left them blown. Decoupled so dark scenes keep their ≤3× boost
    /// cap while bright scenes get ~4.3 stops of darkening headroom. Only
    /// affects scenes bright enough to hit the floor — dark scenes never do.
    var autoExposureMinExposure: Float = 0.05
    /// EMA half-life in seconds for the exposure smoothing. ~0.25 s
    /// matches the SCN feel — fast enough that a scene cut converges
    /// in ~1 s, slow enough that one bright pixel doesn't pump.
    var autoExposureHalfLife: Float = 0.25
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
    var autoExposureHighlightProtection: Float = 0
    /// Where the bright half is asked to land, in log2 linear units, pre-tonemap.
    /// 0 ⇒ 1.0 linear ⇒ ~0.80 through the ACES shoulder — bright, still textured.
    /// Negative pulls the highlights further down. Only read when protection > 0.
    var autoExposureHighlightEV: Float = 0
    var autoExposureMetering: AutoExposureMetering = .mean
    /// The metered percentile under `.percentile` (0…1). 0.5 = the median.
    var autoExposureKeyPercentile: Float = 0.5
    /// The highlight-guard percentile (0…1; 0 = no guard), e.g. 0.95 or 0.99.
    var autoExposureGuardPercentile: Float = 0
    /// Where the guard percentile is asked to land, log2 linear, pre-tonemap.
    var autoExposureGuardEV: Float = 0
    /// Build and publish the histogram even while metering the mean — the INSTRUMENT.
    /// Read it back with `lastExposureHistogram`. Changes no pixel.
    var exposureHistogramEnabled: Bool = false
    /// Scotopic tonemap shaping (see `scotopicDesaturation`): the display luma by which the
    /// night desaturation has faded out (0 = the legacy 0.08) and the Purkinje tint rod
    /// vision is pulled toward (.zero = the legacy neutral grey).
    var scotopicKnee: Float = 0
    var scotopicTint: SIMD3<Float> = .zero
    // ── Phase 4.15: Tonemap + IBL saturation parity ──────────────────
    /// Post-tonemap saturation multiplier. Narkowicz's fitted ACES curve
    /// compresses midtone chroma a little harder than SCN's HDR chain, so a
    /// mild boost keeps parity with the SCN baseline. Was 1.30 while the
    /// tonemap double-encoded sRGB into a non-sRGB background texture (see
    /// [[scenekit-background-texture-double-srgb]]); that washout — the main
    /// cause of the flatness — is now fixed, leaving only genuine ACES chroma
    /// compression to compensate, so 1.10 is enough. 1.0 = no change. Tunable
    /// per-scene if a scene's intentional pastel palette doesn't want the boost.
    var tonemapSaturation: Float = 1.10
    /// **Local tone mapping** — eye-like local adaptation of the scene-referred frame before the
    /// display transform (edge-aware, bilateral-grid base layer; see
    /// `IlluminatoramaLocalToneMapping`). Runs after DOF, before bloom / halation / tonemap, and
    /// after the exposure meter (the meter never sees its own adaptation). `strength == 0`
    /// (default) ⇒ not encoded ⇒ byte-identical frames for every scene that doesn't opt in.
    var localToneMapping = IlluminatoramaLocalToneMapping()
    // ── Tonemap colour-grade (white-balance / tint / contrast / shadows / highlights) ──
    /// White-balance colour temperature in Kelvin (~2000–10000). 6500 = neutral
    /// (channel gain (1,1,1) → exact no-op). Photo-tool convention (DH-0453):
    /// higher K = warmer, lower K = cooler — the inverse of the physical blackbody
    /// sign, matching Lightroom/in-camera WB dials. Applied as a linear-HDR channel
    /// multiply BEFORE exposure + ACES.
    var whiteBalanceK: Float = 6500
    /// Green↔magenta tint on [-1, 1]. 0 = no-op; >0 magenta, <0 green. Luma-
    /// preserving channel gain applied pre-tonemap alongside white-balance.
    var tint: Float = 0
    /// Shadow lift/crush, ~0.5–1.5. 1.0 = no-op; >1 lifts blacks, <1 crushes.
    /// Luma-weighted tone-curve term in the tonemapped (0..1) domain.
    var shadows: Float = 1.0
    /// Highlight lift/roll-off, ~0.5–1.5. 1.0 = no-op. Luma-weighted, post-tonemap.
    var highlights: Float = 1.0
    /// Contrast about mid-grey 0.18, ~0.7–1.3. 1.0 = no-op. Post-tonemap curve.
    var contrast: Float = 1.0
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
    var highlightChromaRolloff: Float = 0
    /// **Split tone, shadow end** — colour temperature (Kelvin) applied to the low-luma
    /// end through the same `whiteBalanceGain` curve as the global white balance
    /// (luma-normalized: it tints, it does not dim). 6500 = neutral (default) → an
    /// exact no-op. BELOW 6500 cools the shadows (photo convention, DH-0453), which
    /// is what gives warm practical lights something to contrast against instead of
    /// every tone sitting in one warm universe.
    var shadowTemperatureK: Float = 6500
    /// **Split tone, highlight end** — the same, masked to the high-luma end.
    /// 6500 = neutral (default) → an exact no-op. Above 6500 keeps highlights warm;
    /// pair it with `highlightChromaRolloff` for the photographic combination —
    /// the scene's own chroma rolls off toward white, and a controlled amount of
    /// warmth goes back in.
    var highlightTemperatureK: Float = 6500
    // ── DH-0715: live-lane look-match ─────────────────────────────────────────
    /// Master enable/blend [0,1] for the live-lane approximation of the still's
    /// RT bounce-GI + RTAO passes (`illumi_tonemap_fs`, applied in the HDR
    /// domain BEFORE white-balance/exposure/ACES/grade — see that file for why
    /// that ordering is what makes this compose with whatever Visuals settings
    /// a scene has, rather than fighting them). 0 (default) = exact no-op. Hosts
    /// gate this live-lane-only — a still keeps the real RT terms.
    var liveLookMatchStrength: Float = 0
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
    var liveLookGIDarken: Float = 0.15
    /// Strength of the fake-bounce warm gain spread (push red up / blue down),
    /// clamped [0, 4] — NOT a 0..1 blend fraction: it scales the tilt's own
    /// magnitude continuously. **0 after the 2026-09-11 multi-scene refit**: the
    /// joint fit across 4 scenes never preferred a nonzero value — the darken
    /// term's own interaction with the existing post-ACES saturation lerp
    /// already carries the chroma match, and adding an explicit R/B tilt on
    /// top only overshot it. Left as a live lever (not deleted) in case a
    /// future fit over more scenes finds it earns its keep after all.
    var liveLookGIWarmth: Float = 0.0
    /// Multiplier the fake-AO term mixes toward, keyed by the EXISTING AO
    /// visibility buffer (`inAO`) rather than a flat screen tint — extra
    /// contact darkening approximating RTAO, riding on real occlusion
    /// geometry. Refit 2026-09-11 alongside `liveLookGIDarken` (see there) —
    /// the error surface was nearly flat over [0.40, 0.70] here, so this
    /// sits in the middle of that range rather than at the exact minimum.
    var liveLookAODarken: Float = 0.50
    /// Opt-in hex-stochastic anti-tiling strength [0,1]. **Default 0 = OFF**, which
    /// is a hard no-op: the G-buffer shader short-circuits every textured sample to a
    /// single plain read, so scenes that leave this at 0 render byte-for-byte
    /// identical to the pre-anti-tiling engine. Raise it (e.g. Daydream Home's large
    /// tiled floors/walls) to break the visible tile-repeat on big planar surfaces —
    /// three stochastically-offset lattice taps blended with variance-preserving
    /// cubic weights, applied consistently across albedo / roughness / normal so the
    /// channels stay registered. 1.0 = full de-repeat; intermediate values crossfade.
    var antiTilingStrength: Float = 0
    /// Saturation boost on the IBL diffuse term in the lighting kernel.
    /// Procedural-gradient backdrops (HotdogDrop+'s peach→tan, every
    /// `+`-tier sky) integrate to a near-grey irradiance, and the
    /// `diffuseIBL = kD · irradiance · albedo` term then multiplies grey
    /// by every albedo → muted indirect lighting. Boosting the
    /// irradiance probe's chroma vs. its luminance preserves the
    /// surface tint through the indirect path too. 1.0 = no change.
    var iblDiffuseSaturation: Float = 1.30
    /// Phase 4.32 — hue-balance DESATURATION of the indirect fill (diffuse-IBL
    /// irradiance + ambient supplement) for highly-saturated probes. 0 = off
    /// (default). Counterpart to `iblDiffuseSaturation`'s boost: a scene whose
    /// IBL is a saturated colored environment (Pizza's red broiler) washes
    /// neutral surfaces toward that hue through the indirect path. This pulls
    /// the monochromatic fill back toward luminance, ramped in only as the
    /// probe's own saturation rises (neutral/pastel IBL is untouched). The
    /// lever the #46 magnitude knobs (intensity/exposure) couldn't move.
    var iblDiffuseDesaturation: Float = 0
    /// Scotopic (Purkinje) desaturation strength, applied in the tonemap. Pulls
    /// only the DIMMEST tonemapped pixels toward their own luminance so dark night
    /// surfaces read neutral moonlit-dark instead of holding their (green grass)
    /// albedo tint; bright pixels (lamps/moon/stars) keep full colour. 0 = OFF
    /// (default) → an exact no-op (Visualizer byte-identical). Hosts typically
    /// fade this in with their night blend.
    var scotopicDesaturation: Float = 0
    /// When true (default), `render()` applies the panel's denoiser flags every
    /// frame, so every Illuminatorama scene — including future ones — inherits them
    /// with zero per-scene wiring (the same policy as `appliesSharedLensFX`, and
    /// the thing that stops 11 controllers each hand-copying the two flags). A
    /// scene that must art-direct its own denoise sets this false.
    var appliesSharedDenoiser: Bool = true
    /// Apply the triangular-PDF debanding dither in the tonemap before the
    /// 8-bit store. On = smooth gradients; off = raw 8-bit quantisation
    /// (exposes contour banding). Exposed for live A/B of gradient banding.
    var debandDitherEnabled: Bool = true
    /// Opt-in: dither in the TRUE sRGB encoding the 8-bit store applies (its linear toe
    /// included) instead of the legacy `pow(1/2.2)` approximation. Near black the two diverge
    /// — the approximation's slope is up to ~5× the real OETF's below code 4, so its "±1 LSB"
    /// lands as ±0.2–0.5 of a real code and a dark gradient's lowest steps are barely
    /// dithered. A night frame whose whole wall sits in codes 0–11 (Digital Clock) needs the
    /// real ±1 LSB there. False (default) ⇒ the legacy dither, byte-identical.
    var debandDitherExactSRGB: Bool = false
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
    var hdrPreExposure: Float = 1
    /// **Hue-stable toe** — opt-in knee [lo, hi] in EXPOSED brightness (the meter's metric,
    /// max(luma, ½·max channel), after exposure and local tone mapping). AgX's per-channel sigmoid
    /// (and its 'punchy' power) behaves like a ~2.3-power in its toe, so every channel RATIO in
    /// the dark end is raised to roughly that power — more with the post-tonemap saturation push
    /// on top (measured on Digital Clock's 19:40 frame: a lavander sky, linear 1 : 0.66 : 1.63,
    /// printed (14, 2, 53)). Below `lo` the display transform runs on the pixel's max-RGB norm
    /// and rescales its colour (the scene's own linear ratios, no per-channel toe) and the
    /// saturation push is off; above `hi` both are exactly the shipped ones; between, a
    /// smoothstep blend. `.zero` (default, `hi == 0`) ⇒ off, byte-identical.
    var hueStableToe: SIMD2<Float> = .zero
    /// Opt-in, with `hueStableToe` and the AgX display transform: the toe's LEVEL comes from AgX's
    /// bare sigmoid instead of its 'punchy' look. The look's power 1.35 acts as a ~2.3-power toe —
    /// 4 stops under mid-grey prints at sRGB 4, 5 stops at 1 — so a locally-adapted night room is
    /// fitted into ~4 printable stops and re-crushed; the bare sigmoid prints 4 stops under at 21,
    /// 5 under at 7, 6 under at 2 (Digital Clock round 3). The toe keeps the scene's colour ratios
    /// either way; above the knee the shipped (punchy) transform is untouched. False (default) ⇒
    /// displayParams.w = 0 ⇒ byte-identical.
    var hueStableToeBareLevel: Bool = false
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
    var taaEnabled: Bool = false
    /// Per-scene escape hatch from the panel's TAA authority. `nil` (default) =
    /// follow the panel. Set to a concrete value ONLY when a scene must force
    /// TAA on/off regardless of the panel (e.g. a perf governor disabling it
    /// under load). Prefer leaving it `nil` — the panel should win.
    var sharedTAAOverride: Bool? = nil
    /// When true (default), `render()` applies the panel's lens-aberration knobs
    /// (chromatic + spherical aberration, fringe, post-FX easing) every frame, so
    /// every Illuminatorama scene — including new ones — inherits them with zero
    /// per-scene wiring. (Bloom is NOT included: scenes art-direct it, so they opt
    /// in via `applySharedPostFX`.) A context that must not be driven by the app
    /// panel (e.g. an isolated AssetLab harness) can set this false.
    var appliesSharedLensFX: Bool = true
    /// Weight of the current-frame sample in the resolve blend. Smaller =
    /// smoother and more denoised but slower convergence on disocclusion.
    /// 0.1 is the standard "good default" — visible AA from camera jitter
    /// within ~10 frames, and the SSAO/SSR noise drops out almost
    /// immediately on a still camera.
    var taaHistoryBlend: Float = 0.1
    /// Magnitude of the sub-pixel projection jitter, in pixels. Standard
    /// TAA uses a Halton sequence within ±0.5 px so successive frames
    /// supersample the same output pixel; values >0.5 push samples beyond
    /// the reconstruction filter footprint, making geometry visibly walk
    /// between frames. With SSAA on (`internalRenderScale > 1`) the AA
    /// role is filled by spatial oversampling — leaving jitter at 0 so
    /// TAA contributes denoise of SSAO/SSR noise without adding any
    /// motion. Set back to 0.5 to re-enable temporal supersample AA.
    var taaJitterPixels: Float = 0.0
    /// The internal → output reduction (Daydream DH-1365): `.legacy` (default, byte-identical) is the
    /// tonemap's 4-tap bilinear; a declared kernel gathers every internal texel within its radius
    /// in linear HDR. See `IlluminatoramaReconstructionFilter`. Rides `FrameUniforms.taaJitterDelta.zw`.
    var reconstructionFilter: IlluminatoramaReconstructionFilter = .legacy
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
    var blockingLightingCompile: Bool = false
    /// Per-frame tap, fired at the end of `present(to:)` with the just-presented
    /// `outputTexture` and this renderer's `commandQueue`. The video recorder
    /// installs it to capture native (MTKView) scenes — there's no SceneKit
    /// render delegate to hook on this path. Copying on the renderer's OWN queue
    /// (not a fresh one) keeps the capture ordered behind the frame's render, so
    /// it reads a fully-written texture. `nil` (default) = no capture overhead.
    /// Set/cleared on the main thread; `present(to:)` runs on main.
    var onFramePresented: ((_ texture: MTLTexture, _ queue: MTLCommandQueue) -> Void)?
    /// 0 = OFF (default) → the tonemap LUT branch is an exact no-op. 1 = full grade.
    var colorLUTAmount: Float = 0
}
