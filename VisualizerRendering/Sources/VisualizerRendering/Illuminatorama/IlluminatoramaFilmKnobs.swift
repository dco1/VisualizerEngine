import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - FilmKnobs

/// Stored state for the renderer's `film` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `FilmKnobs` (`kFilm`) and forwards each name to it from
/// `IlluminatoramaRenderer+FilmKnobs.swift`; callers see no difference.
@MainActor
final class FilmKnobs {


    var bloomThreshold: Float = 1.0
    var bloomIntensity: Float = 0.6
    /// Bloom soft knee, as a FRACTION of `bloomThreshold` (S1.2). 0 restores the
    /// hard `max(0, lum - T)` cut byte-for-byte; the curve above T·(1+knee) is
    /// identical either way, so this only softens the pop as a light crosses the
    /// bar. Default 0.5 — half a stop of ramp, the Unity/Karis convention.
    var bloomSoftKnee: Float = 0.5
    /// Upsample scatter: the CONVEX blend weight toward the blurred low mip on
    /// the way up the pyramid. 0 leaves only the finest level (a tight rim), 1
    /// only the coarsest (a formless wash). Level i ends up weighted
    /// (1-s)·s^i — a geometric series that sums to exactly 1 at ANY level count,
    /// so changing this changes the halo's SHAPE and never its total energy.
    /// Default 0.7.
    var bloomScatter: Float = 0.7
    /// **Lens point-spread bloom (Daydream DH-0992).** 0 (default) ⇒ the geometric up-chain above,
    /// byte-identical. > 0 ⇒ the pyramid's per-level weights are set so the halo's surface
    /// brightness falls as a POWER LAW, I(r) ∝ r^−α with α = this value — the long-tailed glare
    /// spread of a real lens (a tight core over a faint skirt that keeps going), where the
    /// geometric chain is an exponential that dies. Each level reaches 2× the radius of the one
    /// before, so an octave's share of the energy is ∝ r^(2−α); the weights are normalised, so the
    /// total (and `bloomIntensity`'s meaning) is unchanged. Real lenses sit near α ≈ 2.5–3.
    var bloomPSFExponent: Float = 0
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
    var bloomPhysicalGlare: Bool = false
    /// Whose glare it is (Daydream DH-1009): a camera lens's surface scatter (`.cameraLens`, the
    /// default — Ritt et al. 2020's generic double-Gauss, η ≈ 0.025) or the CIE 146:2002 eye
    /// (`.standardObserver` or `.eye(age:pigment:)`, η ≈ 0.30). Both run the same fit + far tail.
    var bloomGlareModel: IlluminatoramaGlareSpread.Model = .cameraLens
    /// TEST-OBSERVABLE: the glare fit the last frame used (nil ⇒ physical glare was off).
    var lastBloomGlareFit: IlluminatoramaGlareSpread.Fit?
    // ── Digital sensor noise (Daydream DH-0992) ───────────────────────
    /// Shot + read noise of a photo-electron count, in the exposed scene signal before the
    /// display transform (see the tonemap's sensor block). 0 (default) ⇒ off, byte-identical;
    /// 1 ⇒ the physical amount for `sensorElectronsPerUnit` and `sensorReadNoise`.
    var sensorNoiseStrength: Float = 0
    /// Photo-electrons per unit of exposed scene radiance at the camera's own ISO, BEFORE the
    /// meter's push (which the shader counts as extra gain). A full-frame sensor at ISO 100 puts
    /// mid-grey (0.18) near 5 000 e⁻ — ~28 000 per unit; divide by ISO/100 for a faster setting.
    var sensorElectronsPerUnit: Float = 28_000
    /// Read noise, electrons (a modern CMOS sensor: ~2–5).
    var sensorReadNoise: Float = 3
    /// How much of the noise the three channels share (demosaicing correlates neighbours' noise
    /// across channels); 0 = independent RGB speckle, 1 = pure luminance noise.
    var sensorChannelCorrelation: Float = 0.5
    /// The capture medium's EMULSION grain (Daydream DH-1000): a relative σ of the exposed signal
    /// — constant in log exposure, which is what "constant in density" means for a negative — added
    /// in the sensor block as a second variance term beside the shot + read noise, on ONE draw, so
    /// the two are a single noise with a single σ (they cannot stack). 0 (default) ⇒ the sensor
    /// block is exactly DH-0992's; with `sensorNoiseStrength` also 0 the block never runs. The
    /// grain is drawn on the `filmGrainSize` cell grid while this is on. Independent of the legacy
    /// encoded-domain `filmGrainStrength`, which a host using this should leave at 0.
    var filmDensityGrain: Float = 0
    /// How much of the emulsion grain the three dye layers share (0.80: the DH-0880 figure).
    /// Per pixel, the draw's correlation is the variance-weighted mix of this and
    /// `sensorChannelCorrelation`, so each mechanism keeps its own colour character.
    var filmDensityGrainCorrelation: Float = 0.80
    /// 3×3 tent radius on the up-chain, in LOW-mip texels. 1.0 is the plain tent;
    /// larger values widen each level's contribution without adding taps.
    var bloomTentRadius: Float = 1.0
    /// Screen-space SUN lens flare strength in the tonemap pass (ghost train +
    /// halo + anamorphic streak, optically occluded by sampling the HDR frame at
    /// the sun's screen position). 0 = OFF (the default for every scene) → an
    /// exact shader no-op. 1.0 is the intended "on" look; values between scale it.
    var lensFlareIntensity: Float = 0
    /// TEST-OBSERVABLE: the lens-flare uniform cluster as computed for the last frame
    /// (x = strength, yz = sun screen uv, w = on-screen weight). Lets a host's GPU test
    /// verify the sun projection without reading the uniform buffer back.
    var lastLensFlareParams: SIMD4<Float> = .zero
    /// Lens-style transverse chromatic aberration strength in the tonemap pass.
    /// 0 = OFF (the default for every scene) → an exact shader no-op. Maps into
    /// the trailing `_padPlush0` slot of `IlluminatoramaFrameUniforms`, so the
    /// struct stride is unchanged.
    var chromaticAberration: Float = 0
    /// Longitudinal / axial chromatic aberration ("purple fringing") strength in
    /// the tonemap pass. 0 = OFF → an exact shader no-op. `fringeTint` is the
    /// dark-side halo colour (linear-ish sRGB components); the bright side gets
    /// its complement. New 16-byte uniform cluster, mirrored byte-for-byte.
    var fringe: Float = 0
    var fringeTint: SIMD3<Float> = SIMD3(0.62, 0.12, 0.92)
    /// Spherical-aberration radial blur strength in the tonemap pass.
    /// 0 = OFF (default) → exact shader no-op. Maps into the former
    /// `_padPlush1` slot of `IlluminatoramaFrameUniforms` (stride unchanged).
    var sphericalAberration: Float = 0
    /// Lens vignette (issue #65): `vignetteStrength` is the corner-darkening amount
    /// (0 = OFF → exact shader no-op) and `vignetteExtent` the normalised radius
    /// kept fully bright before the falloff begins. Film-stock grain:
    /// `filmGrainStrength` is the amplitude (0 = OFF → exact no-op) and
    /// `filmGrainSize` the grain cell size in output px. NEW 16-byte uniform
    /// cluster, mirrored byte-for-byte. Applied directly (not eased) — strength
    /// ramps are gentle enough not to need the glide the blur knobs use.
    var vignetteStrength: Float = 0
    var vignetteExtent: Float = 0.55
    var filmGrainStrength: Float = 0
    var filmGrainSize: Float = 1.5
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
    var diagramMix: Float = 0
    /// How much the flat fills vary with surface orientation, 0…1. 0 = perfectly flat
    /// (a white box is one white silhouette and only the outlines separate its faces);
    /// 1 = a full sky/ground hemisphere term. ~0.35 keeps planes legible without the
    /// image drifting back toward a render.
    var diagramWrap: Float = 0.35
    /// Outline strength, 0…1. 0 = fills only, no line work.
    var diagramOutlineStrength: Float = 1.0
    /// Outline colour (linear RGB). Near-black by default, as an ink line would be.
    var diagramOutlineColor: SIMD3<Float> = SIMD3(0.10, 0.11, 0.13)
    /// How readily a DEPTH step becomes a line. Higher = more lines (silhouettes and
    /// near-coplanar overlaps both draw); lower = silhouettes only.
    var diagramEdgeDepthSensitivity: Float = 1.0
    /// How readily a NORMAL change becomes a line. This is what draws the crease where
    /// two faces of the same object meet — a depth-only detector cannot see it.
    var diagramEdgeNormalSensitivity: Float = 1.0
    /// Line thickness in OUTPUT pixels. The detector samples a cross at this radius, so
    /// non-integer values are legitimate; much above ~2 reads as a smudge, not a pen.
    var diagramEdgeThickness: Float = 1.0
    /// Film **halation**: the warm halo real film wears around blown highlights.
    /// Light punches through the emulsion, reflects off the back of the base and
    /// re-exposes it from behind; what reaches the base is mostly red, hence the tint.
    /// A separate half-res threshold → dense four-scale isotropic gaussian chain from
    /// bloom (see `IlluminatoramaPost.metal` § Halation). 0 = OFF (the default for every scene): the halation
    /// passes are not even encoded and the tonemap branch is skipped, so non-opting
    /// scenes are byte-for-byte unchanged. ~0.4 is a tasteful film amount; 1.0 is strong.
    var halationIntensity: Float = 0
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
    var halationThreshold: Float = 0.6
    /// Halation halo radius as a FRACTION OF FRAME HEIGHT (2σ of the widest scale). Halation is a fixed size on the negative — set by
    /// the base thickness — so it must not depend on the render resolution: the old
    /// radius was in internal texels, and a 2× supersampled still got half the halo.
    /// The PSF is four gaussians an octave apart (σ = R/16 … R/2), a long soft tail.
    var halationRadius: Float = 0.025
    /// Halation TAIL tint (linear RGB) — the deep red where only red survives the round
    /// trip through the base. The core is derived from it in the shader and reads orange.
    var halationTint: SIMD3<Float> = SIMD3(1.0, 0.20, 0.07)
    /// **Natural (cos⁴) vignetting** — `(halfFrameDiagonalMM / focalLengthMM)²`, scaled by how
    /// much of the physical falloff the host wants (1 = all of it). Multiplies the SCENE
    /// before the exposure meter and the tonemap, because it is the lens losing light, not a
    /// grade on the finished image (that is `vignetteStrength`). 0 = off. See DH-0882.
    var naturalVignetteK: Float = 0
    // ── Phase 9 — film-stock LUT (post-tonemap colour grade) ─────────
    /// The film stock's cube as an `MTLTexture3D`, N×N×N. `nil` = stock bypassed
    /// (identity pass-through). Assign from the host via `setFilmLUT(_:strength:)` —
    /// do NOT set this, `filmLUTStrength` or `filmLUTSize` directly.
    var filmLUTTexture: MTLTexture? = nil
    /// Blend weight [0, 1]: 0 = stock fully bypassed, 1 = the stock's own rendering.
    var filmLUTStrength: Float = 1.0
    /// Cells per axis of `filmLUTTexture`, taken from the texture itself rather than
    /// assumed — the shader's half-texel inset is derived from it (DH-0879).
    var filmLUTSize: Float = 16
    /// Velocity-buffer motion blur (issue #65). `motionBlurStrength` scales the
    /// per-frame screen velocity (≈ shutter fraction; ~0.5 is a natural 180° look);
    /// 0 = OFF → the tonemap gather is skipped (exact no-op). `motionBlurMaxPx` caps
    /// the streak length in output px so a fast pan can't smear the whole frame.
    /// Driven by camera AND per-object motion (the renderer ping-pongs the instance
    /// buffer, so spinning/translating instances streak correctly).
    var motionBlurStrength: Float = 0
    var motionBlurMaxPx: Float = 48
    /// Aerial perspective — extinction coefficient σ (1/m) for the distance-haze blend
    /// applied to the deferred lighting pass's FULL composite (never the isolated debug
    /// terms): `mix(airlight, color, exp(-σ·viewDist))`, with the airlight colour sampled
    /// from the prefiltered sky cube at the view azimuth's horizon, so it follows time of
    /// day and sky state with no host plumbing. 0 (the default, and every scene that never
    /// opts in) = an EXACT shader no-op. Physically plausible clear-day Rayleigh is ~1e-4
    /// (≈4 % haze at 400 m) — this is the far-field cue that keeps a distant ground plane
    /// from arriving at the horizon fully saturated, not a fog effect.
    var aerialPerspectiveDensity: Float = 0
    /// Multiplier on the aerial-perspective AIRLIGHT (default 1 = unchanged). The airlight is a
    /// coarse-mip sample of the prefiltered sky cube just above the horizon, so it averages in
    /// the brighter sky (and any moon aureole) above — by day that is the right broad inscatter,
    /// but under a physical night sky, whose horizon the renderer draws per pixel, it measured
    /// ~2.5× the horizon the camera actually sees (UFO Ultra: haze on the far land glowed as a
    /// lilac band under a black horizon). A host whose haze reads brighter than its own horizon
    /// scales it down here until the airlight is sampled from the model the sky pass draws.
    var aerialPerspectiveAirlightScale: Float = 1
    /// Take the airlight from the sky cube's SHARPEST mip just above the horizon — the colour the
    /// camera sees at the horizon — instead of the coarse broad-inscatter mip. A scalar can't fix
    /// a hue mismatch: under a moonlit physical sky the coarse mip is the blue zenith's average
    /// while the horizon band is olive, so far land hazed to blue under an olive horizon.
    var aerialPerspectiveAirlightFromHorizon: Bool = false
    /// Screen-space subsurface scattering (issue #65). Jimenez-style separable SSS
    /// for skin / wax / marble / food. `sssStrength` 0 = OFF → the lighting pass
    /// skips the side-buffer write and the blur/composite passes aren't encoded →
    /// an EXACT no-op (the off-by-default universal pattern). `sssRadius` is the
    /// diffusion mean-free-path in MILLIMETRES; `sssTint` scales the per-channel
    /// scatter distance (default reddish skin profile → warm bleed; marble ≈ 1,1,1;
    /// wax ≈ 1,0.7,0.45). A mesh opts a surface in by tagging its vertices with
    /// colour alpha ∈ [0.90,0.98] (→ `normalRoughness.w` ≈ 0.95h). Driven from the
    /// panel via `applySharedLensFX` (universal off-by-default knob).
    var sssStrength: Float = 0
    var sssRadius: Float = 8.0   // diffusion radius, millimetres
    var sssTint: SIMD3<Float> = SIMD3(1.0, 0.4, 0.25)
    /// TEST-OBSERVABLE: the GPU time of the last `measureBloomImpulseResponse` command buffer
    /// (upload, the whole chain, readback blit) — the A/B a gate times the far-field tail with.
    var lastBloomImpulseGPUMilliseconds: Double = 0
}
