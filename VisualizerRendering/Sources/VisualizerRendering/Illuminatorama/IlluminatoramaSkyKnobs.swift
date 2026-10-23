import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - SkyKnobs

/// Stored state for the renderer's `sky` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `SkyKnobs` (`kSky`) and forwards each name to it from
/// `IlluminatoramaRenderer+SkyKnobs.swift`; callers see no difference.
@MainActor
final class SkyKnobs {
    typealias EmissiveColumn = IlluminatoramaRenderer.EmissiveColumn

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
    var treeWindStrength: Float = 0
    var treeWindHeading: Float = 0
    /// Foliage-wind motion-vector delta, in **seconds** — the simulated time elapsed between
    /// this frame and the previous one (DH-0492). The G-buffer samples the previous frame's
    /// `applyTreeWind` at `time − windPreviousDelta`, so a swaying canopy/blade writes a real
    /// screen-space velocity TAA can reproject instead of crawling between pixels. Leave it at
    /// 0 (the default) and the previous position is sampled at the SAME time as the current —
    /// the pre-DH-0492 no-op, so every scene that never sets it, and every settled capture
    /// (whose clock does not advance between accumulation frames), is byte-identical. The host
    /// feeds it the real per-frame clock delta ONLY on the live, advancing canvas; a frozen
    /// capture must leave it 0 (its previous == current time) to stay still.
    var windPreviousDelta: Float = 0
    // ── Volumetric light shaft (god-rays) ────────────────────────────
    /// Single-scatter ray-march of the sun through hazy air, making the beam
    /// visible in the air. Uses `rtSunDirection` / `rtSunColor` for the sun.
    var volumetricEnabled: Bool = false
    var volFogDensity: Float = 0.05
    var volScatterStrength: Float = 1.0
    var volAnisotropy: Float = 0.6
    var volSteps: Int = 48
    var volMaxDistance: Float = 24.0
    var volWindowX: Float = 4.0
    /// (y0, y1, z0, z1) of the window opening for the in-air shadow test.
    var volWindowRect: SIMD4<Float> = SIMD4(0.9, 2.5, -1.6, 1.6)
    var volFeather: Float = 0.12
    /// Outdoor mode: skips the analytic window shadow test so all air samples
    /// scatter sun light. Use for exterior scenes (forests, fields, open sky).
    /// The Henyey-Greenstein phase still peaks toward the sun, and the depth
    /// buffer still terminates the march at surfaces, so beams glow brightest
    /// when the camera looks toward the low sun through canopy gaps.
    var volOutdoorMode: Bool = false
    // ── Traced volumetric light (Daydream DH-0990) ────────────────────
    /// Single scattering in a faint haze, marched along each view ray against the REAL geometry:
    /// the sun through a shadow ray per sample (so the shafts are exactly what the window openings
    /// cut) plus the sky through one window APERTURE per sample (`IlluminatoramaAreaLight
    /// .isAperture`), confined to ROOFED air (a short ray straight up must meet the house), so an
    /// exterior view never fogs. Monte-Carlo per frame and progressive across frames: a STILL
    /// lane feature — it needs a live TLAS and an accumulating resolve. Default off: no cost.
    var volumetricRTEnabled: Bool = false
    /// Haze extinction = scattering coefficient σ, per metre. A real room's air is ~1e-3; a
    /// photograph's visible shafts need dust, ~1e-2.
    var volRTDensity: Float = 0.01
    /// Henyey–Greenstein g for the haze (> 0 forward: the beams glow toward the sun).
    var volRTAnisotropy: Float = 0.5
    /// March samples per pixel per frame (stratified; a still's frames fill in the rest).
    var volRTSteps: Int = 6
    /// Trace the haze on a random 1/`volRTPixelStride` of the pixels each frame, at that many times the
    /// weight — for a still, which accumulates its frames (1 ⇒ every pixel, as before).
    var volRTPixelStride: Int = 1
    /// The march's reach along a view ray (metres) — an interior is a few metres deep.
    var volRTMaxDistance: Float = 20
    /// How far above an air sample a roof may be for the sample to count as indoors (metres).
    var volRTRoofReach: Float = 8
    /// Separate gains on the sun's and the sky's in-scatter (1 = physical for the density).
    var volRTSunScatter: Float = 1
    var volRTSkyScatter: Float = 1
    /// DH-0999 — SUNLIT DUST: a second, sun-only scatterer (per metre at full strength) whose
    /// amount follows the direct sun measured entering the view's air. A coarse pre-pass marches
    /// the view's own rays and counts the roofed samples that see the sun; with A = sunlit /
    /// roofed, the dust is `volRTDustDensity · min(1, A / volRTDustAdmittanceRef)` — a sunbeam
    /// crossing the room reads as a shaft, a room no sun enters gets none (no grey fog). It adds
    /// only the collimated sun's in-scatter: as a conservative scatterer in the room's
    /// near-uniform diffuse field its extinction and diffuse in-scatter cancel, so it never dims
    /// the room. 0 = off: no pre-pass, the air term alone.
    var volRTDustDensity: Float = 0
    /// HG g of the dust (large particles: strongly forward — the beam glows toward the sun).
    var volRTDustAnisotropy: Float = 0.6
    /// The sunlit fraction of the roofed air in view at which the dust reaches full density.
    var volRTDustAdmittanceRef: Float = 0.1
    /// The admittance lattice's width in rays (height follows the aspect).
    var volRTAdmittanceGrid: Int = 64
    // ── Spot-light beam scattering (issue: CK5 concert beams) ───────
    /// Single-scatter march of every `spotLights` cone through hazy air, so
    /// spots read as BEAMS in the air, not just pools on surfaces. Runs after
    /// the sun volumetric pass, independent of `volumetricEnabled` — a stage
    /// scene wants haze-lit beams with no sun window. Per-ray cost is bounded
    /// per spot by analytic range-sphere clipping, so dead air is ~free; keep
    /// `spotLights.count` in the low tens. Beams are depth-terminated but not
    /// shadow-traced inside the interval (geometry in front hides a beam;
    /// geometry inside one doesn't carve it).
    var volSpotBeamsEnabled: Bool = false
    /// Haze density for the beam march (separate from `volFogDensity` so the
    /// sun shaft and stage haze are tunable independently).
    var volSpotBeamDensity: Float = 0.06
    var volSpotBeamStrength: Float = 1.0
    /// Forward-scatter anisotropy: higher = beams flare when the camera looks
    /// into the fixture, like real stage haze.
    var volSpotBeamAnisotropy: Float = 0.55
    /// Jittered samples per spot interval (TAA accumulates the noise out).
    var volSpotBeamSteps: Int = 20
    var volSpotBeamMaxDistance: Float = 60.0
    /// Carve the beams with the spot shadow maps: each march sample is compared against its
    /// spot's shadow slice, so a beam stops where geometry blocks its light (a cone from inside
    /// a mouth ends at the lips instead of glowing on through the face). Only spots that hold a
    /// shadow slice are carved (`spotShadowsEnabled` + capacity). Default false ⇒ the pass is
    /// byte-identical to before for every existing host.
    var volSpotBeamShadows: Bool = false
    /// Only the first `volSpotBeamSpotLimit` entries of `spotLights` march as haze beams; the
    /// rest light (and shadow) surfaces only. Lets a host carry studio fill / rim cones in
    /// `spotLights` — for their shadow maps — without each one drawing a beam across the frame
    /// (put the beam-worthy spots first). Default `Int.max` ⇒ every spot beams, byte-identical
    /// for every existing host.
    var volSpotBeamSpotLimit: Int = .max
    var volEmissiveColumn: EmissiveColumn? = nil
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
    var nightSkyStarBrightness: Float = 0
    /// Moon disk brightness (HDR). 0 = no moon. ~1.5–3 reads as a luminous moon
    /// that blooms gently at night exposure.
    var nightSkyMoonIntensity: Float = 0
    /// Unit vector from the scene toward the moon (e.g. a real ephemeris).
    var nightSkyMoonDirection: SIMD3<Float> = SIMD3(0, 1, 0)
    /// Unit vector from the scene toward the TRUE sun — below the horizon at
    /// night. Lights the moon sphere per-pixel, so the phase (crescent→gibbous→
    /// full) and its orientation come straight from the geometry.
    var nightSkySunDirection: SIMD3<Float> = SIMD3(0, -1, 0)
    /// Moon angular RADIUS in radians. The real moon is ≈0.0047 (0.27°); the
    /// default is ~2× that for a photographic read. Taste knob.
    var nightSkyMoonAngularRadius: Float = 0.0095
    /// Which analytic night sky (IlluminatoramaNightSky.h). `.legacy` (default) is the
    /// byte-identical hashed grid + sphere-lit disk; `.physical` is the equal-area,
    /// magnitude-distributed, extinguished, twinkling star field with a Milky Way, the
    /// real-sized maria-mapped moon with earthshine and an aureole. In the physical model
    /// the brightnesses are NOT pre-faded by nightBlend: stars wash out against the sky
    /// behind them on their own, so pass `nightSkyStarBrightness` unscaled. Set
    /// `nightSkyMoonAngularRadius` to 0 for the real 0.26°.
    var nightSkyModel: VolumetricCloudRenderer.Params.NightSkyModel = .legacy
    /// Physical model only — gains (1 = physical) on the lunar aureole (raise for haze /
    /// thin cloud), earthshine, the Milky Way and stellar scintillation.
    var nightSkyMoonHalo: Float = 1
    var nightSkyEarthshine: Float = 1
    var nightSkyMilkyWay: Float = 1
    var nightSkyTwinkle: Float = 1
    /// Physical model only — scene radiance of a magnitude-0 flux spread over one steradian:
    /// the ONE scale that puts stars, Milky Way and aureole in their real proportions. Match
    /// the sky dome's `Params.nightRadiance` so the moonlit sky and the stars agree.
    var nightSkyRadiance: Float = 1.5e-5
    /// Physical model only — world → J2000-equatorial rotation as a quaternion
    /// (ix, iy, iz, r); see `NightSkyEphemeris.celestialOrientation`. Identity by default.
    var nightSkyCelestialOrientation: SIMD4<Float> = SIMD4(0, 0, 0, 1)
    /// Physical model only — the faintest star drawn (visual magnitude); 0 (default) = no limit
    /// beyond the per-pixel washout. See `VolumetricCloudRenderer.Params.starLimitingMagnitude`.
    var nightSkyStarLimitingMagnitude: Float = 0
    /// When true AND the three handles below are set, the in-view cloud pass
    /// composites crisp camera-ray clouds into the HDR target (issue #61). OFF
    /// by default → every scene + the dome/IBL path is byte-identical.
    var inViewCloudsEnabled = false
    /// DH-0578 — the ACCUMULATED screen-resolution cumulus pass (`encodeCumulusInViewPass`): a
    /// `.cumulus` deck marched along each pixel's own ray with detail erosion and a fine light
    /// march, one interleave cell per frame while camera and sky are still, swapped in for the
    /// dome's sample once every pixel has one. Needs the same three handles as the legacy pass
    /// (`cloudNoiseTexture` / `cloudSkyUniforms`; `cloudBurstLights` unused) plus `equirectSky`.
    /// OFF by default → byte-identical.
    var cumulusInViewEnabled = false
    /// Accumulator resolution divisor (1 = canvas resolution, 2 = half — a quarter of the rays,
    /// bilinear-resolved; at the photo lane's 2× supersample, 2 is still OUTPUT resolution).
    var cumulusInViewScale = 2
    /// n of the n×n interleave (1, 2 or 4): each frame marches 1/n² of the accumulator.
    var cumulusInViewInterleave = 4
    /// Samples per pixel after which marching stops (the converged sky costs only the resolve).
    var cumulusInViewMaxSamples = 4
    /// Camera-ray steps through the slab, and sun light-march steps per dense sample.
    var cumulusInViewSteps = 128
    var cumulusInViewLightSteps = 7
    /// High-frequency detail erosion (0 = the dome's field, 1 = the ~4 m octave, 2 = + ~1.5 m).
    var cumulusInViewDetail: Float = 2
    /// Keep the last converged accumulator across CAMERA moves and reproject it into each frame
    /// (DH-0578 follow-up): a moving camera keeps the sharp clouds instead of popping back to the
    /// dome, and the new accumulator crossfades in after it stops. OFF → byte-identical (reset on
    /// any camera change, as before).
    var cumulusInViewReprojection = false
    /// Frames over which a freshly converged accumulator crossfades in over the reprojected history
    /// (and, with no history, over the dome). 0 = switch at once (the photo lane: an accumulating
    /// still must not average a ramp in).
    var cumulusInViewFadeFrames = 0
    /// Daydream DH-1462 — the cumulus deck shadows the scene: a low-res transmittance map toward the
    /// sun, baked from the SAME density the sky marches (`volSkyCloudShadowMap`) whenever the sky, sun
    /// or the map's camera-following centre changes, and read by every sun-lit lane (deferred, RT
    /// direct sun, path vertices, secondary hits, glass). Needs `cloudNoiseTexture` +
    /// `cloudSkyUniforms` and a `.cumulus` deck. OFF by default → byte-identical.
    var cloudShadowEnabled = false
    /// Map texels per side, the world square it covers (m, centred on the camera) and sun-ray steps.
    var cloudShadowResolution = 256
    var cloudShadowExtent: Float = 4000
    var cloudShadowSteps = 32
    /// The plane the map is baked on (a point above it slides down the sun ray to it).
    var cloudShadowGroundY: Float = 0
    /// 0…1 — how much of the deck's transmittance reaches the sun term (1 = physical).
    var cloudShadowStrength: Float = 1
    /// Shade only points within this height of `cloudShadowGroundY` (0 = everything — the house too).
    var cloudShadowHeightLimit: Float = 0
    /// The clock the in-view sky animates by (the lava lamp's wax, the cirrus threads' flow) —
    /// `nil` (the default) = `time`, byte-identical. A host that lets those speeds change live
    /// passes its own INTEGRATED phase (Σ speed·dt) and sets the sky's speed factors to 1: the
    /// shader multiplies clock × speed, so a speed step on the frame clock jumps the pattern by
    /// time·Δspeed — hundreds of seconds of animation after a few minutes.
    var inViewSkyTime: Float? = nil
    /// Per-FRAME lava-lamp weight (0…1) for the in-view sky, overriding the cloud renderer's
    /// `lavaFade` (which rides the host-throttled sky uniforms — a fade driven through those
    /// would step at the sky's update rate). `nil` (the default) = the uniform's value.
    var inViewLavaFade: Float? = nil
    /// The cirrus threads' own clock (≥ 0), for a host whose lava lamp and cirrus are on screen
    /// at once (mid-fade) and animate at independent speeds. `nil` = `inViewSkyTime ?? time`.
    var inViewCirrusTime: Float? = nil
    /// Per-FRAME cirrus threads weight (0 veil … 1 threads) for the in-view sky, overriding the
    /// cloud renderer's `cirrusThreadsWeight` (throttled sky uniforms). `nil` = the uniform.
    var inViewCirrusThreads: Float? = nil
    /// Frame-level gain on every instance's `tagGlow.x` weight (petal glow). 1 by default, so a
    /// host that only sets per-instance weights still glows; instances with weight 0 — every
    /// host that never opts in — are untouched by any value.
    var tagGlowGain: Float = 1
    /// Field sink, 0…1: instances with a sink depth (`tagGlow.y`) slide that far below ground at 1
    /// — a host fades a whole field of objects into / out of the ground with one float per frame
    /// (see `fieldSinkOffset` in IlluminatoramaGBuffer.metal). 0 (the default) is an exact no-op.
    /// Applied in the G-buffer and shadow vertex stages only: RT scene data, glass, selection and
    /// impostor paths still see the UN-sunk pose (as with wind sway) — an RT host would reflect
    /// a sunk field. Give a sunk object's parts one `tagGlow.z` key built from WORLD coordinates:
    /// the fallback hash reads the (render-space) origin, which a floating origin moves.
    var fieldSinkProgress: Float = 0
    /// Resolution the in-view cloud + atmosphere march runs at, as a fraction of the canvas
    /// (1 = full res, the default — byte-identical). 0.5 marches a quarter of the rays into an
    /// off-screen target and depth-aware-upsamples it into the sky pixels: clouds and the
    /// nishita sky are smooth at that scale, and the march is the most expensive pass in an
    /// outdoor scene (measured 30 ms of a 2880×1620 frame in Superbloom Hills).
    var inViewCloudsResolutionScale: Float = 1
    /// Baked noise volume + packed SkyUniforms + burst-light buffer, supplied by
    /// the scene's `VolumetricCloudRenderer` (reused, not re-packed).
    var cloudNoiseTexture: MTLTexture?
    var cloudSkyUniforms: MTLBuffer?
    var cloudBurstLights: MTLBuffer?
}
