import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - RTKnobs

/// Stored state for the renderer's `rt` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `RTKnobs` (`kRT`) and forwards each name to it from
/// `IlluminatoramaRenderer+RTKnobs.swift`; callers see no difference.
@MainActor
final class RTKnobs {
    typealias GIRoomFillHandoff = IlluminatoramaRenderer.GIRoomFillHandoff
    typealias RTSunOwnership = IlluminatoramaRenderer.RTSunOwnership

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
    var rtEnabled: Bool = false
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
    var rtOpaqueLightingEnabled: Bool = true
    /// Direction TOWARD the sun (normalised), used by the RT pass.
    var rtSunDirection: SIMD3<Float> = simd_normalize(SIMD3(0.5, 1.0, 0.3))
    /// Sun radiance (pre-multiplied intensity) for the RT direct term.
    var rtSunColor: SIMD3<Float> = SIMD3(1.0, 0.97, 0.92) * 6.0
    /// Sun angular radius in radians — penumbra softness. ~0.0047 rad is the
    /// real sun; larger values give softer contact shadows.
    var rtSunSoftnessRad: Float = 0.02
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
    var rtSunSoftShadowsEnabled: Bool = false
    /// Strength of the one-bounce indirect contribution.
    var rtGIStrength: Float = 1.0
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
    var rtGIReplacesDiffuseSky: Bool = false
    var rtGIOwnsRoomFill: GIRoomFillHandoff = []
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
    var rtGIHitFill: Bool = false
    /// **The path-traced GI lane (Daydream DH-0989; DH-0642's reference is the same lane at a deep
    /// budget).** When true each of `rtGIRays` is a whole multi-bounce path — next-event estimation
    /// of the sun, the local lights and ONE window aperture per vertex, a cosine continuation,
    /// Russian roulette — and it owns the pixel's ENTIRE diffuse indirect: the deferred pass hands
    /// over its diffuse IBL (interior bands and room gains included), its ambient supplement and
    /// its area lights (this host's area lights are window portals, which the path samples as
    /// apertures onto the real sky), and the traced pass subtracts them where it ran. Implies the
    /// `rtGIReplacesDiffuseSky` hand-off. Default `false`: byte-identical for every host that does
    /// not opt in. A host that opts in should run `rtGIStrength` at 1 — the estimate is physical.
    var rtPathTracedGI: Bool = false
    /// Continuation bounces per path (Russian roulette also ends paths from the third vertex).
    var rtPathBounces: Int = 6
    /// Luminance cap on one path's estimate (scene-linear, before exposure); 0 ⇒ no clamp. A
    /// path through a sunlit patch seen past a doorway is the classic firefly; the cap trades a
    /// little energy in those paths for a still that converges in the frames it is given.
    var rtPathClamp: Float = 0
    /// **DH-1011 — how a path vertex samples its lights.** True: the aperture sample and the
    /// cosine continuation are combined by multiple importance sampling (power heuristic) instead
    /// of partitioning the hemisphere — a continuation that leaves through a portal is kept,
    /// weighted, not dropped — which in turn lets the aperture PICK leave out the portals of other
    /// rooms (light-layer mask; the continuation carries their light at full weight); and the area
    /// emitters are ONE pick ∝ their unshadowed contribution with ONE Sobol shadow ray instead of
    /// `secondaryAreaShadowRays` white-noise rays each. Same integrand, so the converged picture
    /// does not move; only its variance does. False: the DH-0989 estimator, byte-identical (the
    /// A/B lever).
    var rtPathLightSampling: Bool = true
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
    var rtPathLampShadows: Bool = true
    /// DH-1011 INSTRUMENTS, OR'd into the path flags — never set by a host. 128: the frame is
    /// replaced by the primary surface's traced local-light estimate alone (traced lamp visibility,
    /// as a ratio of two frames); 64: a continuation that hits an emissive surface adds nothing
    /// (the glowing shades' share of the noise, by its absence); 32 / 16 / 8: no local lights /
    /// no sun / no window apertures at the path's vertices (the noise budget by source).
    var rtPathDebugFlags: UInt32 = 0
    /// Daydream DH-0952 INSTRUMENT — the one-bounce TLAS pass's cost by ablation, OR'd into its path
    /// flags on EVERY lane; never set by a host (the picture is wrong under each bit). 1<<24: no sun
    /// shadow ray at GI / reflection hits · 1<<25: GI hits traced but not shaded · 1<<26: reflection
    /// hits not shaded · 1<<27: GI rays cut at 4 m. See `kRTAblate*` in IlluminatoramaRTInstanced.metal.
    var rtTLASAblationFlags: UInt32 = 0
    /// **Daydream DH-1588 — the sun cascades PROVE a GI / reflection hit's shadow; the TLAS answers the rest.**
    /// The one-bounce TLAS pass traces one hard sun shadow ray at every GI and reflection hit — on a
    /// software-RT GPU ~112 ms of a ~300 ms settled refine, nearly all of it the fixed cost of one
    /// incoherent TLAS descent per ray. With this on, a hit whose 3×3 cascade texels all hold a caster at
    /// least `rtHitSunCascadeMargin` (+ a slope term) nearer the sun is lit as shadowed without a ray;
    /// every other hit (lit, penumbra, outside the cascades, grazing) traces exactly as before. It can
    /// only darken, and only where the RASTER casts a shadow the TLAS lacks (raster-only casters, a gap
    /// under a texel) — see `illumiSunCascadeProvesShadow`. Default false ⇒ byte-identical.
    var rtHitSunCascadeCull: Bool = false
    /// The proof's depth margin (metres) between the cascade's caster and the hit (DH-1588).
    var rtHitSunCascadeMargin: Float = 0.25
    /// The proof's slope term, in cascade texels × (1 + tan θ) (DH-1588).
    var rtHitSunCascadeSlopeTexels: Float = 3
    /// DH-1588 INSTRUMENT — prove, then trace anyway and count (`rtHitSunCascadeCullStats`); the
    /// picture is the shipped one. Never set by a host.
    var rtHitSunCascadeAudit: Bool = false
    /// **Daydream DH-0952 — probe only where the probe can move anything.** The snap probe below
    /// (`rtSurfaceSnapEnabled`) is a full TLAS descent per pixel in every traced pass — measured on
    /// an M1 Max at 1920×1200: 42 ms of the TLAS lighting pass, ~70 ms of the whole frame — and on a
    /// surface that is not a substrate it finds nothing. A host whose substrates all face UP (ground,
    /// paving) sets this to the lowest normal y one of them can show: a pixel whose G-buffer normal
    /// y is at or below it is not probed, unless 4 depth ULPs there span more than
    /// `rtSurfaceSnapClearance` (the probe's other job — depth rounding at range — stays on for
    /// distant walls). Default −2 ⇒ every pixel probed, byte-identical.
    var rtSurfaceSnapMinUpY: Float = -2
    /// The origin clearance (metres) the traced passes' rays already leave with; see `rtSurfaceSnapMinUpY`.
    var rtSurfaceSnapClearance: Float = 0.0005
    /// **DH-1293 INSTRUMENT — floor the lamps' `1/d²` at path vertices** (metres; 0 ⇒ off,
    /// byte-identical). A lamp / can / sconce reaching a path vertex closer than this lights it as
    /// if it were this far away. BIASED by construction — it removes the near-field energy it
    /// floors — so it answers "how much of a still's noise is vertices landing next to a bulb",
    /// and is never set by a host. Path vertices only: the primary's lamps are the deferred pass's.
    /// NEGATIVE ⇒ the exclusion form: a lamp adds nothing to a vertex closer than |value|.
    var rtPathLampMinDistance: Float = 0
    /// **DH-1293 — a finite SOURCE size for the lamps at path vertices** (metres; 0 ⇒ off,
    /// byte-identical). The falloff becomes `1/(d² + r²)` — the exact on-axis irradiance of a
    /// luminous disc of radius r — combined with each light's own `softRadius` by max. A physical
    /// lever (a can's aperture, a shade's opening has size), default off until it earns its place.
    var rtPathLampSourceRadius: Float = 0
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
    var rtPathLampBulbs: Bool = true
    /// Cone-sampled shadow rays per pixel per frame for `rtSunSoftShadowsEnabled`'s traced
    /// sun visibility, host-clamped (`rtSunShadowRayCount`'s write, below) and shader-clamped
    /// (`rtSunSoftVisibility`, IlluminatoramaLighting.metal) to 1…32. DH-0856 measured this
    /// alone does not close the gap to the cascade shadow map's edge quality at any practical
    /// budget — the Monte-Carlo estimator's noise floor drops sharply from 1 to 4 rays, then
    /// only slowly (single-digit percent per doubling) from 4 to 32, so raising the shipped
    /// default costs real per-frame GPU time for little further gain. Default 4, matching the
    /// budget every existing host already renders at.
    var rtShadowRays: Int = 4
    /// **S4.3 — ray-traced PORTAL (area-light) visibility, in the same deferred-kernel variant
    /// as the soft sun shadows.** When > 0 and `rtSunSoftShadowsEnabled` is live, every
    /// shadow-casting area light (`shadowSliceIndex >= 0` — a host's window portals) has its
    /// single-perspective 512² PCF map REPLACED by this many shadow rays per pixel toward
    /// jittered points on the emitting rectangle, traced against the TLAS. Exact in every
    /// direction (a 150° perspective from the portal centre cannot see a niche beside the
    /// window), and a penumbra that scales with the portal's real size — the daytime interior
    /// shadow a room's dominant source has to cast. Per-frame Monte-Carlo like the sun rays,
    /// so it belongs to a still that accumulates. 0 (default) keeps the PCF path byte-identical.
    var rtAreaShadowRays: Int = 0
    /// **Daydream DH-0952 — don't light what the traced pass subtracts.** When a frame hands the
    /// window APERTURES to the traced GI (`rtGIOwnsRoomFill` contains `.apertures`, or the path lane
    /// owns the indirect), the deferred kernel's aperture term — LTC form factor plus one visibility
    /// ray per portal per pixel — goes into `color` and into the hand-off texture the traced pass
    /// takes back out, so it cancels exactly. True (default): the kernel skips it (measured on the
    /// settled refine: the deferred pass's single biggest cost). False: computed and cancelled as before
    /// — the A/B lever. No effect on a frame without the hand-off.
    var skipHandedApertures: Bool = true
    /// **S4.4 — a STILL may build the extracted-scene TLAS under the glass-arm caps.** The
    /// live-RT caps (128 mesh groups / 150 k triangles) exist because a live loop's first
    /// rebuild of a heavy scene never returns; a still that accumulates pays that rebuild
    /// once, off the vsync budget, so it can carry the same 512-group / 2.5 M-triangle
    /// ceiling the glass-only arm already trusts. A furnished document (200+ groups) is
    /// exactly what the live caps auto-disable RT GI on — which is why the shipped photo
    /// export never had bounce. Default `false`: every live host is byte-identical.
    var rtStillCapsRelaxed: Bool = false
    // ── RTAO — ray-traced ambient occlusion on the near field (DH-0528) ────────────────
    /// When true AND a TLAS is live, the AO pass traces `rtaoRays` cosine-weighted hemisphere
    /// rays per half-res texel against the real geometry (`illumi_rtao_tlas`) instead of the
    /// screen-space GTAO march, writing the same raw AO texture into the same bilateral +
    /// temporal chain. The reach is a TRUE world-space distance, so one metre grounds a sofa
    /// foot and stays tight at a drywall junction. Per-frame Monte-Carlo: meant for a still
    /// that accumulates. Falls back to GTAO on any frame without a TLAS (`rtaoDidRunLastFrame`
    /// is the record). Default `false`: byte-identical for every host that never opts in.
    var rtaoEnabled: Bool = false
    /// 0…1 AO strength, the `ssaoIntensity` contract.
    var rtaoIntensity: Float = 1.0
    /// Occlusion reach in world metres (the ray `max_distance`).
    var rtaoRadius: Float = 0.5
    /// Cosine-hemisphere rays per half-res texel per frame (clamped 1…32).
    var rtaoRays: Int = 8
    /// Whether the LAST encoded AO pass was the traced one — a gate's proof that RTAO fired
    /// rather than silently falling back to the screen-space march.
    var rtaoDidRunLastFrame: Bool = false
    var rtGIRays: Int = 4
    /// Soft sun specular strength on the RT direct term.
    var rtSpecStrength: Float = 0.25
    var rtSupported: Bool = false
    /// Who computes the sun's direct term while an RT lighting pass is running.
    /// See `RTSunOwnership`. Default `.automatic`.
    var rtSunOwnership: RTSunOwnership = .automatic
    // ── RT diffuse denoiser ──────────────────────────────────────────
    // The RT pass writes its noisy diffuse (soft shadow + 1-bounce GI) term to
    // its own buffer; this depth+normal-guided bilateral cleans it BEFORE it
    // reaches the HDR composite + TAA, so temporal accumulation gets a far less
    // noisy per-frame input (the visible "dithered" grain under orbit is the
    // raw ~4-spp estimate showing through when TAA history is rejected by
    // motion). The sharp reflection term is composited un-filtered. Off → the
    // raw diffuse is composited straight through (pre-split additive behaviour).
    var rtDenoiseEnabled: Bool = true
    /// Bilateral half-window in pixels (full-res). 2 ⇒ 5×5; 3 ⇒ 7×7.
    var rtDenoiseRadius: Int = 2
    /// Depth edge-stop sensitivity (larger = tighter boundary preservation).
    var rtDenoiseDepthSensitivity: Float = 3.0
    /// Normal edge-stop exponent (larger = more sensitive to curvature).
    var rtDenoiseNormalSharpness: Float = 16.0
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
    var rtGITemporalEnabled: Bool = false
    /// Current-frame weight in steady state. 0.06 ≈ a 16-frame exponential
    /// window. Smaller = cleaner/slower to react; larger = noisier/faster.
    var rtGITemporalBlend: Float = 0.06
    /// Daydream DH-0887 — STILL mode for the GI accumulator: with the camera frozen (a photo, the
    /// settled canvas's refine) average every frame's GI as an unbiased running mean (α = 1/N,
    /// no clamp) instead of the moving-camera EMA, which forgets samples and clamps toward each
    /// noisy frame. The host must reset (`resetTemporalHistory`, or toggle `rtGITemporalEnabled`)
    /// when the picture changes. Default `false`: byte-identical for every other host.
    var rtGIStaticAccumulation: Bool = false
    /// Daydream DH-1388 / DH-1386 — with `rtGIStaticAccumulation`, do NOT keep the GI's own running mean:
    /// hand every frame's own (albedo-demodulated) sample to the à-trous cascade and let the colour
    /// resolve's frozen-camera average be the ONE mean. Otherwise the composite of frame k carries
    /// F(G_k), G_k = mean(g_1..g_k), and the colour average of those composites weights GI sample i by
    /// (H_K − H_{i−1})/K (harmonic) — ~K/2 effective GI samples in a K-frame still instead of K. The
    /// cascade is linear in its input (still lanes have no luminance edge-stop), so filter-then-average
    /// equals average-then-filter; only the weights change. Needs `taaStillAccumulation` to be the
    /// averager. Default `false`: byte-identical for every host.
    var rtGIStillPerFrame: Bool = false
    /// Neighborhood clamp width (sigmas). WIDE on purpose (~4): a tight clamp
    /// re-injects low-frequency crawl by yanking clean history toward the noisy
    /// current 3×3 mean. Only meant to catch gross on-screen disocclusions.
    var rtGITemporalClamp: Float = 4.0
    /// Daydream DH-1344 — the path-traced still's RADIANCE CACHE cell size (m); 0 ⇒ off (byte-identical).
    /// Build frames' paths write the light they find leaving each surface into a hashed world grid; later
    /// frames' paths end at the grid instead of continuing a long random walk — in a pale room, where most
    /// light has bounced four or more times, that walk is where the per-pixel noise (the mottle) came from.
    /// The build frames must be held out of the still's accumulation (their full walks are the noisy ones):
    /// `resetRTGIAccumulation(keepingRadianceCache: true)` after them. Only with `rtPathTracedGI`.
    var rtPathRadianceCacheCell: Float = 0
    /// Frames the cache BUILDS (writes only) after it is cleared, before paths start reading it. 4 measured
    /// the same as 8 and 16 for wall mottle against a 4000-path reference (Oak Credenza Hall 1.52 % at all
    /// three; Sunset Gallery within 0.06 points), and each build frame is a full-walk frame (~3.5 s at hero size).
    var rtPathRadianceCacheBuildFrames: Int = 4
    /// Paths per pixel on a build frame.
    var rtPathRadianceCacheBuildRays: Int = 1
    /// Daydream DH-1344 — whether a STILL's GI filter (`rtGIStaticAccumulation`) keeps the SVGF
    /// luminance edge-stop. The stop weighs neighbours by their NOISY luminance, so it rejects the rare
    /// bright samples a path tracer's mean depends on: against an unfiltered 256-path reference it
    /// darkened every surface by 0.09–0.24 st and grouped the leftover noise into blotches, while
    /// protecting no in-plane indirect shadow edge (Oak Credenza Hall, Sunset Gallery, 2026-10-14).
    /// Depth and normal stops keep geometric edges either way. Default `true`: byte-identical.
    var svgfStillLuminanceStop: Bool = true
    /// Daydream DH-1016 — relative-noise threshold of the still filter's variance gate (`√Var / lum`);
    /// 0 = off (default, byte-identical). See `varGate` in IlluminatoramaSVGF.metal.
    var svgfStillVarianceGate: Float = 0
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
    var rtReflectionsEnabled: Bool = false
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
    var rtTLASForReflections: Bool = false
    /// Master scale on the reflection contribution. 0 = off.
    var rtReflStrength: Float = 1.0
    /// Max reflection ray length, metres.
    var rtReflMaxDistance: Float = 40.0
    /// Surfaces rougher than this skip reflections (glossy fades out).
    var rtReflRoughnessCutoff: Float = 0.6
    /// Glossy samples per pixel (1 = mirror-sharp; TAA denoises the rest).
    var rtReflRays: Int = 1
    /// **Traced rays leave from the surface the camera sees** (Daydream DH-0715). The substrate's
    /// depth bias above buries its depth-rebuilt world position BEHIND the surface — ≈ 10 cm down
    /// the view ray at 50 m — so every traced sun, portal, bounce and AO ray from the ground started
    /// underground and hit it from below: Daydream's settled canvas and exports put a sunlit noon
    /// lawn in full shadow (sRGB luma 40 vs 153 on the shadow-map path). With this on, each traced
    /// pass first probes from the rebuilt point back toward the camera (`illumiSnapToVisibleSurface`)
    /// and moves the ray origin onto the surface it meets; a point not buried meets nothing and is
    /// unchanged. Effective only while `substrateMeshKinds` is non-empty (reach: twice
    /// `substrateDepthBias`), so a host with no substrate is byte-identical.
    var rtSurfaceSnapEnabled: Bool = true
    /// Master switch for the ray-traced glass path. OPT-IN (default off): a scene
    /// that wants AAA glass sets this true, which makes the renderer add its glass
    /// meshes to the TLAS and render them through `illumi_glass_rt_fs` (true
    /// refraction/reflection). Off → the lightweight Fresnel+sky fallback
    /// (`illumi_glass_fallback_fs`), with NO TLAS cost — so existing glass scenes
    /// (incidental tube/wall glass) are unaffected until they opt in. The extractor
    /// sets this automatically when it finds a glass-flagged `SCNMaterial`.
    var rtGlassEnabled: Bool = false
    /// Dielectric bounce cap for the refraction path (entry+exit ≈ 2; raise for
    /// lens-through-lens stacks). Clamped to 10 in-shader.
    var rtGlassMaxBounces: UInt32 = 6
    /// Sun shadow rays cast when an opaque hit inside the glass trace has to be
    /// re-shaded (cache off). 1 is usually enough — TAA cleans the penumbra.
    var rtGlassShadowRays: UInt32 = 1
    /// Global gate for chromatic dispersion. Per-instance `dispersion` still has
    /// to be > 0; this lets a host disable the 3× cost wholesale.
    var rtGlassDispersionEnabled: Bool = true
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
    var rtGlassScreenSpaceTransmission: Bool = true
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
    var rtGlassDefocusTraceFactor: Int = 1
    /// The defocus trace's gate, in LOW-RES texels of CoC radius: a pixel takes the reduced trace
    /// when its |CoC| ≥ this × k px. At 4 a k-px bilinear reconstruction sits under a ≥ 4k-px
    /// disc, which widens the blur by ~2 % (σ² adds: (0.41k)² against (2k)²).
    var rtGlassDefocusMinCoCTexels: Float = 4
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
    var svgfEnabled: Bool = false
    /// Daydream DH-0887 — SVGF for a frozen-camera accumulation (a still, the settled canvas's
    /// refine) regardless of the process-wide `IlluminatoramaSharedSettings.svgfEnabled`, which
    /// is re-read every frame and shared by every window. Runs `svgfStillLevels` levels.
    /// Default `false`: byte-identical for every other host.
    var svgfStillForced: Bool = false
    /// À-trous levels while `svgfStillForced` (2: the still lane's measured trade between
    /// residual noise and edge ringing, DH-0858).
    var svgfStillLevels: Int = 2
    /// Daydream DH-0966 — while the GI is a static running mean, keep it as IRRADIANCE: the temporal pass
    /// divides each frame's G-buffer albedo out of that frame's own samples, and the last à-trous level
    /// multiplies the composing frame's albedo back in. The RT pass bakes the albedo into the GI, so a
    /// wide cascade over the modulated term blurs wood grain and fabric weave along with the noise.
    /// Default `true`; moving-camera frames never enter static accumulation and are untouched.
    /// `VIZ_ILLUMI_SVGF_DEMOD=0` A/Bs it.
    var svgfDemodulateAlbedo: Bool = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF_DEMOD"] != "0"
    /// Number of à-trous cascade levels (1–5). Three levels cover a spatial
    /// reach of 1+2+4 = 7px radius; five levels cover 1+2+4+8+16 = 31px.
    var svgfLevels: Int = 3
    /// Luminance variance weight (σ_L): smaller values produce tighter edge
    /// preservation; larger values allow more spatial averaging across edges.
    var svgfSigmaL: Float = 4.0
    /// Depth edge-stopping weight (σ_Z). Default 1.0.
    var svgfSigmaZ: Float = 1.0
    /// Normal edge-stopping exponent (σ_N). Higher = tighter. The clean bilateral
    /// fallback uses 16; at 128 the `dot(n,nN)^σN` weight amplifies the fp16
    /// oct-encoded normal g-buffer's quantization steps into structured wNorm
    /// jumps — banding contours on a smooth wall (issue #65). 16 matches the
    /// bilateral and keeps the weight smooth across coplanar fp16 normals.
    var svgfSigmaN: Float = 16.0
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
    var svgfLumFloor: Float = 0.6
    /// Times the BLAS cache served an entry built from a DIFFERENT mesh object than the one
    /// asked for — i.e. a recycled address collided with a dead entry. Structurally
    /// impossible while entries own their meshes, and gated at 0 by
    /// `RTMeshCacheIdentityTests`, which is what proved this was happening.
    var rtStaleBLASReuseCount = 0
}
