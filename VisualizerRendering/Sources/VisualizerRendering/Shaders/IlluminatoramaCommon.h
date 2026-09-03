// ── ILLUMINATORAMA ───────────────────────────────────────────────────────────
//
//
// Phase 1: G-buffer pass, deferred PBR resolve, bloom, tonemap.
// Phase 2: SSAO + SSR.
// Phase 2.5: Cascaded shadow maps for the directional sun light.
// Phase 2.7: Motion vectors + temporal anti-aliasing (TAA). G-buffer writes
//          a screen-space velocity attachment; the resolve kernel reprojects
//          the previous frame's HDR and blends it into the current via a
//          neighborhood-clamped exponential history.
// Phase 3: Sky-probe IBL (sky equirect → irradiance cube + GGX-prefiltered
//          specular cube; consumed by the deferred lighting pass as diffuse +
//          specular IBL and as SSR's sky-miss fallback).
// Phase 3.2: Split-sum DFG LUT — pre-integrated GGX BRDF baked once at
//          startup into a 2D (NdotV, roughness) → (scale, bias) texture.
//          Lighting kernel uses `F0 * scale + bias` for the F0-weighted
//          spec response, replacing the Lagarde roughness-Schlick
//          approximation that Phase 3.0 shipped as a placeholder.
// Phase 3.3: DDGI two-bounce. Trace kernel now samples the previous-frame
//          irradiance atlas at the hit point and adds it (Lambertian, ×
//          albedo) to the recorded radiance. Host ping-pongs both atlases
//          so the kernel's read-only `irrAtlasPrev` / `depAtlasPrev` slots
//          never alias the update kernels' read_write writes within a
//          single encoder. Toggle via `ddgi.twoBounceEnabled`.
//
// Swift mirrors of every struct in this file live in IlluminatoramaTypes.swift.
// When changing one side, change both.

#pragma once

// ── ILLUMINATORAMA — SHARED FRAME TYPES AND MATH ────────────────────────────
//
// Included by every Illuminatorama*.metal pass file. Holds what more than one
// pass needs: the frame uniforms, the light/instance/vertex structs, the
// oct-normal codec, depth→position reconstruction, the shared GGX geometry
// term, and the equirect sky sampler. A helper used by exactly ONE pass lives
// in that pass's file, not here.

#include <metal_stdlib>
#include "IlluminatoramaNightSky.h"
using namespace metal;


// ── Shared structs (mirror IlluminatoramaTypes.swift) ────────────────────────

struct FrameUniforms {
    float4x4 viewProjection;
    float4x4 view;
    float4x4 projection;
    float4x4 invViewProjection;
    float4x4 invProjection;
    float4x4 invView;
    float3   cameraWorldPos;
    // Aerial perspective — extinction coefficient σ (1/m) for the distance-haze blend the
    // deferred lighting pass applies to its FULL composite (never the isolated debug
    // terms): `color = mix(airlight, color, exp(-σ·viewDist))`, airlight sampled from the
    // prefiltered sky cube at the view azimuth's horizon so it follows time of day with no
    // extra plumbing. Repurposes the former `_padCamera` slot. 0 — the default, and every
    // scene that never opts in — is an exact no-op. A clear-day Rayleigh σ is ~1e-4
    // (visibility ≈ 39 km: 4 % haze at 400 m); this is the physically-plausible scale,
    // not a fog effect.
    float    aerialPerspectiveDensity;   // was _padCamera
    float3   directionalLightDir;
    float    _padDir;
    float3   directionalLightColor;
    float    _padColor;
    float3   ambientColor;
    float    exposure;
    float    bloomThreshold;
    float    bloomIntensity;
    uint     pointLightCount;
    float    time;
    // Phase 2 — SSAO + SSR knobs.
    float    ssaoIntensity;
    float    ssaoRadius;
    float    ssrIntensity;
    float    ssrMaxDistance;
    float    ssrThickness;
    uint     ssrMaxSteps;
    float    _padPhase2A;
    float    _padPhase2B;
    // Phase 3 — Sky-probe IBL. Sky imagery itself arrives as an external
    // equirect HDR texture (typically VolumetricCloudRenderer's output);
    // these are the per-frame knobs the lighting + bake kernels need.
    float    iblIntensity;
    uint     iblPrefilteredMipCount;
    uint     iblEnabled;
    // Lamp-shade fabric translucency strength (DH-0458). 0 = OFF (default) → the
    // shade thin-sheet transmission branch in illumi_lighting_fs is an EXACT no-op
    // for every scene; only a shade mesh tagged with colour alpha in (0.60, 0.66]
    // (→ normalRoughness.w ≈ 0.62) reads it. Repurposes the former `_padPhase3`
    // pad — same 4 bytes at the same offset, so stride is unchanged. Field-for-
    // field mirror of the Swift IlluminatoramaFrameUniforms.shadeTransmission.
    float    shadeTransmission;
    // Phase 2.5 — Cascaded shadow maps for the directional sun light. Three
    // cascades; each has its own light-space view-projection matrix. The
    // lighting kernel picks a cascade per pixel from view-space Z and does a
    // PCF compare against the appropriate slice of a depth2d_array shadow map.
    float4x4 shadowVP0;
    float4x4 shadowVP1;
    float4x4 shadowVP2;
    // (x,y,z) = far-Z of cascades 0/1/2 in view space, expressed as positive
    // distances (the actual view-space Z values are negative).
    float4   cascadeSplitsView;
    float    shadowBias;
    float    shadowSlopeBias;
    uint     shadowEnabled;
    uint     shadowPcfRadius;
    // Phase 2.7 — Motion vectors + TAA. `previousViewProjection` is the
    // *jittered* VP from the previous frame so the rasterized history sample
    // line up exactly with the motion vector reprojection.
    float4x4 previousViewProjection;
    float    taaHistoryBlend;     // blend factor for the *current* sample (≈0.05–0.10)
    uint     taaEnabled;
    uint     taaIsFirstFrame;     // 1 = no valid history yet, force blend = 1
    // Phase 3.2 — split-sum DFG LUT toggle. 0 = fall back to Lagarde's
    // roughness-Schlick (Phase 3.0 behaviour). The LUT itself is bound at
    // texture(10) of illumi_lighting; bake kernel is illumi_dfg_bake.
    // Repurposes the Phase 2.7 pad slot — same byte width (4 bytes), same
    // 16-byte struct alignment.
    uint     dfgLUTEnabled;
    // Phase 3.6 / 4.10 — spot light count + per-spot shadow bias. Count
    // + float + two uint pads = 16 bytes (one cluster); matches the
    // Swift struct on the byte.
    uint     spotLightCount;
    float    spotShadowBias;
    // Phase 4.15 — saturation knobs. `tonemapSaturation` runs after the
    // ACES curve in `illumi_tonemap`; `iblDiffuseSaturation` runs on the
    // IBL diffuse contribution in the lighting kernel before it composites
    // into `indirect`. Both default to >1.0 in the host struct so a fresh
    // renderer matches SCN's perceived saturation; setting either to 1.0
    // disables it. Repurposes the prior `_padSpot0/_padSpot1` slots
    // (same 4 bytes each, same 16-byte cluster).
    float    tonemapSaturation;
    float    iblDiffuseSaturation;
    // Phase 4.21 — auto-exposure cluster (16 bytes). `autoExposureEnabled`
    // gates the kernel's path; `autoExposureTargetEV` is the log2
    // luminance the exposure aims for (0 = mid-grey 18%, -1 = one stop
    // darker); `autoExposureHalfLife` is how many seconds the EMA takes
    // to halve the gap to the new target.
    uint     autoExposureEnabled;
    float    autoExposureTargetEV;
    float    autoExposureHalfLife;
    // Per-term split-render diagnostic (repurposes the _padPhase421 slot).
    // 0 = normal composite; non-zero isolates ONE term in illumi_lighting:
    //   1 direct sun · 2 point/emissive lights · 3 spot lights ·
    //   4 diffuse IBL · 5 specular IBL · 6 G-buffer emission · 7 ambient supp.
    uint     debugTerm;
    // Desaturates the indirect FILL (diffuse-IBL irradiance + ambient
    // supplement) toward its own luminance for highly-saturated probes.
    // 0 = off. Targets the colored-environment flood (Pizza red): real
    // inter-reflected fill is never as monochromatic as a pure-colour emitter.
    float    iblDiffuseDesaturation;
    // Phase 4.39: AAA spatiotemporal denoiser knobs. Two 16-byte clusters
    // that grow the struct by 32 bytes past the 12-byte implicit padding gap
    // left by iblDiffuseDesaturation (matches the Swift-side comment).
    uint     ssaoDenoiseEnabled;    // gates SSAO bilateral + temporal passes
    float    ssaoTemporalBlend;     // AO history blend (high = more stable)
    uint     ssrDenoiseEnabled;     // gates SSR temporal accumulation
    float    ssrTemporalBlend;      // SSR history blend
    uint     ssaoIsFirstFrame;      // 1 = no valid AO history yet
    uint     ssrIsFirstFrame;       // 1 = no valid SSR history yet
    uint     debandDitherEnabled;   // 1 = TPDF dither before 8-bit store
    // Phase 3.4 — per-pixel DDGI irradiance EMA cache.
    uint     ddgiIrrCacheEnabled;   // 1 = blend with history, 0 = fresh probe blend
    float    ddgiIrrCacheBlend;     // alpha for history blend (0=freeze, 1=always fresh)
    // Leaf thin-sheet transmission strength (issue #58). 0 = OFF (the default for
    // every scene) so the foliage-flag branch in illumi_lighting_fs is a no-op
    // unless a scene opts in. Foliage is flagged in normalRoughness.w. Lands in
    // the 12-byte trailing pad ddgiIrrCacheBlend opened, so stride is unchanged.
    float    leafTransmission;
    // #60 task 5 — rectangular LTC area-light count (buffer 4 of illumi_lighting).
    // Lands in the 4-byte trailing implicit pad; stride stays 960.
    uint     areaLightCount;
    // #60 task 5 increment 2 — 1 ⇒ use the validated LTC specular LUT for area
    // lights, 0 ⇒ most-representative-point fallback. New cluster; stride → 976.
    uint     areaLTCEnabled;
    // #60 task 5 — number of SECONDARY directional lights in the
    // `extraDirectionals` buffer (bound at buffer(5) of illumi_lighting). The
    // PRIMARY (first / key) directional still lives in `directionalLightDir/Color`
    // with the cascade rig; these are the additional fill/back directionals that
    // were formerly collapsed into a flat hemispheric ambient term (the 4.20
    // "ambient fold" shortcut). They now shade with a real NdotL + GGX-specular
    // BRDF (no shadow — SCN fill/back lights ship `castsShadow = false`). Lands in
    // the same trailing 16-byte cluster as areaLightCount/areaLTCEnabled (3rd uint
    // of 4), so the struct stride stays 976.
    uint     directionalLightCount;
    // Plush material (Teddy Bear Press): fur edge-sheen strength + backlit thin-
    // fabric SSS strength. Both 0 by default → the plush branch in illumi_lighting_fs
    // is an EXACT no-op for every scene; only the teddy mesh tags vertices with the
    // plush flag (colour alpha ≈ 0.55 → normalRoughness.w ≈ 0.55h). NEW 16-byte
    // trailing cluster (stride 976 → 992); field-for-field mirror of the Swift
    // IlluminatoramaFrameUniforms.
    float    plushSheen;
    float    plushTransmission;
    // Lens-style transverse chromatic aberration strength (tonemap pass). 0 = OFF
    // (default) → an exact no-op. Repurposes the former _padPlush0 slot; stride
    // is unchanged. Mirrors IlluminatoramaFrameUniforms.chromaticAberration.
    float    chromaticAberration;
    // Spherical-aberration radial blur strength (0 = OFF → exact no-op).
    // Repurposes the former _padPlush1 slot; stride is unchanged.
    float    sphericalAberration;
    // Axial chromatic aberration ("purple fringing"): edge-halo strength (0 = OFF
    // → exact no-op) + dark-side sRGB tint; the bright side gets the complement.
    // NEW 16-byte cluster (stride 992 → 1008). Four scalar floats (NOT float3 +
    // pad) — mirrors IlluminatoramaFrameUniforms.fringe/fringeTintR/G/B.
    float    fringe;
    float    fringeTintR;
    float    fringeTintG;
    float    fringeTintB;
    // Vignette + film grain (issue #65 post-FX). Applied at the very tail of the
    // tonemap pass (after ACES + saturation + fringe, before the deband dither).
    // Both 0 by default → an EXACT no-op for every existing scene. NEW 16-byte
    // cluster (stride 1008 → 1024); field-for-field mirror of the Swift
    // IlluminatoramaFrameUniforms.
    float    vignetteStrength;   // 0 = off; corner-darkening amount (0..1)
    float    vignetteExtent;     // 0..1 radius (frac of half-diagonal) kept bright
    float    filmGrainStrength;  // 0 = off; grain amplitude
    float    filmGrainSize;      // grain cell size in OUTPUT px (>= 1)
    // Issue #65 — 3D colour-grade LUT (texture(2) of illumi_tonemap_fs). 0 = OFF
    // → exact no-op. `colorLUTSize` is the per-axis resolution for the half-texel
    // sampling correction. NEW 16-byte cluster (stride 1024 → 1040); mirror of
    // the Swift IlluminatoramaFrameUniforms.
    float    colorLUTAmount;     // 0 = off; blend of graded over ungraded
    float    colorLUTSize;       // per-axis LUT resolution (e.g. 33)
    // Velocity-buffer motion blur (issue #65). Repurposes the two trailing post-FX
    // pads — stride unchanged. 0 strength → exact no-op (the tonemap keeps its plain
    // 4-tap path). `motionBlurStrength` scales the per-frame screen velocity (≈ a
    // shutter fraction); `motionBlurMaxPx` clamps the streak length in OUTPUT px.
    float    motionBlurStrength; // was _padPostFX0
    float    motionBlurMaxPx;    // was _padPostFX1
    // Screen-space contact shadows (issue #65). A short screen-space ray march
    // toward the PRIMARY directional sun, in the deferred lighting pass, that
    // catches the fine contact occlusion the cascaded shadow maps + RT soft
    // shadows miss at object-base scale (a chip on felt, an egg on a floor).
    // `contactShadowStrength` 0 = OFF → an EXACT no-op (the lighting kernel
    // skips the march entirely), matching the vignette/grain/LUT post-FX
    // pattern. `contactShadowLength` is the march reach in WORLD units;
    // `contactShadowSteps` the sample count (8–16); `contactShadowThickness`
    // the occluder-depth window in world units (acne / over-darkening guard).
    // NEW 16-byte cluster (stride 1040 → 1056); field-for-field mirror of the
    // Swift IlluminatoramaFrameUniforms.
    float    contactShadowStrength;   // 0 = off; direct-sun attenuation amount (0..1)
    float    contactShadowLength;     // march reach, world units
    uint     contactShadowSteps;      // ray-march sample count
    float    contactShadowThickness;  // occluder depth window, world units
    // Screen-space subsurface scattering (issue #65). Jimenez-style separable SSS
    // for skin / wax / marble / food. SSS pixels are flagged in normalRoughness.w
    // (vertex-colour alpha ∈ [0.90,0.98] → 0.95h). `sssStrength` 0 = OFF → the
    // lighting pass skips the side-buffer write and the host doesn't encode the
    // blur/composite passes → an EXACT no-op for every existing scene. `sssRadius`
    // is the diffusion mean-free-path in MILLIMETRES (projected to screen px per
    // pixel via depth); `sssTint*` scales the per-channel scatter distance (default
    // reddish skin profile → warm bleed). NEW 32-byte trailing region (two 16-byte
    // clusters, stride 1056 → 1088); field-for-field mirror of the Swift
    // IlluminatoramaFrameUniforms. The three pads keep the final cluster aligned.
    float    sssStrength;   // 0 = off; blend of blurred over sharp diffuse (0..1)
    float    sssRadius;     // diffusion radius, millimetres
    float    sssTintR;      // per-channel scatter-distance scale (R)
    float    sssTintG;      // per-channel scatter-distance scale (G)
    float    sssTintB;      // per-channel scatter-distance scale (B)
    float    sssDebugForceAll;  // 1 = treat every opaque pixel as SSS (headless verify)
    // Foliage-wind motion-vector delta (DH-0492). The vegetation gust in `applyTreeWind`
    // is a pure function of `frame.time`, so with the previous-frame position sampled at the
    // SAME time the wind contributed ~0 to the screen-space velocity buffer — TAA could not
    // reproject the swaying sub-pixel canopy/blade edges and they crawled/shimmered between
    // pixels instead of reading as a continuous bend. `illumi_vs` now samples the PREVIOUS
    // frame's wind at `frame.time − windPrevDelta`, so a moving canopy writes a real motion
    // vector TAA can track. 0 = same-time (the pre-DH-0492 behaviour, an EXACT no-op) — every
    // scene that never sets it, and every SETTLED capture (whose clock does not advance
    // between accumulation frames), ships a byte-identical velocity buffer. Repurposes the
    // former `_padSSS1` slot — same 4 bytes at the same offset, stride unchanged.
    float    windPrevDelta;
    float    _padSSS2;
    // Phase 9 — film-stock LUT blend strength. 0 = bypass, 1 = full grade.
    // NEW 16-byte cluster (stride 1088 → 1104). Three float pads fill.
    float    filmLUTStrength;
    float    _padFilmLUT0;
    float    _padFilmLUT1;
    float    _padFilmLUT2;
    // Tonemap colour-grade (white-balance / tint pre-tonemap; contrast / shadows
    // / highlights as a post-tonemap curve). TWO new 16-byte clusters (stride
    // 1104 → 1136). Defaults are neutral: whiteBalanceK 6500 → gain (1,1,1),
    // tint 0, shadows/highlights/contrast 1.0 → the whole grade is a no-op.
    // Mirrors IlluminatoramaFrameUniforms.{whiteBalanceK,tint,shadows,highlights,contrast}.
    float    whiteBalanceK;
    float    tint;
    float    shadows;
    float    highlights;
    float    contrast;
    // Phase 7 — opt-in hex-stochastic anti-tiling strength [0,1]. 0 = OFF (the
    // DEFAULT): every `sampleAtlasHex` short-circuits to a single plain texture
    // read, so the G-buffer is byte-for-byte identical to the pre-anti-tiling
    // shader. >0 mixes in the 3-tap de-repeat blend. Repurposes the former
    // `_padGrade0` slot — same 4 bytes, stride unchanged. Mirrors
    // IlluminatoramaFrameUniforms.antiTilingStrength.
    float    antiTilingStrength;
    // Point-light cubemap shadow depth bias (was `_padGrade1`). Read only for point
    // lights with shadowCubeIndex >= 0; 0 ⇒ no point shadows ⇒ no behaviour change.
    float    pointShadowBias;
    // Scotopic (Purkinje) desaturation strength (was `_padGrade2` — same 4 bytes,
    // stride unchanged). Pulls the DIMMEST tonemapped pixels toward their own
    // luminance so moonlit surfaces read neutral-dark, not (green-albedo) tinted.
    // 0 (DEFAULT) ⇒ the tonemap branch never runs ⇒ byte-identical. Mirrors
    // IlluminatoramaFrameUniforms.scotopicDesaturation.
    float    scotopicDesaturation;
    // ── Interior day-light separation (opt-in; pairs with light-layer masking) ──
    // A host that stamps per-room layer bits (gLayer) can declare those bits INTERIOR
    // via `interiorMask` (OR of every interior room bit). A fragment whose layer is a
    // real room stamp (≠ the 0xFFFFFFFF default) AND intersects the mask then has its
    // sky-IBL indirect scaled by mix(interiorIBLSide, interiorIBLUp, saturate(N.y)) —
    // up-facing floors lose the open-top skylight hardest — and its ambient supplement
    // scaled by `interiorAmbient` (a warm indoor fill can exceed 1). `interiorMask == 0`
    // (the default) skips the whole block: both factors stay exactly 1.0 and ×1.0 is an
    // IEEE no-op, so every scene that never opts in renders byte-identically.
    // NEW 16-byte cluster (stride 1136 → 1152). Mirrors IlluminatoramaFrameUniforms.
    uint     interiorMask;
    float    interiorIBLUp;
    float    interiorIBLSide;
    float    interiorAmbient;
    // ── Analytic night sky (stars + moon at SCREEN resolution) ──────────────
    // The host disables the dome's baked celestials (VolumetricCloudRenderer
    // Params.celestialsInDome = false) and sets these; the lighting kernel then
    // evaluates nightStarField/nightMoonDisk per SKY PIXEL — pixel-sharp stars,
    // a crisp phase-correct moon — instead of sampling their bilinearly-
    // magnified dome bakes. All-zero (the default) ⇒ the sky branch adds
    // exactly nothing, so every non-opting scene renders byte-identically.
    // nightSkyParams: x = starBrightness × nightBlend, y = moonIntensity ×
    // nightBlend, z = moon angular radius (radians), w reserved.
    // THREE new 16-byte clusters (stride 1152 → 1200); field-for-field mirror
    // of IlluminatoramaFrameUniforms.
    float4   nightSkyParams;
    float4   nightMoonDir;   // xyz = unit vector toward the moon
    float4   nightSunDir;    // xyz = unit vector toward the TRUE sun (terminator)
    // ── Lens flare (sun) ─────────────────────────────────────────────────────
    // x = strength (0 = OFF, the default — the tonemap branch is gated on it so
    // non-opting scenes are byte-for-byte unchanged), yz = the sun's screen-space
    // uv, w = on-screen weight (fades at the frame edge, 0 behind the camera).
    // ONE new 16-byte cluster (stride 1200 → 1216); mirror of
    // IlluminatoramaFrameUniforms.lensFlareParams.
    float4   lensFlareParams;
    // ── Halation (film) ──────────────────────────────────────────────────────
    // halationParams: x = intensity (0 = OFF, the default — the host skips the
    // halation passes entirely and the tonemap branch is gated on it, so non-opting
    // scenes are byte-for-byte unchanged), y = HDR luminance threshold above which a
    // highlight scatters, z = halo radius in INTERNAL-resolution texels, w reserved.
    // halationTint: xyz = halo tint (linear RGB), w reserved.
    // TWO new 16-byte clusters (stride 1216 → 1248); mirror of
    // IlluminatoramaFrameUniforms.halationParams/halationTint.
    float4   halationParams;
    float4   halationTint;
    // ── Bloom pyramid (S1.2) ─────────────────────────────────────────────────
    // bloomParams: x = soft-knee width as a FRACTION of `bloomThreshold` (0 =
    // the old hard `max(0, lum - T)` cut, reproduced exactly; the curve is
    // identical to the hard one above T·(1+x) either way), y = upsample scatter
    // (the convex `mix` weight toward the blurred low mip — 0 collapses the
    // pyramid to its finest level, 1 to its coarsest), z = 3×3 tent radius in
    // LOW-mip texels, w reserved.
    // ONE new 16-byte cluster (stride 1248 → 1264); mirror of
    // IlluminatoramaFrameUniforms.bloomParams.
    float4   bloomParams;
    // ── Diagram look (flat shading + outlines) ───────────────────────────────
    // diagramParams: x = mix (0 = OFF, an exact no-op — the tonemap branch is
    // gated on it, so non-opting scenes are byte-for-byte unchanged), y = light
    // wrap (0 = perfectly flat fills, 1 = full sky/ground hemisphere term),
    // z = outline strength (0 = no line work), w = orthographic flag (0/1 — tells
    // the view-vector derivations the eye is at infinity; not a look control).
    // diagramOutline: xyz = outline colour (linear RGB), w reserved.
    // diagramEdge: x = depth sensitivity, y = normal sensitivity, z = line
    // thickness in output pixels, w reserved.
    // THREE new 16-byte clusters (stride 1264 → 1312); mirror of
    // IlluminatoramaFrameUniforms.
    float4   diagramParams;
    float4   diagramOutline;
    float4   diagramEdge;
    // ── TAA jitter, subtracted back out of the motion vectors (S4.2) ─────────
    // xy = THIS frame's sub-pixel projection jitter minus the PREVIOUS frame's,
    // in NDC units; zw reserved.
    //
    // `previousViewProjection` is the previous frame's JITTERED VP and the
    // current VP is jittered too, so `currNDC − prevNDC` carries the difference
    // of two different sample offsets on top of the surface's real motion. Left
    // in, the resolve reprojects the history by exactly that difference and
    // realigns it onto the CURRENT sub-pixel sample — so successive frames stop
    // contributing different samples to a fixed output pixel, which is the whole
    // mechanism of jitter supersampling. Subtracting this delta is the standard
    // formulation: it pins the history to the output pixel grid while the samples
    // move across it.
    //
    // Zero whenever jitter is off (the default, and every frame of every scene
    // that never sets `taaJitterPixels`) ⇒ `− 0` is an IEEE no-op ⇒ the velocity
    // buffer is byte-identical. ONE new 16-byte cluster (stride 1312 → 1328);
    // mirror of IlluminatoramaFrameUniforms.taaJitterDelta.
    float4   taaJitterDelta;
    // ── Interior irradiance bands (the lawn-green-ceiling fix) ───────────────
    // A room's own three-band diffuse environment, host-supplied. The irradiance
    // cube is baked from the OUTDOOR sky dome — sky above, ground below — so an
    // interior ceiling (N.y = −1) samples the ground half and renders in the
    // lawn's colour; no scalar can recolour it. Interior fragments
    // (frame.interiorMask) blend their diffuse irradiance toward these bands by
    // `interiorIrrUp.w` (0 → exact cube sample, the default — every scene that
    // never sets it is byte-identical; hosts ramp w with daylight so night keeps
    // the cube):
    //   N.y = +1 → interiorIrrUp   (floors: ceiling + wall bounce)
    //   N.y =  0 → interiorIrrSide (walls: room mix)
    //   N.y = −1 → interiorIrrDown (ceilings: FLOOR bounce — not lawn)
    // As w → 1 the DIFFUSE interior scalar folds to 1.0 (the bands already carry
    // the up/side weighting `interiorIBLUp/Side` faked); the SPECULAR lobe keeps
    // the scalar and the prefiltered cube. Units are pre-`iblIntensity`
    // irradiance, exactly like the cube sample they replace.
    // THREE new 16-byte clusters (stride 1328 → 1376); mirror of
    // IlluminatoramaFrameUniforms.
    float4   interiorIrrUp;    // xyz = irradiance, w = band blend weight (0 = off)
    float4   interiorIrrSide;  // xyz = irradiance, w unused
    float4   interiorIrrDown;  // xyz = irradiance, w unused
    // ── Ray-traced soft sun shadows in the DEFERRED kernel (S4.1, opt-in) ────
    // Consumed ONLY by the `kLightingRTSunShadow` (function_constant(5)) variant
    // of `illumi_lighting`, which replaces the cascaded-shadow-map visibility
    // with `rtSunShadowRayCount` cone-sampled shadow rays traced against the
    // TLAS bound at buffer(8) — a true penumbra from the sun's angular radius.
    // The non-RT variant (every scene that never opts in, and every Visualizer
    // host) never reads these, so all-zero defaults are an exact no-op.
    // `rtSunShadowSeed` decorrelates the cone samples per frame; the host WALKS
    // it only while a temporal accumulator is live and freezes it otherwise
    // (a walking seed on a static frame is crawling speckle — same contract as
    // the glass pass's frameSeed). ONE new 16-byte cluster (stride 1376 → 1392);
    // field-for-field mirror of IlluminatoramaFrameUniforms.
    uint     rtSunShadowSeed;      // per-frame RNG decorrelation (0 = frozen)
    float    rtSunShadowAngle;     // sun angular radius, radians (cone half-angle)
    uint     rtSunShadowRayCount;  // shadow rays per pixel, host-clamped 1…8
    float    _padRTSunShadow;
    // ── Per-room interior band LEVEL (S3.5 Stage E) ──────────────────────────
    // The bands above are ONE environment for the whole frame, pegged to the host's
    // ambient fill — so a room walled in glass and a windowless closet rendered the
    // same ceiling. These are per-ROOM levels: one gain per light-layer bit (the
    // same 32-bit room identity `interiorMask` and `PointLight.layerMask` use),
    // packed 4 to a float4. A fragment's bits resolve through
    // `interiorRoomBandGain` (IlluminatoramaSecondary.h — shared with the secondary
    // paths so a room through a pane matches the room beside it), which multiplies
    // BOTH the diffuse and the specular band.
    // `interiorRoomGainMeta.x` = 0 (the default, and every scene that never opts in)
    // ⇒ the resolve returns exactly 1.0 and the frame is byte-identical.
    // NINE new 16-byte clusters (stride 1392 → 1536); mirror of
    // IlluminatoramaFrameUniforms.interiorRoomGain0…7 + interiorRoomGainMeta.
    float4   interiorRoomGain[8];   // 32 room gains, bit b at [b >> 2][b & 3]
    float4   interiorRoomGainMeta;  // x = enabled (0 = off); yzw unused
    // ── Photographic finish: highlight chroma roll-off + split tone ──────────
    // Two things a real camera does that a flat post-tonemap saturation multiply
    // undoes, applied at the very end of the tone chain in the display (0..1)
    // domain.
    //
    // `highlightChromaRolloff` fades chroma out as a pixel approaches white.
    // ACES already desaturates toward white, but `tonemapSaturation`'s
    // `mix(luma, colour, S)` is FLAT in luminance, so it pushes that chroma
    // straight back out — which is why a lamp-lit wooden wall clips to saturated
    // yellow instead of a pale warm white. 0 (default) ⇒ the branch never runs.
    //
    // `shadowTemperatureK` / `highlightTemperatureK` are a split tone: the same
    // `whiteBalanceGain` curve the global white balance uses, applied through a
    // luma mask so the two ends of the scale can sit at different temperatures
    // (the classic cool-shadow / warm-highlight separation). 6500 on both
    // (default) ⇒ both gains are exactly (1,1,1) and the branch never runs.
    // ONE new 16-byte cluster (stride 1536 → 1552); mirror of the Swift
    // IlluminatoramaFrameUniforms.
    float    highlightChromaRolloff;  // 0 = off
    float    shadowTemperatureK;      // 6500 = no-op
    float    highlightTemperatureK;   // 6500 = no-op
    float    _padPhotoFinish;
};

// Secondary directional light (#60 task 5). Mirror of Swift
// IlluminatoramaDirectionalLight. `dir` points TOWARD the light (world space,
// pre-normalized); `color` is the premultiplied linear-HDR intensity.
struct DirectionalLight {
    float3 dir;     float _pad0;
    float3 color;   float _pad1;
};

struct PointLight {
    float3 position;
    float  radius;
    float3 color;
    // Light-layer mask (was `_pad`). A light contributes to a fragment only when
    // (light.layerMask & fragment.layer) != 0. Default 0xFFFFFFFF ⇒ affects every
    // fragment (byte-identical to the pre-mask behaviour). Reinterpreted from the
    // former float pad, so the struct stride is unchanged.
    uint   layerMask;
    // ── Point-light cubemap shadows (opt-in) ──────────────────────────────────
    // castsShadow: 1 ⇒ sample the depth cube at shadowCubeIndex for this light.
    // Default 0 ⇒ the shadow branch is skipped entirely (byte-identical). shadowCubeIndex
    // < 0 ⇒ no page this frame (treat as fully visible). Mirrors the Swift struct.
    uint   castsShadow;
    int    shadowCubeIndex;
    int    _padPointShadow0;
    // Source-size term for the near-field falloff (was `_padPointShadow1`; reinterpreted,
    // so the struct stride is unchanged). Attenuation is `1/(d² + softRadius²)` instead of
    // `1/d²`: a point source of zero size blows up as d→0, so a wall a metre from a bulb
    // reads as a hard blob. `softRadius` gives it a finite apparent size — the near field
    // flattens into a soft halo while the far field stays honest inverse-square. Default 0 ⇒
    // exactly `1/d²`, byte-identical to the prior behaviour (Visualizer never sets it).
    float  softRadius;
};

// Rectangular area light (#60 task 5). Mirror of Swift IlluminatoramaAreaLight.
// Rectangle corners = center ± ex ± ey; emitting normal = normalize(cross(ex,ey)).
struct AreaLight {
    float3 center;   float twoSided;   // twoSided: 1 = emit both faces, 0 = front only
    // Light-layer mask (was `_pad0` — same 4 bytes, stride unchanged; same rule as
    // PointLight/SpotLight). Default 0xFFFFFFFF ⇒ affects every fragment,
    // byte-identical to the pre-mask behaviour. This is what CONTAINS a window
    // portal: an unmasked area light has no visibility term and no shadow map, so
    // without it a portal lights the yard through the back of its own wall.
    float3 ex;       uint layerMask;   // half-width edge vector (world) + mask
    float3 ey;       float _pad1;      // half-height edge vector (world)
    float3 color;    float radius;     // premultiplied color + distance-falloff range
};

// A daylight aperture (S3.5 Stage D) — a glazed opening reduced to what the interior
// irradiance gradation needs. Mirror of Swift `IlluminatoramaInteriorAperture` (32 B),
// bound at lighting kernel buffer(7). See the band branch in IlluminatoramaLighting.metal.
struct InteriorAperture {
    float3 center;   float width;      // aperture centre (world) + metres
    float3 inward;   float height;     // unit normal INTO the room + metres
};

struct SpotLight {
    float3   position;
    float    innerCone;       // cos(spotInnerAngle / 2)
    float3   direction;       // points AWAY from apex (same as SCNLight)
    float    outerCone;       // cos(spotOuterAngle / 2)
    float3   color;           // premultiplied intensity
    float    radius;
    // Phase 4.10 — per-spot shadow data. matrix transforms world-space
    // fragment into the spot's light-space NDC (post-divide); slice
    // index points at the depth atlas page where this spot's depth was
    // rasterised. shadowSliceIndex < 0 means "no shadow data", and the
    // spot contributes as fully visible.
    float4x4 shadowMatrix;
    int      shadowSliceIndex;
    // Light-layer mask (was `_padSpot0`). Same masking rule as PointLight; default
    // 0xFFFFFFFF ⇒ affects every fragment. Reinterpreted from a former int pad, so
    // the struct stride is unchanged.
    uint     layerMask;
    // 1 ⇒ this spot may claim a shadow-atlas slice; 0 ⇒ it never does. Host-side only —
    // the kernel still branches on `shadowSliceIndex` — but it lives in the struct so the
    // fact travels WITH the light instead of being inferred from its array position.
    // Reinterpreted from a former int pad, so the struct stride is unchanged.
    int      castsShadow;
    // Source-size term for the near-field falloff (was `_padSpot2`; reinterpreted, so the
    // struct stride is unchanged). Same rule as `PointLight.softRadius`: attenuation is
    // `1/(d² + softRadius²)`, so a cone from a finite-size source flattens into a soft halo
    // near the emitter instead of a hard blob on a nearby wall. Default 0 ⇒ exactly `1/d²`.
    float    softRadius;
};

struct Instance {
    float4x4 modelMatrix;
    float4x4 normalMatrix;
    float3   albedo;
    float    metallic;
    // Phase 7 — clearcoat lobe (polished/lacquered surfaces). Occupies the
    // 12-byte padding gap between metallic (offset 144) and emission (offset 160);
    // stride stays 208. Default 0 = off (no change to existing materials).
    float    clearcoat;              // [0,1] lobe strength
    float    clearcoatRoughness;     // GGX roughness for the clearcoat layer
    float    sheen;                  // Phase 7b — cloth sheen strength [0,1] (was _padClearcoat)
    float3   emission;
    float    roughness;
    // Phase 4.0/4.1 — slice indices into the per-material texture atlases
    // bound at G-buffer fragment shader [[texture(0)]] (sRGB albedo) and
    // [[texture(1)]] (linear non-colour: metallic/roughness/normal share).
    // Negative values mean "no texture; use the corresponding scalar
    // (`albedo`/`metallic`/`roughness`)". Three Int32s use the 12 bytes
    // of trailing pad — Instance stride stays at 192.
    int      albedoTextureSlice;
    int      metallicTextureSlice;
    int      roughnessTextureSlice;
    // Phase 4.5 — tangent-space normal map slice in the non-colour atlas.
    int      normalTextureSlice;
    // Phase 4.9 — emission map slice in the sRGB albedo atlas (emission
    // is colour, so it gets the sRGB-decoded atlas, not the linear one).
    // The five Int32s + 2 pads still form a single 16-byte cluster; stride
    // remains 208.
    int      emissionTextureSlice;
    // Phase 4.27b — multiplier on the emission TEXTURE sample so a texture-
    // driven glow (Pizza's heat coils) renders at its tuned HDR intensity.
    // Repurposes the former `_padSlice0` slot.
    float    emissionIntensity;
    // Former trailing pad — still unused GPU-side (RT exclusion is host-only).
    int      _padSlice1;
    // Phase 7 — detail-normal path. Stride grows from 208 → 224 (next 16-byte
    // boundary) to preserve float4x4 natural alignment.
    int      detailNormalTextureSlice;  // < 0 = disabled
    float    detailNormalUVScale;       // tile frequency relative to macro UV
    float    anisotropy;                // Phase 7c — grain highlight stretch [0,1] (was _padDetail0)
    int      highlight;   // 0 none · 1 selected (blue halo) · 2 hover (yellow halo)
    // Drag/impact sway — generic vertex-shader secondary motion (see applySway).
    // New 16-byte cluster (offsets 224-239): stride grows 224 → 240.
    int      swayMode;    // 0 none · 1 bottom-pivot lean · 2 top-pivot pendulum (hanging)
    float    swayLean;    // mode 1: static lean angle (rad); mode 2: pendulum amplitude (rad)
    float    swayJostle;  // vertical pop (metres), applied in world space
    // Light-layer bitfield (was `_padSway0`). Written into the gLayer G-buffer
    // target by the fragment shader; the deferred lighting kernel masks each light
    // by (light.layerMask & fragment.layer). Default 0xFFFFFFFF ⇒ every light
    // affects this instance (byte-identical to pre-mask behaviour). Reinterpreted
    // from the former float pad, so the struct stride stays 240.
    uint     layer;
    // Per-instance multiplier on frame.antiTilingStrength. The hex-stochastic
    // de-repeat is only valid for STOCHASTIC textures — hosts set 0 for coherent
    // patterns (tile grids, wallpaper, directional wood) whose offset samples
    // would double-print/dice the pattern. NEW 16-byte cluster: stride 240 → 256.
    float    antiTilingScale;
    // ── S2.4 detail-band relief companions (was `_padAntiTiling0/1`) ──────────
    // The detail band used to carry a NORMAL and nothing else, which is a
    // specular-band effect: measured over a whole material library it moved metals
    // 1.2–3.4x and every dielectric ~1.00x, because a matte surface's broad diffuse
    // lobe averages a fine normal perturbation away. These two read the SAME detail
    // slice's BLUE channel — a micro-occlusion the host bakes from the same height
    // field as the normal, in a channel the shader already fetched and discarded.
    //
    // detailOcclusionStrength: how much of that occlusion multiplies ALBEDO (⇒ the
    //   DIFFUSE lobe — the half a dielectric can actually show).
    // detailRoughnessStrength: how much the occluded (pit) texels are additionally
    //   roughened — micro-cavities scatter more than the open surface.
    //
    // Both default 0, and a slice whose blue is 255 yields occ == 1 either way, so a
    // host that never opts in — and any foreign detail-normal image — is byte-identical.
    // Reinterpreted pads: stride stays 256.
    float    detailOcclusionStrength;
    float    detailRoughnessStrength;
    // Tile frequency for the OCCLUSION tap, independent of `detailNormalUVScale` (was
    // `_padAntiTiling2`). It has to be separate, and this is the measured reason:
    // at the normal's 8x on a 2 m macro tile a detail texel is ~0.5 mm and a feature
    // ~2 mm — an order of magnitude below one pixel at any archviz camera. Sampled
    // mip-correctly (which occlusion must be, or it is just noise) that averages to a
    // FLAT tint and shows nothing. The occlusion therefore runs at a coarser, actually
    // resolvable band while the normal keeps its fine one; both read the same baked
    // tile, so this costs no memory. 0 falls back to `detailNormalUVScale`.
    float    detailOcclusionUVScale;
    // ── S2.5 half 2 — per-PATTERN-CELL value jitter (the coherent categories' de-repeat) ──
    // NEW 16-byte cluster (offsets 256-271): stride 256 -> 272.
    //
    // The hex blend above is invalid for a COHERENT pattern (a tile grid, a plank run):
    // offsetting the UV superimposes misaligned copies, so those materials run with
    // antiTilingScale == 0 and had no de-repeat at all. What they actually want is what a
    // real tile/plank floor has — batch-to-batch VALUE variation from unit to unit — and
    // that needs no UV displacement and no extra texture tap:
    //
    //     cell = floor(uv * patternCells)
    //
    // `uv` keeps counting across the whole surface (only the atlas LOOKUP wraps), so the
    // cell index is unique per physical tile/plank over the entire floor even though the
    // texture repeats every UV unit. Hashing it to a +/- `patternJitter` tone multiplier
    // therefore breaks the macro repeat while leaving the grout grid, the plank seams and
    // the grain direction exactly where the bake put them.
    //
    // NOT per UV TILE: a UV tile is ~2 m and holds many physical units, so a per-tile hash
    // paints a 2 m checkerboard with hard seams — one visible grid traded for a coarser,
    // uglier one.
    //
    // patternCells: pattern cells per macro UV tile, per axis. A component of 0 means "no
    //   subdivision on this axis" and yields a constant cell index there — which is what a
    //   plank run wants (one tone down the whole board, varying board to board).
    // patternJitter: half-amplitude of the multiplicative tone jitter (0.05 = +/-5 %).
    //   0 (the default, and every scene that never opts in) is an exact no-op.
    float2   patternCells;
    float    patternJitter;
    // Vegetation-wind opt-in (was `_padPattern0` — same 4 bytes, stride stays 272).
    // Multiplies the frame's global wind strength for THIS instance, so `applyTreeWind`
    // can only reach a draw whose host declared it vegetation. 0 (the default) is an
    // exact no-op. See the Swift twin for why this had to become per-instance.
    float    windScale;
    // ── World-space WOOD KNOTS (see `sampleWoodKnots` in IlluminatoramaMaterial.h) ──
    // NEW 16-byte cluster (offsets 272-287): stride 272 -> 288.
    //
    // A knot is a sparse LANDMARK, and a landmark baked into a tiling texture reappears on a
    // lattice at the tile period. So it is not baked: it is evaluated per pixel on the UNWRAPPED
    // uv, which counts up across the whole surface, and it warps the material UV so the grain
    // flows around it rather than being composited over it.
    //
    //   x = lattice cells per UV unit (0 = disabled — the default, and an exact no-op)
    //   y = knot radius in UV units
    //   z = darkness of the core against the wood it grew through, [0,1]
    //   w = fraction of lattice cells that carry a knot
    float4   woodKnots;
    // ── Per-INSTANCE (per-object) macro material variation ────────────────────────
    // NEW 16-byte cluster (offsets 288-303): two floats + 8 bytes pad, stride 288 -> 304.
    // Two placed pieces wearing the SAME material id share one atlas slice and one mesh, so
    // without a per-instance channel they render the identical texels ("two nightstands are
    // the same pixels twice"). macroTone is an ACHROMATIC multiplier on the FINAL albedo (post
    // atlas + vertex colour); macroRoughnessDelta adds to the resolved roughness. Both identity
    // (1 / 0) by default, so a host that never opts in — every Visualizer scene — is
    // byte-identical.
    float    macroTone;
    float    macroRoughnessDelta;
    // DH-0475 — per-INSTANCE UV phase (translation, tile-UV units) added into `matUV` so two
    // pieces sharing one slice + one mesh don't show the same GRAIN figure in the same place.
    // Fills the 8 bytes that were trailing pad, so stride stays 304; `float2(0)` is an exact
    // no-op (identical to every existing scene). Mirrors `IlluminatoramaInstance.uvPhase`.
    float2   uvPhase;
    // DH-0081 — per-material cloth-sheen ROUGHNESS (the nap WIDTH; sibling of `sheen`, its
    // strength). NEW 16-byte cluster (offsets 304-319): stride 304 -> 320. The G-buffer snaps it
    // to a band and folds the band into the integer part of the sign-multiplexed `sheen`
    // emission.alpha (see clothSheenBandForRoughness); band 0 (default 0.30) packs `-sheen` exactly
    // as before, so a host that never sets it — every Visualizer scene — is byte-identical.
    // Mirrors `IlluminatoramaInstance.sheenRoughness`.
    float    sheenRoughness;
};

// The Swift mirror (`IlluminatoramaInstance._assertStride240`) has always asserted this
// side of the contract; this is the other side, and it costs a compile. A Swift field
// added without its Metal twin used to be caught only by a wrong-looking render.
static_assert(sizeof(Instance) == 320, "Instance must match IlluminatoramaInstance (320 bytes)");

// ── Cloth sheen roughness bands (DH-0081) ─────────────────────────────────────────────────
// The sheen lobe's roughness is carried per-material by folding a small BAND index into the
// INTEGER part of the (negative) sheen magnitude packed in `emission.alpha`, leaving the
// FRACTION for sheen strength: `emission.alpha = -(band + strength)`. Band 0 is the historical
// single constant (0.30), so a material that keeps the default nap packs `-strength` exactly as
// it did before this existed — byte-for-byte. Four curated bands span crisp pile → broad fuzz; a
// material's continuous `sheenRoughness` snaps to the nearest at pack time. Both directions live
// here (this header is included by the G-buffer packer AND the lighting unpacker) so the band
// table has ONE definition.
inline float clothSheenRoughnessForBand(int band) {
    switch (band) {
        case 1:  return 0.18f;   // crisp / tight nap (sateen, silk)
        case 2:  return 0.45f;   // broad soft nap (velvet, carpet pile)
        case 3:  return 0.60f;   // very broad fuzz (wool bouclé, chenille)
        default: return 0.30f;   // default — linen / general plain-weave upholstery
    }
}
inline int clothSheenBandForRoughness(float r) {
    // Nearest band. 0.30 (band 0) is listed first so a default-nap material snaps to it and
    // packs identically to the pre-band encoding. Keep in sync with the switch above.
    float bands[4] = { 0.30f, 0.18f, 0.45f, 0.60f };
    int best = 0; float bestD = fabs(r - bands[0]);
    for (int i = 1; i < 4; ++i) { float d = fabs(r - bands[i]); if (d < bestD) { bestD = d; best = i; } }
    return best;
}

struct Vertex {
    float3 position;
    float  _padPos;
    float3 normal;
    float  _padNrm;
    float2 uv;
    float2 _padUv;
    // Phase 4.5 — tangent (xyz) + handedness (w). Bitangent =
    // cross(normal, tangent.xyz) * tangent.w. Zero tangent means "no
    // normal-map data for this vertex" and the fragment shader falls
    // through to the geometric normal.
    float4 tangent;
    // Phase 4.17 — per-vertex RGBA color, multiplied into albedo at
    // shading time. Default white per vertex on assets that ship no
    // .color semantic. Stride 96 → 112.
    float4 color;
};

// ── G-buffer pass ────────────────────────────────────────────────────────────

struct VSOut {
    float4 clipPos      [[position]];
    float3 worldPos;
    float3 worldNormal;
    float2 uv;
    uint   instanceID   [[flat]];
    // Light-layer bitfield, forwarded flat from the instance so the G-buffer
    // fragment can write it into the gLayer target. Default 0xFFFFFFFF.
    uint   layer        [[flat]];
    // Phase 2.7 — clip-space positions for motion-vector reconstruction.
    // We pass both the current and previous clip-space positions explicitly
    // (rather than reading [[position]] in the fragment, which is post-divide
    // viewport coords) so the fragment shader can do the perspective divide
    // for each and compute a clean screen-space delta.
    float4 currentClip;
    float4 previousClip;
    // Phase 4.5 — tangent in world space + handedness. Fragment shader
    // builds the bitangent as `cross(worldNormal, tangent.xyz) * tangent.w`
    // when sampling a normal map. Zero indicates "no tangent data" and the
    // fragment shader falls through to the geometric normal.
    float4 worldTangent;
    // Phase 4.17 — per-vertex RGBA color, interpolated across the
    // triangle and multiplied into the albedo at shading time. Carries
    // pattern detail that lives in a `.color` semantic on the source
    // SCNGeometry (HotAirBalloon's chevron stripes; GiantGummyBears'
    // candy gradients; anything procedural-coloured per-vertex).
    float4 vertexColor;
};

// Octahedral encoding — packs a normalized vec3 into 2 channels. Standard
// "Survey of Efficient Representations for Independent Unit Vectors" (Cigolle
// et al.) encoding. Decoded in the lighting pass.
static inline float2 octEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.xy;
    if (n.z < 0.0) {
        float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0,
                          n.y >= 0.0 ? 1.0 : -1.0);
        e = (1.0 - abs(e.yx)) * s;
    }
    return e * 0.5 + 0.5;
}

// ── Deferred PBR lighting ────────────────────────────────────────────────────
//
// Standard Cook-Torrance microfacet BRDF: GGX/Trowbridge-Reitz NDF + Schlick
// Fresnel + Smith geometry. Reads G-buffer + depth, reconstructs worldPos from
// NDC, sums one directional + N point lights, writes HDR linear color.

static inline float3 octDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0,
                          n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return normalize(n);
}

static inline float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

static inline float geometrySmith(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

// ── Sky-probe IBL helpers (Phase 3) ──────────────────────────────────────────
//
// We don't bake our own procedural sky — the host hands us an equirect HDR
// texture (typically VolumetricCloudRenderer's output, which already has the
// sun disc, atmosphere gradient, clouds, optional moon/stars). From it we
// pre-compute two cubemaps once per frame:
//
//   • irradianceCube   — cosine-hemisphere convolution → diffuse IBL
//   • prefilteredCube  — GGX-importance-sampled mip chain → specular IBL +
//                        the sky-miss fallback that Phase 2's SSR comment
//                        explicitly deferred to "Phase 3 GI".
//
// All sampling matches VolumetricCloudRenderer's equirect convention:
// u runs east-around from +X, v runs from +Y (north pole) to -Y (south pole).

constant float kInvTwoPi = 1.0 / (2.0 * M_PI_F);
constant float kInvPi    = 1.0 / M_PI_F;

// THE canonical dirToEquirectUV — one convention, everywhere. u wraps so +X
// lands at u = 0, matching the equirect texture's WRITER (volSkyRender in
// VolumetricSky.metal) and every reader: the background-sky raster, both IBL
// bakes, DDGI, forward glass, and (since the 2026-08-16 fix) the RT paths.
// Secondary.h, IlluminatoramaRT.metal and IlluminatoramaSurfaceCache.metal
// carry byte-identical copies (they can't include this header without splicing
// in bulk they don't want); the macro below lets Secondary.h skip its copy when
// this one is already in scope, avoiding a redefinition in translation units
// that include both (the deferred lighting kernel, since S4.1). Historical
// note: Secondary.h/RT/SurfaceCache used to carry a +0.5-offset variant, which
// put every RT sky sample 180° in azimuth from where the raster draws the sky.
#define ILLUMI_COMMON_EQUIRECT_UV 1
static inline float2 dirToEquirectUV(float3 dir) {
    // dir must be normalised. u in [0,1) east-around, v in [0,1] from north
    // pole (v=0) to south pole (v=1). +X column is u=0.
    float u = atan2(dir.z, dir.x) * kInvTwoPi;        // (-0.5, 0.5]
    if (u < 0.0) u += 1.0;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) * kInvPi;
    return float2(u, v);
}

// Lagarde's roughness-aware Schlick fresnel — the cheap IBL approximation
// that skips the proper split-sum DFG LUT. Visually fine for v1; a real LUT
// gets added when we wire up TAA in Phase 2.7.
static inline float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    float3 oneMinusRough = max(float3(1.0 - roughness), F0);
    return F0 + (oneMinusRough - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Sample the equirect HDR sky with bilinear filtering. The `[[texture]]`
// argument is the renderer's external sky source (e.g. VolumetricCloudRenderer's
// equirect output).
static inline float3 sampleSkyEquirect(texture2d<float, access::sample> sky,
                                       float3 dir) {
    constexpr sampler s(filter::linear,
                        s_address::repeat,
                        t_address::clamp_to_edge);
    return sky.sample(s, dirToEquirectUV(normalize(dir))).rgb;
}

// ── Analytic night sky — stars + moon at SCREEN resolution ──────────────────
//
// The equirect sky dome (VolumetricCloudRenderer, 2048×1024) is far coarser than
// the frame: one dome texel covers several screen pixels, so anything baked into
// it — a star, the moon's limb — is bilinearly magnified into a soft blob. The
// host can therefore ask the DOME to skip its celestials
// (`Params.celestialsInDome = false`) and have this pass evaluate them
// analytically per SKY PIXEL instead: pixel-sharp stars, a crisp moon disk with
// a geometrically-correct terminator. Gated on `frame.nightSkyParams` being
// non-zero — every scene that never sets it renders byte-identically.
//
// The star grid (cell layout, fill rate, magnitude/colour hashing) mirrors
// `starField` in VolumetricSky.metal so the two paths agree on WHERE the stars
// are — only the point-spread differs (dome texel there, screen pixel here).

// The four functions that used to live here — `nightHash3`, `nightWorley2`,
// `nightStarField`, `nightMoonDisk` — now live in IlluminatoramaNightSky.h so the
// SECONDARY-ray paths (glass refraction/reflection, TLAS reflections) can draw the
// SAME sky. A window used to show a starless, moonless dome while the sky beside it
// was full of stars, because only this primary-ray branch knew how to draw them.

/// Unpack `FrameUniforms`' night-sky fields into the shared currency. The ONE place
/// that knows the packing order, so the glass pass (which mirrors these fields into
/// its own uniforms) and this pass cannot disagree about which slot holds what.
static inline NightSkyParams frameNightSky(constant FrameUniforms& f) {
    NightSkyParams p;
    p.starBrightness = f.nightSkyParams.x;
    p.moonIntensity  = f.nightSkyParams.y;
    p.moonAngRadius  = f.nightSkyParams.z;
    p.moonDir        = f.nightMoonDir.xyz;
    p.toSun          = f.nightSunDir.xyz;
    return p;
}

// ── Screen-space ambient occlusion ───────────────────────────────────────────
//
// Runs at half-res over the G-buffer (depth + octahedral normal). For each
// pixel we sample N points inside a hemisphere oriented along the surface
// normal, project each back to screen space, and check whether the actual
// scene depth there is closer to the camera than the sample's depth. If yes
// the sample is "occluded." Output is a single-channel visibility texture
// (1 = no occlusion, 0 = fully occluded) consumed by the lighting kernel.
//
// We use a fixed 16-sample low-discrepancy pattern (precomputed Hammersley-ish
// directions) rotated per-pixel by a hash of `gid`. That gives a noisy-but-
// deterministic AO that needs no temporal accumulation to be usable — a TAA
// resolve in a later phase will trade the noise for cleaner soft shadowing.

static inline float3 viewPosFromDepth(float2 ndcXY, float depth, float4x4 invProj) {
    float4 clip = float4(ndcXY, depth, 1.0);
    float4 view = invProj * clip;
    return view.xyz / view.w;
}
