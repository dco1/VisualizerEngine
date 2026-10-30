import Metal
import simd

// ── ILLUMINATORAMA SHARED TYPES ──────────────────────────────────────────────
//
// Swift mirrors of the structs declared in `Shaders/Illuminatorama.metal`.
//
// ALIGNMENT RULE (mirrors the one in PBDSolver.swift): every type below has a
// 1:1 layout-compatible counterpart in the .metal source. SIMD3 lays out as
// 16-byte-aligned in Metal too (it's secretly a SIMD4 with a hidden lane), so
// pack carefully — do NOT mix scalar floats next to SIMD3 without thinking
// about which side of a 16-byte boundary the scalar lands on. The matching
// Metal struct uses `packed_float3` where we want true 12-byte tightness, and
// `float3` (16-byte) where Swift uses `SIMD3<Float>`.

/// Per-frame uniforms. One copy in a `MTLBuffer`, bound at index 0.
public struct IlluminatoramaFrameUniforms {
    public var viewProjection: simd_float4x4
    public var view: simd_float4x4
    public var projection: simd_float4x4
    public var invViewProjection: simd_float4x4
    /// Inverse of `projection`. Pre-computed on the CPU so view-space
    /// reconstruction (SSAO, SSR) doesn't pay for a per-pixel matrix inverse.
    public var invProjection: simd_float4x4
    /// Inverse of `view`. Used by the SSR/IBL paths to transform view-space
    /// directions back into world space for cubemap sampling.
    public var invView: simd_float4x4
    public var cameraWorldPos: SIMD3<Float>
    /// Aerial-perspective extinction σ (1/m); 0 = exact no-op. Repurposes the former
    /// `_padCamera` pad — see the Metal twin in IlluminatoramaCommon.h for the contract.
    public var aerialPerspectiveDensity: Float = 0
    public var directionalLightDir: SIMD3<Float>      // world-space, toward light
    /// DH-0140 — 1 when this frame's G-buffer pass wrote the extended sixth target and the
    /// lighting kernel may read it; 0 keeps the 5-target lighting byte-identical. Set by the
    /// renderer from `extendedGBufferEnabled`, never by a host. Repurposes `_padDir`.
    public var extendedGBuffer: Float = 0
    public var directionalLightColor: SIMD3<Float>    // pre-multiplied intensity
    /// Defocus-aware shading rate's gate (`lightingDefocusShadingMinCoC`; 0 = off). Repurposes the
    /// former `_padColor` pad — same 4 bytes, same offset.
    public var lightingCoarseMinCoC: Float = 0
    public var ambientColor: SIMD3<Float>
    public var exposure: Float
    public var bloomThreshold: Float
    public var bloomIntensity: Float
    public var pointLightCount: UInt32
    public var time: Float
    // ── Phase 2: SSAO + SSR knobs ────────────────────────────────────
    /// Strength of screen-space ambient occlusion (0 = disabled, 1 = full).
    public var ssaoIntensity: Float
    /// World-space radius of AO hemisphere samples, in metres.
    public var ssaoRadius: Float
    /// Strength of screen-space reflections (0 = disabled, 1 = full).
    public var ssrIntensity: Float
    /// View-space max ray length for SSR marches, in metres.
    public var ssrMaxDistance: Float
    /// View-space "thickness" tolerance for treating a depth crossing as a hit.
    public var ssrThickness: Float
    /// Cap on march steps per ray — also drives step length together with
    /// `ssrMaxDistance`.
    public var ssrMaxSteps: UInt32
    /// Pair of pad floats so the Phase-2 group closes on a 16-byte boundary
    /// before the Phase-3 IBL group begins (mirrors the Metal struct layout).
    public var _padPhase2A: Float = 0
    public var _padPhase2B: Float = 0
    // ── Phase 3: Sky-probe IBL ───────────────────────────────────────
    /// Master scale on diffuse + specular IBL contribution.
    public var iblIntensity: Float
    /// Number of mip levels in the prefiltered cube (needed for the
    /// roughness→LOD mapping the lighting kernel does per pixel).
    public var iblPrefilteredMipCount: UInt32
    /// 0 disables IBL (lighting kernel falls back to hemispheric ambient,
    /// sky pixels still come from the equirect texture).
    public var iblEnabled: UInt32
    /// Lamp-shade fabric translucency strength (DH-0458). 0 = OFF (the default for
    /// every scene) → the shade thin-sheet transmission branch in the lighting
    /// kernel is an EXACT no-op unless a host sets it. A paper/linen drum shade is
    /// a thin translucent sheet: this drives the back-lit transmission that lets the
    /// sunset (or the room behind the shade) scatter THROUGH the fabric instead of
    /// landing flat on an opaque cone. Shade fragments are flagged in
    /// `normalRoughness.w` (≈0.62 — a scene tags its shade vertices with colour
    /// alpha in (0.60, 0.66]). Repurposes the former `_padPhase3` pad — same 4
    /// bytes at the same offset, so the struct stride is unchanged. Mirrors
    /// `shadeTransmission` in the Metal `FrameUniforms`.
    public var shadeTransmission: Float = 0
    // ── Phase 2.5: Cascaded shadow maps ──────────────────────────────
    /// Per-cascade light-space view-projection matrices. Three cascades
    /// declared individually rather than as a 3-tuple/array so the layout
    /// matches Metal's three discrete `float4x4` fields with no padding
    /// surprises.
    public var shadowVP0: simd_float4x4
    public var shadowVP1: simd_float4x4
    public var shadowVP2: simd_float4x4
    /// (x,y,z) = far-Z of cascades 0/1/2 in view space, as positive distances
    /// from the camera. The lighting kernel uses `-view.z` (also positive) to
    /// pick which cascade covers each pixel. `.w` unused.
    public var cascadeSplitsView: SIMD4<Float>
    /// Constant depth bias subtracted from the light-space NDC.z before the
    /// PCF compare. Combats shadow acne on flat surfaces.
    public var shadowBias: Float
    /// Additional bias scaled by `(1 - NdotL_sun)`, so grazing-angle surfaces
    /// get more bias (where acne is worst) and head-on surfaces get less
    /// (where peter-panning would otherwise show).
    public var shadowSlopeBias: Float
    /// 0 disables shadows (the visibility term is forced to 1).
    public var shadowEnabled: UInt32
    /// PCF kernel radius: 0 = single tap, 1 = 3×3, 2 = 5×5.
    public var shadowPcfRadius: UInt32
    // ── Phase 2.7: Motion vectors + TAA ──────────────────────────────
    /// Previous frame's jittered view-projection matrix. Used by the G-buffer
    /// vertex shader to compute the per-vertex previous clip-space position,
    /// which the fragment then differences against the current clip-space
    /// position to write a screen-space velocity vector.
    public var previousViewProjection: simd_float4x4
    /// Blend factor for the current-frame sample in the TAA resolve. Smaller
    /// values give a smoother, more stable image but converge more slowly
    /// after disocclusion. Typical range 0.05–0.15.
    public var taaHistoryBlend: Float
    /// 0 disables TAA — resolve kernel writes current straight through to
    /// the bloom/tonemap input.
    public var taaEnabled: UInt32
    /// 1 marks the first frame after enable / resize / scene reset. The TAA
    /// resolve treats history as invalid and writes current through with
    /// full weight, primes the history texture for the next frame.
    public var taaIsFirstFrame: UInt32
    // ── Phase 3.2: Split-sum DFG LUT ─────────────────────────────────
    /// 0 disables the LUT, falling back to Lagarde's roughness-Schlick
    /// approximation for the F0-weighted IBL specular term (Phase 3.0
    /// behaviour). Repurposes the prior `_padPhase27` slot — same byte
    /// width, same 16-byte struct alignment.
    public var dfgLUTEnabled: UInt32
    // ── Phase 3.6: Spot lights ───────────────────────────────────────
    /// Number of `IlluminatoramaSpotLight`s in the spotLights buffer the
    /// lighting kernel reads at `buffer(3)`. Zero is a valid value — the
    /// loop early-outs and no spot-light contribution is added.
    public var spotLightCount: UInt32
    /// Phase 4.10 — constant depth bias for the per-spot shadow compare.
    public var spotShadowBias: Float
    // ── Phase 4.15: Tonemap + IBL saturation parity ──────────────────
    /// Saturation multiplier applied AFTER ACES tonemapping in
    /// `illumi_tonemap`. Narkowicz's fitted ACES still compresses midtone
    /// chroma a little harder than SCN's HDR chain, so a mild boost keeps
    /// parity with the SCN baseline without sacrificing the filmic shoulder.
    /// Was 1.22 while the tonemap double-encoded sRGB (see
    /// [[scenekit-background-texture-double-srgb]]) — that washout is fixed,
    /// so only the genuine ACES chroma compression remains to compensate;
    /// 1.10 covers it. 1.0 disables the boost. Repurposes the prior
    /// `_padSpot0` slot — same 4 bytes, same cluster alignment.
    public var tonemapSaturation: Float = 1.10
    /// Saturation boost applied to the IBL diffuse contribution in the
    /// lighting pass. Procedural-gradient backdrops that get reused as
    /// the IBL probe integrate down to a near-grey irradiance, and
    /// `diffuseIBL = kD · irradiance · albedo` then multiplies grey by
    /// every albedo → muted tints everywhere. Boosting the per-fragment
    /// diffuse-IBL chroma against its luminance preserves the original
    /// albedo's tint through the indirect term too. 1.0 disables the
    /// boost. Repurposes `_padSpot1`.
    public var iblDiffuseSaturation: Float = 1.30
    // ── Phase 4.21: Auto-exposure (16-byte cluster) ──────────────────
    /// 0 disables auto-exposure — tonemap falls back to the static
    /// `exposure` scalar earlier in this struct. 1 enables the
    /// `illumi_exposure_estimate` kernel path.
    public var autoExposureEnabled: UInt32 = 1
    /// Target log2-luminance the exposure adapts toward. SCN's
    /// `wantsExposureAdaptation` default lands near "mid-grey at 18%
    /// luminance" which is roughly EV-2 (log2 0.18 ≈ -2.47). The
    /// kernel uses `exposureScalar = 2^(targetEV - sceneLogLum)`, so
    /// scenes brighter than the target get scaled down and dimmer
    /// scenes get pumped up. -2.0 puts us close to SCN's auto-exposure
    /// for outdoor scenes; lower values (e.g. -3.0) brighten night
    /// scenes more aggressively. Stays within the same 16-byte
    /// alignment cluster as the other 4.21 fields.
    public var autoExposureTargetEV: Float = -2.0
    /// EMA half-life in seconds — how long the smoothing takes to
    /// move the exposure halfway to a new scene's target. ~0.25 s
    /// is the SCN default feel: fast enough to follow a scene cut
    /// without snapping, slow enough that a single bright pixel
    /// doesn't pump the whole image.
    public var autoExposureHalfLife: Float = 0.25
    /// Per-term split-render diagnostic. 0 = normal composite. Non-zero
    /// isolates a single lighting term so a flat/flooded scene can be
    /// decomposed (which term dominates the colour). See `IlluminatoramaRenderer.DebugTerm`.
    /// Repurposes the prior `_padPhase421` slot — same 4 bytes, stride unchanged.
    public var debugTerm: UInt32 = 0
    /// Desaturates the indirect FILL (diffuse-IBL irradiance + ambient
    /// supplement) toward luminance for highly-saturated probes. 0 = off.
    /// The hue-balance lever for colored-environment scenes (Pizza red flood) —
    /// distinct from the magnitude knobs (`iblIntensity`, exposure) that #46
    /// proved couldn't move it. Appended field; stride grows by one scalar.
    public var iblDiffuseDesaturation: Float = 0
    // ── Phase 4.39: AAA spatiotemporal denoiser ──────────────────────
    // Four 4-byte fields fit cleanly in the 12-byte implicit-padding gap
    // that `iblDiffuseDesaturation` (4 bytes past the previous 16-byte
    // boundary) left. Together these two clusters (4+4+4+4 / 4+4+4+4)
    // each fill a 16-byte slot, so the struct stride jumps by 32 bytes —
    // the compiler would have padded 28 bytes anyway; we reclaim them.
    /// 1 = the SSAO spatial kernel filters; 0 = it passes the raw AO through.
    /// (Renamed from `ssaoDenoiseEnabled` when spatial and temporal were split —
    /// same type, same slot, so the struct stride is unchanged. The temporal pass
    /// no longer needs a uniform: it is gated entirely CPU-side by
    /// `IlluminatoramaRenderer.ssaoTemporalEnabled`, which decides whether the
    /// dispatch is encoded at all.)
    public var ssaoSpatialEnabled: UInt32 = 1
    /// History blend weight for the SSAO temporal pass: 0 = always current,
    /// 1 = frozen history. Typical: 0.90.
    public var ssaoTemporalBlend: Float = 0.90
    /// 1 enables SSR temporal accumulation (separate from the HDR TAA pass).
    /// Renamed from `ssrDenoiseEnabled`: SSR's chain is temporal-only, so the old
    /// name implied a spatial stage that never existed. Same type, same slot.
    public var ssrTemporalEnabled: UInt32 = 1
    /// History blend weight for the SSR temporal pass. Typical: 0.85.
    public var ssrTemporalBlend: Float = 0.85
    /// 1 on the first frame after enable/resize — SSAO temporal skips history.
    public var ssaoIsFirstFrame: UInt32 = 1
    /// 1 on the first frame after enable/resize — SSR temporal skips history.
    public var ssrIsFirstFrame: UInt32 = 1
    /// 1 applies the triangular-PDF (TPDF) debanding dither in the tonemap
    /// before the 8-bit store; 0 disables it (raw quantised gradients). Exposed
    /// as a toggle so smooth-gradient banding can be A/B'd live. Repurposes
    /// the former `_padDenoise0` slot.
    public var debandDitherEnabled: UInt32 = 1
    /// DH-1305 — `IlluminatoramaRenderer.textureFilterMode` (was `_padDDGI0`, same slot).
    public var textureFilterMode: UInt32 = 0
    /// Signack S5 — `IlluminatoramaRenderer.leafTransmittanceRatio` (was `_padDDGI1`, same slot).
    public var leafTransmittanceRatio: Float = 0
    /// Leaf thin-sheet transmission strength (issue #58 / #20 item 2). 0 = OFF —
    /// the default for every scene, so the foliage-flag branch in the lighting
    /// kernel is a no-op unless a scene opts in. Foliage fragments are flagged in
    /// `normalRoughness.w` (a scene tags its leaf vertices with colour alpha 0).
    /// Fills the 12-byte trailing pad that the old irradiance-cache blend opened, so the
    /// struct stride is unchanged. Mirrors `leafTransmission` in the Metal
    /// `FrameUniforms` (and `RTUniforms`, which owns the sun under RT).
    public var leafTransmission: Float = 0
    /// #60 task 5 — number of rectangular LTC area lights in the `areaLights`
    /// buffer (bound at index 4 of `illumi_lighting`). Lands in the 4-byte
    /// trailing implicit pad (the struct already closed on a 16-byte boundary at
    /// stride 960), so the stride is unchanged. Mirrors the Metal `FrameUniforms`.
    public var areaLightCount: UInt32 = 0
    /// #60 task 5 increment 2 — 1 when the baked LTC specular LUT validated
    /// against brute-force ground truth; the lighting kernel then uses LTC for
    /// area-light specular, else the most-representative-point fallback. New
    /// 16-byte cluster (stride 960 → 976; mirrored in the Metal `FrameUniforms`).
    public var areaLTCEnabled: UInt32 = 0
    /// #60 task 5 — number of SECONDARY directional lights in the
    /// `extraDirectionals` buffer (bound at buffer(5) of `illumi_lighting`).
    /// These are the fill/back directionals a SCN 3-point rig ships alongside
    /// the key; formerly collapsed into a flat hemispheric ambient term (the 4.20
    /// "ambient fold"), they now shade with a real NdotL + GGX-specular BRDF (no
    /// shadow). Lands as the 3rd uint of the same trailing 16-byte cluster as
    /// `areaLightCount`/`areaLTCEnabled`, so the stride stays 976. Mirrors the
    /// Metal `FrameUniforms`.
    public var directionalLightCount: UInt32 = 0
    // ── Plush material (Teddy Bear Press) ────────────────────────────
    /// Plush fur sheen strength + backlit thin-fabric SSS strength. Both 0 by
    /// default → the plush branch in `illumi_lighting_fs` is an EXACT no-op for
    /// every scene (only the teddy-bear mesh tags its vertices with the plush flag
    /// — colour alpha ≈ 0.55 → `normalRoughness.w` ≈ 0.55h). A NEW 16-byte trailing
    /// cluster (stride 976 → 992), mirrored field-for-field in the Metal
    /// `FrameUniforms`. The two trailing pads keep the 16-byte alignment.
    public var plushSheen: Float = 0
    public var plushTransmission: Float = 0
    /// Lens-style transverse chromatic aberration strength, applied in the
    /// tonemap pass. 0 = OFF (the default) → an exact no-op. Repurposes the
    /// former `_padPlush0` slot, so the struct stride is unchanged. Mirrors
    /// `FrameUniforms.chromaticAberration` in the Metal shader.
    public var chromaticAberration: Float = 0
    /// Spherical-aberration radial blur strength (0 = OFF → exact no-op).
    /// Repurposes the former `_padPlush1` slot — same 4 bytes, stride unchanged.
    public var sphericalAberration: Float = 0
    // ── Axial chromatic aberration ("purple fringing") ───────────────
    /// Strength of the edge-fringing halo (0 = OFF → exact no-op) plus the
    /// dark-side tint in sRGB; the bright side of an edge gets its complement.
    /// A NEW 16-byte trailing cluster (stride 992 → 1008). FOUR scalar floats —
    /// NOT a `SIMD3` + pad — to dodge the SIMD3 (stride-16) vs `float3` mismatch.
    /// Field-for-field mirror of the Metal `FrameUniforms`.
    public var fringe: Float = 0
    public var fringeTintR: Float = 0.62
    public var fringeTintG: Float = 0.12
    public var fringeTintB: Float = 0.92
    // ── Vignette + film grain (issue #65 post-FX) ────────────────────
    /// Lens vignette + film-stock grain, applied at the tail of the tonemap pass.
    /// Both 0 by default → an EXACT shader no-op for every scene. A NEW 16-byte
    /// trailing cluster (stride 1008 → 1024); field-for-field mirror of the Metal
    /// `FrameUniforms`. `vignetteExtent` is the normalised radius kept fully bright;
    /// `filmGrainSize` is the grain cell size in output pixels.
    public var vignetteStrength: Float = 0
    public var vignetteExtent: Float = 0.55
    public var filmGrainStrength: Float = 0
    public var filmGrainSize: Float = 1.5
    // ── Colour-grade LUT (issue #65) ─────────────────────────────────
    /// 3D colour-grading LUT applied in the tonemap pass. `colorLUTAmount` 0 = OFF
    /// → exact shader no-op; `colorLUTSize` is the per-axis LUT resolution for the
    /// half-texel sampling correction. NEW 16-byte trailing cluster (stride 1024 →
    /// 1040, two trailing pads); field-for-field mirror of the Metal `FrameUniforms`.
    public var colorLUTAmount: Float = 0
    public var colorLUTSize: Float = 33
    // Velocity-buffer motion blur (issue #65). Repurposes the two trailing post-FX
    // pads — stride unchanged. 0 strength → exact no-op. `motionBlurStrength` scales
    // the per-frame screen velocity (≈ shutter fraction); `motionBlurMaxPx` clamps
    // the streak length in output px. Field-for-field mirror of the Metal struct.
    public var motionBlurStrength: Float = 0
    public var motionBlurMaxPx: Float = 48
    /// Screen-space contact shadows (issue #65). A short screen-space ray march
    /// toward the PRIMARY directional sun in the deferred lighting pass that
    /// catches the fine contact occlusion the cascaded shadow maps + RT soft
    /// shadows miss at object-base scale (a chip on felt, an egg on a floor).
    /// `contactShadowStrength` 0 = OFF → an EXACT shader no-op (the march is
    /// skipped). `contactShadowLength` is the march reach in WORLD units;
    /// `contactShadowSteps` the sample count (8–16); `contactShadowThickness`
    /// the occluder-depth window in world units (acne / over-darkening guard).
    /// NEW 16-byte trailing cluster (stride 1040 → 1056); field-for-field
    /// mirror of the Metal `FrameUniforms`.
    public var contactShadowStrength: Float = 0
    public var contactShadowLength: Float = 0.05
    public var contactShadowSteps: UInt32 = 12
    public var contactShadowThickness: Float = 0.02
    // ── Screen-space subsurface scattering (issue #65) ───────────────
    /// Jimenez-style separable screen-space SSS for skin / wax / marble / food.
    /// The deferred lighting pass writes the diffuse-lit term of SSS-flagged
    /// pixels (vertex-colour alpha ∈ [0.90, 0.98] → `normalRoughness.w` ≈ 0.95h)
    /// into a side buffer; a separable blur diffuses it by `sssRadius` (the
    /// per-channel diffusion mean-free-path in MILLIMETRES, scaled by `sssTint`);
    /// a composite blends it back: `hdr += sssStrength · (blurred − sharp)`.
    /// `sssStrength` 0 = OFF → the lighting pass skips the side-buffer write and
    /// the blur/composite passes aren't encoded at all → an EXACT no-op for every
    /// existing scene (matches the vignette/grain/motion-blur off-by-default
    /// pattern). `sssTint` biases which wavelengths scatter farthest — the default
    /// reddish skin profile (1.0, 0.4, 0.25) gives the characteristic warm bleed;
    /// marble ≈ (1,1,1), wax ≈ (1, 0.7, 0.45). NEW 32-byte trailing region (two
    /// 16-byte clusters; stride 1056 → 1088). The three trailing pads keep the
    /// final cluster 16-byte-aligned; field-for-field mirror of the Metal
    /// `FrameUniforms`.
    public var sssStrength: Float = 0
    public var sssRadius: Float = 8.0     // diffusion radius, millimetres
    public var sssTintR: Float = 1.0
    public var sssTintG: Float = 0.4
    public var sssTintB: Float = 0.25
    /// Headless-verify convenience (VIZ_ILLUMI_SSS_FORCEALL): 1 = treat EVERY
    /// opaque pixel as SSS-flagged so the effect can be render-verified without
    /// re-tagging a mesh. 0 in normal use → only the [0.90,0.98] flag band scatters.
    public var sssDebugForceAll: Float = 0
    /// Foliage-wind motion-vector delta, seconds (DH-0492). The G-buffer samples the
    /// PREVIOUS frame's `applyTreeWind` at `time − windPrevDelta` so a swaying canopy/blade
    /// writes a real screen-space velocity TAA can reproject (killing the sub-pixel crawl).
    /// 0 (the default) = previous sampled at the SAME time ⇒ the pre-DH-0492 no-op, so every
    /// scene that never sets it and every settled/frozen capture stays byte-identical.
    /// Repurposes the former `_padSSS1` slot — same offset, stride unchanged. Mirror of the
    /// Metal `FrameUniforms.windPrevDelta`.
    public var windPrevDelta: Float = 0
    /// DH-0881 — which scene → display rendering the tonemap uses. 0 = the shipped fitted
    /// ACES curve applied PER CHANNEL in Rec.709 primaries (byte-identical, the default),
    /// 1 = the same curve evaluated in ACEScg/AP1 so its channel clipping happens in a wide
    /// gamut and hue stops rotating as intensity rises, 2 = AgX. Repurposes the former
    /// `_padSSS2` slot — same offset, stride unchanged.
    public var displayTransform: UInt32 = 0
    // Phase 9 — film-stock LUT colour grade. Blends the 3D-LUT-graded result with
    // the ACES-tonemapped result. 0 = LUT fully bypassed (identity), 1 = full grade.
    // NEW 16-byte cluster (stride 1088 → 1104). Three float pads fill the cluster.
    public var filmLUTStrength: Float = 0
    /// DH-0441 — outer radius (metres) of the WIDE AO ring: a second GTAO march whose
    /// occluders are weighted to the annulus beyond `ssaoRadius`. Repurposes the former
    /// `_padFilmLUT0` slot — same offset, stride unchanged. Mirrors the Metal `FrameUniforms`.
    public var ssaoFarRadius: Float = 0.5
    /// DH-0441 — strength of the wide AO ring. 0 = OFF (the default) → the ring is never
    /// marched and the AO field is byte-identical. Repurposes the former `_padFilmLUT1` slot.
    public var ssaoFarIntensity: Float = 0
    /// Cells per axis of the film-stock cube — 16 for the legacy 8-bit PNG strips, 33 for
    /// the float16 `.lutcube` assets (DH-0879). The shader's half-texel inset is derived
    /// from this rather than hardcoded, so a host can ship a finer cube without an engine
    /// change. Repurposes the former `_padFilmLUT2` slot — same offset, stride unchanged.
    public var filmLUTSize: Float = 16
    // ── Tonemap colour-grade ─────────────────────────────────────────
    // White-balance + tint are channel gains applied to LINEAR HDR before
    // exposure/ACES; contrast/shadows/highlights are a tone curve in the
    // tonemapped (0..1) domain. TWO new 16-byte clusters (stride 1104 →
    // 1136) — five scalars + three pads. Defaults are neutral so existing
    // renders are byte-for-byte unchanged: whiteBalanceK 6500 → gain
    // (1,1,1), tint 0, shadows/highlights/contrast 1.0 → exact no-op.
    // Field-for-field mirror of the Metal `FrameUniforms`.
    public var whiteBalanceK: Float = 6500
    public var tint: Float = 0
    public var shadows: Float = 1.0
    public var highlights: Float = 1.0
    public var contrast: Float = 1.0
    // Phase 7 — opt-in hex-stochastic anti-tiling strength [0,1]. 0 = OFF (default):
    // the G-buffer shader short-circuits every `sampleAtlasHex` to a single plain
    // texture read, so opted-out scenes are byte-for-byte unchanged. Repurposes the
    // former `_padGrade0` slot (same 4 bytes, stride unchanged).
    public var antiTilingStrength: Float = 0
    // Point-light cubemap shadow depth bias (repurposes the former `_padGrade1`
    // slot — same 4 bytes, stride unchanged). Only read by the lighting kernel for
    // point lights with `shadowCubeIndex >= 0`; 0 for every scene that never enables
    // point shadows, so the byte layout AND behaviour stay identical. Mirrors the
    // Metal `FrameUniforms.pointShadowBias`.
    public var pointShadowBias: Float = 0.0006
    /// Scotopic (Purkinje) desaturation strength. Pulls the DIMMEST tonemapped
    /// pixels toward their own luminance so dark night surfaces read neutral-
    /// moonlit, not green (the grass-albedo × dim-neutral-light green cast). 0 =
    /// OFF (default) → the tonemap branch is an EXACT no-op, so day frames and
    /// every non-opting scene (Visualizer) are byte-identical. Repurposes the
    /// former `_padGrade2` slot — same 4 bytes, stride unchanged. Mirrors the
    /// Metal `FrameUniforms.scotopicDesaturation`.
    public var scotopicDesaturation: Float = 0
    // ── Interior day-light separation (opt-in; pairs with light-layer masking) ──
    // `interiorMask` = OR of every layer bit the host stamped on INTERIOR rooms; a
    // fragment whose gLayer is a real room stamp (≠ 0xFFFFFFFF) AND intersects the
    // mask gets its sky-IBL indirect scaled by mix(interiorIBLSide, interiorIBLUp,
    // saturate(N.y)) and its ambient supplement scaled by `interiorAmbient`. The
    // default `interiorMask == 0` disables the block — both factors stay exactly 1.0
    // in the kernel, so every scene that never opts in is byte-for-byte unchanged.
    // NEW 16-byte cluster (stride 1136 → 1152); mirror of the Metal `FrameUniforms`.
    public var interiorMask: UInt32 = 0
    public var interiorIBLUp: Float = 1
    public var interiorIBLSide: Float = 1
    public var interiorAmbient: Float = 1
    // ── Analytic night sky (stars + moon at SCREEN resolution) ──────────────
    // Set alongside `VolumetricCloudRenderer.Params.celestialsInDome = false`:
    // the lighting kernel then evaluates the star field + phase-correct moon
    // disk per SKY PIXEL instead of sampling their bilinearly-magnified dome
    // bakes (the "stars are gaussian blobs" artifact). All-zero defaults ⇒ the
    // kernel's sky branch adds exactly nothing — every non-opting scene renders
    // byte-identically. nightSkyParams: x = starBrightness × nightBlend,
    // y = moonIntensity × nightBlend, z = moon angular radius (radians),
    // w reserved. THREE new 16-byte clusters (stride 1152 → 1200); mirror of
    // the Metal `FrameUniforms`.
    public var nightSkyParams: SIMD4<Float> = .zero
    public var nightMoonDir: SIMD4<Float> = .zero
    public var nightSunDir: SIMD4<Float> = .zero
    // ── Lens flare (sun) ─────────────────────────────────────────────────────
    // x = strength (0 = OFF, default — the tonemap branch is gated on it so every
    // non-opting scene renders byte-identically), yz = the sun's screen-space uv,
    // w = on-screen weight (fades at the frame edge, 0 behind the camera). ONE
    // new 16-byte cluster (stride 1200 → 1216); mirror of the Metal FrameUniforms.
    public var lensFlareParams: SIMD4<Float> = .zero
    // ── Halation (film) ──────────────────────────────────────────────────────
    // The wide warm halo film wears around blown highlights: light punches through
    // the emulsion, reflects off the back of the base and re-exposes it, and the
    // anti-halation backing lets red survive that round trip best.
    // `halationParams`: x = intensity (0 = OFF, the default — the renderer skips the
    // halation passes AND the tonemap branch is gated on it, so every non-opting
    // scene renders byte-identically), y = HDR luminance threshold, z = halo radius
    // as a FRACTION OF FRAME HEIGHT, w reserved. `halationTint`: xyz = tail tint
    // (linear RGB), w reserved. TWO new 16-byte clusters (stride 1216 → 1248);
    // mirror of the Metal FrameUniforms.
    public var halationParams: SIMD4<Float> = .zero
    public var halationTint: SIMD4<Float> = .zero
    // ── Bloom pyramid (S1.2) ─────────────────────────────────────────────────
    // The Karis / Call-of-Duty dual-filter pyramid's three knobs.
    // `x` = soft-knee width as a FRACTION of `bloomThreshold`. 0 reproduces the
    //       old hard `max(0, lum - T)` cut exactly; the curve is identical to the
    //       hard one above T·(1+x) at any setting, so this only ever softens the
    //       pop as a light crosses the bar.
    // `y` = upsample scatter — the CONVEX `mix` weight toward the blurred low mip
    //       (0 = finest level only, 1 = coarsest only). Convex is load-bearing:
    //       level i ends up weighted (1-y)·y^i, which sums to exactly 1, so the
    //       chain's DC gain is 1 and `bloomIntensity` keeps its old meaning.
    // `z` = 3×3 tent radius in LOW-mip texels. `w` reserved.
    // ONE new 16-byte cluster (stride 1248 → 1264); mirror of the Metal
    // FrameUniforms.bloomParams.
    public var bloomParams: SIMD4<Float> = SIMD4(0.5, 0.7, 1.0, 0)
    // ── Diagram look (flat shading + outlines) ───────────────────────────────
    // An architectural-drawing mode: the tonemap pass REPLACES the shaded image
    // with hemisphere-wrapped G-buffer albedo and lays screen-space outlines over
    // it, in the same fragment that already has albedo/normal/depth bound for the
    // debug-view readouts (no extra pass, no extra binding, no extra texture).
    //
    // `diagramParams`:
    //   x = mix, 0…1. **0 is OFF and an exact no-op** — the branch never runs and
    //       every existing scene tonemaps byte-identically. 1 is the full drawing.
    //   y = light wrap, 0…1. 0 = perfectly flat fills (every face of a white wall
    //       the same white, so only the outlines separate them); 1 = a full
    //       sky/ground hemisphere term, so up-faces read lighter than side-faces.
    //       The reference look sits around 0.35 — enough for planes to read, not
    //       so much that it turns back into shading.
    //   z = outline strength, 0…1. 0 = no line work.
    //   w = orthographic flag (0/1). NOT a look control: it tells the shaders the
    //       eye is at infinity, so the ones that derive a view vector use the
    //       camera's constant forward instead of `eye − surface`. Set from
    //       `camera.isOrthographic`, never by hand.
    // `diagramOutline`: xyz = outline colour (linear RGB), w reserved.
    // `diagramEdge`: x = depth sensitivity, y = normal sensitivity, z = line
    //       thickness in output pixels, w reserved.
    // THREE new 16-byte clusters (stride 1264 → 1312); mirror of the Metal
    // FrameUniforms.
    public var diagramParams: SIMD4<Float> = .zero
    public var diagramOutline: SIMD4<Float> = SIMD4(0.10, 0.11, 0.13, 0)
    public var diagramEdge: SIMD4<Float> = SIMD4(1.0, 1.0, 1.0, 0)
    /// **This frame's sub-pixel jitter minus the previous frame's, in NDC** (xy;
    /// zw reserved) — the amount the G-buffer and the TAA resolve subtract back
    /// out of `currNDC − prevNDC` so the velocity buffer carries the motion of
    /// the SURFACE and not the motion of the sample pattern (S4.2).
    ///
    /// Both matrices the velocity is differenced from are jittered — the current
    /// VP with this frame's offset, `previousViewProjection` with last frame's —
    /// so without this the resolve reprojects the history by exactly the jitter
    /// delta and realigns it onto the current sub-pixel sample. Successive frames
    /// then never contribute a *different* sample to a fixed output pixel, which
    /// is the entire mechanism of jitter supersampling. Removing the delta is the
    /// standard formulation: it pins the history to the output pixel grid while
    /// the samples move across it.
    ///
    /// Zero whenever `taaJitterPixels` is 0 — the default, and every frame of
    /// every scene that never opts in — so the subtraction is an IEEE no-op and
    /// the velocity buffer is byte-identical. ONE new 16-byte cluster
    /// (stride 1312 → 1328); mirror of the Metal `FrameUniforms.taaJitterDelta`.
    /// `zw` (DH-1365) = `IlluminatoramaRenderer.reconstructionFilter.uniform`: (kernel, radius in
    /// output px); (0, 0) ⇒ the tonemap's legacy 4-tap reduction.
    public var taaJitterDelta: SIMD4<Float> = .zero
    /// Interior irradiance bands — the lawn-green-ceiling fix. xyz = irradiance,
    /// `interiorIrrUp.w` = band blend weight (0 = the exact cube sample, the default;
    /// hosts ramp it with daylight). See the Metal `FrameUniforms` twin for the physics
    /// and the fold of the diffuse interior scalar. THREE new 16-byte clusters
    /// (stride 1328 → 1376); mirror of `FrameUniforms.interiorIrr*`.
    public var interiorIrrUp: SIMD4<Float> = .zero
    public var interiorIrrSide: SIMD4<Float> = .zero
    public var interiorIrrDown: SIMD4<Float> = .zero
    /// Ray-traced soft sun shadows in the deferred kernel (S4.1, opt-in). Read ONLY
    /// by the `kLightingRTSunShadow` pipeline variant of `illumi_lighting` (which
    /// replaces the cascade-map sun visibility with `rtSunShadowRayCount` cone rays
    /// of half-angle `rtSunShadowAngle` against the TLAS); the non-RT variant —
    /// every scene that never opts in — never reads these, so the zero defaults are
    /// an exact no-op. `rtSunShadowSeed` WALKS only while a temporal accumulator is
    /// live and is frozen at 0 otherwise (the glass-pass frameSeed contract: a
    /// walking seed on a static frame is crawling speckle). ONE new 16-byte cluster
    /// (stride 1376 → 1392); field-for-field mirror of the Metal `FrameUniforms`.
    public var rtSunShadowSeed: UInt32 = 0
    public var rtSunShadowAngle: Float = 0
    public var rtSunShadowRayCount: UInt32 = 0
    /// S4.3 — ray-traced portal visibility (see `rtAreaShadowRays`). Occupies the former
    /// `_padRTSunShadow` slot, so the stride is unchanged. 0 ⇒ the PCF slice path, byte-identical.
    public var rtAreaShadowRayCount: UInt32 = 0
    /// Per-room interior band LEVEL (S3.5 Stage E). One gain per light-layer bit —
    /// the same 32-bit room identity `interiorMask` and `PointLight.layerMask` use —
    /// packed 4 to a vector, so a fragment's bits resolve to how much daylight ITS
    /// room admits. Without this the bands are one environment for the whole frame
    /// and a glass-walled room renders the same ceiling as a windowless closet.
    /// `interiorRoomGainMeta.x` = 0 (the default) ⇒ the shader resolve returns
    /// exactly 1.0 ⇒ byte-identical. Declared as eight named vectors rather than a
    /// Swift tuple: identical memory layout to the Metal `float4 [8]`, and indexable
    /// through `setInteriorRoomGains`. NINE new 16-byte clusters (stride 1392 →
    /// 1536); mirror of the Metal `FrameUniforms`.
    public var interiorRoomGain0: SIMD4<Float> = .one
    public var interiorRoomGain1: SIMD4<Float> = .one
    public var interiorRoomGain2: SIMD4<Float> = .one
    public var interiorRoomGain3: SIMD4<Float> = .one
    public var interiorRoomGain4: SIMD4<Float> = .one
    public var interiorRoomGain5: SIMD4<Float> = .one
    public var interiorRoomGain6: SIMD4<Float> = .one
    public var interiorRoomGain7: SIMD4<Float> = .one
    public var interiorRoomGainMeta: SIMD4<Float> = .zero   // x = enabled

    // ── Photographic finish: highlight chroma roll-off + split tone ──────────
    /// Fades chroma out as a pixel approaches white. ACES desaturates toward white
    /// on its own, but `tonemapSaturation` is a FLAT `mix(luma, colour, S)` — it
    /// pushes that chroma straight back out, which is how a lamp-lit wooden wall
    /// clips to saturated yellow instead of a pale warm white. 0 (default) ⇒ the
    /// tonemap branch never runs ⇒ byte-identical for every scene that never opts in.
    public var highlightChromaRolloff: Float = 0
    /// Split-tone temperature of the SHADOW end, in Kelvin, through the same
    /// `whiteBalanceGain` curve the global white balance uses (luma-normalized, so it
    /// tints without dimming) under a low-luma mask. 6500 (default) = gain (1,1,1) =
    /// exact no-op. Below 6500 cools the shadows (photo convention, DH-0453) — the
    /// archviz separation that gives warm practicals something to read against.
    public var shadowTemperatureK: Float = 6500
    /// Split-tone temperature of the HIGHLIGHT end. 6500 (default) = exact no-op.
    /// Pairs with `highlightChromaRolloff`: roll the scene's own chroma off as it
    /// approaches white, then put back a controlled amount of warmth.
    /// ONE new 16-byte cluster (stride 1536 → 1552); mirror of the Metal `FrameUniforms`.
    public var highlightTemperatureK: Float = 6500
    /// DH-0882 — NATURAL (cos⁴) vignetting, as `(halfFrameDiagonalMM / focalLengthMM)²` scaled
    /// by how much of the physical falloff the host wants. Every lens loses light off-axis as
    /// cos⁴ of the field angle, which is why a 16 mm is visibly darker in the corners and a
    /// 100 mm is not; the renderer had no notion of it, while the bokeh's `dofCatsEye` modelled
    /// the same effect. Because it is the LENS losing light it multiplies the scene before
    /// exposure and the tonemap — the shoulder can then recover — unlike `vignetteStrength`,
    /// which is a grade on the finished image. 0 = off (an exact no-op), which is the default,
    /// so every host that never sets it is byte-identical. Repurposes the former
    /// `_padPhotoFinish` slot — same offset, stride unchanged.
    public var naturalVignetteK: Float = 0
    // DH-0715 — live-lane look-match (see `IlluminatoramaTonemap.metal`). 0
    // (default) ⇒ byte-identical for every scene that never opts in. ONE new
    // 16-byte cluster (stride 1552 → 1568); mirror of the Metal `FrameUniforms`.
    public var liveLookMatchStrength: Float = 0
    public var liveLookGIDarken: Float = 0
    public var liveLookGIWarmth: Float = 0
    public var liveLookAODarken: Float = 0
    /// Frame-level gain on `IlluminatoramaInstance.tagGlow.x` (see the renderer's `tagGlowGain`).
    /// ONE new 16-byte cluster (stride 1568 → 1584).
    public var tagGlowGain: Float = 0
    /// Field sink progress 0…1 this frame and last (see the renderer's `fieldSinkProgress`).
    public var fieldSink: Float = 0
    public var prevFieldSink: Float = 0
    /// Aerial-perspective airlight override (was `_padTagGlow2`; same 4 bytes). 0 ⇒ default, so
    /// every scene that never sets it is byte-identical; > 0 ⇒ scale; < 0 ⇒ scale |v| and sample the
    /// horizon at the sharpest mip. See `IlluminatoramaRenderer.aerialPerspectiveAirlightScale` /
    /// `aerialPerspectiveAirlightFromHorizon`.
    public var aerialAirlightScale: Float = 0
    /// Physical night sky (IlluminatoramaNightSky.h) — see the Metal `FrameUniforms`.
    /// FOUR new 16-byte clusters (stride 1584 → 1648). All-zero ⇒ the legacy night sky and
    /// the legacy scotopic knee/tint — byte-identical for every host that never opts in.
    public var nightSkyExtra: SIMD4<Float> = .zero
    public var nightSkyExtra2: SIMD4<Float> = .zero
    public var nightCelestial: SIMD4<Float> = .zero
    public var scotopicParams: SIMD4<Float> = .zero
    /// Display chain for a physically-scaled night — see the Metal `FrameUniforms`: x = HDR
    /// pre-exposure (0 ⇒ 1), yz = the hue-stable toe knee [lo, hi] (z = 0 ⇒ off). ONE new
    /// 16-byte cluster (stride 1648 → 1664). All-zero ⇒ byte-identical.
    public var displayParams: SIMD4<Float> = .zero
    /// Traced-ray origin snap — see the Metal `FrameUniforms.rtSurfaceSnap`: x = the depth-ULP
    /// reach of `illumiSnapToVisibleSurface` (0 ⇒ off, byte-identical). 1664 → 1680.
    public var rtSurfaceSnap: SIMD4<Float> = .zero
    /// Digital sensor noise (Daydream DH-0992) — see the Metal `FrameUniforms.sensorNoise`:
    /// x strength (0 ⇒ off, byte-identical), y photo-electrons per exposed unit at the camera's
    /// ISO, z read noise (e⁻), w channel correlation. 1680 → 1696.
    public var sensorNoise: SIMD4<Float> = .zero
    /// The capture medium's emulsion grain (Daydream DH-1000) — see the Metal
    /// `FrameUniforms.filmMedium`: x relative σ (0 ⇒ off, byte-identical), y dye-layer
    /// correlation. 1696 → 1712.
    public var filmMedium: SIMD4<Float> = .zero

    /// Fill the eight gain vectors from a flat 32-entry table, and stamp the enable.
    public mutating func setInteriorRoomGains(_ gains: [Float], enabled: Bool) {
        let p = InteriorRoomGains.pack(gains, enabled: enabled)
        interiorRoomGain0 = p.v[0]; interiorRoomGain1 = p.v[1]
        interiorRoomGain2 = p.v[2]; interiorRoomGain3 = p.v[3]
        interiorRoomGain4 = p.v[4]; interiorRoomGain5 = p.v[5]
        interiorRoomGain6 = p.v[6]; interiorRoomGain7 = p.v[7]
        interiorRoomGainMeta = p.meta
    }
}
