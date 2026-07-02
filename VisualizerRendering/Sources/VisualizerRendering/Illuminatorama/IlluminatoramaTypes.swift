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
    public var _padCamera: Float = 0
    public var directionalLightDir: SIMD3<Float>      // world-space, toward light
    public var _padDir: Float = 0
    public var directionalLightColor: SIMD3<Float>    // pre-multiplied intensity
    public var _padColor: Float = 0
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
    /// Padding so the struct stride closes on a 16-byte boundary.
    public var _padPhase3: Float = 0
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
    /// 1 enables the SSAO bilateral spatial filter + temporal accumulation.
    public var ssaoDenoiseEnabled: UInt32 = 1
    /// History blend weight for the SSAO temporal pass: 0 = always current,
    /// 1 = frozen history. Typical: 0.90.
    public var ssaoTemporalBlend: Float = 0.90
    /// 1 enables SSR temporal accumulation (separate from the HDR TAA pass).
    public var ssrDenoiseEnabled: UInt32 = 1
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
    /// Phase 3.4 — per-pixel DDGI irradiance EMA cache. 1 = blend with
    /// history (cuts 8-probe blend to 1 texture read in steady state); 0 =
    /// recompute fresh probe blend every frame.
    public var ddgiIrrCacheEnabled: UInt32 = 1
    /// EMA blend alpha: 0 = freeze history, 1 = always fresh. Default 0.05
    /// converges in ~20 frames (~0.33 s at 60fps).
    public var ddgiIrrCacheBlend: Float = 0.05
    /// Leaf thin-sheet transmission strength (issue #58 / #20 item 2). 0 = OFF —
    /// the default for every scene, so the foliage-flag branch in the lighting
    /// kernel is a no-op unless a scene opts in. Foliage fragments are flagged in
    /// `normalRoughness.w` (a scene tags its leaf vertices with colour alpha 0).
    /// Fills the 12-byte trailing pad that `ddgiIrrCacheBlend` opened, so the
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
    // Phase 9 — film-stock LUT colour grade. Blends the 3D-LUT-graded result with
    // the ACES-tonemapped result. 0 = LUT fully bypassed (identity), 1 = full grade.
    // NEW 16-byte cluster (stride 1008 → 1024). Three float pads fill the cluster.
    public var filmLUTStrength: Float = 0
    public var _padFilmLUT0: Float = 0
    public var _padFilmLUT1: Float = 0
    public var _padFilmLUT2: Float = 0
    // ── Tonemap colour-grade ─────────────────────────────────────────
    // White-balance + tint are channel gains applied to LINEAR HDR before
    // exposure/ACES; contrast/shadows/highlights are a tone curve in the
    // tonemapped (0..1) domain. TWO new 16-byte clusters (stride 1024 →
    // 1056) — five scalars + three pads. Defaults are neutral so existing
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
    public var _padGrade1: Float = 0
    public var _padGrade2: Float = 0
}

/// World-space secondary directional light (#60 task 5 — retires the 4.20
/// "ambient fold" that collapsed every directional past the first into a flat
/// hemispheric ambient supplement). `dir` points TOWARD the light (pre-
/// normalized); `color` is the premultiplied linear-HDR intensity. The lighting
/// kernel shades it with the same diffuse + GGX-specular BRDF the primary sun
/// uses — no shadow map (SCN fill/back lights are `castsShadow = false`).
/// Field-for-field mirror of the Metal `DirectionalLight` (SIMD3 + pad, ×2 = 64 B
/// — `float3`/`SIMD3<Float>` are 16-byte-sized, so each (vec + pad) pair is 32 B;
/// the same layout as `IlluminatoramaPointLight`).
public struct IlluminatoramaDirectionalLight {
    public var dir: SIMD3<Float>             // toward the light (world, normalized)
    public var _pad0: Float = 0
    public var color: SIMD3<Float>           // pre-multiplied linear-HDR intensity
    public var _pad1: Float = 0

    public init(dir: SIMD3<Float>, color: SIMD3<Float>) {
        self.dir = dir
        self.color = color
    }
}

/// World-space rectangular area light (#60 task 5 — replaces the 4.24 5-spot
/// `.area` approximation). The rectangle is `center ± ex ± ey`, where `ex`/`ey`
/// are the half-width / half-height EDGE vectors in world space (so the four
/// corners need no extra basis), and the emitting normal is `normalize(cross(ex,
/// ey))`. The lighting kernel evaluates a closed-form polygon clamped-cosine
/// integral for diffuse (exact Lambert form-factor) plus a most-representative-
/// point specular. Field-for-field mirror of the Metal `AreaLight` (4× float4).
public struct IlluminatoramaAreaLight {
    public var center: SIMD3<Float>
    /// 1 = emits from both faces; 0 = front only (the `+normal` hemisphere).
    public var twoSided: Float
    public var ex: SIMD3<Float>              // half-width edge vector (world)
    public var _pad0: Float = 0
    public var ey: SIMD3<Float>              // half-height edge vector (world)
    public var _pad1: Float = 0
    public var color: SIMD3<Float>           // pre-multiplied intensity
    public var radius: Float                 // distance-falloff range (metres)

    public init(center: SIMD3<Float>, ex: SIMD3<Float>, ey: SIMD3<Float>,
                color: SIMD3<Float>, radius: Float, twoSided: Bool = false) {
        self.center = center
        self.ex = ex
        self.ey = ey
        self.color = color
        self.radius = radius
        self.twoSided = twoSided ? 1 : 0
    }
}

public struct IlluminatoramaPointLight {
    public var position: SIMD3<Float>
    public var radius: Float
    public var color: SIMD3<Float>           // pre-multiplied intensity
    public var _pad: Float = 0

    public init(position: SIMD3<Float>, radius: Float, color: SIMD3<Float>) {
        self.position = position
        self.radius = radius
        self.color = color
    }
}

/// World-space spot light. Cone angles are stored as the cosines of the
/// half-angles so the lighting kernel can compare a `dot()` result
/// directly. `direction` points AWAY from the apex (the way light
/// travels), matching the convention `SCNLight` uses for `.spot`.
public struct IlluminatoramaSpotLight {
    public var position: SIMD3<Float>
    /// `cos(spotInnerAngle / 2)` — full intensity inside this cone.
    public var innerCone: Float
    public var direction: SIMD3<Float>
    /// `cos(spotOuterAngle / 2)` — zero intensity outside this cone.
    /// `innerCone >= outerCone` since smaller angles → larger cosines.
    public var outerCone: Float
    public var color: SIMD3<Float>            // pre-multiplied intensity
    public var radius: Float
    /// Phase 4.10 — light-space view-projection for the spot's shadow
    /// map. Computed by the renderer each frame from
    /// `(position, direction, outerCone, radius)`. The lighting kernel
    /// projects each fragment into this space, compares against the
    /// stored depth, and modulates the BRDF contribution by the
    /// visibility term.
    public var shadowMatrix: simd_float4x4 = matrix_identity_float4x4
    /// Slice index into `IlluminatoramaRenderer.spotShadowAtlas`. `< 0`
    /// means "no shadow map for this spot" — the spot still contributes
    /// direct lighting, but with full visibility (no occlusion). Set by
    /// the renderer based on the spot's position in the array and
    /// `spotShadowAtlasCapacity`.
    public var shadowSliceIndex: Int32 = -1
    /// Three explicit pads so the struct closes on a 16-byte boundary.
    /// Stride bumps from 96 → 176.
    public var _padSpot0: Int32 = 0
    public var _padSpot1: Int32 = 0
    public var _padSpot2: Int32 = 0

    public init(position: SIMD3<Float>, direction: SIMD3<Float>,
                innerCone: Float, outerCone: Float,
                color: SIMD3<Float>, radius: Float) {
        self.position = position
        self.direction = direction
        self.innerCone = innerCone
        self.outerCone = outerCone
        self.color = color
        self.radius = radius
    }
}

/// Per-instance draw data. Bound as a structured buffer; vertex stage reads
/// `instances[instance_id]`.
public struct IlluminatoramaInstance {
    public var modelMatrix: simd_float4x4
    public var normalMatrix: simd_float4x4   // upper-3x3 inverse-transpose, padded
    public var albedo: SIMD3<Float>
    public var metallic: Float
    // Phase 7 — clearcoat: a second GGX lobe for polished/lacquered surfaces
    // (terrazzo, marble, glazed tile, lacquered wood). Sits in the 12-byte
    // padding gap between `metallic` and `emission` (offsets 148-159); stride
    // stays 208. Default 0 = no clearcoat (no change to existing materials).
    public var clearcoat: Float = 0          // lobe strength [0, 1]
    public var clearcoatRoughness: Float = 0.10  // GGX alpha^2 for clearcoat
    // Phase 7b — cloth sheen lobe strength [0,1] (velvet/wool/linen). Repurposes the former
    // `_padClearcoat` slot (same offset 156, stride stays 208). Packed as a NEGATIVE
    // emission.alpha in the G-buffer (a surface is polished OR cloth, never both).
    public var sheen: Float = 0
    public var emission: SIMD3<Float>
    public var roughness: Float
    /// Phase 4.0 — slice index into the diffuse albedo atlas. `< 0` means
    /// "no texture, use `albedo` colour".
    public var albedoTextureSlice: Int32 = -1
    /// Phase 4.1 — slice index into the non-colour material atlas (R/G/B
    /// channels each readable independently). `< 0` means "no texture,
    /// use `metallic` scalar".
    public var metallicTextureSlice: Int32 = -1
    /// Phase 4.1 — same atlas as metallic; `< 0` falls back to `roughness`.
    public var roughnessTextureSlice: Int32 = -1
    /// Phase 4.5 — slice index for the tangent-space normal map in the
    /// non-colour atlas. `< 0` means "no normal map, use geometric normal".
    public var normalTextureSlice: Int32 = -1
    /// Phase 4.9 — slice index for the emission map in the sRGB albedo
    /// atlas (emission is colour data, so it shares the sRGB-decoding
    /// atlas with diffuse). `< 0` falls back to `emission` scalar.
    public var emissionTextureSlice: Int32 = -1
    /// Phase 4.27b — multiplier applied to the emission TEXTURE sample so a
    /// texture-driven glow (Pizza's heat coils = emission texture × intensity)
    /// renders at its tuned HDR brightness instead of flat intensity-1. Solid
    /// emission already folds intensity into the `emission` scalar in the
    /// extractor, and has no texture to multiply, so this is a no-op there.
    /// Repurposes the former `_padSlice0` slot — same 4 bytes, stride stays 208.
    public var emissionIntensity: Float = 1.0
    /// Former trailing pad, repurposed (#60 item 7): non-zero ⇒ this instance
    /// is raster-only (its TLAS instance gets mask 0, so RT rays never
    /// intersect it; the RT representation comes from elsewhere — e.g. a
    /// registered `IlluminatoramaCurveSet` twin). Same 4 bytes.
    public var rtExclude: Int32 = 0
    // Phase 7 — detail-normal path for close-range pores/weave/grain.
    // Sampled at `detailNormalUVScale × in.uv` and blended into the
    // macro normal map result. Stride grows from 208 → 224 (next 16-byte
    // boundary) to maintain float4x4 natural alignment.
    /// Slice index for the detail normal map in the non-colour atlas.
    /// `< 0` = no detail normal (pass-through). Default -1.
    public var detailNormalTextureSlice: Int32 = -1
    /// Tile frequency of the detail normal relative to the macro UV.
    /// 8 = eight tiles per macro tile = fine grain/pore detail.
    public var detailNormalUVScale: Float = 8.0
    /// Phase 7c — anisotropy [0,1] (was `_padDetail0`, same offset/stride). Stretches the specular
    /// highlight along the surface grain; packed as (1 + anisotropy) into normalRoughness.w.
    public var anisotropy: Float = 0
    /// Highlight mode for the screen-space halo pass: 0 none · 1 selected (blue
    /// halo) · 2 hover (yellow halo). Only honoured on bounding-box proxies fed
    /// through `highlightMaskInstances`; the detailed scene meshes never set it.
    public var highlight: Int32 = 0
    // ── Drag/impact sway (vertex-shader secondary motion) ─────────────────────
    // A generic, GPU-side rigid bend driven by the host `DragSwayTracker`: the
    // sibling of `applyTreeWind` for non-foliage objects. The shader rotates the
    // vertex about the instance's world bottom-pivot + local-Z axis (both read off
    // `modelMatrix`, so no extra per-instance pivot data), then lifts it by jostle.
    // `swayMode 0` is a hard no-op, so every instance that doesn't opt in is
    // untouched. New 16-byte cluster (offsets 224-239): stride 224 → 240.
    /// Sway kind: 0 none · 1 bottom-pivot lean (books, upright shelf contents) ·
    /// 2 top-pivot pendulum (a hanging pendant: rigid rotation about the model origin —
    /// the ceiling anchor — self-oscillating from the frame time; see `applySway` in
    /// Illuminatorama.metal).
    public var swayMode: Int32 = 0
    /// Mode 1: static lean angle (radians) about local-Z, pivoting at the base.
    /// Mode 2: pendulum AMPLITUDE (max swing, radians) — the shader animates the swing
    /// from the frame time, so the host sets this once (no per-frame drive).
    public var swayLean: Float = 0
    /// Vertical pop (metres) added in world space — a knock hops the object up.
    public var swayJostle: Float = 0
    public var _padSway0: Float = 0

    public init(
        modelMatrix: simd_float4x4,
        albedo: SIMD3<Float> = SIMD3(0.8, 0.8, 0.8),
        metallic: Float = 0.0,
        roughness: Float = 0.5,
        emission: SIMD3<Float> = .zero,
        albedoTextureSlice: Int32 = -1,
        metallicTextureSlice: Int32 = -1,
        roughnessTextureSlice: Int32 = -1,
        normalTextureSlice: Int32 = -1,
        emissionTextureSlice: Int32 = -1,
        emissionIntensity: Float = 1.0
    ) {
        self.modelMatrix = modelMatrix
        self.normalMatrix = Self.normalMatrix(from: modelMatrix)
        self.albedo = albedo
        self.metallic = metallic
        self.roughness = roughness
        self.emission = emission
        self.albedoTextureSlice = albedoTextureSlice
        self.metallicTextureSlice = metallicTextureSlice
        self.roughnessTextureSlice = roughnessTextureSlice
        self.normalTextureSlice = normalTextureSlice
        self.emissionTextureSlice = emissionTextureSlice
        self.emissionIntensity = emissionIntensity
    }

    public mutating func setTransform(_ m: simd_float4x4) {
        self.modelMatrix = m
        self.normalMatrix = Self.normalMatrix(from: m)
    }

    /// Compile-time guard: Swift and Metal structs must agree on 240 bytes.
    /// If this fires, either a Swift field was added without the matching Metal
    /// field (or vice versa), or alignment changed unexpectedly.
    static let _assertStride240: Void = { assert(MemoryLayout<IlluminatoramaInstance>.stride == 240, "IlluminatoramaInstance stride must be 240") }()

    // ── Perfect analytic superquadric impostor — per-instance GPU param ────────
    //
    // Rides in a buffer PARALLEL to the instance buffer (NOT widened into
    // `IlluminatoramaInstance` — that would re-trigger the SIMD3 stride-16 vs
    // packed_float3 hazard and grow the 208 B stride for every instance engine-
    // wide). Indexed by the same grouped instance id as `instances[iid]`. Mirrors
    // `SQParam` in Illuminatorama.metal byte-for-byte (stride 80, 16-aligned).
    struct SuperquadricParam {
        var invModel: simd_float4x4    // world → object (= inverse(modelMatrix))
        var shape: SIMD4<Float>        // x=e1, y=e2, z=isEllipsoid(1/0), w=0
    }

    /// Inverse-transpose of the upper 3×3, padded out to float4x4 so we can
    /// memcpy into a shader struct without a separate `float3x3` codepath.
    /// Metal's `float3x3` has tricky alignment; float4x4 is unambiguous.
    static func normalMatrix(from model: simd_float4x4) -> simd_float4x4 {
        let upper = simd_float3x3(
            SIMD3(model.columns.0.x, model.columns.0.y, model.columns.0.z),
            SIMD3(model.columns.1.x, model.columns.1.y, model.columns.1.z),
            SIMD3(model.columns.2.x, model.columns.2.y, model.columns.2.z)
        )
        let inv = upper.inverse.transpose
        return simd_float4x4(
            SIMD4(inv.columns.0, 0),
            SIMD4(inv.columns.1, 0),
            SIMD4(inv.columns.2, 0),
            SIMD4(0, 0, 0, 1)
        )
    }
}

// ── AAA ray-traced glass (#60) ───────────────────────────────────────────────

/// Per-instance dielectric material for the ray-traced glass pass. The author-
/// facing knobs: index of refraction, a Beer–Lambert absorption tint + strength,
/// surface roughness (frosted glass), an extra reflection scale, and optional
/// chromatic dispersion. A scene flags an object as glass with one of these and
/// the renderer routes it through `illumi_glass_rt_fs` (true entry+exit
/// refraction traced against the scene TLAS).
public struct IlluminatoramaGlassMaterial: Equatable, Sendable {
    /// Beer–Lambert absorption colour — the tint the glass imparts as light
    /// travels through it. `(1,1,1)` is perfectly clear; a colour absorbs the
    /// complementary channels with path length. Also tints the Fresnel reflection.
    public var tint: SIMD3<Float>
    /// Index of refraction. Air ≈ 1.0, water 1.33, window glass 1.5,
    /// sapphire 1.77, diamond 2.42. Drives both the refraction bend and the
    /// Fresnel reflectance (F0 = ((ior−1)/(ior+1))²).
    public var ior: Float
    /// Surface roughness, 0 = optically polished, 1 = heavily frosted. Jitters
    /// the refracted + reflected rays in a cone; denoised by the renderer's TAA.
    public var roughness: Float
    /// Beer–Lambert absorption strength, per metre of path INSIDE the glass.
    /// 0 = no volumetric absorption (only the `tint` on transmission). Higher
    /// values make thick/coloured glass read denser and darker at its core.
    public var density: Float
    /// Scale on the Fresnel reflection term (1 = physical). Lower it to calm a
    /// busy reflection; raise it for a more mirror-like rim.
    public var reflectivity: Float
    /// Chromatic dispersion: 0 = off, larger spreads the per-wavelength IOR so
    /// the glass throws a prism rainbow. Costs 3× the refraction trace when > 0.
    public var dispersion: Float

    public init(tint: SIMD3<Float> = SIMD3(repeating: 1),
                ior: Float = 1.5,
                roughness: Float = 0,
                density: Float = 0,
                reflectivity: Float = 1,
                dispersion: Float = 0) {
        self.tint = tint
        self.ior = ior
        self.roughness = roughness
        self.density = density
        self.reflectivity = reflectivity
        self.dispersion = dispersion
    }

    /// Optically-clear window glass.
    public static let clearGlass = IlluminatoramaGlassMaterial(ior: 1.5)
    /// Water (IOR 1.33), faint blue-green tint.
    public static let water = IlluminatoramaGlassMaterial(
        tint: SIMD3(0.86, 0.95, 0.97), ior: 1.333, density: 0.6)
    /// Sapphire (IOR 1.77), cool blue absorption.
    public static let sapphire = IlluminatoramaGlassMaterial(
        tint: SIMD3(0.45, 0.62, 0.95), ior: 1.77, density: 2.0)
    /// Diamond (IOR 2.42) with strong dispersion (fire).
    public static let diamond = IlluminatoramaGlassMaterial(
        ior: 2.42, dispersion: 1.0)
}

/// One glass instance: a transform plus an `IlluminatoramaGlassMaterial`, packed
/// for the GPU. This is the single glass currency — the raster pass reads it for
/// the rasterised front surface's material, and the TLAS carries it so a
/// refraction/reflection ray that lands on glass knows its IOR/tint. Field
/// layout mirrors `GlassInstance` in IlluminatoramaGlassRT.metal (stride 176).
public struct IlluminatoramaGlassInstance {
    public var modelMatrix: simd_float4x4
    public var normalMatrix: simd_float4x4
    public var tintIor: SIMD4<Float>        // xyz = tint, w = ior
    public var rdrf: SIMD4<Float>           // x = roughness, y = density, z = reflectivity, w = reserved
    public var dispersionPad: SIMD4<Float>  // x = dispersion, yzw reserved

    public init(modelMatrix: simd_float4x4, material: IlluminatoramaGlassMaterial) {
        self.modelMatrix = modelMatrix
        self.normalMatrix = IlluminatoramaInstance.normalMatrix(from: modelMatrix)
        self.tintIor = SIMD4(material.tint, max(1, material.ior))
        self.rdrf = SIMD4(max(0, material.roughness), max(0, material.density),
                          max(0, material.reflectivity), 0)
        self.dispersionPad = SIMD4(max(0, material.dispersion), 0, 0, 0)
    }

    public mutating func setTransform(_ m: simd_float4x4) {
        self.modelMatrix = m
        self.normalMatrix = IlluminatoramaInstance.normalMatrix(from: m)
    }

    /// IOR readback (used by the renderer when packing the TLAS glass-data buffer).
    public var ior: Float { tintIor.w }

    /// Convenience initialiser from loose material knobs — mirrors the old
    /// repurposed-`IlluminatoramaInstance` glass encoding (tint = albedo,
    /// reflectivity = metallic, roughness, …) plus an explicit IOR, so a scene
    /// migrating from the legacy screen-space glass needs a one-line change.
    public init(modelMatrix: simd_float4x4, tint: SIMD3<Float>, ior: Float,
                roughness: Float = 0, reflectivity: Float = 1,
                density: Float = 0, dispersion: Float = 0) {
        self.init(modelMatrix: modelMatrix,
                  material: IlluminatoramaGlassMaterial(
                    tint: tint, ior: ior, roughness: roughness,
                    density: density, reflectivity: reflectivity,
                    dispersion: dispersion))
    }
}

/// GPU material for a glass TLAS hit, indexed by `instance_id - glassInstanceBase`
/// in the bounce loop. Mirror of `RTGlassData` in IlluminatoramaGlassRT.metal
/// (stride 48).
struct IlluminatoramaRTGlassData {
    var tintIor: SIMD4<Float>
    var rdrf: SIMD4<Float>
    var dispersionPad: SIMD4<Float>
}

/// Per-frame uniforms for the RT glass fragment shader. Mirror of
/// `GlassRTUniforms` in IlluminatoramaGlassRT.metal.
struct IlluminatoramaGlassRTUniforms {
    var cameraWorldPos: SIMD3<Float> = .zero; var rayTMin: Float = 0.004
    var sunDir: SIMD3<Float> = SIMD3(0, 1, 0); var sunSoftnessRad: Float = 0.01
    var sunColor: SIMD3<Float> = SIMD3(repeating: 1); var reflStrength: Float = 1
    var skyAmbient: SIMD3<Float> = .zero; var skyIntensity: Float = 1
    var glassInstanceBase: UInt32 = 0
    var maxBounces: UInt32 = 6
    var shadowRays: UInt32 = 1
    var frameSeed: UInt32 = 0
    var surfCacheEnabled: UInt32 = 0
    var surfTriCount: UInt32 = 0
    var surfAtlasW: UInt32 = 0; var surfAtlasH: UInt32 = 0
    var dispersionEnabled: UInt32 = 0
    /// Cheap glass mode for the fallback shader: 0 = plain Fresnel+sky (unchanged
    /// for every other scene), 1 = synthetic tinted+glint, 2 = screen-space refraction.
    var cheapGlassMode: UInt32 = 0
    /// Viewport size in pixels (mode 2 maps `clipPos.xy` → backdrop UV).
    var viewW: Float = 0; var viewH: Float = 0
}

/// Per-frame uniforms for the glass caustics kernels. Mirror of `CausticUniforms`
/// in IlluminatoramaCaustics.metal (stride 272 — each `SIMD3 + Float` pair is 32 B,
/// since both Swift `SIMD3<Float>` and Metal `float3` are 16-aligned/16-size).
struct IlluminatoramaCausticUniforms {
    var invViewProjection: simd_float4x4 = matrix_identity_float4x4
    var cameraWorldPos: SIMD3<Float> = .zero; var rayTMin: Float = 0.004
    var sunDir: SIMD3<Float> = SIMD3(0, 1, 0); var photonEnergy: Float = 1
    var sunColor: SIMD3<Float> = SIMD3(repeating: 1); var decay: Float = 0.9
    var aabbMin: SIMD3<Float> = .zero; var strength: Float = 1
    var aabbMax: SIMD3<Float> = .zero; var _pad0: Float = 0
    var width: UInt32 = 0; var height: UInt32 = 0
    var glassInstanceBase: UInt32 = 0
    var photonCount: UInt32 = 0
    var surfTriCount: UInt32 = 0; var surfAtlasW: UInt32 = 0; var surfAtlasH: UInt32 = 0; var frameSeed: UInt32 = 0
    var maxGlassBounces: UInt32 = 6; var glassDiscCount: UInt32 = 0; var _pad2: UInt32 = 0; var _pad3: UInt32 = 0
}

/// Vertex format consumed by the G-buffer vertex shader. Tightly packed so the
/// `MTLVertexDescriptor` is unambiguous and matches the `Vertex` struct in the
/// shader byte-for-byte. Phase 4.5 added the tangent + handedness slot to
/// support tangent-space normal-map sampling; stride goes 48 → 64 (33%
/// vertex-buffer growth, applied uniformly because the vertex shader can't
/// branch on per-mesh schema).
public struct IlluminatoramaVertex {
    public var position: SIMD3<Float>
    public var _padPos: Float = 0      // align next SIMD3 to 16
    public var normal: SIMD3<Float>
    public var _padNrm: Float = 0
    public var uv: SIMD2<Float>
    public var _padUv: SIMD2<Float> = .zero  // pad to 16 so stride was 48
    /// Tangent in object space (xyz) + handedness in `w`. The fragment
    /// shader builds the bitangent as `cross(normal, tangent) * w` so a
    /// mirrored mesh's TBN frame stays consistent. Zero tangent = "no
    /// normal-map data"; the fragment shader falls through to interpolated
    /// vertex normal.
    public var tangent: SIMD4<Float>
    /// Phase 4.17 — per-vertex RGBA color, multiplied into the
    /// fragment's albedo at shading time. Scenes that paint pattern
    /// detail (HotAirBalloon's chevron stripes, GiantGummyBears' candy
    /// gradients, anything with a `SCNGeometrySource(semantic: .color)`)
    /// store their pattern here so the deferred pass doesn't lose it.
    /// Default `(1, 1, 1, 1)` is a no-op — the fragment shader's
    /// `albedo *= color.rgb` then leaves the material's diffuse
    /// untouched for meshes that don't ship a colour source. Stride
    /// grows 96 → 112 (one float4 column). Synthesised tangents in the
    /// 4.13a repack kernel default the colour to white too.
    public var color: SIMD4<Float> = SIMD4(1, 1, 1, 1)

    public init(position: SIMD3<Float>,
                normal: SIMD3<Float>,
                uv: SIMD2<Float>,
                tangent: SIMD4<Float> = .zero,
                color: SIMD4<Float> = SIMD4(1, 1, 1, 1)) {
        self.position = position
        self.normal = normal
        self.uv = uv
        self.tangent = tangent
        self.color = color
    }
}

/// Camera state the host supplies each frame. Illuminatorama derives all
/// matrices from this — the host doesn't need to compute view/projection.
public struct IlluminatoramaCamera {
    public var position: SIMD3<Float>
    public var target: SIMD3<Float>
    public var up: SIMD3<Float>
    public var fovYRadians: Float
    public var aspect: Float
    public var zNear: Float
    public var zFar: Float

    public init(
        position: SIMD3<Float>,
        target: SIMD3<Float> = .zero,
        up: SIMD3<Float> = SIMD3(0, 1, 0),
        fovYRadians: Float = .pi / 3,
        aspect: Float = 16.0 / 9.0,
        zNear: Float = 0.1,
        zFar: Float = 200
    ) {
        self.position = position
        self.target = target
        self.up = up
        self.fovYRadians = fovYRadians
        self.aspect = aspect
        self.zNear = zNear
        self.zFar = zFar
    }

    public var viewMatrix: simd_float4x4 {
        let f = simd_normalize(target - position)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return simd_float4x4(
            SIMD4( s.x,  u.x, -f.x, 0),
            SIMD4( s.y,  u.y, -f.y, 0),
            SIMD4( s.z,  u.z, -f.z, 0),
            SIMD4(-simd_dot(s, position),
                  -simd_dot(u, position),
                   simd_dot(f, position), 1)
        )
    }

    public var projectionMatrix: simd_float4x4 {
        // Right-handed perspective with depth range [0, 1] (Metal-conventional).
        let yScale = 1 / tan(fovYRadians * 0.5)
        let xScale = yScale / aspect
        let zRange = zFar - zNear
        return simd_float4x4(
            SIMD4(xScale, 0, 0, 0),
            SIMD4(0, yScale, 0, 0),
            SIMD4(0, 0, -zFar / zRange, -1),
            SIMD4(0, 0, -(zFar * zNear) / zRange, 0)
        )
    }
}
