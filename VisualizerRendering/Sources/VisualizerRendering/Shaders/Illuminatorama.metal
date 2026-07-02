// ── ILLUMINATORAMA ───────────────────────────────────────────────────────────
//
// TEMP DIAGNOSTIC (forest-tex session): set to 1 to flat-colour Forest wood by
// species marker (oak=red birch=white maple=blue log=green cap=yellow). REVERT
// to 0 before counting any fix; this is a marker-contract probe only.
#define FOREST_SPECIES_DEBUG 0
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

#include <metal_stdlib>
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
    float    _padCamera;
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
    float    _padPhase3;
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
    // Phase 9 — film-stock LUT blend strength. 0 = bypass, 1 = full grade.
    // NEW 16-byte cluster (stride 1008 → 1024). Three float pads fill.
    float    filmLUTStrength;
    float    _padFilmLUT0;
    float    _padFilmLUT1;
    float    _padFilmLUT2;
    // Tonemap colour-grade (white-balance / tint pre-tonemap; contrast / shadows
    // / highlights as a post-tonemap curve). TWO new 16-byte clusters (stride
    // 1024 → 1056). Defaults are neutral: whiteBalanceK 6500 → gain (1,1,1),
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
    float    _padGrade1;
    float    _padGrade2;
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
    float  _pad;
};

// Rectangular area light (#60 task 5). Mirror of Swift IlluminatoramaAreaLight.
// Rectangle corners = center ± ex ± ey; emitting normal = normalize(cross(ex,ey)).
struct AreaLight {
    float3 center;   float twoSided;   // twoSided: 1 = emit both faces, 0 = front only
    float3 ex;       float _pad0;      // half-width edge vector (world)
    float3 ey;       float _pad1;      // half-height edge vector (world)
    float3 color;    float radius;     // premultiplied color + distance-falloff range
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
    int      _padSpot0;
    int      _padSpot1;
    int      _padSpot2;
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
    float    _padSway0;
};

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

// ── Hierarchical tree wind (#58 #1) ──────────────────────────────────────────
// Vertex-shader vegetation wind (GG3-style): each tree vertex carries
// (swayWeight, phase, flutter) in its tangent (packed by ForestGeometry). The
// trunk base has swayWeight 0 so it stays planted while the canopy sways
// (cantilever). Layered: macro height-weighted bend + a coarse TRAVELLING GUST
// envelope (wind moves across the stand in waves) + high-freq leaf flutter.
// Gated by windStrength > 0 (repurposed frame._padPhase2A) → an exact no-op for
// every scene that doesn't set it; windHeading (frame._padPhase2B) is the dir.
static inline float3 applyTreeWind(float3 wp, float4 windAttr, float time,
                                   float windStrength, float windHeading) {
    float sway = windAttr.x;
    if (sway < 0.0001 || windStrength <= 0.0) return wp;
    float phase   = windAttr.y;
    float flutter = windAttr.z;
    float2 wdir = float2(cos(windHeading), sin(windHeading));
    float gust  = 0.55 + 0.45 * sin(time * 0.43 + dot(wp.xz, wdir) * 0.07);
    float macro = sin(time * 0.9 + phase);
    float bend  = sway * windStrength * (0.5 + gust) * macro;
    wp.xz += wdir * bend;
    wp.y  -= sway * windStrength * 0.18 * fabs(macro);   // slight droop as it bends
    // Leaf flutter: high-freq shimmer on foliage. Kept SMALL (sub-card amplitude)
    // and at LOW spatial frequency so neighbouring outer-rind cards flutter
    // COHERENTLY — at the prior 0.15 coefficient the per-vertex chaotic jitter
    // (~3 cm vs ~8.5 cm cards) slid adjacent rind cards apart and re-opened the
    // sealed crown silhouette (side/top "flake cloud" regression). The macro sway
    // above stays coherent (shared tree phase), so the crown leans as one mass.
    if (flutter > 0.0001) {
        wp.x += flutter * windStrength * 0.05 * sin(time * 6.5 + phase * 4.0 + wp.y * 1.1);
        wp.z += flutter * windStrength * 0.05 * sin(time * 5.7 + phase * 3.0 + wp.x * 1.1);
        wp.y += flutter * windStrength * 0.035 * sin(time * 6.0 + phase * 3.5);
    }
    return wp;
}

// ── Drag/impact sway (generic rigid secondary motion) ────────────────────────
// The non-foliage sibling of applyTreeWind: a placed object the host is dragging
// (or that just knocked into something) leans + hops, driven entirely by the
// per-instance swayMode/swayLean/swayJostle the host's DragSwayTracker fills.
//
// Applied in WORLD space about the instance's bottom-pivot so it composes cleanly
// with the per-instance non-uniform scale already baked into modelMatrix (rotating
// unit-box object space then scaling would shear). Pivot = base centre of the
// box; lean axis = the object's local +Z in world (modelMatrix column 2). swayMode
// 0 ⇒ identity, so every non-swaying instance is an exact no-op.
//
// swayMode reference:
//   0 · none                — hard no-op (all vegetation instances that don't opt in).
//   1 · bottom-pivot lean   — rigid rotation about the box BASE (object y=-0.5) by the
//                             host-supplied static `lean` angle (books, upright shelf
//                             contents, dragged/knocked props). `lean` IS the angle.
//   2 · top-pivot pendulum  — rigid rotation about the model ORIGIN (object y=0), for a
//                             HANGING object (ceiling pendant). The pendant mesh is
//                             authored with its ceiling anchor at object y=0 and its body
//                             hanging DOWN into −Y, and it's placed by an unscaled matrix,
//                             so object y=0 is exactly the ceiling attach point — the
//                             pivot. Self-oscillates in the shader: angle = `lean` *
//                             sin(time·ω + phase), so `lean` here is the AMPLITUDE (max
//                             swing, radians) and the host sets it ONCE (static
//                             per-instance) — no per-frame drive. Phase is derived from
//                             the pivot's world XZ so neighbouring pendants swing out of
//                             step. Displacement is 0 at the top anchor and grows toward
//                             the hanging bottom (pivot about the top) — the inverse of
//                             mode 1 (which pivots at the box base y=-0.5).
//
// Returns the rotated world position; `worldN`/`worldT` are rotated in place by the
// same rigid rotation so lighting/normal-mapping track the lean.
static inline float3 applySway(float3 wp, float4x4 model, int mode,
                               float lean, float jostle, float time,
                               thread float3& worldN, thread float3& worldT) {
    if (mode == 0) return wp;
    // Mode 1 pivots at the box base (y=-0.5) and applies `lean` directly. Mode 2 pivots
    // at the model origin (y=0) — the hanging pendant's ceiling anchor — and
    // self-oscillates `lean` (as amplitude) from `time`.
    float  pivotY = (mode == 2) ? 0.0 : -0.5;
    float3 pivot  = (model * float4(0.0, pivotY, 0.0, 1.0)).xyz;
    float3 axis   = normalize((model * float4(0.0, 0.0, 1.0, 0.0)).xyz);  // local +Z in world
    float  angle  = lean;
    if (mode == 2) {
        // Gentle pendulum ~0.42 Hz (ω ≈ 2.65 rad/s); per-instance phase from the top
        // anchor's world XZ so a row of pendants doesn't swing in lock-step.
        float phase = pivot.x * 1.7 + pivot.z * 2.3;
        angle = lean * sin(time * 2.65 + phase);
    }
    float  c = cos(angle), s = sin(angle);
    // Rodrigues rotation of (wp - pivot) about `axis`, then restore pivot.
    float3 r = wp - pivot;
    float3 rot = r * c + cross(axis, r) * s + axis * dot(axis, r) * (1.0 - c);
    worldN = worldN * c + cross(axis, worldN) * s + axis * dot(axis, worldN) * (1.0 - c);
    worldT = worldT * c + cross(axis, worldT) * s + axis * dot(axis, worldT) * (1.0 - c);
    return pivot + rot + float3(0.0, jostle, 0.0);
}

vertex VSOut illumi_vs(
    uint                       vid           [[vertex_id]],
    uint                       iid           [[instance_id]],
    const device Vertex*       verts         [[buffer(0)]],
    constant FrameUniforms&    frame         [[buffer(1)]],
    const device Instance*     instances     [[buffer(2)]],
    const device Instance*     prevInstances [[buffer(4)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    Instance prevInst = prevInstances[iid];

    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    // Tree wind displacement (no-op unless windStrength>0 and this vert carries a
    // sway weight). Same time for current + previous below: during a settled
    // headless capture time is frozen so the pose is static (no TAA smear); in the
    // live app the per-frame delta is tiny (gentle wind), matching the other
    // deforming-geometry scenes.
    worldP.xyz = applyTreeWind(worldP.xyz, v.tangent, frame.time,
                               frame._padPhase2A, frame._padPhase2B);
    float3 worldN = (inst.normalMatrix * float4(v.normal, 0.0)).xyz;
    // Previous-frame world position uses the previous-frame model matrix —
    // captures per-instance motion (spin, translation) on top of camera
    // motion. The renderer ping-pongs an instance buffer so prevInstances
    // holds last frame's data at the same instance index.
    float4 prevWorldP = prevInst.modelMatrix * float4(v.position, 1.0);
    prevWorldP.xyz = applyTreeWind(prevWorldP.xyz, v.tangent, frame.time,
                                   frame._padPhase2A, frame._padPhase2B);
    // Phase 4.5 — transform the object-space tangent into world. Using
    // `modelMatrix` (not normalMatrix) because a tangent is along the
    // surface — it should track non-uniform scale linearly, unlike a
    // normal. Handedness in .w rides along unchanged.
    float3 worldT = (inst.modelMatrix * float4(v.tangent.xyz, 0.0)).xyz;

    // Drag/impact sway — rigid lean+hop driven by the host DragSwayTracker, no-op
    // unless swayMode != 0. worldN/worldT rotate with it so lighting + normal maps
    // track the lean. The previous frame gets LAST frame's sway (prevInst) so the
    // motion vector captures the swing (TAA/motion-blur correctness).
    worldP.xyz = applySway(worldP.xyz, inst.modelMatrix, inst.swayMode,
                           inst.swayLean, inst.swayJostle, frame.time, worldN, worldT);
    float3 prevN = worldN, prevT = worldT;   // throwaway: prev normal/tangent unused
    // Prev frame uses the same `frame.time` as current (matching applyTreeWind above):
    // during a settled headless capture time is frozen (static pose, no TAA smear); in
    // the live app the per-frame swing delta is tiny, so the motion vector stays clean.
    prevWorldP.xyz = applySway(prevWorldP.xyz, prevInst.modelMatrix, prevInst.swayMode,
                               prevInst.swayLean, prevInst.swayJostle, frame.time, prevN, prevT);

    VSOut o;
    o.clipPos      = frame.viewProjection * worldP;
    o.worldPos     = worldP.xyz;
    o.worldNormal  = worldN;
    o.uv           = v.uv;
    o.instanceID   = iid;
    o.currentClip  = o.clipPos;
    o.previousClip = frame.previousViewProjection * prevWorldP;
    o.worldTangent = float4(worldT, v.tangent.w);
    o.vertexColor  = v.color;
    return o;
}

struct GBufferOut {
    half4 albedoMetallic   [[color(0)]];
    half4 normalRoughness  [[color(1)]];
    half4 emission         [[color(2)]];
    // Phase 2.7 — screen-space motion vector (current_uv - previous_uv), in
    // UV units (range typically ±0.5). RG16Float gives plenty of precision.
    half2 velocity         [[color(3)]];
};

// ── Shadow depth pass (Phase 2.5) ────────────────────────────────────────────
//
// Depth-only vertex shader, run once per cascade. The host binds the cascade's
// light-space view-projection matrix at buffer(3) — we don't read frame
// uniforms because the matrix changes per draw call. No fragment shader: the
// rasterizer writes depth into the assigned slice of the cascade array
// automatically.
vertex float4 illumi_shadow_vs(
    uint                        vid       [[vertex_id]],
    uint                        iid       [[instance_id]],
    const device Vertex*        verts     [[buffer(0)]],
    const device Instance*      instances [[buffer(2)]],
    constant float4x4&          lightVP   [[buffer(3)]],
    constant float&             shadowTime [[buffer(4)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    // Match the visible pose: a swaying object casts its leaned shadow (no-op unless
    // swayMode != 0). Position-only — the depth pass needs no normal/tangent.
    // `shadowTime` mirrors frame.time so a self-oscillating pendulum (swayMode 2) casts
    // its swung shadow in phase with the visible mesh.
    float3 nDummy = float3(0.0), tDummy = float3(0.0);
    worldP.xyz = applySway(worldP.xyz, inst.modelMatrix, inst.swayMode,
                           inst.swayLean, inst.swayJostle, shadowTime, nDummy, tDummy);
    return lightVP * worldP;
}

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

// ── Procedural soil material helpers (#58 dirt items #11/#12/#13) ─────────────
// Cheap value-noise + a finite-difference gradient for world-space macro/micro
// normal detail on the ground, so soil gets roughness/normal variation without a
// baked texture or extra geometry. Used only by the gated soil branch below.
static inline float soilHash(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
static inline float soilValueNoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = soilHash(i), b = soilHash(i + float2(1, 0));
    float c = soilHash(i + float2(0, 1)), d = soilHash(i + float2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// Cellular (Worley F1) field: returns x = distance to the nearest jittered cell
// centre (small = near a crack between cells), y = a per-cell ID in [0,1] (a stable
// random value shared by every pixel inside one cell). Used by the OAK bark branch
// to carve the trunk into DISCRETE shedding PLATES — the cell interior takes a
// coherent ridge tint, the inter-cell distance drives the dark furrow crack. Two
// hashes off the same integer cell give an independent centre offset + ID.
static inline float2 barkCellF1(float2 p) {
    float2 ip = floor(p), fp = fract(p);
    float bestD = 8.0;
    float bestID = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float2 g = float2(float(i), float(j));
            float2 cell = ip + g;
            float2 jitter = float2(soilHash(cell), soilHash(cell + 37.7));
            float2 diff = g + jitter - fp;
            float d = dot(diff, diff);
            if (d < bestD) { bestD = d; bestID = soilHash(cell + 91.3); }
        }
    }
    return float2(sqrt(bestD), bestID);
}
// 2D height gradient summed over a macro (~1.4 m) and a micro (~0.4 m) octave.
static inline float2 soilNormalGrad(float2 p) {
    const float e = 0.15;
    float2 g = float2(0.0);
    float2 pm = p * 0.7;
    g.x += soilValueNoise(pm + float2(e, 0)) - soilValueNoise(pm - float2(e, 0));
    g.y += soilValueNoise(pm + float2(0, e)) - soilValueNoise(pm - float2(0, e));
    float2 pf = p * 2.6;
    g.x += 0.5 * (soilValueNoise(pf + float2(e, 0)) - soilValueNoise(pf - float2(e, 0)));
    g.y += 0.5 * (soilValueNoise(pf + float2(0, e)) - soilValueNoise(pf - float2(0, e)));
    return g / (2.0 * e);
}

// Phase 4.25 (issue #60 item 5) — aspect-correct atlas sample. `uvScale[slice]`
// is the fraction of the square slice that the *letterboxed* source image fills
// (see IlluminatoramaTextureAtlas): (1,1) for a square texture, which takes the
// hardware-`repeat` fast path — bit-identical to the pre-aspect code, no seam.
// For a non-square (letterboxed) slice we tile MANUALLY with fract(uv)*scale and
// inset the lookup by a half-texel so the bilinear footprint never bleeds into
// the empty letterbox band. This manual wrap reintroduces a thin filter seam at
// tile boundaries on aspect-corrected textures only — the declared cost of
// keeping one uniform-size square slice array.
static inline float4 sampleAtlasAspect(texture2d_array<float, access::sample> atlas,
                                       sampler s, float2 uv, uint slice,
                                       const device float2* uvScale) {
    float2 sc = uvScale[slice];
    if (sc.x >= 0.999f && sc.y >= 0.999f) {
        return atlas.sample(s, uv, slice);            // square → hardware repeat
    }
    // Per-axis: the axis that fills the slice (sc == 1) keeps the raw uv so the
    // hardware `repeat` sampler wraps it seamlessly; only the LETTERBOXED axis
    // (sc < 1) is tiled manually with fract()*sc, half-texel-inset so the
    // bilinear footprint never bleeds into the empty band. So a 4:1 wall tiles
    // seamlessly along its long axis and carries the manual-tiling seam only on
    // the short (banded) axis.
    float2 texSize = float2(atlas.get_width(), atlas.get_height());
    float2 halfTexel = 0.5f / texSize;
    float2 manual = clamp(fract(uv) * sc, halfTexel, sc - halfTexel);
    float2 st;
    st.x = (sc.x >= 0.999f) ? uv.x : manual.x;
    st.y = (sc.y >= 0.999f) ? uv.y : manual.y;
    return atlas.sample(s, st, slice);
}

// Phase 7 — hex-stochastic atlas sample. Breaks repeating tiling on large planar
// surfaces (floors, walls) by blending three stochastically-offset samples whose
// cell borders align on a triangular lattice. GPU port of DaydreamCore's
// MaterialChannels.sampleAlbedoHex / hexHash (CPU reference in MaterialChannels.swift).
//
// `uv` — the same UV that sampleAtlasAspect would use (world-metres ÷ tile-metres,
//        so one UV unit = one texture tile). The hex cells and offset vectors are
//        expressed in that same space — the offset shifts into a different region of
//        the infinitely-tiling texture, so it's aspect-safe and letterbox-safe.
static inline float2 hexHash2D(float2 p) {
    // Integer bit-mixing hash — no sin/cos. Same constants as the CPU reference.
    // p is in skewed-lattice coordinates (integer-valued at cell vertices).
    float px = p.x * 73856093.0f + p.y * 19349663.0f;
    float py = p.x * 83492791.0f + p.y * 23994923.0f;
    int ix = int(px); int iy = int(py);
    ix ^= ix >> 11; ix *= 0x45d9f3b; ix ^= ix >> 16;
    iy ^= iy >> 11; iy *= 0x45d9f3b; iy ^= iy >> 16;
    return float2(float(ix & 0xFF) / 128.0f - 1.0f,
                  float(iy & 0xFF) / 128.0f - 1.0f);
}

// `strength` gates the whole effect. When strength <= 0 (the DEFAULT for every
// scene that never opts in) this returns EXACTLY `sampleAtlasAspect(atlas, s, uv,
// slice, uvScale)` — the identical single texture read the pre-anti-tiling shader
// did — so opted-out scenes (Visualizer) are byte-for-byte unchanged. For
// strength in (0,1] the three-tap hex blend is mixed in by `strength`, so the
// caller can dial the de-repetition from subtle to full.
static inline float4 sampleAtlasHex(texture2d_array<float, access::sample> atlas,
                                     sampler s, float2 uv, uint slice,
                                     const device float2* uvScale,
                                     float strength) {
    // Exact single-sample fast path — identical to sampleAtlasAspect, no extra taps.
    float4 single = sampleAtlasAspect(atlas, s, uv, slice, uvScale);
    if (strength <= 0.0f) { return single; }
    const float sq3over2 = 0.8660254f;         // sqrt(3)/2
    // Skew UV to triangular lattice: (u, v) → (u + v*0.5, v*sqrt(3)/2)
    float su = uv.x + uv.y * 0.5f;
    float sv = uv.y * sq3over2;
    float si = floor(su), sj = floor(sv);
    float fu = su - si, fv = sv - sj;

    // Barycentric weights and cell-vertex integer coords in skewed space.
    // Lower triangle: verts at (si,sj), (si+1,sj), (si,sj+1)
    // Upper triangle: verts at (si+1,sj+1), (si,sj+1), (si+1,sj)
    float2 sk0, sk1, sk2;
    float  w0, w1, w2;
    if (fu + fv < 1.0f) {
        sk0 = float2(si,       sj);
        sk1 = float2(si + 1.0f, sj);
        sk2 = float2(si,       sj + 1.0f);
        w0 = 1.0f - fu - fv; w1 = fu; w2 = fv;
    } else {
        sk0 = float2(si + 1.0f, sj + 1.0f);
        sk1 = float2(si,        sj + 1.0f);
        sk2 = float2(si + 1.0f, sj);
        w0 = fu + fv - 1.0f; w1 = 1.0f - fu; w2 = 1.0f - fv;
    }

    // Hash each vertex (in skewed-lattice integer space) to a UV offset.
    float2 h0 = hexHash2D(sk0), h1 = hexHash2D(sk1), h2 = hexHash2D(sk2);

    // Cubic blend weight (w^3 normalised) — smooth at cell boundaries.
    float p0 = w0 * w0 * w0, p1 = w1 * w1 * w1, p2 = w2 * w2 * w2;
    float pSum = max(p0 + p1 + p2, 1e-6f);

    float4 c0 = sampleAtlasAspect(atlas, s, uv + h0, slice, uvScale);
    float4 c1 = sampleAtlasAspect(atlas, s, uv + h1, slice, uvScale);
    float4 c2 = sampleAtlasAspect(atlas, s, uv + h2, slice, uvScale);
    float4 hex = (c0 * p0 + c1 * p1 + c2 * p2) / pSum;
    // Blend the de-repeated hex result toward the plain single sample by strength.
    // strength==1 → full hex, strength==0 handled by the early-out above.
    return mix(single, hex, saturate(strength));
}

fragment GBufferOut illumi_fs(
    VSOut                                       in              [[stage_in]],
    bool                                        frontFacing     [[front_facing]],
    const device Instance*                      instances       [[buffer(2)]],
    // Phase 7 — per-frame uniforms, bound to the fragment stage so the G-buffer
    // shader can read `antiTilingStrength` (the opt-in hex-stochastic de-repeat
    // knob). Same buffer the vertex stage reads at buffer(1); default strength 0
    // makes every hex sample fall through to the plain single texture read, so
    // scenes that never opt in are byte-for-byte unchanged.
    constant FrameUniforms&                     frame           [[buffer(1)]],
    // Phase 4.25 — per-slice UV scale for the two atlases (letterbox fill
    // fraction; (1,1) = square). Indexed by the instance's slice id; drives
    // `sampleAtlasAspect`. `float2` (8-byte stride) matches the host's
    // `SIMD2<Float>` exactly.
    const device float2*                        albedoUVScale   [[buffer(3)]],
    const device float2*                        nonColorUVScale [[buffer(5)]],
    // Phase 4.0 — atlas of diffuse-albedo textures. Each slice is the
    // 512×512 BGRA8-sRGB upload from `IlluminatoramaTextureAtlas`; the
    // texture-format's sRGB→linear decode happens automatically inside
    // `sample()`, so the value we feed into albedo is already linear.
    texture2d_array<float, access::sample>      albedoAtlas     [[texture(0)]],
    // Phase 4.1 — atlas of linear, non-colour material maps. Same shape
    // as the albedo atlas but BGRA8-Unorm (no sRGB decode), so values
    // round-trip as the GPU's exact 0–1 linear quantity. R channel =
    // metallic, G channel = roughness (matches SCN's PBR convention of
    // packing both into one image, e.g. `roughnessMetallicAO.png`).
    texture2d_array<float, access::sample>      nonColorAtlas   [[texture(1)]]
) {
    Instance inst = instances[in.instanceID];
    float3 n = normalize(in.worldNormal);
    // Two-sided meshes (cull .none) rasterize back faces too; flip the normal
    // so they're lit as a real surface instead of going dark/hollow. For
    // single-sided meshes back faces are culled, so this is always front (no-op).
    if (!frontFacing) { n = -n; }

    GBufferOut o;
    constexpr sampler texSampler(filter::linear,
                                 mip_filter::linear,
                                 address::repeat);

    // Phase 4.5 — tangent-space normal-map sampling. The atlas is the
    // same `bgra8Unorm` non-colour atlas as metallic/roughness; the
    // normal map encodes tangent-space (x,y,z) into [0,1] via
    // `0.5*(N+1)`. Decode with `2*sample - 1`, then transform into
    // world via TBN. When no normal map is bound, fall through to the
    // geometric normal `n`.
    if (inst.normalTextureSlice >= 0 &&
        length_squared(in.worldTangent.xyz) > 1e-4) {
        float3 T = normalize(in.worldTangent.xyz);
        float3 B = cross(n, T) * in.worldTangent.w;
        // Hex-stochastic normal map — eliminates the regular tiling pattern
        // on large flat surfaces (walls, floors) by blending three stochastically-
        // offset lattice samples. The blend is linear in encoded space (before
        // decode), which slightly overshoots tangent-space magnitude at boundaries
        // but gives correct appearance after renormalize below.
        float4 nmSample = sampleAtlasHex(nonColorAtlas, texSampler, in.uv,
                                         uint(inst.normalTextureSlice), nonColorUVScale,
                                         frame.antiTilingStrength);
        float3 tangentN = normalize(nmSample.xyz * 2.0 - 1.0);
        // Phase 7 detail normal — blended on top of the macro normal at
        // higher UV frequency (pores, weave, grain). Uses overlay-normal
        // blend: partial-derivative add in tangent space, then renormalize.
        if (inst.detailNormalTextureSlice >= 0) {
            float2 detailUV = in.uv * inst.detailNormalUVScale;
            float4 dnSample = sampleAtlasAspect(nonColorAtlas, texSampler, detailUV,
                                                uint(inst.detailNormalTextureSlice), nonColorUVScale);
            float3 dn = dnSample.xyz * 2.0 - 1.0;
            tangentN = normalize(float3(tangentN.xy + dn.xy, tangentN.z));
        }
        // World normal = T*x + B*y + N*z. Equivalent to a TBN matrix
        // multiply but cheaper to spell out as a single dot per axis.
        n = normalize(T * tangentN.x + B * tangentN.y + n * tangentN.z);
    }

    float3 albedo = inst.albedo;
    if (inst.albedoTextureSlice >= 0) {
        // Phase 7 hex-stochastic anti-tiling: floor/wall surfaces tile seamlessly but
        // don't repeat visibly — three blended samples from stochastically-offset
        // lattice cells. `in.uv` is already in tile-space (world ÷ uvMetres), so
        // the hex cell size equals one texture tile. sampleAtlasAspect handles
        // letterbox padding per-slice; the offset just shifts into a neighbouring
        // infinitely-tiling region, so aspect is preserved.
        float4 tx = sampleAtlasHex(albedoAtlas, texSampler, in.uv,
                                    uint(inst.albedoTextureSlice), albedoUVScale,
                                    frame.antiTilingStrength);
        albedo = tx.rgb;
    }
    // Phase 4.17 — modulate albedo by per-vertex color (default white,
    // so a no-op for meshes that ship no .color source). This is where
    // pattern detail painted into the vertex stream (chevron stripes,
    // candy gradients, anything fed via SCNGeometrySource(semantic:
    // .color)) actually paints the fragment. SceneKit convention is
    // multiplicative: vertex colour and material diffuse compose.
    albedo *= in.vertexColor.rgb;

    float metallic = inst.metallic;
    if (inst.metallicTextureSlice >= 0) {
        // SCN-style packed metallic-roughness usually puts metallic in B
        // (Blue) and roughness in G — but the exact channel depends on
        // how the asset was authored. Falling back to R keeps single-
        // channel grayscale metallic maps working without per-scene wiring.
        float4 tx = sampleAtlasAspect(nonColorAtlas, texSampler, in.uv,
                                      uint(inst.metallicTextureSlice), nonColorUVScale);
        metallic = tx.r;
    }

    float roughness = inst.roughness;
    if (inst.roughnessTextureSlice >= 0) {
        // Hex-stochastic roughness — same three-cell blend as albedo so
        // roughness variation doesn't lag the albedo tile seam.
        float4 tx = sampleAtlasHex(nonColorAtlas, texSampler, in.uv,
                                    uint(inst.roughnessTextureSlice), nonColorUVScale,
                                    frame.antiTilingStrength);
        roughness = tx.g;
    }

    // ── Procedural soil material (#58 #11/#12/#13) ───────────────────────────
    // Ground vertices pack soil data into uv as a NEGATIVE-x marker
    // (uv.x = -(roughness + 0.01), uv.y = wetness). Real UVs are never negative
    // and the branch is gated to UNTEXTURED instances, so this is an exact no-op
    // for every other surface and every other scene. Delivers per-region
    // roughness (#12), wetness darkening/smoothing (#12 — wet = darker, smoother,
    // more reflective), and world-space macro/micro normal relief (#13, #11 —
    // world-space so no UV stretching on slopes).
    if (inst.albedoTextureSlice < 0 && inst.roughnessTextureSlice < 0 &&
        inst.normalTextureSlice < 0 && in.uv.x < -0.005) {
        float soilRough = saturate(-in.uv.x - 0.01);
        float wet = saturate(in.uv.y);
        roughness = mix(soilRough, soilRough * 0.45, wet);   // wet → smoother + reflective
        albedo *= mix(1.0, 0.82, wet);                       // gentle extra darken (vertex colour already cools hollows)
        float2 grad = soilNormalGrad(in.worldPos.xz);
        // Dry ground takes more micro-relief; wet (pooled) ground reads smoother.
        float relief = 0.32 * (1.0 - 0.55 * wet);
        n = normalize(n + float3(grad.x, 0.0, grad.y) * relief);
        // ── GRASS-BASE MOTTLE (round-5 polish): a 20–40 cm value/hue mottle on the
        // ground ALBEDO everywhere (clearing + far) so the mossy floor between blades
        // stops reading as one flat green. Two octaves: ~33 cm patches + ~17 cm. Pulls
        // value ±18 % and shifts a fraction toward yellow-green / damp-dark so the
        // sward base has patchy life under the blades. No-op off the soil marker.
        {
            float2 gp = in.worldPos.xz;
            float mo0 = soilValueNoise(float2(gp.x * 3.0, gp.y * 3.0));   // ≈33 cm
            float mo1 = soilValueNoise(float2(gp.x * 6.0 + 17.0, gp.y * 6.0 - 9.0)); // ≈17 cm
            float mottleV = (mo0 * 0.65 + mo1 * 0.35);
            albedo *= 0.84 + 0.34 * mottleV;                             // value patches
            float3 damp = albedo * float3(0.78, 0.92, 0.70);             // damp-dark green
            float3 dryG = albedo * float3(1.12, 1.08, 0.74);             // yellow-green
            albedo = mix(albedo, damp, smoothstep(0.62, 0.30, mo0) * 0.35);
            albedo = mix(albedo, dryG, smoothstep(0.66, 0.88, mo1) * 0.30);
        }
        // ── FOREST-FLOOR DUFF (round 3, target 4): per-PIXEL leaf-litter ─────────
        // The vertex-baked duff (ForestGeometry.groundColor) can only carry ≈0.6 m
        // features (192-res / 120 m grid), so beyond the grass ring the floor read as
        // FLAT GREEN FELT at the hero/side cameras. Add per-pixel litter HERE where it
        // survives distance: BIG (≈8–15 cm) high-contrast tan/brown/dull-orange oval
        // blotches (x-stretched domain ⇒ leaf-shaped, not round dots), gated to the
        // out-of-clearing band (radius > ≈12 m from the clearing centre at z=−3) so
        // the mossy clearing floor stays clean. World-space → no swim, no UV stretch.
        float2 fp = in.worldPos.xz;
        float rad = length(float2(fp.x, fp.y + 3.0));
        // ROUND-5 floor fix: the band between the grass ring and the treeline
        // (≈8–16 m) read as FLAT SATURATED GREEN — the litter started too far out
        // (10→18 m) and was too weak to survive against the bright mossy-green
        // vertex base there. Start the duff sooner (the mossy CLEARING CENTRE at
        // z=−3 stays clean because the clearing radius is ≈9 m and the litter only
        // reaches full past ≈14 m), and make it strong enough to read.
        // ROUND-8: the 8→14 ramp left a HALF-treated annulus (litter at 20–60%)
        // exactly where the hero camera sees substrate past the grass ring, and it
        // still read as green felt (the top view proves duff fully lands past
        // 14 m). Tighten the ramp so full duff arrives where the dense sward ends.
        float farBand = smoothstep(7.0, 10.0, rad);    // 0 in clearing → 1 past 10 m
        if (farBand > 0.001) {
            // First, DESATURATE the bright-green vertex base toward a duff floor so
            // the litter has a neutral ground to sit on (the green felt was the base
            // colour, not the litter). Pull the saturated green toward a dull
            // olive-brown forest-floor tone, scaled by the band.
            float baseLum = dot(albedo, float3(0.299, 0.587, 0.114));
            // ROUND-8b: the old near-constant duffBase (0.07–0.10 regardless of
            // incoming value) compressed the vertex-baked MACRO patches (2–6 m
            // damp/dry swings — the only floor variation that survives 25 m +
            // atmosphere) down to ~22% of their range. Preserve incoming
            // LUMINANCE and shift only the HUE toward duff-brown, so the macro
            // value patchiness rides through the desaturation.
            float3 duffBase = float3(0.118, 0.096, 0.056) * saturate(baseLum * 3.0);
            albedo = mix(albedo, duffBase, farBand * 0.78);
            // Oval litter mask: x-stretched noise domain so blotches read as leaves.
            // Wider duty (0.42–0.78) so litter blotches are COMMON, not sparse.
            float lit = soilValueNoise(float2(fp.x * 9.0, fp.y * 4.5));   // ≈11/22 cm
            float lit2 = soilValueNoise(float2(fp.x * 3.2 + 50.0, fp.y * 3.2 - 20.0)); // coarse drifts
            float litMask = smoothstep(0.42, 0.74, lit) * (0.55 + 0.45 * lit2) * farBand;
            float litHue = soilValueNoise(float2(fp.x * 5.0 + 13.0, fp.y * 5.0 + 7.0));
            float3 litTan    = float3(0.170, 0.120, 0.058);
            float3 litOrange = float3(0.205, 0.105, 0.034);
            float3 litBrown  = float3(0.085, 0.058, 0.032);
            float3 litCol = (litHue < 0.4)
                ? mix(litBrown, litTan, litHue / 0.4)
                : mix(litTan, litOrange, (litHue - 0.4) / 0.6);
            albedo = mix(albedo, litCol, litMask * 0.80);
            // A few surviving green moss patches break the brown (not pure duff).
            float moss = smoothstep(0.74, 0.90, lit2) * farBand;
            albedo = mix(albedo, float3(0.070, 0.115, 0.045), moss * 0.5);
            // Twig/duff micro-normal: a coarser bump on the littered patches.
            float2 lg = soilNormalGrad(fp * 6.0);
            n = normalize(n + float3(lg.x, 0.0, lg.y) * (0.45 * litMask));
            roughness = mix(roughness, 0.92, litMask);
        }
    }

    // ── Procedural SPECIES-CORRECT bark + mossy-log surface (#58 / round 3) ──
    // ForestGeometry marks WOOD in worldTangent.w with a SPECIES/material code:
    //   2 = oak (deep furrowed PLATES)   3 = birch (pale + lenticel dashes)
    //   4 = maple (shaggy lifting strips) 5 = mossy fallen LOG (moss velvet)
    // Leaves are 1, everything else 0/±1, and NO other scene emits w > 1.5 (every
    // non-Forest mesh ships tangent .zero or, from this same file, wind w==1/2 on
    // the *same* Forest scene only). So `w > 1.5` gates cleanly to Forest wood and
    // is an exact no-op for foliage, ground, every prop without a marker, and every
    // other scene. The detail is COMPUTED (the soup has no texture atlas) and is far
    // crisper than the coarse 18-radial trunk mesh could carry.
    if (inst.normalTextureSlice < 0 && in.worldTangent.w > 1.5) {
        float species = in.worldTangent.w;                 // 2/3/4/5
        float ang = atan2(n.z, n.x);                       // sweeps around the trunk
        float3 tang = normalize(cross(n, float3(0.0, 1.0, 0.0)) + float3(1e-4, 0, 0));
        float yW = in.worldPos.y;
        float e  = 0.14;

#if FOREST_SPECIES_DEBUG
        // TEMP marker probe: flat colour per species marker so the contract
        // (geometry stamp vs shader gate) can be read off one render.
        // oak2=red, birch3=white, maple4=blue, log5=green, cap6=yellow.
        if (species < 2.5)      albedo = float3(1.0, 0.0, 0.0);
        else if (species < 3.5) albedo = float3(1.0, 1.0, 1.0);
        else if (species < 4.5) albedo = float3(0.0, 0.3, 1.0);
        else if (species < 5.5) albedo = float3(0.0, 1.0, 0.0);
        else                    albedo = float3(1.0, 1.0, 0.0);
#else
        if (species < 2.5) {
            // ── OAK — deep furrows broken into discrete shedding PLATE CELLS ────
            // ROUND-3 RE-ARCHITECT (the previous furrow-only field still read as a
            // smooth airbrushed gradient in hero_oakbark_tight — no plate structure
            // survived to 3–6 m). The fix the brief asks for is a real CELL field:
            // a Worley F1 lattice (barkCellF1) carves the trunk into discrete
            // plates, the inter-cell DISTANCE drives a dark furrow crack between
            // them, and each cell takes a COHERENT ridge tint by its per-cell ID.
            // The lattice is ANISOTROPIC (ang×3.4 around, yW×1.1 up ⇒ ≈18 cm wide ×
            // ≈30 cm tall plate cells — oak-correct vertical-ish plates) so it reads
            // as stacked vertical bark scales, not bubbles. A vertical furrow octave
            // grooves WITHIN each plate; a fine octave adds the close-up grain.
            float2 cellP = float2(ang * 3.4, yW * 1.1);
            float2 cell = barkCellF1(cellP);
            float crackDist = cell.x;                 // small ⇒ near a plate crack
            float cellID    = cell.y;                 // coherent per-plate random
            // Crack mask: 1 deep in a plate, 0 in the dark furrow border between
            // plates. Tight band so plates have flat lit faces + sharp dark grooves.
            float plate = smoothstep(0.06, 0.24, crackDist);
            // Vertical grooving within the plate (oak's lengthwise fissures).
            float2 vc  = float2(ang * 9.0, yW * 0.7);
            float vf   = soilValueNoise(vc);
            float vfa  = soilValueNoise(vc + float2(e, 0));
            float vfy  = soilValueNoise(vc + float2(0, e));
            float vGroove = smoothstep(0.38, 0.60, vf);
            float fine = soilValueNoise(float2(ang * 38.0, yW * 9.0)) - 0.5;
            // ── ROUND-5 STRUCTURAL #1: DOMAIN-WARPED BRICK-OFFSET HORIZONTAL CRACKS ─
            // The round-4 field (`hPhase = yW*7 + lowFreqAngNoise`) used a phase that
            // varied only SLOWLY with `ang` (×1.7), so adjacent furrow columns shared
            // nearly the same crack height — the horizontal cracks LINED UP across the
            // trunk into continuous rows. Combined with the regular ×9 vertical groove
            // that reads as a KNITTED MESH / SNAKESKIN grid (the basket-weave defect).
            //
            // The fix the reviewer prescribes: per-furrow-COLUMN brick-offset so plate
            // seams stagger between adjacent columns (never align into a grid), plus
            // low-freq jitter on BOTH crack spacing AND amplitude so some plates are
            // tall, some squat, and some cracks fade out entirely — cells read as
            // IRREGULAR polygons. Derive a discrete column index from `ang`, hash it
            // for an independent phase OFFSET (the brick stagger) and a per-column row
            // SPACING + a per-column crack DEPTH, so no two columns share a row line.
            float colW = 6.5;                                  // ~6–7 furrow columns/m circumference
            float colF = ang * colW;
            float colIdx = floor(colF);                        // furrow-column index
            // Brick stagger: each column gets its own phase offset in [0,1) so the
            // crack rows of neighbouring columns never align into a continuous course.
            float colPhase = soilHash(float2(colIdx, 3.1));
            // Per-column row SPACING jitter (±35 %): some columns stack squat plates,
            // some tall — so the horizontal seams don't form a regular ladder.
            float colSpace = 5.5 + 4.0 * soilHash(float2(colIdx, 11.7));   // 5.5–9.5 rows/m
            // Per-column crack DEPTH jitter: some columns' cracks fade out (shallow),
            // so a few plates merge vertically — irregular polygon cells, not a weave.
            // ROUND-5: raise the FLOOR (0.62) so EVERY column shows real horizontal
            // cracks — the prior 0.45 floor let too many columns merge vertically, so
            // the trunk read as continuous combed strands (shaggy maple) instead of
            // discrete stacked OAK plates. A few still merge (0.62→1.0), keeping the
            // irregular-polygon read without losing the plate-row structure.
            float colDepth = 0.62 + 0.38 * soilHash(float2(colIdx, 23.9));
            // Smoothly blend the per-column phase/spacing across the column boundary so
            // the brick offset doesn't itself read as a hard vertical seam line.
            float colMix = smoothstep(0.0, 1.0, fract(colF));
            float colPhaseB = soilHash(float2(colIdx + 1.0, 3.1));
            float colSpaceB = 5.5 + 4.0 * soilHash(float2(colIdx + 1.0, 11.7));
            float colDepthB = 0.45 + 0.55 * soilHash(float2(colIdx + 1.0, 23.9));
            float phase = mix(colPhase, colPhaseB, colMix);
            float space = mix(colSpace, colSpaceB, colMix);
            float crackDepthCol = mix(colDepth, colDepthB, colMix);
            // Low-freq AMPLITUDE jitter on top (slow in y) so a plate is taller here,
            // squatter there even within a column run.
            float ampJ = 0.7 + 0.6 * soilValueNoise(float2(colIdx * 0.7, yW * 0.6));
            float hPhase = (yW * space) * ampJ + phase * 1.31;
            float hTri = abs(fract(hPhase) - 0.5) * 2.0;        // 0 at crack → 1 mid-cell
            // crackDepthCol scales the crack WIDTH: shallow columns barely groove (the
            // "some cracks fade out" cue) so plates merge into irregular polygons.
            // ROUND-5: WIDEN the crack band (0.10→0.34) so the dark horizontal seam is
            // fat enough to read at hero range and physically separate the plate rows.
            float hCrack = mix(1.0, smoothstep(0.12, 0.34, hTri), crackDepthCol);
            // Combined "is this lit ridge-top vs dark damp floor": a pixel reads as
            // ridge only when inside a plate AND on a vertical ridge AND not in a
            // horizontal crack — so the ridges break into discrete blocky plate cells.
            // ROUND-6 STRUCTURAL #2 (oak side): the round-5 field STILL read as soft
            // painted streaks at hero distance — the horizontal cracks barely resolved
            // into discrete stacked plate CELLS. Two reasons: (a) the `ridge` value
            // floored at plate·0.45·0.18≈0.08 so the cracks never reached a true dark
            // floor, and (b) the albedo split (crackTint 0.22 → ridgeTint ~1.3) was a
            // ~6× spread but applied to a non-zero `ridge` ⇒ washed grey, not damp-dark
            // vs grey-lifted. Push BOTH: zero the floors so cracks cut to a true dark
            // furrow floor, and split harder (darker damp floor, brighter ridge top),
            // so each plate reads as a discrete lit cell with sharp dark seams.
            // The horizontal crack now FULLY gates the ridge (multiplicative, no floor)
            // so a horizontal seam carves the vertical ridge into stacked plate cells —
            // the oak-vs-maple tell the reviewer keeps flagging.
            float ridge = plate * (0.20 + 0.80 * vGroove) * hCrack;
            // ≈5× luminance split: weathered grey-lifted ridge tops ↔ DARK damp furrow
            // floors. Per-plate value jitter so adjacent plates differ.
            float plateVal = 0.78 + 0.5 * cellID;          // 0.78..1.28 per plate
            float3 ridgeTint = float3(1.44, 1.37, 1.22) * plateVal;  // bright weathered grey
            float3 crackTint = float3(0.135, 0.105, 0.075);          // dark damp furrow floor
            albedo *= mix(crackTint, ridgeTint, ridge) * (0.90 + 0.22 * fine);
            // Sparse lichen blotches — pale grey-green, slightly SMOOTHER — only on
            // the lit plate tops, keyed to whole plates (per-cell) so they blotch a
            // plate, not a pixel.
            float lichSel = soilHash(floor(cellP) + 5.1);
            float lich = soilValueNoise(float2(ang * 3.5 + 31.0, yW * 0.9 + 9.0));
            float lichM = step(0.82, lichSel) * smoothstep(0.55, 0.85, lich) * plate;
            albedo = mix(albedo, float3(0.42, 0.47, 0.36), lichM * 0.6);
            roughness = mix(roughness, roughness * 0.82, lichM);
            // Lift plate edges: push the dark cracks IN (recessed grooves) and tilt
            // the vertical grooving within plates.
            float gA = (vfa - vf) / e, gY = (vfy - vf) / e;
            float edge = 1.0 - plate;                       // 1 at the vertical plate crack
            float hEdge = 1.0 - hCrack;                     // 1 at the horizontal crack
            // ROUND-6: deepen the recess on BOTH crack axes so the plate edges lift in
            // relief (discrete cells under the rake light), and tilt the lit ridge top
            // OUTWARD along +n so the grey ridge faces the camera (plate-top read).
            n = normalize(n + tang * (gA * 1.1 + fine * 0.45)
                            + float3(0,1,0) * (gY * 0.30)
                            + n * (ridge * 0.22)                  // ridge tops lift toward camera
                            - n * (edge * 0.55 + hEdge * 0.55));  // recess BOTH crack axes deeper
        } else if (species < 3.5) {
            // ── BIRCH — pale paper bark, DARK HORIZONTAL lenticel dashes ────────
            // ROUND-3 RETUNE: at 3–10 m the old dashes (ang×22, smoothstep 0.62–0.80
            // — a narrow ~18 % duty) were too sparse + too thin to survive, and the
            // warm paper tint pushed the trunk pinkish-tan. Now: FATTER, ~3× DENSER,
            // higher-CONTRAST near-black dashes (lower angular freq ≈10–14 cm dashes,
            // wider duty), a brighter NEUTRAL-COOL paper white, and a dark peel band.
            float2 lc  = float2(ang * 2.4, yW * 0.9);          // band selector (big)
            float band = soilValueNoise(lc);
            // Dash field: MODERATE angular freq (fat dashes) broken into short
            // segments by a vertical mask. Wider smoothstep windows → ~3× the duty
            // cycle of the old field so dashes are common, not rare.
            // ROUND-5 polish #5: hero-facing birch dashes read too FAINT vs the
            // side-view birch that passes — at hero distance the moderate duty + 0.94
            // mix wasn't surviving. Widen the duty windows (denser dashes), and raise
            // the band coverage so dashes appear over more of the trunk, so hero
            // birches match the side one.
            float dashRun = soilValueNoise(float2(ang * 12.0, yW * 0.5));
            float dashGap = soilValueNoise(float2(ang * 4.5,  yW * 5.5));
            float dash = smoothstep(0.40, 0.60, dashRun) * smoothstep(0.34, 0.56, dashGap);
            // Lenticels concentrate in horizontal bands but appear broadly — wider
            // band coverage so dashes carry across more of the hero-facing trunk.
            float lent = dash * smoothstep(0.22, 0.52, band);
            // Occasional darker peel band / triangular scar (low freq, rare).
            float peel = smoothstep(0.76, 0.90, soilValueNoise(float2(ang * 1.6 + 7.0, yW * 0.5)));
            // NEUTRAL-COOL paper — but DON'T push >1.0 hard: the hero/foreground
            // birches were blowing to a featureless WHITE PAINT-DRIP smear (the
            // 1.18–1.22 multiplier compounded with the golden-hour sun term and
            // clipped, washing out the dashes). Hold the paper just under unity so
            // the trunk reads as bright bark with VISIBLE structure, not a clipped
            // white column. A faint per-trunk grey value noise breaks the flat sheet.
            float papGrain = soilValueNoise(float2(ang * 1.3 + 4.0, yW * 0.4)) - 0.5;
            float3 paper = float3(0.96, 0.97, 0.99) * (1.0 + 0.10 * papGrain);
            albedo *= mix(paper, float3(0.50, 0.45, 0.42), peel * 0.65);
            // DARK, high-contrast lenticel — drive nearly to black so the dash reads
            // even against the bright paper (full-weight mix so hero dashes hold).
            albedo = mix(albedo, float3(0.035, 0.032, 0.032), lent * 0.98);
            // Paper-smooth roughness; peel edges lift the normal slightly.
            roughness = clamp(roughness * 0.85, 0.30, 0.70);
            n = normalize(n + float3(0,1,0) * (lent * 0.22) + tang * (peel - 0.5) * 0.14);
        } else if (species < 4.5) {
            // ── MAPLE — narrow BROKEN vertical shaggy strips, lifted/curling edges ─
            // ROUND-5 STRUCTURAL #2: the round-3 tuning read as a smooth tan pole at
            // hero distance — its strips were too wide + too low-contrast to survive,
            // so no foreground trunk read as maple (it looked like generic oak/birch
            // wood). The reviewer wants maple UNMISTAKABLY DISTINCT from oak: NARROW,
            // BROKEN vertical shaggy strips with strong tangential normal at the strip
            // borders (lifted/curling edges) and NO horizontal cracking — it must NOT
            // share the oak crack field. Build a vertical-strip field, BREAK each strip
            // vertically into shaggy segments (a separate vertical break mask), drive a
            // wide albedo split (dark damp furrow ↔ pale grey lifted strip), and lift
            // the curling edges HARD (strong tangential tilt at the strip-vertical
            // borders, away from the strip — the peeling-up shag of maple bark).
            // ROUND-6 STRUCTURAL #2 (maple side): the reviewer says maple's strips
            // don't read NARROWER / more BROKEN than oak. Make the difference
            // unmistakable: narrow the strips further (ang×18 ≈ 6–7 cm vs oak's plate
            // cells), break them harder vertically into short shaggy SEGMENTS, push the
            // albedo split MORE (pure-black furrow ↔ pale lifted strip), and lift the
            // strip BORDERS hard with tangential normal (the peeling-up shag of maple),
            // with ZERO horizontal cracking (no oak crack field touches this branch).
            float2 sc  = float2(ang * 18.0, yW * 0.40);        // narrower strips (~6–7 cm)
            float2 sc2 = float2(ang * 38.0, yW * 1.8);
            float f   = soilValueNoise(sc) * 0.70 + soilValueNoise(sc2) * 0.30;
            float fa  = soilValueNoise(sc + float2(e,0)) * 0.70 + soilValueNoise(sc2 + float2(e,0)) * 0.30;
            float fy  = soilValueNoise(sc + float2(0,e)) * 0.70 + soilValueNoise(sc2 + float2(0,e)) * 0.30;
            float fine = soilValueNoise(float2(ang * 50.0, yW * 14.0)) - 0.5;
            // Vertical BREAK: strips snap into short shaggy SEGMENTS (the maple tell vs
            // oak's stacked plate cells). Stronger break (higher vertical freq, wider
            // gap window) so the strips are visibly BROKEN, not continuous combed lines.
            float vbreak = soilValueNoise(float2(ang * 7.0, yW * 4.2));
            float segMask = smoothstep(0.34, 0.50, vbreak);    // 0 at a strip break gap
            // Sharper, NARROWER strip mask → the shaggy lift concentrates at the borders.
            float strip = smoothstep(0.44, 0.52, f) * segMask;
            float border = (1.0 - abs(smoothstep(0.44, 0.52, f) - 0.5) * 2.0) * segMask;
            // WIDER albedo split, pulled DARKER/greyer than oak so the two species never
            // read alike: near-BLACK damp furrow ↔ pale weathered grey lifted strip.
            albedo *= mix(0.18, 1.22, strip) * (0.80 + 0.32 * fine);
            float gA = (fa - f) / e, gY = (fy - f) / e;
            // CURLING-EDGE lift: very strong tangential tilt at the vertical strip
            // borders (the bark peeling up), plus an outward push at the segment break
            // so the shaggy ends catch the rake light.
            float segEdge = (1.0 - segMask);                   // 1 at a vertical break
            n = normalize(n + tang * (gA * 2.7 + border * 1.5 + fine * 0.60)
                            + float3(0,1,0) * (gY * 0.30)
                            + n * (border * 0.35 - segEdge * 0.22));
        } else if (species < 5.5) {
            // ── MOSSY FALLEN LOG — BARREL (damp moss velvet + length-aligned bark) ─
            // ROUND-4 MATERIAL-GATING FIX: the concentric end-grain RING field was
            // firing on the BARREL circumference as alternating pale/dark candy
            // stripes (the "striped wicker basket" read). Rings now live ONLY on the
            // cut-end cap (marker 6, the `species >= 5.5` branch below) — this branch
            // is the BARREL only: moss velvet on top, bark on the underside, with
            // bark RIDGES running ALONG the log's length (not around it).
            float3 ref = (abs(n.y) > 0.9) ? float3(1,0,0) : float3(0,1,0);
            float3 lt = normalize(cross(n, ref));
            float3 lb = cross(n, lt);
            float3 wp = in.worldPos;
            float mu = dot(wp, lt), mv = dot(wp, lb);
            // ── ROUND-6 STRUCTURAL #1 — BARREL MATERIAL REBUILT ──────────────────
            // DIAGNOSIS (why the previous branch rendered a smooth tan dowel even
            // though the marker-5 gate provably fired): the moss was gated to the
            // GEOMETRIC up component `n.y` via `topness = smoothstep(-0.20,0.50,n.y)`.
            // A fallen log is HORIZONTAL, and the hero/low camera sees its SIDE, where
            // n.y≈0 ⇒ topness≈0.29 ⇒ `surf = mix(bark, velvet, 0.29·0.85+0.15≈0.40)`
            // — i.e. the visible side was 60 % BARK. The moss velvet DID survive to the
            // G-buffer, but only on the thin n.y≈1 crest the camera barely grazes. The
            // tan dowel WAS the bark underside, shown on the side. Plus `bareBark`
            // (coverage<0.46) punched even more bark through.
            // REBUILD: dense moss velvet over the whole UPPER HEMISPHERE (top crest AND
            // both upper flanks the camera actually sees), bark only on the genuine
            // UNDERSIDE (n.y<0), smooth blend at the equator. The split is on `topness`
            // but with a much LOWER, wider ramp so the sides read mossy; bark
            // breakthrough is confined to the underside and a few small dry clumps.
            float topness = smoothstep(-0.55, 0.10, n.y);   // moss covers upper hemi + sides
            // Velvet mottle: COARSE clumps dominate (≈12–30 cm), one fine octave for
            // the close fuzz. Weighted toward the low freq so it survives distance.
            float m0 = soilValueNoise(float2(mu * 4.5,  mv * 4.5));   // ≈20 cm patches
            float m1 = soilValueNoise(float2(mu * 11.0, mv * 11.0));  // ≈9 cm
            float m2 = soilValueNoise(float2(mu * 40.0, mv * 40.0));  // close fuzz
            float mottle = m0 * 0.58 + m1 * 0.28 + m2 * 0.14;
            // HIGH-CONTRAST damp velvet: dark olive-green floor ↔ bright yellow-green
            // tip clumps. Floor lifted off black (a damp forest moss is dark olive).
            float3 mossDamp = float3(0.040, 0.072, 0.030);
            float3 mossTip  = float3(0.245, 0.300, 0.082);
            float3 velvet = mix(mossDamp, mossTip, smoothstep(0.34, 0.70, mottle));
            // Bark breakthrough now SPARSE on the moss: only small DRY clumps on the
            // upper surface (rare), but the underside is fully bark. Tightened so the
            // barrel reads UNMISTAKABLY mossy from the side, not 60 % bark.
            float cover = soilValueNoise(float2(mu * 2.4 + 13.0, mv * 2.4 + 5.0));
            float dryClump = smoothstep(0.30, 0.14, cover);  // rare small dry patch
            // LENGTH-ALIGNED bark ridges on the exposed bark: the bark coordinate
            // `mv` runs roughly along the bole, so high-freq in `mv`, low-freq in `mu`
            // makes ridges that run ALONG the log — not stripes that wrap it.
            float barkRidge = soilValueNoise(float2(mu * 2.2, mv * 16.0));
            float3 barkUnder = float3(0.21, 0.135, 0.090) * (0.6 + 0.7 * m1)
                             * (0.78 + 0.5 * barkRidge);   // ridges run lengthwise
            // Moss dominates the upper hemisphere; bark shows on the underside (low
            // topness) and a few small dry clumps on top. The blend is on `topness`
            // (smooth at the equator) with only a sparse dry-clump break on top.
            float3 mossSurf = mix(velvet, barkUnder, dryClump * 0.5 * topness);
            float3 surf = mix(barkUnder, mossSurf, topness);
            albedo = surf;
            // Bark coverage (for the normal/roughness) is high only where bark wins.
            float barkAmt = (1.0 - topness) + dryClump * 0.5 * topness;
            // Fine fuzz normal (where moss is) so velvet catches the rake light as a
            // soft fur. Coarse mottle bumps the silhouette. Bark grooves on underside.
            float ep = 0.05;
            float gU = (soilValueNoise(float2((mu+ep)*40.0, mv*40.0)) - m2);
            float gV = (soilValueNoise(float2(mu*40.0, (mv+ep)*40.0)) - m2);
            float gC0 = (soilValueNoise(float2((mu+ep)*3.2, mv*3.2)) - m0);   // coarse bump
            float gC1 = (soilValueNoise(float2(mu*3.2, (mv+ep)*3.2)) - m0);
            float bg = (soilValueNoise(float2(mu*2.2 + ep, mv*16.0)) - barkRidge);
            n = normalize(n + (lt * (gU + gC0 * 2.6 + bg * barkAmt * 1.2)
                               + lb * (gV + gC1 * 2.6)) * (0.85 * topness + 0.20 * barkAmt)
                            + float3(0,1,0) * (mottle - 0.5) * 0.20);
            // Velvet is matte (high roughness); damp bark a touch glossier.
            roughness = mix(0.97, 0.90, barkAmt);
        } else {
            // ── MOSSY FALLEN LOG — CUT END-GRAIN CAP (marker 6) ─────────────────
            // ROUND-4: rings fire ONLY here. This face is the sawn/snapped END of the
            // bole (addMossyLog stamps materialMark = 6 on the cut-end disc; no other
            // surface emits w ≈ 6). Build a polar radial coordinate centred on the
            // cap's pith and draw concentric growth rings as a luminance alternation —
            // so rings appear on the cut face and NEVER wrap the barrel circumference.
            // The cap's geometric normal is ≈ axial, so the radial frame is built in
            // the cap plane (perpendicular to n).
            float3 ref = (abs(n.y) > 0.9) ? float3(1,0,0) : float3(0,1,0);
            float3 ct = normalize(cross(n, ref));
            float3 cb = cross(n, ct);
            float3 wp = in.worldPos;
            // Radial distance from the local pith. The cap is small (≈0.3 m), so use a
            // fract-free world projection; the baked vertex gradient handles absolute
            // centring, this adds the crisp ring alternation on top.
            float cu = dot(wp, ct), cv = dot(wp, cb);
            // A coherent per-cap centre from the low-freq field so the rings are
            // roughly concentric without packing the true pith position.
            float2 pith = (soilValueNoise(float2(cu, cv)) - 0.5) * 0.05;
            float rr = length(float2(cu, cv) - pith);
            // ≈5 rings over the cap radius: alternate light sap / dark latewood.
            float ringPhase = rr * 22.0;
            float ring = 0.5 + 0.5 * cos(ringPhase * 2.0 * 3.14159265);
            // Heartwood pith → paler sapwood radial base (the baked vertex colour
            // already trends this way; reinforce + add the ring alternation).
            // Warm wood, not black: the cut face is sap-bright with a darker (but not
            // black) latewood line so the END-GRAIN rings read as warm wood, not a
            // hole punched in the log.
            float3 latewood = float3(0.30, 0.20, 0.115);   // darker growth-ring line
            float3 earlywood = max(albedo, float3(0.34, 0.24, 0.15)) * 1.18;  // bright sap
            albedo = mix(latewood, earlywood, smoothstep(0.30, 0.70, ring));
            // Faint radial checks / rays + slight recess at each dark ring line.
            float ray = soilValueNoise(float2(atan2(cv, cu) * 6.0, rr * 3.0));
            albedo *= 0.88 + 0.24 * ray;
            roughness = mix(0.62, 0.85, ring);             // latewood a touch glossier
            n = normalize(n - n * (1.0 - ring) * 0.06);    // dark rings recess slightly
        }
#endif
    }

    // ── Procedural sausage-casing micro-detail (HotdogDropUltra) ──────────────
    // Gates on vertexColor.a ∈ (0.6, 0.9) — set to 0.75 by pbdTubeExpand.
    // All sausage-casing vertices share alpha=0.75; normal geometry is 1.0,
    // foliage is 0.0. This is a no-op for every other scene.
    //
    // PORT OF THE HotdogDrop+ SAUSAGE MATERIAL (HotdogPlusGeometry.swift):
    // Drop+ drives the frank with (a) a SUBTLE wrinkle/pore normal map
    // (octaves 8/24/64 over a 256px tile that wraps the ~2-unit circumference
    // ⇒ ~4/12/32 cycles per world unit, intensity 1.1 — skin texture, not
    // blisters) and (b) a low-frequency ROUGHNESS mottle map in [0.18, 0.78]
    // (octaves 4/10/28 per tile ⇒ ~2/5/14 c/u) so wet/dry patches catch
    // highlights independently — that spatial gloss variance, not albedo
    // mottle, is where the "just off the grill" glisten lives. Albedo stays
    // a solid per-instance colour, exactly like Drop+'s flat diffuse.
    // Drop+'s clearcoat (0.55 / cc-rough 0.18) has no Illuminatorama analog;
    // the 0.18 glossy floor of the roughness mottle approximates the glaze.
    if (in.vertexColor.a > 0.6 && in.vertexColor.a < 0.9) {
        // Build a CONTINUOUS tangent frame from the surface normal.
        // Previous version used a hard branch on abs(n.y): as a horizontal frank's
        // normal swept around the tube it crossed the 0.9 threshold, snapping the
        // reference vector (0,1,0)→(1,0,0) and rotating the entire noise-input
        // space discontinuously → visible diagonal two-tone seam on the franks.
        // Fix: blend the two candidate reference axes smoothly on n.y so the
        // tangent frame transitions without a step-discontinuity.
        float yBlend    = smoothstep(0.75f, 0.95f, abs(n.y));
        float3 blendRef = normalize(float3(yBlend, 1.0f - yBlend, 0.0f));
        float3 casingTang = normalize(cross(n, blendRef));
        float3 casingBtan = cross(n, casingTang);

        float3 wp = in.worldPos;
        float tu = dot(wp, casingTang);
        float tv = dot(wp, casingBtan);
        // Wrinkle/pore normal — Drop+'s sausageNormal equivalent in world space.
        // Gentle: combined peak tilt ≈ 4-5°, reads as skin texture in specular.
        const float sn0 = 4.0f, sn1 = 12.0f, sn2 = 32.0f;
        const float ep = 0.04f;
        float w0   = soilValueNoise(float2(tu * sn0, tv * sn0));
        float w0dx = soilValueNoise(float2((tu + ep) * sn0, tv * sn0));
        float w0dy = soilValueNoise(float2(tu * sn0, (tv + ep) * sn0));
        float w1   = soilValueNoise(float2(tu * sn1, tv * sn1));
        float w1dx = soilValueNoise(float2((tu + ep) * sn1, tv * sn1));
        float w1dy = soilValueNoise(float2(tu * sn1, (tv + ep) * sn1));
        float w2   = soilValueNoise(float2(tu * sn2, tv * sn2));
        float w2dx = soilValueNoise(float2((tu + ep) * sn2, tv * sn2));
        float w2dy = soilValueNoise(float2(tu * sn2, (tv + ep) * sn2));
        // Drop+ octave weights 0.6 / 0.25 / 0.15 at SUBTLE amplitude. The
        // round-31 1.7× lift made the casing read lumpy under a matte lobe;
        // with the clearcoat now supplying a real glint, faint wrinkle is
        // enough — the coat highlight picks it up (round-31 review).
        float gU = (w0dx - w0) * 0.050f + (w1dx - w1) * 0.022f + (w2dx - w2) * 0.012f;
        float gV = (w0dy - w0) * 0.050f + (w1dy - w1) * 0.022f + (w2dy - w2) * 0.012f;
        n = normalize(n + casingTang * gU + casingBtan * gV);

        // Roughness mottle — Drop+'s sausageRoughness map verbatim: big wet/dry
        // blotches, weights 0.65 / 0.25 / 0.10, range [0.18, 0.78], mean ≈ 0.48.
        // This is the SATIN base layer; the tight glint lives in the clearcoat
        // lobe the deferred lighting pass adds for casing-flagged pixels (see
        // illumi_lighting — flag 0.75 in normalRoughness.w). Folding the glaze
        // into this one lobe instead provably oscillates: [0.16, 0.62] read as
        // lacquered silicone (round 30), [0.30, 0.62] as foam rubber (round 31).
        // Overrides the per-instance scalar for casing pixels only.
        const float sr0 = 2.0f, sr1 = 5.0f, sr2 = 14.0f;
        float rN = soilValueNoise(float2(tu * sr0, tv * sr0)) * 0.65f
                 + soilValueNoise(float2(tu * sr1, tv * sr1)) * 0.25f
                 + soilValueNoise(float2(tu * sr2, tv * sr2)) * 0.10f;
        roughness = 0.18f + saturate(rN) * 0.60f;
    }

    // ── Procedural plush fuzz micro-normal (Teddy Bear Press) ─────────────────
    // Gates on vertexColor.a ∈ [0.5, 0.6] (≈0.55 from TeddyBearGeometry). A fine,
    // high-frequency normal break-up so the silhouette + specular read as short
    // fur fuzz rather than smooth rubber, and the per-instance high roughness stays
    // matte. No-op for every other scene (no other mesh ships alpha in this band).
    if (in.vertexColor.a >= 0.5f && in.vertexColor.a <= 0.6f) {
        float yBlend    = smoothstep(0.75f, 0.95f, abs(n.y));
        float3 blendRef = normalize(float3(yBlend, 1.0f - yBlend, 0.0f));
        float3 fuzzTang = normalize(cross(n, blendRef));
        float3 fuzzBtan = cross(n, fuzzTang);
        float3 wp = in.worldPos;
        float tu = dot(wp, fuzzTang);
        float tv = dot(wp, fuzzBtan);
        // High-frequency fuzz (≈ 60 / 150 cycles per world unit) at small amplitude
        // — a soft micro-tilt, not blisters. The matte roughness scatters it into
        // the velvety sheen the deferred plush lobe rim-lights.
        // Two scales of fur fuzz: a coarse tuft (~28 c/u, reads at the camera distance
        // the bears are seen from) + a fine fibre (~90 c/u). Amplitudes lifted so the
        // matte sheen visibly breaks up the silhouette instead of staying glassy-smooth.
        const float fn0 = 28.0f, fn1 = 90.0f, ep = 0.012f;
        float f0   = soilValueNoise(float2(tu * fn0, tv * fn0));
        float f0dx = soilValueNoise(float2((tu + ep) * fn0, tv * fn0));
        float f0dy = soilValueNoise(float2(tu * fn0, (tv + ep) * fn0));
        float f1   = soilValueNoise(float2(tu * fn1, tv * fn1));
        float f1dx = soilValueNoise(float2((tu + ep) * fn1, tv * fn1));
        float f1dy = soilValueNoise(float2(tu * fn1, (tv + ep) * fn1));
        float gU = (f0dx - f0) * 0.11f + (f1dx - f1) * 0.05f;
        float gV = (f0dy - f0) * 0.11f + (f1dy - f1) * 0.05f;
        n = normalize(n + fuzzTang * gU + fuzzBtan * gV);
    }

    o.albedoMetallic  = half4(half3(albedo), half(metallic));
    float2 oct = octEncode(n);
    // normalRoughness.w is otherwise unused (always 1) — repurpose it as a
    // foliage flag. Scenes that want leaf thin-sheet transmission tag their
    // leaf vertices with colour ALPHA 0 (opaque geometry keeps alpha 1); the
    // deferred lighting pass reads w < 0.5 to add the back-light term. No-op
    // for every existing scene (they all ship vertex alpha 1 → w stays 1).
    half foliageFlag = (in.vertexColor.a < 0.5) ? 0.0h : 1.0h;
    // Sausage casing (vertexColor.a = 0.75 from pbdTubeExpand) forwards its
    // flag into normalRoughness.w so the deferred lighting pass can add the
    // clearcoat glaze lobe. Every foliage test is `w < 0.5` and the matRough
    // cap below is `< 0.5h`, so 0.75 behaves as ordinary opaque geometry
    // everywhere except the clearcoat branch.
    if (in.vertexColor.a > 0.6 && in.vertexColor.a < 0.9) foliageFlag = 0.75h;
    // Plush flag (Teddy Bear Press): colour alpha ≈ 0.55 (band 0.5–0.6) → 0.55h.
    // 0.55 is not < 0.5 (so the foliage roughness cap below leaves plush's high
    // roughness alone) and not in (0.6,0.9) (so it skips the casing clearcoat); the
    // deferred lighting pass reads 0.5h<w<0.6h to add the fur sheen + SSS. No-op for
    // every other scene (no other mesh ships colour alpha in this band).
    if (in.vertexColor.a >= 0.5 && in.vertexColor.a <= 0.6) foliageFlag = 0.55h;
    // Waxy-leaf sheen (#58): a real leaf cuticle is markedly smoother than the
    // matte moss / bark / stone around it, so a single soup roughness reads
    // every surface as the same dry matte and leaves never catch a glint.
    // Cap foliage roughness so backlit / edge-lit leaves pick up a soft
    // specular highlight (the wet-canopy cue) while staying well short of
    // plastic. Gated by the foliage flag (w < 0.5 ⇒ foliage) → exact no-op for
    // every non-foliage surface and every scene that ships vertex alpha 1.
    float matRough = (foliageFlag < 0.5h) ? min(roughness, 0.46) : roughness;
    o.normalRoughness = half4(half(oct.x), half(oct.y), half(matRough), foliageFlag);
    // Phase 4.9 — emission can be a texture (used heavily by Plus scenes
    // for glow effects on rails / lamps / fire) or a scalar. Both are
    // additive on top of the lit colour. Sampling from the sRGB-decoded
    // albedo atlas means the bake gets linear RGB directly.
    float3 emission = inst.emission;
    if (inst.emissionTextureSlice >= 0) {
        float4 tx = sampleAtlasAspect(albedoAtlas, texSampler, in.uv,
                                      uint(inst.emissionTextureSlice), albedoUVScale);
        // Phase 4.27b — scale the emission texture by the material's
        // `emission.intensity` so a texture-driven glow reads at its tuned
        // HDR brightness (Pizza's heat coils were flat at intensity 1).
        emission += tx.rgb * inst.emissionIntensity;
    }
    // Phase 7 — pack clearcoat (≥0) OR cloth sheen (<0) into emission.alpha (was always 1.0,
    // unused). A surface is polished OR cloth, never both, so one channel carries either: > 0 =
    // polished/lacquered second GGX lobe; < 0 = velvet/wool grazing-Fresnel sheen (strength = -a).
    o.emission        = half4(half3(emission), half(inst.clearcoat > 0.0 ? inst.clearcoat : -inst.sheen));
    // Screen-space motion vector. NDC.y is up, UV.y is down → Y is flipped.
    // The result is (currentUV - previousUV), so history reprojection in the
    // TAA kernel is `historyUV = currentUV - velocity`.
    float2 currNDC = in.currentClip.xy  / in.currentClip.w;
    float2 prevNDC = in.previousClip.xy / in.previousClip.w;
    float2 velocityUV = (currNDC - prevNDC) * float2(0.5, -0.5);
    o.velocity        = half2(velocityUV);
    return o;
}

// ── Perfect analytic superquadric (hero primitive) ───────────────────────────
//
// A superquadric is rendered the CORRECT way — by ray–surface intersection in
// the fragment, NOT as a tessellated mesh — so its silhouette is mathematically
// exact at any zoom (no facets, ever). One logical object is drawn as two
// instances (see IlluminatoramaSuperquadric.swift): this IMPOSTOR (a [-1,1]
// bounding box whose fragments ray-trace the analytic surface and write the
// G-buffer + analytic depth + analytic motion vectors) and a moderate-tessellation
// triangle PROXY (raster-skipped, lives only in the TLAS so the object still
// casts RT shadows / appears in RT GI & reflections — the RT path is hardware-
// triangle-only).
//
// KEY TRICK: extents are folded into the instance's modelMatrix SCALE, so the
// object-space shape is always a UNIT superquadric (a=b=c=1, bounded by [-1,1]³).
// That makes step sizes scale-independent and lets the ellipsoid case reduce to a
// closed-form ray–UNIT-SPHERE intersection — the ellipsoid-ness (and the correct
// `normalize(localHit/extents²)` normal) falls straight out of the existing
// modelMatrix / normalMatrix, no special-casing.
//
// Convention (matched byte-for-byte by the proxy mesh + the SDF below):
//   F(x,y,z) = (|x|^(2/e2) + |y|^(2/e2))^(e2/e1) + |z|^(2/e1)
//   surface = F==1 ; inside F<1 ; e1==e2==1 → unit sphere.

struct SQParam {
    float4x4 invModel;   // world → object (precomputed CPU-side per instance)
    float4   shape;      // x=e1 (vertical squareness), y=e2 (horizontal), z=isEllipsoid(1/0), w=0
};

// Superquadric inside–outside function F (object space, unit shape a=b=c=1).
// <1 inside, >1 outside, ==1 on the surface.
static inline float sqField(float3 p, float e1, float e2) {
    float ax = pow(abs(p.x), 2.0 / e2);
    float ay = pow(abs(p.y), 2.0 / e2);
    float az = pow(abs(p.z), 2.0 / e1);
    return pow(ax + ay, e2 / e1) + az;
}

// Slab intersection with the object-space AABB [-1,1]³. Returns the entry/exit
// ray parameters; t1 < t0 means a miss. (oo,od) need not be normalized — t is in
// object-direction units, which is fine because every hit point is reconstructed
// as oo + od*t consistently.
static inline bool sqBoxRange(float3 oo, float3 od, thread float& t0, thread float& t1) {
    float3 inv = 1.0 / od;
    float3 a = (float3(-1.0) - oo) * inv;
    float3 b = (float3( 1.0) - oo) * inv;
    float3 tmin = min(a, b), tmax = max(a, b);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 >= max(t0, 0.0);
}

// Object-space ray–superquadric intersection. tHit is the near hit; oN is the
// outward object-space surface normal (central-difference gradient of F — robust
// at the axis/pole singularities where the analytic derivative of |x|^p (p<1)
// blows up). Returns false on a miss.
static inline bool sqIntersect(float3 oo, float3 od, float4 shape,
                               thread float& tHit, thread float3& oN) {
    float e1 = shape.x, e2 = shape.y;

    // ── Ellipsoid fast path: object space is the UNIT SPHERE. Closed-form. ──
    if (shape.z > 0.5) {
        float A = dot(od, od);
        float B = 2.0 * dot(oo, od);
        float C = dot(oo, oo) - 1.0;
        float disc = B * B - 4.0 * A * C;
        if (disc < 0.0) return false;
        float sq = sqrt(disc);
        float tn = (-B - sq) / (2.0 * A);
        float tf = (-B + sq) / (2.0 * A);
        float t = (tn > 1e-5) ? tn : tf;        // camera-outside → near root; inside → far
        if (t <= 1e-5) return false;
        tHit = t;
        oN = oo + od * t;                        // unit-sphere normal == position
        return true;
    }

    // ── General superquadric: sphere-trace the bounded field inside [-1,1]³. ──
    float t0, t1;
    if (!sqBoxRange(oo, od, t0, t1)) return false;
    float t = max(t0, 0.0);
    float th = -1.0;
    float startF = sqField(oo + od * t, e1, e2);
    if (startF <= 1.0) {
        // BOXY shapes nearly fill their AABB, so a face-on ray enters the solid
        // right at the box boundary (F ≤ 1 already). The entry point IS the
        // surface — without this, the near face is missed and the primitive
        // renders see-through (the green near-cube bug).
        th = t;
    } else {
        // 128 steps: stiff boxy fields (e≈0.1 → |x|^20) cross F=1 in a very thin
        // band, especially for face-grazing rays — too few steps step over it and
        // punch a see-through hole. A hero-count primitive can afford the density.
        const int   STEPS = 128;
        const float dt = (t1 - t) / float(STEPS);
        if (dt <= 0.0) return false;
        // Walk until F first drops to/below 1 (outside→inside), then bisect.
        for (int i = 0; i < STEPS; ++i) {
            float tn = t + dt;
            float f = sqField(oo + od * tn, e1, e2);
            if (f <= 1.0) {
                float lo = t, hi = tn;
                for (int b = 0; b < 12; ++b) {
                    float mid = 0.5 * (lo + hi);
                    if (sqField(oo + od * mid, e1, e2) > 1.0) lo = mid; else hi = mid;
                }
                th = 0.5 * (lo + hi);
                break;
            }
            t = tn;
        }
        if (th < 0.0) return false;
    }
    if (th <= 1e-5) return false;
    // Central-difference gradient of F → outward object-space normal.
    float3 ph = oo + od * th;
    const float h = 2e-3;
    float gx = sqField(ph + float3(h,0,0), e1, e2) - sqField(ph - float3(h,0,0), e1, e2);
    float gy = sqField(ph + float3(0,h,0), e1, e2) - sqField(ph - float3(0,h,0), e1, e2);
    float gz = sqField(ph + float3(0,0,h), e1, e2) - sqField(ph - float3(0,0,h), e1, e2);
    tHit = th;
    oN = float3(gx, gy, gz);
    return true;
}

struct SQImpostorVSOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   iid      [[flat]];
};

// The impostor draws the instance's [-1,1] bounding box; the fragment does the
// real intersection. The box is just there to generate fragments over the
// silhouette's screen footprint — back faces are rendered (cull .front host-side)
// so the primitive survives the camera entering the bounding box.
vertex SQImpostorVSOut illumi_superquadric_impostor_vs(
    uint                       vid       [[vertex_id]],
    uint                       iid       [[instance_id]],
    const device Vertex*       verts     [[buffer(0)]],
    constant FrameUniforms&    frame     [[buffer(1)]],
    const device Instance*     instances [[buffer(2)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    SQImpostorVSOut o;
    o.clipPos  = frame.viewProjection * worldP;   // jittered current VP (matches illumi_vs)
    o.worldPos = worldP.xyz;
    o.iid      = iid;
    return o;
}

// Writes the SAME G-buffer layout as illumi_fs, plus analytic [[depth]]. Renders
// into the same G-buffer encoder/pass. `[[depth(less)]]`: with back faces, the
// analytic surface is always at or in front of the rasterized far-wall depth, so
// the written depth is ≤ the interpolated one — a valid promise (no early-Z gain
// under the `.less` compare, negligible for a hero-count primitive).
struct SQImpostorFSOut {
    half4 albedoMetallic   [[color(0)]];
    half4 normalRoughness  [[color(1)]];
    half4 emission         [[color(2)]];
    half2 velocity         [[color(3)]];
    float depth            [[depth(less)]];
};

fragment SQImpostorFSOut illumi_superquadric_impostor_fs(
    SQImpostorVSOut            in            [[stage_in]],
    constant FrameUniforms&    frame         [[buffer(1)]],
    const device Instance*     instances     [[buffer(2)]],
    const device Instance*     prevInstances [[buffer(4)]],
    const device SQParam*      params        [[buffer(6)]]
) {
    Instance inst = instances[in.iid];
    SQParam  p    = params[in.iid];

    // World-space view ray through this fragment, taken into object space. od is
    // intentionally NOT renormalized after the inverse-model transform — the
    // hit point is reconstructed in object space as oo + od*t, so units cancel.
    float3 ro  = frame.cameraWorldPos;
    float3 rdW = normalize(in.worldPos - ro);
    float3 oo  = (p.invModel * float4(ro,  1.0)).xyz;
    float3 od  = (p.invModel * float4(rdW, 0.0)).xyz;

    float  tHit;
    float3 oN;
    if (!sqIntersect(oo, od, p.shape, tHit, oN)) discard_fragment();

    float3 oHit  = oo + od * tHit;
    float3 wHit  = (inst.modelMatrix * float4(oHit, 1.0)).xyz;
    float3 wN    = normalize((inst.normalMatrix * float4(oN, 0.0)).xyz);

    // Analytic depth — exact ray-hit, NOT the box. Every downstream pass
    // (lighting, SSR, SSAO, DOF, volumetric, TAA) reads this.
    float4 curClip = frame.viewProjection * float4(wHit, 1.0);

    // Motion vector from the analytic hit under the current vs previous instance
    // transform (rigid-motion assumption: the surface point sticks to the object).
    // The proxy-box vertices' motion is meaningless for the impostor, so velocity
    // MUST come from here or TAA smears every moving instance.
    Instance prevInst = prevInstances[in.iid];
    float3 prevW    = (prevInst.modelMatrix * float4(oHit, 1.0)).xyz;
    float4 prevClip = frame.previousViewProjection * float4(prevW, 1.0);
    float2 curNDC  = curClip.xy  / curClip.w;
    float2 prevNDC = prevClip.xy / prevClip.w;
    float2 velUV   = (curNDC - prevNDC) * float2(0.5, -0.5);

    SQImpostorFSOut o;
    o.albedoMetallic  = half4(half3(inst.albedo), half(inst.metallic));
    float2 oct = octEncode(wN);
    // .w tag: 1.0 = opaque (default); 1.0 + anisotropy (>1) carries the grain-highlight stretch
    // for the deferred pass (all existing tag tests are < 1.0, so this reads as opaque to them).
    o.normalRoughness = half4(half(oct.x), half(oct.y), half(inst.roughness), 1.0h + half(inst.anisotropy));
    o.emission        = half4(half3(inst.emission), half(inst.clearcoat > 0.0 ? inst.clearcoat : -inst.sheen));
    o.velocity        = half2(velUV);
    o.depth           = curClip.z / curClip.w;   // Metal NDC z ∈ [0,1]
    return o;
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

static inline float distributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (M_PI_F * d * d + 1e-7);
}

static inline float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

static inline float geometrySmith(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

static inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

static inline float3 worldPosFromDepth(float2 ndcXY, float depth, float4x4 invViewProj) {
    float4 clip = float4(ndcXY, depth, 1.0);
    float4 world = invViewProj * clip;
    return world.xyz / world.w;
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

static inline float2 dirToEquirectUV(float3 dir) {
    // dir must be normalised. u in [0,1) east-around, v in [0,1] from north
    // pole (v=0) to south pole (v=1). +X column is u=0.
    float u = atan2(dir.z, dir.x) * kInvTwoPi;        // (-0.5, 0.5]
    if (u < 0.0) u += 1.0;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) * kInvPi;
    return float2(u, v);
}

// Standard cubemap face → direction. Matches Metal's samplerCube convention
// (face 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z) — written so the direction
// returned here, fed to samplerCube::sample(d), would yield this same texel.
static inline float3 cubeDirForFace(uint face, float2 uv) {
    float s = uv.x * 2.0 - 1.0;
    float t = uv.y * 2.0 - 1.0;
    float3 d;
    switch (face) {
        case 0: d = float3( 1.0, -t, -s); break;   // +X
        case 1: d = float3(-1.0, -t,  s); break;   // -X
        case 2: d = float3( s,  1.0,  t); break;   // +Y
        case 3: d = float3( s, -1.0, -t); break;   // -Y
        case 4: d = float3( s, -t,  1.0); break;   // +Z
        default: d = float3(-s, -t, -1.0); break;  // -Z
    }
    return normalize(d);
}

// Hammersley low-discrepancy 2D sequence in [0,1). Used for GGX importance
// sampling of the prefilter cubemap.
static inline float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
static inline float2 hammersley(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverseVdC(i));
}

// Importance-sample GGX in tangent space, then rotate into the N-aligned
// world-space frame. `roughness` is the surface roughness (linear-space α,
// not perceptual). Returns the half-vector H.
static inline float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * M_PI_F * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float3 H_tan = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    // Build a tangent frame around N. The branch avoids degeneracy when N is
    // close to ±Y (where the standard up = (0,1,0) cross would collapse).
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    return normalize(T * H_tan.x + B * H_tan.y + N * H_tan.z);
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

// ── IBL bake — diffuse irradiance ────────────────────────────────────────────
//
// For each output texel (face, x, y) we compute the cosine-weighted hemisphere
// integral of the sky in the corresponding world direction. Output stores
// I(N)/π so the lighting pass can just multiply by albedo * kD — the
// LearnOpenGL / Karis convention. 16×8 = 128 directional samples per texel at
// 16² output → ~200K equirect lookups per face, ~1.2M total. Cheap on Apple
// Silicon (well under a millisecond).

kernel void illumi_irradiance_bake(
    texture2d<float, access::sample>  sky        [[texture(0)]],
    texturecube<half,  access::write> outCube    [[texture(1)]],
    constant float&                   bakeDesat  [[buffer(0)]],
    uint3                              gid       [[thread_position_in_grid]]
) {
    uint W = outCube.get_width();
    if (gid.x >= W || gid.y >= W || gid.z >= 6u) return;

    float2 uv = (float2(gid.xy) + 0.5) / float(W);
    float3 N = cubeDirForFace(gid.z, uv);

    // Tangent frame around N.
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    const uint N_PHI = 16;
    const uint N_THETA = 8;
    float3 acc = float3(0);
    for (uint p = 0; p < N_PHI; ++p) {
        float phi = (float(p) + 0.5) * (2.0 * M_PI_F / float(N_PHI));
        float cphi = cos(phi), sphi = sin(phi);
        for (uint t = 0; t < N_THETA; ++t) {
            float theta = (float(t) + 0.5) * (0.5 * M_PI_F / float(N_THETA));
            float sT = sin(theta), cT = cos(theta);
            // Tangent-space sample dir → world dir aligned with N.
            float3 dir = T * (sT * cphi) + B * (sT * sphi) + N * cT;
            float3 L = sampleSkyEquirect(sky, dir);
            // Optional desaturation: pull near-monochromatic skies (broiler,
            // vivid sunset) toward luminance so diffuse IBL doesn't flood all
            // surfaces with one hue. Visual sky is baked into the cube at its
            // full colour; only the irradiance integral is pulled neutral here.
            if (bakeDesat > 0.0) {
                float lum = dot(L, float3(0.2126, 0.7152, 0.0722));
                L = mix(L, float3(lum), bakeDesat);
            }
            acc += L * cT * sT;        // L * cos(θ) * sin(θ) (Jacobian)
        }
    }
    // Riemann-sum factor = (π/2 / N_THETA) * (2π / N_PHI), then divide by π
    // to bake in the Lambertian 1/π. Net factor: π / (N_PHI * N_THETA).
    float3 irradiance = acc * (M_PI_F / float(N_PHI * N_THETA));
    outCube.write(half4(half3(irradiance), 1.0h), gid.xy, gid.z);
}

// ── IBL bake — GGX-prefiltered specular ──────────────────────────────────────
//
// One dispatch per mip level. The caller binds a texture VIEW of the single
// mip and passes its roughness via the `Prefilter` struct. mip 0 = roughness
// 0 (mirror), mip mipCount-1 = roughness 1 (matte). Karis split-sum
// approximation (V = N = R) so the sampled cubemap can be re-used across
// view directions in the lighting pass.

struct PrefilterBakeParams {
    float roughness;
    uint  faceWidth;
    uint  sampleCount;
    float bakeDesat;   // was _pad — desaturate equirect samples (0=off, 1=grey)
};

kernel void illumi_prefilter_bake(
    texture2d<float, access::sample>   sky      [[texture(0)]],
    texturecube<half,  access::write>  outMip   [[texture(1)]],
    constant PrefilterBakeParams&      params   [[buffer(0)]],
    uint3                              gid      [[thread_position_in_grid]]
) {
    uint W = params.faceWidth;
    if (gid.x >= W || gid.y >= W || gid.z >= 6u) return;

    float2 uv = (float2(gid.xy) + 0.5) / float(W);
    float3 N = cubeDirForFace(gid.z, uv);
    float3 V = N;
    float roughness = params.roughness;
    float bakeDesat = params.bakeDesat;

    // Mirror surface shortcut — no GGX integration needed.
    if (roughness <= 0.0001) {
        float3 sky_val = sampleSkyEquirect(sky, N);
        if (bakeDesat > 0.0) {
            float lum = dot(sky_val, float3(0.2126, 0.7152, 0.0722));
            sky_val = mix(sky_val, float3(lum), bakeDesat);
        }
        outMip.write(half4(half3(sky_val), 1.0h), gid.xy, gid.z);
        return;
    }

    uint samples = max(params.sampleCount, 4u);
    float3 prefiltered = float3(0);
    float totalWeight = 0.0;
    for (uint i = 0; i < samples; ++i) {
        float2 Xi = hammersley(i, samples);
        float3 H = importanceSampleGGX(Xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = saturate(dot(N, L));
        if (NdotL > 0.0) {
            float3 sky_val = sampleSkyEquirect(sky, L);
            if (bakeDesat > 0.0) {
                float lum = dot(sky_val, float3(0.2126, 0.7152, 0.0722));
                sky_val = mix(sky_val, float3(lum), bakeDesat);
            }
            prefiltered += sky_val * NdotL;
            totalWeight += NdotL;
        }
    }
    prefiltered /= max(totalWeight, 1e-4);
    outMip.write(half4(half3(prefiltered), 1.0h), gid.xy, gid.z);
}

// ── Split-sum DFG LUT bake (Phase 3.2) ──────────────────────────────────────
//
// Pre-integrates the specular BRDF into a 2D LUT keyed on (NdotV, roughness).
// For each texel we integrate over a hemisphere of GGX-importance-sampled half-
// vectors with a fixed N = (0,0,1) and a V reconstructed from NdotV. Output is
// (scale, bias) from the split-sum: F0 * scale + bias = the environment BRDF
// integral. Baked once on renderer init (view-independent), never per-frame.
//
// Reference: Karis 2013 "Real Shading in Unreal Engine 4" — equations 4–5.

kernel void illumi_dfg_bake(
    texture2d<half, access::write> outLUT [[texture(0)]],
    uint2                          gid   [[thread_position_in_grid]]
) {
    uint W = outLUT.get_width();
    uint H = outLUT.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // Texel centres map to (0,1] so the LUT never samples at the degenerate
    // NdotV=0 or roughness=0 poles.
    float NdotV    = (float(gid.x) + 0.5) / float(W);
    float roughness = (float(gid.y) + 0.5) / float(H);

    // Reconstruct a view vector against N = (0,0,1) with the given NdotV.
    float sinTheta = sqrt(max(0.0, 1.0 - NdotV * NdotV));
    float3 V = float3(sinTheta, 0.0, NdotV);
    float3 N = float3(0.0, 0.0, 1.0);

    const uint SAMPLES = 512;
    float scale = 0.0;
    float bias  = 0.0;

    for (uint i = 0; i < SAMPLES; ++i) {
        float2 Xi = hammersley(i, SAMPLES);
        float3 H  = importanceSampleGGX(Xi, N, roughness);
        float3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
            float G     = geometrySmith(NdotV, NdotL, roughness);
            // G_Vis: the split-sum measure-change factor.
            float G_Vis = (G * VdotH) / max(NdotH * NdotV, 1e-6);
            float Fc    = pow(1.0 - VdotH, 5.0);
            scale += (1.0 - Fc) * G_Vis;
            bias  += Fc * G_Vis;
        }
    }

    outLUT.write(
        half4(half(scale / float(SAMPLES)), half(bias / float(SAMPLES)), 0.0h, 1.0h),
        gid
    );
}

// ── Cascaded shadow sampling (Phase 2.5) ─────────────────────────────────────
//
// Per pixel: pick a cascade from the view-space depth, project worldPos into
// that cascade's light space, then PCF-compare against the depth slice. The
// returned float is "sun visibility" in [0, 1] — 1 = lit, 0 = fully shadowed.

static inline uint pickCascade(float viewZ, float3 splits) {
    // viewZ here is a positive distance from the camera (i.e. -view.z), so
    // cascades fall into [0, splits.x), [splits.x, splits.y), [splits.y, splits.z).
    if (viewZ < splits.x) return 0u;
    if (viewZ < splits.y) return 1u;
    return 2u;
}

static inline float sampleCascade(
    depth2d_array<float, access::sample> shadowMap,
    sampler                              shadowSampler,
    float3 worldPos,
    float4x4 cascadeVP,
    uint cascade,
    float bias,
    uint pcfRadius
) {
    float4 lp = cascadeVP * float4(worldPos, 1.0);
    float3 ndc = lp.xyz / lp.w;
    // Outside the cascade's frustum → fall back to "lit" so we don't shadow
    // pixels that simply aren't covered by this cascade.
    if (any(abs(ndc.xy) > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    float refZ = ndc.z - bias;

    if (pcfRadius == 0u) {
        return shadowMap.sample_compare(shadowSampler, uv, cascade, refZ);
    }
    // 3×3 PCF (radius 1) or 5×5 (radius 2). The sample_compare hardware filter
    // gives us bilinear-filtered comparisons per tap, so a small grid here is
    // usually enough for nice soft contact shadows.
    int r = int(min(pcfRadius, 2u));
    float2 texel = 1.0 / float2(shadowMap.get_width(), shadowMap.get_height());
    float sum = 0.0;
    int count = 0;
    for (int j = -r; j <= r; ++j) {
        for (int i = -r; i <= r; ++i) {
            float2 off = float2(float(i), float(j)) * texel;
            sum += shadowMap.sample_compare(shadowSampler, uv + off, cascade, refZ);
            count += 1;
        }
    }
    return sum / float(count);
}

// ── Phase 4.42 — function-constant specialization of illumi_lighting (WWDC23 #10127) ──
//
// These gate the lighting kernel's STABLE per-config feature branches at COMPILE
// time. The Metal compiler constant-folds the branch and dead-code-eliminates the
// untaken path (and drops its register pressure) — the "Optimize GPU renderers with
// Metal" headline technique. The host (IlluminatoramaRenderer) sets them via
// MTLFunctionConstantValues and the shared cache memoises ONE pipeline variant per
// combination, so the lab's runtime A/B toggles still work (a toggle compiles its
// variant once, then hits the cache). Only flags that are constant across frames for
// a given config live here — per-frame-flipping flags (taa/ssao/ssr IsFirstFrame) and
// float intensities stay runtime in FrameUniforms.
//
// INVARIANT: each constant MUST equal the matching `frame.*Enabled` uniform the host
// still writes — both are derived from the same Swift Bool in `updateFrameUniforms`
// and `lightingFeatureConstants()`. The uniforms are kept (struct layout is shared
// Swift↔Metal) but no longer read by this kernel.
constant bool kLightingIBLEnabled          [[function_constant(0)]];
constant bool kLightingShadowEnabled       [[function_constant(1)]];
constant bool kLightingDFGLUTEnabled       [[function_constant(2)]];
constant bool kLightingDDGIEnabled         [[function_constant(3)]];
constant bool kLightingDDGIIrrCacheEnabled [[function_constant(4)]];

static inline float sunVisibility(
    depth2d_array<float, access::sample> shadowMap,
    sampler                              shadowSampler,
    constant FrameUniforms&              frame,
    float3 worldPos,
    float3 N,
    float  NdotL
) {
    if (!kLightingShadowEnabled) return 1.0;  // function_constant(1) — see decls above
    // Pick the cascade from view-space depth. We use -view.z (positive
    // distance) so the splits in frame.cascadeSplitsView are also positive.
    float4 vp = frame.view * float4(worldPos, 1.0);
    float viewDist = -vp.z;
    uint cascade = pickCascade(viewDist, frame.cascadeSplitsView.xyz);

    // Slope-scaled depth bias — surfaces nearly parallel to the light direction
    // need more bias to avoid acne, but biasing too much causes peter-panning.
    float slope = clamp(1.0 - NdotL, 0.0, 1.0);
    float bias = frame.shadowBias + frame.shadowSlopeBias * slope;
    // Outer cascades cover larger world extents, so a fixed bias becomes too
    // tight in light-NDC space — scale by cascade index.
    bias *= (1.0 + float(cascade) * 0.5);

    float4x4 cascadeVP;
    if (cascade == 0u)      cascadeVP = frame.shadowVP0;
    else if (cascade == 1u) cascadeVP = frame.shadowVP1;
    else                    cascadeVP = frame.shadowVP2;

    // Normal-offset bias. Depth bias along the light ray peter-pans grazing
    // surfaces (it scales with 1-NdotL, so walls detach from their contact
    // line — the classic offset artifact). Instead push the *receiver sample*
    // out along its own surface normal by ~1 shadow texel of WORLD size, so a
    // flat top no longer samples its own occluder (kills self-shadow acne)
    // while the in-plane shift barely moves the shadow edge. Texel world size
    // is derived from the cascade's ortho extent: the first matrix row's linear
    // length is 1/radius, and the map spans 2·radius across `width` texels.
    float3 row0 = float3(cascadeVP[0][0], cascadeVP[1][0], cascadeVP[2][0]);
    float  radius = 1.0 / max(length(row0), 1e-6);
    float  texelWorld = 2.0 * radius / float(shadowMap.get_width());
    // Widen mildly on grazing incidence (where acne is worst) but stay bounded
    // — 1–3 texels, never the runaway depth bias produced.
    float3 biasedPos = worldPos + N * texelWorld * (1.0 + 2.0 * slope);

    return sampleCascade(shadowMap, shadowSampler, biasedPos,
                         cascadeVP, cascade, bias, frame.shadowPcfRadius);
}

static inline float3 brdf(
    float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness, float3 lightColor,
    float anisotropy = 0.0, float3 grainT = float3(0.0)
) {
    float3 H = normalize(V + L);
    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0) return float3(0);
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float HdotV = saturate(dot(H, V));

    float3 F0 = mix(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlick(HdotV, F0);
    float  G  = geometrySmith(NdotV, NdotL, roughness);
    float  D;
    if (anisotropy > 0.001 && dot(grainT, grainT) > 0.25) {
        // Anisotropic GGX (Burley): stretch the lobe along the in-plane grain tangent, compress
        // across it — a wood floor / brushed steel reads as a streaked highlight, not a round one.
        float3 B  = normalize(cross(N, grainT));
        float3 T  = normalize(cross(B, N));                 // re-orthogonalize into the plane
        float  a  = max(roughness * roughness, 1e-3);
        float  at = max(a * (1.0 + anisotropy), 1e-3);      // along grain (stretched)
        float  ab = max(a * (1.0 - 0.7 * anisotropy), 1e-3); // across grain (tight)
        float  ToH = dot(T, H), BoH = dot(B, H);
        float  d  = ToH * ToH / (at * at) + BoH * BoH / (ab * ab) + NdotH * NdotH;
        D = 1.0 / (M_PI_F * at * ab * d * d);
    } else {
        D = distributionGGX(NdotH, roughness);
    }

    float3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-7);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 diff = kd * albedo / M_PI_F;
    return (diff + spec) * lightColor * NdotL;
}

// ── Rectangular area light (#60 task 5) ─────────────────────────────────────
// Replaces the 4.24 five-spot `.area` approximation. The DIFFUSE term is the
// EXACT closed-form polygon clamped-cosine integral — Linearly-Transformed-
// Cosines with M = identity (Heitz et al. 2016), i.e. a verifiable Lambert
// form-factor, not an approximation. The SPECULAR term is a most-representative-
// point sample (the rect point closest to the reflection ray) — a declared
// approximation pending the fitted GGX LTC specular LUT (increment 2).

// Vector irradiance of one polygon edge (clamped-cosine), v1/v2 normalised.
static inline float3 ltcIntegrateEdge(float3 v1, float3 v2) {
    float  cosT  = clamp(dot(v1, v2), -1.0, 1.0);
    float  theta = acos(cosT);
    float3 cr    = cross(v1, v2);
    float  s     = (theta > 1e-5) ? theta / sin(theta) : 1.0;   // sinc limit at θ→0
    return cr * s;
}

// Clamped-cosine form factor of the quad (corners p0..p3 CCW, relative to the
// shaded point) seen from a surface with normal N. Returns [0,1]; one-sided
// clamps the receiver to the front hemisphere, two-sided takes |·|.
static inline float ltcPolygonForm(float3 N, float3 p0, float3 p1, float3 p2, float3 p3,
                                   bool twoSided) {
    float3 L0 = normalize(p0), L1 = normalize(p1), L2 = normalize(p2), L3 = normalize(p3);
    float3 vsum = ltcIntegrateEdge(L0, L1) + ltcIntegrateEdge(L1, L2)
                + ltcIntegrateEdge(L2, L3) + ltcIntegrateEdge(L3, L0);
    float z = dot(vsum, N) * (1.0 / (2.0 * M_PI_F));
    return twoSided ? abs(z) : max(0.0, z);
}

// Full rectangular area-light contribution at a shaded point.
static inline float3 evalAreaLight(AreaLight al, float3 worldPos, float3 N, float3 V,
                                   float3 albedo, float metallic, float roughness,
                                   texture2d<float> ltcMat, texture2d<float> ltcMag,
                                   bool ltcEnabled) {
    float3 nL = cross(al.ex, al.ey);
    float  nLlen = length(nL);
    if (nLlen < 1e-8) return float3(0.0);
    nL /= nLlen;
    bool twoSided = al.twoSided > 0.5;

    // One-sided: the receiver must be on the emitting (+nL) face.
    float facing = dot(nL, worldPos - al.center);
    if (!twoSided && facing <= 0.0) return float3(0.0);

    // Distance falloff (centre distance) with a smooth radius window — keeps the
    // light local like the point/spot path rather than lighting the whole scene.
    float dist = length(al.center - worldPos);
    if (dist > al.radius) return float3(0.0);
    float window = saturate(1.0 - pow(dist / al.radius, 4.0));
    window *= window;
    if (window <= 0.0) return float3(0.0);

    // Quad corners relative to the shaded point (CCW seen from +nL).
    float3 p0 = al.center - al.ex - al.ey - worldPos;
    float3 p1 = al.center + al.ex - al.ey - worldPos;
    float3 p2 = al.center + al.ex + al.ey - worldPos;
    float3 p3 = al.center - al.ex + al.ey - worldPos;

    // Diffuse — exact polygon clamped-cosine (LTC, M = identity).
    float  ff = ltcPolygonForm(N, p0, p1, p2, p3, twoSided);
    float3 diffuse = (1.0 - metallic) * albedo * ff;

    float3 spec = float3(0.0);
    float3 F0 = mix(float3(0.04), albedo, metallic);
    if (ltcEnabled) {
        // LTC specular — transform the light polygon by the per-(roughness, NdotV)
        // inverse matrix, then run the SAME clamped-cosine polygon integral. The
        // soft area light becomes a physically-shaped glossy reflection.
        constexpr sampler lutSamp(filter::linear, address::clamp_to_edge);
        float NoV = clamp(dot(N, V), 0.0, 1.0);
        float2 uv = float2(NoV, roughness);                  // (NdotV, roughness)
        float4 t1 = ltcMat.sample(lutSamp, uv);              // Minv (a,b,c,d)
        float2 t2 = ltcMag.sample(lutSamp, uv).xy;           // (scale, bias)
        // Tangent frame: T1 in the view-incidence plane (matches the fit's frame).
        float3 T1 = normalize(V - N * dot(V, N));
        float3 T2 = cross(N, T1);
        float3x3 worldToTan = transpose(float3x3(T1, T2, N)); // rows = T1,T2,N
        float3x3 Minv = float3x3(float3(t1.x, 0.0, t1.y),
                                 float3(0.0,  1.0, 0.0),
                                 float3(t1.z, 0.0, t1.w));
        float3 q0 = normalize(Minv * (worldToTan * p0));
        float3 q1 = normalize(Minv * (worldToTan * p1));
        float3 q2 = normalize(Minv * (worldToTan * p2));
        float3 q3 = normalize(Minv * (worldToTan * p3));
        float3 vsum = ltcIntegrateEdge(q0, q1) + ltcIntegrateEdge(q1, q2)
                    + ltcIntegrateEdge(q2, q3) + ltcIntegrateEdge(q3, q0);
        float ltcSpec = max(0.0, vsum.z / (2.0 * M_PI_F));
        spec = ltcSpec * (F0 * t2.x + (1.0 - F0) * t2.y);
    } else {
        // Fallback — most-representative-point: the rect point closest to the
        // reflection ray, as a punctual GGX sample weighted by the form factor.
        float3 R = reflect(-V, N);
        float  denom = dot(R, nL);
        float  t = (abs(denom) > 1e-4) ? dot(al.center - worldPos, nL) / denom : 0.0;
        float3 hit = worldPos + R * max(t, 0.0);
        float3 d   = hit - al.center;
        float  ux  = length(al.ex), uy = length(al.ey);
        float3 exN = al.ex / max(ux, 1e-6), eyN = al.ey / max(uy, 1e-6);
        float3 rep = al.center + exN * clamp(dot(d, exN), -ux, ux)
                               + eyN * clamp(dot(d, eyN), -uy, uy);
        float3 Ls  = normalize(rep - worldPos);
        float  NdotLs = saturate(dot(N, Ls));
        if (NdotLs > 0.0) {
            float3 H     = normalize(V + Ls);
            float  NdotV = saturate(dot(N, V));
            float  NdotH = saturate(dot(N, H));
            float  HdotV = saturate(dot(H, V));
            float3 F     = fresnelSchlick(HdotV, F0);
            float  D     = distributionGGX(NdotH, roughness);
            float  G     = geometrySmith(NdotV, NdotLs, roughness);
            spec = (D * G * F) / (4.0 * NdotV * NdotLs + 1e-7) * NdotLs * ff;
        }
    }

    return (diffuse + spec) * al.color * window;
}

// ── Phase 3.1: DDGI — Dynamic Diffuse Global Illumination ───────────────────
//
// Analytic one-bounce probe GI. No hardware RT — rays are intersected against
// the scene's unit primitives (box, sphere, ground) in local space via the
// per-instance invModelMatrix stored in DDGIInstanceData[].
//
// Three kernels per frame (see IlluminatoramaRenderer.swift `encodeDDGIFrame`):
//   illumi_ddgi_trace             — fire rays from each probe
//   illumi_ddgi_update_irradiance — integrate → irradiance atlas (rgba16Half)
//   illumi_ddgi_update_depth      — integrate → depth atlas (rg16Half, mean+mean²)
//
// The lighting kernel (below) samples both atlases via trilinear probe blend +
// Chebyshev visibility, replacing the irradianceCube diffuse term when enabled.
//
// Struct layout rules: same SIMD3<Float>+Float=16B grouping as FrameUniforms.
// Mirror structs live in IlluminatoramaRenderer.swift (IlluminatoramaDDGIUniforms,
// DDGIGPUInstanceData). When changing one side, change both.

struct DDGIUniforms {
    float3   gridOrigin;              // world-space position of probe (0,0,0)
    uint     gridDimsX;               // probes along X
    float3   directionalLightDir;     // toward light, world space
    uint     gridDimsY;               // probes along Y
    float3   directionalLightColor;   // linear HDR
    uint     gridDimsZ;               // probes along Z
    float    probeSpacing;            // metres between adjacent probes
    uint     raysPerProbe;            // rays dispatched per probe per frame
    float    hysteresis;              // EMA weight kept from previous irr frame
    float    depthHysteresis;         // same for depth atlas
    float    irradianceScale;         // post-multiplier on final GI contribution
    uint     enabled;                 // 0 = skip probe lookup in lighting pass
    uint     irrTileSize;             // interior octahedral tile width (e.g. 6)
    uint     depthTileSize;           // interior tile width for depth atlas (e.g. 14)
    uint     instanceCount;           // entries in DDGIInstanceData[]
    // Phase 3.3 — two-bounce GI. 0 = single-bounce direct only (Phase 3.1
    // behaviour); 1 = trace kernel ALSO samples the previous-frame
    // irradiance atlas at the hit point and adds it to the recorded
    // radiance. Requires ping-pong of both atlases on the host side.
    uint     twoBounceEnabled;
    uint     emitterCount;            // entries in DDGIPointEmitter[] bound at buffer(3)
    uint     _pad2;
    // Total: 96 bytes, stride 96.
};

struct DDGIInstanceData {
    float4x4 invModelMatrix;   // world→local (transforms the ray)
    float4x4 normalMatrix;     // transpose(inverse(upper3x3(model))), padded
    float3   albedo;
    float    metallic;
    float3   emission;         // pre-multiplied emissive radiance
    float    roughness;
    uint     meshKind;         // 0=box, 1=sphere, 2=ground
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
    // Total: 176 bytes, stride 176.
};

struct DDGIRayRecord {
    float4 dirAndDist;  // xyz=ray dir (world), w=hit dist (-1=miss/sky)
    float4 radiance;    // xyz=HDR radiance at hit, w unused
    // Total: 32 bytes.
};

// An analytic point emitter contributed by a particle field. The trace kernel
// evaluates these per-ray so probes accumulate particle emission and propagate
// it to all scene surfaces as proper indirect light.
// Swift mirror: DDGIGPUEmitter in IlluminatoramaRenderer.swift.
struct DDGIPointEmitter {
    float3 position;  // world-space centroid
    float  radius;    // falloff radius in metres; contribution = 0 outside
    float3 color;     // pre-multiplied HDR irradiance
    float  _pad;
    // Total: 32 bytes.
};

// ── DDGI Helpers ─────────────────────────────────────────────────────────────

// Spherical Fibonacci sampling — deterministic, well-distributed sphere dirs.
static float3 ddgiSphericalFibonacci(uint i, uint n) {
    const float PHI = 1.6180339887498948482f;
    float phi      = 2.0f * M_PI_F * fract(float(i) * (PHI - 1.0f));
    float cosTheta = 1.0f - (2.0f * float(i) + 1.0f) / float(n);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Octahedral encode: unit direction → [-1,1]² (full-sphere projection).
static float2 ddgiOctEncode(float3 v) {
    float3 p = v / (abs(v.x) + abs(v.y) + abs(v.z));
    if (p.z < 0.0f) {
        float2 xy = (1.0f - abs(p.yx)) * float2(p.x >= 0.0f ? 1.0f : -1.0f,
                                                  p.y >= 0.0f ? 1.0f : -1.0f);
        return xy;
    }
    return p.xy;
}

// Octahedral decode: [-1,1]² → unit direction.
static float3 ddgiOctDecode(float2 uv) {
    float3 v = float3(uv, 1.0f - abs(uv.x) - abs(uv.y));
    if (v.z < 0.0f) {
        float2 xy = (1.0f - abs(v.yx)) * float2(v.x >= 0.0f ? 1.0f : -1.0f,
                                                   v.y >= 0.0f ? 1.0f : -1.0f);
        v.xy = xy;
    }
    return normalize(v);
}

// Probe world-space position from flat index.
static float3 ddgiProbePos(uint probeIdx, constant DDGIUniforms& ddgi) {
    uint px = probeIdx % ddgi.gridDimsX;
    uint py = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    return ddgi.gridOrigin + float3(px, py, pz) * ddgi.probeSpacing;
}

// Atlas tile layout: tiles are (probeX + probeZ*gridDimsX, probeY), 1px border.
static uint2 ddgiIrrTexel(uint tx, uint ty, uint probeIdx,
                           constant DDGIUniforms& ddgi) {
    uint pad  = ddgi.irrTileSize + 2;
    uint px   = probeIdx % ddgi.gridDimsX;
    uint py   = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz   = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    return uint2((px + pz * ddgi.gridDimsX) * pad + 1 + tx,
                  py * pad + 1 + ty);
}

static uint2 ddgiDepthTexel(uint tx, uint ty, uint probeIdx,
                              constant DDGIUniforms& ddgi) {
    uint pad = ddgi.depthTileSize + 2;
    uint px  = probeIdx % ddgi.gridDimsX;
    uint py  = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz  = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    return uint2((px + pz * ddgi.gridDimsX) * pad + 1 + tx,
                  py * pad + 1 + ty);
}

// Normalised UV into the irradiance atlas for a given probe + oct-encoded dir.
static float2 ddgiIrrAtlasUV(float2 octNorm, uint probeIdx,
                               constant DDGIUniforms& ddgi) {
    uint pad   = ddgi.irrTileSize + 2;
    uint atlasW = pad * ddgi.gridDimsX * ddgi.gridDimsZ;
    uint atlasH = pad * ddgi.gridDimsY;
    uint px    = probeIdx % ddgi.gridDimsX;
    uint py    = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz    = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    float2 tileOrigin = float2((px + pz * ddgi.gridDimsX) * pad + 1,
                                py * pad + 1);
    float2 interior   = (octNorm + 1.0f) * 0.5f;  // [-1,1]² → [0,1]²
    return (tileOrigin + interior * float(ddgi.irrTileSize)) / float2(atlasW, atlasH);
}

static float2 ddgiDepthAtlasUV(float2 octNorm, uint probeIdx,
                                 constant DDGIUniforms& ddgi) {
    uint pad    = ddgi.depthTileSize + 2;
    uint atlasW = pad * ddgi.gridDimsX * ddgi.gridDimsZ;
    uint atlasH = pad * ddgi.gridDimsY;
    uint px     = probeIdx % ddgi.gridDimsX;
    uint py     = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz     = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    float2 tileOrigin = float2((px + pz * ddgi.gridDimsX) * pad + 1,
                                py * pad + 1);
    float2 interior   = (octNorm + 1.0f) * 0.5f;
    return (tileOrigin + interior * float(ddgi.depthTileSize)) / float2(atlasW, atlasH);
}

// ── Analytic ray–primitive intersections ──────────────────────────────────────
//
// Rays are supplied in the primitive's LOCAL space (transformed via invModelMatrix).
// The returned t is the world-space hit distance because the local direction
// vector |ld| = |invModel * rayDir| and the model's scaling cancels out when
// computing the world-space displacement (model * ld * t = rayDir * t).

// Ray vs unit box ([-0.5, 0.5]³). Returns t > 0 on hit; writes outward normal.
static float rayUnitBox(float3 o, float3 d, thread float3& outNormal) {
    float3 tMin = (-0.5f - o) / d;
    float3 tMax = ( 0.5f - o) / d;
    float3 t1   = min(tMin, tMax);
    float3 t2   = max(tMin, tMax);
    float  tNear = max(max(t1.x, t1.y), t1.z);
    float  tFar  = min(min(t2.x, t2.y), t2.z);
    if (tNear > tFar || tFar < 1e-4f) return -1.0f;
    float  t   = (tNear > 1e-4f) ? tNear : tFar;
    float3 hit = o + d * t;
    float3 ab  = abs(hit) * 2.0f;
    if (ab.x >= ab.y && ab.x >= ab.z)
        outNormal = float3(sign(hit.x), 0.0f, 0.0f);
    else if (ab.y >= ab.x && ab.y >= ab.z)
        outNormal = float3(0.0f, sign(hit.y), 0.0f);
    else
        outNormal = float3(0.0f, 0.0f, sign(hit.z));
    return t;
}

// Ray vs unit sphere (radius 0.5, centred at origin).
static float rayUnitSphere(float3 o, float3 d, thread float3& outNormal) {
    float a    = dot(d, d);
    float b    = 2.0f * dot(o, d);
    float c    = dot(o, o) - 0.25f;   // r² = 0.5² = 0.25
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return -1.0f;
    float sq = sqrt(disc);
    float t0 = (-b - sq) / (2.0f * a);
    float t1 = (-b + sq) / (2.0f * a);
    float t  = (t0 > 1e-4f) ? t0 : t1;
    if (t < 1e-4f) return -1.0f;
    outNormal = normalize(o + d * t);
    return t;
}

// Ray vs unit ground plane (y = 0, local x,z ∈ [-0.5, 0.5]).
// The ground mesh uses scale(14,1,14), so local space is the unit quad.
static float rayUnitGround(float3 o, float3 d, thread float3& outNormal) {
    if (abs(d.y) < 1e-6f) return -1.0f;
    float t = -o.y / d.y;
    if (t < 1e-4f) return -1.0f;
    float3 hit = o + d * t;
    if (abs(hit.x) > 0.5f || abs(hit.z) > 0.5f) return -1.0f;
    outNormal = float3(0.0f, 1.0f, 0.0f);
    return t;
}

// ── DDGI probe irradiance lookup (called from illumi_lighting) ────────────────
//
// Trilinear blend over the 8 corner probes containing worldPos. Each probe
// contributes irradiance weighted by:
//   - trilinear distance weight
//   - back-face penalty (probes behind the surface normal get less weight)
//   - Chebyshev visibility (probes blocked by geometry get less weight)
//
// The surface normal N determines which octahedral texel of the irradiance
// atlas to sample (we want radiance arriving AT worldPos along the hemisphere
// of N).

static float3 sampleDDGIIrradiance(
    float3 worldPos,
    float3 N,
    texture2d<half,  access::sample> irrAtlas,
    texture2d<half,  access::sample> depthAtlas,
    constant DDGIUniforms&           ddgi
) {
    float3 local = (worldPos - ddgi.gridOrigin) / ddgi.probeSpacing;
    int3   base  = int3(floor(local));
    float3 alpha = local - float3(base);
    int3   dims  = int3(int(ddgi.gridDimsX), int(ddgi.gridDimsY), int(ddgi.gridDimsZ));

    constexpr sampler irrSmp(filter::linear, address::clamp_to_edge);
    constexpr sampler depSmp(filter::linear, address::clamp_to_edge);

    float3 irradiance  = float3(0.0f);
    float  totalWeight = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 probe  = base + offset;
        if (any(probe < int3(0)) || any(probe >= dims)) continue;

        uint probeIdx = uint(probe.x)
                      + uint(probe.y) * ddgi.gridDimsX
                      + uint(probe.z) * ddgi.gridDimsX * ddgi.gridDimsY;

        float3 probePos = ddgi.gridOrigin + float3(probe) * ddgi.probeSpacing;
        float3 toPoint  = worldPos - probePos;
        float  dist     = length(toPoint);
        float3 dir      = toPoint / max(dist, 1e-4f);

        // Trilinear weight.
        float bx = (offset.x == 0) ? (1.0f - alpha.x) : alpha.x;
        float by = (offset.y == 0) ? (1.0f - alpha.y) : alpha.y;
        float bz = (offset.z == 0) ? (1.0f - alpha.z) : alpha.z;
        float trilinear = bx * by * bz + 1e-5f;

        // Back-face penalty: probes behind the surface normal contribute less.
        float NdotD = dot(N, -dir);
        float backfaceW = ((NdotD + 1.0f) * 0.5f);
        backfaceW = backfaceW * backfaceW + 0.02f;

        // Chebyshev visibility: depth atlas stores (mean, mean²) of hit dists.
        float2 depUV  = ddgiDepthAtlasUV(ddgiOctEncode(dir), probeIdx, ddgi);
        float2 depSmp2 = float2(depthAtlas.sample(depSmp, depUV).rg);
        float  mean    = depSmp2.r;
        float  mean2   = depSmp2.g;
        float  variance = max(0.0f, mean2 - mean * mean);
        float  cheb = 1.0f;
        if (dist > mean) {
            float diff = dist - mean;
            cheb = variance / (variance + diff * diff);
            cheb = cheb * cheb * cheb;  // crush toward 0 for deep occlusion
        }

        float weight = max(0.0f, trilinear * backfaceW * cheb);

        // Sample irradiance at the surface normal direction.
        float2 irrUV   = ddgiIrrAtlasUV(ddgiOctEncode(N), probeIdx, ddgi);
        float3 probeIrr = float3(irrAtlas.sample(irrSmp, irrUV).rgb);

        irradiance  += probeIrr * weight;
        totalWeight += weight;
    }
    if (totalWeight > 1e-4f) irradiance /= totalWeight;
    return irradiance * ddgi.irradianceScale;
}

// Pull a fill colour toward its own luminance, ramped IN by how saturated it
// already is, scaled by `amount` (0 = no-op). Used to tame the monochromatic
// colour of indirect fill from a saturated environment (Phase 4.32).
static inline float3 desaturateFill(float3 c, float amount) {
    if (amount <= 0.0) return c;
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    float cmax = max(c.r, max(c.g, c.b));
    float sat = (cmax > 1e-4) ? saturate((cmax - lum) / cmax) : 0.0;
    float k = amount * smoothstep(0.35, 0.75, sat);
    return mix(c, float3(lum), k);
}

kernel void illumi_lighting(
    texture2d<half,  access::read>          gAlbedoMet      [[texture(0)]],
    texture2d<half,  access::read>          gNormalRgh      [[texture(1)]],
    texture2d<half,  access::read>          gEmission       [[texture(2)]],
    depth2d<float,   access::read>          gDepth          [[texture(3)]],
    texture2d<half,  access::write>         outHDR          [[texture(4)]],
    texture2d<half,  access::read>          aoTex           [[texture(5)]],
    texture2d<float, access::sample>        skyEquirect     [[texture(6)]],
    texturecube<half, access::sample>       irradianceCube  [[texture(7)]],
    texturecube<half, access::sample>       prefilteredCube [[texture(8)]],
    depth2d_array<float, access::sample>    shadowMap       [[texture(9)]],
    // Phase 3.2 — split-sum DFG LUT. RG channels: (scale, bias) such that
    // specular = prefilteredEnv * (F0 * scale + bias). Baked once on init;
    // `frame.dfgLUTEnabled == 0` falls back to Lagarde's roughness-Schlick.
    texture2d<half,  access::sample>        dfgLUT          [[texture(10)]],
    // Phase 3.1 — DDGI probe atlases. When ddgi.enabled == 0 these are
    // 1×1 dummies; the sampling is cheap and the enabled gate skips the loop.
    texture2d<half,  access::sample>        ddgiIrrAtlas    [[texture(11)]],
    texture2d<half,  access::sample>        ddgiDepthAtlas  [[texture(12)]],
    // Phase 4.10 — per-spot shadow depth atlas.
    depth2d_array<float, access::sample>    spotShadowAtlas [[texture(13)]],
    // Phase 3.4 — per-pixel DDGI irradiance EMA cache (ping-pong).
    texture2d<half,  access::sample>        irrCachePrev    [[texture(14)]],
    texture2d<half,  access::write>         irrCacheCur     [[texture(15)]],
    // #60 task 5 increment 2 — LTC area-light specular LUTs (Minv + magnitude).
    texture2d<float, access::sample>        ltcMat          [[texture(16)]],
    texture2d<float, access::sample>        ltcMag          [[texture(17)]],
    constant FrameUniforms&                 frame           [[buffer(0)]],
    const device PointLight*                pointLights     [[buffer(1)]],
    constant DDGIUniforms&                  ddgi            [[buffer(2)]],
    const device SpotLight*                 spotLights      [[buffer(3)]],
    const device AreaLight*                 areaLights      [[buffer(4)]],
    const device DirectionalLight*          extraDirectionals [[buffer(5)]],
    uint2                                   gid             [[thread_position_in_grid]]
) {
    uint w = outHDR.get_width();
    uint h = outHDR.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float depth = gDepth.read(gid);
    half4 emH = gEmission.read(gid);

    // Sky pixels (depth still cleared at 1.0). Reconstruct a world-space ray
    // through this pixel and sample the equirect HDR sky.
    float2 ndc = (float2(gid) + 0.5) / float2(w, h) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    if (depth >= 0.99999) {
        // Unproject a far-plane point and subtract the camera to get the
        // world-space view ray. (Cheaper than running invViewProj twice.)
        float4 farClip = float4(ndc, 1.0, 1.0);
        float4 farWorld = frame.invViewProjection * farClip;
        float3 dir = normalize(farWorld.xyz / farWorld.w - frame.cameraWorldPos);
        float3 sky = sampleSkyEquirect(skyEquirect, dir);
        outHDR.write(half4(half3(sky), 1.0h), gid);
        return;
    }

    half4 amH = gAlbedoMet.read(gid);
    half4 nrH = gNormalRgh.read(gid);

    float3 albedo    = float3(amH.rgb);
    float  metallic  = float(amH.a);
    float3 N         = octDecode(float2(nrH.rg));
    float  roughness = max(0.045, float(nrH.b));
    float3 emission  = float3(emH.rgb);

    float3 worldPos = worldPosFromDepth(ndc, depth, frame.invViewProjection);
    float3 V = normalize(frame.cameraWorldPos - worldPos);

    // Directional light, attenuated by the cascaded-shadow visibility term.
    // sample_compare returns 1.0 when `reference COMPARE_FUNC sample` holds.
    // We store front-most depths in the shadow map, so the surface is lit when
    // its light-space depth (minus bias) is ≤ the stored depth.
    constexpr sampler shadowSampler(filter::linear,
                                    compare_func::less_equal,
                                    address::clamp_to_edge);
    float3 Ld = normalize(frame.directionalLightDir);
    float NdotL_sun = saturate(dot(N, Ld));
    float visibility = sunVisibility(shadowMap, shadowSampler, frame,
                                     worldPos, N, NdotL_sun);
    // Terms kept separate so the per-term split-render (frame.debugTerm) can
    // isolate any one of them; the normal path just sums them at the end.
    // Phase 7c — grain anisotropy: normalRoughness.w carries (1 + aniso) for wood/brushed-metal
    // pixels (opaque is exactly 1.0). Reconstruct an in-plane grain tangent from a world reference
    // (plan-X for floors/ceilings, horizontal for walls) — approximate (per-instance, not per-
    // plank), but it's the highlight STRETCH that kills the plastic look, not the exact grain angle.
    float aniso = (nrH.a > 1.001h) ? float(nrH.a - 1.0h) : 0.0;
    float3 grainT = float3(0.0);
    if (aniso > 0.001) {
        float3 up = float3(0.0, 1.0, 0.0);
        grainT = (abs(dot(N, up)) > 0.95) ? normalize(float3(1.0, 0.0, 0.0) - N * N.x)
                                          : normalize(cross(N, up));
    }
    float3 directSun = brdf(N, V, Ld, albedo, metallic, roughness,
                            frame.directionalLightColor, aniso, grainT) * visibility;

    // ── Leaf thin-sheet transmission (issue #58 / #20 item 2) ───────────────
    // Leaves are flagged in normalRoughness.w (0 = foliage; opaque geometry is
    // 1). Opaque leaf cards in the G-buffer read "plastic" because they only
    // ever reflect; real foliage SCATTERS the sun through the thin blade, so a
    // back-lit leaf GLOWS. We add that transmitted term, driven by the actual
    // directional irradiance (`frame.directionalLightColor`) — ChatGPT's "tie
    // it to solar irradiance" — gated by the same cascade-shadow `visibility`
    // so a leaf buried in canopy shade doesn't light up. Zero for every other
    // surface (back == 0 when the sun is on the viewer's side of the leaf, and
    // the whole branch is skipped for non-foliage).
    float3 transmission = float3(0.0);
    if (nrH.a < 0.5h && frame.leafTransmission > 0.0) {   // foliage + scene opt-in
        float back    = saturate(dot(-N, Ld));           // sun on the FAR face
        float forward = pow(saturate(dot(V, -Ld)), 3.0); // viewing toward the sun
        // ROUND-6 POLISH #3: BACKLIT-ONLY GATE. The old `through = back·(0.40+0.60·
        // forward)` carried a 0.40 floor that fired whenever the sun touched the far
        // face — including FORWARD-LIT side-view leaves (sun on the camera's side,
        // V·(−Ld)≈0), which then bloomed pale khaki (the side-view bleach). Tie the
        // term to genuine backlight: a smoothstep on dot(V,−Ld) (== viewing toward
        // the sun) rolls the transmission to ~0 for forward-lit leaves while the
        // three-quarter rim (where V·(−Ld) is high) keeps the FULL term. Mirrored in
        // IlluminatoramaRT.metal so the deferred and RT sun paths agree.
        float backlit = smoothstep(0.02, 0.42, dot(V, -Ld));
        float through = back * backlit * (0.30 + 0.70 * forward);
        // ENERGY CLAMP: a leaf re-emits a FRACTION of the light reaching it, never
        // more. `leafTransmission` was an un-clamped (>1) gain, so under a low warm
        // BACKLIGHT the foliage blew past 1.0 and clipped to white while the bark
        // (no transmission) stayed dark — the connected canopy READ as bare sticks.
        // Clamp the transmitted radiance to the incident so backlit leaves glow but
        // never out-shine the sun. (Mirrored in IlluminatoramaRT.metal's sun pass.)
        float3 t = albedo * frame.directionalLightColor
                 * through * visibility * frame.leafTransmission;
        transmission = min(t, albedo * frame.directionalLightColor * visibility);
    }

    // ── Plush backlit thin-fabric SSS (Teddy Bear Press) ────────────────────
    // Plush pixels carry ≈0.55 in normalRoughness.w. A stuffed-fabric bear is a
    // thin shell over fibrefill, so a BACKLIT bear glows warmly through the fabric
    // — the same thin-sheet transmission as a leaf but softer/warmer and with less
    // forward bias (the fill diffuses it). Reuses the energy-clamped leaf machinery,
    // driven by `frame.plushTransmission` (0 ⇒ skipped). Mirrors the leaf gate so
    // it never fires forward-lit. Zero for every non-plush surface / scene.
    if (nrH.a > 0.5h && nrH.a < 0.6h && frame.plushTransmission > 0.0) {
        float back    = saturate(dot(-N, Ld));
        float backlit = smoothstep(0.02, 0.55, dot(V, -Ld));
        float through = back * backlit * (0.55 + 0.45 * pow(saturate(dot(V, -Ld)), 2.0));
        float3 warm   = frame.directionalLightColor * float3(1.06, 0.96, 0.82);  // warmer through fabric
        float3 t      = albedo * warm * through * visibility * frame.plushTransmission;
        transmission  = min(t, albedo * frame.directionalLightColor * visibility);
    }

    float3 pointSum = float3(0.0);
    float3 spotSum  = float3(0.0);

    // Point lights — inverse-square with smooth radius cutoff. (Includes the
    // synthesised emissive-as-light points from the extractor, Phase 4.27.)
    for (uint i = 0; i < frame.pointLightCount; ++i) {
        PointLight pl = pointLights[i];
        float3 toLight = pl.position - worldPos;
        float  dist    = length(toLight);
        if (dist > pl.radius) continue;
        float3 L = toLight / max(dist, 1e-4);
        float atten = 1.0 / max(dist * dist, 1e-4);
        float window = saturate(1.0 - pow(dist / pl.radius, 4.0));
        atten *= window * window;
        pointSum += brdf(N, V, L, albedo, metallic, roughness, pl.color * atten);
    }

    // Spot lights — same distance attenuation as point lights, multiplied
    // by a cone-attenuation term that smoothsteps between the outer and
    // inner cone cosines. Phase 4.10 adds an optional per-spot shadow
    // map: when `shadowSliceIndex >= 0`, the fragment's world position
    // is projected into the spot's light space and PCF-compared against
    // the stored depth. `visibility` modulates the contribution.
    constexpr sampler spotShadowSampler(filter::linear,
                                        compare_func::less_equal,
                                        address::clamp_to_edge);
    for (uint i = 0; i < frame.spotLightCount; ++i) {
        SpotLight sl = spotLights[i];
        float3 toLight = sl.position - worldPos;
        float  dist    = length(toLight);
        if (dist > sl.radius) continue;
        float3 L = toLight / max(dist, 1e-4);
        float coneCos   = dot(normalize(sl.direction), -L);
        float coneAtten = smoothstep(sl.outerCone, sl.innerCone, coneCos);
        if (coneAtten <= 0.0) continue;
        float atten = 1.0 / max(dist * dist, 1e-4);
        float window = saturate(1.0 - pow(dist / sl.radius, 4.0));
        atten *= window * window * coneAtten;

        // Shadow visibility. Skip cheaply when the host hasn't provided a
        // shadow slice for this spot — covers `spotShadowsEnabled = false`
        // (slice forced to -1 by `updateSpotShadows`) and any spot beyond
        // the atlas capacity.
        float visibility = 1.0;
        if (sl.shadowSliceIndex >= 0) {
            float4 lsPos = sl.shadowMatrix * float4(worldPos, 1.0);
            float2 lsNDC = lsPos.xy / lsPos.w;
            float  lsZ   = lsPos.z  / lsPos.w;
            // Flip Y to match texture UV convention.
            float2 shadowUV = float2(lsNDC.x * 0.5 + 0.5,
                                      -lsNDC.y * 0.5 + 0.5);
            // Out-of-frustum fragments stay fully lit — the cone attenuation
            // already drops their contribution to ~zero outside the
            // outer cone, so the shadow compare doesn't need to reject
            // them here.
            bool inFrustum = lsZ > 0.0 && lsZ < 1.0
                          && shadowUV.x >= 0.0 && shadowUV.x <= 1.0
                          && shadowUV.y >= 0.0 && shadowUV.y <= 1.0;
            if (inFrustum) {
                float ref = lsZ - frame.spotShadowBias;
                // 3×3 PCF using hardware bilinear compare. Each
                // `sample_compare` returns the linear-filtered fraction
                // of taps that pass; summing 9 gives a smooth penumbra.
                float w = 1.0 / 512.0; // shadow map texel size
                float sum = 0.0;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        sum += spotShadowAtlas.sample_compare(
                            spotShadowSampler,
                            shadowUV + float2(float(ox), float(oy)) * w,
                            uint(sl.shadowSliceIndex),
                            ref);
                    }
                }
                visibility = sum * (1.0 / 9.0);
            }
        }
        if (visibility <= 0.0) continue;
        spotSum += brdf(N, V, L, albedo, metallic, roughness,
                        sl.color * atten * visibility);
    }

    // Rectangular area lights (#60 task 5) — closed-form polygon diffuse + MRP
    // specular, replacing the old five-spot stand-in.
    float3 areaSum = float3(0.0);
    bool areaLTC = frame.areaLTCEnabled != 0u;
    for (uint i = 0; i < frame.areaLightCount; ++i) {
        areaSum += evalAreaLight(areaLights[i], worldPos, N, V, albedo, metallic, roughness,
                                 ltcMat, ltcMag, areaLTC);
    }

    // Secondary directional lights (#60 task 5) — fill / back lights that a
    // SCN 3-point rig ships alongside the key. Formerly collapsed into a flat
    // hemispheric ambient term (the 4.20 fold), which threw away their direction
    // (no NdotL shading gradient) and their specular entirely. Now each shades
    // with the SAME `brdf` the sun uses — real diffuse + GGX specular — just
    // without a shadow map (SCN fill/back lights are `castsShadow = false`; the
    // primary keeps the cascade rig). `dir` is pre-normalized toward the light.
    float3 dirFillSum = float3(0.0);
    for (uint i = 0; i < frame.directionalLightCount; ++i) {
        DirectionalLight dl = extraDirectionals[i];
        dirFillSum += brdf(N, V, dl.dir, albedo, metallic, roughness, dl.color);
    }

    // SSAO (half-res, gid/2). Only the indirect term is modulated — direct
    // lights have their own shadowing pathway.
    uint aoW = aoTex.get_width();
    uint aoH = aoTex.get_height();
    uint2 aoCoord = min(gid / 2, uint2(aoW - 1, aoH - 1));
    float ao = float(aoTex.read(aoCoord).r);

    // ── Indirect (IBL + optional DDGI for diffuse) ──────────────────
    float3 indirect;
    // Debug accumulators (frame.debugTerm split-render) — populated below.
    float3 dbgDiffuseIBL = float3(0.0);
    float3 dbgSpecularIBL = float3(0.0);
    float3 dbgAmbient = float3(0.0);
    if (kLightingIBLEnabled) {  // function_constant(0)
        constexpr sampler cubeSampler(filter::linear, mip_filter::linear);

        float3 F0 = mix(float3(0.04), albedo, metallic);
        float NdotV = saturate(dot(N, V));
        float3 F  = fresnelSchlickRoughness(NdotV, F0, roughness);
        float3 kD = (1.0 - F) * (1.0 - metallic);

        // Diffuse: use DDGI probe irradiance when available (one-bounce GI),
        // otherwise fall back to the sky-probe irradiance cube.
        // Phase 3.4 — EMA irradiance cache cuts the per-pixel 8-probe blend to
        // 1 texture read in steady state. Fresh probe lookup only on cache miss
        // (first frame, cache disabled, or DDGI off).
        float3 irradianceSrc;
        if (kLightingDDGIEnabled) {  // function_constant(3) — agrees with ddgi.enabled
            // Phase 3.4b — velocity-reproject the cache read. Sampling the
            // history at the *current* UV means a moving camera reads the EMA
            // that belongs to whatever world point used to sit under this pixel,
            // so a fast move smears the cache (it reconverges over ~20 frames).
            // Project THIS pixel's world position by the previous frame's VP to
            // fetch the history for the SAME surface point: fast moves reconverge
            // in ~1 frame. Off-screen reprojection (disocclusion) → fresh lookup.
            float4 prevClip  = frame.previousViewProjection * float4(worldPos, 1.0);
            float2 reprojUV  = (prevClip.xy / prevClip.w) * float2(0.5, -0.5) + 0.5;
            bool   reprojOK  = prevClip.w > 1e-5
                            && all(reprojUV >= float2(0.0))
                            && all(reprojUV <= float2(1.0));
            constexpr sampler cacheSmp(filter::linear, address::clamp_to_edge);
            float3 history  = reprojOK
                            ? float3(irrCachePrev.sample(cacheSmp, reprojUV).rgb)
                            : float3(0.0);
            bool   useCache = kLightingDDGIIrrCacheEnabled  // function_constant(4)
                           && reprojOK
                           && any(history > float3(1e-6f));
            if (useCache) {
                float3 fresh = sampleDDGIIrradiance(worldPos, N,
                                                    ddgiIrrAtlas, ddgiDepthAtlas, ddgi);
                irradianceSrc = mix(history, fresh, frame.ddgiIrrCacheBlend);
            } else {
                irradianceSrc = sampleDDGIIrradiance(worldPos, N,
                                                     ddgiIrrAtlas, ddgiDepthAtlas, ddgi);
            }
            irrCacheCur.write(half4(half3(irradianceSrc), 1.0h), gid);
        } else {
            irradianceSrc = float3(irradianceCube.sample(cubeSampler, N).rgb);
            irrCacheCur.write(half4(0.0h), gid);
        }
        // Phase 4.15 — IBL diffuse saturation boost. Procedural-gradient
        // backdrops that the extractor reuses as the IBL probe integrate
        // down to a near-grey irradiance, and `irradiance * albedo` then
        // multiplies grey × albedo → a muted tint on every indirect-lit
        // surface. We push the IBL diffuse term's chroma outward from
        // its own luminance before composing with albedo, which keeps
        // hue but lifts saturation back toward what the SCN baseline
        // shows. 1.0 disables the boost (no-op); >1.0 boosts.
        //
        // Mathematically: this saturates THE LIGHTING (the irradiance
        // probe), not the surface. The surface's albedo still drives
        // the hue; we're just refusing to let the integration's grey
        // average wash it out.
        // Phase 4.29 — ADAPTIVE saturation. The Phase 4.15 boost amplifies
        // whatever chroma the irradiance probe carries; that rescues grey /
        // pastel procedural-gradient IBL (Forest, FloatingFlowers+), but on
        // an already-warm IBL (Pizza's red oven environment) it pushes the
        // red even further, flooding the frame. Roll the boost off toward
        // 1.0 (no-op) as the source irradiance's own saturation rises, so
        // low-chroma probes get the full lift while saturated probes are
        // left alone.
        float irrLum = dot(irradianceSrc, float3(0.2126, 0.7152, 0.0722));
        float irrMax = max(irradianceSrc.r, max(irradianceSrc.g, irradianceSrc.b));
        float srcSat = (irrMax > 1e-4) ? saturate((irrMax - irrLum) / irrMax) : 0.0;
        float satBoost = mix(frame.iblDiffuseSaturation, 1.0,
                             smoothstep(0.18, 0.5, srcSat));
        // Phase 4.32 — hue-balance DESATURATION for saturated probes. The 4.15 /
        // 4.29 boost only ever lifts chroma (≥1.0); a colored-environment scene
        // (Pizza's red broiler used as IBL) needs the opposite: pull the
        // monochromatic-red irradiance toward its luminance so a cream surface
        // isn't washed pure red. Ramps in only as the probe's own saturation
        // rises, so neutral / pastel IBL (Forest, FloatingFlowers+) is untouched.
        float desat = frame.iblDiffuseDesaturation * smoothstep(0.35, 0.75, srcSat);
        float effBoost = satBoost * (1.0 - desat);
        float3 irradianceSat = max(mix(float3(irrLum), irradianceSrc, effBoost), 0.0);
        float3 diffuseIBL = kD * irradianceSat * albedo;

        // Specular IBL: sample prefilteredCube along the world reflection
        // vector at the roughness-derived mip, then apply the split-sum DFG
        // LUT (Phase 3.2) for the F0-weighted environment BRDF integral:
        //   specular = prefilteredEnv * (F0 * dfg.x + dfg.y).
        // The LUT is baked once at startup (`illumi_dfg_bake`). When
        // `frame.dfgLUTEnabled == 0` we fall back to Lagarde's roughness-
        // Schlick (`F` above) — visually decent on dielectrics, drops energy
        // at high roughness / grazing on metals. Toggle preserved for A/B.
        float3 R = reflect(-V, N);
        float mipCount = float(max(frame.iblPrefilteredMipCount, 1u));
        float lod = roughness * (mipCount - 1.0);
        float3 specEnv = float3(prefilteredCube.sample(cubeSampler, R, level(lod)).rgb);
        float3 specularIBL;
        if (kLightingDFGLUTEnabled) {  // function_constant(2)
            constexpr sampler dfgSampler(filter::linear, address::clamp_to_edge);
            float2 dfg = float2(dfgLUT.sample(dfgSampler, float2(NdotV, roughness)).rg);
            specularIBL = specEnv * (F0 * dfg.x + dfg.y);
        } else {
            specularIBL = specEnv * F;
        }

        indirect = (diffuseIBL + specularIBL) * frame.iblIntensity * ao;
        dbgDiffuseIBL = diffuseIBL * frame.iblIntensity * ao;
        dbgSpecularIBL = specularIBL * frame.iblIntensity * ao;
        // `ambientColor` is now a TRUE ambient term — only SCN `.ambient`
        // lights (uniform, no NdotL) feed it. As of #60 task 5 the secondary
        // SCN directionals (fill, back) are NO LONGER folded in here; they
        // shade with a real BRDF in `dirFillSum` above. The remaining ambient
        // supplement is upness-weighted (40% at the nadir → 100% at the
        // zenith), × albedo, AO-gated. `ambientColor == 0` is the default no-op.
        float upness = saturate(N.y * 0.5 + 0.5);
        float3 ambCol = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation);
        float3 ambSupp = mix(ambCol * 0.4, ambCol, upness) * albedo;
        indirect += ambSupp * ao;
        dbgAmbient = ambSupp * ao;
    } else {
        // Legacy hemispheric ambient — only the diffuse term, no spec.
        float upness = saturate(N.y * 0.5 + 0.5);
        float3 ambCol = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation);
        float3 amb = mix(ambCol * 0.4, ambCol, upness) * albedo;
        indirect = amb * ao;
        dbgAmbient = amb * ao;
    }

    // ── Phase 7 — per-material clearcoat (terrazzo/marble/lacquered wood) ──────
    // Clearcoat strength is packed into emission.alpha in the G-buffer (default 0).
    // A second dielectric GGX lobe (F0=0.04, roughness=0.08) is added for
    // polished surfaces. Strength 0 → no cost (branch exits immediately).
    float3 clearcoat = float3(0.0);
    float houseCC = float(emH.a);
    if (houseCC > 0.001f) {
        const float ccRough = 0.08;    // tight polish (terrazzo/marble)
        const float ccF0    = 0.04;    // dielectric IOR 1.5
        float ccNdotV = saturate(dot(N, V));
        if (NdotL_sun > 0.0) {
            float3 Hcc = normalize(V + Ld);
            float  Dcc = distributionGGX(saturate(dot(N, Hcc)), ccRough);
            float  Gcc = geometrySmith(ccNdotV, NdotL_sun, ccRough);
            float  Fcc = ccF0 + (1.0 - ccF0) * pow(1.0 - saturate(dot(Hcc, V)), 5.0);
            float  spec = (Dcc * Gcc * Fcc) / max(4.0 * ccNdotV * NdotL_sun, 1e-4);
            clearcoat += frame.directionalLightColor * spec * NdotL_sun * visibility;
        }
        if (kLightingIBLEnabled) {
            constexpr sampler ccSampler(filter::linear, mip_filter::linear);
            float3 Rcc  = reflect(-V, N);
            float  mips = float(max(frame.iblPrefilteredMipCount, 1u));
            float3 env  = float3(prefilteredCube.sample(ccSampler, Rcc,
                                                        level(ccRough * (mips - 1.0))).rgb);
            float  Fcc  = ccF0 + (1.0 - ccF0) * pow(1.0 - ccNdotV, 5.0);
            clearcoat += env * Fcc * frame.iblIntensity * ao;
        }
        clearcoat *= houseCC;
        // Energy conservation: clearcoat layer attenuates the base for grazing V.
        float baseAtten = 1.0 - houseCC * (ccF0 + (1.0 - ccF0) * pow(1.0 - saturate(dot(N, V)), 5.0));
        directSun    *= baseAtten;
        indirect     *= baseAtten;
    }

    // ── Sausage-casing clearcoat (HotdogDropUltra) ──────────────────────────
    // Casing pixels carry 0.75 in normalRoughness.w (foliage = 0, opaque = 1).
    // HotdogDrop+'s frank is a SATIN base (roughness mottle mean ≈ 0.48) under
    // a thin wet-glaze clearcoat (SCNMaterial clearCoat 0.55, ccRoughness 0.18)
    // — the glaze is what carries the tight "just off the grill" glint. A
    // single GGX lobe can't be both satin and glinting (rounds 30↔31 oscillated
    // between silicone and foam-rubber trying), so this is a real second lobe:
    // fixed-F0 dielectric GGX on the sun + a tight prefiltered-IBL sample,
    // weighted by Drop+'s 0.55 coat strength. No-op for every other scene.
    float3 hotdogCC = float3(0.0);
    if (nrH.a > 0.6h && nrH.a < 0.9h) {
        const float ccRough    = 0.18;
        const float ccF0       = 0.04;
        const float ccStrength = 0.55;
        float ccNdotV = saturate(dot(N, V));
        if (NdotL_sun > 0.0) {
            float3 Hcc = normalize(V + Ld);
            float  Dcc = distributionGGX(saturate(dot(N, Hcc)), ccRough);
            float  Gcc = geometrySmith(ccNdotV, NdotL_sun, ccRough);
            float  Fcc = ccF0 + (1.0 - ccF0) * pow(1.0 - saturate(dot(Hcc, V)), 5.0);
            float  spec = (Dcc * Gcc * Fcc) / max(4.0 * ccNdotV * NdotL_sun, 1e-4);
            hotdogCC += frame.directionalLightColor * spec * NdotL_sun * visibility;
        }
        if (kLightingIBLEnabled) {
            constexpr sampler ccSampler(filter::linear, mip_filter::linear);
            float3 Rcc  = reflect(-V, N);
            float  mips = float(max(frame.iblPrefilteredMipCount, 1u));
            float3 env  = float3(prefilteredCube.sample(ccSampler, Rcc,
                                                        level(ccRough * (mips - 1.0))).rgb);
            float  Fcc  = ccF0 + (1.0 - ccF0) * pow(1.0 - ccNdotV, 5.0);
            hotdogCC += env * Fcc * frame.iblIntensity * ao;
        }
        hotdogCC *= ccStrength;
    }

    // ── Plush fur sheen rim (Teddy Bear Press) ───────────────────────────────
    // Plush pixels carry ≈0.55 in normalRoughness.w. Short-fibre fur is retro-
    // reflective at grazing angles — the velvety bright edge a real teddy catches.
    // A cheap Fresnel-weighted, NdotL-modulated sheen on the sun + ambient, gated
    // by `frame.plushSheen` (0 ⇒ skipped). No-op for every non-plush surface.
    float3 plushSheenTerm = float3(0.0);
    if (nrH.a > 0.5h && nrH.a < 0.6h && frame.plushSheen > 0.0) {
        float fres = pow(1.0 - saturate(dot(N, V)), 3.0);     // grazing-angle edge
        float3 sheenTint = albedo * 0.5 + float3(0.5);        // warm-white fibre tip
        // Sun-lit sheen + a soft ambient sheen so back/side fur still fuzzes.
        float3 sunSheen = frame.directionalLightColor * (NdotL_sun * visibility);
        float3 ambSheen = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation) * ao;
        plushSheenTerm = sheenTint * fres * frame.plushSheen * (sunSheen + ambSheen);
    }

    // ── Phase 7b — per-material cloth sheen (velvet/wool/linen) ───────────────
    // Fabric packs its sheen strength as a NEGATIVE emission.alpha (reuses the clearcoat
    // channel; a surface is polished OR cloth, never both). A soft grazing-Fresnel
    // retroreflective rim — the velvety glow real cloth catches, which a single GGX lobe
    // can't give. Strength = -emission.alpha. No-op for every non-cloth surface (a ≥ 0).
    float3 clothSheen = float3(0.0);
    float sheenStrength = float(-emH.a);
    if (sheenStrength > 0.001f) {
        float  fres = pow(1.0 - saturate(dot(N, V)), 2.5);    // grazing-angle edge
        float3 sheenTint = albedo * 0.4 + float3(0.6);        // fibre-tip warm white
        float3 sunSheen = frame.directionalLightColor * (NdotL_sun * visibility);
        float3 ambSheen = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation) * ao;
        clothSheen = sheenTint * fres * sheenStrength * (sunSheen + ambSheen);
    }

    float3 color = directSun + transmission + dirFillSum + pointSum + spotSum + areaSum + indirect + emission + clearcoat + hotdogCC + plushSheenTerm + clothSheen;

    // Per-term split-render: isolate ONE contribution so a flooded/flat scene
    // can be decomposed. Surfaces only — sky already returned above.
    switch (frame.debugTerm) {
        case 1u: color = directSun + transmission + dirFillSum; break; // direct sun + leaf SSS + fill directionals
        case 2u: color = pointSum;        break;  // point + synthesised emissive lights
        case 3u: color = spotSum + areaSum; break; // spot + area lights
        case 4u: color = dbgDiffuseIBL;   break;  // diffuse IBL (× iblIntensity × ao)
        case 5u: color = dbgSpecularIBL;  break;  // specular IBL (× iblIntensity × ao)
        case 6u: color = emission;        break;  // G-buffer emissive surface
        case 7u: color = dbgAmbient;      break;  // ambient supplement
        default: break;                           // 0 = full composite
    }
    outHDR.write(half4(half3(color), 1.0h), gid);
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

static inline float ssaoHash12(float2 p) {
    // De Vries-style cheap hash, [0,1).
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

constant float3 kSsaoKernel[16] = {
    float3( 0.5381,  0.1856, -0.4319),
    float3( 0.1379,  0.2486,  0.4430),
    float3( 0.3371,  0.5679, -0.0057),
    float3(-0.6999, -0.0451, -0.0019),
    float3( 0.0689, -0.1598, -0.8547),
    float3( 0.0560,  0.0069, -0.1843),
    float3(-0.0146,  0.1402,  0.0762),
    float3( 0.0100, -0.1924, -0.0344),
    float3(-0.3577, -0.5301, -0.4358),
    float3(-0.3169,  0.1063,  0.0158),
    float3( 0.0103, -0.5869,  0.0046),
    float3(-0.0897, -0.4940,  0.3287),
    float3( 0.7119, -0.0154, -0.0918),
    float3(-0.0533,  0.0596, -0.5411),
    float3( 0.0352, -0.0631,  0.5460),
    float3(-0.4776,  0.2847, -0.0271)
};

kernel void illumi_ssao(
    depth2d<float, access::read>   gDepth      [[texture(0)]],
    texture2d<half, access::read>  gNormalRgh  [[texture(1)]],
    texture2d<half, access::write> outAO       [[texture(2)]],
    constant FrameUniforms&        frame       [[buffer(0)]],
    uint2                          gid         [[thread_position_in_grid]]
) {
    uint outW = outAO.get_width();
    uint outH = outAO.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    // Early-out when SSAO is disabled — write 1 (no occlusion) so the
    // lighting kernel can read unconditionally without branching.
    if (frame.ssaoIntensity <= 0.0) {
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Map half-res gid → full-res pixel for the G-buffer reads. Using the
    // 2×2 top-left avoids needing a sampler; we accept the small loss in
    // accuracy.
    uint fullW = gDepth.get_width();
    uint fullH = gDepth.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float depth = gDepth.read(fullGid);
    if (depth >= 0.99999) {
        // Sky — no surface to occlude. Treat as fully lit.
        outAO.write(half4(1.0h), gid);
        return;
    }

    half4 nrH = gNormalRgh.read(fullGid);
    float3 Nworld = octDecode(float2(nrH.rg));

    // Reconstruct view-space position and normal. SSAO works in view space so
    // sample offsets are commensurate with the camera, not the world.
    float2 ndc = (float2(fullGid) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 Pview = viewPosFromDepth(ndc, depth, frame.invProjection);
    float3 Nview = normalize((frame.view * float4(Nworld, 0.0)).xyz);

    // Build a TBN basis from a random tangent so the sample pattern is
    // rotated per-pixel.
    float rot = ssaoHash12(float2(gid)) * 6.2831853;
    float3 randomVec = normalize(float3(cos(rot), sin(rot), 0.0));
    float3 T = normalize(randomVec - Nview * dot(randomVec, Nview));
    float3 B = cross(Nview, T);
    float3x3 TBN = float3x3(T, B, Nview);

    float radius = max(0.001, frame.ssaoRadius);
    float occlusion = 0.0;
    const uint samples = 16;
    for (uint i = 0; i < samples; ++i) {
        // Move kernel sample into view space oriented along the surface.
        float3 sampleView = Pview + TBN * kSsaoKernel[i] * radius;
        // Project back to NDC to look up the actual depth at that screen
        // position.
        float4 sampleClip = frame.projection * float4(sampleView, 1.0);
        float3 sampleNDC = sampleClip.xyz / sampleClip.w;
        if (any(abs(sampleNDC.xy) > 1.0)) continue;
        float2 sUV = float2(sampleNDC.x, -sampleNDC.y) * 0.5 + 0.5;
        uint2 sPx = uint2(clamp(sUV * float2(fullW, fullH),
                                float2(0.0), float2(fullW - 1, fullH - 1)));
        float sceneDepth = gDepth.read(sPx);
        if (sceneDepth >= 0.99999) continue;
        // Reconstruct view-space scene position at the sample's NDC location.
        float3 scenePos = viewPosFromDepth(sampleNDC.xy, sceneDepth, frame.invProjection);
        // Range check — only count occluders within `radius` of P.
        float rangeCheck = smoothstep(0.0, 1.0, radius / max(1e-4, abs(Pview.z - scenePos.z)));
        // Bias slightly to avoid self-occlusion on flat surfaces.
        // View-space Z is negative going away from the camera, so an occluder
        // is "in front of" the sample when its Z is GREATER (less negative).
        if (scenePos.z > sampleView.z + 0.01) {
            occlusion += rangeCheck;
        }
    }
    float ao = 1.0 - (occlusion / float(samples)) * frame.ssaoIntensity;
    ao = clamp(ao, 0.0, 1.0);
    outAO.write(half4(half(ao)), gid);
}

// ── Screen-space reflections — gather (Phase 2 / 4.39) ───────────────────────
//
// Linear march in view space along the reflection vector. Phase 4.39 splits
// what was previously a single inline composite into three passes:
//
//   illumi_ssr_gather  (this kernel) — marches rays, outputs the pre-weighted
//       SSR delta (RGB = reflection contribution, A = hit mask) into a
//       dedicated raw texture instead of writing directly into the HDR composite.
//
//   illumi_ssr_temporal — temporally accumulates the raw SSR signal across
//       frames using a YCoCg-space variance clamp (same style as the TAA
//       upgrade), dramatically reducing single-sample noise on rough surfaces.
//
//   illumi_ssr_composite — blends the denoised SSR into the HDR composite.
//
// On miss, writes half4(0) so the temporal and composite passes see no delta.
// The IBL specular term (already in hdrTexture from illumi_lighting) covers
// off-screen rays — no double-count.

kernel void illumi_ssr_gather(
    texture2d<half,  access::read>     inHDR           [[texture(0)]],
    texture2d<half,  access::read>     gAlbedoMet      [[texture(1)]],
    texture2d<half,  access::read>     gNormalRgh      [[texture(2)]],
    depth2d<float,   access::read>     gDepth          [[texture(3)]],
    texture2d<half,  access::write>    outSSRRaw       [[texture(4)]],
    constant FrameUniforms&            frame           [[buffer(0)]],
    uint2                              gid             [[thread_position_in_grid]]
) {
    uint W = outSSRRaw.get_width();
    uint H = outSSRRaw.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float depth = gDepth.read(gid);
    if (frame.ssrIntensity <= 0.0 || depth >= 0.99999) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }

    half4 amH = gAlbedoMet.read(gid);
    half4 nrH = gNormalRgh.read(gid);
    float metallic  = float(amH.a);
    float roughness = max(0.045, float(nrH.b));
    if (roughness > 0.7) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }
    if (metallic < 0.02 && max(max(float(amH.r), float(amH.g)), float(amH.b)) < 0.02) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }
    float3 albedo = float3(amH.rgb);
    float3 Nworld = octDecode(float2(nrH.rg));

    float2 ndc = (float2(gid) + 0.5) / float2(W, H) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4x4 invProj = frame.invProjection;
    float3 Pview = viewPosFromDepth(ndc, depth, invProj);
    float3 Nview = normalize((frame.view * float4(Nworld, 0.0)).xyz);
    float3 Vview = normalize(-Pview);
    float3 Rview = reflect(-Vview, Nview);
    if (Rview.z >= 0.0) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }

    uint maxSteps = max(8u, min(frame.ssrMaxSteps, 128u));
    float stepLen = max(0.01, frame.ssrMaxDistance / float(maxSteps));
    float thickness = max(0.01, frame.ssrThickness);

    bool hit = false;
    float2 hitUV = float2(0.0);
    for (uint s = 1; s <= maxSteps; ++s) {
        float3 rayPos = Pview + Rview * (stepLen * float(s));
        if (rayPos.z >= -0.05) break;
        float4 clip = frame.projection * float4(rayPos, 1.0);
        float3 rNDC = clip.xyz / clip.w;
        if (any(abs(rNDC.xy) > 1.0)) break;
        float2 rUV = float2(rNDC.x, -rNDC.y) * 0.5 + 0.5;
        uint2 rPx = uint2(clamp(rUV * float2(W, H),
                                float2(0.0), float2(W - 1, H - 1)));
        float sceneDepth = gDepth.read(rPx);
        if (sceneDepth >= 0.99999) continue;
        float3 scenePos = viewPosFromDepth(
            float2(rNDC.x, rNDC.y), sceneDepth, invProj
        );
        float dz = scenePos.z - rayPos.z;
        if (dz < 0.0 && dz > -thickness) {
            hit = true;
            hitUV = rUV;
            break;
        }
    }

    if (hit) {
        uint2 hPx = uint2(clamp(hitUV * float2(W, H),
                                float2(0.0), float2(W - 1, H - 1)));
        float3 refl = float3(inHDR.read(hPx).rgb);
        float3 F0 = mix(float3(0.04), albedo, metallic);
        float NdotV = saturate(dot(Nview, Vview));
        float3 F = F0 + (1.0 - F0) * pow(saturate(1.0 - NdotV), 5.0);
        float2 fade2 = smoothstep(float2(0.0), float2(0.15),
                                  min(hitUV, 1.0 - hitUV));
        float fade = fade2.x * fade2.y;
        float roughAttn = 1.0 - smoothstep(0.2, 0.7, roughness);
        float3 delta = refl * F * fade * roughAttn * frame.ssrIntensity;
        outSSRRaw.write(half4(half3(delta), 1.0h), gid);
    } else {
        outSSRRaw.write(half4(0.0h), gid);
    }
}

// ── Temporal anti-aliasing (Phase 2.7 / upgraded Phase 4.39) ─────────────────
//
// Reprojects the previous frame's accumulated HDR with the velocity buffer,
// clamps the history into the current-frame neighborhood, and blends in a
// small current-frame contribution. Three upgrades land in Phase 4.39:
//
//  1. YCoCg color space — convert current + neighborhood to YCoCg before
//     building the AABB. Luminance (Y) and chrominance (Co/Cg) are clamped
//     independently, which eliminates the chromatic ghosting that RGB-space
//     clamping produces on color edges (magenta/green halos on moving objects).
//
//  2. Variance-based AABB (mean ± γ·σ) — instead of the hard min/max over
//     the 3×3, compute the per-channel mean and standard deviation and use
//     γ·σ as the clamp radius. This widens the acceptance window on smooth
//     flat surfaces (where min/max is overly tight and throws away perfectly
//     valid history) while staying tight on high-contrast edges.
//
//  3. 5-tap Catmull-Rom history sampling — replaces the bilinear sample of
//     the history buffer, eliminating the temporal blur that bilinear
//     introduces. Uses four bilinear fetches to achieve cubic precision
//     (Brian Karis, "High Quality Temporal Supersampling", SIGGRAPH 2014).
//
//  4. Velocity-magnitude disocclusion ramp — fast-moving pixels are more
//     likely to have invalid history (newly uncovered surface). Ramp the
//     current-frame weight up when the velocity is large so disoccluded
//     pixels reconverge quickly instead of ghosting.

// YCoCg ↔ RGB helpers. The lossless "scaled" variant (sum of absolute
// weights = 1) matches the float range of the history HDR buffer exactly.
static inline float3 RGBtoYCoCg(float3 c) {
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,   // Y  ∈ [0,1]
         0.5  * c.r              - 0.5  * c.b,    // Co ∈ [-0.5, 0.5]
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b     // Cg ∈ [-0.5, 0.5]
    );
}
static inline float3 YCoCgtoRGB(float3 c) {
    float t = c.x - c.z;           // Y - Cg
    // max(0) instead of saturate — HDR pipeline, no upper clamp needed.
    return max(float3(0.0), float3(t + c.y, c.x + c.z, t - c.y));
}

// 9-tap Catmull-Rom bicubic history filter (Brian Karis, "High Quality Temporal
// Supersampling", SIGGRAPH 2014). Reconstructs the history with cubic precision
// using 9 bilinear fetches, eliminating the temporal blur a single bilinear tap
// introduces, without the over-sharpening of a naïve bicubic.
//
// CRITICAL: the fractional coordinate `f` must be measured from `texPos1` (the
// first of the four 1-D samples) and lie in [0,1) — that's the domain the weight
// polynomials w0..w3 are defined on. `texPos1 = floor(samplePos − 0.5) + 0.5`,
// `f = samplePos − texPos1`. Computing `f` as `frac(pos) − 0.5` (∈ [−0.5,0.5))
// evaluates the polynomials off-domain → at a static camera f pins to a constant
// non-zero value, reconstructing a half-pixel-shifted, ringing copy of the
// history every frame → horizontal contour banding on smooth gradients.
static float3 sampleCatmullRom(
    texture2d<half, access::sample> tex,
    sampler samp,
    float2 uv,
    float2 texSize
) {
    float2 samplePos = uv * texSize;
    float2 texPos1   = floor(samplePos - 0.5) + 0.5;
    float2 f         = samplePos - texPos1;   // ∈ [0,1)

    float2 w0 = f * (-0.5 + f * ( 1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * ( 0.5 + f * ( 2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    // Combine the middle two 1-D taps (w1,w2) into one bilinear fetch.
    float2 w12     = w1 + w2;
    float2 offset12 = w2 / w12;

    float2 tc0  = (texPos1 - 1.0)      / texSize;
    float2 tc12 = (texPos1 + offset12) / texSize;
    float2 tc3  = (texPos1 + 2.0)      / texSize;

    float3 r = float3(0.0);
    r += float3(tex.sample(samp, float2(tc0.x,  tc0.y )).rgb) * (w0.x  * w0.y );
    r += float3(tex.sample(samp, float2(tc12.x, tc0.y )).rgb) * (w12.x * w0.y );
    r += float3(tex.sample(samp, float2(tc3.x,  tc0.y )).rgb) * (w3.x  * w0.y );

    r += float3(tex.sample(samp, float2(tc0.x,  tc12.y)).rgb) * (w0.x  * w12.y);
    r += float3(tex.sample(samp, float2(tc12.x, tc12.y)).rgb) * (w12.x * w12.y);
    r += float3(tex.sample(samp, float2(tc3.x,  tc12.y)).rgb) * (w3.x  * w12.y);

    r += float3(tex.sample(samp, float2(tc0.x,  tc3.y )).rgb) * (w0.x  * w3.y );
    r += float3(tex.sample(samp, float2(tc12.x, tc3.y )).rgb) * (w12.x * w3.y );
    r += float3(tex.sample(samp, float2(tc3.x,  tc3.y )).rgb) * (w3.x  * w3.y );

    return max(float3(0.0), r);
}

// Clip a reprojected history sample into the current-frame neighbourhood AABB
// along the ray from the box centre, instead of clamping it per-channel. A
// per-channel clamp can land on a colour that is OFF the centre→history line
// (a hue shift / chroma ghost); the ray clip pulls an out-of-box history onto
// the nearest box face along that line, suppressing ghosts without recolouring.
// (Karis 2014, "High-Quality Temporal Supersampling" — clip-to-AABB.)
static inline float3 clipHistoryToAABB(float3 boxMin, float3 boxMax, float3 hist) {
    float3 center = 0.5 * (boxMax + boxMin);
    float3 extent = 0.5 * (boxMax - boxMin) + float3(1e-5);
    float3 v      = hist - center;
    float3 a      = abs(v / extent);
    float  maxU   = max(a.x, max(a.y, a.z));
    return (maxU > 1.0) ? (center + v / maxU) : hist;
}

kernel void illumi_taa_resolve(
    texture2d<half,  access::read>    currentHDR  [[texture(0)]],
    texture2d<half,  access::sample>  historyHDR  [[texture(1)]],
    texture2d<half,  access::read>    velocityTex [[texture(2)]],
    texture2d<half,  access::write>   outHDR      [[texture(3)]],
    // Phase 4.44 — depth, so the resolve can tell a sky pixel (cleared depth,
    // zero velocity) from genuinely-static geometry and synthesise a
    // camera-only motion vector for the former.
    depth2d<float,   access::read>    gDepth      [[texture(4)]],
    // Previous frame's depth — for disocclusion rejection (see below).
    depth2d<float,   access::read>    gPrevDepth  [[texture(5)]],
    constant FrameUniforms&           frame       [[buffer(0)]],
    uint2                             gid         [[thread_position_in_grid]]
) {
    uint W = outHDR.get_width();
    uint H = outHDR.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float3 current = float3(currentHDR.read(gid).rgb);

    if (frame.taaEnabled == 0u || frame.taaIsFirstFrame != 0u) {
        outHDR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    float2 uv  = (float2(gid) + 0.5) / float2(W, H);

    // ── Velocity dilation ────────────────────────────────────────────────────
    // Reproject using the velocity of the CLOSEST-depth pixel in a 3×3 window,
    // not this pixel's own. At a thin foreground mover's silhouette the centre
    // pixel can carry the background's (near-zero) velocity and trail; pinning
    // the motion vector to the nearest surface stops moving edges smearing.
    float closestDepth = 1e30;
    int2  closestOff   = int2(0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(W) - 1, int(H) - 1));
            float d = gDepth.read(uint2(c));
            if (d < closestDepth) { closestDepth = d; closestOff = int2(i, j); }
        }
    }
    uint2 velCoord = uint2(clamp(int2(gid) + closestOff,
                                 int2(0), int2(int(W) - 1, int(H) - 1)));
    float2 vel = float2(velocityTex.read(velCoord).rg);

    // ── Phase 4.44: camera-only velocity for sky pixels ──────────────────────
    // Sky/far-plane pixels are never rasterised into the G-buffer, so their
    // velocity stays at the cleared 0 — a rotating camera then smears the sky
    // (and moving cloud silhouettes) because the resolve reprojects them as if
    // they were static. Reconstruct the pure-rotation motion vector from the
    // view ray at infinity: unproject the pixel to a world-space direction, then
    // project that DIRECTION (w = 0, so the camera's translation is correctly
    // ignored at infinity) by the previous frame's VP. Same UV convention as the
    // G-buffer's velocity write: (currNDC − prevNDC) · (0.5, −0.5). Gated on
    // THIS pixel's depth, not the dilated neighbour.
    if (gDepth.read(gid) >= 0.99999) {
        float2 ndc = uv * 2.0 - 1.0;
        ndc.y = -ndc.y;
        float4 farWorld = frame.invViewProjection * float4(ndc, 1.0, 1.0);
        float3 dir = farWorld.xyz / farWorld.w - frame.cameraWorldPos;
        float4 prevClip = frame.previousViewProjection * float4(dir, 0.0);
        if (prevClip.w > 1e-5) {
            float2 prevNDC = prevClip.xy / prevClip.w;
            vel = (ndc - prevNDC) * float2(0.5, -0.5);
        }
    }

    float2 historyUV = uv - vel;

    if (any(historyUV < float2(0.0)) || any(historyUV > float2(1.0))) {
        outHDR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    // ── Depth-based disocclusion rejection ────────────────────────────────────
    // Reconstruct this pixel's world position from its CURRENT depth, project it
    // by the PREVIOUS frame's view-projection, and compare the depth it WOULD
    // have had last frame against the depth ACTUALLY stored at the reprojected
    // history texel. A large mismatch means a different surface occupied this
    // screen location last frame (a fast mover vacated it) — its history is
    // invalid even when the two surfaces share luma (the case neighbourhood
    // clamping can't catch, e.g. a grey egg trailing over the grey floor). Skips
    // sky (cleared depth), already handled by the camera-only velocity above.
    float depthDisocc = 0.0;
    float dC = gDepth.read(gid);
    if (dC < 0.99999) {
        float2 ndcC   = (uv * 2.0 - 1.0) * float2(1.0, -1.0);
        float4 worldH = frame.invViewProjection * float4(ndcC, dC, 1.0);
        float3 worldC = worldH.xyz / worldH.w;
        float4 prevClip = frame.previousViewProjection * float4(worldC, 1.0);
        if (prevClip.w > 1e-5) {
            float prevZexpected = prevClip.z / prevClip.w;
            uint2 hpix = uint2(clamp(int2(historyUV * float2(W, H)),
                                     int2(0), int2(int(W) - 1, int(H) - 1)));
            float prevZactual = gPrevDepth.read(hpix);
            depthDisocc = smoothstep(0.0015, 0.01, abs(prevZexpected - prevZactual));
        }
    }

    // ── 5-tap Catmull-Rom history sample ─────────────────────────────────────
    constexpr sampler historySampler(filter::linear, address::clamp_to_edge);
    float3 history = sampleCatmullRom(historyHDR, historySampler,
                                      historyUV, float2(W, H));

    // ── 3×3 neighborhood statistics in YCoCg ─────────────────────────────────
    float3 currentYCoCg = RGBtoYCoCg(current);
    float3 historyYCoCg = RGBtoYCoCg(history);

    float3 m1 = float3(0.0), m2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(W) - 1, int(H) - 1));
            float3 s = RGBtoYCoCg(float3(currentHDR.read(uint2(c)).rgb));
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    // Minimum sigma floor prevents the variance window from collapsing to zero
    // on smooth gradients. Without this, mean ± γ·σ ≈ mean ± 0 on flat surfaces
    // → history snaps hard to mean → step banding on smooth colour gradients.
    // 0.04 in YCoCg ≈ ~1/25 of the signal range; small enough to not ghost,
    // large enough to keep smooth surfaces smooth.
    float3 sigma = max(sqrt(max(float3(0.0), m2 - m1 * m1)), float3(0.04));

    // γ = 1.5 — slightly wider than the original 1.25 to reduce over-clamping on
    // textured surfaces that look flat from the 3×3 window but aren't.
    const float gamma = 1.5;
    float3 minC = m1 - gamma * sigma;
    float3 maxC = m1 + gamma * sigma;

    // Clip (not per-channel clamp) the history into the neighbourhood AABB along
    // the centre→history ray — kills ghosts without the hue shift a per-channel
    // clamp introduces on reprojected reflections / GI that lack a true velocity.
    historyYCoCg = clipHistoryToAABB(minC, maxC, historyYCoCg);

    // ── Velocity ramp ───────────────────────────────────────────────────────
    // Ramp toward mostly-current as screen-space motion grows. A fast mover's
    // history is unreliable — even with an exact motion vector, the Catmull-Rom
    // history resample carries sub-pixel error that compounds frame-over-frame
    // into a smear. The old cap (4×blend ≈ 0.2) kept ~80% history on fast movers
    // and was the dominant cause of the "streaky" look; 0.85 clears the moving
    // pixels while static pixels keep the low blend (so SSAO/SSR/GI still denoise
    // and static edges still converge). Pairs with the depth-disocclusion term
    // below, which handles the TRAILING pixels the mover vacated.
    float velMag = length(vel);
    float disoccBlend = smoothstep(0.0015, 0.015, velMag);
    float alpha = mix(frame.taaHistoryBlend, 0.85, disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    // ── Specular-ghost rejection (Phase 4.44) ────────────────────────────────
    // Variance clipping alone can't catch a specular highlight on a fast mover:
    // the highlight's SCREEN motion ≠ the surface motion vector, so reprojection
    // lands on a stale highlight, and where the *current* neighbourhood is also
    // bright (σ large) the mean ± γ·σ window is wide enough to pass the ghost.
    // Detect it as a luminance DISAGREEMENT between the (clamped) history Y and
    // the current Y — and gate it on velocity, because a STATIC high-contrast
    // edge must still converge/AA (that's the whole point of TAA). On a still
    // camera velMag ≈ 0 → specReject ≈ 0 → this is a no-op.
    float yH = historyYCoCg.x, yC = currentYCoCg.x;
    float lumaDisagree = abs(yH - yC) / (max(yH, yC) + 0.2);
    float specReject   = lumaDisagree * smoothstep(0.004, 0.02, velMag);
    alpha = clamp(mix(alpha, 1.0, 0.5 * specReject), 0.01, 1.0);

    // Disocclusion: force (almost) full current weight where last frame held a
    // different surface — kills fast-mover trails over similar-luma backgrounds.
    alpha = clamp(mix(alpha, 1.0, 0.9 * depthDisocc), 0.01, 1.0);

    // ── HDR luma-weighted (anti-flicker) blend ────────────────────────────────
    // Weight each sample by 1/(1+luma) so a bright firefly in the reprojected
    // history can't dominate the average and smear a trail — the single biggest
    // reduction in streaking on emissive / specular content, where reprojection
    // is least reliable. Reduces to the plain `mix` for dark pixels. (Karis
    // anti-flicker weighting; blend done in YCoCg, weighted by the Y term.)
    float wC = alpha         * (1.0 / (1.0 + max(0.0, currentYCoCg.x)));
    float wH = (1.0 - alpha) * (1.0 / (1.0 + max(0.0, historyYCoCg.x)));
    float3 resultYCoCg = (currentYCoCg * wC + historyYCoCg * wH) / max(1e-5, wC + wH);

    // ── Mild luma sharpen ─────────────────────────────────────────────────────
    // Catmull-Rom history + temporal averaging soften the image; restore a touch
    // of acutance by boosting the result luma toward the current-frame high-pass
    // (current − neighbourhood mean). Luma-only so it can't add chroma fringing;
    // clamped to the neighbourhood luma range so it can't reintroduce a ghost.
    resultYCoCg.x += 0.25 * (currentYCoCg.x - m1.x);
    resultYCoCg.x  = clamp(resultYCoCg.x, minC.x, maxC.x);

    float3 result = max(float3(0.0), YCoCgtoRGB(resultYCoCg));
    outHDR.write(half4(half3(result), 1.0h), gid);
}

// ── Phase 4.39: SSAO bilateral spatial filter ────────────────────────────────
//
// Depth + normal guided bilateral filter over the half-resolution raw AO
// texture. A 3×3 window reads each neighbor's AO value and weights it by:
//
//   exp(−|depth_diff| · kDepth) · pow(max(0, dot(n_center, n_neighbor)), kNorm)
//
// Depth weights reject samples across geometric boundaries (contact edges);
// normal weights reject samples on curved surfaces where orientation diverges.
// Output feeds the temporal pass rather than lighting directly, so the spatial
// pass only needs to be stable enough for temporal to converge cleanly.

kernel void illumi_ssao_spatial(
    texture2d<half, access::read>   rawAO      [[texture(0)]],  // half-res
    depth2d<float,  access::read>   gDepth     [[texture(1)]],  // full-res
    texture2d<half, access::read>   gNormalRgh [[texture(2)]],  // full-res
    texture2d<half, access::write>  outAO      [[texture(3)]],  // half-res
    constant FrameUniforms&         frame      [[buffer(0)]],
    uint2                           gid        [[thread_position_in_grid]]
) {
    uint halfW = outAO.get_width();
    uint halfH = outAO.get_height();
    if (gid.x >= halfW || gid.y >= halfH) return;

    if (frame.ssaoDenoiseEnabled == 0u || frame.ssaoIntensity <= 0.0) {
        outAO.write(rawAO.read(gid), gid);
        return;
    }

    uint fullW = gDepth.get_width();
    uint fullH = gDepth.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float centerDepth = gDepth.read(fullGid);
    if (centerDepth >= 0.99999) {
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Reconstruct center view-Z for depth weighting.
    float2 ndcC = (float2(fullGid) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
    ndcC.y = -ndcC.y;
    float3 PviewC = viewPosFromDepth(ndcC, centerDepth, frame.invProjection);
    float3 NworldC = octDecode(float2(gNormalRgh.read(fullGid).rg));

    const float kDepth = 5.0;   // depth sensitivity (larger = tighter boundary preservation)
    const float kNorm  = 16.0;  // normal exponent  (larger = more sensitive to curvature)

    float totalAO     = 0.0;
    float totalWeight = 0.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 halfN = int2(int(gid.x) + i, int(gid.y) + j);
            halfN = clamp(halfN, int2(0), int2(int(halfW) - 1, int(halfH) - 1));
            uint2 fullN = min(uint2(halfN) * 2u, uint2(fullW - 1, fullH - 1));

            float nd = gDepth.read(fullN);
            if (nd >= 0.99999) continue;

            float2 ndcN = (float2(fullN) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
            ndcN.y = -ndcN.y;
            float3 PviewN   = viewPosFromDepth(ndcN, nd, frame.invProjection);
            float3 NworldN  = octDecode(float2(gNormalRgh.read(fullN).rg));

            float depthDiff  = abs(PviewC.z - PviewN.z);
            float wDepth     = exp(-depthDiff * kDepth);
            float wNorm      = pow(max(0.0, dot(NworldC, NworldN)), kNorm);
            float w          = wDepth * wNorm;

            totalAO     += float(rawAO.read(uint2(halfN)).r) * w;
            totalWeight += w;
        }
    }

    float filtered = (totalWeight > 1e-6) ? (totalAO / totalWeight)
                                           : float(rawAO.read(gid).r);
    outAO.write(half4(half(filtered), 0.0h, 0.0h, 1.0h), gid);
}

// ── RT diffuse denoiser (depth + normal guided bilateral) ─────────────────────
//
// `illumi_rt_lighting` writes its noisy diffuse shadow+GI term into a dedicated
// full-res buffer (`rtDiffuse`) instead of summing it straight into the HDR
// composite. This pass bilateral-filters that buffer — guided by g-buffer depth
// + normal so it can't bleed across geometric edges or curvature — then adds the
// cleaned result into the composite. Running it BEFORE the TAA resolve means
// temporal accumulation receives a far less noisy per-frame input, so it
// converges in a fraction of the frames AND survives camera/subject motion (when
// TAA history is rejected, the raw ~4-spp grain would otherwise show through —
// exactly the "more dithered" artefact under orbit).
//
// Only the diffuse term is filtered; the RT pass already composited the sharp
// reflection term directly (a bilateral that shares depth+normal would smear a
// mirror reflection across a flat floor).
struct RTDenoiseUniforms {
    uint  width; uint height; uint enabled; uint radius;
    float kDepth; float kNorm; float _pad0; float _pad1;
};

kernel void illumi_rt_denoise(
    texture2d<half,  access::read>        rtDiffuse  [[texture(0)]],  // full-res raw RT diffuse
    texture2d<float, access::read>        gDepth     [[texture(1)]],  // full-res
    texture2d<half,  access::read>        gNormalRgh [[texture(2)]],  // full-res
    texture2d<half,  access::read_write>  outHDR     [[texture(3)]],  // composite (add into)
    constant FrameUniforms&               frame      [[buffer(0)]],   // invProjection
    constant RTDenoiseUniforms&           du         [[buffer(1)]],
    uint2                                 gid        [[thread_position_in_grid]]
) {
    if (gid.x >= du.width || gid.y >= du.height) return;

    // Sky pixel: the RT pass left rtDiffuse untouched here (it early-outs on the
    // same depth test), so there's nothing to add. Leave the composite alone.
    float centerDepth = gDepth.read(gid).r;
    if (centerDepth >= 0.99999) return;

    half4 prev = outHDR.read(gid);

    // Denoise disabled (or zero radius): composite the raw diffuse straight in.
    // Identical to the pre-split additive behaviour, just across two passes.
    if (du.enabled == 0u || du.radius == 0u) {
        float3 raw = float3(rtDiffuse.read(gid).rgb);
        outHDR.write(half4(prev.rgb + half3(raw), prev.a), gid);
        return;
    }

    float2 ndcC = (float2(gid) + 0.5) / float2(du.width, du.height) * 2.0 - 1.0;
    ndcC.y = -ndcC.y;
    float3 PviewC  = viewPosFromDepth(ndcC, centerDepth, frame.invProjection);
    float3 NworldC = octDecode(float2(gNormalRgh.read(gid).rg));

    int R = int(du.radius);
    float3 total  = float3(0.0);
    float  totalW = 0.0;

    for (int j = -R; j <= R; ++j) {
        for (int i = -R; i <= R; ++i) {
            int2 p = int2(int(gid.x) + i, int(gid.y) + j);
            p = clamp(p, int2(0), int2(int(du.width) - 1, int(du.height) - 1));
            uint2 np = uint2(p);

            float nd = gDepth.read(np).r;
            if (nd >= 0.99999) continue;   // skip sky neighbours (stale rtDiffuse)

            float2 ndcN = (float2(np) + 0.5) / float2(du.width, du.height) * 2.0 - 1.0;
            ndcN.y = -ndcN.y;
            float3 PviewN  = viewPosFromDepth(ndcN, nd, frame.invProjection);
            float3 NworldN = octDecode(float2(gNormalRgh.read(np).rg));

            float wDepth   = exp(-abs(PviewC.z - PviewN.z) * du.kDepth);
            float wNorm    = pow(max(0.0, dot(NworldC, NworldN)), du.kNorm);
            float r2       = float(i * i + j * j);
            float wSpatial = exp(-r2 / (2.0 * float(R * R) + 1e-3));  // centre-weighted
            float w        = wDepth * wNorm * wSpatial;

            total  += float3(rtDiffuse.read(np).rgb) * w;
            totalW += w;
        }
    }

    float3 filtered = (totalW > 1e-6) ? (total / totalW)
                                      : float3(rtDiffuse.read(gid).rgb);
    outHDR.write(half4(prev.rgb + half3(filtered), prev.a), gid);
}

// ── Phase 4.39: SSAO temporal accumulation ───────────────────────────────────
//
// Reprojects the previous frame's AO history using the full-res velocity
// texture, clamps the history sample into a scalar variance band, and blends
// a high-weight (≈0.9) history with the current spatially-filtered AO. This
// converts the 16-sample single-tap AO into effectively hundreds of samples
// spread over time — AO "for free" from the temporal integration.
//
// Velocity is at full-res; for a half-res pixel at `gid` we read velocity
// at `gid×2` (the matching full-res texel). UV-space velocity is resolution-
// independent so the reprojection math is identical.

kernel void illumi_ssao_temporal(
    texture2d<half, access::read>       filteredAO   [[texture(0)]],  // half-res current
    texture2d<half, access::sample>     historyAO    [[texture(1)]],  // half-res history
    texture2d<half, access::read>       velocity     [[texture(2)]],  // full-res
    texture2d<half, access::write>      outAO        [[texture(3)]],  // half-res output
    texture2d<half, access::read_write> sampleCount  [[texture(4)]],  // half-res r16Float
    constant FrameUniforms&             frame        [[buffer(0)]],
    uint2                               gid          [[thread_position_in_grid]]
) {
    uint halfW = outAO.get_width();
    uint halfH = outAO.get_height();
    if (gid.x >= halfW || gid.y >= halfH) return;

    float current = float(filteredAO.read(gid).r);

    // Reset sample count and pass through on first frame or when disabled.
    if (frame.ssaoDenoiseEnabled == 0u || frame.ssaoIsFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outAO.write(half4(half(current)), gid);
        return;
    }

    // Read accumulated sample count from previous frame.
    float N = float(sampleCount.read(gid).r);

    uint fullW = velocity.get_width();
    uint fullH = velocity.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));
    float2 vel = float2(velocity.read(fullGid).rg);

    float2 halfUV = (float2(gid) + 0.5) / float2(halfW, halfH);
    float2 histUV = halfUV - vel;

    // Screen-edge disocclusion: no history available → reset count.
    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outAO.write(half4(half(current)), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float hist = float(historyAO.sample(samp, histUV).r);

    // Scalar variance clamp over the 3×3 half-res neighborhood.
    float m1 = 0.0, m2 = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(halfW) - 1, int(halfH) - 1));
            float s = float(filteredAO.read(uint2(c)).r);
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float sigma = max(sqrt(max(0.0, m2 - m1 * m1)), 0.03);
    hist = clamp(hist, m1 - 1.5 * sigma, m1 + 1.5 * sigma);

    float velMag = length(vel);
    float disoccBlend = smoothstep(0.004, 0.025, velMag);

    // Adaptive blend: 1/N drives fast convergence in early frames; once N is
    // large enough the steady-state floor (1 - ssaoTemporalBlend) takes over.
    // Velocity disocclusion resets N so newly-revealed surfaces reconverge quickly.
    if (disoccBlend > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float minAlpha = 1.0 - frame.ssaoTemporalBlend;
    float alpha = max(minAlpha, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 4.0), disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    float result = mix(hist, current, alpha);
    outAO.write(half4(half(result)), gid);
}

// ── Phase 4.39: SSR temporal accumulation ────────────────────────────────────
//
// Temporally accumulates the raw SSR gather signal (RGB = weighted reflection
// delta, A = hit mask) using a per-channel YCoCg variance clamp and a high-
// weight history blend (≈0.85). Because SSR is full-resolution and in the same
// UV space as the TAA pass, the reprojection uses the same velocity texture.
//
// Running SSR temporal BEFORE the HDR TAA pass means the denoised signal
// enters the composite with a dedicated high-alpha history that TAA can
// then smooth further — giving effectively two rounds of temporal filtering
// to the most visually noisy signal in the pipeline.

kernel void illumi_ssr_temporal(
    texture2d<half,  access::read>       ssrRaw       [[texture(0)]],  // current gather
    texture2d<half,  access::sample>     histSSR      [[texture(1)]],  // history
    texture2d<half,  access::read>       velocity     [[texture(2)]],  // full-res
    texture2d<half,  access::write>      outSSR       [[texture(3)]],  // denoised output
    texture2d<half,  access::read_write> sampleCount  [[texture(4)]],  // full-res r16Float
    constant FrameUniforms&              frame        [[buffer(0)]],
    uint2                                gid          [[thread_position_in_grid]]
) {
    uint W = outSSR.get_width();
    uint H = outSSR.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float3 current = float3(ssrRaw.read(gid).rgb);

    if (frame.ssrDenoiseEnabled == 0u || frame.ssrIsFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outSSR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    float N = float(sampleCount.read(gid).r);

    float2 uv  = (float2(gid) + 0.5) / float2(W, H);
    float2 vel = float2(velocity.read(gid).rg);
    float2 histUV = uv - vel;

    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outSSR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 hist = float3(histSSR.sample(samp, histUV).rgb);

    // YCoCg variance clamp (same algorithm as the HDR TAA upgrade).
    float3 currentYCoCg = RGBtoYCoCg(current);
    float3 histYCoCg    = RGBtoYCoCg(hist);

    float3 m1 = float3(0.0), m2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(W) - 1, int(H) - 1));
            float3 s = RGBtoYCoCg(float3(ssrRaw.read(uint2(c)).rgb));
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float3 sigma = max(sqrt(max(float3(0.0), m2 - m1 * m1)), float3(0.04));
    histYCoCg = clamp(histYCoCg, m1 - 1.5 * sigma, m1 + 1.5 * sigma);

    float velMag = length(vel);
    float disoccBlend = smoothstep(0.004, 0.025, velMag);

    if (disoccBlend > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float minAlpha = 1.0 - frame.ssrTemporalBlend;
    float alpha = max(minAlpha, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 4.0), disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    float3 resultYCoCg = mix(histYCoCg, currentYCoCg, alpha);
    float3 result = YCoCgtoRGB(resultYCoCg);
    outSSR.write(half4(half3(result), 1.0h), gid);
}

// ── Temporal accumulation of the RT diffuse (1-bounce GI + soft shadow) ───────
//
// The walls are lit almost entirely by the RT 1-bounce GI, whose hemisphere ray
// DIRECTIONS re-jitter every frame. A single frame is a few-sample Monte-Carlo
// estimate; the main TAA converges it only while the camera is static, so the
// instant you move the walls "boil" (temporal crawl). This pass gives the GI
// term its OWN velocity-reprojected exponential history BEFORE the spatial
// denoise — so it keeps converging across frames under motion, where the main
// TAA cannot. The GI is low-frequency, so it tolerates aggressive accumulation
// (and the wide neighborhood clamp) without visible ghosting.
//
// Differs from illumi_ssr_temporal on purpose: a WIDE clamp (gammaClamp, ~4 vs
// 1.5) so a clean accumulated history isn't yanked back toward the noisy current
// 3×3 mean (which for low-frequency GI noise has a tiny sigma → a tight clamp
// would re-inject the very crawl we're removing), and a tunable base blend.
struct RTGITemporalUniforms {
    uint  width; uint height; uint enabled; uint isFirstFrame;
    float blend;       // weight of the CURRENT frame in steady state (e.g. 0.06)
    float gammaClamp;  // neighborhood clamp width in sigmas (wide, e.g. 4.0)
    float _pad0; float _pad1;
};

kernel void illumi_rt_gi_temporal(
    texture2d<half,  access::read>       giRaw       [[texture(0)]],  // current rtDiffuse
    texture2d<half,  access::sample>     histGI      [[texture(1)]],  // previous accumulated
    texture2d<half,  access::read>       velocity    [[texture(2)]],  // full-res motion vectors
    texture2d<half,  access::write>      outGI       [[texture(3)]],  // accumulated → denoise input + next history
    texture2d<half,  access::read_write> sampleCount [[texture(4)]],  // full-res r16Float
    constant RTGITemporalUniforms&       u           [[buffer(0)]],
    uint2                                gid         [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;
    float3 current = float3(giRaw.read(gid).rgb);

    if (u.enabled == 0u || u.isFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outGI.write(half4(half3(current), 1.0h), gid);
        return;
    }

    float N = float(sampleCount.read(gid).r);

    float2 uv  = (float2(gid) + 0.5) / float2(u.width, u.height);
    float2 vel = float2(velocity.read(gid).rg);
    float2 histUV = uv - vel;

    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outGI.write(half4(half3(current), 1.0h), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 hist = float3(histGI.sample(samp, histUV).rgb);

    // Wide YCoCg neighborhood clamp — guards gross ghosting at on-screen
    // disocclusions, but loose enough not to fight low-frequency convergence.
    float3 curY  = RGBtoYCoCg(current);
    float3 histY = RGBtoYCoCg(hist);
    float3 m1 = float3(0.0), m2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            float3 s = RGBtoYCoCg(float3(giRaw.read(uint2(c)).rgb));
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float3 sigma = max(sqrt(max(float3(0.0), m2 - m1 * m1)), float3(0.02));
    histY = clamp(histY, m1 - u.gammaClamp * sigma, m1 + u.gammaClamp * sigma);

    // Adaptive blend: 1/N for fast early-frame convergence; u.blend is the
    // steady-state floor. GI uses a gentle disocclusion ramp (×2 not ×4) so
    // a camera pan keeps accumulating instead of dumping the whole history.
    float velMag = length(vel);
    float disocc = smoothstep(0.02, 0.08, velMag);
    if (disocc > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float alpha = max(u.blend, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 2.0), disocc);
    alpha = clamp(alpha, 0.01, 1.0);

    float3 res = YCoCgtoRGB(mix(histY, curY, alpha));
    outGI.write(half4(half3(max(float3(0.0), res)), 1.0h), gid);
}

// ── SVGF: À-trous spatiotemporal denoiser ─────────────────────────────────────
//
// Schied et al., "Spatiotemporal Variance-Guided Filtering: Real-Time
// Reconstruction for Path-Traced Global Illumination", HPG 2017.
//
// Pipeline (runs after illumi_rt_gi_temporal when svgfEnabled):
//   illumi_svgf_variance  — 3×3 spatial variance estimate of accumulated GI
//   illumi_svgf_atrous ×N — joint bilateral à-trous cascade (stride 2^level)
//
// The key SVGF weight is the luminance term:
//   w_L = exp(−|lum_A − lum_B|² / (σ_L² · var_A + ε))
//
// High local variance → loose constraint → more spatial denoising.
// Low local variance  → tight constraint → preserves converged detail.
//
// Three levels cover a spatial reach of 1+2+4 = 7px radius (≈15×15 footprint)
// at the cost of 25 samples × 3 passes — much cheaper than a direct 15×15
// bilateral (225 samples) with the same effective support.

struct SVGFAtrousUniforms {
    uint  width;   uint  height;  uint stepSize; uint _pad0;
    float sigmaL;  float sigmaZ;  float sigmaN;  float _pad1;
};

// B3-spline 1-D à-trous weights (sum = 1). 2D weight = h[i+2] × h[j+2].
static constant float kAtrousH1[5] = { 1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0 };

// Per-pixel variance estimate from a 3×3 spatial neighbourhood of the
// accumulated GI luminance. Outputs R16Float (single channel).
kernel void illumi_svgf_variance(
    texture2d<half,  access::read>  giAccum  [[texture(0)]],  // temporally accumulated GI
    texture2d<half,  access::write> varOut   [[texture(1)]],  // per-pixel variance (R)
    constant SVGFAtrousUniforms&    u        [[buffer(0)]],
    uint2                           gid      [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;

    float m1 = 0.0, m2 = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            float s = dot(float3(giAccum.read(uint2(c)).rgb),
                          float3(0.2126, 0.7152, 0.0722));
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    varOut.write(half4(half(max(0.0, m2 - m1 * m1))), gid);
}

// One level of the SVGF à-trous cascade. Samples at stride `u.stepSize` using
// B3-spline weights multiplied by depth + normal + luminance-variance
// edge-stopping functions. Filters color and propagates variance.
kernel void illumi_svgf_atrous(
    texture2d<half,  access::read>  giIn      [[texture(0)]],  // color in
    texture2d<half,  access::read>  varIn     [[texture(1)]],  // variance in (R)
    texture2d<float, access::read>  gDepth    [[texture(2)]],
    texture2d<half,  access::read>  gNormal   [[texture(3)]],
    texture2d<half,  access::write> giOut     [[texture(4)]],  // filtered color
    texture2d<half,  access::write> varOut    [[texture(5)]],  // filtered variance
    constant SVGFAtrousUniforms&    u         [[buffer(0)]],
    uint2                           gid       [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;

    float centerDepth = gDepth.read(gid).r;
    // Sky pixels carry no GI; pass through unchanged.
    if (centerDepth >= 0.99999) {
        giOut.write(giIn.read(gid), gid);
        varOut.write(varIn.read(gid), gid);
        return;
    }

    float3 giCenter  = float3(giIn.read(gid).rgb);
    float3 nCenter   = octDecode(float2(gNormal.read(gid).rg));
    float  varCenter = float(varIn.read(gid).r);
    float  lumCenter = dot(giCenter, float3(0.2126, 0.7152, 0.0722));

    float3 colorSum  = float3(0.0);
    float  varSum    = 0.0;
    float  weightSum = 0.0;

    int step = int(u.stepSize);
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            int2 p = int2(int(gid.x) + i * step, int(gid.y) + j * step);
            if (p.x < 0 || p.y < 0 || p.x >= int(u.width) || p.y >= int(u.height)) continue;
            uint2 np = uint2(p);

            float nd = gDepth.read(np).r;
            if (nd >= 0.99999) continue;

            // B3-spline kernel weight for this tap position.
            float hW = kAtrousH1[i + 2] * kAtrousH1[j + 2];

            // Depth edge-stopping (raw hardware depth difference).
            float wDepth = exp(-abs(centerDepth - nd) / (u.sigmaZ * centerDepth + 1e-5));

            // Normal edge-stopping.
            float3 nN   = octDecode(float2(gNormal.read(np).rg));
            float  wNorm = pow(max(0.0, dot(nCenter, nN)), u.sigmaN);

            // Luminance edge-stopping — variance-guided (the SVGF key insight):
            // high variance → wide constraint → more spatial denoising.
            float3 giN  = float3(giIn.read(np).rgb);
            float  lumN = dot(giN, float3(0.2126, 0.7152, 0.0722));
            float  lDif = lumCenter - lumN;
            float  wLum = exp(-(lDif * lDif) / (u.sigmaL * u.sigmaL * varCenter + 1e-4));

            float w    = hW * wDepth * wNorm * wLum;
            colorSum  += giN * w;
            varSum    += float(varIn.read(np).r) * (w * w);
            weightSum += w;
        }
    }

    if (weightSum < 1e-6) {
        giOut.write(half4(half3(giCenter), 1.0h), gid);
        varOut.write(varIn.read(gid), gid);
        return;
    }

    giOut.write(half4(half3(max(float3(0.0), colorSum / weightSum)), 1.0h), gid);
    varOut.write(half4(half(varSum / (weightSum * weightSum))), gid);
}

// ── Phase 4.39: SSR composite ─────────────────────────────────────────────────
//
// Combines the denoised SSR signal (or raw gather when temporal is off) with
// the base HDR lighting output to produce the final HDR composite texture that
// RT, volumetric, particles, TAA, bloom, and tonemap consume.
//
// Output = hdrTexture + ssrSource.rgb. When SSR is disabled (ssrRaw = 0),
// this is a plain copy of hdrTexture into hdrComposite.

kernel void illumi_ssr_composite(
    texture2d<half, access::read>  inHDR     [[texture(0)]],  // base lighting
    texture2d<half, access::read>  ssrSource [[texture(1)]],  // denoised SSR delta
    texture2d<half, access::write> outHDR    [[texture(2)]],  // hdrComposite
    constant FrameUniforms&        frame     [[buffer(0)]],
    uint2                          gid       [[thread_position_in_grid]]
) {
    uint W = outHDR.get_width();
    uint H = outHDR.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float3 base  = float3(inHDR.read(gid).rgb);
    float3 delta = float3(ssrSource.read(gid).rgb);
    outHDR.write(half4(half3(base + delta), 1.0h), gid);
}

// ── Phase 4.13a — DynamicMesh bridge ─────────────────────────────────────────
//
// `DynamicMesh` writes its per-frame compute output into separate
// `packed_float3` position + normal buffers (12-byte stride each) — the
// shape SceneKit expects from a buffer-backed `SCNGeometrySource`.
// Illuminatorama's deferred PBR pipeline reads vertices through one
// interleaved `IlluminatoramaVertex` array (96-byte stride: pos, normal,
// uv, tangent, with hidden-lane padding to match Swift's SIMD3<Float>).
// This kernel runs once per `DynamicMesh` per frame on Illuminatorama's
// own queue, immediately before the G-buffer pass that reads the
// repacked buffer — keeping vertices entirely GPU-resident and avoiding
// any CPU round-trip.
//
// Tangent synthesis: hot dogs / fluid surfaces don't sample normal
// maps, so a stable arbitrary perpendicular to `N` is fine. Pick the
// axis least aligned with N, cross-product it; orthonormalise once.

kernel void illumi_repack_pos_norm(
    const device packed_float3*  inPos       [[buffer(0)]],
    const device packed_float3*  inNorm      [[buffer(1)]],
    device Vertex*               outVertex   [[buffer(2)]],
    constant uint&               count       [[buffer(3)]],
    // Phase 4.17 — optional packed-float2 UV stream. When the host
    // passes a real UV buffer through `IlluminatoramaGPUMeshDescriptor.
    // uvBuffer` we sample it per vertex; otherwise the host binds a
    // throwaway 1-element buffer here and sets `hasUV = 0` so the
    // kernel falls back to the synthetic `(0,0)` default. Bound at
    // buffer(4) with the flag at buffer(5).
    const device packed_float2*  inUV        [[buffer(4)]],
    constant uint&               hasUV       [[buffer(5)]],
    // Optional per-vertex RGBA color stream (stride-16 float4). When the host
    // passes a real color buffer through `IlluminatoramaGPUMeshDescriptor.
    // colorBuffer` we sample it per vertex (→ multiplied into albedo at shading);
    // otherwise the host binds a throwaway buffer and sets hasColor = 0 so the
    // kernel writes identity white (a no-op). Used by the coin soup for per-coin
    // DEBUG tints, and available to any instanced GPU mesh that wants vertex color.
    const device float4*         inColor     [[buffer(6)]],
    constant uint&               hasColor    [[buffer(7)]],
    uint                         gid         [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float3 p = float3(inPos[gid]);
    float3 n = normalize(float3(inNorm[gid]));
    // Stable arbitrary tangent: cross N with whichever axis is least
    // aligned (so we don't pick a near-parallel vector and produce
    // a degenerate cross product).
    float3 axis = abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(axis, n));
    float2 uv = (hasUV != 0u) ? float2(inUV[gid]) : float2(0.0);

    Vertex v;
    v.position = p;
    v._padPos  = 0.0;
    v.normal   = n;
    v._padNrm  = 0.0;
    v.uv       = uv;
    v._padUv   = float2(0.0);
    v.tangent  = float4(t, 1.0);
    // Per-vertex color → albedo (white identity when the mesh ships no color).
    v.color    = (hasColor != 0u) ? inColor[gid] : float4(1.0);
    outVertex[gid] = v;
}

// ── Phase 4.21 — GPU mesh normal / tangent synthesis ─────────────────────────
//
// One-shot, GPU-resident replacement for the CPU `synthesiseNormals` /
// `synthesiseTangents` passes that used to run inside
// `IlluminatoramaMesh.from(scnGeometry:)` on the MAIN THREAD. For a cold-cache
// scene that CPU work — O(triangles) scatter-add of per-vertex tangents, the
// classic derivative-of-UV method — parked the run loop for seconds the first
// frame the Illuminatorama overlay attached (it converts EVERY geometry in one
// synchronous `extractFrame` tick). This moves the math onto the GPU.
//
// The accumulation is a per-vertex sum over the triangles incident on that
// vertex. Rather than float-atomic scatter (which the project deliberately
// avoids — see MLSMPM.metal's contention notes), we build a bounded per-vertex
// adjacency list with INTEGER atomics (the counting-sort idiom from
// mlsCellCount/mlsScatterParticles), then gather per vertex. Two dispatches in
// separate encoders (the second sees the first's writes) plus a blit-clear of
// the count buffer:
//
//   1. illumi_mesh_build_adjacency — one thread per triangle; for each of its
//      3 corners, claim a slot via atomic-inc of that vertex's count and store
//      the triangle id in `vertTriList[v*maxValence + slot]`. Vertices past
//      `maxValence` incident triangles drop the overflow (visually negligible
//      for a smoothed average; real meshes sit well under the cap).
//   2. illumi_mesh_synth — one thread per vertex; loop its incident triangles,
//      recompute the area-weighted face normal and UV-derivative tangent,
//      accumulate, then normalise + Gram-Schmidt the tangent against the
//      normal and write both back into the interleaved vertex buffer.
//
// `idx16`/`idx32` are the SAME index buffer bound to two slots; `isU32` selects
// which is read (the other branch is never executed, so no out-of-bounds read).
// `doNormals` / `doTangents` gate each output independently: SCN geometry that
// ships normals keeps them (doNormals = 0) and only gets tangents synthesised.

kernel void illumi_mesh_build_adjacency(
    device const ushort* idx16        [[buffer(0)]],
    device const uint*   idx32        [[buffer(1)]],
    constant uint&       isU32        [[buffer(2)]],
    constant uint&       triCount     [[buffer(3)]],
    constant uint&       maxValence   [[buffer(4)]],
    device atomic_uint*  vertTriCount [[buffer(5)]],
    device uint*         vertTriList  [[buffer(6)]],
    constant uint&       vertCount    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= triCount) return;
    uint base = tid * 3u;
    uint i0 = (isU32 != 0u) ? idx32[base]      : uint(idx16[base]);
    uint i1 = (isU32 != 0u) ? idx32[base + 1u] : uint(idx16[base + 1u]);
    uint i2 = (isU32 != 0u) ? idx32[base + 2u] : uint(idx16[base + 2u]);
    uint corners[3] = { i0, i1, i2 };
    for (uint k = 0u; k < 3u; ++k) {
        uint v = corners[k];
        // Guard malformed meshes whose indices exceed the vertex count — an
        // out-of-range write here would fault the GPU and blank every later
        // draw. The CPU reference path did the same `i < verts.count` check.
        if (v >= vertCount) continue;
        uint slot = atomic_fetch_add_explicit(&vertTriCount[v], 1u, memory_order_relaxed);
        if (slot < maxValence) vertTriList[v * maxValence + slot] = tid;
    }
}

kernel void illumi_mesh_synth(
    device Vertex*       verts        [[buffer(0)]],
    device const ushort* idx16        [[buffer(1)]],
    device const uint*   idx32        [[buffer(2)]],
    constant uint&       isU32        [[buffer(3)]],
    constant uint&       vertCount    [[buffer(4)]],
    constant uint&       maxValence   [[buffer(5)]],
    device const uint*   vertTriCount [[buffer(6)]],
    device const uint*   vertTriList  [[buffer(7)]],
    constant uint&       doNormals    [[buffer(8)]],
    constant uint&       doTangents   [[buffer(9)]],
    uint vid [[thread_position_in_grid]]
) {
    if (vid >= vertCount) return;
    uint cnt = min(vertTriCount[vid], maxValence);
    float3 nAccum = float3(0.0);
    float3 tAccum = float3(0.0);
    for (uint s = 0u; s < cnt; ++s) {
        uint tid  = vertTriList[vid * maxValence + s];
        uint base = tid * 3u;
        uint i0 = (isU32 != 0u) ? idx32[base]      : uint(idx16[base]);
        uint i1 = (isU32 != 0u) ? idx32[base + 1u] : uint(idx16[base + 1u]);
        uint i2 = (isU32 != 0u) ? idx32[base + 2u] : uint(idx16[base + 2u]);
        // Skip degenerate / out-of-range triangles (mirrors the CPU guard).
        if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
        float3 p0 = verts[i0].position;
        float3 p1 = verts[i1].position;
        float3 p2 = verts[i2].position;
        float3 e1 = p1 - p0;
        float3 e2 = p2 - p0;
        if (doNormals != 0u) {
            // Area-weighted (un-normalised cross) — the standard
            // angle/area-weighted smooth-normal scheme.
            nAccum += cross(e1, e2);
        }
        if (doTangents != 0u) {
            float2 uv0 = verts[i0].uv;
            float2 d1  = verts[i1].uv - uv0;
            float2 d2  = verts[i2].uv - uv0;
            float det  = d1.x * d2.y - d1.y * d2.x;
            if (fabs(det) > 1e-8) {
                tAccum += (e1 * d2.y - e2 * d1.y) * (1.0 / det);
            }
        }
    }
    float3 n = verts[vid].normal;
    if (doNormals != 0u) {
        n = (length(nAccum) > 1e-6) ? normalize(nAccum) : n;
        verts[vid].normal = n;
    }
    if (doTangents != 0u) {
        // Gram-Schmidt against the (possibly just-synthesised) normal.
        float3 t = tAccum - n * dot(tAccum, n);
        float len = length(t);
        if (len < 1e-5) {
            float3 axis = (fabs(n.y) < 0.99) ? float3(0, 1, 0) : float3(1, 0, 0);
            t = normalize(cross(axis, n));
        } else {
            t /= len;
        }
        verts[vid].tangent = float4(t, 1.0);
    }
}

// ── Phase 4.11 — Particles ───────────────────────────────────────────────────
//
// Compute-driven particle integration (kernel) + forward additive HDR
// render (vertex + fragment) for in-flight particles. Layout matches the
// Swift `IlluminatoramaParticle` / `IlluminatoramaParticleFrameUniforms`
// structs byte-for-byte.

struct Particle {
    float3 position;
    float  life;
    float3 velocity;
    float  size;
    float3 color;
    float  _pad;
};

struct ParticleFrameUniforms {
    float  dt;
    float3 gravity;
    float  drag;
    uint   capacity;
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

kernel void illumi_particles_step(
    device Particle*                    particles  [[buffer(0)]],
    constant ParticleFrameUniforms&     pf         [[buffer(1)]],
    uint                                gid        [[thread_position_in_grid]]
) {
    if (gid >= pf.capacity) return;
    Particle p = particles[gid];
    if (p.life <= 0.0) return;
    // Symplectic Euler: velocity first, then position. Stable for gravity
    // + drag.
    p.velocity += pf.gravity * pf.dt;
    p.velocity *= exp(-pf.drag * pf.dt);
    p.position += p.velocity * pf.dt;
    // Decay life at ~1/lifetime. We don't store lifetime per-particle in
    // 4.11 — life decays at a fixed rate so each emit's particles last
    // ~1.0s. Adjust on the host by scaling input `life`.
    p.life -= pf.dt;
    if (p.life < 0.0) p.life = 0.0;
    particles[gid] = p;
}

struct ParticleVSOut {
    float4 clipPos      [[position]];
    float3 color;
    float  life;
    float2 uv;          // (0,0) bottom-left → (1,1) top-right of the streak quad
};

// Phase 4.11 round 4 — velocity-aligned billboard quad per particle.
// Six vertices form two triangles; the vertex shader builds a tangent
// (along screen-space velocity) and bitangent so the quad stretches in
// the direction the spark is moving. This is what makes a single
// snapshot READ as motion — a 6 m/s ember now draws as a 4:1 streak
// rather than a round bead.
//
// Dispatch is `drawPrimitives(.triangle, vertexCount: capacity * 6)`.
// `vid / 6` indexes the particle; `vid % 6` selects the quad corner.
vertex ParticleVSOut illumi_particles_vs(
    uint                       vid        [[vertex_id]],
    const device Particle*     particles  [[buffer(0)]],
    constant FrameUniforms&    frame      [[buffer(1)]]
) {
    uint pid    = vid / 6;
    uint corner = vid % 6;
    Particle p  = particles[pid];

    // Quad corner offsets in (tangent, bitangent) coords. Layout is
    // two triangles sharing the diagonal:
    //   0: (-1,-1)  1: ( 1,-1)  2: (-1, 1)   (first triangle)
    //   3: (-1, 1)  4: ( 1,-1)  5: ( 1, 1)   (second triangle)
    const float2 cornerLUT[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
        float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
    };
    float2 c = cornerLUT[corner];

    ParticleVSOut o;
    if (p.life <= 0.0) {
        // Dead → emit a degenerate vertex that the rasteriser clips.
        o.clipPos = float4(0, 0, -1, -1);
        o.color = float3(0);
        o.uv = float2(0);
        o.life = 0;
        return o;
    }

    // Centre clip-space position.
    float4 cClip = frame.viewProjection * float4(p.position, 1.0);
    // First-order projection of the velocity into NDC. Treats velocity
    // as a small displacement; for streak-rendering speeds this is
    // accurate to within a few percent.
    float4 vClip = frame.viewProjection * float4(p.velocity, 0.0);
    float2 velNDC = vClip.xy / max(cClip.w, 0.1);
    float  velMag = length(velNDC);

    // Tangent / bitangent in NDC space. Fall back to horizontal when
    // the spark is essentially stationary (so a particle at apex still
    // renders as a round dot rather than a zero-width line).
    float2 tangent   = velMag > 0.001 ? velNDC / velMag : float2(1.0, 0.0);
    float2 bitangent = float2(-tangent.y, tangent.x);

    // Convert pixel size to NDC. 1400 ≈ half of internal-resolution
    // 1080p plus a fudge to keep the streak's short axis (~spark
    // thickness) thinner than the round-dot equivalent — anisotropic
    // sprites read better when narrow.
    float wAttn   = sqrt(max(cClip.w, 0.5));
    float ndcShort = (p.size / 1400.0) / wAttn;
    // Streak length along tangent. Linear in velocity above a small
    // floor so:
    //   v = 0       → ndcLong = ndcShort (square round dot)
    //   v moderate  → ndcLong = ndcShort + k·perFrameNDC (elongated)
    //   v very high → capped at ~10× short axis so the streak doesn't
    //                  exit-screen on the first frame.
    // k = 3.0 with shutter time 1.0/60 gives roughly a 4–6:1 streak at
    // the lab's spawn velocities — what the reviewer asked for.
    float perFrameNDC = velMag * (1.0 / 60.0);
    float ndcLong = ndcShort + min(3.0 * perFrameNDC, 9.0 * ndcShort);

    float2 ndcOffset = c.x * tangent * ndcLong
                     + c.y * bitangent * ndcShort;
    o.clipPos = cClip;
    // Add the corner offset in pre-divide space (×cClip.w) so it lands
    // at the intended NDC offset after the perspective divide.
    o.clipPos.xy += ndcOffset * cClip.w;

    // Colour ramp (Planckian-locus mix) × twinkle envelope.
    // Twinkle: 4·t·(1-t) is zero at birth (life=1.0) and at death
    // (life=0.0), peaks at life=0.5. Gives embers a "fade-in, peak,
    // fade-out" envelope that reads as living embers rather than
    // pop-in/pop-out flashes. Reviewer round-4 polish item.
    float t = clamp(p.life, 0.0, 1.0);
    float3 cool = float3(0.6, 0.08, 0.03);
    float3 hot  = float3(1.0, 1.0, 1.0);
    float3 ramp = mix(cool, hot, smoothstep(0.0, 0.85, t));
    float twinkle = 4.0 * t * (1.0 - t);
    o.color = p.color * ramp * twinkle;

    // UV in [0,1] for the fragment falloff.
    o.uv  = c * 0.5 + 0.5;
    o.life = p.life;
    return o;
}

// ── Phase 4.23 — host-buffer point-sprite renderer ───────────────────────────
//
// Many scenes ship their own GPU particle pipelines that write into
// `MTLBuffer`s of `(position, color)` per particle (FireworksUltra's
// starfield + burst particles, Foam's spray clouds, anything that
// previously bound to SceneKit via a `.point` primitive SCNGeometry).
// These can't go through `IlluminatoramaParticleEmitter`'s integrate→draw
// path because the host owns the simulation; we just need to render the
// buffers as additive HDR point sprites.
//
// The VS reads from two `device float*` buffers with HOST-PROVIDED
// strides (in float units, NOT bytes — Metal indexes `float*` in
// floats). FireworksUltra ships `SIMD4<Float>` (stride-4 floats);
// other scenes might ship `packed_float3` (stride-3 floats). The
// indexing math handles both without a per-buffer pipeline variant.
//
// Output is a `.point` primitive with size in pixels via `[[point_size]]`;
// the fragment shader does a soft Gaussian falloff inside `point_coord`
// for the glowing-orb look bloom turns into twinkles.

struct ExtParticleVSOut {
    float4 clipPos    [[position]];
    float  pointSize  [[point_size]];
    float3 color;
};

struct ExtParticleParams {
    uint  posStrideFloats;
    uint  colorStrideFloats;
    uint  posOffsetFloats;    // float offset of position within its stride slot
    uint  colorOffsetFloats;  // float offset of colour within its stride slot
    float pointSize;     // pixels
    float colorScale;    // host-tunable scalar applied to color before output
};

vertex ExtParticleVSOut illumi_extparticle_vs(
    uint                       vid       [[vertex_id]],
    const device float*        positions [[buffer(0)]],
    const device float*        colors    [[buffer(1)]],
    constant FrameUniforms&    frame     [[buffer(2)]],
    constant ExtParticleParams& params   [[buffer(3)]]
) {
    uint pOff = vid * params.posStrideFloats   + params.posOffsetFloats;
    uint cOff = vid * params.colorStrideFloats + params.colorOffsetFloats;
    float3 p = float3(positions[pOff + 0],
                       positions[pOff + 1],
                       positions[pOff + 2]);
    float3 c = float3(colors[cOff + 0],
                       colors[cOff + 1],
                       colors[cOff + 2]);
    ExtParticleVSOut o;
    o.clipPos   = frame.viewProjection * float4(p, 1.0);
    o.color     = c * params.colorScale;
    o.pointSize = params.pointSize;
    return o;
}

fragment float4 illumi_extparticle_fs(
    ExtParticleVSOut in       [[stage_in]],
    float2           pcoord   [[point_coord]]
) {
    // Soft Gaussian inside the point quad — gives the bright-core +
    // bloomable-halo look without an external texture. `pcoord` is
    // (0,0)..(1,1) over the point's footprint; recentre to the disc.
    float2 d = pcoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard_fragment();
    float fall = exp(-r2 * 16.0);
    return float4(in.color * fall, fall);
}

fragment float4 illumi_particles_fs(
    ParticleVSOut             in           [[stage_in]]
) {
    if (in.life <= 0.0) discard_fragment();
    // Soft Gaussian falloff over the quad's local UV. `uv` is mapped
    // from corner offsets so dx runs along tangent (long axis) and dy
    // along bitangent (short axis). The Gaussian gives a hot core with
    // a soft outer halo — the additive blend then produces glowing
    // streaks instead of the round 4 / round 5 "dumpling" look reviewer
    // flagged (hard-edged opaque blobs).
    //
    // Round 6 — debug-pass confirmed that the vertex geometry is in
    // fact producing elongated quads; the structural concern from
    // rounds 4 / 5 was a polish-stage falloff issue, not a geometry
    // bug.
    float2 d = in.uv - 0.5;
    float r2 = d.x * d.x + d.y * d.y;
    if (r2 > 0.25) discard_fragment();
    // Half-width at half-max ~0.2 in UV space; outer ~30% gets soft.
    float falloff = exp(-12.0 * r2);
    return float4(in.color * falloff, falloff);
}

// ── Auto-exposure (Phase 4.21) ───────────────────────────────────────────────
//
// One threadgroup of 256 threads scans the HDR target on a stride pattern
// (every ~8th pixel in each axis), accumulates log-luminance into shared
// memory, threadgroup-reduces, and EMAs the result into a tiny shared
// MTLBuffer the tonemap then reads its `exposure` from. The whole loop is
// GPU-only — no `getBytes`, no `waitUntilCompleted`, no CPU↔GPU sync. The
// EMA absorbs noise so a single bright pixel doesn't make the scene darken
// for a frame; the smoothing constant adapts to the per-frame `dt` the
// host hands in.
//
// Why log-luminance: the human visual system responds to log brightness,
// not linear, and so do every published auto-exposure paper (Reinhard 2002
// onwards). Averaging in log space then exponentiating gives the geometric
// mean, which is a much better "what brightness is this scene" signal
// than the arithmetic mean (which gets pulled to the sky's blown-out
// highlights). Result: outdoor scenes with a bright sky don't crush
// midtones; indoor scenes don't overbrighten because the ceiling lights
// average a few stops above the rest of the room.
//
// Buffer layout (4 floats, see ExposureState struct below):
//   [0] previous-frame target log-luminance (kernel writes here, kept
//       between frames so the EMA has a reference)
//   [1] previous-frame smoothed exposure (the value the tonemap reads)
//   [2] target log-luminance just-computed this frame (debug + diagnostic)
//   [3] per-frame dt the host hands in (controls EMA speed)

struct ExposureState {
    float prevTargetLogLum;
    float smoothedExposure;
    float newTargetLogLum;  // written each frame, surfaced for diagnostics
    float deltaTime;        // host-driven, in seconds
};

// Single threadgroup, 256 threads. Each thread strides across the image
// reading every Nth pixel, accumulates log-luminance into a per-thread
// local sum, then threadgroup-reduces via shared memory.
//
// Sample density: ~32 samples per thread × 256 threads = ~8K samples.
// For a 1920×1080 image that's 1 in ~250 pixels — plenty for a stable
// luminance estimate that doesn't trigger on outlier bright pixels.

kernel void illumi_exposure_estimate(
    texture2d<half, access::sample> inHDR     [[texture(0)]],
    device ExposureState&           state     [[buffer(0)]],
    constant uint2&                 imgSize   [[buffer(1)]],
    constant float4&                params    [[buffer(2)]],  // x=targetEV, y=halfLife, z=maxBoost, w=minBoost
    threadgroup float*              sharedAcc [[threadgroup(0)]],
    threadgroup uint*               sharedCnt [[threadgroup(1)]],
    uint                            tid       [[thread_position_in_threadgroup]],
    uint                            tgSize    [[threads_per_threadgroup]]
) {
    constexpr sampler s(filter::linear,
                        address::clamp_to_edge,
                        coord::normalized);

    // Each thread strides through the image. Total samples per thread
    // is bounded so big images don't slow us down quadratically. Sample
    // log-luminance only when it's finite and above a floor — very dark
    // pixels (< -8 EV ≈ 0.004 linear) get clamped to the floor so a
    // few opaque black pixels don't drag the average into deep shadow.
    float thisAcc = 0.0;
    uint  thisCnt = 0;
    const float minLogLum = -8.0;   // ~0.004 linear
    const float maxLogLum =  8.0;   // ~3000 linear (no over-bright HDR)
    const uint samplesPerThread = 32;
    // Stride pattern: tid handles a regular grid across the image. We
    // pick samples by interpreting `tid + k*tgSize` as a 1D index over
    // a coarse (W/8 × H/8) grid — 8x downsample, then 32 such samples
    // per thread covers ~half of the coarse grid every frame.
    uint coarseW = max(1u, imgSize.x / 8u);
    uint coarseH = max(1u, imgSize.y / 8u);
    uint coarseN = coarseW * coarseH;
    for (uint k = 0; k < samplesPerThread; ++k) {
        uint idx = (tid + k * tgSize) % coarseN;
        uint cx = idx % coarseW;
        uint cy = idx / coarseW;
        // Sample at the centre of the coarse cell, in normalised UV
        // so the kernel doesn't have to know whether HDR is at the
        // internal or output resolution.
        float2 uv = (float2(cx, cy) + 0.5) / float2(coarseW, coarseH);
        float3 rgb = float3(inHDR.sample(s, uv).rgb);
        // Phase 4.30 — metering brightness that accounts for the MAX channel,
        // not just Rec.709 luminance. Pure luma badly under-weights saturated
        // colours (a clipping-bright red reads as only 0.21·R of luma), so a
        // red-dominated scene (Pizza's warm oven IBL × red sauce) metered as
        // "dark" and auto-exposure pushed it brighter — driving the red
        // channel past 1 into a flat clip, and washing the whole frame. Take
        // the larger of luma and half the max channel: a saturated bright
        // channel now lifts the metered brightness (→ less boost, no clip),
        // while a genuinely dark scene (low max channel too) still boosts.
        float lum  = dot(rgb, float3(0.2126, 0.7152, 0.0722));
        float maxc = max(rgb.r, max(rgb.g, rgb.b));
        lum = max(lum, 0.5 * maxc);
        if (isfinite(lum) && lum > 0.0) {
            float ll = log2(lum);
            ll = clamp(ll, minLogLum, maxLogLum);
            thisAcc += ll;
            thisCnt += 1u;
        }
    }
    sharedAcc[tid] = thisAcc;
    sharedCnt[tid] = thisCnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory.
    for (uint stride = tgSize / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sharedAcc[tid] += sharedAcc[tid + stride];
            sharedCnt[tid] += sharedCnt[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint cnt = sharedCnt[0];
        float target;
        if (cnt > 0u) {
            target = sharedAcc[0] / float(cnt);
        } else {
            // No samples landed in the valid range (whole image black).
            // Keep the previous target so the exposure stays where it
            // was rather than snapping to mid-grey.
            target = state.prevTargetLogLum;
        }
        // Auto-exposure: we want the target luminance to land at
        // `2^targetEV`. So the exposure scalar is `2^(targetEV - target)`.
        // Negative target log lum (scene is dim) → positive exposure
        // boost. Positive target (scene is bright) → exposure compression.
        float targetEV = params.x;
        float wantedExposure = exp2(targetEV - target);
        // EMA toward `wantedExposure` with a half-life set by
        // `params.y` seconds. Convert half-life + dt into a per-frame
        // mix factor: `alpha = 1 - 2^(-dt / halfLife)`.
        float halfLife = max(params.y, 1.0 / 60.0);
        float dt = max(state.deltaTime, 1.0 / 240.0);
        float alpha = 1.0 - exp2(-dt / halfLife);
        float prev = state.smoothedExposure;
        // Clamp via the host-provided min/max boost — for dark scenes
        // we don't want auto-exposure to fabricate light into mid-grey
        // (Eggs, FloatingFlowers+ overshoot at 4-6×), and for bright
        // scenes we don't want it to crush below a sensible floor.
        // Defaults (host-side) are ~3× max boost / 0.25× min — that
        // gives ±1.5 stops of headroom, which is the SCN feel.
        float maxBoost = max(params.z, 0.05);
        float minBoost = max(params.w, 0.01);
        wantedExposure = clamp(wantedExposure, minBoost, maxBoost);
        float next = mix(prev, wantedExposure, alpha);
        state.smoothedExposure   = next;
        state.prevTargetLogLum   = target;
        state.newTargetLogLum    = target;
    }
}

// ── Bloom ────────────────────────────────────────────────────────────────────
//
// Crude two-pass separable gaussian on a thresholded copy of the HDR target.
// Phase 1 ships a single blur pass at half-res rather than a full mip chain —
// it gives a "halo around bright pixels" that sells HDR without the bookkeeping
// of a Karis-style downsample/upsample pyramid.

kernel void illumi_bloom_threshold(
    texture2d<half, access::read>  inHDR     [[texture(0)]],
    texture2d<half, access::write> outBright [[texture(1)]],
    constant FrameUniforms&        frame     [[buffer(0)]],
    uint2                          gid       [[thread_position_in_grid]]
) {
    uint w = outBright.get_width();
    uint h = outBright.get_height();
    if (gid.x >= w || gid.y >= h) return;
    // Sample 2×2 from the full-res HDR for a half-res downsample.
    uint2 src = gid * 2;
    half4 a = inHDR.read(src);
    half4 b = inHDR.read(min(src + uint2(1, 0), uint2(inHDR.get_width() - 1, inHDR.get_height() - 1)));
    half4 c = inHDR.read(min(src + uint2(0, 1), uint2(inHDR.get_width() - 1, inHDR.get_height() - 1)));
    half4 d = inHDR.read(min(src + uint2(1, 1), uint2(inHDR.get_width() - 1, inHDR.get_height() - 1)));
    float3 avg = float3((a + b + c + d).rgb) * 0.25;
    float lum = dot(avg, float3(0.2126, 0.7152, 0.0722));
    float t = max(0.0, lum - frame.bloomThreshold);
    float3 bright = avg * (t / max(lum, 1e-4));
    outBright.write(half4(half3(bright), 1.0h), gid);
}

kernel void illumi_bloom_blur_h(
    texture2d<half, access::read>  inTex  [[texture(0)]],
    texture2d<half, access::write> outTex [[texture(1)]],
    uint2                          gid    [[thread_position_in_grid]]
) {
    uint w = outTex.get_width();
    uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;
    // 9-tap gaussian, sigma ≈ 2.0. Hardcoded weights so the shader stays
    // self-contained.
    const float weights[5] = { 0.227027, 0.194595, 0.121622, 0.054054, 0.016216 };
    float3 acc = float3(inTex.read(gid).rgb) * weights[0];
    for (int i = 1; i < 5; ++i) {
        uint2 a = uint2(min(int(gid.x) + i, int(w) - 1), gid.y);
        uint2 b = uint2(max(int(gid.x) - i, 0), gid.y);
        acc += float3(inTex.read(a).rgb) * weights[i];
        acc += float3(inTex.read(b).rgb) * weights[i];
    }
    outTex.write(half4(half3(acc), 1.0h), gid);
}

kernel void illumi_bloom_blur_v(
    texture2d<half, access::read>  inTex  [[texture(0)]],
    texture2d<half, access::write> outTex [[texture(1)]],
    uint2                          gid    [[thread_position_in_grid]]
) {
    uint w = outTex.get_width();
    uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;
    const float weights[5] = { 0.227027, 0.194595, 0.121622, 0.054054, 0.016216 };
    float3 acc = float3(inTex.read(gid).rgb) * weights[0];
    for (int i = 1; i < 5; ++i) {
        uint2 a = uint2(gid.x, min(int(gid.y) + i, int(h) - 1));
        uint2 b = uint2(gid.x, max(int(gid.y) - i, 0));
        acc += float3(inTex.read(a).rgb) * weights[i];
        acc += float3(inTex.read(b).rgb) * weights[i];
    }
    outTex.write(half4(half3(acc), 1.0h), gid);
}

// ── Tonemap + composite ──────────────────────────────────────────────────────
//
// ACES filmic curve (Krzysztof Narkowicz's approximation). Reads HDR + bloom,
// applies exposure, ACES-tonemaps, gamma-encodes, writes final LDR.

static inline float3 aces(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ── Color-grade: white-balance gain from a Kelvin temperature ───────────────
// Maps a correlated colour temperature (~2000–10000 K) to a normalized linear
// RGB channel gain that, when MULTIPLIED into a neutral scene, warms it (low K)
// or cools it (high K). 6500 K → (1,1,1) exactly (no-op default). We use a
// cheap polynomial approximation of the daylight locus' channel response
// rather than a full Planckian/CIE conversion — it only has to read tasteful
// across the slider, not be colorimetrically exact. Normalized so the green
// channel (and the luma) stays ≈1, i.e. the grade tints rather than dims.
static inline float3 whiteBalanceGain(float kelvin) {
    // Reference is 6500 K (D65). Below → push red, pull blue (warm); above →
    // push blue, pull red (cool). A smooth, monotonic curve in 1000s-of-K.
    float t = (kelvin - 6500.0) / 6500.0;        // 0 at D65; ~-0.69 at 2000 K; ~+0.54 at 10000 K
    float r = 1.0 - 0.45 * t;                    // warmer (low K) → more red
    float b = 1.0 + 0.55 * t;                    // warmer (low K) → less blue
    float g = 1.0 - 0.04 * t * t;                // slight green dip away from D65
    float3 gain = float3(max(r, 0.0), max(g, 0.0), max(b, 0.0));
    // Renormalize to unit luma so the white-balance only shifts hue, not exposure.
    float lum = dot(gain, float3(0.2126, 0.7152, 0.0722));
    return (lum > 1e-4) ? gain / lum : float3(1.0);
}

// Green↔magenta tint on the [-1, 1] axis. tint > 0 pushes magenta (boost R+B,
// cut G); tint < 0 pushes green. Luma-preserving by construction (the green
// move is twice the magenta half-move). 0 = no-op.
static inline float3 tintGain(float tint) {
    float g = 1.0 - 0.20 * tint;                 // magenta (tint>0) cuts green
    float rb = 1.0 + 0.10 * tint;                // ...and lifts red+blue
    return float3(rb, g, rb);
}

// Tonemapped-domain tone curve. Operates on a 0..1 LDR colour:
//   • contrast pivots around mid-grey 0.18 (1.0 = no-op)
//   • shadows lifts (>1) / crushes (<1) the low-luma end (1.0 = no-op)
//   • highlights lifts/rolls the high-luma end (1.0 = no-op)
// Shadows/highlights are luma-weighted so mid-tones stay put and the two ends
// move independently. All three default to 1.0 → an exact no-op.
static inline float3 toneCurve(float3 c, float contrast, float shadows, float highlights) {
    // Contrast around mid-grey pivot.
    const float pivot = 0.18;
    c = (c - pivot) * contrast + pivot;
    c = max(c, 0.0);
    // Per-pixel luma drives the shadow/highlight weights.
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    // Smooth low/high masks: shadowW ≈ 1 in blacks → 0 by mid; highW the inverse.
    float shadowW = 1.0 - smoothstep(0.0, 0.5, lum);
    float highW   = smoothstep(0.5, 1.0, lum);
    // Multiplicative lift/crush — keeps hue, scales magnitude per region.
    float scale = mix(1.0, shadows, shadowW) * mix(1.0, highlights, highW);
    return max(c * scale, 0.0);
}

// SSAA downsample. The HDR + bloom textures are sized to the INTERNAL
// resolution (output × `internalRenderScale`); this kernel dispatches over
// the OUTPUT resolution and runs a 4-tap bilinear-hardware downsample per
// output pixel. Each bilinear tap is a "free" 2×2 input-texel average
// thanks to `filter::linear`; placing the 4 taps at ±0.5 input-texel
// offsets around the output pixel's centre covers a 3×3 input-texel
// footprint with a smooth, ringing-free tent kernel — exactly the shape
// you want for 1.5× downscale. At `internalRenderScale == 1.0` the four
// taps collapse to a single bilinear read at the texel centre, so the
// path is correct (if slightly more expensive than the original
// `read(gid)`) at the no-SSAA setting too.
// Phase 4.28 — the tonemap is a RENDER PASS (fullscreen triangle + fragment),
// not a compute kernel. The output texture is sampled by SceneKit as
// `background.contents` on its OWN command queue; a compute `texture.write`
// to a `.private` texture is not reliably made visible to that cross-queue
// read without a CPU completion wait (which would stall the overlay's main-
// thread tick). A render pass with `storeAction = .store` resolves/stores the
// attachment at end-of-pass, which IS visible to the subsequent SceneKit
// sample — same mechanism the (working) `blankSky` clear relies on. The
// tonemap math is unchanged; only the dispatch shape moved from grid threads
// to a fragment over the output attachment.

struct TonemapVSOut {
    float4 position [[position]];
    float2 uv;
};

vertex TonemapVSOut illumi_tonemap_vs(uint vid [[vertex_id]]) {
    // Oversized fullscreen triangle covering the viewport in one primitive:
    // verts (0,0)->(-1,-1 NDC), (2,0)->(3,-1), (0,2)->(-1,3). The UV is in
    // [0,1] across the screen (y flipped so v=0 is the top, matching the
    // texture sampling the compute path used).
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    TonemapVSOut o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

fragment float4 illumi_tonemap_fs(
    TonemapVSOut                    in        [[stage_in]],
    texture2d<half, access::sample> inHDR     [[texture(0)]],
    texture2d<half, access::sample> inBloom   [[texture(1)]],
    // Phase 9 — film-stock LUT: 16×16×16 3D texture (B slices left-to-right in
    // the 256×16 PNG strip). Bound when filmLUTStrength > 0; nil-checked below.
    texture3d<float, access::sample> filmLUT  [[texture(2)]],
    constant FrameUniforms&         frame     [[buffer(0)]],
    // Phase 4.21 — auto-exposure read (see below).
    const device ExposureState&     expoState [[buffer(1)]]
) {
    constexpr sampler downSampler(filter::linear,
                                  address::clamp_to_edge,
                                  coord::normalized);

    float2 inSize  = float2(inHDR.get_width(), inHDR.get_height());
    float2 invInSize = 1.0 / inSize;
    // `in.uv` is the output pixel centre in normalised coords, equivalent to
    // the compute path's `(gid+0.5)/outSize`. The 4-tap offsets are ±0.5
    // INPUT texels, i.e. ±0.5*invInSize in normalised space.
    const float2 offsets[4] = {
        float2(-0.5, -0.5), float2( 0.5, -0.5),
        float2(-0.5,  0.5), float2( 0.5,  0.5)
    };
    float3 hdr = float3(0.0);
    if (frame.chromaticAberration > 0.0) {
        // Lens-style transverse chromatic aberration: split the red channel
        // outward and the blue channel inward along the radius from the frame
        // centre, growing toward the edges (zero in the middle, like a real
        // lens). `0.03` maps a strength of 1.0 to a ~1.5%-of-frame split at the
        // corners. Reduces to the plain 4-tap when the strength is 0 (gated
        // above so the default-off path keeps its 4 samples).
        float2 caOff = (in.uv - 0.5) * (frame.chromaticAberration * 0.03);
        for (int i = 0; i < 4; ++i) {
            float2 uv = in.uv + offsets[i] * invInSize;
            hdr.r += float(inHDR.sample(downSampler, uv + caOff).r);
            hdr.g += float(inHDR.sample(downSampler, uv).g);
            hdr.b += float(inHDR.sample(downSampler, uv - caOff).b);
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            float2 uv = in.uv + offsets[i] * invInSize;
            hdr += float3(inHDR.sample(downSampler, uv).rgb);
        }
    }
    hdr *= 0.25;

    // ── Spherical aberration ───────────────────────────────────────────────────
    // A real lens's outer zones focus at a slightly different plane than the
    // paraxial rays, so off-axis detail loses sharpness while the centre stays
    // crisp — a radial defocus that grows quadratically from the optical axis
    // toward the edges. We blur all channels equally (unlike the transverse CA
    // above, which splits them) in the HDR domain so the softening wraps around
    // bright highlights before tonemapping, reading as the classic dreamy
    // soft-focus halation. 0 → branch skipped (exact no-op).
    if (frame.sphericalAberration > 0.0) {
        float2 d     = in.uv - 0.5;
        float  rNorm = saturate(dot(d, d) * 2.0);   // 0 at centre → 1 at corners
        // Radius (in INTERNAL-resolution texels) scales with both the off-axis
        // distance and the strength. At SA=1 the corners pull a ~14-texel disc;
        // at the SA=3 max it's ~42, a heavy soft-focus. Texel-space so the look
        // is stable across the internal render scale.
        float  blurPx = rNorm * frame.sphericalAberration * 14.0;
        if (blurPx > 0.25) {
            // Filled disc, not a ring: a centre tap plus two concentric 6-tap
            // rings (offset half a step) so the kernel covers the disc instead
            // of leaving a hard double-image. Outer ring at full radius, inner
            // at 55%, centre weighted highest → an approximately Gaussian falloff.
            float2 rOut = blurPx * invInSize;
            float2 rIn  = rOut * 0.55;
            float3 acc  = float3(inHDR.sample(downSampler, in.uv).rgb) * 3.0;
            float  wsum = 3.0;
            for (int i = 0; i < 6; ++i) {
                float aO = (float(i)        ) * (M_PI_F / 3.0);
                float aI = (float(i) + 0.5) * (M_PI_F / 3.0);
                acc += float3(inHDR.sample(downSampler, in.uv + float2(cos(aO), sin(aO)) * rOut).rgb) * 1.0;
                acc += float3(inHDR.sample(downSampler, in.uv + float2(cos(aI), sin(aI)) * rIn).rgb)  * 2.0;
            }
            wsum += 6.0 * 1.0 + 6.0 * 2.0;          // 3 + 6 + 12 = 21
            float3 blurred = acc / wsum;
            // Blend ramps with off-axis distance AND strength, so the centre is
            // always sharp and a higher slider both widens AND deepens the haze.
            float blend = saturate(rNorm * (0.55 + frame.sphericalAberration * 0.45));
            hdr = mix(hdr, blurred, blend);
        }
    }

    // Bloom is at half of the INTERNAL resolution. The same normalised UV
    // works directly because `filter::linear` interpolates across mip0
    // regardless of the source's pixel count.
    float3 bloom = float3(inBloom.sample(downSampler, in.uv).rgb);

    float3 mixed = hdr + bloom * frame.bloomIntensity;
    // Phase 4.21 — read the GPU-computed smoothed exposure from the
    // auto-exposure buffer when the host has the feature on; otherwise
    // fall back to the static scalar in FrameUniforms. The estimator
    // kernel ran earlier in the same frame's command buffer so the
    // value is fresh.
    //
    // `frame.exposure` (the "Exposure" slider) is a MULTIPLICATIVE
    // exposure-compensation applied on top of the auto-exposure result, so the
    // slider always has an effect — previously it was ignored whenever
    // auto-exposure was on (the default for every scene), which made the control
    // appear dead. In manual mode (auto off) the base is 1.0, so `base · exposure`
    // == the old `frame.exposure` behaviour (default slider 1.0 → no change).
    float autoBase = (frame.autoExposureEnabled != 0u)
                     ? expoState.smoothedExposure
                     : 1.0;
    float exposure = autoBase * frame.exposure;
    // ── Color-grade: white-balance + tint on LINEAR HDR, pre-tonemap ──────────
    // Channel-multiply gains are most physical in linear light (they model a
    // sensor/illuminant shift), so they go in before exposure + ACES. Defaults
    // (whiteBalanceK = 6500, tint = 0) make both gains exactly (1,1,1) → no-op.
    float3 graded = mixed * whiteBalanceGain(frame.whiteBalanceK) * tintGain(frame.tint);
    float3 mapped = aces(graded * exposure);
    // Phase 4.15 — post-tonemap saturation boost. Narkowicz's fitted ACES
    // famously compresses midtone chroma harder than SCN's HDR chain, so
    // the deferred pipeline reads consistently flatter than the SCN
    // baseline. Cheap post-curve fix: lerp from luminance back to the
    // tonemapped colour with `frame.tonemapSaturation` (typ. 1.15–1.3).
    // At 1.0 this is a no-op; above 1.0 it pushes chroma outward from
    // the per-pixel luminance, which preserves the filmic shoulder while
    // restoring tint. Rec.709 luminance weights are the standard choice.
    float lum = dot(mapped, float3(0.2126, 0.7152, 0.0722));
    mapped = max(mix(float3(lum), mapped, frame.tonemapSaturation), 0.0);
    // Clamp after the saturation push — boosting past 1.0 can take channels
    // negative on near-greys, and `pow(negative, 1/2.2)` returns NaN that
    // then propagates through any subsequent composite.
    mapped = saturate(mapped);
    // ── Color-grade: contrast / shadows / highlights tone curve ───────────────
    // Applied in the tonemapped (0..1) domain so the pivot, lift and roll-off
    // act on display-referred values. Defaults (contrast = shadows = highlights
    // = 1.0) make this an exact no-op.
    mapped = saturate(toneCurve(mapped, frame.contrast, frame.shadows, frame.highlights));

    // ── Axial chromatic aberration ("purple fringing") ────────────────────────
    // Longitudinal CA: a real lens focuses wavelengths at slightly different
    // depths, so high-contrast edges grow a coloured halo — classically violet on
    // the dark side of a bright edge and green on the bright side. Frame-uniform
    // (unlike the radial lateral CA at the top of this pass). We detect luminance
    // edges in the HDR input, compress the (unbounded HDR) gradient to 0..1, then
    // tint: a 5-tap Laplacian's sign decides dark side (tint) vs bright side
    // (complement). 0 strength → the branch never runs (exact no-op).
    if (frame.fringe > 0.0) {
        float2 px = invInSize;
        float lC = dot(float3(inHDR.sample(downSampler, in.uv).rgb),                       float3(0.2126, 0.7152, 0.0722));
        float lL = dot(float3(inHDR.sample(downSampler, in.uv - float2(px.x, 0.0)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float lR = dot(float3(inHDR.sample(downSampler, in.uv + float2(px.x, 0.0)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float lU = dot(float3(inHDR.sample(downSampler, in.uv - float2(0.0, px.y)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float lD = dot(float3(inHDR.sample(downSampler, in.uv + float2(0.0, px.y)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float gx = lR - lL;
        float gy = lD - lU;
        float edge = sqrt(gx * gx + gy * gy);
        edge = edge / (edge + 1.0);                       // compress HDR contrast → 0..1
        float lap = (lL + lR + lU + lD) - 4.0 * lC;       // > 0 on the dark side of a bright edge
        // Tint comes in as sRGB (picked by eye in the UI); `mapped` is linear, so
        // decode before adding (sister of the known solid-colour sRGB→linear bug).
        float3 tintSRGB = (lap >= 0.0)
            ? float3(frame.fringeTintR, frame.fringeTintG, frame.fringeTintB)
            : (1.0 - float3(frame.fringeTintR, frame.fringeTintG, frame.fringeTintB));
        float3 tintLinear = pow(tintSRGB, 2.2);
        mapped = saturate(mapped + tintLinear * (edge * frame.fringe * 0.5));
    }

    // ── Phase 4.39: debanding dither ──────────────────────────────────────────
    // The output attachment is 8-bit `bgra8Unorm_srgb`. A smooth lit gradient
    // (e.g. the room's off-white walls) quantises into discrete 8-bit steps →
    // visible horizontal contour banding. The standard AAA fix is a triangular-
    // PDF (TPDF) dither of ±1 LSB applied before the store: it decorrelates the
    // quantisation error so the eye spatially averages the gradient back to
    // smooth. The dither must be in the SAME space as the quantisation — i.e.
    // applied to the sRGB-encoded value, since the GPU store quantises after
    // the sRGB OETF. We approximate the sRGB encode, dither, and decode back to
    // linear (the attachment re-encodes on store).
    //
    // TPDF = (two independent uniform randoms) differenced → triangular noise,
    // which is the optimal dither distribution (flat error, no noise modulation).
    if (frame.debandDitherEnabled != 0u) {
        float3 srgb = pow(mapped, float3(1.0 / 2.2));   // approx sRGB encode
        // Interleaved-gradient-noise hashes for two decorrelated uniform samples.
        float2 px = in.position.xy;
        float n0 = fract(52.9829189 * fract(dot(px, float2(0.06711056, 0.00583715))));
        float n1 = fract(52.9829189 * fract(dot(px + 113.0, float2(0.06711056, 0.00583715))));
        float tpdf = (n0 + n1) - 1.0;                    // ∈ [-1, 1], triangular
        srgb += tpdf * (1.0 / 255.0);                    // ±1 LSB at 8-bit
        mapped = pow(saturate(srgb), float3(2.2));       // decode back to linear
    }

    // Phase 9 — film-stock LUT colour grade. Samples a 16×16×16 3D LUT
    // (stored as a 256×16 PNG strip: 16 blue slices, each 16×16, laid left
    // to right; the Swift host unpacks this into a proper MTLTexture3D).
    // The LUT expects Cineon log input but we apply it post-ACES-tonemapper
    // for a film-inspired grade (not technically accurate emulation — per spec).
    // `filmLUTStrength` blends between ungraded and graded result.
    if (frame.filmLUTStrength > 0.001) {
        constexpr sampler lutSampler(filter::linear, address::clamp_to_edge);
        // Remap `mapped` from [0,1] linear into LUT normalised coords. A 16-cell
        // LUT needs a half-texel inset so the sample lands at the cell centre:
        // coord = (mapped * (N-1) + 0.5) / N  where N = 16.
        float3 uvw = (mapped * 15.0 + 0.5) / 16.0;
        float3 graded = filmLUT.sample(lutSampler, uvw).rgb;
        mapped = mix(mapped, graded, frame.filmLUTStrength);
        mapped = saturate(mapped);
    }

    // Write LINEAR. The output attachment is `.bgra8Unorm_srgb`, so the GPU
    // store applies the sRGB OETF (with sRGB-distributed 8-bit precision in
    // the darks). SceneKit then samples that sRGB texture as `background.
    // contents`, decodes it back to linear, and applies its OWN output sRGB
    // encode for the bgra8Unorm drawable — one correct round-trip. A manual
    // `pow(1/2.2)` here used to double-encode against SceneKit's pass, which
    // washed out blacks and flattened chroma (issue: Illuminatorama colours
    // read desaturated vs the SCN-native `+` scenes).
    return float4(mapped, 1.0);
}

// ── Phase 3.1 / 3.3: DDGI Trace kernel ───────────────────────────────────────
//
// Dispatch: (raysPerProbe, probeCount, 1). Each thread fires one analytic ray
// from one probe and writes a DDGIRayRecord to outRays[probeIdx*raysPerProbe+rayIdx].
//
// Rays miss → sky HDR sample stored.
// Rays hit  → Lambertian direct light + emission at the hit surface, *plus*
//             (when `ddgi.twoBounceEnabled != 0`) one bounce of indirect
//             light read from the previous-frame irradiance atlas. The hit
//             point's hemisphere-integrated incoming irradiance × albedo / π
//             is the standard Lambertian re-emission; the DDGI atlas already
//             stores `irradiance / π` (the update kernel divides by Σ
//             weights, and the lighting kernel multiplies straight onto
//             albedo without further dividing by π), so we multiply by
//             albedo directly.

kernel void illumi_ddgi_trace(
    device DDGIRayRecord*               outRays      [[buffer(0)]],
    constant DDGIUniforms&              ddgi         [[buffer(1)]],
    const device DDGIInstanceData*      instances    [[buffer(2)]],
    const device DDGIPointEmitter*      emitters     [[buffer(3)]],
    texture2d<float, access::sample>    skyEquirect  [[texture(0)]],
    // Phase 3.3 — previous-frame atlases for the second bounce. The host
    // ping-pongs irradiance + depth atlases, so the atlas slots bound here
    // hold the previous frame's writes (read-only this frame). When
    // `twoBounceEnabled == 0` they're still bound but never sampled.
    texture2d<half,  access::sample>    irrAtlasPrev [[texture(1)]],
    texture2d<half,  access::sample>    depAtlasPrev [[texture(2)]],
    uint2                               gid          [[thread_position_in_grid]]
) {
    uint rayIdx   = gid.x;
    uint probeIdx = gid.y;
    uint probeCount = ddgi.gridDimsX * ddgi.gridDimsY * ddgi.gridDimsZ;
    if (rayIdx >= ddgi.raysPerProbe || probeIdx >= probeCount) return;

    float3 probePos = ddgiProbePos(probeIdx, ddgi);

    // Rotate fibonacci sample by a per-probe golden-angle offset to break up
    // the repeating lattice pattern across adjacent probes.
    float3 fibDir = ddgiSphericalFibonacci(rayIdx, ddgi.raysPerProbe);
    float  angle  = float(probeIdx) * 2.399963229f;  // ~golden angle in rad
    float  sa = sin(angle), ca = cos(angle);
    float3 rayDir = float3(fibDir.x * ca - fibDir.y * sa,
                           fibDir.x * sa + fibDir.y * ca,
                           fibDir.z);

    float  bestDist   = 1.0e10f;
    float3 bestNormal = float3(0.0f, 1.0f, 0.0f);
    uint   hitInst    = 0xFFFFFFFFu;

    for (uint i = 0; i < ddgi.instanceCount; ++i) {
        DDGIInstanceData inst = instances[i];
        float3 lo = (inst.invModelMatrix * float4(probePos, 1.0f)).xyz;
        float3 ld = (inst.invModelMatrix * float4(rayDir,   0.0f)).xyz;
        float3 localNormal = float3(0.0f);
        float  t = -1.0f;
        // Phase 2.6 — meshKind=3 marks a host-extracted custom mesh with
        // no analytic intersection available. DDGI silently treats it as
        // "no geometry here" until Phase 4 brings a real BVH / `MTLAccelerationStructure`
        // for trace-against-arbitrary-meshes. Direct lighting, IBL, and SSR all
        // still operate on the mesh normally — only DDGI's second bounce skips
        // it. The else branch is now guarded so the unit-ground intersection
        // doesn't fire for unknown meshKinds.
        if (inst.meshKind == 0u) {
            t = rayUnitBox(lo, ld, localNormal);
        } else if (inst.meshKind == 1u) {
            t = rayUnitSphere(lo, ld, localNormal);
        } else if (inst.meshKind == 2u) {
            t = rayUnitGround(lo, ld, localNormal);
        }
        // else: meshKind ≥ 3 — leave t = -1 so the (t > 1e-4f) gate skips.
        if (t > 1e-4f && t < bestDist) {
            bestDist   = t;
            bestNormal = normalize((inst.normalMatrix * float4(localNormal, 0.0f)).xyz);
            hitInst    = i;
        }
    }

    uint outIdx = probeIdx * ddgi.raysPerProbe + rayIdx;
    if (hitInst == 0xFFFFFFFFu) {
        float3 sky = sampleSkyEquirect(skyEquirect, rayDir);

        // Emitters visible along this miss ray: treat each as a small sphere.
        // Rays that pass within the emitter's radius pick up its radiance,
        // so probes looking toward a glowing particle field accumulate it.
        for (uint e = 0; e < ddgi.emitterCount; ++e) {
            float3 toE   = emitters[e].position - probePos;
            float  tE    = dot(toE, rayDir);
            if (tE < 0.0f) continue;
            float3 perp  = toE - rayDir * tE;
            float  perpD = length(perp);
            float  r     = max(emitters[e].radius, 0.001f);
            float  atten = max(0.0f, 1.0f - perpD / r);
            atten       *= atten;
            sky          += emitters[e].color * atten;
        }

        outRays[outIdx].dirAndDist = float4(rayDir, -1.0f);
        outRays[outIdx].radiance   = float4(sky, 0.0f);
    } else {
        DDGIInstanceData inst = instances[hitInst];
        float NdotL   = max(0.0f, dot(bestNormal, ddgi.directionalLightDir));
        float3 directLight = inst.albedo * ddgi.directionalLightColor * NdotL
                           * (1.0f - inst.metallic);

        // Phase 3.3 — second bounce. Sample the previous-frame irradiance
        // atlas at the hit point along the hit normal, then re-emit as
        // Lambertian indirect: L_out = albedo · irradiance · (1 - metallic).
        // The atlas read uses the existing trilinear+Chebyshev sampler so
        // probes behind the wall don't bleed onto an exterior hit.
        // Bias the worldPos a hair off the surface so depth-atlas
        // visibility tests don't classify the hit point as "inside" its
        // own surface.
        float3 indirectBounce = float3(0.0f);
        if (ddgi.twoBounceEnabled != 0u) {
            float3 hitWorld = probePos + rayDir * bestDist;
            float3 hitBiased = hitWorld + bestNormal * 0.02f;
            float3 prevIrr = sampleDDGIIrradiance(
                hitBiased, bestNormal, irrAtlasPrev, depAtlasPrev, ddgi);
            indirectBounce = inst.albedo * prevIrr * (1.0f - inst.metallic);
        }

        // Emitter point-light contribution at the hit surface. Each emitter
        // acts as a point light: Lambertian re-emission proportional to NdotL
        // toward the emitter and quadratic attenuation within its radius.
        float3 hitWorld = probePos + rayDir * bestDist;
        float3 emitterLight = float3(0.0f);
        for (uint e = 0; e < ddgi.emitterCount; ++e) {
            float3 toE = emitters[e].position - hitWorld;
            float  d   = length(toE);
            float  r   = max(emitters[e].radius, 0.001f);
            float  atten = max(0.0f, 1.0f - d / r);
            atten       *= atten;
            if (atten < 0.001f) continue;
            float NdotE = max(0.0f, dot(bestNormal, toE / max(d, 0.001f)));
            emitterLight += emitters[e].color * atten * NdotE
                          * inst.albedo * (1.0f - inst.metallic);
        }

        outRays[outIdx].dirAndDist = float4(rayDir, bestDist);
        outRays[outIdx].radiance   = float4(
            directLight + indirectBounce + inst.emission + emitterLight, 1.0f);
    }
}

// ── Phase 3.1: DDGI Update irradiance atlas ───────────────────────────────────
//
// Dispatch: (irrTileSize, irrTileSize, probeCount). For each interior texel of
// each probe's octahedral tile, integrates incoming ray radiance weighted by
// cos(texelDir, rayDir), then exponential-moving-averages into the atlas.

kernel void illumi_ddgi_update_irradiance(
    const device DDGIRayRecord*             rays     [[buffer(0)]],
    constant DDGIUniforms&                  ddgi     [[buffer(1)]],
    texture2d<half, access::read_write>     irrAtlas [[texture(0)]],
    uint3                                   gid      [[thread_position_in_grid]]
) {
    uint tileX    = gid.x;
    uint tileY    = gid.y;
    uint probeIdx = gid.z;
    uint probeCount = ddgi.gridDimsX * ddgi.gridDimsY * ddgi.gridDimsZ;
    if (tileX >= ddgi.irrTileSize || tileY >= ddgi.irrTileSize ||
        probeIdx >= probeCount) return;

    float2 octNorm  = (float2(tileX, tileY) + 0.5f) / float(ddgi.irrTileSize) * 2.0f - 1.0f;
    float3 texelDir = ddgiOctDecode(octNorm);

    float3 irradiance  = float3(0.0f);
    float  totalWeight = 0.0f;
    uint   base = probeIdx * ddgi.raysPerProbe;
    for (uint r = 0; r < ddgi.raysPerProbe; ++r) {
        DDGIRayRecord rec = rays[base + r];
        float3 rayDir = rec.dirAndDist.xyz;
        float  w = max(0.0f, dot(texelDir, rayDir));
        if (w > 0.0f) {
            irradiance  += float3(rec.radiance.xyz) * w;
            totalWeight += w;
        }
    }
    if (totalWeight > 1e-6f) irradiance /= totalWeight;

    uint2  coord    = ddgiIrrTexel(tileX, tileY, probeIdx, ddgi);
    float3 existing = float3(irrAtlas.read(coord).rgb);
    float3 blended  = mix(irradiance, existing, ddgi.hysteresis);
    irrAtlas.write(half4(half3(blended), 1.0h), coord);
}

// ── Phase 3.1: DDGI Update depth atlas ───────────────────────────────────────
//
// Dispatch: (depthTileSize, depthTileSize, probeCount). Stores (mean, mean²) of
// weighted hit distances per octahedral texel. The lighting kernel's Chebyshev
// visibility test reads both channels to bound the probability that a shaded
// point can be "seen" from the probe without obstruction.

kernel void illumi_ddgi_update_depth(
    const device DDGIRayRecord*             rays       [[buffer(0)]],
    constant DDGIUniforms&                  ddgi       [[buffer(1)]],
    texture2d<half, access::read_write>     depthAtlas [[texture(0)]],
    uint3                                   gid        [[thread_position_in_grid]]
) {
    uint tileX    = gid.x;
    uint tileY    = gid.y;
    uint probeIdx = gid.z;
    uint probeCount = ddgi.gridDimsX * ddgi.gridDimsY * ddgi.gridDimsZ;
    if (tileX >= ddgi.depthTileSize || tileY >= ddgi.depthTileSize ||
        probeIdx >= probeCount) return;

    float2 octNorm  = (float2(tileX, tileY) + 0.5f) / float(ddgi.depthTileSize) * 2.0f - 1.0f;
    float3 texelDir = ddgiOctDecode(octNorm);

    float sumD  = 0.0f;
    float sumD2 = 0.0f;
    float totalWeight = 0.0f;
    uint  base = probeIdx * ddgi.raysPerProbe;
    for (uint r = 0; r < ddgi.raysPerProbe; ++r) {
        DDGIRayRecord rec  = rays[base + r];
        float  dist = rec.dirAndDist.w;
        if (dist < 0.0f) continue;  // sky miss
        float3 rayDir = rec.dirAndDist.xyz;
        float  w = max(0.0f, dot(texelDir, rayDir));
        if (w > 0.0f) {
            sumD  += dist * w;
            sumD2 += dist * dist * w;
            totalWeight += w;
        }
    }
    float mean  = (totalWeight > 1e-6f) ? sumD  / totalWeight : 1.0e4f;
    float mean2 = (totalWeight > 1e-6f) ? sumD2 / totalWeight : 1.0e8f;

    uint2  coord    = ddgiDepthTexel(tileX, tileY, probeIdx, ddgi);
    float2 existing = float2(depthAtlas.read(coord).rg);
    float2 blended  = mix(float2(mean, mean2), existing, ddgi.depthHysteresis);
    depthAtlas.write(half4(half2(blended), 0.0h, 1.0h), coord);
}

// ── Transparent reflective glass pane (forward, alpha-blended) ────────────────
//
// One glass sheet rendered AFTER the opaque HDR composite and BEFORE TAA/bloom,
// so its reflections bloom + tonemap naturally. Reuses the Vertex / Instance /
// FrameUniforms structs and the equirect helpers above. Repurposed Instance
// fields (so the host needs no new upload path):
//   albedo     = glass tint (multiplies the reflection + a faint base sheen)
//   metallic   = reflectivity (0..1 strength of the env reflection)
//   roughness  = base alpha (min opacity head-on; edges add fresnel on top)
//   emission.x = fresnel power (higher → reflection only at grazing angles)
// Depth-TESTED (coins in front occlude the glass) but no depth WRITE; src-alpha
// blended over the lit pile already in the composite — the pile shows through.

// Per-glass-pass uniforms: the positionable key light (a controllable specular
// glint, since the forward glass pass otherwise only reflects the sky) plus the
// screen-space refraction knobs. xyz/w packed so the Swift struct round-trips
// without SIMD3 stride surprises. Stride 48 — matches GlassKeyLightUniform.
struct GlassKeyLight {
    float4 positionIntensity;   // xyz = world position, w = intensity
    float4 colorShininess;      // rgb = colour, w = shininess (glint tightness)
    uint   keyEnabled;
    uint   refractEnabled;      // screen-space refraction on/off
    float  ior;                 // index of refraction (glass ≈ 1.5)
    float  refractStrength;     // artistic scale on the screen-space UV offset
};

struct GlassVSOut {
    float4 clipPos [[position]];
    float3 worldPos;
    float3 worldNormal;
    // Per-instance index, flat-interpolated so the fragment shader can read
    // its own instance's material. `instance_id` is vertex-stage only, so we
    // forward it here — this is what lets the glass pass draw MANY panes
    // (orbiting lenses) in one instanced draw instead of a single sheet.
    uint   instanceID [[flat]];
};

vertex GlassVSOut illumi_glass_vs(
    uint                    vid       [[vertex_id]],
    uint                    iid       [[instance_id]],
    const device Vertex*    verts     [[buffer(0)]],
    constant FrameUniforms& frame     [[buffer(1)]],
    const device Instance*  instances [[buffer(2)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    GlassVSOut o;
    o.clipPos     = frame.viewProjection * worldP;
    o.worldPos    = worldP.xyz;
    o.worldNormal = (inst.normalMatrix * float4(v.normal, 0.0)).xyz;
    o.instanceID  = iid;
    return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// APPROACH B (later — GG3·17 "Robust Multiple Specular Reflections & Refractions",
// tracked on issue #57): true ray-traced refraction through the dielectric.
// Instead of the screen-space grab below, put the lens geometry in the RT TLAS
// and, per primary ray: refract at the FRONT surface (Snell, η = 1/IOR), march
// through the glass to the BACK surface, refract again on exit (or total-internal-
// reflect when past the critical angle), then continue to sky/scene. This gives
// physically-correct magnification/inversion, real lens-through-lens stacking, and
// caustics — none of which the screen-space approximation can do. It needs a
// dielectric BSDF + multi-bounce loop added to the RT kernel (which today only
// does diffuse GI + soft shadows) and the lenses moved from this forward pass into
// the RT scene representation. Big lift; keep the screen-space path as the
// realtime default and gate B behind its own toggle when built.
// ─────────────────────────────────────────────────────────────────────────────
fragment float4 illumi_glass_fs(
    GlassVSOut              in        [[stage_in]],
    constant FrameUniforms& frame     [[buffer(1)]],
    const device Instance*  instances [[buffer(2)]],
    constant GlassKeyLight& key       [[buffer(3)]],
    texture2d<float, access::sample> sky        [[texture(0)]],
    texture2d<float, access::sample> background [[texture(1)]]
) {
    Instance inst = instances[in.instanceID];
    float3 N = normalize(in.worldNormal);
    float3 V = normalize(frame.cameraWorldPos - in.worldPos);
    if (dot(N, V) < 0.0) N = -N;                 // two-sided: face the viewer
    float  ndv     = max(dot(N, V), 0.0);
    float  fresPow = max(inst.emission.x, 1.0);
    float  fresnel = pow(1.0 - ndv, fresPow);    // 0 head-on → 1 grazing
    float3 R       = reflect(-V, N);
    float3 refl    = sampleSkyEquirect(sky, R) * inst.metallic;
    float3 tint    = inst.albedo;
    // Visible glass over a DARK interior: the env reflection alone is too dim to
    // read, so add a FRESNEL-WEIGHTED tint sheen — the pane's grazing edges catch a
    // cool highlight while it stays mostly see-through head-on. Sells "there's glass
    // here" without occluding the pile.
    float  sheen   = 0.03 + fresnel * 0.55;

    float3 color;
    float  alpha;
    if (key.refractEnabled != 0u) {
        // ── APPROACH A — screen-space refraction (GG2·19, issue #47) ──────────
        // Sample the pre-glass composite (sky + opaque scene, blitted into
        // `background` before this pass) along the REFRACTED view ray. We perturb
        // the screen UV by the refracted direction expressed in view space; the
        // lens's analytic concave normals make the offset DIVERGE outward, so the
        // background reads minified/upright — a real biconcave (diverging) lens,
        // not a fake warp. With refraction on the lens is opaque: the background
        // is already folded in here, so alpha = 1 (alpha-blending it again would
        // double-count the background).
        //
        // Limitation (the reason Approach B exists): the grab is frozen BEFORE the
        // glass pass, so a lens does not refract OTHER lenses behind it — only the
        // sky/opaque scene. Overlapping lenses simply occlude front-over-back.
        constexpr sampler bgS(filter::linear, address::clamp_to_edge);
        float2 screenUV = in.clipPos.xy /
                          float2(background.get_width(), background.get_height());
        float  eta = 1.0 / max(1.0, key.ior);
        float3 T   = refract(-V, N, eta);                  // world-space refracted ray
        float3 Tv  = (frame.view * float4(T, 0.0)).xyz;    // → view space
        float2 uvOff = Tv.xy * key.refractStrength;
        uvOff.y = -uvOff.y;                                // view +y is up, UV +y is down
        float2 ruv = clamp(screenUV + uvOff, 0.0, 1.0);
        float3 bg  = background.sample(bgS, ruv).rgb;
        float3 transmit = bg * tint;                       // glass tints transmitted light
        // Fresnel mix: see-through (refracted) head-on, sky reflection at grazing.
        color = mix(transmit, refl + tint * sheen, fresnel);
        alpha = 1.0;
    } else {
        // Translucent reflective glass (no refraction): see-through pane that
        // catches the sky + fresnel sheen, alpha-blended over the scene.
        color = refl * tint + tint * sheen;
        alpha = clamp(inst.roughness + fresnel * 0.85, 0.0, 1.0);
    }

    // ── Positionable key light: a Blinn specular glint that sweeps across the
    // lens as it tumbles. Pure highlight (no diffuse — glass has none); it both
    // brightens the colour and locally firms up alpha so the glint reads as a
    // crisp hotspot rather than a faint wash.
    if (key.keyEnabled != 0u) {
        float3 Lp        = key.positionIntensity.xyz;
        float  intensity = key.positionIntensity.w;
        float3 kColor    = key.colorShininess.rgb;
        float  shininess = max(1.0, key.colorShininess.w);
        float3 L = normalize(Lp - in.worldPos);
        float3 H = normalize(L + V);
        float  spec = pow(max(dot(N, H), 0.0), shininess);
        // Tint the glint by the lens colour so a coloured lens (e.g. Clock mode's
        // hour/minute/second hands) keeps its hue under the key light at night,
        // instead of washing to the light's own colour. Clear glass (tint ≈ white)
        // is unaffected.
        float3 glint = kColor * tint * (intensity * spec);
        color += glint;
        alpha  = clamp(alpha + spec * intensity * 0.5, 0.0, 1.0);
    }

    return float4(color, alpha);                 // straight (non-premultiplied) src-alpha blend
}

// ── GPU-resident instance write (opt-in; IlluminatoramaRenderer.onEncodeGPUInstances) ──
//
// Writes full `Instance` structs straight into the renderer's instance buffer
// from a GPU-resident transform buffer (e.g. EggMotion's output), computing a
// 1/r² rail-bulb irradiance tint as the emission term — so per-egg transform,
// material and lighting tint are produced entirely on the GPU with no CPU
// readback / rebuild. One thread per instance; it overwrites `groupStart + i`.
//
// The transform is the 3×4 column layout EggMotion.metal writes (col0..2 =
// rotation·scale columns, col3 = world position). Eggs are uniform-scaled
// rotations, so normalMatrix = transpose(inverse(upper3×3)) = basis / scale²
// (scale = column length) — matching IlluminatoramaInstance.normalMatrix.

struct InstWriteXform { float4 c0; float4 c1; float4 c2; float4 c3; };

struct EggInstanceWriteUniforms {
    uint  count;        // egg count (grid size)
    uint  bulbCount;    // bulbs to sum for the irradiance tint
    uint  groupStart;   // first slot of this mesh group in the instance buffer
    float minDistSq;    // 1/r² floor (scaled-world units²)
    float tintScale;    // raw irradiance → HDR emission gain
    float tintCap;      // per-egg emission ceiling
    float metallic;     // egg material metalness
    float roughness;    // egg material roughness
};

kernel void eggs_write_instances(
    constant EggInstanceWriteUniforms& U   [[buffer(0)]],
    device const InstWriteXform*  xforms   [[buffer(1)]],   // motion output (completed buffer)
    device const float4*          colors   [[buffer(2)]],   // per-egg albedo (xyz)
    device const float4*          bulbPos  [[buffer(3)]],   // bulb world positions (xyz)
    device const float4*          bulbCol  [[buffer(4)]],   // bulb LUT colours (xyz)
    device Instance*              outInst  [[buffer(5)]],   // renderer instance buffer
    uint i                                 [[thread_position_in_grid]]
) {
    if (i >= U.count) return;

    InstWriteXform M = xforms[i];
    float4x4 model = float4x4(M.c0, M.c1, M.c2, M.c3);
    float3x3 basis = float3x3(M.c0.xyz, M.c1.xyz, M.c2.xyz);
    float  s     = max(1e-6, length(M.c0.xyz));
    float  invS2 = 1.0 / (s * s);
    float4x4 normalMat = float4x4(
        float4(basis[0] * invS2, 0.0),
        float4(basis[1] * invS2, 0.0),
        float4(basis[2] * invS2, 0.0),
        float4(0.0, 0.0, 0.0, 1.0));

    // Rail-bulb irradiance tint: 1/r² sum over every bulb's LUT colour at the
    // egg's world position, normalised to a saturated hue and capped (mirrors
    // the former CPU EggsControllerUltra.updateEggRailTints).
    float3 p   = M.c3.xyz;
    float3 sum = float3(0.0);
    for (uint b = 0; b < U.bulbCount; ++b) {
        float3 d  = p - bulbPos[b].xyz;
        float  r2 = max(U.minDistSq, dot(d, d));
        sum += bulbCol[b].xyz / r2;
    }
    float  mag  = max(sum.x, max(sum.y, sum.z));
    float3 tint = float3(0.0);
    if (mag > 1e-4) {
        tint = (sum / mag) * min(mag * U.tintScale, U.tintCap);
    }

    Instance inst;
    inst.modelMatrix           = model;
    inst.normalMatrix          = normalMat;
    inst.albedo                = colors[i].xyz;
    inst.metallic              = U.metallic;
    inst.emission              = tint;
    inst.roughness             = U.roughness;
    inst.albedoTextureSlice    = -1;
    inst.metallicTextureSlice  = -1;
    inst.roughnessTextureSlice = -1;
    inst.normalTextureSlice    = -1;
    inst.emissionTextureSlice  = -1;
    inst.emissionIntensity     = 1.0;
    inst._padSlice1            = 0;

    outInst[U.groupStart + i] = inst;
}

// ── Bulb sphere instances (GPU-resident) ─────────────────────────────────────
// One thread per bulb: static position + the per-tick LUT colour → an emissive
// sphere `Instance`. Dark bulbs (LUT colour ≈ 0) collapse to a scale-0 transform
// so they vanish without leaving a black dot — the instance count stays fixed
// (TAA-stable), unlike the old CPU path that skipped dark bulbs.

struct BulbInstanceWriteUniforms {
    uint  bulbCount;
    uint  groupStart;
    float displayRadius;
    float emissionScale;
    float darkThreshold;
    float _pad0; float _pad1; float _pad2;
};

kernel void bulbs_write_instances(
    constant BulbInstanceWriteUniforms& U [[buffer(0)]],
    device const float4* bulbPos          [[buffer(1)]],
    device const float4* bulbCol          [[buffer(2)]],
    device Instance*     outInst          [[buffer(3)]],
    uint i                                [[thread_position_in_grid]]
) {
    if (i >= U.bulbCount) return;
    float3 c   = bulbCol[i].xyz;
    float3 p   = bulbPos[i].xyz;
    bool   lit = length(c) >= U.darkThreshold;
    float  s   = lit ? U.displayRadius : 0.0;          // dark → scale 0 → invisible
    float  invS = (s > 1e-6) ? (1.0 / s) : 0.0;

    Instance inst;
    inst.modelMatrix  = float4x4(float4(s,0,0,0), float4(0,s,0,0),
                                 float4(0,0,s,0), float4(p, 1.0));
    inst.normalMatrix = float4x4(float4(invS,0,0,0), float4(0,invS,0,0),
                                 float4(0,0,invS,0), float4(0,0,0,1));
    inst.albedo                = float3(0.0);
    inst.metallic              = 0.0;
    inst.emission              = lit ? (c * U.emissionScale) : float3(0.0);
    inst.roughness             = 0.0;
    inst.albedoTextureSlice    = -1;
    inst.metallicTextureSlice  = -1;
    inst.roughnessTextureSlice = -1;
    inst.normalTextureSlice    = -1;
    inst.emissionTextureSlice  = -1;
    inst.emissionIntensity     = 1.0;
    inst._padSlice1            = 0;

    outInst[U.groupStart + i] = inst;
}

// ── Glow strip emission (GPU-resident) ───────────────────────────────────────
// One thread per glow-strip slice. Each slice is a single-instance mesh kind
// (split off at the 60k-vert / UInt16-index limit) and the slices are laid out
// CONTIGUOUSLY in the instance buffer, so thread `i` writes `groupStart + i`.
// The emission is sampled from the same per-bulb LUT the bulb spheres use,
// spread across the network (slot = i/sliceCount · bulbCount) so a rainbow
// reads as a colour gradient along the rails — and so switching the pattern /
// hue visibly drives the dominant rail element, not just the bulbs. The strip
// geometry is already world-space, so the model + normal matrices are identity.

struct GlowInstanceWriteUniforms {
    uint  sliceCount;
    uint  bulbCount;
    uint  groupStart;
    float bloomScale;     // LUT colour multiplier; 0 when the rails aren't glowing
};

kernel void glow_write_instances(
    constant GlowInstanceWriteUniforms& U [[buffer(0)]],
    device const float4* bulbCol          [[buffer(1)]],
    device Instance*     outInst          [[buffer(2)]],
    uint i                                [[thread_position_in_grid]]
) {
    if (i >= U.sliceCount) return;
    uint slot = (U.sliceCount > 1 && U.bulbCount > 0)
        ? min(U.bulbCount - 1u, uint(float(i) / float(U.sliceCount) * float(U.bulbCount)))
        : 0u;
    float3 c = bulbCol[slot].xyz * U.bloomScale;

    // Glow strip vertices are world-space (identity model matrix). When the
    // rails aren't glowing (bloomScale 0) collapse the model matrix to the
    // origin so the full-ring shell draws nothing — otherwise an opaque
    // zero-emission tube would hide the chrome rail it wraps. (A partial arc
    // could get away with emission 0; a full ring can't.)
    bool glowing = U.bloomScale > 0.0;
    float g = glowing ? 1.0 : 0.0;

    Instance inst;
    inst.modelMatrix           = float4x4(float4(g,0,0,0), float4(0,g,0,0),
                                          float4(0,0,g,0), float4(0,0,0,1));
    inst.normalMatrix          = float4x4(float4(1,0,0,0), float4(0,1,0,0),
                                          float4(0,0,1,0), float4(0,0,0,1));
    inst.albedo                = float3(0.0);
    inst.metallic              = 0.0;
    inst.emission              = c;
    inst.roughness             = 0.0;
    inst.albedoTextureSlice    = -1;
    inst.metallicTextureSlice  = -1;
    inst.roughnessTextureSlice = -1;
    inst.normalTextureSlice    = -1;
    inst.emissionTextureSlice  = -1;
    inst.emissionIntensity     = 1.0;
    inst._padSlice1            = 0;

    outInst[U.groupStart + i] = inst;
}

// ── Camera-near bulb point lights (GPU cull) ─────────────────────────────────
// One thread per bulb: a lit bulb within cullRadius of the camera atomically
// claims a point-light slot (capped at maxLights) so the chrome rails pick up
// the marquee colour as real coloured highlights. Unclaimed slots keep the
// zeroed CPU placeholders → skipped by the deferred loop's radius cutoff.
// `counter` is reset to 0 (blit fill) before this dispatch.

struct BulbCullUniforms {
    float4 cameraCull;    // xyz = camera world pos, w = cullRadius²
    uint   bulbCount;
    uint   maxLights;
    uint   lightOffset;   // first reserved point-light slot
    float  darkThreshold;
    float  lightRadius;
    float  gain;
    float  _pad0; float _pad1;
};

kernel void bulbs_write_pointlights(
    constant BulbCullUniforms& U   [[buffer(0)]],
    device const float4* bulbPos   [[buffer(1)]],
    device const float4* bulbCol   [[buffer(2)]],
    device atomic_uint*  counter   [[buffer(3)]],
    device PointLight*   outLights [[buffer(4)]],
    uint i                         [[thread_position_in_grid]]
) {
    if (i >= U.bulbCount) return;
    float3 c = bulbCol[i].xyz;
    if (length(c) < U.darkThreshold) return;
    float3 d = bulbPos[i].xyz - U.cameraCull.xyz;
    if (dot(d, d) > U.cameraCull.w) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot >= U.maxLights) return;
    PointLight pl;
    pl.position = bulbPos[i].xyz;
    pl.radius   = U.lightRadius;
    pl.color    = c * U.gain;
    pl._pad     = 0.0;
    outLights[U.lightOffset + slot] = pl;
}

// ── Highlight outline mask pass ───────────────────────────────────────────────
//
// One render pipeline feeds a separable max-filter dilation + composite. Every
// highlighted element (selected OR hovered) supplies a SOLID bounding-box proxy
// to the mask buffer — never its detailed mesh. A box is a closed solid, so its
// screen-space mask has no internal holes; dilate − original then yields a clean
// outer ring with no internal edges and no fill. (Rasterizing the real mesh —
// open sofa frames, see-through bookshelf bays — produced internal rings and,
// once the small gaps merged under dilation, a full-object wash.)
//
//   illumi_selection_box_vs/fs   — draws boxes whose inst.highlight == wantMode.
//
// Then two compute passes:
//   illumi_selection_dilate_h/v  — separable max-filter dilation by `radius` px.
//   illumi_selection_composite   — ring = dilated − original; additive HDR blend.
//
// The pass runs once per mode: wantMode 1 (selected, blue) then 2 (hover, yellow).

struct SelMaskVSOut {
    float4 position [[position]];
};

// Mask boxes, filtered to the mode being composited this invocation: an instance
// whose highlight tag differs from wantMode is clipped (NDC z > 1 → zero fragments).
vertex SelMaskVSOut illumi_selection_box_vs(
    uint                       vid       [[vertex_id]],
    uint                       iid       [[instance_id]],
    const device Vertex*       verts     [[buffer(0)]],
    constant FrameUniforms&    frame     [[buffer(1)]],
    const device Instance*     instances [[buffer(2)]],
    constant int&              wantMode  [[buffer(3)]]
) {
    if (instances[iid].highlight != wantMode) {
        return { float4(2, 2, 2, 1) };   // outside clip → zero fragments
    }
    float4 worldP = instances[iid].modelMatrix * float4(verts[vid].position, 1.0);
    return { frame.viewProjection * worldP };
}

fragment float illumi_selection_mask_fs(SelMaskVSOut in [[stage_in]]) {
    return 1.0;
}

// Horizontal max-filter dilation pass (reads selectionMask, writes dilateH).
kernel void illumi_selection_dilate_h(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int&                   radius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(inTex.get_width()), h = int(inTex.get_height());
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float v = 0;
    for (int dx = -radius; dx <= radius; dx++) {
        int px = clamp(int(gid.x) + dx, 0, w - 1);
        v = max(v, inTex.read(uint2(px, gid.y)).r);
    }
    outTex.write(float4(v, 0, 0, 0), gid);
}

// Vertical max-filter dilation pass (reads dilateH, writes dilatedFinal).
kernel void illumi_selection_dilate_v(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int&                   radius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(inTex.get_width()), h = int(inTex.get_height());
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float v = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        int py = clamp(int(gid.y) + dy, 0, h - 1);
        v = max(v, inTex.read(uint2(gid.x, py)).r);
    }
    outTex.write(float4(v, 0, 0, 0), gid);
}

// Composites ring = (dilated − original) additively into the HDR texture.
struct SelectionOutlineParams {
    float4 colorIntensity;   // xyz = glow color, w = intensity (push past bloom threshold)
    int    width;
    int    height;
    int    _pad0;
    int    _pad1;
};

kernel void illumi_selection_composite(
    texture2d<float, access::read>       maskTex    [[texture(0)]],
    texture2d<float, access::read>       dilatedTex [[texture(1)]],
    texture2d<float, access::read_write> hdrTex     [[texture(2)]],
    constant SelectionOutlineParams&     p          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (int(gid.x) >= p.width || int(gid.y) >= p.height) return;
    float orig   = maskTex.read(gid).r;
    float dilated = dilatedTex.read(gid).r;
    float ring   = saturate(dilated - orig);
    float4 hdr   = hdrTex.read(gid);
    float3 glow  = ring * p.colorIntensity.xyz * p.colorIntensity.w;
    hdrTex.write(float4(hdr.rgb + glow, hdr.a), gid);
}
