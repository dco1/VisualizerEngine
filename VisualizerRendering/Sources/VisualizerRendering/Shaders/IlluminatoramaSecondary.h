#pragma once

// ── ILLUMINATORAMA — ONE SURFACE SHADER FOR EVERY SECONDARY RAY ───────────────
//
// A "secondary ray" is any ray fired from an already-shaded point: a refraction
// or reflection ray out of the AAA glass pass, a glossy reflection ray out of the
// deferred TLAS lighting kernel, a GI bounce. When such a ray lands on an opaque
// triangle, SOMETHING has to work out how bright that triangle is — and every
// path in this engine used to answer that question with its own private copy of
// the answer.
//
// They diverged, exactly as duplicated shading always does. The glass pass and
// the deferred reflection path both re-shaded a hit with the instance's MEAN
// ALBEDO — no texture tap, so a plank floor, a tiled wall and a rug behind a pane
// each collapsed to one flat colour — plus a bare `albedo * skyAmbient` fill at
// exterior strength and no local lights at all. The glass pass was fixed first
// (measured on the real document: local detail retained through a pane went
// 0.53 → 0.83 of the same view with the glazing removed, of which the texture tap
// is 0.27). Copying that fix into the reflection path would have produced two
// near-identical shading bodies that would drift apart again on the next feature.
//
// So the body lives here, ONCE, and every path calls it:
//
//     shadeSecondarySurface(isect, accel, hit, params, scene, irrCube, atlas, seed)
//
//   = textured albedo                     (`secondaryAlbedo`)
//   × [ sky IBL + ambient, interior-split (`secondaryIndirectFill`)
//     + local point/spot lights           (`secondaryLocalLightFill`) ]
//   + Lambert direct sun with its shadow ray.
//
// These are the same terms, at the same strengths, the deferred lighting pass
// applies to a directly-visible surface — which is the whole contract: the world
// seen through a pane, or in a reflection, must be shaded like the world beside
// it. A future secondary ray gets all of it by construction.
//
// Metal has NO cross-file linkage for `static inline`, so a shared header is the
// only mechanism available; the alternative (`#include` of a .metal, or a fourth
// hand-kept copy) is strictly worse. Callers include this AFTER declaring nothing
// of their own by these names — the mirror structs, RNG, cone and surface-cache
// helpers that used to be duplicated per file now live here too, so "keep in
// lockstep with the sibling kernel" stops being a maintenance instruction.
//
// The intersector type is a TEMPLATE parameter: the glass pass traces
// `intersector<triangle_data, instancing>` and the deferred kernel traces
// `intersector<triangle_data, instancing, curve_data>`. Both instantiate the same
// source.

#include <metal_stdlib>
#include <metal_raytracing>
#include "IlluminatoramaNightSky.h"

using namespace metal;
using namespace raytracing;

// ── Mirror structs (field-for-field with the Swift side) ─────────────────────

/// Compact per-instance RT data; grouped order == TLAS `instance_id`. `nrm0..2`
/// are the columns of the 3×3 normal matrix, `nrm0.w` the albedo-atlas slice PLUS
/// ONE (so a memset-0 default decodes to −1 = untextured), `nrm1.w` the bit-cast
/// light-layer mask, `albedoTriBase.xyz` the instance's mean albedo and
/// `albedoTriBase.w` its base offset into the concatenated `objNormal`/`objUV`.
///
/// `emissionPad.xyz` is the instance's SELF-LIT radiance. It is here because a
/// whole class of surfaces in this engine are lit by a G-buffer emission term and
/// by nothing else — a lamp shade, a TV panel, a night window glow — and a
/// secondary ray that ignores emission renders every one of them dim or absent.
/// Seen through a window, the lamp shade lighting the room simply went missing.
/// Note this is the emission SCALAR only: a texture-driven glow
/// (`emissionTextureSlice`) still needs its atlas tap, which no secondary path
/// does yet.
///
/// Mirror of Swift `IlluminatoramaRenderer.RTInstanceData` — and of the copy in
/// IlluminatoramaCaustics.metal, which reads the same buffer.
struct RTInstanceData {
    float4 nrm0; float4 nrm1; float4 nrm2;
    float4 albedoTriBase;
    float4 emissionPad;      // xyz = emission radiance, w reserved
};

/// Surface-cache card material. Mirror of `SurfCard` in
/// IlluminatoramaSurfaceCache.metal — the full layout is needed so `cards[card]`
/// strides correctly even though only albedo/emission/normal.w are read here.
struct SurfCard {
    float4 origin; float4 uAxis; float4 vAxis; float4 normal;
    float4 albedo; float4 emission;
    float4 originB; float4 uAxisB; float4 vAxisB; float4 normalB;
    float4 albedoB; float4 emissionB;
};

/// Per-mesh card-frame UVs baked into the BLAS `primitive_data` (#60 item 6).
/// Mirror of Swift `IlluminatoramaRenderer.IlluminatoramaPrimUV` (3×float2, 24 B).
struct PrimUV { float2 uvA; float2 uvB; float2 uvC; };

/// Local-light currency. Mirrors `PointLight` / `SpotLight` in Illuminatorama.metal
/// and the Swift `IlluminatoramaPointLight` / `IlluminatoramaSpotLight`.
struct RTPointLight {
    float3 position;  float radius;
    float3 color;     uint  layerMask;
    uint   castsShadow; int shadowCubeIndex; int _pad0; int _pad1;
};
struct RTSpotLight {
    float3   position;   float innerCone;
    float3   direction;  float outerCone;
    float3   color;      float radius;
    float4x4 shadowMatrix;
    int      shadowSliceIndex;
    uint     layerMask;
    int      _pad1; int _pad2;
};

// ── RNG + sampling ───────────────────────────────────────────────────────────

static inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline float rnd(thread uint& seed) { seed = pcgHash(seed); return float(seed) * (1.0 / 4294967296.0); }

/// Orthonormal basis around `n` (Duff et al., branchless).
static inline void onb(float3 n, thread float3& t, thread float3& b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0; float a = -1.0 / (s + n.z); float d = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}
/// Uniform direction inside a cone of half-angle `theta` about `dir`.
/// `theta == 0` returns `dir` exactly (a hard ray), so a caller that wants no
/// softening does not need a separate branch.
static inline float3 coneSample(float3 dir, float theta, float u1, float u2) {
    float cosT = mix(cos(theta), 1.0, u1); float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 2.0 * M_PI_F * u2; float3 t, b; onb(dir, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + dir * cosT);
}
static inline float3 cosineSample(float3 n, float u1, float u2) {
    float r = sqrt(u1); float phi = 2.0 * M_PI_F * u2; float3 t, b; onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}
static inline float2 dirToEquirectUV(float3 d) {
    return float2(atan2(d.z, d.x) * (1.0 / (2.0 * M_PI_F)) + 0.5,
                  acos(clamp(d.y, -1.0, 1.0)) * (1.0 / M_PI_F));
}
/// What a secondary ray sees when it escapes to the sky: the equirect dome PLUS the
/// analytic celestials.
///
/// The celestials are not optional decoration here — the dome is deliberately baked
/// WITHOUT stars or moon (`Params.celestialsInDome = false`, because dome texels
/// magnify a star into a blob), so a path that samples only the dome renders the night
/// sky as flat black. That is exactly what a window used to do: standing in a room at
/// night, the sky above the roofline carried the whole star field (a primary ray, drawn
/// by Illuminatorama.metal's sky branch) and the sky through the glass was empty
/// (a refracted ray, drawn here). Sampling the dome and the celestials in ONE function
/// is what keeps the two views of the same sky the same sky.
///
/// `night` zeroed ⇒ `nightCelestials` returns 0 ⇒ byte-identical to the dome-only
/// sample, so every daytime scene and every host that never sets the night params is
/// unaffected. `pixAngle` sizes the star point-spread; a secondary ray uses the primary
/// pixel angle, which is correct to within the ray's own (small) divergence.
static inline float3 sampleSky(texture2d<float, access::sample> sky, float3 dir, float scale,
                               NightSkyParams night, float pixAngle) {
    constexpr sampler s(filter::linear, s_address::repeat, t_address::clamp_to_edge);
    float3 d = normalize(dir);
    return sky.sample(s, dirToEquirectUV(d)).rgb * scale
         + nightCelestials(d, night, pixAngle);
}

// ── The scatter cone: ONE definition of how wide it is, whether it is worth
//    sampling, and how many samples resolve it ──────────────────────────────────
//
// A rough surface scatters a transmitted/reflected ray inside a cone whose
// half-angle grows with roughness². Below a certain width that cone cannot frost
// or blur ANYTHING — it displaces the image by a fraction of a pixel — but a
// stochastic sample still pays the displacement as per-pixel NOISE. Architectural
// glazing lives exactly there: a polished pane at roughness 0.02 gets a 5.6e-4 rad
// cone, about a third of a pixel at a normal lens and frame size, and bought
// visibly ragged silhouettes seen through the glass with it (measured: 0.21 px of
// excess edge jitter vs the same view with the pane removed, and +36 % high-
// frequency speckle on textured surfaces behind it).
//
// So: skip the cone entirely when it is sub-pixel, and when it IS wide enough to
// see, sample it enough times to resolve it. Both questions are answered from the
// same number. The sample count used to be `uint(roughness * k + 0.5)`, which
// TRUNCATES to a single sample for every roughness below 1/k — i.e. for every
// surface that could actually frost. A cone worth tracing is a cone worth
// averaging.
//
// The two width coefficients differ by path and that is deliberate, so they sit
// side by side rather than in two files: dielectric transmission spreads slightly
// wider than the GGX α the deferred glossy lobe is harmonised to (the soup kernel
// in IlluminatoramaRT.metal uses the same 1.2).
constant float kSecondaryMinConeRad = 1.0e-3;   // ≈0.06°, about half a pixel at a normal lens
constant float kGlassConeK          = 1.4;      // glass refraction / reflection
constant float kReflConeK           = 1.2;      // deferred glossy reflection (GGX α)

static inline float secondaryConeRad(float roughness, float k) { return roughness * roughness * k; }
static inline bool  secondaryConeVisible(float roughness, float k) {
    return secondaryConeRad(roughness, k) > kSecondaryMinConeRad;
}
/// Stochastic samples to average for a cone of this width. Exactly 1 when the cone
/// is sub-pixel (and then nothing jitters, so the one sample is deterministic).
static inline uint secondaryConeSamples(float roughness, float k, float perRoughness, uint cap) {
    if (!secondaryConeVisible(roughness, k)) { return 1u; }
    return min(cap, 1u + uint(ceil(roughness * perRoughness)));
}

// ── Hit attributes ───────────────────────────────────────────────────────────

/// World normal of a triangle hit, from the instance's normal matrix.
static inline float3 hitWorldNormal(uint iid, uint prim,
                                    const device RTInstanceData* insts,
                                    const device float4* objNormal) {
    RTInstanceData d = insts[iid];
    uint triBase = uint(d.albedoTriBase.w);
    float3 nObj = objNormal[triBase + prim].xyz;
    float3x3 nm = float3x3(d.nrm0.xyz, d.nrm1.xyz, d.nrm2.xyz);
    float3 n = nm * nObj;
    float len = length(n);
    return len > 1e-8 ? n / len : float3(0.0, 1.0, 0.0);
}

/// The instance's light-layer bits (bit-cast into the normal matrix's dead
/// `nrm1.w` lane by `rebuildRTAccel`). Drives local-light masking and the interior
/// day-light separation, exactly as `fragLayer` does in the deferred kernel.
static inline uint secondaryLayerBits(uint iid, const device RTInstanceData* insts) {
    return as_type<uint>(insts[iid].nrm1.w);
}

/// The instance's self-lit radiance — see `RTInstanceData.emissionPad`. Added to a
/// hit's outgoing radiance directly (it is not reflected light, so it is NOT
/// modulated by albedo, N·L or visibility).
static inline float3 secondaryEmission(uint iid, const device RTInstanceData* insts) {
    return insts[iid].emissionPad.xyz;
}

// ── Surface cache ────────────────────────────────────────────────────────────

/// Outgoing radiance at a cached triangle hit: the atlas stores albedo-free
/// IRRADIANCE, so reconstruct L_out = albedo·irr + emission from the card (a card
/// seam is then not amplified by the albedo multiply). `primData` is the hit's
/// `res.primitive_data`: non-null ⇒ the BLAS carries baked per-mesh card UVs
/// (#60 item 6) and the `triUVa`/`triUVc` dependent loads (32 B/hit) are skipped.
static inline float3 sampleSurfCacheRT(
    texture2d<float, access::sample> atlas, uint prim, float2 bary,
    const device SurfCard* cards,
    const device uint* triCard, const device float4* triUVa, const device float4* triUVc,
    const device void* primData, const device float4* cardRect, uint atlasW, uint atlasH)
{
    uint card = triCard[prim];
    float4 rect = cardRect[card];          // (x, y, w, h) in atlas px
    float2 uvA, uvB, uvC;
    if (primData != nullptr) {
        const device PrimUV* pd = (const device PrimUV*)primData;
        uvA = pd->uvA; uvB = pd->uvB; uvC = pd->uvC;
    } else {
        float4 a = triUVa[prim], c = triUVc[prim];
        uvA = a.xy; uvB = a.zw; uvC = c.xy;
    }
    float w0 = 1.0 - bary.x - bary.y;
    float2 uv = saturate(w0 * uvA + bary.x * uvB + bary.y * uvC);
    float2 inset = 0.5 / max(float2(1.0), rect.zw);
    uv = clamp(uv, inset, 1.0 - inset);
    float2 px = rect.xy + uv * rect.zw;
    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 irr = atlas.sample(samp, px / float2(atlasW, atlasH)).rgb;
    SurfCard sc = cards[card];
    // Frame-B membership: frame A has uvA=(0,0) (sum 0), frame B has uvA=(1,1) (sum 2).
    bool useB = sc.normal.w > 0.5 && (uvA.x + uvA.y > 1.0);
    float3 albedo   = useB ? sc.albedoB.xyz   : sc.albedo.xyz;
    float3 emission = useB ? sc.emissionB.xyz : sc.emission.xyz;
    return albedo * irr + emission;
}

// ── What the shading needs, in one place per call site ───────────────────────

/// Everything that is NOT a texture or a buffer: each caller fills this from its
/// OWN uniform block, so the two uniform layouts can keep diverging while the
/// shading contract stays single-sourced.
///
/// Every field has a neutral value that reproduces the old, poorer behaviour, so a
/// caller can opt in term by term (which is what makes the terms independently
/// ablatable against a gate).
struct SecondaryShadeParams {
    float3 sunDir;          float sunSoftnessRad;   // 0 ⇒ a hard shadow ray
    float3 sunColor;        float skyIntensity;     // scales the irradiance cube
    float3 skyAmbient;      uint  shadowRays;       // 0 ⇒ no direct sun at the hit
    // Interior day-light separation — mirrors FrameUniforms.interiorMask /
    // interiorIBL* / interiorAmbient. 0 mask ⇒ exterior strength everywhere
    // (both factors exactly 1.0, an IEEE no-op).
    uint   interiorMask;
    float  interiorIBLUp; float interiorIBLSide; float interiorAmbient;
    // Albedo atlas: 0 ⇒ fall back to the instance's mean albedo.
    uint   albedoAtlasEnabled;
    uint   objUVCount;      // bound of objUV in float2 entries
    // Local lights bound in `SecondaryScene`. 0/0 ⇒ sun + sky only.
    uint   pointLightCount; uint spotLightCount;
};

/// The device buffers the shading reads. Textures are passed separately —
/// Metal only allows textures inside a struct for argument buffers.
struct SecondaryScene {
    const device RTInstanceData* insts;
    const device float4*         objNormal;
    const device float2*         objUV;        // 3 per triangle, `objNormal` order
    const device float2*         uvScale;      // albedo-atlas per-slice letterbox
    const device RTPointLight*   pointLights;
    const device RTSpotLight*    spotLights;
};

/// One opaque triangle hit, however the ray that found it was fired.
struct SecondaryHit {
    float3 P;             // world position
    float3 N;             // world normal, ALREADY faced against the incoming ray
    float2 bary;          // triangle barycentrics (→ mesh UV → atlas tap)
    uint   instanceID;    // TLAS instance id
    uint   primitiveID;   // triangle within the instance's mesh
};

// ── Albedo ───────────────────────────────────────────────────────────────────

/// Aspect-correct atlas sample. Mirror of `sampleAtlasAspect` in
/// Illuminatorama.metal: `uvScale[slice]` is the fraction of the square slice the
/// letterboxed source fills; (1,1) takes the hardware-repeat fast path, otherwise
/// the letterboxed axis is tiled manually and half-texel inset so filtering never
/// bleeds into the empty band.
static inline float4 secondaryAtlasSample(texture2d_array<float, access::sample> atlas,
                                          sampler s, float2 uv, uint slice,
                                          const device float2* uvScale) {
    float2 sc = uvScale[slice];
    if (sc.x >= 0.999f && sc.y >= 0.999f) { return atlas.sample(s, uv, slice); }
    float2 texSize = float2(atlas.get_width(), atlas.get_height());
    float2 halfTexel = 0.5f / texSize;
    float2 manual = clamp(fract(uv) * sc, halfTexel, sc - halfTexel);
    float2 st;
    st.x = (sc.x >= 0.999f) ? uv.x : manual.x;
    st.y = (sc.y >= 0.999f) ? uv.y : manual.y;
    return atlas.sample(s, st, slice);
}

/// Albedo at a secondary hit — the atlas texel when the instance carries a slice
/// AND the mesh's UVs are resident, else the per-instance mean (the old behaviour,
/// which stays the fallback for untextured / GPU-private meshes).
///
/// THIS is the term that made the world behind a pane read as flat slabs, and it
/// is the same term missing from every other secondary path: a hit shaded with
/// `insts[iid].albedoTriBase.xyz` collapses a plank floor, a tiled wall and a rug
/// to one colour each.
///
/// Deliberately a SINGLE tap: no hex anti-tiling blend, no normal/roughness maps —
/// this is a secondary ray, and three extra atlas reads per bounce is not worth it.
static inline float3 secondaryAlbedo(SecondaryHit h,
                                     SecondaryShadeParams p,
                                     SecondaryScene sc,
                                     texture2d_array<float, access::sample> albedoAtlas) {
    RTInstanceData d = sc.insts[h.instanceID];
    float3 mean = d.albedoTriBase.xyz;
    if (p.albedoAtlasEnabled == 0u) { return mean; }
    // Slice is stored PLUS ONE so the memset-0 default (curve / unslotted
    // instances) decodes to −1 = untextured, instead of silently sampling slice 0.
    int slice = int(d.nrm0.w) - 1;
    if (slice < 0) { return mean; }
    uint base = (uint(d.albedoTriBase.w) + h.primitiveID) * 3u;
    if (base + 2u >= p.objUVCount) { return mean; }   // UVs weren't readable
    float2 uvA = sc.objUV[base], uvB = sc.objUV[base + 1u], uvC = sc.objUV[base + 2u];
    float w0 = 1.0 - h.bary.x - h.bary.y;
    float2 uv = w0 * uvA + h.bary.x * uvB + h.bary.y * uvC;
    constexpr sampler texSmp(filter::linear, mip_filter::linear, address::repeat);
    return secondaryAtlasSample(albedoAtlas, texSmp, uv, uint(slice), sc.uvScale).rgb;
}

// ── Indirect fill ────────────────────────────────────────────────────────────

/// Sky IBL + ambient at a secondary hit — the SAME terms the deferred lighting
/// pass applies, so a surface seen through glass or in a reflection matches the
/// same surface seen directly:
///   • diffuse IBL — the renderer's cosine-convolved `irradianceCube` sampled
///     along the hit normal, × `skyIntensity` (mirrors `diffuseIBL *
///     frame.iblIntensity`; kD ≈ 1 for the rough dielectrics a secondary ray
///     typically lands on). The old flat `albedo * skyAmbient` skipped this
///     entirely, so transmitted scenery lost the sky's cool fill and read hotter
///     and more saturated than the deferred render of the same surface.
///   • ambient supplement — upness-weighted 40 % → 100 %, exactly the deferred
///     `mix(ambCol * 0.4, ambCol, upness)` shaping of `frame.ambientColor`.
///   • interior separation — a hit whose instance layer intersects `interiorMask`
///     gets the SAME treatment the deferred kernel gives an interior fragment.
///     Without it a secondary path fills interiors at raw exterior strength, which
///     is several times off the room beside the pane, in the wrong direction on
///     both terms.
static inline float3 secondaryIndirectFill(float3 hitN, uint hitLayerBits,
                                           SecondaryShadeParams p,
                                           texturecube<float, access::sample> irrCube)
{
    constexpr sampler cubeSmp(filter::linear);
    float iblK = 1.0, ambK = 1.0;
    if (p.interiorMask != 0u && hitLayerBits != 0xFFFFFFFFu &&
        (hitLayerBits & p.interiorMask) != 0u) {
        iblK = mix(p.interiorIBLSide, p.interiorIBLUp, saturate(hitN.y));
        ambK = p.interiorAmbient;
    }
    float3 irr = irrCube.sample(cubeSmp, hitN).rgb * max(0.0, p.skyIntensity) * iblK;
    float upness = saturate(hitN.y * 0.5 + 0.5);
    return irr + mix(p.skyAmbient * 0.4, p.skyAmbient, upness) * ambK;
}

/// Outgoing radiance of a surface-cache card that is KNOWN but not resident
/// (budget streaming zeroed its atlas rect, so reading the atlas would sample
/// (0,0)). Emission + the card's albedo under the same indirect fill a re-shade
/// would get — never black, and never the exterior-strength flat ambient the
/// deferred path used to substitute here.
static inline float3 secondaryCardFallback(SurfCard card, float3 hitN, uint hitLayerBits,
                                           SecondaryShadeParams p,
                                           texturecube<float, access::sample> irrCube)
{
    return card.emission.xyz + card.albedo.xyz * secondaryIndirectFill(hitN, hitLayerBits, p, irrCube);
}

// ── Local lights ─────────────────────────────────────────────────────────────

/// Local point + spot lights at a secondary hit — the same falloff and cone math
/// the deferred lighting kernel applies, LAMBERT-only (albedo/π · N·L) and with NO
/// shadow rays: this is a secondary ray, and a shadow ray per light per bounce is
/// not worth it. The layer mask IS honoured (unlike the deferred path, which lets
/// a shadow-mapped light ignore it) because without a shadow map the mask is the
/// only thing keeping a lamp from lighting the room on the other side of a wall.
/// Returns the irradiance-weighted sum; the caller multiplies by albedo.
static inline float3 secondaryLocalLightFill(float3 P, float3 N, uint layerBits,
                                             SecondaryShadeParams p, SecondaryScene sc)
{
    float3 sum = float3(0.0);
    for (uint i = 0u; i < p.pointLightCount; ++i) {
        RTPointLight pl = sc.pointLights[i];
        if ((pl.layerMask & layerBits) == 0u) continue;
        float3 toL = pl.position - P;
        float dist = length(toL);
        if (dist > pl.radius) continue;
        float3 L = toL / max(dist, 1e-4);
        float nl = saturate(dot(N, L));
        if (nl <= 0.0) continue;
        float atten = 1.0 / max(dist * dist, 1e-4);
        float window = saturate(1.0 - pow(dist / pl.radius, 4.0));
        sum += pl.color * (atten * window * window * nl);
    }
    for (uint i = 0u; i < p.spotLightCount; ++i) {
        RTSpotLight sl = sc.spotLights[i];
        if ((sl.layerMask & layerBits) == 0u) continue;
        float3 toL = sl.position - P;
        float dist = length(toL);
        if (dist > sl.radius) continue;
        float3 L = toL / max(dist, 1e-4);
        float coneAtten = smoothstep(sl.outerCone, sl.innerCone,
                                     dot(normalize(sl.direction), -L));
        if (coneAtten <= 0.0) continue;
        float nl = saturate(dot(N, L));
        if (nl <= 0.0) continue;
        float atten = 1.0 / max(dist * dist, 1e-4);
        float window = saturate(1.0 - pow(dist / sl.radius, 4.0));
        sum += sl.color * (atten * window * window * coneAtten * nl);
    }
    return sum * (1.0 / M_PI_F);
}

// ── Direct sun ───────────────────────────────────────────────────────────────

/// Fraction of the sun disc visible from a secondary hit, by shadow ray(s) against
/// the OPAQUE mask (0x01) — glass (0x02) never casts a solid shadow. `theta == 0`
/// (`sunSoftnessRad`) degenerates to a hard ray, so a caller that cannot afford a
/// soft penumbra pays exactly one trace and gets the old behaviour.
///
/// Templated on the intersector so the glass fragment path
/// (`intersector<triangle_data, instancing>`) and the deferred kernel
/// (`…, curve_data`) share one body.
template <typename Isect>
static inline float secondarySunVisibility(thread Isect& isect,
                                           instance_acceleration_structure accel,
                                           float3 P, float3 N, float3 Ld,
                                           SecondaryShadeParams p,
                                           thread uint& seed)
{
    if (p.shadowRays == 0u) { return 1.0; }
    isect.accept_any_intersection(true);
    uint hits = 0u;
    for (uint s = 0u; s < p.shadowRays; ++s) {
        ray sr;
        sr.origin = P + N * 2e-3;
        sr.direction = coneSample(Ld, p.sunSoftnessRad, rnd(seed), rnd(seed));
        sr.min_distance = 2e-3; sr.max_distance = 1e4;
        if (isect.intersect(sr, accel, 0x01u).type != intersection_type::none) hits++;
    }
    isect.accept_any_intersection(false);
    return 1.0 - float(hits) / float(p.shadowRays);
}

// ── THE shared secondary-surface shader ──────────────────────────────────────

/// Outgoing radiance of an opaque triangle hit reached by a secondary ray.
///
///   emission
/// + textured albedo × ( sky IBL + ambient, interior-separated
///                     + local point/spot lights )
/// + Lambert direct sun with its shadow ray.
///
/// Anything missing here is visible as the world through a pane — or in a
/// reflection — being shaded differently from the world beside it.
///
/// A caller that wants a SUBSET (a GI bounce, say, whose receiving surface already
/// carries its own ambient/IBL term and would double-count it) suppresses terms by
/// zeroing the corresponding `SecondaryShadeParams` fields rather than by keeping a
/// second copy of the body. That is the whole point of this function.
///
/// Callers that have a resident surface-cache card should prefer it
/// (`sampleSurfCacheRT` returns full multi-bounce radiance for one atlas read);
/// this is the re-shade for everything the cache does not cover.
template <typename Isect>
static inline float3 shadeSecondarySurface(thread Isect& isect,
                                           instance_acceleration_structure accel,
                                           SecondaryHit h,
                                           SecondaryShadeParams p,
                                           SecondaryScene sc,
                                           texturecube<float, access::sample> irrCube,
                                           texture2d_array<float, access::sample> albedoAtlas,
                                           thread uint& seed)
{
    uint layerBits = secondaryLayerBits(h.instanceID, sc.insts);
    float3 A = secondaryAlbedo(h, p, sc, albedoAtlas);
    float3 rad = secondaryEmission(h.instanceID, sc.insts);
    rad += A * secondaryIndirectFill(h.N, layerBits, p, irrCube);
    rad += A * secondaryLocalLightFill(h.P, h.N, layerBits, p, sc);
    float3 Ld = normalize(p.sunDir);
    float nl = saturate(dot(h.N, Ld));
    if (nl > 0.0 && p.shadowRays > 0u) {
        float vis = secondarySunVisibility(isect, accel, h.P, h.N, Ld, p, seed);
        rad += A * (1.0 / M_PI_F) * p.sunColor * nl * vis;
    }
    return rad;
}
