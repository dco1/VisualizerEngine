#pragma once

// ── ILLUMINATORAMA — THE LAYERED BSDF'S ENERGY BOOKKEEPING (Daydream DH-1014) ─────────────────
//
// OpenPBR (and MaterialX's `standard_surface` before it) describes a material as a STACK of
// layers — fuzz (sheen) over coat over specular over a metal/dielectric base — and says how
// much of the light each layer lets through to the one below: whatever the layer above
// reflects is gone from the layer below ("albedo scaling", Kulla & Conty 2017 §3, Estevez &
// Kulla 2017 §4). That bookkeeping is what makes a white material in a white furnace return
// exactly 1.0 whatever its roughness, and it is what this engine's lobes were missing:
//
//   • the single-scatter GGX lobe drops the light that bounces more than once between
//     microfacets — a white metal under a uniform LIGHT returned 0.68 at roughness 0.2 and
//     0.41 at roughness 1 (mean over N·V 1…0.15; 0.26 at grazing) — and only the IBL arm had
//     S1.3b's compensation, so the SAME surface lost energy under the sun, a lamp or a window
//     portal and kept it under the sky;
//   • the diffuse base was weighted by a per-light Schlick (`1 − F(H·V)`) or a roughness-
//     Schlick (`1 − F_r(N·V)`), or nothing at all (area lights), none of which is what the
//     specular lobe actually reflected — a white dielectric returned 0.87…1.05;
//   • the path-traced lane (Maximum) weighted its traced diffuse by `albedo` alone, so a white
//     METAL returned 2.000 there (a full diffuse it has no lobe for) and a dielectric up to 1.46;
//   • cloth sheen was added ON TOP of a base that kept all its energy, and its environment
//     arm was a hand-fitted curve (DH-0597) — white velvet returned up to 1.49 at grazing.
//
// Everything here is the PER-PIXEL half of that stack, evaluated from one read of the split-sum
// DFG LUT (`illumi_dfg_bake`, IlluminatoramaIBLBake.metal), whose four channels are:
//
//   .r  A  — env-lobe Fresnel scale    } the split sum of the IBL-remapped GGX lobe (k = α/2):
//   .g  B  — env-lobe Fresnel bias     } E_env(F0) = F0·A + B
//   .b  Ed — the DIRECT lobe's directional albedo at F = 1: the lobe `brdf()` actually evaluates
//            for a punctual/area light (Schlick-GGX with the analytic k = (r+1)²/8 remap), so
//            its multiple-scattering compensation is exact for THAT lobe, not for a cousin of it
//   .a  Es — the Charlie/Neubelt cloth-sheen lobe's directional albedo, keyed (N·V, α_sheen)
//            (DH-0597 — Filament's DFV_Charlie; the same `clothSheenD/V` as the direct lobe)
//
// The consumers are the deferred lighting kernel (every light type + the environment) and the
// ray-traced lighting kernel (the path-traced GI lane's diffuse weight and the RT reflection's
// Fresnel), which is what keeps Maximum and the live lane in agreement: both weight the same
// surface by the same numbers.
//
// Every entry point is gated by the host (`FrameUniforms.frameFlags` kFrameFlagLayered* /
// `RTInstUniforms.pathFlags` kPathFlagLayered*): with the flags clear the callers keep their
// historical expressions verbatim, so a host that never opts in is byte-identical.

#include <metal_stdlib>
using namespace metal;

/// FrameUniforms.frameFlags bit 2 — multiple-scattering GGX energy compensation on every
/// specular lobe + the diffuse base weighted by what the specular layer did NOT reflect.
#define kFrameFlagLayeredMultiscatter 4u
/// FrameUniforms.frameFlags bit 3 — cloth sheen read from the LUT's fitted albedo (both
/// environment arms) and energy-conserving against the base (albedo scaling).
#define kFrameFlagLayeredSheen        8u
/// FrameUniforms.frameFlags bit 7 (128) — DIFFUSE ROUGHNESS (Daydream DH-1301): the master switch for
/// the EON lobe (IlluminatoramaDiffuseRoughness.h). The per-pixel value rides the sign of
/// `albedoMetallic.a`, so with the flag set a pixel whose material never set one is still Lambert,
/// bit for bit. Mirrored for the ray-traced lighting kernel as RTInstUniforms.pathFlags
/// kPathFlagDiffuseRoughness (the traced primary's sun, its GI weight, every secondary / path
/// vertex). Clear ⇒ every lane is Lambert whatever the materials carry.
#define kFrameFlagDiffuseRoughness    128u
/// RTInstUniforms.pathFlags bit 14 (16384) — the same switch for the ray-traced lighting kernel.
#define kPathFlagDiffuseRoughness     16384u

#include "IlluminatoramaDiffuseRoughness.h"
#include "IlluminatoramaMaterialLayers.h"   // DH-1300 — the per-material specular F0

// ── Cloth sheen lobe (moved here from IlluminatoramaLighting.metal so the LUT bake integrates
//    the IDENTICAL functions the direct lobe evaluates — the two arms cannot drift) ──────────
//
// **Estevez & Kulla 2017 ("Production Friendly Microfacet Sheen BRDF", SIGGRAPH talk)**
// replace GGX's Beckmann-like NDF with an inverted-Gaussian "Charlie" distribution, whose
// density is highest for microfacets standing PERPENDICULAR to the surface. Paired with
// Ashikhmin & Premoze / Neubelt & Pettineo's velvet visibility term (which stays finite as
// either cosine goes to zero, exactly where GGX's Smith term collapses), it is four lines and
// no texture.
//
// The `alpha → 0` guard matters: `invAlpha` becomes the exponent, so a zero roughness is an
// infinite power. Clamped at 1e-3.
static inline float clothSheenD(float alpha, float NdotH) {
    float invAlpha = 1.0 / max(alpha, 1e-3);
    float cos2h = NdotH * NdotH;
    float sin2h = max(1.0 - cos2h, 1e-7);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * M_PI_F);
}

/// Ashikhmin/Neubelt velvet visibility — finite at grazing, where Smith-GGX goes to zero and
/// takes the whole sheen lobe with it.
static inline float clothSheenV(float NdotV, float NdotL) {
    return 1.0 / max(4.0 * (NdotL + NdotV - NdotL * NdotV), 1e-5);
}

/// Fibre-tip colour: cloth sheen is scattered by the pale tips of the nap, not by the dyed
/// core, so it is markedly whiter than the base albedo. Same mix the Phase-7b bolt-on used.
static inline float3 clothSheenColor(float3 albedo) { return albedo * 0.4 + float3(0.6); }

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

// ── The per-pixel energy terms ────────────────────────────────────────────────────────────────

/// What one LUT read buys a shaded pixel. Built by `layeredEnergy` (or `layeredOff`); read by
/// `brdf`, `brdfDiffuse`, `evalAreaLight` and the environment arm.
struct LayeredEnergy {
    /// Turquin 2019 multiple-scattering factor for the DIRECT GGX lobe: `1 + F0·(1 − Ed)/Ed`.
    /// Exact (furnace = 1) at F0 = 1 by construction; 1 when multiscatter is off.
    float3 specDirectMS;
    /// Diffuse weight under a punctual / area light, `(1 − m)·(1 − E_spec,direct(μo))`. Replaces
    /// `(1 − F(H·V))(1 − m)` — the base receives exactly what the specular layer passes down.
    float3 diffuseDirect;
    /// The environment BRDF: specular albedo incl. Fdez-Agüera multiple scattering,
    /// `FssEss + FmsEms` — what the prefiltered radiance is multiplied by.
    float3 envSpec;
    /// Diffuse weight under the environment, `(1 − m)·(1 − envSpec)` (Fdez-Agüera 2019 §5).
    float3 diffuseEnv;
    /// The sheen layer's albedo scaling of everything under it: `1 − s·max(c_sheen)·Es(μo)`.
    float  baseUnderSheen;
    /// Directional albedo of the sheen lobe at μo — the environment arm's magnitude.
    float  sheenAlbedo;
    /// Daydream DH-1301 — the pixel's diffuse roughness (OpenPBR `base_diffuse_roughness`, EON);
    /// 0 ⇒ Lambert. Read by `brdf` / `brdfDiffuse` (the full lobe, per light direction).
    float  diffuseRoughness;
    /// E_EON(μo)/ρ — what every INTEGRATED source (area lights, IBL, ambient) multiplies ρ by; 1 ⇒ Lambert.
    float3 eonAlbedoRatio;
    /// Whether the multiscatter / diffuse-weight half is live (else callers keep the legacy terms).
    bool   multiscatter;
    /// Daydream DH-1300 — the dielectric base's F0 (OpenPBR `specular_weight · F0(specular_ior)`,
    /// tinted by `specular_color`; IlluminatoramaMaterialLayers.h). What every lobe's
    /// `mix(F0, albedo, metallic)` mixes from; the literal 0.04 unless the material set one.
    float3 dielectricF0;
};

/// Fdez-Agüera 2019 ("A Multiple-Scattering Microfacet Model for Real-Time Image-Based
/// Lighting", JCGT 8.1): the single-scatter split sum `F0·A + B`, plus the energy lost to
/// single scattering `Ems = 1 − (A + B)` re-emitted after an average Fresnel
/// `Favg = F0 + (1 − F0)/21`, summed over all further bounces `1/(1 − Favg·Ems)`.
/// This is the S1.3b expression the lighting kernel has shipped since 2026-08, unchanged —
/// it moved here so the RT kernel can weight its reflections by the same number.
static inline float3 layeredEnvSpecular(float3 F0, float2 dfg) {
    float3 FssEss = F0 * dfg.x + dfg.y;
    float  Ems    = saturate(1.0 - (dfg.x + dfg.y));
    float3 Favg   = F0 + (1.0 - F0) * (1.0 / 21.0);
    float3 FmsEms = (Ems * FssEss * Favg) / max(1.0 - Favg * Ems, 1e-4);
    return FssEss + FmsEms;
}

/// The neutral terms: every lobe's historical expression. What a caller that never opts in
/// passes (the default argument of `brdf` & co.).
///
/// **Passed BY VALUE, never as a `thread` pointer with a null sentinel.** The first cut handed
/// `brdf` a `thread const LayeredEnergy*` defaulting to `nullptr`, and the furnace probe showed
/// the compensation silently OFF with the flags set: `&le` of a thread-local can be address 0
/// in the thread address space — the null pointer's representation — so `le != nullptr` was
/// false for a perfectly valid local. (The `thread float3 *sheenOut = nullptr` debug accumulator
/// in IlluminatoramaLighting.metal shares the hazard — DH-1307.)
static inline LayeredEnergy layeredOff() {
    LayeredEnergy e;
    e.specDirectMS = float3(1.0);
    e.diffuseDirect = float3(1.0);
    e.envSpec = float3(0.0);
    e.diffuseEnv = float3(1.0);
    e.baseUnderSheen = 1.0;
    e.sheenAlbedo = 0.0;
    e.multiscatter = false;
    e.diffuseRoughness = 0.0;
    e.eonAlbedoRatio = float3(1.0);
    e.dielectricF0 = float3(0.04);
    return e;
}

/// The per-pixel energy terms from one DFG LUT texel (`lut` sampled at (N·V, roughness)) and,
/// for cloth, the sheen texel (`sheenLut.a` sampled at (N·V, sheenRoughness)).
///
/// `multiscatter` / `sheen` are the host's flags; with both false every field is the neutral
/// value (1 / legacy) — callers ALSO keep their legacy expressions on that path, so this
/// function is never what makes an opted-out host byte-identical; the callers are.
static inline LayeredEnergy layeredEnergy(float3 albedo, float metallic,
                                          float4 lut, float sheenLutA,
                                          float sheenStrength,
                                          bool multiscatter, bool sheen,
                                          float3 dielectricF0 = float3(0.04)) {
    LayeredEnergy e;
    e.dielectricF0 = dielectricF0;     // DH-1300
    float3 F0 = mix(dielectricF0, albedo, metallic);
    e.multiscatter = multiscatter;
    e.diffuseRoughness = 0.0;          // set by `layeredWithDiffuseRoughness`
    e.eonAlbedoRatio = float3(1.0);
    e.envSpec = layeredEnvSpecular(F0, lut.xy);
    if (multiscatter) {
        float Ed = clamp(lut.z, 1e-3, 1.0);
        e.specDirectMS = 1.0 + F0 * ((1.0 - Ed) / Ed);
        // The direct lobe's Fresnel-weighted albedo. The LUT carries the direct lobe's total
        // (Ed, F = 1) but not its Fresnel split, so the split is taken from the env lobe's
        // (A, B) at the same texel — G is the only difference between the two lobes and it
        // scales both halves of the split alike. Exact at F0 = 1 (ratio 1) and at F0 = 0
        // (ratio B/(A+B) is the only term); the direct furnace for a white dielectric reads
        // 0.990…1.000 from roughness 0.2 up (IlluminatoramaLayeredBSDFFurnaceTests).
        float  AB = max(lut.x + lut.y, 1e-4);
        float3 EssDirect = Ed * (F0 * lut.x + lut.y) / AB;
        float3 EspecDirect = saturate(EssDirect * e.specDirectMS);
        e.diffuseDirect = (1.0 - metallic) * (1.0 - EspecDirect);
        e.diffuseEnv    = (1.0 - metallic) * (1.0 - saturate(e.envSpec));
    } else {
        e.specDirectMS  = float3(1.0);
        e.diffuseDirect = float3(1.0 - metallic);   // not read on the legacy path
        e.diffuseEnv    = float3(1.0 - metallic);   // not read on the legacy path
    }
    e.sheenAlbedo = saturate(sheenLutA);
    e.baseUnderSheen = 1.0;
    if (sheen && sheenStrength > 0.0) {
        float3 c = clothSheenColor(albedo);
        e.baseUnderSheen = saturate(1.0 - sheenStrength * max(c.r, max(c.g, c.b)) * e.sheenAlbedo);
    }
    return e;
}

/// Daydream DH-1301 — stamp a pixel's diffuse roughness onto its energy terms: the EON lobe for
/// every light with a direction (`brdf`, `brdfDiffuse`) and the lobe's directional albedo for
/// every integrated source (`eonAlbedoRatio`). r ≤ 0 leaves `e` exactly as it was.
static inline LayeredEnergy layeredWithDiffuseRoughness(LayeredEnergy e, float3 albedo,
                                                        float diffuseRoughness, float NdotV) {
    if (!(diffuseRoughness > 0.0)) return e;
    e.diffuseRoughness = diffuseRoughness;
    e.eonAlbedoRatio = illumiEONAlbedoRatio(albedo, diffuseRoughness, NdotV);
    return e;
}

// ── The ray-traced lighting kernel's half (IlluminatoramaRTInstanced.metal) ─────────────────
//
// The traced GI (one-bounce or the path-traced Maximum lane) REPLACES the deferred pass's
// diffuse environment where it runs, and the traced reflection REPLACES its specular IBL where
// a ray hits. For Maximum and the live lane to agree, each replacement has to be weighted by
// the number the deferred term it replaces was weighted by:
//
//   • the traced diffuse by `(1 − m)(1 − envSpec)·baseUnderSheen` — it was `albedo` alone, so a
//     METAL received a full diffuse indirect in Maximum (the deferred gives it none) and every
//     dielectric received the Fresnel share twice (once as specular IBL, once as diffuse);
//   • the traced reflection by `envSpec`, the split-sum env BRDF the specular IBL uses — it was
//     a plain Schlick at N·V, which tends to 1 at grazing however rough the surface.
//
// RTInstUniforms.pathFlags bits (beside DH-1011's 1/2/4 and the 8–128 instruments; set by the
// host whether or not the path lane runs, since the one-bounce GI replaces the same terms).
#define kPathFlagLayeredMultiscatter 256u
#define kPathFlagLayeredSheen        512u

struct LayeredRTWeights {
    float3 diffuse;      // × albedo × incoming irradiance
    float3 reflection;   // × traced reflected radiance
};

/// `fresLegacy` is the kernel's historical reflection Fresnel, returned unchanged when the
/// multiscatter bit is clear (diffuse weight then 1 — the historical `× albedo`).
static inline LayeredRTWeights layeredRTWeights(uint pathFlags, float3 albedo, float metallic,
                                                float roughness, float NdotV, half emissionAlpha,
                                                texture2d<half, access::sample> dfgLUT,
                                                float3 fresLegacy, float diffuseRoughness = 0.0,
                                                float3 dielectricF0 = float3(0.04)) {
    LayeredRTWeights w;
    // DH-1301 — the traced GI replaces the deferred diffuse ENVIRONMENT, which a rough diffuse
    // surface weights by its EON directional albedo; the replacement takes the same factor
    // (exactly 1 at diffuse roughness 0, with or without the layered switches).
    w.diffuse = illumiEONAlbedoRatio(albedo, diffuseRoughness, NdotV);
    w.reflection = fresLegacy;
    bool ms = (pathFlags & kPathFlagLayeredMultiscatter) != 0u;
    bool sh = (pathFlags & kPathFlagLayeredSheen) != 0u;
    if (!ms && !sh) return w;
    // The sheen decode is the deferred kernel's (`emission.alpha = -(band + strength)`).
    float sheenStrength = 0.0, sheenRoughness = clothSheenRoughnessForBand(0);
    if (emissionAlpha < -0.001h) {
        float m = float(-emissionAlpha);
        int band = int(floor(m + 1e-3f));
        sheenStrength = m - float(band);
        sheenRoughness = clothSheenRoughnessForBand(band);
    }
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float4 lut = float4(dfgLUT.sample(s, float2(NdotV, roughness)));
    float sheenA = (sh && sheenStrength > 0.0) ? float(dfgLUT.sample(s, float2(NdotV, sheenRoughness)).a) : 0.0;
    LayeredEnergy e = layeredEnergy(albedo, metallic, lut, sheenA, sheenStrength, ms, sh, dielectricF0);
    if (ms) { w.diffuse *= e.diffuseEnv; w.reflection = e.envSpec; }
    w.diffuse *= e.baseUnderSheen;
    w.reflection *= e.baseUnderSheen;
    return w;
}

// ── The path lane's VERTEX BSDF (Daydream DH-1308) ───────────────────────────────────────────
//
// Until DH-1308 every vertex of the path-traced lane past the primary was Lambert × albedo: a
// chrome fixture, a mirror or a lacquered floor met in the second bounce scattered the light
// like matte paint. Energy was right (a white furnace returns 1 either way) but the LOBE was
// wrong — no specular inter-reflection, so a polished floor never threw a window's light onto
// the wall opposite and a metal's coloured F0 bounced diffusely.
//
// With `kPathFlagGlossyVertices` a path vertex is the SAME layered stack the deferred kernel
// shades (and `layeredRTWeights` weights the primary by), sampled rather than weighted:
//
//   coat     — dielectric GGX (F0 0.04, the material's clearcoat roughness), albedo
//              Ec = cc · E_env(0.04, r_coat)                         [only where clearcoat > 0]
//   specular — GGX with F0 = mix(0.04, albedo, m), α = r², VNDF-sampled, its single-scatter
//              albedo put back to the layered model's `envSpec` (Fdez-Agüera multiple scattering)
//   diffuse  — Lambert × albedo × (1 − m)(1 − envSpec)  (exactly `layeredEnergy.diffuseEnv`)
//   …with everything under the coat × (1 − Ec) (albedo scaling, as the sheen layer does).
//
// ONE lobe is picked per vertex ∝ its albedo (a one-sample estimator: each lobe's integral is
// estimated by its own strategies and divided by its pick probability, so the sum is unbiased).
// The GGX lobe's G is the IBL-remapped separable Smith the DFG LUT's (A, B) integrate
// (`layeredSmithG_IBL`, k = α/2) and it is sampled through the EXACT Smith G1 visible-normal
// distribution: the estimator F·G_IBL/G1 then has expectation F0·A + B — the LUT's own number —
// so a white dielectric returns (1 − envSpec) + envSpec = 1 and a white metal envSpec = 1.

/// RTInstUniforms.pathFlags bit — sample the layered BSDF at path vertices (needs light
/// sampling + a bound DFG LUT; the host only sets it with both). Independent of the deferred
/// pass's `kPathFlagLayeredMultiscatter`.
#define kPathFlagGlossyVertices 2048u

/// Separable Smith-GGX with the IBL remap k = α/2 — THE G the DFG LUT's (A, B) integrate
/// (`illumi_dfg_bake` calls this), so a lobe built on it has the LUT's albedo by construction.
static inline float layeredSmithG_IBL(float NdotV, float NdotL, float roughness) {
    float alpha = roughness * roughness;
    float k = alpha * 0.5;
    return (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
}

/// GGX NDF with a RELATIVE guard. The deferred `distributionGGX`'s `+1e-7` clips the peak below
/// roughness ~0.2 (DH-1306); at a path vertex the lobe is sampled and evaluated against its own
/// pdf, so a clipped D would bias every polished bounce.
static inline float pathGGXD(float alpha, float NdotH) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(M_PI_F * d * d, 1e-30);
}

/// The exact Smith-GGX masking term G1 — the normaliser of the visible-normal distribution.
static inline float pathSmithG1(float alpha, float NdotX) {
    float c = max(NdotX, 1e-6);
    float a2 = alpha * alpha;
    return 2.0 * c / (c + sqrt(a2 + (1.0 - a2) * c * c));
}

/// One GGX lobe at a path vertex: the view, the Fresnel, the roughness, and the scale that turns
/// the sampled single-scatter lobe into the layered model's share (multiscatter × layer weight).
struct PathGlossyLobe {
    float3 N;         // the vertex's shading normal (faces the incoming ray)
    float3 wo;        // toward where the path came from
    float3 F0;
    float3 scale;     // layerWeight · envSpec / (F0·A + B)
    float  roughness; // perceptual (the LUT's axis); α = roughness²
    float  alpha;
    float  NdotV;
    float  G1v;       // pathSmithG1(alpha, NdotV)
};

static inline PathGlossyLobe pathGlossyLobe(float3 N, float3 wo, float3 F0, float roughness,
                                            float3 layerWeight, float4 lut) {
    PathGlossyLobe b;
    b.N = N; b.wo = wo; b.F0 = F0;
    b.roughness = roughness;
    b.alpha = roughness * roughness;
    b.NdotV = max(dot(N, wo), 1e-4);
    b.G1v = pathSmithG1(b.alpha, b.NdotV);
    float3 ss = max(F0 * lut.x + lut.y, float3(1e-4));
    b.scale = layerWeight * layeredEnvSpecular(F0, lut.xy) / ss;
    return b;
}

/// f(wi, wo) of the lobe — the BRDF value, NOT × cos — and the VNDF sampler's density of `wi`
/// per steradian, for MIS against the vertex's light sampling. Zero below either horizon.
static inline float3 pathGlossyEval(PathGlossyLobe b, float3 wi, thread float& pdf) {
    pdf = 0.0;
    float NdotL = dot(b.N, wi);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(wi + b.wo);
    float NdotH = saturate(dot(b.N, H));
    float VdotH = saturate(dot(b.wo, H));
    float D = pathGGXD(b.alpha, NdotH);
    pdf = D * b.G1v / (4.0 * b.NdotV);
    float G = layeredSmithG_IBL(b.NdotV, NdotL, b.roughness);
    float3 F = b.F0 + (1.0 - b.F0) * pow(1.0 - VdotH, 5.0);
    return F * b.scale * (D * G / (4.0 * b.NdotV * NdotL));
}

/// DH-1395 — `pathGlossyEval` against a SPHERICAL SOURCE of angular radius `rho` (radians, as seen from the
/// shaded point: radius / distance), instead of a point. A lamp here is a point light with a physical
/// bulb radius, and the lobe's value toward a point is the lobe's peak whenever the point sits in the
/// mirror direction: 1/(π α²) is 8e4 at the 0.045 roughness floor, over a patch of surface a few
/// millimetres across — so a diffuse path that lands in it carries a value 10⁴ times the mean, and the
/// frame is full of fireflies (a polished counter reflecting a can onto a plain door: soft pale discs
/// once the GI filter has spread them). Averaging the lobe over the source's disc IS the physical
/// quantity — the source subtends ρ, so the half-vector sweeps ±ρ/2 — and it is a convolution of two
/// narrow distributions: for GGX (width α in tanθ_h) the result is a lobe of width √(α² + (ρ/2)²) that
/// still integrates to 1, so there is no energy factor to fix. Only the NDF widens (G and F are slow).
/// `rho = 0`, `minAlpha = 0` is `pathGlossyEval` exactly. `minAlpha` floors the lobe's own α for THIS light
/// only (the lamp regularisation, `RTInstUniforms.pathGlossyLampMinRoughness`). Used for lamps only: the sun's cone and the apertures are
/// sampled by the continuation too, and MIS already bounds them.
static inline float3 pathGlossyEvalSource(PathGlossyLobe b, float3 wi, float rho, float minAlpha, thread float& pdf) {
    pdf = 0.0;
    float NdotL = dot(b.N, wi);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(wi + b.wo);
    float NdotH = saturate(dot(b.N, H));
    float VdotH = saturate(dot(b.wo, H));
    float half_ = 0.5 * rho;
    float a0 = max(b.alpha, minAlpha);   // the lamp-NEE regularisation floor (DH-1395), on top of the source's size
    float alphaEff = min(1.0, sqrt(a0 * a0 + half_ * half_));
    float D = pathGGXD(alphaEff, NdotH);
    pdf = pathGGXD(b.alpha, NdotH) * b.G1v / (4.0 * b.NdotV);
    float G = layeredSmithG_IBL(b.NdotV, NdotL, b.roughness);
    float3 F = b.F0 + (1.0 - b.F0) * pow(1.0 - VdotH, 5.0);
    return F * b.scale * (D * G / (4.0 * b.NdotV * NdotL));
}

/// A tangent frame about `n` (Duff et al.) — local to this header so it has no dependencies.
static inline void pathBasis(float3 n, thread float3& t, thread float3& bt) {
    float s = n.z >= 0.0 ? 1.0 : -1.0; float a = -1.0 / (s + n.z); float d = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * d, -s * n.x);
    bt = float3(d, s + n.y * n.y * a, -n.y);
}

/// Visible-normal sample (Heitz 2018 stretch; Dupuy & Benyoub 2023 spherical caps) in the
/// local frame (z = N), from a view `Vl` with Vl.z > 0.
static inline float3 pathSampleGGXVNDF(float3 Vl, float alpha, float2 u) {
    float3 Vh = normalize(float3(alpha * Vl.x, alpha * Vl.y, Vl.z));
    float phi = 2.0 * M_PI_F * u.x;
    float z = (1.0 - u.y) * (1.0 + Vh.z) - Vh.z;
    float sinT = sqrt(saturate(1.0 - z * z));
    float3 Hh = float3(sinT * cos(phi), sinT * sin(phi), z) + Vh;
    return normalize(float3(alpha * Hh.x, alpha * Hh.y, max(Hh.z, 0.0)));
}

/// Sample the lobe: returns the throughput weight f·cos/pdf (= F·G_IBL/G1 · scale), the
/// direction and its density; weight 0 (and pdf 0) when the reflection falls below the horizon.
static inline float3 pathGlossySample(PathGlossyLobe b, float2 u, thread float3& wi, thread float& pdf) {
    float3 t, bt; pathBasis(b.N, t, bt);
    float3 Vl = float3(dot(b.wo, t), dot(b.wo, bt), b.NdotV);
    float3 Hl = pathSampleGGXVNDF(Vl, b.alpha, u);
    float3 H = normalize(t * Hl.x + bt * Hl.y + b.N * Hl.z);
    wi = reflect(-b.wo, H);
    float NdotL = dot(b.N, wi);
    pdf = 0.0;
    if (NdotL <= 0.0) return float3(0.0);
    float VdotH = saturate(dot(b.wo, H));
    pdf = pathGGXD(b.alpha, saturate(Hl.z)) * b.G1v / (4.0 * b.NdotV);
    float3 F = b.F0 + (1.0 - b.F0) * pow(1.0 - VdotH, 5.0);
    return F * b.scale * (layeredSmithG_IBL(b.NdotV, NdotL, b.roughness) / b.G1v);
}
