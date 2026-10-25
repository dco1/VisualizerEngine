// ── ILLUMINATORAMA — DIFFUSE ROUGHNESS: the energy-preserving Oren-Nayar lobe (Daydream DH-1301) ──
//
// OpenPBR's `base_diffuse_roughness` (r ∈ [0, 1]; 0 = Lambert). A Lambert surface looks flat and
// plastic from a low angle: real rough diffuse (plaster, carpet, brick, velvet, unglazed clay)
// BACKSCATTERS — it is brighter with the light behind the eye and darker head-on under a raking
// light — and its multiple scattering between the facets saturates its colour.
//
// The model is EON — "EON: A practical energy-preserving rough diffuse BRDF", Portsmouth, Kutz &
// Hill, JCGT 14(1) 2025 — the one the OpenPBR 1.1 specification adopts:
//
//   f_ss  = (ρ/π) · A_F · (1 + r · s/t)                  Fujii's improved Oren-Nayar (FON),
//           A_F = 1 / (1 + c₁ r),   s = L·V − μi μo,      t = max(μi, μo) for s > 0, else 1
//   f_ms  = (ρ_ms/π) · (1 − E_F(μo)) (1 − E_F(μi)) / (1 − Ē_F)        the energy FON loses, put back
//   ρ_ms  = ρ² Ē_F / (1 − ρ (1 − Ē_F))                    its colour after all further bounces
//   Ē_F   = A_F (1 + c₂ r)                                FON's cosine-weighted mean albedo
//   E_F(μ) = (1 + r · G(μ)/π) / (1 + c₁ r)                FON's directional albedo (the paper's
//                                                         quartic fit to G/π — `E_FON_approx`)
//   c₁ = 1/2 − 2/(3π),  c₂ = 2/3 − 28/(15π)
//
// …so that a WHITE surface (ρ = 1) reflects exactly what it receives at every r and every view —
// the white furnace stays 1.000 (IlluminatoramaDiffuseRoughnessFurnaceTests). The directional
// albedo of the whole lobe is closed-form:
//
//   E_EON(μo) = ρ E_F(μo) + ρ_ms (1 − E_F(μo))
//
// HOW IT IS APPLIED, and the one approximation this engine makes, stated:
//   • every light that HAS a direction — the sun, point / spot lamps, the fill directionals, the
//     ray-traced primary's sun — evaluates the FULL lobe, as a factor on the Lambert term it already
//     had: `illumiEONLambertRatio` = f_EON · π / ρ;
//   • every source that is INTEGRATED over solid angle — LTC area / window-portal lights, the
//     diffuse IBL and its interior bands + ambient, the traced GI the path lane returns to the
//     primary, and every path / secondary-ray VERTEX — takes the lobe's directional albedo
//     `illumiEONAlbedoRatio` = E_EON(μo) / ρ in place of ρ. That is EXACT under uniform incident
//     light (it is the lobe's own integral) and it is what OpenPBR's reference viewers do for the
//     environment; what it leaves out is the lobe's ANGULAR redistribution within an extended
//     source (the backscatter peak inside one window's footprint). Both lanes make the same
//     choice for the same source, so the live canvas and the path-traced export stay on the same
//     footing.
//
// Byte identity: r ≤ 0 returns the literal 1.0 from both ratios, and every caller multiplies, so a
// material that never sets a diffuse roughness — every Visualizer scene, every glossy material —
// renders bit-for-bit as before.
//
// CARRIAGE (no new G-buffer target). The live G-buffer and the photo lane's extended target are
// both FULL (DH-1014 survey), so `r` rides the SIGN of `albedoMetallic.a`, which holds metalness:
//   a ≥ 0 ⇒ metallic = a, r = 0 (every existing pixel);   a < 0 ⇒ metallic = 0, r = −a.
// Every target family (plaster, paint, brick, carpet, velvet, unglazed clay) is a pure dielectric,
// and a metal's diffuse lobe is weighted by (1 − m), so a PURE-dielectric-only carriage loses
// nothing a material ships; a partly metallic surface keeps its metalness and is Lambert. The ray
// traced rows carry it in the same place: the base-layer lane (`emissionPad.w`, DH-1308) codes
// metalness in its integer part 0…1000, and 1001…2001 means "metallic 0, diffuse roughness
// (n − 1001)/1000". The material-table indirection DH-1014 recommends for the full parameter-model
// migration needs an instance id in the G-buffer, which has no room either (velocity is RG16F,
// the layer target is the room light-layer mask); this carriage is one channel's sign, read
// through the two helpers below by every reader of that channel.
#ifndef ILLUMINATORAMA_DIFFUSE_ROUGHNESS_H
#define ILLUMINATORAMA_DIFFUSE_ROUGHNESS_H

#include <metal_stdlib>
using namespace metal;

#define kEONc1 0.28779341f   // 1/2 − 2/(3π)
#define kEONc2 0.07248821f   // 2/3 − 28/(15π)

/// FON directional albedo E_F(μ, r) — the EON paper's polynomial fit (`E_FON_approx`).
static inline float illumiEONFonAlbedo(float mu, float r) {
    float mc = 1.0f - saturate(mu);
    float gOverPi = mc * (0.0571085289f + mc * (0.491881867f + mc * (-0.332181442f + mc * 0.0714429953f)));
    return (1.0f + r * gOverPi) / (1.0f + kEONc1 * r);
}

/// FON's cosine-weighted mean albedo Ē_F(r).
static inline float illumiEONFonMean(float r) {
    return (1.0f + kEONc2 * r) / (1.0f + kEONc1 * r);
}

/// ρ_ms / ρ = ρ Ē_F / (1 − ρ (1 − Ē_F)) — finite at ρ = 0, 1 at ρ = 1.
static inline float3 illumiEONMsOverRho(float3 rho, float meanF) {
    float3 p = saturate(rho);
    return p * meanF / max(1.0f - p * (1.0f - meanF), float3(1e-4f));
}

/// f_EON · π / ρ — the factor on a Lambert term `ρ/π · cos θi` under a light from L.
/// `LdotV` is the full dot product of the two (unit) directions. r ≤ 0 ⇒ exactly 1.
static inline float3 illumiEONLambertRatio(float3 rho, float r, float NdotL, float NdotV, float LdotV) {
    if (!(r > 0.0f)) return float3(1.0f);
    r = min(r, 1.0f);
    float mi = saturate(NdotL), mo = saturate(NdotV);
    float s = LdotV - mi * mo;
    float sOverT = s > 0.0f ? s / max(max(mi, mo), 1e-4f) : s;
    float AF = 1.0f / (1.0f + kEONc1 * r);
    float ss = AF * (1.0f + r * sOverT);
    float meanF = illumiEONFonMean(r);
    float ms = max(1e-7f, 1.0f - illumiEONFonAlbedo(mo, r)) * max(1e-7f, 1.0f - illumiEONFonAlbedo(mi, r))
             / max(1e-7f, 1.0f - meanF);
    return ss + illumiEONMsOverRho(rho, meanF) * ms;
}

/// E_EON(μo) / ρ — the lobe's directional albedo relative to Lambert's: what an INTEGRATED source
/// (area light, IBL, traced GI, a path vertex) multiplies ρ by. r ≤ 0 ⇒ exactly 1.
static inline float3 illumiEONAlbedoRatio(float3 rho, float r, float NdotV) {
    if (!(r > 0.0f)) return float3(1.0f);
    r = min(r, 1.0f);
    float EF = illumiEONFonAlbedo(NdotV, r);
    return EF + illumiEONMsOverRho(rho, illumiEONFonMean(r)) * (1.0f - EF);
}

// ── The G-buffer carriage (see the header comment) ─────────────────────────────────────────────

/// Pack metalness + diffuse roughness into `albedoMetallic.a`. Only a PURE dielectric carries r.
static inline half illumiPackMetallicDiffuseRoughness(float metallic, float diffuseRoughness) {
    return (diffuseRoughness > 0.0f && metallic <= 0.0f) ? half(-min(diffuseRoughness, 1.0f)) : half(metallic);
}
/// Metalness out of `albedoMetallic.a` — every reader of that channel goes through this.
static inline float illumiUnpackMetallic(half a) { return max(float(a), 0.0f); }
/// Diffuse roughness out of `albedoMetallic.a` (0 for every metal and every legacy pixel).
static inline float illumiUnpackDiffuseRoughness(half a) { return max(-float(a), 0.0f); }

#endif
