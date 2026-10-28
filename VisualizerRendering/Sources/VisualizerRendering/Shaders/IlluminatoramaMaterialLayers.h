#pragma once

// ── ILLUMINATORAMA — THE PER-MATERIAL LAYER PARAMETERS (Daydream DH-1300 / DH-1309 / DH-1298 / DH-1299) ──
//
// OpenPBR's parameter model has more per-material numbers than the G-buffer has channels: the
// specular layer's IOR, weight and colour (DH-1300), and — for the coat — its IOR, colour and the
// darkening of the base beneath it (DH-1298). DH-1014's survey found every target full and
// recommended a material table indexed from the G-buffer; DH-1301 rejected that only because no
// target had room for an instance id. This header is the answer to both halves:
//
//   • THE CARRIAGE — a seventh G-buffer target, `GBufferOut.materialLayer` (R32Uint, color 7),
//     holding the fragment's INSTANCE SLOT — or `kMaterialLayerNone` when the instance carries no
//     layer parameters, which is every instance of every host that never sets one. The "table" is
//     the instance buffer itself (no host-side table to keep in step): a consumer reads the
//     instance's few layer float4s straight out of it by slot. Every lane reads the SAME numbers
//     through the SAME decode below:
//       – the deferred lighting kernel (every light type + the environment) by the G-buffer slot;
//       – the ray-traced kernel's PRIMARY (its GI weights + reflection Fresnel) by the same slot;
//       – the ray-traced kernel's SECONDARY / path vertices by the TLAS instance id, which is the
//         raster slot by construction (`rebuildRTAccel` lays the TLAS out in the grouped order of
//         `currentInstanceBuffer`; ids ≥ the instance count are curves / glass and decode to none).
//     A per-pixel channel was the alternative and lost on count: specular IOR + weight + colour and
//     coat IOR + colour + darkening are nine numbers, and none of them varies per texel.
//   • THE DECODE — `MaterialLayers`, with every field's legacy value when the slot is none, so a
//     consumer that keeps its historical literal (F0 0.04) on the `!any` path is byte-identical.
//
// The slot is read as raw float4s (`kInstanceF4*`) rather than through `Instance` so the RT kernel,
// which does not include IlluminatoramaCommon.h, reads the same record; IlluminatoramaCommon.h
// static_asserts the offsets against the struct.

#include <metal_stdlib>
using namespace metal;

/// FrameUniforms.frameFlags bit 8 (256) — DH-1300: the per-material SPECULAR layer (IOR, weight,
/// colour) in the deferred lighting kernel. Mirrored for the ray-traced kernel as
/// RTInstUniforms.pathFlags kPathFlagLayeredSpecular. Clear ⇒ every dielectric is F0 0.04.
#define kFrameFlagLayeredSpecular  256u
#define kPathFlagLayeredSpecular   65536u
/// FrameUniforms.frameFlags bit 9 (512) — DH-1309: the COAT as a real layer (IlluminatoramaBSDF.h
/// `layeredWithCoat`): its lobe under every light (sun, points, spots, fills, area / portal lights)
/// and the environment, the base below attenuated by the coat lobe's own directional albedo, the
/// traced reflection taking the coat's share. Mirrored as RTInstUniforms.pathFlags
/// kPathFlagLayeredCoat. Clear ⇒ the Phase-7 clearcoat (sun + IBL, base × 1 − cc·Schlick(N·V)).
#define kFrameFlagLayeredCoat      512u
#define kPathFlagLayeredCoat       131072u

/// FrameUniforms.frameFlags bit 12 (4096) — DH-1014 thin-film: iridescence on OPAQUE surfaces (an
/// interference film of `thinFilm` thickness / IOR over the base), as a view-angle tint on the base's
/// F0 under every light, the environment, SSR and the traced primary's reflection. Mirrored as
/// RTInstUniforms.pathFlags kPathFlagThinFilm. Clear ⇒ every F0 is untouched, bit for bit. Path
/// vertices past the primary keep a plain F0 (declared approximation).
#define kFrameFlagThinFilm         4096u
#define kPathFlagThinFilm          524288u

/// FrameUniforms.frameFlags bit 13 (8192) — DH-1302: per-material thin-sheet TRANSLUCENCY (weight +
/// colour, OpenPBR transmission_weight / transmission_color for a thin sheet) on EVERY lane — the
/// instance's `translucency` record read through the material-layer target, where `thinTransmission`
/// reached only the photo lane's extended G-buffer. Deferred kernel only (the traced lanes own the
/// indirect, not this term). Clear ⇒ the DH-0140 slice-3 behaviour, bit for bit.
#define kFrameFlagTranslucency     8192u

/// `GBufferOut.materialLayer`'s clear value and the "no layer parameters" slot.
#define kMaterialLayerNone 0xFFFFFFFFu

/// The `Instance` record as float4s (static_asserted in IlluminatoramaCommon.h).
#define kInstanceF4Stride       29u   // sizeof(Instance) / 16
#define kInstanceF4Coat          9u   // (metallic, clearcoat, clearcoatRoughness, sheen)
#define kInstanceF4MaterialExt  24u   // (diffuseRoughness, specularIOR, coatIOR, coatDarkening)
#define kInstanceF4SpecularTint 25u   // (specular_color.rgb, specular_weight); all 0 ⇒ (1, 1, 1, 1)
#define kInstanceF4CoatTint     26u   // (coat_color.rgb, —); all 0 ⇒ (1, 1, 1)
#define kInstanceF4ThinFilm     27u   // (weight, thickness nm, film IOR, —); weight 0 ⇒ no film
#define kInstanceF4Translucency 28u   // (weight, colour.rgb); weight 0 ⇒ none, colour 0 ⇒ the albedo

/// One instance's layer parameters, decoded. `any == false` ⇒ every field is its legacy value and
/// the caller keeps its historical expression.
struct MaterialLayers {
    bool   any;
    /// The specular parameters are set (else the dielectric F0 is the legacy literal 0.04).
    bool   spec;
    /// DH-1309 — the instance's coat (`clearcoat`, `clearcoatRoughness`); 0 ⇒ none.
    float  coat;
    float  coatRoughness;
    /// DH-1298 — the coat's optics are set (else the coat is the DH-1309 default: IOR 1.5 / F0 0.04,
    /// clear, no darkening, and the base below keeps its own IOR — bit for bit).
    bool   coatParams;
    float  coatIOR;           // OpenPBR coat_ior (1.5)
    float3 coatColor;         // OpenPBR coat_color (1)
    float  coatDarkening;     // OpenPBR coat_darkening [0, 1] (0 here — see the Swift accessor)
    /// OpenPBR `specular_color × specular_weight` (1 ⇒ untinted, full strength).
    float3 specularTint;
    /// OpenPBR `specular_ior` (1.5 ⇒ F0 0.04).
    float  specularIOR;
    /// DH-1014 — OpenPBR `thin_film_weight` / `thin_film_thickness` (nm) / `thin_film_ior`. `film` false ⇒ none.
    bool   film;
    float  filmWeight;
    float  filmThicknessNm;
    float  filmIOR;
    /// DH-1302 — thin-sheet translucency: the fraction of the light reaching the FAR face that comes
    /// through (`transWeight`), tinted by `transColor` (the albedo when none was set). `trans` false ⇒ none.
    bool   trans;
    float  transWeight;
    float3 transColor;
    bool   transColorSet;
};

static inline MaterialLayers illumiMaterialLayersNone() {
    MaterialLayers m;
    m.any = false;
    m.spec = false;
    m.coat = 0.0;
    m.coatRoughness = 0.08;
    m.coatParams = false;
    m.coatIOR = 1.5;
    m.coatColor = float3(1.0);
    m.coatDarkening = 0.0;
    m.specularTint = float3(1.0);
    m.specularIOR = 1.5;
    m.film = false; m.filmWeight = 0.0; m.filmThicknessNm = 400.0; m.filmIOR = 1.4;
    m.trans = false; m.transWeight = 0.0; m.transColor = float3(1.0); m.transColorSet = false;
    return m;
}

/// Whether an instance carries anything the layer target must point at. Zero in every new field
/// (the Swift defaults, `illumiDefaultInstance`, a memset row) and no coat ⇒ false ⇒ the G-buffer
/// writes none. `coatRec` is the record's `kInstanceF4Coat` float4 (metallic, clearcoat,
/// clearcoatRoughness, sheen): a coated instance points at its record so the coat layer (DH-1309)
/// reads its own roughness on every lane, and a coat can coexist with a sheen (DH-1299).
static inline bool illumiHasMaterialLayers(float4 coatRec, float4 materialExt, float4 specularTint,
                                           float4 coatTint = float4(0.0), float4 thinFilm = float4(0.0),
                                           float4 translucency = float4(0.0)) {
    return coatRec.y > 0.0 || any(materialExt.yzw != float3(0.0)) || any(specularTint != float4(0.0))
        || any(coatTint.xyz != float3(0.0)) || thinFilm.x > 0.0 || translucency.x > 0.0;
}

static inline MaterialLayers illumiDecodeMaterialLayers(float4 coatRec, float4 materialExt, float4 specularTint,
                                                       float4 coatTint = float4(0.0), float4 thinFilm = float4(0.0),
                                                       float4 translucency = float4(0.0)) {
    MaterialLayers m = illumiMaterialLayersNone();
    if (!illumiHasMaterialLayers(coatRec, materialExt, specularTint, coatTint, thinFilm, translucency)) return m;
    if (translucency.x > 0.0) {
        m.trans = true;
        m.transWeight = saturate(translucency.x);
        m.transColorSet = any(translucency.yzw != float3(0.0));
        if (m.transColorSet) m.transColor = saturate(translucency.yzw);
    }
    if (thinFilm.x > 0.0) {
        m.film = true;
        m.filmWeight = saturate(thinFilm.x);
        if (thinFilm.y > 0.0) m.filmThicknessNm = thinFilm.y;
        if (thinFilm.z > 0.0) m.filmIOR = max(thinFilm.z, 1.0);
    }
    m.coatParams = materialExt.z != 0.0 || materialExt.w != 0.0 || any(coatTint.xyz != float3(0.0));
    if (materialExt.z > 0.0) m.coatIOR = materialExt.z;
    m.coatDarkening = saturate(materialExt.w);
    if (any(coatTint.xyz != float3(0.0))) m.coatColor = saturate(coatTint.xyz);
    m.any = true;
    m.spec = materialExt.y != 0.0 || any(specularTint != float4(0.0));
    m.coat = max(coatRec.y, 0.0);
    m.coatRoughness = coatRec.z > 0.0 ? coatRec.z : 0.08;
    if (materialExt.y > 0.0) m.specularIOR = materialExt.y;
    if (any(specularTint != float4(0.0))) m.specularTint = max(specularTint.rgb, 0.0) * max(specularTint.w, 0.0);
    return m;
}

/// The layer record at `slot` of an instance buffer viewed as float4s. `slot == kMaterialLayerNone`
/// or `slot >= count` (a TLAS id past the raster instances: a curve set, a glass row) ⇒ none.
static inline MaterialLayers illumiMaterialLayersAt(const device float4* inst, uint slot, uint count) {
    if (slot == kMaterialLayerNone || slot >= count || inst == nullptr) return illumiMaterialLayersNone();
    const device float4* r = inst + slot * kInstanceF4Stride;
    return illumiDecodeMaterialLayers(r[kInstanceF4Coat], r[kInstanceF4MaterialExt], r[kInstanceF4SpecularTint],
                                      r[kInstanceF4CoatTint], r[kInstanceF4ThinFilm], r[kInstanceF4Translucency]);
}

/// Normal-incidence Fresnel reflectance of a dielectric interface of relative index `n`.
static inline float illumiF0FromIOR(float n) {
    float r = (n - 1.0) / (n + 1.0);
    return r * r;
}

/// DH-1300 — the dielectric base's F0 (what `mix(F0, albedo, metallic)` mixes from): OpenPBR's
/// `specular_weight · F0(specular_ior)`, tinted by `specular_color`. EXACTLY the legacy literal
/// `float3(0.04)` when the instance carries no layer parameters.
///
/// The tint applies to F0, so the base below takes the complement per channel (the diffuse weight
/// is `1 − E_spec(F0)`, computed from this same F0): a white surface under a uniform light still
/// returns exactly 1 per channel whatever the tint — energy-conserving by construction. (OpenPBR
/// leaves the transmitted share untinted, which loses energy for a tinted specular; this engine
/// keeps the furnace exact instead.)
static inline float3 illumiDielectricF0(MaterialLayers m) {
    if (!m.spec) return float3(0.04);
    return saturate(m.specularTint * illumiF0FromIOR(max(m.specularIOR, 1.0)));
}

// ── The coat's optics (Daydream DH-1298 — OpenPBR coat_ior / coat_color / coat_darkening) ─────────
//
// • coat_ior sets the coat interface's F0, and the base below it no longer meets air but the coat:
//   its specular sees the RELATIVE index n_base / n_coat, blended by the coat's weight (OpenPBR §coat).
//   A clear lacquer (1.5) over a dielectric (1.5) leaves the base almost no reflection of its own —
//   all the gloss is the coat's — and water (1.33) over stone (1.56) leaves a little.
// • coat_color tints everything beneath the coat (the light crosses it on the way in and out).
// • coat_darkening: light the base scatters back up is partly trapped by internal reflection at the
//   coat's underside (total internal reflection beyond the critical angle) and returns to the base,
//   which absorbs a fraction again — so a coated diffuse base reads darker and more saturated, the
//   way wet stone or varnished wood does. The diffuse albedo ρ is replaced by ρ(1 − K)/(1 − ρK) with
//   K = 1 − (1 − Ē_c)/n_c² the coat's internal diffuse reflectance (Ē_c its hemispherical albedo,
//   here the Schlick average F0 + (1 − F0)/21) — OpenPBR's form, Lekner & Dorf 1988 — blended by
//   coat weight × coat_darkening. White (ρ = 1) is unchanged, so the furnace stays exact.

/// The coat interface's F0: the legacy literal 0.04 unless the coat's optics were set.
static inline float illumiCoatF0(MaterialLayers m) {
    return m.coatParams ? illumiF0FromIOR(max(m.coatIOR, 1.0)) : 0.04;
}

/// The base dielectric F0 under a coat of weight `w`: OpenPBR's relative index, n_base / n_coat,
/// blended by the coat weight. Exactly `illumiDielectricF0(m)` unless the coat's optics were set.
static inline float3 illumiDielectricF0UnderCoat(MaterialLayers m, float w) {
    if (!m.coatParams || !(w > 0.0)) return illumiDielectricF0(m);
    float nb = max(m.specularIOR, 1.0), nc = max(m.coatIOR, 1.0);
    float n = mix(nb, nb / nc, saturate(w));
    float3 tint = m.spec ? m.specularTint : float3(1.0);
    return saturate(tint * illumiF0FromIOR(n));
}

/// The base's diffuse darkening under the coat (see the section comment); float3(1) unless set.
static inline float3 illumiCoatDarkening(MaterialLayers m, float3 albedo, float w) {
    if (!m.coatParams || !(m.coatDarkening > 0.0) || !(w > 0.0)) return float3(1.0);
    float f0 = illumiCoatF0(m);
    float Eavg = f0 + (1.0 - f0) / 21.0;
    float nc = max(m.coatIOR, 1.0);
    float K = saturate(1.0 - (1.0 - Eavg) / (nc * nc));
    float3 rho = saturate(albedo);
    float3 d = (1.0 - K) / max(1.0 - rho * K, 1e-4);
    return mix(float3(1.0), d, saturate(w * m.coatDarkening));
}

/// The coat tint: float3(1) unless set.
static inline float3 illumiCoatColor(MaterialLayers m) { return m.coatParams ? m.coatColor : float3(1.0); }

// ── Thin-film iridescence on opaque surfaces (Daydream DH-1014 — OpenPBR thin_film_*) ─────────────
//
// A film of index n2 and thickness d on a base of index n3 reflects by two-beam interference
// (Airy): with r12 = (1 − n2)/(1 + n2), r23 = (n2 − n3)/(n2 + n3) and the phase
// δ(λ) = 4π·n2·d·cosθ2 / λ, the film's reflectance is
//     R(λ) = (r12² + r23² + 2·r12·r23·cos δ) / (1 + r12²·r23² + 2·r12·r23·cos δ),
// with cosθ2 from Snell inside the film. Evaluated at the three channel wavelengths it is a colour
// that SHIFTS WITH VIEW ANGLE (the oil-slick look). It is applied as a TINT on the base's F0 — the
// Filament / Belcour–Barla form: Schlick's F0 is replaced by the film's reflectance at N·V — so one
// per-pixel evaluation serves every lobe. The tint is normalised to unit MEAN across the channels
// (the film redistributes the reflected energy between channels, it does not invent any), and
// blended in by `thin_film_weight`; the caller saturates the resulting F0, so no lobe exceeds 1.
// Real-valued Fresnel coefficients (no polarisation split) — the standard real-time simplification.
static inline float3 illumiThinFilmTint(MaterialLayers m, float cosTheta, float metallic) {
    if (!m.film) return float3(1.0);
    float n2 = max(m.filmIOR, 1.0);
    float n3 = mix(max(m.specularIOR, 1.0), 2.8, saturate(metallic));
    float sin2 = saturate(1.0 - cosTheta * cosTheta);
    float cos2 = sqrt(saturate(1.0 - sin2 / (n2 * n2)));
    float r12 = (1.0 - n2) / (1.0 + n2);
    float r23 = (n2 - n3) / (n2 + n3);
    const float3 lambda = float3(610.0, 550.0, 465.0);        // R, G, B channel wavelengths (nm)
    float3 cd = cos(12.5663706 * n2 * m.filmThicknessNm * cos2 / lambda);   // cos δ, 4π = 12.566…
    float  c  = 2.0 * r12 * r23;
    float3 R  = (r12 * r12 + r23 * r23 + c * cd) / (1.0 + r12 * r12 * r23 * r23 + c * cd);
    float  mean = max((R.x + R.y + R.z) * (1.0 / 3.0), 1e-5);
    return mix(float3(1.0), R / mean, m.filmWeight);
}
