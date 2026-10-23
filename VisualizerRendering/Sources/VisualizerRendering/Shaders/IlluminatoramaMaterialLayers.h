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

/// `GBufferOut.materialLayer`'s clear value and the "no layer parameters" slot.
#define kMaterialLayerNone 0xFFFFFFFFu

/// The `Instance` record as float4s (static_asserted in IlluminatoramaCommon.h).
#define kInstanceF4Stride       26u   // sizeof(Instance) / 16
#define kInstanceF4Coat          9u   // (metallic, clearcoat, clearcoatRoughness, sheen)
#define kInstanceF4MaterialExt  24u   // (diffuseRoughness, specularIOR, —, —)
#define kInstanceF4SpecularTint 25u   // (specular_color.rgb, specular_weight); all 0 ⇒ (1, 1, 1, 1)

/// One instance's layer parameters, decoded. `any == false` ⇒ every field is its legacy value and
/// the caller keeps its historical expression.
struct MaterialLayers {
    bool   any;
    /// OpenPBR `specular_color × specular_weight` (1 ⇒ untinted, full strength).
    float3 specularTint;
    /// OpenPBR `specular_ior` (1.5 ⇒ F0 0.04).
    float  specularIOR;
};

static inline MaterialLayers illumiMaterialLayersNone() {
    MaterialLayers m;
    m.any = false;
    m.specularTint = float3(1.0);
    m.specularIOR = 1.5;
    return m;
}

/// Whether an instance carries anything the layer target must point at. Zero in every new field
/// (the Swift defaults, `illumiDefaultInstance`, a memset row) ⇒ false ⇒ the G-buffer writes none.
static inline bool illumiHasMaterialLayers(float4 materialExt, float4 specularTint) {
    return materialExt.y != 0.0 || any(specularTint != float4(0.0));
}

static inline MaterialLayers illumiDecodeMaterialLayers(float4 materialExt, float4 specularTint) {
    MaterialLayers m = illumiMaterialLayersNone();
    if (!illumiHasMaterialLayers(materialExt, specularTint)) return m;
    m.any = true;
    if (materialExt.y > 0.0) m.specularIOR = materialExt.y;
    if (any(specularTint != float4(0.0))) m.specularTint = max(specularTint.rgb, 0.0) * max(specularTint.w, 0.0);
    return m;
}

/// The layer record at `slot` of an instance buffer viewed as float4s. `slot == kMaterialLayerNone`
/// or `slot >= count` (a TLAS id past the raster instances: a curve set, a glass row) ⇒ none.
static inline MaterialLayers illumiMaterialLayersAt(const device float4* inst, uint slot, uint count) {
    if (slot == kMaterialLayerNone || slot >= count || inst == nullptr) return illumiMaterialLayersNone();
    const device float4* r = inst + slot * kInstanceF4Stride;
    return illumiDecodeMaterialLayers(r[kInstanceF4MaterialExt], r[kInstanceF4SpecularTint]);
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
    if (!m.any) return float3(0.04);
    return saturate(m.specularTint * illumiF0FromIOR(max(m.specularIOR, 1.0)));
}
