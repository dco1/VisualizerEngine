// ── ILLUMINATORAMA — THIN-LEAF OPTICS (shared) ────────────────────────────────────────────────
//
// Signack S5 (Daydream DH-1461 / DH-1361): what a thin leaf does with light that reaches its FAR
// face. A leaf is a pigmented slab a fraction of a millimetre thick; most of what enters is
// scattered by the mesophyll and leaves either face roughly diffusely. Seen from the dark side, a
// sunlit leaf therefore glows with TRANSMITTED light — the "backlit canopy" that makes a tree read
// as alive rather than as a cutout. This header is the one model of that, read by every lane:
//
//   • the deferred lighting kernel (live canvas + photo lane) — `leafSunTransmission` for the sun
//     behind the leaf, `leafTransmittance` × the irradiance cube along −N for the sky behind it;
//   • the path lane's vertices (IlluminatoramaRTInstanced.metal) — `thinSheetTransmittance`, the
//     same pigment model at the default thickness, for every instance that carries a thin
//     transmission ratio (`RTInstanceData.nrm2.w`), so light passes THROUGH a canopy in an export.
//
// Physics, in three numbers:
//   1. τ/ρ — the transmittance-to-reflectance ratio at the pigment's own peak (green for a leaf).
//      Broadleaf measurements (LOPEX / PROSPECT) put it at 0.6–1.0 in the visible; the host
//      passes it (`FrameUniforms.leafTransmittanceRatio`, a per-instance ratio on the path lane).
//   2. Colour: light that crosses the leaf travels further through the pigment than light
//      scattered back out of the near face (which also went some way in), so the transmitted
//      spectrum is the reflected one × the pigment's normalised transmittance over the EXTRA
//      path — about half the blade per unit thickness: aN^(d/2), aN the albedo over its peak. Chlorophyll absorbs red and blue, so this is why a
//      backlit leaf is a SATURATED yellow-green and not the dull green of its sunlit face.
//   3. Thickness d (1 = the reference blade): thicker → deeper and darker, thinner → paler and
//      yellower. A leaf can carry its own d through the G-buffer class tag (see `leafThickness`).
//
// Angular shape: mostly Lambertian (the mesophyll scrambles direction) plus a mild forward lobe
// about the straight-through direction, normalised so the leaf never re-emits more than τ·E.
//
// LIMITS (stated, per DH-1361): the sky term reads the outdoor irradiance cube along −N attenuated
// by the near side's ambient occlusion — no traced visibility of what is behind the leaf; inside a
// dense crown it stands in for the inner leaves' own shadowing. The path lane traces it at
// secondary vertices; at the PRIMARY every lane uses the deferred estimate, so all three agree.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// Largest class-tag value a foliage pixel may carry: the tag must stay < 0.5 (every foliage test
/// in the engine is `normalRoughness.w < 0.5`). (0, kLeafThicknessCodeMax] encodes a thickness.
#define kLeafThicknessCodeMax 0.45f

/// A foliage pixel is shaded with this model when its tag carries a thickness (> this) and the
/// frame's `leafTransmittanceRatio` > 0; tag 0 (grass, legacy writers) keeps the #58 backlight.
#define kLeafPhysicalTagMin 0.004f

/// Forward-lobe share and exponent of the transmitted light (the rest is Lambertian).
constant float kLeafForwardShare = 0.35f;
constant float kLeafForwardPower = 4.0f;

/// The leaf's relative thickness from its G-buffer class tag: 0 (every legacy foliage writer —
/// grass ribbons, tile fields, plants) is the reference blade d = 1; a vertex alpha in
/// (0, kLeafThicknessCodeMax] maps linearly onto d ∈ [0.6, 1.4].
static inline float leafThickness(float tag) {
    if (tag <= 0.004f) return 1.0f;
    return mix(0.6f, 1.4f, saturate((tag - 0.05f) / (kLeafThicknessCodeMax - 0.05f)));
}

/// τ — the leaf's diffuse transmittance (the fraction of irradiance on the far face that leaves
/// the near face), from its reflectance colour `albedo`, its relative thickness `d` and the τ/ρ
/// ratio at the pigment peak. Never more than `ratio × albedo`; neutral albedo ⇒ `ratio × albedo`
/// (a white sheet has no pigment to saturate through), so this IS also the thin-sheet model.
static inline float3 leafTransmittance(float3 albedo, float d, float ratio) {
    float m = max(max(albedo.r, albedo.g), max(albedo.b, 1e-4f));
    float3 aN = saturate(albedo / m);
    return ratio * albedo * pow(aN, float3(0.5f * d)) * (2.0f / (1.0f + d));
}

/// The path lane's thin sheet: the same pigment model at the reference thickness.
static inline float3 thinSheetTransmittance(float3 albedo, float ratio) {
    return leafTransmittance(albedo, 1.0f, ratio);
}

/// Radiance leaving the near face toward `V` from a sun of irradiance-colour `sunColor` behind the
/// leaf (direction to the sun `Ld`, near-face normal `N` facing the viewer), shadowed by
/// `visibility`. Lambert + the normalised forward lobe; 0 when the sun is on the viewer's side.
static inline float3 leafSunTransmission(float3 N, float3 V, float3 Ld, float3 sunColor,
                                         float visibility, float3 tau) {
    float back = saturate(dot(-N, Ld));
    if (back <= 0.0f || visibility <= 0.0f) return float3(0.0f);
    float through = pow(saturate(dot(V, -Ld)), kLeafForwardPower);
    float shape = (1.0f - kLeafForwardShare) + kLeafForwardShare * 0.5f * (kLeafForwardPower + 1.0f) * through;
    return tau * (1.0f / M_PI_F) * sunColor * (back * visibility * shape);
}

/// The path lane's thin-transmission lane: `RTInstanceData.nrm2.w` = 1 + ratio when the row has no
/// clearcoat (a coat row is negative, an untouched row 0). Returns the τ/ρ ratio, 0 for none.
static inline float thinSheetRatioFromLane(float w) {
    return w >= 1.0f ? min(w - 1.0f, 1.0f) : 0.0f;
}
