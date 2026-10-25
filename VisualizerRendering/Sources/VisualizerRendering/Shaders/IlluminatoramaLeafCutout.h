// IlluminatoramaLeafCutout.h — a species leaf cut out of a foliage card, per pixel (Daydream DH-1487).
//
// A tree's crown-FILL leaves are 4-triangle cards (tens of thousands per tree). Six outline points
// cannot carry a lobed leaf: a maple or oak outline in geometry is straight-edged spikes. So the card
// stays a roomy hexagon and the leaf is a smooth outline evaluated in the card's own coordinates,
// carried by the vertex's `cardUV`:
//
//   cardUV = (0, 0)              no cutout (every mesh but a fill card)
//   cardUV = (u, v + 2·code)     u ∈ [−1, 1] across the card in half-widths, v ∈ [0, 1] base → tip;
//                                code 1 oak · 2 maple · 3 birch · 4 elliptic (citrus, elderberry)
//
// Read by the G-buffer's cutout variant (`kLeafCutout`) and the shadow-map cutout stage, so a leaf's
// shadow has the leaf's outline. The host mirror is `IlluminatoramaLeafCutout` (Swift): the same
// functions, for tests and for the bake's card scale (each leaf covers the area the ovate card did).
// Self-contained: metal_stdlib only.
#pragma once
#include <metal_stdlib>
using namespace metal;

inline float illumiCutoutSmooth(float a, float b, float x) {
    float t = saturate((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

/// Is the card point `cardUV` inside its species' leaf? (Always true off a fill card.)
inline bool illumiLeafCutoutKeep(float2 cardUV) {
    int code = int(floor(cardUV.y * 0.5f));
    if (code <= 0) return true;
    float v = cardUV.y - 2.0f * float(code);
    float u = cardUV.x, au = abs(u);
    if (v <= 0.0f || v >= 1.0f) return false;
    if (code == 1) {        // OAK — obovate, four irregular rounded lobes a side, sinuses deepening upward
        float env = 0.90f * pow(sin(M_PI_F * pow(v, 1.55f)), 0.85f);
        float side = u > 0.0f ? 0.035f : -0.035f;           // the two sides' lobes don't line up
        float lobe = 0.5f + 0.5f * cos(2.0f * M_PI_F * (v - 0.14f + side) / 0.19f);
        float depth = 0.30f + 0.22f * illumiCutoutSmooth(0.2f, 0.7f, v);
        float w = env * (1.0f - depth + depth * sqrt(lobe)) * illumiCutoutSmooth(0.0f, 0.07f, v);
        return au < w;
    }
    if (code == 2) {        // MAPLE — palmate: five acuminate lobes, U sinuses, a tooth pair per lobe
        float x = u * 0.494f, y = v - 0.48f;                 // physical frame (half-width / length)
        float r = length(float2(x, y)), a = abs(atan2(x, y));
        const float r0 = 0.22f, spike = 0.35f, tooth = 0.10f;
        const float3 lobes[3] = { float3(0.0f, 0.30f, 0.70f), float3(1.12f, 0.26f, 0.66f), float3(2.2f, 0.11f, 0.5f) };
        float rb = r0;
        for (int i = 0; i < 3; ++i) {
            float d = abs(a - lobes[i].x) / lobes[i].z;
            if (d < 1.0f) {
                float p = 1.0f - d, A = lobes[i].y;
                rb = max(rb, r0 + A * p * (1.0f - spike) + A * spike * pow(p, 7.0f)
                             + tooth * A * max(0.0f, 1.0f - abs(d - 0.5f) / 0.1f));
            }
        }
        if (a > 2.75f) rb = min(rb, r0 * (0.55f + 0.45f * (M_PI_F - a) / 0.39f));   // petiole notch
        return r < rb;
    }
    if (code == 3) {        // BIRCH — ovate, acuminate, serrate
        float env = 0.86f * sin(M_PI_F * pow(v, 1.25f)) * (1.0f - 0.25f * v);
        return au < env * (1.0f - 0.08f * fract(v * 18.0f));
    }
    return au < 0.80f * sin(M_PI_F * pow(v, 0.92f));       // ELLIPTIC
}
