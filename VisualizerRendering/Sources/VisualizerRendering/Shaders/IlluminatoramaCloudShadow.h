// IlluminatoramaCloudShadow.h — the cumulus deck's shadow on the scene (Daydream DH-1462).
//
// ONE buffer of floats, written by `volSkyCloudShadowMap` (VolumetricSky.metal) from the SAME density
// field the sky marches (`cumulusDensityHD`, the same extinction), and read by every lane that lights
// with the sun: the deferred lighting pass, the RT direct sun, path-lane vertices and secondary hits
// (GI bounces, reflections, glass). It is the transmittance of the deck along the sun direction over
// a square of world XZ at `groundY`, so a point at any height looks it up by sliding down the sun ray
// to that plane (exact below the cloud base). Low-res and baked only when the sky or sun changes, so
// the term is smooth (no per-pixel march, no noise) and costs one bilinear tap.
//
//   [0] centre X   [1] centre Z   [2] extent (m)      [3] enabled (> 0)
//   [4] resolution [5] groundY    [6] strength (0…1)  [7] height limit above groundY (0 = none)
//   [8 …] resolution² transmittances, row-major (row = Z, column = X)
//
// Disabled (or the 8-float dummy): every lookup is exactly 1.
#pragma once
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kCloudShadowHeader = 8u;

inline float illumiCloudShadowTexel(const device float* cs, int x, int y, int res) {
    x = clamp(x, 0, res - 1); y = clamp(y, 0, res - 1);
    return cs[kCloudShadowHeader + uint(y * res + x)];
}

/// Transmittance of the cloud deck between `P` and the sun (`Ld` = unit direction TOWARD the sun).
inline float illumiCloudSunTransmittance(const device float* cs, float3 P, float3 Ld) {
    if (cs[3] <= 0.0f || Ld.y <= 0.02f) return 1.0f;
    float h = P.y - cs[5];
    float limit = cs[7];
    float hw = limit > 0.0f ? 1.0f - smoothstep(limit, limit + 0.5f, h) : 1.0f;
    if (hw <= 0.0f) return 1.0f;
    float ext = max(cs[2], 1.0f);
    int res = max(int(cs[4]), 2);
    float2 g = P.xz - Ld.xz * (h / Ld.y);
    float2 uv = (g - float2(cs[0], cs[1])) / ext + 0.5f;
    float edge = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y));
    if (edge <= 0.0f) return 1.0f;
    float2 p = uv * float(res) - 0.5f;
    int2 p0 = int2(floor(p));
    float2 f = p - float2(p0);
    float t = mix(mix(illumiCloudShadowTexel(cs, p0.x, p0.y, res), illumiCloudShadowTexel(cs, p0.x + 1, p0.y, res), f.x),
                  mix(illumiCloudShadowTexel(cs, p0.x, p0.y + 1, res), illumiCloudShadowTexel(cs, p0.x + 1, p0.y + 1, res), f.x),
                  f.y);
    t = mix(1.0f, t, smoothstep(0.0f, 0.08f, edge));   // no hard border where the map ends
    return mix(1.0f, t, saturate(cs[6]) * hw);
}
