#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA SNOW ──────────────────────────────────────────────────────
//
// GPU falling-snow field. Each flake is a real inertial particle: its velocity RELAXES toward
// (local wind + its own terminal fall speed + flutter) with time constant `relaxTau` — a Stokes-
// drag first-order response, so a gust accelerates flakes over ~half a second instead of
// teleporting them (the kinematic "velocity := field" advection that made the dust field read as
// a jet stream, see IlluminatoramaDust.metal). Terminal speed varies per flake (dry powder
// ~0.5 m/s .. wet aggregates ~1.4 m/s) and a small tumbling flutter rides on top. Wind is a slow
// spatial gust field on a base wind, not a uniform shove.
//
// Colour is the flake's own shading, computed here from the real scene lights handed in by the
// host (a cold moon term + up to eight warm point lights, e.g. a string of tree lights) — so a
// flake that drifts past the tree glows amber and the same flake in the dark is moon-blue.
// The renderer draws the buffers as additive HDR point sprites at a fixed pixel size; a flake
// smaller than that on screen has its energy scaled by its true projected area instead
// (sub-pixel energy conservation), so distant snow is a fine haze, not a field of bright dots.

struct SnowUniforms {
    float dt; float time; uint count; float relaxTau;
    float4 boundsMin;   // xyz, w = mean terminal fall speed (m/s)
    float4 boundsMax;   // xyz, w = turbulence (m/s)
    float4 wind;        // xyz base wind (m/s), w = gust amplitude (fraction of base, 0..1)
    float4 cam;         // xyz camera position, w = pixel angle (rad per pixel)
    float4 moon;        // rgb moonlit-flake radiance, w = flake diameter (m)
    float4 pointPx;     // x = point size in pixels, y = edge fade width (m), z = flutter (m/s), w = sparkle
    float4 lightPos[8];
    float4 lightCol[8]; // rgb radiant intensity (pre-multiplied); 0 = unused
};

static inline float hash13(float3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
static inline float vnoise(float3 x) {
    float3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i), n100 = hash13(i + float3(1,0,0)), n010 = hash13(i + float3(0,1,0)), n110 = hash13(i + float3(1,1,0));
    float n001 = hash13(i + float3(0,0,1)), n101 = hash13(i + float3(1,0,1)), n011 = hash13(i + float3(0,1,1)), n111 = hash13(i + float3(1,1,1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z) * 2.0 - 1.0;
}
static inline float3 noiseVec3(float3 x) {
    return float3(vnoise(x), vnoise(x + float3(31.4, 17.2, 47.1)), vnoise(x + float3(-19.3, 53.7, 11.9)));
}
static inline float wrapAxis(float v, float lo, float hi) {
    float range = hi - lo;
    if (range <= 1e-5) return v;
    float t = (v - lo) / range;
    return lo + (t - floor(t)) * range;
}

kernel void illumi_snow_step(
    device float4*          positions  [[buffer(0)]],   // xyz, seed
    device float4*          velocities [[buffer(1)]],   // xyz, unused
    device float4*          colors     [[buffer(2)]],
    constant SnowUniforms&  u          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.count) return;
    float3 p = positions[gid].xyz;
    float seed = positions[gid].w;
    float3 v = velocities[gid].xyz;

    float h1 = hash13(float3(seed * 91.7, 3.1, 7.7));
    float h2 = hash13(float3(seed * 17.3, 11.9, 5.3));

    // Terminal fall speed: per flake, 0.45x .. 1.45x the mean.
    float fall = u.boundsMin.w * (0.45 + 1.0 * h1);

    // Wind: a base wind modulated by a slow, large-scale gust field (not a uniform shove).
    float gust = vnoise(p * 0.12 + float3(0.0, 0.0, u.time * 0.35));
    float3 wind = u.wind.xyz * (1.0 + u.wind.w * gust);

    // Small-scale turbulence: curl of a churning noise field (divergence-free, fast churn so it
    // has no permanent attractors for flakes to pool on).
    const float e = 0.15;
    float3 q = p * 0.45 + float3(0.0, u.time * 0.5, 0.0);
    float3 dx = float3(e,0,0), dy = float3(0,e,0), dz = float3(0,0,e);
    float3 px0 = noiseVec3(q - dx), px1 = noiseVec3(q + dx);
    float3 py0 = noiseVec3(q - dy), py1 = noiseVec3(q + dy);
    float3 pz0 = noiseVec3(q - dz), pz1 = noiseVec3(q + dz);
    float3 curl = float3((py1.z - py0.z) - (pz1.y - pz0.y),
                         (pz1.x - pz0.x) - (px1.z - px0.z),
                         (px1.y - px0.y) - (py1.x - py0.x)) / (2.0 * e);
    float3 turb = curl * u.boundsMax.w;

    // Flutter: a flake is a tumbling plate, so it swings side to side as it falls.
    float ph = u.time * (2.0 + 3.0 * h2) + seed * 6.2831;
    float3 flutter = float3(cos(ph), 0.25 * sin(ph * 1.7), sin(ph * 0.83)) * u.pointPx.z * (0.4 + 0.8 * h2);

    float3 target = wind + turb + flutter + float3(0.0, -fall, 0.0);
    float k = 1.0 - exp(-u.dt / max(u.relaxTau, 1e-3));
    v += (target - v) * k;
    p += v * u.dt;

    p.x = wrapAxis(p.x, u.boundsMin.x, u.boundsMax.x);
    p.y = wrapAxis(p.y, u.boundsMin.y, u.boundsMax.y);
    p.z = wrapAxis(p.z, u.boundsMin.z, u.boundsMax.z);
    positions[gid]  = float4(p, seed);
    velocities[gid] = float4(v, 0.0);

    // ── Shading ──────────────────────────────────────────────────
    float3 lit = u.moon.rgb;
    for (int i = 0; i < 8; ++i) {
        float3 L = u.lightPos[i].xyz - p;
        float d2 = dot(L, L) + 0.15;
        lit += u.lightCol[i].rgb / d2;
    }
    // Sparkle: a faceted crystal flashes as it tumbles.
    float sp = 1.0 + u.pointPx.w * (pow(max(0.0, sin(ph * 1.3 + h1 * 40.0)), 8.0) * 3.0 - 0.4);
    lit *= max(0.2, sp) * (0.8 + 0.4 * h2);

    // Sub-pixel energy conservation: true projected size vs the fixed-size sprite.
    float dist = max(0.05, length(p - u.cam.xyz));
    float apparentPx = u.moon.w / (dist * max(u.cam.w, 1e-6));
    float cover = min(1.0, (apparentPx * apparentPx) / max(u.pointPx.x * u.pointPx.x, 1e-3));

    // Fade at the volume's edges so the wrap never pops.
    float3 lo = p - u.boundsMin.xyz, hi = u.boundsMax.xyz - p;
    float edge = min(min(min(lo.x, hi.x), min(lo.z, hi.z)), min(lo.y, hi.y));
    float fade = smoothstep(0.0, max(u.pointPx.y, 1e-3), edge);

    colors[gid] = float4(lit * cover * fade, 1.0);
}
