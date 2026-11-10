// IlluminatoramaLightBounds.metal — per-light screen bounds for the deferred lighting kernel (Daydream DH-1625).
//
// `illumi_lighting` walks every point, spot and area light for every pixel. Each light has an exact reach —
// outside it the light contributes exactly 0 — but finding that out costs the pixel a distance (and for a cone a
// normalise, a dot and a smoothstep) per light. On 4000 Sunset (34 cones, 21 points, 17 portals) those loops were
// 4.0 ms of a 5.2 ms orbit lighting pass and 13.3 ms of a 17.4 ms settled one, though a pixel is lit by a handful.
//
// This kernel runs once per frame, one thread per LIGHT: it boxes the light's reach in world space, projects the
// box's eight corners with the frame's own `viewProjection` (the matrix whose inverse the lighting kernel uses to
// rebuild a pixel's world position, so the rectangle is in that kernel's NDC) and writes
// `float4(minX, minY, maxX, maxY)`. The lighting kernel skips a light whose rectangle does not contain the pixel
// before reading the light at all. Conservative by construction, so the picture is unchanged:
//   point  — the ball `radius`                       (`if (dist > pl.radius) continue`)
//   area   — the ball `radius` about the centre      (`evalAreaLight`: `if (dist > al.radius) return 0`)
//   spot   — the cylinder about the axis that holds BOTH laws' lit sets: the apex law's spherical sector
//            (axial 0…R, lateral ≤ R·sin θ) and the opening-disc law's (range R from the disc's centre at
//            `openingOffset`; every tap lies on the disc, so lateral ≤ r + (R + r)·sin θ). A cone of half-angle
//            ≥ ~87° falls back to the ball `radius + |openingOffset|`.
// A box that reaches behind the eye gets the whole screen. Layout: points, then spots, then area lights.
#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

constant float4 kLightBoundsWholeScreen = float4(-2.0, -2.0, 2.0, 2.0);
// NDC margin: the TAA jitter (≤ 1 px) plus the pixel's own footprint at any resolution this renderer runs
// (0.01 NDC = 2.6 px at 528 wide).
constant float kLightBoundsMargin = 0.01;

static inline float4 lightBoundsOfBox(float3 lo, float3 hi, float4x4 vp) {
    if (!all(isfinite(lo)) || !all(isfinite(hi))) return kLightBoundsWholeScreen;
    float2 mn = float2(1e30), mx = float2(-1e30);
    for (uint k = 0u; k < 8u; ++k) {
        float3 c = float3((k & 1u) != 0u ? hi.x : lo.x, (k & 2u) != 0u ? hi.y : lo.y, (k & 4u) != 0u ? hi.z : lo.z);
        float4 clip = vp * float4(c, 1.0);
        if (clip.w <= 1e-4) return kLightBoundsWholeScreen;
        float2 p = clip.xy / clip.w;
        mn = min(mn, p); mx = max(mx, p);
    }
    return float4(mn - kLightBoundsMargin, mx + kLightBoundsMargin);
}

kernel void illumi_light_bounds(
    constant FrameUniforms&   frame       [[buffer(0)]],
    const device PointLight*  pointLights [[buffer(1)]],
    const device SpotLight*   spotLights  [[buffer(3)]],
    const device AreaLight*   areaLights  [[buffer(4)]],
    device float4*            lightBounds [[buffer(12)]],
    constant uint&            enabled     [[buffer(13)]],
    uint                      tid         [[thread_position_in_grid]])
{
    uint nP = frame.pointLightCount, nS = frame.spotLightCount, nA = frame.areaLightCount;
    if (tid >= nP + nS + nA) return;
    if (enabled == 0u) { lightBounds[tid] = kLightBoundsWholeScreen; return; }
    float3 lo, hi;
    if (tid < nP) {
        float3 c = pointLights[tid].position; float r = pointLights[tid].radius;
        lo = c - r; hi = c + r;
    } else if (tid < nP + nS) {
        uint i = tid - nP;
        float3 pos = spotLights[i].position;
        float  R = spotLights[i].radius, off = spotLights[i].openingOffset;
        float  ro = max(spotLights[i].openingRadius, 0.0), oc = spotLights[i].outerCone;
        float3 d = spotLights[i].direction;
        float  dl = length(d);
        if (oc <= 0.05 || dl < 1e-6) {
            float r = R + abs(off) + ro;
            lo = pos - r; hi = pos + r;
        } else {
            float3 ax = d / dl;
            float  rho = ro + (R + ro) * sqrt(max(1.0 - oc * oc, 0.0));
            float3 a0 = pos + ax * min(off, 0.0), a1 = pos + ax * (max(off, 0.0) + R);
            float3 ext = rho * sqrt(max(1.0 - ax * ax, 0.0));   // a disc ⊥ ax, per world axis
            lo = min(a0, a1) - ext; hi = max(a0, a1) + ext;
        }
    } else {
        uint i = tid - nP - nS;
        float3 c = areaLights[i].center; float r = areaLights[i].radius;
        lo = c - r; hi = c + r;
    }
    lightBounds[tid] = lightBoundsOfBox(lo, hi, frame.viewProjection);
}
