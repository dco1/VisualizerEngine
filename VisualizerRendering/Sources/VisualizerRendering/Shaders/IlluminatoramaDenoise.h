#pragma once

// ── ILLUMINATORAMA — DENOISE COLOUR SPACE ───────────────────────────────────
//
// YCoCg is the working space for every temporal history clamp in the engine —
// TAA, SSR temporal and RT-GI temporal all clip in it, so the transform pair
// is shared rather than copied three times.

#include <metal_stdlib>
using namespace metal;


// YCoCg ↔ RGB helpers. The lossless "scaled" variant (sum of absolute
// weights = 1) matches the float range of the history HDR buffer exactly.
static inline float3 RGBtoYCoCg(float3 c) {
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,   // Y  ∈ [0,1]
         0.5  * c.r              - 0.5  * c.b,    // Co ∈ [-0.5, 0.5]
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b     // Cg ∈ [-0.5, 0.5]
    );
}
static inline float3 YCoCgtoRGB(float3 c) {
    float t = c.x - c.z;           // Y - Cg
    // max(0) instead of saturate — HDR pipeline, no upper clamp needed.
    return max(float3(0.0), float3(t + c.y, c.x + c.z, t - c.y));
}
