// ── STARFIELD WASHOUT ────────────────────────────────────────────────────────
//
// Per-tick kernel that attenuates star colours based on the burst-light field.
// Reads a pristine baseColor buffer + the star positions + the BurstLight
// buffer; writes the attenuated colour to the renderColor buffer SceneKit
// binds to the starfield's `.color` semantic. The pristine base never gets
// touched by the kernel; that's the source of truth from FireworksPlusStarfield
// .populate().
//
// Why a kernel and not a CPU memcopy + tweak each tick: with 4096 stars and
// a dozen active bursts, doing the inner attenuation loop on the CPU per
// frame burns budget for no good reason. On the GPU this is a few µs.

#include <metal_stdlib>
using namespace metal;

struct BurstLight {
    float4 positionIntensity;
    float4 colorAge;
    float4 radiusLifePad;
};

struct StarWashUniforms {
    float4 counts;          // x = star count (uint bits), y = burst count (uint bits)
                            // z = washout reach scale (radius multiplier),
                            // w = washout strength (0..1 max attenuation)
};

kernel void starApplyBurstWashout(
    device const float4*       basePositions [[ buffer(0) ]],
    device const float4*       baseColors    [[ buffer(1) ]],
    device       float4*       outColors     [[ buffer(2) ]],
    device const BurstLight*   bursts        [[ buffer(3) ]],
    constant     StarWashUniforms& u          [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint starCount  = as_type<uint>(u.counts.x);
    uint burstCount = as_type<uint>(u.counts.y);
    if (id >= starCount) return;

    float reachScale = u.counts.z;
    float maxStrength = u.counts.w;

    float4 base = baseColors[id];
    if (burstCount == 0u) {
        outColors[id] = base;
        return;
    }

    // Project the star onto a unit direction from the world origin (stars
    // live on a 250 m sphere) and compare that direction with each burst's
    // direction-from-origin. Angular proximity to a burst → wash the star.
    // This is much cheaper (and more correct for an "infinity-far" sky)
    // than world-space distance.
    float3 sp = basePositions[id].xyz;
    float spLen = length(sp);
    if (spLen < 1e-3f) {
        outColors[id] = base;
        return;
    }
    float3 starDir = sp / spLen;

    float maxWash = 0.0f;
    for (uint i = 0u; i < burstCount; ++i) {
        BurstLight L = bursts[i];
        float intensity = L.positionIntensity.w;
        float maxR      = L.radiusLifePad.x;
        float life      = L.radiusLifePad.y;
        if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

        float age = L.colorAge.a;
        if (age >= life) continue;

        float3 bp = L.positionIntensity.xyz;
        float bpLen = length(bp);
        if (bpLen < 1e-3f) continue;
        float3 burstDir = bp / bpLen;

        // Cosine angular distance; convert to "near = 1, far = 0" via the
        // burst's reach scale (an angular size proxy — 30° gives a generous
        // 0.5 cos, 5° gives 0.996 cos).
        float cosA = dot(starDir, burstDir);
        // The "reach" controls how angularly large each burst's washout
        // halo is. cosA > cosineReach means "this star is inside the halo."
        // We translate to a 0..1 falloff.
        float cosReach = cos(reachScale * 0.4f);  // reachScale=1 → ~23° halo
        if (cosA < cosReach) continue;
        float t = (cosA - cosReach) / (1.0f - cosReach);  // 0 at edge, 1 at centre

        // Burst age decay
        float ageT = age / life;
        float ageScale;
        if (ageT < 0.08f) {
            ageScale = 1.0f;
        } else {
            float k = (ageT - 0.08f) / 0.92f;
            ageScale = (1.0f - k);
            ageScale = ageScale * ageScale;
        }

        float wash = t * ageScale * (intensity / 8.0f);  // normalise so default intensity=8 → 1.0
        maxWash = max(maxWash, wash);
    }

    float attenuation = 1.0f - saturate(maxWash) * maxStrength;
    outColors[id] = float4(base.rgb * attenuation, base.a);
}
