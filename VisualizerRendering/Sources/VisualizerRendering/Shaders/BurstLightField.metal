// ── BURST LIGHT FIELD ────────────────────────────────────────────────────────
//
// Shared GPU sampler for "what colour glow is the show casting on this world
// position?". Built by `BurstLightField.swift` on the host; consumed by ocean
// glow, smoke, starfield washout, and any other downstream kernel that wants
// to be lit by the fireworks rather than a static light rig.
//
// `sampleBurstField` is defined as an inline function so consumer kernels can
// just call it (the metal compiler links across translation units in the
// same library — no header bookkeeping required).
//
// The 48-byte BurstLight struct mirrors the Swift `BurstLight` exactly.
// All shared fields are float4 per the project ALIGNMENT RULE.

#include <metal_stdlib>
using namespace metal;

struct BurstLight {
    float4 positionIntensity;  // xyz = world position (m), w = peak intensity (linear)
    float4 colorAge;           // rgb = base burst colour,   a = age (s) since detonate
    float4 radiusLifePad;      // x = max influence radius (m),
                               // y = lifespan (s) — burst fully decayed by this age,
                               // zw = pad
};

// Sum the burst-glow contribution at `worldPos` across every active burst.
// Falloff: quadratic radial × quadratic age-decay (peak at age 0, zero at
// lifespan). Returns a linear RGB additive contribution suitable for adding
// to any consumer's emission / diffuse output.
//
// The kernel calling this is responsible for passing the actual live count
// — the rest of the buffer is left untouched between bursts.
inline float3 sampleBurstField(
    float3 worldPos,
    device const BurstLight* lights,
    uint count
) {
    float3 sum = float3(0.0f);
    for (uint i = 0u; i < count; ++i) {
        BurstLight L = lights[i];
        float intensity = L.positionIntensity.w;
        float maxR      = L.radiusLifePad.x;
        float life      = L.radiusLifePad.y;
        if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

        float3 d = worldPos - L.positionIntensity.xyz;
        float dist = length(d);
        if (dist >= maxR) continue;

        float age = L.colorAge.a;
        if (age >= life) continue;

        // Radial: 1 at centre, 0 at maxR, quadratic in between.
        float rT = 1.0f - dist / maxR;
        float rFall = rT * rT;

        // Age: hot for the first 8% (the flash), then quadratic decay across
        // the rest of life. Matches the firework integrator's brightness
        // profile so the field strength tracks the burst's visible peak.
        float ageT = age / life;
        float ageScale;
        if (ageT < 0.08f) {
            ageScale = 1.0f;
        } else {
            float k = (ageT - 0.08f) / 0.92f;
            ageScale = (1.0f - k);
            ageScale = ageScale * ageScale;
        }

        sum += L.colorAge.rgb * (intensity * rFall * ageScale);
    }
    return sum;
}

// Variant that also returns proximity to nearest burst (for star washout that
// needs to attenuate alpha based on "am I near ANY hot burst?"). Returns the
// strongest single contribution magnitude across the buffer.
inline float sampleNearestBurstStrength(
    float3 worldPos,
    device const BurstLight* lights,
    uint count
) {
    float strongest = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        BurstLight L = lights[i];
        float intensity = L.positionIntensity.w;
        float maxR      = L.radiusLifePad.x;
        float life      = L.radiusLifePad.y;
        if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

        float dist = distance(worldPos, L.positionIntensity.xyz);
        if (dist >= maxR) continue;

        float age = L.colorAge.a;
        if (age >= life) continue;

        float rT = 1.0f - dist / maxR;
        float ageT = age / life;
        float ageScale = saturate(1.0f - ageT * 1.4f);
        float v = intensity * rT * ageScale;
        strongest = max(strongest, v);
    }
    return strongest;
}
