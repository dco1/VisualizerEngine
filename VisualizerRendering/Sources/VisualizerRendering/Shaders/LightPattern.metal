#include <metal_stdlib>
using namespace metal;

// ── LightPattern.metal ───────────────────────────────────────────────────────
//
// One thread per slot computes the slot's HDR colour for the active pattern
// (glow / chase / marquee / rainbow / sparkle), capped at `emissionCap`, and
// writes RGBA into a 1×N `rgba32Float` texture. Identical math to the
// previous CPU loop in EggsRailLights.apply() — kept bit-for-bit so the
// visual is unchanged.
//
// Swift mirror: `LightPatternUniforms` in LightPatternKernel.swift. Update
// both sides together. Same alignment rule as elsewhere — float4 only in
// shared structs.

struct LightPatternUniforms {
    float4 baseColor;     // rgb (.a unused)
    float4 accentColor;   // rgb (.a unused)
    float  time;
    float  speed;
    float  intensity;
    float  emissionCap;
    uint   pattern;       // 0=glow 1=chase 2=marquee 3=rainbow 4=sparkle
    uint   bulbCount;
    float  _pad0;
    float  _pad1;
};

// HSV→RGB, saturation=0.85, brightness=1.0 — matches `colorFor(hue:)` in the
// CPU path so the rainbow gradient lines up across the migration.
static float3 hueToRGB(float h) {
    h = fract(h);
    float r = abs(h * 6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h * 6.0 - 2.0);
    float b = 2.0 - abs(h * 6.0 - 4.0);
    float3 rgb = saturate(float3(r, g, b));
    // saturation 0.85, value 1.0
    return mix(float3(1.0), rgb, 0.85);
}

kernel void light_pattern_write(
    constant LightPatternUniforms& U   [[buffer(0)]],
    texture2d<float, access::write> lut [[texture(0)]],
    uint slot                          [[thread_position_in_grid]]
) {
    if (slot >= U.bulbCount) return;

    float3 color = U.baseColor.rgb;
    float  scalar = 0.85;

    switch (U.pattern) {
    case 0: // glow
        scalar = 0.85;
        break;
    case 1: { // chase
        float stride = 8.0;
        float phase  = U.time * U.speed;
        float p      = fract(float(slot) / stride - phase);
        float pulse  = max(0.0, 1.0 - p * 4.0);
        scalar = 0.05 + pulse;
        break;
    }
    case 2: { // marquee
        uint flip   = uint(floor(U.time * U.speed)) & 1u;
        uint pair   = slot / 2u;
        uint parity = (pair + flip) & 1u;
        color  = (parity == 0u) ? U.baseColor.rgb : U.accentColor.rgb;
        scalar = 0.95;
        break;
    }
    case 3: { // rainbow
        float pairCount = max(1.0, float(U.bulbCount / 2u));
        float phase     = U.time * U.speed * 0.20;
        float pair      = float(slot / 2u);
        float hue       = fract(pair / pairCount + phase);
        color  = hueToRGB(hue);
        scalar = 0.95;
        break;
    }
    case 4: { // sparkle
        // Knuth multiplicative hash → [0,1) — same constant as the CPU path.
        uint h = (slot * 2654435761u) & 0xFFFFu;
        float fh = float(h) / 65535.0;
        float phase = U.time * U.speed + fh * 6.283185307179586;
        float pulse = 0.5 + 0.5 * sin(phase * 2.0);
        scalar = 0.10 + pulse * 0.90;
        break;
    }
    default:
        break;
    }

    float scale  = max(0.0, U.intensity * scalar);
    float capped = min(scale, U.emissionCap);
    float3 outRGB = color * capped;
    lut.write(float4(outRGB, 1.0), uint2(slot, 0));
}
