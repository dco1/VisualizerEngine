#include <metal_stdlib>
using namespace metal;

// ── Jalapeño lava-lamp colour field ──────────────────────────────────────────
//
// An animated GPU metaball field rendered into a 2D RGBA texture, sampled by
// the jalapeño material to drive its surface colour. Per the project's
// "no faked scrolling textures" rule, the blobs are a real evaluated metaball
// potential advected over time — not a pre-baked texture scrolled across UVs.
//
// `mode`: 0 = Green, 1 = Red, 2 = Mix. Selects the palette the potential maps
// into. The field itself (blob positions, motion) is identical across modes so
// the toggle is purely a colour remap of the same animated structure.

struct JalapenoFieldUniforms {
    float  time;        // seconds
    float  speed;       // motion-speed multiplier
    uint   mode;        // 0 green, 1 red, 2 mix
    uint   blobCount;   // active blobs (<= 8)
    float2 texSize;     // width, height in px
    float2 pad;
};

// 8 metaballs, each driven by its own slow Lissajous orbit. Deterministic so
// the field is reproducible.
kernel void jalapenoColorField(texture2d<float, access::write> outTex [[texture(0)]],
                               constant JalapenoFieldUniforms& u [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.texSize.x) || gid.y >= uint(u.texSize.y)) return;

    float2 uv = (float2(gid) + 0.5) / u.texSize;   // 0..1
    float t = u.time * u.speed;

    // Accumulate metaball potential.
    float field = 0.0;
    for (uint i = 0; i < u.blobCount; ++i) {
        float fi = float(i);
        // Per-blob orbit centre — slow Lissajous, phases spread by index.
        float2 c = float2(
            0.5 + 0.34 * sin(t * (0.21 + fi * 0.013) + fi * 1.7),
            0.5 + 0.34 * cos(t * (0.17 + fi * 0.017) + fi * 2.3)
        );
        float2 d = uv - c;
        float r2 = dot(d, d) + 1e-4;
        // Per-blob radius wobble.
        float radius = 0.10 + 0.04 * sin(t * 0.3 + fi);
        field += (radius * radius) / r2;
    }

    // Normalise the potential into a smooth 0..1 ramp.
    float v = smoothstep(0.55, 2.4, field);

    // Palette remap.
    float3 col;
    if (u.mode == 0u) {
        // Green: dark jalapeño green → bright lime highlight.
        float3 lo = float3(0.05, 0.22, 0.04);
        float3 hi = float3(0.45, 0.85, 0.15);
        col = mix(lo, hi, v);
    } else if (u.mode == 1u) {
        // Red: deep red → bright chili red/orange.
        float3 lo = float3(0.28, 0.03, 0.02);
        float3 hi = float3(0.95, 0.28, 0.08);
        col = mix(lo, hi, v);
    } else {
        // Mix: blend green and red across the field so blobs ripen.
        float3 g = mix(float3(0.05, 0.22, 0.04), float3(0.45, 0.85, 0.15), v);
        float3 r = mix(float3(0.28, 0.03, 0.02), float3(0.95, 0.28, 0.08), v);
        // Spatial-temporal ripen factor.
        float ripen = 0.5 + 0.5 * sin(uv.x * 6.0 + uv.y * 4.0 + t * 0.4);
        col = mix(g, r, ripen);
    }

    outTex.write(float4(col, 1.0), gid);
}
