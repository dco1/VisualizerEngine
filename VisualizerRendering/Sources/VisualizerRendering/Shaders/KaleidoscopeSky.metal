#include <metal_stdlib>
using namespace metal;

// ── KALEIDOSCOPE SKY ─────────────────────────────────────────────────────────
//
// Writes an animated kaleidoscope pattern into an equirect HDR texture. Used as
// an Illuminatorama `equirectSky` so the pattern both lights the scene (IBL,
// baked into irradiance + prefiltered cubes) and fills the background pixels.
//
// The longitude axis (u) wraps seamlessly because the wedge fold is angle-based:
// the texture's left and right edges meet without a visible seam in the panorama.
// Latitude (v) maps to a radial coordinate so the poles read as the kaleidoscope
// centre/rim.
//
// Uniforms (buffer 0): { time, segments, gain, _pad }
struct KaleidoUniforms {
    float time;
    float segments;   // mirrored wedge count (e.g. 6, 8, 12)
    float gain;       // HDR scale on the output colour (>1 for stronger IBL)
    float _pad;
};

// A smooth rainbow palette (Inigo Quilez cosine palette).
static inline float3 palette(float t) {
    const float3 a = float3(0.5, 0.5, 0.5);
    const float3 b = float3(0.5, 0.5, 0.5);
    const float3 c = float3(1.0, 1.0, 1.0);
    const float3 d = float3(0.00, 0.33, 0.67);
    return a + b * cos(6.28318530718 * (c * t + d));
}

kernel void kaleidoscope_sky(texture2d<float, access::write> outTex [[texture(0)]],
                             constant KaleidoUniforms &U          [[buffer(0)]],
                             uint2 gid                            [[thread_position_in_grid]]) {
    const uint w = outTex.get_width();
    const uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Equirect parameterisation.
    float2 uv = float2(float(gid.x) / float(w), float(gid.y) / float(h));
    float ang = uv.x * 6.28318530718;     // longitude, wraps at 0/2π
    float rad = uv.y;                      // 0 (south pole) .. 1 (north pole)

    // Kaleidoscope wedge fold: triangle-wave the angle into `segments` mirrored
    // wedges so each wedge is a mirror image of its neighbour (the classic look).
    float seg = 6.28318530718 / max(2.0, U.segments);
    float fa = fabs(fract(ang / seg) - 0.5) * 2.0;   // 0..1 within wedge

    // Slow rotation so the whole kaleidoscope turns over time.
    float t = U.time;

    // Layered petals: angular ripples × radial rings, both animated.
    float petals = sin(fa * 6.28318530718 * 3.0 + t * 0.9) * 0.5 + 0.5;
    float rings  = sin(rad * 6.28318530718 * 4.0 - t * 0.6) * 0.5 + 0.5;
    float m = petals * rings;

    // A second, finer layer offset in phase for visual richness.
    float fine = sin((fa + rad) * 6.28318530718 * 6.0 + t * 1.4) * 0.5 + 0.5;
    m = mix(m, fine, 0.35);

    // Colour from the palette, slowly cycling hue over time.
    float3 col = palette(m + t * 0.05);

    // Darken toward the poles a touch so the IBL has a soft top/bottom falloff
    // (avoids a uniform flood that washes the eggs out).
    float poleFade = smoothstep(0.0, 0.18, rad) * smoothstep(1.0, 0.82, rad);
    col *= mix(0.35, 1.0, poleFade);

    // HDR gain so the brightest petals exceed 1.0 and drive bloom + IBL.
    col *= U.gain;

    outTex.write(float4(col, 1.0), gid);
}
