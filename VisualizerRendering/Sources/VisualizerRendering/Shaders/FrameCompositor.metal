// ── FRAME COMPOSITOR SHADERS ────────────────────────────────────────────────
//
// Post-process chain for the reusable FrameCompositor module:
//   1. fcBloomThreshold  — full-res HDR → half-res bright-only
//   2. fcBloomBlurH      — 9-tap horizontal Gaussian
//   3. fcBloomBlurV      — 9-tap vertical Gaussian
//   4. fcTonemap         — HDR + bloom → exposure → ACES → rgba8 display
//
// Same techniques as the IlluminatoramaRenderer bloom + tonemap, simplified
// to a single half-res bloom layer (vs Illuminatorama's multi-resolution
// cascade) because the particle/volumetric scenes this module targets don't
// need the cinematic-grade glow Illuminatorama goes for.
//
// All compute kernels — single threadgroup pattern (8×8) covering each
// output texture. Compute simplifies the resize/aspect handling vs
// fragment + viewport setup.

#include <metal_stdlib>
using namespace metal;

constexpr sampler bilinearClamp(filter::linear, address::clamp_to_edge);

// ── 1. Bloom threshold (full → half res, bright pixels only) ───────────────
//
// Reads the HDR colour at the 2×2 footprint of the half-res output pixel,
// averages, subtracts threshold, clamps to zero, writes to the half-res
// bloom scratch. Below threshold → black (no bloom contribution).

kernel void fcBloomThreshold(
    texture2d<float, access::sample> hdr        [[ texture(0) ]],
    texture2d<float, access::write>  bloomOut   [[ texture(1) ]],
    constant float&                  threshold  [[ buffer(0) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = bloomOut.get_width();
    uint h = bloomOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Sample the corresponding 2×2 HDR footprint via a bilinear tap at the
    // centre of the 2×2 region — cheap downsample.
    float2 uv = (float2(gid) + 0.5f) / float2(w, h);
    float4 hdrColor = hdr.sample(bilinearClamp, uv);

    // Subtract threshold using a soft knee so values just above threshold
    // bloom faintly rather than cliffing in. Then clamp to [0, ∞).
    float luma = max(hdrColor.r, max(hdrColor.g, hdrColor.b));
    float soft = max(0.0f, luma - threshold);
    float weight = soft / max(luma, 1e-4f);

    bloomOut.write(float4(hdrColor.rgb * weight, hdrColor.a), gid);
}

// ── 2. Horizontal Gaussian blur (9-tap) ────────────────────────────────────

kernel void fcBloomBlurH(
    texture2d<float, access::sample> src  [[ texture(0) ]],
    texture2d<float, access::write>  dst  [[ texture(1) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = dst.get_width();
    uint h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float2 uv = (float2(gid) + 0.5f) / float2(w, h);
    float2 texelStep = float2(1.0f / float(w), 0.0f);

    // 9-tap 1D Gaussian σ ≈ 2 texels. Weights normalized to sum to 1.
    const float weights[5] = { 0.2270f, 0.1945f, 0.1216f, 0.0540f, 0.0162f };
    float4 sum = src.sample(bilinearClamp, uv) * weights[0];
    for (int i = 1; i < 5; ++i) {
        float2 o = texelStep * float(i);
        sum += src.sample(bilinearClamp, uv + o) * weights[i];
        sum += src.sample(bilinearClamp, uv - o) * weights[i];
    }
    dst.write(sum, gid);
}

// ── 3. Vertical Gaussian blur (9-tap) ──────────────────────────────────────

kernel void fcBloomBlurV(
    texture2d<float, access::sample> src  [[ texture(0) ]],
    texture2d<float, access::write>  dst  [[ texture(1) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = dst.get_width();
    uint h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float2 uv = (float2(gid) + 0.5f) / float2(w, h);
    float2 texelStep = float2(0.0f, 1.0f / float(h));

    const float weights[5] = { 0.2270f, 0.1945f, 0.1216f, 0.0540f, 0.0162f };
    float4 sum = src.sample(bilinearClamp, uv) * weights[0];
    for (int i = 1; i < 5; ++i) {
        float2 o = texelStep * float(i);
        sum += src.sample(bilinearClamp, uv + o) * weights[i];
        sum += src.sample(bilinearClamp, uv - o) * weights[i];
    }
    dst.write(sum, gid);
}

// ── 4. Tonemap (HDR + bloom → exposure → ACES → rgba8) ─────────────────────

// Scalar ACES filmic tonemap fit (Krzysztof Narkowicz). Returns the
// tonemapped value for a single luminance scalar.
inline float aces_scalar(float x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// SATURATION-PRESERVING ACES tonemap — hybrid.
//
// Two failure modes book-end the design space:
//
//   • PER-CHANNEL ACES (`aces_scalar` applied independently to R/G/B).
//     Desaturates bright colours: orange (10, 5, 0) → (0.94, 0.89, 0),
//     a yellow-white blob with no orange punch. Wrong for fireworks.
//
//   • PURE CHROMA-PRESERVING (tonemap Lmax, scale RGB by same ratio).
//     Preserves hue perfectly but the secondary channels stay LOW:
//     orange (10, 5, 0) → (0.94, 0.47, 0) — accurate red-orange, but
//     dimmer than per-channel so the user perceives it as "less vivid"
//     when their reference is a bright bloomed Fireworks+ scene.
//
// Hybrid blends the two with a per-channel weight of 0.35: most of
// the chromaticity preservation, plus enough per-channel lift to
// keep mid-saturation secondaries bright. Same orange:
//
//   chromaPreserving = (0.94, 0.47, 0)
//   perChannel       = (0.94, 0.89, 0)
//   mix(chroma, perCh, 0.35) = (0.94, 0.62, 0)
//
// — a bright vivid orange that reads as fireworks-vivid without
// collapsing toward yellow-white.
inline float3 aces(float3 x) {
    float Lmax = max(max(x.r, x.g), x.b);
    if (Lmax < 1e-4f) return float3(0.0f);
    float Lout = aces_scalar(Lmax);
    float3 chromaPreserving = x * (Lout / Lmax);
    float3 perChannel = float3(aces_scalar(x.r), aces_scalar(x.g), aces_scalar(x.b));
    // Mix dropped 0.35 → 0.15 — leans HARD on chromaticity preservation
    // so the bright bursts read as pure saturated hue (a red spark is
    // PURE RED on screen, not the per-channel-ACES yellow-orange blob).
    // The 15% per-channel contribution keeps mid-saturation secondaries
    // from going completely dim without re-introducing the wash-to-
    // white failure mode.
    return clamp(mix(chromaPreserving, perChannel, 0.15f), 0.0f, 1.0f);
}

kernel void fcTonemap(
    texture2d<float, access::sample> hdr         [[ texture(0) ]],
    texture2d<float, access::sample> bloom       [[ texture(1) ]],
    texture2d<float, access::write>  ldrOut      [[ texture(2) ]],
    constant float2&                 params      [[ buffer(0) ]],  // x = exposure, y = bloomIntensity
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = ldrOut.get_width();
    uint h = ldrOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float2 uv = (float2(gid) + 0.5f) / float2(w, h);
    float4 hdrColor = hdr.sample(bilinearClamp, uv);
    // Bilinear upsample of the half-res bloom layer.
    float4 bloomColor = bloom.sample(bilinearClamp, uv);

    float exposure = params.x;
    float bloomI   = params.y;

    float3 combined = hdrColor.rgb * exposure + bloomColor.rgb * bloomI;
    float3 mapped = aces(combined);

    // Approximate gamma 2.2 → sRGB encoding for the rgba8 display texture.
    // (ACES output is in linear; SCN samples rgba8Unorm as sRGB depending
    // on the material setup, but explicitly gamma-encoding here keeps it
    // consistent regardless of caller's material config.)
    float3 srgb = pow(mapped, float3(1.0f / 2.2f));

    // Alpha encodes per-pixel "coverage" — luminance of the tonemapped
    // result, clamped to [0, 1]. The presenter quad uses standard alpha
    // blending, so bright compositor pixels (streaks, shells, cloud) are
    // opaque and overwrite the SCN backdrop, while regions the compositor
    // didn't touch (luminance ≈ 0) stay transparent and let the SCN scene
    // below show through unaltered. This is the correct way to overlay a
    // sparse Metal render on top of an existing SCN scene — `blendMode = .add`
    // makes SCN treat the quad as opaque and skip the background draw
    // behind it, even when most pixels would have added zero.
    // Premultiplied alpha — RGB scaled by per-pixel coverage (luminance
    // of the tonemapped result), alpha carries the same coverage. The
    // presenter quad uses standard alpha blending, so bright compositor
    // pixels (streaks, shells, cloud) composite opaquely over the SCN
    // scene while regions the compositor didn't touch (coverage ≈ 0)
    // stay transparent and let SCN content show through unchanged.
    float coverage = clamp(dot(srgb, float3(0.299f, 0.587f, 0.114f)), 0.0f, 1.0f);
    ldrOut.write(float4(srgb * coverage, coverage), gid);
}
