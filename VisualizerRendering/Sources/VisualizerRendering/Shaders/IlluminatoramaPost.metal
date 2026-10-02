// ── ILLUMINATORAMA — BLOOM AND HALATION ─────────────────────────────────────
//
// The Karis/COD dual-filter bloom pyramid (prefilter → down → up) and the
// halation threshold + separable blur.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Bloom — Karis / Call-of-Duty dual-filter mip pyramid (S1.2) ──────────────
//
// Phase 1 shipped a threshold prepass plus a SINGLE 9-tap separable gaussian
// (σ ≈ 2) at half the internal resolution. A σ≈2 kernel at half res is dead by
// ~11 output pixels, so a blown highlight wore a tight RIM rather than a wide,
// gentle halo — measured on the app's night-lamp fixture, Δ(64)/Δ(4) came back
// at exactly 0.0000 (see HouseRenderBridgeGPUTests+BloomProfile).
//
// This is the dual-filter pyramid instead (Jorge Jimenez / Call of Duty
// "Next Generation Post Processing", SIGGRAPH 2014, plus Brian Karis's firefly
// weighting):
//
//   prefilter : full-res HDR → mip0 (internal/2), soft-knee threshold + 13-tap
//               box + Karis partial average
//   down ×N   : mip N-1 → mip N, plain 13-tap box
//   up   ×N-1 : mix(down[i], tent(up[i+1]), scatter) → up[i]
//
// **The upsample blend is CONVEX (`mix`), never additive.** Unrolled, level i
// ends up weighted (1-s)·s^i with the last taking s^(N-1) — a geometric series
// that sums to EXACTLY 1 for any level count. So the whole chain has a DC gain
// of 1, the same as the normalised gaussian it replaces, and every existing
// `bloomIntensity` keeps its meaning. The `out = high + tent(low)` form that
// looks equivalent multiplies the added light by the level count (~6×) and
// would blow out the engine default, all four app presets and every Visualizer
// scene on the frame it landed.
//
// The FILTER is the whole width story: each level halves the resolution, so a
// fixed 13-tap box on level i reaches 2^(i+1) internal texels. Five levels is
// a ~100-texel body with a tail past 200 — the "gentle" in gentle bloom.

// Rec. 709 luma of a linear-HDR colour. Shared by the threshold curve and the
// Karis firefly weight so "how bright is this" means one thing in this section.
static inline float illumiBloomLuma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

/// Soft-knee threshold. `knee` is an ABSOLUTE luma width (the host passes
/// `bloomThreshold × bloomParams.x`), and the curve is deliberately pinned at
/// both ends:
///   • knee == 0  ⇒ `clamp(·, 0, 0)` kills the quadratic term and this is
///     EXACTLY the old hard `max(0, lum - T)` cut, to the bit.
///   • lum ≥ T + knee ⇒ the quadratic saturates at `knee`, which is below
///     `lum - T`, so `max` picks the hard cut — identical to the old curve
///     above T·(1 + kneeFraction).
/// Between them it ramps quadratically, which is what stops a light slowly
/// crossing the threshold from popping its halo on in one frame.
static inline float3 illumiBloomKnee(float3 c, float threshold, float knee) {
    float lum  = illumiBloomLuma(c);
    float soft = clamp(lum - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    // Scale the COLOUR by the luma-space excess rather than thresholding each
    // channel, so the halo keeps the highlight's hue instead of going white.
    return c * (max(soft, lum - threshold) / max(lum, 1e-4));
}

/// The 13 taps of the dual-filter downsample, in SOURCE texel units around the
/// destination texel's uv: nine on a ±2 lattice (a…i, row-major from the top)
/// plus four on a ±1 lattice (j…m). `tx` is one source texel in uv.
static inline void illumiBloomFetch13(texture2d<half, access::sample> src,
                                      sampler smp, float2 uv, float2 tx,
                                      thread float3 *t) {
    t[0] = float3(src.sample(smp, uv + float2(-2, -2) * tx).rgb);  // a
    t[1] = float3(src.sample(smp, uv + float2( 0, -2) * tx).rgb);  // b
    t[2] = float3(src.sample(smp, uv + float2( 2, -2) * tx).rgb);  // c
    t[3] = float3(src.sample(smp, uv + float2(-2,  0) * tx).rgb);  // d
    t[4] = float3(src.sample(smp, uv                       ).rgb); // e (centre)
    t[5] = float3(src.sample(smp, uv + float2( 2,  0) * tx).rgb);  // f
    t[6] = float3(src.sample(smp, uv + float2(-2,  2) * tx).rgb);  // g
    t[7] = float3(src.sample(smp, uv + float2( 0,  2) * tx).rgb);  // h
    t[8] = float3(src.sample(smp, uv + float2( 2,  2) * tx).rgb);  // i
    t[9]  = float3(src.sample(smp, uv + float2(-1, -1) * tx).rgb); // j
    t[10] = float3(src.sample(smp, uv + float2( 1, -1) * tx).rgb); // k
    t[11] = float3(src.sample(smp, uv + float2(-1,  1) * tx).rgb); // l
    t[12] = float3(src.sample(smp, uv + float2( 1,  1) * tx).rgb); // m
}

/// Combine the 13 taps into five overlapping 2×2 quads: the inner (±1) quad at
/// weight 0.5 and the four outer (±2) quads at 0.125 each — 0.5 + 4×0.125 = 1
/// exactly, so the plain form is energy-conserving by construction.
///
/// `karis` additionally weights each QUAD by 1/(1 + luma(quad)) and renormalises
/// by the weight sum (Karis's partial average). That is what stops one 200-nit
/// speck from becoming a full-frame flicker as it moves sub-pixel. It is applied
/// on the FIRST downsample ONLY: by mip1 every tap is already an average of 16+
/// source pixels, so re-weighting them is no longer firefly rejection — it is a
/// non-energy-conserving bias that drags the whole halo dark.
static inline float3 illumiBloomCombine13(thread const float3 *t, bool karis, float invPre = 1.0) {
    float3 q0 = (t[9] + t[10] + t[11] + t[12]) * 0.25;   // inner ±1 quad
    float3 q1 = (t[0] + t[1] + t[3] + t[4]) * 0.25;      // top-left
    float3 q2 = (t[1] + t[2] + t[4] + t[5]) * 0.25;      // top-right
    float3 q3 = (t[3] + t[4] + t[6] + t[7]) * 0.25;      // bottom-left
    float3 q4 = (t[4] + t[5] + t[7] + t[8]) * 0.25;      // bottom-right
    if (!karis) {
        return q0 * 0.5 + (q1 + q2 + q3 + q4) * 0.125;
    }
    float w0 = 0.5   / (1.0 + illumiBloomLuma(q0) * invPre);
    float w1 = 0.125 / (1.0 + illumiBloomLuma(q1) * invPre);
    float w2 = 0.125 / (1.0 + illumiBloomLuma(q2) * invPre);
    float w3 = 0.125 / (1.0 + illumiBloomLuma(q3) * invPre);
    float w4 = 0.125 / (1.0 + illumiBloomLuma(q4) * invPre);
    // Renormalised, so a FLAT input reproduces itself exactly (every weight
    // scales by the same 1/(1+L) and it cancels) — the suppression only bites
    // where the quads disagree, which is what a firefly is.
    float wsum = w0 + w1 + w2 + w3 + w4;
    return (q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4) / max(wsum, 1e-6);
}

/// 3×3 tent (1 2 1 / 2 4 2 / 1 2 1, ÷16) on the LOW mip. `r` is the radius in
/// LOW-mip texels (`bloomParams.z`), so widening it costs nothing but reach.
static inline float3 illumiBloomTent(texture2d<half, access::sample> low,
                                     sampler smp, float2 uv, float2 tx, float r) {
    float2 o = tx * r;
    float3 acc = float3(low.sample(smp, uv + float2(-o.x, -o.y)).rgb) * 1.0;
    acc += float3(low.sample(smp, uv + float2( 0.0, -o.y)).rgb) * 2.0;
    acc += float3(low.sample(smp, uv + float2( o.x, -o.y)).rgb) * 1.0;
    acc += float3(low.sample(smp, uv + float2(-o.x,  0.0)).rgb) * 2.0;
    acc += float3(low.sample(smp, uv                      ).rgb) * 4.0;
    acc += float3(low.sample(smp, uv + float2( o.x,  0.0)).rgb) * 2.0;
    acc += float3(low.sample(smp, uv + float2(-o.x,  o.y)).rgb) * 1.0;
    acc += float3(low.sample(smp, uv + float2( 0.0,  o.y)).rgb) * 2.0;
    acc += float3(low.sample(smp, uv + float2( o.x,  o.y)).rgb) * 1.0;
    return acc * (1.0 / 16.0);
}

/// Prefilter: full-res HDR → mip0 (internal/2). Thresholds each of the 13 taps
/// BEFORE they are averaged — thresholding the average instead (what the old
/// kernel did on its 2×2 box) lets three dark neighbours drag a genuinely blown
/// pixel back under the bar, which is how a small bright source loses its halo.
kernel void illumi_bloom_prefilter(
    texture2d<half, access::sample> inHDR   [[texture(0)]],
    texture2d<half, access::write>  outMip0 [[texture(1)]],
    constant FrameUniforms&         frame   [[buffer(0)]],
    uint2                           gid     [[thread_position_in_grid]]
) {
    uint w = outMip0.get_width();
    uint h = outMip0.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler smp(filter::linear, address::clamp_to_edge, coord::normalized);
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float2 tx = 1.0 / float2(inHDR.get_width(), inHDR.get_height());

    float3 t[13];
    illumiBloomFetch13(inHDR, smp, uv, tx, t);
    float threshold = frame.bloomThreshold;
    float knee = max(0.0, frame.bloomParams.x) * threshold;
    for (int i = 0; i < 13; ++i) {
        t[i] = illumiBloomKnee(max(t[i], 0.0), threshold, knee);
    }
    // The Karis weight is 1/(1 + luma) of the UNSCALED frame: divide an HDR pre-exposure back
    // out (`hdrPreExposure`; displayParams.x = 0 ⇒ 1 ⇒ the multiply by 1.0 is exact).
    float invPre = frame.displayParams.x > 0.0 ? 1.0 / frame.displayParams.x : 1.0;
    outMip0.write(half4(half3(illumiBloomCombine13(t, true, invPre)), 1.0h), gid);
}

/// Down-chain: mip N-1 → mip N. Plain 13-tap box, no threshold (mip0 already
/// carries only the above-threshold excess) and no Karis weighting (see
/// `illumiBloomCombine13`).
kernel void illumi_bloom_down(
    texture2d<half, access::sample> inMip  [[texture(0)]],
    texture2d<half, access::write>  outMip [[texture(1)]],
    uint2                           gid    [[thread_position_in_grid]]
) {
    uint w = outMip.get_width();
    uint h = outMip.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler smp(filter::linear, address::clamp_to_edge, coord::normalized);
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float2 tx = 1.0 / float2(inMip.get_width(), inMip.get_height());

    float3 t[13];
    illumiBloomFetch13(inMip, smp, uv, tx, t);
    outMip.write(half4(half3(illumiBloomCombine13(t, false)), 1.0h), gid);
}

/// Up-chain: tent-blur the accumulated LOW level and blend it CONVEXLY with the
/// same-resolution down-chain level. `inLow` is the previous up-chain output
/// (or, seeding the chain, the smallest down level); `inHigh` is down[i]; the
/// result lands in a SEPARATE up[i] texture, so no level is ever read and
/// written in the same dispatch.
kernel void illumi_bloom_up(
    texture2d<half, access::sample> inLow  [[texture(0)]],
    texture2d<half, access::sample> inHigh [[texture(1)]],
    texture2d<half, access::write>  outTex [[texture(2)]],
    constant FrameUniforms&         frame  [[buffer(0)]],
    // Daydream DH-0992 — THIS level's scatter, so the host can shape the halo's radial profile
    // level by level (a lens PSF's power-law tail) instead of the one geometric ratio. < 0 ⇒ the
    // shared `bloomParams.y`, byte-identical. Any per-level set in [0,1] stays CONVEX, so the
    // chain's DC gain is still exactly 1.
    constant float&                 levelScatter [[buffer(1)]],
    uint2                           gid    [[thread_position_in_grid]]
) {
    uint w = outTex.get_width();
    uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler smp(filter::linear, address::clamp_to_edge, coord::normalized);
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float2 tx = 1.0 / float2(inLow.get_width(), inLow.get_height());

    float3 high = float3(inHigh.sample(smp, uv).rgb);
    float3 low  = illumiBloomTent(inLow, smp, uv, tx, max(0.0, frame.bloomParams.z));
    float  scatter = clamp(levelScatter >= 0.0 ? levelScatter : frame.bloomParams.y, 0.0, 1.0);
    // CONVEX — see the section header. `high + low` here is the classic
    // dual-filter mistake and multiplies the added light by the level count.
    outTex.write(half4(half3(mix(high, low, scatter)), 1.0h), gid);
}

// ── Halation (film) ──────────────────────────────────────────────────────────
//
// On real film, light from a highlight passes THROUGH the emulsion, reflects off the
// back of the base, and re-exposes the emulsion from behind — a soft second exposure
// around the highlight. What reaches the base is what the blue and green layers (and
// the yellow filter between them) did not absorb, i.e. overwhelmingly RED light, and
// on the way back up it meets the red-sensitive layer first. So halation is:
//
//   • SOURCED from the red end of the highlight — a pure-blue light barely halates;
//   • ISOTROPIC — a disc of scatter around every point, the same in every direction;
//   • LONG-TAILED — a bright tight core (the first bounce) under a wide faint tail
//     (oblique bounces and multiple reflections), never a single hard-edged blob;
//   • COLOUR-GRADED by distance — orange where the core still carries some green-layer
//     re-exposure, deepening to red in the tail, where only red survives.
//   • a fixed size ON THE NEGATIVE — set by the base thickness — so it is specified
//     as a fraction of FRAME HEIGHT, never in texels (a texel radius made the halo
//     shrink by the supersample factor in a photo-lane still).
//
// Implementation (all at internal/2):
//   threshold : internal HDR → per-texel soft-knee excess (thresholded BEFORE the 2×2
//               average, so a one-texel-wide blown edge keeps its halo), collapsed to
//               one scalar "halation exposure" weighted toward red.
//   blur_h    : that scalar, convolved with FOUR normalised gaussians an octave apart
//               at EVERY texel — out to 3σ, no tap skipping — → rgba = the four scales.
//   blur_v    : the same, vertically; the four separable gaussians make four isotropic
//               ones, tinted orange (inner) → red (outer) and summed → rgb halo.
//
// History: the first version was a 9-tap gaussian whose taps were spaced radius/16
// quarter-res texels apart (4 quarter texels ≈ 16 internal px at the default). A
// smooth blob survives that; a THIN highlight does not — each tap re-printed the
// door edge as a separate copy, and the halo came out as a comb of discrete orange
// stripes with dark gaps (published capture `halation-sunset-bedroom-on`, 2026-10-01).
// Dense taps make that structurally impossible.
//
// `halationParams.x == 0` (the default) ⇒ the host does not encode these passes at
// all AND the tonemap branch is skipped ⇒ non-opting scenes are byte-identical.

/// How much of each channel of a highlight reaches the base and comes back as halation
/// exposure. Normalised to sum 1, so a neutral highlight's halation exposure equals its
/// excess radiance.
constant float3 kHalationSourceWeight = float3(0.80, 0.18, 0.02);

kernel void illumi_halation_threshold(
    texture2d<half, access::read>  inHDR     [[texture(0)]],
    texture2d<half, access::write> outBright [[texture(1)]],
    constant FrameUniforms&        frame     [[buffer(0)]],
    uint2                          gid       [[thread_position_in_grid]]
) {
    uint w = outBright.get_width();
    uint h = outBright.get_height();
    if (gid.x >= w || gid.y >= h) return;
    uint2 maxC = uint2(inHDR.get_width() - 1, inHDR.get_height() - 1);
    uint2 src  = gid * 2;
    float threshold = frame.halationParams.y;
    // A soft knee half the threshold wide: a light easing across the bar grows its halo
    // continuously instead of popping it on (same curve bloom uses).
    float knee = 0.5 * threshold;
    float e = 0.0;
    for (uint j = 0; j < 2; ++j) {
        for (uint i = 0; i < 2; ++i) {
            float3 c = max(float3(inHDR.read(min(src + uint2(i, j), maxC)).rgb), 0.0);
            e += dot(illumiBloomKnee(c, threshold, knee), kHalationSourceWeight);
        }
    }
    outBright.write(half4(half(e * 0.25), 0.0h, 0.0h, 1.0h), gid);
}

/// The halo's PSF is a sum of FOUR normalised gaussians an octave apart — σ = R/16,
/// R/8, R/4, R/2 (R = `halationParams.z` × frame height, so 2σ of the widest is R). With
/// comparable weight per octave the sum falls off roughly as 1/r over the whole range —
/// the long, soft tail of a real halation PSF — where a single gaussian would end in a
/// visible rim (it did: one σ = R/2 gaussian read as an orange sleeve with an edge).
static inline float4 halationSigmas(constant FrameUniforms& frame, float rows) {
    float r = frame.halationParams.z * rows;
    return max(float4(0.35), r * float4(1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0));
}

/// Hard cap on the half-width of the 1-D kernel (texels). 3σ of the default widest
/// scale on a 2400-row photo-lane still is ~45, so this only bites on absurd radii.
constant int kHalationMaxReach = 128;

/// One separable pass of all four gaussians. `axis` = (1,0) or (0,1). The horizontal
/// pass blurs the single exposure scalar in .r by all four; the vertical pass blurs
/// each of the four channels by its own σ.
static inline float4 halationBlur1D(texture2d<half, access::read> inTex, int2 p, int2 axis,
                                    float4 sigma, bool separate) {
    int2 size = int2(inTex.get_width(), inTex.get_height());
    int reach = min(int(ceil(3.0 * sigma.w)), kHalationMaxReach);
    float4 inv2s2 = 1.0 / (2.0 * sigma * sigma);
    float4 acc = 0.0, wsum = 0.0;
    for (int k = -reach; k <= reach; ++k) {
        int2 q = p + axis * k;
        // Out-of-frame texels are SKIPPED and the weights renormalised — clamping
        // would re-read the border texel and brighten every halo that touches an edge.
        if (any(q < 0) || any(q >= size)) continue;
        float4 v = float4(inTex.read(uint2(q)));
        float4 wk = exp(-float(k * k) * inv2s2);
        acc  += wk * (separate ? v : float4(v.x));
        wsum += wk;
    }
    return acc / max(wsum, float4(1e-6));
}

kernel void illumi_halation_blur_h(
    texture2d<half, access::read>  inTex  [[texture(0)]],
    texture2d<half, access::write> outTex [[texture(1)]],
    constant FrameUniforms&        frame  [[buffer(0)]],
    uint2                          gid    [[thread_position_in_grid]]
) {
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    float4 sigma = halationSigmas(frame, float(outTex.get_height()));
    outTex.write(half4(halationBlur1D(inTex, int2(gid), int2(1, 0), sigma, false)), gid);
}

kernel void illumi_halation_blur_v(
    texture2d<half, access::read>  inTex  [[texture(0)]],
    texture2d<half, access::write> outTex [[texture(1)]],
    constant FrameUniforms&        frame  [[buffer(0)]],
    uint2                          gid    [[thread_position_in_grid]]
) {
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    float4 sigma = halationSigmas(frame, float(outTex.get_height()));
    float4 s = halationBlur1D(inTex, int2(gid), int2(0, 1), sigma, true);
    // `halationTint` is the colour of the WIDEST scale — the deep red where only red
    // survives the trip through the base. Nearer scales carry progressively more
    // green-layer re-exposure and read orange; the colour grades with distance.
    float3 tail = frame.halationTint.rgb;
    float3 c0 = tail * float3(1.0, 2.4, 1.8);
    float3 c1 = tail * float3(1.0, 1.8, 1.4);
    float3 c2 = tail * float3(1.0, 1.3, 1.15);
    // Energy per octave: the inner octaves a little heavier (the first bounce is the
    // brightest), summing to 1.
    float3 halo = c0 * (0.30 * s.x) + c1 * (0.28 * s.y) + c2 * (0.24 * s.z) + tail * (0.18 * s.w);
    outTex.write(half4(half3(halo), 1.0h), gid);
}
