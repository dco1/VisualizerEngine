// ── ILLUMINATORAMA — RAY-TRACED DENOISE (SVGF) ──────────────────────────────
//
// The RT reflection denoiser, the RT-GI temporal pass, and the SVGF variance +
// à-trous wavelet filter.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaDenoise.h"
using namespace metal;

// ── RT diffuse denoiser (depth + normal guided bilateral) ─────────────────────
//
// `illumi_rt_lighting` writes its noisy diffuse shadow+GI term into a dedicated
// full-res buffer (`rtDiffuse`) instead of summing it straight into the HDR
// composite. This pass bilateral-filters that buffer — guided by g-buffer depth
// + normal so it can't bleed across geometric edges or curvature — then adds the
// cleaned result into the composite. Running it BEFORE the TAA resolve means
// temporal accumulation receives a far less noisy per-frame input, so it
// converges in a fraction of the frames AND survives camera/subject motion (when
// TAA history is rejected, the raw ~4-spp grain would otherwise show through —
// exactly the "more dithered" artefact under orbit).
//
// Only the diffuse term is filtered; the RT pass already composited the sharp
// reflection term directly (a bilateral that shares depth+normal would smear a
// mirror reflection across a flat floor).
struct RTDenoiseUniforms {
    uint  width; uint height; uint enabled; uint radius;
    float kDepth; float kNorm; float _pad0; float _pad1;
};

kernel void illumi_rt_denoise(
    texture2d<half,  access::read>        rtDiffuse  [[texture(0)]],  // full-res raw RT diffuse
    texture2d<float, access::read>        gDepth     [[texture(1)]],  // full-res
    texture2d<half,  access::read>        gNormalRgh [[texture(2)]],  // full-res
    texture2d<half,  access::read_write>  outHDR     [[texture(3)]],  // composite (add into)
    constant FrameUniforms&               frame      [[buffer(0)]],   // invProjection
    constant RTDenoiseUniforms&           du         [[buffer(1)]],
    uint2                                 gid        [[thread_position_in_grid]]
) {
    if (gid.x >= du.width || gid.y >= du.height) return;

    // Sky pixel: the RT pass left rtDiffuse untouched here (it early-outs on the
    // same depth test), so there's nothing to add. Leave the composite alone.
    float centerDepth = gDepth.read(gid).r;
    if (centerDepth >= 0.99999) return;

    half4 prev = outHDR.read(gid);

    // Denoise disabled (or zero radius): composite the raw diffuse straight in.
    // Identical to the pre-split additive behaviour, just across two passes.
    if (du.enabled == 0u || du.radius == 0u) {
        float3 raw = float3(rtDiffuse.read(gid).rgb);
        outHDR.write(half4(prev.rgb + half3(raw), prev.a), gid);
        return;
    }

    float2 ndcC = (float2(gid) + 0.5) / float2(du.width, du.height) * 2.0 - 1.0;
    ndcC.y = -ndcC.y;
    float3 PviewC  = viewPosFromDepth(ndcC, centerDepth, frame.invProjection);
    float3 NworldC = octDecode(float2(gNormalRgh.read(gid).rg));

    int R = int(du.radius);
    float3 total  = float3(0.0);
    float  totalW = 0.0;

    for (int j = -R; j <= R; ++j) {
        for (int i = -R; i <= R; ++i) {
            int2 p = int2(int(gid.x) + i, int(gid.y) + j);
            p = clamp(p, int2(0), int2(int(du.width) - 1, int(du.height) - 1));
            uint2 np = uint2(p);

            float nd = gDepth.read(np).r;
            if (nd >= 0.99999) continue;   // skip sky neighbours (stale rtDiffuse)

            float2 ndcN = (float2(np) + 0.5) / float2(du.width, du.height) * 2.0 - 1.0;
            ndcN.y = -ndcN.y;
            float3 PviewN  = viewPosFromDepth(ndcN, nd, frame.invProjection);
            float3 NworldN = octDecode(float2(gNormalRgh.read(np).rg));

            float wDepth   = exp(-abs(PviewC.z - PviewN.z) * du.kDepth);
            float wNorm    = pow(max(0.0, dot(NworldC, NworldN)), du.kNorm);
            float r2       = float(i * i + j * j);
            float wSpatial = exp(-r2 / (2.0 * float(R * R) + 1e-3));  // centre-weighted
            float w        = wDepth * wNorm * wSpatial;

            total  += float3(rtDiffuse.read(np).rgb) * w;
            totalW += w;
        }
    }

    float3 filtered = (totalW > 1e-6) ? (total / totalW)
                                      : float3(rtDiffuse.read(gid).rgb);
    outHDR.write(half4(prev.rgb + half3(filtered), prev.a), gid);
}

// ── Temporal accumulation of the RT diffuse (1-bounce GI + soft shadow) ───────
//
// The walls are lit almost entirely by the RT 1-bounce GI, whose hemisphere ray
// DIRECTIONS re-jitter every frame. A single frame is a few-sample Monte-Carlo
// estimate; the main TAA converges it only while the camera is static, so the
// instant you move the walls "boil" (temporal crawl). This pass gives the GI
// term its OWN velocity-reprojected exponential history BEFORE the spatial
// denoise — so it keeps converging across frames under motion, where the main
// TAA cannot. The GI is low-frequency, so it tolerates aggressive accumulation
// (and the wide neighborhood clamp) without visible ghosting.
//
// Differs from illumi_ssr_temporal on purpose: a WIDE clamp (gammaClamp, ~4 vs
// 1.5) so a clean accumulated history isn't yanked back toward the noisy current
// 3×3 mean (which for low-frequency GI noise has a tiny sigma → a tight clamp
// would re-inject the very crawl we're removing), and a tunable base blend.
struct RTGITemporalUniforms {
    uint  width; uint height; uint enabled; uint isFirstFrame;
    float blend;       // weight of the CURRENT frame in steady state (e.g. 0.06)
    float gammaClamp;  // neighborhood clamp width in sigmas (wide, e.g. 4.0)
    float _pad0; float _pad1;
};

kernel void illumi_rt_gi_temporal(
    texture2d<half,  access::read>       giRaw       [[texture(0)]],  // current rtDiffuse
    texture2d<half,  access::sample>     histGI      [[texture(1)]],  // previous accumulated
    texture2d<half,  access::read>       velocity    [[texture(2)]],  // full-res motion vectors
    texture2d<half,  access::write>      outGI       [[texture(3)]],  // accumulated → denoise input + next history
    texture2d<half,  access::read_write> sampleCount [[texture(4)]],  // full-res r16Float
    constant RTGITemporalUniforms&       u           [[buffer(0)]],
    uint2                                gid         [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;
    float3 current = float3(giRaw.read(gid).rgb);

    if (u.enabled == 0u || u.isFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outGI.write(half4(half3(current), 1.0h), gid);
        return;
    }

    float N = float(sampleCount.read(gid).r);

    float2 uv  = (float2(gid) + 0.5) / float2(u.width, u.height);
    float2 vel = float2(velocity.read(gid).rg);
    float2 histUV = uv - vel;

    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outGI.write(half4(half3(current), 1.0h), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 hist = float3(histGI.sample(samp, histUV).rgb);

    // Wide YCoCg neighborhood clamp — guards gross ghosting at on-screen
    // disocclusions, but loose enough not to fight low-frequency convergence.
    float3 curY  = RGBtoYCoCg(current);
    float3 histY = RGBtoYCoCg(hist);
    float3 m1 = float3(0.0), m2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            float3 s = RGBtoYCoCg(float3(giRaw.read(uint2(c)).rgb));
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float3 sigma = max(sqrt(max(float3(0.0), m2 - m1 * m1)), float3(0.02));
    histY = clamp(histY, m1 - u.gammaClamp * sigma, m1 + u.gammaClamp * sigma);

    // Adaptive blend: 1/N for fast early-frame convergence; u.blend is the
    // steady-state floor. GI uses a gentle disocclusion ramp (×2 not ×4) so
    // a camera pan keeps accumulating instead of dumping the whole history.
    float velMag = length(vel);
    float disocc = smoothstep(0.02, 0.08, velMag);
    if (disocc > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float alpha = max(u.blend, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 2.0), disocc);
    alpha = clamp(alpha, 0.01, 1.0);

    float3 res = YCoCgtoRGB(mix(histY, curY, alpha));
    outGI.write(half4(half3(max(float3(0.0), res)), 1.0h), gid);
}

// ── SVGF: À-trous spatiotemporal denoiser ─────────────────────────────────────
//
// Schied et al., "Spatiotemporal Variance-Guided Filtering: Real-Time
// Reconstruction for Path-Traced Global Illumination", HPG 2017.
//
// Pipeline (runs after illumi_rt_gi_temporal when svgfEnabled):
//   illumi_svgf_variance  — 3×3 spatial variance estimate of accumulated GI
//   illumi_svgf_atrous ×N — joint bilateral à-trous cascade (stride 2^level)
//
// The key SVGF weight is the luminance term:
//   w_L = exp(−|lum_A − lum_B|² / (σ_L² · var_A + ε))
//
// High local variance → loose constraint → more spatial denoising.
// Low local variance  → tight constraint → preserves converged detail.
//
// Three levels cover a spatial reach of 1+2+4 = 7px radius (≈15×15 footprint)
// at the cost of 25 samples × 3 passes — much cheaper than a direct 15×15
// bilateral (225 samples) with the same effective support.

struct SVGFAtrousUniforms {
    uint  width;   uint  height;  uint stepSize; uint _pad0;
    float sigmaL;  float sigmaZ;  float sigmaN;  float lumFloor;
};

// B3-spline 1-D à-trous weights (sum = 1). 2D weight = h[i+2] × h[j+2].
static constant float kAtrousH1[5] = { 1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0 };

// 3×3 Gaussian weights [[1,2,1],[2,4,2],[1,2,1]]/16 for prefiltering variance
// before it drives the luminance edge-stop (canonical SVGF — Schied 2017 §4.2).
// Without this the per-pixel variance carries its own spatial structure straight
// into the weight field. Indexed h3x3[j+1][i+1].
static constant float kGauss3[3] = { 0.25, 0.5, 0.25 };  // separable 1-D (sum=1)

// Per-pixel variance estimate from a 3×3 spatial neighbourhood of the
// accumulated GI luminance. Outputs R16Float (single channel).
kernel void illumi_svgf_variance(
    texture2d<half,  access::read>  giAccum  [[texture(0)]],  // temporally accumulated GI
    texture2d<half,  access::write> varOut   [[texture(1)]],  // per-pixel variance (R)
    constant SVGFAtrousUniforms&    u        [[buffer(0)]],
    uint2                           gid      [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;

    float m1 = 0.0, m2 = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            float s = dot(float3(giAccum.read(uint2(c)).rgb),
                          float3(0.2126, 0.7152, 0.0722));
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    varOut.write(half4(half(max(0.0, m2 - m1 * m1))), gid);
}

// One level of the SVGF à-trous cascade. Samples at stride `u.stepSize` using
// B3-spline weights multiplied by depth + normal + luminance-variance
// edge-stopping functions. Filters color and propagates variance.
kernel void illumi_svgf_atrous(
    texture2d<half,  access::read>  giIn      [[texture(0)]],  // color in
    texture2d<half,  access::read>  varIn     [[texture(1)]],  // variance in (R)
    texture2d<float, access::read>  gDepth    [[texture(2)]],
    texture2d<half,  access::read>  gNormal   [[texture(3)]],
    texture2d<half,  access::write> giOut     [[texture(4)]],  // filtered color
    texture2d<half,  access::write> varOut    [[texture(5)]],  // filtered variance
    constant SVGFAtrousUniforms&    u         [[buffer(0)]],
    constant FrameUniforms&         frame     [[buffer(1)]],   // invProjection
    uint2                           gid       [[thread_position_in_grid]]
) {
    if (gid.x >= u.width || gid.y >= u.height) return;

    float centerDepth = gDepth.read(gid).r;
    // Sky pixels carry no GI; pass through unchanged.
    if (centerDepth >= 0.99999) {
        giOut.write(giIn.read(gid), gid);
        varOut.write(varIn.read(gid), gid);
        return;
    }

    // Center view-space (linear eye) depth + its screen-space gradient. The
    // gradient lets the depth edge-stop normalise by the EXPECTED depth change
    // from the surface slope (canonical SVGF — Schied 2017 §4.1), so a smooth
    // receding plane passes freely at ANY à-trous stride. The old raw-hardware-
    // depth weight used a fixed denominator: as the stride dilates, the depth
    // delta grows but the threshold doesn't, so wDepth fell off in a stride-
    // structured, view-dependent way → the horizontal masonry on the wall (#65).
    float2 invWH = 1.0 / float2(u.width, u.height);
    float2 ndcC  = (float2(gid) + 0.5) * invWH * 2.0 - 1.0; ndcC.y = -ndcC.y;
    float  zC    = viewPosFromDepth(ndcC, centerDepth, frame.invProjection).z;
    // Forward differences for ∂z/∂x, ∂z/∂y (clamped at the right/bottom edge).
    uint2  gx    = uint2(min(gid.x + 1u, u.width  - 1u), gid.y);
    uint2  gy    = uint2(gid.x, min(gid.y + 1u, u.height - 1u));
    float  dX    = gDepth.read(gx).r, dY = gDepth.read(gy).r;
    float2 ndcX  = (float2(gx) + 0.5) * invWH * 2.0 - 1.0; ndcX.y = -ndcX.y;
    float2 ndcY  = (float2(gy) + 0.5) * invWH * 2.0 - 1.0; ndcY.y = -ndcY.y;
    float  dzdx  = (dX >= 0.99999) ? 0.0 : (viewPosFromDepth(ndcX, dX, frame.invProjection).z - zC);
    float  dzdy  = (dY >= 0.99999) ? 0.0 : (viewPosFromDepth(ndcY, dY, frame.invProjection).z - zC);

    float3 giCenter  = float3(giIn.read(gid).rgb);
    float3 nCenter   = octDecode(float2(gNormal.read(gid).rg));
    float  lumCenter = dot(giCenter, float3(0.2126, 0.7152, 0.0722));

    // Canonical SVGF: prefilter the variance with a separable 3×3 Gaussian before
    // it drives the luminance edge-stop, so the weight field is smooth rather than
    // carrying the raw variance's own structure.
    float gVar = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            gVar += kGauss3[i + 1] * kGauss3[j + 1] * float(varIn.read(uint2(c)).r);
        }
    }

    // Luminance edge-stop bandwidth (Schied 2017 §4.1): σ = σ_L·√Var + ε, used
    // LINEARLY against |Δlum| (not Δlum² over Var — that squared form collapses to
    // a near-binary contour gate as Var→0). This RT-GI is deterministic per frame,
    // so on a converged surface Var genuinely vanishes; a fixed 1e-4 ε then gates
    // the smooth vertical ramp into horizontal masonry bands (issue #65). The floor
    // is therefore luminance-RELATIVE (`u.lumFloor·max(lumCenter,·)`): a smooth
    // low-amplitude gradient blurs freely while a real high-contrast GI edge
    // (shadow boundary) — where the variance estimate is large — still survives.
    float lumSigma = u.sigmaL * sqrt(max(gVar, 0.0))
                   + u.lumFloor * max(lumCenter, 0.02);

    float3 colorSum  = float3(0.0);
    float  varSum    = 0.0;
    float  weightSum = 0.0;

    int step = int(u.stepSize);
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            int2 p = int2(int(gid.x) + i * step, int(gid.y) + j * step);
            if (p.x < 0 || p.y < 0 || p.x >= int(u.width) || p.y >= int(u.height)) continue;
            uint2 np = uint2(p);

            float nd = gDepth.read(np).r;
            if (nd >= 0.99999) continue;

            // B3-spline kernel weight for this tap position.
            float hW = kAtrousH1[i + 2] * kAtrousH1[j + 2];

            // Depth edge-stopping — gradient-normalised LINEAR (view-space) depth.
            // Expected eye-depth change from the surface slope over this tap offset;
            // the denominator scales with the stride so a smooth plane stays w≈1 at
            // every cascade level, while a true depth discontinuity (different
            // surface) still rejects. ε keeps fronto-parallel surfaces stable.
            float  zN       = viewPosFromDepth(
                ((float2(np) + 0.5) * invWH * 2.0 - 1.0) * float2(1.0, -1.0),
                nd, frame.invProjection).z;
            float  expDz    = dzdx * float(i * step) + dzdy * float(j * step);
            float  wDepth   = exp(-abs(zN - zC) / (u.sigmaZ * abs(expDz) + 1e-2));

            // Normal edge-stopping.
            float3 nN   = octDecode(float2(gNormal.read(np).rg));
            float  wNorm = pow(max(0.0, dot(nCenter, nN)), u.sigmaN);

            // Luminance edge-stopping — variance-guided (the SVGF key insight):
            // high variance → wide constraint → more spatial denoising. Linear
            // |Δlum|/σ form with the prefiltered, relative-floored bandwidth above.
            float3 giN  = float3(giIn.read(np).rgb);
            float  lumN = dot(giN, float3(0.2126, 0.7152, 0.0722));
            float  lDif = abs(lumCenter - lumN);
            float  wLum = exp(-lDif / lumSigma);

            float w    = hW * wDepth * wNorm * wLum;
            colorSum  += giN * w;
            varSum    += float(varIn.read(np).r) * (w * w);
            weightSum += w;
        }
    }

    if (weightSum < 1e-6) {
        giOut.write(half4(half3(giCenter), 1.0h), gid);
        varOut.write(varIn.read(gid), gid);
        return;
    }

    giOut.write(half4(half3(max(float3(0.0), colorSum / weightSum)), 1.0h), gid);
    varOut.write(half4(half(varSum / (weightSum * weightSum))), gid);
}
