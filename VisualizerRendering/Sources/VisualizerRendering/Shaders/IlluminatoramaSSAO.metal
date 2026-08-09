// ── ILLUMINATORAMA — SSAO ───────────────────────────────────────────────────
//
// Hemisphere-kernel screen-space ambient occlusion plus its spatial and temporal
// filters.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

static inline float ssaoHash12(float2 p) {
    // De Vries-style cheap hash, [0,1).
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Cosine-weighted HEMISPHERE kernel, +Z = the surface normal.
//
// EVERY z is strictly positive, and that is the whole point. This kernel replaces a
// SPHERE kernel (2026-08-05) whose 10 negative-z vectors were placed BELOW the surface
// plane — i.e. inside the wall — by the `TBN` basis below. A point inside a wall is
// trivially "occluded" by that wall: the depth buffer at its screen position is the
// wall itself, the occlusion test passes, and `rangeCheck` returns 1.0 because the depth
// difference is ~0. Every flat surface in the scene therefore carried a constant
// occlusion pedestal of 1 − 8/16 = 0.500 (two of the ten sank less than the old fixed
// 0.01 m bias). Measured: a bare wall in an open room read AO 0.503, dead flat, when it
// should read 1.0. That is Danny's "painted with a sponge… black mold", and it is why
// raising `ssaoIntensity` deepened it while shrinking `ssaoRadius` pulled the sunken
// samples back under the bias and made it retreat.
//
// The sphere kernel is the Crytek/NVIDIA-SDK-era one, which is only correct in the
// original formulation that rescales occlusion about 0.5. A normal-oriented hemisphere
// estimator like this one requires a hemisphere kernel; the two are not interchangeable.
//
// Generated deterministically (see the doc comment on
// `HouseRenderBridgeGPUTests_WallSplotch.testSSAOFieldIsUnoccludedOnAFlatWall`):
//   u1 = (i+0.5)/16, u2 = bitReverse4(i)/16
//   dir = (sqrt(u1)·cos 2πu2, sqrt(u1)·sin 2πu2, sqrt(1−u1))    ← cosine-weighted
//   len = 0.35 + 0.65·((bitReverse4(i)+0.5)/16)²                ← clusters near contact
// Cosine weighting is not cosmetic: AO is ∫V(ω)cosθ dω/π, so with cosine-distributed
// directions the estimator below is exactly `occlusion / samples` with unit weights.
// The length uses the BIT-REVERSED index so the most grazing samples are not also the
// longest — that pairing is what lets a normal-map tilt push a sample under the true
// geometric surface. As generated, the shallowest sample still sits 0.075 m above the
// plane at radius 0.5, ~5× the bias below.
constant float3 kSsaoKernel[16] = {
    float3( 0.061984,  0.000000,  0.345113),
    float3(-0.163334,  0.000000,  0.507827),
    float3( 0.000000,  0.158674,  0.368724),
    float3( 0.000000, -0.349250,  0.660021),
    float3( 0.137201,  0.137201,  0.310181),
    float3(-0.261156, -0.261156,  0.510303),
    float3(-0.206091,  0.206091,  0.352354),
    float3( 0.427886, -0.427886,  0.644202),
    float3( 0.239532,  0.099218,  0.243540),
    float3(-0.412295, -0.170778,  0.369137),
    float3(-0.132314,  0.319434,  0.250238),
    float3( 0.263683, -0.636588,  0.431023),
    float3( 0.128907,  0.311210,  0.178245),
    float3(-0.241067, -0.581988,  0.271083),
    float3(-0.433441,  0.179537,  0.150895),
    float3( 0.872965, -0.361594,  0.169707)
};

kernel void illumi_ssao(
    depth2d<float, access::read>   gDepth      [[texture(0)]],
    texture2d<half, access::read>  gNormalRgh  [[texture(1)]],
    texture2d<half, access::write> outAO       [[texture(2)]],
    constant FrameUniforms&        frame       [[buffer(0)]],
    uint2                          gid         [[thread_position_in_grid]]
) {
    uint outW = outAO.get_width();
    uint outH = outAO.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    // Early-out when SSAO is disabled — write 1 (no occlusion) so the
    // lighting kernel can read unconditionally without branching.
    if (frame.ssaoIntensity <= 0.0) {
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Map half-res gid → full-res pixel for the G-buffer reads. Using the
    // 2×2 top-left avoids needing a sampler; we accept the small loss in
    // accuracy.
    uint fullW = gDepth.get_width();
    uint fullH = gDepth.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float depth = gDepth.read(fullGid);
    if (depth >= 0.99999) {
        // Sky — no surface to occlude. Treat as fully lit.
        outAO.write(half4(1.0h), gid);
        return;
    }

    half4 nrH = gNormalRgh.read(fullGid);
    float3 Nworld = octDecode(float2(nrH.rg));

    // Reconstruct view-space position and normal. SSAO works in view space so
    // sample offsets are commensurate with the camera, not the world.
    float2 ndc = (float2(fullGid) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 Pview = viewPosFromDepth(ndc, depth, frame.invProjection);
    float3 Nview = normalize((frame.view * float4(Nworld, 0.0)).xyz);

    // Build a TBN basis from a random tangent so the sample pattern is
    // rotated per-pixel.
    float rot = ssaoHash12(float2(gid)) * 6.2831853;
    float3 randomVec = normalize(float3(cos(rot), sin(rot), 0.0));
    float3 T = normalize(randomVec - Nview * dot(randomVec, Nview));
    float3 B = cross(Nview, T);
    float3x3 TBN = float3x3(T, B, Nview);

    float radius = max(0.001, frame.ssaoRadius);
    // Self-occlusion guard, in view-space metres. Scales with BOTH the sample radius
    // and the view depth, because the two error sources it absorbs do: a sample's
    // height above the surface is proportional to `radius` (so a fixed bias cancels
    // real contact signal at small radii and none at large ones), and the half-res
    // G-buffer read plus depth quantisation grow with distance. The old value was a
    // fixed 0.01 m, which combined with the sphere kernel above to make `ssaoRadius`
    // behave as a hidden pedestal control — shrinking the radius pulled sunken samples
    // back under the bias, which is exactly the "lowering the radius makes them retreat
    // to the edges" behaviour Danny reported and worked around.
    float bias = radius * 0.02 + 0.0015 * abs(Pview.z);
    float occlusion = 0.0;
    const uint samples = 16;
    for (uint i = 0; i < samples; ++i) {
        // Move kernel sample into view space oriented along the surface.
        float3 sampleView = Pview + TBN * kSsaoKernel[i] * radius;
        // Project back to NDC to look up the actual depth at that screen
        // position.
        float4 sampleClip = frame.projection * float4(sampleView, 1.0);
        float3 sampleNDC = sampleClip.xyz / sampleClip.w;
        if (any(abs(sampleNDC.xy) > 1.0)) continue;
        float2 sUV = float2(sampleNDC.x, -sampleNDC.y) * 0.5 + 0.5;
        uint2 sPx = uint2(clamp(sUV * float2(fullW, fullH),
                                float2(0.0), float2(fullW - 1, fullH - 1)));
        float sceneDepth = gDepth.read(sPx);
        if (sceneDepth >= 0.99999) continue;
        // Reconstruct view-space scene position at the sample's NDC location.
        float3 scenePos = viewPosFromDepth(sampleNDC.xy, sceneDepth, frame.invProjection);
        // Range check — only count occluders within `radius` of P.
        float rangeCheck = smoothstep(0.0, 1.0, radius / max(1e-4, abs(Pview.z - scenePos.z)));
        // View-space Z is negative going away from the camera, so an occluder
        // is "in front of" the sample when its Z is GREATER (less negative).
        // With the hemisphere kernel every sample sits ABOVE its own surface, so a
        // plane can no longer occlude itself at any orientation and `bias` is back to
        // guarding quantisation only — it is not load-bearing for flatness.
        if (scenePos.z > sampleView.z + bias) {
            occlusion += rangeCheck;
        }
    }
    float ao = 1.0 - (occlusion / float(samples)) * frame.ssaoIntensity;
    ao = clamp(ao, 0.0, 1.0);
    outAO.write(half4(half(ao)), gid);
}

// ── Phase 4.39: SSAO bilateral spatial filter ────────────────────────────────
//
// Depth + normal guided bilateral filter over the half-resolution raw AO
// texture. A 3×3 window reads each neighbor's AO value and weights it by:
//
//   exp(−|depth_diff| · kDepth) · pow(max(0, dot(n_center, n_neighbor)), kNorm)
//
// Depth weights reject samples across geometric boundaries (contact edges);
// normal weights reject samples on curved surfaces where orientation diverges.
// Output feeds the temporal pass rather than lighting directly, so the spatial
// pass only needs to be stable enough for temporal to converge cleanly.

kernel void illumi_ssao_spatial(
    texture2d<half, access::read>   rawAO      [[texture(0)]],  // half-res
    depth2d<float,  access::read>   gDepth     [[texture(1)]],  // full-res
    texture2d<half, access::read>   gNormalRgh [[texture(2)]],  // full-res
    texture2d<half, access::write>  outAO      [[texture(3)]],  // half-res
    constant FrameUniforms&         frame      [[buffer(0)]],
    uint2                           gid        [[thread_position_in_grid]]
) {
    uint halfW = outAO.get_width();
    uint halfH = outAO.get_height();
    if (gid.x >= halfW || gid.y >= halfH) return;

    if (frame.ssaoDenoiseEnabled == 0u || frame.ssaoIntensity <= 0.0) {
        outAO.write(rawAO.read(gid), gid);
        return;
    }

    uint fullW = gDepth.get_width();
    uint fullH = gDepth.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float centerDepth = gDepth.read(fullGid);
    if (centerDepth >= 0.99999) {
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Reconstruct center view-Z for depth weighting.
    float2 ndcC = (float2(fullGid) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
    ndcC.y = -ndcC.y;
    float3 PviewC = viewPosFromDepth(ndcC, centerDepth, frame.invProjection);
    float3 NworldC = octDecode(float2(gNormalRgh.read(fullGid).rg));

    const float kDepth = 5.0;   // depth sensitivity (larger = tighter boundary preservation)
    const float kNorm  = 16.0;  // normal exponent  (larger = more sensitive to curvature)

    float totalAO     = 0.0;
    float totalWeight = 0.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 halfN = int2(int(gid.x) + i, int(gid.y) + j);
            halfN = clamp(halfN, int2(0), int2(int(halfW) - 1, int(halfH) - 1));
            uint2 fullN = min(uint2(halfN) * 2u, uint2(fullW - 1, fullH - 1));

            float nd = gDepth.read(fullN);
            if (nd >= 0.99999) continue;

            float2 ndcN = (float2(fullN) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
            ndcN.y = -ndcN.y;
            float3 PviewN   = viewPosFromDepth(ndcN, nd, frame.invProjection);
            float3 NworldN  = octDecode(float2(gNormalRgh.read(fullN).rg));

            float depthDiff  = abs(PviewC.z - PviewN.z);
            float wDepth     = exp(-depthDiff * kDepth);
            float wNorm      = pow(max(0.0, dot(NworldC, NworldN)), kNorm);
            float w          = wDepth * wNorm;

            totalAO     += float(rawAO.read(uint2(halfN)).r) * w;
            totalWeight += w;
        }
    }

    float filtered = (totalWeight > 1e-6) ? (totalAO / totalWeight)
                                           : float(rawAO.read(gid).r);
    outAO.write(half4(half(filtered), 0.0h, 0.0h, 1.0h), gid);
}

// ── Phase 4.39: SSAO temporal accumulation ───────────────────────────────────
//
// Reprojects the previous frame's AO history using the full-res velocity
// texture, clamps the history sample into a scalar variance band, and blends
// a high-weight (≈0.9) history with the current spatially-filtered AO. This
// converts the 16-sample single-tap AO into effectively hundreds of samples
// spread over time — AO "for free" from the temporal integration.
//
// Velocity is at full-res; for a half-res pixel at `gid` we read velocity
// at `gid×2` (the matching full-res texel). UV-space velocity is resolution-
// independent so the reprojection math is identical.

kernel void illumi_ssao_temporal(
    texture2d<half, access::read>       filteredAO   [[texture(0)]],  // half-res current
    texture2d<half, access::sample>     historyAO    [[texture(1)]],  // half-res history
    texture2d<half, access::read>       velocity     [[texture(2)]],  // full-res
    texture2d<half, access::write>      outAO        [[texture(3)]],  // half-res output
    texture2d<half, access::read_write> sampleCount  [[texture(4)]],  // half-res r16Float
    constant FrameUniforms&             frame        [[buffer(0)]],
    uint2                               gid          [[thread_position_in_grid]]
) {
    uint halfW = outAO.get_width();
    uint halfH = outAO.get_height();
    if (gid.x >= halfW || gid.y >= halfH) return;

    float current = float(filteredAO.read(gid).r);

    // Reset sample count and pass through on first frame or when disabled.
    if (frame.ssaoDenoiseEnabled == 0u || frame.ssaoIsFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outAO.write(half4(half(current)), gid);
        return;
    }

    // Read accumulated sample count from previous frame.
    float N = float(sampleCount.read(gid).r);

    uint fullW = velocity.get_width();
    uint fullH = velocity.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));
    float2 vel = float2(velocity.read(fullGid).rg);

    float2 halfUV = (float2(gid) + 0.5) / float2(halfW, halfH);
    float2 histUV = halfUV - vel;

    // Screen-edge disocclusion: no history available → reset count.
    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outAO.write(half4(half(current)), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float hist = float(historyAO.sample(samp, histUV).r);

    // Scalar variance clamp over the 3×3 half-res neighborhood.
    float m1 = 0.0, m2 = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(halfW) - 1, int(halfH) - 1));
            float s = float(filteredAO.read(uint2(c)).r);
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float sigma = max(sqrt(max(0.0, m2 - m1 * m1)), 0.03);
    hist = clamp(hist, m1 - 1.5 * sigma, m1 + 1.5 * sigma);

    float velMag = length(vel);
    float disoccBlend = smoothstep(0.004, 0.025, velMag);

    // Adaptive blend: 1/N drives fast convergence in early frames; once N is
    // large enough the steady-state floor (1 - ssaoTemporalBlend) takes over.
    // Velocity disocclusion resets N so newly-revealed surfaces reconverge quickly.
    if (disoccBlend > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float minAlpha = 1.0 - frame.ssaoTemporalBlend;
    float alpha = max(minAlpha, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 4.0), disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    float result = mix(hist, current, alpha);
    outAO.write(half4(half(result)), gid);
}
