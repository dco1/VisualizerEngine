// ── ILLUMINATORAMA — SCREEN-SPACE REFLECTIONS ───────────────────────────────
//
// Ray-march gather, temporal accumulation and the composite back into the HDR
// target.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaDenoise.h"
using namespace metal;

// ── Screen-space reflections — gather (Phase 2 / 4.39) ───────────────────────
//
// Linear march in view space along the reflection vector. Phase 4.39 splits
// what was previously a single inline composite into three passes:
//
//   illumi_ssr_gather  (this kernel) — marches rays, outputs the pre-weighted
//       SSR delta (RGB = reflection contribution, A = hit mask) into a
//       dedicated raw texture instead of writing directly into the HDR composite.
//
//   illumi_ssr_temporal — temporally accumulates the raw SSR signal across
//       frames using a YCoCg-space variance clamp (same style as the TAA
//       upgrade), dramatically reducing single-sample noise on rough surfaces.
//
//   illumi_ssr_composite — blends the denoised SSR into the HDR composite.
//
// On miss, writes half4(0) so the temporal and composite passes see no delta.
// The IBL specular term (already in hdrTexture from illumi_lighting) covers
// off-screen rays — no double-count.

kernel void illumi_ssr_gather(
    texture2d<half,  access::read>     inHDR           [[texture(0)]],
    texture2d<half,  access::read>     gAlbedoMet      [[texture(1)]],
    texture2d<half,  access::read>     gNormalRgh      [[texture(2)]],
    depth2d<float,   access::read>     gDepth          [[texture(3)]],
    texture2d<half,  access::write>    outSSRRaw       [[texture(4)]],
    constant FrameUniforms&            frame           [[buffer(0)]],
    uint2                              gid             [[thread_position_in_grid]]
) {
    uint W = outSSRRaw.get_width();
    uint H = outSSRRaw.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float depth = gDepth.read(gid);
    if (frame.ssrIntensity <= 0.0 || depth >= 0.99999) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }

    half4 amH = gAlbedoMet.read(gid);
    half4 nrH = gNormalRgh.read(gid);
    float metallic  = float(amH.a);
    float roughness = max(0.045, float(nrH.b));
    if (roughness > 0.7) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }
    if (metallic < 0.02 && max(max(float(amH.r), float(amH.g)), float(amH.b)) < 0.02) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }
    float3 albedo = float3(amH.rgb);
    float3 Nworld = octDecode(float2(nrH.rg));

    float2 ndc = (float2(gid) + 0.5) / float2(W, H) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4x4 invProj = frame.invProjection;
    float3 Pview = viewPosFromDepth(ndc, depth, invProj);
    float3 Nview = normalize((frame.view * float4(Nworld, 0.0)).xyz);
    float3 Vview = normalize(-Pview);
    float3 Rview = reflect(-Vview, Nview);
    if (Rview.z >= 0.0) {
        outSSRRaw.write(half4(0.0h), gid);
        return;
    }

    uint maxSteps = max(8u, min(frame.ssrMaxSteps, 128u));
    float stepLen = max(0.01, frame.ssrMaxDistance / float(maxSteps));
    float thickness = max(0.01, frame.ssrThickness);

    bool hit = false;
    float2 hitUV = float2(0.0);
    for (uint s = 1; s <= maxSteps; ++s) {
        float3 rayPos = Pview + Rview * (stepLen * float(s));
        if (rayPos.z >= -0.05) break;
        float4 clip = frame.projection * float4(rayPos, 1.0);
        float3 rNDC = clip.xyz / clip.w;
        if (any(abs(rNDC.xy) > 1.0)) break;
        float2 rUV = float2(rNDC.x, -rNDC.y) * 0.5 + 0.5;
        uint2 rPx = uint2(clamp(rUV * float2(W, H),
                                float2(0.0), float2(W - 1, H - 1)));
        float sceneDepth = gDepth.read(rPx);
        if (sceneDepth >= 0.99999) continue;
        float3 scenePos = viewPosFromDepth(
            float2(rNDC.x, rNDC.y), sceneDepth, invProj
        );
        float dz = scenePos.z - rayPos.z;
        if (dz < 0.0 && dz > -thickness) {
            hit = true;
            hitUV = rUV;
            break;
        }
    }

    if (hit) {
        uint2 hPx = uint2(clamp(hitUV * float2(W, H),
                                float2(0.0), float2(W - 1, H - 1)));
        float3 refl = float3(inHDR.read(hPx).rgb);
        float3 F0 = mix(float3(0.04), albedo, metallic);
        float NdotV = saturate(dot(Nview, Vview));
        float3 F = F0 + (1.0 - F0) * pow(saturate(1.0 - NdotV), 5.0);
        float2 fade2 = smoothstep(float2(0.0), float2(0.15),
                                  min(hitUV, 1.0 - hitUV));
        float fade = fade2.x * fade2.y;
        float roughAttn = 1.0 - smoothstep(0.2, 0.7, roughness);
        float3 delta = refl * F * fade * roughAttn * frame.ssrIntensity;
        outSSRRaw.write(half4(half3(delta), 1.0h), gid);
    } else {
        outSSRRaw.write(half4(0.0h), gid);
    }
}

// ── Phase 4.39: SSR temporal accumulation ────────────────────────────────────
//
// Temporally accumulates the raw SSR gather signal (RGB = weighted reflection
// delta, A = hit mask) using a per-channel YCoCg variance clamp and a high-
// weight history blend (≈0.85). Because SSR is full-resolution and in the same
// UV space as the TAA pass, the reprojection uses the same velocity texture.
//
// Running SSR temporal BEFORE the HDR TAA pass means the denoised signal
// enters the composite with a dedicated high-alpha history that TAA can
// then smooth further — giving effectively two rounds of temporal filtering
// to the most visually noisy signal in the pipeline.

kernel void illumi_ssr_temporal(
    texture2d<half,  access::read>       ssrRaw       [[texture(0)]],  // current gather
    texture2d<half,  access::sample>     histSSR      [[texture(1)]],  // history
    texture2d<half,  access::read>       velocity     [[texture(2)]],  // full-res
    texture2d<half,  access::write>      outSSR       [[texture(3)]],  // denoised output
    texture2d<half,  access::read_write> sampleCount  [[texture(4)]],  // full-res r16Float
    constant FrameUniforms&              frame        [[buffer(0)]],
    uint2                                gid          [[thread_position_in_grid]]
) {
    uint W = outSSR.get_width();
    uint H = outSSR.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float3 current = float3(ssrRaw.read(gid).rgb);

    if (frame.ssrDenoiseEnabled == 0u || frame.ssrIsFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outSSR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    float N = float(sampleCount.read(gid).r);

    float2 uv  = (float2(gid) + 0.5) / float2(W, H);
    float2 vel = float2(velocity.read(gid).rg);
    float2 histUV = uv - vel;

    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outSSR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 hist = float3(histSSR.sample(samp, histUV).rgb);

    // YCoCg variance clamp (same algorithm as the HDR TAA upgrade).
    float3 currentYCoCg = RGBtoYCoCg(current);
    float3 histYCoCg    = RGBtoYCoCg(hist);

    float3 m1 = float3(0.0), m2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(W) - 1, int(H) - 1));
            float3 s = RGBtoYCoCg(float3(ssrRaw.read(uint2(c)).rgb));
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float3 sigma = max(sqrt(max(float3(0.0), m2 - m1 * m1)), float3(0.04));
    histYCoCg = clamp(histYCoCg, m1 - 1.5 * sigma, m1 + 1.5 * sigma);

    float velMag = length(vel);
    float disoccBlend = smoothstep(0.004, 0.025, velMag);

    if (disoccBlend > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float minAlpha = 1.0 - frame.ssrTemporalBlend;
    float alpha = max(minAlpha, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 4.0), disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    float3 resultYCoCg = mix(histYCoCg, currentYCoCg, alpha);
    float3 result = YCoCgtoRGB(resultYCoCg);
    outSSR.write(half4(half3(result), 1.0h), gid);
}

// ── Phase 4.39: SSR composite ─────────────────────────────────────────────────
//
// Combines the denoised SSR signal (or raw gather when temporal is off) with
// the base HDR lighting output to produce the final HDR composite texture that
// RT, volumetric, particles, TAA, bloom, and tonemap consume.
//
// Output = hdrTexture + ssrSource.rgb. When SSR is disabled (ssrRaw = 0),
// this is a plain copy of hdrTexture into hdrComposite.

kernel void illumi_ssr_composite(
    texture2d<half, access::read>  inHDR     [[texture(0)]],  // base lighting
    texture2d<half, access::read>  ssrSource [[texture(1)]],  // denoised SSR delta
    texture2d<half, access::write> outHDR    [[texture(2)]],  // hdrComposite
    constant FrameUniforms&        frame     [[buffer(0)]],
    uint2                          gid       [[thread_position_in_grid]]
) {
    uint W = outHDR.get_width();
    uint H = outHDR.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float3 base  = float3(inHDR.read(gid).rgb);
    float3 delta = float3(ssrSource.read(gid).rgb);
    outHDR.write(half4(half3(base + delta), 1.0h), gid);
}
