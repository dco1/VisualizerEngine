// ── ILLUMINATORAMA — IBL BAKES ──────────────────────────────────────────────
//
// Sky-probe irradiance cube, GGX-prefiltered specular cube, and the split-sum
// DFG LUT. All three run once when the sky changes, not per frame.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// Standard cubemap face → direction. Matches Metal's samplerCube convention
// (face 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z) — written so the direction
// returned here, fed to samplerCube::sample(d), would yield this same texel.
static inline float3 cubeDirForFace(uint face, float2 uv) {
    float s = uv.x * 2.0 - 1.0;
    float t = uv.y * 2.0 - 1.0;
    float3 d;
    switch (face) {
        case 0: d = float3( 1.0, -t, -s); break;   // +X
        case 1: d = float3(-1.0, -t,  s); break;   // -X
        case 2: d = float3( s,  1.0,  t); break;   // +Y
        case 3: d = float3( s, -1.0, -t); break;   // -Y
        case 4: d = float3( s, -t,  1.0); break;   // +Z
        default: d = float3(-s, -t, -1.0); break;  // -Z
    }
    return normalize(d);
}

// Hammersley low-discrepancy 2D sequence in [0,1). Used for GGX importance
// sampling of the prefilter cubemap.
static inline float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
static inline float2 hammersley(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverseVdC(i));
}

// Importance-sample GGX in tangent space, then rotate into the N-aligned
// world-space frame. `roughness` is the surface roughness (linear-space α,
// not perceptual). Returns the half-vector H.
static inline float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * M_PI_F * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float3 H_tan = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    // Build a tangent frame around N. The branch avoids degeneracy when N is
    // close to ±Y (where the standard up = (0,1,0) cross would collapse).
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    return normalize(T * H_tan.x + B * H_tan.y + N * H_tan.z);
}

// ── IBL bake — diffuse irradiance ────────────────────────────────────────────
//
// For each output texel (face, x, y) we compute the cosine-weighted hemisphere
// integral of the sky in the corresponding world direction. Output stores
// I(N)/π so the lighting pass can just multiply by albedo * kD — the
// LearnOpenGL / Karis convention. 16×8 = 128 directional samples per texel at
// 16² output → ~200K equirect lookups per face, ~1.2M total. Cheap on Apple
// Silicon (well under a millisecond).

kernel void illumi_irradiance_bake(
    texture2d<float, access::sample>  sky        [[texture(0)]],
    texturecube<half,  access::write> outCube    [[texture(1)]],
    constant float&                   bakeDesat  [[buffer(0)]],
    uint3                              gid       [[thread_position_in_grid]]
) {
    uint W = outCube.get_width();
    if (gid.x >= W || gid.y >= W || gid.z >= 6u) return;

    float2 uv = (float2(gid.xy) + 0.5) / float(W);
    float3 N = cubeDirForFace(gid.z, uv);

    // Tangent frame around N.
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    const uint N_PHI = 16;
    const uint N_THETA = 8;
    float3 acc = float3(0);
    for (uint p = 0; p < N_PHI; ++p) {
        float phi = (float(p) + 0.5) * (2.0 * M_PI_F / float(N_PHI));
        float cphi = cos(phi), sphi = sin(phi);
        for (uint t = 0; t < N_THETA; ++t) {
            float theta = (float(t) + 0.5) * (0.5 * M_PI_F / float(N_THETA));
            float sT = sin(theta), cT = cos(theta);
            // Tangent-space sample dir → world dir aligned with N.
            float3 dir = T * (sT * cphi) + B * (sT * sphi) + N * cT;
            float3 L = sampleSkyEquirect(sky, dir);
            // Optional desaturation: pull near-monochromatic skies (broiler,
            // vivid sunset) toward luminance so diffuse IBL doesn't flood all
            // surfaces with one hue. Visual sky is baked into the cube at its
            // full colour; only the irradiance integral is pulled neutral here.
            if (bakeDesat > 0.0) {
                float lum = dot(L, float3(0.2126, 0.7152, 0.0722));
                L = mix(L, float3(lum), bakeDesat);
            }
            acc += L * cT * sT;        // L * cos(θ) * sin(θ) (Jacobian)
        }
    }
    // Riemann-sum factor = (π/2 / N_THETA) * (2π / N_PHI), then divide by π
    // to bake in the Lambertian 1/π. Net factor: π / (N_PHI * N_THETA).
    float3 irradiance = acc * (M_PI_F / float(N_PHI * N_THETA));
    outCube.write(half4(half3(irradiance), 1.0h), gid.xy, gid.z);
}

// ── IBL bake — GGX-prefiltered specular ──────────────────────────────────────
//
// One dispatch per mip level. The caller binds a texture VIEW of the single
// mip and passes its roughness via the `Prefilter` struct. mip 0 = roughness
// 0 (mirror), mip mipCount-1 = roughness 1 (matte). Karis split-sum
// approximation (V = N = R) so the sampled cubemap can be re-used across
// view directions in the lighting pass.

struct PrefilterBakeParams {
    float roughness;
    uint  faceWidth;
    uint  sampleCount;
    float bakeDesat;   // was _pad — desaturate equirect samples (0=off, 1=grey)
};

kernel void illumi_prefilter_bake(
    texture2d<float, access::sample>   sky      [[texture(0)]],
    texturecube<half,  access::write>  outMip   [[texture(1)]],
    constant PrefilterBakeParams&      params   [[buffer(0)]],
    uint3                              gid      [[thread_position_in_grid]]
) {
    uint W = params.faceWidth;
    if (gid.x >= W || gid.y >= W || gid.z >= 6u) return;

    float2 uv = (float2(gid.xy) + 0.5) / float(W);
    float3 N = cubeDirForFace(gid.z, uv);
    float3 V = N;
    float roughness = params.roughness;
    float bakeDesat = params.bakeDesat;

    // Mirror surface shortcut — no GGX integration needed.
    if (roughness <= 0.0001) {
        float3 sky_val = sampleSkyEquirect(sky, N);
        if (bakeDesat > 0.0) {
            float lum = dot(sky_val, float3(0.2126, 0.7152, 0.0722));
            sky_val = mix(sky_val, float3(lum), bakeDesat);
        }
        outMip.write(half4(half3(sky_val), 1.0h), gid.xy, gid.z);
        return;
    }

    uint samples = max(params.sampleCount, 4u);
    float3 prefiltered = float3(0);
    float totalWeight = 0.0;
    for (uint i = 0; i < samples; ++i) {
        float2 Xi = hammersley(i, samples);
        float3 H = importanceSampleGGX(Xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = saturate(dot(N, L));
        if (NdotL > 0.0) {
            float3 sky_val = sampleSkyEquirect(sky, L);
            if (bakeDesat > 0.0) {
                float lum = dot(sky_val, float3(0.2126, 0.7152, 0.0722));
                sky_val = mix(sky_val, float3(lum), bakeDesat);
            }
            prefiltered += sky_val * NdotL;
            totalWeight += NdotL;
        }
    }
    prefiltered /= max(totalWeight, 1e-4);
    outMip.write(half4(half3(prefiltered), 1.0h), gid.xy, gid.z);
}

// ── Split-sum DFG LUT bake (Phase 3.2) ──────────────────────────────────────
//
// Pre-integrates the specular BRDF into a 2D LUT keyed on (NdotV, roughness).
// For each texel we integrate over a hemisphere of GGX-importance-sampled half-
// vectors with a fixed N = (0,0,1) and a V reconstructed from NdotV. Output is
// (scale, bias) from the split-sum: F0 * scale + bias = the environment BRDF
// integral. Baked once on renderer init (view-independent), never per-frame.
//
// Reference: Karis 2013 "Real Shading in Unreal Engine 4" — equations 4–5.

kernel void illumi_dfg_bake(
    texture2d<half, access::write> outLUT [[texture(0)]],
    uint2                          gid   [[thread_position_in_grid]]
) {
    uint W = outLUT.get_width();
    uint H = outLUT.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // Texel centres map to (0,1] so the LUT never samples at the degenerate
    // NdotV=0 or roughness=0 poles.
    float NdotV    = (float(gid.x) + 0.5) / float(W);
    float roughness = (float(gid.y) + 0.5) / float(H);

    // Reconstruct a view vector against N = (0,0,1) with the given NdotV.
    float sinTheta = sqrt(max(0.0, 1.0 - NdotV * NdotV));
    float3 V = float3(sinTheta, 0.0, NdotV);
    float3 N = float3(0.0, 0.0, 1.0);

    const uint SAMPLES = 512;
    float scale = 0.0;
    float bias  = 0.0;

    for (uint i = 0; i < SAMPLES; ++i) {
        float2 Xi = hammersley(i, SAMPLES);
        float3 H  = importanceSampleGGX(Xi, N, roughness);
        float3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
            float G     = geometrySmith(NdotV, NdotL, roughness);
            // G_Vis: the split-sum measure-change factor.
            float G_Vis = (G * VdotH) / max(NdotH * NdotV, 1e-6);
            float Fc    = pow(1.0 - VdotH, 5.0);
            scale += (1.0 - Fc) * G_Vis;
            bias  += Fc * G_Vis;
        }
    }

    outLUT.write(
        half4(half(scale / float(SAMPLES)), half(bias / float(SAMPLES)), 0.0h, 1.0h),
        gid
    );
}
