#pragma once

// ── ILLUMINATORAMA — DDGI PROBE FIELD: TYPES AND SAMPLING ───────────────────
//
// A HEADER, not just a .metal, because `sampleDDGIIrradiance` is called from
// BOTH the deferred lighting kernel (IlluminatoramaLighting.metal) and the
// probe trace kernel (IlluminatoramaDDGI.metal). The probe atlas addressing
// helpers it is built on travel with it.

#include <metal_stdlib>
using namespace metal;

// ── Phase 3.1: DDGI — Dynamic Diffuse Global Illumination ───────────────────
//
// Analytic one-bounce probe GI. No hardware RT — rays are intersected against
// the scene's unit primitives (box, sphere, ground) in local space via the
// per-instance invModelMatrix stored in DDGIInstanceData[].
//
// Three kernels per frame (see IlluminatoramaRenderer.swift `encodeDDGIFrame`):
//   illumi_ddgi_trace             — fire rays from each probe
//   illumi_ddgi_update_irradiance — integrate → irradiance atlas (rgba16Half)
//   illumi_ddgi_update_depth      — integrate → depth atlas (rg16Half, mean+mean²)
//
// The lighting kernel (below) samples both atlases via trilinear probe blend +
// Chebyshev visibility, replacing the irradianceCube diffuse term when enabled.
//
// Struct layout rules: same SIMD3<Float>+Float=16B grouping as FrameUniforms.
// Mirror structs live in IlluminatoramaRenderer.swift (IlluminatoramaDDGIUniforms,
// DDGIGPUInstanceData). When changing one side, change both.

struct DDGIUniforms {
    float3   gridOrigin;              // world-space position of probe (0,0,0)
    uint     gridDimsX;               // probes along X
    float3   directionalLightDir;     // toward light, world space
    uint     gridDimsY;               // probes along Y
    float3   directionalLightColor;   // linear HDR
    uint     gridDimsZ;               // probes along Z
    float    probeSpacing;            // metres between adjacent probes
    uint     raysPerProbe;            // rays dispatched per probe per frame
    float    hysteresis;              // EMA weight kept from previous irr frame
    float    depthHysteresis;         // same for depth atlas
    float    irradianceScale;         // post-multiplier on final GI contribution
    uint     enabled;                 // 0 = skip probe lookup in lighting pass
    uint     irrTileSize;             // interior octahedral tile width (e.g. 6)
    uint     depthTileSize;           // interior tile width for depth atlas (e.g. 14)
    uint     instanceCount;           // entries in DDGIInstanceData[]
    // Phase 3.3 — two-bounce GI. 0 = single-bounce direct only (Phase 3.1
    // behaviour); 1 = trace kernel ALSO samples the previous-frame
    // irradiance atlas at the hit point and adds it to the recorded
    // radiance. Requires ping-pong of both atlases on the host side.
    uint     twoBounceEnabled;
    uint     emitterCount;            // entries in DDGIPointEmitter[] bound at buffer(3)
    uint     _pad2;
    // Total: 96 bytes, stride 96.
};

// Octahedral encode: unit direction → [-1,1]² (full-sphere projection).
static float2 ddgiOctEncode(float3 v) {
    float3 p = v / (abs(v.x) + abs(v.y) + abs(v.z));
    if (p.z < 0.0f) {
        float2 xy = (1.0f - abs(p.yx)) * float2(p.x >= 0.0f ? 1.0f : -1.0f,
                                                  p.y >= 0.0f ? 1.0f : -1.0f);
        return xy;
    }
    return p.xy;
}

// Normalised UV into the irradiance atlas for a given probe + oct-encoded dir.
static float2 ddgiIrrAtlasUV(float2 octNorm, uint probeIdx,
                               constant DDGIUniforms& ddgi) {
    uint pad   = ddgi.irrTileSize + 2;
    uint atlasW = pad * ddgi.gridDimsX * ddgi.gridDimsZ;
    uint atlasH = pad * ddgi.gridDimsY;
    uint px    = probeIdx % ddgi.gridDimsX;
    uint py    = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz    = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    float2 tileOrigin = float2((px + pz * ddgi.gridDimsX) * pad + 1,
                                py * pad + 1);
    float2 interior   = (octNorm + 1.0f) * 0.5f;  // [-1,1]² → [0,1]²
    return (tileOrigin + interior * float(ddgi.irrTileSize)) / float2(atlasW, atlasH);
}

static float2 ddgiDepthAtlasUV(float2 octNorm, uint probeIdx,
                                 constant DDGIUniforms& ddgi) {
    uint pad    = ddgi.depthTileSize + 2;
    uint atlasW = pad * ddgi.gridDimsX * ddgi.gridDimsZ;
    uint atlasH = pad * ddgi.gridDimsY;
    uint px     = probeIdx % ddgi.gridDimsX;
    uint py     = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz     = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    float2 tileOrigin = float2((px + pz * ddgi.gridDimsX) * pad + 1,
                                py * pad + 1);
    float2 interior   = (octNorm + 1.0f) * 0.5f;
    return (tileOrigin + interior * float(ddgi.depthTileSize)) / float2(atlasW, atlasH);
}

// ── DDGI probe irradiance lookup (called from illumi_lighting) ────────────────
//
// Trilinear blend over the 8 corner probes containing worldPos. Each probe
// contributes irradiance weighted by:
//   - trilinear distance weight
//   - back-face penalty (probes behind the surface normal get less weight)
//   - Chebyshev visibility (probes blocked by geometry get less weight)
//
// The surface normal N determines which octahedral texel of the irradiance
// atlas to sample (we want radiance arriving AT worldPos along the hemisphere
// of N).

static float3 sampleDDGIIrradiance(
    float3 worldPos,
    float3 N,
    texture2d<half,  access::sample> irrAtlas,
    texture2d<half,  access::sample> depthAtlas,
    constant DDGIUniforms&           ddgi
) {
    float3 local = (worldPos - ddgi.gridOrigin) / ddgi.probeSpacing;
    int3   base  = int3(floor(local));
    float3 alpha = local - float3(base);
    int3   dims  = int3(int(ddgi.gridDimsX), int(ddgi.gridDimsY), int(ddgi.gridDimsZ));

    constexpr sampler irrSmp(filter::linear, address::clamp_to_edge);
    constexpr sampler depSmp(filter::linear, address::clamp_to_edge);

    float3 irradiance  = float3(0.0f);
    float  totalWeight = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 probe  = base + offset;
        if (any(probe < int3(0)) || any(probe >= dims)) continue;

        uint probeIdx = uint(probe.x)
                      + uint(probe.y) * ddgi.gridDimsX
                      + uint(probe.z) * ddgi.gridDimsX * ddgi.gridDimsY;

        float3 probePos = ddgi.gridOrigin + float3(probe) * ddgi.probeSpacing;
        float3 toPoint  = worldPos - probePos;
        float  dist     = length(toPoint);
        float3 dir      = toPoint / max(dist, 1e-4f);

        // Trilinear weight.
        float bx = (offset.x == 0) ? (1.0f - alpha.x) : alpha.x;
        float by = (offset.y == 0) ? (1.0f - alpha.y) : alpha.y;
        float bz = (offset.z == 0) ? (1.0f - alpha.z) : alpha.z;
        float trilinear = bx * by * bz + 1e-5f;

        // Back-face penalty: probes behind the surface normal contribute less.
        float NdotD = dot(N, -dir);
        float backfaceW = ((NdotD + 1.0f) * 0.5f);
        backfaceW = backfaceW * backfaceW + 0.02f;

        // Chebyshev visibility: depth atlas stores (mean, mean²) of hit dists.
        float2 depUV  = ddgiDepthAtlasUV(ddgiOctEncode(dir), probeIdx, ddgi);
        float2 depSmp2 = float2(depthAtlas.sample(depSmp, depUV).rg);
        float  mean    = depSmp2.r;
        float  mean2   = depSmp2.g;
        float  variance = max(0.0f, mean2 - mean * mean);
        float  cheb = 1.0f;
        if (dist > mean) {
            float diff = dist - mean;
            cheb = variance / (variance + diff * diff);
            cheb = cheb * cheb * cheb;  // crush toward 0 for deep occlusion
        }

        float weight = max(0.0f, trilinear * backfaceW * cheb);

        // Sample irradiance at the surface normal direction.
        float2 irrUV   = ddgiIrrAtlasUV(ddgiOctEncode(N), probeIdx, ddgi);
        float3 probeIrr = float3(irrAtlas.sample(irrSmp, irrUV).rgb);

        irradiance  += probeIrr * weight;
        totalWeight += weight;
    }
    if (totalWeight > 1e-4f) irradiance /= totalWeight;
    return irradiance * ddgi.irradianceScale;
}
