// ── ILLUMINATORAMA — DDGI PROBE UPDATE KERNELS ──────────────────────────────
//
// Trace, irradiance update and depth update. The probe types and the irradiance
// SAMPLER they share with the lighting kernel live in IlluminatoramaDDGI.h.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaDDGI.h"
using namespace metal;

struct DDGIInstanceData {
    float4x4 invModelMatrix;   // world→local (transforms the ray)
    float4x4 normalMatrix;     // transpose(inverse(upper3x3(model))), padded
    float3   albedo;
    float    metallic;
    float3   emission;         // pre-multiplied emissive radiance
    float    roughness;
    uint     meshKind;         // 0=box, 1=sphere, 2=ground
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
    // Total: 176 bytes, stride 176.
};

struct DDGIRayRecord {
    float4 dirAndDist;  // xyz=ray dir (world), w=hit dist (-1=miss/sky)
    float4 radiance;    // xyz=HDR radiance at hit, w unused
    // Total: 32 bytes.
};

// An analytic point emitter contributed by a particle field. The trace kernel
// evaluates these per-ray so probes accumulate particle emission and propagate
// it to all scene surfaces as proper indirect light.
// Swift mirror: DDGIGPUEmitter in IlluminatoramaRenderer.swift.
struct DDGIPointEmitter {
    float3 position;  // world-space centroid
    float  radius;    // falloff radius in metres; contribution = 0 outside
    float3 color;     // pre-multiplied HDR irradiance
    float  _pad;
    // Total: 32 bytes.
};

// ── DDGI Helpers ─────────────────────────────────────────────────────────────

// Spherical Fibonacci sampling — deterministic, well-distributed sphere dirs.
static float3 ddgiSphericalFibonacci(uint i, uint n) {
    const float PHI = 1.6180339887498948482f;
    float phi      = 2.0f * M_PI_F * fract(float(i) * (PHI - 1.0f));
    float cosTheta = 1.0f - (2.0f * float(i) + 1.0f) / float(n);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Octahedral decode: [-1,1]² → unit direction.
static float3 ddgiOctDecode(float2 uv) {
    float3 v = float3(uv, 1.0f - abs(uv.x) - abs(uv.y));
    if (v.z < 0.0f) {
        float2 xy = (1.0f - abs(v.yx)) * float2(v.x >= 0.0f ? 1.0f : -1.0f,
                                                   v.y >= 0.0f ? 1.0f : -1.0f);
        v.xy = xy;
    }
    return normalize(v);
}

// Probe world-space position from flat index.
static float3 ddgiProbePos(uint probeIdx, constant DDGIUniforms& ddgi) {
    uint px = probeIdx % ddgi.gridDimsX;
    uint py = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    return ddgi.gridOrigin + float3(px, py, pz) * ddgi.probeSpacing;
}

// Atlas tile layout: tiles are (probeX + probeZ*gridDimsX, probeY), 1px border.
static uint2 ddgiIrrTexel(uint tx, uint ty, uint probeIdx,
                           constant DDGIUniforms& ddgi) {
    uint pad  = ddgi.irrTileSize + 2;
    uint px   = probeIdx % ddgi.gridDimsX;
    uint py   = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz   = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    return uint2((px + pz * ddgi.gridDimsX) * pad + 1 + tx,
                  py * pad + 1 + ty);
}

static uint2 ddgiDepthTexel(uint tx, uint ty, uint probeIdx,
                              constant DDGIUniforms& ddgi) {
    uint pad = ddgi.depthTileSize + 2;
    uint px  = probeIdx % ddgi.gridDimsX;
    uint py  = (probeIdx / ddgi.gridDimsX) % ddgi.gridDimsY;
    uint pz  = probeIdx / (ddgi.gridDimsX * ddgi.gridDimsY);
    return uint2((px + pz * ddgi.gridDimsX) * pad + 1 + tx,
                  py * pad + 1 + ty);
}

// ── Analytic ray–primitive intersections ──────────────────────────────────────
//
// Rays are supplied in the primitive's LOCAL space (transformed via invModelMatrix).
// The returned t is the world-space hit distance because the local direction
// vector |ld| = |invModel * rayDir| and the model's scaling cancels out when
// computing the world-space displacement (model * ld * t = rayDir * t).

// Ray vs unit box ([-0.5, 0.5]³). Returns t > 0 on hit; writes outward normal.
static float rayUnitBox(float3 o, float3 d, thread float3& outNormal) {
    float3 tMin = (-0.5f - o) / d;
    float3 tMax = ( 0.5f - o) / d;
    float3 t1   = min(tMin, tMax);
    float3 t2   = max(tMin, tMax);
    float  tNear = max(max(t1.x, t1.y), t1.z);
    float  tFar  = min(min(t2.x, t2.y), t2.z);
    if (tNear > tFar || tFar < 1e-4f) return -1.0f;
    float  t   = (tNear > 1e-4f) ? tNear : tFar;
    float3 hit = o + d * t;
    float3 ab  = abs(hit) * 2.0f;
    if (ab.x >= ab.y && ab.x >= ab.z)
        outNormal = float3(sign(hit.x), 0.0f, 0.0f);
    else if (ab.y >= ab.x && ab.y >= ab.z)
        outNormal = float3(0.0f, sign(hit.y), 0.0f);
    else
        outNormal = float3(0.0f, 0.0f, sign(hit.z));
    return t;
}

// Ray vs unit sphere (radius 0.5, centred at origin).
static float rayUnitSphere(float3 o, float3 d, thread float3& outNormal) {
    float a    = dot(d, d);
    float b    = 2.0f * dot(o, d);
    float c    = dot(o, o) - 0.25f;   // r² = 0.5² = 0.25
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return -1.0f;
    float sq = sqrt(disc);
    float t0 = (-b - sq) / (2.0f * a);
    float t1 = (-b + sq) / (2.0f * a);
    float t  = (t0 > 1e-4f) ? t0 : t1;
    if (t < 1e-4f) return -1.0f;
    outNormal = normalize(o + d * t);
    return t;
}

// Ray vs unit ground plane (y = 0, local x,z ∈ [-0.5, 0.5]).
// The ground mesh uses scale(14,1,14), so local space is the unit quad.
static float rayUnitGround(float3 o, float3 d, thread float3& outNormal) {
    if (abs(d.y) < 1e-6f) return -1.0f;
    float t = -o.y / d.y;
    if (t < 1e-4f) return -1.0f;
    float3 hit = o + d * t;
    if (abs(hit.x) > 0.5f || abs(hit.z) > 0.5f) return -1.0f;
    outNormal = float3(0.0f, 1.0f, 0.0f);
    return t;
}

// ── Phase 3.1 / 3.3: DDGI Trace kernel ───────────────────────────────────────
//
// Dispatch: (raysPerProbe, probeCount, 1). Each thread fires one analytic ray
// from one probe and writes a DDGIRayRecord to outRays[probeIdx*raysPerProbe+rayIdx].
//
// Rays miss → sky HDR sample stored.
// Rays hit  → Lambertian direct light + emission at the hit surface, *plus*
//             (when `ddgi.twoBounceEnabled != 0`) one bounce of indirect
//             light read from the previous-frame irradiance atlas. The hit
//             point's hemisphere-integrated incoming irradiance × albedo / π
//             is the standard Lambertian re-emission; the DDGI atlas already
//             stores `irradiance / π` (the update kernel divides by Σ
//             weights, and the lighting kernel multiplies straight onto
//             albedo without further dividing by π), so we multiply by
//             albedo directly.

kernel void illumi_ddgi_trace(
    device DDGIRayRecord*               outRays      [[buffer(0)]],
    constant DDGIUniforms&              ddgi         [[buffer(1)]],
    const device DDGIInstanceData*      instances    [[buffer(2)]],
    const device DDGIPointEmitter*      emitters     [[buffer(3)]],
    texture2d<float, access::sample>    skyEquirect  [[texture(0)]],
    // Phase 3.3 — previous-frame atlases for the second bounce. The host
    // ping-pongs irradiance + depth atlases, so the atlas slots bound here
    // hold the previous frame's writes (read-only this frame). When
    // `twoBounceEnabled == 0` they're still bound but never sampled.
    texture2d<half,  access::sample>    irrAtlasPrev [[texture(1)]],
    texture2d<half,  access::sample>    depAtlasPrev [[texture(2)]],
    uint2                               gid          [[thread_position_in_grid]]
) {
    uint rayIdx   = gid.x;
    uint probeIdx = gid.y;
    uint probeCount = ddgi.gridDimsX * ddgi.gridDimsY * ddgi.gridDimsZ;
    if (rayIdx >= ddgi.raysPerProbe || probeIdx >= probeCount) return;

    float3 probePos = ddgiProbePos(probeIdx, ddgi);

    // Rotate fibonacci sample by a per-probe golden-angle offset to break up
    // the repeating lattice pattern across adjacent probes.
    float3 fibDir = ddgiSphericalFibonacci(rayIdx, ddgi.raysPerProbe);
    float  angle  = float(probeIdx) * 2.399963229f;  // ~golden angle in rad
    float  sa = sin(angle), ca = cos(angle);
    float3 rayDir = float3(fibDir.x * ca - fibDir.y * sa,
                           fibDir.x * sa + fibDir.y * ca,
                           fibDir.z);

    float  bestDist   = 1.0e10f;
    float3 bestNormal = float3(0.0f, 1.0f, 0.0f);
    uint   hitInst    = 0xFFFFFFFFu;

    for (uint i = 0; i < ddgi.instanceCount; ++i) {
        DDGIInstanceData inst = instances[i];
        float3 lo = (inst.invModelMatrix * float4(probePos, 1.0f)).xyz;
        float3 ld = (inst.invModelMatrix * float4(rayDir,   0.0f)).xyz;
        float3 localNormal = float3(0.0f);
        float  t = -1.0f;
        // Phase 2.6 — meshKind=3 marks a host-extracted custom mesh with
        // no analytic intersection available. DDGI silently treats it as
        // "no geometry here" until Phase 4 brings a real BVH / `MTLAccelerationStructure`
        // for trace-against-arbitrary-meshes. Direct lighting, IBL, and SSR all
        // still operate on the mesh normally — only DDGI's second bounce skips
        // it. The else branch is now guarded so the unit-ground intersection
        // doesn't fire for unknown meshKinds.
        if (inst.meshKind == 0u) {
            t = rayUnitBox(lo, ld, localNormal);
        } else if (inst.meshKind == 1u) {
            t = rayUnitSphere(lo, ld, localNormal);
        } else if (inst.meshKind == 2u) {
            t = rayUnitGround(lo, ld, localNormal);
        }
        // else: meshKind ≥ 3 — leave t = -1 so the (t > 1e-4f) gate skips.
        if (t > 1e-4f && t < bestDist) {
            bestDist   = t;
            bestNormal = normalize((inst.normalMatrix * float4(localNormal, 0.0f)).xyz);
            hitInst    = i;
        }
    }

    uint outIdx = probeIdx * ddgi.raysPerProbe + rayIdx;
    if (hitInst == 0xFFFFFFFFu) {
        float3 sky = sampleSkyEquirect(skyEquirect, rayDir);

        // Emitters visible along this miss ray: treat each as a small sphere.
        // Rays that pass within the emitter's radius pick up its radiance,
        // so probes looking toward a glowing particle field accumulate it.
        for (uint e = 0; e < ddgi.emitterCount; ++e) {
            float3 toE   = emitters[e].position - probePos;
            float  tE    = dot(toE, rayDir);
            if (tE < 0.0f) continue;
            float3 perp  = toE - rayDir * tE;
            float  perpD = length(perp);
            float  r     = max(emitters[e].radius, 0.001f);
            float  atten = max(0.0f, 1.0f - perpD / r);
            atten       *= atten;
            sky          += emitters[e].color * atten;
        }

        outRays[outIdx].dirAndDist = float4(rayDir, -1.0f);
        outRays[outIdx].radiance   = float4(sky, 0.0f);
    } else {
        DDGIInstanceData inst = instances[hitInst];
        float NdotL   = max(0.0f, dot(bestNormal, ddgi.directionalLightDir));
        float3 directLight = inst.albedo * ddgi.directionalLightColor * NdotL
                           * (1.0f - inst.metallic);

        // Phase 3.3 — second bounce. Sample the previous-frame irradiance
        // atlas at the hit point along the hit normal, then re-emit as
        // Lambertian indirect: L_out = albedo · irradiance · (1 - metallic).
        // The atlas read uses the existing trilinear+Chebyshev sampler so
        // probes behind the wall don't bleed onto an exterior hit.
        // Bias the worldPos a hair off the surface so depth-atlas
        // visibility tests don't classify the hit point as "inside" its
        // own surface.
        float3 indirectBounce = float3(0.0f);
        if (ddgi.twoBounceEnabled != 0u) {
            float3 hitWorld = probePos + rayDir * bestDist;
            float3 hitBiased = hitWorld + bestNormal * 0.02f;
            float3 prevIrr = sampleDDGIIrradiance(
                hitBiased, bestNormal, irrAtlasPrev, depAtlasPrev, ddgi);
            indirectBounce = inst.albedo * prevIrr * (1.0f - inst.metallic);
        }

        // Emitter point-light contribution at the hit surface. Each emitter
        // acts as a point light: Lambertian re-emission proportional to NdotL
        // toward the emitter and quadratic attenuation within its radius.
        float3 hitWorld = probePos + rayDir * bestDist;
        float3 emitterLight = float3(0.0f);
        for (uint e = 0; e < ddgi.emitterCount; ++e) {
            float3 toE = emitters[e].position - hitWorld;
            float  d   = length(toE);
            float  r   = max(emitters[e].radius, 0.001f);
            float  atten = max(0.0f, 1.0f - d / r);
            atten       *= atten;
            if (atten < 0.001f) continue;
            float NdotE = max(0.0f, dot(bestNormal, toE / max(d, 0.001f)));
            emitterLight += emitters[e].color * atten * NdotE
                          * inst.albedo * (1.0f - inst.metallic);
        }

        outRays[outIdx].dirAndDist = float4(rayDir, bestDist);
        outRays[outIdx].radiance   = float4(
            directLight + indirectBounce + inst.emission + emitterLight, 1.0f);
    }
}

// ── Phase 3.1: DDGI Update irradiance atlas ───────────────────────────────────
//
// Dispatch: (irrTileSize, irrTileSize, probeCount). For each interior texel of
// each probe's octahedral tile, integrates incoming ray radiance weighted by
// cos(texelDir, rayDir), then exponential-moving-averages into the atlas.

kernel void illumi_ddgi_update_irradiance(
    const device DDGIRayRecord*             rays     [[buffer(0)]],
    constant DDGIUniforms&                  ddgi     [[buffer(1)]],
    texture2d<half, access::read_write>     irrAtlas [[texture(0)]],
    uint3                                   gid      [[thread_position_in_grid]]
) {
    uint tileX    = gid.x;
    uint tileY    = gid.y;
    uint probeIdx = gid.z;
    uint probeCount = ddgi.gridDimsX * ddgi.gridDimsY * ddgi.gridDimsZ;
    if (tileX >= ddgi.irrTileSize || tileY >= ddgi.irrTileSize ||
        probeIdx >= probeCount) return;

    float2 octNorm  = (float2(tileX, tileY) + 0.5f) / float(ddgi.irrTileSize) * 2.0f - 1.0f;
    float3 texelDir = ddgiOctDecode(octNorm);

    float3 irradiance  = float3(0.0f);
    float  totalWeight = 0.0f;
    uint   base = probeIdx * ddgi.raysPerProbe;
    for (uint r = 0; r < ddgi.raysPerProbe; ++r) {
        DDGIRayRecord rec = rays[base + r];
        float3 rayDir = rec.dirAndDist.xyz;
        float  w = max(0.0f, dot(texelDir, rayDir));
        if (w > 0.0f) {
            irradiance  += float3(rec.radiance.xyz) * w;
            totalWeight += w;
        }
    }
    if (totalWeight > 1e-6f) irradiance /= totalWeight;

    uint2  coord    = ddgiIrrTexel(tileX, tileY, probeIdx, ddgi);
    float3 existing = float3(irrAtlas.read(coord).rgb);
    float3 blended  = mix(irradiance, existing, ddgi.hysteresis);
    irrAtlas.write(half4(half3(blended), 1.0h), coord);
}

// ── Phase 3.1: DDGI Update depth atlas ───────────────────────────────────────
//
// Dispatch: (depthTileSize, depthTileSize, probeCount). Stores (mean, mean²) of
// weighted hit distances per octahedral texel. The lighting kernel's Chebyshev
// visibility test reads both channels to bound the probability that a shaded
// point can be "seen" from the probe without obstruction.

kernel void illumi_ddgi_update_depth(
    const device DDGIRayRecord*             rays       [[buffer(0)]],
    constant DDGIUniforms&                  ddgi       [[buffer(1)]],
    texture2d<half, access::read_write>     depthAtlas [[texture(0)]],
    uint3                                   gid        [[thread_position_in_grid]]
) {
    uint tileX    = gid.x;
    uint tileY    = gid.y;
    uint probeIdx = gid.z;
    uint probeCount = ddgi.gridDimsX * ddgi.gridDimsY * ddgi.gridDimsZ;
    if (tileX >= ddgi.depthTileSize || tileY >= ddgi.depthTileSize ||
        probeIdx >= probeCount) return;

    float2 octNorm  = (float2(tileX, tileY) + 0.5f) / float(ddgi.depthTileSize) * 2.0f - 1.0f;
    float3 texelDir = ddgiOctDecode(octNorm);

    float sumD  = 0.0f;
    float sumD2 = 0.0f;
    float totalWeight = 0.0f;
    uint  base = probeIdx * ddgi.raysPerProbe;
    for (uint r = 0; r < ddgi.raysPerProbe; ++r) {
        DDGIRayRecord rec  = rays[base + r];
        float  dist = rec.dirAndDist.w;
        if (dist < 0.0f) continue;  // sky miss
        float3 rayDir = rec.dirAndDist.xyz;
        float  w = max(0.0f, dot(texelDir, rayDir));
        if (w > 0.0f) {
            sumD  += dist * w;
            sumD2 += dist * dist * w;
            totalWeight += w;
        }
    }
    float mean  = (totalWeight > 1e-6f) ? sumD  / totalWeight : 1.0e4f;
    float mean2 = (totalWeight > 1e-6f) ? sumD2 / totalWeight : 1.0e8f;

    uint2  coord    = ddgiDepthTexel(tileX, tileY, probeIdx, ddgi);
    float2 existing = float2(depthAtlas.read(coord).rg);
    float2 blended  = mix(float2(mean, mean2), existing, ddgi.depthHysteresis);
    depthAtlas.write(half4(half2(blended), 0.0h, 1.0h), coord);
}
