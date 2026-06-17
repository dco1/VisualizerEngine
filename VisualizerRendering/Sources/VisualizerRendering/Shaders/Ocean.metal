// ── OCEAN ────────────────────────────────────────────────────────────────────
//
// GPU compute kernel that drives a Gerstner-wave heightfield ocean surface.
// One thread per vertex of a square subdivided plane (`resolution²` verts);
// each thread sums N traveling Gerstner waves to produce the world-space
// position and analytic normal of its vertex, writing both into the same
// shared MTLBuffer SceneKit reads from via its vertex / normal sources.
//
// The math is the standard Gerstner / "trochoid" form (see Mark Finch,
// "Effective Water Simulation from Physical Models", GPU Gems 1, Ch. 1):
//
//   For each wave i with direction Dᵢ (unit, xz plane), amplitude Aᵢ,
//   wavelength Lᵢ, speed Sᵢ, steepness Qᵢ ∈ [0..1], and given grid point
//   (x₀, z₀) at time t:
//
//     k    = 2π / Lᵢ
//     ω    = Sᵢ · k
//     φ    = Dᵢ · (x₀, z₀) · k − ω · t
//     pos += ( Qᵢ·Aᵢ·Dᵢ.x·cos(φ),  Aᵢ·sin(φ),  Qᵢ·Aᵢ·Dᵢ.z·cos(φ) )
//
//   Tangent and binormal accumulate the partial derivatives of the
//   displacement w.r.t. x₀ and z₀ respectively; the unit normal is
//   `normalize(cross(binormal, tangent))`. Computing the normal analytically
//   instead of finite-differencing across the grid keeps the lighting crisp
//   even at lower resolutions.
//
// Layout note: every shared field is float4 per the project ALIGNMENT RULE
// (see PBDSolver.swift). The 32-byte OceanVertex mirrors the Swift struct
// in OceanSurface.swift exactly.

#include <metal_stdlib>
using namespace metal;

struct OceanVertex {
    float4 position;       // xyz = world position (m), w = 1
    float4 normal;         // xyz = unit normal,        w = 0
    float4 emissionColor;  // rgb = additive emission (burst-light specular),
                           // a = strongest sample (debug / future use)
};

struct OceanWave {
    float4 dirAmpWavelen;   // x,y = direction.xz (unit), z = amplitude (m), w = wavelength (m)
    float4 speedSteepPad;   // x = phase speed (m/s),     y = steepness Q (0..1), zw = pad
};

struct OceanUniforms {
    float4 originSizeRes;   // x,y = surface centre (x, z) (m),
                            // z   = world size (square edge length, m),
                            // w   = grid resolution (uint bits) — verts per side
    float4 timeCount;       // x = wall time (s),
                            // y = waveCount (uint bits),
                            // z = global amplitude scale (0 = glass calm),
                            // w = burstCount (uint bits)
    float4 cameraPos;       // xyz = camera world position (m), w = pad
};

struct BurstLight {
    float4 positionIntensity;
    float4 colorAge;
    float4 radiusLifePad;
};

// One thread per grid vertex. id mapped row-major: ix = id % res, iz = id / res.
kernel void oceanBuild(
    device       OceanVertex*    verts  [[ buffer(0) ]],
    constant     OceanWave*      waves  [[ buffer(1) ]],
    constant     OceanUniforms&  u      [[ buffer(2) ]],
    device const BurstLight*     bursts [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint res = as_type<uint>(u.originSizeRes.w);
    uint total = res * res;
    if (id >= total) return;

    uint waveCount = as_type<uint>(u.timeCount.y);

    uint ix = id % res;
    uint iz = id / res;
    float fx = float(ix) / float(res - 1u);   // 0..1
    float fz = float(iz) / float(res - 1u);   // 0..1

    float worldSize = u.originSizeRes.z;
    float ox = u.originSizeRes.x;
    float oz = u.originSizeRes.y;
    float x0 = ox + (fx - 0.5f) * worldSize;  // rest-pose grid xz, used as both
    float z0 = oz + (fz - 0.5f) * worldSize;  // the sample point AND the start
                                              // position of the (Gerstner-
                                              // displaced) vertex.

    float t        = u.timeCount.x;
    float ampScale = u.timeCount.z;

    float3 pos = float3(x0, 0.0f, z0);

    // Tangent / binormal start as the un-displaced grid basis: a unit step in
    // x maps to (1,0,0), a unit step in z to (0,0,1). Each wave adds the
    // partial derivative of its displacement w.r.t. (x₀, z₀) — sum across
    // waves, then cross-product for the normal.
    float3 tan_x = float3(1.0f, 0.0f, 0.0f);
    float3 tan_z = float3(0.0f, 0.0f, 1.0f);

    for (uint i = 0u; i < waveCount; ++i) {
        OceanWave w = waves[i];
        float2 dir = w.dirAmpWavelen.xy;
        float  A   = w.dirAmpWavelen.z * ampScale;
        float  L   = max(0.001f, w.dirAmpWavelen.w);
        float  S   = w.speedSteepPad.x;
        float  Q   = w.speedSteepPad.y;

        float k     = 6.28318530718f / L;
        float omega = S * k;
        float phase = (dir.x * x0 + dir.y * z0) * k - omega * t;
        float c = cos(phase);
        float s = sin(phase);

        // Displacement contribution.
        pos.x += Q * A * dir.x * c;
        pos.y += A * s;
        pos.z += Q * A * dir.y * c;

        // d(pos)/d(x₀) — derivative of (Q·A·Dx·cos(φ), A·sin(φ), Q·A·Dz·cos(φ))
        // with respect to x₀, where ∂φ/∂x₀ = Dx · k.
        float WA = A * k;
        tan_x.x += -Q * dir.x * dir.x * WA * s;
        tan_x.y +=      dir.x          * WA * c;
        tan_x.z += -Q * dir.x * dir.y * WA * s;

        // d(pos)/d(z₀), with ∂φ/∂z₀ = Dz · k.
        tan_z.x += -Q * dir.y * dir.x * WA * s;
        tan_z.y +=      dir.y          * WA * c;
        tan_z.z += -Q * dir.y * dir.y * WA * s;
    }

    // cross(tan_z, tan_x) gives the upward (+y) normal for the right-handed
    // grid basis we built. Verified by plugging in a single Z-aligned wave at
    // peak (phase=π/2): tan_x = (1,0,0), tan_z = (0,k·A,1), and the cross is
    // (0, +1, −k·A), normalised — correct outward normal for a forward-tilted
    // wave crest.
    float3 nrm = normalize(cross(tan_z, tan_x));

    // ── BURST-LIT SPECULAR ────────────────────────────────────────────────
    // Per-vertex Blinn-Phong specular response to every active burst. Each
    // burst is a point light above the water; this vertex's wave normal
    // determines whether the burst's light specular-reflects toward the
    // camera. Sharp shininess (~180) gives the speckled-on-water look real
    // fireworks have — bright glints scattered across the wave field, NOT
    // a flat coloured wash.
    uint burstCount = as_type<uint>(u.timeCount.w);
    float3 emission = float3(0.0f);
    if (burstCount > 0u) {
        float3 toCam = u.cameraPos.xyz - pos;
        float camDist = length(toCam);
        float3 viewDir = camDist > 1e-4f ? toCam / camDist : float3(0.0f, 1.0f, 0.0f);

        for (uint i = 0u; i < burstCount; ++i) {
            BurstLight L = bursts[i];
            float intensity = L.positionIntensity.w;
            float maxR      = L.radiusLifePad.x;
            float life      = L.radiusLifePad.y;
            if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

            float age = L.colorAge.a;
            if (age >= life) continue;

            float3 toLight = L.positionIntensity.xyz - pos;
            float lightDist = length(toLight);
            // Reach: a burst N m above the water lights a roughly equal
            // radius of surface beneath it. With default maxR=29 m, 2.2×
            // gives ~64 m of glints — enough to read but not so wide it
            // turns into a wash.
            float reach = maxR * 2.2f;
            if (lightDist >= reach) continue;
            float3 lightDir = toLight / max(lightDist, 1e-4f);

            // Quadratic radial falloff inside reach.
            float dT = 1.0f - lightDist / reach;
            float distFall = dT * dT;

            // Age fade: peak at age 0, zero at lifespan.
            float ageT = age / life;
            float ageScale;
            if (ageT < 0.08f) {
                ageScale = 1.0f;
            } else {
                float k = (ageT - 0.08f) / 0.92f;
                ageScale = (1.0f - k);
                ageScale = ageScale * ageScale;
            }

            // Blinn-Phong specular only — pure glint, no diffuse underwash
            // (the diffuse term reads as paint, the "oil spill" complaint).
            //
            // Shininess MUST stay low enough that adjacent grid vertices on
            // the wave mesh catch the highlight consistently — too sharp a
            // shininess (~360+) with per-vertex evaluation makes the spec
            // term fire at single vertices and SceneKit interpolates
            // linearly across triangles, producing visible triangulation
            // ("matrix glitching"). 70 is broad enough that the highlight
            // smoothly spans multiple verts, narrow enough that the
            // response still reads as glints on water rather than a wash.
            float3 H = normalize(lightDir + viewDir);
            float NdotH = max(0.0f, dot(nrm, H));
            float spec = pow(NdotH, 70.0f);

            float surface = spec * distFall * ageScale;
            // Lower per-sample contribution to compensate for the broader
            // shininess lobe lighting more verts at once.
            emission += L.colorAge.rgb * (intensity * surface * 0.42f);
        }
        // Soft tone-map so multiple overlapping spec hits don't clip to
        // pure white — preserves coloured glints when bursts stack.
        emission = emission / (1.0f + emission * 0.5f);
    }

    verts[id].position      = float4(pos, 1.0f);
    verts[id].normal        = float4(nrm, 0.0f);
    verts[id].emissionColor = float4(emission, length(emission));
}
