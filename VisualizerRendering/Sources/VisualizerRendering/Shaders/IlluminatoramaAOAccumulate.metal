#include <metal_stdlib>
#include <metal_raytracing>
#include "IlluminatoramaSampling.h"
using namespace metal;
using namespace raytracing;

// ── ILLUMINATORAMA — ACCUMULATED ray-traced AO (AO v2, Daydream DH-0887) ─────
//
// The still / settled-canvas AO pipeline, host opt-in (`aoAccumulationEnabled`), which leaves
// the per-frame path (`illumi_rtao_tlas` → `illumi_ssao_spatial` → lighting) byte-identical
// for every host that does not ask for it.
//
// Why a separate pipeline and not a better per-frame RTAO. The per-frame path's only
// accumulator was the lit-colour TAA history: every frame's RTAO was blurred by a 3×3 bilateral
// BEFORE anything averaged it, its per-pixel rotation was white noise re-drawn each frame, and
// the colour history clipped the average back toward each noisy frame's local mean. Blur-then-
// average of white noise leaves low-frequency blobs (the "sponge"); a fresh random draw each
// frame makes an N-frame average converge only as 1/√N; and the clip caps it. The fix is the
// standard progressive-still order:
//
//   illumi_rtao_tlas_v2   raw OCCLUSION: progressive Owen-scrambled Sobol rays from the geometric
//                         surface, each hit weighted 1 − (t/R)² by its distance
//   illumi_ao_accumulate  unbiased running mean weighted by rays, NO clamp (frozen camera), fp32
//   illumi_ao_denoise     XeGTAO depth-edge 3×3 (strength eases from 1 to ½ as the mean fills),
//                         then the grade's strength: ao = 1 − occlusion · intensity
//
// and the lighting reads the denoised accumulation through the renderer's one AO selector.

// ── RTAO v2 ───────────────────────────────────────────────────────────────────

struct RTAOv2Uniforms {
    float4x4 invViewProjection;   // depth → world (this frame's, jittered — the depth's own)
    float4   cameraWorldPos;      // .xyz
    float    radius;              // occlusion reach, world metres
    float    intensity;           // unused by the trace (occlusion is accumulated; see denoise)
    float    rayTMin;             // origin offset floor along the geometric normal, world metres
    uint     rayCount;            // rays per half-res texel this frame
    uint     sampleBase;          // progressive index of this frame's first ray (Σ rays so far)
    uint     scramble;            // stream offset (0 in production; test twins differ)
    uint     transportRayMask;    // 0x01 opaque | 0x04 invisible occluder (glass excluded)
    uint     fullWidth;           // full-res depth dims (AO is half-res)
    uint     fullHeight;
    float    surfaceSnapULPs;     // DH-0715 traced-origin snap reach (0 ⇒ off; was _reserved0)
    uint     _reserved1;
    uint     _reserved2;
};
static_assert(sizeof(RTAOv2Uniforms) == 128, "RTAOv2Uniforms layout must match the Swift mirror");

static inline float3 aoWorldPos(depth2d<float, access::read> gDepth, int2 p, int2 mx,
                                float2 fdim, float4x4 invVP) {
    p = clamp(p, int2(0), mx);
    float z = gDepth.read(uint2(p));
    float2 ndc = (float2(p) + 0.5) / fdim * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = invVP * float4(ndc, z, 1.0);
    return w.xyz / w.w;
}

kernel void illumi_rtao_tlas_v2(
    depth2d<float, access::read>      gDepth   [[texture(0)]],   // full-res
    texture2d<half,  access::write>   outAO    [[texture(1)]],   // half-res raw OCCLUSION (0 = open)
    instance_acceleration_structure   accel    [[buffer(0)]],
    constant RTAOv2Uniforms&          u        [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint outW = outAO.get_width(), outH = outAO.get_height();
    if (gid.x >= outW || gid.y >= outH) return;
    if (u.intensity <= 0.0) { outAO.write(half4(0.0h), gid); return; }

    uint2 fullGid = min(gid * 2, uint2(u.fullWidth - 1, u.fullHeight - 1));
    float depth = gDepth.read(fullGid);
    if (depth >= 0.99999) { outAO.write(half4(0.0h), gid); return; }   // sky

    // The GEOMETRIC surface, reconstructed from depth — the same min-|Δ| pick per axis the GTAO
    // pass uses (IlluminatoramaSSAO.metal), so a silhouette neighbour is not blended in. The
    // bump-mapped G-buffer normal is deliberately not read: a hemisphere tilted by the material's
    // relief sends rays below the real plane into the surface itself, which prints occlusion
    // correlated with the texture (DH-0527 for GTAO, the same defect here).
    float2 fdim = float2(u.fullWidth, u.fullHeight);
    int2 p = int2(fullGid), mx = int2(int(u.fullWidth) - 1, int(u.fullHeight) - 1);
    float3 P  = aoWorldPos(gDepth, p, mx, fdim, u.invViewProjection);
    float3 Pr = aoWorldPos(gDepth, p + int2( 1, 0), mx, fdim, u.invViewProjection);
    float3 Pl = aoWorldPos(gDepth, p + int2(-1, 0), mx, fdim, u.invViewProjection);
    float3 Pd = aoWorldPos(gDepth, p + int2( 0, 1), mx, fdim, u.invViewProjection);
    float3 Pu = aoWorldPos(gDepth, p + int2( 0,-1), mx, fdim, u.invViewProjection);
    // One-sided differences; at the frame edge the clamped neighbour IS this texel, so only the
    // inward side is a real difference (a zero one would zero the normal — DH-0887 review).
    bool hasL = p.x > 0, hasR = p.x < mx.x, hasU = p.y > 0, hasD = p.y < mx.y;
    float3 dx = (hasR && (!hasL || length_squared(Pr - P) < length_squared(P - Pl))) ? (Pr - P) : (P - Pl);
    float3 dy = (hasD && (!hasU || length_squared(Pd - P) < length_squared(P - Pu))) ? (Pd - P) : (P - Pu);
    float3 N = cross(dx, dy);
    float nl = length(N);
    if (nl < 1e-12) { outAO.write(half4(0.0h), gid); return; }
    N /= nl;
    float3 toCam = u.cameraWorldPos.xyz - P;
    if (dot(N, toCam) < 0.0) N = -N;                       // face the camera

    // The origin sits on the GEOMETRIC surface, offset along its normal by a distance that grows
    // with range (reconstructed positions carry depth-quantisation error that grows with it).
    // `min_distance` is tiny on purpose: the old 4 mm guard ignored the near wall for any texel
    // within 4 mm of a crease — a bright line one to three texels wide down every wall/floor
    // junction. With the origin lifted off the true plane, no upper-hemisphere ray can re-hit it.
    // DH-0715 — onto the surface the camera SEES: a substrate's biased depth buries P (the
    // neighbours shift with it, so N above is still the ground's). See `illumiSnapToVisibleSurface`.
    float2 ndcP = (float2(p) + 0.5) / fdim * 2.0 - 1.0;
    ndcP.y = -ndcP.y;
    P = illumiSnapToVisibleSurface(accel, P, u.cameraWorldPos.xyz, ndcP, depth, u.invViewProjection,
                                   u.surfaceSnapULPs);
    toCam = u.cameraWorldPos.xyz - P;
    float dist = length(toCam);
    float3 origin = P + N * max(u.rayTMin, 2e-4 * dist);

    // CLOSEST hit, so each ray can be weighted by its distance: occlusion contribution
    // 1 − (t/R)², the same quadratic falloff the screen-space GTAO uses (`gtaoFalloff`). Measured
    // on a crease model the explicit weight beats a stochastic ray length by 1.2–3.8× in per-ray
    // variance and converges as ~N⁻¹ instead of ~N⁻²ᐟ³.
    intersector<triangle_data, instancing, curve_data> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.accept_any_intersection(false);

    uint rays = max(1u, u.rayCount);
    float R = max(1e-3, u.radius);
    float invR2 = 1.0 / (R * R);
    uint seed = illumiPixelSeed(gid, u.scramble);
    float occ = 0.0;
    for (uint i = 0; i < rays; ++i) {
        float2 q = illumiSobolOwen2DShuffled(u.sampleBase + i, seed);
        ray r;
        r.origin = origin;
        r.direction = illumiCosineHemisphereAxisSafe(N, q);
        r.min_distance = 1e-4;
        r.max_distance = R;
        auto hit = isect.intersect(r, accel, u.transportRayMask);
        if (hit.type != intersection_type::none) {
            occ += saturate(1.0 - hit.distance * hit.distance * invR2);
        }
    }
    // Raw OCCLUSION (not 1 − occ·intensity): the grade's strength is applied at the read, so
    // moving the AO dial never invalidates the accumulated mean.
    outAO.write(half4(half(occ / float(rays))), gid);
}

// ── Accumulation: unbiased running mean ──────────────────────────────────────

struct AOAccumulateUniforms {
    float weight;       // this frame's share of the mean: rays / (rays already in it + rays); 1 ⇒ restart
    uint  _r0, _r1, _r2;
};

kernel void illumi_ao_accumulate(
    texture2d<half,  access::read>       rawAO [[texture(0)]],
    texture2d<float, access::read_write> acc   [[texture(1)]],   // half-res r32Float
    constant AOAccumulateUniforms&       u     [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= acc.get_width() || gid.y >= acc.get_height()) return;
    float raw = float(rawAO.read(gid).r);
    float mean = raw;
    if (u.weight < 1.0) {
        float prev = acc.read(gid).r;
        mean = prev + (raw - prev) * u.weight;
    }
    acc.write(float4(mean, 0.0, 0.0, 1.0), gid);
}

// ── Denoise: XeGTAO depth edges, strength decaying with the sample count ─────

struct AODenoiseUniforms {
    float4x4 invProjection;   // depth → view space (edges are relative view-z differences)
    float    strength;        // 0 = pass-through, 1 = full 3×3
    uint     fullWidth;
    uint     fullHeight;
    float    intensity;       // the grade's AO strength: ao = 1 − occlusion · intensity
};

static inline float aoViewZ(depth2d<float, access::read> gDepth, int2 halfP, int2 halfMax,
                            float2 fdim, uint2 fmax, float4x4 invProj) {
    halfP = clamp(halfP, int2(0), halfMax);
    uint2 f = min(uint2(halfP) * 2u, fmax);
    float z = gDepth.read(f);
    if (z >= 0.99999) return 1e6;
    float2 ndc = (float2(f) + 0.5) / fdim * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 v = invProj * float4(ndc, z, 1.0);
    return -v.z / v.w;                                     // positive distance along −Z
}

kernel void illumi_ao_denoise(
    texture2d<float, access::read>  acc    [[texture(0)]],   // half-res accumulated OCCLUSION
    depth2d<float,   access::read>  gDepth [[texture(1)]],   // full-res
    texture2d<half,  access::write> outAO  [[texture(2)]],   // half-res, what lighting reads
    constant AODenoiseUniforms&     u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = outAO.get_width(), h = outAO.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float I = saturate(u.intensity);
    float c = acc.read(gid).r;
    if (u.strength <= 0.0) { outAO.write(half4(half(saturate(1.0 - c * I))), gid); return; }

    int2 hp = int2(gid), hmx = int2(int(w) - 1, int(h) - 1);
    float2 fdim = float2(u.fullWidth, u.fullHeight);
    uint2 fmx = uint2(u.fullWidth - 1, u.fullHeight - 1);
    float zC = aoViewZ(gDepth, hp, hmx, fdim, fmx, u.invProjection);
    if (zC >= 1e5) { outAO.write(half4(1.0h), gid); return; }   // sky
    float zL = aoViewZ(gDepth, hp + int2(-1, 0), hmx, fdim, fmx, u.invProjection);
    float zR = aoViewZ(gDepth, hp + int2( 1, 0), hmx, fdim, fmx, u.invProjection);
    float zT = aoViewZ(gDepth, hp + int2( 0,-1), hmx, fdim, fmx, u.invProjection);
    float zB = aoViewZ(gDepth, hp + int2( 0, 1), hmx, fdim, fmx, u.invProjection);

    // XeGTAO edges: a neighbour is "the same surface" when its depth matches the centre's OR
    // the plane through the centre (slope-adjusted), relative to depth — so a receding floor
    // filters along itself and a contact step does not bleed. No normal term: the G-buffer
    // normal is bump-mapped and would make the filter follow the material's relief.
    float dL = zL - zC, dR = zR - zC, dT = zT - zC, dB = zB - zC;
    float sx = 0.5 * (dR - dL), sy = 0.5 * (dB - dT);
    float tol = 0.011 * zC;
    float eL = saturate(1.25 - min(abs(dL), abs(dL + sx)) / tol);
    float eR = saturate(1.25 - min(abs(dR), abs(dR - sx)) / tol);
    float eT = saturate(1.25 - min(abs(dT), abs(dT + sy)) / tol);
    float eB = saturate(1.25 - min(abs(dB), abs(dB - sy)) / tol);

    float aL = acc.read(uint2(clamp(hp + int2(-1, 0), int2(0), hmx))).r;
    float aR = acc.read(uint2(clamp(hp + int2( 1, 0), int2(0), hmx))).r;
    float aT = acc.read(uint2(clamp(hp + int2( 0,-1), int2(0), hmx))).r;
    float aB = acc.read(uint2(clamp(hp + int2( 0, 1), int2(0), hmx))).r;
    float aTL = acc.read(uint2(clamp(hp + int2(-1,-1), int2(0), hmx))).r;
    float aTR = acc.read(uint2(clamp(hp + int2( 1,-1), int2(0), hmx))).r;
    float aBL = acc.read(uint2(clamp(hp + int2(-1, 1), int2(0), hmx))).r;
    float aBR = acc.read(uint2(clamp(hp + int2( 1, 1), int2(0), hmx))).r;
    // Diagonals are reached through two edges (XeGTAO's 0.425 weight).
    float wTL = 0.425 * eL * eT, wTR = 0.425 * eR * eT;
    float wBL = 0.425 * eL * eB, wBR = 0.425 * eR * eB;
    const float beta = 1.2;
    float sum = beta * c + eL * aL + eR * aR + eT * aT + eB * aB
              + wTL * aTL + wTR * aTR + wBL * aBL + wBR * aBR;
    float wsum = beta + eL + eR + eT + eB + wTL + wTR + wBL + wBR;
    float filtered = sum / wsum;
    float o = mix(c, filtered, saturate(u.strength));
    outAO.write(half4(half(saturate(1.0 - o * I))), gid);
}
