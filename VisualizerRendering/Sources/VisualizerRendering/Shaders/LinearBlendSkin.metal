#include <metal_stdlib>
using namespace metal;

// ── LinearBlendSkin.metal ────────────────────────────────────────────────────
//
// Generic linear-blend skinning: every vertex is a weighted sum of up to four
// bone transforms applied to its rest position. Writes packed_float3 position +
// normal streams in the layout `IlluminatoramaGPUMeshDescriptor` consumes, so a
// skinned mesh goes GPU → Illuminatorama repack → G-buffer / BLAS refit with no
// CPU round trip.
//
// Normals are transformed by the COFACTOR of the blended 3×3 (= det · M⁻ᵀ), which
// is exact for non-uniform scale (a stretched tooth, a squashed lip) where
// transforming the normal by M itself would tilt it off the surface. The det
// factor drops out in the normalize; bones must not mirror (det > 0).
//
// Swift mirror: `LinearBlendSkinMesh` (LinearBlendSkinMesh.swift).

kernel void lbs_skin(
    const device packed_float3* restPos     [[buffer(0)]],
    const device packed_float3* restNrm     [[buffer(1)]],
    const device ushort4*       boneIndex   [[buffer(2)]],
    const device float4*        boneWeight  [[buffer(3)]],
    const device float4x4*      bones       [[buffer(4)]],
    constant uint&              count       [[buffer(5)]],
    device packed_float3*       outPos      [[buffer(6)]],
    device packed_float3*       outNrm      [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ushort4 bi = boneIndex[gid];
    float4  bw = boneWeight[gid];
    float4x4 m = bones[bi.x] * bw.x;
    if (bw.y != 0.0f) m += bones[bi.y] * bw.y;
    if (bw.z != 0.0f) m += bones[bi.z] * bw.z;
    if (bw.w != 0.0f) m += bones[bi.w] * bw.w;

    float3 p = float3(restPos[gid]);
    float3 n = float3(restNrm[gid]);
    float4 wp = m * float4(p, 1.0f);

    float3 c0 = m[0].xyz, c1 = m[1].xyz, c2 = m[2].xyz;
    float3 wn = n.x * cross(c1, c2) + n.y * cross(c2, c0) + n.z * cross(c0, c1);
    float len2 = dot(wn, wn);
    wn = len2 > 1e-20f ? wn * rsqrt(len2) : n;

    outPos[gid] = packed_float3(wp.xyz);
    outNrm[gid] = packed_float3(wn);
}
