// ── BARK WIND ────────────────────────────────────────────────────────────────
//
// GPU vertex animation for the Forest scene's merged bark mesh.
//
// Each tree's bark is one merged SCNGeometry built CPU-side from the tree's
// skeleton (see BarkRenderer.swift). The mesh's position + normal buffers are
// MTLBuffers shared with SceneKit; this kernel writes per-vertex animated
// positions + normals into them each frame.
//
// Today's animation is the SAME effective rotation the tree-wrapper SCNNode
// used to apply CPU-side: a uniform (rigid) quaternion rotation of every
// vertex around the tree's local origin. The wrapper's wind euler is zeroed
// when `useGPUBark` is on, and the leaves (LeafField) continue to use the
// matching per-tree TreeTransform quaternion. Bark + leaves stay in lock-step.
//
// Future passes (R3+) will add per-vertex `heightWeight` / `branchPhase` /
// `branchAnchor` attributes for hierarchical wind — per-branch sway with
// different phases, propagated up the parent chain. The kernel signature
// will grow; the dispatch shape (one thread per vertex) and the rest-pose
// buffer layout will stay.

#include <metal_stdlib>
using namespace metal;

// ── Uniforms (must mirror BarkRenderer.swift exactly) ───────────────────────
//
// Per the ALIGNMENT RULE in PBDSolver.swift: float4 slots only (16 B aligned)
// for fields shared with Swift. Trailing scalars (vertexCount) are packed
// into the tail of a slot via bitcast.
struct BarkWindUniforms {
    float4 quat;          // xyz = quaternion axis, w = quaternion w
    uint   vertexCount;   // 4
    uint   pad0;          // 4
    uint   pad1;          // 4
    uint   pad2;          // 4   → total 32 bytes
};

// ── Quaternion rotation helper (Rodrigues form) ─────────────────────────────
inline float3 rotateByQuat(float3 v, float4 q) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// ── Bark wind kernel ────────────────────────────────────────────────────────
//
// One thread per vertex. Reads the static rest pose + rest normal from one
// pair of buffers, writes the rotated position + normal into another. The
// "current" buffers are what SceneKit reads as the geometry's vertex source.
// packed_float3 (12 B/vertex) — Swift writes the rest pose at packed
// stride; SCN reads vertex/normal sources at dataStride 12. See
// BarkRenderer.swift for why this exact triple of layouts is the only
// one SceneKit accepts here.
kernel void barkWind(
    device const packed_float3*  restPos   [[ buffer(0) ]],
    device const packed_float3*  restNorm  [[ buffer(1) ]],
    device       packed_float3*  outPos    [[ buffer(2) ]],
    device       packed_float3*  outNorm   [[ buffer(3) ]],
    constant     BarkWindUniforms& u       [[ buffer(4) ]],
    uint vid [[ thread_position_in_grid ]]
) {
    if (vid >= u.vertexCount) return;

    float3 rp = float3(restPos[vid]);
    float3 rn = float3(restNorm[vid]);

    float3 outP = rotateByQuat(rp, u.quat);
    float3 outN = rotateByQuat(rn, u.quat);

    outPos[vid]  = packed_float3(outP);
    outNorm[vid] = packed_float3(outN);
}
