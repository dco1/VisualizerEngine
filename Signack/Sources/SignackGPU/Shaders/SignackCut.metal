// ── SIGNACK S2 — GPU CUT SELECTION + FRUSTUM CULLING (Daydream DH-1461) ─────────────────────
//
// One placed instance of a cluster LOD DAG per dispatch triple:
//
//   1. signack_cut_classify   one THREAD per cluster: the crack-free cut test (`isInCut`, exactly
//                             `SignackSelection.isInCut` on the CPU) then the frustum test on the
//                             cluster's padded bounds. flag 0 = not in the cut, 1 = in the cut but
//                             off screen (shadow casters only), 2 = in the cut and on screen.
//                             Off-screen indices are counted.
//   2. signack_cut_write      one THREADGROUP per cluster: thread 0 claims an output range with one
//                             atomic, every thread copies one triangle. Off-screen clusters fill
//                             [0, S), on-screen clusters [S, S + P) — so the SHADOW draw is the
//                             whole LOD cut [0, S + P) and the PRIMARY draw is just [S, S + P).
//   3. signack_cut_finalize   one thread: both indirect argument records, then zero the counters
//                             for the next frame.
//
// Self-contained on purpose (no engine header): Signack is its own package and its shaders build
// into its own library (`SignackGPULibrary`). The structs below are byte-for-byte mirrors of
// `SignackGPUCluster` / `SignackGPUCutUniforms` in Swift — the static_asserts pin the sizes.
//
// Fast-math safe: a root's parent error is the finite sentinel `SIGNACK_ROOT_ERROR` (never +inf,
// which fast-math may fold away), and the distance/division use `precise::` so the GPU decision
// matches the CPU reference bit-for-bit away from float ties. Crack-freedom does not depend on
// that parity: every cluster of a group carries IDENTICAL (sphere, error) inputs, so the GPU makes
// the same call for all of them whatever its rounding.

#include <metal_stdlib>
using namespace metal;

constant float SIGNACK_ROOT_ERROR = 1.0e30f;   // ≥ this ⇒ +∞ (a root: never replaced)

struct SignackGPUCluster {
    float4 bounds;        // culling sphere (centre, radius), object space
    float4 lodSphere;     // where this cluster's own error is measured
    float4 parentSphere;  // where the group that replaces it measures its error
    uint   indexStart;    // first index of the cluster's run in the DAG index buffer
    uint   triangleCount;
    float  lodError;      // object units; 0 for level 0
    float  parentError;   // object units; ≥ SIGNACK_ROOT_ERROR for a root
};
static_assert(sizeof(SignackGPUCluster) == 64, "mirror of Swift SignackGPUCluster");

struct SignackGPUCutUniforms {
    float4 planes[6];       // object-space frustum, inside where dot(xyz, p) + w ≥ 0
    float4 camera;          // xyz = camera in object space, w = pixel scale
    float  threshold;       // acceptable projected error (px)
    float  boundsPad;       // object units added to every culling radius (wind reach)
    uint   isOrthographic;
    uint   clusterCount;
    uint   cull;            // 0 = no frustum test (every cut cluster is "on screen")
    uint   capacity;        // indices the output buffer holds
    uint   _pad0;
    uint   _pad1;
};
static_assert(sizeof(SignackGPUCutUniforms) == 144, "mirror of Swift SignackGPUCutUniforms");

struct SignackDrawArgs {     // MTLDrawIndexedPrimitivesIndirectArguments
    uint indexCount;
    uint instanceCount;
    uint indexStart;
    int  baseVertex;
    uint baseInstance;
};

// counters[0] off-screen indices (classify), [1] off-screen cursor, [2] on-screen cursor.

/// `SignackView.projectedError` — conservative pixels of `err` measured at sphere `s`.
static float signack_projected(float4 s, float err, constant SignackGPUCutUniforms& U) {
    if (err <= 0.0f) return 0.0f;
    if (err >= SIGNACK_ROOT_ERROR) return FLT_MAX;
    if (U.isOrthographic != 0) return err * U.camera.w;
    float3 d3 = U.camera.xyz - s.xyz;
    float d = precise::sqrt(dot(d3, d3)) - s.w;
    return d > 1.0e-4f ? precise::divide(err * U.camera.w, d) : FLT_MAX;
}

kernel void signack_cut_classify(
    constant SignackGPUCutUniforms& U          [[buffer(0)]],
    device const SignackGPUCluster* clusters   [[buffer(1)]],
    device uchar*                   flags      [[buffer(2)]],
    device atomic_uint*             counters   [[buffer(3)]],
    uint i                                     [[thread_position_in_grid]])
{
    if (i >= U.clusterCount) return;
    SignackGPUCluster c = clusters[i];
    bool inCut = signack_projected(c.lodSphere, c.lodError, U) <= U.threshold
              && signack_projected(c.parentSphere, c.parentError, U) > U.threshold;
    uchar f = 0;
    if (inCut) {
        f = 2;
        if (U.cull != 0) {
            float r = c.bounds.w + U.boundsPad;
            for (uint p = 0; p < 6; ++p) {
                float4 pl = U.planes[p];
                if (dot(pl.xyz, c.bounds.xyz) + pl.w < -r) { f = 1; break; }
            }
        }
        if (f == 1) atomic_fetch_add_explicit(&counters[0], c.triangleCount * 3, memory_order_relaxed);
    }
    flags[i] = f;
}

kernel void signack_cut_write(
    constant SignackGPUCutUniforms& U          [[buffer(0)]],
    device const SignackGPUCluster* clusters   [[buffer(1)]],
    device const uchar*             flags      [[buffer(2)]],
    device atomic_uint*             counters   [[buffer(3)]],
    device const uint*              dagIndices [[buffer(4)]],
    device uint*                    outIndices [[buffer(5)]],
    uint cid                                   [[threadgroup_position_in_grid]],
    uint t                                     [[thread_position_in_threadgroup]],
    uint tgSize                                [[threads_per_threadgroup]])
{
    if (cid >= U.clusterCount) return;
    uchar f = flags[cid];
    if (f == 0) return;                          // uniform across the threadgroup
    uint start = clusters[cid].indexStart;
    uint n = clusters[cid].triangleCount * 3;
    threadgroup uint base;
    if (t == 0) {
        base = (f == 1)
            ? atomic_fetch_add_explicit(&counters[1], n, memory_order_relaxed)
            : atomic_load_explicit(&counters[0], memory_order_relaxed)
              + atomic_fetch_add_explicit(&counters[2], n, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b = base;
    for (uint k = t; k < n; k += tgSize) {
        uint dst = b + k;
        if (dst < U.capacity) outIndices[dst] = dagIndices[start + k];
    }
}

kernel void signack_cut_finalize(
    constant SignackGPUCutUniforms& U          [[buffer(0)]],
    device atomic_uint*             counters   [[buffer(3)]],
    device SignackDrawArgs*         args       [[buffer(6)]],   // [0] primary, [2] shadow (32 B apart)
    uint i                                     [[thread_position_in_grid]])
{
    if (i != 0) return;
    uint s = atomic_load_explicit(&counters[0], memory_order_relaxed);
    uint p = atomic_load_explicit(&counters[2], memory_order_relaxed);
    uint total = min(s + p, U.capacity);
    uint primary = total > s ? total - s : 0;
    SignackDrawArgs a;
    a.instanceCount = 1; a.baseVertex = 0; a.baseInstance = 0;
    a.indexCount = primary; a.indexStart = min(s, U.capacity);
    device SignackDrawArgs* primaryArgs = (device SignackDrawArgs*)((device uchar*)args + 0);
    device SignackDrawArgs* shadowArgs  = (device SignackDrawArgs*)((device uchar*)args + 32);
    *primaryArgs = a;
    a.indexCount = total; a.indexStart = 0;
    *shadowArgs = a;
    atomic_store_explicit(&counters[0], 0u, memory_order_relaxed);
    atomic_store_explicit(&counters[1], 0u, memory_order_relaxed);
    atomic_store_explicit(&counters[2], 0u, memory_order_relaxed);
}
