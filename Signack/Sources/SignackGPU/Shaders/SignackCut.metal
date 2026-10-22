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
//                             atomic, every thread copies one triangle — and (S3) stamps the
//                             cluster id of each output triangle, so a visibility-buffer triangle id
//                             names its cluster with one load. Off-screen clusters fill
//                             [0, S), on-screen clusters [S, S + P) — so the SHADOW draw is the
//                             LOD cut [0, S + P) and the PRIMARY draw is [S, S + P).
//                             S4 (`softArea > 0`): an on-screen cluster's threadgroup instead tests
//                             EACH triangle's projected size; sub-pixel ones are written from the TOP
//                             of the buffer down, [C − Q, C) (C = capacity), the rest to [S, S + P).
//                             The SOFT record is [C − Q, C); the host draws it with the shadow record
//                             in every shadow pass, and with the primary record wherever it does not
//                             rasterise it in software. With `softArea == 0`, Q = 0: exactly S2.
//   3. signack_cut_finalize   one thread: the indirect argument records (primary, shadow, soft +
//                             its one-thread-per-triangle dispatch), then zero the counters.
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
    float  softArea;        // S4: on-screen triangles projecting < this many px² go soft (0 = none)
    uint   _pad1;
};
static_assert(sizeof(SignackGPUCutUniforms) == 144, "mirror of Swift SignackGPUCutUniforms");

/// S4 — how to measure a triangle on screen: the instance's object → clip transform (unjittered),
/// the target the software rasterizer draws, and where the host's vertex positions are.
struct SignackGPURouteUniforms {
    float4x4 objectToClip;
    float2   viewport;      // px
    float    maxSpan;       // a soft triangle's box must also be narrower than this (px, per axis)
    uint     vertexStride;  // bytes per host vertex; the position is a float3 at offset 0
};
static_assert(sizeof(SignackGPURouteUniforms) == 80, "mirror of Swift SignackGPURouteUniforms");

struct SignackDrawArgs {     // MTLDrawIndexedPrimitivesIndirectArguments
    uint indexCount;
    uint instanceCount;
    uint indexStart;
    int  baseVertex;
    uint baseInstance;
};

struct SignackDispatchArgs {  // MTLDispatchThreadgroupsIndirectArguments
    uint x, y, z;
};

constant uint SIGNACK_SOFT_THREADS = 64;   // SignackGPUDrawArgs.softThreadsPerGroup

// counters[0] off-screen indices (classify), [1] off-screen cursor, [2] on-screen hardware cursor,
// [3] soft cursor (counted down from the top of the buffer).

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
    device uint*                    outCluster [[buffer(7)]],   // S3: cluster id per output triangle
    constant SignackGPURouteUniforms& R        [[buffer(8)]],   // S4 (read only when softArea > 0)
    device const uchar*             vertices   [[buffer(9)]],   // S4: the host's vertex buffer
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
    if (f == 2 && U.softArea > 0.0f) {           // S4: route each triangle (tgSize ≥ 128 ≥ triangles)
        threadgroup atomic_uint tally[2];
        threadgroup uint first[2];
        if (t == 0) { atomic_store_explicit(&tally[0], 0u, memory_order_relaxed);
                      atomic_store_explicit(&tally[1], 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool mine = t * 3u < n;
        uint i0 = 0, i1 = 0, i2 = 0, slot = 0;
        bool soft = false;
        if (mine) {
            i0 = dagIndices[start + 3u * t]; i1 = dagIndices[start + 3u * t + 1u]; i2 = dagIndices[start + 3u * t + 2u];
            float4 c0 = R.objectToClip * float4(float3(*(device const packed_float3*)(vertices + i0 * R.vertexStride)), 1.0f);
            float4 c1 = R.objectToClip * float4(float3(*(device const packed_float3*)(vertices + i1 * R.vertexStride)), 1.0f);
            float4 c2 = R.objectToClip * float4(float3(*(device const packed_float3*)(vertices + i2 * R.vertexStride)), 1.0f);
            if (min3(c0.w, c1.w, c2.w) > 1.0e-6f) {
                float2 s0 = c0.xy / c0.w * 0.5f * R.viewport, s1 = c1.xy / c1.w * 0.5f * R.viewport;
                float2 s2 = c2.xy / c2.w * 0.5f * R.viewport;
                float2 e1 = s1 - s0, e2 = s2 - s0;
                float2 span = max(s0, max(s1, s2)) - min(s0, min(s1, s2));
                soft = 0.5f * fabs(e1.x * e2.y - e1.y * e2.x) < U.softArea && max(span.x, span.y) < R.maxSpan;
            }
            slot = atomic_fetch_add_explicit(&tally[soft ? 1 : 0], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint hw = 3u * atomic_load_explicit(&tally[0], memory_order_relaxed);
            uint sw = 3u * atomic_load_explicit(&tally[1], memory_order_relaxed);
            first[0] = atomic_load_explicit(&counters[0], memory_order_relaxed)
                     + atomic_fetch_add_explicit(&counters[2], hw, memory_order_relaxed);
            uint below = atomic_fetch_add_explicit(&counters[3], sw, memory_order_relaxed) + sw;
            first[1] = below <= U.capacity ? U.capacity - below : U.capacity;   // overflow writes nothing
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mine) {
            uint dst = first[soft ? 1 : 0] + 3u * slot;
            if (dst + 2u < U.capacity) {
                outIndices[dst] = i0; outIndices[dst + 1u] = i1; outIndices[dst + 2u] = i2;
                outCluster[dst / 3u] = cid;
            }
        }
        return;
    }
    if (t == 0) {
        base = (f == 1) ? atomic_fetch_add_explicit(&counters[1], n, memory_order_relaxed)
             : atomic_load_explicit(&counters[0], memory_order_relaxed)
               + atomic_fetch_add_explicit(&counters[2], n, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b = base;
    for (uint k = t; k < n; k += tgSize) {
        uint dst = b + k;
        if (dst < U.capacity) {
            outIndices[dst] = dagIndices[start + k];
            if (k % 3u == 0u) outCluster[dst / 3u] = cid;   // ranges start on a triangle (b % 3 == 0)
        }
    }
}

kernel void signack_cut_finalize(
    constant SignackGPUCutUniforms& U          [[buffer(0)]],
    device atomic_uint*             counters   [[buffer(3)]],
    device SignackDrawArgs*         args       [[buffer(6)]],   // primary @0, shadow @32, soft @64, soft dispatch @96
    uint i                                     [[thread_position_in_grid]])
{
    if (i != 0) return;
    uint s = atomic_load_explicit(&counters[0], memory_order_relaxed);
    uint p = atomic_load_explicit(&counters[2], memory_order_relaxed);
    uint q = min(atomic_load_explicit(&counters[3], memory_order_relaxed), U.capacity);
    uint hwEnd = min(s + p, U.capacity - q);   // a full cut fits level 0, so this never clips
    SignackDrawArgs a;
    a.instanceCount = 1; a.baseVertex = 0; a.baseInstance = 0;
    a.indexCount = hwEnd > s ? hwEnd - s : 0; a.indexStart = min(s, U.capacity);
    device SignackDrawArgs* primaryArgs = (device SignackDrawArgs*)((device uchar*)args + 0);
    device SignackDrawArgs* shadowArgs  = (device SignackDrawArgs*)((device uchar*)args + 32);
    device SignackDrawArgs* softArgs    = (device SignackDrawArgs*)((device uchar*)args + 64);
    device SignackDispatchArgs* softDispatch = (device SignackDispatchArgs*)((device uchar*)args + 96);
    *primaryArgs = a;
    a.indexCount = hwEnd; a.indexStart = 0;
    *shadowArgs = a;
    a.indexCount = q; a.indexStart = U.capacity - q;
    *softArgs = a;
    softDispatch->x = (q / 3u + SIGNACK_SOFT_THREADS - 1u) / SIGNACK_SOFT_THREADS;
    softDispatch->y = 1u; softDispatch->z = 1u;
    for (uint k = 0; k < 4; ++k) atomic_store_explicit(&counters[k], 0u, memory_order_relaxed);
}
