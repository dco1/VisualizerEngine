// ── SIGNACK S2 — GPU CUT SELECTION + FRUSTUM CULLING (Daydream DH-1461) ─────────────────────
//
// One placed instance of a cluster LOD DAG per dispatch triple:
//
//   1. signack_cut_classify   one THREAD per cluster: the crack-free cut test (`isInCut`, exactly
//                             `SignackSelection.isInCut` on the CPU) then the frustum test on the
//                             cluster's padded bounds. flag 0 = not in the cut, 1 = in the cut but
//                             off screen (shadow casters only), 2 = in the cut and on screen,
//                             3 = on screen but OCCLUDED (DH-1466 Hi-Z; shadow casters only, like 1).
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
// DH-1465 — per-light shadow culling. With `V.count > 0` shadow views (sun cascade slices, spot
// cones, point cube faces, as object-space frustums), classify also tests every cut cluster against
// each view and sets bit v of `viewMask` (+ counts the view's indices); `signack_cut_views_prefix`
// (one thread, between classify and write) gives each view a range of the POOL that follows the
// main output ([C, C + pool)) — or none, when the view sees the whole cut (its record is then the
// shadow record, no copy) or the pool is full (same fallback: never wrong, only not culled); write
// copies each cluster once more per view it lands in; finalize writes one record per view slot at
// 128 + 32·v (slots past V.count hold the shadow record). The main [off | on] layout is untouched.
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

/// DH-1465 — the shadow views one cut is culled against (mirror of `SignackGPUShadowViewUniforms`).
struct SignackGPUShadowViews {
    float4 planes[48];      // view v's six object-space planes at [6v, 6v + 6)
    uint   count;           // ≤ 8
    uint   poolCapacity;    // indices in the pool after the main output
    uint   _pad0, _pad1;
};
static_assert(sizeof(SignackGPUShadowViews) == 784, "mirror of Swift SignackGPUShadowViewUniforms");
constant uint SIGNACK_VIEW_FULL = 0xFFFFFFFFu;   // a view base meaning "draw the shadow record"


/// DH-1466 — Hi-Z occlusion: the pyramid's frame (mirror of Swift `SignackGPUOcclusion.Uniforms`).
struct SignackOcclusionUniforms {
    float4x4 toClip;      // the instance's space → the clip space the pyramid's depth was drawn in
    float2   depthSize;   // px of that depth
    uint     mipCount;
    uint     enabled;
    float    dilate;      // px each side (TAA jitter)
    float    _pad0, _pad1, _pad2;
};
static_assert(sizeof(SignackOcclusionUniforms) == 96, "mirror of Swift SignackGPUOcclusion.Uniforms");

/// Is the sphere (c, r) hidden behind everything over its footprint in the pyramid? Conservative:
/// the footprint is the projected bounding CUBE (dilated), its depth the cube's nearest corner, the
/// occluder depth the pyramid's MAX over every texel the footprint touches. Anything crossing the
/// near plane is visible.
static bool signack_occluded(float3 c, float r, constant SignackOcclusionUniforms& O, texture2d<float, access::read> hzb) {
    if (O.enabled == 0 || O.mipCount == 0) return false;
    float2 lo = float2(1.0e30f), hi = float2(-1.0e30f);
    float zmin = 1.0e30f;
    for (uint k = 0; k < 8; ++k) {
        float3 p = c + r * float3((k & 1u) ? 1.0f : -1.0f, (k & 2u) ? 1.0f : -1.0f, (k & 4u) ? 1.0f : -1.0f);
        float4 q = O.toClip * float4(p, 1.0f);
        if (q.w <= 1.0e-4f) return false;
        float3 n = q.xyz / q.w;
        lo = min(lo, n.xy); hi = max(hi, n.xy); zmin = min(zmin, n.z);
    }
    if (zmin <= 0.0f) return false;
    float2 pmin = float2(lo.x * 0.5f + 0.5f, 0.5f - hi.y * 0.5f) * O.depthSize - O.dilate;
    float2 pmax = float2(hi.x * 0.5f + 0.5f, 0.5f - lo.y * 0.5f) * O.depthSize + O.dilate;
    pmin = clamp(pmin, float2(0.0f), O.depthSize - 1.0f);
    pmax = clamp(pmax, float2(0.0f), O.depthSize - 1.0f);
    float ext = max(pmax.x - pmin.x, pmax.y - pmin.y);
    uint L = uint(clamp(ceil(log2(max(ext, 1.0f))) - 1.0f, 0.0f, float(O.mipCount - 1)));
    float s = exp2(float(L + 1));              // source px per level-L texel
    uint2 dim = uint2(hzb.get_width(L), hzb.get_height(L));
    uint2 t1 = min(uint2(pmax / s), dim - 1u), t0 = min(uint2(pmin / s), t1);
    float m = 0.0f;
    for (uint y = t0.y; y <= t1.y; ++y)
        for (uint x = t0.x; x <= t1.x; ++x) m = max(m, hzb.read(uint2(x, y), L).r);
    return zmin > m;
}

// counters[0] off-screen indices (classify), [1] off-screen cursor, [2] on-screen hardware cursor,
// [3] soft cursor (counted down from the top of the buffer), [4] whole-cut indices (classify),
// [5] occluded clusters (DH-1466; finalize copies it to args + 384 for the census);
// DH-1465: [8 + v] view v's indices (classify), [16 + v] its write cursor, [24 + v] its pool base.

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
    constant SignackGPUShadowViews& V          [[buffer(10)]],  // DH-1465
    device uchar*                   viewMask   [[buffer(11)]],
    constant SignackOcclusionUniforms& O       [[buffer(12)]],  // DH-1466
    texture2d<float, access::read>  hzb        [[texture(0)]],
    uint i                                     [[thread_position_in_grid]])
{
    if (i >= U.clusterCount) return;
    SignackGPUCluster c = clusters[i];
    uchar mask = 0;
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
        // DH-1466: on screen but hidden behind last frame's depth → flag 3, drawn by the shadow passes only.
        if (f == 2 && O.enabled != 0 && signack_occluded(c.bounds.xyz, c.bounds.w + U.boundsPad, O, hzb)) {
            f = 3;
            atomic_fetch_add_explicit(&counters[5], 1u, memory_order_relaxed);
        }
        if (f != 2) atomic_fetch_add_explicit(&counters[0], c.triangleCount * 3, memory_order_relaxed);
        if (V.count > 0) {
            atomic_fetch_add_explicit(&counters[4], c.triangleCount * 3, memory_order_relaxed);
            float r = c.bounds.w + U.boundsPad;
            for (uint v = 0; v < V.count; ++v) {
                bool in = true;
                for (uint p = 0; p < 6; ++p) {
                    float4 pl = V.planes[6 * v + p];
                    if (dot(pl.xyz, c.bounds.xyz) + pl.w < -r) { in = false; break; }
                }
                if (in) {
                    mask |= uchar(1u << v);
                    atomic_fetch_add_explicit(&counters[8 + v], c.triangleCount * 3, memory_order_relaxed);
                }
            }
        }
    }
    flags[i] = f;
    viewMask[i] = mask;
}

/// DH-1465 — one thread: give every shadow view its pool range, in view order (cascades first).
kernel void signack_cut_views_prefix(
    constant SignackGPUShadowViews& V          [[buffer(10)]],
    device atomic_uint*             counters   [[buffer(3)]],
    uint i                                     [[thread_position_in_grid]])
{
    if (i != 0) return;
    uint total = atomic_load_explicit(&counters[4], memory_order_relaxed);
    uint cursor = 0;
    for (uint v = 0; v < V.count; ++v) {
        uint n = atomic_load_explicit(&counters[8 + v], memory_order_relaxed);
        uint base = SIGNACK_VIEW_FULL;
        if (n < total && cursor + n <= V.poolCapacity) { base = cursor; cursor += n; }
        atomic_store_explicit(&counters[24 + v], base, memory_order_relaxed);
    }
}

/// DH-1465 — copy one cluster's indices once more into each view's pool range it lands in.
/// `mask` and the bases are uniform across the threadgroup, so the barriers are too.
static void signack_copy_views(uchar mask, uint start, uint n, constant SignackGPUShadowViews& V,
                               constant SignackGPUCutUniforms& U, device atomic_uint* counters,
                               device const uint* dagIndices, device uint* outIndices,
                               threadgroup uint& vbase, uint t, uint tgSize)
{
    for (uint v = 0; v < V.count; ++v) {
        if ((mask & (1u << v)) == 0) continue;
        uint pb = atomic_load_explicit(&counters[24 + v], memory_order_relaxed);
        if (pb == SIGNACK_VIEW_FULL) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) vbase = U.capacity + pb + atomic_fetch_add_explicit(&counters[16 + v], n, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint b = vbase, end = U.capacity + V.poolCapacity;
        for (uint k = t; k < n; k += tgSize) if (b + k < end) outIndices[b + k] = dagIndices[start + k];
    }
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
    constant SignackGPUShadowViews& V          [[buffer(10)]],  // DH-1465
    device const uchar*             viewMask   [[buffer(11)]],
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
    threadgroup uint vbase;
    signack_copy_views(viewMask[cid], start, n, V, U, counters, dagIndices, outIndices, vbase, t, tgSize);
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
        base = (f != 2) ? atomic_fetch_add_explicit(&counters[1], n, memory_order_relaxed)
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
    device SignackDrawArgs*         args       [[buffer(6)]],   // primary @0, shadow @32, soft @64, soft dispatch @96, views @128+32v
    constant SignackGPUShadowViews& V          [[buffer(10)]],  // DH-1465
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
    for (uint v = 0; v < 8; ++v) {   // DH-1465: a view's own range, or the shadow record
        device SignackDrawArgs* viewArgs = (device SignackDrawArgs*)((device uchar*)args + 128 + 32 * v);
        uint pb = v < V.count ? atomic_load_explicit(&counters[24 + v], memory_order_relaxed) : SIGNACK_VIEW_FULL;
        a.indexCount = hwEnd; a.indexStart = 0;
        if (pb != SIGNACK_VIEW_FULL) {
            a.indexCount = atomic_load_explicit(&counters[8 + v], memory_order_relaxed);
            a.indexStart = U.capacity + pb;
        }
        *viewArgs = a;
    }
    device uint* occludedOut = (device uint*)((device uchar*)args + 384);   // DH-1466 census
    *occludedOut = atomic_load_explicit(&counters[5], memory_order_relaxed);
    for (uint k = 0; k < 32; ++k) atomic_store_explicit(&counters[k], 0u, memory_order_relaxed);
}
