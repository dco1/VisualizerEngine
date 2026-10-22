// ── SIGNACK S6 — BATCHED INSTANCE LOD + CULL + COMPACTION (Daydream DH-1461) ───────────────────
//
// Every placed instance of ONE baked type (a species variant of a planting) in two dispatches:
//
//   1. signack_instances_classify   one THREAD per instance: pick its bucket — the coarsest RUNG
//                                   whose object-space error projects under the threshold, or the
//                                   IMPOSTOR bucket when the whole plant is only a few pixels wide —
//                                   and test its bounding sphere against the camera frustum. Then
//                                   copy the instance's record (opaque, `instanceStride` bytes — the
//                                   host's own instance struct) into that bucket's PRIMARY list when
//                                   it is on screen, and always into its SHADOW list (a plant behind
//                                   the camera still shades the lawn in front of it).
//   2. signack_instances_finalize   one thread per bucket: the two indexed-INSTANCED indirect records
//                                   (primary @ b·64, shadow @ b·64+32 — the shadow record two rungs
//                                   coarser): the rung's index range, the
//                                   bucket's instance count, and baseInstance = where its list sits.
//                                   Then zero the counters.
//
// The host draws each bucket as ONE instanced draw reading the compacted records (bound at offset
// 0; Metal's instance_id includes baseInstance), so a species costs (rungs + 1) draws per pass at
// any instance count. Self-contained: no engine header (Signack is its own package).

#include <metal_stdlib>
using namespace metal;

constant uint SIGNACK_MAX_BUCKETS = 8;

struct SignackInstanceUniforms {        // mirror of Swift SignackGPUInstanceUniforms (256 B)
    float4 planes[6];                   // world-space frustum, inside where dot(xyz, p) + w ≥ 0
    float4 camera;                      // xyz = camera (world), w = pixel scale (px per unit at d = 1)
    float4 rungError[2];                // object-space error per rung (8 slots)
    uint4  rungIndexStart[2];
    uint4  rungIndexCount[2];
    float  threshold;                   // acceptable projected error (px)
    float  impostorPixels;              // projected diameter below which → impostor (0 = none)
    uint   rungCount;                   // ≤ 7 (the impostor takes bucket rungCount)
    uint   impostorIndexCount;          // 0 = no impostor mesh
    uint   instanceCount;
    uint   instanceStride;              // bytes, a multiple of 16
    uint   isOrthographic;
    uint   cull;                        // 0 = no frustum test (everything on screen)
    float  maxDistance;                 // beyond this a plant is not drawn at all (0 = no limit)
    uint   _pad0, _pad1, _pad2;
};
static_assert(sizeof(SignackInstanceUniforms) == 256, "mirror of Swift SignackGPUInstanceUniforms");

struct SignackInstanceBounds {          // per instance, world space (32 B)
    float4 sphere;                      // centre, radius (wind reach included)
    float  scale;                       // object → world scale (rung errors are object-space)
    float  _pad0, _pad1, _pad2;
};

struct SignackInstancedArgs {           // MTLDrawIndexedPrimitivesIndirectArguments
    uint indexCount;
    uint instanceCount;
    uint indexStart;
    int  baseVertex;
    uint baseInstance;
};

static float signack_rung_error(constant SignackInstanceUniforms& U, uint k) {
    return U.rungError[k / 4][k % 4];
}

kernel void signack_instances_classify(
    constant SignackInstanceUniforms& U         [[buffer(0)]],
    device const SignackInstanceBounds* bounds  [[buffer(1)]],
    device const uint4*                source   [[buffer(2)]],   // instanceCount × stride
    device uint4*                      lists    [[buffer(3)]],   // 2 · buckets · instanceCount × stride
    device atomic_uint*                counters [[buffer(4)]],   // [list · buckets + bucket]
    uint i                                      [[thread_position_in_grid]])
{
    if (i >= U.instanceCount) return;
    SignackInstanceBounds b = bounds[i];
    float3 c = b.sphere.xyz;
    float r = b.sphere.w;
    float dist = U.isOrthographic != 0 ? 1.0f : max(precise::sqrt(dot(U.camera.xyz - c, U.camera.xyz - c)) - r, 1.0e-3f);
    if (U.maxDistance > 0.0f && dist > U.maxDistance) return;
    float pxPerUnit = U.camera.w / dist;                     // ortho: camera.w is px per unit, dist = 1
    uint buckets = U.rungCount + (U.impostorIndexCount > 0 ? 1u : 0u);
    uint bucket = 0;
    if (U.impostorIndexCount > 0 && 2.0f * r * pxPerUnit < U.impostorPixels) {
        bucket = U.rungCount;
    } else {
        for (uint k = 1; k < U.rungCount; ++k) {
            if (signack_rung_error(U, k) * b.scale * pxPerUnit <= U.threshold) bucket = k;
        }
    }
    bool onScreen = true;
    if (U.cull != 0) {
        for (uint p = 0; p < 6; ++p) {
            float4 pl = U.planes[p];
            if (dot(pl.xyz, c) + pl.w < -r) { onScreen = false; break; }
        }
    }
    uint words = U.instanceStride / 16u;
    device const uint4* src = source + i * words;
    for (uint list = onScreen ? 0u : 1u; list < 2u; ++list) {
        uint slot = atomic_fetch_add_explicit(&counters[list * buckets + bucket], 1u, memory_order_relaxed);
        device uint4* dst = lists + ((list * buckets + bucket) * U.instanceCount + slot) * words;
        for (uint w = 0; w < words; ++w) dst[w] = src[w];
    }
}

kernel void signack_instances_finalize(
    constant SignackInstanceUniforms& U         [[buffer(0)]],
    device atomic_uint*                counters [[buffer(4)]],
    device SignackInstancedArgs*       args     [[buffer(5)]],   // [bucket · 2 + list], 32-B stride
    uint b                                      [[thread_position_in_grid]])
{
    uint buckets = U.rungCount + (U.impostorIndexCount > 0 ? 1u : 0u);
    if (b >= buckets || b >= SIGNACK_MAX_BUCKETS) return;
    bool impostor = b == U.rungCount;
    for (uint list = 0; list < 2u; ++list) {
        SignackInstancedArgs a;
        // Shadow maps draw two rungs coarser (coverage-compensated cards keep a crown's shadow
        // density; a shadow texel is coarser than the camera's pixel anyway).
        uint r = (list == 1u && !impostor) ? min(b + 2u, U.rungCount - 1u) : b;
        a.indexCount = impostor ? U.impostorIndexCount : U.rungIndexCount[r / 4][r % 4];
        a.indexStart = impostor ? 0u : U.rungIndexStart[r / 4][r % 4];
        a.baseVertex = 0;
        a.baseInstance = (list * buckets + b) * U.instanceCount;
        a.instanceCount = atomic_load_explicit(&counters[list * buckets + b], memory_order_relaxed);
        device SignackInstancedArgs* dst = (device SignackInstancedArgs*)((device uchar*)args + (b * 2u + list) * 32u);
        *dst = a;
        atomic_store_explicit(&counters[list * buckets + b], 0u, memory_order_relaxed);
    }
}
