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


/// DH-1466 — Hi-Z occlusion: the pyramid's frame (mirror of Swift `SignackGPUOcclusion.Uniforms`).
struct SignackInstOcclusionUniforms {
    float4x4 toClip;      // the instance's space → the clip space the pyramid's depth was drawn in
    float2   depthSize;   // px of that depth
    uint     mipCount;
    uint     enabled;
    float    dilate;      // px each side (TAA jitter)
    float    _pad0, _pad1, _pad2;
};
static_assert(sizeof(SignackInstOcclusionUniforms) == 96, "mirror of Swift SignackGPUOcclusion.Uniforms");

/// Is the sphere (c, r) hidden behind everything over its footprint in the pyramid? Conservative:
/// the footprint is the projected bounding CUBE (dilated), its depth the cube's nearest corner, the
/// occluder depth the pyramid's MAX over every texel the footprint touches. Anything crossing the
/// near plane is visible.
static bool signack_inst_occluded(float3 c, float r, constant SignackInstOcclusionUniforms& O, texture2d<float, access::read> hzb) {
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

static float signack_rung_error(constant SignackInstanceUniforms& U, uint k) {
    return U.rungError[k / 4][k % 4];
}

kernel void signack_instances_classify(
    constant SignackInstanceUniforms& U         [[buffer(0)]],
    device const SignackInstanceBounds* bounds  [[buffer(1)]],
    device const uint4*                source   [[buffer(2)]],   // instanceCount × stride
    device uint4*                      lists    [[buffer(3)]],   // 2 · buckets · instanceCount × stride
    device atomic_uint*                counters [[buffer(4)]],   // [list · buckets + bucket]
    constant SignackInstOcclusionUniforms& O    [[buffer(6)]],   // DH-1466
    texture2d<float, access::read>     hzb      [[texture(0)]],
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
    // DH-1466: a plant hidden behind last frame's depth is drawn by the shadow passes only.
    if (onScreen && O.enabled != 0 && signack_inst_occluded(c, r, O, hzb)) onScreen = false;
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
