// ── ILLUMINATORAMA — VISIBILITY BUFFER (Signack S3, Daydream DH-1461) ────────────────────────
//
// For meshes a host opts in (`IlluminatoramaMesh.visBuffer` — Signack's GPU-cut trees), the camera
// pass is split in two, inside ONE render pass that loads the G-buffer the conventional pass just
// wrote (see `IlluminatoramaRenderer+VisBuffer.swift`):
//
//   1. illumi_vis_vs / illumi_vis_fs   rasterise the opted-in meshes with the G-buffer's own depth
//      (test + write), writing ONLY a 64-bit id into a memoryless attachment (color 6):
//        x = front-facing (bit 31) | draw slot + 1 (bits 16–30) | software-rasterised (bit 15, S4)
//            | instance within the draw (0–14)
//        y = triangle index in the mesh's index buffer (indirect record's start + primitive id)
//      Every other attachment is write-masked off, so what the conventional pass wrote stays put
//      until step 2 decides who won. No material work at all: overdraw costs one id write.
//   2. illumi_vis_resolve_fs           one fullscreen triangle. Where the id is set (the opted-in
//      geometry won the depth test) it fetches the triangle's three vertices, re-runs the SAME
//      `illumiDeform` the vertex stage ran (model matrix, tree wind at this frame's time and at
//      the previous pose, sway, sink), finds the pixel's perspective-correct barycentrics from the
//      jittered clip positions (2D-homogeneous: no division by w, so a triangle crossing the near
//      plane resolves correctly), interpolates every `VSOut` field, reconstructs the screen-space
//      derivatives the hardware would have taken (quad-parity fine differences on the same
//      triangle plane), and calls the SAME `illumiGBufferShade` as `illumi_fs`. It writes the
//      ordinary G-buffer targets, so lighting, SSAO, TAA, the photo lane and the path lane read
//      exactly what they always read.
// Material evaluation therefore runs once per covered pixel however deep the foliage is.

#include <metal_stdlib>
#include "IlluminatoramaGBufferShade.h"
using namespace metal;

/// Counting variant (overdraw instrument): step 1 counts depth-passing fragments, step 2 counts
/// resolved pixels. Off in every production pipeline.
constant bool kVisCount [[function_constant(20)]];

/// One opted-in draw, as the resolve reads it — mirror of `IlluminatoramaVisBufferDraw` (32 B).
struct VisBufferDraw {
    device const Vertex* vertices;
    device const uint*   indices;      // 32-bit
    device const uint*   clusterIDs;   // per triangle of `indices` (Signack), or null
    uint                 instanceBase; // the draw group's first slot in the instance buffer
    uint                 _pad;
};

/// Mirror of `IlluminatoramaVisBufferResolveParams` (16 B).
struct VisBufferResolveParams {
    float2 targetSize;   // the G-buffer's size in pixels
    uint   debugMode;    // 0 shade · 1 cluster · 2 triangle · 3 draw · 4 rasterizer — neon
    uint   _pad;
};

struct VisVSOut {
    float4 clipPos [[position]];
    uint   iid     [[flat]];
};

vertex VisVSOut illumi_vis_vs(
    uint                    vid           [[vertex_id]],
    uint                    iid           [[instance_id]],
    const device Vertex*    verts         [[buffer(0)]],
    constant FrameUniforms& frame         [[buffer(1)]],
    const device Instance*  instances     [[buffer(2)]],
    const device Instance*  prevInstances [[buffer(4)]])
{
    Vertex v = verts[vid];
    IllumiDeformed d = illumiDeform(v, v.position, instances[iid], prevInstances[iid], frame);
    VisVSOut o;
    o.clipPos = frame.viewProjection * d.worldP;   // exactly `illumi_vs`'s clipPos
    o.iid = iid;
    return o;
}

struct VisFragOut {
    uint2 vis [[color(6)]];
};

[[early_fragment_tests]]
fragment VisFragOut illumi_vis_fs(
    VisVSOut              in       [[stage_in]],
    bool                  ff       [[front_facing]],
    uint                  pid      [[primitive_id]],
    constant uint&        slot     [[buffer(0)]],
    device const uint*    args     [[buffer(1)]],   // the draw's indirect record (or zeros)
    device atomic_uint*   counters [[buffer(2), function_constant(kVisCount)]])
{
    if (kVisCount) { atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed); }
    VisFragOut o;
    // `primitive_id` counts from the draw's first index; the record's indexStart says where that is.
    uint tri = args[2] / 3u + pid;
    o.vis = uint2((ff ? 0x80000000u : 0u) | (((slot + 1u) & 0x7FFFu) << 16) | (in.iid & 0x7FFFu), tri);
    return o;
}

struct VisResolveVSOut {
    float4 position [[position]];
};

vertex VisResolveVSOut illumi_vis_resolve_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VisResolveVSOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}

/// Perspective-correct barycentrics of the point a triangle shows through NDC `p` — 2D homogeneous
/// (Olano & Greer): the columns are the vertices' clip (x, y, w); b ∝ adj(M)·(p, 1).
static inline float3 visBary(float3 a, float3 b, float3 c, float2 p) {
    float3 q = float3(p, 1.0f);
    float3 r = float3(dot(cross(b, c), q), dot(cross(c, a), q), dot(cross(a, b), q));
    float s = r.x + r.y + r.z;
    return r / (fabs(s) > 1e-30f ? s : 1e-30f);
}

/// The same barycentrics the RASTERISER computes: vertices projected to pixels and snapped to its
/// sub-pixel grid (1/256 px) — what it actually interpolates with — then screen-space edge functions
/// and the perspective correction. For a sub-pixel sliver (a twig seen side-on) the snap changes the
/// triangle's shape enough that the exact plane and the hardware's disagree on its normal; this
/// matches the hardware. Only valid with every w > 0 (no near-plane crossing): `visBary` otherwise.
static inline float3 visBarySnapped(float4 c0, float4 c1, float4 c2, float2 pix, float2 size) {
    float2 s0 = (c0.xy / c0.w * float2(0.5f, -0.5f) + 0.5f) * size;
    float2 s1 = (c1.xy / c1.w * float2(0.5f, -0.5f) + 0.5f) * size;
    float2 s2 = (c2.xy / c2.w * float2(0.5f, -0.5f) + 0.5f) * size;
    s0 = round(s0 * 256.0f) / 256.0f; s1 = round(s1 * 256.0f) / 256.0f; s2 = round(s2 * 256.0f) / 256.0f;
    float2 e1 = s1 - s0, e2 = s2 - s0, q = pix - s0;
    float area = e1.x * e2.y - e1.y * e2.x;
    if (fabs(area) < 1e-12f) return float3(1.0f / 3.0f);
    float l1 = (q.x * e2.y - q.y * e2.x) / area;
    float l2 = (e1.x * q.y - e1.y * q.x) / area;
    float3 l = float3(1.0f - l1 - l2, l1, l2) / float3(c0.w, c1.w, c2.w);
    return l / (l.x + l.y + l.z);
}

template <typename T>
static inline T visLerp(float3 b, T a0, T a1, T a2) { return a0 * b.x + a1 * b.y + a2 * b.z; }

/// A neon colour per id (the debug views): golden-ratio hue walk, full saturation.
static inline float3 visNeon(uint id) {
    float h = fract(float(id) * 0.61803398875f + 0.13f);
    float3 k = saturate(abs(fract(h + float3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f) - 1.0f);
    return mix(float3(1.0f), k, 0.92f);
}

fragment GBufferOut illumi_vis_resolve_fs(
    VisResolveVSOut                          pin               [[stage_in]],
    uint2                                    vis               [[color(6)]],
    constant VisBufferResolveParams&         rp                [[buffer(0)]],
    constant FrameUniforms&                  frame             [[buffer(1)]],
    const device Instance*                   instances         [[buffer(2)]],
    const device float2*                     albedoUVScale     [[buffer(3)]],
    const device Instance*                   prevInstances     [[buffer(4)]],
    const device float2*                     nonColorUVScale   [[buffer(5)]],
    const device float4*                     albedoSliceMean   [[buffer(7)]],
    const device float4*                     nonColorSliceMean [[buffer(8)]],
    constant VisBufferDraw*                  draws             [[buffer(10)]],
    device atomic_uint*                      counters          [[buffer(11), function_constant(kVisCount)]],
    texture2d_array<float, access::sample>   albedoAtlas       [[texture(0)]],
    texture2d_array<float, access::sample>   nonColorAtlas     [[texture(1)]])
{
    if (vis.x == 0u) { discard_fragment(); GBufferOut none; return none; }
    if (kVisCount) { atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed); }
    const uint slot = ((vis.x >> 16) & 0x7FFFu) - 1u;
    const uint iid  = vis.x & 0x7FFFu;
    const bool frontFacing = (vis.x >> 31) != 0u;
    const uint tri  = vis.y;
    const VisBufferDraw dr = draws[slot];

    const uint ii = dr.instanceBase + iid;
    const Instance inst = instances[ii];
    const Instance prevInst = prevInstances[ii];
    Vertex v[3];
    IllumiDeformed d[3];
    float4 clip[3], prevClip[3];
    for (uint k = 0; k < 3; ++k) {
        v[k] = dr.vertices[dr.indices[tri * 3u + k]];
        d[k] = illumiDeform(v[k], v[k].position, inst, prevInst, frame);
        clip[k] = frame.viewProjection * d[k].worldP;
        prevClip[k] = frame.previousViewProjection * d[k].prevWorldP;
    }
    const float3 h0 = clip[0].xyw, h1 = clip[1].xyw, h2 = clip[2].xyw;

    // The pixel centre and its quad-mate neighbours in NDC (y up). A hardware dfdx on Apple GPUs
    // is the fine difference inside the 2×2 quad: forward on the even column, backward on the odd.
    const float2 px = pin.position.xy;
    const float2 ndc = float2(px.x / rp.targetSize.x * 2.0f - 1.0f, 1.0f - px.y / rp.targetSize.y * 2.0f);
    const float sx = 2.0f / rp.targetSize.x, sy = -2.0f / rp.targetSize.y;
    const bool oddX = (uint(px.x) & 1u) != 0u, oddY = (uint(px.y) & 1u) != 0u;
    float3 b, bx, by;
    const float2 dpx = float2(oddX ? -1.0f : 1.0f, 0.0f), dpy = float2(0.0f, oddY ? -1.0f : 1.0f);
    if (min(clip[0].w, min(clip[1].w, clip[2].w)) > 1e-6f) {
        b  = visBarySnapped(clip[0], clip[1], clip[2], px, rp.targetSize);
        bx = visBarySnapped(clip[0], clip[1], clip[2], px + dpx, rp.targetSize);
        by = visBarySnapped(clip[0], clip[1], clip[2], px + dpy, rp.targetSize);
    } else {
        b  = visBary(h0, h1, h2, ndc);
        bx = visBary(h0, h1, h2, ndc + float2(dpx.x * sx, 0.0f));
        by = visBary(h0, h1, h2, ndc + float2(0.0f, dpy.y * sy));
    }

    VSOut in;
    in.currentClip  = visLerp(b, clip[0], clip[1], clip[2]);
    in.clipPos      = in.currentClip;
    in.previousClip = visLerp(b, prevClip[0], prevClip[1], prevClip[2]);
    in.worldPos     = visLerp(b, d[0].worldP.xyz, d[1].worldP.xyz, d[2].worldP.xyz);
    in.worldNormal  = visLerp(b, d[0].worldN, d[1].worldN, d[2].worldN);
    in.uv           = visLerp(b, v[0].uv, v[1].uv, v[2].uv);
    in.instanceID   = ii;
    in.layer        = inst.layer;
    in.worldTangent = visLerp(b, float4(d[0].worldT, v[0].tangent.w), float4(d[1].worldT, v[1].tangent.w),
                              float4(d[2].worldT, v[2].tangent.w));
    in.vertexColor  = visLerp(b, v[0].color, v[1].color, v[2].color);

    float2 uvX = visLerp(bx, v[0].uv, v[1].uv, v[2].uv), uvY = visLerp(by, v[0].uv, v[1].uv, v[2].uv);
    float2 duvdx = oddX ? in.uv - uvX : uvX - in.uv;
    float2 duvdy = oddY ? in.uv - uvY : uvY - in.uv;
    float3 pX = visLerp(bx, d[0].worldP.xyz, d[1].worldP.xyz, d[2].worldP.xyz);
    float3 pY = visLerp(by, d[0].worldP.xyz, d[1].worldP.xyz, d[2].worldP.xyz);
    float3 worldPosFwidth = abs(pX - in.worldPos) + abs(pY - in.worldPos);

    InteriorMapParams noInterior = {};
    GBufferOut o = illumiGBufferShade(in, frontFacing, duvdx, duvdy, worldPosFwidth, instances, frame,
                                      albedoUVScale, nonColorUVScale, albedoSliceMean, nonColorSliceMean,
                                      noInterior, albedoAtlas, nonColorAtlas);
    if (rp.debugMode == 4u) {   // S4: which rasterizer wrote it — software magenta, hardware cyan
        float3 c = (vis.x & 0x8000u) != 0u ? float3(1.0f, 0.0f, 0.85f) : float3(0.0f, 0.9f, 1.0f);
        o.albedoMetallic = half4(half3(c), 0.0h);
        o.emission = half4(half3(c * 0.6f), o.emission.w);
    } else if (rp.debugMode != 0u) {
        uint id = rp.debugMode == 1u ? (dr.clusterIDs ? dr.clusterIDs[tri] : tri / 128u)
                : rp.debugMode == 2u ? tri * 2654435761u
                : slot;
        float3 c = visNeon(rp.debugMode == 2u ? (id >> 12) : id);
        o.albedoMetallic = half4(half3(c), 0.0h);
        o.emission = half4(half3(c * 0.6f), o.emission.w);
    }
    return o;
}
