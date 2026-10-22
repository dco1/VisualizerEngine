// ── ILLUMINATORAMA — SOFTWARE-RASTER HOST GLUE (Signack S4, Daydream DH-1461) ─────────────────
//
// The engine's two ends of a software rasterizer (`IlluminatoramaSoftRasterizer`, Signack's
// `SignackRaster` in Daydream) — see `IlluminatoramaRenderer+SoftRaster.swift`:
//
//   illumi_soft_vertex     the VERTEX STAGE, as a compute kernel: one thread per triangle of a
//                          mesh's soft record. Fetches the three vertices, runs the SAME
//                          `illumiDeform` as `illumi_vs` / `illumi_vis_vs` (model matrix, tree wind,
//                          sway, sink) and the same jittered `viewProjection`, and appends the
//                          window-space triangle + its visibility-buffer id to the rasterizer's
//                          list. So the soft path draws exactly the surface the hardware path would.
//   illumi_soft_merge_fs   the MERGE, inside the G-buffer pass right after the hardware id raster:
//                          one fullscreen triangle (scissored) that hands each pixel's nearest soft
//                          depth to the depth test (`[[depth(any)]]`) with its id in color(6). The
//                          depth test against everything already in the pass decides who is
//                          visible, so ids stay in TILE MEMORY and the S3 resolve runs unchanged.
//                          It resets every pixel it reads (the rasterizer's self-clearing contract).
//
// The id layout is S3's: x = front-facing (31) | draw slot + 1 (16–30) | instance (0–14), with
// bit 15 = "written by the software rasterizer" (the debug view reads it); y = triangle index.

#include <metal_stdlib>
#include "IlluminatoramaGBufferDeform.h"
using namespace metal;

/// Mirror of Signack's `SignackRasterTriangle` (48 B) — the rasterizer's input record.
struct IllumiSoftTriangle {
    packed_float3 v0, v1, v2;   // window x, y (px, y down), z/w
    uint idX;                   // bit 31 IN = cull back faces (OUT: front-facing, set by the rasterizer)
    uint idY;
    uint flags;
};
static_assert(sizeof(IllumiSoftTriangle) == 48, "mirror of SignackRasterTriangle");

/// Mirror of `IlluminatoramaSoftVertexParams` (32 B).
struct SoftVertexParams {
    uint   slot;           // the draw's visibility-buffer slot
    uint   instanceBase;   // the draw group's instance (soft draws are single-instance)
    uint   capacity;       // triangles the rasterizer's list holds
    uint   cullBack;       // single-sided mesh: drop back faces
    float2 targetSize;     // the G-buffer (px)
    float2 _pad;
    uint4  scissor;        // x0, y0, x1, y1 — a triangle wholly outside is not appended
};

constant uint kSoftFlag = 0x8000u;

kernel void illumi_soft_vertex(
    constant SoftVertexParams&  P             [[buffer(0)]],
    const device Vertex*        verts         [[buffer(1)]],
    const device uint*          indices       [[buffer(2)]],
    const device uint*          record        [[buffer(3)]],   // the soft record: [0] indexCount, [2] indexStart
    constant FrameUniforms&     frame         [[buffer(4)]],
    const device Instance*      instances     [[buffer(5)]],
    const device Instance*      prevInstances [[buffer(6)]],
    device IllumiSoftTriangle*  out           [[buffer(7)]],
    device atomic_uint*         state         [[buffer(8)]],   // [0] = the rasterizer's append counter
    uint t                                    [[thread_position_in_grid]])
{
    if (t >= record[0] / 3u) return;
    const uint tri = record[2] / 3u + t;
    const Instance inst = instances[P.instanceBase];
    const Instance prevInst = prevInstances[P.instanceBase];
    float3 w[3];
    for (uint k = 0; k < 3; ++k) {
        Vertex v = verts[indices[tri * 3u + k]];
        IllumiDeformed d = illumiDeform(v, v.position, inst, prevInst, frame);
        float4 clip = frame.viewProjection * d.worldP;   // exactly `illumi_vis_vs`'s position
        if (clip.w <= 1e-6f) return;   // crosses the eye plane — never a sub-pixel cluster
        float2 ndc = float2(precise::divide(clip.x, clip.w), precise::divide(clip.y, clip.w));
        w[k] = float3((ndc.x * 0.5f + 0.5f) * P.targetSize.x, (0.5f - ndc.y * 0.5f) * P.targetSize.y,
                      precise::divide(clip.z, clip.w));
    }
    float2 lo = min(w[0].xy, min(w[1].xy, w[2].xy)), hi = max(w[0].xy, max(w[1].xy, w[2].xy));
    if (hi.x < float(P.scissor.x) || hi.y < float(P.scissor.y)
        || lo.x > float(P.scissor.z) || lo.y > float(P.scissor.w)) return;
    uint i = atomic_fetch_add_explicit(&state[0], 1u, memory_order_relaxed);
    if (i >= P.capacity) return;   // counted as dropped by the rasterizer
    IllumiSoftTriangle o;
    o.v0 = w[0]; o.v1 = w[1]; o.v2 = w[2];
    o.idX = (P.cullBack != 0u ? 0x80000000u : 0u) | (((P.slot + 1u) & 0x7FFFu) << 16) | kSoftFlag;
    o.idY = tri;
    o.flags = 0u;
    out[i] = o;
}

struct SoftMergeIn { float4 position [[position]]; };

struct SoftMergeOut {
    uint2 vis   [[color(6)]];
    float depth [[depth(any)]];
};

fragment SoftMergeOut illumi_soft_merge_fs(
    SoftMergeIn         pin    [[stage_in]],
    constant uint&      width  [[buffer(0)]],
    device uint*        depth  [[buffer(1)]],
    const device uint2* ids    [[buffer(2)]])
{
    const uint2 p = uint2(pin.position.xy);
    const uint i = p.y * width + p.x;
    const uint z = depth[i];
    if (z == 0xFFFFFFFFu) { discard_fragment(); SoftMergeOut none; return none; }
    depth[i] = 0xFFFFFFFFu;   // self-clearing: the next frame starts empty here
    SoftMergeOut o;
    o.vis = ids[i];
    o.depth = as_type<float>(z);
    return o;
}
