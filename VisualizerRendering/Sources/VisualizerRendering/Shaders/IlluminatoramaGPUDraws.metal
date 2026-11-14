// GPU-DRIVEN SHADOW DRAW SUBMISSION (Daydream DH-1684)
// ====================================================
//
// The sun cascades and spot/area slices used to be encoded on the CPU one instanced draw per mesh
// group per view: on a furnished house that is ~1,500 `drawIndexedPrimitives` a frame for the three
// cascades alone, and the driver's per-draw encode (plus the cull test and the mesh-table lookups in
// front of it) was ~two thirds of the renderer's CPU time in an orbit.
//
// Here one compute thread per (draw record, view) does what that loop did — the same frustum test
// against the same group bounds, then either RESETS its command (culled) or writes the bindings and
// the draw into an indirect command buffer. The render pass then runs each view's commands with one
// `executeCommandsInBuffer` per cull-mode bucket. A shadow pass writes depth only (`less`), so the
// map is the per-texel minimum of every fragment whatever the draw order: bucketing changes no
// texel. Pipeline state, cull mode, depth bias and depth state are INHERITED from the encoder.
//
// The test is CONSERVATIVE against the CPU one: a group is reset only when its bounds lie outside a
// plane by more than float error at its distance, so the GPU may draw a group the CPU would have
// skipped, never the reverse — culling is a pure skip, so neither choice changes a texel.

#include <metal_stdlib>
using namespace metal;

/// One draw record: a mesh group's geometry, instance slice and cull bounds. 80 bytes — must match
/// `IlluminatoramaGPUShadowRecord` (IlluminatoramaGPUShadowDraws.swift).
struct GPUShadowRecord {
    device const uchar* vertices;
    device const uchar* indices;
    uint   instanceOffset;   // bytes into this frame's instance ring
    uint   indexCount;
    uint   instanceCount;
    uint   flags;            // bit0 uint32 indices · bit1 bounded · bit2 no sun shadow
    float4 lo;
    float4 hi;
    uint   triangles;        // indexCount / 3 × instanceCount (cull census)
    uint   pad0, pad1, pad2;
};

/// One view (a cascade or a spot/area slice). 128 bytes — must match `IlluminatoramaGPUShadowView`.
struct GPUShadowView {
    float4 planes[6];                // world-space, xyz unit length; inside ⇔ dot(n, p) + w ≥ 0
    device const uchar* instances;   // this frame's instance ring
    device const uchar* lightVP;     // float4x4 the vertex stage reads at buffer(3)
    device const uchar* clock;       // float4 (time, wind, heading, sink) at buffer(4)
    uint   commandBase;              // first ICB slot of this view
    uint   flags;                    // bit0 cull · bit1 sun cascade (honours the no-sun-shadow bit)
};

struct GPUShadowICB {
    command_buffer icb [[id(0)]];
};

static bool gpuShadowOutside(float4 p, float3 lo, float3 hi) {
    float3 v = select(lo, hi, p.xyz >= 0);   // the corner furthest along the normal
    float d = dot(p.xyz, v) + p.w;
    // Float error of that dot at this magnitude, with a millimetre on top: only a box clearly
    // outside is reset (the CPU test runs in doubles).
    float mag = dot(abs(p.xyz), abs(v)) + abs(p.w);
    return d < -(1e-3f + 1e-5f * mag);
}

kernel void illumi_gpu_shadow_draws(
    uint2                           tid       [[thread_position_in_grid]],
    device const GPUShadowRecord*   records   [[buffer(0)]],
    device const GPUShadowView*     views     [[buffer(1)]],
    device GPUShadowICB&            container [[buffer(2)]],
    device atomic_uint*             stats     [[buffer(3)]],   // per view: drawn, culled, triangles culled, 0
    constant uint2&                 counts    [[buffer(4)]],   // (records, views)
    uint                            lane      [[thread_index_in_simdgroup]])
{
    // Every lane reaches the census reduction below (a threadgroup is one row: one view), so an
    // out-of-range lane only contributes zeros — one atomic per SIMD-group, not three per thread.
    uint drawn = 0, culled = 0, culledTris = 0;
    if (tid.x < counts.x && tid.y < counts.y) {
        device const GPUShadowRecord& r = records[tid.x];
        device const GPUShadowView& v = views[tid.y];
        render_command cmd(container.icb, v.commandBase + tid.x);
        if ((v.flags & 2u) != 0 && (r.flags & 4u) != 0) {
            // A kind excluded from the sun (lamps): the CPU loop skipped it before the cull test, uncounted.
            cmd.reset();
        } else {
            bool out = false;
            if ((v.flags & 1u) != 0 && (r.flags & 2u) != 0) {
                float3 lo = r.lo.xyz, hi = r.hi.xyz;
                for (uint i = 0; i < 6 && !out; ++i) out = gpuShadowOutside(v.planes[i], lo, hi);
            }
            if (out) {
                cmd.reset();
                culled = 1; culledTris = r.triangles;
            } else {
                drawn = 1;
                cmd.set_vertex_buffer(r.vertices, 0);
                cmd.set_vertex_buffer(v.instances + r.instanceOffset, 2);
                cmd.set_vertex_buffer(v.lightVP, 3);
                cmd.set_vertex_buffer(v.clock, 4);
                if ((r.flags & 1u) != 0) {
                    cmd.draw_indexed_primitives(primitive_type::triangle, r.indexCount,
                                                (device const uint*)r.indices, r.instanceCount, 0, 0);
                } else {
                    cmd.draw_indexed_primitives(primitive_type::triangle, r.indexCount,
                                                (device const ushort*)r.indices, r.instanceCount, 0, 0);
                }
            }
        }
    }
    drawn = simd_sum(drawn); culled = simd_sum(culled); culledTris = simd_sum(culledTris);
    if (lane == 0 && tid.y < counts.y && (drawn | culled) != 0) {
        device atomic_uint* s = stats + tid.y * 4;
        atomic_fetch_add_explicit(s + 0, drawn, memory_order_relaxed);
        atomic_fetch_add_explicit(s + 1, culled, memory_order_relaxed);
        atomic_fetch_add_explicit(s + 2, culledTris, memory_order_relaxed);
    }
}
