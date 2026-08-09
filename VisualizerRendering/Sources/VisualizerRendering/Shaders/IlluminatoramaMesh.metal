// ── ILLUMINATORAMA — MESH GEOMETRY SYNTHESIS ────────────────────────────────
//
// Position/normal repack and the two-kernel GPU normal + tangent synthesis that
// replaced the CPU passes. `IlluminatoramaMeshSynthTests` compiles THIS file.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Phase 4.13a — DynamicMesh bridge ─────────────────────────────────────────
//
// `DynamicMesh` writes its per-frame compute output into separate
// `packed_float3` position + normal buffers (12-byte stride each) — the
// shape SceneKit expects from a buffer-backed `SCNGeometrySource`.
// Illuminatorama's deferred PBR pipeline reads vertices through one
// interleaved `IlluminatoramaVertex` array (96-byte stride: pos, normal,
// uv, tangent, with hidden-lane padding to match Swift's SIMD3<Float>).
// This kernel runs once per `DynamicMesh` per frame on Illuminatorama's
// own queue, immediately before the G-buffer pass that reads the
// repacked buffer — keeping vertices entirely GPU-resident and avoiding
// any CPU round-trip.
//
// Tangent synthesis: hot dogs / fluid surfaces don't sample normal
// maps, so a stable arbitrary perpendicular to `N` is fine. Pick the
// axis least aligned with N, cross-product it; orthonormalise once.

kernel void illumi_repack_pos_norm(
    const device packed_float3*  inPos       [[buffer(0)]],
    const device packed_float3*  inNorm      [[buffer(1)]],
    device Vertex*               outVertex   [[buffer(2)]],
    constant uint&               count       [[buffer(3)]],
    // Phase 4.17 — optional packed-float2 UV stream. When the host
    // passes a real UV buffer through `IlluminatoramaGPUMeshDescriptor.
    // uvBuffer` we sample it per vertex; otherwise the host binds a
    // throwaway 1-element buffer here and sets `hasUV = 0` so the
    // kernel falls back to the synthetic `(0,0)` default. Bound at
    // buffer(4) with the flag at buffer(5).
    const device packed_float2*  inUV        [[buffer(4)]],
    constant uint&               hasUV       [[buffer(5)]],
    // Optional per-vertex RGBA color stream (stride-16 float4). When the host
    // passes a real color buffer through `IlluminatoramaGPUMeshDescriptor.
    // colorBuffer` we sample it per vertex (→ multiplied into albedo at shading);
    // otherwise the host binds a throwaway buffer and sets hasColor = 0 so the
    // kernel writes identity white (a no-op). Used by the coin soup for per-coin
    // DEBUG tints, and available to any instanced GPU mesh that wants vertex color.
    const device float4*         inColor     [[buffer(6)]],
    constant uint&               hasColor    [[buffer(7)]],
    uint                         gid         [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float3 p = float3(inPos[gid]);
    float3 n = normalize(float3(inNorm[gid]));
    // Stable arbitrary tangent: cross N with whichever axis is least
    // aligned (so we don't pick a near-parallel vector and produce
    // a degenerate cross product).
    float3 axis = abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(axis, n));
    float2 uv = (hasUV != 0u) ? float2(inUV[gid]) : float2(0.0);

    Vertex v;
    v.position = p;
    v._padPos  = 0.0;
    v.normal   = n;
    v._padNrm  = 0.0;
    v.uv       = uv;
    v._padUv   = float2(0.0);
    v.tangent  = float4(t, 1.0);
    // Per-vertex color → albedo (white identity when the mesh ships no color).
    v.color    = (hasColor != 0u) ? inColor[gid] : float4(1.0);
    outVertex[gid] = v;
}

// ── Phase 4.21 — GPU mesh normal / tangent synthesis ─────────────────────────
//
// One-shot, GPU-resident replacement for the CPU `synthesiseNormals` /
// `synthesiseTangents` passes that used to run inside
// `IlluminatoramaMesh.from(scnGeometry:)` on the MAIN THREAD. For a cold-cache
// scene that CPU work — O(triangles) scatter-add of per-vertex tangents, the
// classic derivative-of-UV method — parked the run loop for seconds the first
// frame the Illuminatorama overlay attached (it converts EVERY geometry in one
// synchronous `extractFrame` tick). This moves the math onto the GPU.
//
// The accumulation is a per-vertex sum over the triangles incident on that
// vertex. Rather than float-atomic scatter (which the project deliberately
// avoids — see MLSMPM.metal's contention notes), we build a bounded per-vertex
// adjacency list with INTEGER atomics (the counting-sort idiom from
// mlsCellCount/mlsScatterParticles), then gather per vertex. Two dispatches in
// separate encoders (the second sees the first's writes) plus a blit-clear of
// the count buffer:
//
//   1. illumi_mesh_build_adjacency — one thread per triangle; for each of its
//      3 corners, claim a slot via atomic-inc of that vertex's count and store
//      the triangle id in `vertTriList[v*maxValence + slot]`. Vertices past
//      `maxValence` incident triangles drop the overflow (visually negligible
//      for a smoothed average; real meshes sit well under the cap).
//   2. illumi_mesh_synth — one thread per vertex; loop its incident triangles,
//      recompute the area-weighted face normal and UV-derivative tangent,
//      accumulate, then normalise + Gram-Schmidt the tangent against the
//      normal and write both back into the interleaved vertex buffer.
//
// `idx16`/`idx32` are the SAME index buffer bound to two slots; `isU32` selects
// which is read (the other branch is never executed, so no out-of-bounds read).
// `doNormals` / `doTangents` gate each output independently: SCN geometry that
// ships normals keeps them (doNormals = 0) and only gets tangents synthesised.

kernel void illumi_mesh_build_adjacency(
    device const ushort* idx16        [[buffer(0)]],
    device const uint*   idx32        [[buffer(1)]],
    constant uint&       isU32        [[buffer(2)]],
    constant uint&       triCount     [[buffer(3)]],
    constant uint&       maxValence   [[buffer(4)]],
    device atomic_uint*  vertTriCount [[buffer(5)]],
    device uint*         vertTriList  [[buffer(6)]],
    constant uint&       vertCount    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= triCount) return;
    uint base = tid * 3u;
    uint i0 = (isU32 != 0u) ? idx32[base]      : uint(idx16[base]);
    uint i1 = (isU32 != 0u) ? idx32[base + 1u] : uint(idx16[base + 1u]);
    uint i2 = (isU32 != 0u) ? idx32[base + 2u] : uint(idx16[base + 2u]);
    uint corners[3] = { i0, i1, i2 };
    for (uint k = 0u; k < 3u; ++k) {
        uint v = corners[k];
        // Guard malformed meshes whose indices exceed the vertex count — an
        // out-of-range write here would fault the GPU and blank every later
        // draw. The CPU reference path did the same `i < verts.count` check.
        if (v >= vertCount) continue;
        uint slot = atomic_fetch_add_explicit(&vertTriCount[v], 1u, memory_order_relaxed);
        if (slot < maxValence) vertTriList[v * maxValence + slot] = tid;
    }
}

kernel void illumi_mesh_synth(
    device Vertex*       verts        [[buffer(0)]],
    device const ushort* idx16        [[buffer(1)]],
    device const uint*   idx32        [[buffer(2)]],
    constant uint&       isU32        [[buffer(3)]],
    constant uint&       vertCount    [[buffer(4)]],
    constant uint&       maxValence   [[buffer(5)]],
    device const uint*   vertTriCount [[buffer(6)]],
    device const uint*   vertTriList  [[buffer(7)]],
    constant uint&       doNormals    [[buffer(8)]],
    constant uint&       doTangents   [[buffer(9)]],
    uint vid [[thread_position_in_grid]]
) {
    if (vid >= vertCount) return;
    uint cnt = min(vertTriCount[vid], maxValence);
    float3 nAccum = float3(0.0);
    float3 tAccum = float3(0.0);
    for (uint s = 0u; s < cnt; ++s) {
        uint tid  = vertTriList[vid * maxValence + s];
        uint base = tid * 3u;
        uint i0 = (isU32 != 0u) ? idx32[base]      : uint(idx16[base]);
        uint i1 = (isU32 != 0u) ? idx32[base + 1u] : uint(idx16[base + 1u]);
        uint i2 = (isU32 != 0u) ? idx32[base + 2u] : uint(idx16[base + 2u]);
        // Skip degenerate / out-of-range triangles (mirrors the CPU guard).
        if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
        float3 p0 = verts[i0].position;
        float3 p1 = verts[i1].position;
        float3 p2 = verts[i2].position;
        float3 e1 = p1 - p0;
        float3 e2 = p2 - p0;
        if (doNormals != 0u) {
            // Area-weighted (un-normalised cross) — the standard
            // angle/area-weighted smooth-normal scheme.
            nAccum += cross(e1, e2);
        }
        if (doTangents != 0u) {
            float2 uv0 = verts[i0].uv;
            float2 d1  = verts[i1].uv - uv0;
            float2 d2  = verts[i2].uv - uv0;
            float det  = d1.x * d2.y - d1.y * d2.x;
            if (fabs(det) > 1e-8) {
                tAccum += (e1 * d2.y - e2 * d1.y) * (1.0 / det);
            }
        }
    }
    float3 n = verts[vid].normal;
    if (doNormals != 0u) {
        n = (length(nAccum) > 1e-6) ? normalize(nAccum) : n;
        verts[vid].normal = n;
    }
    if (doTangents != 0u) {
        // Gram-Schmidt against the (possibly just-synthesised) normal.
        float3 t = tAccum - n * dot(tAccum, n);
        float len = length(t);
        if (len < 1e-5) {
            float3 axis = (fabs(n.y) < 0.99) ? float3(0, 1, 0) : float3(1, 0, 0);
            t = normalize(cross(axis, n));
        } else {
            t /= len;
        }
        verts[vid].tangent = float4(t, 1.0);
    }
}
