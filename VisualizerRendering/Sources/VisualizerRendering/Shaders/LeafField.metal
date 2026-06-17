// ── LEAF FIELD ───────────────────────────────────────────────────────────────
//
// GPU expansion + per-frame wind animation for the forest's leaves.
//
// Each leaf is one quad (4 vertices, 2 triangles). The CPU side uploads a
// SimBuffer<LeafInstance> at scene build time — one record per leaf with the
// leaf's tree-local position, orientation quaternion, size, and a per-leaf
// random phase used to break wind synchrony across the forest.
//
// Per frame: ForestController updates a SimBuffer<TreeTransform> with each
// tree's current world position + sway quaternion, advances u_time, and
// dispatches `leafExpand`. The kernel:
//
//   1. reads instance i and its tree transform
//   2. computes the 4 quad corners in leaf-local space
//   3. applies per-vertex wind flutter (tip flutters, base near petiole anchored)
//   4. rotates by leaf orientation → tree-local
//   5. translates + rotates by tree transform → world
//   6. writes positions + normals to the SceneKit-readable vertex buffers
//
// SceneKit reads those buffers directly via SCNGeometrySource(buffer:…), so
// there's no CPU round-trip; the GPU writes vertices and the renderer
// consumes them in the same frame.
//
// Replaces the prior per-leaf SCNNode approach (one node per leaf, ~22K
// nodes for a 10-tree forest) with one node per species (3 nodes total) +
// one draw call each. Scales to 100K+ leaves without hitting CPU per-node
// overhead.

#include <metal_stdlib>
using namespace metal;

// ── Data structs (mirrored on the Swift side) ────────────────────────────────

// Per the ALIGNMENT RULE in PBDSolver.swift: structs shared with Swift use
// float4 only — packed_float3 would mismatch Swift's SIMD3<Float> stride
// (16 bytes), and the kernel would read garbage for every instance past
// the first. Per-field accessors below recover the original components.
struct LeafInstance {
    float4 localPosAndTreeIdx;  // xyz = localPos, w = treeIdx (uint bitcast)
    float4 orientQuat;          // leaf rotation in tree-local space
    float4 sizeAndPhase;        // x = sizeW, y = sizeH, z = phase, w = pad
};                              // 48 bytes total

struct TreeTransform {
    float4 worldPosAndPad;      // xyz = tree root world position, w = pad
    float4 worldQuat;           // tree sway rotation (whole-tree wind)
};                              // 32 bytes total

struct LeafExpandUniforms {
    uint  leafCount;            //  4
    float time;                 //  4  seconds (advances monotonically)
    float windAmp;              //  4  per-vertex flutter amplitude (metres)
    float windFreq;             //  4  cycles per second
};                              // 16 bytes total

// ── Quaternion helper ────────────────────────────────────────────────────────
//
// Rotate vec3 v by unit quaternion q. Standard q*v*q⁻¹ formulation reduced
// to the cross-product identity (Rodrigues form for quaternions).
inline float3 rotateByQuat(float3 v, float4 q) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// ── Leaf expand kernel ───────────────────────────────────────────────────────
//
// One thread per leaf instance. Writes 4 sequential vertices + 4 normals.
kernel void leafExpand(
    device const LeafInstance*       instances [[ buffer(0) ]],
    device const TreeTransform*      trees     [[ buffer(1) ]],
    constant     LeafExpandUniforms& u         [[ buffer(2) ]],
    device       packed_float3*      positions [[ buffer(3) ]],
    device       packed_float3*      normals   [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.leafCount) return;

    LeafInstance inst = instances[id];
    uint treeIdx = as_type<uint>(inst.localPosAndTreeIdx.w);
    TreeTransform tree = trees[treeIdx];

    float3 localPos = inst.localPosAndTreeIdx.xyz;
    float  sizeW   = inst.sizeAndPhase.x;
    float  sizeH   = inst.sizeAndPhase.y;
    float  leafPhase = inst.sizeAndPhase.z;

    // The leaf's SCNPlane sits centred on its local origin, in the local XY
    // plane, with the surface normal pointing along +Z. The −Y edge is the
    // petiole attachment (this matches how attachLeaves orients each leaf:
    // leaf-local +Y is the petiole direction, ie the leaf's "tip axis").
    float halfW = sizeW * 0.5;
    float halfH = sizeH * 0.5;

    // 4 corners in leaf-local space. Vertex order matches the index buffer
    // (bottom-left, bottom-right, top-left, top-right).
    float3 corners[4] = {
        float3(-halfW, -halfH, 0),  // 0: petiole-side left
        float3( halfW, -halfH, 0),  // 1: petiole-side right
        float3(-halfW,  halfH, 0),  // 2: tip-side left
        float3( halfW,  halfH, 0)   // 3: tip-side right
    };

    // Per-leaf wind phase. The leaf flutters in its OWN local frame so the
    // motion is along the leaf's surface plane (not in arbitrary world axes).
    float t = u.time * u.windFreq;
    float sway1 = sin(t        + leafPhase)         * u.windAmp;
    float sway2 = cos(t * 1.7  + leafPhase * 1.3)   * u.windAmp * 0.7;

    // Surface normal in leaf-local space (+Z). Rotated through orientation
    // and tree transform below.
    float3 leafNormal = float3(0, 0, 1);
    float3 worldNormal = rotateByQuat(
        rotateByQuat(leafNormal, inst.orientQuat),
        tree.worldQuat
    );

    uint outBase = id * 4;
    for (uint i = 0; i < 4; i++) {
        // Tip vertices (corners[2], corners[3] with y > 0) flutter; petiole
        // vertices (corners[0], corners[1] with y < 0) stay anchored.
        float weight = (corners[i].y > 0.0) ? 1.0 : 0.0;

        float3 local = corners[i];
        local.x += sway1 * weight;
        local.z += sway2 * weight;

        // Stage 1: leaf-local → tree-local
        float3 treeLocal = rotateByQuat(local, inst.orientQuat) + localPos;
        // Stage 2: tree-local → world
        float3 worldPos = rotateByQuat(treeLocal, tree.worldQuat)
                          + tree.worldPosAndPad.xyz;

        positions[outBase + i] = worldPos;
        normals[outBase + i]   = worldNormal;
    }
}
