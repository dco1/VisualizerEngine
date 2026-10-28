// SimSpatialHash — the ONE uniform-grid hash every particle solver's broadphase shares (cloth, soft bodies, ropes).
// A solver's self-collide kernel walks the 27 cells around a particle with `simHashCell`; the build kernels live in
// SimSpatialHash.metal and are encoded by SimSpatialHash.swift. Same function on both sides or buckets disagree.
#ifndef SIM_SPATIAL_HASH_H
#define SIM_SPATIAL_HASH_H

#include <metal_stdlib>
using namespace metal;

static inline int3 simCellCoord(float3 p, float cellSize) { return int3(floor(p / cellSize)); }

static inline uint simHashCell(int3 c, uint tableSize) {
    uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u);
    return h % tableSize;
}

// Where the float4 position sits in a solver's particle record (PBDParticle and SMParticle: stride 32, offset 0).
struct SimHashBuildUniforms {
    uint  count;
    uint  tableSize;
    float cellSize;
    uint  byteStride;
    uint  posOffset;
    uint  _pad0, _pad1, _pad2;
};

#endif
