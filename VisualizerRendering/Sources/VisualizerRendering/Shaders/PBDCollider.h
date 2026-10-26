// PBDCollider — THE rigid-collider record every solver shader reads. One definition (DH-1547).
//
// Matches `PBDCollider` in PBDSolver.swift byte for byte (48 B). Before this header the struct
// was re-declared in PBD.metal, PaperCloth.metal, MLSMPM.metal (as `PBDColliderMLS`) and
// ShapeMatchSoftBody.metal (as `SMCollider`); a layout change meant editing four copies and a
// Swift struct in lockstep. Include this instead of declaring another.
//
//   a.xyz = primary point (centre / capsule endpoint A / SDF-volume origin)
//   a.w   = kind tag, bit-cast from uint (0 sphere, 1 capsule, 2 box, 3 sdfVolume)
//   b.xyz = secondary point (capsule endpoint B / box half-extents); b.w = radius
//           (sdfVolume: b.x = cell size)
//   meta.x = ownerID (skip-self filter); meta.y = a box's yaw about +Y (bit-cast float);
//           sdfVolume: meta.yzw = grid dims.
#pragma once
#include <metal_stdlib>
using namespace metal;

struct PBDCollider {
    float4 a;
    float4 b;
    uint4  meta;
};
