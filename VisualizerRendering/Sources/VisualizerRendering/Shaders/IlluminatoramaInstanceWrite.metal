// ── ILLUMINATORAMA — GPU INSTANCE WRITE KERNELS ─────────────────────────────
//
// Kernels that build draw-instance and point-light buffers on the GPU (eggs,
// bulbs, glows) so the CPU never walks them per frame.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── GPU-resident instance write (opt-in; IlluminatoramaRenderer.onEncodeGPUInstances) ──
//
// Writes full `Instance` structs straight into the renderer's instance buffer
// from a GPU-resident transform buffer (e.g. EggMotion's output), computing a
// 1/r² rail-bulb irradiance tint as the emission term — so per-egg transform,
// material and lighting tint are produced entirely on the GPU with no CPU
// readback / rebuild. One thread per instance; it overwrites `groupStart + i`.
//
// The transform is the 3×4 column layout EggMotion.metal writes (col0..2 =
// rotation·scale columns, col3 = world position). Eggs are uniform-scaled
// rotations, so normalMatrix = transpose(inverse(upper3×3)) = basis / scale²
// (scale = column length) — matching IlluminatoramaInstance.normalMatrix.

struct InstWriteXform { float4 c0; float4 c1; float4 c2; float4 c3; };

struct EggInstanceWriteUniforms {
    uint  count;        // egg count (grid size)
    uint  bulbCount;    // bulbs to sum for the irradiance tint
    uint  groupStart;   // first slot of this mesh group in the instance buffer
    float minDistSq;    // 1/r² floor (scaled-world units²)
    float tintScale;    // raw irradiance → HDR emission gain
    float tintCap;      // per-egg emission ceiling
    float metallic;     // egg material metalness
    float roughness;    // egg material roughness
};

kernel void eggs_write_instances(
    constant EggInstanceWriteUniforms& U   [[buffer(0)]],
    device const InstWriteXform*  xforms   [[buffer(1)]],   // motion output (completed buffer)
    device const float4*          colors   [[buffer(2)]],   // per-egg albedo (xyz)
    device const float4*          bulbPos  [[buffer(3)]],   // bulb world positions (xyz)
    device const float4*          bulbCol  [[buffer(4)]],   // bulb LUT colours (xyz)
    device Instance*              outInst  [[buffer(5)]],   // renderer instance buffer
    uint i                                 [[thread_position_in_grid]]
) {
    if (i >= U.count) return;

    InstWriteXform M = xforms[i];
    float4x4 model = float4x4(M.c0, M.c1, M.c2, M.c3);
    float3x3 basis = float3x3(M.c0.xyz, M.c1.xyz, M.c2.xyz);
    float  s     = max(1e-6, length(M.c0.xyz));
    float  invS2 = 1.0 / (s * s);
    float4x4 normalMat = float4x4(
        float4(basis[0] * invS2, 0.0),
        float4(basis[1] * invS2, 0.0),
        float4(basis[2] * invS2, 0.0),
        float4(0.0, 0.0, 0.0, 1.0));

    // Rail-bulb irradiance tint: 1/r² sum over every bulb's LUT colour at the
    // egg's world position, normalised to a saturated hue and capped (mirrors
    // the former CPU EggsControllerUltra.updateEggRailTints).
    float3 p   = M.c3.xyz;
    float3 sum = float3(0.0);
    for (uint b = 0; b < U.bulbCount; ++b) {
        float3 d  = p - bulbPos[b].xyz;
        float  r2 = max(U.minDistSq, dot(d, d));
        sum += bulbCol[b].xyz / r2;
    }
    float  mag  = max(sum.x, max(sum.y, sum.z));
    float3 tint = float3(0.0);
    if (mag > 1e-4) {
        tint = (sum / mag) * min(mag * U.tintScale, U.tintCap);
    }

    Instance inst;
    inst.modelMatrix           = model;
    inst.normalMatrix          = normalMat;
    inst.albedo                = colors[i].xyz;
    inst.metallic              = U.metallic;
    inst.emission              = tint;
    inst.roughness             = U.roughness;
    inst.albedoTextureSlice    = -1;
    inst.metallicTextureSlice  = -1;
    inst.roughnessTextureSlice = -1;
    inst.normalTextureSlice    = -1;
    inst.emissionTextureSlice  = -1;
    inst.emissionIntensity     = 1.0;
    inst._padSlice1            = 0;

    outInst[U.groupStart + i] = inst;
}

// ── Bulb sphere instances (GPU-resident) ─────────────────────────────────────
// One thread per bulb: static position + the per-tick LUT colour → an emissive
// sphere `Instance`. Dark bulbs (LUT colour ≈ 0) collapse to a scale-0 transform
// so they vanish without leaving a black dot — the instance count stays fixed
// (TAA-stable), unlike the old CPU path that skipped dark bulbs.

struct BulbInstanceWriteUniforms {
    uint  bulbCount;
    uint  groupStart;
    float displayRadius;
    float emissionScale;
    float darkThreshold;
    float _pad0; float _pad1; float _pad2;
};

kernel void bulbs_write_instances(
    constant BulbInstanceWriteUniforms& U [[buffer(0)]],
    device const float4* bulbPos          [[buffer(1)]],
    device const float4* bulbCol          [[buffer(2)]],
    device Instance*     outInst          [[buffer(3)]],
    uint i                                [[thread_position_in_grid]]
) {
    if (i >= U.bulbCount) return;
    float3 c   = bulbCol[i].xyz;
    float3 p   = bulbPos[i].xyz;
    bool   lit = length(c) >= U.darkThreshold;
    float  s   = lit ? U.displayRadius : 0.0;          // dark → scale 0 → invisible
    float  invS = (s > 1e-6) ? (1.0 / s) : 0.0;

    Instance inst;
    inst.modelMatrix  = float4x4(float4(s,0,0,0), float4(0,s,0,0),
                                 float4(0,0,s,0), float4(p, 1.0));
    inst.normalMatrix = float4x4(float4(invS,0,0,0), float4(0,invS,0,0),
                                 float4(0,0,invS,0), float4(0,0,0,1));
    inst.albedo                = float3(0.0);
    inst.metallic              = 0.0;
    inst.emission              = lit ? (c * U.emissionScale) : float3(0.0);
    inst.roughness             = 0.0;
    inst.albedoTextureSlice    = -1;
    inst.metallicTextureSlice  = -1;
    inst.roughnessTextureSlice = -1;
    inst.normalTextureSlice    = -1;
    inst.emissionTextureSlice  = -1;
    inst.emissionIntensity     = 1.0;
    inst._padSlice1            = 0;

    outInst[U.groupStart + i] = inst;
}

// ── Glow strip emission (GPU-resident) ───────────────────────────────────────
// One thread per glow-strip slice. Each slice is a single-instance mesh kind
// (split off at the 60k-vert / UInt16-index limit) and the slices are laid out
// CONTIGUOUSLY in the instance buffer, so thread `i` writes `groupStart + i`.
// The emission is sampled from the same per-bulb LUT the bulb spheres use,
// spread across the network (slot = i/sliceCount · bulbCount) so a rainbow
// reads as a colour gradient along the rails — and so switching the pattern /
// hue visibly drives the dominant rail element, not just the bulbs. The strip
// geometry is already world-space, so the model + normal matrices are identity.

struct GlowInstanceWriteUniforms {
    uint  sliceCount;
    uint  bulbCount;
    uint  groupStart;
    float bloomScale;     // LUT colour multiplier; 0 when the rails aren't glowing
};

kernel void glow_write_instances(
    constant GlowInstanceWriteUniforms& U [[buffer(0)]],
    device const float4* bulbCol          [[buffer(1)]],
    device Instance*     outInst          [[buffer(2)]],
    uint i                                [[thread_position_in_grid]]
) {
    if (i >= U.sliceCount) return;
    uint slot = (U.sliceCount > 1 && U.bulbCount > 0)
        ? min(U.bulbCount - 1u, uint(float(i) / float(U.sliceCount) * float(U.bulbCount)))
        : 0u;
    float3 c = bulbCol[slot].xyz * U.bloomScale;

    // Glow strip vertices are world-space (identity model matrix). When the
    // rails aren't glowing (bloomScale 0) collapse the model matrix to the
    // origin so the full-ring shell draws nothing — otherwise an opaque
    // zero-emission tube would hide the chrome rail it wraps. (A partial arc
    // could get away with emission 0; a full ring can't.)
    bool glowing = U.bloomScale > 0.0;
    float g = glowing ? 1.0 : 0.0;

    Instance inst;
    inst.modelMatrix           = float4x4(float4(g,0,0,0), float4(0,g,0,0),
                                          float4(0,0,g,0), float4(0,0,0,1));
    inst.normalMatrix          = float4x4(float4(1,0,0,0), float4(0,1,0,0),
                                          float4(0,0,1,0), float4(0,0,0,1));
    inst.albedo                = float3(0.0);
    inst.metallic              = 0.0;
    inst.emission              = c;
    inst.roughness             = 0.0;
    inst.albedoTextureSlice    = -1;
    inst.metallicTextureSlice  = -1;
    inst.roughnessTextureSlice = -1;
    inst.normalTextureSlice    = -1;
    inst.emissionTextureSlice  = -1;
    inst.emissionIntensity     = 1.0;
    inst._padSlice1            = 0;

    outInst[U.groupStart + i] = inst;
}

// ── Camera-near bulb point lights (GPU cull) ─────────────────────────────────
// One thread per bulb: a lit bulb within cullRadius of the camera atomically
// claims a point-light slot (capped at maxLights) so the chrome rails pick up
// the marquee colour as real coloured highlights. Unclaimed slots keep the
// zeroed CPU placeholders → skipped by the deferred loop's radius cutoff.
// `counter` is reset to 0 (blit fill) before this dispatch.

struct BulbCullUniforms {
    float4 cameraCull;    // xyz = camera world pos, w = cullRadius²
    uint   bulbCount;
    uint   maxLights;
    uint   lightOffset;   // first reserved point-light slot
    float  darkThreshold;
    float  lightRadius;
    float  gain;
    float  _pad0; float _pad1;
};

kernel void bulbs_write_pointlights(
    constant BulbCullUniforms& U   [[buffer(0)]],
    device const float4* bulbPos   [[buffer(1)]],
    device const float4* bulbCol   [[buffer(2)]],
    device atomic_uint*  counter   [[buffer(3)]],
    device PointLight*   outLights [[buffer(4)]],
    uint i                         [[thread_position_in_grid]]
) {
    if (i >= U.bulbCount) return;
    float3 c = bulbCol[i].xyz;
    if (length(c) < U.darkThreshold) return;
    float3 d = bulbPos[i].xyz - U.cameraCull.xyz;
    if (dot(d, d) > U.cameraCull.w) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot >= U.maxLights) return;
    PointLight pl;
    pl.position = bulbPos[i].xyz;
    pl.radius   = U.lightRadius;
    pl.color    = c * U.gain;
    pl.layerMask = 0xFFFFFFFFu;   // all-bits: GPU-synthesised bulb lights affect every layer
    outLights[U.lightOffset + slot] = pl;
}
