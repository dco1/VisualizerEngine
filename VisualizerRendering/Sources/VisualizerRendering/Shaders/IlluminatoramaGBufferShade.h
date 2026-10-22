// ── ILLUMINATORAMA — G-BUFFER SHADING (shared) ───────────────────────────────────────────────
//
// What a G-buffer sample IS, independent of how the pixel was found: the material evaluation
// (`illumiGBufferShade`), as a sequence of stages that each live in their own header. Included by
// `IlluminatoramaGBuffer.metal` (the conventional raster path) and `IlluminatoramaVisBuffer.metal`
// (Signack's visibility buffer + resolve, Daydream DH-1461), so the two paths share ONE copy of the
// shading — the resolve writes the same G-buffer the raster path does and every lane downstream
// (live, photo, path) is untouched.
//
//   IlluminatoramaGBufferDeform.h   pass constants, VSOut, `illumiDeform` (vertex pose)
//   IlluminatoramaGBufferSurface.h  material uv, normal map, base albedo / metallic / roughness
//   IlluminatoramaGBufferDetail.h   micro-occlusion, wear, wood knots, carpet bands
//   IlluminatoramaGBufferGround.h   the soil-marker ground
//   IlluminatoramaGBufferBark.h     species bark + mossy log
//   IlluminatoramaGBufferSkins.h    vertex-alpha skins (casing, plush)
//   IlluminatoramaGBufferPack.h     GBufferOut + packing (class tag, emission, material, velocity)
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
#include "IlluminatoramaGBufferSurface.h"
#include "IlluminatoramaGBufferDetail.h"
#include "IlluminatoramaGBufferGround.h"
#include "IlluminatoramaGBufferBark.h"
#include "IlluminatoramaGBufferSkins.h"
#include "IlluminatoramaGBufferPack.h"
using namespace metal;

// ── The G-buffer material evaluation, ONCE ───────────────────────────────────────────────────
// The whole of what `illumi_fs` writes, as a function of an interpolated `VSOut` and its screen-
// space derivatives. Two callers (Signack S3, Daydream DH-1461): `illumi_fs` (hardware
// interpolants, hardware derivatives) and the visibility-buffer resolve (attributes and their
// derivatives reconstructed per pixel from the triangle id). The only fragment-stage facts the
// body ever used — `front_facing`, `dfdx/dfdy(uv)` and `fwidth(worldPos)` — are parameters, so
// the body can run in either.
static inline GBufferOut illumiGBufferShade(
    VSOut in, bool frontFacing, float2 duvdx, float2 duvdy, float3 worldPosFwidth,
    const device Instance* instances, constant FrameUniforms& frame,
    const device float2* albedoUVScale, const device float2* nonColorUVScale,
    const device float4* albedoSliceMean, const device float4* nonColorSliceMean,
    InteriorMapParams imap,
    texture2d_array<float, access::sample> albedoAtlas,
    texture2d_array<float, access::sample> nonColorAtlas
) {
    Instance inst = instances[in.instanceID];
    float3 n = normalize(in.worldNormal);
    // Two-sided meshes (cull .none) rasterize back faces too; flip the normal
    // so they're lit as a real surface instead of going dark/hollow. For
    // single-sided meshes back faces are culled, so this is always front (no-op).
    if (!frontFacing) { n = -n; }

    constexpr sampler texSampler(filter::linear,
                                 mip_filter::linear,
                                 address::repeat,
                                 // S1.1 — 8x. The archviz camera's whole problem is the
                                 // grazing floor, where the footprint is a long thin
                                 // ellipse and trilinear picks the mip fitting its MAJOR
                                 // axis, over-blurring the minor one by the same ratio.
                                 // 8x recovers ~3 mip levels along the minor axis. Not
                                 // 16x: the hex blend already triples every atlas read.
                                 // Apple GPUs spend taps proportional to the ACTUAL
                                 // anisotropy, not the cap, so head-on surfaces pay
                                 // nothing.
                                 max_anisotropy(8));

    // S1.1 — ONE screen-space UV derivative pair for the whole fragment, taken here in
    // UNIFORM control flow from the interpolated uv BEFORE any hex-hash offset, fract()
    // wrap or detail-frequency scale. Every atlas read below hands these to gradient2d
    // rather than letting the texture unit infer a LOD from an offset uv. Taking them
    // inside the `if (inst.*TextureSlice >= 0)` branches instead would be undefined:
    // quad-mate lanes can be inactive there.
    // (`duvdx`/`duvdy` arrive as parameters — see `illumi_fs` and the visibility resolve.)

    WoodKnotSample knot;
    float2 matUV = illumiSurfaceMaterialUV(in, inst, frame, knot);
    // ── S2.4 — the detail band's diffuse-visible companion ────────────────────
    // Micro-occlusion sampled from the detail slice's BLUE channel and applied to the
    // albedo/roughness THIS fragment is about to write. Declared out here (not inside the
    // normal-map branch) so the value survives to the albedo and roughness blocks below.
    // 1.0 = fully open = the exact pre-S2.4 result; every path that doesn't bind a detail
    // slice leaves it at 1.0, and ×1.0 / +0.0 are IEEE no-ops.
    float detailOcc = 1.0;
    const bool wantsDetailRelief = (inst.detailOcclusionStrength > 0.0 ||
                                    inst.detailRoughnessStrength > 0.0);
    illumiSurfaceNormalMap(in, inst, frame, matUV, duvdx, duvdy, wantsDetailRelief, n, detailOcc,
                           nonColorUVScale, nonColorSliceMean, nonColorAtlas, texSampler);

    float3 albedo;
    float metallic, roughness;
    illumiSurfaceBase(in, inst, frame, matUV, duvdx, duvdy, albedo, metallic, roughness,
                      albedoUVScale, nonColorUVScale, albedoSliceMean, nonColorSliceMean,
                      albedoAtlas, nonColorAtlas, texSampler);
    illumiSurfaceDetail(in, inst, worldPosFwidth, knot, wantsDetailRelief, detailOcc, n, albedo, roughness);
    illumiSurfaceGround(in, inst, n, albedo, roughness);
    illumiSurfaceBark(in, inst, n, albedo, roughness);
    illumiSurfaceSkins(in, n, roughness);

    return illumiGBufferPack(in, inst, frame, imap, n, albedo, metallic, roughness, matUV, duvdx, duvdy,
                             albedoUVScale, nonColorUVScale, albedoSliceMean, nonColorSliceMean,
                             albedoAtlas, nonColorAtlas, texSampler);
}
