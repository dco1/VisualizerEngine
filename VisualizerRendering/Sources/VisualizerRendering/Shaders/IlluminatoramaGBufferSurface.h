// ── ILLUMINATORAMA — G-BUFFER SURFACE SAMPLING (shared) ───────────────────────────────────────
//
// The texture-driven part of the material evaluation: the material uv (wood-knot flow warp,
// per-instance / per-panel phase, animated domain warp), the tangent-space normal map + detail
// band, and the base albedo / metallic / roughness reads with their per-cell, per-vertex and
// per-instance modulations (leaf venation, hue cycle, macro tone). Split out of
// IlluminatoramaGBufferShade.h (Signack S4, Daydream DH-1461); bodies unchanged.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
using namespace metal;

// ── Leaf venation (DH-0947) ─────────────────────────────────────────────────
// A houseplant leaf's pale lateral veins are lines a millimetre or two across on a blade a
// quarter-metre long — far finer than any vertex grid can paint — so they are drawn here, per
// pixel, from the leaf's own FLAT blade coordinates (x across, y along, in blade lengths), which
// the host packs into `uv` as (10.5 + x, y) (VisualizerVegetation `LeafSheet.bladeUVOrigin`).
// Pattern 1, the fiddle-leaf fig: eight pairs of laterals leaving the midrib at ~60° and bending
// toward the tip as they near the margin (the loop of a brochidodromous leaf), tapering outward
// and fading before the edge. Each vein sits a little off the even ladder — differently on the
// two sides — because an evenly ruled set reads as a ribbed banana leaf, not a fig. Returns 0…1,
// 1 on a vein's centre line. A vein narrower than a pixel is widened to one and dimmed in
// proportion, so a distant leaf fades to its mean tone instead of shimmering.
static inline float leafVeinMask(float x, float y, float2 duvdx, float2 duvdy) {
    const float pairs = 8.0f, from = 0.12f, to = 0.84f;
    const float cotA = 0.58f;        // rise along the midrib per unit out (cot 60°)
    const float sweep = 1.3f;        // the extra rise toward the margin — the loop toward the tip
    float ax = fabs(x);
    float side = x < 0.0f ? 7.7f : 3.1f;
    float rise = ax * cotA + sweep * ax * ax;
    float spacing = (to - from) / (pairs - 1.0f);
    float k0 = clamp(round((y - rise - from) / spacing), 0.0f, pairs - 1.0f);
    float d = 1e3f;
    for (float k = max(k0 - 1.0f, 0.0f); k <= min(k0 + 1.0f, pairs - 1.0f); k += 1.0f) {
        float jitter = fract(sin((k + 1.0f) * 12.9898f + side) * 43758.5453f) - 0.5f;
        float yk = from + (k + 0.34f * jitter) * spacing;
        d = min(d, fabs((y - rise) - yk));
    }
    float slope = cotA + 2.0f * sweep * ax;
    d /= sqrt(1.0f + slope * slope);                            // perpendicular to the vein
    float w = mix(0.0048f, 0.0026f, saturate(ax / 0.34f));      // half-width, tapering outward
    float px = max(length(float2(duvdx.x, duvdy.x)), length(float2(duvdx.y, duvdy.y)));
    float we = max(w, px);                                       // never under a pixel wide
    float line = 1.0f - smoothstep(0.0f, we, d);
    return line * (w / we)                                       // a widened line is dimmer
         * smoothstep(0.016f, 0.045f, ax)                        // grows out of the painted rib
         * (1.0f - smoothstep(0.22f, 0.36f, ax));                // and fades before the margin
}

/// The uv every material sample reads (`in.uv` + knot warp + phases + domain warp); `knot` is the
/// wood-knot field the detail stage draws on top.
static inline __attribute__((always_inline)) float2 illumiSurfaceMaterialUV(VSOut in, Instance inst, constant FrameUniforms& frame,
                                              thread WoodKnotSample& knot) {
    // ── World-space wood knots ────────────────────────────────────────────────
    // `matUV` is the uv every MATERIAL sample below reads, and it is `in.uv` displaced by the
    // knot field's flow-around warp — which is what makes the grain, the pores and the ring
    // figure sweep around a knot instead of running straight through it. With no knots declared
    // (`woodKnots.x == 0`, the default and every other material) the warp is exactly zero and
    // `matUV == in.uv`, so this is byte-identical for every existing scene.
    //
    // `in.uv` itself is kept for the things that must NOT move with the warp: the plank-cell
    // hash (a board index cannot slide), the soil marker, and the derivatives.
    //
    // DH-0475 — `inst.uvPhase` is a per-INSTANCE translation (tile-UV units) folded into the SAME
    // `matUV` seam: two pieces that share one baked slice and one mesh sample the tiling material
    // at different phases, so identical wood twins don't show the same grain figure in the same
    // place. `float2(0)` — the default and every non-opting instance — leaves `matUV == in.uv`.
    // It rides on `matUV` (not `in.uv`), so it moves the figure without sliding the plank-cell
    // grid or the soil marker; the derivatives above are taken from the un-phased `in.uv`, which
    // is correct because a constant offset has zero screen-space derivative.
    knot = sampleWoodKnots(in.uv, inst.woodKnots, inst.patternCells);
    // DH-0513 — per-panel FIGURE phase: `antiTilingScale < 0` flags a coherent wall veneer (the hex
    // blend treats any strength ≤ 0 as the exact single read, so the sign is free to carry this).
    // Each physical panel — `floor(uv.x · patternCells.x)` — samples the seamless slice shifted along
    // the grain by a hashed fraction of the tile, so the cathedral arch lands on a different stretch
    // of the figure in every panel instead of printing at the panel pitch. A deliberate grain
    // discontinuity at each panel edge, like a real joint. Every other instance adds exactly zero.
    float2 panelPhase = float2(0.0f);
    if (inst.antiTilingScale < 0.0f && inst.patternJitter != 0.0f && inst.patternCells.x >= 1.0f) {
        panelPhase.y = panelFigureShift(floor(in.uv.x * inst.patternCells.x));
    }
    float2 matUV = in.uv + knot.warp + inst.uvPhase + panelPhase;
    // `uvWarp.w >= 0.5` extends the animated domain warp to the MATERIAL maps
    // (albedo/roughness/normal) as well as emission. Default 0 = emission only,
    // which is the common case: a glow that morphs on a surface that does not.
    if (inst.uvWarp.w >= 0.5f) matUV = warpUV(matUV, inst.uvWarp, frame.time);
    return matUV;
}

/// Tangent-space normal map (+ the detail normal band and its micro-occlusion tap) onto `n`.
static inline __attribute__((always_inline)) void illumiSurfaceNormalMap(
    VSOut in, Instance inst, constant FrameUniforms& frame, float2 matUV, float2 duvdx, float2 duvdy,
    bool wantsDetailRelief, thread float3& n, thread float& detailOcc,
    const device float2* nonColorUVScale, const device float4* nonColorSliceMean,
    texture2d_array<float, access::sample> nonColorAtlas, sampler texSampler
) {
    // Phase 4.5 — tangent-space normal-map sampling. The atlas is the
    // same `bgra8Unorm` non-colour atlas as metallic/roughness; the
    // normal map encodes tangent-space (x,y,z) into [0,1] via
    // `0.5*(N+1)`. Decode with `2*sample - 1`, then transform into
    // world via TBN. When no normal map is bound, fall through to the
    // geometric normal `n`.

    if (inst.normalTextureSlice >= 0 &&
        length_squared(in.worldTangent.xyz) > 1e-4) {
        float3 T = normalize(in.worldTangent.xyz);
        float3 B = cross(n, T) * in.worldTangent.w;
        // Hex-stochastic normal map — eliminates the regular tiling pattern
        // on large flat surfaces (walls, floors) by blending three stochastically-
        // offset lattice samples. The blend is linear in encoded space (before
        // decode), which slightly overshoots tangent-space magnitude at boundaries
        // but gives correct appearance after renormalize below.
        float4 nmSample = sampleAtlasHex(nonColorAtlas, texSampler, matUV,
                                         uint(inst.normalTextureSlice), nonColorUVScale,
                                         nonColorSliceMean,
                                         frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
        float3 tangentN = normalize(nmSample.xyz * 2.0 - 1.0);
        // Phase 7 detail normal — blended on top of the macro normal at
        // higher UV frequency (pores, weave, grain). Uses overlay-normal
        // blend: partial-derivative add in tangent space, then renormalize.
        if (inst.detailNormalTextureSlice >= 0) {
            float2 detailUV = matUV * inst.detailNormalUVScale;
            // S1.1 — the detail band is handed the UNSCALED gradient pair ON PURPOSE, and
            // this is a deliberate, measured exception to the rule the rest of this shader
            // follows.
            //
            // The mip-correct thing is to scale by `detailNormalUVScale` (8×) to match the
            // frequency actually being sampled. I did that first, and it BROKE the band:
            // `testDetailNormalAddsGrazingMicroRelief` collapsed from ~1.4× to 1.0002× on
            // aluminum and 0.97× on brushed-nickel — i.e. detail normals stopped doing
            // anything at all. Correct LOD for an 8×-frequency band means selecting a mip
            // ~3 levels coarser, which averages the pore/weave/grain perturbation to flat.
            // The band exists precisely to survive there, and per the 2026-08-02
            // whole-library A/B it is the ONE thing that measurably moves metals.
            //
            // Passing the unscaled pair makes this band sample about as sharply as it did
            // when the atlas had no mip chain at all — so it is the STATUS QUO for detail
            // normals, not a new under-filtering defect, while every other atlas read gets
            // proper mips. The honest cost is that this band can still alias at grazing
            // angles.
            //
            // S2.4 UPDATE (2026-08-06): that companion now exists (see just below), and it
            // does NOT retire this exception — it explains it. The companion survives
            // filtering by running at a resolvable frequency; this band does not, because at
            // 8x on a 2 m macro tile a texel is ~0.5 mm. So what "correct LOD kills the
            // effect" actually means here is that **this band's contribution is sub-pixel
            // aliasing** — which is also why it reads on mirror-like metal and on nothing
            // else. Retiring the exception means moving the detail NORMAL to a resolvable
            // frequency too, which is a look change, not a filtering fix.
            float4 dnSample = sampleAtlasAspect(nonColorAtlas, texSampler, detailUV,
                                                uint(inst.detailNormalTextureSlice), nonColorUVScale,
                                                duvdx, duvdy);
            float3 dn = dnSample.xyz * 2.0 - 1.0;
            // NOTE: `dn.z` is deliberately not used — the blend is a partial-derivative add
            // in tangent space, so blue was always dead weight here. S2.4 puts the detail
            // occlusion in it (see below).
            tangentN = normalize(float3(tangentN.xy + dn.xy, tangentN.z));

            // S2.4 — the occlusion is a SECOND tap of the same slice, at its own frequency
            // and with MIP-CORRECT gradients. Both differences are load-bearing:
            //
            //  * Mip-correct, because occlusion is a non-negative quantity: a coarser mip
            //    averages it to its MEAN, so it degrades into a slight uniform darkening
            //    rather than vanishing (the normal's failure mode) or aliasing (the
            //    normal's cost — and the whole reason S1.1 had to hand THIS band unscaled
            //    gradients). Reading it off `dnSample` would inherit that defect.
            //  * Its own frequency, because mip-correct sampling at the NORMAL's 8× shows
            //    nothing: measured, that band is ~0.5 mm per texel on a 2 m macro tile, so
            //    correct filtering is doing exactly its job when it returns a flat mean.
            //    Occlusion has to run where the camera can resolve it to read as relief.
            if (wantsDetailRelief) {
                float occScale = inst.detailOcclusionUVScale > 0.0
                               ? inst.detailOcclusionUVScale : inst.detailNormalUVScale;
                float2 occUV = matUV * occScale;
                float4 occSample = sampleAtlasAspect(nonColorAtlas, texSampler, occUV,
                                                     uint(inst.detailNormalTextureSlice), nonColorUVScale,
                                                     duvdx * occScale, duvdy * occScale);
                detailOcc = occSample.z;   // [0,1], 1 = open. 255-blue slices ⇒ exactly 1.
            }
        }
        // World normal = T*x + B*y + N*z. Equivalent to a TBN matrix
        // multiply but cheaper to spell out as a single dot per axis.
        n = normalize(T * tangentN.x + B * tangentN.y + n * tangentN.z);
    }
}

/// Base albedo (atlas, pattern-cell jitter, vertex colour, venation, hue cycle, macro tone),
/// metallic and roughness.
static inline __attribute__((always_inline)) void illumiSurfaceBase(
    VSOut in, Instance inst, constant FrameUniforms& frame, float2 matUV, float2 duvdx, float2 duvdy,
    thread float3& albedo, thread float& metallic, thread float& roughness,
    const device float2* albedoUVScale, const device float2* nonColorUVScale,
    const device float4* albedoSliceMean, const device float4* nonColorSliceMean,
    texture2d_array<float, access::sample> albedoAtlas,
    texture2d_array<float, access::sample> nonColorAtlas, sampler texSampler
) {
    albedo = inst.albedo;
    if (inst.albedoTextureSlice >= 0) {
        // Phase 7 hex-stochastic anti-tiling: floor/wall surfaces tile seamlessly but
        // don't repeat visibly — three blended samples from stochastically-offset
        // lattice cells. `in.uv` is already in tile-space (world ÷ uvMetres), so
        // the hex cell size equals one texture tile. sampleAtlasAspect handles
        // letterbox padding per-slice; the offset just shifts into a neighbouring
        // infinitely-tiling region, so aspect is preserved.
        float4 tx = sampleAtlasHex(albedoAtlas, texSampler, matUV,
                                    uint(inst.albedoTextureSlice), albedoUVScale,
                                    albedoSliceMean,
                                    frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
        albedo = tx.rgb;
        // S2.5 half 2 — per-PATTERN-CELL value jitter, the COHERENT categories' de-repeat
        // (full contract on `Instance.patternCells` in IlluminatoramaCommon.h). The hex
        // blend above is invalid for a tile grid or a plank run (it superimposes misaligned
        // copies), so those materials ship antiTilingScale = 0 and, until this line, had no
        // de-repeat at all. `in.uv` counts up across the whole surface — only the atlas
        // LOOKUP wraps — so `floor(uv * patternCells)` indexes each physical tile / plank /
        // course uniquely over the entire floor, and a ± tone multiplier hashed from it
        // breaks the macro repeat with no UV displacement (grout and seams stay put) and no
        // extra texture tap. A bonded axis ships patternCells = 0 there: floor(uv·0) = 0
        // keeps that axis's index constant, so a running-bond COURSE varies as a whole and
        // a cell boundary never cuts through an offset tile mid-body.
        // `patternJitter == 0` — the default, and every scene that never opts in — is an
        // exact no-op, so Visualizer is byte-identical.
        if (inst.patternJitter != 0.0f) {
            albedo *= 1.0f + inst.patternJitter * patternCellHash(patternCellIndex(in.uv, inst.patternCells));
        }
    }
    // Phase 4.17 — modulate albedo by per-vertex color (default white,
    // so a no-op for meshes that ship no .color source). This is where
    // pattern detail painted into the vertex stream (chevron stripes,
    // candy gradients, anything fed via SCNGeometrySource(semantic:
    // .color)) actually paints the fragment. SceneKit convention is
    // multiplicative: vertex colour and material diffuse compose.
    albedo *= in.vertexColor.rgb;
    // DH-0947 — the leaf's own pale lateral veins, drawn on top of its painted colour (the rib
    // band, the underside and the leaf's age are already in the vertex paint). `leafVenation == 0`
    // — every instance that never sets it — skips this entirely; the uv RANGE check keeps the
    // pattern off the plant's petioles and anything else in the group that carries no blade
    // coordinates.
    if (inst.leafVenation == 1 && in.uv.x >= 10.0f && in.uv.x < 11.0f) {
        float vein = leafVeinMask(in.uv.x - 10.5f, in.uv.y, duvdx, duvdy);
        albedo = mix(albedo, albedo * 1.5f + float3(0.018f, 0.022f, 0.0f), vein * 0.6f);
    }
    // Animated hue cycle on host-TAGGED vertices (Instance.hueCycle; tangent.w > 0.5 is the tag).
    // Rodrigues rotation about the grey axis: hue turns, brightness and saturation hold.
    // Tag = tangent.w ≈ 1 exactly (0.5…1.5): a mesh's own tangent handedness (±1 on normal-mapped
    // geometry) or other packed codes (forest species 2–5) must not be read as the tag.
    if (inst.hueCycle.x > 0.0f && in.worldTangent.w > 0.5f && in.worldTangent.w < 1.5f) {
        float theta = inst.hueCycle.y + frame.time * inst.hueCycle.z
                    + dot(in.worldPos.xz, float2(0.6f, 0.8f)) * inst.hueCycle.w;
        const float3 k = float3(0.57735027f);
        float c = cos(theta), s = sin(theta);
        float3 rotated = albedo * c + cross(k, albedo) * s + k * dot(k, albedo) * (1.0f - c);
        albedo = mix(albedo, max(rotated, 0.0f), saturate(inst.hueCycle.x));
    }
    // Per-INSTANCE macro tone (the MACRO tier — a slight achromatic per-object lighten/darken so
    // two pieces wearing the same material id don't read as the same pixels twice). Identity (1)
    // by default ⇒ an exact no-op for every instance and host that never opts in.
    albedo *= inst.macroTone;

    metallic = inst.metallic;
    if (inst.metallicTextureSlice >= 0) {
        // SCN-style packed metallic-roughness usually puts metallic in B
        // (Blue) and roughness in G — but the exact channel depends on
        // how the asset was authored. Falling back to R keeps single-
        // channel grayscale metallic maps working without per-scene wiring.
        float4 tx = sampleAtlasAspect(nonColorAtlas, texSampler, matUV,
                                      uint(inst.metallicTextureSlice), nonColorUVScale, duvdx, duvdy);
        metallic = tx.r;
    }

    roughness = inst.roughness;
    if (inst.roughnessTextureSlice >= 0) {
        // Hex-stochastic roughness — same three-cell blend as albedo so
        // roughness variation doesn't lag the albedo tile seam.
        float4 tx = sampleAtlasHex(nonColorAtlas, texSampler, matUV,
                                    uint(inst.roughnessTextureSlice), nonColorUVScale,
                                    nonColorSliceMean,
                                    frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
        roughness = tx.g;
    }
}
