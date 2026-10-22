// ── ILLUMINATORAMA — G-BUFFER SURFACE DETAIL (shared) ─────────────────────────────────────────
//
// What is drawn on top of the sampled surface: the detail band's micro-occlusion, per-instance
// roughness drift, world-space wear scuffs (DH-1275), the wood knot's own end grain and dish, and
// the carpet pile-lay tone bands (DH-0472). Split out of IlluminatoramaGBufferShade.h (Signack S4,
// Daydream DH-1461); body unchanged.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
using namespace metal;

static inline __attribute__((always_inline)) void illumiSurfaceDetail(
    VSOut in, Instance inst, float3 worldPosFwidth, WoodKnotSample knot,
    bool wantsDetailRelief, float detailOcc,
    thread float3& n, thread float3& albedo, thread float& roughness
) {
    // ── S2.4 detail-band relief: apply the micro-occlusion ────────────────────
    // Placed here because both terms land on quantities the G-buffer already writes —
    // albedo (color 0) and roughness (color 1.z). NOTHING new is carried to the lighting
    // pass. That was the design constraint: `normalRoughness.w` is already a
    // material-class tag (foliage / SSS band / plush / 1+anisotropy) and has no room, and
    // a fifth G-buffer target would cost every pixel of every frame for a close-range
    // effect. In a deferred renderer a micro-AO on albedo IS a diffuse-only term for a
    // dielectric (F0 is the fixed 0.04, not the albedo), which is exactly the lobe the
    // detail normal could not reach.
    //
    // `detailOcc` is 1.0 unless a detail slice was bound AND the instance opted in, so
    // this is an exact no-op for every existing scene.
    if (wantsDetailRelief) {
        // Micro-shadowing: the pits between pores/weave/granules lose part of their
        // hemisphere to their own rim.
        albedo *= mix(1.0, detailOcc, saturate(inst.detailOcclusionStrength));
        // Micro-cavity roughening: the same pits scatter more widely than the open
        // surface between them. Survives mip filtering (see the sample site above).
        roughness = saturate(roughness + inst.detailRoughnessStrength * (1.0 - detailOcc));
    }

    // Per-INSTANCE macro roughness drift — the MACRO tier's companion to `macroTint`: a
    // slightly more/less worn surface per object. 0 by default ⇒ an exact no-op.
    roughness = saturate(roughness + inst.macroRoughnessDelta);

    // ── DH-1275 — surface wear: world-space scuffs, HIGH FREQUENCY ONLY ────────────────────────
    // Danny has rejected decimetre tone blotching twice (it reads as grime, not material; DH-1204
    // I3), so this is thin streaks ~4 mm across and a few cm long, drawn in WORLD space (no tile
    // period to grid at) and band-limited by the pixel's own footprint: once a pixel spans more
    // than the streak width the term fades to ZERO (never to a mean), so it cannot alias into
    // moiré or low-frequency mottle at any distance. Walls and risers streak vertically (furniture
    // and shoulders), tops streak along x (things dragged across them); a second, shorter set at
    // another angle breaks the parallel look.
    if (inst.surfaceWear > 0.0f) {
        float fw = length(worldPosFwidth);                      // metres per pixel
        float fade = 1.0f - smoothstep(0.0012f, 0.0035f, fw);
        if (fade > 0.0f) {
            float3 wp = in.worldPos;
            float2 a, b;
            if (fabs(n.y) > 0.7f) { a = float2(wp.z / 0.004f, wp.x / 0.05f);  b = float2(wp.x / 0.003f, wp.z / 0.02f); }
            else { float h = dot(wp.xz, normalize(float2(-n.z, n.x) + 1e-5f));
                   a = float2(h / 0.004f, wp.y / 0.06f);  b = float2(wp.y / 0.003f, h / 0.025f); }
            float s1 = smoothstep(0.74f, 0.95f, wearValueNoise(a));
            float s2 = smoothstep(0.80f, 0.96f, wearValueNoise(b + 17.3f));
            float scuff = saturate(s1 + 0.7f * s2) * fade * saturate(inst.surfaceWear);
            albedo *= 1.0f - 0.11f * scuff;
            roughness = saturate(roughness + 0.18f * scuff);
        }
    }

    // ── World-space wood knots: the branch, drawn on top of the wood it grew through ──────
    // Placed after roughness so both channels are in hand, and expressed as a modification of
    // the SAMPLED wood rather than as a colour of its own — which is what lets one shader term
    // serve oak, cherry and maple without knowing which it is on.
    if (knot.eye > 0.0f || knot.rim > 0.0f || knot.crack > 0.0f) {
        float dark = saturate(inst.woodKnots.z);
        // The branch's own end grain: its light and dark wood, chosen between by the concentric
        // ring field. Mixing toward ONE flat brown is what makes a knot read as a sticker.
        float3 knotLight = albedo * (1.0f - dark * 0.55f);
        float3 knotDark  = albedo * (1.0f - dark);
        // A shade redder, as resin-dense heartwood is.
        float3 knotWood = mix(knotLight, knotDark, knot.rings) * float3(1.06f, 0.98f, 0.94f);
        albedo = mix(albedo, knotWood, knot.eye);
        albedo *= 1.0f - 0.34f * knot.pith;    // deepest at the pith
        albedo *= 1.0f - 0.45f * knot.rim;     // the bark line
        albedo *= 1.0f - 0.72f * knot.crack;   // the checks
        // Knot wood is denser end grain and takes less finish than the face around it; the
        // checks are raw split wood.
        roughness = saturate(roughness + 0.07f * knot.eye + 0.10f * knot.rim + 0.22f * knot.crack);
        // A sanded knot sits a hair below the softer wood around it, so the face dishes very
        // slightly toward the pith and the checks are real openings. The tangent frame is rebuilt
        // here rather than hoisted, because the normal-map block that owns it is skipped whenever
        // no normal slice is bound — and a wood floor with no normal map still has knots.
        float dish = 0.06f * knot.eye + 0.18f * knot.crack;
        if (dish > 0.0f && length_squared(in.worldTangent.xyz) > 1e-4f) {
            float3 kT = normalize(in.worldTangent.xyz);
            float3 kB = cross(n, kT) * in.worldTangent.w;
            n = normalize(n + normalize(kT * knot.radial.x + kB * knot.radial.y) * dish);
        }
    }

    // ── Carpet pile-lay tone bands (DH-0472) ──────────────────────────────────
    // A low-frequency achromatic value field on the UNWRAPPED uv — the metre-scale nap variation a
    // ~0.30 m carpet tile physically cannot carry (its in-tile drift repeats every 0.30 m and then
    // dissolves under the hex de-repeat, so at room distance the rug reads as one flat tone). Drawn
    // here, on `in.uv`, so it never repeats at the tile period — the wood-knot mechanism, one scale
    // up. `carpetMacro.x == 0` (the default, every other material) makes `sampleCarpetMacro` return
    // 1.0, so this is an exact no-op for every existing scene and surface.
    albedo *= sampleCarpetMacro(in.uv, inst.carpetMacro);
}
