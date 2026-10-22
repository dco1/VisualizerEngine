// ── ILLUMINATORAMA — G-BUFFER PROCEDURAL GROUND (shared) ──────────────────────────────────────
//
// The soil-marker ground material (#58): per-region roughness + wetness from the packed uv,
// world-space micro relief, grass-base mottle and the forest-floor duff. Gated to untextured
// instances whose uv.x carries the negative soil marker — an exact no-op elsewhere. Split out of
// IlluminatoramaGBufferShade.h (Signack S4, Daydream DH-1461); body unchanged.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
using namespace metal;

static inline __attribute__((always_inline)) void illumiSurfaceGround(VSOut in, Instance inst,
                                       thread float3& n, thread float3& albedo, thread float& roughness) {
    // ── Procedural soil material (#58 #11/#12/#13) ───────────────────────────
    // Ground vertices pack soil data into uv as a NEGATIVE-x marker
    // (uv.x = -(roughness + 0.01), uv.y = wetness). Real UVs are never negative
    // and the branch is gated to UNTEXTURED instances, so this is an exact no-op
    // for every other surface and every other scene. Delivers per-region
    // roughness (#12), wetness darkening/smoothing (#12 — wet = darker, smoother,
    // more reflective), and world-space macro/micro normal relief (#13, #11 —
    // world-space so no UV stretching on slopes).
    if (inst.albedoTextureSlice < 0 && inst.roughnessTextureSlice < 0 &&
        inst.normalTextureSlice < 0 && in.uv.x < -0.005) {
        float soilRough = saturate(-in.uv.x - 0.01);
        float wet = saturate(in.uv.y);
        roughness = mix(soilRough, soilRough * 0.45, wet);   // wet → smoother + reflective
        albedo *= mix(1.0, 0.82, wet);                       // gentle extra darken (vertex colour already cools hollows)
        float2 grad = soilNormalGrad(in.worldPos.xz);
        // Dry ground takes more micro-relief; wet (pooled) ground reads smoother.
        float relief = 0.32 * (1.0 - 0.55 * wet);
        n = normalize(n + float3(grad.x, 0.0, grad.y) * relief);
        // ── GRASS-BASE MOTTLE (round-5 polish): a 20–40 cm value/hue mottle on the
        // ground ALBEDO everywhere (clearing + far) so the mossy floor between blades
        // stops reading as one flat green. Two octaves: ~33 cm patches + ~17 cm. Pulls
        // value ±18 % and shifts a fraction toward yellow-green / damp-dark so the
        // sward base has patchy life under the blades. No-op off the soil marker.
        {
            float2 gp = in.worldPos.xz;
            float mo0 = soilValueNoise(float2(gp.x * 3.0, gp.y * 3.0));   // ≈33 cm
            float mo1 = soilValueNoise(float2(gp.x * 6.0 + 17.0, gp.y * 6.0 - 9.0)); // ≈17 cm
            float mottleV = (mo0 * 0.65 + mo1 * 0.35);
            albedo *= 0.84 + 0.34 * mottleV;                             // value patches
            float3 damp = albedo * float3(0.78, 0.92, 0.70);             // damp-dark green
            float3 dryG = albedo * float3(1.12, 1.08, 0.74);             // yellow-green
            albedo = mix(albedo, damp, smoothstep(0.62, 0.30, mo0) * 0.35);
            albedo = mix(albedo, dryG, smoothstep(0.66, 0.88, mo1) * 0.30);
        }
        // ── FOREST-FLOOR DUFF (round 3, target 4): per-PIXEL leaf-litter ─────────
        // The vertex-baked duff (ForestGeometry.groundColor) can only carry ≈0.6 m
        // features (192-res / 120 m grid), so beyond the grass ring the floor read as
        // FLAT GREEN FELT at the hero/side cameras. Add per-pixel litter HERE where it
        // survives distance: BIG (≈8–15 cm) high-contrast tan/brown/dull-orange oval
        // blotches (x-stretched domain ⇒ leaf-shaped, not round dots), gated to the
        // out-of-clearing band (radius > ≈12 m from the clearing centre at z=−3) so
        // the mossy clearing floor stays clean. World-space → no swim, no UV stretch.
        float2 fp = in.worldPos.xz;
        float rad = length(float2(fp.x, fp.y + 3.0));
        // ROUND-5 floor fix: the band between the grass ring and the treeline
        // (≈8–16 m) read as FLAT SATURATED GREEN — the litter started too far out
        // (10→18 m) and was too weak to survive against the bright mossy-green
        // vertex base there. Start the duff sooner (the mossy CLEARING CENTRE at
        // z=−3 stays clean because the clearing radius is ≈9 m and the litter only
        // reaches full past ≈14 m), and make it strong enough to read.
        // ROUND-8: the 8→14 ramp left a HALF-treated annulus (litter at 20–60%)
        // exactly where the hero camera sees substrate past the grass ring, and it
        // still read as green felt (the top view proves duff fully lands past
        // 14 m). Tighten the ramp so full duff arrives where the dense sward ends.
        float farBand = smoothstep(7.0, 10.0, rad);    // 0 in clearing → 1 past 10 m
        if (farBand > 0.001) {
            // First, DESATURATE the bright-green vertex base toward a duff floor so
            // the litter has a neutral ground to sit on (the green felt was the base
            // colour, not the litter). Pull the saturated green toward a dull
            // olive-brown forest-floor tone, scaled by the band.
            float baseLum = dot(albedo, float3(0.299, 0.587, 0.114));
            // ROUND-8b: the old near-constant duffBase (0.07–0.10 regardless of
            // incoming value) compressed the vertex-baked MACRO patches (2–6 m
            // damp/dry swings — the only floor variation that survives 25 m +
            // atmosphere) down to ~22% of their range. Preserve incoming
            // LUMINANCE and shift only the HUE toward duff-brown, so the macro
            // value patchiness rides through the desaturation.
            float3 duffBase = float3(0.118, 0.096, 0.056) * saturate(baseLum * 3.0);
            albedo = mix(albedo, duffBase, farBand * 0.78);
            // Oval litter mask: x-stretched noise domain so blotches read as leaves.
            // Wider duty (0.42–0.78) so litter blotches are COMMON, not sparse.
            float lit = soilValueNoise(float2(fp.x * 9.0, fp.y * 4.5));   // ≈11/22 cm
            float lit2 = soilValueNoise(float2(fp.x * 3.2 + 50.0, fp.y * 3.2 - 20.0)); // coarse drifts
            float litMask = smoothstep(0.42, 0.74, lit) * (0.55 + 0.45 * lit2) * farBand;
            float litHue = soilValueNoise(float2(fp.x * 5.0 + 13.0, fp.y * 5.0 + 7.0));
            float3 litTan    = float3(0.170, 0.120, 0.058);
            float3 litOrange = float3(0.205, 0.105, 0.034);
            float3 litBrown  = float3(0.085, 0.058, 0.032);
            float3 litCol = (litHue < 0.4)
                ? mix(litBrown, litTan, litHue / 0.4)
                : mix(litTan, litOrange, (litHue - 0.4) / 0.6);
            albedo = mix(albedo, litCol, litMask * 0.80);
            // A few surviving green moss patches break the brown (not pure duff).
            float moss = smoothstep(0.74, 0.90, lit2) * farBand;
            albedo = mix(albedo, float3(0.070, 0.115, 0.045), moss * 0.5);
            // Twig/duff micro-normal: a coarser bump on the littered patches.
            float2 lg = soilNormalGrad(fp * 6.0);
            n = normalize(n + float3(lg.x, 0.0, lg.y) * (0.45 * litMask));
            roughness = mix(roughness, 0.92, litMask);
        }
    }
}
