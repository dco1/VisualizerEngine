// ── ILLUMINATORAMA — G-BUFFER VERTEX-ALPHA SKINS (shared) ─────────────────────────────────────
//
// Micro-normal skins selected by a vertex-colour alpha band: the sausage casing's wrinkle/pore
// normal + wet/dry roughness mottle (0.66–0.9) and the plush fuzz (0.5–0.6). No other mesh ships
// alpha in those bands. Split out of IlluminatoramaGBufferShade.h (Signack S4, Daydream DH-1461);
// body unchanged.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
using namespace metal;

static inline __attribute__((always_inline)) void illumiSurfaceSkins(VSOut in, thread float3& n, thread float& roughness) {
    // ── Procedural sausage-casing micro-detail (HotdogDropUltra) ──────────────
    // Gates on vertexColor.a ∈ (0.6, 0.9) — set to 0.75 by pbdTubeExpand.
    // All sausage-casing vertices share alpha=0.75; normal geometry is 1.0,
    // foliage is 0.0. This is a no-op for every other scene.
    //
    // PORT OF THE HotdogDrop+ SAUSAGE MATERIAL (HotdogPlusGeometry.swift):
    // Drop+ drives the frank with (a) a SUBTLE wrinkle/pore normal map
    // (octaves 8/24/64 over a 256px tile that wraps the ~2-unit circumference
    // ⇒ ~4/12/32 cycles per world unit, intensity 1.1 — skin texture, not
    // blisters) and (b) a low-frequency ROUGHNESS mottle map in [0.18, 0.78]
    // (octaves 4/10/28 per tile ⇒ ~2/5/14 c/u) so wet/dry patches catch
    // highlights independently — that spatial gloss variance, not albedo
    // mottle, is where the "just off the grill" glisten lives. Albedo stays
    // a solid per-instance colour, exactly like Drop+'s flat diffuse.
    // Drop+'s clearcoat (0.55 / cc-rough 0.18) has no Illuminatorama analog;
    // the 0.18 glossy floor of the roughness mottle approximates the glaze.
    // Lower bound raised 0.6 → 0.66 to leave the (0.60, 0.66] sub-band to the lamp
    // shade fabric (DH-0458), which must NOT pick up the sausage wet-glaze look.
    if (in.vertexColor.a > 0.66 && in.vertexColor.a < 0.9) {
        // Build a CONTINUOUS tangent frame from the surface normal.
        // Previous version used a hard branch on abs(n.y): as a horizontal frank's
        // normal swept around the tube it crossed the 0.9 threshold, snapping the
        // reference vector (0,1,0)→(1,0,0) and rotating the entire noise-input
        // space discontinuously → visible diagonal two-tone seam on the franks.
        // Fix: blend the two candidate reference axes smoothly on n.y so the
        // tangent frame transitions without a step-discontinuity.
        float yBlend    = smoothstep(0.75f, 0.95f, abs(n.y));
        float3 blendRef = normalize(float3(yBlend, 1.0f - yBlend, 0.0f));
        float3 casingTang = normalize(cross(n, blendRef));
        float3 casingBtan = cross(n, casingTang);

        float3 wp = in.worldPos;
        float tu = dot(wp, casingTang);
        float tv = dot(wp, casingBtan);
        // Wrinkle/pore normal — Drop+'s sausageNormal equivalent in world space.
        // Gentle: combined peak tilt ≈ 4-5°, reads as skin texture in specular.
        const float sn0 = 4.0f, sn1 = 12.0f, sn2 = 32.0f;
        const float ep = 0.04f;
        float w0   = soilValueNoise(float2(tu * sn0, tv * sn0));
        float w0dx = soilValueNoise(float2((tu + ep) * sn0, tv * sn0));
        float w0dy = soilValueNoise(float2(tu * sn0, (tv + ep) * sn0));
        float w1   = soilValueNoise(float2(tu * sn1, tv * sn1));
        float w1dx = soilValueNoise(float2((tu + ep) * sn1, tv * sn1));
        float w1dy = soilValueNoise(float2(tu * sn1, (tv + ep) * sn1));
        float w2   = soilValueNoise(float2(tu * sn2, tv * sn2));
        float w2dx = soilValueNoise(float2((tu + ep) * sn2, tv * sn2));
        float w2dy = soilValueNoise(float2(tu * sn2, (tv + ep) * sn2));
        // Drop+ octave weights 0.6 / 0.25 / 0.15 at SUBTLE amplitude. The
        // round-31 1.7× lift made the casing read lumpy under a matte lobe;
        // with the clearcoat now supplying a real glint, faint wrinkle is
        // enough — the coat highlight picks it up (round-31 review).
        float gU = (w0dx - w0) * 0.050f + (w1dx - w1) * 0.022f + (w2dx - w2) * 0.012f;
        float gV = (w0dy - w0) * 0.050f + (w1dy - w1) * 0.022f + (w2dy - w2) * 0.012f;
        n = normalize(n + casingTang * gU + casingBtan * gV);

        // Roughness mottle — Drop+'s sausageRoughness map verbatim: big wet/dry
        // blotches, weights 0.65 / 0.25 / 0.10, range [0.18, 0.78], mean ≈ 0.48.
        // This is the SATIN base layer; the tight glint lives in the clearcoat
        // lobe the deferred lighting pass adds for casing-flagged pixels (see
        // illumi_lighting — flag 0.75 in normalRoughness.w). Folding the glaze
        // into this one lobe instead provably oscillates: [0.16, 0.62] read as
        // lacquered silicone (round 30), [0.30, 0.62] as foam rubber (round 31).
        // Overrides the per-instance scalar for casing pixels only.
        const float sr0 = 2.0f, sr1 = 5.0f, sr2 = 14.0f;
        float rN = soilValueNoise(float2(tu * sr0, tv * sr0)) * 0.65f
                 + soilValueNoise(float2(tu * sr1, tv * sr1)) * 0.25f
                 + soilValueNoise(float2(tu * sr2, tv * sr2)) * 0.10f;
        roughness = 0.18f + saturate(rN) * 0.60f;
    }

    // ── Procedural plush fuzz micro-normal (Teddy Bear Press) ─────────────────
    // Gates on vertexColor.a ∈ [0.5, 0.6] (≈0.55 from TeddyBearGeometry). A fine,
    // high-frequency normal break-up so the silhouette + specular read as short
    // fur fuzz rather than smooth rubber, and the per-instance high roughness stays
    // matte. No-op for every other scene (no other mesh ships alpha in this band).
    if (in.vertexColor.a >= 0.5f && in.vertexColor.a <= 0.6f) {
        float yBlend    = smoothstep(0.75f, 0.95f, abs(n.y));
        float3 blendRef = normalize(float3(yBlend, 1.0f - yBlend, 0.0f));
        float3 fuzzTang = normalize(cross(n, blendRef));
        float3 fuzzBtan = cross(n, fuzzTang);
        float3 wp = in.worldPos;
        float tu = dot(wp, fuzzTang);
        float tv = dot(wp, fuzzBtan);
        // High-frequency fuzz (≈ 60 / 150 cycles per world unit) at small amplitude
        // — a soft micro-tilt, not blisters. The matte roughness scatters it into
        // the velvety sheen the deferred plush lobe rim-lights.
        // Two scales of fur fuzz: a coarse tuft (~28 c/u, reads at the camera distance
        // the bears are seen from) + a fine fibre (~90 c/u). Amplitudes lifted so the
        // matte sheen visibly breaks up the silhouette instead of staying glassy-smooth.
        const float fn0 = 28.0f, fn1 = 90.0f, ep = 0.012f;
        float f0   = soilValueNoise(float2(tu * fn0, tv * fn0));
        float f0dx = soilValueNoise(float2((tu + ep) * fn0, tv * fn0));
        float f0dy = soilValueNoise(float2(tu * fn0, (tv + ep) * fn0));
        float f1   = soilValueNoise(float2(tu * fn1, tv * fn1));
        float f1dx = soilValueNoise(float2((tu + ep) * fn1, tv * fn1));
        float f1dy = soilValueNoise(float2(tu * fn1, (tv + ep) * fn1));
        float gU = (f0dx - f0) * 0.11f + (f1dx - f1) * 0.05f;
        float gV = (f0dy - f0) * 0.11f + (f1dy - f1) * 0.05f;
        n = normalize(n + fuzzTang * gU + fuzzBtan * gV);
    }
}
