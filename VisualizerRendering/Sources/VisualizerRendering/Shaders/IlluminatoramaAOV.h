// IlluminatoramaAOV.h — THE per-term output set (AOVs), defined once for every lane.
//
// Daydream DH-1304. One named set of lighting terms that the deferred lane (`illumi_lighting`)
// and the traced lanes (`illumi_rt_lighting_tlas`: the one-bounce still and the path-traced
// Maximum) all emit under the SAME names, units and exposure, so a Debug Tester can show
// LIVE | MAXIMUM | ERROR per term and a gate can say WHICH term moved instead of "the picture
// differs". The Swift mirror is `IlluminatoramaAOV` (Illuminatorama/IlluminatoramaAOV.swift);
// `IlluminatoramaAOVTests` fails if the two drift. DH-1017 (professional AOV export) is meant
// to export this set unchanged — add a slice here, mirror it there, never rename a slice.
//
// ── Contract ─────────────────────────────────────────────────────────────────────────────
//  • UNITS / EXPOSURE: scene-referred LINEAR radiance in the renderer's HDR working units —
//    exactly the value `outHDR` + `rtDiffuse` carry into the TAA / post chain — BEFORE the
//    camera exposure, white balance, bloom, the display transform and every grade. Both lanes
//    already share those units, which is what makes the terms comparable.
//  • SUM: on every frame, over every pixel, Σ(lighting slices 0..kAOVLightingCount−1) == `beauty`
//    to float tolerance, where `beauty` is the lit HDR the two lighting kernels produced for that
//    pixel (deferred composite + the traced pass's raw diffuse + its reflection delta). That is
//    the gate (`HouseRenderBridgeGPUTests_AOV`). The sum deliberately EXCLUDES the passes after
//    lighting — the glass draw, volumetric haze, caustics, clouds, particles — which are not
//    terms of a surface's lighting; `transmission` is reserved for the glass lane (DH-1304 inc. 3+).
//  • ACCUMULATION: the capture is a per-pixel running mean over the frames since the last reset
//    (`aovBlend` = 1/N, N counted by the host; it restarts whenever the GI running mean does, so
//    a still's held-out warm-up and cache-build frames stay out of it). A mean is linear, so the
//    SUM contract holds for the accumulated set as it does for one frame. It is NOT the exported
//    pixel: the export's denoise (SVGF) and the still resolve are non-linear filters of the same
//    frames, and the tonemap sits after both.
//  • ATTRIBUTION (what goes where):
//      direct       light from the scene's LIGHTS (sun, lamps, cans, sconces, light strips, fill
//                   directionals) arriving at the visible surface with no bounce — its diffuse AND
//                   its specular lobe (highlights), the thin-leaf transmission of the sun, the
//                   clearcoat's sun lobe. Path lane: the traced soft-shadowed sun at the primary.
//      indirect     light that bounced off GEOMETRY at least once before the visible surface,
//                   diffuse lobe: the path lane's vertices past the first (sun / lamps / windows
//                   seen from a bounce, emitters hit along the path, radiance-cache reads); the
//                   one-bounce lane's hits; the live lane's stand-ins for the same (interior
//                   irradiance bands, the ambient supplement, DDGI, the cloth sheen's environment
//                   arm).
//      emission     the visible surface's OWN emission (G-buffer). An emitter seen through a
//                   bounce is `indirect`; an emitter seen in a reflection is `specular`.
//      environment  SKY / environment light arriving at the visible surface with no bounce off
//                   geometry, diffuse lobe — the outdoor irradiance cube's share, a window
//                   aperture sampled from the primary, the sky the primary's first segment finds —
//                   and the sky itself on sky pixels (the background).
//      specular     the visible surface's reflection of its ENVIRONMENT and of the scene — the
//                   specular IBL, the clearcoat / coat IBL lobes, the traced reflections (with
//                   the sky share they replaced taken out). Direct-light highlights are `direct`.
//      transmission reserved: light refracted THROUGH glass (the RT glass lane). Always 0 in
//                   DH-1304 increment 1 — the glass pass is after the lighting kernels.
//      handoff      BOOKKEEPING, not a light: the stand-in share the deferred pass emitted for a
//                   traced pass to REPLACE (`diffSkyOut`), minus what the traced pass took out.
//                   Exactly 0 on every frame where the traced pass ran; non-zero only where a
//                   share was offered and nobody consumed it (a warm-up frame with no TLAS). Kept
//                   as a slice so the SUM contract holds on such a frame instead of hiding the
//                   discrepancy inside `environment`. A finished still reads ≈ 0 here.
//      beauty       the lit HDR (the sum), captured by the same code path so the gate measures
//                   the tagging, not a readback of another texture.
//  • GEOMETRY / MATERIAL AOVs (albedo, normal, depth, roughness, metalness) are NOT slices here:
//    they are the G-buffer targets the lighting already reads, exposed through the Swift enum's
//    `storage` as `.gbuffer` so a consumer reads them from the same renderer frame.
//  • COST: every write is behind a FUNCTION CONSTANT (`kLightingAOVEnabled` 8, `kRTAOVEnabled`
//    31), so the default pipeline variants are the pre-DH-1304 code — no branch, no extra
//    register, no binding. A host pays only while `aovCaptureEnabled`.
#ifndef ILLUMINATORAMA_AOV_H
#define ILLUMINATORAMA_AOV_H

#define kAOVDirect        0u
#define kAOVIndirect      1u
#define kAOVEmission      2u
#define kAOVEnvironment   3u
#define kAOVSpecular      4u
#define kAOVTransmission  5u
#define kAOVHandoff       6u
/// Slices 0..kAOVLightingCount−1 are the lighting terms whose sum is `beauty`.
#define kAOVLightingCount 7u
#define kAOVBeauty        7u
#define kAOVCount         8u

/// The per-frame capture parameters (one `setBytes`, bound only in the AOV variants).
struct AOVUniforms {
    /// 1/N — the running mean's weight for this frame (1 on the first frame after a reset).
    float blend;
    /// 1 ⇒ the deferred pass hands its stand-in diffuse share (`diffSkyOut`) to a traced pass
    /// this frame: that share goes to `kAOVHandoff` instead of its natural slices, and the traced
    /// pass subtracts what it replaced from the same slice.
    uint  handoff;
    uint  _pad0, _pad1;
};

#endif
