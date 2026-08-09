// ── ILLUMINATORAMA — SUBSURFACE SCATTERING ─────────────────────────
//
// The separable screen-space subsurface blur and its composite, run on the
// deferred lighting result for skin/wax/marble surfaces.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Screen-space subsurface scattering (issue #65) ───────────────────────────
//
// A Jimenez-style SEPARABLE screen-space SSS for skin / wax / marble / food.
// The deferred lighting pass peeled the diffuse-lit irradiance off SSS-flagged
// pixels into a side buffer (rgb = diffuse, a = mask). We diffuse it laterally
// with a 1-D blur run twice (horizontal then vertical) — separable being the
// whole point of the technique (Jimenez 2015): a true 2-D diffusion profile is
// closely approximated by two 1-D passes at a fraction of the cost. A composite
// then blends the blurred diffuse back over the sharp diffuse and the rest of
// the lit image (specular / emission / clearcoat) stays untouched.
//
// The kernel is PHYSICAL in two ways the project cares about (no scrolling-
// texture fakery): (1) the blur width is the diffusion mean-free-path in
// MILLIMETRES projected to screen pixels per-pixel from depth + the projection,
// so an object diffuses by a real-world distance regardless of how big it is on
// screen; (2) the profile is a two-lobe Gaussian sum (narrow core + wide tail)
// whose per-channel width scales by `sssTint`, reproducing the wavelength-
// dependent scatter that makes the red bleed reach farther than green/blue —
// the signature of skin/wax. Depth-aware tap rejection stops the diffusion
// bleeding across silhouettes onto the background or a foreground occluder.

// Two-lobe diffusion profile weight at radius `r` (px), per RGB channel. The
// per-channel sigma scales by `tint` so a reddish tint scatters red farthest.
// Not normalised here — the kernel renormalises by the accumulated weight, which
// also handles taps dropped at silhouettes.
static inline float3 sssProfileWeight(float r, float radiusPx, float3 tint) {
    float3 sNarrow = max(radiusPx * 0.30 * tint, float3(0.5));
    float3 sWide   = max(radiusPx * 1.00 * tint, float3(0.5));
    float  r2      = r * r;
    float3 gN = exp(-r2 / (2.0 * sNarrow * sNarrow));
    float3 gW = exp(-r2 / (2.0 * sWide   * sWide));
    return 0.55 * gN + 0.45 * gW;   // core-weighted (sharper centre, soft tail)
}

constant int kSSSTaps = 12;                 // samples per side along the 1-D axis
constant float kSSSMaxRadiusPx = 64.0;      // clamp so a near object can't smear the frame

// One separable pass. `dir` is (1,0) for horizontal, (0,1) for vertical. Non-SSS
// pixels (mask 0) pass straight through so the second pass and the composite see
// the untouched diffuse for the rest of the image.
kernel void illumi_sss_blur(
    texture2d<half,  access::read>  src    [[texture(0)]],   // diffuse rgb + mask a
    depth2d<float,   access::read>  gDepth [[texture(1)]],
    texture2d<half,  access::write> dst    [[texture(2)]],
    constant FrameUniforms&         frame  [[buffer(0)]],
    constant float2&                dir    [[buffer(1)]],
    uint2                           gid    [[thread_position_in_grid]]
) {
    uint w = dst.get_width();
    uint h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;

    half4 center = src.read(gid);
    if (center.a < 0.5h) { dst.write(center, gid); return; }   // not SSS — passthrough

    float depthC = gDepth.read(gid);
    float2 ndc = (float2(gid) + 0.5) / float2(w, h) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 vposC = viewPosFromDepth(ndc, depthC, frame.invProjection);
    float  viewZ = max(1e-3, -vposC.z);

    // mm → screen px at this depth: perspective px-per-metre = 0.5·H·P[1][1]/zEye.
    float pxPerMetre = 0.5 * float(h) * frame.projection[1][1] / viewZ;
    float radiusPx   = clamp(frame.sssRadius * 0.001 * pxPerMetre, 0.0, kSSSMaxRadiusPx);
    if (radiusPx < 0.75) { dst.write(center, gid); return; }   // sub-pixel — nothing to blur

    float3 tint = max(float3(frame.sssTintR, frame.sssTintG, frame.sssTintB), float3(0.0));
    // Depth window for tap rejection: a fraction of the diffusion reach in world
    // units, so the blur stays on the surface and doesn't leak past a silhouette.
    float depthSigma = max(frame.sssRadius * 0.001 * 1.5, 1e-3);

    float3 sum  = float3(0.0);
    float3 wsum = float3(0.0);
    for (int i = -kSSSTaps; i <= kSSSTaps; ++i) {
        float  t     = float(i) / float(kSSSTaps);   // -1 … 1
        float  offPx = t * radiusPx;
        float2 sxy   = float2(gid) + dir * offPx;
        int2   sc    = int2(round(sxy));
        if (sc.x < 0 || sc.y < 0 || sc.x >= int(w) || sc.y >= int(h)) continue;
        uint2  scu = uint2(sc);
        half4  s   = src.read(scu);
        if (s.a < 0.5h) continue;                    // don't pull from non-SSS neighbours

        // Depth-aware: reject taps whose surface sits far from the centre in eye Z.
        float  dS    = gDepth.read(scu);
        float2 sndc  = (float2(scu) + 0.5) / float2(w, h) * 2.0 - 1.0;
        sndc.y = -sndc.y;
        float  vzS   = max(1e-3, -viewPosFromDepth(sndc, dS, frame.invProjection).z);
        float  dz    = vzS - viewZ;
        float  depthW = exp(-(dz * dz) / (2.0 * depthSigma * depthSigma));

        float3 prof = sssProfileWeight(abs(offPx), radiusPx, tint) * depthW;
        sum  += float3(s.rgb) * prof;
        wsum += prof;
    }

    float3 outRGB = float3(center.rgb);
    outRGB.r = wsum.r > 1e-5 ? sum.r / wsum.r : outRGB.r;
    outRGB.g = wsum.g > 1e-5 ? sum.g / wsum.g : outRGB.g;
    outRGB.b = wsum.b > 1e-5 ? sum.b / wsum.b : outRGB.b;
    dst.write(half4(half3(outRGB), center.a), gid);
}

// Composite the blurred diffuse back over the lit HDR. For SSS pixels only,
// `hdr += sssStrength · (blurred − sharp)`: at strength 1 this exactly replaces
// the sharp diffuse (already in HDR) with the blurred diffuse, leaving specular /
// emission / clearcoat as they were; at strength 0 it adds nothing (and the host
// doesn't even encode this pass). read_write at the same gid is per-pixel safe.
kernel void illumi_sss_composite(
    texture2d<half, access::read_write> hdr      [[texture(0)]],
    texture2d<half, access::read>       diffOrig [[texture(1)]],  // sharp diffuse + mask
    texture2d<half, access::read>       diffBlur [[texture(2)]],  // blurred diffuse
    constant FrameUniforms&             frame    [[buffer(0)]],
    uint2                               gid      [[thread_position_in_grid]]
) {
    uint w = hdr.get_width();
    uint h = hdr.get_height();
    if (gid.x >= w || gid.y >= h) return;

    half4 o = diffOrig.read(gid);
    if (o.a < 0.5h) return;                          // non-SSS — leave HDR untouched

    half4  b     = diffBlur.read(gid);
    float3 delta = (float3(b.rgb) - float3(o.rgb)) * frame.sssStrength;
    half4  c     = hdr.read(gid);
    hdr.write(half4(half3(float3(c.rgb) + delta), c.a), gid);
}
