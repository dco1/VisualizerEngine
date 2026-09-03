// ── ILLUMINATORAMA — SSAO (GTAO horizon-integral) ────────────────────────────
//
// Ground-Truth Ambient Occlusion (Jimenez et al. 2016 / Intel XeGTAO) plus its
// spatial and temporal filters.
//
// **Why this replaced the hemisphere kernel (DH-0527, was DH-0440).** The previous
// estimator was a 16-point cosine-weighted hemisphere Monte-Carlo: it fired 16 rays
// into a hash-rotated hemisphere, projected each back to screen, and counted depth
// hits. At a concave junction — the ONLY place AO finds real occlusion — that hash ×
// point-cloud rendered as a foamy 2–4 px cellular crust hugging every wall/wall and
// wall/ceiling corner ("spongy shadows… painted with a sponge", Danny). The
// `+SpongeProbe` bench proved across seven ablations that the foam was NOT sampling
// variance (16→64 samples removed only 20% of it vs the 50% Monte-Carlo demands),
// NOT the normal basis, NOT the buffer resolution, and NOT the denoiser — the single
// lever that moved it was the kernel RADIUS, the fingerprint of "wrong method", not
// "wrong setting". A screen-space point-cloud estimator cannot be tuned out of it.
//
// GTAO estimates occlusion ANALYTICALLY instead of by point sampling. For each of a
// few screen-space slices through the pixel it marches the depth buffer to find the
// two HORIZON angles (the highest occluding angle on each side), projects the surface
// normal into the slice plane, and evaluates the CLOSED-FORM cosine-visibility
// integral over the arc between the horizons. There is no per-pixel point cloud to
// alias, so the corner foam disappears by construction while genuine contact
// darkening (the wash up a real corner) is retained. Same screen-space cost envelope,
// same draft/interactive lane, same G-buffer inputs, same half-res buffer, and the
// existing bilateral + temporal filters below are unchanged.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

static inline float ssaoHash12(float2 p) {
    // De Vries-style cheap hash, [0,1).
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Jimenez Interleaved Gradient Noise, [0,1). Unlike white noise (`ssaoHash12`) its energy
// sits at a spatial frequency the 3×3 bilateral below resolves, so a per-pixel azimuth
// rotation seeded from it reads as a smooth field after one denoise pass rather than the
// white speckle a hash leaves. It gives the live TAA + SSAO-temporal passes a structured,
// per-frame-advancing pattern to integrate (see the seed walk below), which is what converges
// the grazing-flat self-occlusion the no-guard march leaves. The headless capture lane forces
// TAA off, so a single-frame grazing flat still speckles there — verified live, not in the bench.
static inline float ssaoIGN(float2 p) {
    return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}


// ── GTAO — slices, steps, falloff ────────────────────────────────────────────
//
// Six azimuthal slices × four march steps per side is 48 depth reads. That is more
// than the retired 16-sample point cloud, but it stays firmly in the cheap
// screen-space lane (half-res, no ray tracing) — the point of GTAO over RTAO — and
// the extra slices buy a low-variance single frame on concave geometry (corners), which
// is where the headline artefact lived. Grazing FLAT surfaces (a ceiling seen edge-on)
// still self-occlude in a single frame without a co-planarity guard (removed above); the
// live canvas's TAA + SSAO-temporal passes converge that residual, fed by the per-frame
// azimuth advance below. The headless bench forces TAA off (`pinGrade` →
// `sharedTAAOverride = false`) and cannot exercise that convergence.
constant uint  kGtaoSlices = 6;
constant uint  kGtaoSteps  = 4;
constant float kHalfPi     = 1.5707963267948966;

// Minimum march offset, in FULL-RES pixels. A horizon sample must land on a DIFFERENT
// surface point than the one being shaded, and "different pixel" is not enough: at a
// grazing view two adjacent texels of the same flat plane are separated almost entirely
// ALONG the view ray, so `dot(delta/|delta|, V)` is near 1 and the march records a horizon
// at ~0° — a flat floor reporting near-total occlusion of itself. Whether step 0 lands
// inside that first-neighbour band was decided by a per-pixel WHITE-noise jitter
// (`stepJitter`) that also advances every frame, so the false horizon flickered per pixel
// and per frame: white speckle on flat walls and floors, which the temporal pass could only
// reproject into smears ("spongy trails"). The steps are now distributed over
// [kGtaoMinPix, maxPix] instead of [0, maxPix), so no sample is ever taken from the shaded
// pixel's own neighbourhood and the estimate stops depending on the jitter's coin flip.
// This is the same first-sample offset XeGTAO carries, and unlike a co-planarity bias it
// costs nothing at a real corner — the adjacent wall is still there two pixels out.
constant float kGtaoMinPix = 2.5;

// Tangent-plane bias. A sample only occludes if it rises ABOVE the shaded point's own
// surface plane; `dot(delta, Ngeo)/|delta|` is that elevation's sine. It is distance-relative
// on purpose, so it rejects the near-field self-samples a grazing floor march reads out of
// its OWN plane (depth quantisation lifts them a hair above it) without rejecting a real
// occluder further out. Kept small — the earlier attempt at this guard needed 0.045 to work
// because it was fighting the sub-pixel first sample and a bump-tilted arc normal at the same
// time, and at that size it also lifted the wash up a genuine corner.
constant float kPlaneBias = 0.002;
/// Top of the ramp: a tap is a FULL occluder once it stands this far off the shaded pixel's
/// tangent plane, relative to its distance. Between `kPlaneBias` and here it fades in.
constant float kPlaneBiasFull = 0.020;

/// **The tangent-plane guard as a RAMP, not a coin flip (DH-0599).**
///
/// The guard exists because a screen-space march on a flat surface keeps finding the surface
/// itself; DH-0527 added it as a hard test, `dot(d, Ngeo) > kPlaneBias * dist`, and it did stop
/// the grazing-flat speckle. But on a large flat plane that quantity is a knife edge — every tap
/// lies very nearly IN the plane, so `dot(d, Ngeo)` hovers around zero and what actually decides
/// it is depth quantisation and the truncation of the tap to an integer texel. Both of those are
/// functions of DEPTH, so the decision flips along lines of constant depth and the AO term prints
/// contour bands across any receding floor.
///
/// Measured on the launch document's oak floor at a 34° camera: the raw (pre-denoise) AO carried
/// a row-collapsed residual of 7.62 code values on a level of 62 — 12.3 % — and the 3×3 bilateral
/// only pulled that to 1.62, which is the visible banding. Ablating `ssaoIntensity` alone took it
/// to 0.14, so 91 % of the floor's banding was this term. It is invariant to the azimuth/step
/// jitter, to `ssaoRadius`, and to the material maps, and it gets WORSE with more march steps
/// (1.70 at 12 steps vs 1.62 at 4) — every extra tap is another coin flip. That set of
/// invariances is what identifies a hard threshold rather than under-sampling or noise.
///
/// A ramp keeps the guard's purpose — a tap lying in the plane contributes nothing — while making
/// the transition continuous, so there is no depth at which a tap pops in. Real occluders sit far
/// off the plane and reach weight 1 well inside the band, so contact darkening and room-scale AO
/// are unchanged.
static inline float gtaoPlaneWeight(float3 d, float3 Ngeo, float dist) {
    return smoothstep(kPlaneBias * dist, kPlaneBiasFull * dist, dot(d, Ngeo));
}


// **Why there IS a guard, after shipping without one.** DH-0527 first shipped this march
// with no tangent-plane test at all, on the reasoning that the live canvas's TAA +
// SSAO-temporal passes would converge the grazing-flat speckle a screen-space march leaves.
// They cannot, and the live render refuted it within a day (Danny, 2026-08-31: "spongy walls
// and floors … temporal lagging too, like spongy trails"). The residual was not low-amplitude
// dither for an accumulator to average down: whether a horizon sample landed on the shaded
// pixel's own plane was decided by a per-pixel WHITE-noise jitter that also advanced every
// frame, so the field was high-amplitude and re-randomised per frame — the one input a
// temporal filter turns into smears rather than a mean.
//
// The guard is cheap now, at 0.002 rather than the 0.045 the first attempt needed, because it
// is no longer compensating for two other defects: the march started sub-pixel
// (`kGtaoMinPix`) and the arc integral read a bump normal against geometric horizons
// (`Narc`). Fixing those left the guard with only its own job, and at this size it no longer
// lifts the wash up a genuine corner — the profile still reads 1.000 → 0.982 → 0.965 → 0.930
// into a wall/floor junction.

// The closed-form inner arc integral for ONE side of a slice (Jimenez et al. 2016,
// Eq. for the visibility of an arc). `h` is the (already hemisphere-clamped) horizon
// angle, `n` the projected-normal angle, both relative to the view vector in the
// slice plane. Returns the side's contribution before the projected-normal-length
// weight is applied by the caller.
static inline float gtaoArc(float h, float n, float cosN, float sinN) {
    return 0.25 * (cosN + 2.0 * h * sinN - cos(2.0 * h - n));
}

// Distance falloff for a horizon sample: 1 inside the radius, 0 at it. Quadratic, not
// linear. Linear attenuation charges a genuine occluder at a QUARTER of the radius a 25 %
// penalty, and the near field is where contact lives — a chair leg meeting the floor is
// centimetres from the shaded pixel with a half-metre radius dial — so a linear ramp taxes
// exactly the signal AO exists to produce while leaving the far, room-scale term at full
// strength. Squaring holds the near field at ~94 % and does its rolling off near the radius,
// where the cut is meant to happen. Measured: contact localization Δ 0.12 → 0.22.
static inline float gtaoFalloff(float dist, float radius) {
    float x = clamp(dist / radius, 0.0, 1.0);
    return 1.0 - x * x;
}

// View-space position at an integer full-res pixel, clamped to the buffer. Used to
// reconstruct the GEOMETRIC surface normal from depth (below).
static inline float3 gtaoViewPos(depth2d<float, access::read> gDepth, int2 p, int2 mx,
                                 float2 fdim, float4x4 invProj) {
    p = clamp(p, int2(0), mx);
    float z = gDepth.read(uint2(p));
    float2 ndc = (float2(p) + 0.5) / fdim * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return viewPosFromDepth(ndc, z, invProj);
}

kernel void illumi_ssao(
    depth2d<float, access::read>   gDepth      [[texture(0)]],
    texture2d<half, access::read>  gNormalRgh  [[texture(1)]],
    texture2d<half, access::write> outAO       [[texture(2)]],
    constant FrameUniforms&        frame       [[buffer(0)]],
    uint2                          gid         [[thread_position_in_grid]]
) {
    uint outW = outAO.get_width();
    uint outH = outAO.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    // Early-out when SSAO is disabled — write 1 (no occlusion) so the
    // lighting kernel can read unconditionally without branching.
    if (frame.ssaoIntensity <= 0.0) {
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Map half-res gid → full-res pixel for the G-buffer reads. Using the
    // 2×2 top-left avoids needing a sampler; we accept the small loss in
    // accuracy.
    uint fullW = gDepth.get_width();
    uint fullH = gDepth.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float depth = gDepth.read(fullGid);
    if (depth >= 0.99999) {
        // Sky — no surface to occlude. Treat as fully lit.
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Reconstruct view-space position. GTAO works in view space so the horizon deltas are
    // true camera-space vectors and the radius is metres.
    float2 fdim = float2(fullW, fullH);
    float2 ndc = (float2(fullGid) + 0.5) / fdim * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 Pview = viewPosFromDepth(ndc, depth, frame.invProjection);
    float3 V = normalize(-Pview);                 // surface → camera (view space)

    // ONE normal, reconstructed from depth: the true surface plane. The min-|Δz| pick per
    // axis avoids straddling a silhouette, where the reconstruction blends two surfaces.
    // (The bump-mapped G-buffer normal is deliberately NOT read here — see `Narc` below.)
    int2 gpx = int2(fullGid);
    int2 gmx = int2(int(fullW) - 1, int(fullH) - 1);
    float3 Pr = gtaoViewPos(gDepth, gpx + int2( 1, 0), gmx, fdim, frame.invProjection);
    float3 Pl = gtaoViewPos(gDepth, gpx + int2(-1, 0), gmx, fdim, frame.invProjection);
    float3 Pu = gtaoViewPos(gDepth, gpx + int2( 0, 1), gmx, fdim, frame.invProjection);
    float3 Pd = gtaoViewPos(gDepth, gpx + int2( 0,-1), gmx, fdim, frame.invProjection);
    float3 ddx = (abs(Pr.z - Pview.z) < abs(Pview.z - Pl.z)) ? (Pr - Pview) : (Pview - Pl);
    float3 ddy = (abs(Pu.z - Pview.z) < abs(Pview.z - Pd.z)) ? (Pu - Pview) : (Pview - Pd);
    float3 Ngeo = cross(ddx, ddy);
    float nlen = length(Ngeo);
    if (nlen < 1e-8) { outAO.write(half4(1.0h), gid); return; }   // degenerate patch — no occlusion
    Ngeo /= nlen;
    if (dot(Ngeo, V) < 0.0) Ngeo = -Ngeo;       // face the camera

    // The normal the ARC INTEGRAL integrates against. It must be the geometric one.
    //
    // GTAO's closed form is exact only when the projected-normal angle `n` and the two
    // horizon angles describe the same surface. The horizons are marched out of the DEPTH
    // buffer — pure geometry — so feeding `n` the BUMP-mapped G-buffer normal breaks that
    // agreement per texel: on a flat, unoccluded plane the horizons say "nothing above the
    // plane" while `n` is tilted a few degrees by the material's micro-relief, and the
    // integral returns a value just under 1 that varies with the bump pattern itself. The
    // result is occlusion CORRELATED WITH THE TEXTURE — a mottled crust that follows the
    // wood grain on a floor and the roll on a plaster wall, which is precisely the "spongy"
    // read (DH-0527, Danny 2026-08-31, on the shipped no-guard build). The old point-cloud
    // estimator was immune because a small normal tilt cannot conjure a depth hit; GTAO's
    // analytic term multiplies it straight into the output.
    //
    // Bump normals belong in the lighting integral's N·L, not in a visibility arc whose
    // horizons are geometric.
    float3 Narc = Ngeo;

    float radius = max(0.001, frame.ssaoRadius);

    // Screen-space march extent. A view-space offset of `radius` at this pixel's depth
    // projects to `radius · focal · 0.5 · dim / |z|` pixels; because focalX·W == focalY·H
    // for a standard perspective the count is isotropic, so one scalar covers both axes.
    float focalX = frame.projection[0][0];
    float focalY = frame.projection[1][1];
    float invZ   = 1.0 / max(1e-3, -Pview.z);
    // The lower clamp must leave room for `kGtaoMinPix` plus a real march; a 2 px extent
    // would collapse every step onto the first-neighbour band the minimum offset exists to
    // skip, which is the grazing-flat speckle again at distance.
    float maxPix = clamp(radius * focalY * 0.5 * float(fullH) * invZ, kGtaoMinPix + 4.0, 220.0);

    // Per-pixel azimuth rotation + a march-offset jitter, ADVANCED per frame by a
    // golden-ratio step keyed to `rtSunShadowSeed`. Three slices is too few to cover the
    // azimuth in one frame, so consecutive frames must sample DIFFERENT orientations for
    // the SSAO temporal pass to integrate them into a full sweep — otherwise a static
    // camera re-averages identical frames and denoises nothing (the grazing-ceiling
    // speckle). The seed WALKS only while a temporal accumulator is live and is frozen at
    // 0 otherwise, so a truly static frame keeps a stable per-pixel pattern rather than
    // crawling — the same contract the RT sun-shadow and glass passes already honour.
    float frameAdvance = float(frame.rtSunShadowSeed) * 0.61803398875;
    float sliceJitter = fract(ssaoIGN(float2(gid)) + frameAdvance);
    // The step jitter is INTERLEAVED-GRADIENT too, offset half a tile from the azimuth's so the
    // two are not the same number. It used to be white noise (`ssaoHash12`), which put the march
    // offset — and therefore which occluders each pixel finds — at a spatial frequency the 3×3
    // bilateral cannot resolve. The denoiser cannot remove that; it can only pool it into soft
    // blobs a few pixels across, which is the residual structure the launch look still showed on
    // a shaded wall after the march fixes (a negative autocorrelation lobe at lag 3–6). IGN's
    // energy sits where the bilateral CAN see it, which is the whole reason the azimuth already
    // used it.
    float stepJitter  = fract(ssaoIGN(float2(gid) + float2(23.5, 41.5)) + frameAdvance * 1.324717957);

    float visibility = 0.0;
    for (uint s = 0; s < kGtaoSlices; ++s) {
        float phi = (float(M_PI_F) / float(kGtaoSlices)) * (float(s) + sliceJitter);
        float2 omega = float2(cos(phi), sin(phi));        // screen-space (pixel) direction

        // View-space direction for that screen direction. Divide by the focal lengths so
        // an anisotropic FOV does not skew the slice plane; horizons themselves are read
        // from reconstructed 3-D positions and are unaffected by this approximation.
        float3 dirVec = normalize(float3(omega.x / focalX, omega.y / focalY, 0.0));
        float3 orthoDir = dirVec - dot(dirVec, V) * V;
        float3 axis = cross(dirVec, V);
        float axisLen = length(axis);
        if (axisLen < 1e-5) continue;
        axis /= axisLen;

        // Project the surface normal into the slice plane. This is the GEOMETRIC normal on
        // purpose — see `Ngeo` above: the horizons are read out of the depth buffer, so the
        // arc integral is only self-consistent (flat + unoccluded ⇒ visibility exactly 1)
        // when `n` describes the SAME surface the horizons came from.
        float3 projN = Narc - axis * dot(Narc, axis);
        float projLen = length(projN);
        if (projLen < 1e-5) continue;
        float cosNorm = clamp(dot(projN, V) / projLen, -1.0, 1.0);
        float signN = sign(dot(orthoDir, projN));
        float n = signN * acos(cosNorm);
        float sinNorm = sin(n);

        // Horizon search: track the largest cos (smallest angle → most occluding) on
        // each side. Far samples are faded to -1 (no horizon) so nothing beyond `radius`
        // contributes, which is what turns the radius dial into a true world distance.
        float horizonPos = -1.0;   // +omega side
        float horizonNeg = -1.0;   // -omega side
        for (uint k = 0; k < kGtaoSteps; ++k) {
            // Quadratic step spacing (XeGTAO's `pow` distribution). Four samples spread
            // LINEARLY over [kGtaoMinPix, maxPix] put nothing in the near field — at a
            // metre-scale radius the first sample already sits ~30 px out — and the near
            // field is exactly where an object's contact with the floor lives, so linear
            // spacing measures real room-scale ambient occlusion correctly while reporting
            // almost no LOCALIZATION at object bases (`ContactAO` Δ 0.11 → 0.04). Squaring
            // the fraction clusters the taps near the origin and still reaches the full
            // radius on the last one, so contact darkening and room-scale AO come out of
            // the same march.
            float frac = (float(k) + stepJitter) / float(kGtaoSteps);
            float t = mix(kGtaoMinPix, maxPix, frac * frac);
            float2 off = omega * t;

            // +omega
            float2 pPos = float2(fullGid) + off;
            if (pPos.x >= 0.0 && pPos.y >= 0.0 && pPos.x < fdim.x && pPos.y < fdim.y) {
                uint2 sPx = uint2(pPos);
                float sd = gDepth.read(sPx);
                if (sd < 0.99999) {
                    float2 sndc = (float2(sPx) + 0.5) / fdim * 2.0 - 1.0;
                    sndc.y = -sndc.y;
                    float3 Ps = viewPosFromDepth(sndc, sd, frame.invProjection);
                    float3 d = Ps - Pview;
                    float dist = length(d);
                    if (dist > 1e-4) {
                        float fall = gtaoFalloff(dist, radius) * gtaoPlaneWeight(d, Ngeo, dist);
                        float c = dot(d / dist, V);
                        horizonPos = max(horizonPos, mix(-1.0, c, fall));
                    }
                }
            }

            // -omega
            float2 pNeg = float2(fullGid) - off;
            if (pNeg.x >= 0.0 && pNeg.y >= 0.0 && pNeg.x < fdim.x && pNeg.y < fdim.y) {
                uint2 sPx = uint2(pNeg);
                float sd = gDepth.read(sPx);
                if (sd < 0.99999) {
                    float2 sndc = (float2(sPx) + 0.5) / fdim * 2.0 - 1.0;
                    sndc.y = -sndc.y;
                    float3 Ps = viewPosFromDepth(sndc, sd, frame.invProjection);
                    float3 d = Ps - Pview;
                    float dist = length(d);
                    if (dist > 1e-4) {
                        float fall = gtaoFalloff(dist, radius) * gtaoPlaneWeight(d, Ngeo, dist);
                        float c = dot(d / dist, V);
                        horizonNeg = max(horizonNeg, mix(-1.0, c, fall));
                    }
                }
            }
        }

        // Angles from cosines, clamped to the hemisphere around the projected normal.
        float hP =  acos(clamp(horizonPos, -1.0, 1.0));
        float hN = -acos(clamp(horizonNeg, -1.0, 1.0));
        hP = n + clamp(hP - n, -kHalfPi, kHalfPi);
        hN = n + clamp(hN - n, -kHalfPi, kHalfPi);

        visibility += projLen * (gtaoArc(hP, n, cosNorm, sinNorm) +
                                 gtaoArc(hN, n, cosNorm, sinNorm));
    }
    visibility = clamp(visibility / float(kGtaoSlices), 0.0, 1.0);

    // Fold intensity in the same way the old estimator did: occlusion = 1 − visibility.
    float ao = 1.0 - (1.0 - visibility) * frame.ssaoIntensity;
    ao = clamp(ao, 0.0, 1.0);
    outAO.write(half4(half(ao)), gid);
}

// ── Phase 4.39: SSAO bilateral spatial filter ────────────────────────────────
//
// Depth + normal guided bilateral filter over the half-resolution raw AO
// texture. A 3×3 window reads each neighbor's AO value and weights it by:
//
//   exp(−|depth_diff| · kDepth) · pow(max(0, dot(n_center, n_neighbor)), kNorm)
//
// Depth weights reject samples across geometric boundaries (contact edges);
// normal weights reject samples on curved surfaces where orientation diverges.
// Output feeds the temporal pass rather than lighting directly, so the spatial
// pass only needs to be stable enough for temporal to converge cleanly.

kernel void illumi_ssao_spatial(
    texture2d<half, access::read>   rawAO      [[texture(0)]],  // half-res
    depth2d<float,  access::read>   gDepth     [[texture(1)]],  // full-res
    texture2d<half, access::read>   gNormalRgh [[texture(2)]],  // full-res
    texture2d<half, access::write>  outAO      [[texture(3)]],  // half-res
    constant FrameUniforms&         frame      [[buffer(0)]],
    uint2                           gid        [[thread_position_in_grid]]
) {
    uint halfW = outAO.get_width();
    uint halfH = outAO.get_height();
    if (gid.x >= halfW || gid.y >= halfH) return;

    if (frame.ssaoDenoiseEnabled == 0u || frame.ssaoIntensity <= 0.0) {
        outAO.write(rawAO.read(gid), gid);
        return;
    }

    uint fullW = gDepth.get_width();
    uint fullH = gDepth.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float centerDepth = gDepth.read(fullGid);
    if (centerDepth >= 0.99999) {
        outAO.write(half4(1.0h), gid);
        return;
    }

    // Reconstruct center view-Z for depth weighting.
    float2 ndcC = (float2(fullGid) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
    ndcC.y = -ndcC.y;
    float3 PviewC = viewPosFromDepth(ndcC, centerDepth, frame.invProjection);
    float3 NworldC = octDecode(float2(gNormalRgh.read(fullGid).rg));

    const float kDepth = 5.0;   // depth sensitivity (larger = tighter boundary preservation)
    const float kNorm  = 16.0;  // normal exponent  (larger = more sensitive to curvature)

    float totalAO     = 0.0;
    float totalWeight = 0.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 halfN = int2(int(gid.x) + i, int(gid.y) + j);
            halfN = clamp(halfN, int2(0), int2(int(halfW) - 1, int(halfH) - 1));
            uint2 fullN = min(uint2(halfN) * 2u, uint2(fullW - 1, fullH - 1));

            float nd = gDepth.read(fullN);
            if (nd >= 0.99999) continue;

            float2 ndcN = (float2(fullN) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
            ndcN.y = -ndcN.y;
            float3 PviewN   = viewPosFromDepth(ndcN, nd, frame.invProjection);
            float3 NworldN  = octDecode(float2(gNormalRgh.read(fullN).rg));

            float depthDiff  = abs(PviewC.z - PviewN.z);
            float wDepth     = exp(-depthDiff * kDepth);
            float wNorm      = pow(max(0.0, dot(NworldC, NworldN)), kNorm);
            float w          = wDepth * wNorm;

            totalAO     += float(rawAO.read(uint2(halfN)).r) * w;
            totalWeight += w;
        }
    }

    float filtered = (totalWeight > 1e-6) ? (totalAO / totalWeight)
                                           : float(rawAO.read(gid).r);
    outAO.write(half4(half(filtered), 0.0h, 0.0h, 1.0h), gid);
}

// ── Phase 4.39: SSAO temporal accumulation ───────────────────────────────────
//
// Reprojects the previous frame's AO history using the full-res velocity
// texture, clamps the history sample into a scalar variance band, and blends
// a high-weight (≈0.9) history with the current spatially-filtered AO. This
// converts the 16-sample single-tap AO into effectively hundreds of samples
// spread over time — AO "for free" from the temporal integration.
//
// Velocity is at full-res; for a half-res pixel at `gid` we read velocity
// at `gid×2` (the matching full-res texel). UV-space velocity is resolution-
// independent so the reprojection math is identical.

kernel void illumi_ssao_temporal(
    texture2d<half, access::read>       filteredAO   [[texture(0)]],  // half-res current
    texture2d<half, access::sample>     historyAO    [[texture(1)]],  // half-res history
    texture2d<half, access::read>       velocity     [[texture(2)]],  // full-res
    texture2d<half, access::write>      outAO        [[texture(3)]],  // half-res output
    texture2d<half, access::read_write> sampleCount  [[texture(4)]],  // half-res r16Float
    constant FrameUniforms&             frame        [[buffer(0)]],
    uint2                               gid          [[thread_position_in_grid]]
) {
    uint halfW = outAO.get_width();
    uint halfH = outAO.get_height();
    if (gid.x >= halfW || gid.y >= halfH) return;

    float current = float(filteredAO.read(gid).r);

    // Reset sample count and pass through on first frame or when disabled.
    if (frame.ssaoDenoiseEnabled == 0u || frame.ssaoIsFirstFrame != 0u) {
        sampleCount.write(half4(0.0h), gid);
        outAO.write(half4(half(current)), gid);
        return;
    }

    // Read accumulated sample count from previous frame.
    float N = float(sampleCount.read(gid).r);

    uint fullW = velocity.get_width();
    uint fullH = velocity.get_height();
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));
    float2 vel = float2(velocity.read(fullGid).rg);

    float2 halfUV = (float2(gid) + 0.5) / float2(halfW, halfH);
    float2 histUV = halfUV - vel;

    // Screen-edge disocclusion: no history available → reset count.
    if (any(histUV < float2(0.0)) || any(histUV > float2(1.0))) {
        sampleCount.write(half4(1.0h), gid);
        outAO.write(half4(half(current)), gid);
        return;
    }

    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float hist = float(historyAO.sample(samp, histUV).r);

    // Scalar variance clamp over the 3×3 half-res neighborhood.
    float m1 = 0.0, m2 = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(halfW) - 1, int(halfH) - 1));
            float s = float(filteredAO.read(uint2(c)).r);
            m1 += s; m2 += s * s;
        }
    }
    m1 /= 9.0; m2 /= 9.0;
    float sigma = max(sqrt(max(0.0, m2 - m1 * m1)), 0.03);
    hist = clamp(hist, m1 - 1.5 * sigma, m1 + 1.5 * sigma);

    float velMag = length(vel);
    float disoccBlend = smoothstep(0.004, 0.025, velMag);

    // Adaptive blend: 1/N drives fast convergence in early frames; once N is
    // large enough the steady-state floor (1 - ssaoTemporalBlend) takes over.
    // Velocity disocclusion resets N so newly-revealed surfaces reconverge quickly.
    if (disoccBlend > 0.5) N = 0.0;
    N = min(N + 1.0, 32.0);
    sampleCount.write(half4(half(N)), gid);

    float minAlpha = 1.0 - frame.ssaoTemporalBlend;
    float alpha = max(minAlpha, 1.0 / N);
    alpha = mix(alpha, min(1.0, alpha * 4.0), disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    float result = mix(hist, current, alpha);
    outAO.write(half4(half(result)), gid);
}
