// ── ILLUMINATORAMA — TEMPORAL ANTI-ALIASING ─────────────────────────────────
//
// Catmull-Rom history reprojection with a neighbourhood-clamped exponential
// blend.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaDenoise.h"
using namespace metal;

// 9-tap Catmull-Rom bicubic history filter (Brian Karis, "High Quality Temporal
// Supersampling", SIGGRAPH 2014). Reconstructs the history with cubic precision
// using 9 bilinear fetches, eliminating the temporal blur a single bilinear tap
// introduces, without the over-sharpening of a naïve bicubic.
//
// CRITICAL: the fractional coordinate `f` must be measured from `texPos1` (the
// first of the four 1-D samples) and lie in [0,1) — that's the domain the weight
// polynomials w0..w3 are defined on. `texPos1 = floor(samplePos − 0.5) + 0.5`,
// `f = samplePos − texPos1`. Computing `f` as `frac(pos) − 0.5` (∈ [−0.5,0.5))
// evaluates the polynomials off-domain → at a static camera f pins to a constant
// non-zero value, reconstructing a half-pixel-shifted, ringing copy of the
// history every frame → horizontal contour banding on smooth gradients.
static float3 sampleCatmullRom(
    texture2d<half, access::sample> tex,
    sampler samp,
    float2 uv,
    float2 texSize
) {
    float2 samplePos = uv * texSize;
    float2 texPos1   = floor(samplePos - 0.5) + 0.5;
    float2 f         = samplePos - texPos1;   // ∈ [0,1)

    float2 w0 = f * (-0.5 + f * ( 1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * ( 0.5 + f * ( 2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    // Combine the middle two 1-D taps (w1,w2) into one bilinear fetch.
    float2 w12     = w1 + w2;
    float2 offset12 = w2 / w12;

    float2 tc0  = (texPos1 - 1.0)      / texSize;
    float2 tc12 = (texPos1 + offset12) / texSize;
    float2 tc3  = (texPos1 + 2.0)      / texSize;

    float3 r = float3(0.0);
    r += float3(tex.sample(samp, float2(tc0.x,  tc0.y )).rgb) * (w0.x  * w0.y );
    r += float3(tex.sample(samp, float2(tc12.x, tc0.y )).rgb) * (w12.x * w0.y );
    r += float3(tex.sample(samp, float2(tc3.x,  tc0.y )).rgb) * (w3.x  * w0.y );

    r += float3(tex.sample(samp, float2(tc0.x,  tc12.y)).rgb) * (w0.x  * w12.y);
    r += float3(tex.sample(samp, float2(tc12.x, tc12.y)).rgb) * (w12.x * w12.y);
    r += float3(tex.sample(samp, float2(tc3.x,  tc12.y)).rgb) * (w3.x  * w12.y);

    r += float3(tex.sample(samp, float2(tc0.x,  tc3.y )).rgb) * (w0.x  * w3.y );
    r += float3(tex.sample(samp, float2(tc12.x, tc3.y )).rgb) * (w12.x * w3.y );
    r += float3(tex.sample(samp, float2(tc3.x,  tc3.y )).rgb) * (w3.x  * w3.y );

    return max(float3(0.0), r);
}

// Clip a reprojected history sample into the current-frame neighbourhood AABB
// along the ray from the box centre, instead of clamping it per-channel. A
// per-channel clamp can land on a colour that is OFF the centre→history line
// (a hue shift / chroma ghost); the ray clip pulls an out-of-box history onto
// the nearest box face along that line, suppressing ghosts without recolouring.
// (Karis 2014, "High-Quality Temporal Supersampling" — clip-to-AABB.)
static inline float3 clipHistoryToAABB(float3 boxMin, float3 boxMax, float3 hist) {
    float3 center = 0.5 * (boxMax + boxMin);
    float3 extent = 0.5 * (boxMax - boxMin) + float3(1e-5);
    float3 v      = hist - center;
    float3 a      = abs(v / extent);
    float  maxU   = max(a.x, max(a.y, a.z));
    return (maxU > 1.0) ? (center + v / maxU) : hist;
}

// ── Temporal anti-aliasing (Phase 2.7 / upgraded Phase 4.39) ─────────────────
//
// Reprojects the previous frame's accumulated HDR with the velocity buffer,
// clamps the history into the current-frame neighborhood, and blends in a
// small current-frame contribution. Three upgrades land in Phase 4.39:
//
//  1. YCoCg color space — convert current + neighborhood to YCoCg before
//     building the AABB. Luminance (Y) and chrominance (Co/Cg) are clamped
//     independently, which eliminates the chromatic ghosting that RGB-space
//     clamping produces on color edges (magenta/green halos on moving objects).
//
//  2. Variance-based AABB (mean ± γ·σ) — instead of the hard min/max over
//     the 3×3, compute the per-channel mean and standard deviation and use
//     γ·σ as the clamp radius. This widens the acceptance window on smooth
//     flat surfaces (where min/max is overly tight and throws away perfectly
//     valid history) while staying tight on high-contrast edges.
//
//  3. 5-tap Catmull-Rom history sampling — replaces the bilinear sample of
//     the history buffer, eliminating the temporal blur that bilinear
//     introduces. Uses four bilinear fetches to achieve cubic precision
//     (Brian Karis, "High Quality Temporal Supersampling", SIGGRAPH 2014).
//
//  4. Velocity-magnitude disocclusion ramp — fast-moving pixels are more
//     likely to have invalid history (newly uncovered surface). Ramp the
//     current-frame weight up when the velocity is large so disoccluded
//     pixels reconverge quickly instead of ghosting.

kernel void illumi_taa_resolve(
    texture2d<half,  access::read>    currentHDR  [[texture(0)]],
    texture2d<half,  access::sample>  historyHDR  [[texture(1)]],
    texture2d<half,  access::read>    velocityTex [[texture(2)]],
    texture2d<half,  access::write>   outHDR      [[texture(3)]],
    // Phase 4.44 — depth, so the resolve can tell a sky pixel (cleared depth,
    // zero velocity) from genuinely-static geometry and synthesise a
    // camera-only motion vector for the former.
    depth2d<float,   access::read>    gDepth      [[texture(4)]],
    // Previous frame's depth — for disocclusion rejection (see below).
    depth2d<float,   access::read>    gPrevDepth  [[texture(5)]],
    constant FrameUniforms&           frame       [[buffer(0)]],
    uint2                             gid         [[thread_position_in_grid]]
) {
    uint W = outHDR.get_width();
    uint H = outHDR.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float3 current = float3(currentHDR.read(gid).rgb);

    if (frame.taaEnabled == 0u || frame.taaIsFirstFrame != 0u) {
        outHDR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    float2 uv  = (float2(gid) + 0.5) / float2(W, H);

    // ── Velocity dilation ────────────────────────────────────────────────────
    // Reproject using the velocity of the CLOSEST-depth pixel in a 3×3 window,
    // not this pixel's own. At a thin foreground mover's silhouette the centre
    // pixel can carry the background's (near-zero) velocity and trail; pinning
    // the motion vector to the nearest surface stops moving edges smearing.
    float closestDepth = 1e30;
    int2  closestOff   = int2(0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(W) - 1, int(H) - 1));
            float d = gDepth.read(uint2(c));
            if (d < closestDepth) { closestDepth = d; closestOff = int2(i, j); }
        }
    }
    uint2 velCoord = uint2(clamp(int2(gid) + closestOff,
                                 int2(0), int2(int(W) - 1, int(H) - 1)));
    float2 vel = float2(velocityTex.read(velCoord).rg);

    // ── Phase 4.44: camera-only velocity for sky pixels ──────────────────────
    // Sky/far-plane pixels are never rasterised into the G-buffer, so their
    // velocity stays at the cleared 0 — a rotating camera then smears the sky
    // (and moving cloud silhouettes) because the resolve reprojects them as if
    // they were static. Reconstruct the pure-rotation motion vector from the
    // view ray at infinity: unproject the pixel to a world-space direction, then
    // project that DIRECTION (w = 0, so the camera's translation is correctly
    // ignored at infinity) by the previous frame's VP. Same UV convention as the
    // G-buffer's velocity write: (currNDC − prevNDC) · (0.5, −0.5). Gated on
    // THIS pixel's depth, not the dilated neighbour.
    if (gDepth.read(gid) >= 0.99999) {
        float2 ndc = uv * 2.0 - 1.0;
        ndc.y = -ndc.y;
        float4 farWorld = frame.invViewProjection * float4(ndc, 1.0, 1.0);
        float3 dir = farWorld.xyz / farWorld.w - frame.cameraWorldPos;
        float4 prevClip = frame.previousViewProjection * float4(dir, 0.0);
        if (prevClip.w > 1e-5) {
            float2 prevNDC = prevClip.xy / prevClip.w;
            vel = (ndc - prevNDC) * float2(0.5, -0.5);
        }
    }

    float2 historyUV = uv - vel;

    if (any(historyUV < float2(0.0)) || any(historyUV > float2(1.0))) {
        outHDR.write(half4(half3(current), 1.0h), gid);
        return;
    }

    // ── Depth-based disocclusion rejection ────────────────────────────────────
    // Reconstruct this pixel's world position from its CURRENT depth, project it
    // by the PREVIOUS frame's view-projection, and compare the depth it WOULD
    // have had last frame against the depth ACTUALLY stored at the reprojected
    // history texel. A large mismatch means a different surface occupied this
    // screen location last frame (a fast mover vacated it) — its history is
    // invalid even when the two surfaces share luma (the case neighbourhood
    // clamping can't catch, e.g. a grey egg trailing over the grey floor). Skips
    // sky (cleared depth), already handled by the camera-only velocity above.
    float depthDisocc = 0.0;
    float dC = gDepth.read(gid);
    if (dC < 0.99999) {
        float2 ndcC   = (uv * 2.0 - 1.0) * float2(1.0, -1.0);
        float4 worldH = frame.invViewProjection * float4(ndcC, dC, 1.0);
        float3 worldC = worldH.xyz / worldH.w;
        float4 prevClip = frame.previousViewProjection * float4(worldC, 1.0);
        if (prevClip.w > 1e-5) {
            float prevZexpected = prevClip.z / prevClip.w;
            uint2 hpix = uint2(clamp(int2(historyUV * float2(W, H)),
                                     int2(0), int2(int(W) - 1, int(H) - 1)));
            float prevZactual = gPrevDepth.read(hpix);
            depthDisocc = smoothstep(0.0015, 0.01, abs(prevZexpected - prevZactual));
        }
    }

    // ── 5-tap Catmull-Rom history sample ─────────────────────────────────────
    constexpr sampler historySampler(filter::linear, address::clamp_to_edge);
    float3 history = sampleCatmullRom(historyHDR, historySampler,
                                      historyUV, float2(W, H));

    // ── 3×3 neighborhood statistics in YCoCg ─────────────────────────────────
    float3 currentYCoCg = RGBtoYCoCg(current);
    float3 historyYCoCg = RGBtoYCoCg(history);

    float3 m1 = float3(0.0), m2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j),
                           int2(0), int2(int(W) - 1, int(H) - 1));
            float3 s = RGBtoYCoCg(float3(currentHDR.read(uint2(c)).rgb));
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    // Minimum sigma floor prevents the variance window from collapsing to zero
    // on smooth gradients. Without this, mean ± γ·σ ≈ mean ± 0 on flat surfaces
    // → history snaps hard to mean → step banding on smooth colour gradients.
    // 0.04 in YCoCg ≈ ~1/25 of the signal range; small enough to not ghost,
    // large enough to keep smooth surfaces smooth.
    float3 sigma = max(sqrt(max(float3(0.0), m2 - m1 * m1)), float3(0.04));

    // γ = 1.5 — slightly wider than the original 1.25 to reduce over-clamping on
    // textured surfaces that look flat from the 3×3 window but aren't.
    const float gamma = 1.5;
    float3 minC = m1 - gamma * sigma;
    float3 maxC = m1 + gamma * sigma;

    // Clip (not per-channel clamp) the history into the neighbourhood AABB along
    // the centre→history ray — kills ghosts without the hue shift a per-channel
    // clamp introduces on reprojected reflections / GI that lack a true velocity.
    historyYCoCg = clipHistoryToAABB(minC, maxC, historyYCoCg);

    // ── Velocity ramp ───────────────────────────────────────────────────────
    // Ramp toward mostly-current as screen-space motion grows. A fast mover's
    // history is unreliable — even with an exact motion vector, the Catmull-Rom
    // history resample carries sub-pixel error that compounds frame-over-frame
    // into a smear. The old cap (4×blend ≈ 0.2) kept ~80% history on fast movers
    // and was the dominant cause of the "streaky" look; 0.85 clears the moving
    // pixels while static pixels keep the low blend (so SSAO/SSR/GI still denoise
    // and static edges still converge). Pairs with the depth-disocclusion term
    // below, which handles the TRAILING pixels the mover vacated.
    //
    // NOTE `alpha` and `frame.taaHistoryBlend` are both weights of the CURRENT
    // sample (`wC = alpha`, `wH = 1 - alpha` below) — the uniform's name is a
    // misnomer it is too widely referenced to rename here. So the ramp target
    // must be a FLOOR, not a fixed value: `mix(blend, 0.85, disocc)` ramps the
    // wrong way whenever `taaHistoryBlend > 0.85` (reachable — Daydream Home's
    // Noise Reduction dial maps n < 0.156 to blend > 0.85), handing a moving or
    // newly-revealed pixel MORE history than a stable one, which is the exact
    // inverse of what disocclusion rejection is for. Taking `max` keeps the
    // ramp monotonically non-decreasing in `disoccBlend`: motion may only ever
    // ADD current-frame weight, never remove it. Identical to the old
    // expression for every blend ≤ 0.85 (the 0.05–0.10 tuned range), so the
    // shipped look is unchanged. The sibling SSAO/SSR/GI resolves already ramp
    // toward `alpha * k`, which is monotone by construction.
    float velMag = length(vel);
    float disoccBlend = smoothstep(0.0015, 0.015, velMag);
    float velocityFloor = max(frame.taaHistoryBlend, 0.85);
    float alpha = mix(frame.taaHistoryBlend, velocityFloor, disoccBlend);
    alpha = clamp(alpha, 0.01, 1.0);

    // ── Specular-ghost rejection (Phase 4.44) ────────────────────────────────
    // Variance clipping alone can't catch a specular highlight on a fast mover:
    // the highlight's SCREEN motion ≠ the surface motion vector, so reprojection
    // lands on a stale highlight, and where the *current* neighbourhood is also
    // bright (σ large) the mean ± γ·σ window is wide enough to pass the ghost.
    // Detect it as a luminance DISAGREEMENT between the (clamped) history Y and
    // the current Y — and gate it on velocity, because a STATIC high-contrast
    // edge must still converge/AA (that's the whole point of TAA). On a still
    // camera velMag ≈ 0 → specReject ≈ 0 → this is a no-op.
    float yH = historyYCoCg.x, yC = currentYCoCg.x;
    float lumaDisagree = abs(yH - yC) / (max(yH, yC) + 0.2);
    float specReject   = lumaDisagree * smoothstep(0.004, 0.02, velMag);
    alpha = clamp(mix(alpha, 1.0, 0.5 * specReject), 0.01, 1.0);

    // Disocclusion: force (almost) full current weight where last frame held a
    // different surface — kills fast-mover trails over similar-luma backgrounds.
    alpha = clamp(mix(alpha, 1.0, 0.9 * depthDisocc), 0.01, 1.0);

    // ── HDR luma-weighted (anti-flicker) blend ────────────────────────────────
    // Weight each sample by 1/(1+luma) so a bright firefly in the reprojected
    // history can't dominate the average and smear a trail — the single biggest
    // reduction in streaking on emissive / specular content, where reprojection
    // is least reliable. Reduces to the plain `mix` for dark pixels. (Karis
    // anti-flicker weighting; blend done in YCoCg, weighted by the Y term.)
    float wC = alpha         * (1.0 / (1.0 + max(0.0, currentYCoCg.x)));
    float wH = (1.0 - alpha) * (1.0 / (1.0 + max(0.0, historyYCoCg.x)));
    float3 resultYCoCg = (currentYCoCg * wC + historyYCoCg * wH) / max(1e-5, wC + wH);

    // ── Mild luma sharpen ─────────────────────────────────────────────────────
    // Catmull-Rom history + temporal averaging soften the image; restore a touch
    // of acutance by boosting the result luma toward the current-frame high-pass
    // (current − neighbourhood mean). Luma-only so it can't add chroma fringing;
    // clamped to the neighbourhood luma range so it can't reintroduce a ghost.
    resultYCoCg.x += 0.25 * (currentYCoCg.x - m1.x);
    resultYCoCg.x  = clamp(resultYCoCg.x, minC.x, maxC.x);

    float3 result = max(float3(0.0), YCoCgtoRGB(resultYCoCg));
    outHDR.write(half4(half3(result), 1.0h), gid);
}
