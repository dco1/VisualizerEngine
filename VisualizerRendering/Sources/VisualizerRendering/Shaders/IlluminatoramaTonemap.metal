// ── ILLUMINATORAMA — EXPOSURE AND TONEMAP ───────────────────────────────────
//
// Auto-exposure estimation and the final tonemap/grade fragment: ACES, white
// balance, tone curve, film LUT, grain and lens FX.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Auto-exposure (Phase 4.21) ───────────────────────────────────────────────
//
// One threadgroup of 256 threads scans the HDR target on a stride pattern
// (every ~8th pixel in each axis), accumulates log-luminance into shared
// memory, threadgroup-reduces, and EMAs the result into a tiny shared
// MTLBuffer the tonemap then reads its `exposure` from. The whole loop is
// GPU-only — no `getBytes`, no `waitUntilCompleted`, no CPU↔GPU sync. The
// EMA absorbs noise so a single bright pixel doesn't make the scene darken
// for a frame; the smoothing constant adapts to the per-frame `dt` the
// host hands in.
//
// Why log-luminance: the human visual system responds to log brightness,
// not linear, and so do every published auto-exposure paper (Reinhard 2002
// onwards). Averaging in log space then exponentiating gives the geometric
// mean, which is a much better "what brightness is this scene" signal
// than the arithmetic mean (which gets pulled to the sky's blown-out
// highlights). Result: outdoor scenes with a bright sky don't crush
// midtones; indoor scenes don't overbrighten because the ceiling lights
// average a few stops above the rest of the room.
//
// Buffer layout (4 floats, see ExposureState struct below):
//   [0] previous-frame target log-luminance (kernel writes here, kept
//       between frames so the EMA has a reference)
//   [1] previous-frame smoothed exposure (the value the tonemap reads)
//   [2] target log-luminance just-computed this frame (debug + diagnostic)
//   [3] per-frame dt the host hands in (controls EMA speed)

struct ExposureState {
    float prevTargetLogLum;
    float smoothedExposure;
    float newTargetLogLum;  // written each frame, surfaced for diagnostics
    float deltaTime;        // host-driven, in seconds
};

// Single threadgroup, 256 threads. Each thread strides across the image
// reading every Nth pixel, accumulates log-luminance into a per-thread
// local sum, then threadgroup-reduces via shared memory.
//
// Sample density: ~32 samples per thread × 256 threads = ~8K samples.
// For a 1920×1080 image that's 1 in ~250 pixels — plenty for a stable
// luminance estimate that doesn't trigger on outlier bright pixels.

kernel void illumi_exposure_estimate(
    texture2d<half, access::sample> inHDR     [[texture(0)]],
    device ExposureState&           state     [[buffer(0)]],
    constant uint2&                 imgSize   [[buffer(1)]],
    constant float4&                params    [[buffer(2)]],  // x=targetEV, y=halfLife, z=maxBoost, w=minBoost
    constant float4&                params2   [[buffer(3)]],  // x=highlightProtection, y=highlightEV, zw reserved
    threadgroup float*              sharedAcc [[threadgroup(0)]],
    threadgroup uint*               sharedCnt [[threadgroup(1)]],
    uint                            tid       [[thread_position_in_threadgroup]],
    uint                            tgSize    [[threads_per_threadgroup]]
) {
    constexpr sampler s(filter::linear,
                        address::clamp_to_edge,
                        coord::normalized);

    // Each thread strides through the image. Total samples per thread
    // is bounded so big images don't slow us down quadratically. Sample
    // log-luminance only when it's finite and above a floor — very dark
    // pixels (< -8 EV ≈ 0.004 linear) get clamped to the floor so a
    // few opaque black pixels don't drag the average into deep shadow.
    float thisAcc = 0.0;
    uint  thisCnt = 0;
    // Phase S5 — the highlight pass re-walks these same samples once the mean is
    // known, so keep them rather than re-sampling the texture a second time.
    float thisLL[32];
    uint  thisKept = 0;
    const float minLogLum = -8.0;   // ~0.004 linear
    const float maxLogLum =  8.0;   // ~3000 linear (no over-bright HDR)
    const uint samplesPerThread = 32;
    // Stride pattern: tid handles a regular grid across the image. We
    // pick samples by interpreting `tid + k*tgSize` as a 1D index over
    // a coarse (W/8 × H/8) grid — 8x downsample, then 32 such samples
    // per thread covers ~half of the coarse grid every frame.
    uint coarseW = max(1u, imgSize.x / 8u);
    uint coarseH = max(1u, imgSize.y / 8u);
    uint coarseN = coarseW * coarseH;
    for (uint k = 0; k < samplesPerThread; ++k) {
        uint idx = (tid + k * tgSize) % coarseN;
        uint cx = idx % coarseW;
        uint cy = idx / coarseW;
        // Sample at the centre of the coarse cell, in normalised UV
        // so the kernel doesn't have to know whether HDR is at the
        // internal or output resolution.
        float2 uv = (float2(cx, cy) + 0.5) / float2(coarseW, coarseH);
        float3 rgb = float3(inHDR.sample(s, uv).rgb);
        // Phase 4.30 — metering brightness that accounts for the MAX channel,
        // not just Rec.709 luminance. Pure luma badly under-weights saturated
        // colours (a clipping-bright red reads as only 0.21·R of luma), so a
        // red-dominated scene (Pizza's warm oven IBL × red sauce) metered as
        // "dark" and auto-exposure pushed it brighter — driving the red
        // channel past 1 into a flat clip, and washing the whole frame. Take
        // the larger of luma and half the max channel: a saturated bright
        // channel now lifts the metered brightness (→ less boost, no clip),
        // while a genuinely dark scene (low max channel too) still boosts.
        float lum  = dot(rgb, float3(0.2126, 0.7152, 0.0722));
        float maxc = max(rgb.r, max(rgb.g, rgb.b));
        lum = max(lum, 0.5 * maxc);
        if (isfinite(lum) && lum > 0.0) {
            float ll = log2(lum);
            ll = clamp(ll, minLogLum, maxLogLum);
            thisAcc += ll;
            thisCnt += 1u;
            thisLL[thisKept++] = ll;
        }
    }
    sharedAcc[tid] = thisAcc;
    sharedCnt[tid] = thisCnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory.
    for (uint stride = tgSize / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sharedAcc[tid] += sharedAcc[tid + stride];
            sharedCnt[tid] += sharedCnt[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Highlight-protecting meter (opt-in; `params2.x` 0 ⇒ never runs) ──────
    //
    // The mean above is a GEOMETRIC mean of the whole frame, which is the right
    // "how bright is this scene" signal and the wrong "will anything clip" signal.
    // A doll's-house cutaway is half shaded interior: the mean sits low, exposure
    // pumps to lift it, and the sunlit exterior — several stops above the mean —
    // is pushed past the shoulder into flat white. Metering the mean alone cannot
    // see that, because the pixels it blows are exactly the ones it averaged away.
    //
    // So: take a second statistic — the mean log-luminance of the samples ABOVE
    // the frame mean, i.e. the upper half — and cap the exposure so THAT lands at
    // `highlightEV` rather than wherever the mean-based answer puts it. The cap is
    // a `min`, never a boost: it can only ever pull a frame back from clipping, so
    // a scene with no bright half is untouched.
    //
    // Upper-half mean rather than a true p99: the reduction is already here, one
    // more pass over the SAME 32 cached samples costs no texture reads, and a
    // percentile would need a sort or a histogram. The upper half is also the more
    // stable signal frame to frame — a p99 chases single specular pixels.
    // Every thread derives the mean from the ALREADY-REDUCED shared slots rather
    // than one thread broadcasting through a new threadgroup variable: same value,
    // no second barrier, and no uninitialised-read for the compiler to flag. The
    // whole-image-black fallback lives here so the highlight pass below and the
    // EMA at the bottom agree on what "the frame mean" was.
    float frameMean = (sharedCnt[0] > 0u)
                    ? (sharedAcc[0] / float(sharedCnt[0]))
                    : state.prevTargetLogLum;

    float hiAcc = 0.0;
    uint  hiCnt = 0u;
    if (params2.x > 0.0) {
        for (uint k = 0; k < thisKept; ++k) {
            if (thisLL[k] > frameMean) { hiAcc += thisLL[k]; hiCnt += 1u; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sharedAcc[tid] = hiAcc;
    sharedCnt[tid] = hiCnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sharedAcc[tid] += sharedAcc[tid + stride];
            sharedCnt[tid] += sharedCnt[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint  hiN    = sharedCnt[0];
        float hiMean = (hiN > 0u) ? (sharedAcc[0] / float(hiN)) : frameMean;
        // `frameMean` already carries the whole-image-black fallback (it resolves
        // to `state.prevTargetLogLum` when no sample landed in the valid range),
        // because the highlight pass has to see the same number the mean pass did.
        float target = frameMean;
        // Auto-exposure: we want the target luminance to land at
        // `2^targetEV`. So the exposure scalar is `2^(targetEV - target)`.
        // Negative target log lum (scene is dim) → positive exposure
        // boost. Positive target (scene is bright) → exposure compression.
        float targetEV = params.x;
        float wantedExposure = exp2(targetEV - target);
        // Highlight cap. `protect` 0 ⇒ `mix(w, w, 0)` ⇒ bit-identical to the
        // mean-only answer, so this is an exact no-op for every non-opting scene.
        float protect = clamp(params2.x, 0.0, 1.0);
        if (protect > 0.0) {
            float hiWanted = exp2(params2.y - hiMean);
            wantedExposure = mix(wantedExposure, min(wantedExposure, hiWanted), protect);
        }
        // EMA toward `wantedExposure` with a half-life set by
        // `params.y` seconds. Convert half-life + dt into a per-frame
        // mix factor: `alpha = 1 - 2^(-dt / halfLife)`.
        float halfLife = max(params.y, 1.0 / 60.0);
        float dt = max(state.deltaTime, 1.0 / 240.0);
        float alpha = 1.0 - exp2(-dt / halfLife);
        float prev = state.smoothedExposure;
        // Clamp via the host-provided min/max boost — for dark scenes
        // we don't want auto-exposure to fabricate light into mid-grey
        // (Eggs, FloatingFlowers+ overshoot at 4-6×), and for bright
        // scenes we don't want it to crush below a sensible floor.
        // Defaults (host-side) are ~3× max boost / 0.25× min — that
        // gives ±1.5 stops of headroom, which is the SCN feel.
        float maxBoost = max(params.z, 0.05);
        float minBoost = max(params.w, 0.01);
        wantedExposure = clamp(wantedExposure, minBoost, maxBoost);
        float next = mix(prev, wantedExposure, alpha);
        state.smoothedExposure   = next;
        state.prevTargetLogLum   = target;
        state.newTargetLogLum    = target;
    }
}

// ── Tonemap + composite ──────────────────────────────────────────────────────
//
// ACES filmic curve (Krzysztof Narkowicz's approximation). Reads HDR + bloom,
// applies exposure, ACES-tonemaps, gamma-encodes, writes final LDR.

static inline float3 aces(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ── Color-grade: white-balance gain from a Kelvin temperature ───────────────
// Maps a correlated colour temperature (~2000–10000 K) to a normalized linear
// RGB channel gain that, when MULTIPLIED into a neutral scene, warms it (low K)
// or cools it (high K). 6500 K → (1,1,1) exactly (no-op default). We use a
// cheap polynomial approximation of the daylight locus' channel response
// rather than a full Planckian/CIE conversion — it only has to read tasteful
// across the slider, not be colorimetrically exact. Normalized so the green
// channel (and the luma) stays ≈1, i.e. the grade tints rather than dims.
static inline float3 whiteBalanceGain(float kelvin) {
    // Reference is 6500 K (D65). Below → push red, pull blue (warm); above →
    // push blue, pull red (cool). A smooth, monotonic curve in 1000s-of-K.
    float t = (kelvin - 6500.0) / 6500.0;        // 0 at D65; ~-0.69 at 2000 K; ~+0.54 at 10000 K
    float r = 1.0 - 0.45 * t;                    // warmer (low K) → more red
    float b = 1.0 + 0.55 * t;                    // warmer (low K) → less blue
    float g = 1.0 - 0.04 * t * t;                // slight green dip away from D65
    float3 gain = float3(max(r, 0.0), max(g, 0.0), max(b, 0.0));
    // Renormalize to unit luma so the white-balance only shifts hue, not exposure.
    float lum = dot(gain, float3(0.2126, 0.7152, 0.0722));
    return (lum > 1e-4) ? gain / lum : float3(1.0);
}

// Green↔magenta tint on the [-1, 1] axis. tint > 0 pushes magenta (boost R+B,
// cut G); tint < 0 pushes green. Luma-preserving by construction (the green
// move is twice the magenta half-move). 0 = no-op.
static inline float3 tintGain(float tint) {
    float g = 1.0 - 0.20 * tint;                 // magenta (tint>0) cuts green
    float rb = 1.0 + 0.10 * tint;                // ...and lifts red+blue
    return float3(rb, g, rb);
}

// Tonemapped-domain tone curve. Operates on a 0..1 LDR colour:
//   • contrast pivots around mid-grey 0.18 (1.0 = no-op)
//   • shadows lifts (>1) / crushes (<1) the low-luma end (1.0 = no-op)
//   • highlights lifts/rolls the high-luma end (1.0 = no-op)
// Shadows/highlights are luma-weighted so mid-tones stay put and the two ends
// move independently. All three default to 1.0 → an exact no-op.
static inline float3 toneCurve(float3 c, float contrast, float shadows, float highlights) {
    // Contrast around mid-grey pivot.
    const float pivot = 0.18;
    c = (c - pivot) * contrast + pivot;
    c = max(c, 0.0);
    // Per-pixel luma drives the shadow/highlight weights.
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    // Smooth low/high masks: shadowW ≈ 1 in blacks → 0 by mid; highW the inverse.
    float shadowW = 1.0 - smoothstep(0.0, 0.5, lum);
    float highW   = smoothstep(0.5, 1.0, lum);
    // Multiplicative lift/crush — keeps hue, scales magnitude per region.
    float scale = mix(1.0, shadows, shadowW) * mix(1.0, highlights, highW);
    return max(c * scale, 0.0);
}

// SSAA downsample. The HDR + bloom textures are sized to the INTERNAL
// resolution (output × `internalRenderScale`); this kernel dispatches over
// the OUTPUT resolution and runs a 4-tap bilinear-hardware downsample per
// output pixel. Each bilinear tap is a "free" 2×2 input-texel average
// thanks to `filter::linear`; placing the 4 taps at ±0.5 input-texel
// offsets around the output pixel's centre covers a 3×3 input-texel
// footprint with a smooth, ringing-free tent kernel — exactly the shape
// you want for 1.5× downscale. At `internalRenderScale == 1.0` the four
// taps collapse to a single bilinear read at the texel centre, so the
// path is correct (if slightly more expensive than the original
// `read(gid)`) at the no-SSAA setting too.
// Phase 4.28 — the tonemap is a RENDER PASS (fullscreen triangle + fragment),
// not a compute kernel. The output texture is sampled by SceneKit as
// `background.contents` on its OWN command queue; a compute `texture.write`
// to a `.private` texture is not reliably made visible to that cross-queue
// read without a CPU completion wait (which would stall the overlay's main-
// thread tick). A render pass with `storeAction = .store` resolves/stores the
// attachment at end-of-pass, which IS visible to the subsequent SceneKit
// sample — same mechanism the (working) `blankSky` clear relies on. The
// tonemap math is unchanged; only the dispatch shape moved from grid threads
// to a fragment over the output attachment.

struct TonemapVSOut {
    float4 position [[position]];
    float2 uv;
};

vertex TonemapVSOut illumi_tonemap_vs(uint vid [[vertex_id]]) {
    // Oversized fullscreen triangle covering the viewport in one primitive:
    // verts (0,0)->(-1,-1 NDC), (2,0)->(3,-1), (0,2)->(-1,3). The UV is in
    // [0,1] across the screen (y flipped so v=0 is the top, matching the
    // texture sampling the compute path used).
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    TonemapVSOut o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

// ── Diagram look: flat fills + screen-space outlines ─────────────────────────
//
// Turns the G-buffer into an architectural drawing: flat albedo fills shaded only
// by orientation, with ink lines along depth and normal discontinuities. It reads
// the SAME three G-buffer textures the debug-view readouts already bind (albedo,
// oct-normal, depth), so the whole look costs one branch in this fragment — no
// extra pass, no extra attachment, no second forward path.
//
// Two deliberate choices, both about what a drawing is NOT:
//
//   • The fill ignores every light in the scene. A drawing's tone comes from which
//     way a face points, not from where a lamp is — so a fixed "paper light"
//     direction gives the three face families of an axonometric (top / left-facing
//     / right-facing) three distinct tones that hold still as the sun moves.
//
//   • Edges are found from the SECOND difference of linear view depth, not the
//     first. A first difference is large wherever a surface is merely oblique, so
//     a floor seen at a grazing angle inks over solid black; the second difference
//     is ~0 across any flat plane at ANY slope and spikes only where the surface
//     actually breaks — which is the definition of the line we want.
//
// Depth is linearised through `frame.invProjection`, so this is correct under both
// projections; nothing here assumes a perspective frustum.
struct IllumiDiagramTap { float z; float3 n; };

/// One cross-tap: linear view-space depth (positive metres from the eye plane) plus
/// the world normal, at a pixel offset from `uv`. Sky taps read back at ~zFar, which
/// is what makes the building's silhouette against open sky the strongest edge in
/// the frame — no separate sky mask needed.
static inline IllumiDiagramTap illumiDiagramTap(float2 uv, float2 offsetPx,
                                                float2 gSize, float2 dSize,
                                                constant FrameUniforms& frame,
                                                texture2d<half, access::read> gNormalRgh,
                                                depth2d<float,  access::read> gDepth)
{
    float2 tuv = clamp(uv + offsetPx / dSize, 0.0, 0.99999);
    float  d   = gDepth.read(uint2(tuv * dSize));
    float2 ndc = float2(tuv.x * 2.0 - 1.0, 1.0 - tuv.y * 2.0);
    IllumiDiagramTap t;
    t.z = abs(viewPosFromDepth(ndc, d, frame.invProjection).z);
    t.n = octDecode(float2(gNormalRgh.read(uint2(tuv * gSize)).rg));
    return t;
}

static inline float3 illumiDiagramColor(float2 uv,
                                        constant FrameUniforms& frame,
                                        texture2d<half, access::read> gAlbedoMet,
                                        texture2d<half, access::read> gNormalRgh,
                                        depth2d<float,  access::read> gDepth)
{
    float2 gSize = float2(gAlbedoMet.get_width(), gAlbedoMet.get_height());
    float2 dSize = float2(gDepth.get_width(), gDepth.get_height());

    // Sampling the depth texture at a pixel OFFSET means working in its texel grid;
    // `thickness` is specified in output pixels, which is the same grid here (the
    // G-buffer is at internal resolution and this fragment runs per output pixel).
    float thick = max(0.25, frame.diagramEdge.z);

    IllumiDiagramTap c = illumiDiagramTap(uv, float2( 0.0,   0.0),   gSize, dSize, frame, gNormalRgh, gDepth);
    IllumiDiagramTap l = illumiDiagramTap(uv, float2(-thick, 0.0),   gSize, dSize, frame, gNormalRgh, gDepth);
    IllumiDiagramTap r = illumiDiagramTap(uv, float2( thick, 0.0),   gSize, dSize, frame, gNormalRgh, gDepth);
    IllumiDiagramTap u = illumiDiagramTap(uv, float2( 0.0,  -thick), gSize, dSize, frame, gNormalRgh, gDepth);
    IllumiDiagramTap d = illumiDiagramTap(uv, float2( 0.0,   thick), gSize, dSize, frame, gNormalRgh, gDepth);

    // ── Fill ────────────────────────────────────────────────────────────────
    uint2  gp     = uint2(clamp(uv, 0.0, 0.99999) * gSize);
    float3 albedo = float3(gAlbedoMet.read(gp).rgb);
    // A fixed studio direction, well off-axis in all three components so no two
    // face families of an axonometric box land on the same tone.
    const float3 paperLight = normalize(float3(0.36, 0.86, 0.38));
    float  key   = 0.5 + 0.5 * dot(c.n, paperLight);
    float  wrap  = clamp(frame.diagramParams.y, 0.0, 1.0);
    // wrap = 0 ⇒ `shade` is exactly 1.0 ⇒ perfectly flat fills.
    float  shade = mix(1.0, mix(0.62, 1.0, key), wrap);
    float3 fill  = albedo * shade;

    // Open sky is PAPER — the drawing sits on a white page, not under a gradient.
    // Depth 1.0 is the far plane in this [0,1] convention (near = 0), so a pixel
    // nothing was rasterised into reads back at the clear value.
    float  dC    = gDepth.read(uint2(clamp(uv, 0.0, 0.99999) * dSize));
    const float3 paper = float3(1.0);
    float3 base  = (dC >= 0.99999) ? paper : fill;

    // ── Ink ─────────────────────────────────────────────────────────────────
    // Second difference of linear depth, normalised by distance so one threshold
    // works at every zoom (a 2 cm step reads as an edge up close and, correctly,
    // does not from across the yard).
    float depthSens  = max(1e-3, frame.diagramEdge.x);
    float normalSens = max(1e-3, frame.diagramEdge.y);
    float invZ       = 1.0 / max(c.z, 0.05);
    float ddx        = abs(l.z + r.z - 2.0 * c.z) * invZ;
    float ddy        = abs(u.z + d.z - 2.0 * c.z) * invZ;
    float depthEdge  = smoothstep(0.0, 0.006 / depthSens, max(ddx, ddy));

    // Normal discontinuity — the only thing that can see the crease where two
    // faces of one object meet at the same depth (a cabinet's corner, a wall
    // returning). Smooth curvature stays under the threshold and does not ink.
    float nDiff = max(max(1.0 - dot(c.n, l.n), 1.0 - dot(c.n, r.n)),
                      max(1.0 - dot(c.n, u.n), 1.0 - dot(c.n, d.n)));
    float normalEdge = smoothstep(0.06 / normalSens, 0.30 / normalSens, nDiff);

    float  edge = max(depthEdge, normalEdge) * clamp(frame.diagramParams.z, 0.0, 1.0);
    float3 ink  = frame.diagramOutline.rgb;
    return mix(base, ink, edge);
}

fragment float4 illumi_tonemap_fs(
    TonemapVSOut                    in       [[stage_in]],
    texture2d<half, access::sample> inHDR    [[texture(0)]],
    texture2d<half, access::sample> inBloom  [[texture(1)]],
    texture3d<half, access::sample> colorLUT [[texture(2)]],
    texture2d<half, access::sample> inVel    [[texture(3)]],   // screen-space velocity (UV/frame)
    // Issue #65 — G-buffer channels for the Debug-view readouts (texture 4..6).
    // Only read when `frame.debugTerm >= 10` (the G-buffer cases); a default
    // `.normal` render never touches them.
    texture2d<half,  access::read>  gAlbedoMet [[texture(4)]], // .rgb albedo, .a metalness
    texture2d<half,  access::read>  gNormalRgh [[texture(5)]], // .xy oct-normal, .z roughness
    depth2d<float,   access::read>  gDepth     [[texture(6)]],
    // Phase 9 — film-stock LUT: 16×16×16 3D texture (B slices left-to-right in
    // the 256×16 PNG strip). Bound when filmLUTStrength > 0; nil-checked below.
    // (texture(7): 2..6 were taken by colorLUT/inVel/G-buffer in the merge.)
    texture3d<float, access::sample> filmLUT  [[texture(7)]],
    // Quarter-res halation halo. Always bound (the texture exists); its contents are
    // stale when halationParams.x == 0, and the branch below never reads it then.
    texture2d<half, access::sample> inHalation [[texture(8)]],
    // Half-res AO field for `DebugTerm.ssao` (16) — the same texture the lighting
    // pass multiplies its indirect terms by. Only read when debugTerm == 16.
    texture2d<half, access::sample> inAO     [[texture(9)]],
    constant FrameUniforms&         frame    [[buffer(0)]],
    // Phase 4.21 — auto-exposure read (see below).
    const device ExposureState&     expoState [[buffer(1)]]
) {
    constexpr sampler downSampler(filter::linear,
                                  address::clamp_to_edge,
                                  coord::normalized);

    float2 inSize  = float2(inHDR.get_width(), inHDR.get_height());
    float2 invInSize = 1.0 / inSize;

    // ── Velocity / motion-vector debug overlay (issue #65) ───────────────────────
    // Sentinel: a NEGATIVE motionBlurStrength (set by the host only when
    // VIZ_ILLUMI_MV_DEBUG=1) replaces the image with a colourised view of the
    // screen-space velocity buffer — the "per-object velocity overlay" validator.
    // Grey = no motion; red/green shift = +x/+y screen motion; darkening = speed.
    // The ONLY reliable way to confirm which objects write motion vectors (camera,
    // ping-ponged instances) and which don't (MLS-MPM/fluid surfaces) before TAA or
    // motion blur start trusting them. Costs nothing in normal (strength ≥ 0) renders.
    if (frame.motionBlurStrength < 0.0) {
        constexpr sampler velSampler(filter::linear, address::clamp_to_edge, coord::normalized);
        float2 v   = float2(inVel.sample(velSampler, in.uv).rg);
        float  mag = length(v);
        float2 a   = v * 40.0;                       // amplify (per-frame UV motion is tiny)
        float3 c   = float3(0.5 + a.x, 0.5 + a.y, 0.5 - 0.5 * saturate(mag * 40.0));
        return float4(saturate(c), 1.0);
    }

    // ── G-buffer debug views (issue #65) ─────────────────────────────────────────
    // `debugTerm >= 10` replaces the composite with a RAW G-buffer channel, read
    // BEFORE tonemapping so the readout is the true stored value. Cases 0–9 are the
    // deferred-lighting-term isolations handled in `illumi_lighting` (its switch
    // defaults to the full composite for >= 10), so this branch is the ONLY place
    // these draw — and an exact no-op when `debugTerm < 10` (every scene's default
    // is 0). Nearest-texel reads (no filtering) for a faithful per-pixel channel view.
    if (frame.debugTerm >= 10u) {
        uint2 gp = uint2(in.uv * float2(gAlbedoMet.get_width(), gAlbedoMet.get_height()));
        switch (frame.debugTerm) {
            case 10u: // Albedo — base colour (linear).
                return float4(float3(gAlbedoMet.read(gp).rgb), 1.0);
            case 11u: { // World normal, remapped from [-1,1] to [0,1].
                float3 N = octDecode(float2(gNormalRgh.read(gp).rg));
                return float4(N * 0.5 + 0.5, 1.0);
            }
            case 12u: { // Roughness (greyscale).
                float r = float(gNormalRgh.read(gp).b);
                return float4(r, r, r, 1.0);
            }
            case 13u: { // Metalness (greyscale).
                float m = float(gAlbedoMet.read(gp).a);
                return float4(m, m, m, 1.0);
            }
            case 14u: { // Depth — non-linear NDC z, ramped so the foreground is legible.
                uint2 dp = uint2(in.uv * float2(gDepth.get_width(), gDepth.get_height()));
                float d  = saturate(gDepth.read(dp));
                // depth32 crushes most of a perspective scene near z=1; pow() spreads
                // the high range, and the invert makes near = white / far = black.
                float vis = 1.0 - pow(d, 24.0);
                return float4(vis, vis, vis, 1.0);
            }
            case 15u: { // Screen-space velocity — same colourisation as the MV overlay.
                constexpr sampler velS(filter::linear, address::clamp_to_edge, coord::normalized);
                float2 v   = float2(inVel.sample(velS, in.uv).rg);
                float  mag = length(v);
                float2 a   = v * 40.0;
                float3 c   = float3(0.5 + a.x, 0.5 + a.y, 0.5 - 0.5 * saturate(mag * 40.0));
                return float4(saturate(c), 1.0);
            }
            case 16u: { // RAW AO field — the multiplier, pre-tonemap. 1 = unoccluded.
                constexpr sampler aoS(filter::linear, address::clamp_to_edge, coord::normalized);
                float a = float(inAO.sample(aoS, in.uv).r);
                return float4(a, a, a, 1.0);
            }
            default: break;
        }
    }

    // ── Diagram look ─────────────────────────────────────────────────────────
    // At FULL strength the drawing replaces the image outright, so return here and
    // skip the entire tone chain below — exposure, ACES, grade, grain, LUT and the
    // lens FX all operate on a photograph this mode isn't making. That is the common
    // case and it makes the mode cheaper than a photoreal frame rather than dearer.
    //
    // Below full strength the drawing has to cross-fade against the finished image,
    // which does not exist yet — so that case falls through and blends at the end.
    // `mix == 0` (every scene's default) enters neither branch: exact no-op.
    if (frame.diagramParams.x >= 0.999) {
        return float4(illumiDiagramColor(in.uv, frame, gAlbedoMet, gNormalRgh, gDepth), 1.0);
    }

    // `in.uv` is the output pixel centre in normalised coords, equivalent to
    // the compute path's `(gid+0.5)/outSize`. The 4-tap offsets are ±0.5
    // INPUT texels, i.e. ±0.5*invInSize in normalised space.
    const float2 offsets[4] = {
        float2(-0.5, -0.5), float2( 0.5, -0.5),
        float2(-0.5,  0.5), float2( 0.5,  0.5)
    };
    float3 hdr = float3(0.0);
    if (frame.motionBlurStrength > 0.0) {
        // ── Velocity-buffer motion blur (issue #65) ──────────────────────────────
        // Gather inHDR along the per-frame screen velocity — a camera-shutter streak.
        // `inVel` stores (currNDC − prevNDC)·(0.5, −0.5) in UV, written by the
        // G-buffer pass from the CURRENT and PREVIOUS view-projection AND the
        // ping-ponged previous-instance model matrix, so it carries per-object motion
        // (spin/translate) as well as camera motion. Symmetric N-tap centred on the
        // pixel; the streak length scales with `motionBlurStrength` (≈ shutter
        // fraction) and is clamped to `motionBlurMaxPx`. 0 strength skips this branch,
        // preserving the exact CA / 4-tap path below.
        float2 vel = float2(inVel.sample(downSampler, in.uv).rg) * frame.motionBlurStrength;
        float  maxUV = max(0.0, frame.motionBlurMaxPx) * max(invInSize.x, invInSize.y);
        float  vlen = length(vel);
        if (vlen > maxUV && vlen > 1e-6) vel *= maxUV / vlen;
        const int MB = 9;
        float3 acc = float3(0.0);
        for (int s = 0; s < MB; ++s) {
            float t = (float(s) / float(MB - 1)) - 0.5;   // [-0.5, 0.5] across the streak
            acc += float3(inHDR.sample(downSampler, in.uv + vel * t).rgb);
        }
        hdr = acc / float(MB);
    } else if (frame.chromaticAberration > 0.0) {
        // Lens-style transverse chromatic aberration: split the red channel
        // outward and the blue channel inward along the radius from the frame
        // centre, growing toward the edges (zero in the middle, like a real
        // lens). `0.03` maps a strength of 1.0 to a ~1.5%-of-frame split at the
        // corners. Reduces to the plain 4-tap when the strength is 0 (gated
        // above so the default-off path keeps its 4 samples).
        float2 caOff = (in.uv - 0.5) * (frame.chromaticAberration * 0.03);
        for (int i = 0; i < 4; ++i) {
            float2 uv = in.uv + offsets[i] * invInSize;
            hdr.r += float(inHDR.sample(downSampler, uv + caOff).r);
            hdr.g += float(inHDR.sample(downSampler, uv).g);
            hdr.b += float(inHDR.sample(downSampler, uv - caOff).b);
        }
        hdr *= 0.25;
    } else {
        for (int i = 0; i < 4; ++i) {
            float2 uv = in.uv + offsets[i] * invInSize;
            hdr += float3(inHDR.sample(downSampler, uv).rgb);
        }
        hdr *= 0.25;
    }

    // ── Spherical aberration ───────────────────────────────────────────────────
    // A real lens's outer zones focus at a slightly different plane than the
    // paraxial rays, so off-axis detail loses sharpness while the centre stays
    // crisp — a radial defocus that grows quadratically from the optical axis
    // toward the edges. We blur all channels equally (unlike the transverse CA
    // above, which splits them) in the HDR domain so the softening wraps around
    // bright highlights before tonemapping, reading as the classic dreamy
    // soft-focus halation. 0 → branch skipped (exact no-op).
    if (frame.sphericalAberration > 0.0) {
        float2 d     = in.uv - 0.5;
        float  rNorm = saturate(dot(d, d) * 2.0);   // 0 at centre → 1 at corners
        // Radius (in INTERNAL-resolution texels) scales with both the off-axis
        // distance and the strength. At SA=1 the corners pull a ~14-texel disc;
        // at the SA=3 max it's ~42, a heavy soft-focus. Texel-space so the look
        // is stable across the internal render scale.
        float  blurPx = rNorm * frame.sphericalAberration * 14.0;
        if (blurPx > 0.25) {
            // Filled disc, not a ring: a centre tap plus two concentric 6-tap
            // rings (offset half a step) so the kernel covers the disc instead
            // of leaving a hard double-image. Outer ring at full radius, inner
            // at 55%, centre weighted highest → an approximately Gaussian falloff.
            float2 rOut = blurPx * invInSize;
            float2 rIn  = rOut * 0.55;
            float3 acc  = float3(inHDR.sample(downSampler, in.uv).rgb) * 3.0;
            float  wsum = 3.0;
            for (int i = 0; i < 6; ++i) {
                float aO = (float(i)        ) * (M_PI_F / 3.0);
                float aI = (float(i) + 0.5) * (M_PI_F / 3.0);
                acc += float3(inHDR.sample(downSampler, in.uv + float2(cos(aO), sin(aO)) * rOut).rgb) * 1.0;
                acc += float3(inHDR.sample(downSampler, in.uv + float2(cos(aI), sin(aI)) * rIn).rgb)  * 2.0;
            }
            wsum += 6.0 * 1.0 + 6.0 * 2.0;          // 3 + 6 + 12 = 21
            float3 blurred = acc / wsum;
            // Blend ramps with off-axis distance AND strength, so the centre is
            // always sharp and a higher slider both widens AND deepens the haze.
            float blend = saturate(rNorm * (0.55 + frame.sphericalAberration * 0.45));
            hdr = mix(hdr, blurred, blend);
        }
    }

    // Bloom is the TOP of the mip pyramid — still at half of the INTERNAL
    // resolution, still `rgba16Float`, so this binding is unchanged by S1.2.
    // The same normalised UV works directly because `filter::linear`
    // interpolates across mip0 regardless of the source's pixel count. The
    // pyramid's convex up-chain has a DC gain of 1, exactly like the normalised
    // gaussian it replaced, so `bloomIntensity` means what it always meant.
    float3 bloom = float3(inBloom.sample(downSampler, in.uv).rgb);

    float3 mixed = hdr + bloom * frame.bloomIntensity;

    // ── Halation (film) ────────────────────────────────────────────────────────
    // The wide warm halo film wears around blown highlights (see the halation
    // kernels above). Added in the HDR domain, like bloom, so exposure + ACES shape
    // it. The halo is driven mostly by LUMINANCE and takes its colour from
    // `halationTint` — real halation is red because red is what survives the round
    // trip through the anti-halation backing, whatever colour the highlight was —
    // with a quarter of the source hue left in so a strongly coloured highlight
    // still tints its own halo. Intensity 0 (the default) skips the branch.
    if (frame.halationParams.x > 0.0) {
        // Artistic normalisation. The threshold keeps only the EXCESS radiance above the
        // blown point, and the wide gaussian then averages that excess down by the fraction
        // of the kernel a highlight covers — so the raw halo is a small fraction of scene
        // radiance and a 0…1 dial would spend its whole range on "barely visible". This gain
        // makes `halationIntensity = 1.0` the intended full-strength look (the same
        // convention `lensFlareIntensity` uses); it is a constant, not a per-scene tuning.
        constexpr float kHalationGain = 3.0;
        float3 halo    = float3(inHalation.sample(downSampler, in.uv).rgb);
        float  haloLum = dot(halo, float3(0.2126, 0.7152, 0.0722));
        halo = mix(float3(haloLum), halo, 0.25) * frame.halationTint.rgb;
        mixed += halo * (frame.halationParams.x * kHalationGain);
    }

    // ── Lens flare (sun) ───────────────────────────────────────────────────────
    // Screen-space anamorphic streak + ghost train + halo, driven by the sun's
    // projected screen position (frame.lensFlareParams: x = strength, yz = sun uv,
    // w = on-screen weight). Occlusion is OPTICAL, not geometric: the HDR source is
    // sampled at the sun's uv — when a wall hides the sun those taps are stops
    // dimmer than open sky and the flare fades out, exactly like a real camera.
    // Added in the HDR domain so exposure/ACES/grade shape it naturally. Strength 0
    // (the default) skips the branch — non-opting scenes are byte-for-byte unchanged.
    if (frame.lensFlareParams.x > 0.0 && frame.lensFlareParams.w > 0.0) {
        float2 sunUV = frame.lensFlareParams.yz;
        float3 sunTap = float3(inHDR.sample(downSampler, sunUV).rgb);
        float2 tapR = 4.0 * invInSize;
        sunTap += float3(inHDR.sample(downSampler, sunUV + float2( tapR.x, 0)).rgb);
        sunTap += float3(inHDR.sample(downSampler, sunUV + float2(-tapR.x, 0)).rgb);
        sunTap += float3(inHDR.sample(downSampler, sunUV + float2(0,  tapR.y)).rgb);
        sunTap += float3(inHDR.sample(downSampler, sunUV + float2(0, -tapR.y)).rgb);
        float sunLuma = dot(sunTap * 0.2, float3(0.2126, 0.7152, 0.0722));
        float vis = smoothstep(0.35, 1.6, sunLuma) * frame.lensFlareParams.w;
        if (vis > 0.001) {
            float gain = frame.lensFlareParams.x * vis;
            float aspect = inSize.x / inSize.y;
            float2 asp = float2(aspect, 1.0);
            float2 toC = float2(0.5, 0.5) - sunUV;             // sun → frame centre
            float3 flare = float3(0.0);
            // Ghost train: tinted discs strung along the sun–centre axis (the
            // classic multi-element internal-reflection pattern).
            const float  gT[5]    = { 0.45, 0.85, 1.30, 1.70, 2.10 };
            const float  gR[5]    = { 0.020, 0.045, 0.032, 0.065, 0.028 };
            const float3 gTint[5] = { float3(1.00, 0.85, 0.60), float3(0.55, 0.75, 1.00),
                                      float3(1.00, 0.65, 0.45), float3(0.60, 0.90, 1.00),
                                      float3(0.95, 0.80, 1.00) };
            for (int g = 0; g < 5; ++g) {
                float2 pos = sunUV + toC * gT[g];
                float d = length((in.uv - pos) * asp);
                float disc = smoothstep(gR[g], gR[g] * 0.25, d);
                flare += gTint[g] * disc * (0.14 / (1.0 + gT[g]));
            }
            // Halo ring about the frame centre at the sun's mirrored radius.
            float haloR = clamp(length(toC * asp) * 0.8, 0.15, 0.45);
            float dC = length((in.uv - 0.5) * asp);
            float halo = exp(-pow((dC - haloR) * 18.0, 2.0));
            flare += float3(0.55, 0.75, 1.0) * halo * 0.065;
            // Anamorphic streak through the sun — tight vertically, long horizontally.
            float2 dS = (in.uv - sunUV) * asp;
            float streak = exp(-fabs(dS.y) * 90.0) * exp(-fabs(dS.x) * 4.5);
            flare += float3(0.65, 0.80, 1.0) * streak * 1.1;
            // Veiling glare around the sun itself.
            flare += float3(1.0, 0.92, 0.78) * exp(-length(dS) * 7.0) * 0.55;
            mixed += flare * gain;
        }
    }
    // Phase 4.21 — read the GPU-computed smoothed exposure from the
    // auto-exposure buffer when the host has the feature on; otherwise
    // fall back to the static scalar in FrameUniforms. The estimator
    // kernel ran earlier in the same frame's command buffer so the
    // value is fresh.
    //
    // `frame.exposure` (the "Exposure" slider) is a MULTIPLICATIVE
    // exposure-compensation applied on top of the auto-exposure result, so the
    // slider always has an effect — previously it was ignored whenever
    // auto-exposure was on (the default for every scene), which made the control
    // appear dead. In manual mode (auto off) the base is 1.0, so `base · exposure`
    // == the old `frame.exposure` behaviour (default slider 1.0 → no change).
    float autoBase = (frame.autoExposureEnabled != 0u)
                     ? expoState.smoothedExposure
                     : 1.0;
    float exposure = autoBase * frame.exposure;
    // ── Color-grade: white-balance + tint on LINEAR HDR, pre-tonemap ──────────
    // Channel-multiply gains are most physical in linear light (they model a
    // sensor/illuminant shift), so they go in before exposure + ACES. Defaults
    // (whiteBalanceK = 6500, tint = 0) make both gains exactly (1,1,1) → no-op.
    float3 graded = mixed * whiteBalanceGain(frame.whiteBalanceK) * tintGain(frame.tint);
    float3 mapped = aces(graded * exposure);
    // Phase 4.15 — post-tonemap saturation boost. Narkowicz's fitted ACES
    // famously compresses midtone chroma harder than SCN's HDR chain, so
    // the deferred pipeline reads consistently flatter than the SCN
    // baseline. Cheap post-curve fix: lerp from luminance back to the
    // tonemapped colour with `frame.tonemapSaturation` (typ. 1.15–1.3).
    // At 1.0 this is a no-op; above 1.0 it pushes chroma outward from
    // the per-pixel luminance, which preserves the filmic shoulder while
    // restoring tint. Rec.709 luminance weights are the standard choice.
    float lum = dot(mapped, float3(0.2126, 0.7152, 0.0722));
    mapped = max(mix(float3(lum), mapped, frame.tonemapSaturation), 0.0);
    // ── Scotopic (Purkinje) desaturation ──────────────────────────────────────
    // Human rods are colour-blind, so in dim light real vision loses chroma: a
    // moonlit lawn reads neutral-dark, not green — but the green ALBEDO × dim
    // near-neutral night light keeps our render green. Pull ONLY the dimmest
    // tonemapped pixels toward their own luminance, keyed to the display luma;
    // lamps/moon/stars stay bright enough (lum ≳ 0.08) to keep full colour. 0
    // (DEFAULT) ⇒ the branch never runs ⇒ day frames + every non-opting scene are
    // byte-identical. Hosts fade the strength with nightBlend themselves.
    if (frame.scotopicDesaturation > 0.0) {
        float scotLum = dot(mapped, float3(0.2126, 0.7152, 0.0722));
        float scot = frame.scotopicDesaturation * (1.0 - smoothstep(0.0, 0.08, scotLum));
        mapped = max(mix(mapped, float3(scotLum), scot), 0.0);   // 0 → exact no-op
    }
    // Clamp after the saturation push — boosting past 1.0 can take channels
    // negative on near-greys, and `pow(negative, 1/2.2)` returns NaN that
    // then propagates through any subsequent composite.
    mapped = saturate(mapped);
    // ── Color-grade: contrast / shadows / highlights tone curve ───────────────
    // Applied in the tonemapped (0..1) domain so the pivot, lift and roll-off
    // act on display-referred values. Defaults (contrast = shadows = highlights
    // = 1.0) make this an exact no-op.
    mapped = saturate(toneCurve(mapped, frame.contrast, frame.shadows, frame.highlights));

    // ── Photographic finish: highlight chroma roll-off ────────────────────────
    // A camera's brightest values lose colour on the way to white — dye layers and
    // sensor channels clip in turn, so a hot warm surface arrives as pale cream, not
    // as saturated yellow. ACES gives us that for free, and then the FLAT
    // `tonemapSaturation` multiply above takes it straight back out (it lerps away
    // from luma by the same factor at 0.95 as at 0.2). Roll the chroma back off, but
    // only at the top of the scale, so the midtone chroma that multiply exists for
    // survives. 0 (DEFAULT) ⇒ the branch never runs ⇒ byte-identical.
    if (frame.highlightChromaRolloff > 0.0) {
        float hl = dot(mapped, float3(0.2126, 0.7152, 0.0722));
        // Weight starts at mid-grey and reaches full at white, so a correctly-exposed
        // midtone is untouched and only the shoulder loses colour.
        float w = smoothstep(0.45, 1.0, hl) * frame.highlightChromaRolloff;
        mapped = max(mix(mapped, float3(hl), w), 0.0);
    }
    // ── Photographic finish: split tone (shadow / highlight temperature) ──────
    // Two temperatures on one image, masked by luma. Real interiors read as
    // photographs partly because their shadows are lit by a DIFFERENT illuminant
    // from their highlights (skylight in the shadow, tungsten in the pool), and a
    // renderer whose whole frame rides one white balance has no way to say that.
    // Reuses `whiteBalanceGain` — the same curve as the global control, normalized to
    // unit luma, so this shifts hue without changing exposure. 6500/6500 (DEFAULTS)
    // ⇒ both gains are exactly (1,1,1) ⇒ the branch never runs ⇒ byte-identical.
    if (frame.shadowTemperatureK != 6500.0 || frame.highlightTemperatureK != 6500.0) {
        float sl = dot(mapped, float3(0.2126, 0.7152, 0.0722));
        float shadowW = 1.0 - smoothstep(0.0, 0.5, sl);
        float highW   = smoothstep(0.5, 1.0, sl);
        float3 g = mix(float3(1.0), whiteBalanceGain(frame.shadowTemperatureK), shadowW)
                 * mix(float3(1.0), whiteBalanceGain(frame.highlightTemperatureK), highW);
        mapped = saturate(mapped * g);
    }

    // ── Axial chromatic aberration ("purple fringing") ────────────────────────
    // Longitudinal CA: a real lens focuses wavelengths at slightly different
    // depths, so high-contrast edges grow a coloured halo — classically violet on
    // the dark side of a bright edge and green on the bright side. Frame-uniform
    // (unlike the radial lateral CA at the top of this pass). We detect luminance
    // edges in the HDR input, compress the (unbounded HDR) gradient to 0..1, then
    // tint: a 5-tap Laplacian's sign decides dark side (tint) vs bright side
    // (complement). 0 strength → the branch never runs (exact no-op).
    if (frame.fringe > 0.0) {
        float2 px = invInSize;
        float lC = dot(float3(inHDR.sample(downSampler, in.uv).rgb),                       float3(0.2126, 0.7152, 0.0722));
        float lL = dot(float3(inHDR.sample(downSampler, in.uv - float2(px.x, 0.0)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float lR = dot(float3(inHDR.sample(downSampler, in.uv + float2(px.x, 0.0)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float lU = dot(float3(inHDR.sample(downSampler, in.uv - float2(0.0, px.y)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float lD = dot(float3(inHDR.sample(downSampler, in.uv + float2(0.0, px.y)).rgb),   float3(0.2126, 0.7152, 0.0722));
        float gx = lR - lL;
        float gy = lD - lU;
        float edge = sqrt(gx * gx + gy * gy);
        edge = edge / (edge + 1.0);                       // compress HDR contrast → 0..1
        float lap = (lL + lR + lU + lD) - 4.0 * lC;       // > 0 on the dark side of a bright edge
        // Tint comes in as sRGB (picked by eye in the UI); `mapped` is linear, so
        // decode before adding (sister of the known solid-colour sRGB→linear bug).
        float3 tintSRGB = (lap >= 0.0)
            ? float3(frame.fringeTintR, frame.fringeTintG, frame.fringeTintB)
            : (1.0 - float3(frame.fringeTintR, frame.fringeTintG, frame.fringeTintB));
        float3 tintLinear = pow(tintSRGB, 2.2);
        mapped = saturate(mapped + tintLinear * (edge * frame.fringe * 0.5));
    }

    // ── Vignette ────────────────────────────────────────────────────────────
    // Optical lens falloff: brightness tapers toward the frame corners. `extent`
    // is the normalised radius (fraction of the half-diagonal) that stays fully
    // bright; beyond it the image is multiplied down by a smoothstep ramp toward
    // (1 - strength) at the very corner. 0 strength → exact no-op.
    if (frame.vignetteStrength > 0.0) {
        float2 d = in.uv - 0.5;
        float r = length(d) * 1.41421356;            // 0 centre → 1 corner
        float ext = clamp(frame.vignetteExtent, 0.0, 0.99);
        float v = smoothstep(ext, 1.0, r);           // 0 inside extent → 1 at corner
        float fall = 1.0 - frame.vignetteStrength * v;
        mapped *= max(fall, 0.0);
    }

    // ── Film grain ──────────────────────────────────────────────────────────
    // Per-frame film-stock grain: high-frequency noise reseeded every frame (so it
    // reads as natural grain, NOT a crawling low-frequency pattern), modulated by
    // luminance so it peaks in the mids and fades in highlights/blacks the way real
    // film does. `filmGrainSize` quantises the sampling grid into cells of that many
    // output pixels. 0 strength → exact no-op.
    if (frame.filmGrainStrength > 0.0) {
        float cell = max(frame.filmGrainSize, 1.0);
        float2 gpf = floor(max(in.position.xy, 0.0) / cell);
        // ── The hash is an INTEGER bit mix, not `fract(sin(...))` ────────────────
        // It used to be
        //     fract(sin(dot(gp, float2(12.9898, 78.233)) + frame.time * 91.7) * 43758.5453)
        // and that form is unconditionally broken for any host whose clock is not near
        // zero. The ONLY spatial term is `dot(gp, k)` — at most ~3.4e4 over a 480×360
        // frame, and just 13.0 (x) / 78.2 (y) between NEIGHBOURING pixels. It was added
        // to `frame.time * 91.7`, which is unbounded, and the sum was then handed to
        // `sin`, whose argument reduction has no precision left to resolve a 13-wide step
        // riding on 3e7. `n` therefore collapses to a CONSTANT, and "grain" degenerates
        // into a flat `-0.5 * strength * mask` DARKENING of the whole frame — the exact
        // opposite of grain, which is zero-mean and high-frequency by definition.
        //
        // MEASURED on the real Metal path (Daydream Home fixture house, 480×360, grain
        // dial at the top ⇒ strength 0.20), sweeping `frame.time` alone:
        //     t =      0    high-freq energy +10.19   mean luma  −0.62   ← healthy grain
        //     t =    1e4                     +10.47              −1.08
        //     t =    1e5                      +8.33              −5.60   ← degrading
        //     t = 3.4e5                       +0.21             −19.93   ← no noise at all
        // 3.4e5 s is not exotic: it is `CACurrentMediaTime()` on a Mac that has been awake
        // four days, which is what this app hands `renderer.time`. The effect died silently
        // as a function of the USER'S UPTIME.
        //
        // A bit hash has no magnitude sensitivity — every input bit is mixed, so the pixel
        // coordinate can never be swallowed by a large time term. Time enters as its own
        // mixed word rather than as an addend inside a transcendental, and it is taken as
        // RAW IEEE BITS (`as_type`), which extracts the most decorrelation the host's
        // `float` clock can carry: any two frames whose `frame.time` differ at all reseed.
        uint2 gp = uint2(gpf);
        uint  h  = (gp.x * 73856093u) ^ (gp.y * 19349663u) ^ (as_type<uint>(frame.time) * 83492791u);
        h ^= h >> 16; h *= 0x7feb352du;
        h ^= h >> 15; h *= 0x846ca68bu;
        h ^= h >> 16;
        float n = float(h) * (1.0 / 4294967296.0) - 0.5;   // uniform ∈ [-0.5, 0.5)
        float lumG = dot(mapped, float3(0.2126, 0.7152, 0.0722));
        float mask = 4.0 * lumG * (1.0 - lumG);       // parabola: 1 at mid-grey, 0 at ends
        mapped = saturate(mapped + n * frame.filmGrainStrength * mask);
    }

    // ── Colour-grade LUT (issue #65) ──────────────────────────────────────────
    // A 3D LUT applied in DISPLAY space: encode the linear `mapped` to ~sRGB, look
    // it up in the LUT cube (the standard .cube authoring space), decode back to
    // linear, and blend by `colorLUTAmount`. Half-texel correction keeps the
    // lookup inside texel centres so the trilinear filter doesn't clamp-bias the
    // extremes. 0 amount → exact no-op (and the identity-baked default LUT is a
    // no-op even at amount 1).
    if (frame.colorLUTAmount > 0.0) {
        constexpr sampler lutSampler(filter::linear, address::clamp_to_edge, coord::normalized);
        float  n   = max(frame.colorLUTSize, 2.0);
        float3 src = pow(saturate(mapped), float3(1.0 / 2.2));     // linear → ~sRGB
        float3 uvw = src * ((n - 1.0) / n) + (0.5 / n);           // half-texel inset
        float3 graded = float3(colorLUT.sample(lutSampler, uvw).rgb);
        graded = pow(saturate(graded), float3(2.2));               // ~sRGB → linear
        mapped = mix(mapped, graded, saturate(frame.colorLUTAmount));
    }

    // ── Phase 4.39: debanding dither ──────────────────────────────────────────
    // The output attachment is 8-bit `bgra8Unorm_srgb`. A smooth lit gradient
    // (e.g. the room's off-white walls) quantises into discrete 8-bit steps →
    // visible horizontal contour banding. The standard AAA fix is a triangular-
    // PDF (TPDF) dither of ±1 LSB applied before the store: it decorrelates the
    // quantisation error so the eye spatially averages the gradient back to
    // smooth. The dither must be in the SAME space as the quantisation — i.e.
    // applied to the sRGB-encoded value, since the GPU store quantises after
    // the sRGB OETF. We approximate the sRGB encode, dither, and decode back to
    // linear (the attachment re-encodes on store).
    //
    // TPDF = (two independent uniform randoms) differenced → triangular noise,
    // which is the optimal dither distribution (flat error, no noise modulation).
    if (frame.debandDitherEnabled != 0u) {
        float3 srgb = pow(mapped, float3(1.0 / 2.2));   // approx sRGB encode
        // Interleaved-gradient-noise hashes for two decorrelated uniform samples.
        float2 px = in.position.xy;
        float n0 = fract(52.9829189 * fract(dot(px, float2(0.06711056, 0.00583715))));
        float n1 = fract(52.9829189 * fract(dot(px + 113.0, float2(0.06711056, 0.00583715))));
        float tpdf = (n0 + n1) - 1.0;                    // ∈ [-1, 1], triangular
        srgb += tpdf * (1.0 / 255.0);                    // ±1 LSB at 8-bit
        mapped = pow(saturate(srgb), float3(2.2));       // decode back to linear
    }

    // Phase 9 — film-stock LUT colour grade. Samples a 16×16×16 3D LUT
    // (stored as a 256×16 PNG strip: 16 blue slices, each 16×16, laid left
    // to right; the Swift host unpacks this into a proper MTLTexture3D).
    // The LUT expects Cineon log input but we apply it post-ACES-tonemapper
    // for a film-inspired grade (not technically accurate emulation — per spec).
    // `filmLUTStrength` blends between ungraded and graded result.
    if (frame.filmLUTStrength > 0.001) {
        constexpr sampler lutSampler(filter::linear, address::clamp_to_edge);
        // Remap `mapped` from [0,1] linear into LUT normalised coords. A 16-cell
        // LUT needs a half-texel inset so the sample lands at the cell centre:
        // coord = (mapped * (N-1) + 0.5) / N  where N = 16.
        float3 uvw = (mapped * 15.0 + 0.5) / 16.0;
        float3 graded = filmLUT.sample(lutSampler, uvw).rgb;
        mapped = mix(mapped, graded, frame.filmLUTStrength);
        mapped = saturate(mapped);
    }

    // ── Diagram cross-fade ───────────────────────────────────────────────────
    // Partial strengths only (full strength returned above, before the tone chain).
    // The blend is against the FINISHED image, so dragging the dial reads as the
    // photograph resolving into a drawing rather than as two half-graded images.
    if (frame.diagramParams.x > 0.0) {
        float3 diagram = illumiDiagramColor(in.uv, frame, gAlbedoMet, gNormalRgh, gDepth);
        mapped = mix(mapped, diagram, saturate(frame.diagramParams.x));
    }

    // Write LINEAR. The output attachment is `.bgra8Unorm_srgb`, so the GPU
    // store applies the sRGB OETF (with sRGB-distributed 8-bit precision in
    // the darks). SceneKit then samples that sRGB texture as `background.
    // contents`, decodes it back to linear, and applies its OWN output sRGB
    // encode for the bgra8Unorm drawable — one correct round-trip. A manual
    // `pow(1/2.2)` here used to double-encode against SceneKit's pass, which
    // washed out blacks and flattened chroma (issue: Illuminatorama colours
    // read desaturated vs the SCN-native `+` scenes).
    return float4(mapped, 1.0);
}
