#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA LOCAL TONE MAPPING (local adaptation) ─────────────────────
//
// What an eye (or a phone's night mode) does that a single global exposure cannot: a
// dim region sitting next to a bright one is ADAPTED TO locally, so a moonlit window
// 14 stops under a red LED still reads as a deep blue sky instead of printing black.
// It is the Durand & Dorsey (2002) base/detail split, computed Chen, Paris & Durand
// (2007)-style on a BILATERAL GRID, and it acts before the display transform as a
// per-pixel EXPOSURE (a hue-preserving gain on the scene-referred colour):
//
//   L      = log2(exposed brightness)                      (the meter's own metric:
//            max(luma, ½·max channel) × this frame's exposure, so the operator and the
//            auto-exposure agree on what "bright" means — a saturated red LED is not
//            4× dimmer than it looks)
//   B      = the BASE layer: L smoothed over a neighbourhood of `radius`, but only
//            among pixels within ~`edgeStops` of the pixel's own level. That is what
//            makes it edge-aware: across a step of many stops the two sides never
//            average into each other, so neither side gets the other's adaptation —
//            no halo (a plain blur here draws a dark ring round every bright object
//            and a bright ring inside it).
//   d      = anchor − B            (how many stops this neighbourhood sits under the
//                                   level the operator holds fixed)
//   f(d)   = c·d + (1 − c)·k·(1 − e^(−d/k))     c = 1 − strength, k = knee
//            f′(0) = 1: within ~k stops of the anchor contrast is kept as shot (the
//            red pool keeps its fall-off under the digits); far below, the slope
//            falls to c: that tail's range is compressed — an adaptation that is
//            partial and saturating, like a photoreceptor's (Naka–Rushton) rather
//            than the full flattening of a naive local exposure.
//   gain   = ceil(d − f(d)) + (detail − 1)·(L − B)  stops; ceil = a smooth min with
//            `maxLift` (1-stop softness — the eye's adaptation floor, and a bound on how far
//            the operator amplifies the frame's fp16 quantisation), then clamped to ±maxLift
//   out    = rgb · 2^gain
//
// Neighbourhoods at or above the anchor get gain 0 (the global exposure and the
// display transform already own the highlights). Strictly monotone in B (f′ > 0 for
// strength < 1), so a brighter region never prints darker than a dimmer one; with
// detail = 1 the gain is a function of the smooth base only, so local texture is kept
// (texture as strong as `edgeStops` leaks ≈ 5 % into the base — a 1-stop swing comes out
// within 0.04 stops). strength = 0 ⇒ f(d) = d ⇒ gain 0 ⇒ identity (and the host does not
// even encode the pass).
//
// THE GRID. `L` is first reduced to a low-res log image (mean of log2 over a factor×factor
// block — the pyramid level); each grid cell covers cellLow × cellLow of those texels and
// `bins` log-luminance slabs `binStops` apart, holding the homogeneous pair
// (Σ w·L, Σ w) with a tent weight w on |L − bin centre|. A separable [1 4 6 4 1] blur
// over x, y and the range axis (zero-padded: a normalised convolution, so borders and
// empty slabs need no special case), then each full-res pixel slices the grid at
// (x, y, its own L) with a trilinear blend done by hand (the grid is rg32Float — exact
// sums; 32-bit float filtering is not guaranteed on every Apple GPU). An ε·L term in
// the slice falls back to the pixel's own level where the grid holds nothing near it.

struct LTMParams {
    float4 tone;    // x anchorLog2, y strength, z kneeStops, w maxLiftStops
    float4 grid;    // x lMin, y binStops, z detail, w slice epsilon (weight units)
    float4 expo;    // x host exposure, y auto-exposure on (0/1), z lMax, w unused
    uint4  sizes;   // x fullW, y fullH, z lowW, w lowH
    uint4  cells;   // x factor (full px per low texel), y cellLow (low texels per cell), z gridW, w gridH
    uint4  cells2;  // x gridD (bins), y blur axis (0 x, 1 y, 2 range), zw unused
    float4 tail;    // x floor level (scene-referred; 0 = off), y floor slope, zw unused
    float4 meso;    // x mesopic amount (0 = off), y cd/m² per frame unit, zw unused
    float4 mesoTint;// xyz rod-vision tint, w rod model (0 Larson 1997, 1 additive per-primary)
    // ── Round-3 opt-ins (all zero ⇒ the paths above, bit-identical) ──
    float4 adapt;   // x adaptation from the frame ON, y meter target EV, z tolerance (stops), w gain
    float4 hist;    // x histogram tail ON, y gap slope, z populated fraction τ, w min populated scale
    float4 pfloor;  // x floor luminance (cd/m², needs meso.y), y floor print level (exposed), zw unused
    float4 meso2;   // x mesopic model (0 CIE 191 m, 1 blue shift), y log10 lo, z log10 hi, w pixel floor ON
    float4 adapt2;  // x photopic field luminance (cd/m², needs meso.y; 0 = off), y cone floor ON,
                    // z mesopic weight from the spatial adaptation ON, w unused
};

// Stats buffer written by `illumiLTMStats` (one threadgroup) and read by the apply:
//   [0] mean exposed log2 brightness of the frame (the grid's own meter), [1] lift ceiling from
//   the frame's exposure deficit (stops, −1 = not computed), [2] populated-slope scale q of the
//   histogram fit, [3] unused, [4 …] f(d) at d = i·kLTMCurveStep, i < kLTMCurveN.
constant constexpr int   kLTMCurveN    = 161;
constant constexpr float kLTMCurveStep = 0.25;
constant constexpr int   kLTMStatsHead = 4;
constant constexpr int   kLTMMaxBins   = 160;

// The frame's exposure, exactly as the tonemap derives it (IlluminatoramaTonemap.metal:
// `autoBase * frame.exposure`). `expoState[1]` is ExposureState.smoothedExposure —
// written earlier in this same command buffer by the estimate kernel.
static inline float ltmExposure(constant LTMParams& p, device const float* expoState) {
    float autoBase = (p.expo.y > 0.5) ? expoState[1] : 1.0;
    return autoBase * p.expo.x;
}

// The meter's brightness metric (IlluminatoramaTonemap.metal, exposure estimate): the
// larger of Rec.709 luma and half the max channel, in exposed log2 units, clamped to
// the grid's range.
static inline float ltmLogBrightness(float3 rgb, float exposure, constant LTMParams& p) {
    float lum  = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    float maxc = max(rgb.r, max(rgb.g, rgb.b));
    float b = max(lum, 0.5 * maxc) * exposure;
    float lMin = p.grid.x, lMax = p.expo.z;
    if (!isfinite(b) || b <= 0.0) return lMin;
    return clamp(log2(b), lMin, lMax);
}

// 1 ── the pyramid level: mean log2 brightness over each factor×factor block.
kernel void illumiLTMLogLuminance(texture2d<float, access::read>  src    [[texture(0)]],
                                  texture2d<float, access::write> lowLog [[texture(1)]],
                                  constant LTMParams& p                  [[buffer(0)]],
                                  device const float* expoState          [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.sizes.z || gid.y >= p.sizes.w) return;
    const float exposure = ltmExposure(p, expoState);
    const uint f = max(p.cells.x, 1u);
    const uint x0 = gid.x * f, y0 = gid.y * f;
    const uint x1 = min(x0 + f, p.sizes.x), y1 = min(y0 + f, p.sizes.y);
    float acc = 0.0;
    uint n = 0u;
    for (uint y = y0; y < y1; ++y) {
        for (uint x = x0; x < x1; ++x) {
            acc += ltmLogBrightness(src.read(uint2(x, y)).rgb, exposure, p);
            n += 1u;
        }
    }
    lowLog.write(float4(n > 0u ? acc / float(n) : p.grid.x, 0.0, 0.0, 0.0), gid);
}

// 2 ── splat (as a gather): one thread per grid cell (x, y, bin).
kernel void illumiLTMGridBuild(texture2d<float, access::read>  lowLog [[texture(0)]],
                               texture3d<float, access::write> grid   [[texture(1)]],
                               constant LTMParams& p                  [[buffer(0)]],
                               uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cells.z || gid.y >= p.cells.w || gid.z >= p.cells2.x) return;
    const uint c = max(p.cells.y, 1u);
    const uint x0 = gid.x * c, y0 = gid.y * c;
    const uint x1 = min(x0 + c, p.sizes.z), y1 = min(y0 + c, p.sizes.w);
    const float binStops = max(p.grid.y, 1e-3);
    const float lk = p.grid.x + float(gid.z) * binStops;
    float sumWL = 0.0, sumW = 0.0;
    for (uint y = y0; y < y1; ++y) {
        for (uint x = x0; x < x1; ++x) {
            float l = lowLog.read(uint2(x, y)).r;
            float w = max(0.0, 1.0 - fabs(l - lk) / binStops);
            sumWL += w * l;
            sumW  += w;
        }
    }
    grid.write(float4(sumWL, sumW, 0.0, 0.0), gid);
}

// 3 ── separable [1 4 6 4 1]/16 blur along one axis (cells2.y), zero-padded.
kernel void illumiLTMGridBlur(texture3d<float, access::read>  src [[texture(0)]],
                              texture3d<float, access::write> dst [[texture(1)]],
                              constant LTMParams& p               [[buffer(0)]],
                              uint3 gid [[thread_position_in_grid]]) {
    const uint3 dims = uint3(p.cells.z, p.cells.w, p.cells2.x);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) return;
    const uint axis = p.cells2.y;
    const float wts[5] = { 1.0, 4.0, 6.0, 4.0, 1.0 };
    float2 acc = float2(0.0);
    for (int t = -2; t <= 2; ++t) {
        int3 q = int3(gid);
        q[axis] += t;
        if (q[axis] < 0 || q[axis] >= int(dims[axis])) continue;
        acc += src.read(uint3(q)).rg * wts[t + 2];
    }
    dst.write(float4(acc / 16.0, 0.0, 0.0), gid);
}

// The compressive adaptation curve f(d) (see the header), d ≥ 0 stops under the anchor.
static inline float ltmCompress(float d, float strength, float knee) {
    float c = 1.0 - strength;
    if (knee <= 1e-4) return c * d;
    return c * d + (1.0 - c) * knee * (1.0 - exp(-d / knee));
}

// The lift's ceiling `maxLift`, reached smoothly (a smooth-min of 1-stop softness): a hard min
// would put a slope kink in the tone curve at the level where the lift saturates, and a wall's
// gradient crossing that level would show it as a contour.
constant float kLTMCeilingSoftness = 1.0;
static inline float ltmSoftCeiling(float lift, float maxLift) {
    const float w = kLTMCeilingSoftness;
    return max(0.0, lift - w * log(1.0 + exp((lift - maxLift) / w)));
}

// The second segment (opt-in, `floorLevel`): below the floor the curve's slope changes from
// c = 1 − strength to `floorSlope`, through a softplus corner `kLTMFloorSoftness` stops wide —
//   gain(d) = [d − f(d)] − (floorSlope − c)·softplus(d − dF)
// so a region under the floor gets contrast back (its lift grows at 1 − floorSlope per stop, not
// at `strength`). Output log level = anchor − d + gain, whose slope −1 + gain′ stays < 0: still
// strictly monotone.
constant float kLTMFloorSoftness = 1.0;
static inline float ltmSoftplus(float x) {
    const float w = kLTMFloorSoftness;
    return (x / w > 20.0) ? x : w * log(1.0 + exp(x / w));
}

// ── Mesopic vision (opt-in, `mesopic`) ──────────────────────────────────────────────────────
// CIE 191:2010 recommended system (MES2): the mesopic luminance of a field with photopic Lp and
// scotopic Ls (cd/m², scotopic in its own 1699 lm/W units) is
//   L_mes = (m·Lp + (1 − m)·Ls·V′₀) / (m + (1 − m)·V′₀),   V′₀ = 683/1699
// with the adaptation coefficient m = 0.767 + 0.3334·log10(L_mes), clamped to [0, 1] (1 by
// 5 cd/m², 0 by 0.005), solved by the standard fixed-point iteration.
static inline float ltmMesopicM(float Lp, float Ls) {
    const float V0 = 683.0 / 1699.0;
    float m = 0.5;
    for (int i = 0; i < 5; ++i) {
        float Lmes = (m * Lp + (1.0 - m) * Ls * V0) / (m + (1.0 - m) * V0);
        m = clamp(0.767 + 0.3334 * log10(max(Lmes, 1e-9)), 0.0, 1.0);
    }
    return m;
}
// Scotopic luminance from linear sRGB (Larson, Rushmeier & Piatko 1997: V = Y·(1.33·(1 + (Y+Z)/X)
// − 1.68)), returned as the S/P ratio V/Y normalised to CIE's D65 value (Larson's V/Y for D65 is
// 2.573; CIE's S/P for D65 is ≈ 2.47).
static inline float ltmScotopicRatio(float3 c) {
    float X = dot(c, float3(0.4124, 0.3576, 0.1805));
    float Y = dot(c, float3(0.2126, 0.7152, 0.0722));
    float Z = dot(c, float3(0.0193, 0.1192, 0.9505));
    if (!(X > 1e-20) || !(Y > 1e-20)) return 2.47;
    float VY = max(1.33 * (1.0 + (Y + Z) / X) - 1.68, 0.0);
    return VY * (2.47 / 2.573);
}

// Rod luminance (photopic-equivalent, D65 grey keeps its luminance) of a linear-sRGB colour.
// Larson (model 0) is NOT additive — red light added to dim moonlight LOWERS the mixture's value —
// which printed the edge of a red light as a near-black seam (VZ-0194). The additive model (1) sums
// Larson's own per-primary values (D65 grey within 0.3 %): a photoreceptor's response to light is
// linear in it, so adding light never lowers it.
static inline float ltmRodLuminance(float3 cc, float Y, float model) {
    if (model > 0.5) return dot(cc, float3(0.032876, 0.765326, 0.201635));
    return min(Y * ltmScotopicRatio(cc) / 2.47, 4.0 * Y);
}

// ── Round-3: adaptation from the frame + histogram-adjusted tail (opt-in) ─────────────────────
//
// ADAPTATION FROM THE FRAME (`adapt.x`). The eye's adaptation is set by the whole field it sees;
// a camera whose exposure is held down (a bright emitter's cap, or a meter that reads only part
// of the frame) leaves the field UNDER-exposed, and it is exactly that deficit a dark-adapted eye
// makes up. The grid already holds the frame's log-brightness distribution, so its mean is a
// whole-frame meter at the APPLIED exposure: deficit = targetEV − mean. The lift ceiling is
//   ceiling = gain · max(0, deficit − tolerance)
// — 0 while the field is exposed within `tolerance` of the meter's key (the operator then leaves
// the frame EXACTLY as rendered), growing stop for stop with the underexposure as the light
// falls: one adaptation state keyed on what the frame actually is, not on a separate ramp.
//
// HISTOGRAM-ADJUSTED TAIL (`hist.x`, Ward Larson, Rushmeier & Piatko 1997, histogram adjustment).
// The fixed far-tail slope c = 1 − strength compresses every range under the knee alike —
// populated or empty — so a lit facade 2.6 stops under the sky it stands against printed 0.1
// stop under it. Here the slope at each brightness bin comes from the frame's own histogram (the
// grid's unblurred weight per bin, i.e. the low-res log image's distribution): a POPULATED bin
// (≥ τ of the frame) keeps contrast (slope 1 × q), an empty one is compressed (`gap slope`) —
// the digits-to-window gap collapses, the room and the view keep their internal contrast.
//   f′(d) = c(d) + (1 − c(d))·e^(−d/knee),   c(d) = gap + (w(p) · (1 − gap)) · q
// (the knee keeps contrast as shot right under the anchor, as before). q ∈ [min scale, 1] is
// FITTED so the curve lands the absolute floor luminance (`pfloor.x`, cd/m² — ~1e-5, where a
// dark-adapted eye still sees) exactly at the floor PRINT level (`pfloor.y`, the exposed
// brightness the display prints at a chosen low code — the host solves it through the display
// transform's inverse): the range the eye adapts across is fitted into the range the display
// can still print, instead of being aimed at a level the display's toe crushes to black.

/// The tail slope at exposed level `lv` from the per-bin slopes in threadgroup memory (linear
/// between bin centres).
static inline float ltmBinSlope(threadgroup const float* sl, uint D, float lMin, float bin, float lv) {
    float x = clamp((lv - lMin) / bin, 0.0, float(D - 1u));
    uint i0 = uint(floor(x)); uint i1 = min(i0 + 1u, D - 1u);
    return mix(sl[i0], sl[i1], x - float(i0));
}

/// f(d) at `d` — integrated in kLTMCurveStep steps (trapezoid) for the slope scale q.
static inline float ltmTailF(threadgroup const float* sl, uint D, float lMin, float bin, float A,
                             float gap, float q, float knee, float dEnd) {
    float f = 0.0;
    float prev = 1.0;   // f′(0) = 1
    int n = int(ceil(dEnd / kLTMCurveStep));
    for (int i = 1; i <= n; ++i) {
        float d = min(float(i) * kLTMCurveStep, dEnd);
        float c = gap + (ltmBinSlope(sl, D, lMin, bin, A - d) - gap) * q;
        float fp = c + (1.0 - c) * exp(-d / max(knee, 1e-4));
        f += 0.5 * (prev + fp) * (d - float(i - 1) * kLTMCurveStep);
        prev = fp;
    }
    return f;
}

// 2b ── frame statistics + the fitted curve: ONE threadgroup, run on the UNBLURRED grid.
kernel void illumiLTMStats(texture3d<float, access::read> grid  [[texture(4)]],
                           constant LTMParams& p                [[buffer(0)]],
                           device const float* expoState        [[buffer(1)]],
                           device float* stats                  [[buffer(2)]],
                           uint tid    [[thread_index_in_threadgroup]],
                           uint tgSize [[threads_per_threadgroup]]) {
    threadgroup float hW[kLTMMaxBins];
    threadgroup float hWL[kLTMMaxBins];
    threadgroup float sl[kLTMMaxBins];
    const uint D = min(p.cells2.x, uint(kLTMMaxBins));
    for (uint k = tid; k < D; k += tgSize) {
        float w = 0.0, wl = 0.0;
        for (uint y = 0u; y < p.cells.w; ++y) {
            for (uint x = 0u; x < p.cells.z; ++x) {
                float2 g = grid.read(uint3(x, y, k)).rg;
                wl += g.x; w += g.y;
            }
        }
        hW[k] = w; hWL[k] = wl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;

    float sw = 0.0, swl = 0.0;
    for (uint k = 0u; k < D; ++k) { sw += hW[k]; swl += hWL[k]; }
    const float lMin = p.grid.x, bin = max(p.grid.y, 1e-3);
    const float meanL = sw > 0.0 ? swl / sw : lMin;
    stats[0] = meanL;
    // Photopic fields (`adapt2.x`): in cone vision lightness constancy keeps pace with a camera's
    // print, so a field brighter than the photopic level gets one more stop of tolerance per stop
    // it sits above it — the eye re-adapts beyond the print only where the cones give out.
    float tol = p.adapt.z;
    if (p.adapt2.x > 0.0 && p.meso.y > 0.0) {
        const float La = exp2(meanL) / max(ltmExposure(p, expoState), 1e-30) * p.meso.y;
        tol += max(0.0, log2(max(La, 1e-30) / p.adapt2.x));
    }
    stats[1] = (p.adapt.x > 0.5) ? p.adapt.w * max(0.0, (p.adapt.y - meanL) - tol) : -1.0;

    if (p.hist.x > 0.5) {
        const float exposure = ltmExposure(p, expoState);
        const float A = p.tone.x, knee = p.tone.z, gap = clamp(p.hist.y, 0.0, 1.0);
        const float tau = max(p.hist.z, 1e-6);
        // Where the fit's floor sits (exposed log2); with no absolute scale, no fit (q = 1).
        const bool fit = p.pfloor.x > 0.0 && p.meso.y > 0.0 && p.pfloor.y > 0.0;
        const float lF = fit ? log2(max(p.pfloor.x / p.meso.y * exposure, 1e-30)) : lMin;
        // Populated fraction over the bins the curve acts on (under the anchor, above the floor
        // less one knee — the room just under the floor still counts).
        float tot = 0.0;
        for (uint k = 0u; k < D; ++k) {
            float lk = lMin + float(k) * bin;
            if (lk < A && lk >= lF - knee) tot += hW[k];
        }
        for (uint k = 0u; k < D; ++k) {
            float t = saturate((hW[k] / max(tot, 1e-9)) / tau);
            sl[k] = gap + (1.0 - gap) * (t * t * (3.0 - 2.0 * t));
        }
        float q = 1.0;
        if (fit) {
            const float dF = A - lF;
            const float budget = A - log2(p.pfloor.y);
            if (dF > 0.0 && ltmTailF(sl, D, lMin, bin, A, gap, 1.0, knee, dF) > budget) {
                float lo = clamp(p.hist.w, 0.0, 1.0), hi = 1.0;
                for (int it = 0; it < 18; ++it) {
                    float mid = 0.5 * (lo + hi);
                    if (ltmTailF(sl, D, lMin, bin, A, gap, mid, knee, dF) > budget) hi = mid; else lo = mid;
                }
                q = lo;
            }
        }
        stats[2] = q;
        // The LUT: f(d) at every step, one running integral.
        float f = 0.0, prev = 1.0;
        stats[kLTMStatsHead] = 0.0;
        for (int i = 1; i < kLTMCurveN; ++i) {
            float d = float(i) * kLTMCurveStep;
            float c = gap + (ltmBinSlope(sl, D, lMin, bin, A - d) - gap) * q;
            float fp = c + (1.0 - c) * exp(-d / max(knee, 1e-4));
            f += 0.5 * (prev + fp) * kLTMCurveStep;
            prev = fp;
            stats[kLTMStatsHead + i] = f;
        }
    } else {
        stats[2] = 1.0;
    }
}

/// f(d) from the stats LUT (linear; past the table's end, continued at the last slope).
static inline float ltmTailLUT(device const float* stats, float d) {
    float x = d / kLTMCurveStep;
    const float last = float(kLTMCurveN - 1);
    if (x >= last) {
        float fl = stats[kLTMStatsHead + kLTMCurveN - 1];
        float sl = (fl - stats[kLTMStatsHead + kLTMCurveN - 2]) / kLTMCurveStep;
        return fl + sl * (d - last * kLTMCurveStep);
    }
    int i0 = int(floor(x));
    float t = x - float(i0);
    return mix(stats[kLTMStatsHead + i0], stats[kLTMStatsHead + i0 + 1], t);
}

// 3b ── the neighbourhood's PLAIN adaptation level per cell: the blurred grid summed over every
// brightness bin (Σ w·L / Σ w) — its mean log brightness, NOT edge-aware (the eye's adaptation is
// set by its surround, whatever the pixel's own level). Read only by the mesopic weight.
kernel void illumiLTMAdaptation(texture3d<float, access::read>  grid [[texture(3)]],
                                texture2d<float, access::write> outA [[texture(5)]],
                                constant LTMParams& p                [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cells.z || gid.y >= p.cells.w) return;
    float2 acc = float2(0.0);
    for (uint k = 0u; k < p.cells2.x; ++k) acc += grid.read(uint3(gid, k)).rg;
    outA.write(float4(acc.y > 1e-12 ? acc.x / acc.y : p.grid.x, 0.0, 0.0, 0.0), gid);
}

// 4 ── slice + apply: one thread per full-res pixel.
kernel void illumiLTMApply(texture2d<float, access::read>  src  [[texture(0)]],
                           texture2d<float, access::write> dst  [[texture(2)]],
                           texture3d<float, access::read>  grid [[texture(3)]],
                           texture2d<float, access::read>  adaptTex [[texture(5)]],
                           constant LTMParams& p                [[buffer(0)]],
                           device const float* expoState        [[buffer(1)]],
                           device const float* stats            [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.sizes.x || gid.y >= p.sizes.y) return;
    float4 c = src.read(gid);
    if (!all(isfinite(c.rgb))) { dst.write(c, gid); return; }
    // Adaptation from the frame: a field exposed within tolerance of the key is left EXACTLY as
    // rendered (no lift, no mesopic shift).
    float ceilingFromFrame = -1.0;
    if (p.adapt.x > 0.5) {
        ceilingFromFrame = stats[1];
        if (!(ceilingFromFrame > 0.0)) { dst.write(c, gid); return; }
    }
    const float exposure = ltmExposure(p, expoState);
    const float L = ltmLogBrightness(c.rgb, exposure, p);

    // Continuous grid coordinates: integer = a cell / bin CENTRE.
    const float cellPx = float(max(p.cells.x, 1u) * max(p.cells.y, 1u));
    const float3 dims = float3(p.cells.z, p.cells.w, p.cells2.x);
    float3 u = float3((float2(gid) + 0.5) / cellPx - 0.5,
                      (L - p.grid.x) / max(p.grid.y, 1e-3));
    u = clamp(u, float3(0.0), dims - 1.0);
    const uint3 i0 = uint3(floor(u));
    const uint3 i1 = min(i0 + 1u, uint3(dims) - 1u);
    const float3 fr = u - float3(i0);
    float2 s = float2(0.0);
    for (uint k = 0u; k < 8u; ++k) {
        uint3 q = uint3((k & 1u) ? i1.x : i0.x, (k & 2u) ? i1.y : i0.y, (k & 4u) ? i1.z : i0.z);
        float w = ((k & 1u) ? fr.x : 1.0 - fr.x)
                * ((k & 2u) ? fr.y : 1.0 - fr.y)
                * ((k & 4u) ? fr.z : 1.0 - fr.z);
        s += grid.read(q).rg * w;
    }
    const float eps = p.grid.w;
    const float B = (s.x + eps * L) / (s.y + eps);

    const float anchor = p.tone.x, strength = p.tone.y, knee = p.tone.z, maxLift = p.tone.w;
    const float d = max(anchor - B, 0.0);
    float lift;
    if (p.hist.x > 0.5) {
        lift = d - ltmTailLUT(stats, d);
    } else {
        lift = d - ltmCompress(d, strength, knee);
    }
    if (p.hist.x <= 0.5 && p.tail.x > 0.0) {
        // The floor, in stops under the anchor at THIS frame's exposure.
        const float dF = anchor - log2(max(p.tail.x * exposure, 1e-30));
        lift -= (p.tail.y - (1.0 - strength)) * ltmSoftplus(d - dF);
    }
    const float liftCeiling = (ceilingFromFrame >= 0.0) ? min(maxLift, ceilingFromFrame) : maxLift;
    float gain = ltmSoftCeiling(lift, liftCeiling);
    gain += (p.grid.z - 1.0) * (L - B);
    gain = clamp(gain, -maxLift, maxLift);

    float3 rgb = c.rgb;
    // The mesopic shift follows the adaptation state in: fully on once the frame's ceiling is
    // past a stop (1 when the adaptation is not taken from the frame).
    const float mesoAmount = p.meso.x * ((ceilingFromFrame >= 0.0) ? smoothstep(0.0, 1.0, ceilingFromFrame) : 1.0);
    if (mesoAmount > 0.0 && p.meso2.x > 0.5) {
        // BLUE SHIFT (Jensen, Durand, Stark, Premože, Dorsey & Shirley 2000): rod vision reads a
        // desaturated BLUE (the tint — Jensen's CIE xy (0.25, 0.25)), and colour fades with LOG
        // luminance across the mesopic range [lo, hi] (log10 cd/m²) — not with CIE 191's m,
        // which is a luminance weighting (it says how much the rods count toward brightness, not
        // how much colour is lost). The luminance is the pixel's own when it is the brighter of
        // the two (`meso2.w`): the neighbourhood base comes from the dark side of an edge, so on
        // a lit patch's rim it would rod-tint pixels bright enough to be photopic.
        float3 cc = max(rgb, 0.0);
        float Y = dot(cc, float3(0.2126, 0.7152, 0.0722));
        float bright = max(Y, 0.5 * max(cc.r, max(cc.g, cc.b)));
        float lumRatio = bright > 1e-30 ? Y / bright : 1.0;
        float Lp = exp2(B) / max(exposure, 1e-30) * p.meso.y * lumRatio;
        if (p.adapt2.z > 0.5) {
            // The surround's plain adaptation level (bilinear over the cells), no per-pixel ratio.
            float a00 = adaptTex.read(i0.xy).r, a10 = adaptTex.read(uint2(i1.x, i0.y)).r;
            float a01 = adaptTex.read(uint2(i0.x, i1.y)).r, a11 = adaptTex.read(i1.xy).r;
            float Ba = mix(mix(a00, a10, fr.x), mix(a01, a11, fr.x), fr.y);
            Lp = exp2(Ba) / max(exposure, 1e-30) * p.meso.y;
        }
        if (p.meso2.w > 0.5) Lp = max(Lp, (p.adapt2.y > 0.5 ? bright : Y) * p.meso.y);
        float lg = log10(max(Lp, 1e-12));
        float w = mesoAmount * (1.0 - smoothstep(p.meso2.y, p.meso2.z, lg));
        if (w > 0.0) {
            float rodLum = ltmRodLuminance(cc, Y, p.mesoTint.w);
            float3 tint = p.mesoTint.xyz / max(dot(p.mesoTint.xyz, float3(0.2126, 0.7152, 0.0722)), 1e-4);
            rgb = mix(rgb, rodLum * tint, w);
        }
    } else if (mesoAmount > 0.0) {
        // The neighbourhood's photopic luminance (cd/m²): the edge-aware base, un-exposed, with
        // the pixel's own luma / brightness-metric ratio (the base is in the meter's metric,
        // which reads a saturated red at ½·R — ~2.4× its luminance).
        float3 cc = max(rgb, 0.0);
        float Y = dot(cc, float3(0.2126, 0.7152, 0.0722));
        float bright = max(Y, 0.5 * max(cc.r, max(cc.g, cc.b)));
        float lumRatio = bright > 1e-30 ? Y / bright : 1.0;
        float Lp = exp2(B) / max(exposure, 1e-30) * p.meso.y * lumRatio;
        if (p.meso2.w > 0.5) Lp = max(Lp, Y * p.meso.y);
        float sp = ltmScotopicRatio(cc);
        float m = ltmMesopicM(Lp, Lp * sp);
        float w = mesoAmount * (1.0 - m);
        if (w > 0.0) {
            // The rod response: photopic-equivalent scotopic luminance (a D65 grey keeps its
            // luminance), bounded like the legacy scotopic branch, in the rod tint.
            float rodLum = min(Y * sp / 2.47, 4.0 * Y);   // = Larson V / 2.573
            float3 tint = p.mesoTint.xyz / max(dot(p.mesoTint.xyz, float3(0.2126, 0.7152, 0.0722)), 1e-4);
            rgb = mix(rgb, rodLum * tint, w);
        }
    }
    dst.write(float4(rgb * exp2(gain), c.a), gid);
}
