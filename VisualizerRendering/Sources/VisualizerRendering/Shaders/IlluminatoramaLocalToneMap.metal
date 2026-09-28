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
    float4 mesoTint;// xyz rod-vision tint, w unused
};

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

// 4 ── slice + apply: one thread per full-res pixel.
kernel void illumiLTMApply(texture2d<float, access::read>  src  [[texture(0)]],
                           texture2d<float, access::write> dst  [[texture(2)]],
                           texture3d<float, access::read>  grid [[texture(3)]],
                           constant LTMParams& p                [[buffer(0)]],
                           device const float* expoState        [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.sizes.x || gid.y >= p.sizes.y) return;
    float4 c = src.read(gid);
    if (!all(isfinite(c.rgb))) { dst.write(c, gid); return; }
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
    float lift = d - ltmCompress(d, strength, knee);
    if (p.tail.x > 0.0) {
        // The floor, in stops under the anchor at THIS frame's exposure.
        const float dF = anchor - log2(max(p.tail.x * exposure, 1e-30));
        lift -= (p.tail.y - (1.0 - strength)) * ltmSoftplus(d - dF);
    }
    float gain = ltmSoftCeiling(lift, maxLift);
    gain += (p.grid.z - 1.0) * (L - B);
    gain = clamp(gain, -maxLift, maxLift);

    float3 rgb = c.rgb;
    if (p.meso.x > 0.0) {
        // The neighbourhood's photopic luminance (cd/m²): the edge-aware base, un-exposed, with
        // the pixel's own luma / brightness-metric ratio (the base is in the meter's metric,
        // which reads a saturated red at ½·R — ~2.4× its luminance).
        float3 cc = max(rgb, 0.0);
        float Y = dot(cc, float3(0.2126, 0.7152, 0.0722));
        float bright = max(Y, 0.5 * max(cc.r, max(cc.g, cc.b)));
        float lumRatio = bright > 1e-30 ? Y / bright : 1.0;
        float Lp = exp2(B) / max(exposure, 1e-30) * p.meso.y * lumRatio;
        float sp = ltmScotopicRatio(cc);
        float m = ltmMesopicM(Lp, Lp * sp);
        float w = p.meso.x * (1.0 - m);
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
