#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA DEPTH OF FIELD ────────────────────────────────────────────
//
// A LENS, not a blur. Runs on the resolved HDR composite after TAA + the exposure
// estimate and before bloom, so out-of-focus highlights form defined discs that
// then bloom — the bokeh that sells a real camera.
//
// Three things make this optics rather than a depth-weighted smear:
//
//  1. THE CIRCLE OF CONFUSION IS THE THIN-LENS ONE. `cocCoefficient` is the whole
//     lens — focal length, f-number, focus distance and sensor size collapsed into
//     one scalar by `ThinLens.cocCoefficientPixels` — and the per-pixel CoC is that
//     coefficient times |z − z_f| / z. The `1/z` is the part a ramp cannot fake: the
//     near side of the focus plane blurs much faster than the far side, which is why
//     a foreground object reads as destroyed while a background object at the same
//     distance is merely soft.
//
//  2. THE GATHER IS A SCATTER. The question asked of each neighbour is not "how far
//     away is it" but "does ITS confusion disc reach me" — i.e. the footprint test is
//     against the *neighbour's* CoC, which is what makes a bright out-of-focus point
//     spread into an actual disc of its own size instead of a Gaussian smudge. Energy
//     is divided by that footprint's area, so a point spread over a large disc is
//     correspondingly dimmer per pixel, and the total light is conserved.
//
//  3. THE APERTURE HAS A SHAPE. A real iris is a polygon of blades, and near the
//     frame edges the lens barrel clips it into a lemon ("cat's eye" — optical
//     vignetting). The footprint test is against that shape, so bokeh balls are
//     hexagons or heptagons that turn elliptical toward the corners, exactly as a
//     photograph's do. `blades = 0` is a perfect circular iris.
//
// Four passes. `illumi_dof_tile` reduces the CoC to a max per 16×16 tile and
// `illumi_dof_dilate` spreads that max by the gather's own reach, so the gather knows how
// far it must look to find a NEAR-field neighbour whose disc covers it — without that,
// foreground blur can only ever spread as far as the pixel it lands on already knows about,
// and a defocused foreground silhouette stays crisp. The tile map also buys the early-out
// that pays for the rest: a tile whose whole neighbourhood is in focus copies through
// untouched. `illumi_dof_prefilter` builds the half-resolution base of a CoC-weighted mip
// pyramid (the renderer generates the rest), so each tap of the gather reads its colour
// prefiltered to the spacing between taps rather than point-sampling detail finer than that.

constant float kGoldenAngle = 2.39996323f;

struct DOFParams {
    float4x4 invProjection;
    float focusDist;        // metres to the focus plane
    float cocCoefficient;   // CoC DIAMETER in px per unit of |z − z_f| / z — the lens
    float maxRadius;        // px clamp on the CoC radius (perf + sanity, not optics)
    float blades;           // iris blade count; < 3 ⇒ perfectly circular
    float bladeRotation;    // radians, orientation of the blade polygon
    float catsEye;          // 0…1 optical vignetting toward the frame corners
    uint  width;  uint height;
    uint  tileW;  uint tileH; uint tileSize;
    float prefilterScale;   // × each tap's footprint it is prefiltered over; 0 ⇒ point taps
    float cocFloor;         // DIFFRACTION: Airy-disc CoC radius in px, the same everywhere
    float subjectAware;     // > 0.5 ⇒ the SHARP-SUBJECT fixes (opt-in, see the gather); 0 ⇒ as before
    float fastGather;       // > 0.5 (with subjectAware) ⇒ the FAST gather (opt-in, see below); 0 ⇒ as before
    float halfMinCoC;       // fast gather: |CoC| (px) from which a pixel may take the half-res result
    float quarterMinCoC;    // fast gather: |CoC| (px) from which a pixel may take the QUARTER-res
                            // result (opt-in `dofQuarterResMinCoC`); 0 ⇒ no quarter tier, as before
    float partialOcclusion; // > 0.5 (with subjectAware) ⇒ the LAYERED COMPOSITE (opt-in
                            // `dofPartialOcclusion`, see `dof_layered`); 0 ⇒ the gathers as before
    float nearPyramid;      // > 0.5 (with partialOcclusion) ⇒ near-field fronts read the NEAR-FIELD
                            // coverage pyramid (`illumi_dof_prefilter` textures 4/5); 0 ⇒ per texel
};

// View-space distance (positive, metres) of a depth-buffer sample.
static inline float view_z(float depth, uint2 gid, constant DOFParams& p) {
    float2 ndc = (float2(gid) + 0.5f) / float2(p.width, p.height) * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float4 vp = p.invProjection * float4(ndc, depth, 1.0f);
    return abs(vp.z / max(abs(vp.w), 1e-6f));
}

// `view_z` from the projection's z and w rows alone (the layered composite's taps): clip x/y never
// feed view z or w under a perspective or orthographic projection, jittered or shifted or not, so this
// is the same distance in 4 FMAs instead of a matrix product (bit-identical on the Digital Clock).
static inline float view_z_zw(float depth, constant DOFParams& p) {
    const float z = p.invProjection[2].z * depth + p.invProjection[3].z;
    const float w = p.invProjection[2].w * depth + p.invProjection[3].w;
    return abs(z / (abs(w) > 1e-6f ? w : 1e-6f));
}

// Signed circle-of-confusion RADIUS in pixels. Positive = the point sits in FRONT of
// the focus plane (nearer than focus), negative = behind it. The sign is what lets the
// gather honour occlusion: a near neighbour may spread over anything behind it, a far
// one may not spread over something sharp in front.
static inline float coc_radius(float z, constant DOFParams& p) {
    float zz = max(z, 1e-4f);
    float signed_c = 0.5f * p.cocCoefficient * (p.focusDist - zz) / zz;
    // DIFFRACTION (DH-0882). Stopped down, the Airy disc the aperture itself makes exceeds
    // the acceptable circle of confusion and the WHOLE frame softens — on full frame that
    // crossover is around f/16, and f/22 is an ordinary working aperture on 4×5. It is the
    // same size at every distance, including exactly on the focus plane, so it is a FLOOR on
    // the magnitude rather than a larger coefficient: a bigger coefficient would scale with
    // |z − z_f| and leave the focus plane untouched, which is the one place diffraction is
    // most obvious. The sign is kept so the gather's occlusion rules still apply.
    float mag = min(max(abs(signed_c), p.cocFloor), p.maxRadius);
    return (signed_c < 0.0f) ? -mag : mag;
}

// ── Pass 1: max |CoC| per tile ───────────────────────────────────────────────
kernel void illumi_dof_tile(
    texture2d<float, access::read>  gDepth [[texture(0)]],
    texture2d<float, access::write> outTile[[texture(1)]],
    constant DOFParams&             p      [[buffer(0)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= p.tileW || tid.y >= p.tileH) return;
    uint2 base = tid * p.tileSize;
    float maxAbs = 0.0f;
    float maxNear = 0.0f;       // largest NEAR-field (in front of focus) CoC — the fast gather's bound
    for (uint y = 0; y < p.tileSize; ++y) {
        uint py = base.y + y;
        if (py >= p.height) break;
        for (uint x = 0; x < p.tileSize; ++x) {
            uint px = base.x + x;
            if (px >= p.width) break;
            uint2 s = uint2(px, py);
            float c = coc_radius(view_z(gDepth.read(s).r, s, p), p);
            maxAbs = max(maxAbs, abs(c));
            maxNear = max(maxNear, c);
        }
    }
    outTile.write(float4(maxAbs, maxNear, 0, 0), tid);
}

// ── Pass 1, PARALLEL (the renderer uses it with the opt-in fast gather) ──────
//
// `illumi_dof_tile` reduces a 16×16 tile in ONE thread: 256 dependent depth reads, each with a
// view-depth unprojection and a CoC, in series — over the ~41 k tiles of a 4320 × 2430 frame that
// was 1.9 ms of the Digital Clock's frame, for a max. This is the same max over the same texels
// with one THREADGROUP per tile (a thread per pixel, then a SIMD and a threadgroup max). `max` is
// exact and order-independent, so the tile map is bit-identical to the serial kernel's
// (IlluminatoramaDOFSubjectTests). Dispatch: threadgroups = (tileW, tileH); any threadgroup shape
// works (each thread strides the tile), 16 × 16 is one texel per thread.
kernel void illumi_dof_tile_parallel(
    texture2d<float, access::read>  gDepth [[texture(0)]],
    texture2d<float, access::write> outTile[[texture(1)]],
    constant DOFParams&             p      [[buffer(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float2 partial[32];
    float maxAbs = 0.0f;
    float maxNear = 0.0f;
    uint2 base = tgid * p.tileSize;
    for (uint y = ltid.y; y < p.tileSize; y += tptg.y) {
        uint py = base.y + y;
        if (py >= p.height) break;
        for (uint x = ltid.x; x < p.tileSize; x += tptg.x) {
            uint px = base.x + x;
            if (px >= p.width) break;
            uint2 s = uint2(px, py);
            float c = coc_radius(view_z(gDepth.read(s).r, s, p), p);
            maxAbs = max(maxAbs, abs(c));
            maxNear = max(maxNear, c);
        }
    }
    maxAbs = simd_max(maxAbs);
    maxNear = simd_max(maxNear);
    if (lane == 0u) partial[sg] = float2(maxAbs, maxNear);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float2 v = lane < nsg ? partial[lane] : float2(0.0f);
        v.x = simd_max(v.x);
        v.y = simd_max(v.y);
        if (lane == 0u && tgid.x < p.tileW && tgid.y < p.tileH) outTile.write(float4(v.x, v.y, 0, 0), tgid);
    }
}

// ── Defocus SHADING-RATE tiles (opt-in, the lighting pass's `lightingDefocusShadingMinCoC`) ───
//
// The least |CoC| (px) over each tile's GEOMETRY pixels — sky ignored (+∞ for a tile with none) —
// under exactly the gather's lens (`coc_radius`, clamps included). Runs after the G-buffer and
// before lighting, which shades a one-surface 2×2 quad once where its tile's least |CoC| clears
// the gate (IlluminatoramaLighting.metal, `lightingQuadIsCoarse`). A threadgroup per tile, as
// `illumi_dof_tile_parallel`; dispatch threadgroups = (tileW, tileH).
kernel void illumi_dof_shading_tiles(
    texture2d<float, access::read>  gDepth [[texture(0)]],
    texture2d<float, access::write> outTile[[texture(1)]],
    constant DOFParams&             p      [[buffer(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float partial[32];
    float m = INFINITY;
    uint2 base = tgid * p.tileSize;
    for (uint y = ltid.y; y < p.tileSize; y += tptg.y) {
        uint py = base.y + y;
        if (py >= p.height) break;
        for (uint x = ltid.x; x < p.tileSize; x += tptg.x) {
            uint px = base.x + x;
            if (px >= p.width) break;
            uint2 s = uint2(px, py);
            float d = gDepth.read(s).r;
            if (d >= 0.99999f) continue;                       // sky: the lighting shades it per pixel
            m = min(m, abs(coc_radius(view_z(d, s, p), p)));
        }
    }
    m = simd_min(m);
    if (lane == 0u) partial[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < nsg ? partial[lane] : INFINITY;
        v = simd_min(v);
        if (lane == 0u && tgid.x < p.tileW && tgid.y < p.tileH) outTile.write(float4(min(v, 65504.0f)), tgid);
    }
}

// ── Pass 2: dilate the tile map by the gather's own reach ────────────────────
//
// A source whose disc just covers us can sit `maxRadius` pixels away — that is
// `maxRadius / tileSize` TILES, which is more than one at any sane frame height. Reading a
// fixed 3×3 neighbourhood in the gather therefore truncated the outer edge of a fully
// blurred FOREGROUND disc, a hard rim on exactly the near-field bokeh the scatter exists to
// produce. Widening the gather's own neighbourhood would fix it at 121 texture reads per
// PIXEL on the photo lane; doing the same max once per TILE costs the same reads over a
// thousand times fewer threads, and the gather then reads a single texel.
kernel void illumi_dof_dilate(
    texture2d<float, access::read>  inTile [[texture(0)]],
    texture2d<float, access::write> outTile[[texture(1)]],
    constant DOFParams&             p      [[buffer(0)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= p.tileW || tid.y >= p.tileH) return;
    int tr = clamp(int(ceil(p.maxRadius / float(max(p.tileSize, 1u)))), 1, 8);
    float m = 0.0f, mn = 0.0f;
    for (int dy = -tr; dy <= tr; ++dy) {
        for (int dx = -tr; dx <= tr; ++dx) {
            int2 q = clamp(int2(tid) + int2(dx, dy), int2(0),
                           int2(int(p.tileW) - 1, int(p.tileH) - 1));
            float2 v = inTile.read(uint2(q)).rg;
            m = max(m, v.x);
            mn = max(mn, v.y);
        }
    }
    outTile.write(float4(m, mn, 0, 0), tid);
}

// Is `v` — an offset expressed in units of the confusion disc's radius — inside the
// iris? Returns a soft 0…1 so the disc edge is anti-aliased rather than stair-stepped.
// `aa` is one pixel expressed in those same normalised units.
//
// The blade polygon's boundary radius at angle θ is cos(π/n) / cos(θ mod 2π/n − π/n):
// the apothem over the cosine of the angle off the nearest blade's normal. Optical
// vignetting then intersects that with a unit circle pushed toward the frame centre by
// `offAxis`, which clips the outer flank of the disc into the familiar lemon.
static inline float aperture_mask(float2 v, float aa, float2 offAxis, constant DOFParams& p) {
    float r = length(v);
    float bound = 1.0f;
    if (p.blades >= 3.0f) {
        float n = p.blades;
        float seg = 2.0f * M_PI_F / n;
        float theta = atan2(v.y, v.x) + p.bladeRotation;
        float a = fmod(fmod(theta, seg) + seg, seg) - 0.5f * seg;
        bound = cos(M_PI_F / n) / max(cos(a), 1e-3f);
    }
    float m = smoothstep(bound + aa, bound - aa, r);
    if (p.catsEye > 0.0f) {
        // The barrel's opening, seen off-axis, is a unit circle displaced toward the
        // optical axis; the visible aperture is the intersection of the two.
        float rc = length(v - offAxis);
        m *= smoothstep(1.0f + aa, 1.0f - aa, rc);
    }
    return m;
}

// ── Pass 3: the CoC-weighted prefilter pyramid ───────────────────────────────
//
// The gather spends 24–128 taps on a disc that, wide open, has thousands of pixels. A tap that
// point-samples the frame therefore aliases every detail finer than the spacing between taps —
// a small bright flame is hit by one tap in this pixel and missed in the next, which is the
// grain inside a bokeh ball (DH-0728). Rotating the spiral per pixel only turned that aliasing
// from a lattice into noise. The cure is to prefilter: each tap reads the frame averaged over
// its own share of the disc, so it integrates the flame instead of gambling on hitting it.
//
// This writes the pyramid's base at HALF resolution (full-resolution LOD 1 — below that a tap
// reads the frame directly) as premultiplied colour with the weight in alpha; the renderer's
// mip generation box-filters the rest, and the gather divides the weight back out. The weight
// is the texel's own CoC, so IN-FOCUS pixels all but vanish from the coarse levels: a tap
// behind a sharp subject averages its defocused neighbours, not the subject's edge. Without
// that a sharp silhouette would smear a halo of itself into the bokeh behind it.
kernel void illumi_dof_prefilter(
    texture2d<half,  access::read>  inHDR  [[texture(0)]],
    texture2d<float, access::read>  gDepth [[texture(1)]],
    texture2d<half,  access::write> outPre [[texture(2)]],
    texture2d<half,  access::write> outMinCoC [[texture(3)]],   // sharp-subject mode only
    texture2d<half,  access::write> outNear [[texture(4)]],     // near-field pyramid only
    texture2d<half,  access::write> outNearCoC [[texture(5)]],  // near-field pyramid only
    constant DOFParams&             p      [[buffer(0)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= outPre.get_width() || tid.y >= outPre.get_height()) return;
    uint2 last = uint2(p.width - 1, p.height - 1);
    float3 sum = 0.0f;
    float  wsum = 0.0f;
    float3 nSum = 0.0f;         // the NEAR FIELD alone (in front of the focus plane, past 1.5 px)
    float  nW = 0.0f, nC = 0.0f;
    float  cmin = INFINITY;
    float  smin = INFINITY, smax = -INFINITY;   // signed CoC range (the layered composite's purity test)
    for (uint dy = 0; dy < 2; ++dy) {
        for (uint dx = 0; dx < 2; ++dx) {
            uint2 s = min(tid * 2u + uint2(dx, dy), last);
            float cs = coc_radius(view_z(gDepth.read(s).r, s, p), p);
            float c = abs(cs);
            cmin = min(cmin, c);
            smin = min(smin, cs);
            smax = max(smax, cs);
            // SHARP SUBJECT (opt-in, see the gather): a texel sharper than any tap that reads the
            // pyramid (those taps have |CoC| > 2 px) gets weight EXACTLY 0 — a floor weight is no
            // guard against an HDR-bright subject, which outweighs it by its own contrast.
            float w = p.subjectAware > 0.5f ? smoothstep(1.5f, 2.5f, c) : clamp(c * 0.5f, 0.02f, 1.0f);
            const float3 col = float3(inHDR.read(s).rgb);
            sum  += col * w;
            wsum += w;
            const float wn = cs > 0.0f ? smoothstep(1.5f, 2.5f, cs) : 0.0f;
            nSum += col * wn; nW += wn; nC += cs * wn;
        }
    }
    outPre.write(half4(half3(sum * 0.25f), half(wsum * 0.25f)), tid);
    // The NEAR-FIELD pyramid's base (the layered composite): premultiplied colour + COVERAGE of the
    // texels in front of the focus plane, and CoC × coverage. Box mips of both are what a near-field
    // tap reads: the coverage of its whole footprint, not a hit-or-miss texel (see `dof_layered`).
    if (p.nearPyramid > 0.5f && p.partialOcclusion > 0.5f && p.subjectAware > 0.5f) {
        outNear.write(half4(half3(nSum * 0.25f), half(nW * 0.25f)), tid);
        outNearCoC.write(half4(half(min(nC * 0.25f, 60000.0f))), tid);
    }
    // The MIN-CoC pyramid's base (sharp-subject mode): the gather asks it "is anything in this
    // footprint much sharper than me?" in one read instead of probing the depth buffer. G and B carry
    // the footprint's signed CoC range — "is it one layer?" — for the layered composite, which reads
    // them from an rgba16Float pyramid; the r16Float one every other mode allocates drops them.
    if (p.subjectAware > 0.5f) {
        outMinCoC.write(half4(half(min(cmin, 60000.0f)), half(clamp(smin, -60000.0f, 60000.0f)),
                              half(clamp(smax, -60000.0f, 60000.0f)), 0.0h), tid);
    }
}

// ── Pass 3b (sharp-subject mode): the min-CoC pyramid, one level from the one below ─────────
kernel void illumi_dof_mincoc(
    texture2d<half, access::read>  src [[texture(0)]],
    texture2d<half, access::write> dst [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= dst.get_width() || tid.y >= dst.get_height()) return;
    uint2 last = uint2(src.get_width() - 1, src.get_height() - 1);
    // R = least |CoC|, G = least signed CoC, B = greatest signed CoC (G/B only on an rgba16Float
    // pyramid — an r16Float one reads them as 0 and drops the write).
    half3 m = src.read(min(tid * 2u, last)).rgb;
    for (uint k = 1; k < 4; ++k) {
        half3 v = src.read(min(tid * 2u + uint2(k & 1u, k >> 1u), last)).rgb;
        m = half3(min(m.x, v.x), min(m.y, v.y), max(m.z, v.z));
    }
    dst.write(half4(m.x, m.y, m.z, m.x), tid);
}

// Least |CoC| (px) over the 2×2 texels of min-CoC level `m` round `uv` — which covers the 2×2
// bilinear footprint of every level below it.
static inline float dof_min_coc(texture2d<half, access::read> minCoC, float2 uv, uint m) {
    m = min(m, minCoC.get_num_mip_levels() - 1u);
    int2 dim = int2(minCoC.get_width(m), minCoC.get_height(m));
    int2 b = int2(floor(uv * float2(dim) - 0.5f));
    float c = INFINITY;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            uint2 q = uint2(clamp(b + int2(i, j), int2(0), dim - 1));
            c = min(c, float(minCoC.read(q, m).r));
        }
    }
    return c;
}

// ── PARTIAL OCCLUSION, helpers (the layered composite, `dof_layered`) ────────────────────────
//
// How far apart two signed CoCs must be to be different LAYERS (px): a surface's own slant and the
// depth buffer's quantisation stay inside it, a separate object does not.
static inline float dof_layer_gap(float c) {
    return 0.5f + 0.04f * abs(c);
}

// Signed CoC range (px) over the 2×2 texels of min-CoC level `m` round `uv` — which covers the 2×2
// bilinear footprint of every level below it. (G = least, B = greatest; the rgba16Float pyramid.)
static inline float2 dof_minmax_signed(texture2d<half, access::read> minCoC, float2 uv, uint m) {
    m = min(m, minCoC.get_num_mip_levels() - 1u);
    int2 dim = int2(minCoC.get_width(m), minCoC.get_height(m));
    int2 b = int2(floor(uv * float2(dim) - 0.5f));
    float lo = INFINITY, hi = -INFINITY;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            uint2 q = uint2(clamp(b + int2(i, j), int2(0), dim - 1));
            half4 v = minCoC.read(q, m);
            lo = min(lo, float(v.g));
            hi = max(hi, float(v.b));
        }
    }
    return float2(lo, hi);
}

// The aperture's area in units of the disc radius² — the iris polygon clipped by the barrel's
// cat's-eye circle, exactly `aperture_mask`'s shape — i.e. what one fully covered disc weighs in the
// gather's m/c² units (π for a round iris, (n/2)·sin(2π/n) for n blades, ~12 % less at the corners of
// a 0.2 cat's eye). ½∫r(θ)² dθ by the midpoint rule in 24 steps: within 0.75 % of the exact area for
// round, 5-, 6-, 7- and 8-bladed irises at every field position a cat's eye ≤ 0.2 reaches.
static inline float dof_aperture_area(float2 offAxis, constant DOFParams& p) {
    const int K = 24;
    const float dth = 2.0f * M_PI_F / float(K);
    const float n = p.blades;
    const float seg = 2.0f * M_PI_F / max(n, 3.0f);
    const float oo = dot(offAxis, offAxis);
    float a2 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float th = (float(k) + 0.5f) * dth;
        float r = 1.0f;
        if (n >= 3.0f) {
            float a = fmod(fmod(th + p.bladeRotation, seg) + seg, seg) - 0.5f * seg;
            r = cos(M_PI_F / n) / max(cos(a), 1e-3f);
        }
        if (p.catsEye > 0.0f) {
            const float eo = cos(th) * offAxis.x + sin(th) * offAxis.y;
            r = min(r, eo + sqrt(max(eo * eo - oo + 1.0f, 0.0f)));
        }
        a2 += r * r;
    }
    return 0.5f * a2 * dth;
}

// ── Pass 4: the gather ───────────────────────────────────────────────────────
constexpr sampler dofLinear(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler dofTrilinear(coord::normalized, filter::linear, mip_filter::linear,
                               address::clamp_to_edge);

// ── PARTIAL OCCLUSION: the LAYERED COMPOSITE (opt-in `partialOcclusion`, sharp-subject mode) ─────
//
// What the thin lens does at a pixel p whose own surface has signed CoC c_p. Of the rays through the
// pupil, a share α_F is stopped by something NEARER than p's surface — a defocused foreground whose
// disc reaches p. The rest cross p's depth at x = p + u·c_p, inside p's OWN DISC (radius |c_p|), where
// the frame shows either p's own layer (the ray ends there) or something FARTHER (the ray passes the
// layer's edge, or a gap in it, and reaches what lies behind). So
//
//     I(p) = α_F · F  +  (1 − α_F) · D
//
//  • F, α_F — FRONT taps (c_s > c_p + gap) whose own disc reaches p, at their scatter weight m/c_s²:
//    F is their normalised colour and α_F = Σw / A their ABSOLUTE coverage (A: the aperture's area,
//    what a fully covered disc weighs) — composited OVER the rest, not averaged into it.
//  • D — the average over p's own disc (the aperture's shape, radius |c_p|) of what the frame shows
//    there, the front taps' positions left out: p's own layer where it is, and through every HOLE —
//    a position showing something farther — what shows through it. The share of p's rays that pass
//    its layer is then exactly the holes' share of its disc, on either side of the focus plane.
//
// Farther taps OUTSIDE the own disc weigh nothing (their light reaches p only through a hole, which
// is already counted), and own-layer taps outside it never reach p. The frame cannot show what a
// foreground hides, so a hole carries the colour at the hole itself — the background just past the
// edge — rather than the ray's true landing point further out (for layers on opposite sides of focus
// that point is hidden behind p's own surface).
//
// The rules this replaces — a farther tap at |c_p|/|c_s| of its weight, within |c_p| + 1; a nearer
// one averaged in by its weight — measured against an aperture-sampled reference of the Digital
// Clock (`dofReferenceAccumulation`): a defocused wall's edge against the sky kept 97 % wall where the
// lens has 50 % — a hard step at the silhouette; a defocused FOREGROUND member (a crane's mast, whose
// holes show the background behind the focus plane) stayed opaque right to its edge, a lattice cut out
// sharp inside its own soft halo; and a foreground over a sharp subject took 1/3 of the pixel at the
// silhouette where the lens takes 1/2. `IlluminatoramaDOFPartialOcclusionTests` holds each case to
// the exact layered thin-lens integral.
//
// Taps: p's own disc and the near-field disc round p (one spiral, or two when their sizes differ —
// see below). A tap reads the prefilter pyramid only where the footprint belongs to its term (the
// min-CoC pyramid's signed-CoC range): a front tap's must be its own layer — a thin member
// prefiltered together with the sky behind it would carry the sky into its own coverage — and a disc
// tap's must hold nothing nearer than p's layer; else its own texel.
//
// α_F is an UNNORMALISED estimate — each front tap adds its own weight — so its error is the sampling
// of the fronts' discs, and two things keep that sampling fine enough (measured on the Digital Clock
// against the reference: a dotted line along every defocused edge, 32–40 % of the silhouettes' error):
//  • NEAR-FIELD COVERAGE (`nearPyramid`, with the layered composite). A near-field front read as a
//    hit-or-miss texel makes α a count of hits — the edge of a blurred hard hat or a crane's falls
//    against the wall came out crunchy. Where every near-field texel is a front of p (p's own layer is
//    not in the near field), the near spiral reads the NEAR-FIELD pyramid instead: the coverage κ of
//    the tap's footprint by texels in front of the focus plane, their colour and mean CoC — so a tap
//    half on a member counts half, and α integrates the members instead of counting hits.
//  • SMALL FRONTS. A front behind the focus plane has a disc smaller than p's own (it is nearer the
//    focus plane than p); when it is much smaller, the own-disc spiral meets its disc with a couple of
//    taps — a wall 7 px out of focus against a sky 32 px out: α in steps of 0.17, a dotted line along
//    the wall's edge. The own disc only discovers such fronts; a third spiral, fitted to the widest,
//    counts them.
// (And the near spiral counts the near field ONLY: it used to add every front it met — a front
// behind the focus plane inside the near spiral's disc was counted by both spirals, which doubled the
// wall's coverage at the sky's edge wherever a near-field member was within reach: measured 0.30
// where the lens has 0.41.)
static inline float3 dof_layered(float2 pc, float3 center, float centerCoC, float reach, float nearMax,
                                 float tapsPerPx, int minTaps, int maxTaps,
                                 texture2d<half,  access::sample> inHDR,
                                 texture2d<float, access::read>   gDepth,
                                 texture2d<half,  access::sample> prefiltered,
                                 texture2d<half,  access::read>   minCoC,
                                 texture2d<half,  access::sample> nearField,
                                 texture2d<half,  access::sample> nearCoC,
                                 constant DOFParams& p)
{
    const float cp = centerCoC, ap = abs(cp);
    const float gap = dof_layer_gap(cp);
    const float2 texel = 1.0f / float2(p.width, p.height);
    const int2 pix = int2(floor(pc));
    const int2 last = int2(int(p.width) - 1, int(p.height) - 1);
    const float ign = fract(52.9829189f * fract(0.06711056f * float(pix.x) + 0.00583715f * float(pix.y)));
    const float2 offAxis = -(pc * texel * 2.0f - 1.0f) * p.catsEye;
    // Anything NEARER than our layer within the reach? If not, there are no front taps and every
    // footprint of our own disc may be read prefiltered (own layer and holes weigh alike in D).
    const uint mR = uint(ceil(log2(max(2.0f * reach, 1.0f))));
    const float spanHi = dof_minmax_signed(minCoC, pc * texel, mR).y;
    const bool noFront = spanHi <= cp + gap;
    // The OWN DISC (radius |c_p|) holds D and the fronts BEHIND the focus plane (a nearer layer
    // behind focus has a disc no wider than ours); the fronts IN FRONT of the focus plane reach us
    // from within the widest near-field disc round us, `nearTop` (the tighter of the dilated tile
    // bound and the min-CoC pyramid's range over the reach). One spiral covers both while they are of
    // a size. When one is much the larger, each gets a spiral of its own — the sky round a crane's
    // thin falls (own disc 48 px, the falls' 10) would otherwise meet a thin member's disc with two
    // taps (speckle), and a nearly sharp pixel beside a blurred mast (own disc 2 px, the mast's 16)
    // would estimate its own disc from one — and the near spiral alone counts the near-field fronts
    // inside its disc, so every front is counted exactly once.
    const float aO = max(ap, 0.5f);
    const float nearTop = min(min(nearMax, spanHi), p.maxRadius);
    const bool nearFronts = !noFront && nearTop > max(cp + gap, 0.0f);
    const float rD = ap >= 1.0f ? min(ap + 0.5f, reach) : 0.0f;
    const float rN = nearFronts ? min(reach, nearTop + 1.0f) : 0.0f;
    // NEAR-FIELD COVERAGE (`nearPyramid`): when every near-field texel is a front of ours (our own layer
    // is not in the near field — the near-field pyramid's texels are the ones past 1.5 px in front of
    // focus), the near-field fronts are read as COVERAGE from their own spiral, always separate from
    // the own disc, and the own disc leaves every near-field texel out.
    const bool nearPyr = p.nearPyramid > 0.5f && nearFronts && cp + gap < 1.5f;
    const bool split = nearPyr ? rN > 0.0f
                               : (rN > 0.0f && rD > 0.0f && (rN < 0.6f * rD || rD < 0.6f * rN));
    const float r1 = split ? rD : max(rD, rN);   // (0 when our own disc is under a pixel)
    const int n1 = r1 >= 1.0f ? clamp(int(r1 * tapsPerPx), minTaps, maxTaps) : 0;
    const int n2 = split ? clamp(int(rN * tapsPerPx), minTaps, maxTaps) : 0;

    // D: the own disc, seeded with the centre pixel (1 px² of it at the disc's density).
    const float wCentre = 1.0f / max(aO * aO, 1.0f / M_PI_F);
    float3 dSum = center * wCentre;
    float  dW   = wCentre;
    float3 fSum = 0.0f;
    float  fW   = 0.0f;
    // SMALL FRONTS (behind the focus plane, a disc of ≥ 1.5 px but < 0.6 of the spiral that met them):
    // the own disc only DISCOVERS them (and keeps a fallback estimate, for a pixel whose third spiral
    // finds none); a third spiral, fitted to the widest of them, counts them.
    float  rSmall = 0.0f;
    float3 sSum = 0.0f;
    float  sW   = 0.0f;
    for (int set = 0; set < 3; ++set) {
        const float rOut = set == 0 ? r1 : (set == 1 ? rN : min(rSmall + 1.0f, reach));
        const int n = set == 0 ? n1 : (set == 1 ? n2
                    : (rSmall > 0.0f ? clamp(int(rOut * tapsPerPx), minTaps, maxTaps) : 0));
        if (n <= 0) continue;
        if (set == 2) { sSum = 0.0f; sW = 0.0f; }              // the fallback gives way to the real count
        const float area = M_PI_F * rOut * rOut / float(n);          // px² each tap stands for
        const float tapFootprint = p.prefilterScale * sqrt(area);
        const float rot = (ign + 0.37f * float(set)) * 6.28318531f;
        // The near spiral in near-pyramid mode: each tap reads the near field's COVERAGE κ of its
        // footprint, its colour and its mean CoC, and scatters that share of the footprint's area. The
        // pyramid's base is already 2 px and a trilinear read spans two texels, so the level is the
        // footprint's less 2: a footprint as wide as the tap spacing smoothed a bright member's rim a
        // further 3–4 px past its disc (measured on the night mast); half of it does not.
        const float lodN = max(log2(max(min(tapFootprint, nearTop * 0.5f), 1.0f)) - 2.0f, 0.0f);
        for (int i = 0; i < n; ++i) {
            const float ft = clamp((float(i) + 0.5f + (ign - 0.5f)) / float(n), 1e-4f, 1.0f);
            const float rr = sqrt(ft) * rOut;
            if (rr < 0.5f) continue;                                   // the centre pixel: counted above
            const float a = float(i) * kGoldenAngle + rot;
            const float2 off = float2(cos(a), sin(a)) * rr;
            if (set == 1 && nearPyr) {
                const float2 suv = (pc + off) * texel;
                const half4 nf = nearField.sample(dofTrilinear, suv, level(lodN));
                const float kappa = float(nf.a);
                if (kappa < 1e-3f) continue;
                const float cN = max(float(nearCoC.sample(dofTrilinear, suv, level(lodN)).r) / kappa, 0.5f);
                const float wN = area * kappa * aperture_mask(off / cN, 1.0f / cN, offAxis, p) / (cN * cN);
                if (wN <= 0.0f) continue;
                fSum += float3(nf.rgb) * (wN / kappa);
                fW += wN;
                continue;
            }
            const uint2 su = uint2(clamp(int2(floor(pc + off)), int2(0), last));
            const float ci = coc_radius(view_z_zw(gDepth.read(su).r, p), p);
            const float ai = max(abs(ci), 1e-3f);
            const bool front = ci - cp > gap;
            float frontShare = 1.0f;
            // A small front: behind the focus plane (so we are too, and farther), a disc of ≥ 1.5 px that
            // is much smaller than the own-disc spiral.
            const bool small = front && ci < -1.5f && ai < 0.6f * r1;
            if (set == 2 && !small) continue;
            if (set == 0 && small) rSmall = max(rSmall, ai);
            if (front) {
                if (set == 1 && ci <= 0.0f) continue;      // the near spiral counts the near field only
                if (nearPyr) {
                    // The near spiral counts the near field (as coverage); a texel between, nearly in
                    // focus, is counted here for the share the near-field pyramid leaves out.
                    frontShare = ci > 0.0f ? 1.0f - smoothstep(1.5f, 2.5f, ci) : 1.0f;
                    if (frontShare <= 0.0f) continue;
                } else if (set == 0 && split && ci > 0.0f && rr < rN) {
                    continue;                              // a near-field front: the second spiral's
                }
            } else if (set == 1 || rr > rD) {
                continue;                                  // own / farther, outside our own disc
            }
            // A front tap: does its own disc reach us? A disc tap: a point of our own disc. Either way the
            // PUPIL point is the offset over the SIGNED CoC (a point at c is seen from pupil point u at
            // p + c·u): in front of the focus plane the disc is the aperture turned half round.
            const float rad = front ? (ci < 0.0f ? -ai : ai) : (cp < 0.0f ? -aO : aO);
            const float w = frontShare * area * aperture_mask(off / rad, 1.0f / abs(rad), offAxis, p) / (rad * rad);
            if (w <= 0.0f) continue;
            // Colour: prefiltered over the tap's footprint where that footprint belongs to the tap's
            // term — a FRONT tap's footprint must be its own layer (its colour is weighed by its own
            // disc); a disc tap's must hold nothing nearer than our layer and nothing too sharp for the
            // pyramid (own layer and holes weigh alike there). Else the texel itself.
            float3 col;
            bool haveCol = false;
            const float lod = log2(max(min(tapFootprint, ai * 0.5f), 1.0f));
            if (lod > 0.0f) {
                const float2 suv = (pc + off) * texel;
                bool ok = !front && noFront;
                if (!ok) {
                    const float2 fm = dof_minmax_signed(minCoC, suv, uint(ceil(lod - 1.0f)) + 1u);
                    const float tg = dof_layer_gap(ci);
                    ok = front ? (fm.x >= ci - tg && fm.y <= ci + tg) : fm.y <= cp + gap;
                }
                if (ok) {
                    const half4 pf = prefiltered.sample(dofTrilinear, suv, level(max(lod - 1.0f, 0.0f)));
                    if (pf.a > (front ? 0.05h : 0.97h)) {
                        col = float3(pf.rgb) / float(pf.a);
                        if (lod < 1.0f) col = mix(float3(inHDR.read(su).rgb), col, lod);
                        haveCol = true;
                    }
                }
            }
            if (!haveCol) col = float3(inHDR.read(su).rgb);
            if (small)      { sSum += col * w; sW += w; }
            else if (front) { fSum += col * w; fW += w; }
            else            { dSum += col * w; dW += w; }
        }
    }
    fSum += sSum; fW += sW;
    const float3 disc = dSum / dW;
    if (fW <= 0.0f) return disc;
    const float alpha = saturate(fW / dof_aperture_area(offAxis, p));
    return mix(disc, fSum / fW, alpha);
}


// ── FAST GATHER (opt-in, `fastGather`, sharp-subject mode only) ──────────────────────────────
//
// The gather's cost is taps × pixels, and taps follow the REACH — which the dilated tile max sets
// for every pixel within ~maxRadius of anything defocused, so an in-focus subject in front of a far
// blurred wall paid the wall's 100+ taps per pixel (Digital Clock: 65–73 ms at 2880×1620). Three
// changes, each measured against the full gather (IlluminatoramaDOFSubjectTests):
//
//  1. EXACT REACH. In sharp-subject mode a tap contributes only if (a) it lies within this pixel's
//     own blur of it (behind-taps are cut at rr > |c| + 1) or (b) its own disc reaches us — and a
//     tap nearer than us whose disc is wider than ours must be IN FRONT of the focus plane, so the
//     dilated tile max of the near field (`tileMax.g`) bounds it. reach' = max(|c| + 1.5, near + 1)
//     drops every tap that could only ever have weighed 0; the estimate's expectation is unchanged.
//  2. A SPARSER RING. Taps prefiltered over their footprint (the CoC-weighted pyramid) at 3 per px
//     of reach (2 in the half-res pass; 8…64) instead of 6 (24…128): the footprint grows to keep
//     the disc covered.
//  3. HALF-RESOLUTION BACKGROUND. A pixel at least `halfMinCoC` out of focus is a smooth average
//     over ≥ 3 px: it takes the bilinear half-res gather (`illumi_dof_half`) when the four half-res
//     texels it reads were gathered from the same layer (|CoC| within 30 %) and nothing within
//     ~8 px of it is sharper than half its CoC (the min-CoC pyramid) — else it gathers itself, so a
//     bright sharp subject is never carried a half-res texel further into the blur round it.
//  4. QUARTER-RESOLUTION BACKGROUND (opt-in, `quarterMinCoC` > 0; 0 ⇒ none of this runs). Past
//     `quarterMinCoC` (e.g. 12 px) a pixel's disc averages > 450 px, so it takes a bilinear read of a
//     gather made once per 4 × 4 block (`illumi_dof_quarter`) under the same one-layer rules, with
//     the min-CoC guard widened to the quarter tier's reach (~16 px, level 3); the half-res pass
//     skips every texel whose pixels will all take the quarter result (it evaluates exactly the
//     full pass's test), so the far wall of a macro shot costs a quarter of the half-res gathers.

constant float kDOFFastTapsPerPx = 3.0f;
// The half-res pass: a texel stands for 2×2 pixels already averaged by the upsample.
constant float kDOFFastHalfTapsPerPx = 2.0f;
// The quarter-res pass (opt-in `quarterMinCoC`): the same ring as the half-res pass — the tap
// count is capped at 64 either way for the discs it serves (≥ 12 px); what it saves is 3 of every
// 4 GATHERS, not taps per gather.
constant float kDOFFastQuarterTapsPerPx = 2.0f;
constant int   kDOFFastMinTaps   = 8;
// The LAYERED composite's share of those densities (partial occlusion + fast gather): its own-disc,
// near-field and small-front spirals each sample only their own term, and the near field is read as
// prefiltered coverage — 2.5 taps per px of reach instead of 3 measured 0.9 ms less at 2880 × 1620
// (Digital Clock 14:10, 10.2 → 9.4 ms for the whole DOF) for 0.12 codes more mean error at the
// silhouettes against the aperture-sampled reference (3.20 → 3.32).
constant float kDOFFastLayeredTapScale = 0.833f;
constant int   kDOFFastMaxTaps   = 64;

// The gather at continuous full-res position `pc` (a pixel centre is gid + 0.5) — the sharp-subject
// gather of `illumi_dof`, over the exact reach. Returns premultiplied (Σ col·w, Σ w).
static inline float4 dof_fast_core(float2 pc, float3 center, float centerZ, float centerCoC, float tapsPerPx,
                                   texture2d<half,  access::sample> inHDR,
                                   texture2d<float, access::read>   gDepth,
                                   texture2d<float, access::read>   tileMax,
                                   texture2d<half,  access::sample> prefiltered,
                                   texture2d<half,  access::read>   minCoC,
                                   texture2d<half,  access::sample> nearField,
                                   texture2d<half,  access::sample> nearCoC,
                                   constant DOFParams& p)
{
    const float centerAbs = abs(centerCoC);
    const int2 pix = int2(floor(pc));
    uint2 t = clamp(uint2(max(pix, int2(0))) / p.tileSize, uint2(0), uint2(p.tileW - 1, p.tileH - 1));
    const float2 tm = tileMax.read(t).rg;
    float reach = min(max(centerAbs, tm.x), p.maxRadius);
    reach = min(reach, max(centerAbs + 1.5f, tm.y + 1.0f));
    if (reach < 0.75f) return float4(center, 1.0f);
    if (p.partialOcclusion > 0.5f) {
        return float4(dof_layered(pc, center, centerCoC, reach, tm.y, tapsPerPx * kDOFFastLayeredTapScale,
                                  kDOFFastMinTaps, kDOFFastMaxTaps,
                                  inHDR, gDepth, prefiltered, minCoC, nearField, nearCoC, p), 1.0f);
    }
    const int taps = clamp(int(reach * tapsPerPx), kDOFFastMinTaps, kDOFFastMaxTaps);
    const float tapFootprint = p.prefilterScale * reach * sqrt(M_PI_F / float(taps));
    const float ign = fract(52.9829189f * fract(0.06711056f * float(pix.x) + 0.00583715f * float(pix.y)));
    const float rot = ign * 6.28318531f;
    const float2 uv = pc / float2(p.width, p.height) * 2.0f - 1.0f;
    const float2 offAxis = -uv * p.catsEye;
    const float2 texel = 1.0f / float2(p.width, p.height);
    float3 sum = center * 1e-4f;
    float  wsum = 1e-4f;
    const uint mR = uint(ceil(log2(max(2.0f * reach, 1.0f))));
    const bool mixedLayers = dof_min_coc(minCoC, pc * texel, mR) < 0.5f * reach;
    {
        float cA = max(centerAbs, 0.5f);
        float wc = float(taps) / (reach * reach * max(M_PI_F * cA * cA, 1.0f));
        sum  += center * wc;
        wsum += wc;
    }
    for (int i = 0; i < taps; ++i) {
        float ft = clamp((float(i) + 0.5f + (ign - 0.5f)) / float(taps), 1e-4f, 1.0f);
        float rr = sqrt(ft) * reach;
        if (rr < 0.5f) continue;
        float a  = float(i) * kGoldenAngle + rot;
        float2 off = float2(cos(a), sin(a)) * rr;
        float2 suv = (pc + off) * texel;
        int2 s = clamp(int2(floor(pc + off)), int2(0), int2(int(p.width) - 1, int(p.height) - 1));
        uint2 su = uint2(s);
        float tapZ   = view_z(gDepth.read(su).r, su, p);
        float tapCoC = coc_radius(tapZ, p);
        float tapAbs = max(abs(tapCoC), 1e-3f);
        float w  = aperture_mask(off / (tapCoC < 0.0f ? -tapAbs : tapAbs), 1.0f / tapAbs, offAxis, p);
        if (tapZ > centerZ) w *= saturate(centerAbs / tapAbs) * saturate(centerAbs + 1.0f - rr);
        if (w <= 0.0f) continue;
        w /= (tapAbs * tapAbs);
        float lod = log2(max(min(tapFootprint, tapAbs * 0.5f), 1.0f));
        half4 pf = lod > 0.0f
            ? prefiltered.sample(dofTrilinear, suv, level(max(lod - 1.0f, 0.0f)))
            : half4(0.0h);
        if (mixedLayers && lod > 0.0f && pf.a > 0.05h
            && dof_min_coc(minCoC, suv, uint(ceil(lod - 1.0f)) + 1u) < 0.5f * tapAbs) {
            pf = half4(0.0h);
        }
        float3 col;
        if (pf.a > 0.05h && lod >= 1.0f) {
            col = float3(pf.rgb) / float(pf.a);
        } else {
            col = float3(inHDR.read(su).rgb);
            if (pf.a > 0.05h) col = mix(col, float3(pf.rgb) / float(pf.a), lod);
        }
        sum  += col * w;
        wsum += w;
    }
    return float4(sum, wsum);
}

// Bilinear read of a reduced-resolution tier of the fast gather (`step` = 4: the quarter tier) at
// full-res pixel `gid`. True — with the colour in `outCol` — only when all four texels were
// gathered, from this pixel's own layer (the SIGNED CoC within 30 % + ½ px: a texel from the other
// side of the focus plane is another layer however alike its blur), AND the blurred field is SMOOTH
// across them (their luminances within 25 %): a disc blur is smooth except at the RIM of the bokeh
// ball of an HDR-bright point or line, which is as sharp as the aperture's edge — reconstructed
// from a 4-px grid that rim would soften and shift (measured on the Digital Clock's moonlit sill at
// 23:47, up to 28 codes), so a pixel beside one goes to the half tier. False ⇒ the caller falls
// through to the next tier. (The half-res tier applies its own inline test in `dof_fast_full`.)
static inline bool dof_tier_read(texture2d<half, access::read> tier, uint2 gid, float centerCoC, float step,
                                 thread float3 &outCol) {
    const float centerAbs = abs(centerCoC);
    float2 hp = (float2(gid) + 0.5f) / step - 0.5f;
    int2 h0 = int2(floor(hp));
    float2 f = hp - float2(h0);
    int2 hmax = int2(int(tier.get_width()) - 1, int(tier.get_height()) - 1);
    float3 acc = 0.0f;
    float lmin = INFINITY, lmax = 0.0f;
    for (int k = 0; k < 4; ++k) {
        int2 q = clamp(h0 + int2(k & 1, k >> 1), int2(0), hmax);
        float4 v = float4(tier.read(uint2(q)));
        if (!(v.a != 0.0f && abs(v.a - centerCoC) <= 0.3f * centerAbs + 0.5f)) return false;
        float l = dot(v.rgb, float3(0.2126f, 0.7152f, 0.0722f));
        lmin = min(lmin, l); lmax = max(lmax, l);
        acc += v.rgb * (((k & 1) ? f.x : 1.0f - f.x) * ((k >> 1) ? f.y : 1.0f - f.y));
    }
    if (lmax > 1.25f * lmin + 1e-4f) return false;     // a rim or an edge in the blur: finer tier
    outCol = acc;
    return true;
}

// Would full-res pixel `gid` (signed CoC `centerCoC`) take the QUARTER tier? The full pass's exact
// test, shared with the half-res pass so it can skip the texels nobody will read.
static inline bool dof_takes_quarter(uint2 gid, float centerCoC, texture2d<half, access::read> minCoC,
                                     texture2d<half, access::read> quarterRes, constant DOFParams& p,
                                     thread float3 &outCol) {
    const float centerAbs = abs(centerCoC);
    if (!(p.quarterMinCoC > 0.0f) || centerAbs < p.quarterMinCoC) return false;
    // One layer out to the quarter tier's reach: 2×2 texels of min-CoC level 3 cover ≥ ±16 px.
    const float2 uvC = (float2(gid) + 0.5f) / float2(p.width, p.height);
    if (dof_min_coc(minCoC, uvC, 3u) < 0.5f * centerAbs) return false;
    return dof_tier_read(quarterRes, gid, centerCoC, 4.0f, outCol);
}

// Full-res pass of the fast gather: the quarter-res background (opt-in) or the half-res one where
// they apply, else the exact-reach gather at this pixel.
static inline void dof_fast_full(uint2 gid, half3 centerH, float centerZ, float centerCoC,
                                 texture2d<half,  access::sample> inHDR,
                                 texture2d<float, access::read>   gDepth,
                                 texture2d<half,  access::write>  outHDR,
                                 texture2d<float, access::read>   tileMax,
                                 texture2d<half,  access::sample> prefiltered,
                                 texture2d<half,  access::read>   minCoC,
                                 texture2d<half,  access::read>   halfRes,
                                 texture2d<half,  access::read>   quarterRes,
                                 texture2d<half,  access::sample> nearField,
                                 texture2d<half,  access::sample> nearCoC,
                                 constant DOFParams& p)
{
    const float centerAbs = abs(centerCoC);
    {
        float3 qc;
        if (dof_takes_quarter(gid, centerCoC, minCoC, quarterRes, p, qc)) {
            outHDR.write(half4(half3(qc), 1.0h), gid);
            return;
        }
    }
    // Half-res only where the neighbourhood is ONE layer: a sharper subject within ~8 px (2×2
    // texels of min-CoC level 2) would be carried a half-res texel further out by the upsample.
    const float2 uvC = (float2(gid) + 0.5f) / float2(p.width, p.height);
    if (centerAbs >= p.halfMinCoC && dof_min_coc(minCoC, uvC, 2u) >= 0.5f * centerAbs) {
        float2 hp = (float2(gid) + 0.5f) * 0.5f - 0.5f;
        int2 h0 = int2(floor(hp));
        float2 f = hp - float2(h0);
        int2 hmax = int2(int(halfRes.get_width()) - 1, int(halfRes.get_height()) - 1);
        float4 acc = 0.0f;
        bool ok = true;
        for (int k = 0; k < 4 && ok; ++k) {
            int2 q = clamp(h0 + int2(k & 1, k >> 1), int2(0), hmax);
            float4 v = float4(halfRes.read(uint2(q)));
            ok = v.a != 0.0f && abs(v.a - centerCoC) <= 0.3f * centerAbs + 0.5f;     // signed: same layer
            acc += float4(v.rgb, 1.0f) * (((k & 1) ? f.x : 1.0f - f.x) * ((k >> 1) ? f.y : 1.0f - f.y));
        }
        if (ok) { outHDR.write(half4(half3(acc.rgb), 1.0h), gid); return; }
    }
    float4 g = dof_fast_core(float2(gid) + 0.5f, float3(centerH), centerZ, centerCoC, kDOFFastTapsPerPx,
                             inHDR, gDepth, tileMax,
                             prefiltered, minCoC, nearField, nearCoC, p);
    outHDR.write(half4(half3(g.rgb / max(g.a, 1e-6f)), 1.0h), gid);
}

// Half-res pass of the fast gather: one gather per 2×2 block whose four pixels are one layer (same
// side of focus, |CoC| within 20 % and ≥ `halfMinCoC`); alpha = the block's SIGNED CoC, 0 = "gather
// yourself" (the block straddles a silhouette or is near focus).
kernel void illumi_dof_half(
    texture2d<half,  access::sample> inHDR  [[texture(0)]],
    texture2d<float, access::read>   gDepth [[texture(1)]],
    texture2d<half,  access::write>  outHalf[[texture(2)]],
    texture2d<float, access::read>   tileMax[[texture(3)]],
    texture2d<half,  access::sample> prefiltered [[texture(4)]],
    texture2d<half,  access::read>   minCoC [[texture(5)]],
    // The quarter tier's result (read only when `quarterMinCoC` > 0; any texture otherwise).
    texture2d<half,  access::read>   quarterRes [[texture(6)]],
    texture2d<half,  access::sample> nearField [[texture(7)]],   // near-field pyramid only
    texture2d<half,  access::sample> nearCoC [[texture(8)]],     // near-field pyramid only
    constant DOFParams&              p      [[buffer(0)]],
    uint2 hid [[thread_position_in_grid]])
{
    if (hid.x >= outHalf.get_width() || hid.y >= outHalf.get_height()) return;
    uint2 last = uint2(p.width - 1, p.height - 1);
    float cmin = INFINITY, cmax = 0.0f, zs = 0.0f, cs = 0.0f;
    float3 col = 0.0f;
    int pos = 0;
    float ca[4];
    for (uint k = 0; k < 4; ++k) {
        uint2 s = min(hid * 2u + uint2(k & 1u, k >> 1u), last);
        float z = view_z(gDepth.read(s).r, s, p);
        float c = coc_radius(z, p);
        ca[k] = c;
        cmin = min(cmin, abs(c)); cmax = max(cmax, abs(c));
        pos += c > 0.0f ? 1 : 0;
        zs += z; cs += c;
        col += float3(inHDR.read(s).rgb);
    }
    if (cmin < p.halfMinCoC || cmax - cmin > 0.2f * cmax || (pos != 0 && pos != 4)) {
        outHalf.write(half4(0.0h), hid);
        return;
    }
    // QUARTER TIER (opt-in): if every pixel this texel serves will take the quarter-res result —
    // the full pass's own test, evaluated exactly — nobody reads this texel: skip its gather.
    if (p.quarterMinCoC > 0.0f && cmin >= p.quarterMinCoC) {
        bool allQuarter = true;
        for (uint k = 0; k < 4 && allQuarter; ++k) {
            uint2 s = min(hid * 2u + uint2(k & 1u, k >> 1u), last);
            float3 unused;
            allQuarter = dof_takes_quarter(s, ca[k], minCoC, quarterRes, p, unused);
        }
        if (allQuarter) {
            outHalf.write(half4(0.0h), hid);
            return;
        }
    }
    float4 g = dof_fast_core(float2(hid * 2u) + 1.0f, col * 0.25f, zs * 0.25f, cs * 0.25f, kDOFFastHalfTapsPerPx,
                             inHDR, gDepth, tileMax, prefiltered, minCoC, nearField, nearCoC, p);
    outHalf.write(half4(half3(g.rgb / max(g.a, 1e-6f)), half(cs * 0.25f)), hid);        // the block's SIGNED CoC
}

// Quarter-res pass of the fast gather (opt-in, `quarterMinCoC` > 0): one gather per 4×4 block whose
// sixteen pixels are one layer (same side of focus, |CoC| within 20 % and ≥ `quarterMinCoC`);
// alpha = the block's |CoC|, 0 = decline (the half and full tiers take over). A disc of ≥ 12 px
// radius is a smooth average over > 450 px: a bilinear reconstruction from a 4-px grid adds the
// tent's σ (4/√6 = 1.63 px) to the disc's (R/2 ≥ 6 px) — ≤ 3.6 % wider in σ at the 12-px gate, ~1 % at
// a 32-px clamp — and the gather at the block centre keeps every sharp-subject rule. The block's own colour is the prefilter pyramid's
// level-1 texel: every pixel here is past the prefilter's sharp cut (|CoC| ≥ 2.5 px, weight 1), so
// that texel IS their mean. Must run after the prefilter's mips and the min-CoC pyramid, and before
// `illumi_dof_half` (which skips the texels this tier serves).
kernel void illumi_dof_quarter(
    texture2d<half,  access::sample> inHDR  [[texture(0)]],
    texture2d<float, access::read>   gDepth [[texture(1)]],
    texture2d<half,  access::write>  outQuarter [[texture(2)]],
    texture2d<float, access::read>   tileMax[[texture(3)]],
    texture2d<half,  access::sample> prefiltered [[texture(4)]],
    texture2d<half,  access::read>   minCoC [[texture(5)]],
    texture2d<half,  access::sample> nearField [[texture(6)]],   // near-field pyramid only
    texture2d<half,  access::sample> nearCoC [[texture(7)]],     // near-field pyramid only
    constant DOFParams&              p      [[buffer(0)]],
    uint2 qid [[thread_position_in_grid]])
{
    if (qid.x >= outQuarter.get_width() || qid.y >= outQuarter.get_height()) return;
    uint2 last = uint2(p.width - 1, p.height - 1);
    float cmin = INFINITY, cmax = 0.0f, zs = 0.0f, cs = 0.0f;
    int pos = 0;
    for (uint k = 0; k < 16; ++k) {
        uint2 s = min(qid * 4u + uint2(k & 3u, k >> 2u), last);
        float z = view_z(gDepth.read(s).r, s, p);
        float c = coc_radius(z, p);
        cmin = min(cmin, abs(c)); cmax = max(cmax, abs(c));
        pos += c > 0.0f ? 1 : 0;
        zs += z; cs += c;
    }
    if (!(p.quarterMinCoC > 0.0f) || cmin < p.quarterMinCoC || cmax - cmin > 0.2f * cmax
        || (pos != 0 && pos != 16)) {
        outQuarter.write(half4(0.0h), qid);
        return;
    }
    float3 col;
    if (prefiltered.get_num_mip_levels() > 1u) {
        uint2 pq = min(qid, uint2(prefiltered.get_width(1) - 1u, prefiltered.get_height(1) - 1u));
        half4 pf = prefiltered.read(pq, 1);
        col = pf.a > 0.05h ? float3(pf.rgb) / float(pf.a) : float3(inHDR.read(min(qid * 4u + 2u, last)).rgb);
    } else {
        col = float3(inHDR.read(min(qid * 4u + 2u, last)).rgb);
    }
    float4 g = dof_fast_core(float2(qid * 4u) + 2.0f, col, zs * (1.0f / 16.0f), cs * (1.0f / 16.0f),
                             kDOFFastQuarterTapsPerPx, inHDR, gDepth, tileMax, prefiltered, minCoC,
                             nearField, nearCoC, p);
    outQuarter.write(half4(half3(g.rgb / max(g.a, 1e-6f)), half(cs * (1.0f / 16.0f))), qid);  // SIGNED CoC
}

kernel void illumi_dof(
    texture2d<half,  access::sample> inHDR  [[texture(0)]],
    texture2d<float, access::read>   gDepth [[texture(1)]],
    texture2d<half,  access::write>  outHDR [[texture(2)]],
    texture2d<float, access::read>   tileMax[[texture(3)]],
    texture2d<half,  access::sample> prefiltered [[texture(4)]],
    texture2d<half,  access::read>   minCoC [[texture(5)]],   // sharp-subject mode only
    texture2d<half,  access::read>   halfRes [[texture(6)]],  // fast gather only
    texture2d<half,  access::read>   quarterRes [[texture(7)]], // fast gather + `quarterMinCoC` only
    texture2d<half,  access::sample> nearField [[texture(8)]],  // near-field pyramid only
    texture2d<half,  access::sample> nearCoC [[texture(9)]],    // near-field pyramid only
    constant DOFParams&              p      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    half3 center = inHDR.read(gid).rgb;

    float centerZ    = view_z(gDepth.read(gid).r, gid, p);
    float centerCoC  = coc_radius(centerZ, p);
    float centerAbs  = abs(centerCoC);

    if (p.fastGather > 0.5f && p.subjectAware > 0.5f) {
        dof_fast_full(gid, center, centerZ, centerCoC, inHDR, gDepth, outHDR, tileMax, prefiltered, minCoC,
                      halfRes, quarterRes, nearField, nearCoC, p);
        return;
    }
    if (p.partialOcclusion > 0.5f && p.subjectAware > 0.5f) {
        // The layered composite at the full gather's tap density (6 per px of reach, 24…128), over the
        // exact reach: its own disc, and out to the widest near-field disc that can cover it.
        uint2 t = clamp(gid / p.tileSize, uint2(0), uint2(p.tileW - 1, p.tileH - 1));
        const float2 tm = tileMax.read(t).rg;
        float reach = min(max(centerAbs, tm.x), p.maxRadius);
        reach = min(reach, max(centerAbs + 1.5f, tm.y + 1.0f));
        if (reach < 0.75f) { outHDR.write(half4(center, 1.0h), gid); return; }
        float3 c = dof_layered(float2(gid) + 0.5f, float3(center), centerCoC, reach, tm.y, 6.0f, 24, 128,
                               inHDR, gDepth, prefiltered, minCoC, nearField, nearCoC, p);
        outHDR.write(half4(half3(c), 1.0h), gid);
        return;
    }

    // How far must this pixel look to find a neighbour whose disc could cover it? Its own CoC
    // is only half the answer — a sharply-focused pixel can still be buried under a foreground
    // that is wildly defocused. `tileMax` has already been dilated by the gather's reach, so
    // this single texel is that bound.
    uint2 t = clamp(gid / p.tileSize, uint2(0), uint2(p.tileW - 1, p.tileH - 1));
    float reach = min(max(centerAbs, tileMax.read(t).r), p.maxRadius);
    if (reach < 0.75f) {                       // whole neighbourhood in focus
        outHDR.write(half4(center, 1.0h), gid);
        return;
    }

    // Tap count follows the disc's RADIUS. A count that ignored the reach would leave a
    // large disc reconstructed from a handful of copies of its source; one that followed the
    // full area would be unaffordable. This is the compromise, and the jitter below is what
    // makes it hold up.
    int taps = clamp(int(reach * 6.0f), 24, 128);

    // Each tap's share of the disc, as the side of a square of equal area: the width a tap's
    // colour must be prefiltered over for the taps between them to cover the disc.
    float tapFootprint = p.prefilterScale * reach * sqrt(M_PI_F / float(taps));

    // **Per-pixel rotation of the spiral.** Without this every output pixel samples the SAME
    // relative offsets, so a bright out-of-focus point is rebuilt as a periodic lattice of
    // copies of itself — a chicken-wire stipple across every bokeh ball, which is what an
    // undersampled gather always looks like. Rotating the pattern per pixel (interleaved
    // gradient noise — well distributed and cheap) turns that structured lattice into
    // fine-grained noise, which the disc's own overlap then averages away. It is STATIC, not
    // per-frame: DoF runs after TAA, so a per-frame jitter would shimmer on the live canvas
    // with nothing left to filter it.
    float ign  = fract(52.9829189f * fract(0.06711056f * float(gid.x)
                                         + 0.00583715f * float(gid.y)));
    float rot  = ign * 6.28318531f;

    // Direction from this pixel toward the frame centre, for the cat's-eye clip, scaled
    // by how far off-axis we are (nothing at the centre, full at the corners).
    float2 uv       = (float2(gid) + 0.5f) / float2(p.width, p.height) * 2.0f - 1.0f;
    float2 offAxis  = -uv * p.catsEye;

    float3 sum = float3(center) * 1e-4f;        // keeps a fully-masked pixel from going black
    float  wsum = 1e-4f;
    float2 texel = 1.0f / float2(p.width, p.height);

    // SHARP SUBJECT, part 1 (opt-in, `subjectAware`): THE PIXEL'S OWN DISC. The spiral is a Monte
    // Carlo estimate over a disc of radius `reach`, each tap standing for πR²/taps of it. A pixel
    // that is (nearly) in focus is a source whose whole disc is ≲ 1 px — a target no tap reliably
    // hits once a defocused neighbour has pushed `reach` out to 8+ px. Its only surviving weight
    // was then the background taps' (scaled by centerAbs / tapAbs), so an in-focus silhouette in
    // front of a defocused wall went SEE-THROUGH in a band as wide as the wall's CoC (measured:
    // Digital Clock, a hopping worker's base over the far wall). Counted here explicitly — its
    // true weight 1/(π c²) per px², ≤ 1 once the disc fits inside the pixel — in the taps' units
    // (÷ πR²/taps, the /π cancels the taps' own), and the taps then skip the centre pixel.
    const bool subjectAware = p.subjectAware > 0.5f;
    // Part 3's fast path: if nothing within the gather's whole reach (plus a tap's footprint)
    // is sharper than half the reach, no tap here can straddle a sharper layer — skip its probes.
    bool mixedLayers = false;
    if (subjectAware) {
        float2 uvC = (float2(gid) + 0.5f) * texel;
        // Taps reach `reach` px and read ≤ ±reach px round themselves: 2×2 texels of level m
        // cover ≥ ±2^m px, so m = ⌈log2(2·reach)⌉.
        uint mR = uint(ceil(log2(max(2.0f * reach, 1.0f))));
        mixedLayers = dof_min_coc(minCoC, uvC, mR) < 0.5f * reach;
    }
    if (subjectAware) {
        float cA = max(centerAbs, 0.5f);
        float wc = float(taps) / (reach * reach * max(M_PI_F * cA * cA, 1.0f));
        sum  += float3(center) * wc;
        wsum += wc;
    }

    for (int i = 0; i < taps; ++i) {
        // The radial station is staggered by the same hash, so the rings do not line up
        // across neighbouring pixels either.
        float ft = clamp((float(i) + 0.5f + (ign - 0.5f)) / float(taps), 1e-4f, 1.0f);
        float rr = sqrt(ft) * reach;             // sqrt ⇒ uniform over the disc's AREA
        float a  = float(i) * kGoldenAngle + rot;
        float2 off = float2(cos(a), sin(a)) * rr;
        if (subjectAware && rr < 0.5f) continue;   // the centre pixel: counted above

        // Colour is sampled at the tap's true fractional position. Snapping taps to whole
        // texels quantises them onto a grid, which is the other half of the stipple.
        float2 suv = (float2(gid) + 0.5f + off) * texel;

        // Depth stays a nearest read — interpolating depth across a silhouette would invent
        // a surface that is at neither of the two distances actually there.
        int2 s = clamp(int2(gid) + int2(round(off)), int2(0),
                       int2(int(p.width) - 1, int(p.height) - 1));
        uint2 su = uint2(s);

        float tapZ   = view_z(gDepth.read(su).r, su, p);
        float tapCoC = coc_radius(tapZ, p);
        float tapAbs = max(abs(tapCoC), 1e-3f);

        // SCATTER AS GATHER: does the tap's own confusion disc, with the tap's own
        // aperture shape, reach this pixel? The tap is seen from pupil point u = off / c —
        // the SIGNED CoC: a point at c appears at p + c·u, so in front of the focus plane its
        // disc is the aperture turned half round (odd-bladed irises and the cat's eye show it;
        // `-off / |c|`, as before, drew every foreground disc the wrong way up and shifted the
        // near field's blur by c·|offAxis|/2 — 2–3 px at a 0.2 cat's eye).
        float2 v = off / (tapCoC < 0.0f ? -tapAbs : tapAbs);
        float w  = aperture_mask(v, 1.0f / tapAbs, offAxis, p);

        // Occlusion. A tap in FRONT of us may spread over us freely — that is what
        // foreground defocus looks like. A tap BEHIND us may not paint over something
        // sharper than itself, or the background would bleed through the subject's
        // edges; it is admitted only in proportion to how defocused we already are.
        if (tapZ > centerZ) {
            w *= saturate(centerAbs / tapAbs);
            // SHARP SUBJECT (opt-in): …and only within this pixel's OWN blur of the silhouette. A
            // subject whose disc is c px wide is see-through only within ~c of its edge; a tap
            // behind it further out than that is hidden by the subject, however defocused the tap.
            if (subjectAware) w *= saturate(centerAbs + 1.0f - rr);
        }

        // Energy: a point smeared across a disc contributes per-pixel in inverse
        // proportion to that disc's area.
        w /= (tapAbs * tapAbs);

        // …and PREFILTERED over the tap's footprint (see `illumi_dof_prefilter`), never wider
        // than half the tap's own disc — a tap scatters its colour over that disc, so blurring
        // it by less than the disc changes the result by less than the disc's edge. Full-res
        // LOD 0…1 fades from the frame itself into the pyramid's half-resolution base.
        float lod = log2(max(min(tapFootprint, tapAbs * 0.5f), 1.0f));
        float3 col;
        half4 pf = lod > 0.0f
            ? prefiltered.sample(dofTrilinear, suv, level(max(lod - 1.0f, 0.0f)))
            : half4(0.0h);
        // SHARP SUBJECT, part 3 (opt-in): only a footprint of ONE layer is read prefiltered. A
        // slightly-defocused subject (|CoC| 1.5–3 px, e.g. a toy just behind the focus plane)
        // still carries pyramid weight, and next to this far more defocused tap it is a different
        // layer — ~50× brighter under its own light, it dominated the average and was smeared a
        // mip texel wide. The min-CoC pyramid answers for the whole region the trilinear read
        // reaches: anything there sharper than half this tap's CoC and the tap reads its own texel.
        if (mixedLayers && lod > 0.0f && pf.a > 0.05h
            && dof_min_coc(minCoC, suv, uint(ceil(lod - 1.0f)) + 1u) < 0.5f * tapAbs) {
            pf = half4(0.0h);
        }
        // SHARP SUBJECT, part 2 (opt-in): the pyramid carries no sharp texel (see
        // `illumi_dof_prefilter`) — with the legacy CoC-with-a-floor weight an HDR-bright subject
        // (a red-lit toy at night, ~100× the wall) swamped the floor, and every wall tap beside it
        // carried its colour a whole wall-CoC out: an ~8 px red halo that the night grade's rod
        // (mesopic) shift then printed as a BLACK outline (red has almost no scotopic luminance).
        // The direct read is then the NEAREST texel — the one whose depth weighed this tap; a
        // bilinear read straddles the silhouette and carries the subject in just the same.
        if (pf.a > 0.05h && lod >= 1.0f) {
            col = float3(pf.rgb) / float(pf.a);
        } else {
            col = subjectAware ? float3(inHDR.read(su).rgb) : float3(inHDR.sample(dofLinear, suv).rgb);
            if (pf.a > 0.05h) col = mix(col, float3(pf.rgb) / float(pf.a), lod);
        }

        sum  += col * w;
        wsum += w;
    }

    outHDR.write(half4(half3(sum / max(wsum, 1e-6f)), 1.0h), gid);
}

// ── GROUND TRUTH: the lens by aperture sampling (opt-in diagnostics, `dofReferenceAccumulation`) ──
//
// How an offline renderer makes depth of field, and so what the gather above approximates: the
// frame is the average of pinhole renders taken from points spread over the entrance pupil, each
// with the frustum SHEARED (`projectionShiftNDC`) so the focus plane lands on the same pixels in
// every one (Haeberli & Akeley 1990, the accumulation buffer). A point at depth z then moves by
// ½·cocCoefficient·(z − z_f)/z px per unit of pupil offset — its confusion disc is the pupil's own
// shape, drawn by the samples — and nothing is estimated: occlusion is exact (a defocused edge
// reveals what is behind it), there is no clamp, no tile, no tier, no normalisation.
//
// This kernel is the accumulator. The host renders frame k from pupil point `aperture` (units of
// the iris circumradius, oriented like the gather's footprint: x right, y down) and it adds that
// frame's HDR, weighted per pixel by the barrel's cat's-eye clip at that pixel's field position
// (the far-field convention of `aperture_mask`: the pupil point must lie inside the unit circle
// displaced by −uv·catsEye), then writes the running mean for local adaptation / bloom / tonemap.
// `add` = 0 re-presents the mean without adding (settle frames for temporal post state).
struct DOFReferenceParams {
    float2 aperture;
    float  catsEye;
    float  add;
    float  reset;
    uint   width;
    uint   height;
    uint   _pad;
};

kernel void illumi_dof_reference_accumulate(
    texture2d<half,  access::read>       inHDR  [[texture(0)]],
    texture2d<float, access::read_write> accum  [[texture(1)]],
    texture2d<half,  access::write>      outHDR [[texture(2)]],
    constant DOFReferenceParams&         p      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    const float3 c = float3(inHDR.read(gid).rgb);
    float4 a = p.reset > 0.5f ? float4(0.0f) : accum.read(gid);
    if (p.add > 0.5f) {
        float w = 1.0f;
        if (p.catsEye > 0.0f) {
            float2 uv = (float2(gid) + 0.5f) / float2(p.width, p.height) * 2.0f - 1.0f;
            w = length(p.aperture + uv * p.catsEye) <= 1.0f ? 1.0f : 0.0f;
        }
        a += float4(c * w, w);
    }
    if (p.add > 0.5f || p.reset > 0.5f) accum.write(a, gid);
    const float3 m = a.w > 0.0f ? a.rgb / a.w : c;
    outHDR.write(half4(half3(m), 1.0h), gid);
}
