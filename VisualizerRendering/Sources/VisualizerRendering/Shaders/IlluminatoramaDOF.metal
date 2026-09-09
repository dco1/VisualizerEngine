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
// Three passes. `illumi_dof_tile` reduces the CoC to a max per 16×16 tile and
// `illumi_dof_dilate` spreads that max by the gather's own reach, so the gather knows how
// far it must look to find a NEAR-field neighbour whose disc covers it — without that,
// foreground blur can only ever spread as far as the pixel it lands on already knows about,
// and a defocused foreground silhouette stays crisp. The tile map also buys the early-out
// that pays for the rest: a tile whose whole neighbourhood is in focus copies through
// untouched.

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
    float _pad0;
};

// View-space distance (positive, metres) of a depth-buffer sample.
static inline float view_z(float depth, uint2 gid, constant DOFParams& p) {
    float2 ndc = (float2(gid) + 0.5f) / float2(p.width, p.height) * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float4 vp = p.invProjection * float4(ndc, depth, 1.0f);
    return abs(vp.z / max(abs(vp.w), 1e-6f));
}

// Signed circle-of-confusion RADIUS in pixels. Positive = the point sits in FRONT of
// the focus plane (nearer than focus), negative = behind it. The sign is what lets the
// gather honour occlusion: a near neighbour may spread over anything behind it, a far
// one may not spread over something sharp in front.
static inline float coc_radius(float z, constant DOFParams& p) {
    float zz = max(z, 1e-4f);
    float signed_c = 0.5f * p.cocCoefficient * (p.focusDist - zz) / zz;
    return clamp(signed_c, -p.maxRadius, p.maxRadius);
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
    for (uint y = 0; y < p.tileSize; ++y) {
        uint py = base.y + y;
        if (py >= p.height) break;
        for (uint x = 0; x < p.tileSize; ++x) {
            uint px = base.x + x;
            if (px >= p.width) break;
            uint2 s = uint2(px, py);
            maxAbs = max(maxAbs, abs(coc_radius(view_z(gDepth.read(s).r, s, p), p)));
        }
    }
    outTile.write(float4(maxAbs, 0, 0, 0), tid);
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
    float m = 0.0f;
    for (int dy = -tr; dy <= tr; ++dy) {
        for (int dx = -tr; dx <= tr; ++dx) {
            int2 q = clamp(int2(tid) + int2(dx, dy), int2(0),
                           int2(int(p.tileW) - 1, int(p.tileH) - 1));
            m = max(m, inTile.read(uint2(q)).r);
        }
    }
    outTile.write(float4(m, 0, 0, 0), tid);
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

// ── Pass 3: the gather ───────────────────────────────────────────────────────
constexpr sampler dofLinear(coord::normalized, filter::linear, address::clamp_to_edge);

kernel void illumi_dof(
    texture2d<half,  access::sample> inHDR  [[texture(0)]],
    texture2d<float, access::read>   gDepth [[texture(1)]],
    texture2d<half,  access::write>  outHDR [[texture(2)]],
    texture2d<float, access::read>   tileMax[[texture(3)]],
    constant DOFParams&              p      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    half3 center = inHDR.read(gid).rgb;

    float centerZ    = view_z(gDepth.read(gid).r, gid, p);
    float centerCoC  = coc_radius(centerZ, p);
    float centerAbs  = abs(centerCoC);

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

    for (int i = 0; i < taps; ++i) {
        // The radial station is staggered by the same hash, so the rings do not line up
        // across neighbouring pixels either.
        float ft = clamp((float(i) + 0.5f + (ign - 0.5f)) / float(taps), 1e-4f, 1.0f);
        float rr = sqrt(ft) * reach;             // sqrt ⇒ uniform over the disc's AREA
        float a  = float(i) * kGoldenAngle + rot;
        float2 off = float2(cos(a), sin(a)) * rr;

        // Colour is sampled BILINEARLY at the tap's true fractional position. Snapping taps
        // to whole texels quantises them onto a grid, which is the other half of the stipple.
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
        // aperture shape, reach this pixel? `-off` is the vector from the tap to us.
        float2 v = -off / tapAbs;
        float w  = aperture_mask(v, 1.0f / tapAbs, offAxis, p);

        // Occlusion. A tap in FRONT of us may spread over us freely — that is what
        // foreground defocus looks like. A tap BEHIND us may not paint over something
        // sharper than itself, or the background would bleed through the subject's
        // edges; it is admitted only in proportion to how defocused we already are.
        if (tapZ > centerZ) w *= saturate(centerAbs / tapAbs);

        // Energy: a point smeared across a disc contributes per-pixel in inverse
        // proportion to that disc's area.
        w /= (tapAbs * tapAbs);

        sum  += float3(inHDR.sample(dofLinear, suv).rgb) * w;
        wsum += w;
    }

    outHDR.write(half4(half3(sum / max(wsum, 1e-6f)), 1.0h), gid);
}
