#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── FurStrands.metal ──────────────────────────────────────────────────────────
//
// GPU FUR on a BREATHING SKIN. Three kernels, one command buffer per frame:
//
//   furGuideStep    wind dynamics on a coarse GUIDE-HAIR grid over the skin patch: each guide is a
//                   damped spring (tip displacement d, velocity v) pushed by a travelling gust field
//                   and coupled to its neighbours, so a gust is a visible wave front that crosses the
//                   coat. Ping-pong buffers (no read/write race on the neighbour coupling).
//   furSkinGenerate writes the skin grid (breathing-displaced) the strands are rooted on.
//   furStrandGenerate  one thread per strand: root on the SAME analytic skin function (so a breath
//                   moves every root), clump into locks, comb along the flow field, interpolate the
//                   four surrounding guides for its wind bend, and write a camera-facing ribbon whose
//                   NORMAL slot is the strand TANGENT with colour alpha 0.995 — the flag that makes the
//                   deferred lighting pass shade it with the Chiang hair BSDF (IlluminatoramaHairBSDF.h).
//
// Strands are a pure function of (id, uniforms): nothing is stored per strand.

#define FUR_SEGMENTS 5u   // ribbon segments per strand — keep in lockstep with `FurStrandField.segments`

struct FurUniforms {
    float4 patch;      // x = Lx (m, along the spine), y = Lz (m, back→belly), z = arch height (m), w = time
    float4 breath;     // x = breath phase (cycles, continuous), y = depth (m), z = ribcage centre u, w = ribcage radius (uv)
    float4 coat;       // x = undercoat length (m), y = guard length (m), z = guard fraction, w = clump strength 0…1
    float4 look;       // x = strand width (m), y = lock size (m), z = comb angle (rad), w = flow waviness
    float4 wind;       // x = wind heading (rad), y = strength (m/s²-ish), z = gust scale (1/m), w = gust speed (m/s)
    float4 baseColor;  // rgb = tint multiplier on the banded agouti ramp, a = colour jitter
    float4 camera;     // xyz = camera position, w = pixel footprint at 1 m (m)
    uint4  counts;     // x = strands, y = grid cols (strand lattice), z = guide cols, w = guide rows
    uint4  skinGrid;   // x = skin cols, y = skin rows
    float4 sim;        // x = dt, y = spring k, z = damping, w = neighbour coupling
    float4 colRoot;    // rgb = fibre colour at the root (pale agouti band)
    float4 colMid;     // rgb = saturated mid-shaft colour (also the dark undercoat seen between strands)
    float4 colTip;     // rgb = undercoat tip colour
    float4 colGuardTip;// rgb = guard-hair tip colour
    float4 pattern;    // x = tabby stripe strength, y = stripe frequency (1/m), z = belly lightening, w = unused
    float4 touch;      // xyz = contact point (world), w = contact radius (m)
    float4 touchState; // x = pressure 0…1 (eased by the host), yz = contact velocity (m/s, world xz), w = unused
};

static inline uint fur_pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
static inline float fur_h01(uint v) { return float(fur_pcg(v) >> 8) * (1.0f / 16777216.0f); }
static inline uint fur_mix(uint a, uint b) { return fur_pcg(a ^ (b * 0x9E3779B9u + 0x7F4A7C15u)); }

static inline float fur_vnoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    int2 ii = int2(i);
    float a = fur_h01(fur_mix(uint(ii.x), uint(ii.y)));
    float b = fur_h01(fur_mix(uint(ii.x + 1), uint(ii.y)));
    float c = fur_h01(fur_mix(uint(ii.x), uint(ii.y + 1)));
    float d = fur_h01(fur_mix(uint(ii.x + 1), uint(ii.y + 1)));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// ── The skin: a flank patch, arched across (back→belly), with the ribcage breathing ──────────────
// uv ∈ [0,1]². Same function the skin mesh and every strand root evaluate, so they agree exactly.
static inline float3 fur_skin(constant FurUniforms& u, float2 uv) {
    const float x = (uv.x - 0.5f) * u.patch.x;
    const float z = (uv.y - 0.5f) * u.patch.y;
    const float c = 2.0f * (uv.y - 0.5f);
    float y = u.patch.z * (1.0f - c * c);
    // Breath: smooth inhale/exhale (raised cosine of the continuous phase → C¹ through every turn).
    const float b = 0.5f - 0.5f * cos(6.2831853f * u.breath.x);
    const float2 d = (uv - float2(u.breath.z, 0.5f)) / max(u.breath.w, 1e-3f);
    const float rib = exp(-dot(d, d));                       // ribcage bump
    const float flank = 0.35f * exp(-pow((uv.x - u.breath.z) / (1.8f * u.breath.w), 2.0f)); // slight whole-flank rise
    y += u.breath.y * b * (rib + flank);
    // Touch: a fingertip pressed into the coat dents the skin under it (smooth Gaussian, depth ~⅓ of the contact radius).
    {
        const float r = max(u.touch.w, 1e-3f);
        const float2 td = (float2(x, z) - u.touch.xz) / r;
        y -= u.touchState.x * 0.22f * r * exp(-1.6f * dot(td, td));
    }
    // Ribs also push the surface slightly outward sideways (toward the belly) as they expand.
    const float zz = z + 0.25f * u.breath.y * b * rib * c;
    return float3(x, y, zz);
}

static inline void fur_frame(constant FurUniforms& u, float2 uv, thread float3& P,
                             thread float3& dU, thread float3& dV, thread float3& N) {
    const float e = 0.004f;
    P = fur_skin(u, uv);
    dU = normalize(fur_skin(u, uv + float2(e, 0)) - fur_skin(u, uv - float2(e, 0)));
    dV = normalize(fur_skin(u, uv + float2(0, e)) - fur_skin(u, uv - float2(0, e)));
    N = normalize(cross(dV, dU));
    if (N.y < 0.0f) N = -N;
}

static inline float2 fur_wind(constant FurUniforms& u, float2 xz, float t) {
    const float2 w = float2(cos(u.wind.x), sin(u.wind.x));
    const float k = u.wind.z;
    // A gust is a band crossing the coat along the wind heading: a travelling front, modulated by
    // noise across the front so it is ragged, plus a faint steady breeze.
    const float front = dot(xz, w) * k - t * u.wind.w * k;
    const float across = dot(xz, float2(-w.y, w.x)) * k * 0.7f;
    const float band = pow(0.5f + 0.5f * sin(front * 6.2831853f * 0.5f + 1.7f * fur_vnoise(float2(across, front * 0.3f))), 2.0f);
    const float rag = 0.55f + 0.9f * fur_vnoise(float2(across * 1.3f, front * 0.4f + 9.0f));
    return w * u.wind.y * (0.12f + band * rag);
}

kernel void furGuideStep(
    device const float4* src [[ buffer(0) ]],
    device float4*       dst [[ buffer(1) ]],
    constant FurUniforms& u  [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const int C = int(u.counts.z), R = int(u.counts.w);
    if (int(gid.x) >= C || int(gid.y) >= R) return;
    const int i = int(gid.y) * C + int(gid.x);
    const float4 s = src[i];
    float2 d = s.xy, v = s.zw;
    float2 lap = float2(0);
    const int2 offs[4] = { int2(1,0), int2(-1,0), int2(0,1), int2(0,-1) };
    for (int k = 0; k < 4; ++k) {
        const int2 n = clamp(int2(gid) + offs[k], int2(0), int2(C - 1, R - 1));
        lap += src[n.y * C + n.x].xy - d;
    }
    const float2 uv = (float2(gid) + 0.5f) / float2(C, R);
    const float2 xz = (uv - 0.5f) * float2(u.patch.x, u.patch.y);
    const float dt = u.sim.x;
    float2 acc = fur_wind(u, xz, u.patch.w) - u.sim.y * d - u.sim.z * v + u.sim.w * lap;
    {   // Touch: the contact parts the guides radially and a stroke drags them along its velocity.
        const float r = max(u.touch.w, 1e-3f);
        const float2 off = xz - u.touch.xz;
        const float inf = u.touchState.x * exp(-1.2f * dot(off, off) / (r * r));
        const float2 away = off / max(length(off), 1e-4f);
        acc += away * (260.0f * inf) + u.touchState.yz * (900.0f * inf);
    }
    v += acc * dt;
    d += v * dt;
    const float m = length(d);
    if (m > 0.03f) d *= 0.03f / m;      // a fibre cannot bend its tip further than the coat is long
    dst[i] = float4(d, v);
}

kernel void furSkinGenerate(
    device Vertex* out [[ buffer(0) ]],
    constant FurUniforms& u [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const uint C = u.skinGrid.x, R = u.skinGrid.y;
    if (gid.x >= C || gid.y >= R) return;
    const float2 uv = float2(float(gid.x) / float(C - 1u), float(gid.y) / float(R - 1u));
    float3 P, dU, dV, N;
    fur_frame(u, uv, P, dU, dV, N);
    Vertex v;
    v.position = P - N * 0.0006f;            // just under the strand roots (no z-fight)
    v._padPos = 0.0f;
    v.normal = N; v._padNrm = 0.0f;
    v.uv = float2(0.0f); v.cardUV = float2(0.0f);
    v.tangent = float4(dU, 1.0f);
    // Dense undercoat seen between strands: dark, saturated, deep in its own shadow.
    v.color = float4(u.baseColor.rgb * u.colMid.rgb * 0.22f, 1.0f);
    out[gid.y * C + gid.x] = v;
}

kernel void furStrandGenerate(
    device Vertex*              out    [[ buffer(0) ]],
    device const float4*        guides [[ buffer(1) ]],
    constant FurUniforms&       u      [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    const uint Nn = u.counts.x;
    if (id >= Nn) return;
    const uint cols = u.counts.y;
    const uint ix = id % cols, iy = id / cols;
    const uint rows = (Nn + cols - 1u) / cols;
    const uint key = fur_mix(id, 0x51ED270Bu);

    // Root on a jittered lattice over the patch's uv domain.
    float2 uv = (float2(float(ix), float(iy)) + float2(fur_h01(key), fur_h01(key ^ 0xA511E9B3u))) / float2(float(cols), float(rows));
    uv = clamp(uv, 0.002f, 0.998f);
    const bool guard = fur_h01(key ^ 0x5BD1E995u) < u.coat.z;

    // ── Locks: strands gather toward the nearest of a jittered set of lock centres (clumping). Nearest-of-3×3
    //    (a Worley/Voronoi cell) rather than the axis-aligned cell the strand sits in, so locks are organic
    //    blobs, not a visible square grid. ──
    const float2 pm = uv * float2(u.patch.x, u.patch.y);
    const float lockS = max(u.look.y, 1e-3f);
    const float2 cf = pm / lockS;
    const int2 ci = int2(floor(cf));
    float bestD = 1e9f; float2 bestC = float2(0); uint lkey = 0u;
    for (int oy = -1; oy <= 1; ++oy)
        for (int ox = -1; ox <= 1; ++ox) {
            const int2 c = ci + int2(ox, oy);
            const uint k = fur_mix(uint(c.x), uint(c.y) ^ 0x1234567u);
            const float2 ctr = float2(c) + float2(0.15f + 0.7f * fur_h01(k), 0.15f + 0.7f * fur_h01(k ^ 0x68E31DA4u));
            const float2 dd = ctr - cf;
            const float d2 = dot(dd, dd);
            if (d2 < bestD) { bestD = d2; bestC = ctr; lkey = k; }
        }
    const float2 lockCtr = bestC * lockS;
    const float2 toLock = lockCtr - pm;
    const float clump = u.coat.w;
    uv += (toLock * clump * 0.35f) / float2(u.patch.x, u.patch.y);
    uv = clamp(uv, 0.002f, 0.998f);

    float3 P, dU, dV, N;
    fur_frame(u, uv, P, dU, dV, N);

    // ── Flow: comb direction in the tangent plane, wavy + per-lock scatter ──
    const float ang = u.look.z + u.look.w * (fur_vnoise(pm * 9.0f) - 0.5f) * 2.0f
                    + 0.25f * (fur_h01(lkey ^ 0xB5297A4Du) - 0.5f) + 0.10f * (fur_h01(key ^ 0x2545F491u) - 0.5f);
    const float3 F = normalize(dU * cos(ang) + dV * sin(ang));

    // ── Length: layer, per-lock and per-strand variation ──
    const float lockLen = 0.88f + 0.24f * fur_h01(lkey ^ 0x9E3779B1u);
    const float L = (guard ? u.coat.y : u.coat.x) * lockLen * (0.85f + 0.3f * fur_h01(key ^ 0x7F4A7C15u));

    // ── Wind bend from the four surrounding guides ──
    const int GC = int(u.counts.z), GR = int(u.counts.w);
    const float2 gp = uv * float2(GC, GR) - 0.5f;
    const int2 g0 = int2(floor(gp));
    const float2 gf = gp - float2(g0);
    float2 wd = float2(0);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            const int2 g = clamp(g0 + int2(dx, dy), int2(0), int2(GC - 1, GR - 1));
            const float wgt = (dx ? gf.x : 1.0f - gf.x) * (dy ? gf.y : 1.0f - gf.y);
            wd += wgt * guides[g.y * GC + g.x].xy;
        }
    wd *= (guard ? 1.0f : 0.55f) * (L / max(u.coat.y, 1e-4f));

    // ── Strand curve (quadratic Bézier): rises along the skin normal then lays over along the flow ──
    float lay = guard ? 0.74f : 0.90f;            // the soft undercoat lies flatter, the guard hairs stand
    // Touch: strands near the contact lean away from it and lie flat; a stroke combs them along its velocity.
    const float tr = max(u.touch.w, 1e-3f);
    const float2 toff = P.xz - u.touch.xz;
    const float tinf = u.touchState.x * exp(-1.2f * dot(toff, toff) / (tr * tr));
    const float3 tAway = normalize(float3(toff.x, 0.0f, toff.y) + float3(1e-5f, 0.0f, 0.0f));
    const float3 tPush = (tAway * 0.9f + float3(u.touchState.y, 0.0f, u.touchState.z) * 2.5f) * (L * tinf);
    lay = mix(lay, 0.97f, saturate(tinf * 1.5f));
    const float3 lockPull = float3(toLock.x, 0, toLock.y);
    const float3 P0 = P;
    const float3 P1 = P + N * (0.50f * L);
    const float3 P2 = P + N * (L * (1.0f - lay) * 0.9f) + F * (L * lay) + lockPull * clump * 0.5f + tPush;
    const float3 camPos = u.camera.xyz;
    const float3 toCam = camPos - P;
    const float dist = length(toCam);
    const float3 viewDir = toCam / max(dist, 1e-5f);

    // Rendered width: never thinner than ~0.7 px so TAA has something to resolve.
    const float trueW = u.look.x * (guard ? 1.35f : 1.0f);
    const float footprint = u.camera.w * dist;
    const float wRender = max(trueW, 0.7f * footprint);
    const float cover = trueW / wRender;
    // Sub-pixel strands are thinned stochastically in proportion to the area they should cover: a dropped
    // strand is collapsed to a zero-area sliver at its root (the slot stays defined; no triangle rasterises).
    const bool drop = fur_h01(key ^ 0x1B873593u) > min(1.0f, cover * 1.4f);

    // Per-strand colour: agouti banding (pale root → saturated mid → rust tip), lock-coherent variation.
    const float lockV = (fur_h01(lkey ^ 0x6A09E667u) - 0.5f);
    const float jit = u.baseColor.a;
    const float3 sv = float3(1.0f + jit * (fur_h01(key ^ 0x3C6EF372u) - 0.5f) * 0.6f + 0.25f * lockV,
                             1.0f + jit * (fur_h01(key ^ 0xA54FF53Au) - 0.5f) * 1.1f + 0.35f * lockV,
                             1.0f + jit * (fur_h01(key ^ 0x510E527Fu) - 0.5f) * 1.2f);
    float3 rootC = u.colRoot.rgb;
    float3 midC  = u.colMid.rgb;
    float3 tipC  = guard ? u.colGuardTip.rgb : u.colTip.rgb;
    // Pattern. Tabby: broad dark bands across the back that break up with noise and fade toward the belly
    // (uv.y: 0 = spine, 1 = belly). Belly: the coat lightens toward cream on the underside.
    {
        const float warp = 3.0f * fur_vnoise(pm * 7.0f);
        const float sb = sin(uv.x * u.patch.x * u.pattern.y * 6.2831853f + warp);
        const float band = smoothstep(0.15f, 0.55f, sb) * u.pattern.x * (1.0f - smoothstep(0.55f, 0.95f, uv.y));
        const float3 dark = float3(0.45f, 0.40f, 0.35f);
        rootC *= mix(float3(1.0f), dark, band * 0.6f);
        midC  *= mix(float3(1.0f), dark, band);
        tipC  *= mix(float3(1.0f), dark, band);
        const float belly = u.pattern.z * smoothstep(0.62f, 0.98f, uv.y);
        const float3 cream = float3(0.98f, 0.80f, 0.54f);
        rootC = mix(rootC, cream, belly); midC = mix(midC, cream * 0.92f, belly * 0.8f); tipC = mix(tipC, cream * 0.8f, belly * 0.6f);
    }

    const uint SEG = FUR_SEGMENTS;
    const uint vBase = id * (SEG + 1u) * 2u;
    for (uint row = 0; row <= SEG; ++row) {
        const float t = float(row) / float(SEG);
        const float omt = 1.0f - t;
        float3 p = omt * omt * P0 + 2.0f * omt * t * P1 + t * t * P2;
        float3 tan_ = 2.0f * omt * (P1 - P0) + 2.0f * t * (P2 - P1);
        // Wind: tip displacement grows with t², and length is conserved by sinking the tip.
        const float3 wv = float3(wd.x, 0, wd.y);
        if (drop) { p = P0; }
        if (!drop) { p += wv * (t * t); p.y -= dot(wd, wd) * (t * t) / max(2.0f * L, 1e-4f); }
        tan_ += wv * (2.0f * t);
        tan_ = normalize(tan_);
        // Natural wave: each fibre meanders sideways (undercoat crimps more than a guard hair), phase per strand.
        {
            const float3 side = normalize(cross(N, tan_) + float3(1e-5f, 0.0f, 0.0f));
            const float wavAmp = L * (guard ? 0.07f : 0.16f) * t;
            const float ph = 6.2831853f * fur_h01(key ^ 0x2C1B3C6Du);
            if (!drop) p += side * (wavAmp * sin(t * (guard ? 4.0f : 7.0f) + ph));
        }

        float3 perp = cross(tan_, viewDir);
        const float pl = length(perp);
        perp = (pl > 1e-5f) ? perp / pl : float3(1, 0, 0);
        // Tapered: full width over the lower body, fine point at the tip.
        const float w = drop ? 0.0f : 0.5f * wRender * (1.0f - 0.85f * t * t);
        float3 col = (t < 0.5f) ? mix(rootC, midC, t * 2.0f) : mix(midC, tipC, (t - 0.5f) * 2.0f);
        col *= u.baseColor.rgb * sv;
        // Root-to-tip AO ramp: the base of a dense coat is in its neighbours' shadow.
        col *= mix(0.55f, 1.0f, pow(t, 0.6f));
        // An over-widened sub-pixel strand must not out-shine the area it stands for.
        col *= mix(0.45f, 1.0f, saturate(cover * 1.5f));

        Vertex vl, vr;
        vl.position = p - perp * w; vl._padPos = 0.0f;
        vl.normal = tan_; vl._padNrm = 0.0f;       // TANGENT in the normal slot (hair tag below)
        vl.uv = float2(0.0f); vl.cardUV = float2(0.0f);
        vl.tangent = float4(0.0f);
        vl.color = float4(col, 0.995f);
        vr = vl; vr.position = p + perp * w;
        out[vBase + row * 2u]      = vl;
        out[vBase + row * 2u + 1u] = vr;
    }
}
