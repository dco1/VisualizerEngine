#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── GrassTiles.metal ──────────────────────────────────────────────────────────
//
// GPU-GENERATED, VIEW-DEPENDENT GRASS (the "Ghost of Tsushima" scheme, GDC 2021).
//
// Every frame, one thread per blade SLOT writes a camera-facing ribbon along a quadratic
// Bézier. Slots are laid out in RINGS around the camera. Ring r is a toroidal window of
// W×W world cells of size c0·2^r, each holding K slots, so
//
//   · the blade COUNT per cell is constant while the cell AREA quadruples per ring
//     → density falls 4× per ring, and the surviving blades WIDEN (`widenPerRing`) so the
//       far field keeps its mass instead of dissolving into sub-pixel hairs;
//   · a world cell always lands on the same slot (toroidal wrap), and every blade's
//     parameters hash from its ABSOLUTE cell coordinates → the lawn is stable in world space,
//     nothing swims when the camera moves, and a ring boundary is a statistical cross-fade
//     (each ring's keep-probability fades in/out over the overlap, the two summing to 1).
//
// Blade colour is the host's palette (root dark → tip light, divided by the instance albedo
// the same way the PBD ribbons are), varied by a low-frequency CLUMP factor so variation is
// spatially coherent, not per-blade static.

struct GrassTileUniforms {
    float4 camera;     // xyz = camera position, w = time (s)
    float4 shape;      // x = ground Y, y = wind amplitude (m), z = wind heading (rad), w = widen per ring
    float4 sizes;      // x = ring-0 cell size (m), y = base half-width (m), z = height min (m), w = height range (m)
    float4 maskRect;   // x = min X, y = min Z, z = 1/size X, w = 1/size Z
    float4 rootC;
    float4 midC;
    float4 tipC;
    float4 misc;       // x = vertex alpha (0 = foliage thin-sheet flag, 1 = opaque), y = far tint strength (0…1),
                       // z = far brightness lift, w = blade height growth per ring
    uint4  ringInfo[8];    // per ring: x = W (cells per side), y = K (blades per cell)
    uint   ringCount;
    uint   segments;       // S: rows = S + 1
    uint   capacity;       // blades the output buffers can hold (kept blades beyond it are dropped)
    uint   pad1;
    float4 planes[4];      // inward-facing view planes through the camera: xyz = unit normal, w = offset
    float4 tune;           // x,y = far-tint distance band (m); z,w = brightness-lift distance band (m)
    float4 tune2;          // x = root occlusion (1 = none); y = far normal-to-up blend; z,w = its distance band (m)
    float4 tune3;          // steep-view thinning: x,y = cos(elevation) band (keep 0 → 1); z,w = horizontal distance band
                           // over which the thinning comes in (m). x ≥ y disables it.
};

static inline uint gt_pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
static inline float gt_h01(uint v) { return float(gt_pcg(v) >> 8) * (1.0f / 16777216.0f); }
static inline uint gt_mix(uint a, uint b) { return gt_pcg(a ^ (b * 0x9E3779B9u + 0x7F4A7C15u)); }

// Value noise on a hashed lattice — the clump / wind-phase field. World-space, so it is stable.
static inline float gt_vnoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    int2 ii = int2(i);
    float a = gt_h01(gt_mix(uint(ii.x),     uint(ii.y)));
    float b = gt_h01(gt_mix(uint(ii.x + 1), uint(ii.y)));
    float c = gt_h01(gt_mix(uint(ii.x),     uint(ii.y + 1)));
    float d = gt_h01(gt_mix(uint(ii.x + 1), uint(ii.y + 1)));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

kernel void grassTileGenerate(
    device Vertex*               out     [[ buffer(0) ]],
    device atomic_uint*          counter [[ buffer(1) ]],
    constant GrassTileUniforms&  u       [[ buffer(3) ]],
    texture2d<float>             mask    [[ texture(0) ]],
    texture2d<float>             ground  [[ texture(1) ]],   // r32Float surface Y over the same rect as the mask
    uint id [[ thread_position_in_grid ]]
) {
    const uint S = u.segments;
    // Which ring owns this slot: rings own consecutive slot ranges of W²·K.
    uint r = 0u, base = 0u;
    for (uint i = 0u; i < u.ringCount; ++i) {
        const uint n = u.ringInfo[i].x * u.ringInfo[i].x * u.ringInfo[i].y;
        if (id < base + n) { r = i; break; }
        base += n;
        if (i + 1u == u.ringCount) return;   // beyond the last ring: unused slot
    }
    const uint W = u.ringInfo[r].x, K = u.ringInfo[r].y;
    const uint rem = id - base;
    const uint k = rem % K;
    const uint cell = rem / K;
    const uint ix = cell % W, iz = cell / W;

    const float c = u.sizes.x * exp2(float(r));
    const float2 cam = u.camera.xz;
    const int camCx = int(floor(cam.x / c)), camCz = int(floor(cam.y / c));
    const int half_ = int(W / 2u);
    const int loX = camCx - half_, loZ = camCz - half_;
    const int iW = int(W);
    const int cx = loX + (((int(ix) - loX) % iW) + iW) % iW;
    const int cz = loZ + (((int(iz) - loZ) % iW) + iW) % iW;

    const uint key = gt_mix(gt_mix(uint(cx), uint(cz)), r * 977u + k * 131u + 7u);
    const float2 xz = (float2(float(cx), float(cz)) + float2(gt_h01(key), gt_h01(key ^ 0xA511E9B3u))) * c;
    // Host coverage mask (house slab, paving, decks, soft yard edge…) and the ground surface Y
    // (flat plane, or the terrain drape) — one UV over the same world rect.
    const float2 muv = (xz - u.maskRect.xy) * u.maskRect.zw;
    // Outside the host's yard rect the lawn simply continues (the ground plane runs to the horizon),
    // so coverage there is 1 on the flat ground Y — grass reaches as far as the rings do.
    float m = 1.0f;
    float gy = u.shape.x;
    if (all(muv >= 0.0f) && all(muv <= 1.0f)) {
        constexpr sampler ms(filter::linear, address::clamp_to_edge);
        m  = mask.sample(ms, muv).r;
        gy = ground.sample(ms, muv).r;
    }
    const float3 root = float3(xz.x, gy, xz.y);
    const float3 toCam = u.camera.xyz - root;
    const float dist = length(toCam);

    // Ring bands. Rout(r) < the window's half-extent, so the whole disc is inside the window.
    const float rOut  = 0.46f * float(W) * c;
    const float rOutPrev = (r == 0u) ? 0.0f : 0.46f * float(u.ringInfo[r - 1u].x) * (c * 0.5f);
    const float rInThis  = rOutPrev * 0.40f;
    const float rInNext  = rOut * 0.40f;
    // Ring bands are HORIZONTAL radii (the ring windows are laid out on the ground). Using the 3-D
    // distance made ring 0 — the densest, within ~1.2 m — vanish entirely for an eye 1.6 m up, because
    // no ground point is closer than the eye height.
    const float distH = length(xz - cam);
    float wIn  = (r == 0u) ? 1.0f : smoothstep(rInThis, rOutPrev, distH);
    float wOut = (r + 1u == u.ringCount) ? 1.0f - smoothstep(0.82f * rOut, rOut, distH)
                                         : 1.0f - smoothstep(rInNext, rOut, distH);

    // Steep-view thinning (DH-1083): seen from above, a distant blade is a dark stroke on the turf, not
    // grass — the ground tile carries a lawn there (GPUOpen / Tsushima distance hand-off). Seen at a
    // grazing angle the same blades are the lawn's fuzz, so they stay. cosE = horizontal / 3-D distance.
    float steep = 1.0f;
    if (u.tune3.x < u.tune3.y) {
        const float cosE = length(xz - cam) / max(length(float3(xz.x, gy, xz.y) - u.camera.xyz), 1e-4f);
        const float thin = 1.0f - smoothstep(u.tune3.x, u.tune3.y, cosE);
        steep = 1.0f - thin * smoothstep(u.tune3.z, u.tune3.w, length(xz - cam));
    }
    const bool keep = gt_h01(key ^ 0x5BD1E995u) < wIn * wOut * m * steep;

    const uint rows = S + 1u;

    if (!keep) return;   // a culled slot costs nothing: it never claims output space

    // ── Blade parameters (clump-coherent) ──
    const float clump  = gt_vnoise(xz * 2.5f);
    // Far blades grow in height as well as width so a widened blade keeps a blade's proportions (long and
    // narrow) instead of fattening into a cone: aspect stays ~1:6 and above at every ring.
    const float hgt    = (u.sizes.z + u.sizes.w * gt_h01(key ^ 0x68E31DA4u))
                       * (0.75f + 0.5f * clump) * (1.0f + u.misc.w * float(r));
    const float hw     = u.sizes.y * pow(u.shape.w, float(r)) * (0.65f + 0.60f * gt_h01(key ^ 0xB5297A4Du));
    const float az     = gt_h01(key ^ 0x1B873593u) * 6.2831853f;
    const float2 dir   = float2(cos(az), sin(az));
    const float2 wdir  = float2(cos(u.shape.z), sin(u.shape.z));
    const float lean   = hgt * (0.15f + 0.55f * gt_h01(key ^ 0x2545F491u));
    const float ph     = dot(xz, wdir) * 0.35f + u.camera.w * 1.6f + 4.0f * gt_vnoise(xz * 0.1f);
    // Gust (a travelling wave across the yard) + a small per-blade flutter on its own phase, so the
    // sward ripples as a field and each blade still trembles a little on its own. Taller blades
    // bend further (the amplitude is per 5 cm of blade).
    const float flutter = 0.30f * sin(u.camera.w * 5.1f + 6.2831853f * gt_h01(key ^ 0x7F4A7C15u));
    const float sway   = u.shape.y * (0.6f + 0.4f * sin(ph) + flutter) * (hgt / 0.05f);
    const float tint   = (0.85f + 0.30f * gt_h01(key ^ 0x9E3779B1u)) * (0.90f + 0.20f * clump);

    const float3 up = float3(0, 1, 0);
    const float3 P0 = root;
    // A grass blade rises nearly straight, then ARCS over and its tip droops — a clear curve, not a lean.
    const float droop = 0.25f + 0.45f * gt_h01(key ^ 0x6A09E667u);
    const float3 P1 = root + float3(dir.x * lean * 0.25f, 0.0f, dir.y * lean * 0.25f) + up * (0.62f * hgt);
    const float3 P2 = root + float3(dir.x * lean * (1.0f + droop), 0.0f, dir.y * lean * (1.0f + droop)) + up * ((0.90f - 0.12f * droop) * hgt);
    const float3 viewDir = toCam / max(dist, 1e-5f);

    // View cull: a blade outside the (slightly widened) view cone is never written, so it is neither
    // transformed nor shaded by any pass. Blades right under the camera always stay (shadows, turning).
    {
        const float3 ctr = root + up * (0.5f * hgt);
        const float rad  = hgt * 1.4f + sway + 0.05f;
        bool vis = distH < 0.7f;
        if (!vis) {
            vis = true;
            for (uint pi = 0u; pi < 4u; ++pi)
                vis = vis && (dot(u.planes[pi].xyz, ctr) + u.planes[pi].w >= -rad);
        }
        if (!vis) return;
    }
    const uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot >= u.capacity) return;
    const uint vBase = slot * rows * 2u;

    for (uint row = 0; row < rows; ++row) {
        // Rows are packed toward the TIP: with 3 rows the body is one long near-parallel segment and
        // the point is its own short one, so the silhouette is a blade (ribbon + point), not a cone.
        const float uRow = float(row) / float(max(S, 1u));
        const float t = 1.0f - pow(1.0f - uRow, 1.5f);
        const float omt = 1.0f - t;
        float3 p = omt * omt * P0 + 2.0f * omt * t * P1 + t * t * P2;
        p += float3(wdir.x, 0.0f, wdir.y) * (sway * t * t);
        float3 tan_ = 2.0f * omt * (P1 - P0) + 2.0f * t * (P2 - P1)
                    + float3(wdir.x, 0.0f, wdir.y) * (2.0f * sway * t);
        tan_ = normalize(tan_);

        float3 perp = cross(tan_, viewDir);
        float pl = length(perp);
        perp = (pl > 1e-5f) ? perp / pl : float3(1, 0, 0);

        float3 nrm = cross(perp, tan_);
        if (dot(nrm, viewDir) < 0.0f) nrm = -nrm;
        nrm = normalize(nrm);

        // Parallel-sided for the first ~60 % (a hair narrower at the very base), then a point.
        const float body = mix(0.85f, 1.0f, smoothstep(0.0f, 0.25f, t));
        const float tipT = smoothstep(0.55f, 1.0f, t);
        const float w = hw * body * (1.0f - 0.94f * tipT * tipT);
        const float3 pL = p - perp * w, pR = p + perp * w;
        // Fan the edge normals outward so a flat ribbon shades like a rounded blade (the
        // Tsushima/Horizon "rounded normal" trick) instead of a cardboard cut-out.
        const float fan = 0.35f;
        // Shade with the GROUND's normal, mostly (Horizon/Tsushima: grass is lit as a volume, not per
        // blade). A blade's own normal faces the camera, so under a low sun every blade catches
        // far more light than the horizontal ground it grows from and the field reads as glittering
        // confetti. Blending toward up makes blade and ground light alike; the blend grows with
        // the ring so the far field is almost purely ground-lit.
        const float distT = smoothstep(u.tune2.z, u.tune2.w, dist);   // continuous in DISTANCE, not ring index (no visible band)
        const float toUp = mix(0.20f, u.tune2.y, distT);
        const float3 nL = normalize(mix(normalize(nrm - perp * fan), up, toUp));
        const float3 nR = normalize(mix(normalize(nrm + perp * fan), up, toUp));

        float3 col = (t < 0.5f) ? mix(u.rootC.xyz, u.midC.xyz, t * 2.0f)
                                : mix(u.midC.xyz,  u.tipC.xyz, (t - 0.5f) * 2.0f);
        col *= tint;
        // Root occlusion: the base of a dense sward stands in its neighbours' shadow.
        col *= mix(u.tune2.x, 1.0f, smoothstep(0.0f, 0.45f, t));
        // Far rings converge on the palette's MID colour (≈ the ground tile's mean): in the distance
        // the blades are texture breakup on a ground of the same colour, not individually lit
        // flakes — Tsushima tints distant terrain toward the grass colour for the same reason.
        const float far = smoothstep(u.tune.x, u.tune.y, dist) * u.misc.y;
        col = mix(col, u.midC.xyz * tint, far);
        // A dense sward occludes itself: with ground-blended shading each blade still reads darker than
        // the open ground it stands on (AO from its neighbours). Lift the far blades to meet the ground
        // tile's brightness so the field reads as one surface instead of dark flecks on a lighter plane.
        col *= mix(1.0f, u.misc.z, smoothstep(u.tune.z, u.tune.w, dist));
        // alpha 0 = the engine's foliage thin-sheet flag (same as the PBD ribbons).
        const float4 c4 = float4(col, u.misc.x);
        Vertex vl, vr;
        vl.position = pL; vl._padPos = 0.0f; vl.normal = nL; vl._padNrm = 0.0f;
        vl.uv = float2(0.0f); vl._padUv = float2(0.0f);
        vr = vl; vr.position = pR; vr.normal = nR;
        // Arbitrary stable tangent, as the engine's repack kernel writes (grass samples no normal map).
        float3 axL = abs(nL.y) < 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 axR = abs(nR.y) < 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
        vl.tangent = float4(normalize(cross(axL, nL)), 1.0f);
        vr.tangent = float4(normalize(cross(axR, nR)), 1.0f);
        vl.color = c4; vr.color = c4;
        out[vBase + row * 2u]      = vl;
        out[vBase + row * 2u + 1u] = vr;
    }
}

kernel void grassTileReset(device atomic_uint* counter [[ buffer(0) ]]) {
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

// Writes the indirect draw: indexCount = kept blades × indices per blade (clamped to capacity).
kernel void grassTileFinalize(device atomic_uint* counter [[ buffer(0) ]],
                              device uint*        args    [[ buffer(1) ]],
                              constant uint2&     info    [[ buffer(2) ]])   // x = indices per blade, y = capacity
{
    const uint kept = min(atomic_load_explicit(counter, memory_order_relaxed), info.y);
    args[0] = kept * info.x;   // indexCount
    args[1] = 1u;              // instanceCount
    args[2] = 0u;              // indexStart
    args[3] = 0u;              // baseVertex
    args[4] = 0u;              // baseInstance
}
