#include <metal_stdlib>
using namespace metal;

// ── VesselLiquid.metal ────────────────────────────────────────────────────────
//
// The liquid inside a vessel of revolution with a moving free surface (VesselLiquidMesh.swift).
// The vessel's INNER profile r(y) is a table; the free surface is the height field
//
//     η(x, z) = level + tilt·(x, z) + (mode2·(x, z)) · (ρ² − ½)      ρ = r / R_contact
//
// (the first antisymmetric sloshing mode as a tilt, the second as a cubic wobble, both from the
// host's slosh dynamics). Each azimuth finds where the surface meets the wall — the fixed point
// y = η(r(y)·dir) — and every vertex is closed-form from that: the surface as a polar grid out to
// the contact ring, the wall as the vessel's own inner surface from the floor up to the contact
// ring, the floor as a disc. One thread per vertex; packed position + normal, repacked by
// Illuminatorama (and diff-captured for motion vectors) like any GPU mesh.

// Mirror of `VesselLiquidMesh.Params` (float4 ALIGNMENT RULE: 3 × float4 + 4 × uint).
struct VesselLiquidParams {
    float4 surface;     // x = level (y at the axis), y/z = tilt (∂η/∂x, ∂η/∂z), w = floor y
    float4 mode2;       // xy = second-mode amplitude along x/z, z = profile y0, w = profile dy
    float4 counts;      // x = profile samples, y = segments, z = surface rings, w = wall rings
    uint   vertexCount;
    uint   pad0, pad1, pad2;
};

// r(y) and dr/dy from the profile table (uniform in y from y0 by dy).
static inline float2 vlProfile(constant float* rTable, float y, float y0, float dy, uint n) {
    float f = clamp((y - y0) / dy, 0.0f, float(n - 1) - 1e-3f);
    uint i = uint(f);
    float t = f - float(i);
    float r0 = rTable[i], r1 = rTable[min(i + 1u, n - 1u)];
    return float2(mix(r0, r1, t), (r1 - r0) / dy);
}

static inline float vlEta(float2 xz, float rc, constant VesselLiquidParams& p) {
    float rho2 = dot(xz, xz) / max(rc * rc, 1e-6f);
    return p.surface.x + dot(p.surface.yz, xz) + dot(p.mode2.xy, xz) * (rho2 - 0.5f);
}

static inline float3 vlEtaGrad(float2 xz, float rc, constant VesselLiquidParams& p) {
    float inv = 1.0f / max(rc * rc, 1e-6f);
    float rho2 = dot(xz, xz) * inv;
    float m = dot(p.mode2.xy, xz);
    float2 g = p.surface.yz + p.mode2.xy * (rho2 - 0.5f) + m * 2.0f * xz * inv;
    return normalize(float3(-g.x, 1.0f, -g.y));
}

kernel void vesselLiquidMesh(constant float*              rTable [[buffer(0)]],
                             device packed_float3*        outPos [[buffer(1)]],
                             device packed_float3*        outNrm [[buffer(2)]],
                             constant VesselLiquidParams& p      [[buffer(3)]],
                             uint vid [[thread_position_in_grid]]) {
    if (vid >= p.vertexCount) return;
    uint n = uint(p.counts.x), seg = uint(p.counts.y), sr = uint(p.counts.z), wr = uint(p.counts.w);
    float y0 = p.mode2.z, dy = p.mode2.w, yFloor = p.surface.w;

    // Layout: [surface centre][surface rings sr × seg][wall rings wr × seg][floor centre][floor ring seg]
    uint surfCount = 1u + sr * seg;
    uint wallCount = wr * seg;
    uint j = 0u;
    int region;          // 0 surface centre, 1 surface ring, 2 wall, 3 floor centre, 4 floor ring
    uint ring = 0u;
    if (vid == 0u) { region = 0; }
    else if (vid < surfCount) { uint k = vid - 1u; ring = k / seg; j = k - ring * seg; region = 1; }
    else if (vid < surfCount + wallCount) { uint k = vid - surfCount; ring = k / seg; j = k - ring * seg; region = 2; }
    else if (vid == surfCount + wallCount) { region = 3; }
    else { j = vid - surfCount - wallCount - 1u; region = 4; }

    if (region == 3 || region == 4) {
        float rF = vlProfile(rTable, yFloor, y0, dy, n).x;
        float a = float(j) / float(seg) * 2.0f * M_PI_F;
        float3 pos = region == 3 ? float3(0.0f, yFloor, 0.0f)
                                 : float3(rF * cos(a), yFloor, rF * sin(a));
        outPos[vid] = packed_float3(pos);
        outNrm[vid] = packed_float3(0.0f, -1.0f, 0.0f);
        return;
    }

    float a = float(j) / float(seg) * 2.0f * M_PI_F;
    float2 dir = float2(cos(a), sin(a));
    // Where the surface meets the wall at this azimuth: y = η(r(y)·dir), by fixed point (the
    // slope is small, so it contracts fast). The contact radius sets ρ for the second mode.
    float yc = p.surface.x;
    float rc = vlProfile(rTable, yc, y0, dy, n).x;
    for (int it = 0; it < 6; ++it) {
        rc = vlProfile(rTable, yc, y0, dy, n).x;
        yc = clamp(vlEta(dir * rc, rc, p), yFloor + 1e-3f, y0 + dy * float(n - 1));
    }
    rc = vlProfile(rTable, yc, y0, dy, n).x;

    if (region == 0) {
        // The axis: η at the centre, from the mean contact radius of the direction +x.
        float2 xz = float2(0.0f);
        outPos[vid] = packed_float3(0.0f, vlEta(xz, rc, p), 0.0f);
        outNrm[vid] = packed_float3(vlEtaGrad(xz, rc, p));
        return;
    }
    if (region == 1) {
        float rho = float(ring + 1u) / float(sr);
        float2 xz = dir * (rho * rc);
        outPos[vid] = packed_float3(xz.x, vlEta(xz, rc, p), xz.y);
        outNrm[vid] = packed_float3(vlEtaGrad(xz, rc, p));
        return;
    }
    // Wall: the vessel's inner surface from the floor up to the contact ring.
    float t = float(ring) / float(max(wr, 2u) - 1u);
    float y = mix(yFloor, yc, t);
    float2 rr = vlProfile(rTable, y, y0, dy, n);
    outPos[vid] = packed_float3(rr.x * dir.x, y, rr.x * dir.y);
    outNrm[vid] = packed_float3(normalize(float3(dir.x, -rr.y, dir.y)));
}

// ── The free surface from a simulated liquid's density (VesselLiquidMesh.encode(density:)) ──
//
// Same vertex layout and same wall/floor as `vesselLiquidMesh`, but the free surface is not a
// formula: it is where a SIMULATED liquid's density field (an MLS-MPM grid-mass buffer, in the
// vessel's own frame) falls through `iso`, found by walking up the vertical column from the floor
// — the top of the liquid resting on the floor, so a detached droplet above does not stand a
// column up to it. The density is read through a quadratic B-spline (C¹ — trilinear reads kink
// at every grid plane, which a mirror-smooth liquid prints as facets in its reflections) and
// each crossing refined by bisection, so the height field and its normals are smooth.
//
// Near the wall the grid's stencil is cut by the glass, so heights are read no closer than
// `inset` to it and continued to the wall along their own slope; each azimuth's contact line is
// the fixed point y = η(r(y)) of that continued surface against the inner profile (pass 1,
// `vesselLiquidContact`), and the wall rises to meet it exactly (pass 2).

struct VesselDensityGrid {
    float4 origin;      // xyz = node (0,0,0) position, w = node spacing
    uint4  res;         // xyz = node counts
    float4 extra;       // x = iso, y = inset, z = gradient half-step, w = fullness σ
    float4 profile;     // x = profile y0, y = profile dy, z = profile samples, w = floor y
    float4 extra2;      // x = the liquid's standoff from the glass (its first particle layer's gap)
};

static inline float3 vlBSplineW(float fx) {
    return float3(0.5f * (1.5f - fx) * (1.5f - fx),
                  0.75f - (fx - 1.0f) * (fx - 1.0f),
                  0.5f * (fx - 0.5f) * (fx - 0.5f));
}

// The node density through the quadratic B-spline (the MPM's own kernel): 27 taps, C¹.
static inline float vlGridDensity(device const float* rho, float3 p, constant VesselDensityGrid& g) {
    float3 f = (p - g.origin.xyz) / g.origin.w;
    int3 base = int3(floor(f - 0.5f));
    float3 fx = f - float3(base);
    float3 wx = vlBSplineW(fx.x), wy = vlBSplineW(fx.y), wz = vlBSplineW(fx.z);
    int3 res = int3(g.res.xyz);
    float sum = 0.0f;
    for (int k = 0; k < 3; ++k) {
        int z = clamp(base.z + k, 0, res.z - 1);
        for (int j = 0; j < 3; ++j) {
            int y = clamp(base.y + j, 0, res.y - 1);
            float wyz = wy[j] * wz[k];
            int row = (z * res.y + y) * res.x;
            for (int i = 0; i < 3; ++i) {
                int x = clamp(base.x + i, 0, res.x - 1);
                sum += wx[i] * wyz * rho[row + x];
            }
        }
    }
    return sum;
}

// Φ, the unit normal CDF (the tanh form, |error| < 2·10⁻⁴). The argument is clamped: fast-math
// tanh overflows its exp past ≈ 44 and returns NaN — s ≈ 10 (a point 0.6 m above the floor,
// σ = one node) got there, `max(NaN, 0.3)` then read the fullness as 0.3, and the density well
// away from the glass came out 3.3× too high: the surface stood a node above the liquid in the
// middle and fell to its true height near the wall (a dome; VesselDiagTests measured the axis
// crossing at 0.651 against the grid's own 0.590). Φ(±6) is 1 or 0 to 10⁻⁹.
static inline float vlPhi(float s) {
    s = clamp(s, -6.0f, 6.0f);
    return 0.5f * (1.0f + tanh(0.7978845608f * (s + 0.044715f * s * s * s)));
}

// The liquid density at p, corrected for the glass: the read's smoothing (P2G's B-spline, the
// blur, the B-spline read — together ≈ a Gaussian of σ ≈ one node) reaches into the wall and
// floor, where there is no liquid, so a FULL liquid reads ρ₀·Φ(d_wall/σ)·Φ(d_floor/σ) near
// them. Dividing that out keeps the iso crossing — the surface — unbiased up to the glass
// (uncorrected, the surface drooped 2.5 cm over the last ~3 nodes and tilted its normals 8°).
static inline float vlLiquidDensity(device const float* rho, float3 p, constant float* rTable,
                                    constant VesselDensityGrid& g) {
    uint n = uint(g.profile.z);
    float2 rr = vlProfile(rTable, p.y, g.profile.x, g.profile.y, n);
    float dWall = (rr.x - length(p.xz)) * rsqrt(1.0f + rr.y * rr.y);
    float dFloor = p.y - g.profile.w;
    float d0 = g.extra2.x;
    float full = vlPhi((dWall - d0) / g.extra.w) * vlPhi((dFloor - d0) / g.extra.w);
    return vlGridDensity(rho, p, g) / max(full, 0.3f);
}

// Where the column at radius r stands: the floor under the floor disc; out past it, the height
// at which the vessel's flaring wall has cleared the column by `clear`. A round-bellied vessel's
// columns near the wall stand on its curved bottom, not on the floor — walked from the floor's
// height they start inside the glass, read dry, and either drop the contact line to the floor
// or fake a slope that droops the surface into the wall (VesselLiquidMeshTests
// .testBelliedVesselSurfaceStaysLevelToTheWall). A column the wall never clears reads yTop (dry).
static inline float vlColumnBase(constant float* rTable, float r, float yFloor, float yTop, float clear,
                                 constant VesselDensityGrid& g) {
    uint n = uint(g.profile.z);
    float y0 = g.profile.x, dy = g.profile.y;
    float need = r + clear;
    float prevY = yFloor, prevR = vlProfile(rTable, yFloor, y0, dy, n).x;
    if (prevR >= need) return yFloor;
    for (uint i = 0; i < n; ++i) {
        float y = y0 + dy * float(i);
        if (y <= yFloor) continue;
        float ri = rTable[i];
        if (ri >= need) {
            float t = clamp((need - prevR) / max(ri - prevR, 1e-6f), 0.0f, 1.0f);
            return mix(prevY, y, t);
        }
        prevY = y; prevR = ri;
    }
    return yTop;
}

// Height of the liquid column standing on the vessel's bottom at xz; < yFloor + ε means dry.
static inline float vlColumnTop(device const float* rho, float2 xz, float yFloor, float yTop,
                                constant float* rTable, constant VesselDensityGrid& g) {
    float iso = g.extra.x, h = 0.5f * g.origin.w;
    float base = vlColumnBase(rTable, length(xz), yFloor, yTop, 0.25f * g.origin.w, g);
    if (base >= yTop) return yFloor;
    // Into the liquid: start three-quarters of a node up off the bottom, and let the corner —
    // where the glass-fullness correction is weakest — take up to a node more to read full.
    float y = base + 0.75f * g.origin.w;
    float prev = vlLiquidDensity(rho, float3(xz.x, y, xz.y), rTable, g);
    for (int k = 0; k < 2 && prev < iso; ++k) {
        y += h;
        prev = vlLiquidDensity(rho, float3(xz.x, y, xz.y), rTable, g);
    }
    if (prev < iso) return yFloor;
    for (int k = 0; k < 160 && y < yTop; ++k) {
        float y2 = y + h;
        float d2 = vlLiquidDensity(rho, float3(xz.x, y2, xz.y), rTable, g);
        if (d2 < iso) {
            // Bridge a one-sample dip (a pressure ripple, a pinhole) when the liquid resumes
            // right above it; otherwise the surface is in [y, y2]: bisect it.
            float d3 = vlLiquidDensity(rho, float3(xz.x, y2 + h, xz.y), rTable, g);
            if (d3 >= iso) { y = y2 + h; prev = d3; continue; }
            float lo = y, hi = y2;
            for (int b = 0; b < 5; ++b) {
                float mid = 0.5f * (lo + hi);
                if (vlLiquidDensity(rho, float3(xz.x, mid, xz.y), rTable, g) >= iso) lo = mid; else hi = mid;
            }
            return 0.5f * (lo + hi);
        }
        y = y2; prev = d2;
    }
    return min(y, yTop);
}

static inline float2 vlInDisc(float2 q, float r) {
    float l = length(q);
    return l > r ? q * (r / max(l, 1e-6f)) : q;
}

// The surface at xz as (height, ∂η/∂x, ∂η/∂z): read at the nearest point no closer than
// `rMax` to the axis-centred wall disc, its slope from a centred stencil that stays inside it,
// continued linearly out to xz. A dry column there steps inward until it finds the liquid.
static inline float3 vlSurfaceAt(device const float* rho, float2 xz, float rMax, float yFloor, float yTop,
                                 constant float* rTable, constant VesselDensityGrid& g) {
    float hs = g.extra.z;
    float2 q = vlInDisc(xz, rMax);
    float hq = vlColumnTop(rho, q, yFloor, yTop, rTable, g);
    for (int k = 0; k < 4 && hq <= yFloor + 1e-4f && length(q) > hs; ++k) {
        q *= max(0.0f, 1.0f - 1.5f * hs / length(q));
        hq = vlColumnTop(rho, q, yFloor, yTop, rTable, g);
    }
    float2 c = vlInDisc(q, max(rMax - hs, 0.0f));
    float2 ex = float2(hs, 0.0f), ez = float2(0.0f, hs);
    float gx = (vlColumnTop(rho, c + ex, yFloor, yTop, rTable, g) - vlColumnTop(rho, c - ex, yFloor, yTop, rTable, g)) / (2.0f * hs);
    float gz = (vlColumnTop(rho, c + ez, yFloor, yTop, rTable, g) - vlColumnTop(rho, c - ez, yFloor, yTop, rTable, g)) / (2.0f * hs);
    // A column that reads dry has no slope to trust (its height is the floor, not a surface).
    if (hq <= yFloor + 1e-4f) { gx = 0.0f; gz = 0.0f; }
    float h = hq + gx * (xz.x - q.x) + gz * (xz.y - q.y);
    return float3(clamp(h, yFloor, yTop), gx, gz);
}

// Pass 1 — one thread per azimuth: where the (continued) surface meets the inner wall.
kernel void vesselLiquidContact(constant float*              rTable  [[buffer(0)]],
                                device float2*               contact [[buffer(1)]],
                                constant VesselLiquidParams& p       [[buffer(3)]],
                                device const float*          rho     [[buffer(4)]],
                                constant VesselDensityGrid&  g       [[buffer(5)]],
                                uint j [[thread_position_in_grid]]) {
    uint n = uint(p.counts.x), seg = uint(p.counts.y);
    if (j >= seg) return;
    float y0 = p.mode2.z, dy = p.mode2.w, yFloor = p.surface.w;
    float yTop = y0 + dy * float(n - 1);
    float a = float(j) / float(seg) * 2.0f * M_PI_F;
    float2 dir = float2(cos(a), sin(a));
    float yc = vlColumnTop(rho, float2(0.0f), yFloor, yTop, rTable, g);
    float rc = vlProfile(rTable, yc, y0, dy, n).x;
    for (int it = 0; it < 5; ++it) {
        rc = vlProfile(rTable, yc, y0, dy, n).x;
        yc = vlSurfaceAt(rho, dir * rc, max(rc - g.extra.y, 0.0f), yFloor, yTop, rTable, g).x;
    }
    contact[j] = float2(yc, vlProfile(rTable, yc, y0, dy, n).x);
}

// Pass 2 — one thread per vertex.
kernel void vesselLiquidFromDensity(constant float*              rTable  [[buffer(0)]],
                                    device packed_float3*        outPos  [[buffer(1)]],
                                    device packed_float3*        outNrm  [[buffer(2)]],
                                    constant VesselLiquidParams& p       [[buffer(3)]],
                                    device const float*          rho     [[buffer(4)]],
                                    constant VesselDensityGrid&  g       [[buffer(5)]],
                                    device const float2*         contact [[buffer(6)]],
                                    uint vid [[thread_position_in_grid]]) {
    if (vid >= p.vertexCount) return;
    uint n = uint(p.counts.x), seg = uint(p.counts.y), sr = uint(p.counts.z), wr = uint(p.counts.w);
    float y0 = p.mode2.z, dy = p.mode2.w, yFloor = p.surface.w;
    float yTop = y0 + dy * float(n - 1);

    uint surfCount = 1u + sr * seg;
    uint wallCount = wr * seg;
    uint j = 0u;
    int region;
    uint ring = 0u;
    if (vid == 0u) { region = 0; }
    else if (vid < surfCount) { uint k = vid - 1u; ring = k / seg; j = k - ring * seg; region = 1; }
    else if (vid < surfCount + wallCount) { uint k = vid - surfCount; ring = k / seg; j = k - ring * seg; region = 2; }
    else if (vid == surfCount + wallCount) { region = 3; }
    else { j = vid - surfCount - wallCount - 1u; region = 4; }

    if (region == 3 || region == 4) {
        float rF = vlProfile(rTable, yFloor, y0, dy, n).x;
        float a = float(j) / float(seg) * 2.0f * M_PI_F;
        float3 pos = region == 3 ? float3(0.0f, yFloor, 0.0f) : float3(rF * cos(a), yFloor, rF * sin(a));
        outPos[vid] = packed_float3(pos);
        outNrm[vid] = packed_float3(0.0f, -1.0f, 0.0f);
        return;
    }

    float a = float(j) / float(seg) * 2.0f * M_PI_F;
    float2 dir = float2(cos(a), sin(a));
    float2 cr = contact[j];
    float yc = cr.x, rc = cr.y;

    if (region == 2) {
        float t = float(ring) / float(max(wr, 2u) - 1u);
        float y = mix(yFloor, yc, t);
        float2 rr = vlProfile(rTable, y, y0, dy, n);
        outPos[vid] = packed_float3(rr.x * dir.x, y, rr.x * dir.y);
        outNrm[vid] = packed_float3(normalize(float3(dir.x, -rr.y, dir.y)));
        return;
    }

    // Surface: the polar grid out to the contact ring (which is exactly the contact line).
    float rho01 = region == 0 ? 0.0f : float(ring + 1u) / float(sr);
    float2 xz = dir * (rho01 * rc);
    float3 s = vlSurfaceAt(rho, xz, max(rc - g.extra.y, 0.0f), yFloor, yTop, rTable, g);
    float y = (region == 1 && ring + 1u == sr) ? yc : s.x;
    outPos[vid] = packed_float3(xz.x, y, xz.y);
    outNrm[vid] = packed_float3(normalize(float3(-s.y, 1.0f, -s.z)));
}

// One axis of a [1 2 1]/4 blur over a node grid (the density before the column walk): the
// grid's particle-scale ripple is below what the surface can show, and it would print as a
// rough mirror. axis 0/1/2 = x/y/z; edges clamp.
kernel void vesselDensityBlur(device const float*         src  [[buffer(0)]],
                              device float*               dst  [[buffer(1)]],
                              constant VesselDensityGrid& g    [[buffer(2)]],
                              constant uint&              axis [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {
    uint3 res = g.res.xyz;
    if (any(gid >= res)) return;
    uint stride = axis == 0u ? 1u : (axis == 1u ? res.x : res.x * res.y);
    uint c = gid.x + res.x * (gid.y + res.y * gid.z);
    uint coord = axis == 0u ? gid.x : (axis == 1u ? gid.y : gid.z);
    uint lo = coord > 0u ? c - stride : c;
    uint hi = coord + 1u < res[axis] ? c + stride : c;
    dst[c] = 0.25f * src[lo] + 0.5f * src[c] + 0.25f * src[hi];
}
