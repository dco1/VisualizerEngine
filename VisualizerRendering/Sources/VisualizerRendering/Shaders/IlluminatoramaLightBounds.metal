// IlluminatoramaLightBounds.metal — tiled light masks for the deferred lighting kernel (Daydream DH-1625).
//
// `illumi_lighting` walks every point, spot and area light for every pixel. Each light has an exact reach —
// outside it the light contributes exactly 0 — but finding that out costs the pixel a distance (and for a cone a
// normalise, a dot and a smoothstep) per light. On 4000 Sunset (34 cones, 21 points, 17 portals) those loops were
// 4.0 ms of a 5.2 ms orbit lighting pass and ~15 ms of a 21 ms settled one, though a pixel is lit by a handful.
//
// `illumi_light_tiles` runs once per frame, one thread per 16 × 16 pixel TILE. It reads the tile's depth range
// from the G-buffer, unprojects the tile's slab of the view frustum to a world box (the eight corners, through the
// same `invViewProjection` the lighting kernel rebuilds positions with), and tests every light's reach against
// it; the lights that can touch the box get a bit in the tile's mask (IlluminatoramaLightTiles.h). The lighting
// kernel then tests ONE BIT per light instead of the light. Every test is conservative, so the picture is
// unchanged — a light is dropped only where the lighting kernel would have computed exactly 0 for every surface
// in the tile:
//   point  — ball `radius`:                `if (dist > pl.radius) continue`
//   area   — ball `radius` about the centre (`evalAreaLight`: `dist > al.radius → 0`), and for a one-sided light
//            the half-space in front of it  (`!twoSided && facing <= 0 → 0`)
//   spot   — (a) the ball `radius` about the apex and about the opening disc's centre (the two laws' ranges);
//            (b) the cone, from a VIRTUAL apex far enough behind both the apex and the opening disc that one cone
//            of the same half-angle contains the apex law's cone and every disc tap's cone (back by r/tan θ from
//            the disc), tested against the tile box's bounding sphere. Skipped for half-angles ≥ ~87°.
// A tile with no geometry (all sky) gets an empty mask. `enabled == 0` writes all-ones (the A/B lever).
#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaLightTiles.h"
using namespace metal;

/// Squared distance from `p` to the box [lo, hi] (0 inside).
static inline float lightTileDist2(float3 p, float3 lo, float3 hi) {
    float3 d = max(max(lo - p, p - hi), 0.0);
    return dot(d, d);
}

kernel void illumi_light_tiles(
    constant FrameUniforms&       frame       [[buffer(0)]],
    const device PointLight*      pointLights [[buffer(1)]],
    const device SpotLight*       spotLights  [[buffer(3)]],
    const device AreaLight*       areaLights  [[buffer(4)]],
    device uint*                  tileMasks   [[buffer(12)]],
    constant uint&                enabled     [[buffer(13)]],
    depth2d<float, access::read>  gDepth      [[texture(3)]],
    uint2                         tile        [[thread_position_in_grid]])
{
    uint w = gDepth.get_width(), h = gDepth.get_height();
    uint tilesWide = (w + kLightTileSize - 1u) / kLightTileSize;
    uint tilesHigh = (h + kLightTileSize - 1u) / kLightTileSize;
    if (tile.x >= tilesWide || tile.y >= tilesHigh) return;
    uint base = (tile.y * tilesWide + tile.x) * kLightTileWords;
    if (enabled == 0u) {
        for (uint k = 0u; k < kLightTileWords; ++k) tileMasks[base + k] = 0xFFFFFFFFu;
        return;
    }

    // The tile's depth range over its geometry (sky = the cleared 1.0, which the lighting kernel never lights).
    uint2 p0 = tile * kLightTileSize, p1 = min(p0 + kLightTileSize, uint2(w, h));
    float zMin = 1.0, zMax = 0.0;
    for (uint y = p0.y; y < p1.y; ++y) {
        for (uint x = p0.x; x < p1.x; ++x) {
            float z = gDepth.read(uint2(x, y));
            if (z >= 0.99999) continue;
            zMin = min(zMin, z); zMax = max(zMax, z);
        }
    }
    uint mask[kLightTileWords];
    for (uint k = 0u; k < kLightTileWords; ++k) mask[k] = 0u;
    if (zMax < zMin) {   // no geometry
        for (uint k = 0u; k < kLightTileWords; ++k) tileMasks[base + k] = 0u;
        return;
    }

    // World box of the tile's frustum slab: the lighting kernel's own NDC (pixel centres, y up) and depth.
    // The slab is convex, so the box of its eight corners holds every position the kernel can rebuild in it.
    float2 n0 = float2(p0) / float2(w, h) * 2.0 - 1.0, n1 = float2(p1) / float2(w, h) * 2.0 - 1.0;
    float3 lo = float3(1e30), hi = float3(-1e30);
    bool ok = true;
    for (uint k = 0u; k < 8u; ++k) {
        float4 clip = float4((k & 1u) != 0u ? n1.x : n0.x, -((k & 2u) != 0u ? n1.y : n0.y),
                             (k & 4u) != 0u ? zMax : zMin, 1.0);
        float4 wp = frame.invViewProjection * clip;
        if (!(abs(wp.w) > 1e-12)) { ok = false; break; }
        float3 q = wp.xyz / wp.w;
        if (!all(isfinite(q))) { ok = false; break; }
        lo = min(lo, q); hi = max(hi, q);
    }
    if (!ok) {   // cannot bound the tile: every light
        for (uint k = 0u; k < kLightTileWords; ++k) tileMasks[base + k] = 0xFFFFFFFFu;
        return;
    }
    // Rounding slack: a centimetre plus a part in a thousand of the box.
    float3 pad = 0.01 + 1e-3 * (hi - lo);
    lo -= pad; hi += pad;
    float3 bc = 0.5 * (lo + hi);
    float  br = 0.5 * length(hi - lo);

    uint nP = frame.pointLightCount, nS = frame.spotLightCount, nA = frame.areaLightCount;
    uint n = min(nP + nS + nA, kLightTileBits);
    for (uint i = 0u; i < n; ++i) {
        bool reach;
        if (i < nP) {
            float r = pointLights[i].radius;
            reach = !(lightTileDist2(pointLights[i].position, lo, hi) > r * r);
        } else if (i < nP + nS) {
            uint s = i - nP;
            float3 pos = spotLights[s].position, d = spotLights[s].direction;
            float  R = spotLights[s].radius, off = spotLights[s].openingOffset;
            float  ro = max(spotLights[s].openingRadius, 0.0), oc = spotLights[s].outerCone;
            float  dl = length(d);
            float3 ax = dl > 1e-6 ? d / dl : float3(0.0, -1.0, 0.0);
            float3 c = pos + ax * off;   // the opening disc's centre (the disc law's range origin)
            // (a) range: inside the apex law's ball OR the disc law's.
            reach = !(lightTileDist2(pos, lo, hi) > R * R) || !(lightTileDist2(c, lo, hi) > R * R);
            // (b) cone, where the angle bounds anything.
            if (reach && dl > 1e-6 && oc > 0.05 && oc < 0.99999) {
                float sn = sqrt(max(1.0 - oc * oc, 0.0));
                float3 apex = pos + ax * (min(off, 0.0) - ro * oc / sn);
                float3 v = bc - apex;
                float  along = dot(v, ax);
                float  lateral = sqrt(max(dot(v, v) - along * along, 0.0));
                // Signed distance from the box's centre to the cone's surface (positive outside); also behind
                // the apex, where the nearest point is the apex itself.
                float  outside = lateral * oc - along * sn;
                if (outside > br && (along > 0.0 || dot(v, v) > br * br)) reach = false;
            }
        } else {
            uint a = i - nP - nS;
            float3 c = areaLights[a].center;
            float  r = areaLights[a].radius;
            reach = !(lightTileDist2(c, lo, hi) > r * r);
            if (reach && areaLights[a].twoSided <= 0.5) {
                float3 nL = cross(areaLights[a].ex, areaLights[a].ey);
                float  nl = length(nL);
                if (nl > 1e-8) {
                    nL /= nl;
                    // The box's farthest reach along +nL: centre projection + its half-extent's support.
                    float front = dot(nL, bc - c) + dot(abs(nL), 0.5 * (hi - lo));
                    if (front <= 0.0) reach = false;   // every point of the tile is behind the emitting face
                }
            }
        }
        if (reach) mask[i >> 5u] |= 1u << (i & 31u);
    }
    for (uint k = 0u; k < kLightTileWords; ++k) tileMasks[base + k] = mask[k];
}
