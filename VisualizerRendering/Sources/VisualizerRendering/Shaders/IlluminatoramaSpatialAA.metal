#include <metal_stdlib>
using namespace metal;

// ── Spatial anti-aliasing (Daydream, 2026-11-16) ─────────────────────────────────────────────────
//
// Edge anti-aliasing for a frame with NO temporal accumulation — the live canvas, which draws one
// picture whether or not anything moves, so it cannot borrow AA from a history (a history is what
// ghosted). The algorithm is FXAA 3.11's quality path (Lottes 2011): find the local edge from the
// 3 × 3 luma, walk along it both ways until the luma pair it separates ends, and resample the
// pixel across the edge by how far it sits from the nearer end — a staircase becomes a ramp. The
// sub-pixel term (`subpix`) softens single-texel features in proportion to their contrast.
//
// It runs on the scene-referred HDR at the INTERNAL resolution, before exposure, bloom and the
// tonemap's upscale, so it works on the pixels that actually step. Luma is a tone-compressed
// sqrt(Y / (1 + Y)) so a window 50× the wall reads as an edge, not as the whole range.
static inline float illumiAALuma(texture2d<half, access::sample> t, sampler s, float2 uv) {
    float3 c = float3(t.sample(s, uv, level(0)).rgb);
    float y = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 0.0);
    return sqrt(y / (1.0 + y));
}

struct IllumiSpatialAAParams {
    float edgeThreshold;      // relative contrast an edge needs (FXAA "quality edge threshold")
    float edgeThresholdMin;   // absolute floor, so dark noise is left alone
    float subpix;             // 0 = edges only, 1 = full sub-pixel softening
    float pad;
};

constant float kIllumiAAStep[12] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0 };

kernel void illumiSpatialAA(texture2d<half, access::sample> src [[texture(0)]],
                            texture2d<half, access::write>  dst [[texture(1)]],
                            constant IllumiSpatialAAParams& p   [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    uint W = dst.get_width(), H = dst.get_height();
    if (gid.x >= W || gid.y >= H) return;
    constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);
    float2 rcp = 1.0 / float2(W, H);
    float2 uv = (float2(gid) + 0.5) * rcp;

    float lM = illumiAALuma(src, s, uv);
    float lD = illumiAALuma(src, s, uv + float2(0.0, -rcp.y));
    float lU = illumiAALuma(src, s, uv + float2(0.0,  rcp.y));
    float lL = illumiAALuma(src, s, uv + float2(-rcp.x, 0.0));
    float lR = illumiAALuma(src, s, uv + float2( rcp.x, 0.0));
    float hi = max(lM, max(max(lD, lU), max(lL, lR)));
    float lo = min(lM, min(min(lD, lU), min(lL, lR)));
    float range = hi - lo;
    half4 centre = src.sample(s, uv, level(0));
    if (range < max(p.edgeThresholdMin, hi * p.edgeThreshold)) { dst.write(centre, gid); return; }

    float lDL = illumiAALuma(src, s, uv + float2(-rcp.x, -rcp.y));
    float lUR = illumiAALuma(src, s, uv + float2( rcp.x,  rcp.y));
    float lUL = illumiAALuma(src, s, uv + float2(-rcp.x,  rcp.y));
    float lDR = illumiAALuma(src, s, uv + float2( rcp.x, -rcp.y));

    float lDU = lD + lU, lLR = lL + lR;
    float lLc = lDL + lUL, lDc = lDL + lDR, lRc = lDR + lUR, lUc = lUR + lUL;
    float edgeH = abs(-2.0 * lL + lLc) + abs(-2.0 * lM + lDU) * 2.0 + abs(-2.0 * lR + lRc);
    float edgeV = abs(-2.0 * lU + lUc) + abs(-2.0 * lM + lLR) * 2.0 + abs(-2.0 * lD + lDc);
    bool horiz = edgeH >= edgeV;

    float l1 = horiz ? lD : lL, l2 = horiz ? lU : lR;
    float g1 = l1 - lM, g2 = l2 - lM;
    bool steep1 = abs(g1) >= abs(g2);
    float gScaled = 0.25 * max(abs(g1), abs(g2));
    float stepLen = horiz ? rcp.y : rcp.x;
    float lAvg;
    if (steep1) { stepLen = -stepLen; lAvg = 0.5 * (l1 + lM); } else { lAvg = 0.5 * (l2 + lM); }

    float2 cur = uv;
    if (horiz) cur.y += stepLen * 0.5; else cur.x += stepLen * 0.5;
    float2 off = horiz ? float2(rcp.x, 0.0) : float2(0.0, rcp.y);
    float2 uv1 = cur - off, uv2 = cur + off;
    float e1 = illumiAALuma(src, s, uv1) - lAvg;
    float e2 = illumiAALuma(src, s, uv2) - lAvg;
    bool r1 = abs(e1) >= gScaled, r2 = abs(e2) >= gScaled;
    if (!r1) uv1 -= off;
    if (!r2) uv2 += off;
    if (!(r1 && r2)) {
        for (int i = 2; i < 12; ++i) {
            if (!r1) e1 = illumiAALuma(src, s, uv1) - lAvg;
            if (!r2) e2 = illumiAALuma(src, s, uv2) - lAvg;
            r1 = abs(e1) >= gScaled; r2 = abs(e2) >= gScaled;
            if (!r1) uv1 -= off * kIllumiAAStep[i];
            if (!r2) uv2 += off * kIllumiAAStep[i];
            if (r1 && r2) break;
        }
    }
    float d1 = horiz ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float d2 = horiz ? (uv2.x - uv.x) : (uv2.y - uv.y);
    bool dir1 = d1 < d2;
    float dMin = min(d1, d2);
    float thick = max(d1 + d2, 1e-6);
    float pixOff = -dMin / thick + 0.5;
    bool centreSmaller = lM < lAvg;
    bool goodSpan = ((dir1 ? e1 : e2) < 0.0) != centreSmaller;
    float finalOff = goodSpan ? pixOff : 0.0;

    float lAll = (1.0 / 12.0) * (2.0 * (lDU + lLR) + lLc + lRc);
    float sp1 = clamp(abs(lAll - lM) / max(range, 1e-6), 0.0, 1.0);
    float sp2 = (-2.0 * sp1 + 3.0) * sp1 * sp1;
    finalOff = max(finalOff, sp2 * sp2 * p.subpix);

    float2 fuv = uv;
    if (horiz) fuv.y += finalOff * stepLen; else fuv.x += finalOff * stepLen;
    half4 c = src.sample(s, fuv, level(0));
    dst.write(half4(c.rgb, centre.a), gid);
}
