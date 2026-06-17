#include <metal_stdlib>
using namespace metal;

// ── SoftServeSolver.metal ──────────────────────────────────────────────────────
//
// GPU kernels that replace the two dominant CPU hot-paths in main's CURRENT
// (steady-treadmill + baked-flavour) InfiniteSoftServeController:
//
//   ss_coil_extrude      — per-vertex coil POSITION+NORMAL blend + cup flare,
//                          replacing updateCoilExtrusion() (~43ms/frame).
//                          IMPORTANT: this kernel does NOT touch per-vertex
//                          colour — flavour is BAKED ONCE on the CPU
//                          (bakeFlavorColors) and rides the geometry, a
//                          TAA-correctness design that must not be reverted.
//   ss_depth_init        — fill depth table to grooveFloorR (= coilRadius).
//   ss_depth_scatter     — per-coil-vertex forward-scatter (atomic max radius).
//   ss_dilate_max        — 3×3 float dilation (ping-pong).
//   ss_box_blur_float    — 3×3 box blur (ping-pong).
//   ss_cover_threshold   — binary cover mask from film-h > revealFloor.
//   ss_dilate_bin        — 3×3 binary OR.
//   ss_erode_bin         — 3×3 binary AND.
//   ss_coat_displace     — per-shell-vertex radius from smoothed depth +
//                          coverage + edge-taper + finite-diff normal +
//                          per-cell flavour colour (incl. birthday-cake dots).
//
// Together the depth/morph/displace passes replace displaceChocoCoat()
// (~205ms on CPU). Math is ported VERBATIM from main's method bodies — verified
// against the CPU path A/B (VIZ_SOFTSERVE_GPU=0/1).
//
// plausibility: real — exact same math as main's CPU paths, parallelised on GPU.

// ── Shared vertex type — MUST match IlluminatoramaVertex in Swift. stride 80. ──
struct IlluminatoramaVertex {
    float3  position;   // offset  0
    float   _padPos;    // offset 12
    float3  normal;     // offset 16
    float   _padNrm;    // offset 28
    float2  uv;         // offset 32
    float2  _padUv;     // offset 40
    float4  tangent;    // offset 48
    float4  color;      // offset 64
    // total = 80
};

// ── Coil extrusion uniform — Swift mirror: CoilUniforms. 48 bytes. ─────────────
struct CoilUniforms {
    float  tipTurns;            // extruded / pitch
    float  coilStraightTurns;   // LIVE straight-section length (per frame)
    float  pitch;
    float  dieExitY;
    float  tubeRadius;          // settings.tubeRadius  (cupFlare coilOuterR = coilRadius+tubeRadius)
    float  coilRadius;          // Self.coilRadius
    float  funnelStraightTurns; // Self.funnelStraightTurns (ropeCenterlineLocal origin)
    float  funnelRampTurns;     // Self.funnelRampTurns
    float  liveCupRimWorldY;
    float  coneHeight;
    float  coneFloorY;          // CakeConeGeometry.floorY
    uint   vertexCount;
    // 12 × 4 = 48 bytes ✓
};

// ── Coat displace uniform — Swift mirror: CoatUniforms. 64 bytes. ─────────────
struct CoatUniforms {
    float  coilRadius;    // grooveFloorR fallback (= Self.coilRadius)
    float  tubeRadius;    // settings.tubeRadius
    float  gap;           // chocoShellGap = 0.0015
    float  rowH;          // chocoFilmRowH
    float  topY;          // chocoFilmTopY
    float  maxWedge;      // chocoFilmMaxWedge
    float  subRow;        // chocoFilmAdvectAccum (fractional row offset)
    float  dAlpha;        // 2*maxWedge / (cols-1)
    float  hiddenR;       // coilOuterR - 0.03
    float  revealFloor;   // chocoFilmRevealFloor (for the debug branch compare)
    uint   cols;
    uint   rows;
    uint   debug;         // chocoDebug ? 1 : 0
    uint   pad0;
    uint   pad1;
    uint   pad2;
    // 16 × 4 = 64 bytes ✓
};

// ── Depth scatter uniform — Swift mirror: DepthScatterUniforms. 32 bytes. ──────
struct DepthScatterUniforms {
    float  phi;
    float  coilRadius;    // grooveFloorR
    float  dieExitY;
    float  rowH;          // chocoFilmRowH
    float  topY;          // chocoFilmTopY
    float  maxWedge;      // chocoFilmMaxWedge
    uint   cols;
    uint   rows;
    uint   totalCoilVerts;
    uint   pad0;
    uint   pad1;
    uint   pad2;
    // 12 × 4 = 48 bytes ✓
};

// ── Morph/cover uniform. 16 bytes. ────────────────────────────────────────────
struct MorphUniforms {
    uint   rows;
    uint   cols;
    float  revealFloor;   // for ss_cover_threshold
    uint   pad;
};

// ── Steady-treadmill clip uniform — Swift mirror: SteadyClipUniforms. 32 bytes ──
// The DESCEND-phase coil is a rigid 56-turn helix translated straight down. Its
// fixed bottom (plus the yScroll recycle shift) sits well BELOW the cake-cup
// floor, so the swirl pokes out the bottom of the cone. This kernel collapses
// every steady vertex whose LOCAL y is below `cutoffLocalY` (= cup floor in the
// mesh's local frame) onto a single hidden cap point on the helix centreline at
// the cutoff — closing the tube exactly at the cup floor, the same trick
// ss_coil_extrude uses for the funnel tip.
struct SteadyClipUniforms {
    float  cutoffLocalY;  // collapse verts with origPos.y < this
    float  capX;          // collapse target (local space) …
    float  capY;
    float  capZ;
    uint   vertexCount;
    uint   pad0;
    uint   pad1;
    uint   pad2;
    // 8 × 4 = 32 bytes ✓
};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers — ported verbatim from the controller.
// ──────────────────────────────────────────────────────────────────────────────

// coilBlendFrac(tt): smoothstep keyed to the LIVE coilStraightTurns (bw=0.45).
static float coilBlendFrac(float tt, float coilStraightTurns) {
    const float bw = 0.45f;
    float x = saturate((tt - (coilStraightTurns - bw)) / (2.0f * bw));
    return x * x * (3.0f - 2.0f * x);
}

// ropeCenterlineLocal(s): the static helix centreline at arc s.
static float3 ropeCenterlineLocal(float s, constant CoilUniforms& u) {
    float coilTT = max(0.0f, s - u.funnelStraightTurns);
    float theta  = coilTT * 2.0f * M_PI_F;
    float ramp   = saturate(coilTT / u.funnelRampTurns);
    float cr     = u.coilRadius * (ramp * ramp * (3.0f - 2.0f * ramp));   // smoothstep
    return float3(cr * cos(theta), -s * u.pitch, cr * sin(theta));
}

// cupFlare(p): widen XZ to follow the cup taper / mushroom band.
static float3 cupFlare(float3 p, constant CoilUniforms& u) {
    float coilOuterR    = u.coilRadius + u.tubeRadius;            // ~0.076 natural outer
    float rimLocalY     = u.liveCupRimWorldY - u.dieExitY;
    float cupInnerDepth = u.coneHeight - u.coneFloorY;
    float floorLocalY   = rimLocalY - cupInnerDepth;
    const float innerAtRim   = 0.088f;
    const float innerAtFloor = 0.060f;
    const float mushroomBand = 0.05f;
    float y = p.y;
    float targetR;
    if (y <= rimLocalY) {
        float uu = saturate((y - floorLocalY) / max(1e-4f, cupInnerDepth));
        targetR  = innerAtFloor + (innerAtRim - innerAtFloor) * uu;
    } else if (y <= rimLocalY + mushroomBand) {
        float m  = (y - rimLocalY) / mushroomBand;
        float sm = m * m * (3.0f - 2.0f * m);
        targetR  = innerAtRim + (coilOuterR - innerAtRim) * sm;
    } else {
        return p;
    }
    float s = targetR / coilOuterR;
    return float3(p.x * s, p.y, p.z * s);
}

// srgbToLinear — pow(c, 2.2) componentwise.
static float3 srgbToLinear(float3 c) { return powr(saturate(c), 2.2f); }

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 1 — ss_coil_extrude
// One thread per coil vertex. Mirrors updateCoilExtrusion() EXACTLY.
//   tt = arcTurns[i]
//   if tt > tipTurns:  position = tipLocal (collapsed); normal UNTOUCHED.
//   else:              position = cupFlare(blend straight↔coil by coilBlendFrac);
//                      normal   = normalize(blend straight↔coil normals).
// Colour is NOT written here — flavour is baked once on the CPU (TAA-correct).
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_coil_extrude(
    constant CoilUniforms&         u             [[ buffer(0) ]],
    constant float*                arcTurns      [[ buffer(1) ]],
    constant float4*               origPos       [[ buffer(2) ]],
    constant float4*               origNorm      [[ buffer(3) ]],
    constant float4*               straightPos   [[ buffer(4) ]],
    constant float4*               straightNorm  [[ buffer(5) ]],
    device   IlluminatoramaVertex* out           [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= u.vertexCount) return;

    float tt       = arcTurns[gid];
    float tipTurns = u.tipTurns;

    if (tt > tipTurns) {
        // Collapse to the CURRENT TIP — blended straight↔coil, then cup-flared.
        // tipStraight = (0, -tipTurns*pitch, 0)
        float3 tipStraight = float3(0.0f, -tipTurns * u.pitch, 0.0f);
        float3 tipCoil     = ropeCenterlineLocal(tipTurns, u);
        float  tipF        = coilBlendFrac(tipTurns, u.coilStraightTurns);
        float3 tipLocal    = cupFlare(tipStraight + (tipCoil - tipStraight) * tipF, u);
        out[gid].position  = tipLocal;
        // normal intentionally untouched (matches the CPU path).
    } else {
        float  f  = coilBlendFrac(tt, u.coilStraightTurns);
        float3 sp = straightPos[gid].xyz;
        float3 cp = origPos[gid].xyz;
        float3 sn = straightNorm[gid].xyz;
        float3 cn = origNorm[gid].xyz;
        out[gid].position = cupFlare(sp + (cp - sp) * f, u);
        out[gid].normal   = normalize(sn + (cn - sn) * f);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 1b — ss_steady_clip
// One thread per STEADY-mesh vertex. Reads the baked-original steady position
// (a separate, never-mutated buffer) and writes into the live steady vertex
// buffer: verts at/above the cup-floor cutoff pass through unchanged; verts
// below collapse onto the hidden centreline cap. Idempotent (always reads from
// the orig buffer), positions only — colour/uv stay baked.
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_steady_clip(
    constant SteadyClipUniforms&   u       [[ buffer(0) ]],
    constant float4*               origPos [[ buffer(1) ]],
    device   IlluminatoramaVertex* out     [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= u.vertexCount) return;
    float3 p = origPos[gid].xyz;
    out[gid].position = (p.y < u.cutoffLocalY)
        ? float3(u.capX, u.capY, u.capZ)
        : p;
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 2 — ss_depth_init   (fill every cell to grooveFloorR = coilRadius)
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_depth_init(
    constant DepthScatterUniforms& u          [[ buffer(0) ]],
    device   uint*                 depthTable [[ buffer(1) ]],   // float-as-uint
    uint gid [[ thread_position_in_grid ]])
{
    uint cells = u.rows * u.cols;
    if (gid >= cells) return;
    depthTable[gid] = as_type<uint>(u.coilRadius);
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 3 — ss_depth_scatter
// One thread per coil ORIG vertex. Atomic-max XZ radius into the front bin.
// Mirrors the depth-build loop in displaceChocoCoat(). chocoFrontDir=(0,0,1):
//   az = atan2(hx, hz);  front check: wz/hl > 0.
// All-positive radii → atomic_fetch_max on the float bit-pattern is order-safe.
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_depth_scatter(
    constant DepthScatterUniforms& u          [[ buffer(0) ]],
    constant float4*               origPos    [[ buffer(1) ]],   // local-space, xyz
    device   atomic_uint*          depthTable [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= u.totalCoilVerts) return;

    float3 lp   = origPos[gid].xyz;
    float  wy   = lp.y + u.dieExitY;
    float  rRow = (u.topY - wy) / u.rowH;
    if (rRow < 0.0f || rRow > float(u.rows - 1)) return;

    float cphi = cos(u.phi), sphi = sin(u.phi);
    float wx   =  cphi * lp.x + sphi * lp.z;
    float wz   = -sphi * lp.x + cphi * lp.z;

    // chocoAzOffset with front=(0,0,1): az = atan2(hx, hz).
    float hl = max(1e-5f, sqrt(wx * wx + wz * wz));
    float hx = wx / hl, hz = wz / hl;
    float az = atan2(hx, hz);
    if (fabs(az) > u.maxWedge) return;

    // Front-facing only: (wx/hl)*dx + (wz/hl)*dz with d=(0,0,1) → hz < 0 skips.
    if (hz < 0.0f) return;

    int rr = int(rRow + 0.5f);
    float invAz = float(u.cols - 1) / (2.0f * u.maxWedge);
    int cc = int((az + u.maxWedge) * invAz + 0.5f);
    if (rr < 0 || rr >= int(u.rows) || cc < 0 || cc >= int(u.cols)) return;

    uint kk = uint(rr) * u.cols + uint(cc);
    atomic_fetch_max_explicit(&depthTable[kk], as_type<uint>(hl), memory_order_relaxed);
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 4 — ss_dilate_max   (3×3 greyscale dilation)
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_dilate_max(
    constant MorphUniforms& u   [[ buffer(0) ]],
    constant float*         src [[ buffer(1) ]],
    device   float*         dst [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint c = gid.x, r = gid.y;
    if (r >= u.rows || c >= u.cols) return;
    uint r0 = r > 0 ? r - 1 : 0, r1 = r < u.rows - 1 ? r + 1 : u.rows - 1;
    uint c0 = c > 0 ? c - 1 : 0, c1 = c < u.cols - 1 ? c + 1 : u.cols - 1;
    float m = src[r * u.cols + c];
    for (uint rr = r0; rr <= r1; ++rr)
        for (uint cc = c0; cc <= c1; ++cc)
            m = max(m, src[rr * u.cols + cc]);
    dst[r * u.cols + c] = m;
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 5 — ss_box_blur_float   (3×3 box blur, edge-clamped count)
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_box_blur_float(
    constant MorphUniforms& u   [[ buffer(0) ]],
    constant float*         src [[ buffer(1) ]],
    device   float*         dst [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint c = gid.x, r = gid.y;
    if (r >= u.rows || c >= u.cols) return;
    uint r0 = r > 0 ? r - 1 : 0, r1 = r < u.rows - 1 ? r + 1 : u.rows - 1;
    uint c0 = c > 0 ? c - 1 : 0, c1 = c < u.cols - 1 ? c + 1 : u.cols - 1;
    float sum = 0.0f, n = 0.0f;
    for (uint rr = r0; rr <= r1; ++rr)
        for (uint cc = c0; cc <= c1; ++cc) { sum += src[rr * u.cols + cc]; n += 1.0f; }
    dst[r * u.cols + c] = sum / n;
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 6 — ss_cover_threshold   (cover = filmH > revealFloor)
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_cover_threshold(
    constant MorphUniforms& u     [[ buffer(0) ]],
    constant float*         filmH [[ buffer(1) ]],
    device   uint*          cover [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint cells = u.rows * u.cols;
    if (gid >= cells) return;
    cover[gid] = (filmH[gid] > u.revealFloor) ? 1u : 0u;
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 7 — ss_dilate_bin   (3×3 binary OR)
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_dilate_bin(
    constant MorphUniforms& u   [[ buffer(0) ]],
    constant uint*          src [[ buffer(1) ]],
    device   uint*          dst [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint c = gid.x, r = gid.y;
    if (r >= u.rows || c >= u.cols) return;
    uint r0 = r > 0 ? r - 1 : 0, r1 = r < u.rows - 1 ? r + 1 : u.rows - 1;
    uint c0 = c > 0 ? c - 1 : 0, c1 = c < u.cols - 1 ? c + 1 : u.cols - 1;
    uint on = 0;
    for (uint rr = r0; rr <= r1 && on == 0; ++rr)
        for (uint cc = c0; cc <= c1 && on == 0; ++cc)
            if (src[rr * u.cols + cc] != 0) on = 1;
    dst[r * u.cols + c] = on;
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 8 — ss_erode_bin   (3×3 binary AND)
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_erode_bin(
    constant MorphUniforms& u   [[ buffer(0) ]],
    constant uint*          src [[ buffer(1) ]],
    device   uint*          dst [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint c = gid.x, r = gid.y;
    if (r >= u.rows || c >= u.cols) return;
    uint r0 = r > 0 ? r - 1 : 0, r1 = r < u.rows - 1 ? r + 1 : u.rows - 1;
    uint c0 = c > 0 ? c - 1 : 0, c1 = c < u.cols - 1 ? c + 1 : u.cols - 1;
    uint on = 1;
    for (uint rr = r0; rr <= r1 && on == 1; ++rr)
        for (uint cc = c0; cc <= c1 && on == 1; ++cc)
            if (src[rr * u.cols + cc] == 0) on = 0;
    dst[r * u.cols + c] = on;
}

// ── Birthday-cake per-vertex colour (LINEAR) — mirrors birthdayDotColor(i). ────
static float3 birthdayDotColor(int i) {
    float h = fract(sin(float(i) * 12.9898f) * 43758.547f);
    float3 frosting = srgbToLinear(float3(0.95f, 0.93f, 0.90f));
    if (fabs(h) < 0.13f) {
        // 6-entry rainbow palette.
        float3 pal[6] = {
            float3(0.95f, 0.25f, 0.30f), float3(0.97f, 0.62f, 0.18f),
            float3(0.96f, 0.90f, 0.25f), float3(0.35f, 0.80f, 0.40f),
            float3(0.28f, 0.55f, 0.95f), float3(0.70f, 0.38f, 0.92f) };
        int k = int(fabs(fract(sin(float(i) * 78.233f) * 9123.4f)) * 6.0f) % 6;
        return srgbToLinear(pal[k]);
    }
    return frosting;
}

// ── flavorAt / flavorSRGB — flavorOrder = [choco, straw, butter, birthday]. ────
// Returns the flavour's base sRGB. fi is the stored chocoFilmFlavor index.
static float3 flavorSRGB_fromIndex(int fi) {
    // flavorAt clamps out-of-range to .chocolate (index 0).
    if      (fi == 1) return float3(0.86f, 0.26f, 0.40f);   // strawberry
    else if (fi == 2) return float3(0.80f, 0.50f, 0.16f);   // butterscotch
    else if (fi == 3) return float3(0.95f, 0.93f, 0.90f);   // birthday cake (white)
    else              return float3(0.16f, 0.075f, 0.040f); // chocolate (default)
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel 9 — ss_coat_displace
// One thread per shell vertex (rows×cols). Mirrors the vertex loop of
// displaceChocoCoat() EXACTLY, including the edge-taper / hidden / normal /
// flavour-colour branches.
//   depthSmooth = final blurred depth drape
//   cover       = closed coverage mask  (chocoCover after close)
//   inner       = interior mask         (chocoCoverTmp after extra erode)
//   filmHSmooth = blurred film thickness
// ──────────────────────────────────────────────────────────────────────────────
kernel void ss_coat_displace(
    constant CoatUniforms&         u           [[ buffer(0) ]],
    constant float*                depthSmooth [[ buffer(1) ]],
    constant float*                filmH       [[ buffer(2) ]],
    constant float*                filmHSmooth [[ buffer(3) ]],
    constant uint*                 cover       [[ buffer(4) ]],   // closed mask
    constant uint*                 inner       [[ buffer(5) ]],   // interior (eroded) mask
    constant int*                  filmFlavor  [[ buffer(6) ]],
    device   IlluminatoramaVertex* out         [[ buffer(7) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint c = gid.x, r = gid.y;
    if (r >= u.rows || c >= u.cols) return;

    uint  cols = u.cols, rows = u.rows;
    uint  idx  = r * cols + c;
    float y    = u.topY - (float(r) + u.subRow) * u.rowH;

    // dir = chocoFrontDir = (0,0,1). alpha = -maxWedge + c*dAlpha.
    //   wdx =  ca*dir.x + sa*dir.z =  sa
    //   wdz = -sa*dir.x + ca*dir.z =  ca
    float alpha = -u.maxWedge + float(c) * u.dAlpha;
    float ca = cos(alpha), sa = sin(alpha);
    float wdx = sa;
    float wdz = ca;

    if (cover[idx] == 0) {
        // Tuck inside the rope mesh → occluded from the front.
        out[idx].position = float3(wdx * u.hiddenR, y, wdz * u.hiddenR);
        out[idx].normal   = float3(wdx, 0.0f, wdz);
        return;
    }

    float grooveFloorR = u.coilRadius;
    bool  isInner = (inner[idx] == 1);
    float hEff    = isInner ? max(filmH[idx], filmHSmooth[idx]) : 0.0f;
    float surfR   = max(depthSmooth[idx], grooveFloorR);
    float radius  = surfR + (isInner ? (u.gap + hEff) : 0.0f);

    out[idx].position = float3(wdx * radius, y, wdz * radius);

    // Finite-difference gradient normal (mirrors the CPU cross-product).
    uint rp = min(rows - 1, r + 1), rm = (r > 0) ? r - 1 : 0;
    uint cp = min(cols - 1, c + 1), cm = (c > 0) ? c - 1 : 0;
    float dRdr = (depthSmooth[rp * cols + c] - depthSmooth[rm * cols + c]) / float(rp - rm);
    float dRdc = (depthSmooth[r * cols + cp] - depthSmooth[r * cols + cm]) / float(cp - cm);
    float3 tRow = float3(wdx * dRdr, -u.rowH, wdz * dRdr);
    float3 tCol = float3(wdz * u.dAlpha * radius + wdx * dRdc,
                         0.0f,
                         -wdx * u.dAlpha * radius + wdz * dRdc);
    float3 nRaw = cross(tRow, tCol);
    float  nLen = length(nRaw);
    out[idx].normal = (nLen > 1e-6f) ? (nRaw / nLen) : float3(wdx, 0.0f, wdz);

    // Per-cell flavour colour.
    int fi = filmFlavor[idx];
    if (u.debug != 0) {
        if (!isInner)                       out[idx].color = float4(1, 1, 0, 1); // boundary
        else if (filmH[idx] > u.revealFloor) out[idx].color = float4(1, 0, 1, 1); // real coat
        else                                out[idx].color = float4(0, 1, 1, 1); // hole-filled
    } else if (fi == 3) {                    // birthdayCake → per-vertex dots
        out[idx].color = float4(birthdayDotColor(int(idx)), 1.0f);
    } else {
        float3 lin = srgbToLinear(flavorSRGB_fromIndex(fi));
        out[idx].color = float4(lin.x, lin.y, lin.z, 1.0f);
    }
}
