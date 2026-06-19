#include <metal_stdlib>
using namespace metal;

// ── SHALLOW WATER EQUATIONS ─────────────────────────────────────────────────
//
// Real fluid simulation for the Hotdog Waterslide. The grid is path-local:
// the x-axis is arc-length s along the slide's closed loop, the y-axis is
// cross-section width w across the half-pipe floor. Each cell stores
// conserved variables (h, hu, hv) = height, arc-direction momentum, and
// width-direction momentum.
//
// Solver
// ──────
// Lax-Friedrichs flux + forward Euler in time. First order, but rock-stable
// at our spatial scales (dx_s ~16 cm, dx_w ~4 cm) and visually rich because
// the numerical viscosity in LxF gives the water a believable settling
// behaviour without explicit drag.
//
// Body forces
// ───────────
// At each arc cell i the controller pre-bakes (a_s, a_w) from the path's
// local cradle frame: a_s = -g · tangent_y (slope-driven flow downhill),
// a_w = -g · right_y (lateral tilt). Plus optional centripetal can be added
// later. These body forces are what makes the water flow downhill and
// bank — they're the whole physics-of-the-path link.
//
// Boundary conditions
// ───────────────────
// Periodic in arc (the loop is closed), reflective in width (water can't
// cross the half-pipe walls — the kernel mirrors hv at j=0 and j=Nwidth-1).
//
// Outputs to the rendering pipeline
// ─────────────────────────────────
// • `surfaceFromSWE` writes the visible surface mesh's positions + normals
//   from h directly. Surface lifts toward the path centre as h grows; normal
//   is computed from the finite-difference gradient of h, so peaks tilt
//   light toward the camera.
// • `causticsFromSWE` writes the caustic receiver mesh's per-vertex
//   brightness from the 5-point finite-difference Laplacian of h, scaled by
//   the depth-gain factor. Bright bands appear where h is locally concave
//   (rebound waves, troughs around riders); dark gaps where it's convex.
//   Real caustics from the real wave field.

struct SWECell {
    float h;     // depth
    float hu;    // arc-direction momentum  (h × u_s)
    float hv;    // width-direction momentum (h × u_w)
    float pad;   // align to 16 bytes
};

struct SWEUniforms {
    float dt;
    float dxS;
    float dxW;
    float gravity;
    float manning;
    uint  Narc;
    uint  Nwidth;
    uint  riderCount;
    float totalLength;
    float maxDepth;   // hard ceiling on water column h (m); 0 = uncapped. Was pad0.
    float pad1;
    float pad2;
};

// ── SWE advance (one substep) ────────────────────────────────────────────────

kernel void sweAdvance(
    device const SWECell  *current   [[buffer(0)]],
    device       SWECell  *next      [[buffer(1)]],
    constant float2       *bodyForce [[buffer(2)]],   // [Narc] (a_s, a_w)
    constant SWEUniforms  &u         [[buffer(3)]],
    uint2                  gid       [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= u.Narc || j >= u.Nwidth) return;

    uint idx = i * u.Nwidth + j;
    SWECell cc = current[idx];
    // Defensive: any NaN/inf in the live state from a previous tick gets
    // sanitised before it propagates through the flux divergence.
    if (!isfinite(cc.h))  cc.h  = 0.0f;
    if (!isfinite(cc.hu)) cc.hu = 0.0f;
    if (!isfinite(cc.hv)) cc.hv = 0.0f;

    // Periodic in arc, reflective walls in width.
    uint iL = (i == 0u) ? u.Narc - 1u : i - 1u;
    uint iR = (i + 1u) % u.Narc;
    uint jLi = (j == 0u) ? 0u : j - 1u;
    uint jRi = min(j + 1u, u.Nwidth - 1u);

    SWECell cL = current[iL * u.Nwidth + j];
    SWECell cR = current[iR * u.Nwidth + j];
    SWECell cD = current[i  * u.Nwidth + jLi];
    SWECell cU = current[i  * u.Nwidth + jRi];

    // Mirror cross-section momentum at the half-pipe walls so the water can't
    // flow out the sides.
    if (j == 0u)              cD.hv = -cc.hv;
    if (j == u.Nwidth - 1u)   cU.hv = -cc.hv;

    float3 Uc = float3(cc.h, cc.hu, cc.hv);
    float3 UL = float3(cL.h, cL.hu, cL.hv);
    float3 UR = float3(cR.h, cR.hu, cR.hv);
    float3 UD = float3(cD.h, cD.hu, cD.hv);
    float3 UU = float3(cU.h, cU.hu, cU.hv);

    float g = u.gravity;

    // F-flux along arc, G-flux along width.
    //   F(U) = (hu, hu²/h + ½g h², hu·hv/h)
    //   G(U) = (hv, hu·hv/h,        hv²/h + ½g h²)
    auto F = [g](float3 U) {
        float h = max(U.x, 1e-5f);
        return float3(U.y, U.y * U.y / h + 0.5f * g * U.x * U.x, U.y * U.z / h);
    };
    auto G = [g](float3 U) {
        float h = max(U.x, 1e-5f);
        return float3(U.z, U.y * U.z / h, U.z * U.z / h + 0.5f * g * U.x * U.x);
    };

    float3 Fc = F(Uc), FL = F(UL), FR = F(UR);
    float3 Gc = G(Uc), GD = G(UD), GU = G(UU);

    // Lax-Friedrichs interface flux: average of neighbour fluxes minus a
    // numerical-viscosity term proportional to the state jump. The 1D LxF
    // dissipation coefficient is `0.5 · dx/dt`. In 2D, applying that PER
    // DIRECTION sums the two contributions and over-diffuses — for a single-
    // cell perturbation on a uniform background, the central cell flips past
    // the neighbours' value in one step and the noise grows ~2× per substep
    // (h goes from 0.164 → 0.158 toward neighbours of 0.16, an overshoot of
    // 0.002 below the mean). Halving each direction so the sum matches the
    // 1D coefficient restores monotonic damping toward the local average.
    float alphaS = 0.25f * (u.dxS / u.dt);
    float alphaW = 0.25f * (u.dxW / u.dt);

    float3 fluxW = 0.5f * (FL + Fc) - alphaS * (Uc - UL);   // west  face
    float3 fluxE = 0.5f * (Fc + FR) - alphaS * (UR - Uc);   // east  face
    float3 fluxS = 0.5f * (GD + Gc) - alphaW * (Uc - UD);   // south face
    float3 fluxN = 0.5f * (Gc + GU) - alphaW * (UU - Uc);   // north face

    float3 Unew = Uc
                - (u.dt / u.dxS) * (fluxE - fluxW)
                - (u.dt / u.dxW) * (fluxN - fluxS);

    // Body forces — pre-baked from path slope + lateral tilt at this arc cell.
    float2 bf = bodyForce[i];
    Unew.y += u.dt * Uc.x * bf.x;
    Unew.z += u.dt * Uc.x * bf.y;

    // Manning-ish friction — keeps water from accelerating forever down the
    // slope, gives a settling speed.
    float h_safe = max(Unew.x, 1e-5f);
    float speed = length(float2(Unew.y, Unew.z)) / h_safe;
    float frictionFactor = 1.0f / (1.0f + u.dt * u.manning * speed);
    Unew.y *= frictionFactor;
    Unew.z *= frictionFactor;

    // h ≥ 0 — the LxF dissipation can briefly drive a dry-state cell
    // slightly negative; clamp keeps `√(g·h)` real downstream.
    Unew.x = max(Unew.x, 0.0f);

    // h ≤ maxDepth — the solver is a lidless height field, so nothing otherwise
    // stops water piling deeper than the tube it lives in (the closed-loop
    // pooling that lofted franks out the open top). Cap to the tube's holding
    // depth so neither the surface nor the riders can rise above the tube. The
    // velocity clamp below already ASSUMES this h_max for CFL; this enforces it.
    // 0 = uncapped (any other future caller that leaves maxDepth unset).
    if (u.maxDepth > 0.0f) Unew.x = min(Unew.x, u.maxDepth);

    // Velocity clamp — must stay under the CFL bound for the cross-width
    // direction. With dxW ≈ 2.5 cm and the substep dt held to ≤ 4 ms by
    // ShallowWaterSim.maxSubDt, dxW/subDt ≈ 6.25 m/s, and the gravity wave
    // speed at the CFL-bounded h_max ≈ 0.5 m is √(g·h) ≈ 2.2 m/s. So |v| <
    // 4.0 m/s leaves margin for LxF stability. The earlier 8 m/s clamp was
    // ABOVE CFL even at rest (sqrt(g·0.16) + 8 > 6.25) — riders' source
    // kicks could push v past the bound, the scheme went unstable, h
    // exploded across the grid, and probe.lift went to NaN.
    float vMax = 4.0f;
    float h_clamp = max(Unew.x, 1e-5f);
    float u_now = Unew.y / h_clamp;
    float v_now = Unew.z / h_clamp;
    u_now = clamp(u_now, -vMax, vMax);
    v_now = clamp(v_now, -vMax, vMax);
    Unew.y = u_now * h_clamp;
    Unew.z = v_now * h_clamp;

    next[idx].h   = Unew.x;
    next[idx].hu  = Unew.y;
    next[idx].hv  = Unew.z;
    next[idx].pad = 0;
}

// ── Rider source injection ───────────────────────────────────────────────────
//
// A rider produces sloshing by shoving water sideways as it passes — not
// by pressing it down. We inject LATERAL cross-section momentum (hv)
// outward from each rider's centre plus a touch of FORWARD momentum (hu)
// behind it for a wake. The walls reflect the lateral flow back across the
// channel, and the SWE solver naturally produces visible standing waves
// when the rider's source frequency interacts with the cross-section's
// reflection period.
//
// `h` itself isn't depressed — that earlier behaviour created a Gaussian
// dimple under each rider that didn't slosh and also tripped the rider-
// surface coupling (riders sampled their own depression and fell through
// the water).

kernel void sweApplyRiderSources(
    device       SWECell  *state    [[buffer(0)]],
    constant float4       *riders   [[buffer(1)]],   // (arcS, riderRadius, _, _)
    constant SWEUniforms  &u        [[buffer(2)]],
    uint2                  gid      [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= u.Narc || j >= u.Nwidth) return;

    float s_cell = (float(i) + 0.5f) * u.dxS;
    float w_cell = (float(j) + 0.5f) / float(u.Nwidth) - 0.5f;   // [-0.5, +0.5]

    float hv_kick = 0.0f;
    float hu_kick = 0.0f;
    for (uint k = 0u; k < u.riderCount; ++k) {
        float s_r = riders[k].x;
        float rad = max(riders[k].y, 0.05f);

        // Arc-wrap distance.
        float ds = s_cell - s_r;
        ds = ds - u.totalLength * round(ds / u.totalLength);

        float sigmaS = rad * 0.9f;
        float gaussS = exp(-(ds * ds) / (2.0f * sigmaS * sigmaS));
        // Cross-section footprint that fades away from the rider centre,
        // weighted by w_cell so the push is OUTWARD (sign of w_cell).
        // sin profile across width gives strong push at the rider, zero at
        // the centerline (the rider's body), and reversal at the far wall.
        float widthPush = sin(w_cell * 3.14159f);    // ~ [-π/2, +π/2] over half-pipe

        hv_kick += 1.2f * widthPush * gaussS;        // lateral slosh impulse
        hu_kick += 0.4f * sign(ds) * gaussS;         // wake (forward of rider on the +ds side)
    }

    uint idx = i * u.Nwidth + j;
    // Momentum impulses, scaled by dt at the kernel-call site (we treat the
    // accumulated kick as Δ(hv)/dt × dt = Δ(hv), so it's an immediate
    // momentum injection per source pass).
    state[idx].hv += u.dt * hv_kick;
    state[idx].hu += u.dt * hu_kick;
}

// ── Surface mesh kernel ──────────────────────────────────────────────────────

struct SurfaceUniforms {
    uint  rings;
    uint  segments;
    uint  pathSamples;
    float totalLength;
    float floorRadius;       // distance from path centre to the half-pipe floor
    float angleSpan;
    float waterScale;        // visual exaggeration of (h - baselineH) — applied
                             // only to wave activity above/below the rest level
    float baselineH;         // physical resting depth (m). The visible surface
                             // sits at floorRadius - baselineH when the water
                             // is still; waves move it from there.
    float elapsed;           // wall-clock seconds, drives the micro-normal flow
    float microNormalAmp;    // strength of high-frequency normal perturbation
    float microNormalFreq;   // spatial frequency (cycles per metre) of the noise
    float foamMaskGain;      // multiplier on foam-field → vertex-color whiteness
    float clipUpY;           // > 0 → collapse the surface ring to the centerline
                             // where the cradle up.y falls below this (inverted /
                             // steep spans can't hold an open free surface: loop
                             // top, corkscrew, splashdown dive). 0 = never clip.
    float looping;           // 1 = closed loop (wrap the last ring to the first,
                             // seamless recirculation). 0 = ONE-WAY: clamp the
                             // path-sample lookup so the final ring does NOT wrap
                             // back to the inlet — that wrap is what stretched the
                             // splashdown surface into a vertical spike (Descent).
    float pad2;
};

// Cheap value noise — gradient via 4-sample stencil. Cheap enough to evaluate
// per-vertex; gives us anisotropic streak-able micro-detail without a texture
// (shader modifiers can't reliably bind textures on this codepath — see
// caustic_via_emission_slot.md / shader_modifier_uniform_types.md).
static inline float vnoiseHash(float2 p) {
    float n = sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453f;
    return fract(n);
}
static inline float vnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float a = vnoiseHash(i);
    float b = vnoiseHash(i + float2(1.0f, 0.0f));
    float c = vnoiseHash(i + float2(0.0f, 1.0f));
    float d = vnoiseHash(i + float2(1.0f, 1.0f));
    float2 sm = f * f * (3.0f - 2.0f * f);
    return mix(mix(a, b, sm.x), mix(c, d, sm.x), sm.y);
}

kernel void surfaceFromSWE(
    device   float4              *positions    [[buffer(0)]],
    device   float4              *normals      [[buffer(1)]],
    constant float4              *pathPos      [[buffer(2)]],
    constant float4              *pathRight    [[buffer(3)]],
    constant float4              *pathUp       [[buffer(4)]],
    device const SWECell         *state        [[buffer(5)]],
    constant SurfaceUniforms     &u            [[buffer(6)]],
    device const float           *foamField    [[buffer(7)]],
    device   float4              *vertexColors [[buffer(8)]],
    uint                          id           [[thread_position_in_grid]]
) {
    uint ring = id / u.segments;
    uint seg  = id % u.segments;
    if (ring >= u.rings) return;

    float ringF = float(ring) / float(u.rings - 1);
    float fIdx  = ringF * float(u.pathSamples);
    uint  i0, i1;
    if (u.looping != 0.0f) {
        i0 = uint(floor(fIdx)) % u.pathSamples;
        i1 = (i0 + 1u) % u.pathSamples;
    } else {
        // One-way: clamp (no wrap) so the final ring stays at the splashdown end
        // instead of stretching back up to the inlet (the vertical-spike bug).
        i0 = min(uint(floor(fIdx)), u.pathSamples - 1u);
        i1 = min(i0 + 1u, u.pathSamples - 1u);
    }
    float frac  = fIdx - floor(fIdx);

    float3 pos   = mix(pathPos[i0].xyz,   pathPos[i1].xyz,   frac);
    float3 right = normalize(mix(pathRight[i0].xyz, pathRight[i1].xyz, frac));
    float3 up    = normalize(mix(pathUp[i0].xyz,    pathUp[i1].xyz,    frac));

    // Inverted / steep-span clip: where the channel tilts past the threshold an
    // open free surface is unphysical (it would render on the loop ceiling or
    // spike on the splashdown dive). Collapse the whole ring to the centerline
    // (coincident verts → zero-area triangles → invisible) with alpha 0.
    if (u.clipUpY > 0.0f && up.y < u.clipUpY) {
        positions[id]    = float4(pos, 1.0f);
        normals[id]      = float4(up, 0.0f);
        vertexColors[id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // 1:1 cell → vertex map (rings == Narc, segments == Nwidth).
    uint cell = ring * u.segments + seg;
    float h = state[cell].h;
    if (!isfinite(h)) h = u.baselineH;
    h = max(h, 1e-3f);
    // Cap h below the half-pipe rim so the surface never renders above the
    // open top. The rim sits at cradle-y ≈ 0.71·floorRadius (cos(π/4)·R for
    // an angleSpan opening at ±π/4 from up), so h_max ≈ 1.71·floorRadius
    // would touch the rim. Cap at 1.6·floorRadius to leave visual margin.
    h = min(h, u.floorRadius * 1.6f);

    // Horizontal-slab water surface. Each vertex sits at:
    //   • cradle-y = -floorRadius + h  (water level above the tube floor)
    //   • cradle-x = uFrac · chord(h), where chord(h) is the tube's
    //     horizontal cross-section width at the water's vertical level
    //     (the standard circle-chord formula).
    // This makes the visible water naturally WIDER as `h` rises — exactly
    // what real water does — so the deep pool on the flat straightaway
    // reads as a wide filled trough, while the calm helices read at
    // baseline width. Replaces the earlier cylindrical-strip model that
    // shrank the visible surface inward as h grew (counterintuitive: deep
    // water rendering as a thin ribbon along the tube floor).
    float y_cradle = -u.floorRadius + h;
    float chord_hw_sq = u.floorRadius * u.floorRadius - y_cradle * y_cradle;
    float chord_half_width = sqrt(max(chord_hw_sq, 1e-4f));

    float uFrac = (float(seg) / float(u.segments - 1)) - 0.5f;
    float x_cradle = 2.0f * uFrac * chord_half_width;

    float3 worldPos = pos + up * y_cradle + right * x_cradle;

    // ── Coarse normal from finite-difference gradient of h ──────────────
    uint ringL = (ring == 0u) ? u.rings - 1u : ring - 1u;
    uint ringR = (ring + 1u) % u.rings;
    uint segL  = (seg == 0u) ? 0u : seg - 1u;
    uint segR  = min(seg + 1u, u.segments - 1u);

    float hL = state[ringL * u.segments + seg].h;
    float hR = state[ringR * u.segments + seg].h;
    float hD = state[ring * u.segments + segL].h;
    float hU = state[ring * u.segments + segR].h;

    float dh_ds = 0.5f * (hR - hL);
    float dh_du = 0.5f * (hU - hD);

    float3 tangent  = normalize(cross(right, up));

    // Horizontal slab: surface normal is `up` plus a tilt from the local
    // h gradient (real water tilts toward higher-h neighbours).
    float3 N = up
             + tangent * (-dh_ds * u.waterScale * 8.0f)
             + right   * (-dh_du * u.waterScale * 8.0f);

    // ── Micro-normal — flow-mapped procedural noise (2 octaves) ──────
    // Sample noise on two scales, take a finite-difference gradient on each,
    // sum into a tangent-space perturbation. UVs are offset by the local
    // water velocity × time so highlights streak along the flow direction —
    // the canonical "fast water" specular tell. Two octaves at 1× and 2.7×
    // give us both broad ripples and a sub-cm sparkle scale, so the surface
    // never reads as flat plastic at any zoom level.
    SWECell c_here = state[cell];
    float h_here = max(c_here.h, 1e-4f);
    float2 vel = float2(c_here.hu / h_here, c_here.hv / h_here);
    // Project worldPos onto the (tangent, width) plane to get a 2D UV that's
    // continuous along the slide — using world XZ would have seams at the
    // helices' winding planes.
    float2 uvBase = float2(dot(worldPos, tangent), dot(worldPos, right));

    float2 uv1 = uvBase * u.microNormalFreq + vel * u.elapsed * 0.45f;
    float n0_1 = vnoise(uv1);
    float dN_dS_1 = (vnoise(uv1 + float2(0.07f, 0.0f)) - n0_1) / 0.07f;
    float dN_dW_1 = (vnoise(uv1 + float2(0.0f, 0.07f)) - n0_1) / 0.07f;

    float2 uv2 = uvBase * (u.microNormalFreq * 2.7f) + vel * u.elapsed * 0.9f
               + float2(11.3f, 7.7f);
    float n0_2 = vnoise(uv2);
    float dN_dS_2 = (vnoise(uv2 + float2(0.04f, 0.0f)) - n0_2) / 0.04f;
    float dN_dW_2 = (vnoise(uv2 + float2(0.0f, 0.04f)) - n0_2) / 0.04f;

    float dN_dS = dN_dS_1 + 0.55f * dN_dS_2;
    float dN_dW = dN_dW_1 + 0.55f * dN_dW_2;
    N += tangent * (-dN_dS * u.microNormalAmp)
       + right   * (-dN_dW * u.microNormalAmp);

    N = normalize(N);

    // ── Vertex color — depth-driven absorption + foam whitening ─────────
    // Beer-Lambert read: shallow water at the trough edges is a light teal
    // (you see through it to the slide); as the column deepens it saturates to
    // a dark blue-green. This depth gradient is the single biggest cue that
    // the surface has VOLUME rather than being a flat decal.
    float depthT    = clamp(h / (u.floorRadius * 0.85f), 0.0f, 1.0f);
    float3 shallow  = float3(0.22f, 0.62f, 0.66f);   // bright clear teal at thin edges
    float3 deep     = float3(0.04f, 0.30f, 0.46f);   // saturated pool-blue when deep
    float3 baseWater = mix(shallow, deep, depthT * depthT);
    float foamRaw = (cell < u.rings * u.segments) ? foamField[cell] : 0.0f;
    float foamMask = clamp(foamRaw * u.foamMaskGain, 0.0f, 1.0f);
    // Smooth the foam-to-white curve so a low foam value still tints
    // subtly (capillary-foam look) before saturating to full whitecap.
    float foamSmooth = smoothstep(0.05f, 0.85f, foamMask);
    float3 foamWhite = float3(0.95f, 0.98f, 1.00f);
    float3 vColor    = mix(baseWater, foamWhite, foamSmooth);

    positions[id]    = float4(worldPos, 1.0f);
    normals[id]      = float4(N, 0.0f);
    vertexColors[id] = float4(vColor, 1.0f);
}

// ── Foam scalar field — per-cell whiteness, advected by velocity ────────────
//
// Tracks where foam is "currently visible" on the water surface as a smooth
// scalar in [0, 1] per SWE cell. Two pieces:
//
//   • Source per tick is `|curl(v)| + α·|v|² + β·|∇h|·|v|` — the standard
//     whitecap-energy proxy. Cells with high shear (curl), high speed, or
//     fast water on a steep slope all accumulate foam.
//
//   • The existing foam state is advected semi-Lagrangian one tick: lookup
//     the upstream value at `(s - v_s·dt, w - v_w·dt)` and decay it by
//     `exp(-decayRate·dt)`. That's what makes the foam streak along the
//     flow direction instead of pulsing in place.
//
// Output is read by `surfaceFromSWE` as a per-vertex whiteness mask written
// into the surface mesh's vertex color buffer.

struct SWEFoamFieldUniforms {
    uint  Narc;
    uint  Nwidth;
    float dt;
    float dxS;
    float dxW;
    float decayRate;          // 1/s — foam fades over ~1.5 s at 0.7
    float curlGain;
    float speedGain;          // multiplied by |v|² → strong whitecap signal
    float slopeGain;          // multiplied by |∇h|·|v|
    float spawnSpeedThresh;   // |v| below this contributes nothing
    float maxFoam;            // hard cap so a hot cell can't blow out
    float pad0;
};

static inline float bilerpFoam(
    device const float *foam,
    uint Narc, uint Nwidth,
    float sIdx, float wIdx
) {
    // Periodic in s, clamp in w.
    int Narci = int(Narc);
    int s0 = int(floor(sIdx));
    int s1 = s0 + 1;
    s0 = ((s0 % Narci) + Narci) % Narci;
    s1 = ((s1 % Narci) + Narci) % Narci;
    int w0 = clamp(int(floor(wIdx)), 0, int(Nwidth) - 1);
    int w1 = clamp(w0 + 1, 0, int(Nwidth) - 1);
    float fs = fract(sIdx);
    float fw = fract(wIdx);
    if (fs < 0.0f) fs += 1.0f;
    if (fw < 0.0f) fw += 1.0f;
    float a = foam[uint(s0) * Nwidth + uint(w0)];
    float b = foam[uint(s1) * Nwidth + uint(w0)];
    float c = foam[uint(s0) * Nwidth + uint(w1)];
    float d = foam[uint(s1) * Nwidth + uint(w1)];
    return mix(mix(a, b, fs), mix(c, d, fs), fw);
}

kernel void sweFoamFieldAdvect(
    device const SWECell                *state   [[buffer(0)]],
    device const float                  *foamIn  [[buffer(1)]],
    device       float                  *foamOut [[buffer(2)]],
    constant SWEFoamFieldUniforms       &u       [[buffer(3)]],
    uint2                                gid     [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= u.Narc || j >= u.Nwidth) return;
    uint idx = i * u.Nwidth + j;
    SWECell c = state[idx];
    float h = max(c.h, 1e-4f);
    float vel_s = c.hu / h;
    float vel_w = c.hv / h;
    float speed = length(float2(vel_s, vel_w));

    // Semi-Lagrangian upstream lookup. Trace particle position backward by
    // (vel · dt) in cell-index space, then bilerp the previous foam value
    // there. This is what makes the foam streak along the flow.
    float sIdx = float(i) + 0.5f - vel_s * u.dt / u.dxS;
    float wIdx = float(j) + 0.5f - vel_w * u.dt / u.dxW;
    float prev = bilerpFoam(foamIn, u.Narc, u.Nwidth, sIdx - 0.5f, wIdx - 0.5f);

    // Whitecap source: curl + speed² + slope·speed.
    uint iL = (i == 0u) ? u.Narc - 1u : i - 1u;
    uint iR = (i + 1u) % u.Narc;
    uint jL = (j == 0u) ? 0u : j - 1u;
    uint jR = min(j + 1u, u.Nwidth - 1u);
    float h_iL = max(state[iL * u.Nwidth + j].h, 1e-4f);
    float h_iR = max(state[iR * u.Nwidth + j].h, 1e-4f);
    float h_jL = max(state[i  * u.Nwidth + jL].h, 1e-4f);
    float h_jR = max(state[i  * u.Nwidth + jR].h, 1e-4f);
    float u_jL = state[i  * u.Nwidth + jL].hu / h_jL;
    float u_jR = state[i  * u.Nwidth + jR].hu / h_jR;
    float v_iL = state[iL * u.Nwidth + j].hv  / h_iL;
    float v_iR = state[iR * u.Nwidth + j].hv  / h_iR;
    float curl_z = 0.5f * (v_iR - v_iL) / u.dxS
                 - 0.5f * (u_jR - u_jL) / u.dxW;
    float dh_ds = 0.5f * (state[iR * u.Nwidth + j].h - state[iL * u.Nwidth + j].h) / u.dxS;
    float dh_dw = 0.5f * (state[i  * u.Nwidth + jR].h - state[i  * u.Nwidth + jL].h) / u.dxW;
    float slope = length(float2(dh_ds, dh_dw));

    float speedExcess = max(0.0f, speed - u.spawnSpeedThresh);
    float src = (abs(curl_z) * u.curlGain
                 + speedExcess * speedExcess * u.speedGain
                 + slope * speed * u.slopeGain) * u.dt;

    float decayed = prev * exp(-u.decayRate * u.dt);
    float total   = clamp(decayed + src, 0.0f, u.maxFoam);
    foamOut[idx] = total;
}

// ── Rider probe — per-rider readback of (lift, lateral, vTan, vCross, slopeS, slopeW) ──
//
// One thread per rider. Replaces the CPU loop that used to scan ~12 cells
// per rider every tick to compute the visible water state at the rider's
// arc-length. The GPU writes into a small (maxRiders × float4×2) buffer
// the controller reads next frame for posing — same one-tick lag the rest
// of the rendering pipeline already absorbs, and pushes the per-cell loop
// off the main thread.

struct RiderProbe {
    float4 liftLateral;   // x = lift (m, signed offset from path centerline)
                          // y = lateral (m, signed offset along cradle-right)
                          // z = unused
                          // w = unused
    float4 velSlope;      // x = vTan   (m/s, water velocity along path tangent at rider)
                          // y = vCross (m/s, water velocity across cradle width)
                          // z = slopeS (∂h/∂s — wave gradient along arc)
                          // w = slopeW (∂h/∂w — wave gradient across width)
};

struct RiderProbeUniforms {
    uint  Narc;
    uint  Nwidth;
    uint  riderCount;
    float dxS;
    float dxW;
    float totalLength;
    float floorRadius;
    float baselineH;
    float waterScale;
    float riderFollowFraction;
    float lateralScale;
    float lateralLimit;
};

kernel void sweSampleRiders(
    device const SWECell        *state   [[buffer(0)]],
    constant float4             *riders  [[buffer(1)]],   // (arcS, riderRadius, lateralUFrac, _)
    device   RiderProbe         *probes  [[buffer(2)]],
    constant RiderProbeUniforms &u       [[buffer(3)]],
    uint                         id      [[thread_position_in_grid]]
) {
    if (id >= u.riderCount) return;

    float s = riders[id].x;
    // Wrap into [0, totalLength).
    s = s - u.totalLength * floor(s / u.totalLength);
    uint i = uint(s / u.dxS) % u.Narc;
    uint iL = (i == 0u) ? u.Narc - 1u : i - 1u;
    uint iR = (i + 1u) % u.Narc;

    // Lift — average h around the channel midline, then map to visible
    // surface offset using the same formula sampleRiderState used on CPU.
    uint centre = u.Nwidth / 2u;
    uint radius = max(1u, u.Nwidth / 8u);
    uint lo = (centre > radius) ? centre - radius : 0u;
    uint hi = min(u.Nwidth - 1u, centre + radius);

    float liftSum = 0.0f;
    float vTanSum = 0.0f;
    uint  liftCount = 0u;
    for (uint j = lo; j <= hi; ++j) {
        SWECell c = state[i * u.Nwidth + j];
        liftSum += c.h;
        float h_safe = max(c.h, 1e-4f);
        vTanSum += c.hu / h_safe;
        liftCount += 1u;
    }
    float hAvg = liftSum / float(liftCount);
    float vTan = vTanSum / float(liftCount);

    // Rider sits at the resting water level (constant) so deep-water
    // sections submerge the dog beneath the visible surface rather than
    // lifting it past the half-pipe rim. Tiny bob retained for visual
    // life on calm sections.
    float restingSurface = -(u.floorRadius - u.baselineH);
    float dhForLift = clamp(hAvg - u.baselineH, -0.05f, 0.05f);
    float waveLift = dhForLift * u.waterScale * u.riderFollowFraction;
    float lift = restingSurface + waveLift;

    // Lateral — h-weighted COM across the width.
    float wSum = 0.0f;
    float hSum = 0.0f;
    float vCrossSum = 0.0f;
    float vCrossWeight = 0.0f;
    for (uint j = 0u; j < u.Nwidth; ++j) {
        SWECell c = state[i * u.Nwidth + j];
        float h = max(c.h, 0.0f);
        wSum += h * (float(j) + 0.5f);
        hSum += h;
        float h_safe = max(c.h, 1e-4f);
        vCrossSum    += (c.hv / h_safe) * h;
        vCrossWeight += h;
    }
    float midline = float(u.Nwidth) * 0.5f;
    float comOffsetCells = (wSum / max(hSum, 1e-5f)) - midline;
    float lateralRaw = comOffsetCells * u.dxW * u.lateralScale;
    float lateral = clamp(lateralRaw, -u.lateralLimit, u.lateralLimit);

    // h-weighted average cross-flow velocity at the rider's arc cell.
    float vCross = vCrossSum / max(vCrossWeight, 1e-5f);

    // Slope of free surface at rider — used for pitch (along arc) and roll/
    // yaw (across width). Cell-centred averages so a single noisy neighbour
    // doesn't swing the spine.
    float hL = 0.0f, hR = 0.0f, hD = 0.0f, hU = 0.0f;
    uint mid = centre;
    uint midL = (mid > 0u) ? mid - 1u : 0u;
    uint midR = min(mid + 1u, u.Nwidth - 1u);
    hL = state[iL * u.Nwidth + mid].h;
    hR = state[iR * u.Nwidth + mid].h;
    hD = state[i  * u.Nwidth + midL].h;
    hU = state[i  * u.Nwidth + midR].h;
    float slopeS = 0.5f * (hR - hL) / u.dxS;
    float slopeW = 0.5f * (hU - hD) / u.dxW;

    // NaN-sanitise before publishing. Even one NaN read by the controller
    // would propagate to rider.s on CPU and teleport the dog. The SWE
    // advance kernel already sanitises h, but defence-in-depth here costs
    // nothing.
    if (!isfinite(lift))    lift    = 0.0f;
    if (!isfinite(lateral)) lateral = 0.0f;
    if (!isfinite(vTan))    vTan    = 0.0f;
    if (!isfinite(vCross))  vCross  = 0.0f;
    if (!isfinite(slopeS))  slopeS  = 0.0f;
    if (!isfinite(slopeW))  slopeW  = 0.0f;
    probes[id].liftLateral = float4(lift, lateral, 0.0f, 0.0f);
    probes[id].velSlope    = float4(vTan, vCross, slopeS, slopeW);
}

// ── Rider spine pose — kinematic placement of one rider's PBD chain ─────
//
// Replaces the Swift `placeAlong` loop. One dispatch per rider (spineCount
// threads); each thread computes its own spine particle's world-space
// position from the rider's central arc-length, the path frame at the
// particle's offset along arc, and the SWE-derived (lift, lateral, pitch,
// yaw) bend that comes from the rider's current water probe. Writes the
// PBDParticle's `positionAndInvMass.xyz` AND the matching `prevPositionAndPad.xyz`
// so the verlet integrator stays a no-op (invMass is 0 for these kinematic
// chains).
//
// pitchSlope and yawSlope are pre-clamped on the CPU side to ±tan(maxAngle)
// so a transient SWE shockwave can't fold the spine into a Z; the kernel
// applies them directly without further clamping.

struct PBDParticleGPU {
    float4 positionAndInvMass;
    float4 prevPositionAndPad;
};

struct RiderPoseUniforms {
    float centerS;
    float segLen;
    float lift;
    float lateral;
    float pitchSlope;
    float yawSlope;
    uint  spineCount;
    uint  pathSamples;
    float totalLength;
    float endCap;
    // Max radial distance (m) from the path centerline that any spine
    // particle is allowed to sit at. Caller sets this to
    // `tubeRadius − riderRadius − margin` so the rider's tube-mesh wall
    // never pokes through the slide's interior wall, regardless of how
    // deep the SWE wants to drop the rider.
    float maxSpineOffset;
    float pad1;
};

kernel void sweRiderSpinePose(
    device   PBDParticleGPU      *particles [[buffer(0)]],
    constant float4              *pathPos   [[buffer(1)]],
    constant float4              *pathRight [[buffer(2)]],
    constant float4              *pathUp    [[buffer(3)]],
    constant RiderPoseUniforms   &u         [[buffer(4)]],
    uint                          i         [[thread_position_in_grid]]
) {
    if (i >= u.spineCount) return;

    // Spine offset in body-local arc-length (centerIdx is the middle particle).
    float halfIdx = 0.5f * float(u.spineCount - 1u);
    float offsetS = (float(i) - halfIdx) * u.segLen;

    // Sample path at this particle's arc-length, wrapping at totalLength.
    float s = u.centerS + offsetS;
    s = s - u.totalLength * floor(s / u.totalLength);
    float sFrac = s / u.totalLength;
    float fIdx  = sFrac * float(u.pathSamples);
    uint  i0    = uint(floor(fIdx)) % u.pathSamples;
    uint  i1    = (i0 + 1u) % u.pathSamples;
    float frac  = fIdx - floor(fIdx);

    float3 pos   = mix(pathPos[i0].xyz,   pathPos[i1].xyz,   frac);
    float3 right = normalize(mix(pathRight[i0].xyz, pathRight[i1].xyz, frac));
    float3 up    = normalize(mix(pathUp[i0].xyz,    pathUp[i1].xyz,    frac));

    float pitchOffset = clamp(u.pitchSlope * offsetS, -u.endCap, u.endCap);
    float yawOffset   = clamp(u.yawSlope   * offsetS, -u.endCap, u.endCap);
    float upTerm = u.lift    + pitchOffset;
    float rgTerm = u.lateral + yawOffset;
    if (!isfinite(upTerm)) upTerm = 0.0f;
    if (!isfinite(rgTerm)) rgTerm = 0.0f;
    // Constrain the spine particle's cradle-frame offset to a disk of
    // radius `maxSpineOffset` around the path centerline so the rider's
    // tube mesh stays inside the slide wall. Without this, a deep resting
    // water level (lift ≈ -0.374 m with a tubeRadius of 0.55 m) plus a
    // 0.26 m rider radius pushes the rider's belly through the tube floor
    // — visible as hot dogs floating BELOW the slide's underside on the
    // outside of curves.
    float mag = length(float2(upTerm, rgTerm));
    if (mag > u.maxSpineOffset && mag > 1e-5f) {
        float scale = u.maxSpineOffset / mag;
        upTerm *= scale;
        rgTerm *= scale;
    }
    float3 world = pos + up * upTerm + right * rgTerm;
    if (!isfinite(world.x) || !isfinite(world.y) || !isfinite(world.z)) {
        world = pos;
    }

    // Preserve invMass (0 = pinned/kinematic) and the pad slot.
    float invMass = particles[i].positionAndInvMass.w;
    float padW    = particles[i].prevPositionAndPad.w;
    particles[i].positionAndInvMass = float4(world, invMass);
    particles[i].prevPositionAndPad = float4(world, padW);
}

// ── Caustics from SWE state ──────────────────────────────────────────────────

struct CausticsUniforms {
    uint  rings;
    uint  segments;
    uint  pathSamples;
    float totalLength;
    float floorRadius;
    float angleSpan;
    float causticGain;
    float baseline;
    uint  riderCount;
    float shadowStrength;
    float shadowHalfArcCells;
    float shadowHalfWidthCells;
};

kernel void causticsFromSWE(
    device   float4              *positions [[buffer(0)]],
    device   float4              *colours   [[buffer(1)]],
    constant float4              *pathPos   [[buffer(2)]],
    constant float4              *pathRight [[buffer(3)]],
    constant float4              *pathUp    [[buffer(4)]],
    device const SWECell         *state     [[buffer(5)]],
    constant CausticsUniforms    &u         [[buffer(6)]],
    constant float4              *riders    [[buffer(7)]],   // (arcS, radius, lateralUFrac, _)
    uint                          id        [[thread_position_in_grid]]
) {
    uint ring = id / u.segments;
    uint seg  = id % u.segments;
    if (ring >= u.rings) return;

    float ringF = float(ring) / float(u.rings - 1);
    float fIdx  = ringF * float(u.pathSamples);
    uint  i0    = uint(floor(fIdx)) % u.pathSamples;
    uint  i1    = (i0 + 1u) % u.pathSamples;
    float frac  = fIdx - floor(fIdx);

    float3 pos   = mix(pathPos[i0].xyz,   pathPos[i1].xyz,   frac);
    float3 right = normalize(mix(pathRight[i0].xyz, pathRight[i1].xyz, frac));
    float3 up    = normalize(mix(pathUp[i0].xyz,    pathUp[i1].xyz,    frac));

    float uFrac = (float(seg) / float(u.segments - 1)) - 0.5f;
    float angle = M_PI_F + uFrac * u.angleSpan;
    float3 outward = up * cos(angle) + right * sin(angle);
    float3 floorPos = pos + outward * u.floorRadius;

    // 5-point Laplacian of h on the (rings, segments) grid.
    uint ringL = (ring == 0u) ? u.rings - 1u : ring - 1u;
    uint ringR = (ring + 1u) % u.rings;
    uint segL  = (seg == 0u) ? 0u : seg - 1u;
    uint segR  = min(seg + 1u, u.segments - 1u);

    float h_cc = state[ring  * u.segments + seg].h;
    float h_L  = state[ringL * u.segments + seg].h;
    float h_R  = state[ringR * u.segments + seg].h;
    float h_D  = state[ring  * u.segments + segL].h;
    float h_U  = state[ring  * u.segments + segR].h;

    float laplacian = (h_L + h_R + h_D + h_U) - 4.0f * h_cc;

    // Concave surface (∇²h < 0) → bright caustic band; convex → dark.
    float caustic = max(0.0f, -laplacian) * u.causticGain + u.baseline;
    caustic = clamp(caustic, 0.0f, 2.0f);

    // Rider silhouette shadow. Each rider blocks sunlight, so the floor
    // directly under it gets darkened by an elliptical hot-dog footprint
    // centred at the rider's (arcCell, segCell). Falloff is `(1 - r²)`
    // inside the ellipse so the shadow has soft edges that don't look
    // like a hard-painted decal.
    float shadow = 0.0f;
    float ringF_self = float(ring);
    float segF_self  = float(seg);
    for (uint k = 0u; k < u.riderCount; ++k) {
        float r_s   = riders[k].x;
        float r_uF  = riders[k].z;            // [-0.5, +0.5]

        // Rider's continuous (ring, seg) index in this grid.
        float ringF_r = (r_s / u.totalLength) * float(u.rings);
        float segF_r  = (r_uF + 0.5f) * float(u.segments - 1);

        // Arc distance with wrap (the loop is periodic in arc).
        float dRing = ringF_self - ringF_r;
        dRing = dRing - float(u.rings) * round(dRing / float(u.rings));
        float dSeg  = segF_self - segF_r;

        float arcRatio   = dRing / u.shadowHalfArcCells;
        float widthRatio = dSeg  / u.shadowHalfWidthCells;
        float rSq = arcRatio * arcRatio + widthRatio * widthRatio;
        if (rSq < 1.0f) {
            shadow += (1.0f - rSq) * u.shadowStrength;
        }
    }
    // Apply shadow as a multiplicative attenuation so dark regions go to
    // (close to) zero brightness — visible as the dog's outline darkening
    // the dappled-light pattern on the tube floor.
    float shadowMask = clamp(1.0f - shadow, 0.0f, 1.0f);
    caustic *= shadowMask;

    // Cool tint — water filters out red, caustics under it pick up the
    // green-blue end of the spectrum.
    float3 tint = float3(0.74f, 0.95f, 1.06f) * caustic;

    positions[id] = float4(floorPos, 1.0f);
    colours[id]   = float4(tint, 1.0f);
}

// ── Surface pack: float4 → packed_float3 ──────────────────────────────────────
// `surfaceFromSWE` writes positions/normals as float4 (stride 16) for the
// SceneKit SCNGeometrySource path. The native Illuminatorama path feeds the
// surface through `registerGPUMesh`, whose repack kernel (`illumi_repack_pos_norm`)
// reads `packed_float3` (stride 12). This pass re-packs the live surface buffers
// into packed_float3 so the native mesh reads correctly. Same trick as
// `thinFilmWriteCoat`.
kernel void swePackSurface(device       packed_float3 *outPos [[buffer(0)]],
                           device       packed_float3 *outNrm [[buffer(1)]],
                           device const float4         *inPos  [[buffer(2)]],
                           device const float4         *inNrm  [[buffer(3)]],
                           constant     uint           &count  [[buffer(4)]],
                           uint gid [[thread_position_in_grid]])
{
    if (gid >= count) { return; }
    outPos[gid] = packed_float3(inPos[gid].xyz);
    outNrm[gid] = packed_float3(inNrm[gid].xyz);
}
