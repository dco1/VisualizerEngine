#include <metal_stdlib>
using namespace metal;

// ── PBDField.metal ────────────────────────────────────────────────────────────
//
// Compute kernels specific to PBDFieldSolver — the multi-chain shared-buffer
// solver used by grass (and, in the future, fur / kelp / hair).
//
// PBDFieldSolver REUSES the per-particle / per-constraint kernels from
// PBD.metal directly (pbdIntegrate, pbdConstraint, pbdFloorCollide) because
// those kernels are already global-by-id and don't care whether the buffer
// holds one hot-dog spine or 10 000 grass blades. The kernels here add what
// the field case needs and the chain case doesn't:
//
//   1. pbdFieldForces — per-particle wind sampled from a procedural 2D noise
//      field by world XZ + a soft restoring spring toward each particle's
//      rest position (so a wind-bent blade returns to vertical when the
//      gust passes).
//
//   2. pbdGrassExpand — expand spine particles into a camera-facing ribbon.
//      One thread per particle; writes 2 vertices + 2 normals (left / right
//      edges of the blade) per particle. No tube cap rings, no Catmull-Rom —
//      grass is a flat strip, not a cylinder.
//
// Struct layouts MUST match the Swift mirrors in PBDFieldSolver.swift and
// GrassRibbonRenderer.swift. Same ALIGNMENT RULE as elsewhere — no bare
// float3 in shared structs.

struct PBDParticle {
    float4 positionAndInvMass;  // xyz = position, w = invMass (0 = pinned)
    float4 prevPositionAndPad;  // xyz = prevPosition, w = unused
};

struct PBDFieldForcesUniforms {
    uint  particleCount;
    float dt;
    float time;
    float windAmp;        // metres/s² peak acceleration from wind
    float windFreq;       // spatial frequency of the wind noise (1/m)
    float windScroll;     // metres/s the wind pattern drifts in dirX/dirZ
    float windDirX;
    float windDirZ;
    float springK;        // spring stiffness for rest-position restoring force (1/s²)
    float springDamp;     // [0,1] velocity damping toward zero each substep
};

struct GrassExpandUniforms {
    uint   bladeCount;
    uint   particlesPerBlade;
    float  baseHalfWidth;    // half-width at the root (metres)
    float  tipHalfWidthFrac; // tip half-width as a fraction of base (0…1)
    float4 cameraPos;        // xyz = world camera position
};

// ── Procedural 2D wind noise ────────────────────────────────────────────────
//
// Cheap multi-octave sin/cos lattice. Not a true gradient noise; the goal is
// "varies in space and time and looks like wind," not "passes the noise unit
// tests." Sample by world XZ + a directional scroll term so a steady wind
// direction shows as gust patterns drifting across the field.
static float windScalar(float2 p, float t) {
    float a = sin(p.x * 1.0 + t * 0.7) * cos(p.y * 1.3 - t * 0.5);
    float b = sin(p.x * 2.3 + t * 1.1 + 1.7) * cos(p.y * 1.9 + t * 0.9);
    float c = sin((p.x + p.y) * 0.6 + t * 0.3);
    return (a * 0.6 + b * 0.3 + c * 0.4);
}

// ── KERNEL: per-particle wind + restoring spring ─────────────────────────────
//
// Runs AFTER pbdIntegrate and BEFORE the constraint loop. Modifies the
// post-integrate position so the implicit Verlet velocity reflects the
// wind impulse + a damped spring pull toward rest. Pinned particles
// (invMass == 0) are skipped — grass roots stay put regardless of wind.
//
// `rest` is a parallel buffer of starting positions (one per particle). For
// grass this is the perfectly-vertical configuration laid down at scene
// init; for hair / fur it would be the styled rest pose.
// Rest positions arrive as float4 (xyz = rest, w = unused) so Swift's
// SIMD4<Float> 16-byte stride matches Metal's float4 alignment. An earlier
// pass used packed_float3 here — Swift's SIMD3<Float> stride is 16 bytes, not
// 12, so the kernel was reading garbage for every particle past id=0 and
// blades got pulled toward random points (visible as a permanent uniform
// lean regardless of wind tuning).
kernel void pbdFieldForces(
    device       PBDParticle*           particles [[ buffer(0) ]],
    device const float4*                rest      [[ buffer(1) ]],
    constant     PBDFieldForcesUniforms& u        [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned root

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;
    float3 vel  = pos - prev;

    // Wind: scroll the noise lattice along the wind direction so successive
    // gusts visibly travel through the field rather than just flickering.
    float2 scrollOffset = float2(u.windDirX, u.windDirZ) * (u.windScroll * u.time);
    float2 samplePt     = float2(pos.x, pos.z) * u.windFreq - scrollOffset;
    float  w            = windScalar(samplePt, u.time);
    // Wind primarily pushes along the wind direction, with a vertical
    // component so blades occasionally lift / drop. Vertical mix kept small
    // so blades don't visually levitate.
    float3 windAccel = float3(u.windDirX * w, w * 0.15, u.windDirZ * w) * u.windAmp;

    // Restoring spring toward rest position. Tip particles (higher index)
    // get progressively less spring so the tip sways further than the base
    // — which is what real grass does mechanically. The taper happens
    // implicitly: the spring affects all non-root particles equally here,
    // but the base particle is held by its constraint to the pinned root,
    // so it can barely move; the tip has 4-5 constraints between it and
    // the root, each adding a bit of compliance, so it ends up doing the
    // most travelling under wind. No per-particle tuning needed.
    float3 restPos     = rest[id].xyz;
    float3 toRest      = restPos - pos;
    float3 springAccel = toRest * u.springK;

    vel += (windAccel + springAccel) * u.dt;
    // Optional viscous damping on top of the integrate kernel's `damping`
    // multiplier — useful for tuning "grass moves like it's in air" vs.
    // "grass moves like it's underwater." 0 = no extra damping.
    vel *= (1.0 - u.springDamp);

    particles[id].prevPositionAndPad.xyz = pos - vel;
}


// ── KERNEL: grass-blade ribbon expansion ─────────────────────────────────────
//
// One thread per particle. Writes 2 vertices + 2 normals per particle into
// the shared mesh buffer at layout:
//
//   vert[bladeIdx * 2*particlesPerBlade + i*2 + 0] = LEFT edge of blade at i
//   vert[bladeIdx * 2*particlesPerBlade + i*2 + 1] = RIGHT edge of blade at i
//
// Tangent direction at particle i is the central difference of neighbouring
// spine positions (forward/backward at the endpoints). The blade's left↔right
// axis is `cross(tangent, viewDir)` where viewDir = particle → camera; that
// keeps the ribbon facing the camera regardless of which way the blade is
// bent in world space.
//
// Width tapers linearly from `baseHalfWidth` at the root to
// `baseHalfWidth * tipHalfWidthFrac` at the tip. Anything less than
// ~tipHalfWidthFrac = 0.05 produces a degenerate triangle at the tip that
// the rasterizer will cull anyway — fine, blades look more grass-like with a
// nearly-pointed tip.
//
// Normals point toward the camera (`cross(left↔right, tangent)`), so the
// material reads correctly under any lighting setup without needing a
// double-sided shader trick. The host code still sets `isDoubleSided = true`
// to handle the back of a curled-over blade (where this normal would be
// pointing away from the camera).
kernel void pbdGrassExpand(
    device const PBDParticle*           particles   [[ buffer(0) ]],
    device       packed_float3*         verts       [[ buffer(1) ]],
    constant     GrassExpandUniforms&   u           [[ buffer(2) ]],
    device       packed_float3*         normals     [[ buffer(3) ]],
    device const float*                 widthScales [[ buffer(4) ]],  // one per blade (host fills 1.0 when uniform)
    uint id [[ thread_position_in_grid ]]
) {
    uint total = u.bladeCount * u.particlesPerBlade;
    if (id >= total) return;

    uint blade    = id / u.particlesPerBlade;
    uint localIdx = id - blade * u.particlesPerBlade;
    uint base     = blade * u.particlesPerBlade;

    float3 pos = particles[id].positionAndInvMass.xyz;

    // Central-difference tangent along the spine. Endpoints fall back to
    // a one-sided difference so the tip / root don't carry a half-magnitude
    // tangent (which would only matter for normal direction, but better to
    // be consistent).
    float3 tangent;
    if (localIdx == 0u) {
        float3 next = particles[base + 1u].positionAndInvMass.xyz;
        tangent = next - pos;
    } else if (localIdx + 1u >= u.particlesPerBlade) {
        float3 prev = particles[base + localIdx - 1u].positionAndInvMass.xyz;
        tangent = pos - prev;
    } else {
        float3 next = particles[base + localIdx + 1u].positionAndInvMass.xyz;
        float3 prev = particles[base + localIdx - 1u].positionAndInvMass.xyz;
        tangent = next - prev;
    }
    float tLen = length(tangent);
    tangent = (tLen > 1e-6) ? tangent / tLen : float3(0, 1, 0);

    // View direction from particle to camera. cross(tangent, viewDir) is
    // perpendicular to both — i.e. parallel to the screen's horizontal axis
    // for a near-vertical blade — and that's the axis we expand along.
    float3 viewDir = u.cameraPos.xyz - pos;
    float vLen = length(viewDir);
    viewDir = (vLen > 1e-6) ? viewDir / vLen : float3(0, 0, 1);

    float3 perp = cross(tangent, viewDir);
    float pLen = length(perp);
    if (pLen < 1e-6) {
        // Tangent and viewDir near-parallel (looking straight down the
        // blade). Pick an arbitrary perpendicular so the blade doesn't
        // collapse to a line.
        perp = cross(tangent, float3(0, 0, 1));
        pLen = length(perp);
        if (pLen < 1e-6) { perp = float3(1, 0, 0); pLen = 1.0; }
    }
    perp /= pLen;

    // Linear taper from base (t=0) to tip (t=1), scaled per blade (flower
    // heads / reeds can run wider than the field's grass blades).
    float t = float(localIdx) / max(1.0f, float(u.particlesPerBlade - 1u));
    float halfWidth = u.baseHalfWidth * widthScales[blade] *
        mix(1.0f, max(u.tipHalfWidthFrac, 0.0f), t);

    // Normal points "toward the camera" — orthogonal to both the blade
    // direction and the left↔right axis, so it's the front face of the
    // ribbon. Flip if it ended up facing away (handles the cross-product
    // handedness in cases where the blade's tangent has crossed under
    // the camera).
    float3 nrm = cross(perp, tangent);
    if (dot(nrm, viewDir) < 0.0) nrm = -nrm;
    float nLen = length(nrm);
    nrm = (nLen > 1e-6) ? nrm / nLen : float3(0, 0, 1);

    uint outBase = id * 2u;
    verts  [outBase + 0u] = pos - perp * halfWidth;
    verts  [outBase + 1u] = pos + perp * halfWidth;
    normals[outBase + 0u] = nrm;
    normals[outBase + 1u] = nrm;
}
