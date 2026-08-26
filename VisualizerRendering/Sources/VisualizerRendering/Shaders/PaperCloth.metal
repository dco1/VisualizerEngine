#include <metal_stdlib>
using namespace metal;

// ── PaperCloth.metal ──────────────────────────────────────────────────────────
//
// Compute kernels for PaperClothSolver — 2-D XPBD cloth (sheets of paper) that
// bend, fold, twist, self-shadow, and collide with each other. A 2-D extension
// of the 1-D chain work in PBD.metal / PBDField.metal.
//
// REUSE: the solver reuses pbdIntegrate (Verlet) and pbdConstraint (XPBD
// distance) from PBD.metal verbatim for the stretch + shear constraints — those
// kernels are already global-by-id and don't care whether the constraint graph
// is a chain or a grid, as long as the Swift side graph-colours the constraints
// so each dispatched colour group touches disjoint particles.
//
// This file adds what the sheet case needs and the chain case doesn't:
//   • paperWind          — per-particle, upward-biased wind + turbulence.
//   • paperAero          — per-vertex aerodynamic pressure (gather, no atomics).
//   • paperBendConstraint — dihedral hinge XPBD on triangle pairs (CLOTH bend;
//     the skip-one distance groups model PAPER — see the kernel's comment).
//   • paperHash*         — counting-sort spatial hash (self-collision broadphase).
//   • paperSelfCollide   — particle-particle pushout across ALL sheets.
//   • paperWritePositions/paperRecomputeNormals — pack the deformed grid into
//     the packed_float3 position+normal buffers Illuminatorama reads.
//
// Struct layouts MUST match the Swift mirrors in PaperClothSolver.swift exactly.
// ALIGNMENT RULE (see PBDSolver.swift): no bare float3 in a shared struct;
// float4 ↔ SIMD4<Float>, packed_float3 ↔ SIMD3<Float> for vertex arrays.

// Mirrors PBDParticle in PBD.metal / PBDSolver.swift.
struct PBDParticle {
    float4 positionAndInvMass;  // xyz = position, w = invMass (0 = pinned)
    float4 prevPositionAndPad;  // xyz = prevPosition, w = unused
};

// Mirrors PBDCollider in PBD.metal / PBDSolver.swift (each .metal file is its
// own translation unit — PBD.metal's definition is not visible here).
//   a.xyz = centre / endpoint A; a.w = type tag (0 sphere, 1 capsule, 2 box)
//   b.xyz = endpoint B / box halfExtents; b.w = radius
//   meta.x = ownerID
struct PBDCollider {
    float4 a;
    float4 b;
    uint4  meta;
};

// Per-sheet mesh-pack uniforms. The particle buffer is bound with a byte offset
// to this sheet's base, so `id` indexes 0..<vertexCount within the sheet and the
// output buffers are this sheet's own packed_float3 position / normal buffers.
struct PaperMeshUniforms {
    uint vertexCount;   // gridW * gridH
    uint gridW;         // vertices per row
    uint gridH;         // vertices per column
    uint _pad;
};

// Per-frame wind/aero uniforms. Mirror PaperWindUniforms in PaperClothSolver.swift.
struct PaperWindUniforms {
    uint  particleCount;
    uint  verticesPerSheet;   // M = gridW · gridH
    uint  gridW;
    uint  gridH;
    float dt;
    float time;
    float windAmp;            // base wind speed (m/s) along (dirX,0,dirZ)
    float windFreq;           // spatial frequency of the gust noise (1/m)
    float windScroll;         // gust drift speed (m/s)
    float windDirX;
    float windDirZ;
    float turb;               // turbulent velocity amplitude (m/s)
    float aero;               // normal-pressure coefficient (billow / flap strength)
    float drag;               // air-coupling drag toward the wind velocity (1/s)
    // Updraft JET over the screen mouth: peak speed at the mouth, decaying to 0
    // by `jetHeight` above and `jetRadius` out. Pages rise in the jet, exit the
    // top with upward momentum, arc over under gravity, and fall back into it —
    // a self-sustaining fountain (no respawn / readback needed).
    float mouthX;
    float mouthY;
    float mouthZ;
    float jetHeight;          // THIN nozzle height — strong updraft only this far up
    float jetRadius;          // nozzle footprint radius over the mouth
    float jetUpdraft;         // peak upward wind speed at the mouth (m/s)
    float wallRadius;         // lateral soft-wall radius (wider than the nozzle)
    float _pad1;
};

// ── Procedural wind noise (shared with the grass field solver's idea) ─────────
static float paperWindScalar(float2 p, float t) {
    float a = sin(p.x * 1.0 + t * 0.7) * cos(p.y * 1.3 - t * 0.5);
    float b = sin(p.x * 2.3 + t * 1.1 + 1.7) * cos(p.y * 1.9 + t * 0.9);
    float c = sin((p.x + p.y) * 0.6 + t * 0.3);
    return (a * 0.6 + b * 0.3 + c * 0.4);
}

// Smooth per-vertex normal from the (up-to-4) incident grid quads. `g` is the
// GLOBAL particle index; neighbours are found within this vertex's sheet slice.
static float3 paperVertexNormal(device const PBDParticle* P, uint g,
                                uint M, uint W, uint H) {
    uint base  = (g / M) * M;
    uint local = g - base;
    uint x = local % W, y = local / W;
    float3 c = P[g].positionAndInvMass.xyz;
    float3 R = (x + 1u < W) ? P[base + y*W + (x+1u)].positionAndInvMass.xyz : c;
    float3 L = (x > 0u)     ? P[base + y*W + (x-1u)].positionAndInvMass.xyz : c;
    float3 U = (y + 1u < H) ? P[base + (y+1u)*W + x].positionAndInvMass.xyz : c;
    float3 D = (y > 0u)     ? P[base + (y-1u)*W + x].positionAndInvMass.xyz : c;
    float3 eR = R - c, eL = L - c, eU = U - c, eD = D - c;
    float3 n = cross(eU, eR) + cross(eL, eU) + cross(eD, eL) + cross(eR, eD);
    float len = length(n);
    return (len > 1e-8) ? n / len : float3(0, 1, 0);
}

// ── KERNEL: wind + aerodynamics ───────────────────────────────────────────────
//
// Runs AFTER pbdIntegrate, BEFORE the constraint loop (same slot as the grass
// solver's pbdFieldForces). Builds a turbulent, upward-biased wind velocity
// field, then applies two air-coupling forces per particle:
//   • drag        — pulls the particle's velocity toward the local wind velocity
//                   (the bulk loft that carries paper up out of the CRT).
//   • aero pressure — a force along the surface normal proportional to the wind
//                   speed THROUGH the sheet (dot(n, relWind)). This is what makes
//                   a flat sheet billow, luff, and flap instead of sliding rigidly.
// Impulse is injected by shifting `prev` (implicit Verlet velocity), exactly like
// pbdFieldForces. Pinned particles (invMass==0) are skipped.
kernel void paperWind(
    device       PBDParticle*       particles [[ buffer(0) ]],
    constant     PaperWindUniforms& u         [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;   // pinned

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;
    float dt = max(u.dt, 1e-5);
    float3 v = (pos - prev) / dt;                 // actual velocity (m/s)

    // Gust field: scroll a noise lattice along the wind direction.
    float2 sp = float2(pos.x, pos.z) * u.windFreq
              - float2(u.windDirX, u.windDirZ) * (u.windScroll * u.time);
    float w = paperWindScalar(sp, u.time);
    float gust = 0.55 + 0.45 * w;                 // modulate base speed

    // A "fan on the floor": a gentle, CONTINUOUS updraft column rising from the
    // floor up through the whole height, within a radius covering the monitor.
    // Tuned (with strong air-damping below) so the air speed roughly balances
    // paper's fall — pages FLOAT, glide, and shear slowly rather than being
    // kicked ballistically. The updraft tapers to zero near the top so pages
    // settle back down instead of pinning to the ceiling.
    float2 toAxis = float2(pos.x - u.mouthX, pos.z - u.mouthZ);
    float r       = length(toAxis);
    float rFall   = clamp(1.0 - r / max(u.jetRadius, 1e-3), 0.0, 1.0);
    rFall = smoothstep(0.0, 1.0, rFall);
    // Full-column height profile that falls off FAST with height (topFall²) so
    // there's a STABLE floating equilibrium band at mid-height: strong lift low
    // down (net up), near-zero lift high up (net down). Pages collect in the
    // band and float/drift there. `jetUpdraft` sets the band's height.
    float topFall = clamp(1.0 - pos.y / max(u.jetHeight, 1e-3), 0.0, 1.0);
    float up = u.jetUpdraft * rFall * topFall * topFall * (0.8 + 0.2 * gust);

    // Gentle lateral soft-wall so the floating cloud doesn't drift off sideways.
    float2 inward = (r > 1e-4) ? -toAxis / r : float2(0.0);
    float pull = 6.0 * max(0.0, r - u.wallRadius);

    // Wind velocity the page is drag-coupled to: gentle updraft + soft swirl +
    // mild turbulence + the wall. Drag (below) pulls the page toward this, so a
    // balanced updraft makes it hover and drift, not snap.
    float3 windVec = float3(0.0, up, 0.0)
                   + float3(u.windDirX, 0.0, u.windDirZ) * (u.windAmp * gust)
                   + float3(inward.x * pull, 0.0, inward.y * pull)
                   + float3(w, w * 0.25, -w) * u.turb;

    float3 n   = paperVertexNormal(particles, id, u.verticesPerSheet, u.gridW, u.gridH);
    float3 rel = windVec - v;
    // Soft aero pressure (gentle billow/shear, not crumple) + air drag toward
    // the wind — drag is the dominant term, which is what makes paper "float".
    float3 accel = n * (u.aero * dot(n, rel)) + rel * u.drag;

    v += accel * dt;
    particles[id].prevPositionAndPad.xyz = pos - v * dt;
}

// ── DIHEDRAL BEND (XPBD hinge) ────────────────────────────────────────────────
//
// A real bending constraint for CLOTH, as opposed to the skip-one distance
// "bend" groups, which model PAPER. The difference is the small-angle regime:
// a skip-one chord shortens with the *cosine* of the fold angle, so its
// restoring force grows like θ³ — effectively zero for small kinks — and once
// the sheet buckles, the chord is equally happy with all the curvature piled
// into one sharp crease. That is correct behaviour for paper (it creases) and
// wrong for a duvet (it rolls). A hinge on the dihedral ANGLE between the two
// triangles sharing an edge restores like θ: every small kink feels a torque,
// so curvature spreads into an arc instead of concentrating into a fold line.
//
// Formulation: Müller et al., "Position Based Dynamics", the dihedral
// constraint C = acos(n₁·n₂) − φ₀ over four particles — p₁,p₂ the shared edge,
// p₃,p₄ the two wing vertices — lifted to XPBD (per-constraint compliance + λ,
// same lifecycle as pbdConstraint's). φ₀ = π for cloth cut flat.
//
// The Swift side graph-colours hinges so each dispatched colour group touches
// disjoint particles (same contract as every other constraint pass).
struct PaperBendConstraint {
    uint4  v;          // x,y = shared-edge vertices; z,w = wing vertices
    float  restAngle;  // dihedral at rest (π = flat)
    float  compliance; // XPBD α: 0 = rigid crease-holder, large = soft
    float2 _pad;
};

struct PaperBendUniforms {
    uint   constraintCount;
    float  dt;
    // CREASING — the plastic half of fabric bending. Elastic-only hinges give every
    // fold the same generous radius, which reads as RUBBER. Real cloth folded past a
    // yield angle creases: the fold keeps its shape and stops fighting back. When a
    // hinge's deviation from rest exceeds `yieldAngle`, its rest angle creeps toward
    // the folded pose by `creepRate` per solver visit — sustained tight folds become
    // permanent sharp ridges; gentle curvature stays elastic and smooth.
    float  yieldAngle;   // radians of deviation before plastic flow (≥ π = never)
    float  creepRate;    // fraction of the excess absorbed per visit (0 = elastic only)
};

kernel void paperBendConstraint(
    device PBDParticle*                particles   [[ buffer(0) ]],
    device PaperBendConstraint*        constraints [[ buffer(1) ]],
    constant PaperBendUniforms&        u           [[ buffer(2) ]],
    device float*                      lambda      [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.constraintCount) return;
    PaperBendConstraint c = constraints[id];

    float4 P1 = particles[c.v.x].positionAndInvMass;
    float4 P2 = particles[c.v.y].positionAndInvMass;
    float4 P3 = particles[c.v.z].positionAndInvMass;
    float4 P4 = particles[c.v.w].positionAndInvMass;
    float w1 = P1.w, w2 = P2.w, w3 = P3.w, w4 = P4.w;
    if (w1 + w2 + w3 + w4 < 1e-9) return;   // all four pinned

    // Work relative to p1 (Müller Appendix A convention).
    float3 p2 = P2.xyz - P1.xyz;
    float3 p3 = P3.xyz - P1.xyz;
    float3 p4 = P4.xyz - P1.xyz;

    float3 n1r = cross(p2, p3); float l1 = length(n1r);
    float3 n2r = cross(p2, p4); float l2 = length(n2r);
    if (l1 < 1e-12 || l2 < 1e-12) return;   // degenerate triangle — no hinge
    float3 n1 = n1r / l1, n2 = n2r / l2;

    float d   = clamp(dot(n1, n2), -1.0, 1.0);
    float phi = acos(d);
    // Plastic creasing (see PaperBendUniforms). Each thread owns constraints[id],
    // so the write is race-free within a colour group like every other update.
    float dev = phi - c.restAngle;
    if (u.creepRate > 0.0 && abs(dev) > u.yieldAngle) {
        float excess = abs(dev) - u.yieldAngle;
        c.restAngle += sign(dev) * excess * u.creepRate;
        constraints[id].restAngle = c.restAngle;
    }
    float C   = phi - c.restAngle;

    // Müller's qᵢ — note they are −∂d/∂pᵢ (verified numerically), so with
    // C = acos(d): ∇Cᵢ = −∂d/∂pᵢ/sinφ = +qᵢ/sinφ. Getting this sign wrong
    // turns the hinge into an amplifier — the sheet accordions at cell scale.
    float3 q3 = (cross(p2, n2) + cross(n1, p2) * d) / l1;
    float3 q4 = (cross(p2, n1) + cross(n2, p2) * d) / l2;
    float3 q2 = -(cross(p3, n2) + cross(n1, p3) * d) / l1
                -(cross(p4, n1) + cross(n2, p4) * d) / l2;
    float3 q1 = -q2 - q3 - q4;

    float sin2 = max(1.0 - d * d, 1e-8);     // sin²φ — the acos derivative
    float gradSum = (w1 * dot(q1, q1) + w2 * dot(q2, q2)
                   + w3 * dot(q3, q3) + w4 * dot(q4, q4)) / sin2;
    float alphaTilde = c.compliance / (u.dt * u.dt);
    float denom = gradSum + alphaTilde;
    if (denom < 1e-12) return;

    float deltaLambda = (-C - alphaTilde * lambda[id]) / denom;
    lambda[id] += deltaLambda;

    // Δpᵢ = wᵢ · ∇Cᵢ · Δλ, with ∇Cᵢ = +qᵢ/sinφ.
    float scale = deltaLambda * rsqrt(sin2);
    particles[c.v.x].positionAndInvMass.xyz += q1 * (w1 * scale);
    particles[c.v.y].positionAndInvMass.xyz += q2 * (w2 * scale);
    particles[c.v.z].positionAndInvMass.xyz += q3 * (w3 * scale);
    particles[c.v.w].positionAndInvMass.xyz += q4 * (w4 * scale);
}

// ── STATIC FRICTION (contact stick) ───────────────────────────────────────────
//
// The one force that makes cloth REST. The collide passes' `collideFriction`
// is tangential-velocity RETENTION — a rate, with no threshold — so a resting
// sheet is never actually held: it creeps every substep and a long settle
// walks it off whatever it lies on (the bake's pinPatch was invented to fight
// exactly this). Real static friction has a stick regime: below a threshold
// tangential speed, a contact does not slide at all.
//
// Runs after the collide passes. For each particle resting near a collider
// surface (or the floor), if its tangential speed is below `stickSpeed`, the
// tangential velocity is zeroed by moving `prev` (implicit Verlet velocity).
// The normal component survives, so settling, push-out and lift-off all still
// work. Above the threshold the contact slides exactly as before.
struct PaperStickUniforms {
    uint  particleCount;
    uint  colliderCount;
    float dt;
    float floorY;        // -1e9 when the floor is off
    float contactBand;   // distance from a surface that still counts as resting
    float stickSpeed;    // tangential speed below which the contact is static (m/s)
    float selfRadius;    // collider inflation, same value the collide pass uses
    // Breakable STATIC anchors — true static friction. > 0 enables: a particle
    // that comes to rest in contact ANCHORS to that spot and is held tangentially
    // (normal free, so cloth still lifts and settles) until pulled past this slip
    // distance (m). Velocity gating alone cannot do this: positional relaxation
    // moves material faster than any velocity threshold, then leaves it smooth.
    //
    // Anchors arm only where the contact normal is OFF-VERTICAL (|n.y| < 0.85):
    // cloth wrapping an edge is pressed into it by tension × curvature and grips
    // hard; cloth lying flat carries only its own weight and must stay free to
    // buckle (anchoring the flat top froze the quilt slack's dunes at 0.3 mm).
    float anchorBreak;
    // 1 → this pass ARMS anchors for every eligible contact regardless of speed.
    // Needed at t = 0: after a single substep gravity already moves everything
    // faster than any sane capture speed (g·dt ≈ 0.08 m/s), so the speed-gated
    // capture can never catch the initial resting state on its own.
    uint  armAll;
};

// Local copy — PBD.metal's `closestOnSegment` has internal linkage there.
static float3 paperClosestOnSegment(float3 p, float3 a, float3 b) {
    float3 ab = b - a;
    float  t  = dot(p - a, ab) / max(dot(ab, ab), 1e-12);
    return a + ab * clamp(t, 0.0, 1.0);
}

// Distance to (inflated) collider surface + outward normal at `p`.
static float paperColliderDistance(float3 p, PBDCollider col, float inflate,
                                   thread float3& n) {
    uint  kind   = as_type<uint>(col.a.w);
    float radius = col.b.w + inflate;
    if (kind == 2u) {                       // box
        float3 d  = p - col.a.xyz;
        float3 he = col.b.xyz;
        float3 q  = clamp(d, -he, he);
        float3 outside = d - q;
        float  outLen  = length(outside);
        if (outLen > 1e-6) { n = outside / outLen; return outLen - radius; }
        float3 dist = he - abs(d);
        if (dist.x < dist.y && dist.x < dist.z) { n = float3(sign(d.x), 0, 0); return -(dist.x + radius); }
        if (dist.y < dist.z)                    { n = float3(0, sign(d.y), 0); return -(dist.y + radius); }
        n = float3(0, 0, sign(d.z));            return -(dist.z + radius);
    }
    float3 closest = (kind == 1u) ? paperClosestOnSegment(p, col.a.xyz, col.b.xyz)
                                  : col.a.xyz;
    float3 d = p - closest;
    float  l = length(d);
    n = (l > 1e-6) ? d / l : float3(0, 1, 0);
    return l - radius;
}

// Cloth's own SDF collide — replaces the shared pbdSDFCollide for this solver.
// Two reasons: (1) the shared kernel probes NEIGHBOUR particles (midpoint probes
// for sparse hot-dog spines) while those neighbours are being rewritten, a
// scheduling race that makes the bake non-deterministic — a dense cloth grid
// needs no probes; (2) each thread then touches only its own particle, full stop.
struct PaperClothCollideUniforms {
    uint  particleCount;
    uint  colliderCount;
    float stiffness;     // fraction of the penetration resolved per pass
    float maxPush;       // per-step displacement cap (m); climb out coherently
    float selfRadius;    // collider inflation (cloth thickness)
    // Tangential velocity RETENTION on contact (0…1, lower = stickier) — the same
    // kinetic friction the shared pbdSDFCollide applies. Its absence here was a
    // silent regression: with frictionless contact, any in-plane surplus in resting
    // cloth drains laterally off the surface within a few substeps and the settled
    // top irons into a perfect plane.
    float friction;
    float _pad0, _pad1;
};

kernel void paperClothSDFCollide(
    device PBDParticle*                 particles [[ buffer(0) ]],
    device const PBDCollider*           colliders [[ buffer(1) ]],
    constant PaperClothCollideUniforms& u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;   // pinned

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;
    bool hit = false;
    float3 cn = float3(0, 1, 0);
    for (uint c = 0u; c < u.colliderCount; ++c) {
        float3 n;
        float d = paperColliderDistance(pos, colliders[c], u.selfRadius, n);
        if (d < 0.0) {
            float disp = min(-d * u.stiffness, u.maxPush);
            pos  += n * disp;
            prev += n * disp * 0.5;   // bleed off inward velocity, same as the shared kernel
            cn = n;
            hit = true;
        }
    }
    if (hit) {
        // Kinetic contact friction: retain only `friction` of the tangential motion.
        float3 v  = pos - prev;
        float3 vn = cn * dot(v, cn);
        float3 vt = v - vn;
        particles[id].positionAndInvMass.xyz = pos;
        particles[id].prevPositionAndPad.xyz = pos - (vn + vt * u.friction);
    }
}

kernel void paperStaticFriction(
    device PBDParticle*        particles [[ buffer(0) ]],
    device const PBDCollider*  colliders [[ buffer(1) ]],
    constant PaperStickUniforms& u       [[ buffer(2) ]],
    device float4*             anchors   [[ buffer(3) ]],  // xyz anchor, w: 1 = armed
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;   // pinned

    float3 pos = p.positionAndInvMass.xyz;

    // Nearest resting surface within the contact band.
    float  bestD = pos.y - u.floorY;             // floor (−1e9 when off)
    float3 bestN = float3(0, 1, 0);
    for (uint c = 0u; c < u.colliderCount; ++c) {
        float3 n;
        float  d = paperColliderDistance(pos, colliders[c], u.selfRadius, n);
        if (d < bestD) { bestD = d; bestN = n; }
    }
    bool contact = bestD <= u.contactBand;

    float3 vel = (pos - p.prevPositionAndPad.xyz) / u.dt;
    float3 vn  = bestN * dot(vel, bestN);
    float3 vt  = vel - vn;

    if (u.anchorBreak > 0.0) {
        float4 a = anchors[id];
        if (a.w > 0.5) {
            // Anchored: hold the TANGENTIAL position at the anchor; the normal
            // direction stays free so the cloth can still lift off or press in.
            float3 off  = pos - a.xyz;
            float3 offN = bestN * dot(off, bestN);
            float3 offT = off - offN;
            bool lifted = bestD > u.contactBand * 2.0;
            if (lifted || length(offT) > u.anchorBreak) {
                anchors[id] = float4(0);                       // SLIP — release
            } else {
                float3 held = pos - offT;
                particles[id].positionAndInvMass.xyz = held;
                // Kill the tangential velocity too, or it re-slips next step.
                particles[id].prevPositionAndPad.xyz = held - vn * u.dt;
                return;
            }
        }
        bool edgeContact = contact && fabs(bestN.y) < 0.85;
        if (edgeContact && (u.armAll != 0u || length(vt) < u.stickSpeed)) {
            anchors[id] = float4(pos, 1.0);                    // GRIP
        }
        return;
    }

    if (contact && length(vt) < u.stickSpeed) {
        // STICK (velocity mode): tangential motion dies; normal survives.
        particles[id].prevPositionAndPad.xyz = pos - vn * u.dt;
    }
}

// ── SELF-COLLISION (GPU spatial hash) ─────────────────────────────────────────
//
// Counting-sort uniform hash grid over ALL particles of ALL sheets (mirrors the
// integer-atomic broadphase in MLSMPM.metal — no float atomics). Then a Jacobi
// pushout: each thread reads its 27-cell neighbourhood and moves ONLY its own
// particle by half the penetration, so the pass is race-free even though every
// thread reads shared positions. This is what makes the pages "know about each
// other" — different sheets stack instead of interpenetrating, and a sheet can't
// pass through itself when it folds.
struct PaperHashUniforms {
    uint  particleCount;
    uint  tableSize;
    float cellSize;
    float radius;          // collision radius; particles separate to 2·radius
    uint  gridW;
    uint  verticesPerSheet;
    uint  skipRadius;      // skip same-sheet neighbours within this grid distance
    uint  legacy;          // DEBUG A/B: 1 = old summed/unclamped pushout (reproduce the blowup)
    // Cloth-cloth STATIC friction: contacting layers whose relative tangential
    // slip per substep is below this DISPLACEMENT (stickSpeed·dt) stop sliding
    // against each other — a pile of folds HOLDS instead of slithering apart.
    // 0 disables (paper pages slide).
    float stickDisp;
    float _pad;
};

static int3 paperCellCoord(float3 p, float cs) { return int3(floor(p / cs)); }
static uint paperHashCell(int3 c, uint tableSize) {
    uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u);
    return h % tableSize;
}

kernel void paperHashClear(device atomic_uint* counts [[ buffer(0) ]],
                           constant uint& tableSize   [[ buffer(1) ]],
                           uint id [[ thread_position_in_grid ]]) {
    if (id >= tableSize) return;
    atomic_store_explicit(&counts[id], 0u, memory_order_relaxed);
}

kernel void paperHashCount(device const PBDParticle* P  [[ buffer(0) ]],
                           device atomic_uint* counts    [[ buffer(1) ]],
                           constant PaperHashUniforms& u [[ buffer(2) ]],
                           uint id [[ thread_position_in_grid ]]) {
    if (id >= u.particleCount) return;
    uint h = paperHashCell(paperCellCoord(P[id].positionAndInvMass.xyz, u.cellSize), u.tableSize);
    atomic_fetch_add_explicit(&counts[h], 1u, memory_order_relaxed);
}

// Single-thread exclusive prefix sum (tableSize is modest, runs once/frame).
// Re-zeros counts so the scatter pass can reuse it as a write cursor.
kernel void paperHashScan(device atomic_uint* counts [[ buffer(0) ]],
                          device uint* offsets        [[ buffer(1) ]],
                          constant uint& tableSize    [[ buffer(2) ]],
                          uint id [[ thread_position_in_grid ]]) {
    if (id != 0u) return;
    uint acc = 0u;
    for (uint i = 0u; i < tableSize; ++i) {
        uint c = atomic_load_explicit(&counts[i], memory_order_relaxed);
        offsets[i] = acc;
        acc += c;
        atomic_store_explicit(&counts[i], 0u, memory_order_relaxed);
    }
}

kernel void paperHashScatter(device const PBDParticle* P  [[ buffer(0) ]],
                             device atomic_uint* cursor    [[ buffer(1) ]],
                             device const uint* offsets     [[ buffer(2) ]],
                             device uint* sorted            [[ buffer(3) ]],
                             constant PaperHashUniforms& u  [[ buffer(4) ]],
                             uint id [[ thread_position_in_grid ]]) {
    if (id >= u.particleCount) return;
    uint h = paperHashCell(paperCellCoord(P[id].positionAndInvMass.xyz, u.cellSize), u.tableSize);
    uint slot = atomic_fetch_add_explicit(&cursor[h], 1u, memory_order_relaxed);
    sorted[offsets[h] + slot] = id;
}

// `S` is a SNAPSHOT of the particle buffer taken just before this dispatch. All
// reads (own state AND neighbours) come from the snapshot; only P[id] is written.
// Without it the kernel raced its own writes — whether a thread saw a neighbour
// before or after its correction depended on GPU scheduling, which made the bake
// non-deterministic run to run (amplified by chaos over hundreds of substeps).
kernel void paperSelfCollide(device PBDParticle* P          [[ buffer(0) ]],
                             device const uint* offsets      [[ buffer(1) ]],
                             device const uint* counts       [[ buffer(2) ]],  // per-bucket count (post-scatter cursor)
                             device const uint* sorted       [[ buffer(3) ]],
                             constant PaperHashUniforms& u   [[ buffer(4) ]],
                             device const PBDParticle* S     [[ buffer(5) ]],
                             uint id [[ thread_position_in_grid ]]) {
    if (id >= u.particleCount) return;
    PBDParticle pi = S[id];
    if (pi.positionAndInvMass.w == 0.0) return;   // pinned

    float3 pos = pi.positionAndInvMass.xyz;
    int3 base  = paperCellCoord(pos, u.cellSize);
    uint sheet = id / u.verticesPerSheet;
    uint local = id - sheet * u.verticesPerSheet;
    int  x = int(local % u.gridW), y = int(local / u.gridW);

    float minDist = 2.0 * u.radius;
    float3 delta  = float3(0.0);
    float3 nbrVel = float3(0.0);   // summed Δpos of contacting neighbours (velocity·dt)
    int    hits   = 0;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        uint h = paperHashCell(base + int3(dx, dy, dz), u.tableSize);
        uint start = offsets[h], cnt = counts[h];
        // DETERMINISM: the scatter fills buckets through an atomic cursor, so the
        // stored order differs run to run — and float accumulation is not
        // associative, so the same contacts summed in a different order drift the
        // bake by fractions of a millimetre that chaos then amplifies. Iterate each
        // bucket in ascending particle order instead (insertion sort of a small
        // local copy); overflow beyond the local buffer keeps scatter order, which
        // only a degenerate all-in-one-bucket configuration would ever hit.
        uint bucket[64];
        uint m = min(cnt, 64u);
        for (uint k = 0u; k < m; ++k) bucket[k] = sorted[start + k];
        for (uint a = 1u; a < m; ++a) {
            uint v = bucket[a];
            uint b = a;
            while (b > 0u && bucket[b - 1u] > v) { bucket[b] = bucket[b - 1u]; --b; }
            bucket[b] = v;
        }
        for (uint k = 0u; k < cnt; ++k) {
            uint j = (k < m) ? bucket[k] : sorted[start + k];
            if (j == id) continue;
            // Same-sheet pairs: skip only GRID NEIGHBOURS (within skipRadius,
            // Chebyshev), not the whole sheet. The neighbour exclusion is what
            // keeps a sheet whose internal spacing is under the collision
            // radius from puffing itself up; excluding ALL same-sheet pairs —
            // which this kernel did for a while — silently turns SELF-collision
            // off for any lone sheet, and a duvet that cannot touch itself
            // cannot pile, gather or hold a fold: its corner surplus splays
            // into stiff wings (Daydream Home DH-0415). Distant same-sheet
            // particles only ever come within radius when the cloth genuinely
            // folds onto itself, which is exactly the contact we want.
            if (j / u.verticesPerSheet == sheet) {
                uint lj = j - sheet * u.verticesPerSheet;
                int jx = int(lj % u.gridW), jy = int(lj / u.gridW);
                if (uint(max(abs(jx - x), abs(jy - y))) <= u.skipRadius) continue;
            }
            float3 d = pos - S[j].positionAndInvMass.xyz;
            float dist = length(d);
            if (dist < minDist && dist > 1e-6) {
                delta += (d / dist) * ((minDist - dist) * 0.5);
                nbrVel += S[j].positionAndInvMass.xyz - S[j].prevPositionAndPad.xyz;
                hits++;
            }
        }
    }
    if (hits > 0) {
        // AVERAGE the pushout over the contacts, don't SUM it. When the collision
        // radius is larger than the other sheet's inter-vertex spacing, a single
        // vertex straddles a whole PATCH of the other sheet's grid (dozens of
        // neighbours), and a summed half-penetration is then many edge-lengths of
        // displacement in one frame — it yanks individual vertices into spikes
        // ("jagged points") and the prev-shift injects matching velocity so it
        // never settles. Averaging resolves the vertex toward the mean non-
        // penetration position (and naturally centres a sandwiched vertex). A hard
        // clamp to one radius/step is the safety belt against any residual blowup.
        float3 corr;
        if (u.legacy != 0u) {
            corr = delta;                          // OLD: summed half-penetrations, unclamped
        } else {
            corr = delta / float(hits);            // mean non-penetration position
            float cl = length(corr);
            if (cl > u.radius) corr *= u.radius / cl;   // clamp to one radius / pass
        }
        P[id].positionAndInvMass.xyz = pos + corr;
        P[id].prevPositionAndPad.xyz = pi.prevPositionAndPad.xyz + corr * 0.5;  // bleed off inward velocity

        // Static friction BETWEEN layers (see stickDisp above). Relative slip is
        // judged tangentially — the pushout direction is the contact normal — and
        // each side cancels HALF its share, so a symmetric pair stops together.
        if (u.stickDisp > 0.0) {
            float3 vSelf = pos - pi.prevPositionAndPad.xyz;
            float3 rel   = vSelf - nbrVel / float(hits);
            float  cl2   = length(corr);
            if (cl2 > 1e-9) {
                float3 n = corr / cl2;
                rel -= n * dot(rel, n);
            }
            if (length(rel) < u.stickDisp) {
                P[id].prevPositionAndPad.xyz += rel * 0.5;
            }
        }
    }
}

// ── KERNEL: pack deformed positions into the render buffer ────────────────────
//
// Copies the simulated particle positions (xyz of positionAndInvMass) into the
// packed_float3 position buffer the Illuminatorama repack kernel reads. Runs
// once per frame per sheet, AFTER the substep loop. `particles` is bound with a
// byte offset to this sheet's slice, so id is sheet-local.
kernel void paperWritePositions(
    device const PBDParticle* particles [[ buffer(0) ]],
    device packed_float3*     posOut    [[ buffer(1) ]],
    constant PaperMeshUniforms& u       [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.vertexCount) return;
    posOut[id] = particles[id].positionAndInvMass.xyz;
}

// ── KERNEL: recompute smooth per-vertex normals from the deformed grid ────────
//
// Per-vertex gather over the (up-to-4) grid quads incident to vertex (x,y). Each
// quad contributes its face normal (area-weighted by leaving the cross product
// un-normalised); the sum is normalised at the end. Race-free — each thread
// writes only its own normal and only reads neighbour positions.
//
// Orientation: for the seed flat grid (XZ plane, +Y up, U=+row=+Z, R=+col=+X)
// this yields +Y, matching the index winding's front face. The render mesh is
// double-sided (paper has two faces), so the renderer flips the normal for
// back-facing fragments regardless.
kernel void paperRecomputeNormals(
    device const PBDParticle* particles [[ buffer(0) ]],
    device packed_float3*     normOut   [[ buffer(1) ]],
    constant PaperMeshUniforms& u       [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.vertexCount) return;
    uint W = u.gridW, H = u.gridH;
    uint x = id % W;
    uint y = id / W;

    float3 c = particles[id].positionAndInvMass.xyz;
    // Cardinal neighbours, clamped to self at the border (zero edge → zero
    // contribution from the missing quads).
    float3 R = (x + 1u < W) ? particles[id + 1u].positionAndInvMass.xyz : c;
    float3 L = (x > 0u)     ? particles[id - 1u].positionAndInvMass.xyz : c;
    float3 U = (y + 1u < H) ? particles[id + W ].positionAndInvMass.xyz : c;
    float3 D = (y > 0u)     ? particles[id - W ].positionAndInvMass.xyz : c;

    float3 eR = R - c, eL = L - c, eU = U - c, eD = D - c;

    float3 n = float3(0.0);
    n += cross(eU, eR);   // +Z × +X = +Y for the flat seed
    n += cross(eL, eU);
    n += cross(eD, eL);
    n += cross(eR, eD);

    float len = length(n);
    normOut[id] = (len > 1e-8) ? (n / len) : float3(0.0, 1.0, 0.0);
}
