#include <metal_stdlib>
using namespace metal;

// ── CoinDEM.metal ─────────────────────────────────────────────────────────────
//
// GPU rigid-body solver for a coin-pusher pile: hundreds-to-thousands of flat
// coins that stack, slide, lean, get shoved by a reciprocating pusher, and
// cascade off an overhang into a payout trough. There is no general rigid-body
// solver in the project; this is the first. It is a *specialised* DEM solver, not
// a generic engine — see SimEngine.swift for why the project favours specialisation.
//
// FULL ORIENTED RIGID BODY (the cheat we removed)
// ───────────────────────────────────────────────
// An earlier version treated ORIENTATION AS DECORATIVE — coins always collided as
// upright cylinders and snapped flat when supported. That produced tidy, axis-
// aligned pancake "stacks" that never toppled and then froze in place: not a real
// pile. This solver makes orientation PHYSICAL. Each coin is a true rigid body
// (COM x, quaternion q, linear velocity v, angular velocity ω, real disk inertia),
// and every contact is resolved at a real CONTACT POINT, so the correction carries
// torque. Coins therefore rest at genuine tilts, lean and brace on their
// neighbours, tip over ledges, and the heap finds a natural angle of repose and
// micro-avalanches — a real mass of individual coins.
//
// THE CONTACT MODEL — feature points vs analytic capped-cylinder SDF
// ──────────────────────────────────────────────────────────────────
// A coin is a thin solid cylinder (radius R, half-thickness h). For each pair we
// sample one coin's SURFACE as a small set of feature points (both face centres +
// a rim ring on both faces) and test each point against the OTHER coin's analytic
// capped-cylinder signed-distance field. A penetrating point is a contact: its
// world position is the contact point, the SDF gradient is the contact normal, and
// the penetration is the depth. This is exact for the dominant contact cases
// (face-face stacking, face-edge lean, edge-edge) and — crucially — the contact
// point is offset from the COM, so the XPBD correction rotates the coin. Static
// colliders (floor/walls/shelf boxes, the kinematic pusher plate) are resolved the
// same way, per feature point, so a coin tips as it cantilevers off a lip.
//
// TODO(general-rigid-body): this feature-point-vs-cylinder-SDF model is the only
// coin-specific piece. To make this a generic convex rigid-body solver (e.g. drop
// a cube and let a corner contact tip it flat), branch on a per-body shape tag:
// keep the capped-cylinder SDF for disks, and add a box path — sample box corners/
// edges as feature points and test against an oriented-box SDF (or full box–box
// SAT). The integrator, inertia, contact-point XPBD correction, and static box/
// plane colliders below already carry over unchanged.
//
// SOLVER LOOP (one command buffer per frame, NO per-substep waitUntilCompleted)
//   per substep (×substeps):
//     coinIntegrate            predict COM (gravity, damping) + advance orientation
//                              by ω; store prevPos + prevOrient
//     coinCellClear/Count/Scan/Scatter   counting-sort hash on coin COM
//     × iterations:
//        coinContactSolve      feature points vs neighbour cylinders + static
//                              colliders → accumulate XPBD positional + rotational
//                              correction (Jacobi: read→delta→apply, no races)
//        coinApplyDelta        x += Δx·relax ; q ⊕= Δrot·relax   (averaged)
//     coinFinalize             v = (x-prevX)/dt, ω from quaternion delta, friction,
//                              sleep, floor safety
//   coinDeriveTransforms       (x, q, scale) → CoinTransform for the instanced renderer
//
// ALIGNMENT RULE (see PBDSolver.swift): every Swift↔Metal shared struct uses
// only float4 / uint4 lanes — never a bare float3 (Metal float3 is 16-byte
// aligned; Swift SIMD3 is 12). Scalars-only uniform structs are fine.

// ── Shared structs (mirror CoinDEMSolver.swift exactly) ───────────────────────

struct CoinBody {
    float4 posInvMass;  // xyz = COM (world),         w = invMass (0 = inactive slot)
    float4 prevPos;     // xyz = previous COM (scratch within substep), w = collider RADIUS (bounding for a box)
    float4 vel;         // xyz = linear velocity,     w = collider HALF-THICKNESS (min half-extent for a box)
    float4 orient;      // current visual+physical orientation quaternion (x,y,z,w)
    float4 prevOrient;  // orientation at substep start (x,y,z,w) — for deriving ω
    float4 angVel;      // xyz = angular velocity (world frame, rad/s),  w = support/rest flag
    float4 shapeExtents;// w = shape tag (0 disc/cylinder, 1 box, 2 sphere, 3 capsule, 4 hull); box: xyz = half-extents, sphere: x = radius, capsule: x = seg half-length
    float4 hullRef;     // hull only: x = hull index (float), yzw = per-unit-mass INVERSE inertia diag (principal frame)
};

// Plane:    a.xyz = unit normal, a.w = type tag(0);  b.w = plane offset d  (n·x = d)
// Box:      a.xyz = centre,      a.w = type tag(1);  b.xyz = half-extents
// Pusher:   a.xyz = centre,      a.w = type tag(2);  b.xyz = half-extents; vel.xyz = plate vel
// OBox:     a.xyz = centre,      a.w = type tag(3);  b.xyz = half-extents; orient = unit quat (local→world)
struct CoinStaticCollider {
    float4 a;
    float4 b;
    // xyz = surface velocity (kinematic pusher); w = cylinder (kind 4) half-length
    // ONLY — every other kind leaves it 0, but it's occupied there, so material
    // lives in meta.yz instead (see below).
    float4 vel;
    // x = flags (bit0 = one-way: push only when approaching from +normal).
    // y/z = per-collider (μ, e), bit-cast float via as_type<float>() — a NEGATIVE
    // value means "inherit the global uniform", identical convention and combine
    // rule (μ=√(μA·μB), e=max(eA,eB)) to per-body material. w reserved.
    uint4  meta;
    float4 orient; // oriented-box (kind 3) rotation quaternion (x,y,z,w); identity for other kinds
};
// `cdQuatRotate` / `cdQuatRotateInv` are defined below (shared with the body solver).

struct CoinUniforms {
    float dt;
    float gravity;
    float linDamping;     // per-substep linear velocity retention (<1)
    float coinRadius;     // R
    float halfThickness;  // h
    float contactRelax;   // Jacobi relaxation on the AVERAGED per-coin correction
    float friction;       // tangential linear velocity retention on contact (<1)
    float restitution;    // reserved (0 = inelastic, best for piles)
    float frictionCoeff;  // Coulomb μ for POSITIONAL friction-with-torque (0 = off)
    float rollingResistance; // resists spin of a body rolling on a contact (0 = off)
    float floorY;         // hard safety floor
    float sleepLinVel;    // below this combined speed a contacting coin is slept
    float angFriction;    // angular velocity retention on contact (was settleStrength)
    float angDamping;     // per-substep angular velocity retention (was settleDamp)
    float gridMinX;
    float gridMinY;
    float gridMinZ;
    float invCell;        // 1 / cellSize
    uint  coinCount;      // active high-water count (dispatch bound)
    uint  colliderCount;
    uint  gridResX;
    uint  gridResY;
    uint  gridResZ;
    // Velocity caps (finalize). Defaulted to the coin-pusher values by the
    // solver; a fast-ballistic scene (Tennis Ball Painter) raises them so a
    // thrown body can actually fly across the volume and rebound off the wall.
    float maxHSpeed;      // horizontal de-penetration speed cap (m/s)
    float maxSpeed;       // global linear speed cap (m/s)
    float maxOmega;       // angular speed cap (rad/s)
    // Constraint-solver params (Stage 3).
    float contactSlop;    // allowed penetration (m) — recovery only beyond this
    float baumgarteBeta;  // position-bias gain (split-impulse pseudo-velocity)
    float restThreshold;  // restitution only for approach speed beyond this (m/s)
    // ── Realistic-bounce extensions (opt-in; all 0 ⇒ byte-identical legacy) ──
    float restitutionVelFalloff; // COR drop per m/s of impact speed (0 = constant COR)
    float restitutionMinE;       // floor for the velocity-faded COR
    float quadraticDrag;         // ∝v² aerodynamic drag k (accel = −k·|v|·v); 0 = off
    float dragRefRadius;         // radius where quadraticDrag is calibrated; drag scales ∝1/r per body. 0 = flat k (no size scaling)
    float speculativeMargin;     // emit near-contacts within this gap (anti-tunneling); 0 = off
};

// Per-body drag coefficient. Real aerodynamic drag-per-mass A/m ∝ r²/r³ = 1/r, so
// a BIGGER body drags LESS (and falls/flies further) and a smaller one drags more.
// dragRefRadius is the radius at which `quadraticDrag` was tuned; 0 ⇒ flat k.
inline float cdDragK(constant CoinUniforms& u, float bodyRadius) {
    if (u.dragRefRadius <= 0.0) return u.quadraticDrag;
    float r = bodyRadius > 1e-4 ? bodyRadius : u.dragRefRadius;
    return u.quadraticDrag * (u.dragRefRadius / r);
}

// Velocity-dependent coefficient of restitution. Real materials — a fuzzy tennis
// ball especially — rebound with a SMALLER fraction of their speed as the impact
// gets harder (Cross 2002): e falls roughly linearly with impact speed over the
// few–tens-of-m/s range. falloff == 0 reproduces the constant-COR legacy exactly.
inline float cdEffectiveCOR(constant CoinUniforms& u, float impactSpeed) {
    float e = u.restitution - u.restitutionVelFalloff * impactSpeed;
    return clamp(e, min(u.restitutionMinE, u.restitution), u.restitution);
}

// Per-body-material variant: same velocity falloff, but around a per-CONTACT base
// COR (the combined material restitution) instead of the global uniform.
inline float cdEffectiveCORBase(constant CoinUniforms& u, float baseE, float impactSpeed) {
    float e = baseE - u.restitutionVelFalloff * impactSpeed;
    return clamp(e, min(u.restitutionMinE, baseE), baseE);
}

// ── Per-body material (constraint path) ──────────────────────────────────────
// material[slot] = (μ, e); a NEGATIVE lane means "inherit the global uniform"
// (u.frictionCoeff / u.restitution) — the default for every body, so a scene
// that never sets materials is byte-identical. Combine rules are the standard
// production ones: friction = √(μA·μB) (geometric mean), restitution = max.
inline float cdBodyMu(device const float2* material, uint slot, constant CoinUniforms& u) {
    float m = material[slot].x;
    return m >= 0.0 ? m : u.frictionCoeff;
}
inline float cdBodyE(device const float2* material, uint slot, constant CoinUniforms& u) {
    float e = material[slot].y;
    return e >= 0.0 ? e : u.restitution;
}

// ── Per-collider material (constraint path) ───────────────────────────────────
// Mirrors cdBodyMu/cdBodyE exactly, reading CoinStaticCollider.meta.yz (bit-cast
// float) instead of a per-body buffer slot. A static side used to just copy the
// dynamic body's own material (an icy ramp read identical to a rubber floor);
// this lets the collider carry its own (μ, e), combined the same way.
inline float cdColliderMu(device const CoinStaticCollider* colliders, uint idx, constant CoinUniforms& u) {
    float m = as_type<float>(colliders[idx].meta.y);
    return m >= 0.0 ? m : u.frictionCoeff;
}
inline float cdColliderE(device const CoinStaticCollider* colliders, uint idx, constant CoinUniforms& u) {
    float e = as_type<float>(colliders[idx].meta.z);
    return e >= 0.0 ? e : u.restitution;
}

struct CoinTransform {
    float4 col0;  // basis column 0 (xyz, 0)
    float4 col1;
    float4 col2;
    float4 col3;  // world position (xyz, 1)
};

// ── CoinContact — one contact-constraint (a single manifold point) ────────────
//
// THE NEW SOLVER'S CURRENCY. The legacy path (coinContactSolve→coinApplyDelta)
// re-detects contacts and pushes POSITIONS every iteration (Jacobi position-based
// dynamics). The constraint path instead GENERATES contacts ONCE per substep into a
// shared buffer of these records, then runs a graph-colored sequential-impulse
// VELOCITY solve + a split-impulse POSITION solve over them — real Gauss-Seidel
// convergence, warm-startable, with restitution/friction as proper velocity
// constraints. float4 lanes only (the ALIGNMENT RULE); 96 bytes.
struct CoinContact {
    uint4  meta;   // x=bodyA, y=bodyB (CD_STATIC=no dynamic B), z=colliderIdx|featureId, w=pairKey
    float4 nrm;    // xyz = world contact normal (points from B toward A), w = penetration depth (>0)
    float4 rA;     // xyz = contact point − comA,  w = accumulated NORMAL impulse  (warm start)
    float4 rB;     // xyz = contact point − comB,  w = accumulated TANGENT-1 impulse
    float4 tan1;   // xyz = tangent dir 1 (world), w = accumulated TANGENT-2 impulse
    float4 tan2;   // xyz = tangent dir 2 (world), w = assigned colour (−1 until coloured)
    // x = pre-solve approach speed vn₀ along n, captured ONCE by the first solve
    // pass (w = captured flag). The restitution target is −e·vn₀ — recomputing it
    // from the CURRENT vn each iteration let iteration 2 see the already-reflected
    // velocity, gate restitution off, and unwind the bounce via the accumulated-
    // impulse clamp (a resting-stack feature, a bounce killer). yz reserved.
    float4 aux;
};

// One generic constraint-path joint. Types (meta.x): 0 = BALL (anchors
// coincide, 3-DOF point constraint), 1 = HINGE (ball + axis alignment +
// optional angle limits), 2 = DISTANCE (anchor separation = rest length,
// 1-DOF). meta.w bit 0 = enabled, bit 1 = collideConnected (contacts between
// this joint's own bodyA/bodyB are suppressed in coinGenerateContacts unless
// this bit is set — a hinge whose bodies touch at the anchor would otherwise
// fight its own contact). Declared here (not by the joint-solve kernel below,
// where it conceptually lives) so coinGenerateContacts can read it too.
struct CoinJoint {
    uint4  meta;     // x = type, y = bodyA, z = bodyB (CD_STATIC = world), w = enabled|collideConnected bits
    float4 anchorA;  // xyz = anchor in A-local frame;               w = rest length (distance)
    float4 anchorB;  // xyz = anchor in B-local frame (WORLD if z==CD_STATIC)
    float4 axisA;    // xyz = hinge axis in A-local frame;           w = limit lo (rad)
    float4 axisB;    // xyz = hinge axis in B-local (WORLD if world); w = limit hi (rad)
};

// ── Quaternion helpers (mirror EggMotion.metal) ───────────────────────────────

static float3x3 cdQuatToMat3(float4 q) {
    float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    float xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    float wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    return float3x3(
        float3(1.0 - 2.0*(yy+zz),       2.0*(xy+wz),       2.0*(xz-wy)),
        float3(      2.0*(xy-wz), 1.0 - 2.0*(xx+zz),       2.0*(yz+wx)),
        float3(      2.0*(xz+wy),       2.0*(yz-wx), 1.0 - 2.0*(xx+yy))
    );
}

static float4 cdQuatMul(float4 a, float4 b) {
    return float4(
        a.w*b.xyz + b.w*a.xyz + cross(a.xyz, b.xyz),
        a.w*b.w - dot(a.xyz, b.xyz)
    );
}

static float4 cdQuatConj(float4 q) { return float4(-q.xyz, q.w); }

static float3 cdQuatRotate(float4 q, float3 v) {
    // v + 2w(q×v) + 2 q×(q×v)
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Inverse rotation (rotate by the conjugate).
static float3 cdQuatRotateInv(float4 q, float3 v) {
    return cdQuatRotate(cdQuatConj(q), v);
}

// Integrate a unit quaternion by a world-frame angular velocity for time dt.
static float4 cdIntegrateQuat(float4 q, float3 omega, float dt) {
    float4 dq = cdQuatMul(float4(omega * (0.5 * dt), 0.0), q);
    return normalize(q + dq);
}

// Apply a world-frame rotation VECTOR (axis·angle, small) to a quaternion.
static float4 cdApplyRotVec(float4 q, float3 w) {
    float4 dq = cdQuatMul(float4(0.5 * w, 0.0), q);
    return normalize(q + dq);
}

// ── Rigid-body inertia (thin solid disk, symmetry axis = local +Y) ────────────
//
// Inertia uses the body's actual radius/half-thickness AND its mass (via invMass):
// I = m·k, so I⁻¹ = invMass/k. Diameter axes (x,z) share a moment; the symmetry
// axis (y) is twice as easy to spin. For invMass == 1 (the default — a unit-mass
// pile) this is exactly the old fixed-mass tensor, so the disc pile is unchanged;
// a heavier body (invMass < 1) now correctly resists both translation AND rotation.
static float3 cdInvInertiaLocal(float R, float h, float invMass) {
    float Iyy_k = 0.5 * R * R;                          // about the disk normal, per unit mass
    float Ixx_k = 0.25 * R * R + (1.0/3.0) * h * h;     // about a diameter, per unit mass
    return float3(invMass / Ixx_k, invMass / Iyy_k, invMass / Ixx_k);
}

// Solid BOX inverse inertia, half-extents (a,b,c): I_xx = (1/3)m(b²+c²) etc.
static float3 cdBoxInvInertia(float3 he, float invMass) {
    float a2 = he.x*he.x, b2 = he.y*he.y, c2 = he.z*he.z;
    float Ixx_k = (1.0/3.0) * (b2 + c2);
    float Iyy_k = (1.0/3.0) * (a2 + c2);
    float Izz_k = (1.0/3.0) * (a2 + b2);
    return float3(invMass / Ixx_k, invMass / Iyy_k, invMass / Izz_k);
}

// Shape tags ride shapeExtents.w: 0 = disc/capped-cylinder, 1 = box, 2 = sphere,
// 3 = capsule, 4 = convex hull. (Every predicate MUST be a half-open band — a
// bare `> 0.5` box test would mis-collide every later tag as a box.)
static bool cdIsBox(CoinBody c)     { return c.shapeExtents.w > 0.5 && c.shapeExtents.w < 1.5; }
static bool cdIsSphere(CoinBody c)  { return c.shapeExtents.w > 1.5 && c.shapeExtents.w < 2.5; }
static bool cdIsCapsule(CoinBody c) { return c.shapeExtents.w > 2.5 && c.shapeExtents.w < 3.5; }
static bool cdIsHull(CoinBody c)    { return c.shapeExtents.w > 3.5; }

// Capsule lanes: prevPos.w = cross-section RADIUS r (like a disc), vel.w = the
// FULL half-height hl + r (so the legacy capped-cylinder path sees the correct
// bounds), shapeExtents.x = the true SEGMENT half-length hl for exact contacts.
static float cdCapsuleR(CoinBody c)  { return c.prevPos.w > 1e-4 ? c.prevPos.w : 0.02; }
static float cdCapsuleHL(CoinBody c) { return max(c.shapeExtents.x, 0.0); }

// Solid capsule inverse inertia (axis = local +Y). Standard cylinder+hemisphere
// aggregation: mass splits by volume; hemispheres carry parallel-axis terms.
static float3 cdCapsuleInvInertia(float r, float hl, float invMass) {
    float L  = 2.0 * hl;                       // cylinder full length
    float Vc = M_PI_F * r * r * L;             // cylinder volume
    float Vs = (4.0 / 3.0) * M_PI_F * r * r * r;   // both hemispheres = one sphere
    float V  = max(Vc + Vs, 1e-12);
    float mc = Vc / V, ms = Vs / V;            // mass fractions (per unit mass)
    float Iyy_k = mc * 0.5 * r * r + ms * 0.4 * r * r;
    float Ixx_k = mc * (L * L / 12.0 + 0.25 * r * r)
                + ms * (0.4 * r * r + 0.25 * L * L + 0.375 * L * r);
    return float3(invMass / max(Ixx_k, 1e-10),
                  invMass / max(Iyy_k, 1e-10),
                  invMass / max(Ixx_k, 1e-10));
}

// Shape-aware inverse inertia for a body.
static float3 cdBodyInvInertia(CoinBody c, float invMass) {
    if (cdIsBox(c)) return cdBoxInvInertia(c.shapeExtents.xyz, invMass);
    if (cdIsCapsule(c)) return cdCapsuleInvInertia(cdCapsuleR(c), cdCapsuleHL(c), invMass);
    // Hull: the exact per-unit-mass inverse inertia diag (principal frame) was
    // integrated over the solid hull at registration and rides hullRef.yzw.
    if (cdIsHull(c)) return invMass * c.hullRef.yzw;
    if (cdIsSphere(c)) {
        // Solid sphere: I = (2/5) m R², isotropic ⇒ I⁻¹ = invMass / (0.4 R²) on every axis.
        float R = c.prevPos.w > 1e-4 ? c.prevPos.w : 0.12;
        float invI = invMass / max(0.4 * R * R, 1e-8);
        return float3(invI, invI, invI);
    }
    return cdInvInertiaLocal(c.prevPos.w > 1e-4 ? c.prevPos.w : 0.12,
                             c.vel.w > 1e-5 ? c.vel.w : 0.009, invMass);
}

// World-space inverse-inertia applied to a world vector: R · (I⁻¹_local ⊙ (Rᵀ a)).
static float3 cdApplyInvInertiaWorld(float4 q, float3 invIlocal, float3 a) {
    float3 la = cdQuatRotateInv(q, a);
    return cdQuatRotate(q, la * invIlocal);
}

// Flat grid cell index for a world position (clamped into the grid).
static uint cdCellIndex(float3 p, constant CoinUniforms& u) {
    int3 c = int3(floor((p - float3(u.gridMinX, u.gridMinY, u.gridMinZ)) * u.invCell));
    int3 res = int3(int(u.gridResX), int(u.gridResY), int(u.gridResZ));
    c = clamp(c, int3(0), res - 1);
    return uint(c.x) + u.gridResX * (uint(c.y) + u.gridResY * uint(c.z));
}

static float cdScaleOf(CoinBody c) { return c.prevPos.w > 0.01 ? c.prevPos.w : 1.0; }

// Per-body collision dimensions (mixed-asset pile). radius → prevPos.w,
// halfThickness → vel.w; both written at spawn and only ever read here (the
// kernels write .xyz of those lanes, never .w). Fallback to a coin-ish default if
// a body was spawned before the dims were set.
static float cdRadiusOf(CoinBody c)    { return c.prevPos.w > 1e-4 ? c.prevPos.w : 0.12; }
static float cdHalfThickOf(CoinBody c) { return c.vel.w     > 1e-5 ? c.vel.w     : 0.009; }

// ── Segment / closest-point helpers (capsule narrowphase) ─────────────────────

// Closest point on segment [a,b] to point p.
static float3 cdClosestOnSegment(float3 a, float3 b, float3 p) {
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return a + t * ab;
}

// Closest points between segments [p1,q1] / [p2,q2] (Ericson, RTCD §5.1.9).
static void cdClosestSegSeg(float3 p1, float3 q1, float3 p2, float3 q2,
                            thread float3& c1, thread float3& c2) {
    float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s = 0.0, t = 0.0;
    if (a <= 1e-12 && e <= 1e-12) { c1 = p1; c2 = p2; return; }
    if (a <= 1e-12) {
        t = clamp(f / e, 0.0, 1.0);
    } else {
        float c = dot(d1, r);
        if (e <= 1e-12) {
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            s = (denom > 1e-12) ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0)      { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
            else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

// Closest point ON-or-inside shape O's solid volume to the O-local point `lp`
// (box: AABB clamp; disc: radial+axial clamp on the capped cylinder). The shared
// core of every "sphere-probe vs shape" contact (sphere bodies, capsule probe
// centres). For a point INSIDE the shape this returns `lp` itself — callers
// needing an interior push-out use the SDF/box-push routines instead.
static float3 cdClosestInShapeLocal(CoinBody O, float3 lp) {
    if (cdIsBox(O)) {
        float3 heO = O.shapeExtents.xyz;
        return clamp(lp, -heO, heO);
    }
    float Ro = cdRadiusOf(O), hO = cdHalfThickOf(O);
    float radial = length(lp.xz);
    float2 rd = radial > 1e-6 ? lp.xz / radial : float2(1, 0);
    return float3(rd.x * min(radial, Ro), clamp(lp.y, -hO, hO), rd.y * min(radial, Ro));
}

// A capsule's world-space segment endpoints (axis = local +Y, half-length hl).
static void cdCapsuleSegment(CoinBody c, thread float3& a, thread float3& b) {
    float3 ax = cdQuatRotate(c.orient, float3(0, cdCapsuleHL(c), 0));
    a = c.posInvMass.xyz - ax;
    b = c.posInvMass.xyz + ax;
}


// ── KERNEL: integrate (predict COM + advance orientation) ─────────────────────

kernel void coinIntegrate(
    device CoinBody*      coins [[ buffer(0) ]],
    constant CoinUniforms& u    [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) return;  // inactive slot

    // Linear: gravity + drag, predict COM.
    float3 x = c.posInvMass.xyz;
    float3 v = c.vel.xyz;
    v.y -= u.gravity * u.dt;
    // Real ∝v² aerodynamic drag (opt-in): negligible when slow, biting when fast —
    // unlike the uniform `linDamping` bleed, which sheds the same FRACTION of speed
    // at every velocity. With drag enabled a scene sets linDamping≈1 so the only
    // airborne energy loss is this physically-shaped term.
    if (u.quadraticDrag > 0.0) {
        v -= cdDragK(u, c.prevPos.w) * length(v) * v * u.dt;   // drag ∝ 1/r per body
    }
    v   *= u.linDamping;
    coins[id].prevPos.xyz    = x;
    coins[id].posInvMass.xyz = x + v * u.dt;
    coins[id].vel.xyz        = v;

    // Angular: light damping, advance orientation by ω. Store prevOrient so
    // finalize can recover ω from the (contact-corrected) quaternion delta.
    float3 omega = c.angVel.xyz * u.angDamping;
    float4 q = c.orient;
    coins[id].prevOrient = q;
    coins[id].orient     = cdIntegrateQuat(q, omega, u.dt);
    coins[id].angVel.xyz = omega;
}

// ── KERNELS: counting-sort spatial hash on coin COM (clone of MLSMPM pattern) ──

kernel void coinCellClear(
    device atomic_uint* cellCounts [[ buffer(0) ]],
    uint idx [[ thread_position_in_grid ]])
{
    atomic_store_explicit(&cellCounts[idx], 0u, memory_order_relaxed);
}

kernel void coinCellCount(
    device const CoinBody* coins      [[ buffer(0) ]],
    device atomic_uint*    cellCounts [[ buffer(1) ]],
    constant CoinUniforms& u          [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    if (coins[id].posInvMass.w == 0.0) return;
    uint cell = cdCellIndex(coins[id].posInvMass.xyz, u);
    atomic_fetch_add_explicit(&cellCounts[cell], 1u, memory_order_relaxed);
}

kernel void coinCellOffsetsScan(
    device uint*           cellCounts  [[ buffer(0) ]],
    device uint*           cellOffsets [[ buffer(1) ]],
    constant CoinUniforms& u           [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid != 0) return;
    uint total = u.gridResX * u.gridResY * u.gridResZ;
    uint sum = 0;
    for (uint i = 0; i < total; ++i) {
        cellOffsets[i] = sum;
        sum += cellCounts[i];
        cellCounts[i] = 0;   // reused as write cursor by coinScatter
    }
    cellOffsets[total] = sum;
}

kernel void coinScatter(
    device const CoinBody* coins         [[ buffer(0) ]],
    device atomic_uint*    cellCursors   [[ buffer(1) ]],   // == cellCounts, zeroed by scan
    device const uint*     cellOffsets   [[ buffer(2) ]],
    device uint*           sortedIndices [[ buffer(3) ]],
    constant CoinUniforms& u             [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    if (coins[id].posInvMass.w == 0.0) return;
    uint cell = cdCellIndex(coins[id].posInvMass.xyz, u);
    uint slot = atomic_fetch_add_explicit(&cellCursors[cell], 1u, memory_order_relaxed);
    sortedIndices[cellOffsets[cell] + slot] = id;
}

// ── Coin feature points (surface samples used for contact generation) ─────────
//
// 14 points: face centres + an INWARD face ring (radius 0.6R) on both faces +
// an EQUATORIAL rim ring (radius R, y = 0). Local frame: disk normal = +Y, faces
// at y = ±h.
//
// Why the face ring is at 0.6R, not the rim R: for equal-radius coins a point at
// the rim lands on the neighbour's cylindrical SIDE, where the radial penetration
// (≈0) is the minimum-translation direction — so a stacking contact would resolve
// SIDEWAYS and the coins sink together coplanar. Pulling the face samples inward to
// 0.6R puts them safely inside the neighbour's CAP region, so face-face contact
// resolves VERTICALLY (true stacking). The equatorial ring (radius R, y=0) is what
// catches genuine edge/side contacts, which correctly resolve radially. Both faces
// are sampled so a middle coin in a 3-stack is pushed both up (off the coin below)
// and down (off the coin above).
constant int   CD_FRING = 4;          // face-ring samples per face
constant int   CD_EQ    = 4;          // equatorial samples
constant int   CD_NPTS  = 2 + 2 * CD_FRING + CD_EQ;   // 14
constant float CD_FACE_R = 0.6;       // face-ring radius fraction
// Widest per-body probe count the static-collider loop's `nFP` ever reaches — a
// hull's real vertex count, capped (`min(hRange.y, 32u)` at its computation
// site). CD_NPTS (14, box/disc feature points) is narrower; any fixed-size
// per-point cache in that loop must size to THIS, or a >14-vertex hull
// overflows it.
constant int   CD_MAX_STATIC_FP = 32;

static float3 cdFeaturePoint(int i, float R, float h) {
    if (i == 0) return float3(0.0,  h, 0.0);   // top centre
    if (i == 1) return float3(0.0, -h, 0.0);   // bottom centre
    int k = i - 2;
    if (k < 2 * CD_FRING) {
        int ring = k / CD_FRING;               // 0 = top face, 1 = bottom face
        int j = k - ring * CD_FRING;
        float a = (float(j) + (ring == 1 ? 0.5 : 0.0)) / float(CD_FRING) * 6.2831853;
        float y = (ring == 0) ? h : -h;
        return float3(CD_FACE_R * R * cos(a), y, CD_FACE_R * R * sin(a));
    }
    int j = k - 2 * CD_FRING;                  // equatorial sample
    float a = (float(j) + 0.5) / float(CD_EQ) * 6.2831853;
    return float3(R * cos(a), 0.0, R * sin(a));
}

// Box surface feature points (local): the 8 corners + 6 face centres = 14 (== CD_NPTS).
// Sampling corners is what lets a dropped box tip onto a face — one corner contacts
// the floor first, gets pushed out, and the offset correction rotates the box flat.
static float3 cdBoxFeaturePoint(int i, float3 he) {
    if (i < 8) {
        return float3((i & 1) ? he.x : -he.x,
                      (i & 2) ? he.y : -he.y,
                      (i & 4) ? he.z : -he.z);
    }
    int f = i - 8;
    if (f == 0) return float3( he.x, 0, 0);
    if (f == 1) return float3(-he.x, 0, 0);
    if (f == 2) return float3(0,  he.y, 0);
    if (f == 3) return float3(0, -he.y, 0);
    if (f == 4) return float3(0, 0,  he.z);
    return            float3(0, 0, -he.z);
}

// Shape-aware surface feature point in the body's local frame.
static float3 cdBodyFeaturePoint(int i, CoinBody c) {
    if (cdIsBox(c)) return cdBoxFeaturePoint(i, c.shapeExtents.xyz);
    return cdFeaturePoint(i, cdRadiusOf(c), cdHalfThickOf(c));
}

// Signed distance + outward normal (local) of a capped cylinder (axis +Y).
// Returns sd (<0 inside). `nLocal` is the unit push-out direction when inside.
static float cdCappedCylSDF(float3 p, float R, float h, thread float3& nLocal) {
    float radial = length(p.xz);
    float dRad = radial - R;       // <0 inside the infinite cylinder
    float dCap = abs(p.y) - h;     // <0 between the two faces
    if (dRad < 0.0 && dCap < 0.0) {
        // Inside: minimum-translation exit is the nearer face.
        float penRad = -dRad;      // R - radial
        float penCap = -dCap;      // h - |y|
        if (penRad < penCap) {
            float2 rd = (radial > 1e-6) ? (p.xz / radial) : float2(1.0, 0.0);
            nLocal = float3(rd.x, 0.0, rd.y);
            return -penRad;
        } else {
            nLocal = float3(0.0, (p.y >= 0.0) ? 1.0 : -1.0, 0.0);
            return -penCap;
        }
    }
    // Outside: positive distance (we don't need the exterior normal — no contact).
    float2 d2 = float2(max(dRad, 0.0), max(dCap, 0.0));
    nLocal = float3(0.0, 1.0, 0.0);
    return length(d2) + min(max(dRad, dCap), 0.0);
}

// Accumulate one XPBD contact for coin i: contact at world point `pw`, world
// normal `n` (pointing OUT of the other body, toward i), penetration `depth`,
// other body's generalized inverse mass `wOther` (0 for static). Splits the depth
// between the two bodies via their generalized inverse masses; applies i's share.
static void cdAccumContact(float3 pw, float3 n, float depth,
                           float3 xi, float4 qi, float invMi, float3 invIi,
                           float wOther,
                           thread float3& dPos, thread float3& dRot,
                           thread float& count, thread bool& support,
                           thread float3& supportNormal,
                           thread float3& contactNormal)
{
    float3 r = pw - xi;
    float3 rxn = cross(r, n);
    float wi = invMi + dot(rxn, cdApplyInvInertiaWorld(qi, invIi, rxn));
    float denom = wi + wOther;
    if (denom < 1e-8) return;
    float corr = depth / denom;            // shared positional magnitude
    float3 P = corr * n;
    dPos += invMi * P;
    dRot += cdApplyInvInertiaWorld(qi, invIi, cross(r, P));
    count += 1.0;
    // Σ of upward contact normals → leveling target (resting support footprint).
    if (n.y > 0.30) { support = true; supportNormal += n; }
    // Σ of ALL contact normals, depth-weighted → the impact axis finalize bounces
    // along (a wall hit is horizontal, a floor hit vertical — restitution must act
    // along whichever dominates, not always world-Y).
    contactNormal += n * depth;
}

// Generalized inverse mass of a body at contact point pw along normal n.
static float cdGenInvMass(float3 pw, float3 xj, float4 qj, float invMj, float3 invIj, float3 n) {
    float3 r = pw - xj;
    float3 rxn = cross(r, n);
    return invMj + dot(rxn, cdApplyInvInertiaWorld(qj, invIj, rxn));
}

// A body's oriented-box half-extents: a real box's extents, or a disc's bounding box
// (R, h, R — thin axis = local +Y, matching the disc), so a disc can be collided as a
// box against a box without a separate box–cylinder routine.
static float3 cdBodyHalfExtents(CoinBody c) {
    if (cdIsBox(c)) return c.shapeExtents.xyz;
    return float3(cdRadiusOf(c), cdHalfThickOf(c), cdRadiusOf(c));
}

// Oriented box–box SAT (15 axes: 3+3 face normals + 9 edge×edge). Returns the
// minimum-penetration depth, the contact normal `n` (from j toward i), and a contact
// point at the average of the two boxes' support points along n. Used for any pair
// where at least one body is a box (discs collide via their bounding box). Returns
// false if a separating axis exists (no contact).
static bool cdBoxBoxSAT(float3 xi, float4 qi, float3 heI,
                        float3 xj, float4 qj, float3 heJ,
                        thread float& outDepth, thread float3& outN, thread float3& outCp) {
    float3 A[3] = { cdQuatRotate(qi, float3(1,0,0)), cdQuatRotate(qi, float3(0,1,0)), cdQuatRotate(qi, float3(0,0,1)) };
    float3 B[3] = { cdQuatRotate(qj, float3(1,0,0)), cdQuatRotate(qj, float3(0,1,0)), cdQuatRotate(qj, float3(0,0,1)) };
    float3 D = xi - xj;
    float  minDepth = 1e30;
    float3 bestN = float3(0,1,0);
    for (int k = 0; k < 15; ++k) {
        float3 a;
        if      (k < 3) a = A[k];
        else if (k < 6) a = B[k - 3];
        else { int e = k - 6; a = cross(A[e / 3], B[e % 3]); }
        float al = length(a);
        if (al < 1e-6) continue;                 // parallel edges → degenerate axis
        a /= al;
        float rA = abs(dot(A[0], a)) * heI.x + abs(dot(A[1], a)) * heI.y + abs(dot(A[2], a)) * heI.z;
        float rB = abs(dot(B[0], a)) * heJ.x + abs(dot(B[1], a)) * heJ.y + abs(dot(B[2], a)) * heJ.z;
        float depth = (rA + rB) - abs(dot(D, a));
        if (depth <= 0.0) return false;          // separating axis
        if (depth < minDepth) { minDepth = depth; bestN = a * (dot(D, a) >= 0.0 ? 1.0 : -1.0); }
    }
    outN = normalize(bestN);
    float rA = abs(dot(A[0], outN)) * heI.x + abs(dot(A[1], outN)) * heI.y + abs(dot(A[2], outN)) * heI.z;
    float rB = abs(dot(B[0], outN)) * heJ.x + abs(dot(B[1], outN)) * heJ.y + abs(dot(B[2], outN)) * heJ.z;
    outCp = 0.5 * ((xi - outN * rA) + (xj + outN * rB));
    outDepth = minDepth;
    return true;
}

// Push a world point `pw` out of an oriented box (centre cj, orient qj, half-extents
// heJ): if inside, returns the world push-out normal (out of the box, toward pw) and
// the penetration depth along the nearest face. The dynamic counterpart of the static
// box collider's point-push — used to build a box–box contact MANIFOLD (one box's
// corners vs the other box), which a single SAT contact can't: stacks need a support
// footprint (≥3 points) to resist tipping, exactly as box-vs-static rests flat on its
// 4 bottom corners.
static bool cdOrientedBoxPush(float3 pw, float3 cj, float4 qj, float3 heJ,
                              thread float3& outN, thread float& outDepth) {
    float3 lp = cdQuatRotateInv(qj, pw - cj);    // point in box-local frame
    float3 d  = heJ - abs(lp);
    if (d.x <= 0.0 || d.y <= 0.0 || d.z <= 0.0) return false;   // outside
    float3 nLocal; float push;
    if (d.x < d.y && d.x < d.z) { nLocal = float3(sign(lp.x), 0, 0); push = d.x; }
    else if (d.y < d.z)         { nLocal = float3(0, sign(lp.y), 0); push = d.y; }
    else                        { nLocal = float3(0, 0, sign(lp.z)); push = d.z; }
    outN = cdQuatRotate(qj, nLocal);             // world normal, out of box j toward pw
    outDepth = push;
    return true;
}

// Speculative-margin sibling of `cdOrientedBoxPush`: an inside-only test misses a
// fast thin point that tunnels a thin wall between substeps (the point never
// registers as "inside" at either sample). Inside behaves exactly like
// `cdOrientedBoxPush` (positive outDepth, push-out normal). Outside, instead of
// unconditionally returning false, it finds the TRUE exterior closest point
// (clamp to the half-extents — same core as `cdClosestInShapeLocal`'s box case)
// and reports a near-contact (negative outDepth = gap) when that gap is under
// `spec`, using cdEmitContact's `pen > -spec` convention throughout. `spec <= 0`
// degrades to the exact inside-only behaviour of `cdOrientedBoxPush`.
static bool cdOrientedBoxPushSpeculative(float3 pw, float3 cj, float4 qj, float3 heJ, float spec,
                                         thread float3& outN, thread float& outDepth) {
    float3 lp = cdQuatRotateInv(qj, pw - cj);    // point in box-local frame
    float3 d  = heJ - abs(lp);
    if (d.x > 0.0 && d.y > 0.0 && d.z > 0.0) {    // inside: push out (unchanged)
        float3 nLocal; float push;
        if (d.x < d.y && d.x < d.z) { nLocal = float3(sign(lp.x), 0, 0); push = d.x; }
        else if (d.y < d.z)         { nLocal = float3(0, sign(lp.y), 0); push = d.y; }
        else                        { nLocal = float3(0, 0, sign(lp.z)); push = d.z; }
        outN = cdQuatRotate(qj, nLocal);
        outDepth = push;
        return true;
    }
    if (spec <= 0.0) return false;                // no margin ⇒ exact cdOrientedBoxPush behaviour
    float3 cl = clamp(lp, -heJ, heJ);
    float3 dl3 = lp - cl;
    float dist = length(dl3);
    if (dist >= spec || dist < 1e-8) return false;
    outN = cdQuatRotate(qj, dl3 / dist);          // world normal, out of box j toward pw
    outDepth = -dist;                             // negative gap
    return true;
}

// ── KERNEL: contact solve (Jacobi — read state, accumulate per-coin correction) ─

kernel void coinContactSolve(
    device const CoinBody*           coins         [[ buffer(0) ]],
    device const uint*               sortedIndices [[ buffer(1) ]],
    device const uint*               cellOffsets   [[ buffer(2) ]],
    device const CoinStaticCollider* colliders     [[ buffer(3) ]],
    constant CoinUniforms&           u             [[ buffer(4) ]],
    device float4*                   coinDelta     [[ buffer(5) ]],   // 2 per coin: [pos+count, rot+support]
    device const int2*               links         [[ buffer(6) ]],   // articulation: skip the joint partner
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) { return; }
    CoinBody ci = coins[id];
    if (ci.posInvMass.w == 0.0) {
        coinDelta[4*id] = float4(0); coinDelta[4*id+1] = float4(0);
        coinDelta[4*id+2] = float4(0); coinDelta[4*id+3] = float4(0);
        return;
    }
    int jointPartner = links[id].x;   // the one neighbour this body must NOT collide with

    float3 xi = ci.posInvMass.xyz;
    float4 qi = ci.orient;
    float  invMi = ci.posInvMass.w;            // 1 for active coins
    // Per-body collision dimensions: a mixed pile (coins / jewels / franks) packs
    // bodies of different cylinder aspect ratios into ONE solver. radius rides
    // prevPos.w, halfThickness rides vel.w (both written at spawn, preserved by
    // every kernel — they only write .xyz). The SAT below is already general in
    // (R,h) per body, so heterogeneous shapes collide correctly.
    float  Ri = cdRadiusOf(ci);
    float  hi = cdHalfThickOf(ci);
    bool   iIsBox = cdIsBox(ci);
    float3 invIi = cdBodyInvInertia(ci, invMi);

    // Precompute i's feature points in world space (shape-aware: disc rim/face samples
    // or box corners/face-centres).
    float3 fp[CD_NPTS];
    for (int p = 0; p < CD_NPTS; ++p) {
        fp[p] = xi + cdQuatRotate(qi, cdBodyFeaturePoint(p, ci));
    }

    float3 dPos = float3(0), dRot = float3(0);
    float  count = 0.0;
    bool   support = false;
    float3 supportNormal = float3(0);   // Σ of upward contact normals → leveling target
    float3 contactNormal = float3(0);   // Σ of ALL contact normals (depth-weighted) → restitution axis

    // ── Coin–coin via the 3×3×3 neighbourhood of the hash ─────────────────────
    //
    // ANALYTIC oriented-cylinder contact (SAT over the two face normals + the
    // centre-line). Surface-point sampling fails here: two equal coaxial coins put
    // every sample point on the other's surface at ~zero depth, so coplanar is a
    // false equilibrium and they interpenetrate. The SAT overlap gives a real
    // depth (≈2h at coplanar) along the minimum-penetration axis, so they pop apart
    // — applied at the contact centroid so an OFFSET pair also gets the tipping
    // torque (coaxial → r∥n → no torque → stable flat stack).
    float3 ni = normalize(cdQuatRotate(qi, float3(0,1,0)));
    int3 base = int3(floor((xi - float3(u.gridMinX, u.gridMinY, u.gridMinZ)) * u.invCell));
    int3 res  = int3(int(u.gridResX), int(u.gridResY), int(u.gridResZ));
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int3 c = base + int3(dx, dy, dz);
        if (any(c < int3(0)) || any(c >= res)) continue;
        uint cell = uint(c.x) + u.gridResX * (uint(c.y) + u.gridResY * uint(c.z));
        uint start = cellOffsets[cell];
        uint end   = cellOffsets[cell + 1u];
        for (uint s = start; s < end; ++s) {
            uint j = sortedIndices[s];
            if (j == id) continue;
            if (int(j) == jointPartner) continue;   // joint partners overlap by design; don't fight the joint
            CoinBody cj = coins[j];
            if (cj.posInvMass.w == 0.0) continue;
            float3 xj = cj.posInvMass.xyz;
            float4 qj = cj.orient;
            float  Rj = cdRadiusOf(cj);
            float  hj = cdHalfThickOf(cj);
            float3 D  = xi - xj;
            // Cheap reject: bounding spheres (radius √(R²+h²)).
            float reach = sqrt(Ri*Ri + hi*hi) + sqrt(Rj*Rj + hj*hj);
            if (dot(D, D) > reach * reach) continue;
            float  invMj = cj.posInvMass.w;
            float3 invIj = cdBodyInvInertia(cj, invMj);

            // ── Sphere-involved pair: exact, ORIENTATION-INDEPENDENT contact ───────
            // A sphere's closest point is always center − R·n, so the contact is a
            // single point with an exact normal (no rim/cap feature-point sampling,
            // which is what makes a coin-shaped stand-in float/clip at random tilts).
            if (cdIsSphere(ci) && cdIsSphere(cj)) {
                // Sphere↔sphere — exact.
                float  dl  = length(D);
                float  pen = (Ri + Rj) - dl;
                if (pen > 0.0 && dl > 1e-6) {
                    float3 nn = D / dl;                       // points OUT of j, toward i
                    float3 cp = 0.5 * ((xi - nn * Ri) + (xj + nn * Rj));
                    float  wO = cdGenInvMass(cp, xj, qj, invMj, invIj, nn);
                    cdAccumContact(cp, nn, pen, xi, qi, invMi, invIi, wO,
                                   dPos, dRot, count, support, supportNormal, contactNormal);
                }
                continue;
            }
            if (cdIsSphere(ci) || cdIsSphere(cj)) {
                // Sphere ↔ disc/box — REAL closest-point contact (not a bounding-sphere
                // fallback, which over-separated: a marble resting on a flat coin was held
                // a full √(R²+h²)−h ≈ 5 cm too high, then fell back → a permanent limit
                // cycle, the marble-on-coin / marble-on-die rest jitter). The sphere is a
                // point+radius, so the exact contact is the closest point on the OTHER
                // body's real surface (OBB clamp for a box; radial+axial clamp on the
                // capped cylinder for a disc) vs the sphere centre.
                bool   iSphere = cdIsSphere(ci);
                float3 cs   = iSphere ? xi : xj;             // sphere centre
                float  rs   = iSphere ? Ri : Rj;             // sphere radius
                float3 co   = iSphere ? xj : xi;             // shape centre
                float4 qo   = iSphere ? qj : qi;             // shape orient
                CoinBody O  = iSphere ? cj : ci;             // the shape body
                float3 lp   = cdQuatRotateInv(qo, cs - co);  // sphere centre in shape-local
                float3 closestLocal;
                if (cdIsBox(O)) {
                    float3 heO = cdBodyHalfExtents(O);
                    closestLocal = clamp(lp, -heO, heO);
                } else {
                    float Ro = cdRadiusOf(O), hO = cdHalfThickOf(O);   // capped cylinder, axis +Y
                    float radial = length(lp.xz);
                    float2 rd = radial > 1e-6 ? lp.xz / radial : float2(1, 0);
                    closestLocal = float3(rd.x * min(radial, Ro), clamp(lp.y, -hO, hO), rd.y * min(radial, Ro));
                }
                float3 deltaLocal = lp - closestLocal;
                float  dist = length(deltaLocal);
                float  pen  = rs - dist;
                if (pen > 0.0 && dist > 1e-6) {
                    float3 nOut = cdQuatRotate(qo, deltaLocal / dist);   // out of shape, toward sphere
                    float3 cp   = co + cdQuatRotate(qo, closestLocal);   // contact point on the shape
                    float3 n    = iSphere ? nOut : -nOut;               // must point toward i (this thread)
                    float  wO   = cdGenInvMass(cp, xj, qj, invMj, invIj, n);
                    cdAccumContact(cp, n, pen, xi, qi, invMi, invIi, wO,
                                   dPos, dRot, count, support, supportNormal, contactNormal);
                }
                continue;
            }

            float  minDepth;
            float3 n, cp;
            if (iIsBox || cdIsBox(cj)) {
                // ── DISC–BOX / BOX–BOX → a corner MANIFOLD: test this body's feature
                // points (a box's 8 corners) against the OTHER oriented box, each
                // penetrating point a contact. A single SAT contact can't hold a stack
                // (one central point gives no restoring torque, so a tower tips); the
                // multi-point footprint resists tipping, exactly as box-vs-static rests
                // flat on its bottom corners. A disc is collided as its bounding box —
                // kept on purpose because the box engages early/shallow (it circumscribes
                // the cylinder), which is what keeps a dense chaotic fill STABLE (a hard
                // real-cylinder SDF lets deep overlaps form then EJECTS them → the pile
                // explodes). BUT the disc's bounding box has 4 corner COLUMNS, radial
                // R..R√2, that the round disc doesn't occupy; a box corner landing there
                // is a contact the real disc lacks → gravity undoes it every frame → a
                // permanent limit cycle (the coin↔die rest jitter). Reject those — within
                // radius R the box cap and the cylinder cap COINCIDE, so the rest is exact.
                bool jIsDisc = !cdIsBox(cj);   // sphere handled earlier ⇒ j is disc or box
                float3 heJ = cdBodyHalfExtents(cj);
                int added = 0;
                for (int p = 0; p < 8; ++p) {
                    float3 pw = fp[p];
                    float3 nB; float depthB;
                    if (cdOrientedBoxPush(pw, xj, qj, heJ, nB, depthB)) {
                        if (jIsDisc) {
                            float3 lp = cdQuatRotateInv(qj, pw - xj);
                            if (length(lp.xz) > Rj) continue;   // phantom corner of the disc's bbox
                        }
                        float wO = cdGenInvMass(pw, xj, qj, invMj, invIj, nB);
                        cdAccumContact(pw, nB, depthB, xi, qi, invMi, invIi, wO,
                                       dPos, dRot, count, support, supportNormal, contactNormal);
                        added++;
                    }
                }
                // Edge–edge (no corner of i is inside j) → fall back to the single SAT
                // contact so the pair still separates.
                if (added == 0 &&
                    cdBoxBoxSAT(xi, qi, cdBodyHalfExtents(ci), xj, qj, heJ, minDepth, n, cp)) {
                    float wO = cdGenInvMass(cp, xj, qj, invMj, invIj, n);
                    cdAccumContact(cp, n, minDepth, xi, qi, invMi, invIi, wO,
                                   dPos, dRot, count, support, supportNormal, contactNormal);
                }
                continue;   // pair handled by the manifold
            } else {
                // Analytic disc–disc SAT (both face normals + the centre line). Surface-
                // point sampling fails here: two equal coaxial coins put every sample on
                // the other's surface at ~zero depth, a false coplanar equilibrium. The
                // SAT overlap gives a real depth (≈2h coplanar) along the min-penetration
                // axis so they pop apart, applied at the contact centroid so an OFFSET
                // pair also gets the tipping torque (coaxial → r∥n → no torque → stable).
                float3 nj = normalize(cdQuatRotate(qj, float3(0,1,0)));
                float3 axes[3] = { ni, nj, float3(0,1,0) };
                float dlen = length(D);
                if (dlen > 1e-5) axes[2] = D / dlen;
                float  md = 1e30;
                float3 bestN = float3(0,1,0);
                bool   separated = false;
                for (int ax = 0; ax < 3; ++ax) {
                    float3 a = axes[ax];
                    float di = abs(dot(ni, a)), dj = abs(dot(nj, a));
                    float ei = hi * di + Ri * sqrt(max(0.0, 1.0 - di*di));
                    float ej = hj * dj + Rj * sqrt(max(0.0, 1.0 - dj*dj));
                    float proj = dot(D, a);
                    float depth = (ei + ej) - abs(proj);
                    if (depth <= 0.0) { separated = true; break; }
                    if (depth < md) { md = depth; bestN = a * (proj >= 0.0 ? 1.0 : -1.0); }
                }
                if (separated) continue;
                n = normalize(bestN);
                // REAL contact point = average of the two discs' SUPPORT points along n
                // (each disc's surface point deepest into the other) — correct lever arm.
                float dniN = dot(ni, n), dnjN = dot(nj, n);
                float eiN = hi * abs(dniN) + Ri * sqrt(max(0.0, 1.0 - dniN * dniN));
                float ejN = hj * abs(dnjN) + Rj * sqrt(max(0.0, 1.0 - dnjN * dnjN));
                cp = 0.5 * ((xi - n * eiN) + (xj + n * ejN));
                minDepth = md;
            }
            float wOther = cdGenInvMass(cp, xj, qj, invMj, invIj, n);
            cdAccumContact(cp, n, minDepth, xi, qi, invMi, invIi, wOther,
                           dPos, dRot, count, support, supportNormal,
                           contactNormal);
        }
    }

    // ── Static environment ────────────────────────────────────────────────────
    for (uint k = 0u; k < u.colliderCount; ++k) {
        CoinStaticCollider col = colliders[k];
        uint kind = as_type<uint>(col.a.w);

        if (kind == 2u) {
            // Forward-only pusher plate: shove coins the plate has overtaken to just
            // in front of its +Z face, at most plateSpeed·dt per substep (so the
            // contact-derived velocity can't exceed the plate speed and launch
            // coins). Applied at each feature point inside the plate footprint so
            // the shove imparts a little tumble, not a rigid slab translation.
            float3 cc = col.a.xyz;
            float3 he = col.b.xyz;
            float frontZ = cc.z + he.z;
            float backZ  = cc.z - he.z;
            float maxPush = max(0.0, col.vel.z) * u.dt;
            for (int p = 0; p < CD_NPTS; ++p) {
                float3 pw = fp[p];
                if (abs(pw.x - cc.x) < he.x + Ri &&
                    pw.y > cc.y - (he.y + hi) && pw.y < cc.y + (he.y + hi) &&
                    pw.z < frontZ + Ri && pw.z > backZ - Ri) {
                    float depth = clamp((frontZ + Ri) - pw.z, 0.0, maxPush);
                    if (depth > 0.0) {
                        cdAccumContact(pw, float3(0,0,1), depth, xi, qi, invMi, invIi, 0.0,
                                       dPos, dRot, count, support, supportNormal, contactNormal);
                    }
                }
            }
            continue;
        }

        if (kind == 0u) {
            // Half-space n·p ≥ d. Push any feature point behind the plane out.
            float3 n = col.a.xyz;
            float  d = col.b.w;
            if (cdIsSphere(ci)) {
                // Sphere: the single deepest point is center − R·n, so rest clearance
                // is EXACTLY R in every orientation — no tilted rim corner to float on,
                // no cap-flat to clip through. pen = (d + R) − n·center.
                float pen = (d + Ri) - dot(n, xi);
                if (pen > 0.0) {
                    cdAccumContact(xi - Ri * n, n, pen, xi, qi, invMi, invIi, 0.0,
                                   dPos, dRot, count, support, supportNormal, contactNormal);
                } else if (pen > -0.04 * Ri) {
                    // TOUCHING (just-resolved) contact — register PRESENCE without a
                    // positional push. A single sphere contact fully de-penetrates
                    // within the Jacobi iteration loop, so by the LAST iteration pen≤0
                    // and `support`/`contactNormal` would vanish — finalize then never
                    // fires its impact restitution and the ball dead-stops instead of
                    // bouncing (a coin's multi-point manifold never fully clears, which
                    // is why only spheres hit this). Flag the contact within a hair of
                    // the surface so the bounce (floor: `support`; wall/ceiling: `nCl`)
                    // still triggers. No count++ → real corrections aren't diluted.
                    if (n.y > 0.30) { support = true; supportNormal += n; }
                    contactNormal += n * max(Ri * 0.01, 1e-4);   // tiny weight → nCl > 1e-6, axis = n
                }
            } else {
                for (int p = 0; p < CD_NPTS; ++p) {
                    float3 pw = fp[p];
                    float pen = d - dot(n, pw);
                    if (pen > 0.0) {
                        cdAccumContact(pw, n, pen, xi, qi, invMi, invIi, 0.0,
                                       dPos, dRot, count, support, supportNormal, contactNormal);
                    }
                }
            }
        } else {
            // Axis-aligned box. One-way ledges only push from the +normal side.
            float3 ctr = col.a.xyz;
            float3 he  = col.b.xyz;
            bool oneWay = (col.meta.x & 1u) != 0u;
            for (int p = 0; p < CD_NPTS; ++p) {
                float3 pw = fp[p];
                float3 dd = pw - ctr;
                if (abs(dd.x) >= he.x || abs(dd.y) >= he.y || abs(dd.z) >= he.z) continue; // outside
                // Inside: push out along the nearest face.
                float3 dist = he - abs(dd);
                float3 n; float push;
                if (dist.x < dist.y && dist.x < dist.z) { n = float3(sign(dd.x),0,0); push = dist.x; }
                else if (dist.y < dist.z)               { n = float3(0,sign(dd.y),0); push = dist.y; }
                else                                    { n = float3(0,0,sign(dd.z)); push = dist.z; }
                if (oneWay && n.y < 0.5) continue;   // top-only ledge
                cdAccumContact(pw, n, push, xi, qi, invMi, invIi, 0.0,
                               dPos, dRot, count, support, supportNormal, contactNormal);
            }
        }
    }

    coinDelta[4*id]     = float4(dPos, count);
    coinDelta[4*id + 1] = float4(dRot, support ? 1.0 : 0.0);
    coinDelta[4*id + 2] = float4(supportNormal, 0.0);   // Σ upward contact normals (leveling target)
    coinDelta[4*id + 3] = float4(contactNormal, 0.0);   // Σ all contact normals, depth-weighted (restitution axis)
}

// ── KERNEL: apply accumulated correction (separate pass keeps the solve race-free) ─

kernel void coinApplyDelta(
    device CoinBody*       coins     [[ buffer(0) ]],
    device const float4*   coinDelta [[ buffer(1) ]],
    constant CoinUniforms& u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    if (coins[id].posInvMass.w == 0.0) return;
    float4 dp = coinDelta[4*id];
    float  count = dp.w;
    if (count < 0.5) return;
    float inv = u.contactRelax / count;       // AVERAGE the per-point corrections, then relax

    // Linear: TIGHT per-apply clamp (≈ one coin thickness). Bounding the per-apply
    // move is what keeps the distributed face manifold stable under Jacobi — a loose
    // clamp lets one apply inject metres/s of spurious velocity (v = Δx/dt) and the
    // pile jitters. With under-relaxation + extra iterations the small steps still
    // resolve normal penetrations within a substep.
    float3 d = dp.xyz * inv;
    // Per-apply clamp scaled to the body's own thickness (a long frank tolerates a
    // larger de-penetration step than a thin coin).
    float maxStep = 2.0 * cdHalfThickOf(coins[id]);
    float len = length(d);
    if (len > maxStep) d *= (maxStep / len);
    coins[id].posInvMass.xyz += d;

    // Angular: clamp the rotation step so a bad contact can't spin a coin wildly.
    float3 w = coinDelta[4*id + 1].xyz * inv;
    float wlen = length(w);
    const float maxAng = 0.07;                // rad per apply — bounds per-substep spin injection
    if (wlen > maxAng) w *= (maxAng / wlen);
    coins[id].orient = cdApplyRotVec(coins[id].orient, w);
    // (support flag rides coinDelta[4*id+1].w; restitution axis rides [4*id+3], both
    //  read by coinFinalize this substep.)
}

// ── KERNEL: finalize (velocity from position + orientation, friction, sleep) ──

kernel void coinFinalize(
    device CoinBody*       coins     [[ buffer(0) ]],
    device const float4*   coinDelta [[ buffer(1) ]],
    constant CoinUniforms& u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) return;

    float3 x = c.posInvMass.xyz;
    // The body's SMALLEST half-dimension is how low its COM can rest (a coin/jewel
    // on its face → halfThickness; a long frank on its side → radius). Use the min
    // so the safety floor never floats a side-lying prolate body.
    float  floorH = min(cdRadiusOf(c), cdHalfThickOf(c));

    // Hard safety floor (the plane collider does the real work; this catches a fast
    // body that outran a substep). A sphere's COM rests at exactly R, so clamp it to
    // floorY + R — clamping to half that (the disc backstop) is what sank a tunneled
    // ball to its equator. Discs/boxes keep the gentle floorH·0.5 backstop.
    float restY = cdIsSphere(c) ? cdRadiusOf(c) : floorH * 0.5;
    if (x.y < u.floorY + restY) {
        x.y = u.floorY + restY;
    }

    // Linear velocity from the (contact-corrected) position change.
    float ballisticVy = c.vel.y;
    float3 v = (x - c.prevPos.xyz) / u.dt;

    // Angular velocity from the quaternion delta over the substep.
    float4 dq = cdQuatMul(c.orient, cdQuatConj(c.prevOrient));
    if (dq.w < 0.0) dq = -dq;                 // shortest arc
    float3 omega = 2.0 * dq.xyz / u.dt;

    bool support   = (coinDelta[4*id + 1].w > 0.5);   // a real upward (resting) support contact
    float3 sN      = coinDelta[4*id + 2].xyz;          // Σ UNIT upward normals (robust at rest)
    float  sNl     = length(sN);
    float3 nC      = coinDelta[4*id + 3].xyz;          // Σ all contact normals (depth-weighted)
    float  nCl     = length(nC);
    bool   anyContact = support || (nCl > 1e-6);
    if (anyContact) {
        omega  *= u.angFriction;    // contact angular friction → tilts hold, spin bleeds

        // Robust contact axis: the unit-summed support normal (does NOT vanish as the
        // resting penetration converges to ~0, unlike the depth-weighted nC).
        if (u.frictionCoeff > 0.0 && support && sNl > 1e-4) {
            // VELOCITY-LEVEL COULOMB CONTACT FRICTION (friction-with-torque). The old
            // friction was a COM-velocity damp (v.xz *= k): isotropic, normal-force-
            // independent, producing NO torque — so a sliding body never started
            // rolling and a heap's repose came only from geometry. This applies a
            // friction IMPULSE at the CONTACT POINT opposing the contact-point
            // tangential velocity, bounded by the Coulomb cone μ·jₙ. Because the
            // impulse acts off the COM it slows translation AND spins the body up
            // toward rolling-without-slipping, and μ now sets the angle of repose.
            float3 axis = sN / sNl;                          // robust support normal
            float3 bodyN = normalize(cdQuatRotate(c.orient, float3(0,1,0)));
            float  dna = dot(bodyN, axis);
            float  Rb = cdRadiusOf(c), hb = cdHalfThickOf(c);
            float  extent = hb * abs(dna) + Rb * sqrt(max(0.0, 1.0 - dna*dna));
            float3 r = -extent * axis;                       // contact point offset from COM
            float  invMb = c.posInvMass.w;
            float3 invIb = cdBodyInvInertia(c, invMb);
            float3 vcp = v + cross(omega, r);                // contact-point velocity
            float3 vt  = vcp - dot(vcp, axis) * axis;        // tangential slip
            float  vtl = length(vt);
            if (vtl > 1e-5) {
                float3 t = vt / vtl;
                float3 rxt = cross(r, t);
                float  kt = invMb + dot(rxt, cdApplyInvInertiaWorld(c.orient, invIb, rxt));
                float  jStop = vtl / max(kt, 1e-6);          // impulse to fully arrest slip (stick)
                // jₙ ≈ normal momentum exchanged this substep: holding against gravity
                // (g·dt/invM) plus arresting any incoming approach (−vInN/invM).
                float  vInN = dot(c.vel.xyz, axis);
                float  jn = (u.gravity * u.dt + max(0.0, -vInN)) / max(invMb, 1e-6);
                float  jt = min(jStop, u.frictionCoeff * jn);
                float3 P = -jt * t;                          // friction impulse opposes slip
                v     += invMb * P;
                omega += cdApplyInvInertiaWorld(c.orient, invIb, cross(r, P));
            }
        } else {
            v.xz *= u.friction;     // legacy isotropic tangential damp (μ == 0 path)
        }

        // VERTICAL floor restitution + anti-levitation cap — only when actually
        // RESTING on a surface (n.y>0.3). A real downward impact bounces; a gentle
        // position-fix hop is capped so a settled pile doesn't drift upward.
        if (support) {
            // Below restThreshold the impact is too gentle to read as a real bounce —
            // it just hops a few mm in place (the visible "bouncing in place" jitter),
            // so cap it instead of bouncing. A scene with a high restitution wants this
            // raised so the sub-threshold tail of in-place micro-bounces dies cleanly.
            if (ballisticVy > -u.restThreshold) {
                v.y = min(v.y, 0.25);
            } else {
                v.y = -cdEffectiveCOR(u, -ballisticVy) * ballisticVy;  // impact speed = −ballisticVy
            }
        }

        // WALL / CEILING restitution along the IMPACT NORMAL. Restitution used to be
        // vertical-only, so a body slamming a vertical wall just dead-stopped. Now a
        // genuinely-approaching impact against a mostly-horizontal normal reflects
        // along that normal (e.g. a thrown ball rebounding off the back wall). Gated
        // tightly — real fast approach only — so resting contacts and the kinematic
        // pusher (whose forward shove is a position correction, not an approach) are
        // untouched. Applied after tangential friction so the reflected normal speed
        // isn't damped, while along-wall sliding still is.
        if (nCl > 1e-6) {
            float3 axis = nC / nCl;
            if (abs(axis.y) < 0.7) {                 // a wall / ceiling, not the floor
                float vInN = dot(c.vel.xyz, axis);   // pre-contact approach (<0 = into surface)
                if (vInN < -0.8) {
                    float vN = dot(v, axis);
                    v += (-cdEffectiveCOR(u, -vInN) * vInN - vN) * axis;  // impact speed = −vInN
                }
            }
        }

        // ROLLING RESISTANCE: resist the component of spin that rolls the body along
        // the contact (ω perpendicular to the contact normal), so a coin on its edge
        // slows and stops instead of rolling forever under the near-frictionless
        // global angular damping. Spin ABOUT the normal (twisting in place) is left to
        // angFriction. Opt-in (default 0).
        if (u.rollingResistance > 0.0 && support && sNl > 1e-4) {
            float3 axis = sN / sNl;
            float3 omegaRoll = omega - dot(omega, axis) * axis;
            omega -= u.rollingResistance * omegaRoll;
        }

        // SLEEP a resting coin hard so a settled heap goes truly quiet (no micro-
        // jitter). Only a SUPPORTED, slow coin sleeps — a body mid-bounce off a wall
        // (support == false) is never frozen. A real coin at rest doesn't twitch.
        //
        // CRUCIALLY, refuse to sleep a coin that is still significantly PENETRATING
        // (nCl ≈ Σ contact penetration depth): zeroing its velocity mid-overlap froze
        // the residual interpenetration in place (a settled 2-stack locked in at ~1.3h
        // of overlap, because the coin's separating velocity dropped below the sleep
        // threshold while it was still sunk in). Holding it awake until the contact has
        // pushed the overlap below ~⅓ thickness lets the stack reach its true gap.
        float restPenSlop = 0.33 * cdHalfThickOf(c);
        if (support && nCl < restPenSlop &&
            length(v) < u.sleepLinVel && length(omega) < 1.5) {
            v = float3(0);
            omega = float3(0);
        }
    }

    // Horizontal de-penetration cap — a coin pusher has no legitimate fast lateral
    // motion (plate creeps, gravity is vertical), so clamp horizontal speed so a
    // dense pile shoved out of overlap eases sideways instead of launching.
    float2 hv = v.xz;
    float  hspd = length(hv);
    float kMaxHSpeed = u.maxHSpeed;
    if (hspd > kMaxHSpeed) v.xz = hv * (kMaxHSpeed / hspd);
    // Global safety cap against any pathological ejection.
    float spd = length(v);
    if (spd > u.maxSpeed) v *= (u.maxSpeed / spd);
    // Angular speed cap. Coins in a pusher don't legitimately spin fast — a dropped
    // coin tumbles a few rev/s, a resting one not at all. 30 rad/s (~5 rev/s) let
    // contact/leveling corrections pump visible whirling; 11 rad/s (~1.7 rev/s)
    // keeps tumble lively without the "going wild" whirl.
    float ospd = length(omega);
    float kMaxOmega = u.maxOmega;
    if (ospd > kMaxOmega) omega *= (kMaxOmega / ospd);

    coins[id].posInvMass.xyz = x;
    coins[id].vel.xyz   = v;
    coins[id].angVel.xyz = omega;
    coins[id].angVel.w   = support ? 1.0 : 0.0;   // rest/support flag (drives leveling + sleep)
}

// ── KERNEL: gentle support-leveling (distributed-support flattening) ──────────
//
// Orientation is PHYSICAL (integrated in coinIntegrate, corrected by contacts).
// A single analytic contact per pair is stable but can't constrain a coin flat —
// one contact point lets it perch at whatever tilt it landed at. A coin resting on
// a real surface has a SUPPORT FOOTPRINT, and gravity levels it onto that surface
// (minimum PE = lying flat). This pass models that distributed-support torque the
// point contact misses: when a coin is SUPPORTED and SLOW, ease its disc face
// toward the surface it actually rests on (the accumulated contact normal), a small
// fraction per frame. It is NOT the old decorative snap-flat: it's gated on
// rest, gentle (eases over several frames), and aims at the true support normal —
// so coins in motion keep tumbling and leaning, and a settled heap reads calm.
kernel void coinOrient(
    device CoinBody*       coins     [[ buffer(0) ]],
    constant CoinUniforms& u         [[ buffer(1) ]],
    device const float4*   coinDelta [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) return;

    bool supported = c.angVel.w > 0.5;
    float speed = length(c.vel.xyz);
    if (!supported || speed > 0.08) return;          // only calm, resting coins

    float3 sn = coinDelta[4*id + 2].xyz;             // Σ upward contact normals
    float snl = length(sn);
    if (snl < 1e-4) return;
    float3 target = sn / snl;

    float4 q = c.orient;
    float3 axisW = cdQuatRotate(q, float3(0,1,0));    // current disc normal
    if (dot(axisW, target) < 0.0) target = -target;   // symmetric disc: nearest face
    float3 rotAxis = cross(axisW, target);            // small-angle rotation vector
    float s = length(rotAxis);
    if (s < 1e-5) return;
    float angle = asin(clamp(s, 0.0, 1.0));
    // DEADBAND: a coin already lying near-flat is left alone. Without this the pass
    // perpetually nudges settled coins toward "perfectly" flat, and each nudge
    // rotates them a hair into a neighbour → the contact shoves back → the whole
    // pile micro-JITTERS forever. Only correct a genuinely-tilted resting coin.
    if (angle < 0.16) return;                         // ~9° — within this, leave it be
    const float gain = 0.13;                          // gentle: eases tilt out over several frames
    coins[id].orient    = cdApplyRotVec(q, rotAxis * (gain * angle / s));
    coins[id].angVel.xyz = c.angVel.xyz * 0.7;        // bleed spin so it doesn't fight leveling
}

// ── KERNEL: joint solve (articulated franks: 2 segments + 1 bend joint) ───────
//
// Each frankfurter is two capsule SEGMENTS linked end-to-end. `links[i] = (partner
// slot, mySign)`: mySign = +1 means THIS body's +Y end joins the partner's −Y end.
// Only the LOWER-index body of a pair runs the solve (and writes BOTH bodies), so
// for 2-segment links no two threads ever touch the same body — race-free without a
// delta buffer. Enforces (a) the two joint ends COINCIDE (XPBD point constraint with
// generalized inverse mass, like a contact but bilateral) and (b) a soft restoring
// bend toward straight with a HARD angle limit — so the sausage holds its shape but
// creases when something presses on it.
kernel void coinJointSolve(
    device CoinBody*       coins [[ buffer(0) ]],
    device const int2*     links [[ buffer(1) ]],
    constant CoinUniforms& u     [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    int2 li = links[id];
    if (li.x < 0) return;                       // no joint
    uint j = uint(li.x);
    if (j <= id) return;                         // the lower-index body owns the pair
    CoinBody ci = coins[id];
    CoinBody cj = coins[j];
    if (ci.posInvMass.w == 0.0 || cj.posInvMass.w == 0.0) return;
    int2 lj = links[j];

    float hi = cdHalfThickOf(ci), hj = cdHalfThickOf(cj);
    float Ri = cdRadiusOf(ci),    Rj = cdRadiusOf(cj);
    float invMi = ci.posInvMass.w, invMj = cj.posInvMass.w;
    float3 invIi = cdInvInertiaLocal(Ri, hi, invMi);
    float3 invIj = cdInvInertiaLocal(Rj, hj, invMj);
    float signi = float(li.y), signj = float(lj.y);

    // (a) Point constraint: pull the two joint ends together. The joint sits at the
    // CYLINDER SEAM (halfThickness − radius), not the rounded tip, so the two
    // segments' hemispherical caps meet there and merge into a smooth sphere → one
    // continuous sausage rather than a pinch. (The two colliders overlap by ~2R at
    // the joint; contact between joint partners is skipped — see coinContactSolve.)
    float3 ri = cdQuatRotate(ci.orient, float3(0.0, signi * (hi - Ri), 0.0));
    float3 rj = cdQuatRotate(cj.orient, float3(0.0, signj * (hj - Rj), 0.0));
    float3 pA = ci.posInvMass.xyz + ri;
    float3 pB = cj.posInvMass.xyz + rj;
    float3 C  = pB - pA;
    float clen = length(C);
    if (clen > 1e-6) {
        float3 n = C / clen;
        float wi = invMi + dot(cross(ri, n), cdApplyInvInertiaWorld(ci.orient, invIi, cross(ri, n)));
        float wj = invMj + dot(cross(rj, n), cdApplyInvInertiaWorld(cj.orient, invIj, cross(rj, n)));
        float denom = wi + wj;
        if (denom > 1e-8) {
            float3 P = (clen / denom) * n;       // full bilateral correction (mass-weighted)
            ci.posInvMass.xyz += invMi * P;
            cj.posInvMass.xyz -= invMj * P;
            ci.orient = cdApplyRotVec(ci.orient, cdApplyInvInertiaWorld(ci.orient, invIi, cross(ri,  P)));
            cj.orient = cdApplyRotVec(cj.orient, cdApplyInvInertiaWorld(cj.orient, invIj, cross(rj, -P)));
        }
    }

    // (b) Bend: relative rotation (j ← i), restored toward straight — gentle under
    // the limit (holds its shape, flexes a little), hard past it (never folds flat).
    float4 dq = cdQuatMul(cj.orient, cdQuatConj(ci.orient));
    if (dq.w < 0.0) dq = -dq;
    float3 rv = 2.0 * dq.xyz;                    // small-angle rotation vector i→j
    float ang = length(rv);
    const float limit = 0.55;                    // ~31° max bend
    float gain = (ang > limit) ? 0.5 : 0.06;
    float3 corr = rv * gain;
    ci.orient = cdApplyRotVec(ci.orient,  0.5 * corr);
    cj.orient = cdApplyRotVec(cj.orient, -0.5 * corr);

    coins[id] = ci;
    coins[j]  = cj;
}

// ── KERNEL: derive per-coin render transform ──────────────────────────────────

kernel void coinDeriveTransforms(
    device const CoinBody* coins      [[ buffer(0) ]],
    device CoinTransform*  transforms [[ buffer(1) ]],
    constant CoinUniforms& u          [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) {
        // Park inactive slots far off-screen so their instance is invisible.
        transforms[id] = CoinTransform{
            float4(1,0,0,0), float4(0,1,0,0), float4(0,0,1,0),
            float4(0, -100000.0, 0, 1)
        };
        return;
    }
    // Discs: the mesh is authored at the body's TRUE size (radius/halfThickness ARE
    // the size, each asset has its own true-size mesh), so the transform is a pure
    // rotation + translation. BOXES vary in size per body, so bake the per-instance
    // scale (full extents) into the basis columns — a box renderer then draws ONE
    // unit cube mesh (vertices in [-0.5, 0.5]³) and every box comes out the right
    // size. (Box face normals stay correct: a normal lies along a scaled axis, and
    // the expand kernel renormalizes, recovering the exact world face normal.)
    float3x3 m = cdQuatToMat3(c.orient);
    if (cdIsBox(c)) {
        float3 fe = 2.0 * c.shapeExtents.xyz;   // full extents (side lengths)
        m[0] *= fe.x;                           // column k = image of local axis k
        m[1] *= fe.y;
        m[2] *= fe.z;
    }
    transforms[id] = CoinTransform{
        float4(m[0], 0.0),
        float4(m[1], 0.0),
        float4(m[2], 0.0),
        float4(c.posInvMass.xyz, 1.0)
    };
}

// ── KERNEL: expand instances (clone of eggs_expand_instances) ─────────────────
//
// One thread per (instance, vertex). Transforms the shared unit-coin mesh by
// each coin's CoinTransform into the big shared position/normal buffers that one
// SCNGeometry draws in a single call. See CoinInstancedRenderer.swift.

struct CoinExpandUniforms {
    uint  coinCount;
    uint  vertsPerInstance;
    uint  targetType;       // this renderer's asset type
    uint  filterEnabled;    // 1 = only expand bodies whose type matches (mixed pile)
};

kernel void coin_expand_instances(
    constant CoinExpandUniforms& U     [[ buffer(0) ]],
    device const CoinTransform* xforms [[ buffer(1) ]],
    device const float4* unitPos       [[ buffer(2) ]],   // xyz = pos, w unused
    device const float4* unitNrm       [[ buffer(3) ]],   // xyz = normal, w unused
    device packed_float3* outPos       [[ buffer(4) ]],   // packed (stride 12)
    device packed_float3* outNrm       [[ buffer(5) ]],   // packed (stride 12)
    device const uint*    bodyType     [[ buffer(6) ]],   // per-body asset type (mixed pile)
    uint gid [[ thread_position_in_grid ]])
{
    // Output is packed_float3 (stride 12) so the Illuminatorama GPU-mesh repack —
    // which assumes packed_float3 and ignores any stride field — reads it
    // correctly. SceneKit also reads stride-12 .float3 fine.
    if (U.vertsPerInstance == 0u) return;
    uint instance = gid / U.vertsPerInstance;
    uint vertIdx  = gid - instance * U.vertsPerInstance;
    if (instance >= U.coinCount) return;

    // Mixed pile: one solver holds every asset; each asset has its own renderer +
    // mesh. This renderer only draws bodies of its `targetType`; others are parked
    // off-screen so they're invisible in this mesh (the matching renderer draws them).
    if (U.filterEnabled != 0u && bodyType[instance] != U.targetType) {
        uint outIdxPark = instance * U.vertsPerInstance + vertIdx;
        outPos[outIdxPark] = packed_float3(0.0, -100000.0, 0.0);
        outNrm[outIdxPark] = packed_float3(0.0, 1.0, 0.0);
        return;
    }

    CoinTransform M = xforms[instance];
    float3x3 basis = float3x3(M.col0.xyz, M.col1.xyz, M.col2.xyz);
    float3 worldPos = basis * unitPos[vertIdx].xyz + M.col3.xyz;
    float3 nWorld   = normalize(basis * unitNrm[vertIdx].xyz);

    uint outIdx = instance * U.vertsPerInstance + vertIdx;
    outPos[outIdx] = packed_float3(worldPos);
    outNrm[outIdx] = packed_float3(nWorld);
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 6: GJK + EPA general convex narrowphase
// ══════════════════════════════════════════════════════════════════════════════
//
// The general path: any convex pair → GJK (is the Minkowski difference's origin
// enclosed? = overlap) → EPA (expand the polytope to the origin's closest face =
// penetration normal + depth). Works for ANY shape with a support function, so it
// generalizes past the analytic disc/box/sphere routines (e.g. a future arbitrary
// convex hull). Delivered + verified as the primitive; the analytic manifold path
// stays the live narrowphase (one GJK/EPA point per pair can't hold a box stack
// without face-clipping, which is a separate, larger addition).

// Farthest surface point of body `c` along world direction `d` (the support map).
static float3 cdSupport(CoinBody c, float3 d,
                        device const float4* hullVerts, device const uint2* hullRanges) {
    float3 ctr = c.posInvMass.xyz;
    if (cdIsSphere(c)) return ctr + cdRadiusOf(c) * normalize(d);
    float4 q = c.orient;
    float3 dl = cdQuatRotateInv(q, d);     // direction in body-local frame
    if (cdIsHull(c)) {
        // Scan the registered principal-frame vertices (≤64; interior points were
        // dropped at registration, so every scan hit is a true hull vertex).
        uint2 rng = hullRanges[uint(c.hullRef.x + 0.5)];
        float best = -1e30; float3 bp = float3(0.0);
        for (uint i = 0; i < rng.y; ++i) {
            float3 v = float3(hullVerts[rng.x + i].xyz);
            float pr = dot(v, dl);
            if (pr > best) { best = pr; bp = v; }
        }
        return ctr + cdQuatRotate(q, bp);
    }
    float3 pl;
    if (cdIsCapsule(c)) {
        // Segment end nearest the direction, plus the radius along d.
        float hl = cdCapsuleHL(c), r = cdCapsuleR(c);
        return ctr + cdQuatRotate(q, float3(0.0, dl.y >= 0.0 ? hl : -hl, 0.0))
                   + r * normalize(d);
    }
    if (cdIsBox(c)) {
        float3 he = cdBodyHalfExtents(c);
        pl = float3(dl.x >= 0.0 ? he.x : -he.x, dl.y >= 0.0 ? he.y : -he.y, dl.z >= 0.0 ? he.z : -he.z);
    } else {                               // capped cylinder, axis +Y
        float R = cdRadiusOf(c), h = cdHalfThickOf(c);
        float2 rad = float2(dl.x, dl.z); float rl = length(rad);
        float2 rdir = rl > 1e-6 ? rad / rl : float2(1.0, 0.0);
        pl = float3(rdir.x * R, dl.y >= 0.0 ? h : -h, rdir.y * R);
    }
    return ctr + cdQuatRotate(q, pl);
}

// Support point of the Minkowski difference A⊖B along d (and the witness on A).
static float4 cdCSO(CoinBody ci, CoinBody cj, float3 d,
                    device const float4* hullVerts, device const uint2* hullRanges) {
    float3 a = cdSupport(ci, d, hullVerts, hullRanges);
    return float4(a - cdSupport(cj, -d, hullVerts, hullRanges), 0.0);   // .xyz = CSO point
}

// Triple product (a×b)×c.
static float3 cdTriple(float3 a, float3 b, float3 c) { return cross(cross(a, b), c); }

// EXACT penetration depth along unit direction d (defined as: translating body A
// by depth·d just separates the pair). Two support calls; used to (a) resolve the
// degenerate-GJK flat/face-face terminations EPA can't seed from, and (b) cross-
// check EPA's answer against the bodies' own face axes.
static float cdDepthAlong(CoinBody ci, CoinBody cj, float3 d,
                          device const float4* hullVerts, device const uint2* hullRanges) {
    float3 sm = cdCSO(ci, cj, -d, hullVerts, hullRanges).xyz;   // CSO min along d
    return dot(sm, -d);
}

// Directional-sampling narrowphase: minimum exact depth over the two bodies'
// local frame axes + the centre line. For flat face-face contacts (where the GJK
// simplex degenerates and EPA has nothing to seed from) one of these axes IS the
// true separating direction, so the result is exact — and it can only ever
// report a deeper-or-equal depth than the true minimum elsewhere (safe).
static bool cdAxisProbe(CoinBody ci, CoinBody cj,
                        device const float4* hullVerts, device const uint2* hullRanges,
                        thread float3& outN, thread float& outDepth) {
    float3 cand[14];
    int nc = 0;
    for (int a = 0; a < 3; ++a) {
        float3 e = float3(a == 0 ? 1.0 : 0.0, a == 1 ? 1.0 : 0.0, a == 2 ? 1.0 : 0.0);
        float3 wa = cdQuatRotate(ci.orient, e);
        float3 wb = cdQuatRotate(cj.orient, e);
        cand[nc++] = wa; cand[nc++] = -wa;
        cand[nc++] = wb; cand[nc++] = -wb;
    }
    float3 D = ci.posInvMass.xyz - cj.posInvMass.xyz;
    float dl = length(D);
    cand[nc++] = dl > 1e-6 ? D / dl : float3(0, 1, 0);
    cand[nc++] = float3(0, 1, 0);
    float best = 1e30; float3 bestN = float3(0, 1, 0);
    for (int i = 0; i < nc; ++i) {
        float pd = cdDepthAlong(ci, cj, cand[i], hullVerts, hullRanges);
        if (pd < best) { best = pd; bestN = cand[i]; }
    }
    if (best <= 0.0) return false;   // a separating axis exists — no contact
    outN = bestN;                    // pushing A along n separates ⇒ n points B→A
    outDepth = best;
    return true;
}

constant int CD_EPA_MAXV = 32;
constant int CD_EPA_MAXF = 60;

// GJK + EPA. Returns true if `ci`,`cj` overlap, with the penetration NORMAL (world,
// points from B toward A) and DEPTH. Bounded thread-local polytope (no heap).
static bool cdGJKEPA(CoinBody ci, CoinBody cj,
                     device const float4* hullVerts, device const uint2* hullRanges,
                     thread float3& outN, thread float& outDepth) {
    // ── GJK: evolve a simplex toward the origin of the Minkowski difference ────
    float3 sx[4]; int n = 0;
    float3 dir = ci.posInvMass.xyz - cj.posInvMass.xyz;
    if (dot(dir, dir) < 1e-12) dir = float3(1, 0, 0);
    sx[0] = cdCSO(ci, cj, dir, hullVerts, hullRanges).xyz; n = 1;
    dir = -sx[0];
    bool overlap = false;
    for (int iter = 0; iter < 32; ++iter) {
        if (dot(dir, dir) < 1e-12) { overlap = true; break; }
        float3 p = cdCSO(ci, cj, dir, hullVerts, hullRanges).xyz;
        if (dot(p, dir) < 0.0) return false;            // no overlap (separating axis)
        // add p, run do-simplex
        for (int k = n; k > 0; --k) sx[k] = sx[k-1];
        sx[0] = p; n++;
        // Evolve simplex (point already handled; handle line/triangle/tetra).
        if (n == 2) {
            float3 a = sx[0], b = sx[1], ab = b - a, ao = -a;
            dir = cdTriple(ab, ao, ab);
            if (dot(dir, dir) < 1e-12) dir = cross(ab, float3(1,0,0)), dir = dot(dir,dir)<1e-12 ? cross(ab,float3(0,1,0)) : dir;
        } else if (n == 3) {
            float3 a = sx[0], b = sx[1], c = sx[2];
            float3 ab = b - a, ac = c - a, ao = -a, abc = cross(ab, ac);
            if (dot(cross(abc, ac), ao) > 0.0)      { sx[1] = c; n = 2; dir = cdTriple(ac, ao, ac); }
            else if (dot(cross(ab, abc), ao) > 0.0) { n = 2; dir = cdTriple(ab, ao, ab); }
            else { dir = dot(abc, ao) > 0.0 ? abc : -abc; }
        } else { // n == 4: tetrahedron — does it contain the origin?
            float3 a = sx[0], b = sx[1], c = sx[2], d = sx[3], ao = -a;
            float3 abc = cross(b-a, c-a), acd = cross(c-a, d-a), adb = cross(d-a, b-a);
            if (dot(abc, ao) > 0.0)      { sx[3] = c; sx[2] = b; sx[1] = a; n = 3; dir = abc; n = 3; sx[0]=a;sx[1]=b;sx[2]=c; n=3; }
            else if (dot(acd, ao) > 0.0) { sx[1] = c; sx[2] = d; n = 3; }
            else if (dot(adb, ao) > 0.0) { sx[1] = d; sx[2] = b; n = 3; }
            else { overlap = true; break; }
            // recompute dir from the kept triangle
            float3 aa = sx[0], bb = sx[1], cc = sx[2];
            float3 nn = cross(bb-aa, cc-aa); dir = dot(nn, -aa) > 0.0 ? nn : -nn;
        }
    }
    if (!overlap) return false;

    // GJK confirmed overlap but terminated on a DEGENERATE simplex (point / edge
    // / triangle — the flat face-face case). EPA cannot be seeded from it: the
    // old completion added coplanar supports (or left seed vertices
    // uninitialized), and the garbage face normals it produced read as LATERAL
    // contact normals that pumped resting hull stacks apart. The exact
    // directional probe over both bodies' face axes IS the right answer for
    // precisely these flat contacts.
    if (n < 4) return cdAxisProbe(ci, cj, hullVerts, hullRanges, outN, outDepth);

    // ── EPA: expand the simplex's polytope to the origin's closest face ───────
    float3 V[CD_EPA_MAXV]; int VN = 0;
    int F[CD_EPA_MAXF][3]; float3 FN[CD_EPA_MAXF]; float FD[CD_EPA_MAXF]; int FNn = 0;
    // Seed with the GJK tetrahedron (n == 4: it encloses the origin).
    for (int i = 0; i < 4; ++i) { V[i] = sx[i]; }
    VN = 4;
    // faces of the tetra (winding outward)
    int tet[4][3] = {{0,1,2},{0,2,3},{0,3,1},{1,3,2}};
    for (int i = 0; i < 4; ++i) {
        int i0=tet[i][0], i1=tet[i][1], i2=tet[i][2];
        float3 fn = cross(V[i1]-V[i0], V[i2]-V[i0]);
        float fl = length(fn); if (fl < 1e-12) continue; fn /= fl;
        if (dot(fn, V[i0]) < 0.0) { int t=i1; i1=i2; i2=t; fn = -fn; }   // outward
        F[FNn][0]=i0; F[FNn][1]=i1; F[FNn][2]=i2; FN[FNn]=fn; FD[FNn]=dot(fn,V[i0]); FNn++;
    }
    float3 bestN = float3(0,1,0); float bestD = 1e30;
    for (int iter = 0; iter < 24; ++iter) {
        // closest face to origin
        int ci2 = -1; float cd = 1e30;
        for (int f = 0; f < FNn; ++f) if (FD[f] < cd) { cd = FD[f]; ci2 = f; }
        if (ci2 < 0) break;
        bestN = FN[ci2]; bestD = cd;
        float3 sp = cdCSO(ci, cj, FN[ci2], hullVerts, hullRanges).xyz;
        float spd = dot(FN[ci2], sp);
        if (spd - cd < 1e-4 || VN >= CD_EPA_MAXV || FNn + 6 >= CD_EPA_MAXF) break;   // converged
        // Remove all faces the new point can "see", collect the horizon, re-fan.
        int newV = VN; V[VN++] = sp;
        // mark visible faces, build a fresh face list
        int keepF[CD_EPA_MAXF][3]; float3 keepN[CD_EPA_MAXF]; float keepD[CD_EPA_MAXF]; int keepN_n = 0;
        // horizon edges (store as pairs)
        int edges[CD_EPA_MAXF*2][2]; int en = 0;
        for (int f = 0; f < FNn; ++f) {
            bool visible = dot(FN[f], sp) - FD[f] > 1e-6;
            if (!visible) { keepF[keepN_n][0]=F[f][0];keepF[keepN_n][1]=F[f][1];keepF[keepN_n][2]=F[f][2];keepN[keepN_n]=FN[f];keepD[keepN_n]=FD[f];keepN_n++; continue; }
            // add its 3 edges to the horizon (cancel shared)
            for (int e = 0; e < 3; ++e) {
                int a = F[f][e], b = F[f][(e+1)%3];
                bool found = false;
                for (int q = 0; q < en; ++q) if (edges[q][0]==b && edges[q][1]==a) { edges[q][0]=edges[en-1][0]; edges[q][1]=edges[en-1][1]; en--; found=true; break; }
                if (!found && en < CD_EPA_MAXF*2) { edges[en][0]=a; edges[en][1]=b; en++; }
            }
        }
        // rebuild faces = kept + a fan from newV over the horizon
        FNn = 0;
        for (int f = 0; f < keepN_n; ++f) { F[FNn][0]=keepF[f][0];F[FNn][1]=keepF[f][1];F[FNn][2]=keepF[f][2];FN[FNn]=keepN[f];FD[FNn]=keepD[f];FNn++; }
        for (int e = 0; e < en && FNn < CD_EPA_MAXF; ++e) {
            int a = edges[e][0], b = edges[e][1];
            float3 fn = cross(V[b]-V[a], V[newV]-V[a]); float fl = length(fn);
            if (fl < 1e-12) continue; fn /= fl;
            if (dot(fn, V[a]) < 0.0) fn = -fn;          // keep outward
            F[FNn][0]=a; F[FNn][1]=b; F[FNn][2]=newV; FN[FNn]=fn; FD[FNn]=max(dot(fn,V[a]),0.0); FNn++;
        }
    }
    // EPA's outward face normal of the A⊖B polytope points A→B; the rest of the solver
    // uses "out of B toward A" (see cdAccumContact), so negate to match the convention.
    outN = -bestN;
    outDepth = bestD;
    // Cross-check EPA against the exact face-axis probe: near-flat contacts sit at
    // EPA's numerical edge, and a shallower true axis means EPA's normal is off.
    float3 pn; float pdepth;
    if (cdAxisProbe(ci, cj, hullVerts, hullRanges, pn, pdepth) && pdepth < bestD * 0.9) {
        outN = pn;
        outDepth = pdepth;
    }
    return true;
}



// ─────────────────────────────────────────────────────────────────────────────
// coinMeasurePenetration — DIAGNOSTIC pass (no resolution).
//
// Answers, in code, the recurring "these objects are interpenetrating" note for
// GPU-simulated piles. It reuses coinContactSolve's EXACT broadphase (the 3×3×3
// spatial-hash neighbourhood) and disk-vs-disk SAT, but instead of applying a
// positional correction it records the overlap: atomic-max of the deepest
// penetration and an atomic count of pairs deeper than `threshold`. Each pair is
// counted once (`j > id`); joint partners (articulated frank segments, which
// overlap by design) are skipped, exactly as the solver skips them.
//
// `result` is two atomic uints: [0] = max depth in MICROMETRES (depth·1e6, so a
// uint atomic_max gives a correct float max for the small positive depths here),
// [1] = penetrating-pair count. The host reads them back once after the sim has
// settled — this is not a per-frame kernel.
kernel void coinMeasurePenetration(
    device const CoinBody*  coins         [[ buffer(0) ]],
    device const uint*      sortedIndices [[ buffer(1) ]],
    device const uint*      cellOffsets   [[ buffer(2) ]],
    constant CoinUniforms&  u             [[ buffer(3) ]],
    device const int2*      links         [[ buffer(4) ]],
    device atomic_uint*     result        [[ buffer(5) ]],   // [0]=maxDepth µm, [1]=pairCount
    constant float&         threshold     [[ buffer(6) ]],
    device const float4*    hullVerts     [[ buffer(7) ]],
    device const uint2*     hullRanges    [[ buffer(8) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) { return; }
    CoinBody ci = coins[id];
    if (ci.posInvMass.w == 0.0) { return; }
    int jointPartner = links[id].x;

    float3 xi = ci.posInvMass.xyz;
    float4 qi = ci.orient;
    float  Ri = cdRadiusOf(ci);
    float  hi = cdHalfThickOf(ci);
    float3 ni = normalize(cdQuatRotate(qi, float3(0,1,0)));

    int3 base = int3(floor((xi - float3(u.gridMinX, u.gridMinY, u.gridMinZ)) * u.invCell));
    int3 res  = int3(int(u.gridResX), int(u.gridResY), int(u.gridResZ));
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int3 c = base + int3(dx, dy, dz);
        if (any(c < int3(0)) || any(c >= res)) continue;
        uint cell = uint(c.x) + u.gridResX * (uint(c.y) + u.gridResY * uint(c.z));
        uint start = cellOffsets[cell];
        uint end   = cellOffsets[cell + 1u];
        for (uint s = start; s < end; ++s) {
            uint j = sortedIndices[s];
            if (j <= id) continue;                  // count each pair ONCE
            if (int(j) == jointPartner) continue;   // articulated partners overlap by design
            CoinBody cj = coins[j];
            if (cj.posInvMass.w == 0.0) continue;
            float3 xj = cj.posInvMass.xyz;
            float4 qj = cj.orient;
            float  Rj = cdRadiusOf(cj);
            float  hj = cdHalfThickOf(cj);
            float3 D  = xi - xj;
            float reach = sqrt(Ri*Ri + hi*hi) + sqrt(Rj*Rj + hj*hj);
            if (dot(D, D) > reach * reach) continue;

            float minDepth = 1e30;
            bool  separated = false;
            if (cdIsHull(ci) || cdIsHull(cj)) {
                // Hull-involved pair: the same GJK/EPA depth the solver resolves with.
                float3 nH;
                if (!cdGJKEPA(ci, cj, hullVerts, hullRanges, nH, minDepth)) separated = true;
            } else if (cdIsSphere(ci) || cdIsSphere(cj)) {
                // Sphere-involved pair: the EXACT contact the constraint-path
                // narrowphase (coinGenerateContacts) de-penetrates with — sphere↔sphere
                // is centre distance; sphere↔box clamps the centre to the box, sphere↔disc
                // clamps to the capped cylinder, sphere↔capsule clamps to the segment.
                // (The old code used the box's *bounding sphere* here, which matched only
                // the legacy Jacobi solver and grossly over-reported a sphere resting
                // beside a box — a false positive for any constraint-path scene.)
                bool iSphere = cdIsSphere(ci);
                float3 cs = iSphere ? xi : xj;  float rs = iSphere ? Ri : Rj;
                float3 co = iSphere ? xj : xi;  float4 qo = iSphere ? qj : qi;
                CoinBody O = iSphere ? cj : ci;
                if (cdIsSphere(O)) {
                    minDepth = (Ri + Rj) - length(D);
                } else {
                    float3 lp = cdQuatRotateInv(qo, cs - co);
                    float3 closestLocal;
                    float extraR = 0.0;
                    if (cdIsCapsule(O)) {
                        float hlO = cdCapsuleHL(O);
                        closestLocal = float3(0.0, clamp(lp.y, -hlO, hlO), 0.0);
                        extraR = cdCapsuleR(O);
                    } else {
                        closestLocal = cdClosestInShapeLocal(O, lp);
                    }
                    minDepth = rs + extraR - length(lp - closestLocal);
                }
                if (minDepth <= 0.0) separated = true;
            } else if (cdIsCapsule(ci) || cdIsCapsule(cj)) {
                // Capsule-involved pair — the same probe math the constraint
                // narrowphase de-penetrates with (bounding-cylinder SAT here would
                // over-report a capsule resting beside anything: a false positive).
                if (cdIsCapsule(ci) && cdIsCapsule(cj)) {
                    float3 a0, a1, b0, b1;
                    cdCapsuleSegment(ci, a0, a1);
                    cdCapsuleSegment(cj, b0, b1);
                    float3 c1, c2; cdClosestSegSeg(a0, a1, b0, b1, c1, c2);
                    minDepth = (cdCapsuleR(ci) + cdCapsuleR(cj)) - length(c1 - c2);
                } else {
                    bool iCap = cdIsCapsule(ci);
                    CoinBody C = iCap ? ci : cj;
                    CoinBody O = iCap ? cj : ci;
                    float rc = cdCapsuleR(C);
                    float3 s0, s1; cdCapsuleSegment(C, s0, s1);
                    float3 co2 = O.posInvMass.xyz; float4 qo2 = O.orient;
                    minDepth = -1e30;
                    for (int p = 0; p < 3; ++p) {
                        float3 s = mix(s0, s1, float(p) * 0.5);
                        float3 lp = cdQuatRotateInv(qo2, s - co2);
                        float3 cl = cdClosestInShapeLocal(O, lp);
                        float dl = length(lp - cl);
                        float pen;
                        if (dl > 1e-6) {
                            pen = rc - dl;
                        } else if (cdIsBox(O)) {
                            float3 dd = cdBodyHalfExtents(O) - abs(lp);
                            pen = min(dd.x, min(dd.y, dd.z)) + rc;
                        } else {
                            float3 nL;
                            pen = -cdCappedCylSDF(lp, cdRadiusOf(O), cdHalfThickOf(O), nL) + rc;
                        }
                        minDepth = max(minDepth, pen);
                    }
                }
                if (minDepth <= 0.0) separated = true;
            } else if (cdIsBox(ci) || cdIsBox(cj)) {
                // Box-involved pair → the same oriented box–box SAT the solver
                // de-penetrates with (disc as its bounding box).
                float3 nB, cpB;
                if (!cdBoxBoxSAT(xi, qi, cdBodyHalfExtents(ci),
                                 xj, qj, cdBodyHalfExtents(cj), minDepth, nB, cpB)) separated = true;
            } else {
                // Identical SAT to coinContactSolve: face normals + centre line.
                float3 nj = normalize(cdQuatRotate(qj, float3(0,1,0)));
                float3 axes[3] = { ni, nj, float3(0,1,0) };
                float dlen = length(D);
                if (dlen > 1e-5) axes[2] = D / dlen;
                for (int ax = 0; ax < 3; ++ax) {
                    float3 a = axes[ax];
                    float di = abs(dot(ni, a)), dj = abs(dot(nj, a));
                    float ei = hi * di + Ri * sqrt(max(0.0, 1.0 - di*di));
                    float ej = hj * dj + Rj * sqrt(max(0.0, 1.0 - dj*dj));
                    float depth = (ei + ej) - abs(dot(D, a));
                    if (depth <= 0.0) { separated = true; break; }
                    if (depth < minDepth) { minDepth = depth; }
                }
            }
            if (separated || minDepth <= threshold) continue;

            atomic_fetch_max_explicit(&result[0],
                uint(minDepth * 1e6), memory_order_relaxed);
            atomic_fetch_add_explicit(&result[1], 1u, memory_order_relaxed);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 1: contact generation into a persistent buffer
// ══════════════════════════════════════════════════════════════════════════════
//
// `coinGenerateContacts` runs the SAME narrowphase as coinContactSolve but, instead
// of pushing positions, it APPENDS each manifold point to a global `CoinContact`
// buffer (atomic bump-allocator). Each dynamic body pair is processed once (thread =
// body i, neighbours j>i); static colliders are appended with bodyB = CD_STATIC. The
// resulting buffer is what the colour / velocity / position passes consume.

constant uint CD_STATIC = 0xFFFFFFFFu;

// Two orthonormal tangents spanning the plane ⟂ n (for friction).
static void cdContactTangents(float3 n, thread float3& t1, thread float3& t2) {
    t1 = (abs(n.x) > 0.7) ? normalize(float3(-n.y, n.x, 0.0))
                          : normalize(float3(0.0, -n.z, n.y));
    t2 = cross(n, t1);
}

// Pack a stable key for warm-start matching: min/max body (12 bits each) + feature.
static uint cdPairKey(uint a, uint b, uint feature) {
    uint lo = min(a, b), hi = (b == CD_STATIC) ? 0xFFFu : max(a, b);
    return (lo & 0xFFFu) | ((hi & 0xFFFu) << 12) | ((feature & 0xFFu) << 24);
}

// Append one contact (atomic bump). nBtoA points from B's surface toward A.
static void cdEmitContact(device CoinContact* contacts,
                          device atomic_uint* contactCount, uint maxContacts,
                          uint a, uint b, uint feature, uint colliderIdx,
                          float3 nBtoA, float3 cp, float3 xa, float3 xb, float depth) {
    // Depth may be NEGATIVE (a speculative near-contact within the margin) — each
    // call site owns its own `pen > -margin` gate; this only rejects garbage.
    if (isnan(depth)) return;
    uint slot = atomic_fetch_add_explicit(contactCount, 1u, memory_order_relaxed);
    if (slot >= maxContacts) return;                 // buffer full — drop (logged host-side)
    float3 n = normalize(nBtoA);
    float3 t1, t2; cdContactTangents(n, t1, t2);
    CoinContact c;
    c.meta = uint4(a, b, (b == CD_STATIC) ? colliderIdx : feature, cdPairKey(a, b, feature));
    c.nrm  = float4(n, depth);
    c.rA   = float4(cp - xa, 0.0);
    c.rB   = float4((b == CD_STATIC) ? float3(0.0) : (cp - xb), 0.0);
    c.tan1 = float4(t1, 0.0);
    c.tan2 = float4(t2, -1.0);                       // colour = −1 (uncoloured)
    c.aux  = float4(0.0);                            // vn₀ captured by the first solve pass
    contacts[slot] = c;
}


// ── Hull manifold: support sets + polygon clipping ────────────────────────────
//
// One EPA contact per pair can't hold a stack (the doc's "GJK/EPA not wired in"
// caveat) — a resting box needs its support FOOTPRINT. Given the EPA normal we
// gather each body's support set (the vertices within an eps band of its support
// plane along ∓n), project both onto the contact plane, clip one convex polygon
// by the other (Sutherland–Hodgman), and reduce to ≤4 spread points. Both sets
// live in eps-thin slabs ⟂ n, so the EPA depth is valid for every manifold point
// to within ~eps — the standard production approximation.

constant int CD_MSET = 8;

// Support set of body `c` along world direction `dirW` (toward its contact
// face). Returns ≤CD_MSET world-space points. Hull: banded vertex scan;
// box: banded corners; disc: 8 rim samples of the support-side face circle.
static int cdSupportSet(CoinBody c, float3 dirW,
                        device const float4* hullVerts, device const uint2* hullRanges,
                        thread float3* out) {
    float4 q = c.orient;
    float3 ctr = c.posInvMass.xyz;
    float3 dl = normalize(cdQuatRotateInv(q, dirW));
    float eps = 1e-3 + 0.02 * cdRadiusOf(c);
    int n = 0;
    if (cdIsHull(c)) {
        uint2 rng = hullRanges[uint(c.hullRef.x + 0.5)];
        float best = -1e30;
        for (uint i = 0; i < rng.y; ++i) best = max(best, dot(float3(hullVerts[rng.x + i].xyz), dl));
        for (uint i = 0; i < rng.y && n < CD_MSET; ++i) {
            float3 v = float3(hullVerts[rng.x + i].xyz);
            if (best - dot(v, dl) < eps) out[n++] = ctr + cdQuatRotate(q, v);
        }
        return n;
    }
    if (cdIsBox(c)) {
        float3 he = c.shapeExtents.xyz;
        float best = -1e30;
        for (int i = 0; i < 8; ++i) {
            float3 v = float3((i & 1) ? he.x : -he.x, (i & 2) ? he.y : -he.y, (i & 4) ? he.z : -he.z);
            best = max(best, dot(v, dl));
        }
        for (int i = 0; i < 8 && n < CD_MSET; ++i) {
            float3 v = float3((i & 1) ? he.x : -he.x, (i & 2) ? he.y : -he.y, (i & 4) ? he.z : -he.z);
            if (best - dot(v, dl) < eps) out[n++] = ctr + cdQuatRotate(q, v);
        }
        return n;
    }
    // Disc (capped cylinder): sample the rim circle on the support-side face; the
    // eps band keeps all 8 for a face-on contact and 1–2 for an edge-on one.
    {
        float R = cdRadiusOf(c), h = cdHalfThickOf(c);
        float sy = dl.y >= 0.0 ? h : -h;
        float best = -1e30;
        for (int i = 0; i < 8; ++i) {
            float a = float(i) * 0.7853982;
            float3 v = float3(R * cos(a), sy, R * sin(a));
            best = max(best, dot(v, dl));
        }
        for (int i = 0; i < 8 && n < CD_MSET; ++i) {
            float a = float(i) * 0.7853982;
            float3 v = float3(R * cos(a), sy, R * sin(a));
            if (best - dot(v, dl) < eps) out[n++] = ctr + cdQuatRotate(q, v);
        }
        return n;
    }
}

// Order ≤CD_MSET 2D points CCW about their centroid (insertion sort by angle).
static int cdOrderConvex2D(thread float2* pts, int n) {
    if (n < 3) return n;
    float2 cen = float2(0.0);
    for (int i = 0; i < n; ++i) cen += pts[i];
    cen /= float(n);
    float ang[CD_MSET];
    for (int i = 0; i < n; ++i) ang[i] = atan2(pts[i].y - cen.y, pts[i].x - cen.x);
    for (int i = 1; i < n; ++i) {
        float2 pv = pts[i]; float av = ang[i]; int j = i - 1;
        while (j >= 0 && ang[j] > av) { pts[j+1] = pts[j]; ang[j+1] = ang[j]; j--; }
        pts[j+1] = pv; ang[j+1] = av;
    }
    return n;
}

// Order ≤CD_MSET 3D points CCW (as seen along the tangent basis t1/t2) about
// their centroid — the winding cdFitFacePlane's Newell's-method normal needs
// to come out stable and consistently signed.
static void cdOrderConvex3D(thread float3* pts, int n, float3 t1, float3 t2) {
    if (n < 3) return;
    float3 cen = float3(0.0);
    for (int i = 0; i < n; ++i) cen += pts[i];
    cen /= float(n);
    float ang[CD_MSET];
    for (int i = 0; i < n; ++i) {
        float3 d = pts[i] - cen;
        ang[i] = atan2(dot(d, t2), dot(d, t1));
    }
    for (int i = 1; i < n; ++i) {
        float3 pv = pts[i]; float av = ang[i]; int j = i - 1;
        while (j >= 0 && ang[j] > av) { pts[j+1] = pts[j]; ang[j+1] = ang[j]; j--; }
        pts[j+1] = pv; ang[j+1] = av;
    }
}

// Fit a plane (dot(normal, x) == offset) through ≤CD_MSET CCW-ordered 3D
// points via Newell's method. False on a degenerate (near-collinear) set —
// callers must fall back rather than trust a near-zero-length normal.
static bool cdFitFacePlane(thread const float3* pts, int n, thread float3& outNormal, thread float& outOffset) {
    if (n < 3) return false;
    float3 centroid = float3(0.0);
    for (int i = 0; i < n; ++i) centroid += pts[i];
    centroid /= float(n);
    float3 nrm = float3(0.0);
    for (int i = 0; i < n; ++i) nrm += cross(pts[i] - centroid, pts[(i + 1) % n] - centroid);
    float len = length(nrm);
    if (len < 1e-10) return false;
    outNormal = nrm / len;
    outOffset = dot(outNormal, centroid);
    return true;
}

// Sutherland–Hodgman: clip convex polygon A (CCW) by convex polygon B (CCW).
// Returns the clipped vertex count (≤ 2·CD_MSET).
static int cdClipConvex2D(thread float2* A, int nA, thread const float2* B, int nB,
                          thread float2* out) {
    float2 cur[CD_MSET * 2]; int nc = min(nA, CD_MSET);
    for (int i = 0; i < nc; ++i) cur[i] = A[i];
    float2 nxt[CD_MSET * 2];
    for (int e = 0; e < nB; ++e) {
        float2 p0 = B[e], p1 = B[(e + 1) % nB];
        float2 en = float2(-(p1.y - p0.y), p1.x - p0.x);   // inward normal of a CCW edge
        int nn = 0;
        for (int i = 0; i < nc; ++i) {
            float2 a = cur[i], b = cur[(i + 1) % nc];
            float da = dot(a - p0, en), db = dot(b - p0, en);
            if (da >= 0.0 && nn < CD_MSET * 2) nxt[nn++] = a;
            if (da * db < 0.0 && nn < CD_MSET * 2) {
                float t = da / (da - db);
                nxt[nn++] = a + t * (b - a);
            }
        }
        nc = nn;
        for (int i = 0; i < nc; ++i) cur[i] = nxt[i];
        if (nc == 0) break;
    }
    for (int i = 0; i < nc; ++i) out[i] = cur[i];
    return nc;
}

kernel void coinGenerateContacts(
    device const CoinBody*           coins         [[ buffer(0) ]],
    device const uint*               sortedIndices [[ buffer(1) ]],
    device const uint*               cellOffsets   [[ buffer(2) ]],
    device const CoinStaticCollider* colliders     [[ buffer(3) ]],
    constant CoinUniforms&           u             [[ buffer(4) ]],
    device const int2*               links         [[ buffer(5) ]],
    device CoinContact*              contacts      [[ buffer(6) ]],
    device atomic_uint*              contactCount  [[ buffer(7) ]],
    constant uint&                   maxContacts   [[ buffer(8) ]],
    device const float4*             hullVerts     [[ buffer(9) ]],
    device const uint2*              hullRanges    [[ buffer(10) ]],
    device const CoinJoint*          joints        [[ buffer(11) ]],
    constant uint&                   jointCount    [[ buffer(12) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody ci = coins[id];
    if (ci.posInvMass.w == 0.0) return;
    int jointPartner = links[id].x;
    float spec = u.speculativeMargin;   // >0 ⇒ also emit near-contacts (anti-tunneling)

    float3 xi = ci.posInvMass.xyz;
    float4 qi = ci.orient;
    float  Ri = cdRadiusOf(ci);
    float  hi = cdHalfThickOf(ci);
    bool   iIsBox = cdIsBox(ci);

    float3 fp[CD_NPTS];
    for (int p = 0; p < CD_NPTS; ++p) fp[p] = xi + cdQuatRotate(qi, cdBodyFeaturePoint(p, ci));
    float3 ni = normalize(cdQuatRotate(qi, float3(0,1,0)));

    // ── Dynamic–dynamic (each pair once: only j with index > id) ──────────────
    int3 base = int3(floor((xi - float3(u.gridMinX, u.gridMinY, u.gridMinZ)) * u.invCell));
    int3 res  = int3(int(u.gridResX), int(u.gridResY), int(u.gridResZ));
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int3 c = base + int3(dx, dy, dz);
        if (any(c < int3(0)) || any(c >= res)) continue;
        uint cell = uint(c.x) + u.gridResX * (uint(c.y) + u.gridResY * uint(c.z));
        uint start = cellOffsets[cell], end = cellOffsets[cell + 1u];
        for (uint s = start; s < end; ++s) {
            uint j = sortedIndices[s];
            if (j <= id) continue;                       // pair once, lower index owns it
            if (int(j) == jointPartner) continue;
            // Generic joints: skip contact generation for a pair that's the two
            // bodies of a joint with collideConnected off (meta.w bit 1 unset,
            // the default) — otherwise a hinge whose bodies touch at the anchor
            // fights its own contact. Bounded scan: jointCount is tens-to-
            // hundreds (see coinJointSolveCS), and only candidate pairs that
            // already passed the broadphase cell + bounding-sphere reject above
            // reach this line.
            bool jointSkip = false;
            for (uint k = 0; k < jointCount; ++k) {
                CoinJoint jn = joints[k];
                if ((jn.meta.w & 1u) == 0u) continue;        // disabled
                if ((jn.meta.w & 2u) != 0u) continue;        // collideConnected → keep contact
                if ((jn.meta.y == id && jn.meta.z == j) || (jn.meta.y == j && jn.meta.z == id)) {
                    jointSkip = true;
                    break;
                }
            }
            if (jointSkip) continue;
            CoinBody cj = coins[j];
            if (cj.posInvMass.w == 0.0) continue;
            float3 xj = cj.posInvMass.xyz;
            float4 qj = cj.orient;
            float  Rj = cdRadiusOf(cj), hj = cdHalfThickOf(cj);
            float3 D  = xi - xj;
            // Bounding reject — widened by the speculative margin, or near-contacts
            // within the margin would be culled before their branch ever runs.
            float reach = sqrt(Ri*Ri + hi*hi) + sqrt(Rj*Rj + hj*hj) + spec;
            if (dot(D, D) > reach * reach) continue;

            // ── Hull-involved pair: LIVE GJK + EPA + clipped manifold ──────
            // (This is the "wire GJK/EPA into the solver" step the constraint-
            // solver doc lists as missing: the EPA contact gets a real support-
            // polygon manifold, so hull stacks rest on a footprint.)
            if (cdIsHull(ci) || cdIsHull(cj)) {
                float3 n; float depth;
                if (!cdGJKEPA(ci, cj, hullVerts, hullRanges, n, depth)) continue;
                if (depth <= 0.0) continue;
                // n points from j(B) toward i(A) — cdEmitContact's convention.
                bool capsuleInvolved = cdIsCapsule(ci) || cdIsCapsule(cj);

                // Capsule↔hull: 2-point manifold (both segment stations vs the
                // hull's near-face support point along the EPA normal), so a
                // capsule lying across a hull face/edge rests level instead of
                // teetering on the single-point EPA contact.
                if (capsuleInvolved) {
                    bool idIsCapsule = cdIsCapsule(ci);
                    CoinBody capBody = idIsCapsule ? ci : cj;
                    float3 nCapWard = idIsCapsule ? n : -n;   // hull → capsule
                    float capR = cdCapsuleR(capBody);
                    float3 s0, s1; cdCapsuleSegment(capBody, s0, s1);
                    CoinBody hullBody = idIsCapsule ? cj : ci;
                    float3 hullSupport = cdSupport(hullBody, nCapWard, hullVerts, hullRanges);
                    float planeOffset = dot(nCapWard, hullSupport);
                    float3 stations[2] = { s0, s1 };
                    int emitted = 0;
                    for (int p = 0; p < 2; ++p) {
                        float pen = (planeOffset + capR) - dot(nCapWard, stations[p]);
                        if (pen > -spec) {
                            float3 cp = stations[p] - capR * nCapWard;
                            cdEmitContact(contacts, contactCount, maxContacts, id, j, uint(p), 0u, n, cp, xi, xj, pen);
                            emitted++;
                        }
                    }
                    if (emitted > 0) continue;
                    // Neither station registered a plane-relative penetration
                    // (e.g. the capsule tip is the true contact, not a side
                    // rest) — fall back to the single EPA contact below.
                    cdEmitContact(contacts, contactCount, maxContacts, id, j, 0u, 0u, n, 0.5 * (cdSupport(ci, -n, hullVerts, hullRanges) + cdSupport(cj, n, hullVerts, hullRanges)), xi, xj, depth);
                    continue;
                }

                if (!cdIsSphere(ci) && !cdIsSphere(cj)) {
                    float3 setA[CD_MSET], setB[CD_MSET];
                    int nA = cdSupportSet(ci, -n, hullVerts, hullRanges, setA);
                    int nB = cdSupportSet(cj,  n, hullVerts, hullRanges, setB);
                    if (nA >= 3 && nB >= 3) {
                        // Project both support polygons onto the contact plane,
                        // clip, reduce to ≤4 spread points.
                        float3 t1, t2; cdContactTangents(n, t1, t2);
                        float3 origin = 0.5 * (xi + xj);
                        float2 A2[CD_MSET], B2[CD_MSET];
                        for (int m = 0; m < nA; ++m) A2[m] = float2(dot(setA[m] - origin, t1), dot(setA[m] - origin, t2));
                        for (int m = 0; m < nB; ++m) B2[m] = float2(dot(setB[m] - origin, t1), dot(setB[m] - origin, t2));
                        nA = cdOrderConvex2D(A2, nA);
                        nB = cdOrderConvex2D(B2, nB);
                        float2 clipped[CD_MSET * 2];
                        int nC = cdClipConvex2D(A2, nA, B2, nB, clipped);
                        if (nC > 0) {
                            // Reduce: extremes along ±t1 / ±t2 (≤4 spread points).
                            int pick[4]; int nP = 0;
                            for (int axm = 0; axm < 4 && nP < min(nC, 4); ++axm) {
                                float bestv = -1e30; int bi = 0;
                                for (int m = 0; m < nC; ++m) {
                                    float v2 = (axm == 0) ? clipped[m].x : (axm == 1) ? -clipped[m].x
                                             : (axm == 2) ? clipped[m].y : -clipped[m].y;
                                    if (v2 > bestv) { bestv = v2; bi = m; }
                                }
                                bool dup = false;
                                for (int m = 0; m < nP; ++m) if (pick[m] == bi) dup = true;
                                if (!dup) pick[nP++] = bi;
                            }

                            // Fit each shape's actual near-face plane (Newell's
                            // method over its own CCW-ordered support set) so
                            // manifold points get a TRUE per-point depth instead
                            // of sharing the single EPA witness depth — the
                            // shared value is only exact at the EPA witness
                            // points; on a tilted face contact the other
                            // manifold points are off by up to the eps-band
                            // width, which reads back as bias-recovery noise.
                            // Falls back to the shared depth wherever the fit
                            // is degenerate or a face sits edge-on to n.
                            float3 setA3[CD_MSET], setB3[CD_MSET];
                            for (int m = 0; m < nA; ++m) setA3[m] = setA[m];
                            for (int m = 0; m < nB; ++m) setB3[m] = setB[m];
                            cdOrderConvex3D(setA3, nA, t1, t2);
                            cdOrderConvex3D(setB3, nB, t1, t2);
                            float3 planeNA, planeNB; float offA, offB;
                            bool haveA = cdFitFacePlane(setA3, nA, planeNA, offA);
                            bool haveB = cdFitFacePlane(setB3, nB, planeNB, offB);
                            if (haveA && dot(planeNA, n) > 0.0) { planeNA = -planeNA; offA = -offA; }
                            if (haveB && dot(planeNB, n) < 0.0) { planeNB = -planeNB; offB = -offB; }

                            int emittedPts = 0;
                            for (int m = 0; m < nP; ++m) {
                                float2 c2 = clipped[pick[m]];
                                float3 cp = origin + c2.x * t1 + c2.y * t2;
                                float pointDepth = depth;
                                if (haveA && haveB) {
                                    float denomA = dot(planeNA, n), denomB = dot(planeNB, n);
                                    if (abs(denomA) > 0.2 && abs(denomB) > 0.2) {
                                        float tA = (offA - dot(planeNA, cp)) / denomA;
                                        float tB = (offB - dot(planeNB, cp)) / denomB;
                                        // pointOnA = cp + tA*n, pointOnB = cp + tB*n; depth is
                                        // how far B's surface has been pushed past A's along n
                                        // (B sits "ahead" along n by convention: n points B→A).
                                        pointDepth = tB - tA;
                                    }
                                }
                                if (pointDepth > -spec) {
                                    cdEmitContact(contacts, contactCount, maxContacts, id, j, uint(m), 0u, n, cp, xi, xj, pointDepth);
                                    emittedPts++;
                                }
                            }
                            if (emittedPts == 0) {
                                // Every corner read non-penetrating under the
                                // tilted-plane fit (a rare near-equilibrium
                                // contact) — don't drop the pair to zero contacts.
                                // EPA's depth>0 is the authoritative overlap test,
                                // so it wins over the plane-fit approximation here;
                                // `origin` (body-center midpoint) stands in for a
                                // contact point since no per-corner one qualified.
                                cdEmitContact(contacts, contactCount, maxContacts, id, j, 0u, 0u, n, origin, xi, xj, depth);
                            }
                            continue;
                        }
                    }
                }
                // Round partner / degenerate clip → the single EPA contact.
                float3 sa = cdSupport(ci, -n, hullVerts, hullRanges);
                float3 sb = cdSupport(cj,  n, hullVerts, hullRanges);
                cdEmitContact(contacts, contactCount, maxContacts, id, j, 0u, 0u, n, 0.5 * (sa + sb), xi, xj, depth);
                continue;
            }

            // Sphere-involved.
            if (cdIsSphere(ci) && cdIsSphere(cj)) {
                float dl = length(D), pen = (Ri + Rj) - dl;
                if (pen > -spec && dl > 1e-6) {
                    float3 n = D / dl;
                    float3 cp = 0.5 * ((xi - n * Ri) + (xj + n * Rj));
                    cdEmitContact(contacts, contactCount, maxContacts, id, j, 0u, 0u, n, cp, xi, xj, pen);
                }
                continue;
            }
            if (cdIsSphere(ci) || cdIsSphere(cj)) {
                bool iSphere = cdIsSphere(ci);
                float3 cs = iSphere ? xi : xj;  float rs = iSphere ? Ri : Rj;
                float3 co = iSphere ? xj : xi;  float4 qo = iSphere ? qj : qi;
                CoinBody O = iSphere ? cj : ci;
                float3 lp = cdQuatRotateInv(qo, cs - co);
                float3 closestLocal;
                float extraR = 0.0;      // the other shape's own surface radius (capsule)
                if (cdIsCapsule(O)) {
                    // Sphere ↔ capsule: closest point on the capsule's SEGMENT, then
                    // the capsule's radius joins the sphere's in the gap test.
                    float hlO = cdCapsuleHL(O);
                    closestLocal = float3(0.0, clamp(lp.y, -hlO, hlO), 0.0);
                    extraR = cdCapsuleR(O);
                } else {
                    closestLocal = cdClosestInShapeLocal(O, lp);
                }
                float3 deltaLocal = lp - closestLocal;
                float dist = length(deltaLocal), pen = rs + extraR - dist;
                if (pen > -spec && dist > 1e-6) {
                    float3 nOut = cdQuatRotate(qo, deltaLocal / dist);   // out of shape toward sphere
                    float3 cp   = co + cdQuatRotate(qo, closestLocal) + extraR * nOut;
                    float3 nBtoA = iSphere ? nOut : -nOut;               // from j(B) toward i(A)
                    cdEmitContact(contacts, contactCount, maxContacts, id, j, 0u, 0u, nBtoA, cp, xi, xj, pen);
                }
                continue;
            }

            // ── Capsule-involved pair (constraint path gets the EXACT capsule) ──
            // A capsule is a segment + radius, so every contact reduces to sphere
            // probes: the true seg-seg closest pair plus the two END probes for the
            // parallel-rest manifold (capsule↔capsule), or 3 probe centres along
            // the segment vs the other shape's closest-point map (capsule↔box/disc).
            if (cdIsCapsule(ci) || cdIsCapsule(cj)) {
                if (cdIsCapsule(ci) && cdIsCapsule(cj)) {
                    float rI = cdCapsuleR(ci), rJ = cdCapsuleR(cj);
                    float3 a0, a1, b0, b1;
                    cdCapsuleSegment(ci, a0, a1);
                    cdCapsuleSegment(cj, b0, b1);
                    float3 cps[3]; float3 cqs[3];
                    float3 c1, c2; cdClosestSegSeg(a0, a1, b0, b1, c1, c2);
                    cps[0] = c1; cqs[0] = c2;                              // true closest pair
                    cps[1] = a0; cqs[1] = cdClosestOnSegment(b0, b1, a0);  // end probes: the
                    cps[2] = a1; cqs[2] = cdClosestOnSegment(b0, b1, a1);  // parallel-rest manifold
                    for (int k = 0; k < 3; ++k) {
                        if (k > 0 && length(cps[k] - cps[0]) < 0.5 * rI) continue;   // dedupe vs main
                        float3 dS = cps[k] - cqs[k];
                        float dl = length(dS);
                        float pen = rI + rJ - dl;
                        if (pen > -spec && dl > 1e-6) {
                            float3 n = dS / dl;                            // out of j, toward i
                            float3 cp = 0.5 * ((cps[k] - n * rI) + (cqs[k] + n * rJ));
                            cdEmitContact(contacts, contactCount, maxContacts, id, j, uint(k), 0u, n, cp, xi, xj, pen);
                        }
                    }
                    continue;
                }
                bool iCap = cdIsCapsule(ci);
                CoinBody C = iCap ? ci : cj;      // the capsule
                CoinBody O = iCap ? cj : ci;      // the box / disc
                float rc = cdCapsuleR(C);
                float3 s0, s1; cdCapsuleSegment(C, s0, s1);
                float3 co = O.posInvMass.xyz; float4 qo = O.orient;
                for (int k = 0; k < 3; ++k) {
                    float3 s = mix(s0, s1, float(k) * 0.5);          // end, mid, end
                    float3 lp = cdQuatRotateInv(qo, s - co);
                    float3 cl = cdClosestInShapeLocal(O, lp);
                    float3 dL = lp - cl;
                    float dl = length(dL);
                    float3 nOut, cp; float pen;
                    if (dl > 1e-6) {
                        pen = rc - dl;
                        if (pen <= -spec) continue;
                        nOut = cdQuatRotate(qo, dL / dl);            // out of O, toward the probe
                        cp = co + cdQuatRotate(qo, cl);
                    } else {
                        // Probe centre INSIDE the shape: push out along the nearest
                        // face (box) / SDF gradient (disc); the depth gains rc.
                        float d2; float3 n2;
                        if (cdIsBox(O)) {
                            if (!cdOrientedBoxPush(s, co, qo, O.shapeExtents.xyz, n2, d2)) continue;
                        } else {
                            float3 nL;
                            float sd = cdCappedCylSDF(lp, cdRadiusOf(O), cdHalfThickOf(O), nL);
                            if (sd >= 0.0) continue;
                            n2 = cdQuatRotate(qo, nL); d2 = -sd;
                        }
                        nOut = n2; pen = d2 + rc; cp = s;
                    }
                    float3 nBtoA = iCap ? nOut : -nOut;              // toward A (thread's body i)
                    cdEmitContact(contacts, contactCount, maxContacts, id, j, uint(k), 0u, nBtoA, cp, xi, xj, pen);
                }
                continue;
            }

            // Box-involved (box–box manifold, or disc–box with phantom-corner reject).
            // Two-phase, shared-depth near-face manifold: any INSIDE corner emits
            // the full manifold (unchanged); with none inside — a fast thin box
            // can cross a thin box/disc between substeps without any corner ever
            // landing "inside" the other — every corner within a tight tolerance
            // of the CLOSEST corner's depth gets a near-contact, all sharing that
            // one depth (item 5 in the constraint-solver doc's roadmap). Testing
            // every corner independently, or using only the single nearest one,
            // both fail a flush symmetric hit: independent depths mix the near
            // face with the trailing one (over-constrained, inconsistent
            // targets); a single off-center contact induces spurious SPIN instead
            // of clean linear deceleration (part of the impulse goes into
            // rotation to satisfy just that one point's relative velocity).
            if (iIsBox || cdIsBox(cj)) {
                bool jIsDisc = !cdIsBox(cj);
                float3 heJ = cdBodyHalfExtents(cj);
                float3 cPos[8]; float3 cN[8]; float cDepth[8]; int cFeat[8];
                bool anyInside = false;
                int nCand = 0;
                float bestDepth = -1e30;
                for (int p = 0; p < 8; ++p) {
                    float3 pw = fp[p]; float3 nB; float depthB;
                    if (!cdOrientedBoxPushSpeculative(pw, xj, qj, heJ, spec, nB, depthB)) continue;
                    if (jIsDisc) { float3 lp = cdQuatRotateInv(qj, pw - xj); if (length(lp.xz) > Rj) continue; }
                    if (depthB >= 0.0) {
                        anyInside = true;
                        cdEmitContact(contacts, contactCount, maxContacts, id, j, uint(p), 0u, nB, pw, xi, xj, depthB);
                    } else {
                        cPos[nCand] = pw; cN[nCand] = nB; cDepth[nCand] = depthB; cFeat[nCand] = p; nCand++;
                        bestDepth = max(bestDepth, depthB);
                    }
                }
                if (!anyInside && nCand > 0) {
                    float tol = max(1e-6, min(abs(spec), min(Ri, hi)) * 0.05);
                    for (int m = 0; m < nCand; ++m) {
                        if (cDepth[m] < bestDepth - tol) continue;
                        cdEmitContact(contacts, contactCount, maxContacts, id, j, uint(cFeat[m]), 0u, cN[m], cPos[m], xi, xj, bestDepth);
                    }
                }
                if (!anyInside && nCand == 0) {
                    float minDepth; float3 nS, cpS;
                    if (cdBoxBoxSAT(xi, qi, cdBodyHalfExtents(ci), xj, qj, heJ, minDepth, nS, cpS))
                        cdEmitContact(contacts, contactCount, maxContacts, id, j, 14u, 0u, nS, cpS, xi, xj, minDepth);
                }
                continue;
            }

            // Disc–disc analytic SAT (face normals + centre line).
            {
                float3 nj = normalize(cdQuatRotate(qj, float3(0,1,0)));
                float3 axes[3] = { ni, nj, float3(0,1,0) };
                float dlen = length(D); if (dlen > 1e-5) axes[2] = D / dlen;
                float md = 1e30; float3 bestN = float3(0,1,0); bool separated = false;
                for (int ax = 0; ax < 3; ++ax) {
                    float3 a = axes[ax];
                    float di = abs(dot(ni, a)), dj = abs(dot(nj, a));
                    float ei = hi * di + Ri * sqrt(max(0.0, 1.0 - di*di));
                    float ej = hj * dj + Rj * sqrt(max(0.0, 1.0 - dj*dj));
                    float proj = dot(D, a);
                    float depth = (ei + ej) - abs(proj);
                    if (depth <= 0.0) { separated = true; break; }
                    if (depth < md) { md = depth; bestN = a * (proj >= 0.0 ? 1.0 : -1.0); }
                }
                if (separated) continue;
                float3 n = normalize(bestN);
                float dniN = dot(ni, n), dnjN = dot(nj, n);
                float eiN = hi * abs(dniN) + Ri * sqrt(max(0.0, 1.0 - dniN*dniN));
                float ejN = hj * abs(dnjN) + Rj * sqrt(max(0.0, 1.0 - dnjN*dnjN));
                float3 cp = 0.5 * ((xi - n * eiN) + (xj + n * ejN));
                cdEmitContact(contacts, contactCount, maxContacts, id, j, 0u, 0u, n, cp, xi, xj, md);
            }
        }
    }

    // ── Static / kinematic colliders (planes, boxes, pusher) ──────────────────
    // A CAPSULE resolves against every static as sphere probes at 3 segment
    // stations (both ends + mid, radius r) — the exact round-cap surface, not the
    // bounding cylinder's hard rim (which is what the generic feature-point path
    // would test). Probes are precomputed once here.
    bool iCapsule = cdIsCapsule(ci);
    float capR = iCapsule ? cdCapsuleR(ci) : 0.0;
    float3 capP[3];
    if (iCapsule) {
        float3 s0, s1; cdCapsuleSegment(ci, s0, s1);
        capP[0] = s0; capP[1] = 0.5 * (s0 + s1); capP[2] = s1;
    }
    // A HULL body probes statics with its true vertices (≤32), radius 0 — the
    // same point logic as the disc/box feature points, but on the real shape.
    bool iHull = cdIsHull(ci);
    uint2 hRange = uint2(0, 0);
    if (iHull) hRange = hullRanges[uint(ci.hullRef.x + 0.5)];
    int nFP = iHull ? int(min(hRange.y, uint(CD_MAX_STATIC_FP))) : CD_NPTS;
#define CD_PROBE(pp) (iHull ? (xi + cdQuatRotate(qi, float3(hullVerts[hRange.x + uint(pp)].xyz))) : fp[pp])
    for (uint k = 0u; k < u.colliderCount; ++k) {
        CoinStaticCollider col = colliders[k];
        uint kind = as_type<uint>(col.a.w);
        if (kind == 0u) {                                // plane n·p ≥ d
            float3 n = col.a.xyz; float d = col.b.w;
            if (cdIsSphere(ci)) {
                float pen = (d + Ri) - dot(n, xi);
                if (pen > -spec) cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, 0u, k, n, xi - Ri*n, xi, xi, pen);
            } else if (iCapsule) {
                // Both END probes (skip mid: coplanar with the ends, adds nothing) —
                // the 2-point manifold that holds a side-lying capsule level.
                for (int p = 0; p < 3; p += 2) {
                    float pen = (d + capR) - dot(n, capP[p]);
                    if (pen > -spec) cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, capP[p] - capR*n, xi, xi, pen);
                }
            } else {
                for (int p = 0; p < nFP; ++p) {
                    float3 pw = CD_PROBE(p);
                    float pen = d - dot(n, pw);
                    if (pen > -spec) cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, pw, xi, xi, pen);
                }
            }
        } else if (kind == 1u) {                         // axis-aligned box
            float3 ctr = col.a.xyz, he = col.b.xyz;
            bool oneWay = (col.meta.x & 1u) != 0u;
            if (cdIsSphere(ci)) {
                // Sphere vs AABB: exact closest point. Previously a sphere fell
                // into the generic feature-point loop below — that loop samples
                // the DISC surrogate's 14 points (cdBodyFeaturePoint's fallback
                // for a non-box shape), an approximation for a round body; this
                // mirrors kind==3's OBB sphere branch (identity rotation).
                float3 lp = xi - ctr;
                float3 cl = clamp(lp, -he, he);
                float3 dlt = lp - cl;
                float dist2 = dot(dlt, dlt);
                float3 n; float pen;
                if (dist2 > 1e-10) {                      // centre outside the box
                    float dist = sqrt(dist2);
                    pen = Ri - dist;
                    n = dlt / dist;
                } else {                                  // centre inside: least-penetration axis
                    float3 d = he - abs(lp);
                    n = (d.x < d.y && d.x < d.z) ? float3(sign(lp.x),0,0)
                      : (d.y < d.z)              ? float3(0,sign(lp.y),0)
                      :                            float3(0,0,sign(lp.z));
                    pen = min(d.x, min(d.y, d.z)) + Ri;
                }
                if (pen > -spec && !(oneWay && n.y < 0.5)) {
                    cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, 0u, k, n, xi - Ri*n, xi, xi, pen);
                }
            } else if (iCapsule) {
                for (int p = 0; p < 3; ++p) {
                    float3 lp = capP[p] - ctr;
                    float3 cl = clamp(lp, -he, he);
                    float3 dL = lp - cl;
                    float dl = length(dL);
                    float3 n; float pen; float3 cp;
                    if (dl > 1e-6) {
                        pen = capR - dl;
                        if (pen <= -spec) continue;
                        n = dL / dl; cp = ctr + cl;
                    } else {                              // probe centre inside the box
                        float3 dist = he - abs(lp); float push;
                        if (dist.x < dist.y && dist.x < dist.z) { n = float3(sign(lp.x),0,0); push = dist.x; }
                        else if (dist.y < dist.z)               { n = float3(0,sign(lp.y),0); push = dist.y; }
                        else                                    { n = float3(0,0,sign(lp.z)); push = dist.z; }
                        pen = push + capR; cp = capP[p];
                    }
                    if (oneWay && n.y < 0.5) continue;
                    cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, cp, xi, xi, pen);
                }
            } else {
                // Box/disc feature points vs the wall. Two-phase: if any point is
                // genuinely INSIDE, emit the full multi-point manifold (unchanged —
                // needed for resting stability). Otherwise the body isn't touching
                // yet — a fast thin box can still tunnel a thin wall between
                // substeps without any corner ever landing "inside" (item 5 in the
                // constraint-solver doc's roadmap). The near face's corners must
                // share ONE target depth (the closest approach), not each their
                // own: a lone off-center speculative contact induces spurious
                // SPIN instead of clean linear deceleration for a flush hit (part
                // of the correcting impulse goes into rotation to satisfy that one
                // point's own relative velocity), and testing every point
                // independently mixes the near face with the trailing one — an
                // over-constrained, physically-inconsistent target set. So: find
                // the overall closest point, then emit every OTHER point within a
                // tight tolerance of that SAME depth (its own near face; a
                // genuinely farther face differs by ~the body's size, far above
                // the tolerance) using the SHARED depth + normal.
                float4 qId = float4(0, 0, 0, 1);
                float3 cPos[CD_MAX_STATIC_FP]; float3 cN[CD_MAX_STATIC_FP]; float cDepth[CD_MAX_STATIC_FP]; int cFeat[CD_MAX_STATIC_FP];
                bool anyInside = false;
                int nCand = 0;
                float bestDepth = -1e30;
                for (int p = 0; p < nFP; ++p) {
                    float3 pw = CD_PROBE(p);
                    float3 n; float push;
                    if (!cdOrientedBoxPushSpeculative(pw, ctr, qId, he, spec, n, push)) continue;
                    if (oneWay && n.y < 0.5) continue;
                    if (push >= 0.0) {
                        anyInside = true;
                        cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, pw, xi, xi, push);
                    } else if (nCand < CD_MAX_STATIC_FP) {
                        cPos[nCand] = pw; cN[nCand] = n; cDepth[nCand] = push; cFeat[nCand] = p; nCand++;
                        bestDepth = max(bestDepth, push);
                    }
                }
                if (!anyInside) {
                    // Bounded by both the margin AND the querying body's own
                    // bounding scale, so a projectile THINNER than 5% of the
                    // margin can't have its trailing face misread as "near" too.
                    float tol = max(1e-6, min(abs(spec), min(Ri, hi)) * 0.05);
                    for (int m = 0; m < nCand; ++m) {
                        if (cDepth[m] < bestDepth - tol) continue;   // a farther face, not this near-contact
                        cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(cFeat[m]), k, cN[m], cPos[m], xi, xi, bestDepth);
                    }
                }
            }
        } else if (kind == 3u) {                         // oriented box (plank / ramp)
            float3 ctr = col.a.xyz, he = col.b.xyz;
            float4 q = col.orient;
            if (cdIsSphere(ci)) {
                // Sphere vs OBB: closest point on the box to the sphere centre,
                // in the box's local frame, then back to world.
                float3 lp = cdQuatRotateInv(q, xi - ctr);
                float3 cl = clamp(lp, -he, he);
                float3 dlt = lp - cl;
                float dist2 = dot(dlt, dlt);
                if (dist2 > 1e-10) {                      // centre outside the box
                    float dist = sqrt(dist2);
                    float pen = Ri - dist;
                    if (pen > -spec) {
                        float3 n = cdQuatRotate(q, dlt / dist);
                        cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, 0u, k, n, xi - Ri*n, xi, xi, pen);
                    }
                } else {                                  // centre inside: push out least-penetration axis
                    float3 d = he - abs(lp);
                    float3 nL = (d.x < d.y && d.x < d.z) ? float3(sign(lp.x),0,0)
                              : (d.y < d.z)              ? float3(0,sign(lp.y),0)
                              :                            float3(0,0,sign(lp.z));
                    float pen = min(d.x, min(d.y, d.z)) + Ri;
                    float3 n = cdQuatRotate(q, nL);
                    cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, 0u, k, n, xi - Ri*n, xi, xi, pen);
                }
            } else if (iCapsule) {
                // Capsule vs OBB: the 3 segment probes in the box's local frame.
                for (int p = 0; p < 3; ++p) {
                    float3 lp = cdQuatRotateInv(q, capP[p] - ctr);
                    float3 cl = clamp(lp, -he, he);
                    float3 dL = lp - cl;
                    float dl = length(dL);
                    float3 nL; float pen; float3 cp;
                    if (dl > 1e-6) {
                        pen = capR - dl;
                        if (pen <= -spec) continue;
                        nL = dL / dl; cp = ctr + cdQuatRotate(q, cl);
                    } else {
                        float3 dd = he - abs(lp); float push;
                        if (dd.x < dd.y && dd.x < dd.z) { nL = float3(sign(lp.x),0,0); push = dd.x; }
                        else if (dd.y < dd.z)           { nL = float3(0,sign(lp.y),0); push = dd.y; }
                        else                            { nL = float3(0,0,sign(lp.z)); push = dd.z; }
                        pen = push + capR; cp = capP[p];
                    }
                    float3 n = cdQuatRotate(q, nL);
                    cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, cp, xi, xi, pen);
                }
            } else {
                // Two-phase, shared-depth near-face manifold — see the kind==1u
                // generic loop's comment.
                float3 cPos[CD_MAX_STATIC_FP]; float3 cN[CD_MAX_STATIC_FP]; float cDepth[CD_MAX_STATIC_FP]; int cFeat[CD_MAX_STATIC_FP];
                bool anyInside = false;
                int nCand = 0;
                float bestDepth = -1e30;
                for (int p = 0; p < nFP; ++p) {
                    float3 pw = CD_PROBE(p);
                    float3 n; float push;
                    if (!cdOrientedBoxPushSpeculative(pw, ctr, q, he, spec, n, push)) continue;
                    if (push >= 0.0) {
                        anyInside = true;
                        cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, pw, xi, xi, push);
                    } else if (nCand < CD_MAX_STATIC_FP) {
                        cPos[nCand] = pw; cN[nCand] = n; cDepth[nCand] = push; cFeat[nCand] = p; nCand++;
                        bestDepth = max(bestDepth, push);
                    }
                }
                if (!anyInside) {
                    float tol = max(1e-6, min(abs(spec), min(Ri, hi)) * 0.05);
                    for (int m = 0; m < nCand; ++m) {
                        if (cDepth[m] < bestDepth - tol) continue;
                        cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(cFeat[m]), k, cN[m], cPos[m], xi, xi, bestDepth);
                    }
                }
            }
        } else if (kind == 4u) {                         // cylinder segment (pipe / half-pipe)
            // a.xyz = axis centre, b.xyz = axis dir (unit), b.w = radius R,
            // vel.xyz = "up" dir, vel.w = half-length, meta.x bit0 = lower-half only.
            // A sphere is one probe (its centre, radius Ri); a capsule its 3 segment
            // probes (radius capR); a disc/box its surface FEATURE POINTS (radius 0)
            // — so every shape rides the pipe, not just marbles.
            float3 ctr = col.a.xyz, ax = col.b.xyz, up = col.vel.xyz;
            float R = col.b.w, halfLen = col.vel.w;
            bool lowerOnly = (col.meta.x & 1u) != 0u;
            bool sphereLike = cdIsSphere(ci) || iCapsule;
            int nProbes = iCapsule ? 3 : (cdIsSphere(ci) ? 1 : nFP);
            for (int p = 0; p < nProbes; ++p) {
                float3 s  = sphereLike ? (iCapsule ? capP[p] : xi) : CD_PROBE(p);
                float  rs = sphereLike ? (iCapsule ? capR    : Ri) : 0.0;
                float3 d = s - ctr;
                float along = dot(d, ax);
                if (along < -halfLen || along > halfLen) continue;   // outside this segment's length
                float3 radial = d - ax * along;
                float dr = length(radial);
                if (dr <= 1e-5 || dr >= R) continue;       // ONLY when the centre is INSIDE the tube
                float3 rn = radial / dr;                    // outward from axis
                bool active = !lowerOnly || dot(rn, up) < 0.05;   // half-pipe: lower hemisphere
                float pen = (dr + rs) - R;                  // surface vs inner wall (>0 ⇒ touching)
                if (active && pen > -spec) {
                    float3 n = -rn;                         // push toward the axis
                    cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, s + rn * rs, xi, xi, pen);
                }
            }
        } else if (kind == 2u) {
            // ── Reciprocating pusher plate (constraint-path parity) ────────────
            // Legacy semantics preserved: a body the plate has overtaken is pushed
            // to just in front of its +Z face, at most plateSpeed·dt per substep —
            // the depth clamp means the split-impulse recovery can never move a
            // body faster than the plate itself (no launching). Probes: sphere =
            // its centre (radius Ri), capsule = 3 segment probes (radius capR),
            // disc/box = feature points with the legacy (Ri, hi) margins.
            float3 cc2 = col.a.xyz, he2 = col.b.xyz;
            float frontZ = cc2.z + he2.z;
            float backZ  = cc2.z - he2.z;
            float maxPush = max(0.0, col.vel.z) * u.dt;
            if (maxPush <= 0.0) continue;
            // The shove arrives through the split-impulse bias, which recovers
            // β·(depth − slop)/dt — so cap the DEPTH such that the recovery speed
            // caps at the plate speed (depth ≤ push/β + slop), not the raw push
            // (which β·slop would eat: the plate would ghost through the pile).
            float maxDepth = maxPush / max(u.baumgarteBeta, 0.05) + u.contactSlop;
            bool sphereLike = cdIsSphere(ci) || iCapsule;
            int nProbes = iCapsule ? 3 : (cdIsSphere(ci) ? 1 : nFP);
            for (int p = 0; p < nProbes; ++p) {
                float3 s  = sphereLike ? (iCapsule ? capP[p] : xi) : CD_PROBE(p);
                float  rs = sphereLike ? (iCapsule ? capR    : Ri) : (iHull ? 0.0 : Ri);
                float  ry = sphereLike ? rs : (iHull ? 0.0 : hi);
                if (abs(s.x - cc2.x) < he2.x + rs &&
                    s.y > cc2.y - (he2.y + ry) && s.y < cc2.y + (he2.y + ry) &&
                    s.z < frontZ + rs && s.z > backZ - rs) {
                    float depth = clamp((frontZ + rs) - s.z, 0.0, maxDepth);
                    if (depth > 0.0) {
                        cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k,
                                      float3(0, 0, 1), s, xi, xi, depth);
                    }
                }
            }
        }
    }
#undef CD_PROBE
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 2: graph colouring (for GPU Gauss-Seidel)
// ══════════════════════════════════════════════════════════════════════════════
//
// The velocity/position solves write to BOTH bodies of a contact, so two contacts
// that share a body must NOT run in parallel (race) — and solving them sequentially
// (Gauss-Seidel) is what gives real convergence on a dense pile, unlike Jacobi. We
// partition contacts into COLOURS such that no two contacts in a colour share a body;
// the solver then runs the colours in sequence (Gauss-Seidel between colours) and the
// contacts within a colour fully in parallel (race-free).
//
// Method: build per-body contact lists, then Jones-Plassmann-Luby greedy colouring —
// a contact colours itself in a round iff its random PRIORITY beats every still-
// uncolored neighbour's, taking the lowest colour no coloured neighbour uses. Random
// priority (a hash of the contact index) colours ~half the frontier per round, so it
// converges in ~log(#contacts) rounds; a fixed round budget is encoded with no
// per-round readback (mirrors how the legacy solver batches its iterations).

constant uint CD_MAX_BODY_CONTACTS = 48u;   // per-body contact-list capacity
constant uint CD_MAX_COLORS        = 32u;   // colour fits in a uint mask

static uint cdHashU(uint x) {               // priority hash (deterministic, no RNG state)
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}

// Append `cid` to one body's contact list (atomic, bounded).
static void cdAppendBodyContact(device atomic_uint* bodyContactCount,
                                device uint* bodyContacts, uint body, uint cid) {
    uint slot = atomic_fetch_add_explicit(&bodyContactCount[body], 1u, memory_order_relaxed);
    if (slot < CD_MAX_BODY_CONTACTS) bodyContacts[body * CD_MAX_BODY_CONTACTS + slot] = cid;
}

kernel void coinClearBodyContacts(
    device atomic_uint* bodyContactCount [[ buffer(0) ]],
    uint id [[ thread_position_in_grid ]])
{
    atomic_store_explicit(&bodyContactCount[id], 0u, memory_order_relaxed);
}

kernel void coinBuildBodyContacts(
    device const CoinContact* contacts        [[ buffer(0) ]],
    device const atomic_uint& contactCount    [[ buffer(1) ]],
    device uint*              bodyContacts     [[ buffer(2) ]],
    device atomic_uint*       bodyContactCount [[ buffer(3) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    cdAppendBodyContact(bodyContactCount, bodyContacts, c.meta.x, cid);
    if (c.meta.y != CD_STATIC) cdAppendBodyContact(bodyContactCount, bodyContacts, c.meta.y, cid);
}

// One colouring round. A contact colours itself iff its priority is the strict max
// among still-uncoloured neighbours (ties → larger cid), choosing the lowest colour
// no coloured neighbour occupies. Only local-priority-maxima write, and a writer's
// neighbours never write the same round, so single-buffered colours are race-free.
kernel void coinColorRound(
    device CoinContact*       contacts         [[ buffer(0) ]],
    device const atomic_uint& contactCount     [[ buffer(1) ]],
    device const uint*        bodyContacts     [[ buffer(2) ]],
    device const uint*        bodyContactCount [[ buffer(3) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    if (c.tan2.w >= 0.0) return;                 // already coloured
    uint myPri = cdHashU(cid);

    uint bodies[2]; int nb = 0;
    bodies[nb++] = c.meta.x;
    if (c.meta.y != CD_STATIC) bodies[nb++] = c.meta.y;

    uint usedMask = 0u;
    for (int bi = 0; bi < nb; ++bi) {
        uint body = bodies[bi];
        uint cnt = min(bodyContactCount[body], CD_MAX_BODY_CONTACTS);
        for (uint k = 0; k < cnt; ++k) {
            uint ncid = bodyContacts[body * CD_MAX_BODY_CONTACTS + k];
            if (ncid == cid) continue;
            float ncolor = contacts[ncid].tan2.w;
            if (ncolor < 0.0) {                  // uncoloured neighbour competes for this round
                uint nPri = cdHashU(ncid);
                if (nPri > myPri || (nPri == myPri && ncid > cid)) return;   // not our turn
            } else {
                uint cc = uint(ncolor);
                if (cc < CD_MAX_COLORS) usedMask |= (1u << cc);
            }
        }
    }
    uint color = 0u;
    while (color < CD_MAX_COLORS && (usedMask & (1u << color))) color++;
    if (color < CD_MAX_COLORS) contacts[cid].tan2.w = float(color);   // else stays −1 (overflow; solved atomically)
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 3: sequential-impulse velocity solve + split-impulse
// ══════════════════════════════════════════════════════════════════════════════
//
// The real engine. Per substep:
//   coinIntegrateVelocityCS   v += g·dt (no position move yet); store prevPos/Orient;
//                             clear the per-body BIAS (pseudo) velocity.
//   × iterations, × colour:
//     coinSolveVelocityColor  for each contact in this colour (race-free — no shared
//                             body): solve the NORMAL velocity constraint (relative
//                             normal velocity → 0, with restitution above a threshold),
//                             a SEPARATE split-impulse BIAS constraint (drives a pseudo
//                             velocity that recovers penetration beyond `slop`, never
//                             touching real velocity → zero energy injection at rest),
//                             and two-axis COULOMB friction (|jt| ≤ μ·jn).
//   coinIntegratePositionCS   x += (v + biasLin)·dt ; q ⊕= (ω + biasAng)·dt  — the bias
//                             velocity moves position then is discarded.
//   coinFinalizeCS            sleep a slow supported body; clamp speeds.
//
// Contacts carry accumulated impulses in rA.w / rB.w / tan1.w (warm-startable, Stage 4);
// they're regenerated each substep so the accumulation is per-substep until then.

// Per-body bias (pseudo) velocity: [0]=linear.xyz, [1]=angular.xyz. One pair per body.
// (Separate buffer so the split-impulse recovery never aliases real velocity.)

kernel void coinIntegrateVelocityCS(
    device CoinBody*       coins  [[ buffer(0) ]],
    device float4*         bias   [[ buffer(1) ]],
    constant CoinUniforms& u      [[ buffer(2) ]],
    device const uint*     asleep [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    bias[2*id] = float4(0.0); bias[2*id+1] = float4(0.0);
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) return;
    if (asleep[id] != 0u) {                  // frozen: no gravity, no drift
        coins[id].prevPos.xyz = c.posInvMass.xyz; coins[id].prevOrient = c.orient;
        coins[id].vel.xyz = float3(0.0); coins[id].angVel.xyz = float3(0.0);
        return;
    }
    float3 v = c.vel.xyz;
    v.y -= u.gravity * u.dt;
    if (u.quadraticDrag > 0.0) {
        v -= cdDragK(u, c.prevPos.w) * length(v) * v * u.dt;   // ∝v² drag, ∝1/r per body
    }
    v   *= u.linDamping;
    coins[id].prevPos.xyz = c.posInvMass.xyz;     // for finalize sleep / diagnostics
    coins[id].prevOrient  = c.orient;
    coins[id].vel.xyz     = v;
    coins[id].angVel.xyz  = c.angVel.xyz * u.angDamping;
}

// Write the indirect-dispatch threadgroup count for the contact-iterating kernels:
// only the ACTUAL contact count needs threads, not the buffer capacity (a settled
// 176-pile has ~600 contacts vs ~12k capacity — a ~19× over-dispatch otherwise).
kernel void coinWriteSolveArgs(
    device const atomic_uint& contactCount [[ buffer(0) ]],
    device uint*              args         [[ buffer(1) ]],   // 3 separate uints, NOT a
    constant uint&            tgSize       [[ buffer(2) ]],   // uint3 (16-byte) into a 12-byte buf
    uint id [[ thread_position_in_grid ]])
{
    if (id != 0) return;
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    args[0] = max((n + tgSize - 1u) / tgSize, 1u);            // threadgroupsPerGrid.x
    args[1] = 1u;                                             // .y
    args[2] = 1u;                                             // .z
}

// Solve all contacts of one colour. `currentColor` is passed per-dispatch (setBytes).
kernel void coinSolveVelocityColor(
    device CoinBody*          coins        [[ buffer(0) ]],
    device float4*            bias         [[ buffer(1) ]],
    device CoinContact*       contacts     [[ buffer(2) ]],
    device const atomic_uint& contactCount [[ buffer(3) ]],
    constant CoinUniforms&    u            [[ buffer(4) ]],
    constant uint&            currentColor [[ buffer(5) ]],
    device const uint*        asleep       [[ buffer(6) ]],
    device const float2*      material     [[ buffer(7) ]],   // per-body (μ, e); <0 = inherit
    device const CoinStaticCollider* colliders [[ buffer(8) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint ncon = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= ncon) return;
    CoinContact c = contacts[cid];
    if (c.tan2.w < 0.0 || uint(c.tan2.w) != currentColor) return;

    uint A = c.meta.x, B = c.meta.y;
    bool bStatic = (B == CD_STATIC);
    // A sleeping body acts as immovable (invMass 0) for an awake neighbour; if BOTH
    // ends are inert there's nothing to solve — skip (the island-sleep perf win).
    bool aSleep = (asleep[A] != 0u);
    bool bSleep = bStatic || (asleep[B] != 0u);
    if (aSleep && bSleep) return;

    CoinBody a = coins[A];
    float invMa = aSleep ? 0.0 : a.posInvMass.w;
    float3 invIa = aSleep ? float3(0.0) : cdBodyInvInertia(a, a.posInvMass.w);
    float4 qa = a.orient;
    float3 vA = a.vel.xyz, wA = a.angVel.xyz;
    float3 bvA = bias[2*A].xyz, bwA = bias[2*A+1].xyz;

    float invMb = 0.0; float3 invIb = float3(0.0); float4 qb = float4(0,0,0,1);
    float3 vB = float3(0.0), wB = float3(0.0), bvB = float3(0.0), bwB = float3(0.0);
    CoinBody b;
    if (!bStatic) {
        b = coins[B];
        invMb = bSleep ? 0.0 : b.posInvMass.w;
        invIb = bSleep ? float3(0.0) : cdBodyInvInertia(b, b.posInvMass.w);
        qb = b.orient;
        vB = b.vel.xyz; wB = b.angVel.xyz; bvB = bias[2*B].xyz; bwB = bias[2*B+1].xyz;
    }

    float3 n = c.nrm.xyz, rA = c.rA.xyz, rB = c.rB.xyz, t1 = c.tan1.xyz, t2 = c.tan2.xyz;
    float depth = c.nrm.w;

    // Effective mass along a unit direction d at this contact.
    float3 raXn1 = cross(rA, t1), raXn2 = cross(rA, t2), raXnn = cross(rA, n);
    float3 rbXn1 = cross(rB, t1), rbXn2 = cross(rB, t2), rbXnn = cross(rB, n);
    float kN = invMa + invMb
             + dot(raXnn, cdApplyInvInertiaWorld(qa, invIa, raXnn))
             + (bStatic ? 0.0 : dot(rbXnn, cdApplyInvInertiaWorld(qb, invIb, rbXnn)));
    kN = max(kN, 1e-8);

    // n points from B toward A: contact-point relative velocity (A − B).
    float3 vpA = vA + cross(wA, rA);
    float3 vpB = bStatic ? float3(0.0) : (vB + cross(wB, rB));
    float3 vrel = vpA - vpB;
    float vn = dot(vrel, n);                       // <0 = closing along the normal

    // ── Per-contact material: combine the two sides' (μ, e). A static side
    // reads the COLLIDER's own material (c.meta.z is the collider index for a
    // static contact — see cdEmitContact/cdPairKey), not a mirror of A's — an
    // icy ramp next to a rubber floor now actually differs. Same combine rule
    // and "<0 = inherit the global uniform" convention as per-body material, so
    // the all-defaults case still reduces exactly to the global uniforms.
    float muA = cdBodyMu(material, A, u);
    float eA  = cdBodyE(material, A, u);
    float muB = bStatic ? cdColliderMu(colliders, c.meta.z, u) : cdBodyMu(material, B, u);
    float eB  = bStatic ? cdColliderE(colliders, c.meta.z, u) : cdBodyE(material, B, u);
    float muC = sqrt(max(muA * muB, 0.0));      // geometric mean (Box2D convention)
    float eC  = max(eA, eB);                    // bounciest material wins

    // ── REAL normal impulse (restitution only above the threshold) ────────────
    // The restitution target is anchored to the PRE-SOLVE approach speed vn₀,
    // captured once on this contact's first solve pass — every later iteration
    // then drives toward the same −e·vn₀ instead of re-deriving it from the
    // already-reflected current velocity (which zeroed the bounce).
    if (c.aux.w < 0.5) { c.aux.x = vn; c.aux.w = 1.0; }
    float vn0 = c.aux.x;
    float restE = (vn0 < -u.restThreshold) ? cdEffectiveCORBase(u, eC, -vn0) : 0.0;
    // SPECULATIVE near-contact (depth < 0): don't stop the body at the current
    // gap — only cap its approach so it can close AT MOST the gap this substep
    // (vn ≥ depth/dt). A real contact (depth ≥ 0) keeps the plain vn → −e·vn₀.
    float allowedVn = min(depth, 0.0) / max(u.dt, 1e-6);
    float jnOld = c.rA.w;
    float dJn = -(vn - allowedVn + restE * vn0) / kN;   // drive vn → allowedVn − e·vn₀
    float jnNew = max(jnOld + dJn, 0.0);           // accumulated, non-adhesive
    dJn = jnNew - jnOld;
    float3 Pn = dJn * n;
    vA += invMa * Pn; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pn));
    if (!bStatic) { vB -= invMb * Pn; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pn)); }
    c.rA.w = jnNew;

    // ── SPLIT-IMPULSE bias: recover penetration beyond slop via pseudo velocity ─
    // No per-contact accumulator (the per-body bias velocity carries convergence);
    // the depth>slop target is ≥0, so the pseudo impulse stays separating.
    float3 bvpA = bvA + cross(bwA, rA);
    float3 bvpB = bStatic ? float3(0.0) : (bvB + cross(bwB, rB));
    float bvn = dot(bvpA - bvpB, n);
    float biasTarget = u.baumgarteBeta * max(depth - u.contactSlop, 0.0) / max(u.dt, 1e-6);
    float dJb = (biasTarget - bvn) / kN;
    float3 Pb = dJb * n;
    bvA += invMa * Pb; bwA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pb));
    if (!bStatic) { bvB -= invMb * Pb; bwB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pb)); }

    // ── Two-axis Coulomb friction (accumulated jt1→rB.w, jt2→tan1.w; |jt|≤μ·jn) ─
    float mu = muC;
    if (mu > 0.0) {
        float kT1 = invMa + invMb + dot(raXn1, cdApplyInvInertiaWorld(qa, invIa, raXn1))
                  + (bStatic ? 0.0 : dot(rbXn1, cdApplyInvInertiaWorld(qb, invIb, rbXn1)));
        float kT2 = invMa + invMb + dot(raXn2, cdApplyInvInertiaWorld(qa, invIa, raXn2))
                  + (bStatic ? 0.0 : dot(rbXn2, cdApplyInvInertiaWorld(qb, invIb, rbXn2)));
        float bound = mu * c.rA.w;
        // tangent 1
        vrel = (vA + cross(wA, rA)) - (bStatic ? float3(0.0) : (vB + cross(wB, rB)));
        float jt1Old = c.rB.w;
        float jt1New = clamp(jt1Old - dot(vrel, t1) / max(kT1, 1e-8), -bound, bound);
        float3 Pt1 = (jt1New - jt1Old) * t1;
        vA += invMa * Pt1; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pt1));
        if (!bStatic) { vB -= invMb * Pt1; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pt1)); }
        c.rB.w = jt1New;
        // tangent 2
        vrel = (vA + cross(wA, rA)) - (bStatic ? float3(0.0) : (vB + cross(wB, rB)));
        float jt2Old = c.tan1.w;
        float jt2New = clamp(jt2Old - dot(vrel, t2) / max(kT2, 1e-8), -bound, bound);
        float3 Pt2 = (jt2New - jt2Old) * t2;
        vA += invMa * Pt2; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pt2));
        if (!bStatic) { vB -= invMb * Pt2; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pt2)); }
        c.tan1.w = jt2New;
    }

    // ── Rolling resistance (constraint path): a physical angular constraint ────
    // Opposes the RELATIVE rolling spin (ω ⟂ n) with an angular impulse bounded
    // by μᵣ · jₙ · r (coefficient × normal impulse × contact lever) — so a ball
    // rolls out and STOPS on a level floor instead of coasting forever, and the
    // stopping torque scales with how hard the contact is loaded, exactly like
    // real rolling friction. Twist ABOUT n is friction's job, not this.
    if (u.rollingResistance > 0.0 && c.rA.w > 0.0) {
        float3 wrel = wA - wB;
        float3 wRoll = wrel - dot(wrel, n) * n;
        float wl = length(wRoll);
        if (wl > 1e-5) {
            float3 axis = wRoll / wl;
            float kR = dot(axis, cdApplyInvInertiaWorld(qa, invIa, axis))
                     + (bStatic ? 0.0 : dot(axis, cdApplyInvInertiaWorld(qb, invIb, axis)));
            if (kR > 1e-9) {
                float lever = max(length(rA), 1e-4);
                float jR = min(wl / kR, u.rollingResistance * c.rA.w * lever);
                float3 T = -jR * axis;
                wA += cdApplyInvInertiaWorld(qa, invIa, T);
                if (!bStatic) wB -= cdApplyInvInertiaWorld(qb, invIb, T);
            }
        }
    }

    // Write back (an inert/asleep end carried no impulse, so leave it untouched).
    if (!aSleep) {
        coins[A].vel.xyz = vA; coins[A].angVel.xyz = wA;
        bias[2*A] = float4(bvA, 0.0); bias[2*A+1] = float4(bwA, 0.0);
    }
    if (!bSleep) {
        coins[B].vel.xyz = vB; coins[B].angVel.xyz = wB;
        bias[2*B] = float4(bvB, 0.0); bias[2*B+1] = float4(bwB, 0.0);
    }
    contacts[cid] = c;
}

kernel void coinIntegratePositionCS(
    device CoinBody*       coins  [[ buffer(0) ]],
    device const float4*   bias   [[ buffer(1) ]],
    constant CoinUniforms& u      [[ buffer(2) ]],
    device const uint*     asleep [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) return;
    if (asleep[id] != 0u) return;                 // frozen: no position move
    float3 v = c.vel.xyz + bias[2*id].xyz;        // real + pseudo for the position move
    float3 w = c.angVel.xyz + bias[2*id+1].xyz;
    coins[id].posInvMass.xyz = c.posInvMass.xyz + v * u.dt;
    coins[id].orient = cdIntegrateQuat(c.orient, w, u.dt);
    // bias is discarded (zeroed next substep) — it never persists into real velocity.
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 4: warm-started persistent manifolds
// ══════════════════════════════════════════════════════════════════════════════
//
// A contact re-found next substep should start from the impulse it converged to last
// substep, not from zero — this is the single biggest convergence win for resting
// stacks (the stack settles ONCE and stays, instead of re-solving from scratch and
// re-jiggling every step). We snapshot the solved contacts, hash them by `pairKey`
// (open addressing), and on the next substep copy the matching old impulses into the
// fresh contacts and apply them before iterating.

constant uint CD_HASH_EMPTY = 0xFFFFFFFFu;

kernel void coinClearHash(
    device uint* pairHash [[ buffer(0) ]],
    uint i [[ thread_position_in_grid ]])
{
    pairHash[i] = CD_HASH_EMPTY;
}

// Snapshot the solved contacts into `prev` (so they survive the next regeneration),
// and insert each into the open-addressing hash (pairKey → prev slot).
kernel void coinSnapshotContacts(
    device const CoinContact*  contacts     [[ buffer(0) ]],
    device const atomic_uint&  contactCount [[ buffer(1) ]],
    device CoinContact*        prev         [[ buffer(2) ]],
    device atomic_uint*        pairHash     [[ buffer(3) ]],
    constant uint&             hashSize     [[ buffer(4) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    prev[cid] = c;
    uint key = c.meta.w;
    uint h = cdHashU(key) & (hashSize - 1u);
    for (uint probe = 0; probe < 32u; ++probe) {       // linear probe, bounded
        uint slot = (h + probe) & (hashSize - 1u);
        uint expected = CD_HASH_EMPTY;
        if (atomic_compare_exchange_weak_explicit(&pairHash[slot], &expected, cid,
                memory_order_relaxed, memory_order_relaxed)) return;
    }
}

// For each FRESH contact, find last substep's matching contact by pairKey and copy its
// converged impulses (normal→rA.w, friction1→rB.w, friction2→tan1.w) as the warm seed.
kernel void coinWarmStartMatch(
    device CoinContact*        contacts     [[ buffer(0) ]],
    device const atomic_uint&  contactCount [[ buffer(1) ]],
    device const CoinContact*  prev         [[ buffer(2) ]],
    device const uint*         pairHash     [[ buffer(3) ]],
    constant uint&             hashSize     [[ buffer(4) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= n) return;
    uint key = contacts[cid].meta.w;
    uint h = cdHashU(key) & (hashSize - 1u);
    for (uint probe = 0; probe < 32u; ++probe) {
        uint slot = pairHash[(h + probe) & (hashSize - 1u)];
        if (slot == CD_HASH_EMPTY) break;
        if (prev[slot].meta.w == key) {
            contacts[cid].rA.w   = prev[slot].rA.w;     // warm seed (the solve continues from here)
            contacts[cid].rB.w   = prev[slot].rB.w;
            contacts[cid].tan1.w = prev[slot].tan1.w;
            return;
        }
    }
}

// Apply the warm-start impulses to the bodies' velocities (per colour, race-free).
kernel void coinWarmStartApply(
    device CoinBody*          coins        [[ buffer(0) ]],
    device CoinContact*       contacts     [[ buffer(1) ]],
    device const atomic_uint& contactCount [[ buffer(2) ]],
    constant uint&            currentColor [[ buffer(3) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint ncon = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= ncon) return;
    CoinContact c = contacts[cid];
    if (c.tan2.w < 0.0 || uint(c.tan2.w) != currentColor) return;
    float jn = c.rA.w, jt1 = c.rB.w, jt2 = c.tan1.w;
    if (jn == 0.0 && jt1 == 0.0 && jt2 == 0.0) return;
    uint A = c.meta.x, B = c.meta.y; bool bStatic = (B == CD_STATIC);
    float3 P = jn * c.nrm.xyz + jt1 * c.tan1.xyz + jt2 * c.tan2.xyz;
    CoinBody a = coins[A]; float invMa = a.posInvMass.w; float3 invIa = cdBodyInvInertia(a, invMa);
    coins[A].vel.xyz    += invMa * P;
    coins[A].angVel.xyz += cdApplyInvInertiaWorld(a.orient, invIa, cross(c.rA.xyz, P));
    if (!bStatic) {
        CoinBody b = coins[B]; float invMb = b.posInvMass.w; float3 invIb = cdBodyInvInertia(b, invMb);
        coins[B].vel.xyz    -= invMb * P;
        coins[B].angVel.xyz -= cdApplyInvInertiaWorld(b.orient, invIb, cross(c.rB.xyz, P));
    }
}

kernel void coinFinalizeCS(
    device CoinBody*       coins  [[ buffer(0) ]],
    constant CoinUniforms& u      [[ buffer(1) ]],
    device const uint*     asleep [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) return;
    if (asleep[id] != 0u) { coins[id].vel.xyz = float3(0.0); coins[id].angVel.xyz = float3(0.0); return; }
    float3 v = c.vel.xyz, w = c.angVel.xyz;

    // Hard safety floor (a fast body that outran a substep). Half the min extent
    // for EVERY shape — deliberately BELOW rest height. The constraint path
    // generates contacts from the PREVIOUS substep's end state, so a backstop AT
    // rest height (the old sphere special case, floorY + R) dead-stopped a falling
    // sphere before a floor contact could ever exist: the plane contact — and with
    // it restitution — never fired, and every ball landed dead. At half-extent the
    // backstop only catches true tunnelers; the contact solve owns the landing.
    float floorH = min(cdRadiusOf(c), cdHalfThickOf(c));
    float restY = floorH * 0.5;
    float3 x = c.posInvMass.xyz;
    if (x.y < u.floorY + restY) { x.y = u.floorY + restY; if (v.y < 0.0) v.y = 0.0; }
    coins[id].posInvMass.xyz = x;

    // Speed caps.
    float sp = length(v); if (sp > u.maxSpeed) v *= u.maxSpeed / sp;
    float os = length(w); if (os > u.maxOmega) w *= u.maxOmega / os;

    // Sleep a slow body to a dead stop (no per-frame micro-creep).
    if (length(v) < u.sleepLinVel && length(w) < 0.6) { v = float3(0.0); w = float3(0.0); }

    coins[id].vel.xyz = v;
    coins[id].angVel.xyz = w;
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 7: generic joints (ball / hinge / distance)
// ══════════════════════════════════════════════════════════════════════════════
//
// Bilateral constraints between two dynamic bodies (or a body and the WORLD),
// solved at velocity level exactly like contacts — real velocity toward the
// constraint manifold, position drift recovered through the SAME split-impulse
// bias velocities (never mixed into real velocity, so a settled articulated
// assembly carries zero restoring energy). One thread solves all joints
// serially (true Gauss-Seidel): joint counts are tens-to-hundreds, not the
// thousands contacts reach, so a serial loop is cheaper and simpler than
// colouring them, and it is exactly race-free.
//
// Types (meta.x): 0 = BALL (anchors coincide, 3-DOF point constraint),
//                 1 = HINGE (ball + axis alignment + optional angle limits),
//                 2 = DISTANCE (anchor separation = rest length, 1-DOF).
// (CoinJoint itself is declared earlier, alongside CoinContact, so
// coinGenerateContacts can also read it for the collideConnected check.)

// Invert a symmetric positive-definite 3×3 (cofactor expansion).
static float3x3 cdInvert3x3(float3x3 m) {
    float3 c0 = cross(m[1], m[2]);
    float3 c1 = cross(m[2], m[0]);
    float3 c2 = cross(m[0], m[1]);
    float det = dot(m[0], c0);
    if (abs(det) < 1e-12) return float3x3(0.0);
    float inv = 1.0 / det;
    // rows of the inverse are the cofactor columns / det (m is symmetric here).
    return float3x3(c0 * inv, c1 * inv, c2 * inv);
}

kernel void coinJointSolveCS(
    device CoinBody*        coins      [[ buffer(0) ]],
    device float4*          bias       [[ buffer(1) ]],
    device const CoinJoint* joints     [[ buffer(2) ]],
    constant uint&          jointCount [[ buffer(3) ]],
    constant CoinUniforms&  u          [[ buffer(4) ]],
    device const uint*      asleep     [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]])
{
    if (tid != 0) return;                    // serial Gauss-Seidel over all joints
    for (uint k = 0; k < jointCount; ++k) {
        CoinJoint jn = joints[k];
        if (jn.meta.w == 0u) continue;       // disabled slot
        uint A = jn.meta.y, B = jn.meta.z;
        bool bWorld = (B == CD_STATIC);
        CoinBody a = coins[A];
        if (a.posInvMass.w == 0.0) continue;
        bool aSleep = (asleep[A] != 0u);
        bool bSleep = bWorld || (asleep[B] != 0u);
        if (aSleep && bSleep) continue;

        float invMa = aSleep ? 0.0 : a.posInvMass.w;
        float3 invIa = aSleep ? float3(0.0) : cdBodyInvInertia(a, a.posInvMass.w);
        float4 qa = a.orient;
        float3 vA = a.vel.xyz, wA = a.angVel.xyz;
        float3 bvA = bias[2*A].xyz, bwA = bias[2*A+1].xyz;

        CoinBody b;
        float invMb = 0.0; float3 invIb = float3(0.0); float4 qb = float4(0,0,0,1);
        float3 vB = float3(0.0), wB = float3(0.0), bvB = float3(0.0), bwB = float3(0.0);
        float3 xb = float3(0.0);
        if (!bWorld) {
            b = coins[B];
            if (b.posInvMass.w == 0.0) continue;
            invMb = bSleep ? 0.0 : b.posInvMass.w;
            invIb = bSleep ? float3(0.0) : cdBodyInvInertia(b, b.posInvMass.w);
            qb = b.orient;
            vB = b.vel.xyz; wB = b.angVel.xyz;
            bvB = bias[2*B].xyz; bwB = bias[2*B+1].xyz;
            xb = b.posInvMass.xyz;
        }

        // World anchors + lever arms. For a world joint, anchorB IS world space.
        float3 pA = a.posInvMass.xyz + cdQuatRotate(qa, jn.anchorA.xyz);
        float3 pB = bWorld ? jn.anchorB.xyz : (xb + cdQuatRotate(qb, jn.anchorB.xyz));
        float3 rA = pA - a.posInvMass.xyz;
        float3 rB = bWorld ? float3(0.0) : (pB - xb);
        uint type = jn.meta.x;

        if (type == 2u) {
            // ── DISTANCE: |pA − pB| = rest, 1-DOF along the anchor axis ────────
            float3 d = pA - pB;
            float len = length(d);
            if (len < 1e-6) continue;
            float3 n = d / len;
            float C = len - jn.anchorA.w;                 // signed stretch
            float3 raXn = cross(rA, n), rbXn = cross(rB, n);
            float kJ = invMa + invMb
                     + dot(raXn, cdApplyInvInertiaWorld(qa, invIa, raXn))
                     + (bWorld ? 0.0 : dot(rbXn, cdApplyInvInertiaWorld(qb, invIb, rbXn)));
            if (kJ < 1e-9) continue;
            // Real velocity along n → 0 (bilateral, unclamped).
            float vn = dot((vA + cross(wA, rA)) - (vB + cross(wB, rB)), n);
            float jI = -vn / kJ;
            float3 P = jI * n;
            vA += invMa * P; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, P));
            if (!bWorld) { vB -= invMb * P; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, P)); }
            // Bias velocity recovers the stretch (split-impulse; never real energy).
            float bvn = dot((bvA + cross(bwA, rA)) - (bvB + cross(bwB, rB)), n);
            float jB = (-u.baumgarteBeta * C / max(u.dt, 1e-6) - bvn) / kJ;
            float3 Pb = jB * n;
            bvA += invMa * Pb; bwA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pb));
            if (!bWorld) { bvB -= invMb * Pb; bwB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pb)); }
        } else {
            // ── BALL core (types 0 and 1): anchors coincide, 3-DOF ─────────────
            // K built by columns: K = (ΣinvM)·I − [rA]× IA⁻¹ [rA]× − [rB]× IB⁻¹ [rB]×.
            float3x3 K;
            for (int j = 0; j < 3; ++j) {
                float3 e = float3(j == 0 ? 1.0 : 0.0, j == 1 ? 1.0 : 0.0, j == 2 ? 1.0 : 0.0);
                float3 col = (invMa + invMb) * e;
                col -= cross(rA, cdApplyInvInertiaWorld(qa, invIa, cross(rA, e)));
                if (!bWorld) col -= cross(rB, cdApplyInvInertiaWorld(qb, invIb, cross(rB, e)));
                K[j] = col;
            }
            float3x3 Kinv = cdInvert3x3(K);
            // Real: relative anchor velocity → 0.
            float3 vrel = (vA + cross(wA, rA)) - (vB + cross(wB, rB));
            float3 P = Kinv * (-vrel);
            vA += invMa * P; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, P));
            if (!bWorld) { vB -= invMb * P; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, P)); }
            // Bias: recover the anchor gap C = pA − pB.
            float3 C = pA - pB;
            float3 bvrel = (bvA + cross(bwA, rA)) - (bvB + cross(bwB, rB));
            float3 Pb = Kinv * (-u.baumgarteBeta * C / max(u.dt, 1e-6) - bvrel);
            bvA += invMa * Pb; bwA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pb));
            if (!bWorld) { bvB -= invMb * Pb; bwB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pb)); }

            if (type == 1u) {
                // ── HINGE: kill relative rotation ⟂ the axis + align the axes ──
                float3 aW = normalize(cdQuatRotate(qa, jn.axisA.xyz));
                float3 bW = bWorld ? normalize(jn.axisB.xyz) : normalize(cdQuatRotate(qb, jn.axisB.xyz));
                // Two directions spanning ⟂ aW.
                float3 b1 = (abs(aW.x) > 0.7) ? normalize(float3(-aW.y, aW.x, 0.0))
                                              : normalize(float3(0.0, -aW.z, aW.y));
                float3 b2 = cross(aW, b1);
                float3 axisErr = cross(aW, bW);           // sin-scaled misalignment
                for (int t = 0; t < 2; ++t) {
                    float3 dir = (t == 0) ? b1 : b2;
                    float kA = dot(dir, cdApplyInvInertiaWorld(qa, invIa, dir))
                             + (bWorld ? 0.0 : dot(dir, cdApplyInvInertiaWorld(qb, invIb, dir)));
                    if (kA < 1e-9) continue;
                    // Real: relative angular velocity ⟂ axis → 0.
                    float wrel = dot(wA - wB, dir);
                    float jA = -wrel / kA;
                    wA += cdApplyInvInertiaWorld(qa, invIa, jA * dir);
                    if (!bWorld) wB -= cdApplyInvInertiaWorld(qb, invIb, jA * dir);
                    // Bias: rotate the axes back into alignment.
                    float bwrel = dot(bwA - bwB, dir);
                    float errT = dot(axisErr, dir);
                    float jAb = (-u.baumgarteBeta * errT / max(u.dt, 1e-6) - bwrel) / kA;
                    bwA += cdApplyInvInertiaWorld(qa, invIa, jAb * dir);
                    if (!bWorld) bwB -= cdApplyInvInertiaWorld(qb, invIb, jAb * dir);
                }
                // Angle limits about the hinge axis (lo < hi enables them).
                float lo = jn.axisA.w, hi = jn.axisB.w;
                if (lo < hi) {
                    // Twist of B relative to A about the WORLD hinge axis
                    // (swing-twist projection of the relative quaternion).
                    float4 qrel = bWorld ? cdQuatConj(qa) : cdQuatMul(qb, cdQuatConj(qa));
                    float twist = 2.0 * atan2(dot(qrel.xyz, aW), qrel.w);
                    float overLo = twist - lo;             // < 0 ⇒ below the low stop
                    float overHi = twist - hi;             // > 0 ⇒ past the high stop
                    float kT = dot(aW, cdApplyInvInertiaWorld(qa, invIa, aW))
                             + (bWorld ? 0.0 : dot(aW, cdApplyInvInertiaWorld(qb, invIb, aW)));
                    if (kT > 1e-9 && (overLo < 0.0 || overHi > 0.0)) {
                        float err = (overLo < 0.0) ? overLo : overHi;
                        // Real: stop the twist velocity heading further past the stop.
                        float wt = dot(wB - wA, aW);        // d(twist)/dt
                        if ((err < 0.0 && wt < 0.0) || (err > 0.0 && wt > 0.0)) {
                            float jT = -wt / kT;
                            wA -= cdApplyInvInertiaWorld(qa, invIa, jT * aW);
                            if (!bWorld) wB += cdApplyInvInertiaWorld(qb, invIb, jT * aW);
                        }
                        // Bias: rotate back inside the limit.
                        float bwt = dot(bwB - bwA, aW);
                        float jTb = (-u.baumgarteBeta * err / max(u.dt, 1e-6) - bwt) / kT;
                        bwA -= cdApplyInvInertiaWorld(qa, invIa, jTb * aW);
                        if (!bWorld) bwB += cdApplyInvInertiaWorld(qb, invIb, jTb * aW);
                    }
                }
            }
        }

        // Write back (a sleeping/world end carried no impulse — leave it be).
        if (!aSleep) {
            coins[A].vel.xyz = vA; coins[A].angVel.xyz = wA;
            bias[2*A] = float4(bvA, 0.0); bias[2*A+1] = float4(bwA, 0.0);
        }
        if (!bWorld && !bSleep) {
            coins[B].vel.xyz = vB; coins[B].angVel.xyz = wB;
            bias[2*B] = float4(bvB, 0.0); bias[2*B+1] = float4(bwB, 0.0);
        }
    }
}

// Union the two bodies of each ENABLED joint into one island, so an articulated
// assembly sleeps and wakes as a unit (mirror of coinIslandUnion over contacts).
kernel void coinIslandUnionJoints(
    device atomic_uint*     label      [[ buffer(0) ]],
    device const CoinJoint* joints     [[ buffer(1) ]],
    constant uint&          jointCount [[ buffer(2) ]],
    uint k [[ thread_position_in_grid ]])
{
    if (k >= jointCount) return;
    CoinJoint jn = joints[k];
    if (jn.meta.w == 0u || jn.meta.z == CD_STATIC) return;
    uint a = jn.meta.y, b = jn.meta.z;
    uint la = atomic_load_explicit(&label[a], memory_order_relaxed);
    uint lb = atomic_load_explicit(&label[b], memory_order_relaxed);
    uint lo = min(la, lb);
    atomic_fetch_min_explicit(&label[a], lo, memory_order_relaxed);
    atomic_fetch_min_explicit(&label[b], lo, memory_order_relaxed);
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 5: island detection + sleeping
// ══════════════════════════════════════════════════════════════════════════════
//
// A settled heap should cost ~nothing. We label connected components (ISLANDS) of the
// contact graph by union-find, track how long each body has been slow, and DEACTIVATE
// a whole island once every body in it has been slow for `sleepFrames` — then the
// integrate/solve/integrate-position kernels skip it. Per-ISLAND (not per-body) so one
// twitching body keeps its neighbours awake, and a new contact from an awake body
// unions it into a sleeping island and wakes the lot. Run once per FRAME (sleep state
// changes slowly), so the union-find cost is amortised. `label`/`islandMin` are the
// connectivity scratch; `sleepTimer` persists across frames; `asleep` gates the solver.

kernel void coinIslandInit(
    device uint*           label    [[ buffer(0) ]],
    constant CoinUniforms& u        [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    label[id] = id;
}

// Union the two bodies of each contact (atomic-min connectivity; converges over rounds).
kernel void coinIslandUnion(
    device atomic_uint*       label        [[ buffer(0) ]],
    device const CoinContact* contacts     [[ buffer(1) ]],
    device const atomic_uint& contactCount [[ buffer(2) ]],
    uint cid [[ thread_position_in_grid ]])
{
    uint n = atomic_load_explicit(&contactCount, memory_order_relaxed);
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    if (c.meta.y == CD_STATIC) return;                 // statics aren't solver DOFs
    uint a = c.meta.x, b = c.meta.y;
    uint la = atomic_load_explicit(&label[a], memory_order_relaxed);
    uint lb = atomic_load_explicit(&label[b], memory_order_relaxed);
    uint lo = min(la, lb);
    atomic_fetch_min_explicit(&label[a], lo, memory_order_relaxed);
    atomic_fetch_min_explicit(&label[b], lo, memory_order_relaxed);
}

// Pointer-jump: flatten label[id] toward its component root (run a few times after union).
kernel void coinIslandJump(
    device uint*           label [[ buffer(0) ]],
    constant CoinUniforms& u     [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    uint l = label[id];
    label[id] = label[l];
}

// Per body: tick the slow-frame counter; then reset the per-island min accumulator.
kernel void coinSleepTick(
    device const CoinBody* coins      [[ buffer(0) ]],
    device uint*           sleepTimer [[ buffer(1) ]],
    device uint*           islandMin  [[ buffer(2) ]],
    constant CoinUniforms& u          [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    islandMin[id] = 0xFFFFFFFFu;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) { sleepTimer[id] = 0u; return; }
    bool slow = length(c.vel.xyz) < u.sleepLinVel * 2.5 && length(c.angVel.xyz) < 1.0;
    // Hysteresis: a slow frame INCREMENTS, a fast frame DECREMENTS (not a hard reset) —
    // so a transient contact kick doesn't keep an otherwise-settled island awake, while
    // a genuinely moving body (sustained fast frames) still decays to 0 and wakes it.
    uint t = sleepTimer[id];
    sleepTimer[id] = slow ? min(t + 1u, 100000u) : (t > 6u ? t - 6u : 0u);
}

// Reduce each island's MIN slow-frame count (an island is only as asleep as its
// most-recently-moved body — so any motion anywhere keeps the whole island awake).
kernel void coinIslandMinReduce(
    device const uint*  label      [[ buffer(0) ]],
    device const uint*  sleepTimer [[ buffer(1) ]],
    device atomic_uint* islandMin  [[ buffer(2) ]],
    constant CoinUniforms& u       [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    atomic_fetch_min_explicit(&islandMin[label[id]], sleepTimer[id], memory_order_relaxed);
}

// Mark a body asleep iff its whole island has been slow for ≥ sleepFrames.
kernel void coinSleepMark(
    device const uint*     label     [[ buffer(0) ]],
    device const uint*     islandMin [[ buffer(1) ]],
    device uint*           asleep    [[ buffer(2) ]],
    constant CoinUniforms& u         [[ buffer(3) ]],
    constant uint&         sleepFrames [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= u.coinCount) return;
    asleep[id] = (islandMin[label[id]] >= sleepFrames) ? 1u : 0u;
}

// Diagnostic probe: run GJK+EPA on coins[a],coins[b]; write [overlap, depthµm, nx,ny,nz].
kernel void coinGJKEPAProbe(
    device const CoinBody* coins      [[ buffer(0) ]],
    constant uint2&        pair       [[ buffer(1) ]],
    device float4*         result     [[ buffer(2) ]],
    device const float4*   hullVerts  [[ buffer(3) ]],
    device const uint2*    hullRanges [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id != 0) return;
    float3 nrm; float depth;
    bool hit = cdGJKEPA(coins[pair.x], coins[pair.y], hullVerts, hullRanges, nrm, depth);
    result[0] = float4(hit ? 1.0 : 0.0, depth, 0.0, 0.0);
    result[1] = float4(nrm, 0.0);
}
