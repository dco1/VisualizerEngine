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
    float4 shapeExtents;// w = shape tag (0 disc/cylinder, 1 box, 2 sphere, 3 capsule, 4 hull, 5 ovoid); box: xyz = half-extents, sphere: x = radius, capsule: x = seg half-length, ovoid: x = fat radius, y = tip radius, z = tip-centre offset
    float4 hullRef;     // hull: x = hull index (float); ovoid: x = fat-centre offset; both: yzw = per-unit-mass INVERSE inertia diag (principal frame)
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
    // Opt-in solver behaviour bits (0 ⇒ every existing scene is byte-identical):
    //   CD_FLAG_SCALE_AWARE_DEADSTOP   — finalize/sleep "slow" test uses the body's
    //                                    fastest surface speed |v| + |ω|·R_bound (VZ-0149)
    //   CD_FLAG_ACCUM_ROLLING          — rolling resistance clamps the ACCUMULATED
    //                                    per-contact impulse, not each iteration (VZ-0155)
    //   CD_FLAG_NO_COLLIDER_CULL       — TEST SEAM: disable coinGenerateContacts' per-
    //                                    collider bounding reject, so a test can prove the
    //                                    reject never changes the contact set (VZ-0151)
    uint  solverFlags;
    // Largest tight bounding radius of any body spawned so far (CoinDEMSolver.noteBodyBound):
    // sizes the broadphase neighbourhood when a body outgrows the cell (VZ-0161).
    float maxBodyBound;
    // Gauss–Seidel passes over a grouped manifold's points per velocity iteration
    // (CD_FLAG_MANIFOLD_SOLVE only; CoinDEMSolver.manifoldInnerPasses).
    uint  manifoldPasses;
};
constant uint CD_FLAG_SCALE_AWARE_DEADSTOP = 1u;
constant uint CD_FLAG_ACCUM_ROLLING        = 2u;
constant uint CD_FLAG_NO_COLLIDER_CULL     = 4u;
//   CD_FLAG_MANIFOLD_SOLVE — a multi-point manifold is emitted contiguously, coloured as
//                            ONE unit and solved by one thread (cdSolveContactVelocity) —
//                            VZ-0157 / VZ-0163. Off: every contact is its own unit, as ever.
constant uint CD_FLAG_MANIFOLD_SOLVE       = 8u;
//   CD_FLAG_TORSION        — torsional (point) friction rows about each contact normal, from the
//                            per-body contact patch radius (engine plan 5a, stage B2; see
//                            cdSolveContactTorsion). Set only while some body has a patch > 0.
constant uint CD_FLAG_TORSION              = 16u;
//   CD_FLAG_SEPARATING_BIAS — the split-impulse BIAS row only ever pushes a contact's bodies APART
//                            (VZ-02xx, Digital Clock stock): its pseudo impulse is clamped ≥ 0, and a
//                            speculative contact (depth < 0, a real gap) only resists a bias approach
//                            that would close more than its gap this substep. Off: the row drives the
//                            relative bias velocity to its target in BOTH directions — so a contact
//                            that is merely within the speculative margin (two bars 0.5 mm apart, a
//                            bar and a stake) or within the slop PULLS its bodies together whenever
//                            another contact's recovery moves one of them away: a bilateral glue in
//                            the position channel between bodies that do not touch. A 2 × 5 on-edge
//                            bar stack on a dynamic stake pallet (margin 2.5 mm) never slept and
//                            walked ≥ 1 mm (10 of 20 bars off their places in 10 s); at margins
//                            below its 0.5 mm gaps it slept in 0.6 s. Opt-in (every other world is
//                            bit-identical).
constant uint CD_FLAG_SEPARATING_BIAS      = 128u;
//   CD_FLAG_KEEP_ASLEEP_CONTACTS — a sleeping island keeps its warm start: every substep the
//                            contacts whose ends are both inert (asleep, or asleep vs a static) are
//                            CARRIED from the last substep's list into this one (coinCarryDormant —
//                            copied, not regenerated: VZ-0152 still skips their narrowphase), marked
//                            DORMANT (ext.z = 1): not coloured, not solved, kept only so that the
//                            substep a touch wakes the island, its fresh contacts find their
//                            converged impulses in the warm-start table. Without it every wake is
//                            a cold start (Digital Clock's bar stock: a woken 2 × 5 stack sank at
//                            14–48 mm/s in its first frame and crept 0.7–1.4 mm). Opt-in.
constant uint CD_FLAG_KEEP_ASLEEP_CONTACTS = 256u;

// The split-impulse bias row's pseudo impulse (see CD_FLAG_SEPARATING_BIAS): `legacy` is the
// default row's (target − bvn)/kN, returned untouched unless the flag is set.
static inline float cdSeparatingBias(float legacy, float depth, float bvn, float kN, constant CoinUniforms& u) {
    if ((u.solverFlags & CD_FLAG_SEPARATING_BIAS) == 0u) return legacy;
    float target = (depth < 0.0) ? depth / max(u.dt, 1e-6)
                                 : u.baumgarteBeta * max(depth - u.contactSlop, 0.0) / max(u.dt, 1e-6);
    return max((target - bvn) / kN, 0.0);
}
// colorContacts entries: contact index | (grouped-manifold point count << 28).
constant uint CD_CC_SHIFT = 28u;
constant uint CD_CC_MASK  = 0x0FFFFFFFu;

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

/// Surface velocity of a KINEMATIC collider: the pusher plate (kind 2) and host-moved
/// boxes / oriented boxes (kinds 1, 3 — e.g. a running character's body re-placed every
/// frame, which must hand its real velocity to what it hits, or a kick only ever shoves).
/// Every existing box / oriented box carries vel = 0, so they are unchanged. Planes are
/// world-fixed, and kind 4 (cylinder segment) reuses `vel.xyz` as its "up" axis, so this
/// must switch on the kind rather than reading `vel` blindly.
inline float3 cdColliderVelocity(device const CoinStaticCollider* colliders, uint idx) {
    CoinStaticCollider c = colliders[idx];
    uint kind = as_type<uint>(c.a.w);
    return (kind == 1u || kind == 2u || kind == 3u) ? c.vel.xyz : float3(0.0);
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
// constraints. float4 lanes only (the ALIGNMENT RULE); 128 bytes.
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
    // impulse clamp (a resting-stack feature, a bounce killer). yz = accumulated
    // rolling-resistance impulse in the (tan1, tan2) basis (CD_FLAG_ACCUM_ROLLING only).
    // w BEFORE the first solve pass is the manifold marker (CD_FLAG_MANIFOLD_SOLVE): 0 an
    // ungrouped contact, −n the HEAD of a grouped n-point manifold (its points are the n
    // contiguous contacts from here), −0.25 a non-head point; the solve's capture then
    // overwrites it with 1 (every value is < 0.5, so the capture test is unchanged).
    float4 aux;
    // Stage B2 (engine plan 5a, CD_FLAG_TORSION): x = accumulated TORSIONAL impulse about n
    // (N·m·s, + on A / − on B; warm-started with the other rows), y = the torsion capacity
    // per unit normal impulse this contact was solved with, μ·r_patch·share (m; 0 = no
    // torsion row), zw reserved. Written 0 by every emitter; nothing reads it with the flag
    // clear. (128 bytes.)
    float4 ext;
};
static uint cdManifoldCount(CoinContact c)  { return c.aux.w <= -0.5 ? uint(-c.aux.w + 0.5) : 1u; }
static bool cdIsManifoldMember(CoinContact c) { return c.aux.w > -0.5 && c.aux.w < 0.0; }
// A contact carried through a sleep (CD_FLAG_KEEP_ASLEEP_CONTACTS, coinCarryDormant — ext.z = 1):
// only the warm start reads it (it is never coloured or solved; both its ends are inert).
static bool cdIsDormant(CoinContact c) { return c.ext.z > 0.5; }

// One generic constraint-path joint. Types (meta.x): 0 = BALL (anchors
// coincide, 3-DOF point constraint), 1 = HINGE (ball + axis alignment +
// optional angle limits + optional motor), 2 = DISTANCE (anchor separation =
// rest length, 1-DOF), 3 = PRISMATIC (axis alignment + rotation fully locked
// + optional linear limits + optional motor — 1-DOF translation only),
// 4 = WELD (ball + rotation fully locked: 0-DOF, VZ-0150).
// meta.w bit 0 = enabled, bit 1 = collideConnected (contacts between this
// joint's own bodyA/bodyB are suppressed in coinGenerateContacts unless this
// bit is set — a hinge whose bodies touch at the anchor would otherwise fight
// its own contact). Declared here (not by the joint-solve kernel below, where
// it conceptually lives) so coinGenerateContacts can read it too. 96 bytes.
struct CoinJoint {
    uint4  meta;     // x = type, y = bodyA, z = bodyB (CD_STATIC = world), w = enabled|collideConnected bits
    // xyz = anchor in A-local frame. w: DISTANCE only, rest length; HINGE or
    // PRISMATIC only, motor target velocity (rad/s or m/s) — otherwise dead
    // weight (BALL has no use for it either).
    float4 anchorA;
    // xyz = anchor in B-local frame (WORLD if z==CD_STATIC). w: HINGE or
    // PRISMATIC, motor max torque/force (>0 enables the motor); DISTANCE, the
    // swing-friction arm c (m, VZ-0168 — 0 = a frictionless rod); otherwise dead weight.
    float4 anchorB;
    float4 axisA;    // xyz = hinge/slide axis in A-local frame;      w = limit lo (rad or m)
    float4 axisB;    // xyz = hinge/slide axis in B-local (WORLD if world); w = limit hi (rad or m)
    // Relative orientation at creation, conj(qA)·qB (B's orientation in A's frame; a
    // world joint: conj(qA)), unit quaternion (x, y, z, w). The WELD / PRISMATIC rotation
    // lock and the HINGE twist (its limits) are measured against it (VZ-0156), so a
    // joint made between bodies at ANY relative orientation holds that pose.
    float4 ref;
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
// 3 = capsule, 4 = convex hull, 5 = ovoid (egg), 6 = compound of boxes (cdIsCompound,
// CoinDEMNarrowphase.h). (Every predicate MUST be a
// half-open band — a bare `> 0.5` box test would mis-collide every later tag as
// a box, and the old bare `> 3.5` hull test would have read an ovoid's
// hullRef.x — its fat-sphere offset — as a hull-table index.)
static bool cdIsBox(CoinBody c)     { return c.shapeExtents.w > 0.5 && c.shapeExtents.w < 1.5; }
static bool cdIsSphere(CoinBody c)  { return c.shapeExtents.w > 1.5 && c.shapeExtents.w < 2.5; }
static bool cdIsCapsule(CoinBody c) { return c.shapeExtents.w > 2.5 && c.shapeExtents.w < 3.5; }
static bool cdIsHull(CoinBody c)    { return c.shapeExtents.w > 3.5 && c.shapeExtents.w < 4.5; }
static bool cdIsEgg(CoinBody c)     { return c.shapeExtents.w > 4.5 && c.shapeExtents.w < 5.5; }

// Capsule lanes: prevPos.w = cross-section RADIUS r (like a disc), vel.w = the
// FULL half-height hl + r (so the legacy capped-cylinder path sees the correct
// bounds), shapeExtents.x = the true SEGMENT half-length hl for exact contacts.
static float cdCapsuleR(CoinBody c)  { return c.prevPos.w > 1e-4 ? c.prevPos.w : 0.02; }
static float cdCapsuleHL(CoinBody c) { return max(c.shapeExtents.x, 0.0); }

// ── OVOID (egg) — tag 5: a sphere-swept cone ─────────────────────────────────
// The convex hull of TWO spheres of different radii on the local +Y axis: the
// fat sphere (radius rFat, centre at yFat < 0) and the tip sphere (radius rTip,
// centre at yTip > 0), with the tangent cone flank between them. Smooth
// everywhere, so it rolls and wobbles like a real egg — no facets. Offsets are
// measured from the COM (the body's position), which the CPU integrates over
// the true solid of revolution at spawn, so the asymmetric mass distribution —
// the egg's signature settle-to-the-fat-end wobble — is real, not staged.
//
// Lanes: shapeExtents = (rFat, rTip, yTip, 5); hullRef.x = yFat, hullRef.yzw =
// the per-unit-mass INVERSE inertia diagonal (integrated at spawn, hull's
// convention); prevPos.w = bounding radius; vel.w = full half-height.
//
// Contact strategy (constraint path ONLY, like hulls): every query treats the
// egg as sphere probes on its axis segment with a linearly-varying radius.
// Against a PLANE the two end-sphere probes are EXACT — the swept surface's
// signed plane distance is linear in the sweep parameter, so its minimum is at
// an endpoint. Against boxes/discs/cylinders it is the capsule 3-station probe
// scheme with the station's own radius. GJK/EPA pairs get the exact support
// map (the deeper of the two end-sphere supports).
static float  cdEggRFat(CoinBody c) { return max(c.shapeExtents.x, 1e-4); }
static float  cdEggRTip(CoinBody c) { return max(c.shapeExtents.y, 1e-4); }
// World-space sphere centres: a = fat end, b = tip end.
static void cdEggSegment(CoinBody c, thread float3& a, thread float3& b) {
    float3 x = c.posInvMass.xyz;
    a = x + cdQuatRotate(c.orient, float3(0.0, c.hullRef.x, 0.0));
    b = x + cdQuatRotate(c.orient, float3(0.0, c.shapeExtents.z, 0.0));
}
// Surface radius at sweep parameter t (0 = fat end, 1 = tip end).
static float cdEggRadiusAt(CoinBody c, float t) {
    return mix(cdEggRFat(c), cdEggRTip(c), clamp(t, 0.0, 1.0));
}

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
    // Ovoid: the same convention — integrated over the solid of revolution at
    // spawn (only hullRef.x differs: hull index vs fat-sphere offset).
    // Compound (tag 6): the same convention — the parallel-axis sum over its boxes,
    // diagonalized at registration (CoinCompound.swift). The band is spelled out here
    // because cdIsCompound lives in CoinDEMNarrowphase.h, included further down.
    if (cdIsHull(c) || cdIsEgg(c) || (c.shapeExtents.w > 5.5 && c.shapeExtents.w < 6.5)) return invMass * c.hullRef.yzw;
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

// ── Swept-sphere shapes (capsule + egg unified) ──────────────────────────────
// Both are "a segment swept by a sphere"; the capsule's radius is constant, the
// egg's varies linearly fat→tip. Narrowphase code that probes a capsule's
// segment generalizes verbatim by asking the radius AT the probe.
static bool cdIsSwept(CoinBody c) { return cdIsCapsule(c) || cdIsEgg(c); }
static void cdSweptSegment(CoinBody c, thread float3& a, thread float3& b,
                           thread float& ra, thread float& rb) {
    if (cdIsEgg(c)) {
        cdEggSegment(c, a, b);
        ra = cdEggRFat(c); rb = cdEggRTip(c);
    } else {
        cdCapsuleSegment(c, a, b);
        ra = cdCapsuleR(c); rb = ra;
    }
}
// Swept radius at the segment parameter nearest world point p.
static float cdSweptRadiusNear(float3 a, float3 b, float ra, float rb, float3 p) {
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return mix(ra, rb, t);
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

// ── Cell-offset prefix sum — HIERARCHICAL, not serial ─────────────────────────
// The original coinCellOffsetsScan was one GPU thread walking EVERY cell. At bin
// scale (thousands of cells) that was invisible; at HOUSE scale (Daydream's egg
// rain: a lot-sized volume ⇒ millions of cells) the single latency-bound lane
// took whole seconds per substep and tripped the GPU watchdog
// (kIOGPUCommandBufferCallbackErrorTimeout — measured, 2026-08-27). Three-phase
// scan instead: parallel per-block sums, ONE short serial pass over the block
// sums (cells/1024 iterations — thousands, not millions), parallel per-block
// offset apply. Same output contract as before: cellOffsets is the exclusive
// prefix sum with the grand total in cellOffsets[totalCells], and cellCounts is
// zeroed for reuse as coinScatter's write cursor.
constant uint CD_SCAN_BLOCK = 1024u;

kernel void coinCellBlockSums(
    device const uint*     cellCounts [[ buffer(0) ]],
    device uint*           blockSums  [[ buffer(1) ]],
    constant CoinUniforms& u          [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint total = u.gridResX * u.gridResY * u.gridResZ;
    uint start = gid * CD_SCAN_BLOCK;
    if (start >= total) return;
    uint end = min(start + CD_SCAN_BLOCK, total);
    uint s = 0;
    for (uint i = start; i < end; ++i) s += cellCounts[i];
    blockSums[gid] = s;
}

kernel void coinCellBlockScan(
    device uint*           blockSums [[ buffer(0) ]],
    constant CoinUniforms& u         [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid != 0) return;
    uint total = u.gridResX * u.gridResY * u.gridResZ;
    uint blocks = (total + CD_SCAN_BLOCK - 1u) / CD_SCAN_BLOCK;
    uint sum = 0;
    for (uint b = 0; b < blocks; ++b) {
        uint t = blockSums[b];
        blockSums[b] = sum;
        sum += t;
    }
}

kernel void coinCellOffsetsApply(
    device uint*           cellCounts  [[ buffer(0) ]],
    device uint*           cellOffsets [[ buffer(1) ]],
    device const uint*     blockSums   [[ buffer(2) ]],
    constant CoinUniforms& u           [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint total = u.gridResX * u.gridResY * u.gridResZ;
    uint start = gid * CD_SCAN_BLOCK;
    if (start >= total) return;
    uint end = min(start + CD_SCAN_BLOCK, total);
    uint run = blockSums[gid];
    for (uint i = start; i < end; ++i) {
        cellOffsets[i] = run;
        run += cellCounts[i];
        cellCounts[i] = 0;   // reused as write cursor by coinScatter
    }
    if (end == total) cellOffsets[total] = run;   // the tail block owns the grand total
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
    float3 cl = clamp(lp, -heJ, heJ);
    float3 dl3 = lp - cl;
    float dist = length(dl3);
    // Inclusive at both ends, with a rounding allowance (VZ-0163): a point exactly ON the
    // surface (dist 0 — a box corner resting exactly on a face or edge) and a point
    // exactly AT the margin are contacts. The old `dist >= spec || dist < 1e-8` dropped
    // both, so two boxes stacked square (every corner exactly on the other's edge) had no
    // corner contact at all.
    float tolB = 1e-6 * (length(cj) + length(heJ)) + 1e-9;
    if (dist > max(spec, 0.0) + tolB) return false;
    if (dist < 1e-8) {                             // on the surface: the face it lies on
        uint k = (d.x <= d.y && d.x <= d.z) ? 0u : ((d.y <= d.z) ? 1u : 2u);
        float sk = lp[k] >= 0.0 ? 1.0 : -1.0;
        outN = cdQuatRotate(qj, float3(k == 0u ? sk : 0.0, k == 1u ? sk : 0.0, k == 2u ? sk : 0.0));
        outDepth = 0.0;
        return true;
    }
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
    float restY = cdIsSphere(c) ? cdRadiusOf(c)
                : cdIsEgg(c)    ? min(cdEggRFat(c), cdEggRTip(c))   // side-lying egg rests ≥ its thinner end
                :                 floorH * 0.5;
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

static inline void cdDeriveTransformBody(uint id, device const CoinBody* coins, device CoinTransform* transforms,
                                         constant CoinUniforms& u)
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

kernel void coinDeriveTransforms(
    device const CoinBody* coins      [[ buffer(0) ]],
    device CoinTransform*  transforms [[ buffer(1) ]],
    constant CoinUniforms& u          [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdDeriveTransformBody(id, coins, transforms, u);
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
    if (cdIsEgg(c)) {
        // EXACT support of the sphere-swept cone (the hull of its two end
        // spheres): whichever end sphere reaches farther along d.
        float3 nd = normalize(d);
        float3 a, b; cdEggSegment(c, a, b);
        float pa = dot(a, nd) + cdEggRFat(c);
        float pb = dot(b, nd) + cdEggRTip(c);
        return pa >= pb ? (a + cdEggRFat(c) * nd) : (b + cdEggRTip(c) * nd);
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
    //
    // An axis probe that finds a SEPARATING axis (best ≤ 0) PROVES the pair apart — GJK's
    // tetrahedron test misclassifies a flat face–face Minkowski difference by rounding,
    // and the EPA it seeds then invents a depth. The old `probe && pdepth < 0.9·bestD`
    // silently kept that depth whenever the probe said "separated": two Digital Clock bars
    // stacked 0.01 mm apart came back 0.2587 mm deep, were shoved apart, and toppled.
    float3 pn; float pdepth;
    if (!cdAxisProbe(ci, cj, hullVerts, hullRanges, pn, pdepth)) return false;
    if (pdepth < bestD * 0.9) {
        outN = pn;
        outDepth = pdepth;
    }
    return true;
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
// For a STATIC contact the "hi body" field is free, so it carries the COLLIDER index
// instead: without it, a body resting in a bin corner produced the SAME key for its
// floor contact and its wall contact (both feature 0), so `coinWarmStartMatch` seeded
// one with the other's converged impulse — and which one won the hash slot depended on
// the racing insert order. Collider 0 keeps the historical key.
// The pairKey feature byte reserved for polytope-narrowphase contacts (CoinDEMNarrowphase.h):
// their key is the pair alone, and the warm start matches them by identity, else by position.
constant uint CD_KEY_POSMATCH = 0xFFu << 24;
static uint cdPairKey(uint a, uint b, uint feature, uint colliderIdx) {
    uint lo = min(a, b);
    uint hi = (b == CD_STATIC) ? (0xFFFu - min(colliderIdx, 0xFFFu)) : max(a, b);
    return (lo & 0xFFFu) | ((hi & 0xFFFu) << 12) | ((feature & 0xFFu) << 24);
}


// ── Manifold reduction ────────────────────────────────────────────────────────
//
// A rigid body resting against one surface is fully constrained by ~4 well-spread
// contact points; every extra sample is a redundant constraint that costs real solver
// throughput. Not because of the arithmetic — because the graph colouring gives every
// contact sharing a body its OWN colour, so the per-body contact DEGREE sets both the
// number of colours (each an extra serialized solve pass) and the number of colouring
// rounds. Measured on the shipped 176-body mixed pile: a coin lying flat emitted up to
// 20 points against the bin, max degree hit 48, and the pile needed 48 colours.
//
// So each manifold is reduced to at most CD_MANIFOLD_MAX points: the DEEPEST (the
// constraint that matters most), then greedily the point farthest from those already
// kept — the standard deepest+spread reduction, which preserves the support polygon that
// keeps a body level. Ties resolve to the lowest index, so the choice is deterministic.
constant int CD_MANIFOLD_MAX = 4;

static int cdReduceManifold(thread const float3* pos, thread const float* depth, int n,
                            thread int* keep) {
    if (n <= CD_MANIFOLD_MAX) { for (int i = 0; i < n; ++i) keep[i] = i; return max(n, 0); }
    int best = 0;
    for (int i = 1; i < n; ++i) if (depth[i] > depth[best]) best = i;
    int m = 0;
    keep[m++] = best;
    while (m < CD_MANIFOLD_MAX) {
        int pick = -1; float bestD = -1.0;
        for (int i = 0; i < n; ++i) {
            bool taken = false;
            for (int j = 0; j < m; ++j) if (keep[j] == i) { taken = true; break; }
            if (taken) continue;
            float dmin = 1e30;
            for (int j = 0; j < m; ++j) dmin = min(dmin, distance_squared(pos[i], pos[keep[j]]));
            if (dmin > bestD) { bestD = dmin; pick = i; }
        }
        if (pick < 0) break;
        keep[m++] = pick;
    }
    return m;
}

// Write one contact record into `slot`. `auxW` is the manifold marker (0 ungrouped).
static void cdWriteContact(device CoinContact* contacts, uint slot,
                           uint a, uint b, uint feature, uint colliderIdx,
                           float3 nBtoA, float3 cp, float3 xa, float3 xb, float depth, float auxW) {
    float3 n = normalize(nBtoA);
    float3 t1, t2; cdContactTangents(n, t1, t2);
    CoinContact c;
    c.meta = uint4(a, b, (b == CD_STATIC) ? colliderIdx : feature, cdPairKey(a, b, feature, colliderIdx));
    c.nrm  = float4(n, depth);
    c.rA   = float4(cp - xa, 0.0);
    c.rB   = float4((b == CD_STATIC) ? float3(0.0) : (cp - xb), 0.0);
    c.tan1 = float4(t1, 0.0);
    c.tan2 = float4(t2, -1.0);                       // colour = −1 (uncoloured)
    c.aux  = float4(0.0, 0.0, 0.0, auxW);            // vn₀ captured by the first solve pass
    c.ext  = float4(0.0);                            // no torsional impulse yet
    contacts[slot] = c;
}

// The order a GROUPED manifold's points are written in is the order the manifold solve
// visits them (its Gauss–Seidel sweep), so it must be canonical — ascending feature id —
// not the order the narrowphase happened to find them in. cdReduceManifold keeps the
// deepest point first; on a flat face-to-face rest the four corner depths tie to float
// noise, so which corner came first (and with it the sweep order, and the split of the
// redundant 4-point system's impulse that the warm start then carries on) flipped with the
// bodies' absolute position: a warm-started, manifold-solved 5-cube tower rocked by up to
// 3.2° while awake at 2 of 24 spawn positions and 0.025° at the median one. BOTH emitters
// use it — cdEmitGroup below and cdEmitGroupF (CoinDEMNarrowphase.h, every polytope
// manifold); sorting only one of them measured worse than sorting neither (a three-bar
// stack slid 10.8 mm at 1 of 12 positions). Both sorted, over 40 spawn positions each (sleep
// off, 10 s): towers ≤ 0.019°, three-bar stacks ≤ 0.0003 mm / 0.00007°, and the fence's
// 16 stocked bars ≤ 0.052° cold / 0.015° warm. `ord` gets the permutation (n ≤ CD_MANIFOLD_MAX).
static void cdCanonicalManifoldOrder(thread const uint* feature, int n, thread int* ord) {
    for (int i = 0; i < n; ++i) ord[i] = i;
    for (int i = 1; i < n; ++i)
        for (int j = i; j > 0 && feature[ord[j - 1]] > feature[ord[j]]; --j) { int t = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = t; }
}

// Append a MANIFOLD of n ≤ 4 points of one body pair. With CD_FLAG_MANIFOLD_SOLVE and
// n > 1 it is allocated contiguously, in canonical order (cdCanonicalManifoldOrder), and
// marked (head −n, points −0.25) so the colouring and the solve treat it as one unit;
// otherwise each point is an independent contact, exactly as cdEmitContact appends it.
static void cdEmitGroup(device CoinContact* contacts, device atomic_uint* contactCount, uint maxContacts,
                        constant CoinUniforms& u, uint a, uint b, uint colliderIdx, int n,
                        thread const uint* feature, thread const float3* nBtoA, thread const float3* cp,
                        float3 xa, float3 xb, thread const float* depth) {
    bool grouped = (u.solverFlags & CD_FLAG_MANIFOLD_SOLVE) != 0u && n > 1 && n <= CD_MANIFOLD_MAX;
    if (grouped) {
        for (int i = 0; i < n; ++i) if (isnan(depth[i])) return;
        int ord[CD_MANIFOLD_MAX];
        cdCanonicalManifoldOrder(feature, n, ord);
        uint slot = atomic_fetch_add_explicit(contactCount, uint(n), memory_order_relaxed);
        // Straddling the end of the buffer: the points that fit go in as ungrouped contacts
        // (a slot inside the buffer is never left holding a stale contact).
        bool fits = slot + uint(n) <= maxContacts;
        for (int i = 0; i < n && slot + uint(i) < maxContacts; ++i) {
            int k = ord[i];
            cdWriteContact(contacts, slot + uint(i), a, b, feature[k], colliderIdx, nBtoA[k], cp[k], xa, xb, depth[k],
                           !fits ? 0.0 : (i == 0 ? -float(n) : -0.25));
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (isnan(depth[i])) continue;
        uint slot = atomic_fetch_add_explicit(contactCount, 1u, memory_order_relaxed);
        if (slot >= maxContacts) return;
        cdWriteContact(contacts, slot, a, b, feature[i], colliderIdx, nBtoA[i], cp[i], xa, xb, depth[i], 0.0);
    }
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
    c.meta = uint4(a, b, (b == CD_STATIC) ? colliderIdx : feature, cdPairKey(a, b, feature, colliderIdx));
    c.nrm  = float4(n, depth);
    c.rA   = float4(cp - xa, 0.0);
    c.rB   = float4((b == CD_STATIC) ? float3(0.0) : (cp - xb), 0.0);
    c.tan1 = float4(t1, 0.0);
    c.tan2 = float4(t2, -1.0);                       // colour = −1 (uncoloured)
    c.aux  = float4(0.0);                            // vn₀ captured by the first solve pass
    c.ext  = float4(0.0);                            // no torsional impulse yet
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

// Per body: 1 iff it is one end of an ENABLED, non-collideConnected joint between two
// dynamic bodies — the only kind that suppresses a contact pair in coinGenerateContacts.
// Run once per frame (joints change only between frames, host-side), so the generate
// kernel's per-pair joint scan runs only when BOTH bodies carry the bit (VZ-0153).
// The joint-list scans below read the list EIGHT slots at a time (the index clamped to the last
// slot, so a short tail re-reads it): the eight (list → joint) dependent device loads are then in
// flight together instead of one pair after another — a scan used to cost ≈ 0.6 µs per listed
// joint, and the Digital Clock lists 30 (9 crane joints + 21 latch welds), so a crane body pair
// with no joint between them (the jib against the block, rail, puck, held bar) walked ≈ 18 µs of
// loads in contact generation every substep (stage B3). The answer is the same boolean: any
// listed joint that matches.
static inline bool cdJointMarksBody(uint4 m, uint id) {
    return (m.w & 1u) != 0u && (m.w & 2u) == 0u && m.z != CD_STATIC && (m.y == id || m.z == id);
}
static inline bool cdJointSuppressesPair(uint4 m, uint a, uint b) {
    return (m.w & 1u) != 0u && (m.w & 2u) == 0u && ((m.y == a && m.z == b) || (m.y == b && m.z == a));
}

static inline void cdMarkJointedBody(uint id, device const CoinJoint* joints, uint jointCount,
                                     device uint* jointedBody, constant CoinUniforms& u,
                                     device const uint* jointList)
{
    if (id >= u.coinCount) return;
    bool flag = false;
    for (uint k = 0u; k < jointCount && !flag; k += 8u) {
        uint last = jointCount - 1u;
        uint sl[8];
        #pragma clang loop unroll(full)
        for (uint t = 0u; t < 8u; ++t) sl[t] = jointList[min(k + t, last)];
        #pragma clang loop unroll(full)
        for (uint t = 0u; t < 8u; ++t) flag = flag | cdJointMarksBody(joints[sl[t]].meta, id);
    }
    jointedBody[id] = flag ? 1u : 0u;
}

// Whether an enabled, non-collideConnected listed joint joins bodies a and b (see above).
static inline bool cdJointScanPair(device const CoinJoint* joints, device const uint* jointList, uint jointCount,
                                   uint a, uint b)
{
    bool hit = false;
    for (uint k = 0u; k < jointCount && !hit; k += 8u) {
        uint last = jointCount - 1u;
        uint sl[8];
        #pragma clang loop unroll(full)
        for (uint t = 0u; t < 8u; ++t) sl[t] = jointList[min(k + t, last)];
        #pragma clang loop unroll(full)
        for (uint t = 0u; t < 8u; ++t) hit = hit | cdJointSuppressesPair(joints[sl[t]].meta, a, b);
    }
    return hit;
}

kernel void coinMarkJointedBodies(
    device const CoinJoint* joints      [[ buffer(0) ]],
    constant uint&          jointCount  [[ buffer(1) ]],   // ACTIVE joints (the list length)
    device uint*            jointedBody [[ buffer(2) ]],
    constant CoinUniforms&  u           [[ buffer(3) ]],
    device const uint*      jointList   [[ buffer(4) ]],   // enabled slots (coinJointListUpload)
    uint id [[ thread_position_in_grid ]])
{
    cdMarkJointedBody(id, joints, jointCount, jointedBody, u, jointList);
}

// The exact polytope narrowphase (boxes, topology hulls, compounds; swept spheres vs
// polytopes) and its kernels coinWritePolyArgs / coinPolyNarrow — stage B1 (VZ-0163, T5,
// T8, 5c).
#include "CoinDEMNarrowphase.h"

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

    int K = cdScanCells(ci, u);                  // the generate kernels' neighbourhood (VZ-0161)
    int3 base = int3(floor((xi - float3(u.gridMinX, u.gridMinY, u.gridMinZ)) * u.invCell));
    int3 res  = int3(int(u.gridResX), int(u.gridResY), int(u.gridResZ));
    for (int dz = -K; dz <= K; ++dz)
    for (int dy = -K; dy <= K; ++dy)
    for (int dx = -K; dx <= K; ++dx) {
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
            if (cdPairIsPoly(ci, cj, hullVerts, hullRanges)) {
                // A pair the polytope narrowphase owns: the SAME exact geometry it resolves
                // with (SAT for polytope pieces, the swept-sphere distance otherwise).
                minDepth = cdPolyPairDepth(ci, cj, u, hullVerts, hullRanges);
                if (minDepth <= 0.0) separated = true;
            } else if (cdIsHull(ci) || cdIsHull(cj)) {
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
                    } else if (cdIsEgg(O)) {
                        // Sphere ↔ egg: the same swept-radius segment probe the
                        // narrowphase (coinGenerateContacts) de-penetrates with.
                        float yA = O.hullRef.x, yB = O.shapeExtents.z;
                        float t = clamp((lp.y - yA) / max(yB - yA, 1e-6), 0.0, 1.0);
                        closestLocal = float3(0.0, mix(yA, yB, t), 0.0);
                        extraR = cdEggRadiusAt(O, t);
                    } else {
                        closestLocal = cdClosestInShapeLocal(O, lp);
                    }
                    minDepth = rs + extraR - length(lp - closestLocal);
                }
                if (minDepth <= 0.0) separated = true;
            } else if (cdIsSwept(ci) || cdIsSwept(cj)) {
                // Swept-involved pair (capsule or egg) — the same probe math the
                // constraint narrowphase de-penetrates with (bounding-cylinder SAT
                // here would over-report a capsule/egg resting beside anything: a
                // false positive).
                if (cdIsSwept(ci) && cdIsSwept(cj)) {
                    float3 a0, a1, b0, b1; float rI0, rI1, rJ0, rJ1;
                    cdSweptSegment(ci, a0, a1, rI0, rI1);
                    cdSweptSegment(cj, b0, b1, rJ0, rJ1);
                    float3 c1, c2; cdClosestSegSeg(a0, a1, b0, b1, c1, c2);
                    minDepth = (cdSweptRadiusNear(a0, a1, rI0, rI1, c1)
                              + cdSweptRadiusNear(b0, b1, rJ0, rJ1, c2)) - length(c1 - c2);
                } else {
                    bool iCap = cdIsSwept(ci);
                    CoinBody C = iCap ? ci : cj;
                    CoinBody O = iCap ? cj : ci;
                    float3 s0, s1; float r0, r1;
                    cdSweptSegment(C, s0, s1, r0, r1);
                    float3 co2 = O.posInvMass.xyz; float4 qo2 = O.orient;
                    minDepth = -1e30;
                    for (int p = 0; p < 3; ++p) {
                        float tS = float(p) * 0.5;
                        float3 s = mix(s0, s1, tS);
                        float rc = mix(r0, r1, tS);
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


// One body's contact generation (thread `id`): its dynamic pairs (each pair once, from the
// lower index) and its static colliders. The body of coinGenerateContacts; the small-world
// frame kernel calls it too, over a one-cell grid (every body in cell 0 — all pairs).
//
// `part` of `nParts`: the body's work split across threads — its dynamic partners j with
// j % nParts == part, its colliders k ≡ part (mod nParts). Each pair / collider is still handled
// entirely by one thread with the same code, and only the ORDER of the atomic appends changes,
// which nothing downstream reads (colour priorities hash identities, manifolds are written in
// canonical order, the uncoloured bucket is sorted) — so any split steps bit-identically. The
// multi-dispatch kernel runs part 0 of 1; the small-world kernel spreads a body over the
// threads it would otherwise leave idle (one thread per body left ~200 of 256 idle while the
// busiest body walked every collider).
static inline void cdGenerateContactsBody(
    uint                             id,
    uint                             part,
    uint                             nParts,
    device const CoinBody*           coins,
    device const uint*               sortedIndices,
    device const uint*               cellOffsets,
    device const CoinStaticCollider* colliders,
    constant CoinUniforms&           u,
    device const int2*               links,
    device CoinContact*              contacts,
    device atomic_uint*              contactCount,
    constant uint&                   maxContacts,
    device const float4*             hullVerts,
    device const uint2*              hullRanges,
    device const CoinJoint*          joints,
    constant uint&                   jointCount,    // ACTIVE joints (the list length)
    device const uint*               asleep,
    device const uint*               jointedBody,   // coinMarkJointedBodies
    device const uint*               jointList,     // enabled slots (coinJointListUpload)
    device uint4*                    polyPairs,     // the polytope narrowphase's pair list (coinPolyNarrow)
    device atomic_uint*              polyPairCount,
    constant uint&                   maxPolyPairs)  // 0 while no list is bound (no box / hull / compound)
{
    if (id >= u.coinCount) return;
    CoinBody ci = coins[id];
    if (ci.posInvMass.w == 0.0) return;
    int jointPartner = links[id].x;
    float spec = u.speculativeMargin;   // >0 ⇒ also emit near-contacts (anti-tunneling)
    // An asleep body is inert (invMass 0 in the solve, no integration), so a contact
    // between two asleep bodies — or between an asleep body and a static — is skipped
    // by coinSolveVelocityColor anyway. Generating it only cost narrowphase time
    // (asleep–asleep hull GJK/EPA, the whole static loop) and colouring degree (VZ-0152).
    // Island connectivity among asleep bodies is kept by the sleep-group edges instead
    // (coinIslandSleepHub / coinIslandUnion), so a woken pile still wakes as one island.
    bool iAsleep = asleep[id] != 0u;
    // Only a body that is one end of an ENABLED, non-collideConnected two-body joint can
    // have a pair suppressed by the joint scan below (VZ-0153).
    bool iJointed = (jointCount > 0u) && (jointedBody[id] != 0u);

    float3 xi = ci.posInvMass.xyz;
    float4 qi = ci.orient;
    float  Ri = cdRadiusOf(ci);
    float  hi = cdHalfThickOf(ci);
    bool   iIsBox = cdIsBox(ci);

    float3 fp[CD_NPTS];
    for (int p = 0; p < CD_NPTS; ++p) fp[p] = xi + cdQuatRotate(qi, cdBodyFeaturePoint(p, ci));
    float3 ni = normalize(cdQuatRotate(qi, float3(0,1,0)));

    // ── Dynamic–dynamic (each pair once: only j with index > id) ──────────────
    // Neighbourhood: ±1 cell (the classic 3×3×3) whenever the cell covers the bodies;
    // wider only for a body that outgrew it (cdScanCells, VZ-0161).
    int K = cdScanCells(ci, u);
    int3 base = int3(floor((xi - float3(u.gridMinX, u.gridMinY, u.gridMinZ)) * u.invCell));
    int3 res  = int3(int(u.gridResX), int(u.gridResY), int(u.gridResZ));
    for (int dz = -K; dz <= K; ++dz)
    for (int dy = -K; dy <= K; ++dy)
    for (int dx = -K; dx <= K; ++dx) {
        int3 c = base + int3(dx, dy, dz);
        if (any(c < int3(0)) || any(c >= res)) continue;
        uint cell = uint(c.x) + u.gridResX * (uint(c.y) + u.gridResY * uint(c.z));
        uint start = cellOffsets[cell], end = cellOffsets[cell + 1u];
        for (uint s = start; s < end; ++s) {
            uint j = sortedIndices[s];
            if (j <= id) continue;                       // pair once, lower index owns it
            if (j % nParts != part) continue;            // another part of this body's work
            if (int(j) == jointPartner) continue;
            if (iAsleep && asleep[j] != 0u) continue;    // both inert — see iAsleep above
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
            // Generic joints: skip contact generation for a pair that's the two
            // bodies of a joint with collideConnected off (meta.w bit 1 unset,
            // the default) — otherwise a hinge whose bodies touch at the anchor
            // fights its own contact. The scan reads the ACTIVE joint list only (a
            // disabled pool slot is never read, VZ-0150 item 4c), and runs only for a
            // pair that passed the bounding reject above AND whose two bodies both
            // belong to such a joint (VZ-0153: it used to run first, for every
            // candidate j in the 3×3×3 cells — #bodies × #slots loads per thread).
            // (Disabled → no skip; collideConnected → keep the contact. cdJointScanPair.)
            if (iJointed && jointedBody[j] != 0u && cdJointScanPair(joints, jointList, jointCount, id, j)) continue;

            // Polytope pairs (box, topology hull, compound; and those against a swept
            // sphere) go to the pair list — coinPolyNarrow generates them, exactly once.
            if (cdPairIsPoly(ci, cj, hullVerts, hullRanges)) {
                cdAppendPolyPairs(id, j, ci, cj, spec, hullVerts, hullRanges, polyPairs, polyPairCount, maxPolyPairs);
                continue;
            }

            // ── Hull-involved pair: LIVE GJK + EPA + clipped manifold ──────
            // (This is the "wire GJK/EPA into the solver" step the constraint-
            // solver doc lists as missing: the EPA contact gets a real support-
            // polygon manifold, so hull stacks rest on a footprint.)
            if (cdIsHull(ci) || cdIsHull(cj)) {
                float3 n; float depth;
                if (!cdGJKEPA(ci, cj, hullVerts, hullRanges, n, depth)) continue;
                if (depth <= 0.0) continue;
                // n points from j(B) toward i(A) — cdEmitContact's convention.
                bool capsuleInvolved = cdIsCapsule(ci) || cdIsCapsule(cj)
                                    || cdIsEgg(ci)     || cdIsEgg(cj);

                // Capsule/egg↔hull: 2-point manifold (both segment stations vs
                // the hull's near-face support point along the EPA normal), so a
                // capsule/egg lying across a hull face/edge rests level instead
                // of teetering on the single-point EPA contact. An egg's two
                // stations carry their own radii (fat end / tip end).
                if (capsuleInvolved) {
                    bool idIsCapsule = cdIsCapsule(ci) || cdIsEgg(ci);
                    CoinBody capBody = idIsCapsule ? ci : cj;
                    float3 nCapWard = idIsCapsule ? n : -n;   // hull → capsule
                    float3 s0, s1; float r0, r1;
                    if (cdIsEgg(capBody)) {
                        cdEggSegment(capBody, s0, s1);
                        r0 = cdEggRFat(capBody); r1 = cdEggRTip(capBody);
                    } else {
                        cdCapsuleSegment(capBody, s0, s1);
                        r0 = cdCapsuleR(capBody); r1 = r0;
                    }
                    CoinBody hullBody = idIsCapsule ? cj : ci;
                    float3 hullSupport = cdSupport(hullBody, nCapWard, hullVerts, hullRanges);
                    float planeOffset = dot(nCapWard, hullSupport);
                    float3 stations[2] = { s0, s1 };
                    float  stationR[2] = { r0, r1 };
                    int emitted = 0;
                    for (int p = 0; p < 2; ++p) {
                        float pen = (planeOffset + stationR[p]) - dot(nCapWard, stations[p]);
                        if (pen > -spec) {
                            float3 cp = stations[p] - stationR[p] * nCapWard;
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
                            uint gF[4]; float3 gN[4], gP[4]; float gD[4];
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
                                    gF[emittedPts] = uint(m); gN[emittedPts] = n; gP[emittedPts] = cp; gD[emittedPts] = pointDepth;
                                    emittedPts++;
                                }
                            }
                            cdEmitGroup(contacts, contactCount, maxContacts, u, id, j, 0u, emittedPts, gF, gN, gP, xi, xj, gD);
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
                float extraR = 0.0;      // the other shape's own surface radius (capsule/egg)
                if (cdIsCapsule(O)) {
                    // Sphere ↔ capsule: closest point on the capsule's SEGMENT, then
                    // the capsule's radius joins the sphere's in the gap test.
                    float hlO = cdCapsuleHL(O);
                    closestLocal = float3(0.0, clamp(lp.y, -hlO, hlO), 0.0);
                    extraR = cdCapsuleR(O);
                } else if (cdIsEgg(O)) {
                    // Sphere ↔ egg: closest point on the egg's centre SEGMENT
                    // (fat-sphere centre → tip-sphere centre, both offset from the
                    // COM), with the swept radius at that parameter. The lerped
                    // radius slightly underestimates the flank by the taper's sag —
                    // millimetres at egg scale, recovered by the solver's slop.
                    float yA = O.hullRef.x, yB = O.shapeExtents.z;
                    float t = clamp((lp.y - yA) / max(yB - yA, 1e-6), 0.0, 1.0);
                    closestLocal = float3(0.0, mix(yA, yB, t), 0.0);
                    extraR = cdEggRadiusAt(O, t);
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

            // ── Swept-involved pair (capsule OR egg — the constraint path gets the
            // EXACT swept shape). A swept body is a segment + [varying] radius, so
            // every contact reduces to sphere probes: the true seg-seg closest pair
            // plus the two END probes for the parallel-rest manifold (swept↔swept),
            // or 3 probe centres along the segment vs the other shape's
            // closest-point map (swept↔box/disc). A capsule is the constant-radius
            // special case — its maths here are bit-identical to the old branch.
            if (cdIsSwept(ci) || cdIsSwept(cj)) {
                if (cdIsSwept(ci) && cdIsSwept(cj)) {
                    float3 a0, a1, b0, b1; float rI0, rI1, rJ0, rJ1;
                    cdSweptSegment(ci, a0, a1, rI0, rI1);
                    cdSweptSegment(cj, b0, b1, rJ0, rJ1);
                    float3 cps[3]; float3 cqs[3];
                    float3 c1, c2; cdClosestSegSeg(a0, a1, b0, b1, c1, c2);
                    cps[0] = c1; cqs[0] = c2;                              // true closest pair
                    cps[1] = a0; cqs[1] = cdClosestOnSegment(b0, b1, a0);  // end probes: the
                    cps[2] = a1; cqs[2] = cdClosestOnSegment(b0, b1, a1);  // parallel-rest manifold
                    for (int k = 0; k < 3; ++k) {
                        if (k > 0 && length(cps[k] - cps[0]) < 0.5 * rI0) continue;   // dedupe vs main
                        float rI = cdSweptRadiusNear(a0, a1, rI0, rI1, cps[k]);
                        float rJ = cdSweptRadiusNear(b0, b1, rJ0, rJ1, cqs[k]);
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
                bool iSw = cdIsSwept(ci);
                CoinBody C = iSw ? ci : cj;      // the swept body
                CoinBody O = iSw ? cj : ci;      // the box / disc
                float3 s0, s1; float r0, r1;
                cdSweptSegment(C, s0, s1, r0, r1);
                float3 co = O.posInvMass.xyz; float4 qo = O.orient;
                for (int k = 0; k < 3; ++k) {
                    float tS = float(k) * 0.5;                       // end, mid, end
                    float3 s = mix(s0, s1, tS);
                    float rc = mix(r0, r1, tS);                      // the probe's own radius
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
                    float3 nBtoA = iSw ? nOut : -nOut;               // toward A (thread's body i)
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
                float3 iPos[8]; float3 iN[8]; float iDep[8]; int iFeat[8];
                int nIn = 0;                     // INSIDE corners, reduced to a manifold below
                bool anyInside = false;
                int nCand = 0;
                float bestDepth = -1e30;
                for (int p = 0; p < 8; ++p) {
                    float3 pw = fp[p]; float3 nB; float depthB;
                    if (!cdOrientedBoxPushSpeculative(pw, xj, qj, heJ, spec, nB, depthB)) continue;
                    if (jIsDisc) { float3 lp = cdQuatRotateInv(qj, pw - xj); if (length(lp.xz) > Rj) continue; }
                    if (depthB >= 0.0) {
                        anyInside = true;
                        iPos[nIn] = pw; iN[nIn] = nB; iDep[nIn] = depthB; iFeat[nIn] = p; nIn++;
                    } else {
                        cPos[nCand] = pw; cN[nCand] = nB; cDepth[nCand] = depthB; cFeat[nCand] = p; nCand++;
                        bestDepth = max(bestDepth, depthB);
                    }
                }
                if (anyInside) {                 // ≤4 spread corners hold a resting box
                    int keepI[CD_MANIFOLD_MAX];
                    int mI = cdReduceManifold(iPos, iDep, nIn, keepI);
                    uint gF[4]; float3 gN[4], gP[4]; float gD[4];
                    for (int i = 0; i < mI; ++i) {
                        int q = keepI[i];
                        gF[i] = uint(iFeat[q]); gN[i] = iN[q]; gP[i] = iPos[q]; gD[i] = iDep[q];
                    }
                    cdEmitGroup(contacts, contactCount, maxContacts, u, id, j, 0u, mI, gF, gN, gP, xi, xj, gD);
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
    // A CAPSULE or EGG resolves against every static as sphere probes at 3
    // segment stations (both ends + mid, each with its own swept radius) — the
    // exact round surface, not the bounding cylinder's hard rim (which is what
    // the generic feature-point path would test). For an egg the end radii
    // differ (fat end vs tip); against a PLANE the two end probes are exact.
    // Probes are precomputed once here.
    if (iAsleep) return;                 // asleep vs static: inert on both sides (VZ-0152)
    // A polytope body's static BOXES (kinds 1, 3) go to the pair list (SAT + clipping in
    // coinPolyNarrow); a compound's planes / tubes / pusher are its children's corner probes.
    bool iCompound = cdIsCompound(ci);
    bool polyStatic = cdIsPolyBody(ci, hullVerts, hullRanges);
    bool iCapsule = cdIsSwept(ci);
    float3 capP[3];
    float capRs[3] = { 0.0, 0.0, 0.0 };
    if (iCapsule) {
        float3 s0, s1; float r0, r1;
        cdSweptSegment(ci, s0, s1, r0, r1);
        capP[0] = s0;  capP[1] = 0.5 * (s0 + s1); capP[2] = s1;
        capRs[0] = r0; capRs[1] = 0.5 * (r0 + r1); capRs[2] = r1;
    }
    // A HULL body probes statics with its true vertices (≤32), radius 0 — the
    // same point logic as the disc/box feature points, but on the real shape.
    bool iHull = cdIsHull(ci);
    uint2 hRange = uint2(0, 0);
    if (iHull) hRange = hullRanges[uint(ci.hullRef.x + 0.5)];
    int nFP = iHull ? int(min(hRange.y, uint(CD_MAX_STATIC_FP))) : CD_NPTS;
#define CD_PROBE(pp) (iHull ? (xi + cdQuatRotate(qi, float3(hullVerts[hRange.x + uint(pp)].xyz))) : fp[pp])

    // ── Per-collider bounding reject (VZ-0151) ─────────────────────────────────
    // Every probe the branches below test — a feature point / hull vertex (radius
    // 0), a swept station with its own radius, or the sphere centre with Ri — lies
    // inside the ball B(xi, rProbe). Each branch emits only when a probe comes within
    // `spec` of the collider (plane: pen > −spec; box/OBB: the Euclidean gap of
    // cdOrientedBoxPushSpeculative < spec, or a sphere-probe gap < r + spec; tube: the
    // probe inside the pipe within r + spec of its wall), and the distance from a
    // probe to a convex collider differs from the distance from xi by at most the
    // probe's offset (1-Lipschitz). So a collider farther than rProbe + spec from xi
    // cannot produce a contact, and is skipped before any per-probe work — a hull used
    // to run all 32 vertices through the box push for every uploaded collider
    // (≈19 µs per collider per pass, measured). rCull pads rProbe for float rounding
    // (1e-4 relative + 1 µm), so the emitted contact set is unchanged.
    float rProbe;
    if (cdIsSphere(ci)) {
        rProbe = Ri;
    } else if (iCapsule) {
        rProbe = 0.0;
        for (int p = 0; p < 3; ++p) rProbe = max(rProbe, length(capP[p] - xi) + capRs[p]);
    } else if (iHull) {
        rProbe = 0.0;
        for (int p = 0; p < nFP; ++p) rProbe = max(rProbe, length(float3(hullVerts[hRange.x + uint(p)].xyz)));
    } else if (iCompound) {
        rProbe = Ri;                                      // prevPos.w: the union's bounding radius
    } else {
        rProbe = 0.0;
        for (int p = 0; p < CD_NPTS; ++p) rProbe = max(rProbe, length(fp[p] - xi));
    }
    float rCull = rProbe * 1.0001 + 1e-6 + max(spec, 0.0);
    bool cull = (u.solverFlags & CD_FLAG_NO_COLLIDER_CULL) == 0u;

    for (uint k = part; k < u.colliderCount; k += nParts) {
        CoinStaticCollider col = colliders[k];
        uint kind = as_type<uint>(col.a.w);
        if (cull) {
            if (kind == 0u) {                            // plane: signed distance of the centre
                if (dot(col.a.xyz, xi) - col.b.w >= rCull * length(col.a.xyz)) continue;   // (|n| = 1 from .plane)
            } else if (kind == 1u) {                     // AABB: Euclidean gap of the centre
                float3 g = max(abs(xi - col.a.xyz) - col.b.xyz, float3(0.0));
                if (dot(g, g) >= rCull * rCull) continue;
            } else if (kind == 3u) {                     // OBB: the same gap in its frame
                float3 g = max(abs(cdQuatRotateInv(col.orient, xi - col.a.xyz)) - col.b.xyz, float3(0.0));
                if (dot(g, g) >= rCull * rCull) continue;
            } else if (kind == 4u) {                     // tube: along its axis, radially
                float3 d = xi - col.a.xyz;
                float along = dot(d, col.b.xyz);
                float dr = length(d - col.b.xyz * along);
                if (abs(along) >= col.vel.w + rCull) continue;          // past either end
                if (dr >= col.b.w + rCull) continue;                     // wholly outside the tube
                if (dr + rCull <= col.b.w) continue;                     // wholly inside, off the wall
            }
            // kind 2 (the pusher plate) inflates disc/box probes by the legacy (Ri, hi)
            // margins, so it is never culled — it is one collider in one scene.
        }
        if (polyStatic && (kind == 1u || kind == 3u)) {  // SAT + clipping, coinPolyNarrow
            cdAppendStaticBoxPairs(id, ci, k, col, spec, hullVerts, hullRanges, polyPairs, polyPairCount, maxPolyPairs);
            continue;
        }
        if (iCompound) {                                 // children's corners vs plane / tube / pusher
            cdCompoundStaticProbes(id, ci, k, col, kind, u, hullVerts, hullRanges, contacts, contactCount, maxContacts);
            continue;
        }
        if (kind == 0u) {                                // plane n·p ≥ d
            float3 n = col.a.xyz; float d = col.b.w;
            if (cdIsSphere(ci)) {
                float pen = (d + Ri) - dot(n, xi);
                if (pen > -spec) cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, 0u, k, n, xi - Ri*n, xi, xi, pen);
            } else if (iCapsule) {
                // Both END probes (skip mid: the swept surface's plane distance is
                // linear along the axis, so the mid station is never the deepest) —
                // the 2-point manifold that holds a side-lying capsule/egg level.
                for (int p = 0; p < 3; p += 2) {
                    float pen = (d + capRs[p]) - dot(n, capP[p]);
                    if (pen > -spec) cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, capP[p] - capRs[p]*n, xi, xi, pen);
                }
            } else {
                // Gather the touching feature points, then keep only a reduced manifold
                // (a flat coin otherwise emits all 14 of its samples against one plane).
                float3 mPos[CD_MAX_STATIC_FP]; float mDep[CD_MAX_STATIC_FP]; int mFeat[CD_MAX_STATIC_FP];
                int nc = 0;
                // Inclusive margin with a rounding allowance (VZ-0163): a corner exactly ON the
                // plane with no margin, or exactly AT the margin, is a contact.
                float lim = spec + 1e-6 * (abs(d) + rProbe) + 1e-9;
                for (int p = 0; p < nFP && nc < CD_MAX_STATIC_FP; ++p) {
                    float3 pw = CD_PROBE(p);
                    float pen = d - dot(n, pw);
                    if (pen >= -lim) { mPos[nc] = pw; mDep[nc] = pen; mFeat[nc] = p; nc++; }
                }
                int keep[CD_MANIFOLD_MAX];
                int m = cdReduceManifold(mPos, mDep, nc, keep);
                uint gF[4]; float3 gN[4], gP[4]; float gD[4];
                for (int i = 0; i < m; ++i) {
                    int q = keep[i];
                    gF[i] = uint(mFeat[q]); gN[i] = n; gP[i] = mPos[q]; gD[i] = mDep[q];
                }
                cdEmitGroup(contacts, contactCount, maxContacts, u, id, CD_STATIC, k, m, gF, gN, gP, xi, xi, gD);
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
                        pen = capRs[p] - dl;
                        if (pen <= -spec) continue;
                        n = dL / dl; cp = ctr + cl;
                    } else {                              // probe centre inside the box
                        float3 dist = he - abs(lp); float push;
                        if (dist.x < dist.y && dist.x < dist.z) { n = float3(sign(lp.x),0,0); push = dist.x; }
                        else if (dist.y < dist.z)               { n = float3(0,sign(lp.y),0); push = dist.y; }
                        else                                    { n = float3(0,0,sign(lp.z)); push = dist.z; }
                        pen = push + capRs[p]; cp = capP[p];
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
                float3 iPos[CD_MAX_STATIC_FP]; float3 iN[CD_MAX_STATIC_FP]; float iDep[CD_MAX_STATIC_FP]; int iFeat[CD_MAX_STATIC_FP];
                int nIn = 0;                     // INSIDE points, reduced to a manifold below
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
                        if (nIn < CD_MAX_STATIC_FP) { iPos[nIn] = pw; iN[nIn] = n; iDep[nIn] = push; iFeat[nIn] = p; nIn++; }
                    } else if (nCand < CD_MAX_STATIC_FP) {
                        cPos[nCand] = pw; cN[nCand] = n; cDepth[nCand] = push; cFeat[nCand] = p; nCand++;
                        bestDepth = max(bestDepth, push);
                    }
                }
                if (anyInside) {                 // keep ≤4 spread points of the real manifold
                    int keepI[CD_MANIFOLD_MAX];
                    int mI = cdReduceManifold(iPos, iDep, nIn, keepI);
                    uint gF[4]; float3 gN[4], gP[4]; float gD[4];
                    for (int i = 0; i < mI; ++i) {
                        int q = keepI[i];
                        gF[i] = uint(iFeat[q]); gN[i] = iN[q]; gP[i] = iPos[q]; gD[i] = iDep[q];
                    }
                    cdEmitGroup(contacts, contactCount, maxContacts, u, id, CD_STATIC, k, mI, gF, gN, gP, xi, xi, gD);
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
                // Capsule/egg vs OBB: the 3 segment probes in the box's local frame.
                for (int p = 0; p < 3; ++p) {
                    float3 lp = cdQuatRotateInv(q, capP[p] - ctr);
                    float3 cl = clamp(lp, -he, he);
                    float3 dL = lp - cl;
                    float dl = length(dL);
                    float3 nL; float pen; float3 cp;
                    if (dl > 1e-6) {
                        pen = capRs[p] - dl;
                        if (pen <= -spec) continue;
                        nL = dL / dl; cp = ctr + cdQuatRotate(q, cl);
                    } else {
                        float3 dd = he - abs(lp); float push;
                        if (dd.x < dd.y && dd.x < dd.z) { nL = float3(sign(lp.x),0,0); push = dd.x; }
                        else if (dd.y < dd.z)           { nL = float3(0,sign(lp.y),0); push = dd.y; }
                        else                            { nL = float3(0,0,sign(lp.z)); push = dd.z; }
                        pen = push + capRs[p]; cp = capP[p];
                    }
                    float3 n = cdQuatRotate(q, nL);
                    cdEmitContact(contacts, contactCount, maxContacts, id, CD_STATIC, uint(p), k, n, cp, xi, xi, pen);
                }
            } else {
                // Two-phase, shared-depth near-face manifold — see the kind==1u
                // generic loop's comment.
                float3 cPos[CD_MAX_STATIC_FP]; float3 cN[CD_MAX_STATIC_FP]; float cDepth[CD_MAX_STATIC_FP]; int cFeat[CD_MAX_STATIC_FP];
                float3 iPos[CD_MAX_STATIC_FP]; float3 iN[CD_MAX_STATIC_FP]; float iDep[CD_MAX_STATIC_FP]; int iFeat[CD_MAX_STATIC_FP];
                int nIn = 0;                     // INSIDE points, reduced to a manifold below
                bool anyInside = false;
                int nCand = 0;
                float bestDepth = -1e30;
                for (int p = 0; p < nFP; ++p) {
                    float3 pw = CD_PROBE(p);
                    float3 n; float push;
                    if (!cdOrientedBoxPushSpeculative(pw, ctr, q, he, spec, n, push)) continue;
                    if (push >= 0.0) {
                        anyInside = true;
                        if (nIn < CD_MAX_STATIC_FP) { iPos[nIn] = pw; iN[nIn] = n; iDep[nIn] = push; iFeat[nIn] = p; nIn++; }
                    } else if (nCand < CD_MAX_STATIC_FP) {
                        cPos[nCand] = pw; cN[nCand] = n; cDepth[nCand] = push; cFeat[nCand] = p; nCand++;
                        bestDepth = max(bestDepth, push);
                    }
                }
                if (anyInside) {                 // keep ≤4 spread points of the real manifold
                    int keepI[CD_MANIFOLD_MAX];
                    int mI = cdReduceManifold(iPos, iDep, nIn, keepI);
                    uint gF[4]; float3 gN[4], gP[4]; float gD[4];
                    for (int i = 0; i < mI; ++i) {
                        int q = keepI[i];
                        gF[i] = uint(iFeat[q]); gN[i] = iN[q]; gP[i] = iPos[q]; gD[i] = iDep[q];
                    }
                    cdEmitGroup(contacts, contactCount, maxContacts, u, id, CD_STATIC, k, mI, gF, gN, gP, xi, xi, gD);
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
                float3 s  = sphereLike ? (iCapsule ? capP[p]  : xi) : CD_PROBE(p);
                float  rs = sphereLike ? (iCapsule ? capRs[p] : Ri) : 0.0;
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
                float3 s  = sphereLike ? (iCapsule ? capP[p]  : xi) : CD_PROBE(p);
                float  rs = sphereLike ? (iCapsule ? capRs[p] : Ri) : (iHull ? 0.0 : Ri);
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
    constant uint&                   jointCount    [[ buffer(12) ]],   // ACTIVE joints (the list length)
    device const uint*               asleep        [[ buffer(13) ]],
    device const uint*               jointedBody   [[ buffer(14) ]],   // coinMarkJointedBodies
    device const uint*               jointList     [[ buffer(15) ]],   // enabled slots (coinJointListUpload)
    // The polytope narrowphase's pair list (coinPolyNarrow), and whether it is bound this
    // substep (the host binds a real list only while boxes / hulls / compounds exist).
    device uint4*                    polyPairs     [[ buffer(16) ]],
    device atomic_uint*              polyPairCount [[ buffer(17) ]],
    constant uint&                   maxPolyPairs  [[ buffer(18) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdGenerateContactsBody(id, 0u, 1u, coins, sortedIndices, cellOffsets, colliders, u, links, contacts, contactCount,
                           maxContacts, hullVerts, hullRanges, joints, jointCount, asleep, jointedBody, jointList,
                           polyPairs, polyPairCount, maxPolyPairs);
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

// Per-body contact-list capacity. It must exceed the true per-body contact DEGREE: a
// truncated list hides neighbours from the colouring, which then gives two contacts of
// the same body the same colour, and those two threads read-modify-write that body's
// velocity in parallel. Measured on the shipped 176-body mixed pile the degree peaks at
// 13, so 64 is ~5× headroom; overflow is counted into `colorStats.listOverflow` (gated
// by the tests) rather than silently truncating. The colouring loop is bounded by each
// body's ACTUAL degree, so the cap costs memory (maxCoins × cap × 4 B), not time.
constant uint CD_MAX_BODY_CONTACTS = 64u;
// Colour cap. A body of degree d forces d distinct colours among its contacts, and a
// contact conflicts with both its bodies' lists, so the count needed is O(degree). The
// measured pile uses 13 colours against a peak degree of 13; 64 (the width of the
// `ulong` usedMask below) is the headroom. Anything that doesn't fit stays uncoloured —
// counted into `colorStats.uncolored` and solved by the tail pass after the colours, up to
// CD_UNCOLORED_SUB_MAX per substep (it used to go unsolved; see CD_UNCOLORED_BUCKET).
// The host does NOT dispatch all 64: greedy colouring always takes the lowest free
// colour, so the live colours are 0…maxUsed and the solve sweeps only that many.
constant uint CD_MAX_COLORS        = 64u;
constant uint CD_COLOR_NONE        = 0xFFFFFFFFu;   // "not yet coloured"
constant uint CD_COLOR_SKIP        = 0xFFFFFFFEu;   // a grouped manifold's non-head point: solved with its head

static uint cdHashU(uint x) {               // priority hash (deterministic, no RNG state)
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}

// A contact's colouring PRIORITY, hashed from its stable IDENTITY — bodyA, bodyB,
// collider|feature, pairKey — and NOT from its buffer slot `cid`. Slots are handed out
// by an atomic bump allocator in `cdEmitContact`, so the same physical contact lands in
// a different slot every run; keying the priority off the slot made the colour
// assignment (and therefore the Gauss-Seidel order, and therefore the settled pile) a
// different lottery each run. Identity is a pure function of the geometry, so the whole
// colouring is now reproducible. `meta` covers both cases: for a dynamic pair meta.z is
// the feature, for a static contact meta.z is the collider and the feature rides in
// meta.w's pairKey.
static uint cdContactPriority(uint4 meta) {
    return cdHashU(meta.x ^ cdHashU(meta.y ^ cdHashU(meta.z ^ cdHashU(meta.w))));
}

// Strict lexicographic order on that identity — the tie-break when two priorities hash
// equal. Identities are unique per contact, so this is a total order (and if a duplicate
// ever were emitted, the caller falls back to `cid` so the round can still make progress).
static bool cdIdGreater(uint4 a, uint4 b) {
    if (a.x != b.x) return a.x > b.x;
    if (a.y != b.y) return a.y > b.y;
    if (a.z != b.z) return a.z > b.z;
    return a.w > b.w;
}

// Append `cid` to one body's contact list (atomic, bounded). An overflow is counted so
// the truncation can't come back silently (`CoinDEMSolver.colorStats.listOverflow`).
static void cdAppendBodyContact(device atomic_uint* bodyContactCount,
                                device uint* bodyContacts, uint body, uint cid,
                                device atomic_uint* stats) {
    uint slot = atomic_fetch_add_explicit(&bodyContactCount[body], 1u, memory_order_relaxed);
    if (slot < CD_MAX_BODY_CONTACTS) bodyContacts[body * CD_MAX_BODY_CONTACTS + slot] = cid;
    else atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
}

// Zero the contact append cursor ON THE GPU TIMELINE. This has to be a dispatch, not a
// CPU write to the shared buffer: a frame encodes all `maxSubsteps` substeps into ONE
// command buffer, so any CPU-side reset happens while ENCODING — i.e. before the GPU has
// run a single substep. The counter then never reset between substeps, and each substep's
// narrowphase APPENDED to the previous one's contacts: at 4 substeps the buffer carried
// ~4× the live contacts, three quarters of them stale (generated from positions the pile
// had already left), and the solve applied impulses for all of them. It also multiplied
// the per-body contact degree by 4, and the degree is what sets the colour count — the
// shipped mixed pile needed 48 colours (48 serialized solve passes per iteration) where
// its true contact graph needs ~13.
kernel void coinClearContactCount(
    device atomic_uint* contactCount [[ buffer(0) ]],
    device atomic_uint* pairCount    [[ buffer(1) ]],   // the polytope narrowphase's pair cursor (or buffer 0 again)
    uint id [[ thread_position_in_grid ]])
{
    if (id == 0) {
        atomic_store_explicit(&contactCount[0], 0u, memory_order_relaxed);
        atomic_store_explicit(&pairCount[0], 0u, memory_order_relaxed);
    }
}

kernel void coinClearBodyContacts(
    device atomic_uint* bodyContactCount [[ buffer(0) ]],
    uint id [[ thread_position_in_grid ]])
{
    atomic_store_explicit(&bodyContactCount[id], 0u, memory_order_relaxed);
}

static inline void cdBuildBodyContactsBody(uint cid, uint n, device const CoinContact* contacts,
                                           device uint* bodyContacts, device atomic_uint* bodyContactCount,
                                           device atomic_uint* stats)
{
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    if (cdIsManifoldMember(c)) return;            // a grouped manifold is one node: its head
    if (cdIsDormant(c)) return;                   // carried through a sleep: not coloured (never solved)
    cdAppendBodyContact(bodyContactCount, bodyContacts, c.meta.x, cid, stats);
    if (c.meta.y != CD_STATIC) cdAppendBodyContact(bodyContactCount, bodyContacts, c.meta.y, cid, stats);
}

kernel void coinBuildBodyContacts(
    device const CoinContact* contacts        [[ buffer(0) ]],
    device const atomic_uint& contactCount    [[ buffer(1) ]],
    device uint*              bodyContacts     [[ buffer(2) ]],
    device atomic_uint*       bodyContactCount [[ buffer(3) ]],
    device atomic_uint*       stats            [[ buffer(4) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdBuildBodyContactsBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts,
                            bodyContacts, bodyContactCount, stats);
}

// Seed a colouring pass: every live contact starts uncoloured, with its priority
// precomputed from its identity (so a round reads one uint per neighbour instead of
// re-hashing four).
static inline void cdColorInitBody(uint cid, uint n, device const CoinContact* contacts,
                                   device uint* priority, device uint* color)
{
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    priority[cid] = cdContactPriority(c.meta);
    color[cid]    = (cdIsManifoldMember(c) || cdIsDormant(c)) ? CD_COLOR_SKIP : CD_COLOR_NONE;
}

kernel void coinColorInit(
    device const CoinContact* contacts     [[ buffer(0) ]],
    device const atomic_uint& contactCount [[ buffer(1) ]],
    device uint*              priority     [[ buffer(2) ]],
    device uint*              color        [[ buffer(3) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorInitBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, priority, color);
}

// One colouring round, as a PURE FUNCTION of the previous round: it reads every
// neighbour's colour from `colorIn` and writes this contact's colour to `colorOut`
// (ping-pong). The single-buffered version this replaces read colours that its
// neighbours were writing in the same dispatch — safe (a writer's neighbours never
// write the same round) but timing-dependent: a neighbour that happened to observe
// this contact's fresh colour could colour itself a round earlier and land on a
// different colour. Together with the identity-derived priority, the colouring is now
// a function of the contact GRAPH alone, so two runs of the same scene colour it
// identically.
//
// A contact colours itself iff its priority is the strict max among still-uncoloured
// neighbours, taking the lowest colour no coloured neighbour occupies.
static inline void cdColorRoundBody(uint cid, uint n,
    device const CoinContact* contacts,
    device const uint*        bodyContacts,
    device const uint*        bodyContactCount,
    device const uint*        priority,
    device const uint*        colorIn,
    device uint*              colorOut)
{
    if (cid >= n) return;
    uint mine = colorIn[cid];
    if (mine != CD_COLOR_NONE) { colorOut[cid] = mine; return; }   // already coloured

    CoinContact c = contacts[cid];
    uint4 myId = c.meta;
    uint myPri = priority[cid];

    uint bodies[2]; int nb = 0;
    bodies[nb++] = c.meta.x;
    if (c.meta.y != CD_STATIC) bodies[nb++] = c.meta.y;

    ulong usedMask = 0ul;
    for (int bi = 0; bi < nb; ++bi) {
        uint body = bodies[bi];
        uint cnt = min(bodyContactCount[body], CD_MAX_BODY_CONTACTS);
        for (uint k = 0; k < cnt; ++k) {
            uint ncid = bodyContacts[body * CD_MAX_BODY_CONTACTS + k];
            if (ncid == cid) continue;
            uint ncolor = colorIn[ncid];
            if (ncolor == CD_COLOR_NONE) {       // uncoloured neighbour competes for this round
                uint nPri = priority[ncid];
                if (nPri > myPri) { colorOut[cid] = CD_COLOR_NONE; return; }
                if (nPri == myPri) {             // hash tie → exact identity order
                    uint4 nId = contacts[ncid].meta;
                    bool greater = all(nId == myId) ? (ncid > cid) : cdIdGreater(nId, myId);
                    if (greater) { colorOut[cid] = CD_COLOR_NONE; return; }
                }
            } else if (ncolor < CD_MAX_COLORS) {
                usedMask |= (1ul << ncolor);
            }
        }
    }
    uint color = 0u;
    while (color < CD_MAX_COLORS && (usedMask & (1ul << color))) color++;
    colorOut[cid] = (color < CD_MAX_COLORS) ? color : CD_COLOR_NONE;
}

kernel void coinColorRound(
    device const CoinContact* contacts         [[ buffer(0) ]],
    device const atomic_uint& contactCount     [[ buffer(1) ]],
    device const uint*        bodyContacts     [[ buffer(2) ]],
    device const uint*        bodyContactCount [[ buffer(3) ]],
    device const uint*        priority         [[ buffer(4) ]],
    device const uint*        colorIn          [[ buffer(5) ]],
    device uint*              colorOut         [[ buffer(6) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorRoundBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, bodyContacts,
                     bodyContactCount, priority, colorIn, colorOut);
}

// ── VZ-0160: speculative RANKED colouring (opt-in, CoinDEMSolver.coloringScheme = .speculative) ─
//
// Jones–Plassmann (above) colours a contact only when it outranks every still-uncoloured
// neighbour, so a round colours at most ONE contact per body: a body of degree d needs ≥ d
// rounds, and in practice the rounds follow the longest decreasing-priority chain through the
// contact graph — ≈ 2.4 × the max degree (a graph model of VZ-0160's heaps: a 7 × 7 grid of
// 10 mm cubes, degree 20: 51 rounds; egg heaps of degree 21–30: 48–63 rounds). At the usual
// 16–24 rounds a Daydream-Home-sized heap leaves thousands of contacts to the uncoloured tail,
// and past its 256 per substep they go unsolved: the DH-scale 800-egg rain at 24 rounds left
// 2.15 M contacts unsolved over 1500 frames and ended with 170 pairs > 10 mm deep (44 mm).
//
// Here every round colours MANY contacts of a body at once, in two passes:
//   • coinColorTentative — each uncoloured contact counts how many uncoloured contacts of each
//     of its two bodies outrank it (the same priority / identity order as JP) and bids for the
//     k-th colour that no COLOURED neighbour holds, k = the larger of the two counts: a body's
//     contacts spread over distinct colours instead of all queueing for the lowest free one.
//   • coinColorResolve — a contact keeps its bid unless an uncoloured neighbour bid the same
//     colour and outranks it; losers bid again next round. Two neighbours can never both keep
//     one colour (the higher-ranked one wins), so the colouring stays proper.
// Both passes are pure functions of the previous round's colours (the bids live in their own
// buffer; colours ping-pong as in JP), so the colouring is reproducible run to run. Measured
// on the GPU (CoinDEMColoringTests): 6 rounds (12 dispatches) colour every contact of a
// resting 7 × 7 cube grid (degree 24) and of a 300-egg heap (degree 40), of which JP leaves
// 215 of 473 at 16 rounds and 2909 of 4333 at 24; the price is ≈ 15–35 % more colours than a
// complete JP colouring
// (28 for 24, 47 for 40) — each an extra dispatch per velocity iteration — and a colour set
// that may skip a number (the sweep covers 0…max used, an empty colour is a 0-group dispatch).
// The same 800-egg rain with it: 0 uncoloured, 0 unsolved, 49 colours, 49 pairs > 10 mm
// (27 mm; not the colouring — VZ-0190), frame p50 +1.1 ms over JP at 24 rounds (which skips
// what it cannot colour) and −4.7 ms against JP at 64 rounds (which still left 115 k unsolved).

// ncid outranks cid in the colouring order (JP's rule: priority, then identity, then slot).
static bool cdOutranks(uint ncid, uint cid, uint myPri, uint4 myId,
                       device const uint* priority, device const CoinContact* contacts) {
    uint nPri = priority[ncid];
    if (nPri != myPri) return nPri > myPri;
    uint4 nId = contacts[ncid].meta;
    return all(nId == myId) ? (ncid > cid) : cdIdGreater(nId, myId);
}

static inline void cdColorTentativeBody(uint cid, uint n,
    device const CoinContact* contacts,
    device const uint*        bodyContacts,
    device const uint*        bodyContactCount,
    device const uint*        priority,
    device const uint*        colorIn,
    device uint*              bid)
{
    if (cid >= n) return;
    if (colorIn[cid] != CD_COLOR_NONE) return;          // coloured, or a manifold's non-head point
    uint4 myId = contacts[cid].meta;
    uint myPri = priority[cid];
    uint bodies[2]; int nb = 0;
    bodies[nb++] = myId.x;
    if (myId.y != CD_STATIC) bodies[nb++] = myId.y;
    ulong usedMask = 0ul;
    uint rank = 0u;
    for (int bi = 0; bi < nb; ++bi) {
        uint body = bodies[bi];
        uint cnt = min(bodyContactCount[body], CD_MAX_BODY_CONTACTS);
        uint r = 0u;
        for (uint k = 0; k < cnt; ++k) {
            uint ncid = bodyContacts[body * CD_MAX_BODY_CONTACTS + k];
            if (ncid == cid) continue;
            uint ncolor = colorIn[ncid];
            if (ncolor == CD_COLOR_NONE) {
                if (cdOutranks(ncid, cid, myPri, myId, priority, contacts)) r++;
            } else if (ncolor < CD_MAX_COLORS) {
                usedMask |= (1ul << ncolor);
            }
        }
        rank = max(rank, r);
    }
    uint t = CD_COLOR_NONE;                              // the rank-th colour no coloured neighbour holds
    for (uint col = 0u; col < CD_MAX_COLORS; ++col) {
        if ((usedMask & (1ul << col)) != 0ul) continue;
        if (rank == 0u) { t = col; break; }
        rank--;
    }
    bid[cid] = t;
}

kernel void coinColorTentative(
    device const CoinContact* contacts         [[ buffer(0) ]],
    device const atomic_uint& contactCount     [[ buffer(1) ]],
    device const uint*        bodyContacts     [[ buffer(2) ]],
    device const uint*        bodyContactCount [[ buffer(3) ]],
    device const uint*        priority         [[ buffer(4) ]],
    device const uint*        colorIn          [[ buffer(5) ]],
    device uint*              bid              [[ buffer(6) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorTentativeBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, bodyContacts,
                         bodyContactCount, priority, colorIn, bid);
}

static inline void cdColorResolveBody(uint cid, uint n,
    device const CoinContact* contacts,
    device const uint*        bodyContacts,
    device const uint*        bodyContactCount,
    device const uint*        priority,
    device const uint*        colorIn,
    device const uint*        bid,
    device uint*              colorOut)
{
    if (cid >= n) return;
    uint mine = colorIn[cid];
    if (mine != CD_COLOR_NONE) { colorOut[cid] = mine; return; }
    uint t = bid[cid];
    if (t == CD_COLOR_NONE) { colorOut[cid] = CD_COLOR_NONE; return; }
    uint4 myId = contacts[cid].meta;
    uint myPri = priority[cid];
    uint bodies[2]; int nb = 0;
    bodies[nb++] = myId.x;
    if (myId.y != CD_STATIC) bodies[nb++] = myId.y;
    for (int bi = 0; bi < nb; ++bi) {
        uint body = bodies[bi];
        uint cnt = min(bodyContactCount[body], CD_MAX_BODY_CONTACTS);
        for (uint k = 0; k < cnt; ++k) {
            uint ncid = bodyContacts[body * CD_MAX_BODY_CONTACTS + k];
            if (ncid == cid || colorIn[ncid] != CD_COLOR_NONE || bid[ncid] != t) continue;
            if (cdOutranks(ncid, cid, myPri, myId, priority, contacts)) { colorOut[cid] = CD_COLOR_NONE; return; }
        }
    }
    colorOut[cid] = t;
}

kernel void coinColorResolve(
    device const CoinContact* contacts         [[ buffer(0) ]],
    device const atomic_uint& contactCount     [[ buffer(1) ]],
    device const uint*        bodyContacts     [[ buffer(2) ]],
    device const uint*        bodyContactCount [[ buffer(3) ]],
    device const uint*        priority         [[ buffer(4) ]],
    device const uint*        colorIn          [[ buffer(5) ]],
    device const uint*        bid              [[ buffer(6) ]],
    device uint*              colorOut         [[ buffer(7) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorResolveBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, bodyContacts,
                       bodyContactCount, priority, colorIn, bid, colorOut);
}

// Publish the finished colouring onto the contacts (tan2.w keeps carrying the colour for
// every other reader) and count anything left uncoloured — with enough rounds and
// CD_MAX_COLORS clearing the real per-body degree this should be zero; what isn't is
// solved by the tail pass (CD_UNCOLORED_BUCKET, up to CD_UNCOLORED_SUB_MAX per substep) and
// shows in `CoinDEMSolver.colorStats.uncolored`.
static inline void cdColorWritebackBody(uint cid, uint n, device CoinContact* contacts,
                                        device const uint* color, device atomic_uint* stats,
                                        uint dispatchedColors)
{
    if (cid >= n) return;
    uint c = color[cid];
    if (c == CD_COLOR_SKIP) { contacts[cid].tan2.w = -1.0; return; }   // solved with its manifold head
    contacts[cid].tan2.w = (c == CD_COLOR_NONE) ? -1.0 : float(c);
    if (c == CD_COLOR_NONE) { atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed); return; }
    // stats[2] = highest colour ever used → the host sizes the next frame's colour sweep
    // from it (Jones–Plassmann always takes the lowest free colour, so colours 0…maxUsed
    // are exactly the non-empty ones; the speculative scheme can leave one of them empty,
    // which costs a 0-group dispatch, never a contact). stats[3] counts contacts that fell
    // beyond the per-colour dispatches the host encoded — they are solved by the serial
    // tail pass (coinSolveVelocityTail), so it is a perf signal, not a loss (VZ-0154).
    atomic_fetch_max_explicit(&stats[2], c + 1u, memory_order_relaxed);
    if (c >= dispatchedColors) atomic_fetch_add_explicit(&stats[3], 1u, memory_order_relaxed);
}

kernel void coinColorWriteback(
    device CoinContact*       contacts     [[ buffer(0) ]],
    device const atomic_uint& contactCount [[ buffer(1) ]],
    device const uint*        color        [[ buffer(2) ]],
    device atomic_uint*       stats        [[ buffer(3) ]],
    constant uint&            dispatchedColors [[ buffer(4) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorWritebackBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, color, stats,
                         dispatchedColors);
}

// ── Compaction: bucket the contacts BY COLOUR ─────────────────────────────────
//
// The solve used to dispatch every live contact once per colour and let 63/64 of the
// threads return on a colour mismatch — so covering all the colours the colouring can
// emit would have cost 64 full passes. Instead we counting-sort the contacts by colour
// (exactly the coinCellCount / coinCellOffsetsScan / coinScatter pattern the broadphase
// uses) into `colorContacts`, and each colour's solve dispatches ONLY its own slice.
// Total threads per velocity iteration drops from colours×contacts to contacts.
//
// Bucket CD_UNCOLORED_BUCKET (= CD_MAX_COLORS, one past the last colour) collects the
// contacts the fixed round budget left UNCOLOURED. They used to be dropped here, i.e.
// never solved: measured in CoinDEMGenericEngineTests' dense 200-body hull/capsule pile,
// 600–12 000 uncoloured contacts per 420-frame run (default 24 rounds), and an unsolved
// cube–cube contact let one cube fall THROUGH another at gravity speed (−0.14 → −0.57
// m/s over 6 frames, 68 mm deep, jₙ = 0, colour −1) before the pile slept with it frozen
// in. The tail pass now solves this bucket after every colour, in the deterministic
// sub-colours coinWriteColorArgs gives it (VZ-0154 follow-through), up to
// CD_UNCOLORED_SUB_MAX contacts per substep — a larger bucket is left unsolved as
// before and counted (colorStats[4]); colorOffset[CD_UNCOLORED_BUCKET + 2] carries the
// sub-colour count (0 ⇒ not solved).
constant uint CD_UNCOLORED_BUCKET = CD_MAX_COLORS;

kernel void coinColorBucketClear(
    device atomic_uint* colorCount [[ buffer(0) ]],
    uint i [[ thread_position_in_grid ]])
{
    atomic_store_explicit(&colorCount[i], 0u, memory_order_relaxed);
}

static inline void cdColorBucketCountBody(uint cid, uint n, device const uint* color, device atomic_uint* colorCount)
{
    if (cid >= n) return;
    uint c = color[cid];
    if (c == CD_COLOR_SKIP) return;
    atomic_fetch_add_explicit(&colorCount[min(c, CD_UNCOLORED_BUCKET)], 1u, memory_order_relaxed);
}

kernel void coinColorBucketCount(
    device const uint*        color        [[ buffer(0) ]],
    device const atomic_uint& contactCount [[ buffer(1) ]],
    device atomic_uint*       colorCount   [[ buffer(2) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorBucketCountBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), color, colorCount);
}

// Prefix-sum the 64 colour buckets + the uncoloured one, and re-zero the counts so
// they serve as write cursors. colorOffset[CD_MAX_COLORS + 1] is the total.
static inline void cdColorBucketScanBody(device uint* colorCount, device uint* colorOffset)
{
    uint sum = 0;
    for (uint c = 0; c <= CD_UNCOLORED_BUCKET; ++c) {
        colorOffset[c] = sum;
        sum += colorCount[c];
        colorCount[c] = 0;                      // reused as the scatter cursor
    }
    colorOffset[CD_UNCOLORED_BUCKET + 1u] = sum;
}

// The same scan on the FIRST SIMD group of a threadgroup (all of its lanes call it together;
// `lane` = the thread's index in it, `w` its width): a lane per bucket, a SIMD prefix sum per
// w buckets. Integer sums, so the offsets are the serial scan's exactly. (Stage B3: the serial
// walk was one thread's dependent device loads and stores, ≈ 10 µs of the colouring per substep.)
static inline void cdColorBucketScanSG(uint lane, uint w, device uint* colorCount, device uint* colorOffset)
{
    uint base = 0u;
    for (uint c0 = 0u; c0 <= CD_UNCOLORED_BUCKET; c0 += w) {          // uniform across the group
        uint c = c0 + lane;
        bool in = c <= CD_UNCOLORED_BUCKET;
        uint cnt = in ? colorCount[c] : 0u;
        uint pre = simd_prefix_exclusive_sum(cnt);
        if (in) { colorOffset[c] = base + pre; colorCount[c] = 0u; }   // (reused as the scatter cursor)
        base += simd_sum(cnt);
    }
    if (lane == 0u) colorOffset[CD_UNCOLORED_BUCKET + 1u] = base;
}

kernel void coinColorBucketScan(
    device uint* colorCount  [[ buffer(0) ]],
    device uint* colorOffset [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid != 0) return;
    cdColorBucketScanBody(colorCount, colorOffset);
}

static inline void cdColorBucketScatterBody(uint cid, uint n, device const uint* color, device atomic_uint* colorCursor,
                                            device const uint* colorOffset, device uint* colorContacts,
                                            device const CoinContact* contacts)
{
    if (cid >= n) return;
    uint col = color[cid];
    if (col == CD_COLOR_SKIP) return;                // a manifold point rides its head's entry
    uint c = min(col, CD_UNCOLORED_BUCKET);          // uncoloured → the serial bucket
    uint slot = atomic_fetch_add_explicit(&colorCursor[c], 1u, memory_order_relaxed);
    // The entry carries the manifold's point count for the solve (1 for an ungrouped contact).
    colorContacts[colorOffset[c] + slot] = cid | (min(cdManifoldCount(contacts[cid]), 15u) << CD_CC_SHIFT);
}

kernel void coinColorBucketScatter(
    device const uint*        color         [[ buffer(0) ]],
    device const atomic_uint& contactCount  [[ buffer(1) ]],
    device atomic_uint*       colorCursor   [[ buffer(2) ]],
    device const uint*        colorOffset   [[ buffer(3) ]],
    device uint*              colorContacts [[ buffer(4) ]],
    device const CoinContact* contacts      [[ buffer(5) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdColorBucketScatterBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), color, colorCursor,
                             colorOffset, colorContacts, contacts);
}

// The uncoloured bucket (a round-budget shortfall: Jones–Plassmann colours a body's
// d contacts at best one per round, so a body of degree ≳ colorRounds leaves some)
// conflicts only with ITSELF once every colour has been solved. coinWriteColorArgs's
// first threadgroup orders it by (identity priority, identity) — the scatter's atomic
// cursor fills it in a racy order — and greedy-colours it in that order into
// sub-colours (lowest free among earlier bucket contacts sharing a body), so the tail
// solves each sub-colour in parallel, race-free and reproducibly.
//
// COST BOUND. The ordering + greedy used to run on ONE thread straight out of device
// memory (an insertion sort and an O(n²) greedy, every step a dependent device load):
// measured ≈ 0.15 µs·n² per substep — 2.1 ms at 128, 9.6 ms at 256 — and past the cap
// the bucket was solved ONE CONTACT PER PASS on one thread (≈ 3.8 µs per contact per
// velocity iteration): a Daydream-Home-scale egg rain (800 eggs, ~1 800 uncoloured per
// substep) went from 11–12 ms to 55–81 ms of GPU per frame. Now the bucket is staged in
// threadgroup memory, ranked in parallel (a stable rank, so the order is exactly the old
// insertion sort's), and each greedy step's scan is spread across the threadgroup (same
// sub-colours, 2 barriers per contact). A bucket larger than
// CD_UNCOLORED_SUB_MAX — or one that would need ≥ 64 sub-colours — is left UNSOLVED for
// the substep, exactly as every uncoloured contact was before the tail existed, and
// counted into colorStats[4] (`CoinDEMSolver.uncoloredUnsolved`): the fix for a scene
// that lands there is enough colouring rounds (VZ-0160), not a serial detour.
constant uint CD_UNCOLORED_SUB_MAX = 256u;

// Order + greedy sub-colour the uncoloured bucket (see COST BOUND above) — ONE threadgroup,
// `tgN` threads, every thread must call it (it has barriers). The staging arrays are the
// caller's threadgroup memory (coinWriteColorArgs declares them; the small-world frame
// kernel carves them out of its shared scratch).
static void cdUncolouredBucketTG(uint tid, uint tgN,
    device uint*              colorOffset,
    device uint*              colorContacts,
    device const uint*        priority,
    device const CoinContact* contacts,
    device uint*              uncolSub,
    device atomic_uint*       stats,
    threadgroup uint*         sCid,     // [CD_UNCOLORED_SUB_MAX] staged, scatter order
    threadgroup uint*         sPri,     // [CD_UNCOLORED_SUB_MAX]
    threadgroup uint4*        sId,      // [CD_UNCOLORED_SUB_MAX]
    threadgroup uint2*        sAB,      // [CD_UNCOLORED_SUB_MAX] the two bodies, SORTED order
    threadgroup uint*         sSub,     // [CD_UNCOLORED_SUB_MAX] sub-colour, SORTED order
    threadgroup atomic_uint*  sUsed,    // [4] greedy step i's used mask, [slot i&1][lo, hi]
    threadgroup uint&         sFail)
{
    {
        uint b0 = colorOffset[CD_UNCOLORED_BUCKET], b1 = colorOffset[CD_UNCOLORED_BUCKET + 1u];
        uint n = b1 - b0;
        bool staged = n > 0u && n <= CD_UNCOLORED_SUB_MAX;
        if (staged) {
            for (uint i = tid; i < n; i += tgN) {
                uint e = colorContacts[b0 + i];
                uint cid = e & CD_CC_MASK;
                sCid[i] = e; sPri[i] = priority[cid]; sId[i] = contacts[cid].meta;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (staged) {
            // Stable rank by (priority, identity), ties by scatter position — the same
            // ascending order the single-thread insertion sort produced.
            for (uint i = tid; i < n; i += tgN) {
                uint pi = sPri[i]; uint4 mi = sId[i];
                uint rank = 0u;
                for (uint j = 0u; j < n; ++j) {
                    uint pj = sPri[j]; uint4 mj = sId[j];
                    bool before = (pj < pi) || (pj == pi && (cdIdGreater(mi, mj) || (!cdIdGreater(mj, mi) && j < i)));
                    rank += before ? 1u : 0u;
                }
                colorContacts[b0 + rank] = sCid[i];
                sAB[rank] = mi.xy;
            }
        }
        if (tid == 0u) {
            for (uint k = 0u; k < 4u; ++k) atomic_store_explicit(&sUsed[k], 0u, memory_order_relaxed);
            sFail = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Greedy sub-colours in that order — contact i takes the lowest sub-colour no
        // earlier bucket contact sharing a body holds. Sequential in i (that is what makes
        // it the greedy), but each step's scan over j < i is spread across the threadgroup
        // and OR-reduced, and thread 0 only picks: the same sub-colours a single thread
        // computes, for 2 barriers per contact instead of an O(n²) serial loop.
        uint nSub = 0u;                                  // (thread 0's copy is the result)
        uint nSteps = staged ? n : 0u;
        for (uint i = 0u; i < nSteps; ++i) {             // no early exit: every thread meets every barrier
            uint slot = (i & 1u) * 2u;
            bool live = (sFail == 0u);                   // written before the previous step's last barrier
            uint2 ai = sAB[i];
            uint mineLo = 0u, mineHi = 0u;
            if (live) {
                for (uint j = tid; j < i; j += tgN) {
                    uint2 aj = sAB[j];
                    if (ai.x == aj.x || ai.x == aj.y || (ai.y != CD_STATIC && (ai.y == aj.x || ai.y == aj.y))) {
                        uint sj = sSub[j];
                        if (sj < 32u) mineLo |= (1u << sj); else mineHi |= (1u << (sj - 32u));
                    }
                }
            }
            if (mineLo != 0u) atomic_fetch_or_explicit(&sUsed[slot],      mineLo, memory_order_relaxed);
            if (mineHi != 0u) atomic_fetch_or_explicit(&sUsed[slot + 1u], mineHi, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u && live) {
                uint lo = atomic_load_explicit(&sUsed[slot], memory_order_relaxed);
                uint hi = atomic_load_explicit(&sUsed[slot + 1u], memory_order_relaxed);
                atomic_store_explicit(&sUsed[slot], 0u, memory_order_relaxed);        // reused at i + 2
                atomic_store_explicit(&sUsed[slot + 1u], 0u, memory_order_relaxed);
                uint sub = (lo != 0xFFFFFFFFu) ? ctz(~lo) : ((hi != 0xFFFFFFFFu) ? 32u + ctz(~hi) : 64u);
                if (sub >= 64u) {
                    sFail = 1u;                          // (degree ≥ 64 inside the bucket) → unsolved
                } else {
                    sSub[i] = sub;
                    uncolSub[b0 + i] = sub;
                    nSub = max(nSub, sub + 1u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            if (sFail != 0u) nSub = 0u;
            colorOffset[CD_UNCOLORED_BUCKET + 2u] = nSub;   // 0 ⇒ the tail skips the bucket
            if (n > 0u && nSub == 0u) atomic_fetch_add_explicit(&stats[4], n, memory_order_relaxed);
        }
    }
}

// One indirect-dispatch argument triple per colour, sized to that colour's slice.
// An empty colour gets 0 threadgroups, so unused colours are free. The first
// threadgroup also orders and sub-colours the uncoloured bucket (above).
kernel void coinWriteColorArgs(
    device uint*       colorOffset [[ buffer(0) ]],
    device uint*       args        [[ buffer(1) ]],   // 4 uints per colour (16-B stride)
    constant uint&     tgSize      [[ buffer(2) ]],
    device uint*              colorContacts [[ buffer(3) ]],
    device const uint*        priority      [[ buffer(4) ]],
    device const CoinContact* contacts      [[ buffer(5) ]],
    device uint*              uncolSub      [[ buffer(6) ]],   // sub-colour per bucket POSITION
    device atomic_uint*       stats         [[ buffer(7) ]],   // colorStats; [4] = left unsolved
    uint c    [[ thread_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tgN  [[ threads_per_threadgroup ]],
    uint tgId [[ threadgroup_position_in_grid ]])
{
    threadgroup uint  sCid[CD_UNCOLORED_SUB_MAX];   // staged, scatter order
    threadgroup uint  sPri[CD_UNCOLORED_SUB_MAX];
    threadgroup uint4 sId[CD_UNCOLORED_SUB_MAX];
    threadgroup uint2 sAB[CD_UNCOLORED_SUB_MAX];    // the two bodies, SORTED order
    threadgroup uint  sSub[CD_UNCOLORED_SUB_MAX];   // sub-colour, SORTED order
    threadgroup atomic_uint sUsed[4];                // greedy step i's used mask, [slot i&1][lo, hi]
    threadgroup uint  sFail;
    if (tgId == 0u)                                  // uniform per threadgroup: barriers are safe
        cdUncolouredBucketTG(tid, tgN, colorOffset, colorContacts, priority, contacts, uncolSub, stats,
                             sCid, sPri, sId, sAB, sSub, sUsed, sFail);
    if (c >= CD_MAX_COLORS) return;
    uint count = colorOffset[c + 1] - colorOffset[c];
    args[c * 4 + 0] = (count + tgSize - 1u) / tgSize;   // threadgroupsPerGrid.x (0 when empty)
    args[c * 4 + 1] = 1u;
    args[c * 4 + 2] = 1u;
    args[c * 4 + 3] = 0u;                               // padding to a 16-B stride
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

// Each constraint-path kernel below is a thin wrapper around a `cd…Body` / `cd…TG` device
// function, so the multi-dispatch path and the small-world frame kernel (CoinDEMSmallWorld.h,
// stage B3) run ONE implementation of every step.
static inline void cdIntegrateVelocityBody(uint id, device CoinBody* coins, device float4* bias,
                                           constant CoinUniforms& u, device const uint* asleep)
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

kernel void coinIntegrateVelocityCS(
    device CoinBody*       coins  [[ buffer(0) ]],
    device float4*         bias   [[ buffer(1) ]],
    constant CoinUniforms& u      [[ buffer(2) ]],
    device const uint*     asleep [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIntegrateVelocityBody(id, coins, bias, u, asleep);
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

// Threadgroup count for a pass that iterates the LIVE contacts. Every contact-indexed
// kernel guards `cid >= contactCount`, so the partial tail is covered; this just stops
// them launching over the whole buffer CAPACITY (12k) when a settled pile has ~2.7k —
// which matters most for the colouring, whose rounds are the densest pass in the solver.
kernel void coinWriteContactArgs(
    device atomic_uint&       contactCount [[ buffer(0) ]],
    device uint*              args         [[ buffer(1) ]],
    constant uint&            tgSize       [[ buffer(2) ]],
    constant uint&            maxContacts  [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id != 0) return;
    // Clamp the append cursor to the buffer: every append past `maxContacts` was dropped,
    // but the cursor kept counting, and every kernel after this one reads it as "contacts
    // 0 ..< n" — past the end of the buffer on an overflow. This is the first dispatch
    // after generation, so from here on the count is the number of contacts that exist.
    uint n = min(atomic_load_explicit(&contactCount, memory_order_relaxed), maxContacts);
    atomic_store_explicit(&contactCount, n, memory_order_relaxed);
    args[0] = (n + tgSize - 1u) / tgSize;
    args[1] = 1u;
    args[2] = 1u;
}

// The per-contact velocity solve of ONE contact — the pre-B1 function verbatim (its
// collider index masked to meta.z's low 16 bits, where the polytope narrowphase folds a
// feature id into the high half). Every ungrouped contact is solved by exactly this code.
// It is kept as a separate straight-line copy on purpose: Metal compiles with fast math,
// so the rounding follows the shape of the code, and folding it into the manifold loop
// below (the same arithmetic, restructured) stepped a Daydream-Home-style egg rain
// differently from frame 6 on. As it is, a world without the manifold solve steps
// bit-identically to the pre-B1 engine (egg rain, a mixed sphere / capsule / disc / egg
// pile: 240 frames, every body's state bit-for-bit). Change the arithmetic here and in
// cdSolveContactVelocity's manifold loop together.
static inline void cdSolveContactVelocityOne(
    uint                      cid,
    device CoinBody*          coins,
    device float4*            bias,
    device CoinContact*       contacts,
    constant CoinUniforms&    u,
    device const uint*        asleep,
    device const float2*      material,   // per-body (μ, e); <0 = inherit
    device const CoinStaticCollider* colliders)
{
    CoinContact c = contacts[cid];

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
    // A static collider is immovable but not necessarily STILL: the kinematic pusher
    // plate carries the pile forward, so a contact against it targets the PLATE's surface
    // velocity, not zero. With zero, the plate is just a wall the pile leans on and only
    // the split-impulse position channel nudges bodies along — which is what the
    // accumulated duplicate contacts used to paper over (the nudge landed once per stale
    // copy still sitting in the buffer).
    float3 vStatic = bStatic ? cdColliderVelocity(colliders, c.meta.z & 0xFFFFu) : float3(0.0);
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
    float3 vpB = bStatic ? vStatic : (vB + cross(wB, rB));
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
    float muB = bStatic ? cdColliderMu(colliders, c.meta.z & 0xFFFFu, u) : cdBodyMu(material, B, u);
    float eB  = bStatic ? cdColliderE(colliders, c.meta.z & 0xFFFFu, u) : cdBodyE(material, B, u);
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
    float dJb = cdSeparatingBias((biasTarget - bvn) / kN, depth, bvn, kN, u);
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
        vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
        float jt1Old = c.rB.w;
        float jt1New = clamp(jt1Old - dot(vrel, t1) / max(kT1, 1e-8), -bound, bound);
        float3 Pt1 = (jt1New - jt1Old) * t1;
        vA += invMa * Pt1; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pt1));
        if (!bStatic) { vB -= invMb * Pt1; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pt1)); }
        c.rB.w = jt1New;
        // tangent 2
        vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
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
    //
    // CD_FLAG_ACCUM_ROLLING (opt-in, VZ-0155): the bound applies to the ACCUMULATED
    // rolling impulse of this contact over the substep (aux.yz, in the (t1, t2) basis;
    // reset to 0 when the contact is generated), clamped to a disc of radius
    // μᵣ·jₙ·|rA| — the standard sequential-impulse treatment the friction rows above
    // already get. The legacy branch below clamps each ITERATION's impulse on its own,
    // so its per-substep cap is velocityIterations × μᵣ·jₙ·|rA| (a Weeble parks at
    // asin(iterations·μᵣ·h_c/d)); every shipping scene is tuned against that, so it
    // stays the default.
    if ((u.solverFlags & CD_FLAG_ACCUM_ROLLING) != 0u) {
        // Not gated on jₙ > 0 alone: when a later iteration unloads this contact to
        // jₙ = 0 (a manifold point whose neighbours took the weight, a separating
        // bounce) the bound is 0 and the clamp must hand back what earlier iterations
        // applied — as the friction rows do. Gating on jₙ > 0 left that impulse on
        // bodies the contact no longer pressed (A1 verifier: 76 of 7415 solved contacts
        // in a tumbling box/sphere pile ended at jₙ = 0 still carrying up to 1.5e-7 N·m·s,
        // ≈ 70 % of a resting 10 g body's whole per-substep rolling bound).
        if (u.rollingResistance > 0.0 && (c.rA.w > 0.0 || any(c.aux.yz != 0.0))) {
            float k1 = dot(t1, cdApplyInvInertiaWorld(qa, invIa, t1))
                     + (bStatic ? 0.0 : dot(t1, cdApplyInvInertiaWorld(qb, invIb, t1)));
            float k2 = dot(t2, cdApplyInvInertiaWorld(qa, invIa, t2))
                     + (bStatic ? 0.0 : dot(t2, cdApplyInvInertiaWorld(qb, invIb, t2)));
            if (k1 > 1e-9 && k2 > 1e-9) {
                float3 wrel = wA - wB;
                float2 accOld = c.aux.yz;
                float2 acc = accOld - float2(dot(wrel, t1) / k1, dot(wrel, t2) / k2);
                float bound = u.rollingResistance * c.rA.w * max(length(rA), 1e-4);
                float al = length(acc);
                if (al > bound) acc *= bound / al;
                float2 d = acc - accOld;
                float3 T = d.x * t1 + d.y * t2;
                wA += cdApplyInvInertiaWorld(qa, invIa, T);
                if (!bStatic) wB -= cdApplyInvInertiaWorld(qb, invIb, T);
                c.aux.yz = acc;
            }
        }
    } else if (u.rollingResistance > 0.0 && c.rA.w > 0.0) {
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
    // bias[2*i+1].w is the joint pass's motor-driven marker — carried through.
    if (!aSleep) {
        coins[A].vel.xyz = vA; coins[A].angVel.xyz = wA;
        bias[2*A] = float4(bvA, 0.0); bias[2*A+1] = float4(bwA, bias[2*A+1].w);
    }
    if (!bSleep) {
        coins[B].vel.xyz = vB; coins[B].angVel.xyz = wB;
        bias[2*B] = float4(bvB, 0.0); bias[2*B+1] = float4(bwB, bias[2*B+1].w);
    }
    contacts[cid] = c;
}

// ── Torsional (point) friction — CD_FLAG_TORSION (engine plan 5a, stage B2) ───────────────
//
// A real contact is a small PATCH pressed with normal force N, and Coulomb friction across it
// resists spin ABOUT THE NORMAL with a moment of order μ·N times the patch's radius (a
// uniformly loaded disc of radius a: (2/3)·μ·N·a; a Hertzian patch: (3π/16)·μ·N·a). The
// narrowphase's point contact has no lever about n, so a body spinning about its contact
// normal — a Weeble given a yaw rate — never slowed (the rolling row leaves twist alone): > 70 %
// of a 5 rad/s spin was left after 1 s (CoinDEMActuationTests #8). This is the row that stops
// it: ONE angular row about n per contact, the relative spin (ω_A − ω_B)·n driven to 0 with the
// effective mass 1 / (n·I_A⁻¹·n + n·I_B⁻¹·n) (world inverse inertia, movable ends only), its
// impulse ACCUMULATED over the substep (ext.x, warm-started with the contact's other rows)
// inside ±μ·r·Λn·share — Λn the contact's accumulated normal impulse, μ its combined Coulomb μ,
// r = max(r_A, r_B) the per-body EFFECTIVE patch radius (the moment arm that multiplies μ·N; a
// static side has no patch of its own), share = 1/n on each point of an n-point grouped
// manifold (a flat face resists spin mostly through its spread points' tangential friction;
// the patch term is shared across them, not multiplied — plan 5a). Like the accumulated
// rolling row it runs while Λn > 0 OR it still holds an impulse, so a contact a later iteration
// unloads hands back what it applied.
//
// The spring foot's pad row (CoinDEMFoot.h) is the same law — |Λ| ≤ μ_spin·r_patch·Λn_leg, its
// OWN normal impulse — so a worker standing on its shell AND a preloaded pad has
// μ·r·(Λn_shell + Λn_leg) = μ·r·m·g·dt of torsion in all: each patch counted once, by its own
// share of the load (CoinDEMTorsionTests).
//
// An UNGROUPED contact runs this as its own step right after cdSolveContactVelocityOne, on the
// velocities that function wrote back (race-free: the colour gives this thread both bodies),
// so the pre-B2 single-contact code — and every world without a patch — is untouched.
static inline void cdSolveContactTorsion(
    uint                      cid,
    device CoinBody*          coins,
    device CoinContact*       contacts,
    constant CoinUniforms&    u,
    device const uint*        asleep,
    device const float2*      material,
    device const CoinStaticCollider* colliders,
    device const float*       patch)
{
    CoinContact c = contacts[cid];
    uint A = c.meta.x, B = c.meta.y;
    bool bStatic = (B == CD_STATIC);
    bool aSleep = (asleep[A] != 0u);
    bool bSleep = bStatic || (asleep[B] != 0u);
    if (aSleep && bSleep) return;
    float r = max(patch[A], bStatic ? 0.0 : patch[B]);
    float lamOld = c.ext.x;
    if (!(r > 0.0) && lamOld == 0.0) return;
    uint colIdx = c.meta.z & 0xFFFFu;
    float muA = cdBodyMu(material, A, u);
    float muB = bStatic ? cdColliderMu(colliders, colIdx, u) : cdBodyMu(material, B, u);
    float cap = sqrt(max(muA * muB, 0.0)) * max(r, 0.0);      // torsion per unit normal impulse (m)
    float3 n = c.nrm.xyz;
    CoinBody a = coins[A];
    float3 dA = cdApplyInvInertiaWorld(a.orient, aSleep ? float3(0.0) : cdBodyInvInertia(a, a.posInvMass.w), n);
    float3 wA = a.angVel.xyz, wB = float3(0.0), dB = float3(0.0);
    if (!bStatic) {
        CoinBody b = coins[B];
        dB = cdApplyInvInertiaWorld(b.orient, bSleep ? float3(0.0) : cdBodyInvInertia(b, b.posInvMass.w), n);
        wB = b.angVel.xyz;
    }
    float k = dot(n, dA) + dot(n, dB);                         // n·I_A⁻¹·n + n·I_B⁻¹·n
    if (!(k > 0.0)) return;
    float bound = cap * max(c.rA.w, 0.0);
    float lamNew = clamp(lamOld - dot(wA - wB, n) / k, -bound, bound);
    float d = lamNew - lamOld;
    if (!aSleep) coins[A].angVel.xyz = wA + d * dA;
    if (!bSleep) coins[B].angVel.xyz = wB - d * dB;
    contacts[cid].ext.xy = float2(lamNew, cap);
}


// The per-contact velocity solve, shared by the per-colour dispatch and the
// serial tail pass below (a contact is solved identically by either).
//
// MANIFOLD SOLVE (CD_FLAG_MANIFOLD_SOLVE, opt-in). `entry` is a colorContacts entry: the
// contact index, and in its top 4 bits the number of contiguous contacts of one grouped
// manifold (cdEmitGroup) — 1 for every ungrouped contact, which goes to
// cdSolveContactVelocityOne, the path it always took. A grouped manifold
// (≤ 4 points of one body pair, allocated contiguously and coloured as ONE unit) is solved
// by one thread with the pair's velocities in registers: `manifoldPasses` Gauss–Seidel
// passes over its points' normal / bias / friction rows, then one rolling pass. Why: a
// 4-point manifold is a REDUNDANT system (rank 3 on a rigid body), and one Gauss–Seidel
// visit per point per iteration — the points in four different colours — leaves it
// asymmetric: a 10 mm cube resting on a cube got ω = 0.48 rad/s from its very first
// cold 6-iteration solve (0 at 30 iterations), so a 5-cube tower fell at frame 3 and a lone
// resting box yawed ~1°/s (VZ-0157, VZ-0163). Solved as a unit, each manifold is (near)
// exact on every visit; the colour count also drops (one node per manifold, not per point).
static inline void cdSolveContactVelocity(
    uint                      entry,
    device CoinBody*          coins,
    device float4*            bias,
    device CoinContact*       contacts,
    constant CoinUniforms&    u,
    device const uint*        asleep,
    device const float2*      material,   // per-body (μ, e); <0 = inherit
    device const CoinStaticCollider* colliders,
    device const float*       patch)      // per-body contact patch radius (m), CD_FLAG_TORSION
{
    uint cid0 = entry & CD_CC_MASK;
    uint nPts = max(entry >> CD_CC_SHIFT, 1u);
    if (nPts == 1u) {
        cdSolveContactVelocityOne(cid0, coins, bias, contacts, u, asleep, material, colliders);
        if ((u.solverFlags & CD_FLAG_TORSION) != 0u)
            cdSolveContactTorsion(cid0, coins, contacts, u, asleep, material, colliders, patch);
        return;
    }
    CoinContact c = contacts[cid0];

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
    // A static collider is immovable but not necessarily STILL: the kinematic pusher
    // plate carries the pile forward, so a contact against it targets the PLATE's surface
    // velocity, not zero. With zero, the plate is just a wall the pile leans on and only
    // the split-impulse position channel nudges bodies along — which is what the
    // accumulated duplicate contacts used to paper over (the nudge landed once per stale
    // copy still sitting in the buffer).
    // Static contacts carry the collider index in meta.z's low 16 bits (the polytope
    // narrowphase folds a feature id into the high half; every other path leaves it 0).
    uint colIdx = c.meta.z & 0xFFFFu;
    float3 vStatic = bStatic ? cdColliderVelocity(colliders, colIdx) : float3(0.0);
    CoinBody b;
    if (!bStatic) {
        b = coins[B];
        invMb = bSleep ? 0.0 : b.posInvMass.w;
        invIb = bSleep ? float3(0.0) : cdBodyInvInertia(b, b.posInvMass.w);
        qb = b.orient;
        vB = b.vel.xyz; wB = b.angVel.xyz; bvB = bias[2*B].xyz; bwB = bias[2*B+1].xyz;
    }

    // ── Per-contact material: combine the two sides' (μ, e). A static side
    // reads the COLLIDER's own material (c.meta.z is the collider index for a
    // static contact — see cdEmitContact/cdPairKey), not a mirror of A's — an
    // icy ramp next to a rubber floor now actually differs. Same combine rule
    // and "<0 = inherit the global uniform" convention as per-body material, so
    // the all-defaults case still reduces exactly to the global uniforms.
    float muA = cdBodyMu(material, A, u);
    float eA  = cdBodyE(material, A, u);
    float muB = bStatic ? cdColliderMu(colliders, colIdx, u) : cdBodyMu(material, B, u);
    float eB  = bStatic ? cdColliderE(colliders, colIdx, u) : cdBodyE(material, B, u);
    float muC = sqrt(max(muA * muB, 0.0));      // geometric mean (Box2D convention)
    float eC  = max(eA, eB);                    // bounciest material wins
    // Torsion (CD_FLAG_TORSION, cdSolveContactTorsion): each point of the n carries 1/n of
    // the pair's patch term — μ·r·Λn_i/n — solved on the registers with the other rows.
    bool torsionOn = (u.solverFlags & CD_FLAG_TORSION) != 0u;
    float tCap = torsionOn ? muC * max(max(patch[A], bStatic ? 0.0 : patch[B]), 0.0) / float(nPts) : 0.0;

    uint passes = (nPts == 1u) ? 1u : max(u.manifoldPasses, 1u);
    for (uint pass = 0u; pass < passes; ++pass) {
    for (uint k = 0u; k < nPts; ++k) {
    if (nPts > 1u) c = contacts[cid0 + k];
    bool lastPass = (pass + 1u == passes);
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
    float3 vpB = bStatic ? vStatic : (vB + cross(wB, rB));
    float3 vrel = vpA - vpB;
    float vn = dot(vrel, n);                       // <0 = closing along the normal


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
    float dJb = cdSeparatingBias((biasTarget - bvn) / kN, depth, bvn, kN, u);
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
        vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
        float jt1Old = c.rB.w;
        float jt1New = clamp(jt1Old - dot(vrel, t1) / max(kT1, 1e-8), -bound, bound);
        float3 Pt1 = (jt1New - jt1Old) * t1;
        vA += invMa * Pt1; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pt1));
        if (!bStatic) { vB -= invMb * Pt1; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pt1)); }
        c.rB.w = jt1New;
        // tangent 2
        vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
        float jt2Old = c.tan1.w;
        float jt2New = clamp(jt2Old - dot(vrel, t2) / max(kT2, 1e-8), -bound, bound);
        float3 Pt2 = (jt2New - jt2Old) * t2;
        vA += invMa * Pt2; wA += cdApplyInvInertiaWorld(qa, invIa, cross(rA, Pt2));
        if (!bStatic) { vB -= invMb * Pt2; wB -= cdApplyInvInertiaWorld(qb, invIb, cross(rB, Pt2)); }
        c.tan1.w = jt2New;
    }

    // ── Torsional friction (CD_FLAG_TORSION; the law of cdSolveContactTorsion, 1/n share) ─
    if (torsionOn && (tCap > 0.0 || c.ext.x != 0.0)) {
        float3 dA = cdApplyInvInertiaWorld(qa, invIa, n);
        float3 dB = bStatic ? float3(0.0) : cdApplyInvInertiaWorld(qb, invIb, n);
        float kT = dot(n, dA) + dot(n, dB);
        if (kT > 0.0) {
            float tBound = tCap * max(c.rA.w, 0.0);
            float lamOld = c.ext.x;
            float lamNew = clamp(lamOld - dot(wA - wB, n) / kT, -tBound, tBound);
            float dl = lamNew - lamOld;
            wA += dl * dA;
            if (!bStatic) wB -= dl * dB;
            c.ext.xy = float2(lamNew, tCap);
        }
    }

    // ── Rolling resistance (constraint path): a physical angular constraint ────
    // Opposes the RELATIVE rolling spin (ω ⟂ n) with an angular impulse bounded
    // by μᵣ · jₙ · r (coefficient × normal impulse × contact lever) — so a ball
    // rolls out and STOPS on a level floor instead of coasting forever, and the
    // stopping torque scales with how hard the contact is loaded, exactly like
    // real rolling friction. Twist ABOUT n is friction's job, not this.
    //
    // CD_FLAG_ACCUM_ROLLING (opt-in, VZ-0155): the bound applies to the ACCUMULATED
    // rolling impulse of this contact over the substep (aux.yz, in the (t1, t2) basis;
    // reset to 0 when the contact is generated), clamped to a disc of radius
    // μᵣ·jₙ·|rA| — the standard sequential-impulse treatment the friction rows above
    // already get. The legacy branch below clamps each ITERATION's impulse on its own,
    // so its per-substep cap is velocityIterations × μᵣ·jₙ·|rA| (a Weeble parks at
    // asin(iterations·μᵣ·h_c/d)); every shipping scene is tuned against that, so it
    // stays the default.
    if (!lastPass) {
        // rolling resistance: once per outer iteration (the last manifold pass), so its
        // per-iteration bound is what it always was
    } else if ((u.solverFlags & CD_FLAG_ACCUM_ROLLING) != 0u) {
        // Not gated on jₙ > 0 alone: when a later iteration unloads this contact to
        // jₙ = 0 (a manifold point whose neighbours took the weight, a separating
        // bounce) the bound is 0 and the clamp must hand back what earlier iterations
        // applied — as the friction rows do. Gating on jₙ > 0 left that impulse on
        // bodies the contact no longer pressed (A1 verifier: 76 of 7415 solved contacts
        // in a tumbling box/sphere pile ended at jₙ = 0 still carrying up to 1.5e-7 N·m·s,
        // ≈ 70 % of a resting 10 g body's whole per-substep rolling bound).
        if (u.rollingResistance > 0.0 && (c.rA.w > 0.0 || any(c.aux.yz != 0.0))) {
            float k1 = dot(t1, cdApplyInvInertiaWorld(qa, invIa, t1))
                     + (bStatic ? 0.0 : dot(t1, cdApplyInvInertiaWorld(qb, invIb, t1)));
            float k2 = dot(t2, cdApplyInvInertiaWorld(qa, invIa, t2))
                     + (bStatic ? 0.0 : dot(t2, cdApplyInvInertiaWorld(qb, invIb, t2)));
            if (k1 > 1e-9 && k2 > 1e-9) {
                float3 wrel = wA - wB;
                float2 accOld = c.aux.yz;
                float2 acc = accOld - float2(dot(wrel, t1) / k1, dot(wrel, t2) / k2);
                float bound = u.rollingResistance * c.rA.w * max(length(rA), 1e-4);
                float al = length(acc);
                if (al > bound) acc *= bound / al;
                float2 d = acc - accOld;
                float3 T = d.x * t1 + d.y * t2;
                wA += cdApplyInvInertiaWorld(qa, invIa, T);
                if (!bStatic) wB -= cdApplyInvInertiaWorld(qb, invIb, T);
                c.aux.yz = acc;
            }
        }
    } else if (u.rollingResistance > 0.0 && c.rA.w > 0.0) {
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

    contacts[cid0 + k] = c;
    }   // points
    }   // passes

    // Write back (an inert/asleep end carried no impulse, so leave it untouched).
    // bias[2*i+1].w is the joint pass's motor-driven marker — carried through.
    if (!aSleep) {
        coins[A].vel.xyz = vA; coins[A].angVel.xyz = wA;
        bias[2*A] = float4(bvA, 0.0); bias[2*A+1] = float4(bwA, bias[2*A+1].w);
    }
    if (!bSleep) {
        coins[B].vel.xyz = vB; coins[B].angVel.xyz = wB;
        bias[2*B] = float4(bvB, 0.0); bias[2*B+1] = float4(bwB, bias[2*B+1].w);
    }
}

// Solve all contacts of one colour. The dispatch is sized to this colour's slice of the
// compacted `colorContacts` list, so every thread has real work (no colour filtering).
// `currentColor` is passed per-dispatch (setBytes).
kernel void coinSolveVelocityColor(
    device CoinBody*          coins        [[ buffer(0) ]],
    device float4*            bias         [[ buffer(1) ]],
    device CoinContact*       contacts     [[ buffer(2) ]],
    device const uint*        colorContacts [[ buffer(3) ]],
    constant CoinUniforms&    u            [[ buffer(4) ]],
    constant uint&            currentColor [[ buffer(5) ]],
    device const uint*        asleep       [[ buffer(6) ]],
    device const float2*      material     [[ buffer(7) ]],   // per-body (μ, e); <0 = inherit
    device const CoinStaticCollider* colliders [[ buffer(8) ]],
    device const uint*        colorOffset  [[ buffer(9) ]],
    device const float*       patch        [[ buffer(11) ]],  // per-body patch radius (CD_FLAG_TORSION)
    uint i [[ thread_position_in_grid ]])
{
    uint start = colorOffset[currentColor], end = colorOffset[currentColor + 1u];
    if (start + i >= end) return;                // partial tail of the last threadgroup
    cdSolveContactVelocity(colorContacts[start + i], coins, bias, contacts, u, asleep, material, colliders, patch);
}

// The colour sweep's fail-safe (VZ-0154). The host dispatches one coinSolveVelocityColor
// per colour for colours [0, firstColor) — sized from the most colours the GPU has
// reported so far, +1 — and this ONE threadgroup then solves every colour from
// `firstColor` to CD_MAX_COLORS in order, a colour at a time with a device-memory
// barrier between colours: the same Gauss–Seidel order and the same per-contact solve,
// so the result does not depend on where the host put the split. A colour the host
// didn't dispatch used to be skipped for the substep (the `beyondSweep` count); now it
// costs serial time on one threadgroup instead, and nothing when it is empty (every
// thread reads the same prefix-sum offsets, so the early-out and the barriers are
// uniform across the threadgroup).
static void cdSolveVelocityTailTG(uint firstColor, uint tid, uint tgSize,
    device CoinBody*          coins,
    device float4*            bias,
    device CoinContact*       contacts,
    device const uint*        colorContacts,
    constant CoinUniforms&    u,
    device const uint*        asleep,
    device const float2*      material,
    device const CoinStaticCollider* colliders,
    device const uint*        colorOffset,
    device const uint*        uncolSub,
    device const float*       patch)
{
    uint first = min(firstColor, CD_UNCOLORED_BUCKET);
    if (colorOffset[first] >= colorOffset[CD_UNCOLORED_BUCKET + 1u]) return;
    // Passes, in order: every colour from `first` up; then the uncoloured bucket (after
    // every coloured contact of this iteration) as its sub-colours (coinWriteColorArgs).
    // A bucket coinWriteColorArgs could not sub-colour (nSub 0: over CD_UNCOLORED_SUB_MAX)
    // is skipped — it used to be solved one contact per pass here, ≈ 3.8 µs per contact
    // per iteration on one thread, the Daydream-Home-scale cliff.
    // ONE call site of cdSolveContactVelocity, like coinSolveVelocityColor, so both
    // kernels compile the per-contact solve the same way and WHERE the host splits the
    // sweep cannot change a single bit of the result.
    uint b0 = colorOffset[CD_UNCOLORED_BUCKET], b1 = colorOffset[CD_UNCOLORED_BUCKET + 1u];
    uint nSub = colorOffset[CD_UNCOLORED_BUCKET + 2u];
    uint nColourPasses = CD_UNCOLORED_BUCKET - first;
    uint nPasses = nColourPasses + (b1 > b0 ? nSub : 0u);
    for (uint pass = 0u; pass < nPasses; ++pass) {
        uint start, end, sub = 0xFFFFFFFFu;
        if (pass < nColourPasses) {
            start = colorOffset[first + pass];
            // Every coloured contact is behind us (the offsets are cumulative and b0 is where
            // the coloured ones end), so every colour from here up is empty: go straight to the
            // uncoloured bucket's passes. Stage B3: walking the empty colours cost two dependent
            // device loads each — ≈ 20 µs per call on the in-order GPU, i.e. per velocity
            // iteration for a world whose contacts fill colour 0 (the small-world kernel sweeps
            // from colour 0; the multi-dispatch tail from its split). They did nothing — no work,
            // no barrier — so skipping them changes no bit of the result.
            if (start >= b0) { pass = nColourPasses - 1u; continue; }
            end = colorOffset[first + pass + 1u];
        } else { start = b0; end = b1; sub = pass - nColourPasses; }
        if (start >= end) continue;
        for (uint k = start + tid; k < end; k += tgSize)
            if (sub == 0xFFFFFFFFu || uncolSub[k] == sub)
                cdSolveContactVelocity(colorContacts[k], coins, bias, contacts, u, asleep, material, colliders, patch);
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void coinSolveVelocityTail(
    device CoinBody*          coins        [[ buffer(0) ]],
    device float4*            bias         [[ buffer(1) ]],
    device CoinContact*       contacts     [[ buffer(2) ]],
    device const uint*        colorContacts [[ buffer(3) ]],
    constant CoinUniforms&    u            [[ buffer(4) ]],
    constant uint&            firstColor   [[ buffer(5) ]],
    device const uint*        asleep       [[ buffer(6) ]],
    device const float2*      material     [[ buffer(7) ]],
    device const CoinStaticCollider* colliders [[ buffer(8) ]],
    device const uint*        colorOffset  [[ buffer(9) ]],
    device const uint*        uncolSub     [[ buffer(10) ]],
    device const float*       patch        [[ buffer(11) ]],  // per-body patch radius (CD_FLAG_TORSION)
    uint tid    [[ thread_index_in_threadgroup ]],
    uint tgSize [[ threads_per_threadgroup ]])
{
    cdSolveVelocityTailTG(firstColor, tid, tgSize, coins, bias, contacts, colorContacts, u, asleep, material,
                          colliders, colorOffset, uncolSub, patch);
}

static inline void cdIntegratePositionBody(uint id, device CoinBody* coins, device const float4* bias,
                                           constant CoinUniforms& u, device const uint* asleep)
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

kernel void coinIntegratePositionCS(
    device CoinBody*       coins  [[ buffer(0) ]],
    device const float4*   bias   [[ buffer(1) ]],
    constant CoinUniforms& u      [[ buffer(2) ]],
    device const uint*     asleep [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIntegratePositionBody(id, coins, bias, u, asleep);
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
// ── KERNEL: capture every contact's pre-solve approach speed (manifoldSolve) ─────────
//
// The restitution target −e·vn₀ is anchored to the approach speed vn₀ the contact had
// BEFORE the solve. The solve captures it lazily, on the contact's own first visit — in
// colour order, so every contact after the first on a body reads an approach speed its
// predecessors have already cut. A 12 mm L dropped flat from 3 mm (0.24 m/s): its first
// corner captured the full impact and bounced (e 0.15), later corners captured less than
// the 0.14 m/s rest threshold and did not, and the one-sided bounce tipped the L 0.5° in
// the impact frame — at 60 iterations as at 20 (the capture, not convergence). Under
// CD_FLAG_MANIFOLD_SOLVE every contact captures here instead, from the velocities before
// the warm start and the first colour (Box2D's prepare stage); the solve's lazy capture
// then finds aux.w = 1 and keeps it. Runs after the colouring, which has consumed the
// manifold marker aux.w carried until now (it is packed into the colour entries).
static inline void cdCaptureApproachBody(uint cid, uint n, device CoinContact* contacts,
                                         device const CoinBody* coins, device const CoinStaticCollider* colliders)
{
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    uint A = c.meta.x, B = c.meta.y;
    CoinBody a = coins[A];
    float3 vpA = a.vel.xyz + cross(a.angVel.xyz, c.rA.xyz);
    float3 vpB;
    if (B == CD_STATIC) vpB = cdColliderVelocity(colliders, c.meta.z & 0xFFFFu);
    else { CoinBody b = coins[B]; vpB = b.vel.xyz + cross(b.angVel.xyz, c.rB.xyz); }
    contacts[cid].aux.x = dot(vpA - vpB, c.nrm.xyz);
    contacts[cid].aux.w = 1.0;
}

kernel void coinCaptureApproach(
    device CoinContact*               contacts     [[ buffer(0) ]],
    device const atomic_uint&         contactCount [[ buffer(1) ]],
    device const CoinBody*            coins        [[ buffer(2) ]],
    device const CoinStaticCollider*  colliders    [[ buffer(3) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdCaptureApproachBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, coins, colliders);
}

static inline void cdSnapshotContactBody(uint cid, uint n, device const CoinContact* contacts,
                                         device CoinContact* prev, device atomic_uint* pairHash, uint hashSize)
{
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

// CD_FLAG_KEEP_ASLEEP_CONTACTS: carry one entry of the last substep's contact table into this
// substep's list while BOTH its ends are inert now (asleep, or asleep vs a static — generation
// skipped the pair, VZ-0152), marked dormant (ext.z = 1: never coloured, never solved). The
// snapshot then carries it on to the next substep, so a sleeping island's converged impulses stay
// in the warm-start table for as long as it sleeps; the substep a touch wakes it, its fresh
// contacts match them. One thread per hash slot: the table indexes exactly the last snapshot.
// A carried contact is the same record (identity, points, normal, impulses): nothing it touches
// moved while asleep. Appends like an emitter; the colouring's cursor clamp covers an overflow.
static inline void cdCarryDormantBody(uint slot, uint hashSize, device const uint* pairHash,
                                      device const CoinContact* prev, device const CoinBody* coins,
                                      device const uint* asleep, device CoinContact* contacts,
                                      device atomic_uint* contactCount, uint maxContacts)
{
    if (slot >= hashSize) return;
    uint idx = pairHash[slot];
    if (idx == CD_HASH_EMPTY || idx >= maxContacts) return;
    CoinContact c = prev[idx];
    uint A = c.meta.x, B = c.meta.y;
    if (A >= maxContacts || asleep[A] == 0u || coins[A].posInvMass.w == 0.0) return;
    if (B != CD_STATIC && (asleep[B] == 0u || coins[B].posInvMass.w == 0.0)) return;
    uint k = atomic_fetch_add_explicit(contactCount, 1u, memory_order_relaxed);
    if (k >= maxContacts) return;                    // buffer full — dropped (the clamp follows)
    c.ext.z = 1.0;                                   // dormant
    c.tan2.w = -1.0;                                 // uncoloured
    contacts[k] = c;
}

kernel void coinCarryDormant(
    device const uint*         pairHash     [[ buffer(0) ]],
    constant uint&             hashSize     [[ buffer(1) ]],
    device const CoinContact*  prev         [[ buffer(2) ]],
    device const CoinBody*     coins        [[ buffer(3) ]],
    device const uint*         asleep       [[ buffer(4) ]],
    device CoinContact*        contacts     [[ buffer(5) ]],
    device atomic_uint*        contactCount [[ buffer(6) ]],
    constant uint&             maxContacts  [[ buffer(7) ]],
    uint slot [[ thread_position_in_grid ]])
{
    cdCarryDormantBody(slot, hashSize, pairHash, prev, coins, asleep, contacts, contactCount, maxContacts);
}

kernel void coinSnapshotContacts(
    device const CoinContact*  contacts     [[ buffer(0) ]],
    device const atomic_uint&  contactCount [[ buffer(1) ]],
    device CoinContact*        prev         [[ buffer(2) ]],
    device atomic_uint*        pairHash     [[ buffer(3) ]],
    constant uint&             hashSize     [[ buffer(4) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdSnapshotContactBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, prev,
                          pairHash, hashSize);
}

// For each FRESH contact, find last substep's matching contact by pairKey and copy its
// converged impulses (normal→rA.w, friction1→rB.w, friction2→tan1.w) as the warm seed.
// `torsionOn`: 1 while CD_FLAG_TORSION is set this substep. The torsion impulse (ext.x) is
// carried only then: the torsion row is what clamps / hands back a carried impulse, so once
// the last patch is cleared (the flag drops) a carried ext.x would be re-applied by the warm
// start every substep and never re-solved — a resting sphere cleared mid-slide went from
// ω 3.50 to −7.68 rad/s in 30 substeps (0.373 rad/s per substep, without end).
static inline void cdWarmStartMatchBody(uint cid, uint n, device CoinContact* contacts,
                                        device const CoinContact* prev, device const uint* pairHash,
                                        uint hashSize, device const CoinBody* coins, uint torsionOn)
{
    if (cid >= n) return;
    uint key = contacts[cid].meta.w;
    uint h = cdHashU(key) & (hashSize - 1u);
    CoinContact me = contacts[cid];
    bool posMatch = (key & 0xFF000000u) == CD_KEY_POSMATCH;
    // Position fallback (polytope contacts only): the old point of the SAME pair nearest to
    // this one in A's frame offset, within a tenth of the smaller body's half-thickness,
    // with the normal within ~18°. VZ-0163: a 10 mm cube stacked square on another puts
    // its corners exactly on the other's side planes, so slight wobble swaps a corner for
    // two clip points and back every substep — an exact-id match then cold-started the
    // manifold every other substep and a warm-started 5-cube tower still fell (frame 33);
    // with the fallback it stands (worst tilt < 0.1°).
    float tolMatch = 1e30;
    if (posMatch) {
        float hA = cdHalfThickOf(coins[me.meta.x]);
        float hB = (me.meta.y == CD_STATIC) ? hA : cdHalfThickOf(coins[me.meta.y]);
        tolMatch = 0.1 * min(hA, hB);
    }
    uint best = CD_HASH_EMPTY; float bestD = tolMatch;
    for (uint probe = 0; probe < 32u; ++probe) {
        uint slot = pairHash[(h + probe) & (hashSize - 1u)];
        if (slot == CD_HASH_EMPTY) break;
        CoinContact o = prev[slot];
        // The WHOLE identity must match, not just the 32-bit key: the key packs 12-bit
        // body ids and an 8-bit feature (or, for a polytope contact, the pair alone), so
        // two contacts can share it — and a key collision seeded one contact with
        // ANOTHER's impulse, at a different point.
        if (all(o.meta == me.meta)) { best = slot; break; }
        if (posMatch && o.meta.w == key && o.meta.x == me.meta.x && o.meta.y == me.meta.y
            && (me.meta.y != CD_STATIC || (o.meta.z & 0xFFFFu) == (me.meta.z & 0xFFFFu))
            && dot(o.nrm.xyz, me.nrm.xyz) > 0.95) {
            float d = distance(o.rA.xyz, me.rA.xyz);
            if (d < bestD) { bestD = d; best = slot; }
        }
    }
    if (best != CD_HASH_EMPTY) {
        CoinContact o = prev[best];
        if (posMatch) {
            // Re-project the old friction impulse onto this contact's tangent basis.
            float3 Pt = o.rB.w * o.tan1.xyz + o.tan1.w * o.tan2.xyz;
            contacts[cid].rA.w   = o.rA.w * max(dot(o.nrm.xyz, me.nrm.xyz), 0.0);
            contacts[cid].rB.w   = dot(Pt, me.tan1.xyz);
            contacts[cid].tan1.w = dot(Pt, me.tan2.xyz);
            contacts[cid].ext.x  = (torsionOn != 0u) ? o.ext.x * max(dot(o.nrm.xyz, me.nrm.xyz), 0.0) : 0.0;   // torsion about n
        } else {
            contacts[cid].rA.w   = o.rA.w;     // warm seed (the solve continues from here)
            contacts[cid].rB.w   = o.rB.w;
            contacts[cid].tan1.w = o.tan1.w;
            contacts[cid].ext.x  = (torsionOn != 0u) ? o.ext.x : 0.0;    // torsion (CD_FLAG_TORSION only)
        }
    }
}

kernel void coinWarmStartMatch(
    device CoinContact*        contacts     [[ buffer(0) ]],
    device const atomic_uint&  contactCount [[ buffer(1) ]],
    device const CoinContact*  prev         [[ buffer(2) ]],
    device const uint*         pairHash     [[ buffer(3) ]],
    constant uint&             hashSize     [[ buffer(4) ]],
    device const CoinBody*     coins        [[ buffer(5) ]],
    constant uint&             torsionOn    [[ buffer(6) ]],   // see cdWarmStartMatchBody
    uint cid [[ thread_position_in_grid ]])
{
    cdWarmStartMatchBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, prev, pairHash,
                         hashSize, coins, torsionOn);
}

// Apply the warm-start impulses to the bodies' velocities (per colour, race-free).
//
// One contact's warm-start impulse, applied to its two bodies — the pre-B1 code, its two
// body updates now guarded by the VZ-0162 asleep test and otherwise untouched. Kept in
// this exact shape, and called on its own for a single contact: Metal compiles with fast
// math, so the rounding follows the shape of the code, and a loop-restructured copy (the
// same arithmetic) stepped a warm-started pile differently from the pre-B1 engine. As it
// is, a warm-started world with no body asleep steps bit-identically to it (240 frames of
// a sphere / capsule / disc / egg pile, every body's state bit for bit).
//
// An ASLEEP end is immovable here exactly as in the solve (VZ-0162): the warm start used
// to add its impulse to an asleep body's velocity with the body's real inverse mass — a
// velocity nothing integrates (the body is frozen) but that the solve then READ as that
// body's motion (vB), so an awake neighbour resting on a sleeping pile was solved against
// a pile that "moved"; and a host-held kinematic body (setKinematicHold) got kicked.
static inline void cdWarmStartApplyOne(uint cid, device CoinBody* coins, device CoinContact* contacts,
                                       device const uint* asleep) {
    CoinContact c = contacts[cid];
    float jn = c.rA.w, jt1 = c.rB.w, jt2 = c.tan1.w;
    if (jn == 0.0 && jt1 == 0.0 && jt2 == 0.0) return;
    uint A = c.meta.x, B = c.meta.y; bool bStatic = (B == CD_STATIC);
    float3 P = jn * c.nrm.xyz + jt1 * c.tan1.xyz + jt2 * c.tan2.xyz;
    if (asleep[A] == 0u) {
        CoinBody a = coins[A]; float invMa = a.posInvMass.w; float3 invIa = cdBodyInvInertia(a, invMa);
        coins[A].vel.xyz    += invMa * P;
        coins[A].angVel.xyz += cdApplyInvInertiaWorld(a.orient, invIa, cross(c.rA.xyz, P));
    }
    if (!bStatic && asleep[B] == 0u) {
        CoinBody b = coins[B]; float invMb = b.posInvMass.w; float3 invIb = cdBodyInvInertia(b, invMb);
        coins[B].vel.xyz    -= invMb * P;
        coins[B].angVel.xyz -= cdApplyInvInertiaWorld(b.orient, invIb, cross(c.rB.xyz, P));
    }
}

// The torsion row's warm start (CD_FLAG_TORSION): its carried impulse about n onto the awake
// ends' spin — a separate step after cdWarmStartApplyOne, so that code keeps its exact shape.
// ext.x is 0 unless the torsion row ran (every emitter writes 0; the match carries it only while
// CD_FLAG_TORSION is set), so in a world without a patch — including one whose last patch was
// just cleared — this reads one float and returns.
static inline void cdWarmStartTorsionOne(uint cid, device CoinBody* coins, device const CoinContact* contacts,
                                         device const uint* asleep) {
    float lam = contacts[cid].ext.x;
    if (lam == 0.0) return;
    uint A = contacts[cid].meta.x, B = contacts[cid].meta.y;
    float3 L = lam * contacts[cid].nrm.xyz;
    if (asleep[A] == 0u) {
        CoinBody a = coins[A];
        coins[A].angVel.xyz = a.angVel.xyz + cdApplyInvInertiaWorld(a.orient, cdBodyInvInertia(a, a.posInvMass.w), L);
    }
    if (B != CD_STATIC && asleep[B] == 0u) {
        CoinBody b = coins[B];
        coins[B].angVel.xyz = b.angVel.xyz - cdApplyInvInertiaWorld(b.orient, cdBodyInvInertia(b, b.posInvMass.w), L);
    }
}

// `entry` is a colorContacts entry: a grouped manifold's points are applied together.
static inline void cdWarmStartApplyContact(uint entry, device CoinBody* coins, device CoinContact* contacts,
                                           device const uint* asleep) {
    uint cid0 = entry & CD_CC_MASK, nPts = max(entry >> CD_CC_SHIFT, 1u);
    if (nPts == 1u) { cdWarmStartApplyOne(cid0, coins, contacts, asleep); cdWarmStartTorsionOne(cid0, coins, contacts, asleep); return; }
    for (uint k = 0u; k < nPts; ++k) { cdWarmStartApplyOne(cid0 + k, coins, contacts, asleep); cdWarmStartTorsionOne(cid0 + k, coins, contacts, asleep); }
}

kernel void coinWarmStartApply(
    device CoinBody*          coins         [[ buffer(0) ]],
    device CoinContact*       contacts      [[ buffer(1) ]],
    device const uint*        colorContacts [[ buffer(2) ]],
    device const uint*        colorOffset   [[ buffer(4) ]],
    constant uint&            currentColor  [[ buffer(5) ]],
    device const uint*        asleep        [[ buffer(6) ]],
    uint i [[ thread_position_in_grid ]])
{
    uint start = colorOffset[currentColor], end = colorOffset[currentColor + 1u];
    if (start + i >= end) return;
    cdWarmStartApplyContact(colorContacts[start + i], coins, contacts, asleep);
}

// Serial tail of the warm-start sweep — coinSolveVelocityTail's twin (VZ-0154).
static void cdWarmStartApplyTailTG(uint firstColor, uint tid, uint tgSize,
    device CoinBody*          coins,
    device CoinContact*       contacts,
    device const uint*        colorContacts,
    device const uint*        colorOffset,
    device const uint*        asleep,
    device const uint*        uncolSub)
{
    uint first = min(firstColor, CD_UNCOLORED_BUCKET);
    if (colorOffset[first] >= colorOffset[CD_UNCOLORED_BUCKET + 1u]) return;
    uint b0 = colorOffset[CD_UNCOLORED_BUCKET], b1 = colorOffset[CD_UNCOLORED_BUCKET + 1u];
    uint nSub = colorOffset[CD_UNCOLORED_BUCKET + 2u];
    uint nColourPasses = CD_UNCOLORED_BUCKET - first;
    uint nPasses = nColourPasses + (b1 > b0 ? nSub : 0u);   // nSub 0: bucket left unsolved
    for (uint pass = 0u; pass < nPasses; ++pass) {
        uint start, end, sub = 0xFFFFFFFFu;
        if (pass < nColourPasses) {
            start = colorOffset[first + pass];
            if (start >= b0) { pass = nColourPasses - 1u; continue; }   // the colours above are empty (see cdSolveVelocityTailTG)
            end = colorOffset[first + pass + 1u];
        } else { start = b0; end = b1; sub = pass - nColourPasses; }
        if (start >= end) continue;
        for (uint k = start + tid; k < end; k += tgSize)
            if (sub == 0xFFFFFFFFu || uncolSub[k] == sub) cdWarmStartApplyContact(colorContacts[k], coins, contacts, asleep);
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void coinWarmStartApplyTail(
    device CoinBody*          coins         [[ buffer(0) ]],
    device CoinContact*       contacts      [[ buffer(1) ]],
    device const uint*        colorContacts [[ buffer(2) ]],
    device const uint*        colorOffset   [[ buffer(4) ]],
    constant uint&            firstColor    [[ buffer(5) ]],
    device const uint*        asleep        [[ buffer(6) ]],
    device const uint*        uncolSub      [[ buffer(10) ]],
    uint tid    [[ thread_index_in_threadgroup ]],
    uint tgSize [[ threads_per_threadgroup ]])
{
    cdWarmStartApplyTailTG(firstColor, tid, tgSize, coins, contacts, colorContacts, colorOffset, asleep, uncolSub);
}

// A body's bounding radius about its COM for the surface-speed tests: prevPos.w is
// the bounding radius (box, hull, egg) or the round radius (sphere, disc); vel.w is
// a capsule's full half-height (hl + r) — so the larger of the two bounds every shape.
static float cdBoundRadiusOf(CoinBody c) { return max(cdRadiusOf(c), cdHalfThickOf(c)); }

static inline void cdFinalizeBody(uint id, device CoinBody* coins, constant CoinUniforms& u,
                                  device const uint* asleep, device const float4* bias)
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
    //  • Never a body a motor is actively driving (the joint pass marks it): a
    //    commanded slew slower than the threshold is motion, not creep (VZ-0149).
    //  • Legacy test (default): |v| < sleepLinVel && |ω| < 0.6 rad/s — the angular
    //    part is absolute, so ANY body turning about a still COM slower than
    //    0.6 rad/s is zeroed (a 0.52 m jib at 0.4 rad/s: a 0.1 m/s tip speed).
    //  • CD_FLAG_SCALE_AWARE_DEADSTOP: the body's fastest surface speed,
    //    |v| + |ω|·R_bound < sleepLinVel — one length-scaled threshold for every size.
    bool motorDriven = bias[2*id+1].w > 0.5;
    bool slow = ((u.solverFlags & CD_FLAG_SCALE_AWARE_DEADSTOP) != 0u)
              ? (length(v) + length(w) * cdBoundRadiusOf(c) < u.sleepLinVel)
              : (length(v) < u.sleepLinVel && length(w) < 0.6);
    if (slow && !motorDriven) { v = float3(0.0); w = float3(0.0); }

    coins[id].vel.xyz = v;
    coins[id].angVel.xyz = w;
}

kernel void coinFinalizeCS(
    device CoinBody*       coins  [[ buffer(0) ]],
    constant CoinUniforms& u      [[ buffer(1) ]],
    device const uint*     asleep [[ buffer(2) ]],
    device const float4*   bias   [[ buffer(3) ]],   // .w of [2*id+1] = motor-driven marker
    uint id [[ thread_position_in_grid ]])
{
    cdFinalizeBody(id, coins, u, asleep, bias);
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTRAINT SOLVER — Stage 7: generic joints, BLOCK-solved and warm-started
// ══════════════════════════════════════════════════════════════════════════════
//
// Bilateral constraints between two dynamic bodies (or a body and the WORLD), solved
// at velocity level like contacts: real velocity toward the constraint manifold,
// position drift recovered through the SAME split-impulse bias velocities (never mixed
// into real velocity, so a settled articulated assembly carries no restoring energy).
//
// Types (meta.x): 0 = BALL (anchors coincide, 3 rows), 1 = HINGE (ball + the 2 axis-
// alignment rows; optional motor / angle limits about the axis), 2 = DISTANCE (anchor
// separation = rest length, 1 row), 3 = PRISMATIC (the 2 translation rows ⟂ the axis +
// all 3 rotation rows; optional motor / travel limits along the axis), 4 = WELD (ball +
// all 3 rotation rows).
//
// WHY A BLOCK (VZ-0150). A joint's rows are coupled through the lever arm r from each
// COM to the anchor: an impulse on the ball core turns the body (I⁻¹·(r × P)), which
// changes the angular rows' error, whose impulse moves the anchor again. Solving the
// rows one after another (ball core, then each angular row — the pre-VZ-0150 kernel)
// is a block Gauss–Seidel whose contraction per pass is about 1 − I_com/I_pivot: the
// §A1 jib (COM 0.17 m off its hinge, I_com/I_pivot = 0.44) kept 69.9 / 11.8 /
// 1.77 mm/s of residual velocity at 1 / 3 / 6 iterations and never slept, and the §C1
// crane chain pitched its plunger rail 26° and ran its wrist into the stop at 6. Each
// joint here is ONE linear system over all of its rows, K·Δλ = −(J·v − target) with
// K = J·M⁻¹·Jᵀ (the full lever coupling through both bodies' mass and world inertia),
// factored LDLᵀ once per substep (K changes only when poses do) — exact for a single
// joint in one pass. The joint's motor (or, near a stop, its limit) is the block's last
// row, a clamped one: solve all rows; if the free row's ACCUMULATED impulse leaves its
// bounds, pin it at the bound and re-solve the equality rows with the leading factor
// (exact for one bounded row). A braked hinge is then a 6-row block — a weld while the
// brake holds — and a lever-loaded servo holds any load within its rating.
//
// WHY WARM START. The accumulated impulse of every row is kept per joint slot across
// substeps (as world impulse vectors, re-projected onto the new rows each substep) and
// applied before the first iteration, so a static load is carried from the first pass
// and the iterations only correct what changed. Joint-to-joint coupling in a chain is
// still Gauss–Seidel (serial, in list order), but at rest the warm-started impulses are
// already the solution.
//
// Pipeline per substep: coinJointPrepare (parallel: rows, K, LDLᵀ, re-projected
// impulses; then one thread applies the warm start) → per velocity iteration
// coinJointSolveCS (one thread, serial over the ACTIVE list, `passes` times).
// The active list (VZ-0150 item 4c) is uploaded once per frame by coinJointListUpload
// from the host's scan of enabled slots, so a disabled pool slot is never read by the
// joint kernels, the contact generator's collideConnected scan or the island union.
// (CoinJoint itself is declared earlier, alongside CoinContact.)

//
// Iteration cost. A single GPU thread is ALU-latency bound, so everything that depends
// only on poses is done once per substep, in parallel over joints, by coinJointPrepare:
// with y = [dv; dw] the joint's relative velocity (dv = (vA + ωA×rA) − (vB + ωB×rB) at
// the constraint point, dw = ωB − ωA) and z = [P; L] its generalized impulse (P linear
// at the point, + on A / − on B; L angular, − on A / + on B), a joint's rows are
// u = G·y, and the exact block solve x = K⁻¹(t − G·y), K = G·M̃·Gᵀ, applies
//     z = Gᵀx = c − W·y,     W = Gᵀ·K⁻¹·G (6×6, symmetric),  c = Gᵀ·K⁻¹·t.
// An iteration is then one 6×6 mat-vec for the real velocities and one for the bias
// velocities (plus, when the free row's accumulated impulse leaves its bounds, the
// equality-only system W_e with the free impulse pinned: z = −W_e·y + h·Δλ).

// (type 0 = BALL: the ball-core branch, no constant needed)
constant uint CD_JOINT_HINGE     = 1u;
constant uint CD_JOINT_DISTANCE  = 2u;
constant uint CD_JOINT_PRISMATIC = 3u;
constant uint CD_JOINT_WELD      = 4u;
constant uint CD_JLIST_RESET     = 0x80000000u;   // list entry flag: (re)activated slot, cold start

// Per-slot block data, rebuilt each substep (private GPU buffer). A symmetric 6×6 is
// packed as its upper triangle row by row (00 01 02 03 04 05 11 12 13 14 15 22 23 24 25
// 33 34 35 44 45 55), then 6 extra values, then a spare: 7 float4.
struct CoinJointPrep {
    float4 hdr;      // x = #equality rows, y = free row (0 none, 1 motor, 2 limit), z = motor pre-row (0/1), w = active this substep (0/1)
    float4 mass;     // x = A, y = B (uint bits), z = invMa, w = invMb (0 = an immovable end: asleep or world)
    float4 rA;       // xyz = A's lever arm to the constraint point, w = bias uses the full system (0/1)
    float4 rB;       // xyz = B's lever arm,                           w = motor-driven marker (0/1)
    float4 iA0, iA1; // A's world inverse inertia (xx, yy, zz, xy), (xz, yz, −, −)
    float4 iB0, iB1; // B's
    float4 acc0;     // xyz = accumulated linear impulse of the equality rows (world), w = motor impulse
    float4 acc1;     // xyz = accumulated angular impulse of the equality rows,         w = limit impulse
    float4 gF0;      // xyz = free row's linear part (signed), w = (K⁻¹t)₅
    float4 gF1;      // xyz = free row's angular part,         w = 1/K of the motor pre-row
    float4 rF0;      // xyz = linear part of (K⁻¹G)₅,          w = lower bound on the free row's accumulated impulse
    float4 rF1;      // xyz = angular part,                     w = upper bound
    float4 mCfg;     // x = motor target, y = motor bound (max·dt), z = limit side (±1; the free row's sign), w = spare
    float4 cb0, cb1; // bias constant c_b (linear / angular)
    float4 W[7];     // full system W (+ c in the 6 extra lanes)
    float4 We[7];    // equality rows only W_e (+ h in the 6 extra lanes)
};
static_assert(sizeof(CoinJointPrep) == 496, "CoinJointPrep must match CoinDEMSolver.jointPrepStride");

// LDLᵀ of the leading n×n block of the symmetric 6×6 K (lower triangle read). A pivot that has
// lost all but 1e-7 of its diagonal is a dependent row: it is dropped (invD 0, L column 0).
//
// Everything here is FULLY UNROLLED over the six row slots (`#pragma unroll`, constant trip
// counts), so every array index is a compile-time constant and the arrays live in registers.
// Stage B3 (VZ-0169): the rolled loops before indexed K / L / D / Y in threadgroup scratch with
// runtime indices and kept nine private arrays (gl, ga, pw, pe, c, h, r5, sb, cb) as stack
// allocas — 157 basic blocks, 139 φ-nodes, every step a dependent memory round trip on ONE
// thread: ≈ 86 µs to prepare a single hinge (the one-joint pendulum, per substep), the joint
// prepare's whole critical path. `n` (the row count) stays a runtime value: a slot ≥ n is
// skipped by a uniform predicate, as before.
#define CD_LP(i, k) ((i) * ((i) - 1) / 2 + (k))
static inline void cdLDLFactorReg(thread const float (&K)[6][6], int n, thread float (&Lp)[15],
                                  thread float (&invD)[6], thread float (&D)[6]) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) D[i] = 0.0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 15; ++i) Lp[i] = 0.0;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 6; ++j) {
        invD[j] = 0.0;
        if (j >= n) continue;
        float kjj = K[j][j];
        float d = kjj;
        #pragma clang loop unroll(full)
        for (int k = 0; k < 5; ++k) { if (k < j) { float l = Lp[CD_LP(j, k)]; d -= l * l * D[k]; } }
        bool ok = kjj > 0.0 && d > 1e-7 * kjj;
        D[j] = ok ? d : 0.0;
        invD[j] = ok ? 1.0 / d : 0.0;
        #pragma clang loop unroll(full)
        for (int i = 1; i < 6; ++i) {
            if (i <= j || i >= n) continue;
            float s = K[i][j];
            #pragma clang loop unroll(full)
            for (int k = 0; k < 5; ++k) { if (k < j) s -= Lp[CD_LP(i, k)] * Lp[CD_LP(j, k)] * D[k]; }
            Lp[CD_LP(i, j)] = ok ? s * invD[j] : 0.0;
        }
    }
}

// Symmetric 3×3 (xx, yy, zz, xy), (xz, yz) times v.
static inline float3 cdSymMul(float4 i0, float4 i1, float3 v) {
    return float3(i0.x * v.x + i0.w * v.y + i1.x * v.z,
                  i0.w * v.x + i0.y * v.y + i1.y * v.z,
                  i1.x * v.x + i1.y * v.y + i0.z * v.z);
}
// World inverse inertia R·diag(d)·Rᵀ of a body at orientation q.
static void cdWorldInvInertia(float4 q, float3 d, thread float4& i0, thread float4& i1) {
    float3x3 R = cdQuatToMat3(q);   // columns = local axes in world
    float3 r0 = float3(R[0][0], R[1][0], R[2][0]);
    float3 r1 = float3(R[0][1], R[1][1], R[2][1]);
    float3 r2 = float3(R[0][2], R[1][2], R[2][2]);
    i0 = float4(dot(r0 * d, r0), dot(r1 * d, r1), dot(r2 * d, r2), dot(r0 * d, r1));
    i1 = float4(dot(r0 * d, r2), dot(r1 * d, r2), 0.0, 0.0);
}
// Apply a generalized impulse (P linear at the constraint point, + on A / − on B; Lg
// angular, − on A / + on B).
static inline void cdJApply(float3 P, float3 Lg, float3 rA, float3 rB, float invMa, float invMb,
                            float4 iA0, float4 iA1, float4 iB0, float4 iB1,
                            thread float3& vA, thread float3& wA, thread float3& vB, thread float3& wB) {
    vA += invMa * P; wA += cdSymMul(iA0, iA1, cross(rA, P) - Lg);
    vB -= invMb * P; wB += cdSymMul(iB0, iB1, Lg - cross(rB, P));
}
// −W·y for a packed symmetric 6×6 (y = [yl; ya]); the result as (linear, angular).
static inline void cdSym6NegMul(float4 m0, float4 m1, float4 m2, float4 m3, float4 m4, float4 m5,
                                float3 yl, float3 ya, thread float3& zl, thread float3& za) {
    // m0 = 00 01 02 03 | m1 = 04 05 11 12 | m2 = 13 14 15 22 | m3 = 23 24 25 33 | m4 = 34 35 44 45 | m5.x = 55
    zl.x = -(m0.x * yl.x + m0.y * yl.y + m0.z * yl.z + m0.w * ya.x + m1.x * ya.y + m1.y * ya.z);
    zl.y = -(m0.y * yl.x + m1.z * yl.y + m1.w * yl.z + m2.x * ya.x + m2.y * ya.y + m2.z * ya.z);
    zl.z = -(m0.z * yl.x + m1.w * yl.y + m2.w * yl.z + m3.x * ya.x + m3.y * ya.y + m3.z * ya.z);
    za.x = -(m0.w * yl.x + m2.x * yl.y + m3.x * yl.z + m3.w * ya.x + m4.x * ya.y + m4.y * ya.z);
    za.y = -(m1.x * yl.x + m2.y * yl.y + m3.y * yl.z + m4.x * ya.x + m4.z * ya.y + m4.w * ya.z);
    za.z = -(m1.y * yl.x + m2.z * yl.y + m3.z * yl.z + m4.y * ya.x + m4.w * ya.y + m5.x * ya.z);
}
// Two unit directions spanning the plane ⟂ a (the construction the joint rows have always
// used, so an axis that does not move keeps its basis).
static void cdPerpBasis(float3 a, thread float3& b1, thread float3& b2) {
    b1 = (abs(a.x) > 0.7) ? normalize(float3(-a.y, a.x, 0.0)) : normalize(float3(0.0, -a.z, a.y));
    b2 = cross(a, b1);
}

// Relative orientation of B in A's frame against the one captured at creation (ref):
// qe = (conj(qa)·qb)·conj(ref), sign-canonicalised to w ≥ 0 so it never wraps (a
// rotation and its 2π-offset twin are the same pose). Identity at creation.
static float4 cdJointRelErr(float4 qa, float4 qb, float4 ref) {
    float4 qe = cdQuatMul(cdQuatMul(cdQuatConj(qa), qb), cdQuatConj(ref));
    return (qe.w < 0.0) ? -qe : qe;
}

// Build one joint's block for this substep: rows, K, its LDLᵀ, W / c / W_e / h / c_b,
// and the accumulated impulses projected onto the current rows. Reads bodies (pose +
// velocity), writes only prep[slot].
static void cdJointPrepareOne(uint slot, device const CoinJoint* joints, device const CoinBody* coins,
                              constant CoinUniforms& u, device const uint* asleep,
                              device CoinJointPrep* prep)
{
    CoinJoint jn = joints[slot];
    device CoinJointPrep& P = prep[slot];
    // The previous impulses are world vectors (the list upload zeroes them for a cold start).
    float4 oa0 = P.acc0, oa1 = P.acc1;
    float3 Pw = oa0.xyz, Lw = oa1.xyz;
    float lamM = oa0.w, lamL = oa1.w, sideOld = P.mCfg.z;
    // A DISTANCE joint's swing-friction impulse (VZ-0168; rF1.xyz, a world vector ⟂ the fall).
    // Every other type writes rF1 with its free-row data, a distance joint without friction
    // writes 0, and the list upload zeroes it on a cold start — so a fall starts from 0.
    float3 fricOld = P.rF1.xyz;
    float fricArm = 0.0, fricLen = 1.0;

    uint A = jn.meta.y, B = jn.meta.z;
    bool bWorld = (B == CD_STATIC);
    CoinBody a = coins[A];
    CoinBody b;
    bool live = ((jn.meta.w & 1u) != 0u) && a.posInvMass.w != 0.0;
    if (!bWorld) { b = coins[B]; live = live && b.posInvMass.w != 0.0; }
    bool aSleep = asleep[A] != 0u;
    bool bSleep = bWorld || (asleep[B] != 0u);
    // Nothing to solve: keep the impulses for the next active substep.
    if (!live || (aSleep && bSleep)) { P.hdr.w = 0.0; return; }

    float  invMa = aSleep ? 0.0 : a.posInvMass.w;
    float4 qa = a.orient;
    float3 xa = a.posInvMass.xyz;
    float4 iA0 = float4(0.0), iA1 = float4(0.0), iB0 = float4(0.0), iB1 = float4(0.0);
    if (!aSleep) cdWorldInvInertia(qa, cdBodyInvInertia(a, a.posInvMass.w), iA0, iA1);
    float  invMb = 0.0;
    float4 qb = float4(0.0, 0.0, 0.0, 1.0);
    float3 xb = float3(0.0), vB = float3(0.0), wB = float3(0.0);
    if (!bWorld) {
        invMb = bSleep ? 0.0 : b.posInvMass.w;
        qb = b.orient; xb = b.posInvMass.xyz;
        vB = b.vel.xyz; wB = b.angVel.xyz;
        if (!bSleep) cdWorldInvInertia(qb, cdBodyInvInertia(b, b.posInvMass.w), iB0, iB1);
    }
    float3 vA = a.vel.xyz, wA = a.angVel.xyz;

    float3 pA = xa + cdQuatRotate(qa, jn.anchorA.xyz);
    float3 pB = bWorld ? jn.anchorB.xyz : (xb + cdQuatRotate(qb, jn.anchorB.xyz));
    float3 rA = pA - xa;
    float3 rB = bWorld ? float3(0.0) : (pB - xb);
    float invDt = 1.0 / max(u.dt, 1e-6);
    float beta = u.baumgarteBeta;
    uint type = jn.meta.x;
    const float3 EX = float3(1.0, 0.0, 0.0), EY = float3(0.0, 1.0, 0.0), EZ = float3(0.0, 0.0, 1.0);

    // Rows: gl[i] / ga[i] = the linear / angular part of row i (u_i = gl·dv + ga·dw).
    float3 gl[6], ga[6];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) { gl[i] = float3(0.0); ga[i] = float3(0.0); }
    float biasT[6] = { 0, 0, 0, 0, 0, 0 };
    int   nEq = 0;
    bool  axial = false, freeAngular = false;
    float3 D3 = EZ;                // free axis
    float freePos = 0.0;           // twist (rad, from creation) or slide (m)

    if (type == CD_JOINT_DISTANCE) {
        float3 d = pA - pB;
        float len = length(d);
        if (len < 1e-6) { P.hdr.w = 0.0; return; }
        gl[0] = d / len;
        biasT[0] = -beta * (len - jn.anchorA.w) * invDt;
        nEq = 1;
        fricArm = max(jn.anchorB.w, 0.0);   // swing friction arm (m), VZ-0168; 0 = frictionless
        fricLen = len;
    } else if (type == CD_JOINT_PRISMATIC) {
        float3 aW = normalize(cdQuatRotate(qa, jn.axisA.xyz));
        float3 b1, b2; cdPerpBasis(aW, b1, b2);
        // Coincident-point Jacobian: A's lever arm reaches pB (the point of A's slide line
        // that sits there), so a pair turning rigidly together has zero relative velocity
        // at any slide.
        float3 gap = pA - pB;
        rA = pB - xa;
        float3 C = cdQuatRotate(qa, 2.0 * cdJointRelErr(qa, qb, jn.ref).xyz);
        gl[0] = b1; gl[1] = b2; ga[2] = EX; ga[3] = EY; ga[4] = EZ;
        biasT[0] = -beta * dot(gap, b1) * invDt;
        biasT[1] = -beta * dot(gap, b2) * invDt;
        biasT[2] = -beta * C.x * invDt; biasT[3] = -beta * C.y * invDt; biasT[4] = -beta * C.z * invDt;
        nEq = 5;
        axial = true; freeAngular = false; D3 = aW;
        freePos = dot(gap, aW);
    } else {
        // BALL core (ball, hinge, weld): anchors coincide.
        float3 C = pA - pB;
        gl[0] = EX; gl[1] = EY; gl[2] = EZ;
        biasT[0] = -beta * C.x * invDt; biasT[1] = -beta * C.y * invDt; biasT[2] = -beta * C.z * invDt;
        nEq = 3;
        if (type == CD_JOINT_HINGE) {
            float3 aW = normalize(cdQuatRotate(qa, jn.axisA.xyz));
            float3 bW = bWorld ? normalize(jn.axisB.xyz) : normalize(cdQuatRotate(qb, jn.axisB.xyz));
            float3 b1, b2; cdPerpBasis(aW, b1, b2);
            // axisErr = aW × bW ≈ the rotation of B relative to A ⟂ the axis (for A turned by
            // φ about d, axisErr·d = −φ), so the bias target on the row u = (ωB − ωA)·d is
            // −β·(axisErr·d)/dt — restoring ×(1 − β) per substep (VZ-0147's sign, unchanged).
            float3 axisErr = cross(aW, bW);
            ga[3] = b1; ga[4] = b2;
            biasT[3] = -beta * dot(axisErr, b1) * invDt;
            biasT[4] = -beta * dot(axisErr, b2) * invDt;
            nEq = 5;
            axial = true; freeAngular = true; D3 = aW;
            // Twist of B relative to A about the axis, from the pose at creation (VZ-0156):
            // canonical, so it spans (−π, π] and never jumps by 4π.
            float4 qe = cdJointRelErr(qa, qb, jn.ref);
            freePos = 2.0 * atan2(dot(qe.xyz, normalize(jn.axisA.xyz)), qe.w);
        } else if (type == CD_JOINT_WELD) {
            float3 Cr = cdQuatRotate(qa, 2.0 * cdJointRelErr(qa, qb, jn.ref).xyz);
            ga[3] = EX; ga[4] = EY; ga[5] = EZ;
            biasT[3] = -beta * Cr.x * invDt; biasT[4] = -beta * Cr.y * invDt; biasT[5] = -beta * Cr.z * invDt;
            nEq = 6;
        }
    }

    // ── Free row (row 5 of a hinge / prismatic): motor and/or limits along D3 ────────
    int   freeKind = 0;            // 0 none, 1 motor, 2 limit
    bool  motorPre = false;
    float fTarget = 0.0, fLo = 0.0, fHi = 0.0, fBias = 0.0, fSign = 1.0;
    bool  fBiasOn = false;
    float mTarget = 0.0, mBound = 0.0;
    float driven = 0.0;
    if (axial) {
        float motorTarget = jn.anchorA.w, maxDrive = jn.anchorB.w;
        bool  hasMotor = maxDrive > 0.0;
        float lo = jn.axisA.w, hi = jn.axisB.w;
        bool  hasLimit = lo < hi;
        float3 dv = (vA + cross(wA, rA)) - (vB + cross(wB, rB));
        float uf = freeAngular ? dot(wB - wA, D3) : dot(dv, D3);   // free-axis rate now
        mTarget = motorTarget; mBound = maxDrive * u.dt;
        // Motor-driven marker (VZ-0149): a motor with a non-zero target moves its bodies —
        // unless it is pushing INTO a stop it has reached (holding, not moving).
        if (hasMotor && motorTarget != 0.0) {
            bool stalled = false;
            if (hasLimit) {
                float tol = abs(motorTarget) * u.dt;
                stalled = (motorTarget > 0.0) ? (freePos >= hi - tol) : (freePos <= lo + tol);
            }
            if (!stalled) driven = 1.0;
        }
        bool engaged = false;
        float Cs = 0.0, side = 1.0;
        if (hasLimit) {
            float cLo = freePos - lo, cHi = hi - freePos;
            side = (cLo <= cHi) ? 1.0 : -1.0;
            Cs = min(cLo, cHi);
            // Engage the stop while it is within reach this substep (twice the current rate
            // or the motor's, plus a tolerance) or penetrated. The limit row is speculative:
            // the pair may close the remaining gap exactly and no further.
            float tol = freeAngular ? 0.005 : 1e-4;
            float band = 2.0 * max(abs(uf), abs(motorTarget)) * u.dt + tol;
            engaged = Cs < band;
        }
        if (engaged) {
            freeKind = 2; fSign = side;
            fTarget = (Cs > 0.0) ? -Cs * invDt : 0.0;
            fLo = 0.0; fHi = 3.0e38;                           // one-sided (finite: fast math)
            fBiasOn = Cs < 0.0;
            fBias = fBiasOn ? -beta * Cs * invDt : 0.0;
            lamL = (sideOld == side) ? max(lamL, 0.0) : 0.0;
            motorPre = hasMotor;
            lamM = hasMotor ? clamp(lamM, -mBound, mBound) : 0.0;
        } else {
            lamL = 0.0;
            if (hasMotor) {
                freeKind = 1; fSign = 1.0;
                fTarget = motorTarget; fLo = -mBound; fHi = mBound;
                lamM = clamp(lamM, -mBound, mBound);
            } else {
                lamM = 0.0;
            }
        }
        if (freeKind != 0) {
            gl[5] = freeAngular ? float3(0.0) : fSign * D3;
            ga[5] = freeAngular ? fSign * D3 : float3(0.0);
        }
    } else {
        lamM = 0.0; lamL = 0.0;
    }
    int n = nEq + (freeKind != 0 ? 1 : 0);
    bool hasFree = freeKind != 0;

    // The previous impulses projected onto this substep's equality rows (their directions
    // are orthonormal within the linear and within the angular block).
    float3 Pn = float3(0.0), Ln = float3(0.0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) {
        if (i >= nEq) continue;
        float l = dot(gl[i], Pw) + dot(ga[i], Lw);
        Pn += l * gl[i]; Ln += l * ga[i];
    }

    // K, its factor, Y and the sums below are registers: every loop is fully unrolled over
    // the six row slots, so every index is a compile-time constant (see cdLDLFactorReg).
    //
    // K = G·M̃·Gᵀ: row i's unit impulse → the body response → every row's velocity.
    float K[6][6];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) K[i][j] = 0.0;
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) {
        if (i >= n) continue;
        float3 dvA = float3(0.0), dwA = float3(0.0), dvB = float3(0.0), dwB = float3(0.0);
        cdJApply(gl[i], ga[i], rA, rB, invMa, invMb, iA0, iA1, iB0, iB1, dvA, dwA, dvB, dwB);
        float3 rdv = (dvA + cross(dwA, rA)) - (dvB + cross(dwB, rB));
        float3 rdw = dwB - dwA;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) {
            if (j > i) continue;
            float k = dot(gl[j], rdv) + dot(ga[j], rdw);
            K[i][j] = k; K[j][i] = k;
        }
    }
    float kMotor = K[5][5];
    float Lp[15], invD[6], Dd[6];
    cdLDLFactorReg(K, n, Lp, invD, Dd);

    // Y = L⁻¹·G (forward substitution down each of G's 6 columns; rows ≥ n are 0). Then,
    // with K⁻¹ = L⁻ᵀ·D⁻¹·L⁻¹ and the free row LAST:
    //   W   = Gᵀ·K⁻¹·G      = Σ_{i<n}   Yᵢ·Yᵢᵀ/Dᵢ      W_e = the same sum over i < nEq
    //   c   = Gᵀ·K⁻¹·(t₅e₅) = t₅/D₅·Y₅               (K⁻¹t)₅ = t₅/D₅
    //   (K⁻¹G)₅ = Y₅/D₅                                h = g₅ − G_eᵀK_ee⁻¹K_e5 = Y₅
    float Y[6][6];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) {
        Y[i][0] = gl[i].x; Y[i][1] = gl[i].y; Y[i][2] = gl[i].z;
        Y[i][3] = ga[i].x; Y[i][4] = ga[i].y; Y[i][5] = ga[i].z;
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < 6; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) {
            float s = (i < n) ? Y[i][j] : 0.0;
            #pragma clang loop unroll(full)
            for (int k = 0; k < 5; ++k) { if (k < i && i < n) s -= Lp[CD_LP(i, k)] * Y[k][j]; }
            Y[i][j] = s;
        }
    }
    float pw[21], pe[21];
    #pragma clang loop unroll(full)
    for (int k = 0; k < 6; ++k) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) {
            if (j < k) continue;
            const int idx = k * 6 - k * (k - 1) / 2 + (j - k);   // row-major upper triangle
            float se = 0.0;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 6; ++i) { if (i < nEq) se += invD[i] * Y[i][k] * Y[i][j]; }
            pe[idx] = se;
            pw[idx] = se + (hasFree ? invD[5] * Y[5][k] * Y[5][j] : 0.0);
        }
    }
    float t5 = hasFree ? fTarget : 0.0;
    float c[6], h[6], r5[6];
    #pragma clang loop unroll(full)
    for (int k = 0; k < 6; ++k) {
        float y5k = hasFree ? Y[5][k] : 0.0;
        c[k] = t5 * invD[5] * y5k;
        r5[k] = invD[5] * y5k;
        h[k] = y5k;
    }
    // Bias: the equality rows toward −β·C/dt, plus the limit row while its stop is
    // penetrated (only then is the free row part of the bias system): c_b = Y_bᵀ·D⁻¹·L⁻¹t_b.
    bool bFull = !hasFree || fBiasOn;
    int  nb = bFull ? n : nEq;
    float sb[6] = { biasT[0], biasT[1], biasT[2], biasT[3], biasT[4], hasFree ? fBias : biasT[5] };
    #pragma clang loop unroll(full)
    for (int i = 1; i < 6; ++i) {
        float s = sb[i];
        #pragma clang loop unroll(full)
        for (int k = 0; k < 5; ++k) { if (k < i) s -= Lp[CD_LP(i, k)] * sb[k]; }
        sb[i] = s;
    }
    float cb[6];
    #pragma clang loop unroll(full)
    for (int k = 0; k < 6; ++k) {
        float s = 0.0;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 6; ++i) { if (i < nb) s += invD[i] * Y[i][k] * sb[i]; }
        cb[k] = s;
    }

    // ── Store ────────────────────────────────────────────────────────────────────
    P.hdr = float4(float(nEq), float(freeKind), motorPre ? 1.0 : 0.0, 1.0);
    P.mass = float4(as_type<float>(A), as_type<float>(B), invMa, invMb);
    P.rA = float4(rA, bFull ? 1.0 : 0.0);
    P.rB = float4(rB, driven);
    P.iA0 = iA0; P.iA1 = iA1; P.iB0 = iB0; P.iB1 = iB1;
    P.acc0 = float4(Pn, lamM);
    P.acc1 = float4(Ln, lamL);
    P.gF0 = float4(gl[5], hasFree ? t5 * invD[5] : 0.0);
    P.gF1 = float4(ga[5], (motorPre && kMotor > 0.0) ? 1.0 / kMotor : 0.0);
    P.rF0 = float4(r5[0], r5[1], r5[2], fLo);
    P.rF1 = float4(r5[3], r5[4], r5[5], fHi);
    P.mCfg = float4(mTarget, mBound, fSign, 0.0);
    P.cb0 = float4(cb[0], cb[1], cb[2], 0.0);
    P.cb1 = float4(cb[3], cb[4], cb[5], 0.0);
    P.W[0] = float4(pw[0], pw[1], pw[2], pw[3]);   P.We[0] = float4(pe[0], pe[1], pe[2], pe[3]);
    P.W[1] = float4(pw[4], pw[5], pw[6], pw[7]);   P.We[1] = float4(pe[4], pe[5], pe[6], pe[7]);
    P.W[2] = float4(pw[8], pw[9], pw[10], pw[11]); P.We[2] = float4(pe[8], pe[9], pe[10], pe[11]);
    P.W[3] = float4(pw[12], pw[13], pw[14], pw[15]); P.We[3] = float4(pe[12], pe[13], pe[14], pe[15]);
    P.W[4] = float4(pw[16], pw[17], pw[18], pw[19]); P.We[4] = float4(pe[16], pe[17], pe[18], pe[19]);
    P.W[5] = float4(pw[20], c[0], c[1], c[2]);     P.We[5] = float4(pe[20], h[0], h[1], h[2]);
    P.W[6] = float4(c[3], c[4], c[5], 0.0);        P.We[6] = float4(h[3], h[4], h[5], 0.0);

    // ── Swing friction of a DISTANCE joint (VZ-0168) ─────────────────────────────────
    // A rope fall that runs over sheaves turns them as it swings (a hook block swinging by θ
    // rolls the rope θ round every sheave in its plane), and each sheave's pin resists with a
    // Coulomb moment μ_pin·r_pin·(pin load ≈ 2T). Lumped per fall as M = c·T (c = anchorB.w,
    // the friction ARM, m): a force pair ⟂ the fall at its two anchors of up to M / L = (c/L)·T,
    // i.e. two friction rows on the anchors' relative velocity ⟂ the fall, their ACCUMULATED
    // impulse clamped to a disc of radius (c/L)·|Λ_axial| (Λ_axial the fall's own accumulated
    // tension impulse) — the contact friction law with the rope tension as the normal load.
    // Solved after the block each iteration (cdJointSolveReal); a free (distance) joint has no
    // free row, so the free-row lanes carry it: gF0 = (fall direction, c/L), rF0.xyz = e1,
    // gF1 = (K_t⁻¹: 11, 12, 22, on-flag), rF1.xyz = the accumulated impulse (world, ⟂ the fall).
    if (type == CD_JOINT_DISTANCE && fricArm > 0.0) {
        float3 dU = gl[0];
        float3 e1, e2; cdPerpBasis(dU, e1, e2);
        float k11, k12, k22;
        {
            float3 dvA = float3(0.0), dwA = float3(0.0), dvB = float3(0.0), dwB = float3(0.0);
            cdJApply(e1, float3(0.0), rA, rB, invMa, invMb, iA0, iA1, iB0, iB1, dvA, dwA, dvB, dwB);
            float3 rdv = (dvA + cross(dwA, rA)) - (dvB + cross(dwB, rB));
            k11 = dot(e1, rdv); k12 = dot(e2, rdv);
        }
        {
            float3 dvA = float3(0.0), dwA = float3(0.0), dvB = float3(0.0), dwB = float3(0.0);
            cdJApply(e2, float3(0.0), rA, rB, invMa, invMb, iA0, iA1, iB0, iB1, dvA, dwA, dvB, dwB);
            float3 rdv = (dvA + cross(dwA, rA)) - (dvB + cross(dwB, rB));
            k22 = dot(e2, rdv);
        }
        float det = k11 * k22 - k12 * k12;
        bool ok = det > 1e-12 * max(k11 * k22, 1e-30);
        float muEff = fricArm / fricLen;
        // Warm start: last substep's impulse, re-projected ⟂ this substep's fall and kept inside
        // the disc of the fall's warm-started tension.
        float3 Pf = fricOld - dot(fricOld, dU) * dU;
        float cap = muEff * abs(dot(Pn, dU));
        float pl = length(Pf);
        if (pl > cap) Pf *= cap / max(pl, 1e-30);
        P.gF0 = float4(dU, muEff);
        P.rF0 = float4(e1, 0.0);
        P.gF1 = ok ? float4(k22 / det, -k12 / det, k11 / det, 1.0) : float4(0.0);
        P.rF1 = float4(ok ? Pf : float3(0.0), 0.0);
    }
}

// Warm start one joint: its accumulated impulses onto the REAL velocities of its movable
// ends, and its motor-driven marker.
static void cdJointWarmStartOne(uint slot, device CoinBody* coins, device float4* bias,
                                device const CoinJointPrep* prep)
{
    device const CoinJointPrep& P = prep[slot];
    float4 hdr = P.hdr;
    if (hdr.w < 0.5) return;
    float4 mass = P.mass, rA4 = P.rA, rB4 = P.rB, iA0 = P.iA0, iA1 = P.iA1, iB0 = P.iB0, iB1 = P.iB1;
    float4 a0 = P.acc0, a1 = P.acc1, g0 = P.gF0, g1 = P.gF1, mc = P.mCfg;
    uint A = as_type<uint>(mass.x), B = as_type<uint>(mass.y);
    bool bWorld = (B == CD_STATIC);
    int fk = int(hdr.y);
    float3 Pi = a0.xyz, Li = a1.xyz;
    if (fk != 0) { float l = (fk == 1) ? a0.w : a1.w; Pi += l * g0.xyz; Li += l * g1.xyz; }
    if (hdr.z > 0.5) { float l = a0.w * mc.z; Pi += l * g0.xyz; Li += l * g1.xyz; }   // motor pre-row (unsigned)
    if (hdr.x < 1.5 && g1.w > 0.5) Pi += P.rF1.xyz;   // a DISTANCE joint's swing friction (VZ-0168), at the anchors
    float3 vA = coins[A].vel.xyz, wA = coins[A].angVel.xyz;
    float3 vB = float3(0.0), wB = float3(0.0);
    if (!bWorld) { vB = coins[B].vel.xyz; wB = coins[B].angVel.xyz; }
    cdJApply(Pi, Li, rA4.xyz, rB4.xyz, mass.z, mass.w, iA0, iA1, iB0, iB1, vA, wA, vB, wB);
    bool driven = rB4.w > 0.5;
    if (mass.z > 0.0) {
        coins[A].vel.xyz = vA; coins[A].angVel.xyz = wA;
        if (driven) bias[2*A+1].w = 1.0;
    }
    if (!bWorld && mass.w > 0.0) {
        coins[B].vel.xyz = vB; coins[B].angVel.xyz = wB;
        if (driven) bias[2*B+1].w = 1.0;
    }
}

// Per substep, ONE threadgroup: every thread prepares a share of the active joints in parallel
// (independent per slot, all in registers — cdJointPrepareOne), then thread 0 applies the warm
// starts serially (joints share bodies).
// The joints ACTIVE this substep (the prepare's hdr.w — both ends live, not both asleep), in list
// order: list[CD_JLIST_ACTIVE + i] = the list index of the i-th, list[CD_JLIST_NACTIVE] = how many.
// Written by the prepare's serial warm-start pass, read by every velocity iteration's solve, which
// then stages, walks and writes back only those — the Digital Clock's 21 latch welds hold bars
// that are asleep for the whole job, and their blocks were staged, walked and written back every
// iteration for no work. Measured on the clock world (stage B3): the saving is small — within the
// run-to-run noise of the joint solve (≤ 0.15 ms of a 1/180 s × 6 frame); the solve's cost is the
// ACTIVE joints' serial Gauss–Seidel walk. The Gauss–Seidel order of the active joints is the list
// order either way, so the result is unchanged (bit-identical on every parity world).
constant uint CD_JLIST_ACTIVE  = 3072u;
constant uint CD_JLIST_NACTIVE = 4096u;

// The prepare pass for one threadgroup: the active joints in parallel, then their warm starts.
//
// PARALLEL PREPARE. Joint k (list order) runs on SIMD group k mod G, lane k / G (G = the group's
// SIMD groups): the first G joints each get a SIMD group of their own, the next G share them as
// second lanes, and so on — so any frame with ≤ G·w joints prepares them in ONE round. The cost
// model behind it (stage B3, measured on an M1 Max): the prepare is ~1 500 instructions of
// straight-line code (cdJointPrepareOne, fully unrolled), executed once per substep, so it runs
// COLD out of the instruction cache — ≈ 12–22 ns per instruction against ≈ 2–4 ns warm (a
// straight-line probe: 512 instructions warm 1.8 ns, 1 024+ never warm) — and the cost of a
// SIMD group's stream does not grow with its active lanes (one fetch serves all 32) while the
// groups fetch the same code concurrently. What costs is a group running a SECOND joint after the
// first: the Digital Clock crane's 9 active joints on 8 groups (one lane each) left group 0
// preparing two, back to back — ≈ 25 µs per substep of the ≈ 52.
//
// WARM STARTS. Serial on one thread in list order (joints share bodies; the order of their adds
// is the result's), but only over the ACTIVE joints: SIMD group 0 reads 32 joints' active flags
// at once (a ballot), writes their active-list entries in parallel, and its first lane applies
// the set ones in order. It used to walk every listed joint — the clock's 21 latch welds hold
// asleep bars — paying two dependent device loads (list → block header) per inactive joint on
// the serial path.
//
// Every thread of the group must call it (one barrier). `tid` / `tgs` as in the kernel; `w` the
// SIMD width.
static void cdJointPrepareTG(uint tid, uint tgs, uint w,
    device CoinBody*        coins,
    device float4*          bias,
    device const CoinJoint* joints,
    device uint*            list,
    uint                    count,
    constant CoinUniforms&  u,
    device const uint*      asleep,
    device CoinJointPrep*   prep)
{
    uint W = max(w, 1u), nsg = max(1u, tgs / W), sg = tid / W, lane = tid % W;
    if (sg < nsg)
        for (uint k = sg + nsg * lane; k < count; k += nsg * W)
            cdJointPrepareOne(list[k], joints, coins, u, asleep, prep);
    threadgroup_barrier(mem_flags::mem_device);
    if (sg == 0u) {
        uint nActive = 0u;
        for (uint k0 = 0u; k0 < count; k0 += W) {             // uniform across the SIMD group
            uint k = k0 + lane;
            bool act = (k < count) && prep[list[k]].hdr.w > 0.5;
            ulong m = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(act)));   // lane l → bit l (W ≤ 64)
            if (act) list[CD_JLIST_ACTIVE + nActive + uint(popcount(m & ((1ul << lane) - 1ul)))] = k;
            if (lane == 0u)
                for (ulong mm = m; mm != 0ul; mm &= mm - 1ul)
                    cdJointWarmStartOne(list[k0 + uint(ctz(mm))], coins, bias, prep);
            nActive += uint(popcount(m));
        }
        if (lane == 0u) list[CD_JLIST_NACTIVE] = nActive;
    }
}

kernel void coinJointPrepare(
    device CoinBody*        coins  [[ buffer(0) ]],
    device float4*          bias   [[ buffer(1) ]],
    device const CoinJoint* joints [[ buffer(2) ]],
    device uint*            list   [[ buffer(3) ]],   // + the substep's active list (CD_JLIST_ACTIVE)
    constant uint&          count  [[ buffer(4) ]],
    constant CoinUniforms&  u      [[ buffer(5) ]],
    device const uint*      asleep [[ buffer(6) ]],
    device CoinJointPrep*   prep   [[ buffer(7) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]],
    uint w   [[ threads_per_simdgroup ]])
{
    cdJointPrepareTG(tid, tgs, w, coins, bias, joints, list, count, u, asleep, prep);
}

// One joint's velocity-iteration solve on register values: an exact block solve of its
// real rows (the free row clamped on its accumulated impulse) and then of its bias rows —
// two 6×6 mat-vecs from the substep's precomputed W / c (see the header above). `P` is
// the joint's block (device or threadgroup memory); the body velocities and the two
// accumulators are the caller's.
//
// TWO INDEPENDENT CHAINS (stage B3, VZ-0169). The real rows (with the motor pre-row and a
// fall's swing friction) read and write only the REAL velocities and the accumulators; the
// bias rows (split impulse) read and write only the BIAS velocities and read the block. So
// the serial Gauss–Seidel walk over the joints is two walks that never exchange a value, and
// cdJointSolveTG runs them on two SIMD groups at once — each chain the same instructions in
// the same joint order as when one thread interleaved them, so the result is unchanged. (One
// SIMD group on a core issues ≈ one instruction per 4 ns and waits ≈ 16 ns on a dependent
// one — measured on an M1 Max — so the joint walk is instruction-bound: halving the stream
// on the critical SIMD group is what makes it faster.)
struct CdJBodies { float3 vA, wA, bvA, bwA, vB, wB, bvB, bwB; };

template <typename PrepPtr>
static inline void cdJointSolveReal(PrepPtr P, thread CdJBodies& b, thread float4& a0, thread float4& a1)
{
    float4 hdr = P->hdr, mass = P->mass, rA4 = P->rA, rB4 = P->rB;
    float4 iA0 = P->iA0, iA1 = P->iA1, iB0 = P->iB0, iB1 = P->iB1;
    float4 w0 = P->W[0], w1 = P->W[1], w2 = P->W[2], w3 = P->W[3], w4 = P->W[4], w5 = P->W[5], w6 = P->W[6];
    int fk = int(hdr.y);
    float4 g0 = float4(0.0), g1 = float4(0.0), f0 = float4(0.0), f1 = float4(0.0), mc = float4(0.0);
    if (fk != 0) { g0 = P->gF0; g1 = P->gF1; f0 = P->rF0; f1 = P->rF1; mc = P->mCfg; }
    float invMa = mass.z, invMb = mass.w;
    float3 rA = rA4.xyz, rB = rB4.xyz;

    // (0) The motor as a scalar row ahead of the block (only while at a stop): along
    // the free axis, unsigned.
    if (hdr.z > 0.5) {
        float3 ml = mc.z * g0.xyz, ma = mc.z * g1.xyz;
        float3 dv = (b.vA + cross(b.wA, rA)) - (b.vB + cross(b.wB, rB));
        float um = dot(ml, dv) + dot(ma, b.wB - b.wA);
        float accOld = a0.w;
        float accNew = clamp(accOld + (mc.x - um) * g1.w, -mc.y, mc.y);
        a0.w = accNew;
        cdJApply((accNew - accOld) * ml, (accNew - accOld) * ma, rA, rB, invMa, invMb,
                 iA0, iA1, iB0, iB1, b.vA, b.wA, b.vB, b.wB);
    }

    // (1) Real rows: z = c − W·y — or, when the free row's step would leave its bounds, the free
    // row pinned at the bound and the equality rows absorbing the rest exactly: z = −W_e·y + h·Δλ.
    // The free row's step x5 = t₅/D₅ − (K⁻¹G)₅·y does not depend on the block's product, so it is
    // decided first and ONE of the two products is formed (the same values as forming W's and then
    // replacing it); W_e is read only on that rare branch, so its 28 floats are never live across
    // the rest of the chain (the serial walk runs on one thread: its registers are the budget).
    {
        float3 dv = (b.vA + cross(b.wA, rA)) - (b.vB + cross(b.wB, rB));
        float3 dw = b.wB - b.wA;
        float x5 = 0.0, got = 0.0;
        bool pinned = false;
        if (fk != 0) {
            x5 = g0.w - (dot(f0.xyz, dv) + dot(f1.xyz, dw));
            float accF = (fk == 1) ? a0.w : a1.w;
            float want = accF + x5;
            got = clamp(want, f0.w, f1.w);
            pinned = got != want;
            if (pinned) x5 = got - accF;
        }
        float3 zl, za;
        if (!pinned) {
            cdSym6NegMul(w0, w1, w2, w3, w4, w5, dv, dw, zl, za);
            zl += float3(w5.y, w5.z, w5.w); za += w6.xyz;
        } else {
            float4 e0 = P->We[0], e1 = P->We[1], e2 = P->We[2], e3 = P->We[3], e4 = P->We[4], e5 = P->We[5], e6 = P->We[6];
            cdSym6NegMul(e0, e1, e2, e3, e4, e5, dv, dw, zl, za);
            zl += x5 * float3(e5.y, e5.z, e5.w); za += x5 * e6.xyz;
        }
        float3 El = zl, Ea = za;                            // the equality rows' share
        if (fk != 0) {
            if (fk == 1) a0.w = got; else a1.w = got;
            El = zl - x5 * g0.xyz; Ea = za - x5 * g1.xyz;
        }
        a0.xyz += El; a1.xyz += Ea;
        cdJApply(zl, za, rA, rB, invMa, invMb, iA0, iA1, iB0, iB1, b.vA, b.wA, b.vB, b.wB);
    }

    // (1b) A DISTANCE joint's swing friction (VZ-0168; see the prepare): the anchors' relative
    // velocity ⟂ the fall driven to 0, the accumulated impulse inside the disc of radius
    // (c/L)·|Λ_axial| — Λ_axial the tension impulse the block has just accumulated.
    if (hdr.x < 1.5 && P->gF1.w > 0.5) {
        float4 dm = P->gF0, kk = P->gF1;
        float3 e1 = P->rF0.xyz, e2 = cross(dm.xyz, e1);
        float3 dv = (b.vA + cross(b.wA, rA)) - (b.vB + cross(b.wB, rB));
        float2 ut = float2(dot(e1, dv), dot(e2, dv));
        float3 Pold = P->rF1.xyz;
        float2 ln = float2(dot(e1, Pold), dot(e2, Pold)) - float2(kk.x * ut.x + kk.y * ut.y, kk.y * ut.x + kk.z * ut.y);
        float cap = dm.w * abs(dot(a0.xyz, dm.xyz));
        float l = length(ln);
        if (l > cap) ln *= cap / max(l, 1e-30);
        float3 Pnew = ln.x * e1 + ln.y * e2;
        cdJApply(Pnew - Pold, float3(0.0), rA, rB, invMa, invMb, iA0, iA1, iB0, iB1, b.vA, b.wA, b.vB, b.wB);
        P->rF1 = float4(Pnew, 0.0);
    }
}

// (2) Bias rows (split impulse): z_b = c_b − W_b·y_b. Not accumulated. W_b is the full W
// unless the joint has a free row whose stop is not penetrated (then the equality rows' W_e).
template <typename PrepPtr>
static inline void cdJointSolveBias(PrepPtr P, thread CdJBodies& b)
{
    float4 hdr = P->hdr, mass = P->mass, rA4 = P->rA, rB4 = P->rB;
    float4 iA0 = P->iA0, iA1 = P->iA1, iB0 = P->iB0, iB1 = P->iB1, cb0 = P->cb0, cb1 = P->cb1;
    int fk = int(hdr.y);
    bool bFull = rA4.w > 0.5;
    float4 m0, m1, m2, m3, m4, m5;
    if (bFull || fk == 0) { m0 = P->W[0];  m1 = P->W[1];  m2 = P->W[2];  m3 = P->W[3];  m4 = P->W[4];  m5 = P->W[5]; }
    else                  { m0 = P->We[0]; m1 = P->We[1]; m2 = P->We[2]; m3 = P->We[3]; m4 = P->We[4]; m5 = P->We[5]; }
    float invMa = mass.z, invMb = mass.w;
    float3 rA = rA4.xyz, rB = rB4.xyz;
    float3 dv = (b.bvA + cross(b.bwA, rA)) - (b.bvB + cross(b.bwB, rB));
    float3 dw = b.bwB - b.bwA;
    float3 zl, za;
    cdSym6NegMul(m0, m1, m2, m3, m4, m5, dv, dw, zl, za);
    zl += cb0.xyz; za += cb1.xyz;
    cdJApply(zl, za, rA, rB, invMa, invMb, iA0, iA1, iB0, iB1, b.bvA, b.bwA, b.bvB, b.bwB);
}

// Threadgroup-cached solve: when the frame's active joints and the distinct bodies they
// join fit (the host decides; see CoinDEMSolver.encodeJointListUpload), every thread of
// the group stages a share of the joints' blocks and of the bodies' velocities into
// threadgroup memory, thread 0 runs the serial Gauss–Seidel entirely out of it, and the
// group writes the results back. The serial chain then never waits on device memory: it
// used to pay three dependent device-memory latencies per joint (list → block → bodies)
// plus the stores' ordering before the next joint's loads (~3 µs a joint-iteration).
// The staged body tables come from the host's scan at ENCODE; the blocks from the joint
// table as coinJointPrepare read it. If they disagree (a slot re-bound to other bodies
// while the frame was in flight), the pass falls back to the direct path, which takes the
// bodies from the block, so a joint's impulse never lands on bodies it does not join.
constant uint CD_JSOLVE_MAXJ = 40u;        // CoinDEMSolver.jointTGMaxJoints
constant uint CD_JSOLVE_MAXB = 80u;        // CoinDEMSolver.jointTGMaxBodies
constant uint CD_JLIST_PAIRS = 1024u;      // list[1024 + k] = joint k's (local A | local B << 16)
constant uint CD_JLIST_BODIES = 2048u;     // list[2048 + i] = cached body i's slot
constant uint CD_JLOCAL_WORLD = 0xFFFFu;   // "no body" (a world end)

// One velocity iteration of every active joint: serial Gauss–Seidel over the list
// (`passes` times), as two chains (real / bias) on two SIMD groups. One threadgroup of 64
// threads; `tgBodies` > 0 selects the threadgroup-cached path, 0 the direct one (straight
// from device memory).
// Threadgroup scratch of the cached pass: CD_JSOLVE_MAXJ blocks, then CD_JSOLVE_MAXB bodies × 4
// float4 (v, ω, bias-v, bias-ω + marker) — 24 960 bytes.
constant uint CD_JSOLVE_TG_BYTES = CD_JSOLVE_MAXJ * 496u + CD_JSOLVE_MAXB * 4u * 16u;

// One velocity iteration of the joints for one threadgroup (see coinJointSolveCS); every
// thread of the group must call it. `cached` is uniform across the group (read after a
// barrier), so the early return of the direct path is taken by every thread together.
//
// RESIDENT BLOCKS (the small-world kernel, stage B3). Between two velocity iterations of one
// substep nothing but this pass reads or writes a joint's block (the contact colours and the
// feet touch bodies only), so a caller that keeps the threadgroup scratch across the iterations
// passes `stageBlocks` only on the first and `writeBlocks` only on the last: the blocks (and the
// stale check) stay in threadgroup memory, the accumulated impulses are written back once, and
// each iteration re-stages only the bodies' velocities — the values the contacts changed. Every
// joint still reads exactly the block values it would have read back from device memory, so the
// result is the same bit for bit. The multi-dispatch kernel passes both on every call. A cached
// body's slot rides in the .w of its first staged float4 (the solve reads .xyz only), so the later
// iterations' staging and every write-back read it from threadgroup memory, not from the list.
static void cdJointSolveTG(uint tid, uint tgs,
    device CoinBody*       coins,
    device float4*         bias,
    device const uint*     list,
    uint                   count,
    device CoinJointPrep*  prep,
    uint                   passes,
    uint                   tgBodies,
    threadgroup CoinJointPrep* tgP,      // [CD_JSOLVE_MAXJ]
    threadgroup float4*        tgB,      // [CD_JSOLVE_MAXB * 4]
    threadgroup atomic_uint&   tgStale,
    bool                   stageBlocks = true,
    bool                   writeBlocks = true)
{
    // Only the joints active this substep (CD_JLIST_ACTIVE, list order): tgP[i] holds the i-th.
    uint nActive = list[CD_JLIST_NACTIVE];
    if (nActive == 0u) return;                      // uniform: every thread reads the same count
    device const uint* act = list + CD_JLIST_ACTIVE;
    uint nb = tgBodies;
    bool cached = nb != 0u && nb <= CD_JSOLVE_MAXB && count <= CD_JSOLVE_MAXJ;

    if (cached && stageBlocks) {
        // Threadgroup-cached path: stage …
        if (tid == 0) atomic_store_explicit(&tgStale, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < nActive; i += tgs) {
            uint k = act[i];
            tgP[i] = prep[list[k]];
            // The body tables were built by the host from the joint table at ENCODE; the
            // block was prepared from the table as it is NOW. A slot re-bound to other
            // bodies in between (removeJoint + add*Joint re-using it while this frame was
            // in flight) would get the OLD bodies' cached velocities: then this pass runs
            // the direct path instead, which reads the bodies from the block itself.
            if (tgP[i].hdr.w > 0.5) {
                uint pr = list[CD_JLIST_PAIRS + k];
                // The serial pass reads the pair word from here (mCfg.w is spare: the solve core
                // reads mCfg.xyz, and the write-back never copies mCfg) instead of two dependent
                // device loads (act[i] → list[…]) per joint per pass.
                tgP[i].mCfg.w = as_type<float>(pr);
                uint la = pr & 0xFFFFu, lb = pr >> 16;
                uint A = as_type<uint>(tgP[i].mass.x), B = as_type<uint>(tgP[i].mass.y);
                bool okB = (lb == CD_JLOCAL_WORLD) ? (B == CD_STATIC) : (list[CD_JLIST_BODIES + lb] == B);
                if (list[CD_JLIST_BODIES + la] != A || !okB) atomic_store_explicit(&tgStale, 1u, memory_order_relaxed);
            }
        }
        for (uint i = tid; i < nb; i += tgs) {
            uint id = list[CD_JLIST_BODIES + i];
            tgB[4*i]     = float4(coins[id].vel.xyz, as_type<float>(id));   // .w: the slot (see above)
            tgB[4*i + 1] = float4(coins[id].angVel.xyz, 0.0);
            tgB[4*i + 2] = bias[2*id];
            tgB[4*i + 3] = bias[2*id + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cached = atomic_load_explicit(&tgStale, memory_order_relaxed) == 0u;   // uniform after the barrier
    } else if (cached) {
        // … or, the blocks resident from an earlier iteration of this substep, only the bodies
        // (whose slots the first staging kept; its stale verdict stands — nothing re-binds a slot
        // mid-frame).
        for (uint i = tid; i < nb; i += tgs) {
            uint id = as_type<uint>(tgB[4*i].w);
            tgB[4*i].xyz = coins[id].vel.xyz;
            tgB[4*i + 1] = float4(coins[id].angVel.xyz, 0.0);
            tgB[4*i + 2] = bias[2*id];
            tgB[4*i + 3] = bias[2*id + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cached = atomic_load_explicit(&tgStale, memory_order_relaxed) == 0u;
    }

    // The two chains (see cdJointSolveReal): the real one on thread 0, the bias one on the
    // first thread of the NEXT SIMD group when the group has one (else thread 0 runs both, one
    // after the other — the same values, since the chains never exchange one).
    const uint realT = 0u, biasT = (tgs > 32u) ? 32u : 0u;

    if (!cached) {
        // Direct path: everything a joint needs is loaded before anything is stored.
        if (tid == realT)
        for (uint pass = 0; pass < passes; ++pass)
        for (uint i = 0; i < nActive; ++i) {
            device CoinJointPrep* P = &prep[list[act[i]]];
            if (P->hdr.w < 0.5) continue;
            float4 mass = P->mass, a0 = P->acc0, a1 = P->acc1;
            uint A = as_type<uint>(mass.x), B = as_type<uint>(mass.y);
            bool bWorld = (B == CD_STATIC);
            CdJBodies bd;
            bd.vA = coins[A].vel.xyz; bd.wA = coins[A].angVel.xyz;
            bd.vB = float3(0.0); bd.wB = float3(0.0);
            if (!bWorld) { bd.vB = coins[B].vel.xyz; bd.wB = coins[B].angVel.xyz; }
            cdJointSolveReal(P, bd, a0, a1);
            // Store: impulses, then the movable ends (an asleep or world end carried no impulse).
            P->acc0 = a0; P->acc1 = a1;
            if (mass.z > 0.0) { coins[A].vel.xyz = bd.vA; coins[A].angVel.xyz = bd.wA; }
            if (!bWorld && mass.w > 0.0) { coins[B].vel.xyz = bd.vB; coins[B].angVel.xyz = bd.wB; }
        }
        if (tid == biasT)
        for (uint pass = 0; pass < passes; ++pass)
        for (uint i = 0; i < nActive; ++i) {
            device CoinJointPrep* P = &prep[list[act[i]]];
            if (P->hdr.w < 0.5) continue;
            float4 mass = P->mass;
            uint A = as_type<uint>(mass.x), B = as_type<uint>(mass.y);
            bool bWorld = (B == CD_STATIC);
            CdJBodies bd;
            float4 bA0 = bias[2*A], bA1 = bias[2*A+1];
            bd.bvA = bA0.xyz; bd.bwA = bA1.xyz;
            float4 bB1 = float4(0.0);
            bd.bvB = float3(0.0); bd.bwB = float3(0.0);
            if (!bWorld) { bd.bvB = bias[2*B].xyz; bB1 = bias[2*B+1]; bd.bwB = bB1.xyz; }
            cdJointSolveBias(P, bd);
            // bias .w of [2·id + 1] is the motor-driven marker — carried through.
            if (mass.z > 0.0) { bias[2*A] = float4(bd.bvA, 0.0); bias[2*A+1] = float4(bd.bwA, bA1.w); }
            if (!bWorld && mass.w > 0.0) { bias[2*B] = float4(bd.bvB, 0.0); bias[2*B+1] = float4(bd.bwB, bB1.w); }
        }
        return;
    }

    // … solve serially out of threadgroup memory: the real chain …
    if (tid == realT) {
        for (uint pass = 0; pass < passes; ++pass)
        for (uint i = 0; i < nActive; ++i) {
            threadgroup CoinJointPrep* P = &tgP[i];
            if (P->hdr.w < 0.5) continue;
            uint pr = as_type<uint>(P->mCfg.w);             // staged above
            uint la = pr & 0xFFFFu, lb = pr >> 16;
            bool bWorld = (lb == CD_JLOCAL_WORLD);
            float4 mass = P->mass, a0 = P->acc0, a1 = P->acc1;
            CdJBodies bd;
            bd.vA = tgB[4*la].xyz; bd.wA = tgB[4*la + 1].xyz;
            bd.vB = float3(0.0); bd.wB = float3(0.0);
            if (!bWorld) { bd.vB = tgB[4*lb].xyz; bd.wB = tgB[4*lb + 1].xyz; }
            cdJointSolveReal(P, bd, a0, a1);
            P->acc0 = a0; P->acc1 = a1;
            if (mass.z > 0.0) { tgB[4*la].xyz = bd.vA; tgB[4*la + 1].xyz = bd.wA; }
            if (!bWorld && mass.w > 0.0) { tgB[4*lb].xyz = bd.vB; tgB[4*lb + 1].xyz = bd.wB; }
        }
    }
    // … and, at the same time, the bias chain.
    if (tid == biasT) {
        for (uint pass = 0; pass < passes; ++pass)
        for (uint i = 0; i < nActive; ++i) {
            threadgroup CoinJointPrep* P = &tgP[i];
            if (P->hdr.w < 0.5) continue;
            uint pr = as_type<uint>(P->mCfg.w);
            uint la = pr & 0xFFFFu, lb = pr >> 16;
            bool bWorld = (lb == CD_JLOCAL_WORLD);
            float4 mass = P->mass;
            CdJBodies bd;
            bd.bvA = tgB[4*la + 2].xyz; bd.bwA = tgB[4*la + 3].xyz;
            bd.bvB = float3(0.0); bd.bwB = float3(0.0);
            if (!bWorld) { bd.bvB = tgB[4*lb + 2].xyz; bd.bwB = tgB[4*lb + 3].xyz; }
            cdJointSolveBias(P, bd);
            if (mass.z > 0.0) { tgB[4*la + 2] = float4(bd.bvA, 0.0); tgB[4*la + 3].xyz = bd.bwA; }   // .w: the marker, kept
            if (!bWorld && mass.w > 0.0) { tgB[4*lb + 2] = float4(bd.bvB, 0.0); tgB[4*lb + 3].xyz = bd.bwB; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // … and write back (an untouched body is rewritten with the values it had) — the blocks'
    // impulses on the last iteration only when they stay resident.
    if (writeBlocks)
    for (uint i = tid; i < nActive; i += tgs) {
        device CoinJointPrep* P = &prep[list[act[i]]];
        P->acc0 = tgP[i].acc0; P->acc1 = tgP[i].acc1;
        P->rF1 = tgP[i].rF1;   // a DISTANCE joint's swing-friction impulse (VZ-0168); else unchanged
    }
    for (uint i = tid; i < nb; i += tgs) {
        uint id = as_type<uint>(tgB[4*i].w);
        coins[id].vel.xyz = tgB[4*i].xyz;
        coins[id].angVel.xyz = tgB[4*i + 1].xyz;
        bias[2*id] = tgB[4*i + 2];
        bias[2*id + 1] = tgB[4*i + 3];
    }
}

kernel void coinJointSolveCS(
    device CoinBody*       coins    [[ buffer(0) ]],
    device float4*         bias     [[ buffer(1) ]],
    device const uint*     list     [[ buffer(2) ]],
    constant uint&         count    [[ buffer(3) ]],
    device CoinJointPrep*  prep     [[ buffer(4) ]],
    constant uint&         passes   [[ buffer(5) ]],
    constant uint&         tgBodies [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]])
{
    threadgroup CoinJointPrep tgP[CD_JSOLVE_MAXJ];
    threadgroup float4        tgB[CD_JSOLVE_MAXB * 4];     // v, ω, bias-v, bias-ω (+ marker)
    threadgroup atomic_uint   tgStale;
    cdJointSolveTG(tid, tgs, coins, bias, list, count, prep, passes, tgBodies, tgP, tgB, tgStale);
}

// Once per frame: the host's compact list of ENABLED joint slots (setBytes, ≤ 4 KB) →
// the private list every joint kernel reads. An entry flagged CD_JLIST_RESET is a slot
// that was disabled at the previous upload or whose bodies / anchors / axes / reference
// changed since: its warm-start state is cleared so it starts cold. When the host sends
// the threadgroup-cache tables (nBodies > 0) — each joint's two local body indices and
// the distinct bodies — they are copied alongside.
kernel void coinJointListUpload(
    constant uint*        src     [[ buffer(0) ]],
    device uint*          list    [[ buffer(1) ]],
    device CoinJointPrep* prep    [[ buffer(2) ]],
    constant uint&        count   [[ buffer(3) ]],
    constant uint*        pairs   [[ buffer(4) ]],
    constant uint*        bodies  [[ buffer(5) ]],
    constant uint&        nBodies [[ buffer(6) ]],
    uint i [[ thread_position_in_grid ]])
{
    if (nBodies > 0u) {
        if (i < count) list[CD_JLIST_PAIRS + i] = pairs[i];
        for (uint b = i; b < nBodies; b += max(count, 1u)) list[CD_JLIST_BODIES + b] = bodies[b];
    }
    if (i >= count) return;
    uint e = src[i];
    uint slot = e & ~CD_JLIST_RESET;
    list[i] = slot;
    if ((e & CD_JLIST_RESET) != 0u) {
        prep[slot].hdr = float4(0.0);
        prep[slot].acc0 = float4(0.0);
        prep[slot].acc1 = float4(0.0);
        prep[slot].mCfg = float4(0.0);
        prep[slot].rF1 = float4(0.0);    // a DISTANCE joint's swing-friction impulse (VZ-0168)
    }
}

// Union the two bodies of each ACTIVE joint into one island, so an articulated
// assembly sleeps and wakes as a unit (mirror of coinIslandUnion over contacts).
static inline void cdIslandUnionJointBody(uint k, device atomic_uint* label, device const CoinJoint* joints,
                                          uint count, device const uint* list)
{
    if (k >= count) return;
    CoinJoint jn = joints[list[k]];
    if (jn.meta.w == 0u || jn.meta.z == CD_STATIC) return;
    uint a = jn.meta.y, b = jn.meta.z;
    uint la = atomic_load_explicit(&label[a], memory_order_relaxed);
    uint lb = atomic_load_explicit(&label[b], memory_order_relaxed);
    uint lo = min(la, lb);
    atomic_fetch_min_explicit(&label[a], lo, memory_order_relaxed);
    atomic_fetch_min_explicit(&label[b], lo, memory_order_relaxed);
}

kernel void coinIslandUnionJoints(
    device atomic_uint*     label  [[ buffer(0) ]],
    device const CoinJoint* joints [[ buffer(1) ]],
    constant uint&          count  [[ buffer(2) ]],
    device const uint*      list   [[ buffer(3) ]],
    uint k [[ thread_position_in_grid ]])
{
    cdIslandUnionJointBody(k, label, joints, count, list);
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

// ── Sleep groups: island connectivity among ASLEEP bodies (VZ-0152) ─────────────
// coinGenerateContacts no longer emits asleep–asleep contacts, so the contact graph
// alone would split a sleeping heap into singletons, and a heap woken through one body
// would wake one contact layer per frame. But an asleep body cannot move, so the
// connectivity it had when it fell asleep is still exact: coinSleepMark records each
// sleeping body's island label as its `sleepKey`, and every frame the asleep, live
// bodies of one key are unioned through a hub (the smallest such id) alongside the
// contact and joint edges. An awake body touching ANY member (a real contact, since
// one end is awake) therefore joins the whole group, whose island minimum then drops
// and wakes it as a unit, in the frame of the touch — exactly what the asleep–asleep
// contacts used to provide, without their narrowphase. A dead slot (despawned while
// asleep) never joins, so a despawn does not wake a heap (it didn't before either).
constant uint CD_NO_KEY = 0xFFFFFFFFu;

static inline void cdIslandInitBody(uint id, device uint* label, constant CoinUniforms& u, device uint* sleepHub)
{
    if (id >= u.coinCount) return;
    label[id] = id;
    sleepHub[id] = CD_NO_KEY;
}

kernel void coinIslandInit(
    device uint*           label    [[ buffer(0) ]],
    constant CoinUniforms& u        [[ buffer(1) ]],
    device uint*           sleepHub [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIslandInitBody(id, label, u, sleepHub);
}

// Hub of each sleep group = its smallest asleep, live member (after coinIslandInit).
static inline void cdIslandSleepHubBody(uint id, device const CoinBody* coins, device const uint* asleep,
                                        device const uint* sleepKey, device atomic_uint* sleepHub,
                                        constant CoinUniforms& u)
{
    if (id >= u.coinCount) return;
    uint key = sleepKey[id];
    if (asleep[id] == 0u || key >= u.coinCount || coins[id].posInvMass.w == 0.0) return;
    atomic_fetch_min_explicit(&sleepHub[key], id, memory_order_relaxed);
}

kernel void coinIslandSleepHub(
    device const CoinBody* coins    [[ buffer(0) ]],
    device const uint*     asleep   [[ buffer(1) ]],
    device const uint*     sleepKey [[ buffer(2) ]],
    device atomic_uint*    sleepHub [[ buffer(3) ]],
    constant CoinUniforms& u        [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIslandSleepHubBody(id, coins, asleep, sleepKey, sleepHub, u);
}

static void cdUnionLabels(device atomic_uint* label, uint a, uint b) {
    uint la = atomic_load_explicit(&label[a], memory_order_relaxed);
    uint lb = atomic_load_explicit(&label[b], memory_order_relaxed);
    uint lo = min(la, lb);
    atomic_fetch_min_explicit(&label[a], lo, memory_order_relaxed);
    atomic_fetch_min_explicit(&label[b], lo, memory_order_relaxed);
}

// Union the two bodies of each contact (atomic-min connectivity; converges over rounds).
// Threads below coinCount ALSO apply their body's sleep-group edge (body → its group's
// hub), so the rounds converge contact, joint and sleep-group connectivity together
// with no extra dispatch (the grid is maxContacts ≥ maxCoins threads).
// `n` = the contact count (the caller's read of the append cursor).
static inline void cdIslandUnionBody(uint cid, uint n,
    device atomic_uint*       label,
    device const CoinContact* contacts,
    device const CoinBody*    coins,
    device const uint*        asleep,
    device const uint*        sleepKey,
    device const uint*        sleepHub,
    constant CoinUniforms&    u)
{
    if (cid < u.coinCount && asleep[cid] != 0u) {
        uint key = sleepKey[cid];
        if (key < u.coinCount && coins[cid].posInvMass.w != 0.0) {
            uint hub = sleepHub[key];
            if (hub < u.coinCount && hub != cid) cdUnionLabels(label, cid, hub);
        }
    }
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    if (c.meta.y == CD_STATIC) return;                 // statics aren't solver DOFs
    cdUnionLabels(label, c.meta.x, c.meta.y);
}

kernel void coinIslandUnion(
    device atomic_uint*       label        [[ buffer(0) ]],
    device const CoinContact* contacts     [[ buffer(1) ]],
    device const atomic_uint& contactCount [[ buffer(2) ]],
    device const CoinBody*    coins        [[ buffer(3) ]],
    device const uint*        asleep       [[ buffer(4) ]],
    device const uint*        sleepKey     [[ buffer(5) ]],
    device const uint*        sleepHub     [[ buffer(6) ]],
    constant CoinUniforms&    u            [[ buffer(7) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdIslandUnionBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), label, contacts, coins,
                      asleep, sleepKey, sleepHub, u);
}

// Pointer-jump: flatten label[id] toward its component root (run a few times after union).
static inline void cdIslandJumpBody(uint id, device uint* label, constant CoinUniforms& u)
{
    if (id >= u.coinCount) return;
    uint l = label[id];
    label[id] = label[l];
}

kernel void coinIslandJump(
    device uint*           label [[ buffer(0) ]],
    constant CoinUniforms& u     [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIslandJumpBody(id, label, u);
}

// Per body: tick the slow-frame counter; then reset the per-island min accumulator.
static inline void cdSleepTickBody(uint id, device const CoinBody* coins, device uint* sleepTimer,
                                   device uint* islandMin, constant CoinUniforms& u, device const float4* bias)
{
    if (id >= u.coinCount) return;
    islandMin[id] = 0xFFFFFFFFu;
    CoinBody c = coins[id];
    if (c.posInvMass.w == 0.0) { sleepTimer[id] = 0u; return; }
    // Same two modes as coinFinalizeCS's dead-stop, at 2.5× its speed; a body a motor
    // is driving (non-zero target) is never slow, or a slow slew would freeze its
    // whole island after sleepFrames (VZ-0149).
    bool slow = ((u.solverFlags & CD_FLAG_SCALE_AWARE_DEADSTOP) != 0u)
              ? (length(c.vel.xyz) + length(c.angVel.xyz) * cdBoundRadiusOf(c) < u.sleepLinVel * 2.5)
              : (length(c.vel.xyz) < u.sleepLinVel * 2.5 && length(c.angVel.xyz) < 1.0);
    if (bias[2*id+1].w > 0.5) slow = false;
    // Hysteresis: a slow frame INCREMENTS, a fast frame DECREMENTS (not a hard reset) —
    // so a transient contact kick doesn't keep an otherwise-settled island awake, while
    // a genuinely moving body (sustained fast frames) still decays to 0 and wakes it.
    uint t = sleepTimer[id];
    sleepTimer[id] = slow ? min(t + 1u, 100000u) : (t > 6u ? t - 6u : 0u);
}

kernel void coinSleepTick(
    device const CoinBody* coins      [[ buffer(0) ]],
    device uint*           sleepTimer [[ buffer(1) ]],
    device uint*           islandMin  [[ buffer(2) ]],
    constant CoinUniforms& u          [[ buffer(3) ]],
    device const float4*   bias       [[ buffer(4) ]],   // last substep's motor-driven marker (.w of [2*id+1])
    uint id [[ thread_position_in_grid ]])
{
    cdSleepTickBody(id, coins, sleepTimer, islandMin, u, bias);
}

// Reduce each island's MIN slow-frame count (an island is only as asleep as its
// most-recently-moved body — so any motion anywhere keeps the whole island awake).
static inline void cdIslandMinReduceBody(uint id, device const uint* label, device const uint* sleepTimer,
                                         device atomic_uint* islandMin, constant CoinUniforms& u)
{
    if (id >= u.coinCount) return;
    atomic_fetch_min_explicit(&islandMin[label[id]], sleepTimer[id], memory_order_relaxed);
}

kernel void coinIslandMinReduce(
    device const uint*  label      [[ buffer(0) ]],
    device const uint*  sleepTimer [[ buffer(1) ]],
    device atomic_uint* islandMin  [[ buffer(2) ]],
    constant CoinUniforms& u       [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdIslandMinReduceBody(id, label, sleepTimer, islandMin, u);
}

// Mark a body asleep iff its whole island has been slow for ≥ sleepFrames.
//
// FREEZING ZEROES THE VELOCITY. A frozen body's position is not integrated
// (coinIntegratePositionCS returns for asleep[id]), so whatever was left in `vel` /
// `angVel` at the moment it froze would sit there forever describing motion that is
// not happening: measured on the settled Pile of Mess pile, a body frozen holding
// 0.152 m/s still read 0.152 m/s 600 frames later having moved 0.0000 m. Every
// consumer of `vel` — CoinDiagnostics' rest-KE gate, any velocity-driven renderer —
// was reading that lie, and which bodies froze mid-motion is decided by the
// non-deterministic contact ordering, so the "rest KE" of a dead-still pile came out
// as a run-to-run lottery (see RigidPileFieldTests.testPileOfMessRestQuiet). Zeroing
// on freeze is also what Box2D/Bullet do: asleep means at rest, so v ≡ 0. It cannot
// lose momentum, because a frozen body was already not moving.
static inline void cdSleepMarkBody(uint id, device const uint* label, device const uint* islandMin,
                                   device uint* asleep, constant CoinUniforms& u, uint sleepFrames,
                                   device CoinBody* coins, device uint* sleepKey)
{
    if (id >= u.coinCount) return;
    bool sleeping = (islandMin[label[id]] >= sleepFrames);
    asleep[id] = sleeping ? 1u : 0u;
    // The label is the sleep decision's own key (islandMin is indexed by it), so the
    // bodies that fall asleep together share it — the sleep group (see coinIslandSleepHub).
    sleepKey[id] = sleeping ? label[id] : CD_NO_KEY;
    if (sleeping) {
        coins[id].vel.xyz    = float3(0.0);
        coins[id].angVel.xyz = float3(0.0);
    }
}

kernel void coinSleepMark(
    device const uint*     label     [[ buffer(0) ]],
    device const uint*     islandMin [[ buffer(1) ]],
    device uint*           asleep    [[ buffer(2) ]],
    constant CoinUniforms& u         [[ buffer(3) ]],
    constant uint&         sleepFrames [[ buffer(4) ]],
    device CoinBody*       coins     [[ buffer(5) ]],
    device uint*           sleepKey  [[ buffer(6) ]],
    uint id [[ thread_position_in_grid ]])
{
    cdSleepMarkBody(id, label, islandMin, asleep, u, sleepFrames, coins, sleepKey);
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

// FOOT HOOK — the spring-foot actuator's kernels (CoinDEMSolver+Foot.swift). Last in the file:
// they use the structs and helpers above. A solver without an active foot never dispatches them.
#include "CoinDEMFoot.h"

// The multi-dispatch substep's small kernels, fused (stage B3, engine plan 3b).
#include "CoinDEMFusedKernels.h"

// Prepared contact rows (stage B3, opt-in CoinDEMSolver.preparedContactSolve): the pose-constant
// factors of every contact row computed once per substep.
#include "CoinDEMPreparedSolve.h"

// The SMALL-WORLD frame kernel (stage B3, engine plan 3c; CoinDEMSolver+SmallWorld.swift): a whole
// constraint-path frame of a small world in one threadgroup dispatch, over the same step functions.
#include "CoinDEMSmallWorld.h"
