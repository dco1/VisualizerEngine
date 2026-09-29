// ── CoinDEMFoot.h — the SPRING-FOOT actuator (engine plan item 5b) ────────────
//
// plausibility: real — a spring-loaded foot is a FORCE ELEMENT: a massless telescoping leg
// whose spring-damper force k·δ + c·δ̇ (push only, capped) is integrated EXACTLY over each
// substep, and whose reaction goes into whatever the foot stands on. Nothing is kicked,
// nothing is kinematic: a hop is the spring's own stored energy ½·k·Δ² turned into motion
// by the solver, and a turn in place is a yaw motor fighting real pad friction.
//
// Included at the END of CoinDEM.metal (it uses the solver's structs and helpers), so its
// kernels live in the same library; CoinDEMSolver+Foot.swift is the host side. A solver
// with no foot — every existing scene, Daydream Home's egg rain — never dispatches any of
// this, and the rest of CoinDEM.metal is untouched by it.
//
// THE MODEL: a SLIP leg. Each foot is a massless leg hinged at a HIP point on its body (the
// mount; the COM for a Weeble worker). While it is in the air the leg points along its
// body-local axis (the host cants it and yaws it to aim the next hop), with rest length L0
// (retracted = inside the shell, i.e. no contact at all; extended = past the surface). When
// the leg reaches a static collider or a dynamic box / sphere within L0 it PLANTS there: the
// tip is pinned to the support (in the support's frame) and, like the spring-loaded
// inverted pendulum every hopping-robot model uses, the leg swings freely about the hip —
// its force is always along the line tip → hip. With the hip at the COM that force has no
// moment arm, so a push can never spin the body: flight spin ≈ 0 BY CONSTRUCTION, however
// the leg is canted. (A leg rigid in the body would not have that property — gravity's
// share across a canted leg topples the body about the pinned tip during the push, 1–2
// rad/s for a 12 mm hop at 11°; the SLIP hip is what the plan's "fire through the COM"
// needs.)
//
//   • Grip — the pad holds only while the leg's line of force lies inside the support's
//     friction cone, |u_t| ≤ μ·(u·n), μ = √(μ_pad·μ_support) (the solver's combine rule).
//     Outside it the massless leg cannot push at all: the foot SKIDS (no force; counted).
//   • Spring — δ = L0 − |tip − hip|, F = clamp(k·δ + c·δ̇, 0, Fmax) (a massless leg cannot
//     pull; it re-plants when it comes back). Liftoff when |tip − hip| ≥ L0.
//   • Torsion — a planted pad of radius r_patch resists spin about the support normal up to
//     μ_spin·r_patch·N, and through it the hip's YAW MOTOR (target rate, max torque) turns
//     the body in place: one row about n per velocity iteration, its ACCUMULATED impulse
//     clamped to min(μ_spin·r_patch·Λn, τ_max·dt), Λn the substep's normal leg impulse.
//
// WHY NOT A CATTO SOFT ROW. The implicit soft constraint (γ = 1/(h(c + hk)), β = hk/(c + hk))
// is implicit Euler of the spring: at the Digital Clock's 1/180 s substep a worker leg
// (k 230 N/m, ω = √(k/m) ≈ 125 rad/s, ω·h ≈ 0.7) loses about HALF of ½kΔ² to numerical
// damping in the 2–3 substeps of a push (a 1-D model of the solver step: 5.9 mm apex instead
// of 11.9 mm). So each planted leg is integrated EXACTLY instead: once per substep, its 1-DOF
// relative coordinate (effective mass m_eff along the leg, gravity continuous) is integrated
// with fine RK4 steps (ω·dt ≤ 0.25, ≥ 8 steps — error < 1e-7 of the stroke), the resulting
// impulse P = ∫F dt is applied to the body at the hip and −P to a dynamic support at the tip,
// and — after the substep's contact and joint rows — the POSITION part of that exact solution
// goes in through the split-impulse bias channel (x += (v + bias)·dt, bias discarded), because
// the solver's drift x += v_end·dt would otherwise move the body as if it had had its END
// velocity for the whole substep and throw away a fifth of the stroke (9.4 mm apex). The
// correction is applied only while the leg's DOF moved freely (nothing else changed the
// relative velocity along the leg this substep); when a contact shares the load (a worker
// standing on its round bottom AND its foot) the contact owns the motion and it is skipped.
// That 1-D model gives 11.75 mm against the continuous 11.9 mm; at rest on a leg the scheme
// is exactly steady (P = m·g·dt, no drift, v = 0 — so a standing worker can sleep).
//
// SLEEP. A pulse still armed (a pending release) or a planted foot turning under its yaw
// motor sets the joint pass's motor-driven marker (bias[2i+1].w = 1): the finalize dead-stop
// and the sleep test leave the body alone. A foot parked retracted costs nothing and the
// worker sleeps exactly as a footless one. A planted foot joins its body's island to a
// dynamic support's (coinFootIslandUnion), and a pushing foot wakes an asleep dynamic support
// (sleep on; with sleep off an asleep body is a host kinematic hold and stays immovable).

constant uint CF_MAX_FEET       = 64u;
constant uint CF_MODE_RETRACTED = 0u;   // leg inside the shell: no contact, no force
constant uint CF_MODE_EXTENDED  = 1u;   // leg held out at its extended length (a stand / a pogo)
constant uint CF_MODE_PULSE     = 2u;   // extended until the next liftoff, then latched retracted

// Host-written configuration (CoinFootGPU in CoinDEMSolver+Foot.swift). 80 bytes.
struct CoinFoot {
    uint4  meta;    // x = body slot, y = enabled (0/1), z = mode (CF_MODE_*), w = armSeq (a pulse fires once per value)
    float4 mount;   // xyz = body-local hip (m), w = extended rest length L0 (m)
    float4 axis;    // xyz = body-local leg direction (unit, cant + yaw applied), w = retracted length (drawing only)
    float4 spring;  // x = k (N/m), y = c (N·s/m), z = max force (N), w = pad μ (< 0: the body's own material μ)
    float4 spin;    // x = yaw target (rad/s, body relative to support about n), y = yaw max torque (N·m), z = μ_spin, w = patch radius (m)
};

// GPU-written, persistent (CoinFootStateGPU). 160 bytes.
struct CoinFootState {
    uint4  meta;    // x = planted (0/1), y = support (CD_STATIC or body slot), z = static collider index, w = firedSeq (pulse)
    float4 tipL;    // xyz = planted tip, support frame (world for a static), w = compression δ at this substep's start (m)
    float4 nrmL;    // xyz = support normal at the tip, support frame (world for a static), w = leg length |tip − hip| (m)
    float4 point;   // xyz = world tip, w = leg force this substep (N) = P / dt
    float4 dir;     // xyz = world push direction u (tip → hip), w = normal share of the leg impulse Λn = P·(u·n) (N·s)
    float4 imp;     // x = leg impulse P (N·s), y = exact end compression x(h), z = exact end compression rate x'(h), w = m_eff (kg)
    float4 nrmW;    // xyz = world support normal, w = position correction applied this substep (m, along u)
    float4 row;     // x = Σ n·I⁻¹·n, y = torsion bound (N·m·s), z = spin target (rad/s), w = accumulated torsion (this substep)
    float4 total;   // xyz = cumulative leg impulse on the body (world, N·s), w = cumulative torsion on the body about n (N·m·s)
    float4 stats;   // x = plants, y = liftoffs, z = skids, w = impulse delivered during the current plant (N·s)
};

// setBytes uniforms (CoinFootUniformsGPU). 32 bytes.
struct CoinFootUniforms {
    float dt;
    float gravity;
    float globalMu;      // CoinUniforms.frictionCoeff: what a "< 0 = inherit" μ falls back to
    uint  footCount;
    uint  colliderCount;
    uint  bodyCount;     // the solver's high-water mark
    uint  sleepEnabled;
    uint  lastIteration; // coinFootIterate: 1 on the substep's final velocity iteration
};

static_assert(sizeof(CoinFoot) == 80, "CoinFoot must match CoinFootGPU (CoinDEMSolver+Foot.swift)");
static_assert(sizeof(CoinFootState) == 160, "CoinFootState must match CoinFootStateGPU");
static_assert(sizeof(CoinFootUniforms) == 32, "CoinFootUniforms must match CoinFootUniformsGPU");

// ── Leg rays ───────────────────────────────────────────────────────────────────

// Ray (o, d unit) against an axis-aligned box centred at the origin with half-extents he,
// origin OUTSIDE: the entry distance and the entry face's outward normal. An origin inside
// the box (a hip buried in a collider) is no hit — the leg cannot plant from in there.
static bool cfRaySlab(float3 o, float3 d, float3 he, float tMax, thread float& tHit, thread float3& nLocal) {
    float tNear = -1e30, tFar = 1e30;
    int axis = -1;
    for (int i = 0; i < 3; ++i) {
        float oi = o[i], di = d[i], hi = he[i];
        if (fabs(di) < 1e-9) {
            if (fabs(oi) > hi) return false;
            continue;
        }
        float inv = 1.0 / di;
        float t1 = (-hi - oi) * inv, t2 = (hi - oi) * inv;
        float tn = min(t1, t2), tf = max(t1, t2);
        if (tn > tNear) { tNear = tn; axis = i; }
        tFar = min(tFar, tf);
    }
    if (axis < 0 || tNear > tFar || tNear < 0.0 || tNear > tMax) return false;
    nLocal = float3(0.0);
    nLocal[axis] = d[axis] > 0.0 ? -1.0 : 1.0;
    tHit = tNear;
    return true;
}

// Ray against a sphere, origin outside.
static bool cfRaySphere(float3 o, float3 d, float3 c, float r, float tMax,
                        thread float& tHit, thread float3& n) {
    float3 oc = o - c;
    float b = dot(oc, d);
    float cc = dot(oc, oc) - r * r;
    if (cc < 0.0 || b > 0.0) return false;
    float disc = b * b - cc;
    if (disc < 0.0) return false;
    float t = -b - sqrt(disc);
    if (t < 0.0 || t > tMax) return false;
    tHit = t;
    n = normalize(o + t * d - c);
    return true;
}

// The leg's nearest support along the ray hip + t·a, t ∈ [0, tMax]: static planes, boxes and
// oriented boxes, and dynamic boxes / spheres (not the foot's own body). Other shapes are not
// leg supports (a worker hops on the table, not on another worker's hat). Returns the support
// (CD_STATIC or a slot), the collider index for a static, the distance and the world normal.
static bool cfCastLeg(float3 o, float3 a, float tMax, uint self,
                      device const CoinBody* coins, device const CoinStaticCollider* colliders,
                      constant CoinFootUniforms& fu,
                      thread uint& sup, thread uint& col, thread float& tHit, thread float3& nHit) {
    bool hit = false;
    float best = tMax;
    for (uint i = 0; i < fu.colliderCount; ++i) {
        CoinStaticCollider C = colliders[i];
        uint kind = as_type<uint>(C.a.w);
        float t; float3 n;
        bool h = false;
        if (kind == 0u) {                                   // plane n·x = d
            float3 pn = C.a.xyz;
            float dn = dot(a, pn);
            float s = dot(o, pn) - C.b.w;
            if (dn < -1e-6 && s >= 0.0) { t = s / -dn; n = pn; h = t <= best; }
        } else if (kind == 1u) {                            // axis-aligned box
            float3 nl;
            h = cfRaySlab(o - C.a.xyz, a, C.b.xyz, best, t, nl);
            n = nl;
        } else if (kind == 3u) {                            // oriented box
            float3 nl;
            h = cfRaySlab(cdQuatRotateInv(C.orient, o - C.a.xyz), cdQuatRotateInv(C.orient, a), C.b.xyz, best, t, nl);
            n = cdQuatRotate(C.orient, nl);
        }
        if (h && t <= best) { best = t; sup = CD_STATIC; col = i; nHit = n; hit = true; }
    }
    for (uint j = 0; j < fu.bodyCount; ++j) {
        if (j == self) continue;
        CoinBody O = coins[j];
        if (O.posInvMass.w == 0.0) continue;
        bool isBox = cdIsBox(O), isSphere = cdIsSphere(O);
        if (!isBox && !isSphere) continue;
        // Bounding reject: the body's bounding sphere against the segment [o, o + a·best].
        float3 oc = O.posInvMass.xyz - o;
        float tc = clamp(dot(oc, a), 0.0, best);
        float R = cdRadiusOf(O);
        if (length_squared(oc - tc * a) > R * R) continue;
        float t; float3 n;
        bool h;
        if (isBox) {
            float3 nl;
            h = cfRaySlab(cdQuatRotateInv(O.orient, o - O.posInvMass.xyz), cdQuatRotateInv(O.orient, a),
                          O.shapeExtents.xyz, best, t, nl);
            n = cdQuatRotate(O.orient, nl);
        } else {
            h = cfRaySphere(o, a, O.posInvMass.xyz, R, best, t, n);
        }
        if (h && t <= best) { best = t; sup = j; col = 0u; nHit = n; hit = true; }
    }
    tHit = best;
    return hit;
}

// ── The leg's exact substep ────────────────────────────────────────────────────

// Push-only, capped spring-damper force for compression x (m) and compression rate xd (m/s).
static float cfLegForce(float x, float xd, float k, float c, float fMax) {
    if (x <= 0.0) return 0.0;
    return clamp(k * x + c * xd, 0.0, fMax);
}

// Integrate the leg's relative coordinate over one substep h: compression x, rate xd, with
// x'' = gRel − F(x, x')/mEff (gRel = gravity's pull along the leg on the body relative to its
// support). Fine RK4 steps (ω·dt ≤ 0.25, at least 8). Out: the end compression and rate.
static void cfLegStep(float x0, float xd0, float gRel, float k, float c, float fMax, float mEff, float h,
                      thread float& xh, thread float& xdh) {
    float invM = 1.0 / mEff;
    float rate = max(sqrt(max(k, 0.0) * invM), max(c, 0.0) * invM);
    int n = clamp(int(ceil(rate * h * 4.0)), 8, 256);
    float dt = h / float(n);
    float x = x0, v = xd0;
    for (int i = 0; i < n; ++i) {
        float a1 = gRel - cfLegForce(x, v, k, c, fMax) * invM;
        float xa = x + 0.5 * dt * v,  va = v + 0.5 * dt * a1;
        float a2 = gRel - cfLegForce(xa, va, k, c, fMax) * invM;
        float xb = x + 0.5 * dt * va, vb = v + 0.5 * dt * a2;
        float a3 = gRel - cfLegForce(xb, vb, k, c, fMax) * invM;
        float xc = x + dt * vb,       vc = v + dt * a3;
        float a4 = gRel - cfLegForce(xc, vc, k, c, fMax) * invM;
        x += dt / 6.0 * (v + 2.0 * va + 2.0 * vb + vc);
        v += dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
    }
    xh = x; xdh = v;
}

static float cfMaterialMu(float m, float globalMu) { return m >= 0.0 ? m : globalMu; }

// ── Phase 1 of coinFootSubstep: one foot's geometry, plant / liftoff / skid, and its
//    exact leg impulse (written to the foot's state; applied in phase 2). ────────────────
static void cfFootPrepare(uint f, device const CoinBody* coins, device const CoinFoot* feet,
                          device CoinFootState* state, device const CoinStaticCollider* colliders,
                          device const float2* material, device const uint* asleep,
                          constant CoinFootUniforms& fu) {
    CoinFoot F = feet[f];
    CoinFootState S = state[f];
    // This substep's outputs start empty.
    S.imp = float4(0.0);
    S.row = float4(0.0);
    S.dir.w = 0.0; S.point.w = 0.0; S.nrmW.w = 0.0; S.tipL.w = 0.0;
    uint b = F.meta.x;
    if (F.meta.y == 0u || b >= fu.bodyCount || coins[b].posInvMass.w == 0.0) {
        S.meta.x = 0u; state[f] = S; return;
    }
    if (asleep[b] != 0u) { state[f] = S; return; }            // frozen: keeps its plant, moves nothing
    bool pulse = F.meta.z == CF_MODE_PULSE;
    bool active = F.meta.z == CF_MODE_EXTENDED || (pulse && S.meta.w != F.meta.w);
    if (!active) { S.meta.x = 0u; state[f] = S; return; }       // retracted: the leg is inside the shell

    CoinBody B = coins[b];
    float L0 = F.mount.w;
    float3 x = B.posInvMass.xyz;
    float4 q = B.orient;
    float3 hip = x + cdQuatRotate(q, F.mount.xyz);
    float muPad = F.spring.w >= 0.0 ? F.spring.w : cfMaterialMu(material[b].x, fu.globalMu);

    bool planted = S.meta.x != 0u;
    uint sup = S.meta.y, col = S.meta.z;
    float3 p = float3(0.0), n = float3(0.0, 1.0, 0.0);
    if (planted) {
        if (sup == CD_STATIC) {
            p = S.tipL.xyz; n = S.nrmL.xyz;
        } else if (sup < fu.bodyCount && coins[sup].posInvMass.w != 0.0) {
            CoinBody O = coins[sup];
            p = O.posInvMass.xyz + cdQuatRotate(O.orient, S.tipL.xyz);
            n = normalize(cdQuatRotate(O.orient, S.nrmL.xyz));
        } else {
            planted = false;                                    // the support was despawned
            S.meta.x = 0u;
        }
    }
    float L;
    if (planted) {
        L = length(hip - p);
        if (L >= L0 || L < 1e-7) {                              // leg at its rest length: the tip lifts off
            S.meta.x = 0u;
            S.stats.y += 1.0;
            if (pulse && S.stats.w > 0.0) S.meta.w = F.meta.w;  // the pulse has pushed: latch retracted
            state[f] = S;
            return;
        }
    } else {
        float3 a = normalize(cdQuatRotate(q, F.axis.xyz));
        float t; float3 nh; uint s = CD_STATIC, c = 0u;
        if (!cfCastLeg(hip, a, L0, b, coins, colliders, fu, s, c, t, nh)) { state[f] = S; return; }
        p = hip + t * a; n = nh; sup = s; col = c; L = t;
    }
    float3 u = (hip - p) / max(L, 1e-7);                       // push direction on the body: tip → hip

    // Grip: the leg's line of force must lie inside the support's friction cone.
    float muS;
    if (sup == CD_STATIC) muS = cfMaterialMu(as_type<float>(colliders[col].meta.y), fu.globalMu);
    else                  muS = cfMaterialMu(material[sup].x, fu.globalMu);
    float mu = sqrt(max(muPad * muS, 0.0));
    float un = dot(u, n);
    if (un <= 1e-4 || length(u - un * n) > mu * un) {
        // A stance that loses its grip AFTER pushing has ended as surely as one that reaches
        // its rest length: a pulse latches here too. Otherwise the still-extended leg re-plants
        // at the landing and fires a second, uncommanded stroke (a 27° hop: the leg's line of
        // force drifts ~1° outward during the push, skids at the end of it, then the landing
        // re-plant knocked the worker over — 4 plants, 9 bounces, 130° tilt for ONE pulse).
        if (planted && pulse && S.stats.w > 0.0) S.meta.w = F.meta.w;
        S.meta.x = 0u;
        S.stats.z += 1.0;
        state[f] = S;
        return;
    }
    if (!planted) {                                             // plant: pin the tip in the support's frame
        S.meta = uint4(1u, sup, col, S.meta.w);
        if (sup == CD_STATIC) { S.tipL.xyz = p; S.nrmL.xyz = n; }
        else {
            CoinBody O = coins[sup];
            S.tipL.xyz = cdQuatRotateInv(O.orient, p - O.posInvMass.xyz);
            S.nrmL.xyz = cdQuatRotateInv(O.orient, n);
        }
        S.stats.x += 1.0;
        S.stats.w = 0.0;
    }

    // Effective mass along u: the body at the hip, a movable dynamic support at the tip.
    float3 rH = hip - x;
    float invMb = B.posInvMass.w;
    float3 invIb = cdBodyInvInertia(B, invMb);
    float3 rHu = cross(rH, u);
    float kInv = invMb + dot(rHu, cdApplyInvInertiaWorld(q, invIb, rHu));
    float3 vp = float3(0.0);
    bool supKicked = false;                                     // did intVel give the support this substep's gravity?
    float kT = dot(n, cdApplyInvInertiaWorld(q, invIb, n));
    if (sup != CD_STATIC) {
        CoinBody O = coins[sup];
        bool sAsleep = asleep[sup] != 0u;
        supKicked = !sAsleep;
        if (!sAsleep || fu.sleepEnabled != 0u) {                // asleep + sleep on: woken in phase 2
            float invMs = O.posInvMass.w;
            float3 invIs = cdBodyInvInertia(O, invMs);
            float3 rP = p - O.posInvMass.xyz;
            float3 rPu = cross(rP, u);
            kInv += invMs + dot(rPu, cdApplyInvInertiaWorld(O.orient, invIs, rPu));
            kT += dot(n, cdApplyInvInertiaWorld(O.orient, invIs, n));
            vp = O.vel.xyz + cross(O.angVel.xyz, rP);           // an asleep one had v zeroed by intVel
        }
    } else {
        vp = cdColliderVelocity(colliders, col);                // a host-moved kinematic box carries its velocity
    }
    float mEff = 1.0 / max(kInv, 1e-12);
    float3 vh = B.vel.xyz + cross(B.angVel.xyz, rH);
    float delta = L0 - L;
    float xd0 = -dot(vh - vp, u);                               // compression rate, AFTER this substep's gravity kick
    // Gravity along the leg, relative to the support: the body's pull, unless the support
    // fell with it this substep (a free dynamic support). The kick intVel already applied is
    // taken back out and gravity integrated continuously, so a leg holding a body at rest is
    // EXACTLY steady (P = m·g·dt, no creep).
    float gRel = fu.gravity * u.y * (supKicked ? 0.0 : 1.0);
    float xh, xdh;
    cfLegStep(delta, xd0 - gRel * fu.dt, gRel, F.spring.x, F.spring.y, F.spring.z, mEff, fu.dt, xh, xdh);
    float P = max(mEff * (xd0 - xdh), 0.0);                    // = ∫F dt over the substep

    S.tipL.w = delta;
    S.nrmL.w = L;
    S.point = float4(p, P / fu.dt);
    S.dir = float4(u, P * un);
    S.imp = float4(P, xh, xdh, mEff);
    S.nrmW = float4(n, 0.0);
    S.stats.w += P;
    // The pad's torsion / yaw-motor row for this substep's iterations.
    float bound = min(F.spin.z * F.spin.w * P * un, max(F.spin.y, 0.0) * fu.dt);
    if (bound > 0.0 && kT > 1e-12) S.row = float4(kT, bound, F.spin.x, 0.0);
    state[f] = S;
}

// A planted foot whose support is a DYNAMIC body touches two bodies; every other foot touches
// only its own (one foot per body — the host API keys feet by body). The passes below run the
// latter in parallel, one thread per foot, and the former serially on thread 0 afterwards.
static bool cfOnDynamicSupport(CoinFootState S) { return S.meta.x != 0u && S.meta.y != CD_STATIC; }

// Phase 2 of coinFootSubstep for one foot: the motor-driven marker, then the leg impulse onto
// the body at the hip and its reaction onto a dynamic support at the tip (waking it if asleep).
static void cfFootApply(uint f, device CoinBody* coins, device float4* bias, device const CoinFoot* feet,
                        device CoinFootState* state, device uint* asleep, device uint* sleepTimer,
                        constant CoinFootUniforms& fu) {
    CoinFoot F = feet[f];
    uint b = F.meta.x;
    if (F.meta.y == 0u || b >= fu.bodyCount || asleep[b] != 0u || coins[b].posInvMass.w == 0.0) return;
    CoinFootState S = state[f];
    bool pending = F.meta.z == CF_MODE_PULSE && S.meta.w != F.meta.w;
    bool planted = S.meta.x != 0u;
    bool turning = planted && S.row.y > 0.0 && F.spin.x != 0.0;
    if (pending || turning) bias[2 * b + 1].w = 1.0;           // not dead-stopped, not "slow" for sleep
    float P = S.imp.x;
    if (!planted || P <= 0.0) return;

    CoinBody B = coins[b];
    float3 u = S.dir.xyz, p = S.point.xyz;
    float3 rH = cdQuatRotate(B.orient, F.mount.xyz);
    float invMb = B.posInvMass.w;
    float3 J = P * u;
    coins[b].vel.xyz    = B.vel.xyz + invMb * J;
    coins[b].angVel.xyz = B.angVel.xyz + cdApplyInvInertiaWorld(B.orient, cdBodyInvInertia(B, invMb), cross(rH, J));
    uint s = S.meta.y;
    if (s != CD_STATIC && s < fu.bodyCount) {
        if (asleep[s] != 0u && fu.sleepEnabled != 0u) {         // a pushed support wakes; its island follows at the frame's end
            asleep[s] = 0u;
            sleepTimer[s] = 0u;
        }
        if (asleep[s] == 0u) {
            CoinBody O = coins[s];
            float invMs = O.posInvMass.w;
            coins[s].vel.xyz    = O.vel.xyz - invMs * J;
            coins[s].angVel.xyz = O.angVel.xyz - cdApplyInvInertiaWorld(O.orient, cdBodyInvInertia(O, invMs),
                                                                        cross(p - O.posInvMass.xyz, J));
        }
    }
    state[f].total.xyz = S.total.xyz + J;
}

// ── KERNEL: once per substep, after coinIntegrateVelocityCS (gravity + damping) and before
//    contact generation. ONE threadgroup. Phase 1 (one thread per foot): plant / liftoff /
//    skid and the exact leg impulse. Phase 2: apply them — parallel for feet on statics,
//    then thread 0 alone for feet on dynamic supports. ─────────────────────────────────────
// The substep pass for one threadgroup (every thread of the group must call it: two barriers).
static void cfFootSubstepTG(uint tid, uint tgs,
    device CoinBody*                  coins,
    device float4*                    bias,
    device const CoinFoot*            feet,
    device CoinFootState*             state,
    device const CoinStaticCollider*  colliders,
    device const float2*              material,
    device uint*                      asleep,
    device uint*                      sleepTimer,
    constant CoinFootUniforms&        fu,
    threadgroup atomic_uint&          anyDynamic)
{
    if (tid == 0u) atomic_store_explicit(&anyDynamic, 0u, memory_order_relaxed);
    uint count = min(fu.footCount, CF_MAX_FEET);
    for (uint f = tid; f < count; f += tgs)
        cfFootPrepare(f, coins, feet, state, colliders, material, asleep, fu);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint f = tid; f < count; f += tgs) {
        if (cfOnDynamicSupport(state[f])) atomic_store_explicit(&anyDynamic, 1u, memory_order_relaxed);
        else cfFootApply(f, coins, bias, feet, state, asleep, sleepTimer, fu);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid != 0u || atomic_load_explicit(&anyDynamic, memory_order_relaxed) == 0u) return;
    for (uint f = 0; f < count; ++f)
        if (cfOnDynamicSupport(state[f])) cfFootApply(f, coins, bias, feet, state, asleep, sleepTimer, fu);
}

kernel void coinFootSubstep(
    device CoinBody*                  coins      [[ buffer(0) ]],
    device float4*                    bias       [[ buffer(1) ]],
    device const CoinFoot*            feet       [[ buffer(2) ]],
    device CoinFootState*             state      [[ buffer(3) ]],
    device const CoinStaticCollider*  colliders  [[ buffer(4) ]],
    device const float2*              material   [[ buffer(5) ]],
    device uint*                      asleep     [[ buffer(6) ]],
    device uint*                      sleepTimer [[ buffer(7) ]],
    constant CoinFootUniforms&        fu         [[ buffer(8) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]])
{
    threadgroup atomic_uint anyDynamic;
    cfFootSubstepTG(tid, tgs, coins, bias, feet, state, colliders, material, asleep, sleepTimer, fu, anyDynamic);
}

// One foot's velocity-iteration work: the pad's torsion / yaw-motor row and, on the substep's
// last iteration, the position half of the leg's exact substep (see the header).
static void cfFootIterateOne(uint f, device CoinBody* coins, device float4* bias, device const CoinFoot* feet,
                             device CoinFootState* state, device const uint* asleep,
                             constant CoinFootUniforms& fu, device const CoinStaticCollider* colliders) {
    CoinFoot F = feet[f];
    uint b = F.meta.x;
    if (F.meta.y == 0u || b >= fu.bodyCount || asleep[b] != 0u) return;
    CoinFootState S = state[f];
    if (S.meta.x == 0u) return;
    uint s = S.meta.y;
    bool dyn = s != CD_STATIC && s < fu.bodyCount && asleep[s] == 0u;

    // Torsion / yaw motor about the support normal: drive (ω_body − ω_support)·n to the
    // target, the ACCUMULATED impulse inside ±min(μ_spin·r_patch·Λn, τ_max·dt).
    if (S.row.y > 0.0) {
        CoinBody B = coins[b];
        float3 n = S.nrmW.xyz;
        float3 wS = dyn ? coins[s].angVel.xyz : float3(0.0);
        float wrel = dot(B.angVel.xyz - wS, n);
        float lamOld = S.row.w;
        float lamNew = clamp(lamOld - (wrel - S.row.z) / S.row.x, -S.row.y, S.row.y);
        float dl = lamNew - lamOld;
        if (dl != 0.0) {
            float3 T = dl * n;
            coins[b].angVel.xyz = B.angVel.xyz + cdApplyInvInertiaWorld(B.orient, cdBodyInvInertia(B, B.posInvMass.w), T);
            if (dyn) {
                CoinBody O = coins[s];
                coins[s].angVel.xyz = O.angVel.xyz - cdApplyInvInertiaWorld(O.orient, cdBodyInvInertia(O, O.posInvMass.w), T);
            }
            state[f].row.w = lamNew;
            state[f].total.w = S.total.w + dl;
        }
    }

    // Last iteration: put the POSITION part of the leg's exact substep in through the
    // split-impulse bias — only as far as the leg's DOF moved freely (w → 0 when a contact or
    // joint changed the relative velocity along the leg: then that constraint owns the motion).
    float P = S.imp.x, mEff = S.imp.w;
    if (fu.lastIteration == 0u || P <= 0.0 || mEff <= 0.0) return;
    CoinBody B = coins[b];
    float3 u = S.dir.xyz, p = S.point.xyz;
    float3 rH = cdQuatRotate(B.orient, F.mount.xyz);
    float3 vh = B.vel.xyz + cross(B.angVel.xyz, rH);
    float3 bh = bias[2 * b].xyz + cross(bias[2 * b + 1].xyz, rH);
    float3 vp = float3(0.0), bp = float3(0.0);
    if (dyn) {
        CoinBody O = coins[s];
        float3 rP = p - O.posInvMass.xyz;
        vp = O.vel.xyz + cross(O.angVel.xyz, rP);
        bp = bias[2 * s].xyz + cross(bias[2 * s + 1].xyz, rP);
    } else if (s == CD_STATIC && S.meta.z < fu.colliderCount) {
        vp = cdColliderVelocity(colliders, S.meta.z);            // a host-moved kinematic box
    }
    float sepNow = dot(vh - vp, u);                              // the solver's separation rate now
    float sepExact = -S.imp.z;                                   // what the leg alone would have produced
    float legDv = P / mEff;
    float w = saturate(1.0 - fabs(sepNow - sepExact) / (0.1 * legDv + 1e-4));
    if (w <= 0.0) return;
    float xSolver = S.tipL.w - (sepNow + dot(bh - bp, u)) * fu.dt;
    float D = (S.imp.y - xSolver) * w;                           // exact end compression − the solver's
    float Pb = -mEff * D / fu.dt;                                // bias impulse along u (+ on the body)
    float3 J = Pb * u;
    float invMb = B.posInvMass.w;
    bias[2 * b].xyz += invMb * J;
    bias[2 * b + 1].xyz += cdApplyInvInertiaWorld(B.orient, cdBodyInvInertia(B, invMb), cross(rH, J));
    if (dyn) {
        CoinBody O = coins[s];
        float invMs = O.posInvMass.w;
        bias[2 * s].xyz -= invMs * J;
        bias[2 * s + 1].xyz -= cdApplyInvInertiaWorld(O.orient, cdBodyInvInertia(O, invMs), cross(p - O.posInvMass.xyz, J));
    }
    state[f].nrmW.w = -D;
}

// ── KERNEL: once per velocity iteration (after that iteration's contact colours and joint
//    pass). ONE threadgroup: feet on statics in parallel, then feet on dynamic supports
//    serially on thread 0. ──────────────────────────────────────────────────────────────────
// The velocity-iteration pass for one threadgroup (every thread of the group must call it).
static void cfFootIterateTG(uint tid, uint tgs,
    device CoinBody*                  coins,
    device float4*                    bias,
    device const CoinFoot*            feet,
    device CoinFootState*             state,
    device const uint*                asleep,
    constant CoinFootUniforms&        fu,
    device const CoinStaticCollider*  colliders,
    threadgroup atomic_uint&          anyDynamic)
{
    if (tid == 0u) atomic_store_explicit(&anyDynamic, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = min(fu.footCount, CF_MAX_FEET);
    for (uint f = tid; f < count; f += tgs) {
        if (cfOnDynamicSupport(state[f])) atomic_store_explicit(&anyDynamic, 1u, memory_order_relaxed);
        else cfFootIterateOne(f, coins, bias, feet, state, asleep, fu, colliders);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid != 0u || atomic_load_explicit(&anyDynamic, memory_order_relaxed) == 0u) return;
    for (uint f = 0; f < count; ++f)
        if (cfOnDynamicSupport(state[f])) cfFootIterateOne(f, coins, bias, feet, state, asleep, fu, colliders);
}

kernel void coinFootIterate(
    device CoinBody*                  coins     [[ buffer(0) ]],
    device float4*                    bias      [[ buffer(1) ]],
    device const CoinFoot*            feet      [[ buffer(2) ]],
    device CoinFootState*             state     [[ buffer(3) ]],
    device const uint*                asleep    [[ buffer(4) ]],
    constant CoinFootUniforms&        fu        [[ buffer(5) ]],
    device const CoinStaticCollider*  colliders [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]])
{
    threadgroup atomic_uint anyDynamic;
    cfFootIterateTG(tid, tgs, coins, bias, feet, state, asleep, fu, colliders, anyDynamic);
}

// ── KERNEL: once per island-union round — a planted foot joins its body's island to a
//    DYNAMIC support's, so a worker standing on a plate sleeps and wakes with it. ───────────
static inline void cfFootIslandUnionBody(uint f, device atomic_uint* label, device const CoinFoot* feet,
                                         device const CoinFootState* state, constant CoinFootUniforms& fu)
{
    if (f >= min(fu.footCount, CF_MAX_FEET)) return;
    CoinFoot F = feet[f];
    CoinFootState S = state[f];
    if (F.meta.y == 0u || S.meta.x == 0u) return;
    uint b = F.meta.x, s = S.meta.y;
    if (s == CD_STATIC || s >= fu.bodyCount || b >= fu.bodyCount) return;
    cdUnionLabels(label, b, s);
}

kernel void coinFootIslandUnion(
    device atomic_uint*          label [[ buffer(0) ]],
    device const CoinFoot*       feet  [[ buffer(1) ]],
    device const CoinFootState*  state [[ buffer(2) ]],
    constant CoinFootUniforms&   fu    [[ buffer(3) ]],
    uint f [[ thread_position_in_grid ]])
{
    cfFootIslandUnionBody(f, label, feet, state, fu);
}
