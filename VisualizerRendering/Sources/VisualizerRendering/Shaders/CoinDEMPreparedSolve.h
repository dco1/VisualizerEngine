// ── CoinDEMPreparedSolve.h — PREPARED contact rows (opt-in, stage B3) ──────────────────────────
//
// CoinDEMSolver.preparedContactSolve (CD_FLAG_PREPARED_CONTACTS). Every row of the contact solve
// (cdSolveContactVelocityOne, the manifold loop of cdSolveContactVelocity, cdSolveContactTorsion)
// needs the contact's effective masses — k = m_A⁻¹ + m_B⁻¹ + (r_A×d)·I_A⁻¹(r_A×d) + … — and the
// angular response of each body to a unit impulse along the row, I⁻¹(r×d). Both depend only on the
// bodies' POSES, which do not change during a substep's velocity iterations (positions integrate
// after them), yet the default solve evaluates them on every row of every iteration — and of every
// manifold pass: ~16–22 world-inverse-inertia products (two quaternion rotations each) per contact
// point per pass. Measured on the Digital Clock's perf fence (small-world path, stage B3): the
// crane's 3–4-point box manifold spent ≈ 35 µs per velocity iteration in the solve on one thread
// (224 µs per substep), the p95 frames' largest single cost; the four Weebles' single-point
// contacts ≈ 10 µs per iteration.
//
// Here a PREPARE pass (cdPrepareContactBody, one thread per contact, once per substep after the
// colouring and the restitution capture) stores those quantities per contact point, and the solve
// (cdSolveContactPrepared) reads them: each row is then a dot product, a clamp and an FMA per body.
// The physics is the same — the same rows in the same order, the same clamps, bounds and targets,
// the same accumulated impulses and warm start — but the arithmetic is regrouped
// (I⁻¹(r×(λ·d)) becomes λ·I⁻¹(r×d)), and Metal's fast math rounds the two differently: so it is
// OPT-IN, and the default kernels (coinSolveVelocityColor / Tail, every world that does not set it:
// Daydream Home, the pile scenes) are untouched. A world that sets it runs these kernels on BOTH
// paths (the small-world kernel and the multi-dispatch fallback), so the two stay bit-identical.
//
// Not prepared (computed per row as before, it depends on the current velocities): the LEGACY
// rolling row's axis (the relative rolling spin) — its effective mass along that axis is still
// evaluated per row, as it always was. The accumulated rolling row (CD_FLAG_ACCUM_ROLLING) reads
// its two directions' responses from the prepare too.
//
// plausibility: real — the same sequential-impulse rows, bounds and targets as the default solve;
// only where each pose-constant factor is computed moves (once per substep instead of per row).

constant uint CD_FLAG_PREPARED_CONTACTS = 64u;
constant uint CD_PREP_F4 = 12u;               // float4 per contact point (CoinDEMSolver.contactPrepStride / 16)

// Per contact point, CD_PREP_F4 float4:
//   [0] I_A⁻¹(r_A×n),  kN          [1] I_B⁻¹(r_B×n),  kT1        [2] I_A⁻¹(r_A×t1), kT2
//   [3] I_B⁻¹(r_B×t1), μ (combined)[4] I_A⁻¹(r_A×t2), e (combined) [5] I_B⁻¹(r_B×t2), kTors
//   [6] I_A⁻¹n, r_patch             [7] I_B⁻¹n, 0                  [8] I_A⁻¹t1, k1 (accum. rolling)
//   [9] I_A⁻¹t2, k2                 [10] I_B⁻¹t1, 0                [11] I_B⁻¹t2, 0
// (an asleep end's inverse mass and inertia are 0 — the same as the solve's `aSleep ? 0 : …`).
// NOT inlined (and neither is cdSolveContactPrepared): each kernel that runs the prepared rows —
// the small-world frame, coinPrepareContacts, the per-colour and tail solves — then calls ONE
// compiled body, so the two paths agree bit for bit. Inlined, the copies rounded differently
// (fast math follows the code around them): the small-world and multi-dispatch runs of a tumbling
// box / hull / compound pile parted in the 7th digit of a velocity after 24 frames (measured).
__attribute__((noinline)) static void cdPrepareContactBody(uint cid, uint n,
    device const CoinContact*        contacts,
    device const CoinBody*           coins,
    constant CoinUniforms&           u,
    device const uint*               asleep,
    device const float2*             material,
    device const CoinStaticCollider* colliders,
    device const float*              patch,
    device float4*                   prep)
{
    if (cid >= n) return;
    CoinContact c = contacts[cid];
    uint A = c.meta.x, B = c.meta.y;
    bool bStatic = (B == CD_STATIC);
    bool aSleep = (asleep[A] != 0u);
    bool bSleep = bStatic || (asleep[B] != 0u);
    CoinBody a = coins[A];
    float invMa = aSleep ? 0.0 : a.posInvMass.w;
    float3 invIa = aSleep ? float3(0.0) : cdBodyInvInertia(a, a.posInvMass.w);
    float4 qa = a.orient;
    float invMb = 0.0; float3 invIb = float3(0.0); float4 qb = float4(0, 0, 0, 1);
    if (!bStatic) {
        CoinBody b = coins[B];
        invMb = bSleep ? 0.0 : b.posInvMass.w;
        invIb = bSleep ? float3(0.0) : cdBodyInvInertia(b, b.posInvMass.w);
        qb = b.orient;
    }
    float3 nn = c.nrm.xyz, rA = c.rA.xyz, rB = c.rB.xyz, t1 = c.tan1.xyz, t2 = c.tan2.xyz;
    float3 raXnn = cross(rA, nn), raXn1 = cross(rA, t1), raXn2 = cross(rA, t2);
    float3 rbXnn = cross(rB, nn), rbXn1 = cross(rB, t1), rbXn2 = cross(rB, t2);
    float3 aN = cdApplyInvInertiaWorld(qa, invIa, raXnn), aT1 = cdApplyInvInertiaWorld(qa, invIa, raXn1),
           aT2 = cdApplyInvInertiaWorld(qa, invIa, raXn2);
    float3 bN = float3(0.0), bT1 = float3(0.0), bT2 = float3(0.0);
    if (!bStatic) {
        bN = cdApplyInvInertiaWorld(qb, invIb, rbXnn); bT1 = cdApplyInvInertiaWorld(qb, invIb, rbXn1);
        bT2 = cdApplyInvInertiaWorld(qb, invIb, rbXn2);
    }
    float kN  = max(invMa + invMb + dot(raXnn, aN) + (bStatic ? 0.0 : dot(rbXnn, bN)), 1e-8);
    float kT1 = invMa + invMb + dot(raXn1, aT1) + (bStatic ? 0.0 : dot(rbXn1, bT1));
    float kT2 = invMa + invMb + dot(raXn2, aT2) + (bStatic ? 0.0 : dot(rbXn2, bT2));
    uint colIdx = c.meta.z & 0xFFFFu;
    float muA = cdBodyMu(material, A, u);
    float eA  = cdBodyE(material, A, u);
    float muB = bStatic ? cdColliderMu(colliders, colIdx, u) : cdBodyMu(material, B, u);
    float eB  = bStatic ? cdColliderE(colliders, colIdx, u) : cdBodyE(material, B, u);
    float muC = sqrt(max(muA * muB, 0.0));
    float eC  = max(eA, eB);
    float3 dA = cdApplyInvInertiaWorld(qa, invIa, nn);
    float3 dB = bStatic ? float3(0.0) : cdApplyInvInertiaWorld(qb, invIb, nn);
    float kTors = dot(nn, dA) + dot(nn, dB);
    float rPatch = max(patch[A], bStatic ? 0.0 : patch[B]);
    float3 rA1 = cdApplyInvInertiaWorld(qa, invIa, t1), rA2 = cdApplyInvInertiaWorld(qa, invIa, t2);
    float3 rB1 = bStatic ? float3(0.0) : cdApplyInvInertiaWorld(qb, invIb, t1);
    float3 rB2 = bStatic ? float3(0.0) : cdApplyInvInertiaWorld(qb, invIb, t2);
    float k1 = dot(t1, rA1) + dot(t1, rB1), k2 = dot(t2, rA2) + dot(t2, rB2);
    device float4* P = prep + CD_PREP_F4 * cid;
    P[0] = float4(aN, kN);    P[1] = float4(bN, kT1);  P[2]  = float4(aT1, kT2); P[3]  = float4(bT1, muC);
    P[4] = float4(aT2, eC);   P[5] = float4(bT2, kTors); P[6] = float4(dA, rPatch); P[7] = float4(dB, 0.0);
    P[8] = float4(rA1, k1);   P[9] = float4(rA2, k2);  P[10] = float4(rB1, 0.0); P[11] = float4(rB2, 0.0);
}

// The torsion row about n (CD_FLAG_TORSION): |Λs| ≤ μ·r·Λn·share, accumulated in ext.x
// (cdSolveContactTorsion's law; a manifold point carries 1/n of the patch term).
static inline void cdPrepTorsionRow(device const float4* P, float3 n, float share, bool bStatic,
                                    thread float3& wA, thread float3& wB, thread CoinContact& c) {
    float4 p3 = P[3], p5 = P[5], p6 = P[6], p7 = P[7];
    float tCap = p3.w * max(p6.w, 0.0) * share;
    float kT = p5.w;
    if (!((tCap > 0.0 || c.ext.x != 0.0) && kT > 0.0)) return;
    float tBound = tCap * max(c.rA.w, 0.0);
    float lamOld = c.ext.x;
    float lamNew = clamp(lamOld - dot(wA - wB, n) / kT, -tBound, tBound);
    float dl = lamNew - lamOld;
    wA += dl * p6.xyz;
    if (!bStatic) wB -= dl * p7.xyz;
    c.ext.xy = float2(lamNew, tCap);
}

// The rolling-resistance row: accumulated (CD_FLAG_ACCUM_ROLLING, the disc of radius μᵣ·jₙ·|r_A|)
// or the legacy per-iteration clamp about the current rolling axis.
static inline void cdPrepRollingRow(device const float4* P, float3 n, float3 rA, float3 t1, float3 t2, bool accumRolling,
                                    bool bStatic, device const CoinBody* coins, uint A, uint B, bool aSleep, bool bSleep,
                                    constant CoinUniforms& u,
                                    thread float3& wA, thread float3& wB, thread CoinContact& c) {
    if (!(u.rollingResistance > 0.0)) return;
    if (accumRolling) {
        if (!(c.rA.w > 0.0 || any(c.aux.yz != 0.0))) return;
        float4 p8 = P[8], p9 = P[9], p10 = P[10], p11 = P[11];
        float k1 = p8.w, k2 = p9.w;
        if (!(k1 > 1e-9 && k2 > 1e-9)) return;
        float3 wrel = wA - wB;
        float2 accOld = c.aux.yz;
        float2 acc = accOld - float2(dot(wrel, t1) / k1, dot(wrel, t2) / k2);
        float bound = u.rollingResistance * c.rA.w * max(length(rA), 1e-4);
        float al = length(acc);
        if (al > bound) acc *= bound / al;
        float2 d = acc - accOld;
        wA += d.x * p8.xyz + d.y * p9.xyz;
        if (!bStatic) wB -= d.x * p10.xyz + d.y * p11.xyz;
        c.aux.yz = acc;
        return;
    }
    if (!(c.rA.w > 0.0)) return;
    float3 wrel = wA - wB;
    float3 wRoll = wrel - dot(wrel, n) * n;
    float wl = length(wRoll);
    if (!(wl > 1e-5)) return;
    float3 axis = wRoll / wl;
    // The legacy row's axis follows the spin, so its response is not prepared: the poses are read
    // here, where the row needs them, rather than held in registers through the point loop.
    CoinBody a = coins[A];
    float3 invIa = aSleep ? float3(0.0) : cdBodyInvInertia(a, a.posInvMass.w);
    float3 ra = cdApplyInvInertiaWorld(a.orient, invIa, axis);
    float3 rb = float3(0.0);
    if (!bStatic) {
        CoinBody b = coins[B];
        float3 invIb = bSleep ? float3(0.0) : cdBodyInvInertia(b, b.posInvMass.w);
        rb = cdApplyInvInertiaWorld(b.orient, invIb, axis);
    }
    float kR = dot(axis, ra) + dot(axis, rb);
    if (!(kR > 1e-9)) return;
    float jR = min(wl / kR, u.rollingResistance * c.rA.w * max(length(rA), 1e-4));
    wA -= jR * ra;
    if (!bStatic) wB += jR * rb;
}

// The prepared solve of one colorContacts entry: a single contact (its rows + its torsion row) or a
// grouped manifold (manifoldPasses Gauss–Seidel passes over its points) — cdSolveContactVelocity's
// rows and order, with the pose-constant factors read from `prep`.
__attribute__((noinline)) static void cdSolveContactPrepared(
    uint                             entry,
    device CoinBody*                 coins,
    device float4*                   bias,
    device CoinContact*              contacts,
    constant CoinUniforms&           u,
    device const uint*               asleep,
    device const CoinStaticCollider* colliders,
    device const float4*             prep)
{
    uint cid0 = entry & CD_CC_MASK;
    uint nPts = max(entry >> CD_CC_SHIFT, 1u);
    uint4 meta0 = contacts[cid0].meta;
    uint A = meta0.x, B = meta0.y;
    bool bStatic = (B == CD_STATIC);
    bool aSleep = (asleep[A] != 0u);
    bool bSleep = bStatic || (asleep[B] != 0u);
    if (aSleep && bSleep) return;

    CoinBody a = coins[A];
    float invMa = aSleep ? 0.0 : a.posInvMass.w;
    float3 vA = a.vel.xyz, wA = a.angVel.xyz;
    float3 bvA = bias[2*A].xyz, bwA = bias[2*A+1].xyz;
    float invMb = 0.0;
    float3 vB = float3(0.0), wB = float3(0.0), bvB = float3(0.0), bwB = float3(0.0);
    uint colIdx = meta0.z & 0xFFFFu;
    float3 vStatic = bStatic ? cdColliderVelocity(colliders, colIdx) : float3(0.0);
    if (!bStatic) {
        CoinBody b = coins[B];
        invMb = bSleep ? 0.0 : b.posInvMass.w;
        vB = b.vel.xyz; wB = b.angVel.xyz; bvB = bias[2*B].xyz; bwB = bias[2*B+1].xyz;
    }
    bool torsionOn = (u.solverFlags & CD_FLAG_TORSION) != 0u;
    bool accumRolling = (u.solverFlags & CD_FLAG_ACCUM_ROLLING) != 0u;
    // A grouped manifold shares the patch term: 1/n on each point (cdSolveContactVelocity); a
    // single contact carries it whole (cdSolveContactTorsion).
    float share = 1.0 / float(nPts);
    uint passes = (nPts == 1u) ? 1u : max(u.manifoldPasses, 1u);
    float invDt = 1.0 / max(u.dt, 1e-6);

    // Each point's fields are read fresh on every visit and only what the rows change is written
    // back (the impulses, the restitution capture and rolling accumulators, the torsion lanes): the
    // point loop runs on one thread, and whatever it keeps live through a pass — a whole contact
    // struct, both bodies' poses for the rolling row — is register pressure on that thread.
    for (uint pass = 0u; pass < passes; ++pass) {
    for (uint k = 0u; k < nPts; ++k) {
        device CoinContact& cr = contacts[cid0 + k];
        CoinContact c;
        c.nrm = cr.nrm; c.rA = cr.rA; c.rB = cr.rB; c.tan1 = cr.tan1; c.tan2 = cr.tan2; c.aux = cr.aux; c.ext = cr.ext;
        device const float4* P = prep + CD_PREP_F4 * (cid0 + k);
        float4 p0 = P[0], p1 = P[1], p2 = P[2], p3 = P[3], p4 = P[4], p5 = P[5];
        bool lastPass = (pass + 1u == passes);
        float3 n = c.nrm.xyz, rA = c.rA.xyz, rB = c.rB.xyz, t1 = c.tan1.xyz, t2 = c.tan2.xyz;
        float depth = c.nrm.w;
        float kN = p0.w, kT1 = p1.w, kT2 = p2.w, muC = p3.w, eC = p4.w;

        // Normal (restitution anchored to the pre-solve approach speed, speculative cap).
        float3 vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
        float vn = dot(vrel, n);
        if (c.aux.w < 0.5) { c.aux.x = vn; c.aux.w = 1.0; }
        float vn0 = c.aux.x;
        float restE = (vn0 < -u.restThreshold) ? cdEffectiveCORBase(u, eC, -vn0) : 0.0;
        float allowedVn = min(depth, 0.0) * invDt;
        float jnOld = c.rA.w;
        float jnNew = max(jnOld - (vn - allowedVn + restE * vn0) / kN, 0.0);
        float dJn = jnNew - jnOld;
        vA += (invMa * dJn) * n; wA += dJn * p0.xyz;
        if (!bStatic) { vB -= (invMb * dJn) * n; wB -= dJn * p1.xyz; }
        c.rA.w = jnNew;

        // Split-impulse bias (not accumulated).
        float bvn = dot((bvA + cross(bwA, rA)) - (bStatic ? float3(0.0) : (bvB + cross(bwB, rB))), n);
        float dJb = cdSeparatingBias((u.baumgarteBeta * max(depth - u.contactSlop, 0.0) * invDt - bvn) / kN, depth, bvn, kN, u);
        bvA += (invMa * dJb) * n; bwA += dJb * p0.xyz;
        if (!bStatic) { bvB -= (invMb * dJb) * n; bwB -= dJb * p1.xyz; }

        // Two-axis Coulomb friction, |jt| ≤ μ·jn.
        if (muC > 0.0) {
            float bound = muC * c.rA.w;
            vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
            float jt1Old = c.rB.w;
            float jt1New = clamp(jt1Old - dot(vrel, t1) / max(kT1, 1e-8), -bound, bound);
            float d1 = jt1New - jt1Old;
            vA += (invMa * d1) * t1; wA += d1 * p2.xyz;
            if (!bStatic) { vB -= (invMb * d1) * t1; wB -= d1 * p3.xyz; }
            c.rB.w = jt1New;
            vrel = (vA + cross(wA, rA)) - (bStatic ? vStatic : (vB + cross(wB, rB)));
            float jt2Old = c.tan1.w;
            float jt2New = clamp(jt2Old - dot(vrel, t2) / max(kT2, 1e-8), -bound, bound);
            float d2 = jt2New - jt2Old;
            vA += (invMa * d2) * t2; wA += d2 * p4.xyz;
            if (!bStatic) { vB -= (invMb * d2) * t2; wB -= d2 * p5.xyz; }
            c.tan1.w = jt2New;
        }

        // Torsion and rolling in each path's original order: a grouped manifold solves torsion then
        // rolling inside its point loop (cdSolveContactVelocity); a single contact runs its rolling row
        // in cdSolveContactVelocityOne and its torsion row after it (cdSolveContactTorsion).
        if (nPts > 1u) {
            if (torsionOn) cdPrepTorsionRow(P, n, share, bStatic, wA, wB, c);
            if (lastPass) cdPrepRollingRow(P, n, rA, t1, t2, accumRolling, bStatic, coins, A, B, aSleep, bSleep, u, wA, wB, c);
        } else {
            cdPrepRollingRow(P, n, rA, t1, t2, accumRolling, bStatic, coins, A, B, aSleep, bSleep, u, wA, wB, c);
            if (torsionOn) cdPrepTorsionRow(P, n, share, bStatic, wA, wB, c);
        }
        cr.rA = c.rA; cr.rB = c.rB; cr.tan1 = c.tan1; cr.aux = c.aux; cr.ext = c.ext;
    }   // points
    }   // passes

    // Write back (an inert/asleep end carried no impulse). bias[2i+1].w: the motor marker, kept.
    if (!aSleep) {
        coins[A].vel.xyz = vA; coins[A].angVel.xyz = wA;
        bias[2*A] = float4(bvA, 0.0); bias[2*A+1] = float4(bwA, bias[2*A+1].w);
    }
    if (!bSleep) {
        coins[B].vel.xyz = vB; coins[B].angVel.xyz = wB;
        bias[2*B] = float4(bvB, 0.0); bias[2*B+1] = float4(bwB, bias[2*B+1].w);
    }
}

// cdSolveVelocityTailTG with the prepared solve (the same passes: colours from `firstColor`, the
// empty colours above the last coloured contact skipped, then the uncoloured bucket's sub-colours).
static void cdSolveVelocityTailTGPrep(uint firstColor, uint tid, uint tgSize,
    device CoinBody*                 coins,
    device float4*                   bias,
    device CoinContact*              contacts,
    device const uint*               colorContacts,
    constant CoinUniforms&           u,
    device const uint*               asleep,
    device const CoinStaticCollider* colliders,
    device const uint*               colorOffset,
    device const uint*               uncolSub,
    device const float4*             prep)
{
    uint first = min(firstColor, CD_UNCOLORED_BUCKET);
    if (colorOffset[first] >= colorOffset[CD_UNCOLORED_BUCKET + 1u]) return;
    uint b0 = colorOffset[CD_UNCOLORED_BUCKET], b1 = colorOffset[CD_UNCOLORED_BUCKET + 1u];
    uint nSub = colorOffset[CD_UNCOLORED_BUCKET + 2u];
    uint nColourPasses = CD_UNCOLORED_BUCKET - first;
    uint nPasses = nColourPasses + (b1 > b0 ? nSub : 0u);
    for (uint pass = 0u; pass < nPasses; ++pass) {
        uint start, end, sub = 0xFFFFFFFFu;
        if (pass < nColourPasses) {
            start = colorOffset[first + pass];
            if (start >= b0) { pass = nColourPasses - 1u; continue; }   // the colours above are empty
            end = colorOffset[first + pass + 1u];
        } else { start = b0; end = b1; sub = pass - nColourPasses; }
        if (start >= end) continue;
        for (uint k = start + tid; k < end; k += tgSize)
            if (sub == 0xFFFFFFFFu || uncolSub[k] == sub)
                cdSolveContactPrepared(colorContacts[k], coins, bias, contacts, u, asleep, colliders, prep);
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ── Multi-dispatch kernels (the same bindings as their default twins, + the prepare buffer) ──

kernel void coinPrepareContacts(
    device const CoinContact*        contacts     [[ buffer(0) ]],
    device const atomic_uint&        contactCount [[ buffer(1) ]],
    device const CoinBody*           coins        [[ buffer(2) ]],
    constant CoinUniforms&           u            [[ buffer(3) ]],
    device const uint*               asleep       [[ buffer(4) ]],
    device const float2*             material     [[ buffer(5) ]],
    device const CoinStaticCollider* colliders    [[ buffer(6) ]],
    device const float*              patch        [[ buffer(7) ]],
    device float4*                   prep         [[ buffer(8) ]],
    uint cid [[ thread_position_in_grid ]])
{
    cdPrepareContactBody(cid, atomic_load_explicit(&contactCount, memory_order_relaxed), contacts, coins, u, asleep,
                         material, colliders, patch, prep);
}

kernel void coinSolveVelocityColorPrep(
    device CoinBody*                 coins         [[ buffer(0) ]],
    device float4*                   bias          [[ buffer(1) ]],
    device CoinContact*              contacts      [[ buffer(2) ]],
    device const uint*               colorContacts [[ buffer(3) ]],
    constant CoinUniforms&           u             [[ buffer(4) ]],
    constant uint&                   currentColor  [[ buffer(5) ]],
    device const uint*               asleep        [[ buffer(6) ]],
    device const CoinStaticCollider* colliders     [[ buffer(8) ]],
    device const uint*               colorOffset   [[ buffer(9) ]],
    device const float4*             prep          [[ buffer(12) ]],
    uint i [[ thread_position_in_grid ]])
{
    uint start = colorOffset[currentColor], end = colorOffset[currentColor + 1u];
    if (start + i >= end) return;
    cdSolveContactPrepared(colorContacts[start + i], coins, bias, contacts, u, asleep, colliders, prep);
}

kernel void coinSolveVelocityTailPrep(
    device CoinBody*                 coins         [[ buffer(0) ]],
    device float4*                   bias          [[ buffer(1) ]],
    device CoinContact*              contacts      [[ buffer(2) ]],
    device const uint*               colorContacts [[ buffer(3) ]],
    constant CoinUniforms&           u             [[ buffer(4) ]],
    constant uint&                   firstColor    [[ buffer(5) ]],
    device const uint*               asleep        [[ buffer(6) ]],
    device const CoinStaticCollider* colliders     [[ buffer(8) ]],
    device const uint*               colorOffset   [[ buffer(9) ]],
    device const uint*               uncolSub      [[ buffer(10) ]],
    device const float4*             prep          [[ buffer(12) ]],
    uint tid    [[ thread_index_in_threadgroup ]],
    uint tgSize [[ threads_per_threadgroup ]])
{
    cdSolveVelocityTailTGPrep(firstColor, tid, tgSize, coins, bias, contacts, colorContacts, u, asleep, colliders,
                              colorOffset, uncolSub, prep);
}
