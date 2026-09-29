// ── CoinDEMSmallWorld.h — the SMALL-WORLD frame kernel (engine plan item 3c, stage B3) ─────────
//
// A constraint-path frame of a small world — a few dozen bodies, a few hundred contacts, the
// Digital Clock's toy crane and crew — costs almost nothing in arithmetic and a great deal in
// DISPATCHES: each substep of the multi-dispatch path (CoinDEMSolver.encodeConstraintSubstep) is
// ~40 dependent dispatches plus ~5 per velocity iteration, and every one of them drains the GPU
// before the next can start. Measured on the clock's fence world (47 bodies, ~12 contacts,
// 1/180 s × 6 iterations at 30 fps): ≈ 70 dispatches per substep, 5–9 ms of GPU per frame, for
// perhaps a millisecond of actual work.
//
// So a small world runs its WHOLE FRAME here, in ONE threadgroup: every substep (velocity
// integration, spring feet, contact generation incl. the polytope narrowphase, the colouring,
// restitution capture, warm start, the joint blocks, the colour-by-colour velocity solve with its
// torsion / foot / joint rows, the warm-start snapshot, position integration and finalize), then
// the island-sleep update and the render transforms — one dispatch per frame. Phases are separated
// by threadgroup barriers with device-memory ordering, which is what a single threadgroup needs to
// see its own device writes (the solve's serial tail pass has always relied on it).
//
// ONE IMPLEMENTATION. Every phase calls the SAME device function the multi-dispatch kernel is a
// thin wrapper around (cdGenerateContactsBody, cdColorRoundBody, cdSolveVelocityTailTG,
// cdJointSolveTG, cfFootIterateTG, …), in the same order, with the same per-substep decisions the
// host makes there (CdSWParams). What differs, and why the results still match:
//   • Broadphase: a ONE-CELL grid (the uniforms' grid is overridden to 1 × 1 × 1 cell with
//     invCell 0, and the cell lists every body in index order), so each body's thread tests every
//     higher-index body with the same bounding reject — the contact SET the grid neighbourhood
//     finds (the grid is sized so every pair within reach shares a ±K neighbourhood, VZ-0161),
//     with no clear / count / scan / scatter phases. Only the append ORDER differs, and nothing
//     downstream depends on it: colour priorities hash each contact's identity, a manifold is
//     written in canonical order, the uncoloured bucket is sorted, warm-start matching compares
//     whole identities.
//   • Colours: every colour is solved by the fused loop cdSolveVelocityTailTG runs for the tail
//     (first colour 0) — the same per-contact function, the same colour order; the host's per-
//     colour split never changed a bit of the result (VZ-0154). `beyondSweep` stays 0 (there is
//     no per-colour dispatch to fall beyond).
// CoinDEMSmallWorldTests steps the same worlds both ways and compares every body bit for bit.
//
// WHEN. Opt-in (CoinDEMSolver.smallWorldPath), and only while the world is small: ≤ 64 bodies and
// ≤ 2048 contacts at the last completed frame (CoinDEMSolver+SmallWorld.swift gives the
// measurements behind the limits). A world that outgrows them — or uses the legacy path, leveling
// or a test seam the kernel does not model — falls back to the multi-dispatch path by itself.
//
// THREADGROUP MEMORY: 25 264 bytes (the pipeline's staticThreadgroupMemoryLength, asserted ≤ 32 KB
// by CoinDEMSmallWorldTests) — one scratch region shared by the phases that need one (the joint
// solve's cached blocks + bodies, 24 960 B, the largest — resident across a substep's velocity
// iterations, which use no other scratch; the uncoloured bucket's staging, 9 216 B), plus 8 atomics,
// a flag and the colouring's 64 per-round "still uncoloured" flags. (The joint prepare needs none:
// its factors are registers.) Under the 32 KB of every A11-or-later GPU and every Apple silicon Mac
// — the class the engine already needed: the multi-dispatch joint solve's threadgroup cache
// (coinJointSolveCS) is 24 976 B, so the 16 KB GPUs (A8–A10X: Apple TV HD, Apple TV 4K 1st
// generation) could not run the constraint path before this kernel either. This pipeline is
// OPTIONAL (Pipelines.smallWorld): where it cannot be built the path is simply never taken. It
// compiles for tvOS as it is (xcrun -sdk appletvos metal): Apple TV 4K 2nd generation (A12) and
// later run it.

// Host-written parameters of one frame (setBytes; mirrors CoinSmallWorldParamsGPU in
// CoinDEMSolver+SmallWorld.swift). 224 bytes.
struct CdSWParams {
    uint steps;                 // substeps this frame
    uint velocityIterations;
    uint colorScheme;           // 0 = Jones–Plassmann (colorRounds rounds), 1 = speculative
    uint colorRounds;           // rounds of the scheme in use
    uint warmStart;             // 0 / 1
    uint clearHashFirst;        // 1: the pair hash was never cleared (first warm-started substep)
    uint hashSize;
    uint torsionOn;             // CD_FLAG_TORSION is set (the warm start carries ext.x)
    uint maxContacts;
    uint maxPolyPairs;          // the polytope pair list's capacity; 0 = no list (no polytope body)
    uint jointCount;            // the frame's ACTIVE joints (coinJointListUpload's list length)
    uint jointTGBodies;         // > 0: the joint solve's threadgroup cache tables are in the list
    uint jointPasses;
    uint footCount;             // the foot table's high-water mark; 0 = no foot work this frame
    uint footEveryIteration;    // 1: the pads' rows every velocity iteration, else the last only
    uint sleepEnabled;          // run the island-sleep update after the last substep
    uint sleepFrames;
    uint islandUnionRounds;
    uint phaseMask;             // TEST SEAM (cost attribution): CD_SW_PH_* bits; all set in use
    uint firstColor;            // 0: the colour sweeps start at colour 0 (the tail functions' first colour)
    // Transient scratch: byte offsets of each array in buffer `arena` (16-byte aligned).
    uint offSorted, offCellOffsets, offBodyContacts, offBodyContactCount;
    uint offPriority, offColor0, offColor1, offBid;
    uint offColorCount, offColorOffset, offColorContacts, offUncolSub;
    uint offPolyPairs, offIslandLabel, offIslandMin, offSleepHub;
    uint offJointedBody, offPolySurv, pad2, pad3;   // offPolySurv: the SAT pairs past their first face (uint4 × maxPolyPairs)
    CoinFootUniforms footU[2];  // [0] = any velocity iteration but the last, [1] = the last one
};
static_assert(sizeof(CdSWParams) == 224, "CdSWParams must match CoinSmallWorldParamsGPU (CoinDEMSolver+SmallWorld.swift)");

// Phase bits of CdSWParams.phaseMask — a TEST SEAM for per-part cost attribution only (a frame
// with a phase switched off is not a valid simulation). Every scene runs with all bits set.
constant uint CD_SW_PH_FEET      = 1u << 0;   // the spring-foot substep pass
constant uint CD_SW_PH_GENERATE  = 1u << 1;   // contact generation (dynamic pairs + statics)
constant uint CD_SW_PH_POLY      = 1u << 2;   // the polytope narrowphase
constant uint CD_SW_PH_COLOR     = 1u << 3;   // body lists + colouring rounds + buckets
constant uint CD_SW_PH_WARM      = 1u << 4;   // restitution capture + warm-start match / apply
constant uint CD_SW_PH_JPREP     = 1u << 5;   // joint prepare (blocks + warm start)
constant uint CD_SW_PH_SOLVE     = 1u << 6;   // the per-iteration contact colour sweep
constant uint CD_SW_PH_SLEEP     = 1u << 7;   // the island-sleep update
constant uint CD_SW_PH_INTVEL    = 1u << 8;   // velocity integration
constant uint CD_SW_PH_INTPOS    = 1u << 9;   // position integration + finalize
constant uint CD_SW_PH_FEETIT    = 1u << 10;  // the spring-foot iteration pass
constant uint CD_SW_PH_JSOLVE    = 1u << 11;  // the per-iteration joint solve
constant uint CD_SW_PH_SNAPSHOT  = 1u << 12;  // the warm-start snapshot
constant uint CD_SW_PH_TRANSFORM = 1u << 13;  // the render transforms
constant uint CD_SW_PH_PREP      = 1u << 14;  // the prepared contact rows (CD_FLAG_PREPARED_CONTACTS)

// The shared threadgroup scratch, in float4s: the cached joint solve's CD_JSOLVE_MAXJ blocks
// (31 float4 each) + CD_JSOLVE_MAXB × 4 body float4s = CD_JSOLVE_TG_BYTES — the largest phase.
constant uint CD_SW_TG_F4 = CD_JSOLVE_TG_BYTES / 16u;
static_assert(sizeof(CoinJointPrep) == 31u * 16u, "the small-world scratch carve assumes 31 float4 per joint block");

#define CD_SW_SYNC() threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)
#define CD_SW_ON(bit) ((sp.phaseMask & (bit)) != 0u)

// ── The phases ─────────────────────────────────────────────────────────────────────────────
//
// Each phase is a function the kernel calls in order; every thread of the group enters every
// phase (several contain barriers). Two compile-shape facts decide how they are written — both
// measured, and together they are what makes this kernel bit-identical to the multi-dispatch path
// (CoinDEMSmallWorldTests):
//   • The phases are INLINED, so the frame's uniforms `u` stay what they are in every standalone
//     kernel: a constant kernel argument. The compiler evaluates arithmetic on such UNIFORM values
//     on its own and rounds it separately; passed into a non-inlined function as a pointer, the same
//     expression is per-thread and gets contracted into a fused multiply-add. Measured: with the
//     phases as real calls, the velocity integration's `v.y -= u.gravity * u.dt` came out as
//     fma(−g, dt, v) (bits …e3) where coinIntegrateVelocityCS computes v − round(g·dt) (…e2) — a
//     1-ulp launch-velocity difference a pile then amplifies (and the quaternion integration's
//     0.5·dt did the same) — while inlined, every world stepped bit for bit.
//   • EXCEPT the polytope narrowphase, which is a real call (cdSWPolyNarrow): inlined into the
//     same function as cdGenerateContactsBody it changed how the latter compiled — a box resting on
//     a plane emitted 1 of its 4 corner contacts (the plane branch's gather of feature points into
//     its per-thread arrays), though the narrowphase never even ran; with the narrowphase a real
//     call, all 4. (Its own arithmetic reads only uniform tolerances, and the polytope worlds
//     still step bit for bit.)
#define CD_SW_PHASE static inline void

CD_SW_PHASE cdSWIntegrateVelocity(uint tid, uint tgs, uint nb, device CoinBody* coins, device float4* bias,
                                  constant CoinUniforms& u, device const uint* asleep)
{
    for (uint id = tid; id < nb; id += tgs) cdIntegrateVelocityBody(id, coins, bias, u, asleep);
}

CD_SW_PHASE cdSWFootSubstep(uint tid, uint tgs, device CoinBody* coins, device float4* bias,
                            device const CoinFoot* feet, device CoinFootState* footState,
                            device const CoinStaticCollider* colliders, device const float2* material,
                            device uint* asleep, device uint* sleepTimer, constant CdSWParams& sp,
                            threadgroup atomic_uint& anyDynamic)
{
    cfFootSubstepTG(tid, tgs, coins, bias, feet, footState, colliders, material, asleep, sleepTimer, sp.footU[0], anyDynamic);
}

CD_SW_PHASE cdSWGenerate(uint tid, uint tgs, uint nb,
    device const CoinBody* coins, device const uint* sorted, device const uint* cellOffsets,
    device const CoinStaticCollider* colliders, constant CoinUniforms& u, device const int2* links,
    device CoinContact* contacts, device atomic_uint* contactCount, constant CdSWParams& sp,
    device const float4* hullVerts, device const uint2* hullRanges, device const CoinJoint* joints,
    device const uint* asleep, device const uint* jointedBody, device const uint* jointList,
    device uint4* polyPairs, device atomic_uint* polyPairCount)
{
    // Each body split into `parts` work items (cdGenerateContactsBody), one pass of the group.
    uint parts = max(1u, tgs / max(nb, 1u));
    for (uint t = tid; t < nb * parts; t += tgs)
        cdGenerateContactsBody(t % nb, t / nb, parts, coins, sorted, cellOffsets, colliders, u, links, contacts,
                               contactCount, sp.maxContacts, hullVerts, hullRanges, joints, sp.jointCount, asleep,
                               jointedBody, jointList, polyPairs, polyPairCount, sp.maxPolyPairs);
}

// The polytope narrowphase (VZ-0198), in two passes with one barrier:
//   1. one THREAD per listed pair: the swept-sphere / GJK kinds whole (cdPolyNarrowThreadBody);
//      for the kinds that run the separating-axis test, its first face (cdSATPairFirstFace) —
//      where nearly every listed pair ends (the clock's jib against every bar under it: 56
//      listed, 0 touching) — and a pair that survives it is appended to `surv` with its f0 / s0;
//   2. one SIMD GROUP per survivor: the rest of its SAT on every lane, the manifold on lane 0
//      (cdSATPairSolveSG).
// Before: one thread per pair for everything — a touching box pair walked 144 edge pairs on one
// thread through stack-memory arrays (≈ 0.12 ms per substep); and a SIMD group per LISTED pair
// took 7 rounds of 8 groups for the clock's 56 far-apart pairs (≈ 70 µs per substep).
__attribute__((noinline)) static void cdSWPolyNarrow(uint tid, uint tgs, uint lane, uint sg, uint nsg, uint w,
                           device const CoinBody* coins, device const CoinStaticCollider* colliders,
                           constant CoinUniforms& u, device CoinContact* contacts, device atomic_uint* contactCount,
                           constant CdSWParams& sp, device const float4* hullVerts, device const uint2* hullRanges,
                           device const uint4* polyPairs, device atomic_uint* polyPairCount,
                           device uint4* surv, threadgroup atomic_uint& survCount)
{
    uint np = min(atomic_load_explicit(polyPairCount, memory_order_relaxed), sp.maxPolyPairs);
    bool serial = (u.solverFlags & CD_FLAG_POLY_SERIAL) != 0u;
    if (tid == 0u) atomic_store_explicit(&survCount, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < np; p += tgs) {
        cdPolyNarrowThreadBody(p, np, coins, u, contacts, contactCount, sp.maxContacts, hullVerts, hullRanges, polyPairs);
        CdSATPair Q;
        if (!cdSATPairSetup(p, np, coins, colliders, hullVerts, hullRanges, polyPairs, Q)) continue;
        uint f0 = 0u; float s0 = 0.0;
        if (!serial && !cdSATPairFirstFace(Q, u, hullVerts, f0, s0)) continue;
        uint k = atomic_fetch_add_explicit(&survCount, 1u, memory_order_relaxed);
        surv[k] = uint4(p, f0, as_type<uint>(s0), 0u);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    uint ns = atomic_load_explicit(&survCount, memory_order_relaxed);
    for (uint k = sg; k < ns; k += nsg) {
        uint4 e = surv[k];
        CdSATPair Q;
        cdSATPairSetup(e.x, np, coins, colliders, hullVerts, hullRanges, polyPairs, Q);   // (a SAT pair — pass 1 listed it)
        cdSATPairSolveSG(Q, e.y, as_type<float>(e.z), serial, lane, w, u, contacts, contactCount, sp.maxContacts, hullVerts);
    }
}

// Body lists, the colouring rounds, the colour buckets and the uncoloured bucket's sub-colours.
//
// The rounds stop WORKING once a round leaves nothing uncoloured: every later round would copy
// each colour unchanged (a coloured contact keeps its colour — see coinColorRound / Resolve), so
// the colours are bit-identical to running them all, and a small world converges in a few of the
// scene's rounds (the clock's 16 Jones–Plassmann rounds cost ≈ 10 µs each on one threadgroup).
// Each round has its own flag (roundLeft[r], cleared on entry), set by any contact the round
// leaves uncoloured and read by every thread after the round's barrier, so the stop is uniform
// and no flag is reused while another thread may still read it. Past CD_SW_MAX_FLAGGED_ROUNDS
// the rounds just run.
constant uint CD_SW_MAX_FLAGGED_ROUNDS = 64u;
CD_SW_PHASE cdSWColor(uint tid, uint tgs, uint w, uint nb, uint nc, constant CdSWParams& sp,
    device CoinContact* contacts, device uint* bodyContacts, device uint* bodyCount, device atomic_uint* bodyCountA,
    device uint* priority, device uint* color0, device uint* color1, device uint* bid,
    device uint* colorCount, device atomic_uint* colorCountA, device uint* colorOffset, device uint* colorContacts,
    device uint* uncolSub, device atomic_uint* colorStats,
    threadgroup uint* sCid, threadgroup uint* sPri, threadgroup uint4* sId, threadgroup uint2* sAB, threadgroup uint* sSub,
    threadgroup atomic_uint* sUsed, threadgroup uint& sFail, threadgroup atomic_uint* roundLeft)
{
    for (uint b = tid; b < nb; b += tgs) bodyCount[b] = 0u;
    for (uint c = tid; c <= CD_UNCOLORED_BUCKET; c += tgs) colorCount[c] = 0u;
    for (uint r = tid; r < CD_SW_MAX_FLAGGED_ROUNDS; r += tgs) atomic_store_explicit(&roundLeft[r], 0u, memory_order_relaxed);
    CD_SW_SYNC();
    for (uint cid = tid; cid < nc; cid += tgs) {
        cdBuildBodyContactsBody(cid, nc, contacts, bodyContacts, bodyCountA, colorStats);
        cdColorInitBody(cid, nc, contacts, priority, color0);
    }
    CD_SW_SYNC();
    device uint* cIn = color0;
    device uint* cOut = color1;
    uint left = 1u;                                          // uniform: some contact may still be uncoloured
    if (sp.colorScheme == 1u) {
        for (uint r = 0u; r < sp.colorRounds; ++r) {
            if (left == 0u) continue;
            for (uint cid = tid; cid < nc; cid += tgs)
                cdColorTentativeBody(cid, nc, contacts, bodyContacts, bodyCount, priority, cIn, bid);
            CD_SW_SYNC();
            for (uint cid = tid; cid < nc; cid += tgs) {
                cdColorResolveBody(cid, nc, contacts, bodyContacts, bodyCount, priority, cIn, bid, cOut);
                if (cOut[cid] == CD_COLOR_NONE) atomic_store_explicit(&roundLeft[min(r, CD_SW_MAX_FLAGGED_ROUNDS - 1u)], 1u, memory_order_relaxed);
            }
            CD_SW_SYNC();
            device uint* t = cIn; cIn = cOut; cOut = t;
            left = (r < CD_SW_MAX_FLAGGED_ROUNDS - 1u) ? atomic_load_explicit(&roundLeft[r], memory_order_relaxed) : 1u;
        }
    } else {
        for (uint r = 0u; r < sp.colorRounds; ++r) {
            if (left == 0u) continue;
            for (uint cid = tid; cid < nc; cid += tgs) {
                cdColorRoundBody(cid, nc, contacts, bodyContacts, bodyCount, priority, cIn, cOut);
                if (cOut[cid] == CD_COLOR_NONE) atomic_store_explicit(&roundLeft[min(r, CD_SW_MAX_FLAGGED_ROUNDS - 1u)], 1u, memory_order_relaxed);
            }
            CD_SW_SYNC();
            device uint* t = cIn; cIn = cOut; cOut = t;
            left = (r < CD_SW_MAX_FLAGGED_ROUNDS - 1u) ? atomic_load_explicit(&roundLeft[r], memory_order_relaxed) : 1u;
        }
    }
    for (uint cid = tid; cid < nc; cid += tgs) {
        cdColorWritebackBody(cid, nc, contacts, cIn, colorStats, CD_MAX_COLORS);   // nothing is "beyond"
        cdColorBucketCountBody(cid, nc, cIn, colorCountA);
    }
    CD_SW_SYNC();
    if (tid < w) cdColorBucketScanSG(tid, w, colorCount, colorOffset);   // the first SIMD group
    CD_SW_SYNC();
    for (uint cid = tid; cid < nc; cid += tgs)
        cdColorBucketScatterBody(cid, nc, cIn, colorCountA, colorOffset, colorContacts, contacts);
    CD_SW_SYNC();
    cdUncolouredBucketTG(tid, tgs, colorOffset, colorContacts, priority, contacts, uncolSub, colorStats,
                         sCid, sPri, sId, sAB, sSub, sUsed, sFail);
}

CD_SW_PHASE cdSWCaptureApproach(uint tid, uint tgs, uint nc, device CoinContact* contacts, device const CoinBody* coins,
                                device const CoinStaticCollider* colliders)
{
    for (uint cid = tid; cid < nc; cid += tgs) cdCaptureApproachBody(cid, nc, contacts, coins, colliders);
}

// The warm start: (first use) clear the pair hash, match, then apply colour by colour.
CD_SW_PHASE cdSWWarmStart(uint tid, uint tgs, uint nc, bool clearHash, constant CdSWParams& sp,
    device CoinBody* coins, device CoinContact* contacts, device const CoinContact* prevContacts,
    device atomic_uint* pairHashA, device const uint* pairHash, device const uint* colorContacts,
    device const uint* colorOffset, device const uint* asleep, device const uint* uncolSub)
{
    if (clearHash) {
        for (uint i = tid; i < sp.hashSize; i += tgs) atomic_store_explicit(&pairHashA[i], CD_HASH_EMPTY, memory_order_relaxed);
        CD_SW_SYNC();
    }
    for (uint cid = tid; cid < nc; cid += tgs)
        cdWarmStartMatchBody(cid, nc, contacts, prevContacts, pairHash, sp.hashSize, coins, sp.torsionOn);
    CD_SW_SYNC();
    cdWarmStartApplyTailTG(sp.firstColor, tid, tgs, coins, contacts, colorContacts, colorOffset, asleep, uncolSub);
}

CD_SW_PHASE cdSWJointPrepare(uint tid, uint tgs, uint w, device CoinBody* coins, device float4* bias, device const CoinJoint* joints,
                             device uint* jointList, constant CdSWParams& sp, constant CoinUniforms& u,
                             device const uint* asleep, device CoinJointPrep* jointPrep)
{
    cdJointPrepareTG(tid, tgs, w, coins, bias, joints, jointList, sp.jointCount, u, asleep, jointPrep);
}

// Every colour, in order, then the uncoloured bucket's sub-colours: coinSolveVelocityTail's own
// function, from colour sp.firstColor (0).
CD_SW_PHASE cdSWSolve(uint tid, uint tgs, constant CdSWParams& sp, device CoinBody* coins, device float4* bias,
                      device CoinContact* contacts,
                      device const uint* colorContacts, constant CoinUniforms& u, device const uint* asleep,
                      device const float2* material, device const CoinStaticCollider* colliders,
                      device const uint* colorOffset, device const uint* uncolSub, device const float* patch,
                      device const float4* contactPrep)
{
    if ((u.solverFlags & CD_FLAG_PREPARED_CONTACTS) != 0u)
        cdSolveVelocityTailTGPrep(sp.firstColor, tid, tgs, coins, bias, contacts, colorContacts, u, asleep, colliders,
                                  colorOffset, uncolSub, contactPrep);
    else
        cdSolveVelocityTailTG(sp.firstColor, tid, tgs, coins, bias, contacts, colorContacts, u, asleep, material, colliders,
                              colorOffset, uncolSub, patch);
}

// The blocks stay resident in the threadgroup scratch across the substep's velocity iterations
// (staged on the first, written back on the last — cdJointSolveTG): nothing between them uses it.
CD_SW_PHASE cdSWJointSolve(uint tid, uint tgs, device CoinBody* coins, device float4* bias, device const uint* jointList,
                           constant CdSWParams& sp, device CoinJointPrep* jointPrep,
                           threadgroup CoinJointPrep* tgP, threadgroup float4* tgB, threadgroup atomic_uint& tgStale,
                           bool firstIteration, bool lastIteration)
{
    cdJointSolveTG(tid, tgs, coins, bias, jointList, sp.jointCount, jointPrep, sp.jointPasses, sp.jointTGBodies,
                   tgP, tgB, tgStale, firstIteration, lastIteration);
}

CD_SW_PHASE cdSWFootIterate(uint tid, uint tgs, bool last, device CoinBody* coins, device float4* bias,
                            device const CoinFoot* feet, device CoinFootState* footState, device const uint* asleep,
                            constant CdSWParams& sp, device const CoinStaticCollider* colliders,
                            threadgroup atomic_uint& anyDynamic)
{
    cfFootIterateTG(tid, tgs, coins, bias, feet, footState, asleep, sp.footU[last ? 1 : 0], colliders, anyDynamic);
}

// The warm-start snapshot: the solved contacts and a fresh pair hash for the next substep.
CD_SW_PHASE cdSWSnapshot(uint tid, uint tgs, uint nc, constant CdSWParams& sp, device const CoinContact* contacts,
                         device CoinContact* prevContacts, device atomic_uint* pairHashA)
{
    for (uint i = tid; i < sp.hashSize; i += tgs) atomic_store_explicit(&pairHashA[i], CD_HASH_EMPTY, memory_order_relaxed);
    CD_SW_SYNC();
    for (uint cid = tid; cid < nc; cid += tgs) cdSnapshotContactBody(cid, nc, contacts, prevContacts, pairHashA, sp.hashSize);
}

// Position (real + bias velocity), then finalize — both per body, so one pass.
CD_SW_PHASE cdSWIntegratePosition(uint tid, uint tgs, uint nb, device CoinBody* coins, device const float4* bias,
                                  constant CoinUniforms& u, device const uint* asleep)
{
    for (uint id = tid; id < nb; id += tgs) {
        cdIntegratePositionBody(id, coins, bias, u, asleep);
        cdFinalizeBody(id, coins, u, asleep, bias);
    }
}

// Island sleep (once per frame, on the last substep's contacts): union-find over contacts,
// joints, planted feet and the asleep sleep groups; tick; per-island minimum; mark.
CD_SW_PHASE cdSWSleep(uint tid, uint tgs, uint nb, constant CdSWParams& sp, constant CoinUniforms& u,
    device CoinBody* coins, device uint* asleep, device const CoinContact* contacts, device atomic_uint* contactCount,
    device const CoinJoint* joints, device const uint* jointList, device const CoinFoot* feet,
    device const CoinFootState* footState, device uint* sleepTimer, device uint* sleepKey, device const float4* bias,
    device uint* label, device atomic_uint* labelA, device uint* islandMin, device atomic_uint* islandMinA,
    device uint* sleepHub, device atomic_uint* sleepHubA)
{
    uint nc = atomic_load_explicit(contactCount, memory_order_relaxed);
    for (uint id = tid; id < nb; id += tgs) cdIslandInitBody(id, label, u, sleepHub);
    CD_SW_SYNC();
    for (uint id = tid; id < nb; id += tgs) cdIslandSleepHubBody(id, coins, asleep, sleepKey, sleepHubA, u);
    CD_SW_SYNC();
    uint nUnion = max(nc, nb);
    for (uint r = 0u; r < sp.islandUnionRounds; ++r) {
        for (uint cid = tid; cid < nUnion; cid += tgs)
            cdIslandUnionBody(cid, nc, labelA, contacts, coins, asleep, sleepKey, sleepHub, u);
        CD_SW_SYNC();
        if (sp.jointCount > 0u) {
            for (uint k = tid; k < sp.jointCount; k += tgs) cdIslandUnionJointBody(k, labelA, joints, sp.jointCount, jointList);
            CD_SW_SYNC();
        }
        if (sp.footCount > 0u) {
            for (uint f = tid; f < sp.footCount; f += tgs) cfFootIslandUnionBody(f, labelA, feet, footState, sp.footU[0]);
            CD_SW_SYNC();
        }
        for (uint id = tid; id < nb; id += tgs) cdIslandJumpBody(id, label, u);
        CD_SW_SYNC();
    }
    for (uint id = tid; id < nb; id += tgs) cdSleepTickBody(id, coins, sleepTimer, islandMin, u, bias);
    CD_SW_SYNC();
    for (uint id = tid; id < nb; id += tgs) cdIslandMinReduceBody(id, label, sleepTimer, islandMinA, u);
    CD_SW_SYNC();
    for (uint id = tid; id < nb; id += tgs) cdSleepMarkBody(id, label, islandMin, asleep, u, sp.sleepFrames, coins, sleepKey);
}

kernel void coinSmallWorldFrame(
    device CoinBody*                  coins         [[ buffer(0) ]],
    device float4*                    bias          [[ buffer(1) ]],
    constant CoinUniforms&            u             [[ buffer(2) ]],   // the frame's uniforms over a ONE-CELL grid
    device uint*                      asleep        [[ buffer(3) ]],
    device const CoinStaticCollider*  colliders     [[ buffer(4) ]],
    device const int2*                links         [[ buffer(5) ]],
    device CoinContact*               contacts      [[ buffer(6) ]],
    device atomic_uint*               contactCount  [[ buffer(7) ]],
    device const float4*              hullVerts     [[ buffer(8) ]],
    device const uint2*               hullRanges    [[ buffer(9) ]],
    device const CoinJoint*           joints        [[ buffer(10) ]],
    device uint*                      jointList     [[ buffer(11) ]],   // + the substep's active list (CD_JLIST_ACTIVE)
    device atomic_uint*               colorStats    [[ buffer(12) ]],
    device const float2*              material      [[ buffer(13) ]],
    device const float*               patch         [[ buffer(14) ]],
    device CoinContact*               prevContacts  [[ buffer(15) ]],
    device atomic_uint*               pairHashA     [[ buffer(16) ]],   // the warm-start hash: inserts / clears …
    device CoinJointPrep*             jointPrep     [[ buffer(17) ]],
    device uint*                      sleepTimer    [[ buffer(18) ]],
    device uint*                      sleepKey      [[ buffer(19) ]],
    device CoinTransform*             transforms    [[ buffer(20) ]],
    device const CoinFoot*            feet          [[ buffer(21) ]],
    device CoinFootState*             footState     [[ buffer(22) ]],
    constant CdSWParams&              sp            [[ buffer(23) ]],
    device uchar*                     arena         [[ buffer(24) ]],   // transient scratch (offsets in sp)
    device atomic_uint*               polyPairCount [[ buffer(25) ]],   // shared: CoinDEMSolver.polyPairCount reads it
    device const uint*                pairHash      [[ buffer(26) ]],   // … and the same buffer, read by the matches
    device float4*                    contactPrep   [[ buffer(27) ]],   // prepared contact rows (CD_FLAG_PREPARED_CONTACTS)
    uint tid   [[ thread_position_in_threadgroup ]],
    uint tgs   [[ threads_per_threadgroup ]],      // a whole number of SIMD groups (the host rounds it)
    uint lane  [[ thread_index_in_simdgroup ]],
    uint sg    [[ simdgroup_index_in_threadgroup ]],
    uint nsg   [[ simdgroups_per_threadgroup ]],
    uint simdW [[ threads_per_simdgroup ]])
{
    threadgroup float4      tgArena[CD_SW_TG_F4];
    threadgroup atomic_uint tgAtom[8];     // [0] joint-cache stale flag, [1] foot anyDynamic, [2..5] bucket used masks, [6] SAT survivors
    threadgroup uint        tgFail;        // the uncoloured bucket's failure flag
    threadgroup atomic_uint tgRoundLeft[CD_SW_MAX_FLAGGED_ROUNDS];   // cdSWColor's per-round "still uncoloured" flags

    device uint*        sorted        = reinterpret_cast<device uint*>(arena + sp.offSorted);
    device uint*        cellOffsets   = reinterpret_cast<device uint*>(arena + sp.offCellOffsets);
    device uint*        bodyContacts  = reinterpret_cast<device uint*>(arena + sp.offBodyContacts);
    device uint*        bodyCount     = reinterpret_cast<device uint*>(arena + sp.offBodyContactCount);
    device atomic_uint* bodyCountA    = reinterpret_cast<device atomic_uint*>(arena + sp.offBodyContactCount);
    device uint*        priority      = reinterpret_cast<device uint*>(arena + sp.offPriority);
    device uint*        color0        = reinterpret_cast<device uint*>(arena + sp.offColor0);
    device uint*        color1        = reinterpret_cast<device uint*>(arena + sp.offColor1);
    device uint*        bid           = reinterpret_cast<device uint*>(arena + sp.offBid);
    device uint*        colorCount    = reinterpret_cast<device uint*>(arena + sp.offColorCount);
    device atomic_uint* colorCountA   = reinterpret_cast<device atomic_uint*>(arena + sp.offColorCount);
    device uint*        colorOffset   = reinterpret_cast<device uint*>(arena + sp.offColorOffset);
    device uint*        colorContacts = reinterpret_cast<device uint*>(arena + sp.offColorContacts);
    device uint*        uncolSub      = reinterpret_cast<device uint*>(arena + sp.offUncolSub);
    device uint4*       polyPairs     = reinterpret_cast<device uint4*>(arena + sp.offPolyPairs);
    device uint*        label         = reinterpret_cast<device uint*>(arena + sp.offIslandLabel);
    device atomic_uint* labelA        = reinterpret_cast<device atomic_uint*>(arena + sp.offIslandLabel);
    device uint*        islandMin     = reinterpret_cast<device uint*>(arena + sp.offIslandMin);
    device atomic_uint* islandMinA    = reinterpret_cast<device atomic_uint*>(arena + sp.offIslandMin);
    device uint*        sleepHub      = reinterpret_cast<device uint*>(arena + sp.offSleepHub);
    device atomic_uint* sleepHubA     = reinterpret_cast<device atomic_uint*>(arena + sp.offSleepHub);
    device uint*        jointedBody   = reinterpret_cast<device uint*>(arena + sp.offJointedBody);
    device uint4*       polySurv      = reinterpret_cast<device uint4*>(arena + sp.offPolySurv);

    // Carves of the shared threadgroup scratch (one phase uses it at a time).
    threadgroup CoinJointPrep* tgP     = reinterpret_cast<threadgroup CoinJointPrep*>(tgArena);
    threadgroup float4*        tgB     = tgArena + CD_JSOLVE_MAXJ * 31u;
    threadgroup uint*          sCid    = reinterpret_cast<threadgroup uint*>(tgArena);
    threadgroup uint*          sPri    = reinterpret_cast<threadgroup uint*>(tgArena + 64u);
    threadgroup uint4*         sId     = reinterpret_cast<threadgroup uint4*>(tgArena + 128u);
    threadgroup uint2*         sAB     = reinterpret_cast<threadgroup uint2*>(tgArena + 384u);
    threadgroup uint*          sSub    = reinterpret_cast<threadgroup uint*>(tgArena + 512u);

    const uint nb = u.coinCount;

    // ── Once per frame: the one-cell grid (every slot in cell 0, index order) and the bodies
    //    that belong to an enabled two-body joint (coinMarkJointedBodies).
    for (uint i = tid; i < nb; i += tgs) sorted[i] = i;
    if (tid == 0u) { cellOffsets[0] = 0u; cellOffsets[1] = nb; }
    if (sp.jointCount > 0u)
        for (uint id = tid; id < nb; id += tgs) cdMarkJointedBody(id, joints, sp.jointCount, jointedBody, u, jointList);
    CD_SW_SYNC();

    for (uint step = 0u; step < sp.steps; ++step) {
        // (1) Velocity: gravity, drag, damping; the split-impulse bias cleared.
        if (CD_SW_ON(CD_SW_PH_INTVEL)) cdSWIntegrateVelocity(tid, tgs, nb, coins, bias, u, asleep);
        CD_SW_SYNC();

        // (2) Spring feet: plant / liftoff / skid and each planted leg's exact substep impulse.
        if (sp.footCount > 0u && CD_SW_ON(CD_SW_PH_FEET)) {
            cdSWFootSubstep(tid, tgs, coins, bias, feet, footState, colliders, material, asleep, sleepTimer, sp, tgAtom[1]);
            CD_SW_SYNC();
        }

        // (3) Contacts: every body's pairs + statics, then the listed polytope pairs.
        if (tid == 0u && CD_SW_ON(CD_SW_PH_GENERATE)) {
            atomic_store_explicit(contactCount, 0u, memory_order_relaxed);
            atomic_store_explicit(polyPairCount, 0u, memory_order_relaxed);
        }
        CD_SW_SYNC();
        if (CD_SW_ON(CD_SW_PH_GENERATE))
            cdSWGenerate(tid, tgs, nb, coins, sorted, cellOffsets, colliders, u, links, contacts, contactCount, sp,
                         hullVerts, hullRanges, joints, asleep, jointedBody, jointList, polyPairs, polyPairCount);
        CD_SW_SYNC();
        if (sp.maxPolyPairs > 0u && CD_SW_ON(CD_SW_PH_POLY)) {
            cdSWPolyNarrow(tid, tgs, lane, sg, nsg, simdW, coins, colliders, u, contacts, contactCount, sp, hullVerts, hullRanges,
                           polyPairs, polyPairCount, polySurv, tgAtom[6]);
            CD_SW_SYNC();
        }
        // (3b) CD_FLAG_KEEP_ASLEEP_CONTACTS: a sleeping island's last contacts, carried as dormant
        //      records (the warm start's memory — coinCarryDormant), once the pair hash exists.
        if ((u.solverFlags & CD_FLAG_KEEP_ASLEEP_CONTACTS) != 0u && sp.warmStart != 0u && sp.clearHashFirst == 0u
            && CD_SW_ON(CD_SW_PH_GENERATE)) {
            for (uint i = tid; i < sp.hashSize; i += tgs)
                cdCarryDormantBody(i, sp.hashSize, pairHash, prevContacts, coins, asleep, contacts, contactCount, sp.maxContacts);
            CD_SW_SYNC();
        }
        // The append cursor clamped to the buffer (coinWriteContactArgs): from here on it is the
        // number of contacts that exist.
        uint nc = min(atomic_load_explicit(contactCount, memory_order_relaxed), sp.maxContacts);
        CD_SW_SYNC();
        if (tid == 0u) atomic_store_explicit(contactCount, nc, memory_order_relaxed);

        // (4) Colouring.
        if (CD_SW_ON(CD_SW_PH_COLOR)) {
            cdSWColor(tid, tgs, simdW, nb, nc, sp, contacts, bodyContacts, bodyCount, bodyCountA, priority, color0, color1, bid,
                      colorCount, colorCountA, colorOffset, colorContacts, uncolSub, colorStats,
                      sCid, sPri, sId, sAB, sSub, &tgAtom[2], tgFail, tgRoundLeft);
        } else if (CD_SW_ON(CD_SW_PH_GENERATE)) {
            // (Test seam only.) Contacts generated this dispatch but not coloured: nothing to solve.
            // A per-phase profile dispatch (generation off too) keeps the colouring the colour
            // phase's own dispatch wrote, so its solve / warm start measure real work.
            for (uint c = tid; c < CD_UNCOLORED_BUCKET + 3u; c += tgs) colorOffset[c] = 0u;
        }
        CD_SW_SYNC();

        // (5) Restitution capture, before the first impulse (manifold solve).
        if ((u.solverFlags & CD_FLAG_MANIFOLD_SOLVE) != 0u && CD_SW_ON(CD_SW_PH_WARM)) {
            cdSWCaptureApproach(tid, tgs, nc, contacts, coins, colliders);
            CD_SW_SYNC();
        }

        // (5b) Prepared contact rows (opt-in): each contact's pose-constant factors, once.
        if ((u.solverFlags & CD_FLAG_PREPARED_CONTACTS) != 0u && CD_SW_ON(CD_SW_PH_PREP)) {
            for (uint cid = tid; cid < nc; cid += tgs)
                cdPrepareContactBody(cid, nc, contacts, coins, u, asleep, material, colliders, patch, contactPrep);
            CD_SW_SYNC();
        }

        // (6) Warm start: last substep's impulses onto the matching fresh contacts, applied colour
        //     by colour.
        if (sp.warmStart != 0u && CD_SW_ON(CD_SW_PH_WARM)) {
            cdSWWarmStart(tid, tgs, nc, step == 0u && sp.clearHashFirst != 0u, sp, coins, contacts, prevContacts,
                          pairHashA, pairHash, colorContacts, colorOffset, asleep, uncolSub);
            CD_SW_SYNC();
        }

        // (7) Joints: each active joint's block for this substep's poses, then their warm start.
        if (sp.jointCount > 0u && CD_SW_ON(CD_SW_PH_JPREP)) {
            cdSWJointPrepare(tid, tgs, simdW, coins, bias, joints, jointList, sp, u, asleep, jointPrep);
            CD_SW_SYNC();
        }

        // (8) Velocity iterations: the contact colours, the joint blocks, the pads' rows.
        for (uint it = 0u; it < sp.velocityIterations; ++it) {
            if (CD_SW_ON(CD_SW_PH_SOLVE))
                cdSWSolve(tid, tgs, sp, coins, bias, contacts, colorContacts, u, asleep, material, colliders, colorOffset,
                          uncolSub, patch, contactPrep);
            CD_SW_SYNC();
            if (sp.jointCount > 0u && CD_SW_ON(CD_SW_PH_JSOLVE)) {
                cdSWJointSolve(tid, tgs, coins, bias, jointList, sp, jointPrep, tgP, tgB, tgAtom[0],
                               it == 0u, it + 1u == sp.velocityIterations);
                CD_SW_SYNC();
            }
            bool last = (it + 1u == sp.velocityIterations);
            if (sp.footCount > 0u && (last || sp.footEveryIteration != 0u) && CD_SW_ON(CD_SW_PH_FEETIT)) {
                cdSWFootIterate(tid, tgs, last, coins, bias, feet, footState, asleep, sp, colliders, tgAtom[1]);
                CD_SW_SYNC();
            }
        }

        // (9) Warm-start snapshot.
        if (sp.warmStart != 0u && CD_SW_ON(CD_SW_PH_SNAPSHOT)) {
            cdSWSnapshot(tid, tgs, nc, sp, contacts, prevContacts, pairHashA);
            CD_SW_SYNC();
        }

        // (10) Position, finalize.
        if (CD_SW_ON(CD_SW_PH_INTPOS)) cdSWIntegratePosition(tid, tgs, nb, coins, bias, u, asleep);
        CD_SW_SYNC();
    }

    if (sp.sleepEnabled != 0u && CD_SW_ON(CD_SW_PH_SLEEP)) {
        cdSWSleep(tid, tgs, nb, sp, u, coins, asleep, contacts, contactCount, joints, jointList, feet, footState,
                  sleepTimer, sleepKey, bias, label, labelA, islandMin, islandMinA, sleepHub, sleepHubA);
        CD_SW_SYNC();
    }

    // ── Render transforms.
    if (CD_SW_ON(CD_SW_PH_TRANSFORM))
        for (uint id = tid; id < nb; id += tgs) cdDeriveTransformBody(id, coins, transforms, u);
}

#undef CD_SW_SYNC
#undef CD_SW_ON
#undef CD_SW_PHASE
