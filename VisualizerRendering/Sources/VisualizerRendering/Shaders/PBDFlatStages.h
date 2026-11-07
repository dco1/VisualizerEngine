// PBDFlatStages.h — the fully batched tube substep (DH-1623). Included ONCE, at the end of
// PBD.metal, because it reuses that file's structs and helpers (PBDParticle, sdfContactBatch,
// TubeExpandUniforms, pbdTubeExpandRing). Not a standalone header.
//
// WHY: `PBDTubeBatch` used to run integrate / build-colliders / copy-to-flat / copy-from-flat /
// floor / tube-expand as ONE DISPATCH PER TUBE (8–48 threads each), ~10 per tube per substep.
// A serial compute encoder runs those nearly back to back, so a substep cost ~0.16 ms per tube
// whatever the tube did — 160 franks ≈ 1 650 dispatches ≈ 26 ms a substep. These kernels run
// each of those stages ONCE over every tube's particles in the batch's flat buffer, so a
// substep is O(colour groups) dispatches, not O(tubes).
//
// Every kernel here computes exactly what its per-tube twin computed, in the same order per
// particle (see each one); the A/B lever is `PBDTubeBatch.batchesEveryStage`.
#pragma once

// One tube of the flat batch. `particles` is the tube's OWN particle buffer (a GPU address the
// host made resident with useResources) — the flat buffer is gathered from it at the start of
// a substep and scattered back at the end, so everything that reads or writes a tube's own
// buffer between substeps (spawns, CPU reads, other scenes' passes) is unchanged.
struct PBDFlatTube {
    device PBDParticle* particles;
    uint start;          // first index of this tube in the flat buffer
    uint count;          // particles in this tube (≥ 2)
    uint colliderBase;   // first of this tube's (count − 1) capsules in the collider buffer
    uint ownerID;        // the solver's ownerID (stamped into prevPositionAndPad.w)
};

struct PBDFlatBuildUniforms {
    uint  tubeCount;
    float radius;        // capsule radius (every tube's, homogeneous by eligibility)
    float reach;         // capsule radius + the SDF pass's selfRadius + a rounding margin
    uint  _pad;
};

struct PBDSDFBroadUniforms {
    uint  totalParticles;
    uint  colliderCount;
    float selfRadius;
    float collideStiffness;
    float collideFriction;
    uint  tubeCount;
    uint  staticCount;   // colliders [0, staticCount) are the scene's statics, always tested
    uint  countTests;    // != 0 → add this pass's tests to `counters` (diagnostic only)
};

// ── Gather: every tube's own buffer → the flat buffer, ownerID stamped into .w ──────────────
// Same bytes `pbdCopyToFlat` wrote, one dispatch for every tube.
kernel void pbdFlatGather(
    device       PBDParticle*  flat   [[ buffer(0) ]],
    device const PBDFlatTube*  tubes  [[ buffer(1) ]],
    device const uint*         tubeOf [[ buffer(2) ]],
    constant     uint&         total  [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= total) return;
    PBDFlatTube t = tubes[tubeOf[id]];
    PBDParticle p = t.particles[id - t.start];
    p.prevPositionAndPad.w = as_type<float>(t.ownerID);
    flat[id] = p;
}

// ── Scatter: the flat buffer → every tube's own buffer, stamp cleared ───────────────────────
// Same bytes `pbdCopyFromFlat` wrote.
kernel void pbdFlatScatter(
    device const PBDParticle*  flat   [[ buffer(0) ]],
    device const PBDFlatTube*  tubes  [[ buffer(1) ]],
    device const uint*         tubeOf [[ buffer(2) ]],
    constant     uint&         total  [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= total) return;
    PBDFlatTube t = tubes[tubeOf[id]];
    PBDParticle p = flat[id];
    p.prevPositionAndPad.w = 0.0;
    t.particles[id - t.start] = p;
}

// ── Capsule colliders for every tube, plus each tube's broadphase box ───────────────────────
// One thread per TUBE: writes the same (count − 1) capsules `pbdBuildCapsuleColliders` wrote
// (a = particle i, b = particle i+1, radius, ownerID) at the same collider slots, and the
// tube's axis-aligned box over those capsule endpoints. `bounds[2t].w` carries the reach a
// probe must come within for any of these capsules to touch it.
kernel void pbdBuildCapsuleCollidersFlat(
    device const PBDParticle*        flat   [[ buffer(0) ]],
    device       PBDCollider*        target [[ buffer(1) ]],
    device const PBDFlatTube*        tubes  [[ buffer(2) ]],
    device       float4*             bounds [[ buffer(3) ]],
    constant     PBDFlatBuildUniforms& u    [[ buffer(4) ]],
    uint t [[ thread_position_in_grid ]]
) {
    if (t >= u.tubeCount) return;
    PBDFlatTube tb = tubes[t];
    float3 lo = flat[tb.start].positionAndInvMass.xyz;
    float3 hi = lo;
    for (uint i = 0u; i + 1u < tb.count; ++i) {
        float3 a = flat[tb.start + i].positionAndInvMass.xyz;
        float3 b = flat[tb.start + i + 1u].positionAndInvMass.xyz;
        lo = min(lo, b); hi = max(hi, b);
        PBDCollider c;
        c.a    = float4(a, as_type<float>((uint)1));
        c.b    = float4(b, u.radius);
        c.meta = uint4(tb.ownerID, 0u, 0u, 0u);
        target[tb.colliderBase + i] = c;
    }
    bounds[2u * t]      = float4(lo, u.reach);
    bounds[2u * t + 1u] = float4(hi, 0.0);
}

// ── SDF contact with a per-tube broadphase ──────────────────────────────────────────────────
// `pbdSDFCollideBatch` tested every probe against EVERY collider: particles × colliders, the
// quadratic term the frank cap was holding down. Here a particle first tests each OTHER tube's
// box (one test per tube, not per capsule per probe) and keeps the tubes its probes can reach;
// then each probe runs the statics, then the kept tubes' capsules, in ascending collider order
// — exactly the old loop with the colliders that cannot touch it left out. `sdfContactBatch`
// returns before touching pos/prev for any capsule farther than its radius, and the box is
// conservative (closest point on a segment lies inside its endpoints' box; `reach` adds a
// margin for rounding), so the skipped calls were no-ops: the result is the old kernel's.
// The own tube is skipped by index: its capsules carry this particle's ownerID, which
// `sdfContactBatch` already skips. More than kMaxCandidates reachable tubes → the old full
// loop, unchanged.
constant uint kMaxCandidates = 48u;

kernel void pbdSDFCollideBroad(
    device       PBDParticle*        particles [[ buffer(0) ]],
    device const PBDCollider*        colliders [[ buffer(1) ]],
    constant     PBDSDFBroadUniforms& u        [[ buffer(2) ]],
    device const PBDFlatTube*        tubes     [[ buffer(3) ]],
    device const uint*               tubeOf    [[ buffer(4) ]],
    device const float4*             bounds    [[ buffer(5) ]],
    device       atomic_uint*        counters  [[ buffer(6) ]],   // [0] narrow tests, [1] box tests
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.totalParticles) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned

    float3 pos   = p.positionAndInvMass.xyz;
    float3 prev  = p.prevPositionAndPad.xyz;
    uint   ownID = as_type<uint>(p.prevPositionAndPad.w);
    bool   hit   = false;

    uint myT = tubeOf[id];
    uint tubeStart = tubes[myT].start;
    uint tubeEnd   = tubeStart + tubes[myT].count;

    float3 probes[3];
    uint   numProbes = 0u;
    probes[numProbes++] = pos;
    if (id > tubeStart) {
        probes[numProbes++] = (particles[id - 1u].positionAndInvMass.xyz + pos) * 0.5;
    }
    if (id + 1u < tubeEnd) {
        probes[numProbes++] = (pos + particles[id + 1u].positionAndInvMass.xyz) * 0.5;
    }
    float3 lo = probes[0], hi = probes[0];
    for (uint k = 1u; k < numProbes; ++k) { lo = min(lo, probes[k]); hi = max(hi, probes[k]); }

    uint cand[kMaxCandidates];
    uint nc = 0u;
    bool overflow = false;
    for (uint t = 0u; t < u.tubeCount; ++t) {
        if (t == myT) continue;
        float4 bl = bounds[2u * t];
        float3 bh = bounds[2u * t + 1u].xyz;
        float  r  = bl.w;
        if (any(hi < bl.xyz - r) || any(lo > bh + r)) continue;
        if (nc == kMaxCandidates) { overflow = true; break; }
        cand[nc++] = t;
    }

    uint narrow = 0u;
    for (uint k = 0u; k < numProbes; ++k) {
        for (uint c = 0u; c < u.staticCount; ++c) {
            sdfContactBatch(probes[k], colliders[c], ownID,
                            u.selfRadius, u.collideStiffness, u.collideFriction, pos, prev, hit);
        }
        narrow += u.staticCount;
        if (overflow) {
            for (uint c = u.staticCount; c < u.colliderCount; ++c) {
                sdfContactBatch(probes[k], colliders[c], ownID,
                                u.selfRadius, u.collideStiffness, u.collideFriction, pos, prev, hit);
            }
            narrow += u.colliderCount - u.staticCount;
        } else {
            for (uint j = 0u; j < nc; ++j) {
                PBDFlatTube tb = tubes[cand[j]];
                uint end = tb.colliderBase + tb.count - 1u;
                for (uint c = tb.colliderBase; c < end; ++c) {
                    sdfContactBatch(probes[k], colliders[c], ownID,
                                    u.selfRadius, u.collideStiffness, u.collideFriction, pos, prev, hit);
                }
                narrow += tb.count - 1u;
            }
        }
    }

    if (hit) {
        particles[id].positionAndInvMass.xyz = pos;
        particles[id].prevPositionAndPad.xyz  = prev;
    }
    if (u.countTests != 0u) {
        atomic_fetch_add_explicit(&counters[0], narrow, memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[1], u.tubeCount, memory_order_relaxed);
    }
}

// ── Tube expand for every tube in one dispatch ──────────────────────────────────────────────
// Grid = (visual rings, tubes). Tube t reads its spine straight out of the flat buffer and
// writes its slot of the shared batched mesh — the same `pbdTubeExpandRing` the per-tube
// kernel runs, with the per-tube uniforms from `uniforms[t]`.
kernel void pbdTubeExpandBatch(
    device const PBDParticle*    flat        [[ buffer(0) ]],
    device packed_float3*        verts       [[ buffer(1) ]],
    constant TubeExpandUniforms* uniforms    [[ buffer(2) ]],
    device packed_float3*        normals     [[ buffer(3) ]],
    device float4*               colors      [[ buffer(4) ]],
    device const PBDFlatTube*    tubes       [[ buffer(5) ]],
    device const uint*           vertexBase  [[ buffer(6) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint t  = gid.y;
    uint vb = vertexBase[t];
    pbdTubeExpandRing(flat + tubes[t].start, verts + vb, uniforms[t], normals + vb, colors + vb, gid.x);
}
