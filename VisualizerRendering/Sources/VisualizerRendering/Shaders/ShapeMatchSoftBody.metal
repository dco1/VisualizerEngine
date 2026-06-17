#include <metal_stdlib>
using namespace metal;

// ── SHAPE-MATCH SOFT BODY ────────────────────────────────────────────────────
//
// GPU kernels for a volumetric, highly-compressible XPBD soft body built from
// OVERLAPPING shape-matching clusters (Müller et al. 2005, "Meshless Deformations
// Based on Shape Matching"), with the rotation extracted per cluster by the
// robust analytic polar decomposition of Müller 2016 ("A Robust Method to Extract
// the Rotational Part of Deformations"), warm-started across frames.
//
// Why shape matching and not distance+volume XPBD: a stuffed-plush body needs to
// compress a LOT (fold flat) and spring back to a fixed rest pose with overshoot.
// Shape matching gives that for free — the per-cluster rest pose IS the goal, a
// single per-body stiffness `beta` dials squishy↔springy, and underdamped Verlet
// relaxation produces the re-inflation overshoot + secondary jiggle with no extra
// machinery. The clusters overlap (a particle belongs to several) so the body
// deforms locally (a limb folds independently) while the whole bear still recovers.
//
// All N bodies share ONE flat particle buffer; body b occupies the contiguous
// slice [b·P, (b+1)·P) where P = particlesPerBody. The rest pose (restLocal,
// cluster membership, cluster rest centroids) is TEMPLATE data shared by every
// body — kernels index it by the LOCAL particle/cluster index (global % P, etc.)
// and add the body base to reach the per-instance world state. Cluster rotations
// live in a per-(body,cluster) state buffer, warm-started frame to frame.
//
// ALIGNMENT RULE (see PBDSolver.swift): every Swift↔Metal shared struct uses
// float4 / packed_float3, never a bare float3.

// One simulation particle. pos.w = invMass (0 = pinned), prev.w = bodyID (float).
struct SMParticle {
    float4 pos;
    float4 prev;
};

// Per-(body,cluster) state, written by sm_cluster_fit, read by sm_apply_goals
// and sm_skin_write. centroid.xyz = current world centroid; quat = rotation
// (x,y,z,w), warm-started across frames.
struct SMClusterState {
    float4 centroid;
    float4 quat;
};

// Per-body dynamic parameters (the controller drives the compress→inflate ease
// here). p.x = beta (shape-match stiffness), p.y = restScale (isotropic rest-pose
// scale; <1 vacuum-packed, 1 full, transient >1 overshoot), p.z = gravityScale
// (floaty fall), p.w = unused.
struct SMBodyParams {
    float4 p;
};

struct SMUniforms {
    uint  particleCount;     // N·P
    uint  particlesPerBody;  // P
    uint  clusterCount;      // C (per body)
    uint  bodyCount;         // N
    float dt;
    float damping;
    float gravity;
    float floorY;            // hard safety floor (void) — clamp below this
    uint  colliderCount;
    float collideStiffness;
    float collideFriction;
    float _pad0;
};

// Box collider — same {float4 a; float4 b; uint4 meta} layout as PBDCollider so
// the scene reuses PBDCollider.box(...). a.xyz = center, b.xyz = halfExtents,
// meta.x = kind (2 = box; we only consume boxes here).
struct SMCollider {
    float4 a;
    float4 b;
    uint4  meta;
};

struct SMHashUniforms {
    uint  particleCount;
    uint  tableSize;
    float cellSize;
    float radius;          // particle collision radius (separate to 2·radius)
    uint  particlesPerBody;
    uint  _pad0, _pad1, _pad2;
};

// ── Quaternion helpers (x,y,z,w) ─────────────────────────────────────────────

static inline float4 quatMul(float4 a, float4 b) {
    return float4(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

// Rotate vector v by unit quaternion q.
static inline float3 quatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float  s = q.w;
    return 2.0 * dot(u, v) * u + (s*s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

// ── Analytic Polar Decomposition (Müller 2016) ───────────────────────────────
//
// Extract the rotation quaternion that best aligns the rest frame to the current
// covariance `A` (3 columns a0,a1,a2 = Σ (x-c) ⊗ (rest-r0)). `q` is warm-started.
static float4 extractRotation(float3 a0, float3 a1, float3 a2, float4 q) {
    for (int it = 0; it < 16; ++it) {
        // Columns of R(q).
        float3 r0 = quatRotate(q, float3(1, 0, 0));
        float3 r1 = quatRotate(q, float3(0, 1, 0));
        float3 r2 = quatRotate(q, float3(0, 0, 1));
        float denom = fabs(dot(r0, a0) + dot(r1, a1) + dot(r2, a2)) + 1.0e-9;
        float3 omega = (cross(r0, a0) + cross(r1, a1) + cross(r2, a2)) / denom;
        float w = length(omega);
        if (w < 1.0e-9) break;
        float3 axis = omega / w;
        float halfAngle = w * 0.5;
        float4 dq = float4(axis * sin(halfAngle), cos(halfAngle));
        q = normalize(quatMul(dq, q));
    }
    return q;
}

// ── Integrate (Verlet + gravity + damping) ───────────────────────────────────

kernel void sm_integrate(
    device SMParticle*         particles [[buffer(0)]],
    device const SMBodyParams* bodies    [[buffer(1)]],
    constant SMUniforms&       u         [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= u.particleCount) return;
    SMParticle p = particles[id];
    if (p.pos.w == 0.0) return;   // pinned

    uint body = id / u.particlesPerBody;
    float gScale = bodies[body].p.z;

    float3 pos  = p.pos.xyz;
    float3 prev = p.prev.xyz;
    float3 vel  = (pos - prev) * u.damping;
    float3 next = pos + vel;
    next.y -= u.gravity * gScale * u.dt * u.dt;

    // Hard safety floor far below the void so a spilled body never rests on a
    // phantom plane — it falls freely until the host culls it.
    if (next.y < u.floorY) next.y = u.floorY;

    particles[id].prev.xyz = pos;
    particles[id].pos.xyz  = next;
}

// ── Cluster fit — one thread per (body, cluster) ─────────────────────────────

kernel void sm_cluster_fit(
    device SMClusterState*     state         [[buffer(0)]],
    device const SMParticle*   particles     [[buffer(1)]],
    device const uint*         memberStart   [[buffer(2)]],   // [C]
    device const uint*         memberCount   [[buffer(3)]],   // [C]
    device const uint*         members       [[buffer(4)]],   // flat LOCAL indices
    device const packed_float3* restLocal    [[buffer(5)]],   // [P]
    device const packed_float3* clusterRest  [[buffer(6)]],   // [C] rest centroids
    constant SMUniforms&       u             [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = u.bodyCount * u.clusterCount;
    if (gid >= total) return;
    uint body = gid / u.clusterCount;
    uint c    = gid % u.clusterCount;
    uint base = body * u.particlesPerBody;

    uint start = memberStart[c];
    uint cnt   = memberCount[c];
    if (cnt == 0u) { return; }

    // Current centroid.
    float3 cm = float3(0.0);
    for (uint k = 0; k < cnt; ++k) {
        uint lp = members[start + k];
        cm += particles[base + lp].pos.xyz;
    }
    cm /= float(cnt);

    float3 rc = float3(clusterRest[c]);

    // Covariance columns A = Σ (x - cm) ⊗ (rest - rc).
    float3 a0 = float3(0.0), a1 = float3(0.0), a2 = float3(0.0);
    for (uint k = 0; k < cnt; ++k) {
        uint lp = members[start + k];
        float3 x = particles[base + lp].pos.xyz - cm;
        float3 r = float3(restLocal[lp]) - rc;
        a0 += x * r.x;
        a1 += x * r.y;
        a2 += x * r.z;
    }

    float4 q = state[gid].quat;
    if (dot(q, q) < 0.5) q = float4(0, 0, 0, 1);   // first frame → identity
    q = extractRotation(a0, a1, a2, q);

    state[gid].centroid = float4(cm, 0.0);
    state[gid].quat = q;
}

// ── Apply goals — one thread per (body, particle) ────────────────────────────

kernel void sm_apply_goals(
    device SMParticle*          particles    [[buffer(0)]],
    device const SMClusterState* state       [[buffer(1)]],
    device const SMBodyParams*  bodies       [[buffer(2)]],
    device const uint*          pcStart      [[buffer(3)]],   // [P] per-particle cluster-list start
    device const uint*          pcCount      [[buffer(4)]],   // [P]
    device const uint*          pcList       [[buffer(5)]],   // flat cluster indices
    device const packed_float3* restLocal    [[buffer(6)]],   // [P]
    device const packed_float3* clusterRest  [[buffer(7)]],   // [C]
    constant SMUniforms&        u            [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= u.particleCount) return;
    if (particles[id].pos.w == 0.0) return;

    uint body = id / u.particlesPerBody;
    uint lp   = id % u.particlesPerBody;
    float beta      = bodies[body].p.x;
    float restScale = bodies[body].p.y;

    uint start = pcStart[lp];
    uint cnt   = pcCount[lp];
    if (cnt == 0u) return;

    float3 rl = float3(restLocal[lp]);
    float3 goal = float3(0.0);
    for (uint k = 0; k < cnt; ++k) {
        uint c = pcList[start + k];
        SMClusterState cs = state[body * u.clusterCount + c];
        float3 off = (rl - float3(clusterRest[c])) * restScale;
        goal += cs.centroid.xyz + quatRotate(cs.quat, off);
    }
    goal /= float(cnt);

    float3 x = particles[id].pos.xyz;
    particles[id].pos.xyz = x + beta * (goal - x);
}

// ── Box collider pushout — one thread per particle ───────────────────────────

kernel void sm_collide(
    device SMParticle*         particles [[buffer(0)]],
    device const SMCollider*   colliders [[buffer(1)]],
    constant SMUniforms&       u         [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= u.particleCount) return;
    if (u.colliderCount == 0u) return;
    if (particles[id].pos.w == 0.0) return;

    float3 pos  = particles[id].pos.xyz;
    float3 prev = particles[id].prev.xyz;
    bool hit = false;

    for (uint c = 0; c < u.colliderCount; ++c) {
        SMCollider col = colliders[c];
        float3 center = col.a.xyz;
        float3 he     = col.b.xyz;        // half extents
        float3 d = pos - center;
        // Inside the box if |d| < he on every axis.
        float3 q = abs(d) - he;
        if (q.x < 0.0 && q.y < 0.0 && q.z < 0.0) {
            // Push out along the axis of least penetration.
            // q is negative inside; the largest (closest to 0) axis is nearest face.
            float3 pen = -q;              // positive penetration depth per axis
            float minPen = pen.x; int axis = 0;
            if (pen.y < minPen) { minPen = pen.y; axis = 1; }
            if (pen.z < minPen) { minPen = pen.z; axis = 2; }
            float sgn = d[axis] >= 0.0 ? 1.0 : -1.0;
            pos[axis] = center[axis] + sgn * he[axis];
            // Tangential friction: damp the slide along the face.
            float3 vel = pos - prev;
            vel[axis] = 0.0;
            prev = pos - vel * (1.0 - u.collideFriction);
            hit = true;
        }
    }

    if (hit) {
        particles[id].pos.xyz  = pos;
        particles[id].prev.xyz = prev;
    }
}

// ── Self-collision spatial hash (bear↔bear) ──────────────────────────────────
// Counting-sort broadphase + Jacobi pairwise pushout, skipping same-body pairs.

static inline uint sm_hashCell(int3 cell, uint tableSize) {
    // Large primes (Teschner et al.).
    int h = (cell.x * 92837111) ^ (cell.y * 689287499) ^ (cell.z * 283923481);
    return uint(h & 0x7fffffff) % tableSize;
}

kernel void sm_hashClear(
    device atomic_uint* counts [[buffer(0)]],
    constant uint&      tableSize [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= tableSize) return;
    atomic_store_explicit(&counts[id], 0u, memory_order_relaxed);
}

kernel void sm_hashCount(
    device const SMParticle* particles [[buffer(0)]],
    device atomic_uint*      counts    [[buffer(1)]],
    constant SMHashUniforms& h         [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= h.particleCount) return;
    float3 p = particles[id].pos.xyz;
    int3 cell = int3(floor(p / h.cellSize));
    uint b = sm_hashCell(cell, h.tableSize);
    atomic_fetch_add_explicit(&counts[b], 1u, memory_order_relaxed);
}

kernel void sm_hashScan(
    device uint*       counts  [[buffer(0)]],
    device uint*       offsets [[buffer(1)]],
    constant uint&     tableSize [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Single-thread prefix sum, then RESET counts to 0 so the same buffer serves
    // as the zero-initialised scatter write cursor; after scatter it holds the
    // per-bucket count again, which sm_selfCollide reads. (PaperCloth idiom.)
    if (id != 0u) return;
    uint acc = 0u;
    for (uint i = 0; i < tableSize; ++i) {
        offsets[i] = acc;
        acc += counts[i];
        counts[i] = 0u;
    }
}

kernel void sm_hashScatter(
    device const SMParticle* particles [[buffer(0)]],
    device atomic_uint*      cursor    [[buffer(1)]],   // reused counts as write cursor
    device const uint*       offsets   [[buffer(2)]],
    device uint*             sorted    [[buffer(3)]],
    constant SMHashUniforms& h         [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= h.particleCount) return;
    float3 p = particles[id].pos.xyz;
    int3 cell = int3(floor(p / h.cellSize));
    uint b = sm_hashCell(cell, h.tableSize);
    uint slot = atomic_fetch_add_explicit(&cursor[b], 1u, memory_order_relaxed);
    sorted[offsets[b] + slot] = id;
}

kernel void sm_selfCollide(
    device SMParticle*       particles [[buffer(0)]],
    device const uint*       offsets   [[buffer(1)]],
    device const uint*       counts    [[buffer(2)]],   // per-bucket count post-scatter
    device const uint*       sorted    [[buffer(3)]],
    constant SMHashUniforms& h         [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= h.particleCount) return;
    if (particles[id].pos.w == 0.0) return;

    float3 pos = particles[id].pos.xyz;
    uint myBody = uint(particles[id].prev.w + 0.5);
    float minDist = 2.0 * h.radius;
    int3 base = int3(floor(pos / h.cellSize));

    float3 push = float3(0.0);
    uint nPush = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        uint b = sm_hashCell(base + int3(dx, dy, dz), h.tableSize);
        uint o = offsets[b];
        uint n = counts[b];
        for (uint k = 0; k < n; ++k) {
            uint j = sorted[o + k];
            if (j == id) continue;
            if (uint(particles[j].prev.w + 0.5) == myBody) continue;   // skip same body
            float3 d = pos - particles[j].pos.xyz;
            float dist = length(d);
            if (dist < minDist && dist > 1.0e-6) {
                push += (d / dist) * (minDist - dist) * 0.5;
                nPush++;
            }
        }
    }
    if (nPush > 0u) {
        // Averaged + clamped pushout (avoids the summed-overshoot blowup the
        // PaperCloth fix documents).
        float3 avg = push / float(nPush);
        float maxStep = h.radius;
        if (length(avg) > maxStep) avg = normalize(avg) * maxStep;
        particles[id].pos.xyz = pos + avg;
    }
}

// ── Skinned mesh write — one thread per (body, vertex) ───────────────────────
// Linear-blend skinning to the K nearest cluster frames. Writes packed_float3
// position + normal into the per-body render buffers the renderer reads.

struct SMSkinUniforms {
    uint vertexCount;       // V
    uint clusterCount;      // C
    uint bodyBase;          // body * C — offset into clusterState for this body
    uint K;                 // weights per vertex
};

kernel void sm_skin_write(
    device const SMClusterState* state      [[buffer(0)]],
    device const packed_float3*  vRest       [[buffer(1)]],   // [V] rest pos (body-local)
    device const packed_float3*  vRestNorm   [[buffer(2)]],   // [V] rest normal (body-local)
    device const uint*           skinIdx     [[buffer(3)]],   // [V*K] cluster indices
    device const float*          skinW       [[buffer(4)]],   // [V*K] weights
    device const packed_float3*  clusterRest [[buffer(5)]],   // [C] rest centroids
    device packed_float3*        outPos      [[buffer(6)]],   // [V]
    device packed_float3*        outNorm     [[buffer(7)]],   // [V]
    constant SMSkinUniforms&     s           [[buffer(8)]],
    uint vid [[thread_position_in_grid]])
{
    if (vid >= s.vertexCount) return;
    float3 rest = float3(vRest[vid]);
    float3 rn   = float3(vRestNorm[vid]);
    float3 pos = float3(0.0);
    float3 nrm = float3(0.0);
    for (uint k = 0; k < s.K; ++k) {
        uint c = skinIdx[vid * s.K + k];
        float w = skinW[vid * s.K + k];
        SMClusterState cs = state[s.bodyBase + c];
        float3 off = rest - float3(clusterRest[c]);
        pos += w * (cs.centroid.xyz + quatRotate(cs.quat, off));
        nrm += w * quatRotate(cs.quat, rn);
    }
    float nl = length(nrm);
    outPos[vid]  = packed_float3(pos);
    outNorm[vid] = packed_float3(nl > 1.0e-6 ? nrm / nl : float3(0, 1, 0));
}
