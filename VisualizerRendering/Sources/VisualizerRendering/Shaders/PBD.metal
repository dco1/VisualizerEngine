#include <metal_stdlib>
using namespace metal;

// ── PBD.metal ─────────────────────────────────────────────────────────────────
//
// Position-Based Dynamics compute kernels. Seven kernels:
//   1.  pbdIntegrate              — Verlet integration (gravity, damping)
//   2.  pbdConstraint             — XPBD distance constraint solver
//   3.  pbdFloorCollide           — keep particles above floorY
//   4.  pbdSDFCollide             — push particles out of SDF colliders
//   4b. pbdVelocitySolve          — XPBD per-constraint velocity post-solve
//   5.  pbdBuildCapsuleColliders  — rebuild capsule colliders from particles
//   6.  pbdTubeExpand             — expand spine particles into a cylindrical tube mesh
//
// Struct layouts MUST match the Swift definitions in PBDSolver.swift exactly.
// See the ALIGNMENT RULE comment in that file (float4 for shared structs,
// packed_float3 for vertex arrays that Swift reads as SIMD3<Float>).

// ── Shared structs ──────────────────────────────────────────────────────────

struct PBDParticle {
    float4 positionAndInvMass;  // xyz = position, w = invMass (0 = pinned)
    float4 prevPositionAndPad;  // xyz = prevPosition, w = unused
};

// XPBD compliance formulation — `compliance = α` has units of inverse spring
// constant (m/N). Zero is rigid; large is soft. See the Swift mirror in
// PBDSolver.swift for the full doc-comment on compliance ranges and how to
// translate the ergonomic stiffness-0-to-1 API into compliance.
struct PBDConstraint {
    uint  i;
    uint  j;
    float restLength;
    float compliance;
};

struct PBDUniforms {
    float dt;
    float gravity;
    float damping;
    float floorY;
    uint  particleCount;
    uint  constraintCount;
    float floorBand;        // tolerance above floorY: kill upward velocity in this zone
    uint  colliderCount;    // length of the SDF collider buffer (Phase 3)
    uint  ownerID;          // colliders with this ID are skipped (self-collision)
    float collideStiffness; // 0…1 scale on the per-substep SDF push-out
    float collideFriction;  // 0…1 *retention* of tangential vel on contact (lower = stickier)
    float selfRadius;       // this body's tube radius. Added to collider.radius for the
                            // effective contact distance so two R-tubes separate at 2R,
                            // not R. Particles are spine centres, not surface points.
    float velRestitution;   // XPBD velocity-solve restitution along constraint gradient.
                            // 0 = fully inelastic (kill normal-component oscillation),
                            // 1 = perfectly elastic (perpetual ring), ~0.2 = quick decay.
    float floorRestitution; // Coefficient of restitution against the floor.
                            // 0 = fully inelastic (impact energy absorbed, original
                            // behaviour), 1 = perfectly elastic, ~0.2 = real-world hot
                            // dog. Without this the chain has nowhere to put impact
                            // energy except bending, which reads as kinking.
    uint  ringCount;        // > 0 → ring treadmill; skip seam constraints. 0 = off.
    uint  ringHead;         // current pinned-anchor index; open seam is head-1 ↔ head.
    float maxSDFPush;       // > 0 → clamp the per-substep SDF push-out to this many
                            // metres. A thin sheet buried deep in a solid box would
                            // otherwise be teleported the full half-thickness to the
                            // surface in ONE step while its still-outside neighbour
                            // stays put — tearing that edge to tens of × rest length
                            // ("jagged points"). Clamping lets the buried patch climb
                            // out coherently over a few frames. 0 = unlimited (legacy).
};

// True if a distance/bend constraint (i↔j, ring-stride `stride`) spans the open
// seam between `ringHead-1` and `ringHead` in a ring of `ringCount` particles.
// The cut sits between the two consecutive ring indices (ringHead-1, ringHead);
// a stride-S constraint from `a` crosses it iff that cut lies in (a, a+S] in
// ring order. Returns false when the ring treadmill is off (ringCount == 0).
static bool ringSeamCrosses(uint i, uint j, uint stride, constant PBDUniforms& u) {
    if (u.ringCount == 0u) return false;
    uint N = u.ringCount;
    // Identify the lower ring-endpoint `a` such that the other endpoint is a+stride
    // (mod N). For a forward ring link, j == (i+stride)%N OR i == (j+stride)%N.
    uint a = ( (i + stride) % N == j ) ? i : j;
    // Cut index c: the link crosses the cut iff (ringHead) ∈ {a+1, …, a+stride} (mod N).
    for (uint s = 1u; s <= stride; ++s) {
        if ((a + s) % N == u.ringHead) return true;
    }
    return false;
}

// SDF collider — one of {sphere, capsule, box} packed into 48 bytes. Matches
// `PBDCollider` in PBDSolver.swift exactly; keep the two in lockstep.
//   a.xyz = primary point (centre or endpoint A); a.w = type tag, bit-cast from
//           uint (0 = sphere, 1 = capsule, 2 = box).
//   b.xyz = secondary point (capsule endpoint B / box halfExtents); b.w = radius.
//   meta.x = ownerID (skip-self filter); meta.yzw unused.
struct PBDCollider {
    float4 a;
    float4 b;
    uint4  meta;
};

struct TubeExpandUniforms {
    uint  ringSegments;
    uint  spineCount;
    float radius;
    uint  subSegments;  // visual rings per spine segment (Catmull-Rom oversampling)
    // Optional caller-supplied "up" axis for the ring-frame seed. When
    // `useFixedUp != 0`, frameFromTangent uses (fixedUpX, fixedUpY, fixedUpZ)
    // instead of world-Y. Lets a 2-D-plane caller (e.g. Hotdog Font glyphs in
    // the XY plane) pick an out-of-plane up so the bitangent never flips as
    // the tangent rotates inside the plane — kills the chevron pucker that
    // appears when the bare-frame fallback toggles mid-curve.
    uint  useFixedUp;
    float fixedUpX;
    float fixedUpY;
    float fixedUpZ;
    // Ring-treadmill logical-order offset. When `spineWrap != 0`, the spine is a
    // RING of `spineCount` particles and the logical open rope starts at index
    // `spineOffset`: ring sample k reads particles[(spineOffset + k) % spineCount].
    // This puts the open seam at the rope's ENDS (where the end-caps go) instead
    // of drawing a spurious segment across the buffer wrap. spineWrap == 0 keeps
    // the plain linear-buffer behaviour for every chain caller.
    uint  spineWrap;
    uint  spineOffset;
    // Tip-taper fraction: 0 = perfect cylinder; 0.25 = tips 25% narrower than
    // the body centre. Body cross-sections scale as the sin(π·bodyT) profile;
    // both end-cap hemispheres scale by (1 − taperFactor) to match the tip radius.
    float taperFactor;
    // Straight-line mode (HotdogDropUltra): when 1, ignore Catmull-Rom and
    // render a perfect capsule along the spine[0]→spine[N-1] axis. Eliminates
    // silhouette lobes at knot bends and frame-seam artifacts. Physics capsules
    // (pbdBuildCapsuleColliders) are unaffected — they use raw particle positions.
    // 0 = Catmull-Rom mode (default; used by SoftServe, HotdogFont, etc.).
    uint  useStraightLine;
    // Dip-coat waterline (HotdogDropUltra mustard bath). When dipStrength > 0,
    // vertices below dipY (and within dipRadius of the world Y axis) blend
    // their vertex RGB toward (dipR,dipG,dipB) — a per-channel albedo
    // MULTIPLIER chosen by the host to turn the body colour into the fluid
    // colour. The 12 cm MC fluid grid cannot mesh a millimetre coating film,
    // so the "wet third" of a submerged body is carried on the body itself —
    // driven by the sim's actual surface height, not a faked decal.
    float dipY;
    float dipRadius;
    float dipStrength;
    float dipR;
    float dipG;
    float dipB;
    // Per-frank BODY colour multiplier (default 1,1,1 = white identity). The
    // tube-expand kernel writes this as the DRY vertex colour; the dip-coat then
    // blends bodyColor → (dipR,dipG,dipB) below the waterline. The BATCHED tube
    // path sets bodyColor to the frank's per-frank albedo so a single white
    // InstanceRef + the shared colour buffer reproduce per-frank tone in one
    // draw. Keep in lockstep with PBDSolver.swift TubeExpandUniforms.
    float bodyR;
    float bodyG;
    float bodyB;
    // Rotation-minimizing frame (RMF). When 1 AND useStraightLine == 0, the ring
    // cross-section frame is PROPAGATED along the centerline (rotation-minimizing)
    // instead of rebuilt per-ring by frameFromTangent. frameFromTangent computes
    // each ring's frame independently by Gram-Schmidt against a fixed world-up,
    // which DISCONTINUOUSLY switches the up-reference when the tangent sweeps near
    // vertical — across that switch adjacent rings rotate ~90°, the tube triangle
    // strip threads through itself, and the cross-section collapses to a pinched
    // waist. A stuffed tube of bent franks hits this constantly. The RMF carries a
    // single coherent reference vector from ring 0, eliminating both the pinch and
    // any twist while KEEPING the Catmull-Rom bend. Default 0 = legacy per-ring
    // frame (SoftServe, HotdogFont, straight-line Ultra — all byte-identical).
    uint  useRMF;
};

// Dip-coat multiplier for a tube vertex (see TubeExpandUniforms.dipY). The
// waterline gets ±5 cm of per-vertex hash jitter and a 15 cm soft band so the
// coat edge reads as a paste-dipped line, not a CSG cut.
static inline float3 dipCoatMultiplier(float3 vert, constant TubeExpandUniforms& u, uint seed) {
    // DRY colour is the per-frank body colour (default 1,1,1 = white identity,
    // so legacy per-instance scenes are byte-identical). The dip-coat blends the
    // body colour toward the fluid colour below the waterline.
    float3 body = float3(u.bodyR, u.bodyG, u.bodyB);
    if (u.dipStrength <= 0.0f) return body;
    float r2 = vert.x * vert.x + vert.z * vert.z;
    if (r2 >= u.dipRadius * u.dipRadius) return body;
    float jitter = (float((seed * 747796405u) >> 9u) / float(1u << 23u) - 0.5f) * 0.10f;
    float below  = saturate((u.dipY + jitter - vert.y) / 0.15f);
    return mix(body, float3(u.dipR, u.dipG, u.dipB), below * u.dipStrength);
}

// Wrapped spine sampler for the ring treadmill. Reads logical index `k`
// (0…spineCount-1) of the open rope, mapping through the ring offset when
// `spineWrap != 0`. For plain chains it's a straight `spine[k]`.
static float3 sampleSpine(device const PBDParticle* spine,
                          constant TubeExpandUniforms& u, uint k) {
    uint idx = (u.spineWrap != 0u) ? (u.spineOffset + k) % u.spineCount : k;
    return spine[idx].positionAndInvMass.xyz;
}

// ── Roadmap struct sketches ───────────────────────────────────────────────────
//
// Angle-based bending constraint — add when you implement `pbdBend`. The
// existing distance-based skip-one approximation works but fights axial
// compression; an angle constraint penalises hinge deflection directly.
//
//   struct PBDBendConstraint {
//       uint  i0;           // left particle
//       uint  i1;           // hinge particle
//       uint  i2;           // right particle
//       float restAngle;    // rest angle at hinge in radians
//       float compliance;   // XPBD-style α
//       uint  pad;
//   };
//
// Shape-matching constraint — for blob soft bodies. Needs a `restPositions`
// buffer alongside the particle buffer. Use the same XPBD λ-per-constraint
// pattern as the distance solver.
//
//   struct PBDShapeParams {
//       uint  particleStart;  // first particle index of this body in the buffer
//       uint  particleCount;  // number of particles in this body
//       float compliance;     // XPBD-style α
//       uint  pad;
//   };


// ── KERNEL 1: Verlet integration ──────────────────────────────────────────────
//
// Advances position by Verlet integration:
//   velocity = (pos - prevPos) / dt
//   velocity.y -= gravity * dt        ← gravity
//   velocity   *= damping             ← per-step drag
//   newPos      = pos + velocity * dt
//   prevPos     = pos
//
// Verlet is preferred over explicit Euler for PBD because:
//   • Implicit velocity = constraint corrections to positions are reflected
//     automatically in the next step's velocity.
//   • Energy-conserving (symplectic) — chains don't gain energy over time.
//
// Pinned particles (invMass == 0) are skipped completely.
//
// EXTENSION — Wind / external forces:
//   Add a float3 windAccel to PBDUniforms. Before damping:
//     velocity += windAccel * dt
//   Wind becomes an animated field you update from Swift each frame.
//   For per-blade grass wind, pass a 2-D wind noise texture and sample it
//   by world XZ position.

kernel void pbdIntegrate(
    device PBDParticle*   particles [[ buffer(0) ]],
    device PBDConstraint* unused    [[ buffer(1) ]],
    constant PBDUniforms& u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;

    float3 vel  = (pos - prev) / u.dt;
    vel.y      -= u.gravity * u.dt;
    vel        *= u.damping;

    float3 newPos = pos + vel * u.dt;

    particles[id].prevPositionAndPad.xyz  = pos;
    particles[id].positionAndInvMass.xyz  = newPos;
}


// ── KERNEL 2: XPBD distance constraint solver ────────────────────────────────
//
// Müller-2020 Extended Position Based Dynamics. For each constraint between
// particles i and j with rest length L₀:
//
//   C(x)         = |x_i − x_j| − L₀                  (signed length error)
//   α            = compliance                          (constraint-stored, units m/N)
//   α̃            = α / dt²                              (dt-scaled compliance)
//   Δλ           = (−C − α̃ · λ) / (w_i + w_j + α̃)     (Lagrange-multiplier update)
//   λ           += Δλ                                  (accumulates across iters
//                                                       WITHIN a substep; reset 0
//                                                       BETWEEN substeps)
//   Δx_i         =  w_i · n · Δλ                       (n = (x_i − x_j)/|x_i − x_j|)
//   Δx_j         = −w_j · n · Δλ
//
// Why XPBD over classic PBD:
//   • Stiffness becomes iteration-count *and* timestep independent. Classic
//     PBD's "stiffness 0.3" effectively becomes nearly rigid by ~10 iters.
//     XPBD's compliance value behaves the same after 4 iters as after 40.
//   • Soft constraints actually *yield* when fighting another constraint
//     (in our case an SDF contact push) rather than fully restoring per iter
//     and forcing the other constraint into compromise.
//   • Position corrections are properly separated from velocity corrections —
//     the implicit-velocity ghost impulse that contributed to end-on kinking
//     in classic PBD is no longer there.
//
// GPU RACE CONDITION NOTE
// Two constraints that share a particle would race; PBDSolver handles this
// via graph colouring (even/odd primary, 3-colour bending) so each subpass
// dispatch only touches disjoint particle pairs.
//
// LAMBDA BUFFER LIFECYCLE
// The Swift side blit-fills lambda to zero at the start of each substep,
// before the integrate kernel. Within the substep, `lambda[id]` accumulates
// across all `constraintIterations` iterations for this constraint. Each
// per-colour-group dispatch sees both the constraint buffer AND the lambda
// buffer offset by the colour's `start`, so `id` indexes consistently into
// both — see encodeConstraintSubpass in PBDSolver.swift.

kernel void pbdConstraint(
    device PBDParticle*   particles   [[ buffer(0) ]],
    device PBDConstraint* constraints [[ buffer(1) ]],
    constant PBDUniforms& u           [[ buffer(2) ]],
    device float*         lambda      [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.constraintCount) return;
    PBDConstraint c = constraints[id];

    // Ring treadmill: skip the one (or two, for bend) constraint(s) spanning the
    // open seam, so the closed index-ring behaves as an OPEN rope. No-op when
    // ringCount == 0 (every chain caller). Ring stride = min forward/back gap.
    if (u.ringCount != 0u) {
        uint N = u.ringCount;
        uint fwd = (c.j + N - c.i) % N;
        uint stride = min(fwd, N - fwd);
        if (ringSeamCrosses(c.i, c.j, stride, u)) return;
    }

    float3 pi = particles[c.i].positionAndInvMass.xyz;
    float3 pj = particles[c.j].positionAndInvMass.xyz;
    float  wi = particles[c.i].positionAndInvMass.w;
    float  wj = particles[c.j].positionAndInvMass.w;

    float3 delta = pi - pj;            // points from j → i so n is +∇_i C
    float  dist  = length(delta);
    if (dist < 1e-6) return;

    float  wSum  = wi + wj;
    if (wSum < 1e-6) return;           // both particles pinned

    float3 n = delta / dist;
    float  C = dist - c.restLength;

    // Compliance is per-constraint (in PBDConstraint). dt² scaling lets the
    // same compliance produce the same behaviour at any substep rate.
    float  alphaTilde  = c.compliance / (u.dt * u.dt);
    float  deltaLambda = (-C - alphaTilde * lambda[id])
                        / (wSum + alphaTilde);
    lambda[id] += deltaLambda;

    float3 correction = n * deltaLambda;
    particles[c.i].positionAndInvMass.xyz += correction * wi;
    particles[c.j].positionAndInvMass.xyz -= correction * wj;
}


// ── KERNEL 3: Floor collision ─────────────────────────────────────────────────
//
// Projects particles that have fallen below floorY back to floorY and zeroes
// their downward velocity implicitly (by setting prevPos.y ≥ pos.y at floor).
//
// Lateral friction is applied directly in this kernel: when a particle enters
// the floor band, its tangential velocity is multiplied by a strong damping
// factor so contacts settle quickly. See the floorBand explanation below.
//
// Arbitrary SDF collisions are handled by pbdSDFCollide (KERNEL 4); the floor
// itself stays as a fast hard-clamp here.
//
// EXTENSION — Spatial hash for inter-particle collision:
//   Two particles in different chains can pass through each other when chains
//   get long or dense. Add a buildHash kernel + a self-collide kernel that
//   reads nearby cells. The same spatial hash is the substrate for SPH fluid
//   neighbour queries — build it once and reuse.

kernel void pbdFloorCollide(
    device PBDParticle*   particles [[ buffer(0) ]],
    device PBDConstraint* unused    [[ buffer(1) ]],
    constant PBDUniforms& u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned

    // Act on particles at or within floorBand above the floor.
    // The band catches particles that were clamped to floorY by a previous
    // floor pass and then pushed slightly upward by the constraint solver —
    // without it, the constraint's position delta becomes ε/dt artificial
    // upward velocity in the next integrate step.
    if (p.positionAndInvMass.y < u.floorY + u.floorBand) {
        float3 vel = p.positionAndInvMass.xyz - p.prevPositionAndPad.xyz;

        if (p.positionAndInvMass.y < u.floorY) {
            // Below floor: clamp position, REFLECT y velocity with restitution.
            // Original behaviour was vel.y = 0 (fully inelastic) — that meant
            // impact energy went entirely into the chain via constraint
            // corrections, which on an end-on landing has nowhere to go
            // except bending (kinks). Bouncing returns the energy as upward
            // motion instead.
            //   restitution = 0   → vel.y_new = 0       (original, kink-prone)
            //   restitution = 0.2 → vel.y_new = +0.2|vy|  (real-world hot dog)
            //   restitution = 1   → vel.y_new = +|vy|   (lossless, perpetual)
            // Only reflect if actually impacting; a previous-substep bounce
            // travelling up through the band shouldn't get re-reflected back
            // down into the floor.
            p.positionAndInvMass.y = u.floorY;
            if (vel.y < 0) {
                vel.y = -u.floorRestitution * vel.y;
            }
            // Moderate lateral friction. We used to retain just 12 % of
            // horizontal velocity (≈88 % friction) so hot dogs stopped dead
            // on contact, but that pinned settled tubes against the floor
            // and prevented the SDF push from sliding pile-mates apart —
            // settled tubes ended up with 9–14 % sustained interpenetration
            // because lateral motion died before it could relax the pile.
            // 0.40 retention (60 % friction) still settles a fresh landing
            // within ~5 contact ticks but lets pile-pressure dissipate
            // horizontally over many ticks, which is what unwinds stuck
            // pile geometry.
            vel.xz *= 0.40;
        } else {
            // In the band above floor: constraint pushed particle slightly up.
            // Kill only TINY upward components — anything below 3× a gravity
            // tick is a constraint kick or numerical ε; anything larger is a
            // real bounce that must survive or restitution has no effect.
            //
            // gravityKickPos = g · dt²   ≈ 6.8e-4 m at 9.8 m/s² and dt=1/120,
            // so threshold ≈ 2e-3 m  ≈ 0.24 m/s upward — well below a
            // restitution-0.2 bounce from 2 m/s impact (1.7 mm in positional
            // delta units) but well above a constraint correction artefact.
            float gravityKickPos = u.gravity * u.dt * u.dt;
            float upThreshold    = 3.0 * gravityKickPos;
            if (vel.y < upThreshold) {
                vel.y = min(vel.y, 0.0);
            }
            vel.xz *= 0.60;
        }
        p.prevPositionAndPad.xyz = p.positionAndInvMass.xyz - vel;
        particles[id] = p;
    }
}


// ── KERNEL 4b: Particle self-collision (single-rope coiling) ─────────────────
//
// Liquid-rope coiling (soft-serve, honey, drizzled paint) needs the rope to
// physically STACK on itself: a falling viscous rope buckles into coils only
// because the coils already on the pile block the descending segment's path.
// Distance + bending constraints alone never produce this — with no
// inter-particle contact the rope just passes through its own pile and falls
// flat. This kernel is the missing contact term.
//
// Every particle carries a swept sphere of radius `selfRadius`. Two particles
// whose centres are closer than 2·selfRadius get pushed apart, UNLESS they are
// chain neighbours within `selfSkip` indices — those are deliberately held at
// segLength (< 2·selfRadius) by the distance constraints, so colliding them
// would fight the rope's own structure and make it explode. Excluding the
// near-neighbour band leaves exactly the coil-on-coil and coil-on-pile contacts
// that build the stack.
//
// TOPOLOGY / PERF: this is an honest O(n²) pass — each thread walks the whole
// particle buffer. For the prototype's few-hundred-particle rope (e.g. 280 →
// ~78K distance tests, once per substep) it's well under the GPU budget and
// measured to hold 60 fps. The documented scale-up (thousands of particles) is
// a uniform spatial hash: a build-bins kernel + a nearby-cell walk — the SAME
// hash the SPH-fluid roadmap wants, built once and shared. Until a real
// particle count forces it, O(n²) is the simplest correct thing.
//
// RACE SAFETY: each thread writes ONLY its own particle slot. The push is
// symmetric — this thread moves particle `id` by half the overlap toward its
// side; particle `j`'s own thread independently moves `j` by the other half.
// No two threads write the same slot, so no atomics needed.

struct PBDSelfCollideUniforms {
    uint  particleCount;
    float selfRadius;      // contact at 2·selfRadius centre separation
    float stiffness;       // 0…1 push scale per dispatch (yields to constraints)
    float friction;        // 0…1 tangential-velocity retention on contact
    uint  selfSkip;        // exclude |i-j| <= selfSkip (chain neighbours)
    // ── Cup / cone wall confinement (inside-out cylinder SDF) ────────────────
    // A vertical cylinder the rope is kept INSIDE: particles whose XZ distance
    // from (wallCx, wallCz) exceeds (wallRadius − selfRadius) are pushed back
    // toward the axis, so coils can only build UPWARD into a tight column
    // instead of spreading flat. wallEnabled == 0 disables (open-floor mode).
    uint  wallEnabled;
    float wallRadius;      // inner radius of the cup (m)
    float wallCx;          // cup-axis X
    float wallCz;          // cup-axis Z
    float wallBottomY;     // confinement applies only above this Y (cup floor)
};

kernel void pbdSelfCollide(
    device PBDParticle*               particles [[ buffer(0) ]],
    constant PBDSelfCollideUniforms&  u         [[ buffer(1) ]],
    device const PBDParticle*         others    [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    // JACOBI double-buffer: read EVERY position (own + neighbours) from the
    // frozen `others` snapshot taken before this pass, write only to
    // `particles[id]`. This is what makes the 0.5-split contact push symmetric:
    // for a pair (i,j), thread i computes (pos_i − pos_j) and thread j computes
    // (pos_j − pos_i) from the SAME frozen positions, so the two half-pushes are
    // exact mirrors and inject zero net energy. The old in-place version read
    // live, mutating neighbours → asymmetric pushes → the pile buzzed forever
    // (diagnosed via the neon per-substep-speed heatmap, 2026-06-06).
    PBDParticle pa = others[id];
    if (pa.positionAndInvMass.w == 0.0) return;  // pinned (nozzle anchor)

    float3 pos  = pa.positionAndInvMass.xyz;
    float3 prev = pa.prevPositionAndPad.xyz;

    float  contact  = 2.0 * u.selfRadius;
    float  contact2 = contact * contact;

    float3 pushSum = float3(0.0);   // accumulated separation (≈ contact normal × mag)
    float  hits    = 0.0;

    for (uint j = 0; j < u.particleCount; ++j) {
        uint d = (j > id) ? (j - id) : (id - j);
        if (d <= u.selfSkip) continue;            // self + chain neighbours

        float4 pbb   = others[j].positionAndInvMass;
        if (pbb.w == 0.0) continue;   // skip PINNED neighbours: parked/un-extruded
                                      // cream stacked at the nozzle + the anchor
                                      // must not shove the live rope.
        float3 pb    = pbb.xyz;
        float3 delta = pos - pb;
        float  dist2 = dot(delta, delta);
        if (dist2 >= contact2 || dist2 < 1e-12) continue;

        float  dist = sqrt(dist2);
        float3 n    = delta / dist;
        // Half the overlap to this particle; j's thread applies the mirror half.
        float  pen  = (contact - dist) * 0.5 * u.stiffness;
        pushSum += n * pen;
        hits    += 1.0;
    }

    if (hits > 0.0) {
        // AVERAGE the push over contact count (2026-06-06). Because the pass is
        // Jacobi (every thread reads the same frozen snapshot), a particle and
        // ALL its overlapping neighbours move away from each other based on the
        // same old positions — so SUMMING per-pair pushes double-counts and
        // over-relaxes catastrophically in a dense pile (measured: ~20 mm/substep
        // churn flinging particles 5 m up; self-collision-OFF bisect was dead
        // still at 0.002 mm/substep). Averaging makes each particle move by the
        // MEAN separation it owes, which converges over the 14 iterations instead
        // of exploding. A safety clamp bounds any residual spike.
        pushSum /= hits;
        float maxPush = 0.25 * u.selfRadius;
        float plen    = length(pushSum);
        if (plen > maxPush) pushSum *= (maxPush / plen);
        pos += pushSum;
        // Damp the velocity component ALONG the net push direction (the contact
        // normal) so impact energy dissipates inelastically, and retain a
        // fraction of the tangential component as friction so the pile settles
        // without writhing but coils can still slide into place.
        float3 vel = pos - prev;
        float  pl  = length(pushSum);
        if (pl > 1e-6) {
            float3 nrm  = pushSum / pl;
            float  vN   = dot(vel, nrm);
            float3 velT = vel - nrm * vN;
            // FULLY INELASTIC normal (2026-06-06). The old `max(vN, 0.0)` PRESERVED
            // the outward separation velocity, so every `pos += pushSum` became
            // retained kinetic energy → the pile pumped itself into a sustained
            // ~15 mm/substep churn that flung particles metres up (measured via the
            // stats CSV). Killing the normal component entirely makes the contact a
            // pure position projection: overlap is resolved, no velocity injected.
            // Tangential is still retained (× friction) so coils slide into place.
            vel = velT * u.friction;
        } else {
            vel *= u.friction;
        }
        prev = pos - vel;
    }

    // ── Cup-wall confinement (inside-out cylinder) ───────────────────────────
    // Applied AFTER the self-contact push (and regardless of whether there was
    // a contact) so the column is kept inside the cup every iteration. This is
    // a real one-sided SDF: only particles that have escaped past the inner
    // wall radius feel an inward correction, with the radial velocity damped so
    // they don't ping-pong. The cup is what turns a spreading mound into a tall
    // stacked column — exactly the confinement the real soft-serve cup provides.
    if (u.wallEnabled != 0u && pos.y > u.wallBottomY) {
        float2 r2   = float2(pos.x - u.wallCx, pos.z - u.wallCz);
        float  rLen = length(r2);
        float  rMax = u.wallRadius - u.selfRadius;
        if (rLen > rMax && rLen > 1e-6) {
            float2 inward = -r2 / rLen;            // toward the axis
            float  pen    = (rLen - rMax) * u.stiffness;
            pos.x += inward.x * pen;
            pos.z += inward.y * pen;
            // Kill the outward radial velocity component so the particle settles
            // against the wall instead of bouncing back out.
            float3 vel = pos - prev;
            float  vR  = vel.x * inward.x + vel.z * inward.y;   // along inward
            if (vR < 0.0) {                        // moving outward — cancel it
                vel.x -= inward.x * vR;
                vel.z -= inward.y * vR;
                prev = pos - vel;
            }
        }
    }

    particles[id].positionAndInvMass.xyz = pos;
    particles[id].prevPositionAndPad.xyz = prev;
}


// ── KERNEL 4c: Endless-feed recycle (treadmill respool) ──────────────────────
//
// Turns the finite rope into an ENDLESS column on a FIXED particle budget, with
// O(1) GPU work per recycle (one thread, a handful of buffer writes) — no
// growing budget, no per-frame CPU particle loop, no GPU→CPU readback.
//
// MODEL — a moving-pin ring treadmill. The chain is a RING of N particles
// (distance + bend constraints connect i ↔ (i+1)%N and i ↔ (i+2)%N). Exactly
// ONE seam is held open so the ring behaves as an OPEN rope: the seam
// constraints (the ones spanning the cut between `head-1` and `head`) are
// disabled on the CPU by writing them a near-infinite compliance — O(1), no
// kernel branch. `particles[head]` is the pinned nozzle anchor; the logical
// rope runs head → head+1 → … → head-1, whose free end (head-1) is the buried
// tip at the bottom of the confined column.
//
// One recycle step (this kernel, dispatched once per cadence inside the tick
// command buffer, AFTER the constraint/contact loop so it sees settled state):
//   • oldAnchor = head            → released (invMass 1): becomes the freshest
//     extruded segment just below the nozzle.
//   • newAnchor = (head+N-1)%N    → the buried tip: respooled to a tight spool
//     slot at the nozzle and PINNED (invMass 0). Because it was buried/occluded
//     at the bottom of the column, the teleport is invisible.
// The CPU then advances head := newAnchor and moves the open seam one notch, so
// the constraint that was the seam (tip↔oldAnchor) becomes a live chain link
// and the next link up becomes the new seam — the rope stays topologically open
// and contiguous. Net effect: rope is continuously consumed at the bottom and
// re-paid-out at the top, forever, at constant cost.
//
// Driven open-loop on a fixed cadence tuned so respool rate ≈ pay-out rate
// (gravity/damping), keeping the live rope length — and therefore the column
// height and the GPU cost — bounded and steady. No settle-detection readback.

struct PBDRecycleUniforms {
    uint  oldAnchor;    // index released this step (was the pinned nozzle anchor)
    uint  newAnchor;    // index respooled + pinned this step (was the buried tip)
    float spoolX;       // nozzle spool slot position
    float spoolY;
    float spoolZ;
    float seedAngle;    // tiny helical seed so the fresh segment breaks symmetry
    float seedRadius;
    float pad0;
};

kernel void pbdRecycle(
    device PBDParticle*              particles [[ buffer(0) ]],
    constant PBDRecycleUniforms&     u         [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id != 0u) return;   // single-thread O(1) bookkeeping

    // Release the old anchor so it falls as fresh rope. Keep its position; just
    // give it mass and zero velocity (prev == pos) so it starts from rest.
    float3 oa = particles[u.oldAnchor].positionAndInvMass.xyz;
    particles[u.oldAnchor].positionAndInvMass = float4(oa, 1.0);
    particles[u.oldAnchor].prevPositionAndPad = float4(oa, 0.0);

    // Respool the buried tip to a tight spool slot at the nozzle and pin it.
    float3 spool = float3(u.spoolX + cos(u.seedAngle) * u.seedRadius,
                          u.spoolY,
                          u.spoolZ + sin(u.seedAngle) * u.seedRadius);
    particles[u.newAnchor].positionAndInvMass = float4(spool, 0.0);  // pinned
    particles[u.newAnchor].prevPositionAndPad = float4(spool, 0.0);
}


// ── KERNEL 4d: World-space sinking-floor conveyor ────────────────────────────
//
// THE piece that makes the endless column hold a STEADY TALL height instead of
// draining or overflowing. A ring treadmill alone respools the buried tip back
// to the nozzle, but the cup is already full when it does — so the respooled
// cream has nowhere to go and coils UPWARD at the nozzle, choking the feed and
// letting the column drain (the documented open-loop failure).
//
// The fix is a sinking floor: every substep, translate the WHOLE settled column
// downward by `sinkDelta` (= sinkRate · dt). This opens a `sinkDelta` gap below
// the nozzle every step that the active feed fills with fresh coiling rope, so
// the top always has headroom — new cream always has somewhere to go. The
// settled coils ride the conveyor down toward the recycle plane; once the
// buried tip crosses it (detected by the async probe on the CPU), it respools.
//
// Rigid translation: shift `position.y` and `prevPosition.y` by the SAME amount
// so the Verlet velocity (pos − prev) is unchanged — the sink injects no
// spurious velocity, it just slides the frame of reference down. Pinned
// particles (the nozzle anchor) are NOT sunk — they hold the feed point fixed.
//
// Only sinks particles ABOVE `sinkFloorY` (the recycle plane): once a particle
// is at/below the plane it's about to be recycled, so further sinking would just
// drive it through the floor before the CPU probe catches it.

struct PBDConveyorUniforms {
    uint  particleCount;
    float sinkDelta;     // metres to lower the column this substep (sinkRate · dt)
    float sinkFloorY;    // don't sink particles already at/below this (recycle plane)
    float pad0;
};

kernel void pbdConveyorSink(
    device PBDParticle*            particles [[ buffer(0) ]],
    constant PBDConveyorUniforms&  u         [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;          // pinned nozzle anchor
    if (p.positionAndInvMass.y <= u.sinkFloorY) return; // already at the plane

    particles[id].positionAndInvMass.y -= u.sinkDelta;
    particles[id].prevPositionAndPad.y -= u.sinkDelta;  // same shift → velocity preserved
}


// ── KERNEL 4e: Column-depth probe (closed-loop rate control) ─────────────────
//
// Reduces the live particle column into two scalars the controller reads back
// ASYNCHRONOUSLY (completion handler, never a per-frame waitUntilCompleted) to
// self-regulate the conveyor:
//   • columnTopY     — highest live (unpinned) particle Y. The conveyor's sink
//                      rate is servo'd to hold this at a target height: column
//                      too tall → sink faster; too short → sink slower.
//   • buriedTipMinY  — lowest live particle Y. When it crosses the recycle plane
//                      the controller asks for one recycle next tick.
// Both are computed with a single-thread linear pass — the rope is only a few
// hundred particles, so a fused reduction isn't worth the threadgroup-atomics
// complexity. O(n), one dispatch per probe cadence (not every substep).
//
// Output is a tiny shared buffer; the controller reads it in the command
// buffer's completion handler. The probe encodes near the END of the tick so it
// sees the post-sink, post-constraint settled state.

struct PBDProbeUniforms {
    uint  particleCount;
    float recycleY;       // particles at/below this are counted as "owed a respool"
    float pad0;
    float pad1;
};

struct PBDColumnProbe {
    float columnTopY;       // max Y over live particles (servo target)
    float buriedTipMinY;    // min Y over live particles
    uint  liveCount;        // number of unpinned particles (diagnostic)
    uint  belowPlaneCount;  // live particles at/below recycleY (= respools owed)
};

kernel void pbdProbeColumn(
    device const PBDParticle*    particles [[ buffer(0) ]],
    constant PBDProbeUniforms&   u         [[ buffer(1) ]],
    device PBDColumnProbe*       out       [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id != 0u) return;   // single-thread linear reduction (few-hundred rope)
    float topY = -1e30;
    float minY =  1e30;
    uint  live = 0u;
    uint  below = 0u;
    for (uint i = 0; i < u.particleCount; ++i) {
        PBDParticle p = particles[i];
        if (p.positionAndInvMass.w == 0.0) continue;  // skip the pinned anchor
        float y = p.positionAndInvMass.y;
        topY = max(topY, y);
        minY = min(minY, y);
        if (y <= u.recycleY) below += 1u;
        live += 1u;
    }
    if (live == 0u) { topY = 0.0; minY = 0.0; }
    out[0].columnTopY      = topY;
    out[0].buriedTipMinY   = minY;
    out[0].liveCount       = live;
    out[0].belowPlaneCount = below;
}


// ── KERNEL 4: SDF inter-body collision (Phase 3) ─────────────────────────────
//
// For each particle, walk the collider buffer and push the particle out of any
// collider whose `meta.x` (ownerID) differs from `u.ownerID`. Treat the push as
// a position correction (PBD-style): also rewind `prevPos` so the implicit
// Verlet velocity reflects the contact, preventing the next integrate step
// from re-tunneling the particle straight back in.
//
// The SDF gradient is the unit vector from the closest point on the collider to
// the particle. For a sphere it's `normalize(p - centre)`; for a capsule it's
// `normalize(p - closestPointOnSegment(p, a, b))`; for a box it's the analytic
// gradient of the standard exterior box SDF.
//
// `u.collideStiffness ∈ [0,1]` scales every push so the controller can dial
// hardness against the bending/long-range constraints. 1.0 = full correction in
// one substep (rigid), lower values let the colliders feel slightly squishy.
//
// EXTENSION — Spatial hash:
//   This kernel is O(particles × colliders). With 250 hot dogs × 4 spine
//   particles each, and ~750 capsule colliders, that's ~750K ops/iter — fine on
//   GPU, but won't scale to thousands of bodies. When it bottlenecks, add a
//   spatial hash so each particle only walks colliders in nearby cells.

// Squared distance from point p to the segment [a, b].
static float3 closestOnSegment(float3 p, float3 a, float3 b) {
    float3 ab = b - a;
    float  t  = saturate(dot(p - a, ab) / max(dot(ab, ab), 1e-12));
    return a + t * ab;
}

// One iteration of the SDF contact: tests `probe` against `col`, applies any
// resulting push to `pos` (the actual particle, not the probe), and updates
// `prev` so the next Verlet velocity reflects a proper inelastic contact.
// Pulled out of pbdSDFCollide so we can call it for the particle position AND
// for the two segment-midpoint probes that catch crossing-segment overlaps.
static void sdfContact(float3 probe,
                       PBDCollider col,
                       constant PBDUniforms& u,
                       thread float3& pos,
                       thread float3& prev,
                       thread bool& hit) {
    if (col.meta.x == u.ownerID) return;

    uint  kind   = as_type<uint>(col.a.w);
    float radius = col.b.w + u.selfRadius;
    // Per-step displacement cap (metres). 0 = unlimited (legacy / non-cloth callers).
    float maxPush = (u.maxSDFPush > 0.0) ? u.maxSDFPush : 1e30;

    float3 closest;
    if (kind == 0u) {
        // Sphere.
        closest = col.a.xyz;
    } else if (kind == 1u) {
        // Capsule.
        closest = closestOnSegment(probe, col.a.xyz, col.b.xyz);
    } else {
        // Box (kind == 2u). Probe-vs-box uses the same standard logic, just
        // with `probe` in place of the historical pos.
        float3 d  = probe - col.a.xyz;
        float3 he = col.b.xyz;
        float3 q  = clamp(d, -he, he);
        float3 outside = d - q;
        float  outLen  = length(outside);
        if (outLen > 1e-6) {
            if (outLen >= radius) return;
            float3 n    = outside / outLen;
            float  disp = min((radius - outLen) * u.collideStiffness, maxPush);
            pos  += n * disp;
            prev += n * disp * 0.5;
            hit   = true;
            return;
        } else {
            float3 dist = he - abs(d);
            float3 n;
            float  push;
            if (dist.x < dist.y && dist.x < dist.z) {
                n = float3(sign(d.x), 0, 0); push = dist.x + radius;
            } else if (dist.y < dist.z) {
                n = float3(0, sign(d.y), 0); push = dist.y + radius;
            } else {
                n = float3(0, 0, sign(d.z)); push = dist.z + radius;
            }
            // CLAMP: a deep-interior sheet vertex would otherwise be shoved the
            // full half-thickness in one step, tearing it from its outside
            // neighbour. Climb out coherently instead (see maxSDFPush docstring).
            float disp = min(push * u.collideStiffness, maxPush);
            pos  += n * disp;
            prev += n * disp * 0.5;
            hit   = true;
            return;
        }
    }

    float3 delta = probe - closest;
    float  dist  = length(delta);
    if (dist >= radius) return;

    float3 n;
    if (dist < 1e-6) {
        // Perpendicular-to-segment fallback with ownerID asymmetry; see the
        // long comment in pbdSDFCollide (preserved above for git-archaeology).
        float3 perp;
        if (kind == 1u) {
            float3 ab    = col.b.xyz - col.a.xyz;
            float  ablen = length(ab);
            if (ablen > 1e-6) {
                float3 t  = ab / ablen;
                perp = cross(t, float3(0, 1, 0));
                if (length(perp) < 1e-3) {
                    perp = cross(t, float3(1, 0, 0));
                }
                perp = normalize(perp);
            } else {
                perp = float3(0, 1, 0);
            }
        } else {
            perp = float3(0, 1, 0);
        }
        float s = (u.ownerID < col.meta.x) ? 1.0 : -1.0;
        n = perp * s;
        dist = 0.0;
    } else {
        n = delta / dist;
    }

    // ── STACKED-CONTACT DOWNWARD-PUSH REDIRECTION ────────────────────────────
    //
    // When the SDF gradient is "stacked" (mostly vertical), the lower body
    // gets pushed down by the upper one. That downward push is wasted: the
    // floor (or another tube further below) cancels it, gravity returns the
    // upper body, bodies stay interlocked indefinitely. The QA log showed
    // pairs stuck at 12-14 % overlap for >15 seconds because of this.
    //
    // Two conditions trigger the redirect:
    //
    //  (A) Floor-bound particle (probe.y ≤ floorY + floorBand) with any
    //      downward push (n.y < -0.1). The floor will absorb the push, so
    //      ANY downward component is wasted.
    //
    //  (B) Steeply downward gradient (n.y < -0.7) regardless of floor
    //      proximity. This catches mid-pile cases where the "lower" body
    //      isn't on the floor itself but is being squeezed between an upper
    //      tube and yet another tube below it. The 0.7 threshold (≈ 45° below
    //      horizontal) is steep enough that falling tubes that haven't
    //      stacked yet still get their natural downward push — their
    //      gradients are typically -0.3 to -0.5, slanted.
    //
    // Redirect the push into the horizontal plane so the body slides OUT
    // from under instead of impossibly pushing into a blocked direction.
    //
    // Direction priority:
    //   1. If gradient has a non-trivial horizontal component, amplify it.
    //   2. Otherwise (purely vertical stack), use the perpendicular to the
    //      upper body's segment in the floor plane, with ownerID-asymmetric
    //      sign so the two bodies always push opposite ways.
    bool floorBlocked = (probe.y <= u.floorY + u.floorBand) && n.y < -0.1;
    bool steepStack   = n.y < -0.7;
    if (floorBlocked || steepStack) {
        float3 horiz    = float3(n.x, 0, n.z);
        float  horizMag = length(horiz);
        if (horizMag > 0.1) {
            n = horiz / horizMag;
        } else if (kind == 1u) {
            float3 ab    = col.b.xyz - col.a.xyz;
            float3 abH   = float3(ab.x, 0, ab.z);
            float  abLen = length(abH);
            if (abLen > 1e-3) {
                float3 t    = abH / abLen;
                float3 perp = float3(-t.z, 0, t.x);
                float  s    = (u.ownerID < col.meta.x) ? 1.0 : -1.0;
                n = perp * s;
            }
            // else: upper body's spine is itself vertical and aligned over
            // ours — a flag-pole-on-flag-pole scenario. Leave n alone; the
            // ownerID-asymmetric fallback in the dist<1e-6 branch handles
            // exact coincidence, and non-coincident cases are too rare to
            // optimise for.
        }
    }

    // Pre-push velocity decomposition — must come BEFORE the position update
    // or the push reads back as a 19 m/s outward kick in the next Verlet step
    // (the bug that produced the "wild bounce on impact" QA report). Position
    // is the contact response, velocity must not double-count it.
    float3 vel  = pos - prev;
    float  vN   = dot(vel, n);
    float3 velT = vel - n * vN;

    float push = min((radius - dist) * u.collideStiffness, maxPush);
    pos += n * push;

    float  vNNew  = max(vN, 0.0);
    velT *= u.collideFriction;
    float3 velNew = velT + n * vNNew;
    prev = pos - velNew;
    hit  = true;
}

kernel void pbdSDFCollide(
    device PBDParticle*       particles [[ buffer(0) ]],
    device const PBDCollider* colliders [[ buffer(1) ]],
    constant PBDUniforms&     u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    if (u.colliderCount == 0u)  return;

    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;
    bool   hit  = false;

    // ── Probe points ────────────────────────────────────────────────────────
    //
    // Per-particle SDF alone misses crossings that fall in the GAP between
    // particles. With pbdSpine = 4 and a ~4 m hot dog, particles are ~1.3 m
    // apart on the spine and 2R = 0.64 m collision range, so two hot dogs
    // crossing on the floor between a body's particles produced sep = 0 and
    // visible interpenetration in the diagnostic log (pair A50↔B52).
    //
    // Add two extra probes per particle: the midpoint of the segment to the
    // previous particle, and the midpoint of the segment to the next particle.
    // The contact push always targets THIS thread's particle. The neighbour
    // thread runs the mirror check from the other side of the same segment,
    // so when a midpoint hit is detected the whole segment translates
    // uniformly — both endpoints push by the same amount, no race because
    // each thread writes only to its own particle slot.
    //
    // Probes are precomputed once at the start of the kernel so subsequent
    // pushes (which modify `pos`) don't drift the midpoint location during
    // the inner collider loop.
    float3 probes[3];
    uint   numProbes = 0u;
    probes[numProbes++] = pos;
    if (id > 0u) {
        float3 prevP = particles[id - 1u].positionAndInvMass.xyz;
        probes[numProbes++] = (prevP + pos) * 0.5;
    }
    if (id + 1u < u.particleCount) {
        float3 nextP = particles[id + 1u].positionAndInvMass.xyz;
        probes[numProbes++] = (pos + nextP) * 0.5;
    }

    for (uint k = 0u; k < numProbes; ++k) {
        for (uint c = 0u; c < u.colliderCount; ++c) {
            sdfContact(probes[k], colliders[c], u, pos, prev, hit);
        }
    }

    if (hit) {
        particles[id].positionAndInvMass.xyz = pos;
        particles[id].prevPositionAndPad.xyz = prev;
    }
}


// ── Batch SDF structs ────────────────────────────────────────────────────────

struct PBDSDFBatchUniforms {
    uint  totalParticles;
    uint  colliderCount;
    float selfRadius;
    float collideStiffness;
    float collideFriction;   // 0…1 tangential-vel RETENTION on contact (lower = stickier)
};

struct PBDTubeBoundary {
    uint start;   // first index in flat buffer for this tube
    uint count;   // number of particles in this tube
};

// ── sdfContactBatch ──────────────────────────────────────────────────────────
//
// Identical logic to `sdfContact` above, but takes ownerID / selfRadius /
// collideStiffness as explicit scalar arguments instead of reading them from
// PBDUniforms. Used by pbdSDFCollideBatch where a single dispatch covers all
// tubes and each particle carries its owner via prevPositionAndPad.w.
//
// collideFriction is hard-wired to 0.70 (the same default as PBDSolver) since
// the batch path doesn't carry per-tube friction into the flat buffer — the
// per-tube `dispatchSDFCollide` path still has it.  A future extension could
// stamp friction into a second unused float field if per-tube friction variance
// matters at 100+ body counts.
static void sdfContactBatch(float3 probe,
                             PBDCollider col,
                             uint ownerID,
                             float selfRadius,
                             float collideStiffness,
                             float collideFriction,
                             thread float3& pos,
                             thread float3& prev,
                             thread bool& hit) {
    if (col.meta.x == ownerID) return;

    uint  kind   = as_type<uint>(col.a.w);
    float radius = col.b.w + selfRadius;

    float3 closest;
    if (kind == 0u) {
        // Sphere.
        closest = col.a.xyz;
    } else if (kind == 1u) {
        // Capsule.
        closest = closestOnSegment(probe, col.a.xyz, col.b.xyz);
    } else {
        // Box (kind == 2u).
        float3 d  = probe - col.a.xyz;
        float3 he = col.b.xyz;
        float3 q  = clamp(d, -he, he);
        float3 outside = d - q;
        float  outLen  = length(outside);
        if (outLen > 1e-6) {
            if (outLen >= radius) return;
            float3 n    = outside / outLen;
            float  push = (radius - outLen) * collideStiffness;
            // Tangential friction (same model as the capsule path below) — without
            // it the box WALLS are frictionless and a stuffed column just slides
            // down/around them. Lower collideFriction = stickier walls = the column
            // jams and holds, riding up as a mass.
            float3 vel  = pos - prev;
            float  vN   = dot(vel, n);
            float3 velT = vel - n * vN;
            pos += n * push;
            velT *= collideFriction;
            prev = pos - (velT + n * max(vN, 0.0));
            hit   = true;
            return;
        } else {
            float3 dist = he - abs(d);
            float3 n;
            float  push;
            if (dist.x < dist.y && dist.x < dist.z) {
                n = float3(sign(d.x), 0, 0); push = dist.x + radius;
            } else if (dist.y < dist.z) {
                n = float3(0, sign(d.y), 0); push = dist.y + radius;
            } else {
                n = float3(0, 0, sign(d.z)); push = dist.z + radius;
            }
            push *= collideStiffness;
            float3 vel  = pos - prev;
            float  vN   = dot(vel, n);
            float3 velT = vel - n * vN;
            pos += n * push;
            velT *= collideFriction;
            prev = pos - (velT + n * max(vN, 0.0));
            hit   = true;
            return;
        }
    }

    float3 delta = probe - closest;
    float  dist  = length(delta);
    if (dist >= radius) return;

    float3 n;
    if (dist < 1e-6) {
        float3 perp;
        if (kind == 1u) {
            float3 ab    = col.b.xyz - col.a.xyz;
            float  ablen = length(ab);
            if (ablen > 1e-6) {
                float3 t  = ab / ablen;
                perp = cross(t, float3(0, 1, 0));
                if (length(perp) < 1e-3) {
                    perp = cross(t, float3(1, 0, 0));
                }
                perp = normalize(perp);
            } else {
                perp = float3(0, 1, 0);
            }
        } else {
            perp = float3(0, 1, 0);
        }
        float s = (ownerID < col.meta.x) ? 1.0 : -1.0;
        n = perp * s;
        dist = 0.0;
    } else {
        n = delta / dist;
    }

    // Stacked-contact downward-push redirection (same logic as sdfContact).
    // floorY / floorBand not available here — we only apply the steepStack check.
    bool steepStack = n.y < -0.7;
    if (steepStack) {
        float3 horiz    = float3(n.x, 0, n.z);
        float  horizMag = length(horiz);
        if (horizMag > 0.1) {
            n = horiz / horizMag;
        } else if (kind == 1u) {
            float3 ab    = col.b.xyz - col.a.xyz;
            float3 abH   = float3(ab.x, 0, ab.z);
            float  abLen = length(abH);
            if (abLen > 1e-3) {
                float3 t    = abH / abLen;
                float3 perp = float3(-t.z, 0, t.x);
                float  s    = (ownerID < col.meta.x) ? 1.0 : -1.0;
                n = perp * s;
            }
        }
    }

    float3 vel  = pos - prev;
    float  vN   = dot(vel, n);
    float3 velT = vel - n * vN;

    float push = (radius - dist) * collideStiffness;
    pos += n * push;

    float  vNNew  = max(vN, 0.0);
    velT *= collideFriction;
    float3 velNew = velT + n * vNNew;
    prev = pos - velNew;
    hit  = true;
}


// ── KERNEL: pbdCopyToFlat ────────────────────────────────────────────────────
//
// Per-tube kernel: copies one tube's particles into a scene-level flat buffer
// at `dstBase`, stamping the tube's `ownerId` into `prevPositionAndPad.w`
// (normally unused/zero). The batch SDF kernel reads this stamp to skip
// self-collisions without per-tube PBDUniforms.

kernel void pbdCopyToFlat(
    device const PBDParticle* src     [[ buffer(0) ]],
    device       PBDParticle* dst     [[ buffer(1) ]],
    constant     uint&        dstBase [[ buffer(2) ]],
    constant     uint&        ownerId [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    PBDParticle p = src[id];
    p.prevPositionAndPad.w = as_type<float>(ownerId);
    dst[dstBase + id] = p;
}


// ── KERNEL: pbdSDFCollideBatch ───────────────────────────────────────────────
//
// Single dispatch covering ALL tubes' particles at once. Reads ownerID from
// prevPositionAndPad.w to skip self-collisions; respects tube boundaries for
// midpoint probes. Replaces N × 4-thread per-tube dispatchSDFCollide calls
// with one dispatch of N*4 threads at much better GPU occupancy.

kernel void pbdSDFCollideBatch(
    device       PBDParticle*         particles  [[ buffer(0) ]],
    device const PBDCollider*         colliders  [[ buffer(1) ]],
    constant     PBDSDFBatchUniforms& u          [[ buffer(2) ]],
    device const PBDTubeBoundary*     boundaries [[ buffer(3) ]],
    constant     uint&                tubeCount  [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.totalParticles) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned

    float3 pos   = p.positionAndInvMass.xyz;
    float3 prev  = p.prevPositionAndPad.xyz;
    uint   ownID = as_type<uint>(p.prevPositionAndPad.w);
    bool   hit   = false;

    // Find tube boundary for this particle (linear scan — N<=100 in practice).
    uint tubeStart = 0u, tubeEnd = u.totalParticles;
    for (uint t = 0u; t < tubeCount; ++t) {
        uint s = boundaries[t].start;
        uint e = s + boundaries[t].count;
        if (id >= s && id < e) { tubeStart = s; tubeEnd = e; break; }
    }

    float3 probes[3];
    uint   numProbes = 0u;
    probes[numProbes++] = pos;
    if (id > tubeStart) {
        probes[numProbes++] = (particles[id - 1u].positionAndInvMass.xyz + pos) * 0.5;
    }
    if (id + 1u < tubeEnd) {
        probes[numProbes++] = (pos + particles[id + 1u].positionAndInvMass.xyz) * 0.5;
    }

    for (uint k = 0u; k < numProbes; ++k) {
        for (uint c = 0u; c < u.colliderCount; ++c) {
            sdfContactBatch(probes[k], colliders[c], ownID,
                            u.selfRadius, u.collideStiffness, u.collideFriction, pos, prev, hit);
        }
    }

    if (hit) {
        particles[id].positionAndInvMass.xyz = pos;
        particles[id].prevPositionAndPad.xyz  = prev;
        // ownerID stamp (.w) stays in flat buffer; stripped by pbdCopyFromFlat
    }
}


// ── KERNEL: pbdCopyFromFlat ──────────────────────────────────────────────────
//
// Per-tube kernel: copies updated particles back from the flat buffer at
// `srcBase` into the per-tube particle buffer, clearing the ownerID stamp
// in prevPositionAndPad.w so the buffer is clean for the next substep.

kernel void pbdCopyFromFlat(
    device const PBDParticle* src     [[ buffer(0) ]],
    device       PBDParticle* dst     [[ buffer(1) ]],
    constant     uint&        srcBase [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    PBDParticle p = src[srcBase + id];
    p.prevPositionAndPad.w = 0.0;  // clear ownerID stamp
    dst[id] = p;
}


// The following block is the original inline SDF body — leaving the original
// numeric/control flow on disk lets future archaeology trace the per-collider
// logic that was extracted into `sdfContact` above. Compile-only no-op.
#if 0
kernel void pbdSDFCollide_inlineLegacy(
    device PBDParticle*       particles [[ buffer(0) ]],
    device const PBDCollider* colliders [[ buffer(1) ]],
    constant PBDUniforms&     u         [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    if (u.colliderCount == 0u)  return;

    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;
    bool   hit  = false;

    for (uint c = 0u; c < u.colliderCount; ++c) {
        PBDCollider col = colliders[c];
        if (col.meta.x == u.ownerID) continue;

        uint   kind = as_type<uint>(col.a.w);
        // Effective contact distance = collider's surface radius + this body's
        // radius. Particles are at the centreline of an R-radius tube, not on
        // its surface — so two R-tubes must keep spine centres 2R apart for
        // their surfaces to just touch. Without this term we get the visible
        // half-overlap reported in QA: the particles think they're separated
        // when in fact their swept tube surfaces are 50 % interpenetrated.
        float  radius = col.b.w + u.selfRadius;

        float3 closest;
        if (kind == 0u) {
            // Sphere — closest point is the centre.
            closest = col.a.xyz;
        } else if (kind == 1u) {
            // Capsule — closest point on the spine segment.
            closest = closestOnSegment(pos, col.a.xyz, col.b.xyz);
        } else {
            // Box (kind == 2u, axis-aligned in world space).
            // Project to box surface: if inside, find the nearest face; if
            // outside, clamp to half-extents.
            float3 d  = pos - col.a.xyz;
            float3 he = col.b.xyz;
            float3 q  = clamp(d, -he, he);
            // For an inside hit (q == d), the SDF gradient is the axis with the
            // smallest face distance. Handle outside (q != d) first.
            float3 outside = d - q;
            float  outLen  = length(outside);
            if (outLen > 1e-6) {
                // Outside the box. Pretend the box has a 0 radius unless the
                // caller stored one in b.w (rounded box).
                if (outLen >= radius) continue;
                float3 n = outside / outLen;
                float  push = radius - outLen;
                pos  += n * push * u.collideStiffness;
                prev += n * push * u.collideStiffness * 0.5;
                hit   = true;
                continue;
            } else {
                // Inside the box — push out along the nearest face normal.
                float3 dist = he - abs(d);
                float3 n;
                float  push;
                if (dist.x < dist.y && dist.x < dist.z) {
                    n = float3(sign(d.x), 0, 0); push = dist.x + radius;
                } else if (dist.y < dist.z) {
                    n = float3(0, sign(d.y), 0); push = dist.y + radius;
                } else {
                    n = float3(0, 0, sign(d.z)); push = dist.z + radius;
                }
                pos  += n * push * u.collideStiffness;
                prev += n * push * u.collideStiffness * 0.5;
                hit   = true;
                continue;
            }
        }

        float3 delta = pos - closest;
        float  dist  = length(delta);
        if (dist >= radius) continue;

        // Degenerate case: particle coincident with the closest point on the
        // other body's segment.
        //
        // Two earlier attempts failed:
        //   1. n = (0,1,0) for both bodies — the symmetric world-up fallback.
        //      Both bodies translate together, sep stays at 0 forever (seen
        //      as the A1↔B4 99% lock in the diagnostic log).
        //   2. n = (±1,0,0) per-ownerID — asymmetric world-X fallback. Fixed
        //      the stacked case, but TWO PARALLEL BODIES WHOSE SPINES LIE
        //      ALONG WORLD-X just slide along their own length: every push
        //      moves a particle along its own segment, so the next iteration
        //      it's still on the other body's segment and stays degenerate.
        //
        // Real fix: pick a direction PERPENDICULAR to the collider's segment.
        // That way, no matter which world axis the bodies happen to align
        // with, the push is always orthogonal to their shared direction and
        // each iteration actually increases the spine-to-spine distance.
        // Asymmetric sign per ownerID guarantees opposite push for the two
        // bodies so they pull apart rather than drift together.
        float3 n;
        if (dist < 1e-6) {
            float3 perp;
            if (kind == 1u) {
                // Capsule: use the capsule's own segment direction.
                float3 ab    = col.b.xyz - col.a.xyz;
                float  ablen = length(ab);
                if (ablen > 1e-6) {
                    float3 t  = ab / ablen;
                    // cross(t, world-up) is perpendicular to both; degenerate
                    // when t is near-vertical, in which case cross with
                    // world-X gives a different valid perpendicular.
                    perp = cross(t, float3(0, 1, 0));
                    if (length(perp) < 1e-3) {
                        perp = cross(t, float3(1, 0, 0));
                    }
                    perp = normalize(perp);
                } else {
                    perp = float3(0, 1, 0);
                }
            } else {
                // Sphere / box / unknown: no segment direction available,
                // fall back to world-up. Asymmetric sign still separates.
                perp = float3(0, 1, 0);
            }
            float s = (u.ownerID < col.meta.x) ? 1.0 : -1.0;
            n = perp * s;
            dist = 0.0;
        } else {
            n = delta / dist;
        }

        // Decompose the PRE-push velocity (Verlet implicit vel = pos - prev)
        // BEFORE moving the particle. Doing this after the push would treat
        // the position correction as an additional impulse — at substep dt of
        // 1/120s a depth-0.16m push reads as a 19 m/s outward kick — which
        // is exactly the "wild reaction on impact" symptom in QA, and the
        // root cause of pile creep (each substep's tiny gravity overlap got
        // a matching velocity kick that the constraint solver propagated up
        // through the chain). The PBD-correct response is: position is the
        // contact response, velocity must not double-count it.
        float3 vel  = pos - prev;
        float  vN   = dot(vel, n);
        float3 velT = vel - n * vN;

        // Apply the position correction.
        float push = (radius - dist) * u.collideStiffness;
        pos += n * push;

        // Reconstruct prev from the pre-push velocity decomposition:
        //   • Normal:    kill any inward component (vN ≤ 0 → 0). Outward
        //                motion is preserved — particles already separating
        //                don't get an artificial brake.
        //   • Tangent:   retain `collideFriction` fraction so sliding/grinding
        //                contacts bleed energy each substep. Default 0.35
        //                retention means a settling pile loses ~65 % of any
        //                lateral motion per contact tick.
        float  vNNew = max(vN, 0.0);
        velT *= u.collideFriction;
        float3 velNew = velT + n * vNNew;
        prev = pos - velNew;
        hit  = true;
    }

    if (hit) {
        particles[id].positionAndInvMass.xyz = pos;
        particles[id].prevPositionAndPad.xyz = prev;
    }
}
#endif  // close the legacy archaeological block opened above


// ── KERNEL 4b: XPBD velocity post-solve ──────────────────────────────────────
//
// After the position pass converges each substep, walk the distance constraints
// once more and damp the *relative* velocity component along each constraint's
// gradient. This is the velocity step of Müller-2020 XPBD (§3.5), the second
// part of the two-part end-on-kinking fix paired with non-zero primary
// compliance.
//
// For each constraint between particles i and j:
//
//   n      = (x_i − x_j) / |x_i − x_j|             (unit gradient at solved pos)
//   v_i    = x_i − prev_i                          (Verlet positional delta)
//   v_j    = x_j − prev_j
//   v_rel  = v_i − v_j
//   v_N    = dot(v_rel, n)                         (relative-velocity normal cmp)
//
//   v_N'   = −e · v_N                              (e = restitution coefficient)
//   Δv_N   = v_N' − v_N
//
//   Δv_i   = +w_i · n · Δv_N / (w_i + w_j)         (momentum-conserving split)
//   Δv_j   = −w_j · n · Δv_N / (w_i + w_j)
//
//   prev_i' = x_i − (v_i + Δv_i)                   (encode new velocity in prev)
//   prev_j' = x_j − (v_j + Δv_j)
//
// Restitution semantics:
//   e = 0   → fully inelastic, normal-component relative velocity zeroed.
//             Chain oscillations die in ONE substep. Best for piles.
//   e = 0.2 → 20 % bounce-back. The hot dog flexes on impact, rings briefly,
//             settles within ~5 substeps. The "feels alive, doesn't kink" sweet
//             spot called out in the design brief.
//   e = 1   → perpetual ring. Don't use for distance constraints.
//
// Why this fixes end-on kinking:
//   Pure XPBD position solve produces an implicit-velocity ghost: each
//   constraint correction adds Δx to a particle's position without touching
//   its prev, so the next integrate's Verlet velocity contains that Δx as a
//   free impulse. On an end-on landing the (0,1) constraint repeatedly corrects
//   up while the (1,2) constraint corrects down — the residual normal
//   velocities alternate and the chain accordions. Restituting the
//   normal-velocity component damps that exact oscillation channel without
//   affecting bulk motion (gravity, sliding) which lives in tangential
//   directions.
//
// GPU RACE CONDITION:
//   Writes prev_i AND prev_j. Two constraints sharing a particle would race.
//   The Swift side dispatches per colour group (even/odd primary, 3-colour
//   bending, uncoloured long-range) exactly like pbdConstraint.

kernel void pbdVelocitySolve(
    device PBDParticle*   particles   [[ buffer(0) ]],
    device PBDConstraint* constraints [[ buffer(1) ]],
    constant PBDUniforms& u           [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.constraintCount) return;
    PBDConstraint c = constraints[id];

    float3 posi  = particles[c.i].positionAndInvMass.xyz;
    float3 posj  = particles[c.j].positionAndInvMass.xyz;
    float  wi    = particles[c.i].positionAndInvMass.w;
    float  wj    = particles[c.j].positionAndInvMass.w;
    float  wSum  = wi + wj;
    if (wSum < 1e-6) return;                       // both particles pinned

    float3 delta = posi - posj;
    float  dist  = length(delta);
    if (dist < 1e-6) return;                       // degenerate; nothing to project onto
    float3 n = delta / dist;

    float3 previ = particles[c.i].prevPositionAndPad.xyz;
    float3 prevj = particles[c.j].prevPositionAndPad.xyz;
    float3 vi    = posi - previ;
    float3 vj    = posj - prevj;
    float  vN    = dot(vi - vj, n);

    // Target relative normal velocity = −restitution · current. Restitution 0
    // kills it entirely (the strongest damping); restitution 1 mirrors it
    // (perpetual oscillation — do not use).
    float  dvN  = (-u.velRestitution * vN) - vN;
    float3 imp  = n * (dvN / wSum);

    // Momentum-conserving impulse split. Encoded by shifting prev positions so
    // the next integrate step's Verlet velocity reflects the change.
    float3 viNew = vi + imp * wi;
    float3 vjNew = vj - imp * wj;
    particles[c.i].prevPositionAndPad.xyz = posi - viNew;
    particles[c.j].prevPositionAndPad.xyz = posj - vjNew;
}


// ── KERNEL 5: GPU collider rebuild (capsules from spine particles) ───────────
//
// Writes one PBDCollider capsule per spine segment for a single body, into a
// caller-supplied target buffer at a caller-supplied offset. Reads particles
// IN-FLIGHT from the same command buffer that just integrated them — so the
// SDF pass that follows sees post-integrate capsules, not the pre-integrate
// stale snapshot that the previous CPU-side rebuild produced.
//
// Why this kernel exists (and not a CPU rebuild):
//   The CPU rebuild that this replaces read `particles.contents` from a shared
//   MTLBuffer *while the GPU was still executing the previous substep's
//   command buffer* (we never called waitUntilCompleted, and didn't want to).
//   So colliders were one-substep stale at best, partially-written at worst.
//   That stale data showed up as a tiny systematic mis-correction in the SDF
//   pass — small per substep, but accumulating across many frames as visible
//   creep of one tube into a neighbour after both had settled.
//
//   Doing the rebuild on the GPU inside the same command buffer eliminates
//   both problems: ordering is enforced (integrate ▸ build ▸ constraints
//   ▸ SDF), and the read is from the GPU's own freshly-written particles.
//
// Layout per dispatch:
//   - threads = body's spineCount − 1 (one per capsule)
//   - reads particles[0..N], writes targetColliders[targetOffset .. + N-1]

struct BuildCollidersUniforms {
    uint  particleCount;   // = body's spineCount
    uint  targetOffset;    // index into targetColliders to start writing
    uint  ownerID;
    float radius;
};

kernel void pbdBuildCapsuleColliders(
    device const PBDParticle*     particles  [[ buffer(0) ]],
    device PBDCollider*           target     [[ buffer(1) ]],
    constant BuildCollidersUniforms& u       [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id + 1u >= u.particleCount) return;

    float3 a = particles[id].positionAndInvMass.xyz;
    float3 b = particles[id + 1u].positionAndInvMass.xyz;

    // Capsule layout: a.xyz = endpoint A, a.w = type tag (1.0 bit-cast for capsule),
    //                 b.xyz = endpoint B, b.w = radius,
    //                 meta.x = ownerID.
    PBDCollider c;
    c.a    = float4(a, as_type<float>((uint)1));
    c.b    = float4(b, u.radius);
    c.meta = uint4(u.ownerID, 0u, 0u, 0u);
    target[u.targetOffset + id] = c;
}


// ── KERNEL 6: Tube expansion (Catmull-Rom) ───────────────────────────────────
//
// Converts spineCount physics particles into a smooth cylindrical tube mesh
// by evaluating a Catmull-Rom spline at subSegments points per spine segment.
// One thread per VISUAL ring (not per spine particle).
//
// Total visual rings = (spineCount - 1) * subSegments + 1
// Vertex count       = totalRings * ringSegments + 2  (+ 2 flat end-cap centres)
//
// Catmull-Rom passes exactly through the physics particle positions (unlike
// Bézier), so the mesh surface touches every simulated point while the curve
// between them is smooth. With subSegments=4 and spineCount=4 you get 13
// visual rings — enough to look like a rubber sausage rather than a faceted
// polygon chain.
//
// TANGENT FRAME
//   Derived analytically from the Catmull-Rom first derivative so the tangent
//   is accurate everywhere along the curve, not just at the physics particles.
//   Fallback to world-forward when the tangent is nearly world-up to avoid
//   cross-product degeneracy.

// Catmull-Rom position at parameter t ∈ [0,1] through segment p1→p2
// with neighbouring control points p0 and p3.
static float3 crPos(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5 * (  2.0*p1
                  + (-p0 + p2) * t
                  + ( 2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2
                  + (-p0 + 3.0*p1 - 3.0*p2 + p3) * t3);
}

// Catmull-Rom first derivative (tangent direction, unnormalised).
static float3 crTan(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
    float t2 = t * t;
    return 0.5 * (  (-p0 + p2)
                  + 2.0*( 2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t
                  + 3.0*(-p0 + 3.0*p1 - 3.0*p2 + p3) * t2);
}

// Compute a stable perpendicular frame (bitangent, ringNorm) for a given
// tangent. Falls back to a perpendicular up when the tangent is near `worldUp`.
//
// `worldUp` is now caller-supplied so a 2-D-plane geometry (Hotdog Font
// glyphs in the XY plane) can pin it to Z and keep the ring orientation
// flip-free as the tangent rotates inside the plane. Pass (0, 1, 0) to get
// the original default.
static void frameFromTangent(float3 tangent, float3 worldUp,
                             thread float3& bitangent,
                             thread float3& ringNorm) {
    if (abs(dot(tangent, worldUp)) > 0.99) {
        // Pick a fallback that isn't parallel to the current worldUp.
        worldUp = (abs(worldUp.y) > 0.5) ? float3(0, 0, 1) : float3(0, 1, 0);
    }
    bitangent = normalize(cross(tangent, worldUp));
    ringNorm  = cross(bitangent, tangent);
}

// ── Rotation-minimizing frame (RMF) support ─────────────────────────────────
//
// Unit tangent of the RENDERED centerline at ring station `sid` ∈ [0,totalRings),
// matching the per-region tangents the kernel uses to PLACE each ring (cap rings
// use the constant endpoint-segment tangent; body rings use the Catmull-Rom
// tangent). Used to propagate the rotation-minimizing reference frame so it is
// consistent with the geometry the kernel actually emits.
static float3 stationTangent(device const PBDParticle* spine,
                             constant TubeExpandUniforms& u,
                             uint sid, uint totalRings, uint K) {
    float3 spineStart = sampleSpine(spine, u, 0u);
    float3 spineEnd   = sampleSpine(spine, u, u.spineCount - 1u);
    if (sid < K) {
        float3 t0 = sampleSpine(spine, u, 1u) - spineStart;
        return (length(t0) < 1e-6) ? float3(0, 0, 1) : normalize(t0);
    }
    if (sid >= totalRings - K) {
        float3 t1 = spineEnd - sampleSpine(spine, u, u.spineCount - 2u);
        return (length(t1) < 1e-6) ? float3(0, 0, 1) : normalize(t1);
    }
    uint  bodyIdx   = sid - K;
    uint  bodyRings = totalRings - 2u * K;
    float bodyT     = float(bodyIdx) / float(bodyRings - 1u);
    float spineT    = bodyT * float(u.spineCount - 1u);
    uint  segIdx    = uint(floor(spineT));
    if (segIdx >= u.spineCount - 1u) segIdx = u.spineCount - 2u;
    float t  = spineT - float(segIdx);
    uint  i0 = (segIdx > 0u) ? segIdx - 1u : 0u;
    uint  i1 = segIdx;
    uint  i2 = segIdx + 1u;
    uint  i3 = (segIdx + 2u < u.spineCount) ? segIdx + 2u : u.spineCount - 1u;
    float3 tg = crTan(sampleSpine(spine, u, i0), sampleSpine(spine, u, i1),
                      sampleSpine(spine, u, i2), sampleSpine(spine, u, i3), t);
    return (length(tg) < 1e-6) ? float3(0, 0, 1) : normalize(tg);
}

// Build the ring frame at station `id` by PROPAGATING a rotation-minimizing
// reference from station 0. Each thread independently walks 0→id, rotating the
// reference by the minimal rotation between consecutive station tangents (never
// degenerates: parallel tangents → no rotation, so no pinch and no twist). The
// reference is re-orthogonalized against this ring's tangent and the bitangent is
// built with the SAME handedness as frameFromTangent (bitangent = cross(tangent,
// ringNorm)), so winding/normals are unchanged — only the frame is now coherent.
static void rmfFrame(device const PBDParticle* spine,
                     constant TubeExpandUniforms& u,
                     uint id, uint totalRings, uint K,
                     float3 tangent,
                     thread float3& bitangent, thread float3& ringNorm) {
    float3 t0  = stationTangent(spine, u, 0u, totalRings, K);
    float3 a   = (abs(t0.x) < 0.9f) ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 ref = normalize(cross(t0, a));     // any stable perpendicular seed
    float3 tprev = t0;
    for (uint j = 1u; j <= id; j++) {
        float3 tj   = stationTangent(spine, u, j, totalRings, K);
        float3 axis = cross(tprev, tj);
        float  s    = length(axis);
        if (s > 1e-6f) {
            axis /= s;
            float ang = atan2(s, dot(tprev, tj));
            float ca = cos(ang), sa = sin(ang);
            // Rodrigues: rotate ref about axis by ang.
            ref = ref * ca + cross(axis, ref) * sa + axis * dot(axis, ref) * (1.0f - ca);
            ref = normalize(ref);
        }
        tprev = tj;
    }
    ringNorm  = normalize(ref - dot(ref, tangent) * tangent);
    bitangent = cross(tangent, ringNorm);
}

kernel void pbdTubeExpand(
    device const PBDParticle*    spine   [[ buffer(0) ]],
    device packed_float3*        verts   [[ buffer(1) ]],
    constant TubeExpandUniforms& u       [[ buffer(2) ]],
    device packed_float3*        normals [[ buffer(3) ]],
    device float4*               colors  [[ buffer(4) ]],  // uniform white + casing flag (a=0.75)
    uint id [[ thread_position_in_grid ]]
) {
    uint totalRings = (u.spineCount - 1u) * u.subSegments + 1u;
    if (id >= totalRings) return;

    // ── Straight-line mode (useStraightLine == 1, HotdogDropUltra) ───────────
    //
    // Pre-compute the tube's overall axis from spine[0] → spine[N-1].
    // In straight-line mode, ALL three regions (start cap, body, end cap)
    // use this axis instead of local Catmull-Rom tangents. The result is a
    // perfect constant-radius capsule (cylinder + two hemispheres) whose
    // orientation tracks the head-to-tail axis of the PBD chain.
    //
    // Why this fixes the lobing artifacts:
    //   With Catmull-Rom, any bend in the PBD chain at an interior knot creates
    //   a localized high-curvature region. A constant-radius tube swept along a
    //   bent centerline produces a visible silhouette bulge at each bend (more
    //   outer-radius surface is visible at the outer side of the curve). With
    //   4 interior knots, the result is 4 lobes. Straight-line rendering
    //   eliminates all bends → no bulges → smooth capsule.
    //
    // Physics capsules (pbdBuildCapsuleColliders) are NOT affected — they use
    // the raw PBD particle positions for accurate collision response. The visual
    // discrepancy for a nearly-rigid chain (pbdBendStiffness ~0.85) is <1 cm.
    float3 spineStart  = sampleSpine(spine, u, 0u);
    float3 spineEnd    = sampleSpine(spine, u, u.spineCount - 1u);
    float3 tubeAxisRaw = spineEnd - spineStart;
    float  tubeLen     = length(tubeAxisRaw);
    float3 tubeAxis    = (tubeLen > 1e-6f) ? (tubeAxisRaw / tubeLen) : float3(0, 0, 1);

    // ── Region split ─────────────────────────────────────────────────────────
    //
    // Reserve K rings at each end for the hemispherical cap. The remaining
    // (totalRings − 2K) rings are the cylindrical body and interpolate the
    // spine via Catmull-Rom. K is clamped so very short chains still render
    // (need at least 2 body rings to interpolate between).
    //
    // For each cap ring k ∈ [0, K-1]:
    //     α = (k + 1) / (K + 1)  ·  π/2
    //     ringRadius   = R · sin α
    //     tangentDrop  = R · cos α      (distance back from spine endpoint
    //                                    toward the pole)
    // Pole sits at full distance R past the spine endpoint. With K=3 the
    // hemisphere has 3 stratification rings + 1 pole fan, which reads as
    // smoothly rounded for tube radii at the scale of the hot dog.
    //
    // The cap ring normal is the true sphere normal: (vert − sphereCenter)/R,
    // which is the only way to make IBL/specular highlights wrap correctly
    // around the dome.
    // K=5 gives 5 stratification rings per hemisphere before the pole fan,
    // producing a smooth 5-ring dome instead of the faceted 3-ring dome (K=3).
    // maxK ensures at least 2 body rings remain: K ≤ (totalRings-2)/2.
    // With totalRings=29: maxK = (29-2)/3 = 9 → K = min(5,9) = 5. ✓
    uint maxK = max(1u, (totalRings - 2u) / 3u);
    uint K    = min(5u, maxK);

    float3 spinePos;
    float3 tangent;
    float  ringRadius;
    float3 capCenter   = float3(0);  // sphere centre for cap-ring normals
    bool   isCapRing   = false;
    // Saved body parameter for the normal-wrinkle pass below (set in body
    // region only; caps leave it at 0 so the wrinkle is suppressed there).
    float  savedBodyT  = 0.0f;

    if (id < K) {
        // ── Start cap region ──
        float3 s0 = spineStart;

        if (u.useStraightLine != 0u) {
            // Straight-line mode: use the overall tube axis for cap orientation.
            // Ensures the cap is perpendicular to the rendered cylinder axis,
            // not to the first spine segment (which may differ on a bent chain).
            tangent = tubeAxis;
        } else {
            // Catmull-Rom mode: end tangent from the first spine segment.
            // Derivative at t=0 with p0=p1=spine[0], p2=spine[1], p3=spine[2]:
            // simplifies to 0.5 * (spine[1] − spine[0]) — same direction as simple.
            float3 s1 = sampleSpine(spine, u, 1u);
            float3 t0 = s1 - s0;
            if (length(t0) < 1e-6) t0 = float3(0, 0, 1);
            tangent = normalize(t0);
        }

        float  alpha = (float(id) + 1.0f) / (float(K) + 1.0f) * (M_PI_F * 0.5f);
        float  s     = sin(alpha);
        float  c     = cos(alpha);
        // Tip radius = body radius × (1 − taperFactor); the cap hemisphere
        // scales to match the tapered end of the body rather than the raw radius.
        float  capR  = u.radius * max(1.0f - u.taperFactor, 0.01f);
        spinePos     = s0 - tangent * (capR * c);
        ringRadius   = capR * s;
        capCenter    = s0;
        isCapRing    = true;
    } else if (id >= totalRings - K) {
        // ── End cap region ──
        // Mirror of the start cap. kEnd = 0 is the ring nearest the body
        // (α ≈ π/2 ε short), kEnd = K-1 is the ring nearest the pole.
        uint kEnd = (totalRings - 1u) - id;  // 0 … K-1
        float3 sNm1 = spineEnd;

        if (u.useStraightLine != 0u) {
            // Straight-line mode: use the overall tube axis (same direction as
            // start cap) for consistent end-cap orientation.
            tangent = tubeAxis;
        } else {
            float3 sNm2 = sampleSpine(spine, u, u.spineCount - 2u);
            float3 t1   = sNm1 - sNm2;
            if (length(t1) < 1e-6) t1 = float3(0, 0, 1);
            tangent = normalize(t1);
        }

        float  alpha = (float(kEnd) + 1.0f) / (float(K) + 1.0f) * (M_PI_F * 0.5f);
        float  s     = sin(alpha);
        float  c     = cos(alpha);
        float  capR  = u.radius * max(1.0f - u.taperFactor, 0.01f);
        spinePos     = sNm1 + tangent * (capR * c);
        ringRadius   = capR * s;
        capCenter    = sNm1;
        isCapRing    = true;
    } else {
        // ── Body region ──
        uint bodyIdx = id - K;
        uint bodyRings = totalRings - 2u * K;        // ≥ 2 by clamp on K
        float  bodyT = float(bodyIdx) / float(bodyRings - 1u);  // ∈ [0, 1]

        if (u.useStraightLine != 0u) {
            // ── Straight-line mode: perfect constant-radius cylinder ─────────
            // Position is a linear interpolation along the spine[0]→spine[N-1]
            // axis. The tangent is constant (= tubeAxis) so the ring frame is
            // identical at every station — zero frame-seam artifacts.
            // Combined with no bends, this guarantees a smooth capsule silhouette.
            float halfLen  = tubeLen * 0.5f;
            float s        = (bodyT - 0.5f) * tubeLen;   // −halfLen … +halfLen
            float3 center  = (spineStart + spineEnd) * 0.5f;
            spinePos  = center + tubeAxis * s;
            tangent   = tubeAxis;
        } else {
            // ── Catmull-Rom mode (SoftServe, HotdogFont, etc.) ──────────────
            float  spineT = bodyT * float(u.spineCount - 1u);
            uint   segIdx = uint(floor(spineT));
            if (segIdx >= u.spineCount - 1u) segIdx = u.spineCount - 2u;
            float  t = spineT - float(segIdx);

            uint i0 = (segIdx > 0u)                ? segIdx - 1u : 0u;
            uint i1 = segIdx;
            uint i2 = segIdx + 1u;
            uint i3 = (segIdx + 2u < u.spineCount) ? segIdx + 2u : u.spineCount - 1u;

            float3 p0 = sampleSpine(spine, u, i0);
            float3 p1 = sampleSpine(spine, u, i1);
            float3 p2 = sampleSpine(spine, u, i2);
            float3 p3 = sampleSpine(spine, u, i3);

            spinePos = crPos(p0, p1, p2, p3, t);
            float3 tg = crTan(p0, p1, p2, p3, t);
            if (length(tg) < 1e-6) tg = float3(0, 0, 1);
            tangent = normalize(tg);
        }

        // sin(π·bodyT) peaks at 1.0 at the midpoint and approaches 0 at both
        // ends. taperFactor=0 → perfect constant-radius cylinder (HotdogDropUltra).
        float sinT     = sin(M_PI_F * bodyT);
        float taperedR = u.radius * ((1.0f + 0.0f * sinT) - u.taperFactor * (1.0f - sinT));
        ringRadius     = taperedR;
        savedBodyT     = bodyT;
    }

    float3 worldUp = (u.useFixedUp != 0u)
        ? float3(u.fixedUpX, u.fixedUpY, u.fixedUpZ)
        : float3(0, 1, 0);
    float3 bitangent, ringNorm;
    if (u.useRMF != 0u && u.useStraightLine == 0u) {
        // Coherent rotation-minimizing frame — no per-ring discontinuity, no pinch.
        rmfFrame(spine, u, id, totalRings, K, tangent, bitangent, ringNorm);
    } else {
        frameFromTangent(tangent, worldUp, bitangent, ringNorm);
    }

    // ── Cap pucker controls ──────────────────────────────────────
    //
    // Real tied/twisted frankfurter ends show ~6 radial creases running from
    // the rim toward the pole. We bake these into the cap-ring normals only —
    // the body element keeps its smooth cylindrical normal.
    //
    // The perturbation is in the AROUND-THE-RING tangent direction (the cap's
    // surface tangent perpendicular to both the radial and the tube axis), so
    // the normal tilts side-to-side as you walk around the ring. That's what
    // creates the alternating bright/dark stripes of a gathered casing.
    // Axial perturbation (toward/away from the tube) would create CONCENTRIC
    // rings around the pole, which is the wrong texture.
    //
    // Amplitude grows linearly toward the pole so the rim is undisturbed
    // (clean join to the body) and the apex carries the visible gather. The
    // value here is intentionally aggressive — IBL + bloom + clearcoat all
    // wash subtle normal detail flat, so anything below ~0.6 reads as smooth.
    const float puckerFolds = 6.0;
    // puckerMaxAmp at the RIM (not the pole). poleProgress runs 0 at pole → 1 at
    // rim, so the apex is always smooth (zero-gradient, per pepperoni-centre fix).
    // This eliminates the "puckered point / polar singularity" at the cap apex that
    // read as a dead CG tell in the low/top cells (see docs/known-issues/pepperoni-centre.md).
    // 0.6 amplitude at the rim gives visible gathered creases where the casing is
    // tied, without distorting the apex normals past 90°.
    //
    // Applies in BOTH spline and straight-line modes: the pucker is normals-only
    // (vertices stay on the exact hemisphere), so the straight-line capsule
    // silhouette invariant (no bulge/ridges/lobes) is untouched — only the cap
    // shading gains the gathered tied-end look that HotdogDrop+'s franks carry.
    //
    // Amplitude is mode-split: the Catmull-Rom path (Plus) keeps its original
    // 0.6 — under Plus's warm key + clearcoat the creases read as a tied end.
    // The straight-line path (Ultra, Illuminatorama) uses 0.35: round-30 review
    // found 0.6 under the deferred satin lobe shades into a deep navel/button
    // at the cap centre (balloon-knot read). Lower amplitude keeps the ring of
    // gathered creases near the cap base and leaves the apex near-smooth.
    const float puckerMaxAmp = (u.useStraightLine == 0u) ? 0.6f : 0.35f;
    float capBaseR = u.radius * max(1.0f - u.taperFactor, 0.01f);
    // poleProgress = 0 at the apex (ringRadius → 0), = 1 at the rim (ringRadius ≈ capR).
    // Pucker grows FROM the pole TO the rim, keeping the apex derivative-free.
    float poleProgress = saturate(ringRadius / max(capBaseR, 1e-4f));
    float puckerAmp = puckerMaxAmp * poleProgress;

    // ── Per-ring wrinkle phase (body rings only) ──────────────────────────────
    //
    // Hash the ring index to a random phase in [0, 2π) so adjacent rings have
    // independent pucker phases — no helical groove alignment, reads as organic.
    // Hoisted outside the loop since it depends only on `id`, not on `r`.
    // Cap rings also get this hash but the isCapRing branch ignores it.
    uint  ringSeed  = id * 1664525u + 1013904223u;   // LCG hash of ring index
    float ringPhase = float(ringSeed >> 8u) / float(1u << 24u) * 2.0f * M_PI_F;

    for (uint r = 0; r < u.ringSegments; r++) {
        float  theta   = (float(r) / float(u.ringSegments)) * 2.0 * M_PI_F;
        float  cosT    = cos(theta);
        float  sinT    = sin(theta);
        float3 radial  = cosT * ringNorm + sinT * bitangent;
        // tRing = d/dθ(radial). Unit length, tangent to the ring at this
        // vertex, perpendicular to both radial and the tube axial tangent.
        float3 tRing   = -sinT * ringNorm + cosT * bitangent;

        float3 vert;
        float3 normal;

        vert = spinePos + ringRadius * radial;

        if (isCapRing) {
            // Sphere normal from the cap centre, then tilted around the
            // pole axis with `cos(N·θ)` so the lighting reads as N radial
            // gathered creases. sin (rather than cos) here would just
            // rotate the crease pattern; cos keeps the first crease at θ=0.
            float3 n = normalize(vert - capCenter);
            float  bump = cos(puckerFolds * theta) * puckerAmp;
            normal = normalize(n + tRing * bump);
        } else {
            // Smooth constant-radius cylinder body — base normal is exactly radial.
            // Per-fragment grain (cooked-skin micro-texture) is applied in the
            // Illuminatorama G-buffer pass via soilValueNoise on world position.
            // It cannot be authored here at vertex frequency — barycentric
            // interpolation averages opposite-sign neighbours to ≈zero before
            // reaching a fragment. The casing flag (vertexColor.a = 0.75) below
            // tells the G-buffer pass which vertices belong to sausage casing.
            normal = radial;
        }

        verts  [id * u.ringSegments + r] = vert;
        normals[id * u.ringSegments + r] = normal;

        // ── Casing vertex colour: uniform white + casing flag ───────────────
        //
        // Matches HotdogDrop+'s sausage material: the frank is one solid
        // per-instance colour (hue-jittered in the controller); there is no
        // tip-char ramp and no per-vertex mottle in the reference. Surface
        // variation lives in the G-buffer pass's per-fragment wrinkle normals
        // + roughness mottle (the Drop+ texture maps, evaluated procedurally).
        //
        // Alpha = 0.75: casing flag for the Illuminatorama G-buffer pass. The pass
        // gates its per-fragment casing detail on alpha ∈ (0.6, 0.9) to distinguish
        // sausage casing (0.75) from normal geometry (1.0) and foliage (0.0).
        // RGB: dip-coat waterline multiplier (1,1,1 when the dip is disabled).
        float3 dipMul = dipCoatMultiplier(vert, u, id * 7919u + r);
        colors[id * u.ringSegments + r] = float4(dipMul, 0.75f);
    }

    // End-cap pole vertices. Apex of the start hemisphere sits at full R from
    // spine[0]; same for the end hemisphere from spine[N-1]. Pole normals are
    // ±tangent (the sphere normal at the apex).
    uint capBase = totalRings * u.ringSegments;
    float poleR = u.radius * max(1.0f - u.taperFactor, 0.01f);
    if (id == 0u) {
        float3 poleAxis;
        if (u.useStraightLine != 0u) {
            // Straight-line mode: pole is on the overall tube axis.
            poleAxis = tubeAxis;
        } else {
            float3 s0 = spineStart;
            float3 s1 = sampleSpine(spine, u, 1u);
            float3 t0 = s1 - s0;
            if (length(t0) < 1e-6) t0 = float3(0, 0, 1);
            poleAxis = normalize(t0);
        }
        verts  [capBase] = spineStart - poleAxis * poleR;
        normals[capBase] = -poleAxis;
        // Uniform casing colour + casing flag (matches the body rings — a
        // mismatched pole alpha would interpolate across the cap fan and
        // partially defeat the (0.6, 0.9) casing gate).
        colors [capBase] = float4(dipCoatMultiplier(verts[capBase], u, 0xA001u), 0.75f);
    }
    if (id == totalRings - 1u) {
        float3 poleAxis;
        if (u.useStraightLine != 0u) {
            // Straight-line mode: pole is on the overall tube axis.
            poleAxis = tubeAxis;
        } else {
            float3 sNm2 = sampleSpine(spine, u, u.spineCount - 2u);
            float3 t1   = spineEnd - sNm2;
            if (length(t1) < 1e-6) t1 = float3(0, 0, 1);
            poleAxis = normalize(t1);
        }
        verts  [capBase + 1u] = spineEnd + poleAxis * poleR;
        normals[capBase + 1u] = poleAxis;
        // Uniform casing colour + casing flag (see start-pole comment).
        colors [capBase + 1u] = float4(dipCoatMultiplier(verts[capBase + 1u], u, 0xA002u), 0.75f);
    }
}


// ── KERNEL: swept-tube interior confinement (Hotdog Waterslide Descent) ──────
//
// Keeps a frank's spine particles INSIDE the analytic swept tube — an inside-out
// cylinder of radius `tubeRadius` swept along the path centerline. The SDF
// `PBDCollider` primitives (sphere/capsule/box) can only push particles OUT of a
// solid shape, so a *curved tube interior* needs a dedicated one-sided
// constraint: the swept-tube generalisation of the vertical cup-wall already in
// `pbdSelfCollide` (`wallEnabled` / `wallRadius`).
//
// Per particle: search a LOCAL arc window around the owning frank's current
// arc-centre `sCenter` (passed per dispatch) for the nearest path sample, build
// that sample's cross-section basis (right, up), and if the particle's radial
// offset exceeds `tubeRadius − bodyRadius`, push it back inward and cancel the
// outward radial velocity. Windowing by the frank's OWN sCenter is what makes a
// path that passes near itself safe — a frank is only ever confined to ITS
// strand of track, never grabbed by a different nearby strand.
//
// Dispatched per frank (one tube's spine = one dispatch), so `particles` is that
// frank's spine buffer and `sCenter` is its scalar arc-centre. The path buffers
// are shared, read-only, uniform-arc samples (sample i at arc = totalLength·i/N).
struct PBDSweptTubeUniforms {
    uint  particleCount;   // spine particles for THIS frank
    uint  pathSamples;     // number of uniform-arc samples in the path buffers
    float totalLength;     // path length (m)
    float sCenter;         // this frank's current arc-centre (m)
    float windowArc;       // ± arc searched around sCenter for the nearest sample (m)
    float tubeRadius;      // flume interior radius (m)
    float bodyRadius;      // frank radius (m) — spine centres confined to R − r
    float stiffness;       // 0…1 fraction of the penetration corrected per substep
    float maxPush;         // > 0 → clamp per-substep inward push (m); 0 = unlimited
    float wallFriction;    // 0…1 RETENTION of the circumferential (around-the-tube)
                           // velocity per substep while in contact with the wall.
                           // 1 = frictionless (the marble-in-a-bowl orbiting bug);
                           // <1 damps the slosh so the dog settles to the channel
                           // floor and rides there. Does NOT touch along-path speed.
    float pad1; float pad2;
};

kernel void pbdSweptTubeConfine(
    device PBDParticle*            particles [[ buffer(0) ]],
    device const float4*           pathPos   [[ buffer(1) ]],   // xyz = centerline pos
    device const float4*           pathRight [[ buffer(2) ]],   // xyz = cradle right
    device const float4*           pathUp    [[ buffer(3) ]],   // xyz = cradle up
    constant PBDSweptTubeUniforms& u         [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= u.particleCount) return;
    if (u.pathSamples < 2u) return;
    PBDParticle p = particles[id];
    if (p.positionAndInvMass.w == 0.0) return;  // pinned (cheap guard; franks are dynamic)

    float3 pos  = p.positionAndInvMass.xyz;
    float3 prev = p.prevPositionAndPad.xyz;

    float dxS = u.totalLength / float(u.pathSamples);
    if (dxS < 1e-6) return;
    int last    = int(u.pathSamples) - 1;
    int iCenter = int(round(u.sCenter / dxS));
    int halfN   = int(ceil(u.windowArc / dxS)) + 1;
    int iLo = max(0,    iCenter - halfN);
    int iHi = min(last, iCenter + halfN);

    // Nearest centerline sample in the local window (one-way path, no wrap).
    int   best   = iLo;
    float bestD2 = 1e30;
    for (int i = iLo; i <= iHi; ++i) {
        float3 d  = pos - pathPos[i].xyz;
        float  d2 = dot(d, d);
        if (d2 < bestD2) { bestD2 = d2; best = i; }
    }

    float3 c     = pathPos[best].xyz;
    float3 right = pathRight[best].xyz;
    float3 up    = pathUp[best].xyz;
    float3 off   = pos - c;
    float  a     = dot(off, right);
    float  b     = dot(off, up);
    float  rLen  = sqrt(a * a + b * b);
    float  rMax  = max(0.0f, u.tubeRadius - u.bodyRadius);

    if (rLen > 1e-6) {
        float2 radial = float2(a, b) / rLen;              // outward unit in (right,up)
        float3 inward = -(radial.x * right + radial.y * up);
        float3 vel    = pos - prev;

        // Radial confinement: push back inside the wall + kill the outward radial
        // velocity so the frank settles against the wall instead of pinging out.
        if (rLen > rMax) {
            float maxPush = (u.maxPush > 0.0) ? u.maxPush : 1e30;
            float pen     = min((rLen - rMax) * u.stiffness, maxPush);
            pos += inward * pen;
            vel = pos - prev;
            float vR = dot(vel, inward);                  // > 0 = inward, < 0 = outward
            if (vR < 0.0) vel -= inward * vR;
        }

        // Circumferential WALL FRICTION: while in contact with the lower wall,
        // damp the velocity component that orbits AROUND the tube (perpendicular
        // to both the radial and the path tangent) so the dog settles to the
        // channel floor and rides there, instead of sloshing around the pipe like
        // a marble in a bowl. This is cross-sectional only — it never touches the
        // along-path (tangent) speed, so the current still carries the dog.
        if (rLen > 0.55f * rMax && u.wallFriction < 1.0f) {
            float3 circ = -radial.y * right + radial.x * up;   // ⟂ radial, in the cross-section
            float  vC   = dot(vel, circ);
            vel -= circ * vC * (1.0f - u.wallFriction);
        }

        prev = pos - vel;
    }

    particles[id].positionAndInvMass.xyz = pos;
    particles[id].prevPositionAndPad.xyz = prev;
}
