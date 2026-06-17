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
//   • paperBend          — dihedral hinge XPBD on triangle pairs (+ creasing).
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

kernel void paperSelfCollide(device PBDParticle* P          [[ buffer(0) ]],
                             device const uint* offsets      [[ buffer(1) ]],
                             device const uint* counts       [[ buffer(2) ]],  // per-bucket count (post-scatter cursor)
                             device const uint* sorted       [[ buffer(3) ]],
                             constant PaperHashUniforms& u   [[ buffer(4) ]],
                             uint id [[ thread_position_in_grid ]]) {
    if (id >= u.particleCount) return;
    PBDParticle pi = P[id];
    if (pi.positionAndInvMass.w == 0.0) return;   // pinned

    float3 pos = pi.positionAndInvMass.xyz;
    int3 base  = paperCellCoord(pos, u.cellSize);
    uint sheet = id / u.verticesPerSheet;
    uint local = id - sheet * u.verticesPerSheet;
    int  x = int(local % u.gridW), y = int(local / u.gridW);

    float minDist = 2.0 * u.radius;
    float3 delta  = float3(0.0);
    int    hits   = 0;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        uint h = paperHashCell(base + int3(dx, dy, dz), u.tableSize);
        uint start = offsets[h], cnt = counts[h];
        for (uint k = 0u; k < cnt; ++k) {
            uint j = sorted[start + k];
            if (j == id) continue;
            // Collide only between DIFFERENT sheets ("pages know about each
            // other"). Skipping all same-sheet pairs avoids a small sheet whose
            // internal spacing is below the collision radius puffing itself up.
            if (j / u.verticesPerSheet == sheet) continue;
            float3 d = pos - P[j].positionAndInvMass.xyz;
            float dist = length(d);
            if (dist < minDist && dist > 1e-6) {
                delta += (d / dist) * ((minDist - dist) * 0.5);
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
        P[id].prevPositionAndPad.xyz += corr * 0.5;    // bleed off inward velocity
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
