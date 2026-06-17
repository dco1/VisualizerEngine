#include <metal_stdlib>
using namespace metal;

// ── Foam.metal ────────────────────────────────────────────────────────────────
//
// Secondary particles ("spray / foam / mist") that ride on top of the MLS-MPM
// fluid. They are independent of the MPM solver after birth — pure ballistic
// integration with gravity + air drag, no grid coupling. The point is *visual*:
// what the fluid surface alone can't show is the bright disconnected droplets
// that fly off a splash, and the lingering surface foam that doesn't reset its
// position the moment the fluid recedes.
//
// Two kernels:
//
//   1. foamSpawn    — one thread per fluid particle. Hash-based RNG decides
//                     whether the particle is "energetic enough" to seed a
//                     foam droplet this substep. Energetic = fast + thin
//                     (low density estimate → near the surface). If so,
//                     claims a slot in the ring buffer via atomic_fetch_add
//                     and writes a fresh foam particle there.
//   2. foamAdvect   — one thread per foam slot. Integrates ballistically;
//                     kills (parks below scene) when life hits zero or the
//                     particle drops below the floor.
//
// Ring buffer: there is no free list. nextSlot wraps mod capacity, so the
// oldest live particles get clobbered first. Capacity sized so this happens
// well after natural death at typical spawn rates.

// ── Shared structs ──────────────────────────────────────────────────────────
//
// Must match the Swift mirrors in FoamSolver.swift. All float4 / uint4 —
// no bare float3 (same alignment rule as MLSParticle).

struct FoamParticle {
    float4 positionLife;   // xyz = position, w = remaining life in seconds
    float4 velocityAge;    // xyz = velocity, w = age in seconds
};

// Matches MLSMPM.metal's MLSParticle exactly so we can sample fluid state
// directly from the solver's buffer. Layout is fixed by the Swift mirror.
struct MLSParticle {
    float4   positionMass;
    float4   velocityJp;
    float3x3 C;
    float3x3 F;
    float4   misc;
};

struct FoamUniforms {
    uint4  counts;         // x = fluidCount, y = foamCapacity, z = randSeed, w = pad
    float4 dtTime;         // x = dt, y = elapsed, z/w = pad
    float4 gravity;        // xyz = gravity, w = drag (1/s)
    float4 bounds;         // xyz = boundsMin, w = lifeMax (s)
    float4 boundsMax;      // xyz = boundsMax, w = parkY (where dead particles sit)
    float4 spawn;          // x = minSpeed, y = maxDensity, z = spawnProb, w = spawnSpeedJitter
};

// ── Tiny PCG hash + RNG ─────────────────────────────────────────────────────
//
// We need cheap independent randoms per (particle, frame). PCG-style mix from
// Jarzynski & Olano 2020 "Hash Functions for GPU Rendering" — single uint in,
// single uint out, ~6 ALU ops, no state. Folding the seed in lets every frame
// produce a different random for the same particle without keeping per-particle
// RNG state.

inline uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand01(uint a, uint b) {
    return float(pcg_hash(a ^ (b * 0x9E3779B9u))) * (1.0f / 4294967296.0f);
}

inline float3 rand3(uint a, uint b) {
    return float3(
        rand01(a, b ^ 0x12345u),
        rand01(a, b ^ 0xABCDEu),
        rand01(a, b ^ 0x5A5A5u)
    );
}

// ── KERNEL 1 — Spawn ────────────────────────────────────────────────────────
//
// One thread per fluid particle. Decides probabilistically whether this
// particle should emit a foam droplet this substep. Criteria:
//
//   speed > minSpeed                  — only fast particles produce spray
//   density < maxDensity              — only thin / near-surface particles
//                                        (bulk interior particles never spray)
//   rand01() < baseProb * energyBoost — probabilistic, with extra weight for
//                                        upward-moving particles (splashes)
//
// If accepted, atomically grabs the next slot in the ring buffer and writes
// the new foam particle there, with velocity = fluid velocity + small jitter
// (so the foam doesn't move exactly in lock-step with the parent particle —
// reads as droplet break-off).

kernel void foamSpawn(
    device const MLSParticle* fluid       [[ buffer(0) ]],
    device FoamParticle*      foam        [[ buffer(1) ]],
    device atomic_uint*       nextSlot    [[ buffer(2) ]],
    constant FoamUniforms&    U           [[ buffer(3) ]],
    uint                      pid         [[ thread_position_in_grid ]])
{
    if (pid >= U.counts.x) return;

    MLSParticle p = fluid[pid];
    float3 vel = p.velocityJp.xyz;
    float  speed = length(vel);
    float  density = p.misc.x;

    if (speed < U.spawn.x) return;
    if (density > U.spawn.y) return;

    // Frame seed mixed with particle id → independent per-frame, per-particle RNG.
    uint frameSeed = U.counts.z;
    float r = rand01(pid, frameSeed);

    // Energy boost: particles going UP get a heftier spawn probability —
    // splashes are the visually-loud event, and a sheet of fluid sliding
    // sideways shouldn't emit foam at the same rate as a vertical crown.
    float upBoost = max(0.0f, vel.y) * 0.5f + 1.0f;     // [1, ∞)
    float prob = U.spawn.z * upBoost;
    if (r > prob) return;

    // Claim a ring-buffer slot. Atomic-inc with no mod; we let the uint wrap
    // naturally and reduce mod capacity when reading.
    uint slotRaw = atomic_fetch_add_explicit(nextSlot, 1u, memory_order_relaxed);
    uint slot = slotRaw % U.counts.y;

    // Position: at the parent particle, with a small offset to avoid stacking
    // multiple foam particles at exactly the same point (which renders as a
    // single bright dot regardless of count).
    float3 jitter = (rand3(pid, frameSeed ^ 0xF00Du) - 0.5f) * 0.03f;
    float3 pos = p.positionMass.xyz + jitter;

    // Velocity: parent velocity plus a small random nudge so droplets fan out
    // rather than tracking the parent path.
    float3 vJit = (rand3(pid, frameSeed ^ 0xBEEFu) - 0.5f) * U.spawn.w;
    float3 fvel = vel + vJit;

    // Life: spread over [0.4, 1.0] × lifeMax so they don't all die in sync.
    float life = U.bounds.w * mix(0.4f, 1.0f, rand01(pid, frameSeed ^ 0xC0FFEEu));

    FoamParticle fp;
    fp.positionLife = float4(pos, life);
    fp.velocityAge  = float4(fvel, 0.0f);
    foam[slot] = fp;
}

// ── KERNEL 2 — Advect ───────────────────────────────────────────────────────
//
// One thread per foam slot. Skips dead slots (life ≤ 0). For live ones:
// integrate ballistically with gravity and exponential drag, decrement life,
// kill (park below floor) when life expires or the particle drops below the
// floor.
//
// "Park below floor" = move position to U.boundsMax.w (which the host sets to
// well below boundsMin.y). The renderer draws all capacity particles every
// frame; parked ones are off-screen.

kernel void foamAdvect(
    device FoamParticle*    foam     [[ buffer(0) ]],
    constant FoamUniforms&  U        [[ buffer(1) ]],
    uint                    fid      [[ thread_position_in_grid ]])
{
    if (fid >= U.counts.y) return;

    FoamParticle p = foam[fid];
    float life = p.positionLife.w;
    if (life <= 0.0f) {
        // Already dead. Make sure it stays parked even if some other write
        // path ever resurrects its position field.
        p.positionLife = float4(0.0f, U.boundsMax.w, 0.0f, 0.0f);
        foam[fid] = p;
        return;
    }

    float dt = U.dtTime.x;
    float drag = U.gravity.w;
    float3 g = U.gravity.xyz;

    float3 vel = p.velocityAge.xyz;
    float3 pos = p.positionLife.xyz;

    // Exponential drag — `vel *= exp(-drag * dt)` is the exact solution to
    // dv/dt = -drag·v, which keeps the system stable at any dt. (1 - drag*dt)
    // is only first-order accurate and goes negative once drag*dt > 1.)
    vel = vel * exp(-drag * dt) + g * dt;
    pos = pos + vel * dt;

    life -= dt;
    float age = p.velocityAge.w + dt;

    // Kill if expired, or if hit floor / left domain. Foam hitting the side
    // walls would normally bounce, but for the MVP we don't try to render
    // wall splashes — let them die quietly.
    bool dead = (life <= 0.0f)
             || (pos.y < U.bounds.y)
             || (pos.x < U.bounds.x) || (pos.x > U.boundsMax.x)
             || (pos.z < U.bounds.z) || (pos.z > U.boundsMax.z);

    if (dead) {
        p.positionLife = float4(0.0f, U.boundsMax.w, 0.0f, 0.0f);
        p.velocityAge  = float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        p.positionLife = float4(pos, life);
        p.velocityAge  = float4(vel, age);
    }
    foam[fid] = p;
}
