#include <metal_stdlib>
using namespace metal;

// ── SWE-DRIVEN FOAM ─────────────────────────────────────────────────────────
//
// Replaces the old WaterFlow point-particle "blue dust" with real secondary
// foam born from the shallow-water solver's state. Two kernels:
//
//   • sweFoamSpawn  — one thread per SWE cell (Narc × Nwidth). Computes the
//                     local water speed and free-surface slope; cells whose
//                     speed × slope crests a threshold are whitecap candidates.
//                     A hash-RNG gate emits a foam droplet from those cells
//                     into a ring buffer (atomic cursor). Spawn position is
//                     the cell's world-space surface point with a sub-cell
//                     jitter; initial velocity is the water's local velocity
//                     converted to world plus an upward splash kick scaled
//                     by crest strength.
//
//   • sweFoamAdvect — one thread per foam slot. Ballistic gravity + drag
//                     integration; life decrement; dead particles parked at
//                     `parkY` so the renderer keeps drawing the full capacity
//                     but the audience never sees them.
//
// All world-space conversions go through the SWE's existing path-sample
// buffers (pos / right / up) — no per-cell CPU work.

struct SWECell {
    float h;
    float hu;
    float hv;
    float pad;
};

struct SWEFoamParticle {
    float4 positionLife;   // xyz = world position, w = life remaining (s)
    float4 velocityS;      // xyz = world velocity (m/s), w = arc-length along path
                           //       (stored so advect can clamp the particle back
                           //        into the tube wedge each tick without an
                           //        O(pathSamples) nearest-point search).
};

struct SWEFoamUniforms {
    uint   Narc;
    uint   Nwidth;
    uint   pathSamples;
    uint   capacity;
    float  totalLength;
    float  dxS;
    float  dxW;
    float  floorRadius;
    float  angleSpan;
    float  baselineH;
    float  waterScale;
    float  dt;
    float  elapsed;
    float  spawnSpeedThreshold;
    float  spawnSlopeThreshold;
    float  spawnProb;
    float  lifeMin;
    float  lifeMax;
    float  parkY;
    float  drag;
    float  gravity;
    float  upKickMin;
    float  upKickMax;
    float  velJitter;
    float  tubeRadius;     // hard clamp on particle distance from path centerline
    uint   randSeed;
    uint   pad0;
    uint   pad1;
};

static inline float wfHash(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15);
    return float(x & 0x00ffffffu) / float(0x01000000);
}

// Looks up the path frame at a continuous sample index (sFrac ∈ [0,1)),
// returns (pos, right, up, tangent). All buffers are pathSamples-long
// SIMD4 polyline samples baked at scene init.
static inline void samplePathFrame(
    float sFrac, uint pathSamples,
    constant float4 *pathPos,
    constant float4 *pathRight,
    constant float4 *pathUp,
    thread float3 &pos,
    thread float3 &right,
    thread float3 &up,
    thread float3 &tangent
) {
    float fIdx = sFrac * float(pathSamples);
    uint  i0   = uint(floor(fIdx)) % pathSamples;
    uint  i1   = (i0 + 1u) % pathSamples;
    float frac = fIdx - floor(fIdx);
    pos     = mix(pathPos[i0].xyz,   pathPos[i1].xyz,   frac);
    right   = normalize(mix(pathRight[i0].xyz, pathRight[i1].xyz, frac));
    up      = normalize(mix(pathUp[i0].xyz,    pathUp[i1].xyz,    frac));
    tangent = normalize(cross(right, up));
}

kernel void sweFoamSpawn(
    device const SWECell        *state     [[buffer(0)]],
    device       SWEFoamParticle *foam     [[buffer(1)]],
    device       atomic_uint     *nextSlot [[buffer(2)]],
    constant float4              *pathPos  [[buffer(3)]],
    constant float4              *pathRight[[buffer(4)]],
    constant float4              *pathUp   [[buffer(5)]],
    constant SWEFoamUniforms     &u        [[buffer(6)]],
    uint2                         gid      [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= u.Narc || j >= u.Nwidth) return;

    uint cellID = i * u.Nwidth + j;
    SWECell c = state[cellID];
    float h = max(c.h, 1e-4f);
    float uv = c.hu / h;
    float vv = c.hv / h;
    float speed = length(float2(uv, vv));

    // Periodic in arc, clamped in width — matches the SWE solver's BCs.
    uint iL = (i == 0u) ? u.Narc - 1u : i - 1u;
    uint iR = (i + 1u) % u.Narc;
    uint jL = (j == 0u) ? 0u : j - 1u;
    uint jR = min(j + 1u, u.Nwidth - 1u);

    // Whitecap criterion = |curl(v)| + α·|v|² + β·|∇h|·|v|. Foam appears at
    // banking turns (curl), high-shear collision fronts (speed²), and where
    // fast water rolls down a steep gradient (slope·speed). Replaces the
    // earlier speed×slope-only proxy, which missed turn-induced whitecaps
    // entirely.
    float h_iL = max(state[iL * u.Nwidth + j].h, 1e-4f);
    float h_iR = max(state[iR * u.Nwidth + j].h, 1e-4f);
    float h_jL = max(state[i  * u.Nwidth + jL].h, 1e-4f);
    float h_jR = max(state[i  * u.Nwidth + jR].h, 1e-4f);
    float u_jL = state[i  * u.Nwidth + jL].hu / h_jL;
    float u_jR = state[i  * u.Nwidth + jR].hu / h_jR;
    float v_iL = state[iL * u.Nwidth + j].hv  / h_iL;
    float v_iR = state[iR * u.Nwidth + j].hv  / h_iR;
    float curl_z = 0.5f * (v_iR - v_iL) / u.dxS
                 - 0.5f * (u_jR - u_jL) / u.dxW;
    float dh_ds = 0.5f * (state[iR * u.Nwidth + j].h - state[iL * u.Nwidth + j].h) / u.dxS;
    float dh_dw = 0.5f * (state[i  * u.Nwidth + jR].h - state[i  * u.Nwidth + jL].h) / u.dxW;
    float slope = length(float2(dh_ds, dh_dw));

    if (speed < u.spawnSpeedThreshold) return;

    float speedExcess = max(0.0f, speed - u.spawnSpeedThreshold);
    // spawnSlopeThreshold is now repurposed as the "minimum potential energy"
    // gate, mapped via the same uniform name to avoid churning the struct.
    float potential = abs(curl_z)
                    + 0.5f * speedExcess * speedExcess
                    + 0.6f * slope * speed;
    if (potential < u.spawnSlopeThreshold) return;

    float crest = potential;
    float prob = min(1.0f, u.spawnProb * (1.0f + crest * 4.0f));

    // Per-cell-per-frame hash-RNG gate. randSeed varies across frames so we
    // don't lock into a static pattern of spawning cells.
    uint seed = cellID * 2654435761u + u.randSeed * 374761393u;
    if (wfHash(seed) > prob) return;

    float r2 = wfHash(seed + 17u);
    float r3 = wfHash(seed + 31u);
    float r4 = wfHash(seed + 53u);
    float r5 = wfHash(seed + 71u);
    float r6 = wfHash(seed + 97u);
    float r7 = wfHash(seed + 113u);

    // Continuous (sFrac, wFrac) sample point inside the cell.
    float sFrac = (float(i) + r2) / float(u.Narc);
    float wFrac = ((float(j) + r3) / float(u.Nwidth)) - 0.5f;

    float3 pos, right, up, tangent;
    samplePathFrame(sFrac, u.pathSamples, pathPos, pathRight, pathUp,
                    pos, right, up, tangent);
    float angle = M_PI_F + wFrac * u.angleSpan;
    float3 outward = up * cos(angle) + right * sin(angle);

    // Surface-of-water height matches the surfaceFromSWE kernel's formula.
    float dh = clamp(h - u.baselineH, -u.baselineH, 0.5f * u.baselineH);
    float displayDepth = clamp(u.baselineH + dh * u.waterScale, 0.0f, u.floorRadius * 0.9f);
    float effR = max(u.floorRadius - displayDepth, 0.05f);
    float3 worldPos = pos + outward * effR;

    // World velocity = water's local velocity in cradle frame → world basis.
    float3 worldVel = tangent * uv + right * vv;
    // Splash kick along the outward (out-of-water) direction, proportional
    // to crest strength so calm flow puffs gently and whitecaps spray hard.
    // Direction is -outward (out of the half-pipe interior toward the air).
    float kick = mix(u.upKickMin, u.upKickMax, saturate(crest * 2.0f));
    worldVel += (-outward) * kick;
    // Symmetric velocity jitter so the spray isn't a perfect fan.
    worldVel += (float3(r4, r5, r6) - 0.5f) * 2.0f * u.velJitter;

    float life = mix(u.lifeMin, u.lifeMax, r7);

    // Write the particle's arc-length into velocityS.w so advect can look
    // up the local path frame without a global nearest-point search.
    float s_world = sFrac * u.totalLength;

    uint slot = atomic_fetch_add_explicit(nextSlot, 1u, memory_order_relaxed) % u.capacity;
    foam[slot].positionLife = float4(worldPos, life);
    foam[slot].velocityS    = float4(worldVel, s_world);
}

kernel void sweFoamAdvect(
    device       SWEFoamParticle *foam      [[buffer(0)]],
    constant SWEFoamUniforms     &u         [[buffer(1)]],
    constant float4              *pathPos   [[buffer(2)]],
    constant float4              *pathRight [[buffer(3)]],
    constant float4              *pathUp    [[buffer(4)]],
    uint                          id        [[thread_position_in_grid]]
) {
    if (id >= u.capacity) return;
    SWEFoamParticle p = foam[id];
    float life = p.positionLife.w - u.dt;
    if (life <= 0.0f) {
        // Park well off-screen; renderer keeps drawing every slot.
        p.positionLife = float4(0.0f, u.parkY, 0.0f, 0.0f);
        p.velocityS    = float4(0.0f);
        foam[id] = p;
        return;
    }
    float3 vel = p.velocityS.xyz;
    vel.y -= u.gravity * u.dt;
    vel *= exp(-u.drag * u.dt);
    float3 pos = p.positionLife.xyz + vel * u.dt;

    // ── Tube-interior clamp ───────────────────────────────────────────
    // Update arc-length s by the tangent component of velocity (cheap
    // local update — no nearest-point search needed). Then re-sample
    // the path frame at the new s, compute the particle's offset from
    // the centerline in the (right, up) plane, and clamp it inside the
    // tube radius. Without this, gravity drags spawned foam straight
    // through the slide floor and they "leak" beneath the tube
    // silhouette — exactly the visual bug we're fixing.
    float s = p.velocityS.w;

    float sFrac0 = s / u.totalLength;
    sFrac0 = sFrac0 - floor(sFrac0);
    float fIdx0  = sFrac0 * float(u.pathSamples);
    uint  i0a    = uint(floor(fIdx0)) % u.pathSamples;
    uint  i1a    = (i0a + 1u) % u.pathSamples;
    float fra    = fIdx0 - floor(fIdx0);
    float3 tan0  = normalize(cross(
        normalize(mix(pathRight[i0a].xyz, pathRight[i1a].xyz, fra)),
        normalize(mix(pathUp[i0a].xyz,    pathUp[i1a].xyz,    fra))
    ));
    float vTanComp = dot(vel, tan0);
    s = s + vTanComp * u.dt;
    s = s - u.totalLength * floor(s / u.totalLength);

    float sFrac = s / u.totalLength;
    float fIdx  = sFrac * float(u.pathSamples);
    uint  pi0   = uint(floor(fIdx)) % u.pathSamples;
    uint  pi1   = (pi0 + 1u) % u.pathSamples;
    float prf   = fIdx - floor(fIdx);
    float3 cPos   = mix(pathPos[pi0].xyz,   pathPos[pi1].xyz,   prf);
    float3 cRight = normalize(mix(pathRight[pi0].xyz, pathRight[pi1].xyz, prf));
    float3 cUp    = normalize(mix(pathUp[pi0].xyz,    pathUp[pi1].xyz,    prf));

    float3 offset = pos - cPos;
    float  lateralOff = dot(offset, cRight);
    float  upOff      = dot(offset, cUp);
    float  r2 = lateralOff * lateralOff + upOff * upOff;
    float  maxR = u.tubeRadius * 0.95f;
    if (r2 > maxR * maxR) {
        float r = sqrt(r2);
        float scale = maxR / r;
        float lat2 = lateralOff * scale;
        float up2  = upOff * scale;
        pos = cPos + cRight * lat2 + cUp * up2;
        // Soft inelastic reflection: kill the outward-pointing component
        // of velocity so the particle slides along the wall instead of
        // tunnelling through next tick.
        float3 outDir = (cRight * lateralOff + cUp * upOff) / max(r, 1e-4f);
        float vOut = dot(vel, outDir);
        if (vOut > 0.0f) {
            vel -= outDir * vOut * 1.35f;
        }
    }

    p.positionLife = float4(pos, life);
    p.velocityS    = float4(vel, s);
    foam[id] = p;
}
