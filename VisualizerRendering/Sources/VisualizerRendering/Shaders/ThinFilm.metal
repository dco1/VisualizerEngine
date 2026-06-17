#include <metal_stdlib>
using namespace metal;

// ── THIN-FILM (LUBRICATION) SURFACE FLOW ─────────────────────────────────────
//
// Animates a thin viscous coating (buffalo sauce) on a fixed object surface,
// per the lubrication-theory / thin-film literature:
//   • Vantzos, Azencot, Wardetzky, Rumpf, Ben-Chen, "Functional Thin Films on
//     Surfaces" (SCA 2015 / IEEE TVCG 2017)
//   • Vantzos & Raz, "Real-Time Viscous Thin Films" (SIGGRAPH Asia 2018)
//   • Herson et al., "Dripping Thin Films for Real-time Digital Painting" (CGF 2024)
//
// The film is a THICKNESS field h(node) living on the object's surface, NOT a 3D
// volume fluid. The surface is the object's own structured grid: W columns
// (around, wrapping) × H rows (along), node n = row*W + col. Per node we store
// the fixed world-space base position + outward normal; the film flows over it.
//
// Physics per step (explicit, finite-difference on the grid using true edge
// lengths from the base positions as the metric):
//   ∂h/∂t = -div( M(h) · ∇Φ ) + D·Δh
//   M(h)  = max(0, h - hMin)³ / (3μ)      lubrication mobility — only the part of
//                                         the film ABOVE the adhesion floor hMin
//                                         is mobile, so a residual layer always
//                                         CLINGS (stick). This is the disjoining-
//                                         pressure / adhesion term.
//   Φ     = gravity · y                    gravity potential — sauce flows downhill
//   D·Δh                                   surface-tension smoothing (also seeds
//                                         the viscous-fingering into rivulets)
// Mass is (near-)conserved: edge fluxes are antisymmetric between neighbours.

struct ThinFilmUniforms {
    uint  W;             // columns (around, wraps)
    uint  H;             // rows (along)
    uint  N;             // total vertex count (body grid W*H + end-cap centres)
    uint  bodyN;         // W*H — the film grid; vertices >= bodyN are cap centres
    float dt;            // substep dt
    float gravity;       // m/s² (positive)
    float mu;            // lubrication viscosity coefficient
    float hMin;          // adhesion floor — residual clinging thickness (m)
    float surfTension;   // D, the Δh smoothing coefficient
    float displaceScale; // outward displacement = h * displaceScale
    float wetFloor;      // hard wetting floor — the wing stays at least this wet
                         // everywhere the coat exists (full coverage; 0 at Dry)
    float pad2;
};

// Neighbour indices on the grid: columns wrap, rows clamp.
static inline void neighbours(uint n, constant ThinFilmUniforms &u,
                              thread uint &nL, thread uint &nR,
                              thread uint &nU, thread uint &nD) {
    uint W = u.W, H = u.H;
    uint ri = n / W, ai = n % W;
    uint aL = (ai + W - 1) % W;
    uint aR = (ai + 1) % W;
    uint rU = ri == 0 ? 0 : ri - 1;
    uint rD = ri + 1 >= H ? H - 1 : ri + 1;
    nL = ri * W + aL;
    nR = ri * W + aR;
    nU = rU * W + ai;
    nD = rD * W + ai;
}

// One explicit thin-film step. Reads h, writes hNew.
kernel void thinFilmStep(device const float4   *basePos  [[buffer(0)]],
                         device const float    *h        [[buffer(1)]],
                         device       float    *hNew     [[buffer(2)]],
                         constant ThinFilmUniforms &u    [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= u.bodyN) return;          // film evolves only on the body grid
    float hn = h[gid];
    float3 pn = basePos[gid].xyz;

    uint nb[4];
    neighbours(gid, u, nb[0], nb[1], nb[2], nb[3]);

    float dHdt = 0.0;
    for (uint k = 0; k < 4; ++k) {
        uint m = nb[k];
        if (m == gid) continue;            // clamped edge (top/bottom row) — no flow
        float hm = h[m];
        float3 pm = basePos[m].xyz;
        float len2 = max(1e-8, distance_squared(pn, pm));
        // Mobile thickness at the edge — only the part ABOVE the adhesion floor
        // flows, so a residual layer always clings (disjoining-pressure proxy).
        float hEdge = max(0.0, 0.5 * (hn + hm) - u.hMin);
        float M = hEdge * hEdge * hEdge / (3.0 * u.mu);
        // Gravity transport along the surface: sauce moves toward the LOWER-y
        // neighbour. (pm.y - pn.y) < 0 when the neighbour is lower → h[gid] loses
        // to it; the neighbour's own pass gains symmetrically (≈ conservative).
        dHdt += M * u.gravity * (pm.y - pn.y) / len2;
        // Surface-tension smoothing (also seeds rivulet fingering).
        dHdt += u.surfTension * (hm - hn) / len2;
    }
    float hn1 = hn + u.dt * dHdt;
    // Hard wetting floor: where any coat exists, the wing stays at least this
    // wet (full coverage, glossy sheen). Excess above it is what flows + drips.
    hNew[gid] = max(hn1, u.wetFloor);
}

// Write the displaced coat surface (base position pushed out along the base
// normal by the film thickness) + a recomputed shading normal. Output is
// `packed_float3` (stride 12) so the SAME buffers feed both a buffer-backed
// SCNGeometry AND Illuminatorama's `illumi_repack_pos_norm` (which reads
// `packed_float3`), enabling a native-Illuminatorama coat via registerGPUMesh.
kernel void thinFilmWriteCoat(device       packed_float3 *outPos [[buffer(0)]],
                              device       packed_float3 *outNrm [[buffer(1)]],
                              device const float4 *basePos [[buffer(2)]],
                              device const float4 *baseNrm [[buffer(3)]],
                              device const float  *h       [[buffer(4)]],
                              constant ThinFilmUniforms &u [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= u.N) return;

    // End-cap centre vertices (gid >= bodyN) have no grid neighbours — just
    // displace along the base normal and keep the base normal for shading. This
    // closes the coat tube so the camera never sees into an open end.
    if (gid >= u.bodyN) {
        outPos[gid] = packed_float3(basePos[gid].xyz + baseNrm[gid].xyz * (h[gid] * u.displaceScale));
        outNrm[gid] = packed_float3(baseNrm[gid].xyz);
        return;
    }

    uint nb[4];
    neighbours(gid, u, nb[0], nb[1], nb[2], nb[3]);

    // Inline-displaced positions (recompute neighbours to avoid a read race with
    // the concurrent writes to outPos).
    float3 dp[5];
    float3 bn = baseNrm[gid].xyz;
    dp[4] = basePos[gid].xyz + bn * (h[gid] * u.displaceScale);
    for (uint k = 0; k < 4; ++k) {
        uint m = nb[k];
        dp[k] = basePos[m].xyz + baseNrm[m].xyz * (h[m] * u.displaceScale);
    }
    // nb order: L, R, U, D
    float3 t1 = dp[1] - dp[0];   // R - L  (around)
    float3 t2 = dp[3] - dp[2];   // D - U  (along)
    float3 nrm = cross(t2, t1);
    float nl = length(nrm);
    nrm = nl > 1e-6 ? nrm / nl : bn;
    if (dot(nrm, bn) < 0.0) nrm = -nrm;    // keep outward

    outPos[gid] = packed_float3(dp[4]);
    outNrm[gid] = packed_float3(nrm);
}

// ── DRIPS ────────────────────────────────────────────────────────────────────
//
// Where the coat pools on a DOWNWARD-facing part of the wing past the adhesion's
// holding capacity, it pinches off into a falling drip (Rayleigh-Taylor /
// Plateau instability, the "some drips" half). Each drip is a ballistic particle
// that falls under gravity and pools on the plate. A small fraction of the
// excess thickness is moved into the drip and removed from the film.

struct DripParticle {
    float4 posLife;   // xyz = world position, w = life (>0 alive)
    float4 velSize;   // xyz = velocity, w = render size
};

struct DripUniforms {
    uint  bodyN;
    uint  dripCap;
    uint  frame;       // per-frame seed for stochastic shedding
    uint  pad0;
    float dt;
    float gravity;
    float shedCap;     // hold capacity — excess above this can pinch into a drip
    float plateY;      // floor the drips pool on
    float shedRate;    // 0..1 per-node shed probability throttle
    float dripSize;
    float displaceScale;
    float lifeDecay;
    float camX, camY, camZ;   // camera position (for billboard facing)
    float baseLen;            // teardrop length at rest (pendant drop)
    float stretch;            // extra length per unit fall speed (strand)
    float width;             // teardrop half-width
    float pad1, pad2;
};

static inline uint wangHash(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u; x = x ^ (x >> 4); x *= 0x27d4eb2du; x = x ^ (x >> 15);
    return x;
}

// Shed pass — runs AFTER the film steps, on the current h buffer. Reads/writes h.
kernel void thinFilmShed(device const float4 *basePos [[buffer(0)]],
                         device const float4 *baseNrm [[buffer(1)]],
                         device       float  *h       [[buffer(2)]],
                         device DripParticle *drips    [[buffer(3)]],
                         device atomic_uint  *head     [[buffer(4)]],
                         constant DripUniforms &u      [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= u.bodyN) return;
    if (baseNrm[gid].y > 0.15) return;        // sides + undersides drip (downward-ish)
    float hv = h[gid];
    if (hv < u.shedCap) return;
    uint rnd = wangHash(gid ^ (u.frame * 2654435761u));
    if ((rnd & 1023u) > uint(saturate(u.shedRate) * 1023.0)) return;

    uint slot = atomic_fetch_add_explicit(head, 1u, memory_order_relaxed) % max(1u, u.dripCap);
    float3 wp = basePos[gid].xyz + baseNrm[gid].xyz * (hv * u.displaceScale);
    float jitter = (float(wangHash(rnd) & 255u) / 255.0 - 0.5) * 0.01;
    drips[slot].posLife = float4(wp + float3(jitter, 0.0, jitter), 1.0);
    drips[slot].velSize = float4(0.0, -0.15, 0.0, u.dripSize);
    // remove the shed mass (leave the holding layer behind)
    h[gid] = max(u.shedCap * 0.6, hv - (hv - u.shedCap) * 0.6);
}

// Integrate the drips: gravity fall, pool on the plate, fade out.
kernel void thinFilmDripStep(device DripParticle *drips [[buffer(0)]],
                             constant DripUniforms &u    [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= u.dripCap) return;
    float life = drips[gid].posLife.w;
    if (life <= 0.0) { drips[gid].posLife.y = -1000.0; return; }   // hidden
    float3 pos = drips[gid].posLife.xyz;
    float3 vel = drips[gid].velSize.xyz;
    if (pos.y > u.plateY) {
        vel.y -= u.gravity * u.dt;
        pos += vel * u.dt;
        life -= u.dt * u.lifeDecay;
    } else {
        pos.y = u.plateY;                 // pooled on the plate
        vel = float3(0.0);
        life -= u.dt * u.lifeDecay * 2.0; // fade the puddle faster
    }
    drips[gid].posLife = float4(pos, max(0.0, life));
    drips[gid].velSize = float4(vel, drips[gid].velSize.w);
}

// Build a camera-facing TEARDROP billboard quad (4 verts) per drip, stretched
// along its fall direction — a hanging pendant drop (low speed) reads short and
// round; a falling one elongates into a strand. The teardrop SHAPE comes from
// the alpha texture (pointy trailing top, round leading bottom); this kernel
// only places the quad. Dead drips collapse offscreen.
kernel void thinFilmDripBillboard(device const DripParticle *drips [[buffer(0)]],
                                  device       float4       *verts  [[buffer(1)]],
                                  constant DripUniforms      &u      [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= u.dripCap) return;
    uint base = gid * 4;
    float life = drips[gid].posLife.w;
    if (life <= 0.0) {
        float4 hidden = float4(0.0, -1000.0, 0.0, 0.0);
        verts[base + 0] = hidden; verts[base + 1] = hidden;
        verts[base + 2] = hidden; verts[base + 3] = hidden;
        return;
    }
    float3 P = drips[gid].posLife.xyz;
    float3 vel = drips[gid].velSize.xyz;
    float size = drips[gid].velSize.w;
    float speed = length(vel);
    float3 fall = speed > 0.05 ? (vel / speed) : float3(0.0, -1.0, 0.0);
    float3 cam = float3(u.camX, u.camY, u.camZ);
    float3 toCam = normalize(cam - P);
    float3 right = cross(fall, toCam);
    float rl = length(right);
    right = rl > 1e-4 ? right / rl : normalize(cross(fall, float3(1, 0, 0)));

    float halfLen = min(0.035, u.baseLen + speed * u.stretch);  // clamp strand length
    float halfW = u.width * max(0.4, size);
    float3 tip = P - fall * (halfLen * 0.45);   // trailing pointy end
    float3 bot = P + fall * (halfLen * 0.55);   // leading round end

    verts[base + 0] = float4(tip - right * halfW, 0.0);  // uv (0,0)
    verts[base + 1] = float4(tip + right * halfW, 0.0);  // uv (1,0)
    verts[base + 2] = float4(bot - right * halfW, 0.0);  // uv (0,1)
    verts[base + 3] = float4(bot + right * halfW, 0.0);  // uv (1,1)
}

