// ── FIREWORKS ────────────────────────────────────────────────────────────────
//
// GPU compute kernels for the Fireworks+ scene's spark simulation. One big
// MTLBuffer of `FWParticle` (80 bytes, all float4) holds every live spark in
// the scene; the host allocates contiguous slot ranges for each burst and
// dispatches `fwSpawnBurst` once per burst, then runs `fwIntegrate` every
// frame to advance position / velocity / age / brightness.
//
// SceneKit reads `positionAge.xyz` (vertex semantic) and `displayColor`
// (color semantic) directly out of the same MTLBuffer — zero copies, no CPU
// snapshot. The buffer layout MUST match `FireworkParticle` in the Swift
// solver exactly. Per the project's "no bare float3 in shared structs" rule
// (see ALIGNMENT RULE in PBDSolver.swift), every field is float4.
//
// Pattern handling: `fwSpawnBurst` takes a `kind` enum in the uniforms and
// produces an initial velocity for each particle accordingly. The shape of
// the burst (chrysanthemum sphere vs willow droop vs ring vs heart) is
// expressed as initial velocity + per-particle drag/gravity multipliers —
// the integrator doesn't know about patterns.

#include <metal_stdlib>
using namespace metal;

// ── Shared structs ──────────────────────────────────────────────────────────

struct FWParticle {
    float4 positionAge;   // xyz = world position (m), w = age (s)
    float4 velocityLife;  // xyz = velocity (m/s),     w = lifespan (s)
    float4 displayColor;  // rgb = emissive color × current brightness, a = 1
    float4 baseDrag;      // rgb = base color,         w = drag (/s exponential)
    float4 paramsExtra;   // x = turbulence amp,
                          // y = gravity multiplier,
                          // z = visual size hint (unused on points but ported
                          //     to per-vertex aux for future shader use),
                          // w = pad
    float4 colorTransition;  // rgb = SECONDARY color the star transitions to,
                          // w   = transitionAt ∈ [0, 1] (age fraction).
                          //       w == 0 means "no transition" — keep
                          //       baseDrag.rgb for the entire life.
};

struct FWIntegrateUniforms {
    float4 dtTimeCount;   // x = dt, y = wall time, z = count (uint bits), w = pad
    float4 gravity;       // xyz = base gravity (m/s²), w = pad
    float4 wind;          // xyz = base wind vector (m/s), w = pad
};

struct FWSpawnUniforms {
    float4 centerKind;          // xyz = burst center (m),
                                // w = kind (uint bits): 0=sphere, 1=ring, 2=heart
    float4 baseColor;           // rgb = base color, a = colorJitter (0..1)
    float4 speedSizeLife;       // x = base speed, y = speed jitter,
                                // z = size, w = base life (s)
    float4 dragTurbGravLife;    // x = drag, y = turbulence,
                                // z = gravity multiplier, w = life jitter (s)
    float4 firstCountSeedTime;  // x = firstIndex (uint bits),
                                // y = count (uint bits),
                                // z = seed (uint bits), w = time
    float4 ringAxis;            // xyz = orientation axis for ring/heart (unit),
                                // w = ring radial thickness (0..1)
    float4 secondaryColor;      // rgb = colour the star transitions INTO,
                                // a = per-spark probability of transitioning
                                //     (0 = no transition for this burst)
};

// Per-vertex streak vertex written by fwBuildStreaks and read by SceneKit as
// the geometry's position + color sources. 32 bytes — SCN reads the first 12
// as float3 position and bytes 16..<28 as float3 color (alpha lives in w but
// the additive blend mode ignores it, so the color source is float3).
struct FWStreakVertex {
    float4 position;   // xyz = world-space corner (m), w = 1
    float4 color;      // rgb = displayColor × falloff, a = falloff
};

struct FWStreakUniforms {
    float4 cameraCount;    // xyz = camera world position, w = count (uint bits)
    float4 streakParams;   // x = streakDuration (s) — how long a streak the
                           //                          spark drags behind itself
                           // y = streakWidth (m) — billboard width in world units
                           // z = minLength (m)  — floor so still sparks render
                           //                      as squares, not zero-length lines
                           // w = pad
};

// ── Spark emit (shell-rise trail) ───────────────────────────────────────────
//
// Per-emit record consumed by `fwSpawnSparks`. The host fills a small SimBuffer
// with one entry per spark to spawn this tick (one per active climbing shell,
// or a few per shell at higher trail densities), then dispatches a kernel
// pass that writes the resulting particles directly into the shared particle
// buffer. No CPU integration, no SCNParticleSystem.

struct FWSparkEmit {
    float4 positionSeed;   // xyz = world position the spark starts at,
                           // w   = per-emit seed bits (uint)
};

struct FWSparkUniforms {
    float4 baseColorJitter;        // rgb = base color (pre-multiplied to land
                                   //       inside HDR after the brightness
                                   //       profile in fwIntegrate),
                                   // a   = color jitter amplitude (0..1)
    float4 speedLifeJitter;        // x = base speed, y = speed jitter,
                                   // z = base life, w = life jitter
    float4 dragTurbGrav;           // x = drag, y = turbulence, z = gravityMul,
                                   // w = visual size hint
    float4 firstCountSeedTime;     // x = firstIndex (uint bits),
                                   // y = count (uint bits),
                                   // z = global seed (uint bits),
                                   // w = wall time
};

// ── PCG random ──────────────────────────────────────────────────────────────
//
// Cheap deterministic per-particle RNG. Seeded by `seed ^ (slot * golden)`
// so consecutive particles in the same burst don't read sequential states.

inline uint pcg_step(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float frand(thread uint& state) {
    return float(pcg_step(state)) * (1.0f / 4294967296.0f);
}

// ── Direction builders ──────────────────────────────────────────────────────

// Uniform-area random unit vector on the sphere.
inline float3 dir_sphere(thread uint& state) {
    float z = frand(state) * 2.0f - 1.0f;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = frand(state) * 6.2831853f;
    return float3(r * cos(phi), r * sin(phi), z);
}

// Build an orthonormal basis (e1, e2) perpendicular to `axis`. Used so the
// ring + heart patterns can be oriented arbitrarily — default axis is +Y so
// the ring sits in the XZ plane, but the host can rotate it to face the camera.
inline void basis_from_axis(float3 axis, thread float3& e1, thread float3& e2) {
    float3 a = normalize(axis);
    float3 ref = abs(a.y) < 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    e1 = normalize(cross(a, ref));
    e2 = cross(a, e1);
}

// ── Integrate kernel ────────────────────────────────────────────────────────
//
// One thread per particle slot. Advances position by velocity, velocity by
// gravity+drag+turbulence, and age by dt. Brightness follows a "hot start /
// long ember tail" profile keyed on normalized age. Dead particles (age ≥ life)
// are parked at y = −10000 with displayColor zeroed so SCN draws them outside
// the visible volume.

kernel void fwIntegrate(
    device       FWParticle*          particles [[ buffer(0) ]],
    constant     FWIntegrateUniforms& u         [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint count = as_type<uint>(u.dtTimeCount.z);
    if (id >= count) return;

    FWParticle p = particles[id];
    float age  = p.positionAge.w;
    float life = p.velocityLife.w;

    // Dead-slot fast path: park offscreen, kill output color, leave for the
    // host to recycle the slot at the next burst.
    if (life <= 0.0f || age >= life) {
        p.positionAge   = float4(0.0f, -10000.0f, 0.0f, life + 1.0f);
        p.displayColor  = float4(0.0f, 0.0f, 0.0f, 0.0f);
        particles[id]   = p;
        return;
    }

    float dt = u.dtTimeCount.x;
    float t  = u.dtTimeCount.y;

    float3 vel  = p.velocityLife.xyz;
    float3 pos  = p.positionAge.xyz;
    float  drag = p.baseDrag.w;
    float  turb = p.paramsExtra.x;
    float  gMul = p.paramsExtra.y;

    // Gravity: per-particle multiplier so willow can pull harder than peony.
    vel += u.gravity.xyz * gMul * dt;
    // Constant wind drift.
    vel += u.wind.xyz * dt;
    // Exponential drag.
    vel *= exp(-drag * dt);
    // Turbulence: cheap sin-noise jitter so willow trails wander rather than
    // falling in clean parabolas. Phase keyed on position so neighbours
    // diverge.
    if (turb > 0.0f) {
        float3 n = float3(
            sin(pos.x * 0.31f + t * 0.73f + 1.7f),
            cos(pos.y * 0.27f + t * 0.61f + 2.3f),
            sin(pos.z * 0.29f + t * 0.83f + 0.9f)
        );
        vel += n * turb * dt;
    }

    pos += vel * dt;
    age += dt;

    // Brightness profile: hot for the first 8% of life (the spark is fresh
    // from the burst), then a long fade to zero. The curve is biased so the
    // ember phase lingers and the very end drops quickly. Both peak and
    // plateau sit well above 1.0 so HDR + bloom pick the sparks up as glowing
    // arcs rather than dim points.
    //
    // Hot-start peak BACKED OFF from 5.0× to 3.2× — at the higher value
    // the dense cluster of newly-spawned sparks at burst time formed a
    // single overpowering white blob that drowned out the individual
    // colored streaks erupting from it. 3.2× still pops past the bloom
    // threshold (1.0 in the compositor) so the burst registers as a
    // bright event without washing out.
    float u_age = saturate(age / life);
    float bright;
    if (u_age < 0.08f) {
        bright = mix(3.2f, 2.0f, u_age / 0.08f);
    } else {
        float k = saturate((u_age - 0.08f) / 0.92f);
        bright = 2.0f * (1.0f - pow(k, 1.8f));
    }

    // Per-particle TWINKLE — magnesium/aluminum sparks rapidly modulate
    // brightness as they oxidize unevenly (~5..15 Hz flicker per particle).
    // Without this every spark fades on a smooth curve and reads as a
    // CG particle system, not a chemical reaction. Each spark gets its own
    // phase + rate so they don't twinkle in lockstep. Floor at 0.55 so the
    // spark never disappears between twinkles (lossy on perceptual brightness
    // but keeps the particle visible).
    //
    // Late-life CRACKLE: as the chemistry destabilises near end of life, the
    // twinkle frequency ramps up to ~2.5× — that fast flicker is what reads
    // as the audible "crackle" of magnesium-based stars even in silence.
    uint twinkleSeed = id * 2654435761u;
    float twinkleHash = float(twinkleSeed >> 8u) * (1.0f / 16777216.0f);  // 0..1
    float baseTwinkleRate = mix(7.0f, 17.0f, twinkleHash);                // Hz × 2π
    float lateLife = saturate((u_age - 0.70f) / 0.30f);                   // 0..1 in last 30%
    float twinkleRate = baseTwinkleRate * (1.0f + lateLife * 1.6f);       // 1× → 2.6×
    float twinklePhase = twinkleHash * 6.2831853f;
    float twinkle = 0.78f + 0.22f * sin(t * twinkleRate + twinklePhase);
    bright *= twinkle;

    // ── GLITTERING (metal-flake flash) ──────────────────────────────────
    // Real titanium/aluminum-flake compositions emit very bright sub-frame
    // flashes at random times — the unmistakable "sparkler crackle" close-up.
    // Per-particle Poisson roll each frame: small probability of a 4-6×
    // multiplier for one frame. Probability ramps up modestly with age so
    // the post-burst ember tail crackles more than the initial burst flash
    // (which is already bright). Hash uses (id, frame index) so the flashes
    // dance frame-to-frame rather than picking the same particles each tick.
    uint frameBucket = uint(t * 60.0f);
    uint flashSeed = id * 1664525u + frameBucket * 1013904223u;
    flashSeed = (flashSeed >> ((flashSeed >> 28u) + 4u)) ^ flashSeed;
    flashSeed = flashSeed * 277803737u;
    float flashRoll = float((flashSeed >> 22u) ^ flashSeed) * (1.0f / 4294967296.0f);
    float flashProb = 0.018f + u_age * u_age * 0.060f;                    // 1.8% → 7.8%
    if (flashRoll < flashProb) {
        bright *= 4.0f + flashRoll * 80.0f;                               // 4..~5.5×
    }

    // ── DEATH FLASH ──────────────────────────────────────────────────────
    // ~30% of sparks (deterministic per-id) get a single bright pulse 90..97%
    // through their life — the "PFF" pop of a star just before extinguishing.
    // Pulse is a smooth tent so it's not a hard frame-step.
    uint deathHash = (id * 0x9E3779B9u) >> 24u;                           // 0..255
    if (deathHash < 77u && u_age > 0.90f && u_age < 0.98f) {              // ~30% of sparks
        float pulseCenter = 0.94f;
        float pulse = saturate(1.0f - abs(u_age - pulseCenter) * 30.0f);   // tent, width ~0.066
        bright *= 1.0f + pulse * 2.5f;                                    // 1× → 3.5×
    }

    // ── Color-changing STARS ────────────────────────────────────────────
    // Real fireworks "transition stars" have a layered chemistry: the outer
    // layer burns first (one colour), then the inner core ignites in a
    // second colour. We model this by blending baseDrag.rgb → colorTransition
    // .rgb over a smooth 10%-of-life window centred on transitionAt.
    // transitionAt == 0 is the sentinel "this star never transitions" (the
    // typical case for trail sparks / non-transitioning bursts).
    float transitionAt = p.colorTransition.w;
    float3 baseRGB = p.baseDrag.rgb;
    if (transitionAt > 0.0f) {
        float halfWindow = 0.05f;
        float tBlend = saturate((u_age - (transitionAt - halfWindow))
                                / (2.0f * halfWindow));
        // smoothstep for a softer chemistry crossover than linear.
        tBlend = tBlend * tBlend * (3.0f - 2.0f * tBlend);
        baseRGB = mix(baseRGB, p.colorTransition.rgb, tBlend);
    }

    // White-hot CORE — at peak brightness a real spark is so bright the eye
    // (or camera sensor) saturates locally, washing the colour toward white.
    // Blend the (post-transition) colour toward (1,1,1) in proportion to
    // brightness above the threshold so the hot moment desaturates and the
    // ember phase returns to the saturated tint.
    //
    // Threshold RAISED from (1.5..5.0) to (3.5..8.0) — at the lower
    // threshold most of the burst's hot-start cluster was reading as a
    // white blob with the chemistry colour only visible in the cooling
    // tail. The higher threshold means most sparks keep their red /
    // green / blue identity even at peak brightness; only the brightest
    // glitter / flash particles wash to white. Reads as "vivid coloured
    // burst" instead of "white explosion fading to colour."
    float whiteAmount = saturate((bright - 3.5f) / 4.5f);  // 0 below 3.5×, 1 above 8×
    float3 hotRGB = mix(baseRGB, float3(1.0f, 0.97f, 0.92f), whiteAmount);

    // ── EMBER BLACKBODY COOLING ────────────────────────────────────────
    // Real spark embers cool from white-hot → orange → deep red as the
    // metal/chemistry burns out. Approximate the blackbody curve in
    // three steps keyed on normalised age. Trail sparks (no chemistry
    // tint) cool through the full ramp; chemistry stars (red, green,
    // etc.) get partial cooling — they keep their tint but pick up a
    // warm cast at end of life like real strontium/copper stars do.
    //
    // Mix is gated by `coolBlend` — chemistry stars (highly saturated
    // baseRGB) blend less to keep their colour; trail sparks (near-
    // white baseRGB) blend more.
    float3 emberHot  = float3(1.00f, 0.95f, 0.78f);  // ~5000K white-yellow
    float3 emberMid  = float3(1.00f, 0.55f, 0.20f);  // ~2800K orange
    float3 emberCold = float3(0.78f, 0.18f, 0.04f);  // ~1500K deep red
    float ageRamp = smoothstep(0.30f, 0.90f, u_age);
    float3 emberRGB;
    if (ageRamp < 0.5f) {
        emberRGB = mix(emberHot, emberMid, ageRamp * 2.0f);
    } else {
        emberRGB = mix(emberMid, emberCold, (ageRamp - 0.5f) * 2.0f);
    }
    // Saturation of baseRGB: max-min over channels. Trail sparks are
    // near-white (low saturation) → cool fully; chemistry stars (high
    // saturation) → cool only ~25 %.
    float baseMax = max(baseRGB.r, max(baseRGB.g, baseRGB.b));
    float baseMin = min(baseRGB.r, min(baseRGB.g, baseRGB.b));
    float baseSat = (baseMax > 1e-4f) ? (baseMax - baseMin) / baseMax : 0.0f;
    float coolBlend = ageRamp * mix(0.85f, 0.25f, baseSat);
    hotRGB = mix(hotRGB, emberRGB, coolBlend);

    p.positionAge  = float4(pos, age);
    p.velocityLife = float4(vel, life);
    p.displayColor = float4(hotRGB * bright, 1.0f);
    particles[id]  = p;
}

// ── Spawn-burst kernel ──────────────────────────────────────────────────────
//
// One thread per particle being spawned. The host pre-allocates a contiguous
// slot range [firstIndex, firstIndex+count) and dispatches `count` threads.
// Each thread seeds its RNG from `seed XOR (localId * golden)`, builds a
// direction according to the burst kind, picks a speed and life, and writes
// the slot atomically (each slot is owned by one thread — no contention).

kernel void fwSpawnBurst(
    device       FWParticle*       particles [[ buffer(0) ]],
    constant     FWSpawnUniforms&  u         [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint count = as_type<uint>(u.firstCountSeedTime.y);
    if (id >= count) return;

    uint firstIndex = as_type<uint>(u.firstCountSeedTime.x);
    uint seed       = as_type<uint>(u.firstCountSeedTime.z);
    uint kind       = as_type<uint>(u.centerKind.w);
    uint slot       = firstIndex + id;

    uint rng = seed ^ (id * 2654435761u);

    float3 dir;
    if (kind == 1u) {
        // Ring: random angle around the configured axis. A compass-perfect
        // circle of identical-radius stars reads as a procedurally-stamped CG
        // ring (reviewer pass 14); a real ring shell is lumpy and slightly
        // non-circular with a band of depth. So:
        //   • modulate the radius with two low-frequency sinusoids in phi
        //     (some sectors bulge, some pinch) → a wobbled, non-circular ring,
        //     with the phase keyed on the burst seed so adjacent rings differ;
        //   • widen the per-star radial jitter so stars sit at varied radii;
        //   • widen the out-of-plane scatter so the ring is a BAND of stars,
        //     not a razor-thin wire.
        // Per-star radial-SPEED variance (stars overtaking / lagging → varied
        // streak lengths) comes from the larger speedJitter the host sets for
        // the ring pattern, applied below.
        float3 e1, e2;
        basis_from_axis(u.ringAxis.xyz, e1, e2);
        float phi    = frand(rng) * 6.2831853f;
        float seedF  = float(seed & 0xFFFFu) * (6.2831853f / 65536.0f);
        float wobble = 1.0f + 0.12f * sin(phi * 3.0f + seedF)
                            + 0.07f * sin(phi * 5.0f + 1.3f);
        float r      = wobble * (1.0f + (frand(rng) - 0.5f) * (u.ringAxis.w + 0.20f));
        float a      = (frand(rng) - 0.5f) * 0.16f;  // wider out-of-plane band
        dir = normalize(e1 * cos(phi) * r + e2 * sin(phi) * r + u.ringAxis.xyz * a);
    } else if (kind == 2u) {
        // Heart curve: x = 16 sin³(t), y = 13 cos(t) − 5 cos(2t) − 2 cos(3t) − cos(4t).
        // Sample uniformly in t, project into the plane perpendicular to ringAxis.
        float3 e1, e2;
        basis_from_axis(u.ringAxis.xyz, e1, e2);
        float t = frand(rng) * 6.2831853f;
        float s = sin(t);
        float c = cos(t);
        float hx = 16.0f * s * s * s;
        float hy = 13.0f * c - 5.0f * cos(2.0f * t) - 2.0f * cos(3.0f * t) - cos(4.0f * t);
        // Normalize so the heart fits a unit sphere — the heart curve's
        // max radius is ~17, so divide by 17.
        float3 raw = e1 * (hx / 17.0f) + e2 * (hy / 17.0f);
        // Radial thickness gives the heart visible depth instead of a flat curve.
        float thick = (frand(rng) - 0.5f) * u.ringAxis.w;
        raw += u.ringAxis.xyz * thick;
        dir = length(raw) > 0.0001f ? normalize(raw) : float3(0, 1, 0);
    } else if (kind == 3u) {
        // FLAG: Spawn particles on a 2D rectangular grid in the plane
        // perpendicular to ringAxis, colored to read as the US flag —
        // 13 stripes (top red), blue canton in the upper-left covering
        // 7/13 height × 2/5 width, white stars sprinkled across the canton.
        //
        // This case is NOT a velocity-field shape like the others; the
        // particle is teleported to its grid cell and given near-zero
        // velocity. The integrator's gravity multiplier (set to ~0 by the
        // host) holds the tableau in place until the brightness curve
        // fades it out. Position is encoded in `dir` and then scaled by
        // `speed` in the common path below — speed becomes a half-width
        // for the flag.
        //
        // basis_from_axis's e1/e2 are orthonormal to the axis but have an
        // arbitrary roll — for a camera-facing axis (+Z), e2 lands on
        // world -Y, which would flip the flag vertically. We override the
        // basis so e1 is the world-horizontal axis (in-plane × world-up)
        // and e2 is world-up projected onto the plane — that pins
        // "row 0 = canton on TOP-LEFT" regardless of axis direction.
        float3 a  = normalize(u.ringAxis.xyz);
        float3 wu = float3(0.0f, 1.0f, 0.0f);
        // If the axis is nearly vertical, fall back to a world-X up so the
        // basis stays defined; the flag will face a bird's-eye viewer.
        if (abs(a.y) > 0.95f) wu = float3(1.0f, 0.0f, 0.0f);
        float3 e1 = normalize(cross(wu, a));  // horizontal in-plane axis (world-right-ish)
        float3 e2 = normalize(cross(a, e1));  // up in-plane axis (world-up-projected)
        // 60 cols × 32 rows of cells. Aspect 60/32 ≈ 1.875 — close to the
        // real flag's 1.9:1. Each thread maps deterministically to a cell;
        // any threads beyond 60×32 = 1920 wrap and overlay (id % total).
        // Doubled vs the original 30×16 so the GPU streak-quads (default
        // streakWidth 0.085 m) tile the flag densely at the 14 m half-width.
        const uint COLS = 60u;
        const uint ROWS = 32u;
        uint total = COLS * ROWS;
        uint cell  = id % total;
        uint col   = cell % COLS;
        uint row   = cell / COLS;
        // u, v in [-1, 1] across the flag's width / height. Width is the
        // long axis; height is half (1.9:1).
        float uu = (float(col) + 0.5f) / float(COLS) * 2.0f - 1.0f;
        // Flip row so row=0 is the TOP of the flag (canton on top-left).
        float vv = ((float(ROWS - 1u - row) + 0.5f) / float(ROWS)) * 2.0f - 1.0f;
        // Aspect: the height is half the width in world units. The host
        // multiplies `dir` by `speed` for offset, so encode an in-plane
        // offset directly. We'll set speed = halfWidth on the host side
        // so uu = ±1 lands at ±halfWidth and vv = ±1 lands at ±halfWidth/2
        // (i.e. height = width / 1.875).
        const float ASPECT_INV = 1.0f / 1.875f;
        // Small per-cell jitter for organic feel.
        float jx = (frand(rng) - 0.5f) * 0.03f;
        float jy = (frand(rng) - 0.5f) * 0.03f;
        // Compose in-plane position (still in normalized [-1, 1] x [-1, 1]).
        float px = uu + jx;
        float py = vv * ASPECT_INV + jy * ASPECT_INV;
        // Tiny out-of-plane scatter so the tableau has slight depth.
        float pz = (frand(rng) - 0.5f) * 0.02f;
        dir = e1 * px + e2 * py + u.ringAxis.xyz * pz;
        // Determine cell colour. Canton covers top 7/13 of height and
        // left 2/5 of width.
        // ROWS = 32, so canton rows 0..17 (top 18/32 = 0.5625, close to 7/13 ≈ 0.538).
        // COLS = 60, so canton cols 0..23 (left 24/60 = 0.4 = 2/5 exact).
        // Stripes outside the canton alternate red/white; row 0 = red.
        float3 cellColor;
        bool inCanton = (row < 18u) && (col < 24u);
        if (inCanton) {
            // Star sub-grid inside the canton — a star every 4×4 cells
            // (so ~6 cols × 4 rows of stars across the canton). Offset
            // alternating rows by 2 cols for the classic staggered look.
            uint cantonCol = col;
            uint cantonRow = row;
            uint colOff = ((cantonRow / 4u) % 2u) * 2u;  // 0 or 2
            bool isStar = (((cantonCol + colOff) % 4u) == 2u)
                       && ((cantonRow % 4u) == 2u);
            if (isStar) {
                // White stars — push past 1.0 so the bloom makes them pop.
                cellColor = float3(1.20f, 1.15f, 1.00f);
            } else {
                // Patriotic blue.
                cellColor = float3(0.15f, 0.25f, 0.85f);
            }
        } else {
            // Stripes. Group: each stripe is roughly ROWS/13 rows thick.
            // With 32 rows / 13 stripes ≈ 2.46 rows per stripe.
            float rowT = (float(row) + 0.5f) / float(ROWS);
            uint stripe = uint(rowT * 13.0f);
            if (stripe > 12u) stripe = 12u;
            bool red = (stripe % 2u) == 0u;
            cellColor = red
                ? float3(0.95f, 0.18f, 0.18f)
                : float3(0.98f, 0.98f, 0.98f);
        }
        // Stash the per-cell colour by overriding `base` later — but we
        // need it now while we know the cell. Trick: replace `u.baseColor`
        // for this thread by writing into a local override and skipping
        // the host-provided colour for the flag case. We do that below
        // by jumping to a post-direction code path.
        // Speed → magnitude (positional). Lifespan is host-set; gravity
        // multiplier is host-set to ~0 so the tableau holds.
        float baseSpeed = u.speedSizeLife.x;  // half-width in metres
        float3 vel = float3(0.0f, 0.0f, 0.0f);
        float baseLife = u.speedSizeLife.w;
        float lifeJit  = u.dragTurbGravLife.w;
        float life = max(0.05f, baseLife + (frand(rng) * 2.0f - 1.0f) * lifeJit);
        float size = u.speedSizeLife.z * (0.85f + frand(rng) * 0.3f);
        FWParticle p;
        p.positionAge     = float4(u.centerKind.xyz + dir * baseSpeed, 0.0f);
        p.velocityLife    = float4(vel, life);
        p.displayColor    = float4(cellColor * 3.5f, 1.0f);
        p.baseDrag        = float4(cellColor, u.dragTurbGravLife.x);
        p.paramsExtra     = float4(u.dragTurbGravLife.y, u.dragTurbGravLife.z, size, 0.0f);
        p.colorTransition = float4(0.0f);
        particles[slot] = p;
        return;
    } else {
        // 0 = uniform sphere — chrysanthemum, peony, willow all use this and
        // differentiate via integrator params (drag, gravity, life).
        dir = dir_sphere(rng);
    }

    float baseSpeed = u.speedSizeLife.x;
    float speedJit  = u.speedSizeLife.y;
    float speed = max(0.0f, baseSpeed + (frand(rng) * 2.0f - 1.0f) * speedJit);

    // ── Burst ASYMMETRY ────────────────────────────────────────────────
    // A real shell never bursts as a perfect sphere — uneven powder packing
    // gives gappy/clumpy radial distribution, wind shears the moment of
    // bursting into an off-axis lean, and the cluster opens with a faint
    // dent pattern. Three terms together (only on sphere/ring/heart, NOT
    // on flag which has its own logic + returns earlier):
    //   - per-burst wind shear: a small fixed offset to every spark in this
    //     burst (seed-derived so adjacent bursts lean differently)
    //   - per-direction speed modulation: 3D sin-noise of the direction
    //     scales speed, creating clusters and gaps
    //   - per-direction direction nudge: small noise-aligned tangent offset
    //     so the cluster edges aren't a clean sphere boundary
    {
        uint sub = seed * 1664525u + 1013904223u;
        float sx = (float(sub >> 16u) / 32768.0f) - 1.0f;
        float sy = (float((sub * 7u) >> 16u) / 32768.0f) - 1.0f;
        float sz = (float((sub * 13u) >> 16u) / 32768.0f) - 1.0f;
        float3 shellWind = float3(sx, sy * 0.4f, sz) * 0.10f;  // 10% of base speed

        // Direction-keyed low-freq noise (1.6 wavelength along each axis)
        float n = sin(dir.x * 1.6f + 1.7f)
                * cos(dir.y * 1.4f + 0.3f)
                * sin(dir.z * 1.9f + 2.6f);
        float speedScale = 0.78f + 0.32f * n;                      // 0.46..1.10
        speed *= speedScale;

        // Tangential dir nudge — break the perfect-sphere silhouette
        float3 tangent = float3(-dir.z, dir.x * 0.5f, dir.y);       // arbitrary perp-ish
        tangent = normalize(tangent - dir * dot(tangent, dir));     // project off dir
        dir = normalize(dir + tangent * (n * 0.18f));

        float3 vel = dir * speed + shellWind * baseSpeed;
        speed = length(vel);
        dir = vel / max(speed, 1e-4f);
    }

    float3 vel = dir * speed;

    float baseLife = u.speedSizeLife.w;
    float lifeJit  = u.dragTurbGravLife.w;
    float life = max(0.05f, baseLife + (frand(rng) * 2.0f - 1.0f) * lifeJit);

    // Per-particle color jitter: nudge a random fraction toward white so the
    // burst has some hot-cored sparks mixed in with the saturated tint.
    float3 base = u.baseColor.rgb;
    float colJitAmp = u.baseColor.w;
    float jit = frand(rng) * colJitAmp;
    base = mix(base, float3(1.0f, 0.95f, 0.85f), jit);

    // paramsExtra.z is the per-particle size hint consumed by the streak
    // kernel as a floor on streakWidth/minLen. Set to 0 for regular burst
    // sparks so the user-tuned `streakWidth` slider alone controls billboard
    // size — the .flag case above writes a non-zero hint because each flag
    // pixel needs a fat billboard to read as a tableau pixel rather than a
    // 5 cm sliver.

    // colorTransition — per-particle roll against the burst's transition
    // chance. Real fireworks colour-changers transition almost-all stars
    // together (one chemistry layer burns out across the cloud at once),
    // so the host typically sets chance to 0 OR ~0.95. Per-particle here
    // gives soft sprinkles of "unchanged" stars for organic feel.
    // transitionAt randomised in [0.35, 0.65] so the crossover doesn't
    // happen at exactly the same age across the cloud.
    float transitionChance = u.secondaryColor.a;
    float4 colorTransition = float4(0.0f);
    if (transitionChance > 0.0f && frand(rng) < transitionChance) {
        float transitionAt = 0.35f + frand(rng) * 0.30f;
        colorTransition = float4(u.secondaryColor.rgb, transitionAt);
    }

    // ── Per-spark DRAG + GRAVITY stratification ─────────────────────────
    // Real fireworks star packing is never uniform — heavier sparks fall
    // faster, lighter ones hang. Per-particle jitter on drag (0.6..1.5×)
    // and gravity (0.7..1.3×) makes the ember cloud stratify into a curtain
    // over time instead of staying a coherent expanding sphere. Free
    // realism. The host's burst-level drag/gravity values stay the centre
    // of the distribution; this just spreads particles around them.
    float dragJitter = 0.6f + frand(rng) * 0.9f;
    float gravJitter = 0.7f + frand(rng) * 0.6f;
    float perSparkDrag = u.dragTurbGravLife.x * dragJitter;
    float perSparkGrav = u.dragTurbGravLife.z * gravJitter;

    FWParticle p;
    p.positionAge     = float4(u.centerKind.xyz, 0.0f);
    p.velocityLife    = float4(vel, life);
    p.displayColor    = float4(base * 3.5f, 1.0f);  // hot start
    p.baseDrag        = float4(base, perSparkDrag);
    p.paramsExtra     = float4(u.dragTurbGravLife.y, perSparkGrav, 0.0f, 0.0f);
    p.colorTransition = colorTransition;
    particles[slot] = p;
}

// ── Streak-build kernel ─────────────────────────────────────────────────────
//
// Once per frame, after `fwIntegrate`, build a 4-vertex view-aligned billboard
// quad per particle so SceneKit can draw the spark as a velocity-stretched
// streak instead of a uniform point. The "motion blur" the eye reads from real
// fireworks comes from the spark's photographic trail; with point primitives
// we can't fake that — every spark renders as the same little square. Streaks
// are the fix.
//
// Quad layout (head = current position, tail = current − vel·duration):
//
//        head-left (1) ── head-right (0)
//                |             |
//        tail-left (2) ── tail-right (3)
//
// Triangles drawn by the renderer's pre-filled index buffer: (0,1,2) + (0,2,3).
// Width axis is `cross(streakDir, viewDir)` — degenerate when the spark is
// flying straight at the camera; we fall back to any in-screen perpendicular
// so the quad still has area.
//
// Tail vertices carry a brightness fade (alpha ramp) so the streak reads as a
// hot head dragging a cooling tail, even with hard-edge additive blending.

kernel void fwBuildStreaks(
    device       FWParticle*       particles [[ buffer(0) ]],
    device       FWStreakVertex*   streaks   [[ buffer(1) ]],
    device       float2*           streakUVs [[ buffer(2) ]],
    constant     FWStreakUniforms& u         [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint count = as_type<uint>(u.cameraCount.w);
    if (id >= count) return;

    uint base = id * 4u;

    FWParticle p = particles[id];
    float age  = p.positionAge.w;
    float life = p.velocityLife.w;

    // Dead → park all four corners at the same offscreen sentinel. SceneKit
    // still draws the two triangles but they collapse to zero area outside
    // the visible volume. UVs don't matter for offscreen verts, but write a
    // safe value so a stale buffer doesn't tile-bomb the first visible frame.
    if (life <= 0.0f || age >= life) {
        FWStreakVertex dead;
        dead.position = float4(0.0f, -10000.0f, 0.0f, 1.0f);
        dead.color    = float4(0.0f, 0.0f, 0.0f, 0.0f);
        streaks[base + 0u] = dead;
        streaks[base + 1u] = dead;
        streaks[base + 2u] = dead;
        streaks[base + 3u] = dead;
        streakUVs[base + 0u] = float2(0.0f, 0.0f);
        streakUVs[base + 1u] = float2(0.0f, 0.0f);
        streakUVs[base + 2u] = float2(0.0f, 0.0f);
        streakUVs[base + 3u] = float2(0.0f, 0.0f);
        return;
    }

    float3 pos = p.positionAge.xyz;
    float3 vel = p.velocityLife.xyz;
    float3 rgb = p.displayColor.rgb;

    // Atmospheric DEPTH HAZE — far sparks soften and cool. Pure black sky
    // means real distance attenuation is mostly invisible without this; with
    // it, bursts at z = -55 read distinctly farther than bursts at z = -10
    // and the over-water staging gains depth.
    float distToCam = distance(pos, u.cameraCount.xyz);
    float fogT = saturate((distToCam - 25.0f) / 90.0f);           // 0 near, 1 far
    float fogDim = mix(1.0f, 0.55f, fogT);                         // far = 55% brightness
    float3 fogTint = float3(0.55f, 0.62f, 0.78f);                  // cool blue
    rgb = mix(rgb * fogDim, fogTint * fogDim * length(rgb) * 0.5f, fogT * 0.45f);

    float duration = u.streakParams.x;
    float width    = u.streakParams.y;
    float minLen   = u.streakParams.z;

    // Per-particle size override: paramsExtra.z carries the spawn-time size
    // hint. When > 0 it scales the streak's width and minimum-length floor —
    // useful for the .flag burst kind where each spark sits still and needs
    // a larger billboard to read as a flag-pixel rather than a 0.085 m sliver.
    float sizeHint = p.paramsExtra.z;
    if (sizeHint > 0.0f) {
        width  = max(width,  sizeHint);
        minLen = max(minLen, sizeHint);
    }

    float3 toCam   = u.cameraCount.xyz - pos;
    float  camDist = length(toCam);
    float3 viewDir = camDist > 1e-4f ? toCam / camDist : float3(0.0f, 0.0f, 1.0f);

    float speed = length(vel);
    float3 streakDir = speed > 1e-4f ? vel / speed : float3(0.0f, 1.0f, 0.0f);

    // Age-based streak length scaling: a real long-exposure burst only
    // motion-blurs during the explosion frame. After the peak the sparks
    // become individual dying embers — each one a hot dot falling through
    // the sky, NOT a continuous line. Without this taper, gravity-falling
    // embers at 5..15 m/s drag 20..60 cm streaks behind them and read as
    // rain/lines, defeating the "collection of dying particles" look.
    //
    // 0..5%  of life : full streak (the burst frame — sells the explosion)
    // 5..45% of life : tapers smoothly to zero
    // 45%+   of life : collapses to minLen (just the soft sprite dot)
    float ageFrac  = age / life;
    float ageScale = 1.0f - smoothstep(0.05f, 0.45f, ageFrac);
    float ageScaleStreakLen = speed * duration * ageScale;

    // ── PREVENT THE "PRE-EXPLOSION IMPLOSION" ARTEFACT ──────────────────
    //
    // Every burst particle spawns AT the burst centre and immediately
    // gets a full outward velocity. The previous formulation set
    //   tailPos = pos - streakDir * (speed * duration * ageScale)
    // — i.e. the tail extended a FULL streak length in the OPPOSITE
    // direction of motion, regardless of how far the particle had
    // actually moved. At age=0 the head was at centre, the tail was
    // a full streak length BEHIND centre (i.e. ON THE OTHER SIDE of
    // the burst), and the quad straddled the spawn point.
    //
    // Sum that across hundreds of particles firing in every
    // direction from a single point and the midpoints of all the
    // streak quads cluster RIGHT AT the burst centre with mid-bright
    // interpolated colours. The eye reads that as a brief bright
    // "implosion" or pre-bang flash, immediately before the
    // particles' heads pull out far enough that the streaks fall
    // entirely outside the centre.
    //
    // Fix: clamp the streak length to the distance the particle has
    // ACTUALLY travelled since spawn (`age * speed`). The tail can't
    // extend further back than the spawn point — there's no way for
    // it to reach across the centre into the wrong hemisphere.
    float traveled = age * speed;
    float streakLen = max(minLen, min(ageScaleStreakLen, traveled));
    float3 headPos = pos;
    float3 tailPos = pos - streakDir * streakLen;

    // Width axis perpendicular to both motion and view. If the spark is
    // travelling straight at / away from the camera, those two are parallel
    // and the cross collapses — pick an in-screen perpendicular instead.
    float3 wAxis = cross(streakDir, viewDir);
    float wLen   = length(wAxis);
    if (wLen < 1e-4f) {
        float3 ref = abs(viewDir.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f)
                                           : float3(1.0f, 0.0f, 0.0f);
        wAxis = normalize(cross(viewDir, ref));
    } else {
        wAxis /= wLen;
    }
    float halfW = width * 0.5f;

    float3 v0 = headPos + wAxis * halfW;   // head-right
    float3 v1 = headPos - wAxis * halfW;   // head-left
    float3 v2 = tailPos - wAxis * halfW;   // tail-left
    float3 v3 = tailPos + wAxis * halfW;   // tail-right

    // Tail fade tracks the same age curve as streak length: while the
    // spark is in its streaking phase (ageScale ~ 1) the tail goes to zero
    // so the quad reads as a comet head (bright sprite at the head, fading
    // to invisible at the tail). Once the streak has collapsed to minLen
    // (ageScale ~ 0) the tail equals the head so the sprite renders as a
    // uniform soft round dot — the dying-ember look.
    //
    // The RGB pre-multiplication by tailFade is preserved here for
    // backwards compatibility with the SCN-pipeline Fireworks+ scene,
    // which renders the streak quad with a sprite-tiled material whose
    // additive blend uses RGB as the brightness signal. Fireworks+
    // reads only `color.rgb` (vertex format float3 in its
    // SCNGeometrySource), so it never sees the alpha channel.
    //
    // ── COLOR.A REPURPOSED AS STRETCH FACTOR ────────────────────────
    //
    // FireworksUltra's custom Metal compositor reads `color` as a full
    // float4 so the fragment shader can pull `vTile` out of `color.a`.
    // This is the "stretch factor" of the streak quad — how many
    // multiples of `width` the streak spans along its velocity axis.
    //
    // We need this per-fragment so the SDF can be made ANISOTROPIC
    // (stretched along the velocity direction) to render proper
    // cinematic motion blur: fast-moving particles become elongated
    // capsules; slow particles stay spherical. Without per-fragment
    // access to vTile the shader can't tell static dots apart from
    // fast streaks and they all render the same.
    //
    // We write the SAME vTile value at BOTH head and tail vertices —
    // it's a per-quad constant, not a per-vertex gradient — so the
    // fragment shader gets the same value regardless of which side
    // of the quad it interpolates from.
    // vTile (streak stretch factor) is declared up here so it can flow
    // into BOTH the per-vertex color.a (consumed by FireworksUltra's
    // SDF fragment shader as the anisotropic stretch factor) AND the
    // streak UV.y values below (consumed by the SCN-pipeline scenes
    // for sprite tiling along the streak).
    float vTile = max(1.0f, streakLen / max(width, 1e-4f));
    float tailFade = 1.0f - ageScale;
    float4 headColor = float4(rgb,             vTile);
    float4 tailColor = float4(rgb * tailFade,  vTile);

    FWStreakVertex h0; h0.position = float4(v0, 1.0f); h0.color = headColor;
    FWStreakVertex h1; h1.position = float4(v1, 1.0f); h1.color = headColor;
    FWStreakVertex t2; t2.position = float4(v2, 1.0f); t2.color = tailColor;
    FWStreakVertex t3; t3.position = float4(v3, 1.0f); t3.color = tailColor;

    streaks[base + 0u] = h0;
    streaks[base + 1u] = h1;
    streaks[base + 2u] = t2;
    streaks[base + 3u] = t3;

    // Tiled-sprite UVs: V on the head verts is (streakLen / width) so the
    // Gaussian sprite repeats once per sprite-width along the streak. With
    // the material's wrap mode = .repeat this paints a chain of round dots
    // along the velocity vector instead of one stretched ellipse. U stays
    // 0..1 across width because there's no need to tile across width — the
    // streak is one sprite wide. minimum 1.0 keeps the head sprite intact
    // when the streak is shorter than the sprite (collapses to a dot).
    // `vTile` is already declared above (it's also needed in color.a).
    streakUVs[base + 0u] = float2(1.0f, vTile);  // head-right
    streakUVs[base + 1u] = float2(0.0f, vTile);  // head-left
    streakUVs[base + 2u] = float2(0.0f, 0.0f);   // tail-left
    streakUVs[base + 3u] = float2(1.0f, 0.0f);   // tail-right
}

// ── Spawn-sparks kernel (shell-rise trail) ──────────────────────────────────
//
// One thread per emit. Reads the emit's world position + seed, picks a random
// unit-sphere direction at low speed, and writes one FWParticle into the
// shared particle buffer at `firstIndex + id`. After this kernel runs, the
// per-frame `fwIntegrate` pass advances the spark exactly like any other
// particle — same brightness profile, same streak-quad build pass — so trail
// sparks bloom alongside burst sparks with zero extra renderer code.
//
// Why a separate kernel from `fwSpawnBurst`: bursts spawn `count` particles
// from a single center; trails spawn one particle from each of `count`
// per-emit positions. Folding both into one kernel would muddy the spawn
// uniform layout for negligible code savings.

kernel void fwSpawnSparks(
    device       FWParticle*       particles [[ buffer(0) ]],
    constant     FWSparkUniforms&  u         [[ buffer(1) ]],
    device const FWSparkEmit*      emits     [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint count = as_type<uint>(u.firstCountSeedTime.y);
    if (id >= count) return;

    uint firstIndex = as_type<uint>(u.firstCountSeedTime.x);
    uint globalSeed = as_type<uint>(u.firstCountSeedTime.z);
    uint slot       = firstIndex + id;

    FWSparkEmit emit = emits[id];
    uint emitSeed = as_type<uint>(emit.positionSeed.w);

    // Mix global + per-emit seed with the slot index so two emits at the same
    // world position from the same call still get different RNG streams.
    uint rng = globalSeed ^ emitSeed ^ (id * 2654435761u);

    float3 dir = dir_sphere(rng);

    float baseSpeed = u.speedLifeJitter.x;
    float speedJit  = u.speedLifeJitter.y;
    float speed = max(0.0f, baseSpeed + (frand(rng) * 2.0f - 1.0f) * speedJit);
    float3 vel = dir * speed;

    float baseLife = u.speedLifeJitter.z;
    float lifeJit  = u.speedLifeJitter.w;
    float life = max(0.05f, baseLife + (frand(rng) * 2.0f - 1.0f) * lifeJit);

    // Per-spark color jitter toward warm white, same nudge the burst kernel
    // uses so a handful of sparks read as hotter than the warm base tone.
    float3 base = u.baseColorJitter.rgb;
    float colJitAmp = u.baseColorJitter.a;
    float jit = frand(rng) * colJitAmp;
    base = mix(base, float3(1.0f, 0.95f, 0.85f), jit);

    // Trail sparks ride the user-tuned streakWidth too — leave paramsExtra.z
    // at 0 so the streak kernel doesn't max-clamp them up to the spawn-time
    // `size` field. (Only the .flag burst kind needs a per-particle size
    // override.) Trail sparks never colour-transition.
    FWParticle p;
    p.positionAge     = float4(emit.positionSeed.xyz, 0.0f);
    p.velocityLife    = float4(vel, life);
    p.displayColor    = float4(base * 3.5f, 1.0f);
    p.baseDrag        = float4(base, u.dragTurbGrav.x);
    p.paramsExtra     = float4(u.dragTurbGrav.y, u.dragTurbGrav.z, 0.0f, 0.0f);
    p.colorTransition = float4(0.0f);
    particles[slot] = p;
}
