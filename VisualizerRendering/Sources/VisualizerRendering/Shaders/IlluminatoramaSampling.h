// ── ILLUMINATORAMA — progressive, per-pixel stratified sampling ──────────────
//
// ONE sampler for the stochastic terms a still / settled accumulator averages: traced AO, the
// traced sun cone, window-portal visibility, one-bounce GI (Daydream DH-0887). It replaces the
// pattern all of them used — a per-pixel WHITE-noise hash re-randomised every frame
// (`pcgHash`/`rnd`), whose N-frame average converges only as 1/√N.
//
// Ray `i` of accumulated frame `k` takes point `k·rays + i` of the pixel's OWN Owen-scrambled
// Sobol sequence, so an average over the frames of a still is one stratified set that keeps
// improving at every power of two (≈N⁻¹ on a distance-weighted visibility integrand, measured
// on a crease model: 4 rays × 32 frames beats 16 white-noise rays per frame by 1.5–4×). The
// index must be CONTIGUOUS per estimator — every other Sobol index covers only half the square.
//
// Rejected, measured on the same model: a Kronecker/R2 sequence with a Hilbert-curve spatial
// dither (XeGTAO's layout) — ~2× the per-pixel error for a blue-noise advantage worth 10–25 %
// only while a filter runs every frame; and R3 with a stochastic ray length — near-linear
// relations between its dimensions (3α₁+α₃ ≈ 3) left the length a function of the direction.

#ifndef ILLUMI_SAMPLING_H
#define ILLUMI_SAMPLING_H

#include <metal_stdlib>
using namespace metal;

/// Concentric (Shirley–Chiu) square → disk map: adjacent strata stay adjacent.
static inline float2 illumiConcentricDisk(float2 u) {
    float2 o = 2.0 * u - 1.0;
    if (o.x == 0.0 && o.y == 0.0) return float2(0.0);
    float r, theta;
    if (abs(o.x) > abs(o.y)) { r = o.x; theta = (M_PI_F * 0.25) * (o.y / o.x); }
    else                     { r = o.y; theta = (M_PI_F * 0.5) - (M_PI_F * 0.25) * (o.x / o.y); }
    return r * float2(cos(theta), sin(theta));
}

// ── Owen-scrambled Sobol (Burley 2020, "Practical Hash-based Owen Scrambling") ──────────────
//
// The per-pixel sequence for a term whose error is judged PER PIXEL after accumulation (traced
// AO): measured on a planar-crease model, 2-D Owen-scrambled Sobol with a distance-weighted
// estimator reaches, at 4 rays × 32 frames, 1.5–4× lower RMS than 16 white-noise rays per frame,
// and it keeps the progressive property — samples 0…N−1 of a pixel are a (0,2)-sequence prefix
// in natural index order, so ANY stopping point is stratified. Each pixel gets its own hashed
// scramble, so there is no tiling in the residual.

static inline uint illumiPCG(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Laine–Karras permutation — Burley's hash-based nested uniform scramble.
static inline uint illumiLKPermutation(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}
static inline uint illumiOwenScramble(uint x, uint seed) {
    return reverse_bits(illumiLKPermutation(reverse_bits(x), seed));
}

/// Point `index` of the 2-D Sobol sequence, Owen-scrambled by `seed`, in [0,1)².
static inline float2 illumiSobolOwen2D(uint index, uint seed) {
    uint s0 = reverse_bits(index);                        // dimension 0: van der Corput
    uint s1 = 0u, v = 0x80000000u;                        // dimension 1: primitive poly x + 1
    for (uint i = index; i != 0u; i >>= 1u) {
        if ((i & 1u) != 0u) s1 ^= v;
        v ^= v >> 1u;
    }
    uint a = illumiOwenScramble(s0, illumiPCG(seed));
    uint b = illumiOwenScramble(s1, illumiPCG(seed ^ 0x9e3779b9u));
    return float2(float(a >> 8), float(b >> 8)) * (1.0 / 16777216.0);
}

/// Point `index` of the pixel's sequence with the INDEX Owen-shuffled too (Burley 2020 §4). A
/// nested-uniform scramble of the index maps every aligned power-of-two block of indices onto an
/// aligned block, so every power-of-two prefix is still a (0,m,2)-net — but the ORDER inside each
/// block becomes pseudo-random per pixel. That order matters whenever the index is a frame
/// counter shared with something else base-2: the TAA jitter is Halton(k+1, 2), whose first
/// digit is the complement of vdC(k)'s, so an unshuffled index paired each pixel's left/right
/// jitter half with a FIXED half of the sample square and a still converged to a wrong value on
/// every one-pixel penumbra (DH-0887 review: RMS stalled at 0.065 by 256 frames; shuffled 0.019).
static inline float2 illumiSobolOwen2DShuffled(uint index, uint seed) {
    return illumiSobolOwen2D(illumiOwenScramble(index, illumiPCG(seed ^ 0x5BD1E995u)), seed);
}

/// A per-pixel scramble seed (not a linear function of the coordinates, so no lattice shows).
static inline uint illumiPixelSeed(uint2 pixel, uint salt) {
    return illumiPCG(illumiPCG(pixel.x ^ (pixel.y << 16u)) ^ salt);
}

/// Orthonormal basis whose seam is OFF the axis-aligned normals. `illumiONB` (Duff) flips on the
/// sign of n.z, and a y-up interior's floors, ceilings and x-facing walls all sit at n.z ≈ ±0 —
/// a normal rebuilt from depth then flips its disk 180° per pixel at random.
static inline void illumiONBAxisSafe(float3 n, thread float3& t, thread float3& b) {
    float3 ref = abs(n.y) > 0.5 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    t = normalize(cross(ref, n));
    b = cross(n, t);
}

/// Cosine-weighted direction about `n` (Malley's method on the concentric disk), axis-safe frame.
static inline float3 illumiCosineHemisphereAxisSafe(float3 n, float2 u) {
    float2 d = illumiConcentricDisk(u);
    float z = sqrt(max(0.0, 1.0 - dot(d, d)));
    float3 t, b; illumiONBAxisSafe(n, t, b);
    return t * d.x + b * d.y + n * z;
}

#endif
