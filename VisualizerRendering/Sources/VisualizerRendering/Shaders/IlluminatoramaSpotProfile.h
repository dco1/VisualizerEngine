#ifndef ILLUMINATORAMA_SPOT_PROFILE_H
#define ILLUMINATORAMA_SPOT_PROFILE_H
#include <metal_stdlib>
using namespace metal;

// Daydream DH-0540 — a spot light's PHOTOMETRIC PROFILE. Every spot is an analytic cone (a smoothstep rim feather);
// a real luminaire's light has a measured angular distribution — a can's crisp reflector cutoff with faint spill, a
// sconce's scallop, a pendant's downlight-with-halo. The host hands the renderer a table of rows (one per curated
// luminaire + hemisphere, `ILLUMI_SPOT_PROFILE_SAMPLES` samples uniformly over 0..90° off the cone axis, flux-matched
// to the cone they replace), and a spot names its row with `profileIndex` (−1 ⇒ the analytic cone, byte-identical).
// The row is MULTIPLIED into the cone term after its own rim feather: it shapes intensity within the cone, it never
// moves the cone, the shadow map or the wall trap.
#define ILLUMI_SPOT_PROFILE_SAMPLES 128u

static inline float illumiSpotProfile(const device float* rows, int row, float coneCos) {
    if (row < 0) return 1.0;
    float theta = acos(clamp(coneCos, -1.0, 1.0));
    float x = saturate(theta / (0.5 * M_PI_F)) * float(ILLUMI_SPOT_PROFILE_SAMPLES - 1u);
    uint i0 = min(uint(x), ILLUMI_SPOT_PROFILE_SAMPLES - 2u);
    uint base = uint(row) * ILLUMI_SPOT_PROFILE_SAMPLES;
    return mix(rows[base + i0], rows[base + i0 + 1u], x - float(i0));
}
#endif
