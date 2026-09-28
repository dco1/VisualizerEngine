// ── SKY RADIANCE PROBE ──────────────────────────────────────────────────────
//
// `VolumetricCloudRenderer.radianceProbes` (opt-in; a host that sets none never dispatches this):
// the cosine-weighted mean radiance of the dome over the hemisphere around a plane's normal —
// E(n)/π, the irradiance the plane receives from the sky (and the ground below the horizon, as
// the dome draws it), expressed as the radiance of a Lambertian surface emitting that flux. A
// window facing n passes exactly this flux into a room, so it is the radiance of the window's
// portal light. One threadgroup per probe; encoded right after the dome it reads, in the same
// command buffer, so it always sees the sky the dome shows. The host reads the few float4s back
// from a completion handler — no wait.
//
// Sampling: kProbeTheta cosine-weighted rings (sin²θ uniform) × kProbePhi azimuths, the odd rings
// offset half a step — 4096 bilinear dome samples. The mean of cosine-weighted samples IS E/π, so
// there are no weights to carry. An optional excluded cone (the dome's stylised sun disc + halo,
// whose light the host already carries as its directional) is dropped and the mean taken over the
// rest; w reports the fraction of samples kept.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

/// Swift mirror: `VolumetricCloudRenderer.RadianceProbeGPU` (32 bytes).
struct SkyRadianceProbeIn {
    float4 normal;    // xyz = unit normal (the hemisphere's axis), w unused
    float4 exclude;   // xyz = unit direction of an excluded cone, w = cos(half-angle); w > 1 = none
};

constant uint kProbeThreads = 256;
constant uint kProbeTheta   = 32;
constant uint kProbePhi     = 128;

kernel void skyRadianceProbe(texture2d<float, access::sample> sky    [[texture(0)]],
                             constant SkyRadianceProbeIn*    probes [[buffer(0)]],
                             device float4*                  out    [[buffer(1)]],
                             uint probe [[threadgroup_position_in_grid]],
                             uint tid   [[thread_index_in_threadgroup]]) {
    threadgroup float4 partial[kProbeThreads];
    const float3 N = normalize(probes[probe].normal.xyz);
    const float3 up = abs(N.y) < 0.999f ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 T = normalize(cross(up, N));
    const float3 B = cross(N, T);
    const float3 ex = probes[probe].exclude.xyz;
    const float exCos = probes[probe].exclude.w;

    float3 acc = float3(0.0f);
    float kept = 0.0f;
    const uint total = kProbeTheta * kProbePhi;
    for (uint s = tid; s < total; s += kProbeThreads) {
        uint i = s / kProbePhi, j = s % kProbePhi;
        float sinT = sqrt((float(i) + 0.5f) / float(kProbeTheta));
        float cosT = sqrt(max(0.0f, 1.0f - sinT * sinT));
        float phi = (float(j) + 0.5f + 0.5f * float(i & 1u)) * (2.0f * M_PI_F / float(kProbePhi));
        float3 d = T * (sinT * cos(phi)) + B * (sinT * sin(phi)) + N * cosT;
        if (exCos <= 1.0f && dot(d, ex) > exCos) continue;
        float3 L = sampleSkyEquirect(sky, d);
        if (any(isnan(L)) || any(isinf(L))) continue;
        acc += max(L, float3(0.0f));
        kept += 1.0f;
    }
    partial[tid] = float4(acc, kept);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kProbeThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float4 p = partial[0];
        out[probe] = float4(p.xyz / max(p.w, 1.0f), p.w / float(total));
    }
}
