#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA DEPTH OF FIELD ────────────────────────────────────────────
//
// Gather depth-of-field on the resolved HDR composite, after TAA + exposure
// estimate and before bloom (so out-of-focus bright dust motes spread into soft
// discs that then bloom — the bokeh that sells a low, cinematic camera). The
// circle of confusion grows with the view-space distance of a pixel from the
// focus plane; a golden-angle disc of taps, sized by that CoC, does the blur.
// Bright taps are weighted up slightly so highlights form defined bokeh rather
// than washing out.

struct DOFParams {
    float4x4 invProjection;
    float focusDist; float aperture; float maxRadius; float focusRange;
    uint width; uint height; float _p0; float _p1;
};

kernel void illumi_dof(
    texture2d<half,  access::read>  inHDR  [[texture(0)]],
    texture2d<float, access::read>  gDepth [[texture(1)]],
    texture2d<half,  access::write> outHDR [[texture(2)]],
    constant DOFParams&             p      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    half3 center = inHDR.read(gid).rgb;

    float depth = gDepth.read(gid).r;
    float2 ndc = (float2(gid) + 0.5) / float2(p.width, p.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 vp = p.invProjection * float4(ndc, depth, 1.0);
    float viewZ = abs(vp.z / vp.w);

    float radius = clamp(abs(viewZ - p.focusDist) / max(p.focusRange, 0.01),
                         0.0, 1.0) * p.maxRadius * p.aperture;
    if (radius < 0.6) {
        outHDR.write(half4(center, 1.0h), gid);
        return;
    }

    float3 sum = float3(0.0);
    float wsum = 0.0;
    const int TAPS = 32;
    for (int i = 0; i < TAPS; ++i) {
        float t = (float(i) + 0.5) / float(TAPS);
        float rr = sqrt(t) * radius;
        float a = float(i) * 2.39996323;        // golden angle
        int2 o = int2(round(cos(a) * rr), round(sin(a) * rr));
        int2 s = clamp(int2(gid) + o, int2(0), int2(int(p.width) - 1, int(p.height) - 1));
        float3 c = float3(inHDR.read(uint2(s)).rgb);
        float lum = dot(c, float3(0.299, 0.587, 0.114));
        float w = 1.0 + lum * 0.15;             // bokeh: favour bright samples
        sum += c * w;
        wsum += w;
    }
    outHDR.write(half4(half3(sum / max(wsum, 1e-3)), 1.0h), gid);
}
