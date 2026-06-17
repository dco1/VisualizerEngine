#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA VOLUMETRIC LIGHT SHAFT ────────────────────────────────────
//
// Single-scatter ray-march of the sun through the room's hazy air, so the
// sunbeam is visible IN THE AIR (god-rays), not only where it lands on the
// floor. This is the physically-grounded answer the project's "don't fake light
// shafts with scrolling textures" rule calls for: for each view ray we step
// from the camera to the first surface, and at each step test whether direct
// sun reaches that air sample (the SAME window shadow test the floor + dust
// use — trace toward the sun, see if it exits through the window opening). The
// in-scattered radiance is weighted by a Henyey-Greenstein phase function so
// the beam glows brightest when the camera looks toward the sun. A per-pixel
// jittered start offset breaks up banding; the renderer's TAA accumulates it
// out across frames.

struct VolUniforms {
    float4x4 invViewProjection;
    float3 cameraWorldPos; float fogDensity;
    float3 sunDir;         float scatterStrength;   // sunDir = toward the sun
    float3 sunColor;       float anisotropy;
    float windowX; float winY0; float winY1; float winZ0;
    float winZ1; float maxDist; uint steps; uint frameSeed;
    uint width; uint height; float feather; uint isOutdoor; // outdoor: skip window test
};

static inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static inline float sunShaft(float3 P, constant VolUniforms& u) {
    // Outdoor mode: no window to test through — all air is sun-lit.
    // Mie scatter is still view-angle weighted via HG phase below.
    if (u.isOutdoor) return 1.0;
    if (u.sunDir.x <= 1e-3) return 0.0;
    float s = (u.windowX - P.x) / u.sunDir.x;
    if (s <= 0.0) return 0.0;
    float yc = P.y + s * u.sunDir.y;
    float zc = P.z + s * u.sunDir.z;
    float fy = smoothstep(u.winY0 - u.feather, u.winY0 + u.feather, yc) *
               (1.0 - smoothstep(u.winY1 - u.feather, u.winY1 + u.feather, yc));
    float fz = smoothstep(u.winZ0 - u.feather, u.winZ0 + u.feather, zc) *
               (1.0 - smoothstep(u.winZ1 - u.feather, u.winZ1 + u.feather, zc));
    return fy * fz;
}

kernel void illumi_volumetric(
    texture2d<float, access::read>       gDepth [[texture(0)]],
    texture2d<half,  access::read_write> outHDR [[texture(1)]],
    constant VolUniforms&                u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;

    float2 ndc = (float2(gid) + 0.5) / float2(u.width, u.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 farClip = float4(ndc, 1.0, 1.0);
    float4 fw = u.invViewProjection * farClip;
    float3 ro = u.cameraWorldPos;
    float3 rd = normalize(fw.xyz / fw.w - ro);

    float depth = gDepth.read(gid).r;
    float tEnd = u.maxDist;
    if (depth < 0.99999) {
        float4 c = float4(ndc, depth, 1.0);
        float4 w = u.invViewProjection * c;
        tEnd = min(u.maxDist, length(w.xyz / w.w - ro));
    }
    if (tEnd <= 0.01) return;

    uint N = max(1u, min(u.steps, 96u));
    float dtStep = tEnd / float(N);
    float jitter = float(pcgHash(gid.x + gid.y * u.width + u.frameSeed * 2654435761u))
                   * (1.0 / 4294967296.0);

    float g = clamp(u.anisotropy, -0.95, 0.95);
    float cosT = dot(rd, normalize(u.sunDir));
    float hg = (1.0 - g * g) /
               (4.0 * M_PI_F * pow(max(1e-3, 1.0 + g * g - 2.0 * g * cosT), 1.5));
    // Blend in a small isotropic floor so the beam reads as a cone even when
    // the camera isn't looking straight down the sun vector (a pure HG lobe
    // makes the shaft vanish off-axis).
    float phase = mix(1.0 / (4.0 * M_PI_F), hg, 0.82);

    float accum = 0.0;
    for (uint i = 0; i < N; ++i) {
        float t = (float(i) + jitter) * dtStep;
        float3 P = ro + rd * t;
        accum += sunShaft(P, u);
    }
    accum *= dtStep * u.fogDensity * phase * u.scatterStrength;

    half4 prev = outHDR.read(gid);
    outHDR.write(half4(prev.rgb + half3(u.sunColor * accum), prev.a), gid);
}
