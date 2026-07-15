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

// ── SPOT-LIGHT BEAM SCATTERING ───────────────────────────────────────────────
//
// Single-scatter march of every spot light's cone through hazy air, so a
// moving-head fixture reads as a BEAM, not just a pool of light where it
// lands. Same physical framing as the sun pass above (in-scatter weighted by
// a Henyey-Greenstein phase, per-pixel jitter that TAA accumulates out), but
// the march is bounded per spot: each ray is clipped to the spot's range
// SPHERE analytically, and only that interval is sampled. That keeps cost
// proportional to beam coverage rather than steps × spots × full ray length,
// and puts full sample density inside the cone where it matters.
//
// The cone test happens per-sample (smoothstep between outer/inner cosines)
// instead of solving the ray/cone quadratic — the quadratic's edge cases
// (camera inside the cone, ray ∥ axis, degenerate half-angle) all vanish and
// the soft cone edge falls out for free. Beams are NOT depth-occluded inside
// the interval (no per-sample shadow trace); the interval end is still
// clamped to the G-buffer depth so geometry in front of a beam hides it.
//
// Mirrors the Swift `IlluminatoramaSpotLight` / lighting-kernel `SpotLight`
// layout — keep in lockstep (stride 176).

struct VolSpotLight {
    float3   position;
    float    innerCone;
    float3   direction;       // away from apex, normalized host-side
    float    outerCone;
    float3   color;           // premultiplied intensity
    float    radius;
    float4x4 shadowMatrix;    // unused here
    int      shadowSliceIndex;
    int      _padSpot0;
    int      _padSpot1;
    int      _padSpot2;
};

struct VolSpotUniforms {
    float4x4 invViewProjection;
    float3 cameraWorldPos; float density;
    float anisotropy; float strength; uint spotCount; uint steps;
    float maxDist; uint frameSeed; uint width; uint height;
};

kernel void illumi_volumetric_spots(
    texture2d<float, access::read>       gDepth [[texture(0)]],
    texture2d<half,  access::read_write> outHDR [[texture(1)]],
    constant VolSpotUniforms&            u      [[buffer(0)]],
    const device VolSpotLight*           spots  [[buffer(1)]],
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

    uint N = max(4u, min(u.steps, 48u));
    // Per-pixel seed; each STEP re-jitters below (decorrelating samples along
    // the ray averages the variance within one frame — a single whole-ray
    // offset left visible per-pixel speckle that TAA alone couldn't settle).
    uint seed = pcgHash(gid.x + gid.y * u.width + u.frameSeed * 2654435761u);
    float g = clamp(u.anisotropy, -0.95, 0.95);

    float3 accum = float3(0.0);
    for (uint s = 0; s < u.spotCount; ++s) {
        VolSpotLight sl = spots[s];
        if ((sl.color.r + sl.color.g + sl.color.b) <= 1e-4) continue;

        // Clip the view ray to the spot's range sphere — the beam can't
        // scatter outside it, so that's the only interval worth sampling.
        float3 oc = ro - sl.position;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sl.radius * sl.radius;
        float disc = b * b - c;
        if (disc <= 0.0) continue;
        float sq = sqrt(disc);
        float t0 = max(0.0, -b - sq);
        float t1 = min(tEnd, -b + sq);
        if (t1 <= t0) continue;

        // Concentrate samples where the ray actually crosses the cone: the
        // range sphere is tens of metres but the cone is thin, so sampling
        // the whole sphere interval wastes almost every sample and the few
        // hits read as speckle. March a window around the ray's closest
        // approach to the beam AXIS, sized by the cone's local radius; when
        // the ray runs nearly parallel to the axis (looking down the beam)
        // the window degenerates to the full interval, which is correct.
        float rdd = dot(rd, sl.direction);
        float sin2 = 1.0 - rdd * rdd;                    // sin² of ray/axis angle
        float bAxis = dot(oc, rd) - rdd * dot(oc, sl.direction);
        float tc = (sin2 > 1e-4) ? (-bAxis / sin2) : 0.5 * (t0 + t1);
        float axialDist = max(0.0, dot(oc + rd * tc, sl.direction));
        float tanOuter = sqrt(max(0.0, 1.0 - sl.outerCone * sl.outerCone))
                         / max(0.05, sl.outerCone);
        float coneRadius = axialDist * tanOuter;
        float halfW = (sin2 > 1e-4)
            ? (2.0 * coneRadius / sqrt(sin2) + 0.75)
            : (t1 - t0);
        float s0 = max(t0, tc - halfW);
        float s1 = min(t1, tc + halfW);
        if (s1 <= s0) continue;                          // never comes near the cone
        t0 = s0;
        t1 = s1;

        float seg = t1 - t0;
        float dtStep = seg / float(N);
        float3 inscatter = float3(0.0);
        for (uint i = 0; i < N; ++i) {
            seed = pcgHash(seed + i);
            float jitter = float(seed) * (1.0 / 4294967296.0);
            float t = t0 + (float(i) + jitter) * dtStep;
            float3 P = ro + rd * t;
            float3 L = P - sl.position;          // light propagation at P
            float distL = length(L);
            if (distL < 1e-4) continue;
            float3 Ln = L / distL;
            // Soft cone edge: full inside innerCone, zero outside outerCone.
            float cone = smoothstep(sl.outerCone, sl.innerCone, dot(Ln, sl.direction));
            if (cone <= 0.0) continue;
            // Softened inverse-square: physically 1/d², but that kills a
            // beam a few metres out; stage haze reads better when the shaft
            // carries. The range-sphere fade below still ends it cleanly.
            float atten = 1.0 / (0.6 + 0.3 * distL * distL);
            float rangeFade = 1.0 - (distL / sl.radius) * (distL / sl.radius);
            rangeFade = rangeFade * rangeFade;
            // HG phase: light propagates along Ln and scatters toward the
            // camera (-rd), so forward scatter (g > 0) peaks when the camera
            // looks INTO the beam — same convention as the sun pass.
            float cosT = -dot(Ln, rd);
            float hg = (1.0 - g * g) /
                       (4.0 * M_PI_F * pow(max(1e-3, 1.0 + g * g - 2.0 * g * cosT), 1.5));
            float phase = mix(1.0 / (4.0 * M_PI_F), hg, 0.85);
            inscatter += sl.color * (cone * atten * rangeFade * phase);
        }
        accum += inscatter * dtStep;
    }

    accum *= u.density * u.strength;
    if ((accum.x + accum.y + accum.z) <= 0.0) return;
    half4 prev = outHDR.read(gid);
    outHDR.write(half4(prev.rgb + half3(accum), prev.a), gid);
}
