#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA DUST ──────────────────────────────────────────────────────
//
// GPU dust-mote simulation for the Illuminatorama Room (issue #28). Motes drift
// on a divergence-free curl-noise flow — believable turbulent air currents, not
// straight-line velocities — recycle within a bounding box, and brighten sharply
// when they cross the sunbeam. The "in the beam" test is NOT faked with a
// scrolling texture: it's a real hard-shadow test against the window opening
// (trace the mote back toward the sun; if it would exit through the window
// rectangle, the sun reaches it). The renderer draws the output position/colour
// buffers as additive HDR point sprites, so bloom turns lit motes into the
// glowing flecks you see in a real sunlit room.

struct DustUniforms {
    float dt; float time; uint count; float curlScale;
    float3 boundsMin; float drift;
    float3 boundsMax; float windowX;     // +X window plane
    float3 sunDir;    float shaftGlow;    // sunDir = toward the sun
    float3 sunColor;  float baseBrightness;
    float winY0; float winY1; float winZ0; float winZ1;
    float feather; float skyFill; float twinkleAmt; float outdoor;  // outdoor>0.5 ⇒ no window, dappled sun
};

// ── value noise + curl ───────────────────────────────────────────────────────

static inline float hash13(float3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

static inline float vnoise(float3 x) {
    float3 i = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i + float3(0,0,0));
    float n100 = hash13(i + float3(1,0,0));
    float n010 = hash13(i + float3(0,1,0));
    float n110 = hash13(i + float3(1,1,0));
    float n001 = hash13(i + float3(0,0,1));
    float n101 = hash13(i + float3(1,0,1));
    float n011 = hash13(i + float3(0,1,1));
    float n111 = hash13(i + float3(1,1,1));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z) * 2.0 - 1.0;
}

static inline float3 noiseVec3(float3 x) {
    return float3(vnoise(x),
                  vnoise(x + float3(31.4, 17.2, 47.1)),
                  vnoise(x + float3(-19.3, 53.7, 11.9)));
}

static inline float3 curlNoise(float3 p) {
    const float e = 0.12;
    float3 dx = float3(e, 0, 0), dy = float3(0, e, 0), dz = float3(0, 0, e);
    float3 px0 = noiseVec3(p - dx), px1 = noiseVec3(p + dx);
    float3 py0 = noiseVec3(p - dy), py1 = noiseVec3(p + dy);
    float3 pz0 = noiseVec3(p - dz), pz1 = noiseVec3(p + dz);
    float x = (py1.z - py0.z) - (pz1.y - pz0.y);
    float y = (pz1.x - pz0.x) - (px1.z - px0.z);
    float z = (px1.y - px0.y) - (py1.x - py0.x);
    return float3(x, y, z) / (2.0 * e);
}

static inline float wrap(float v, float lo, float hi) {
    float range = hi - lo;
    if (range <= 1e-5) return v;
    float t = (v - lo) / range;
    t = t - floor(t);
    return lo + t * range;
}

kernel void illumi_dust_step(
    device float4*          positions [[buffer(0)]],
    device float4*          colors    [[buffer(1)]],
    constant DustUniforms&  u         [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.count) return;
    float3 p = positions[gid].xyz;
    float seed = positions[gid].w;     // per-mote phase, set at seed time

    // Curl-noise drift + a gentle settling.
    // The time term churns the field fast enough (0.40) that it does NOT read as
    // a quasi-static flow. A near-static curl field has permanent stagnation
    // manifolds that forward-Euler advection collapses motes onto — measured as
    // runaway clustering (Var/Mean 1→236 over 60 s, dust-physics-probe.swift).
    // Fast churn keeps the divergence-free field's no-clumping property intact,
    // so motes stay uniformly suspended instead of draining into filaments.
    float3 flow = curlNoise(p * u.curlScale + float3(0.0, u.time * 0.40, 0.0));
    flow.y -= 0.02;                    // gentle Stokes-like settling (~0.1 cm/s at drift 0.05)
    p += flow * u.drift * u.dt;

    // Recycle within the room box.
    p.x = wrap(p.x, u.boundsMin.x, u.boundsMax.x);
    p.y = wrap(p.y, u.boundsMin.y, u.boundsMax.y);
    p.z = wrap(p.z, u.boundsMin.z, u.boundsMax.z);
    positions[gid] = float4(p, seed);

    // ── Light reactivity: real window shadow test ────────────────
    // Trace from the mote toward the sun; if it crosses the +X window plane
    // inside the opening rectangle, direct sun reaches the mote.
    float shaft = 0.0;
    if (u.outdoor > 0.5) {
        // Outdoor (forest) mode: no window plane. The low golden sun reaches
        // motes through CANOPY GAPS — approximated by a slow, drifting, low-freq
        // spatial dapple so motes brighten as they float through light patches
        // and dim in the shade between, instead of a hard window-rectangle test.
        // (An honest stand-in for per-mote canopy occlusion without the shadow
        // map: motes still react to "where the light is," just via a soft field.)
        float dap = vnoise(p * 0.11 + float3(0.0, 0.0, u.time * 0.02)) * 0.5 + 0.5;
        shaft = smoothstep(0.30, 0.80, dap);
    } else if (u.sunDir.x > 1e-3) {
        float s = (u.windowX - p.x) / u.sunDir.x;
        if (s > 0.0) {
            float yc = p.y + s * u.sunDir.y;
            float zc = p.z + s * u.sunDir.z;
            float fy = smoothstep(u.winY0 - u.feather, u.winY0 + u.feather, yc) *
                       (1.0 - smoothstep(u.winY1 - u.feather, u.winY1 + u.feather, yc));
            float fz = smoothstep(u.winZ0 - u.feather, u.winZ0 + u.feather, zc) *
                       (1.0 - smoothstep(u.winZ1 - u.feather, u.winZ1 + u.feather, zc));
            shaft = fy * fz;
        }
    }

    // Per-mote variation + slow twinkle so the field reads alive.
    float tw = 1.0 + u.twinkleAmt * sin(u.time * 1.7 + seed * 6.2831);
    float variance = 0.6 + 0.8 * hash13(float3(seed * 91.7, 3.1, 7.7));

    float3 ambientCol = float3(0.42, 0.45, 0.52) * u.baseBrightness * u.skyFill;
    float3 shaftCol = u.sunColor * (u.shaftGlow * shaft);
    colors[gid] = float4((ambientCol + shaftCol) * (variance * tw), 1.0);
}
