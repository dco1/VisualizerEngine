#pragma once

// ── ILLUMINATORAMA — THE ANALYTIC NIGHT SKY, IN ONE PLACE ────────────────────
//
// Stars and the moon are drawn ANALYTICALLY at screen resolution rather than
// baked into the equirect sky dome. The dome is 2048x1024: one of its texels
// spans several screen pixels, so a baked star is bilinearly magnified into a
// soft blob and the moon loses its limb. Hosts therefore set
// `Params.celestialsInDome = false` and let the shading paths evaluate the
// celestials per ray direction instead — pixel-sharp stars, a crisp moon disk
// with a geometrically-correct terminator.
//
// WHY THIS IS A SHARED HEADER. These functions used to be private to
// Illuminatorama.metal's deferred sky branch, which only ever runs for a PRIMARY
// ray that missed all geometry. Every other way of seeing the sky — a refraction
// through a window pane, a reflection off glass or a glossy floor — reaches it
// through `sampleSky` in IlluminatoramaSecondary.h, which sampled the dome and
// nothing else. So the dome's celestials were off (correctly) and the analytic
// ones were unreachable (incorrectly): standing in a room at night, the sky
// above the roofline was full of stars and the sky through the window was flat
// black. One definition, called from both, is the fix.
//
// `NightSkyParams` is the currency both callers pass; a default-constructed
// (all-zero) value makes `nightCelestials` return exactly float3(0), so every
// scene that never opts in is byte-identical.
//
// The star grid (cell layout, fill rate, magnitude/colour hashing) mirrors
// `starField` in VolumetricSky.metal so the dome and the analytic path agree on
// WHERE the stars are — only the point-spread differs (dome texel vs screen pixel).

#include <metal_stdlib>
using namespace metal;

/// Everything the analytic celestials need, in the order the host packs it into
/// `FrameUniforms.nightSkyParams` / `nightMoonDir` / `nightSunDir`.
struct NightSkyParams {
    float3 moonDir;        // unit, toward the moon
    float3 toSun;          // unit, toward the (below-horizon) sun — sets the phase
    float  starBrightness; // 0 = no stars
    float  moonIntensity;  // 0 = no moon
    float  moonAngRadius;  // radians
};

/// Zeroed params — the exact no-op every non-night scene gets.
static inline NightSkyParams nightSkyOff() {
    NightSkyParams p;
    p.moonDir = float3(0, 1, 0); p.toSun = float3(0, -1, 0);
    p.starBrightness = 0; p.moonIntensity = 0; p.moonAngRadius = 0;
    return p;
}

static inline uint nightHash3(int3 p) {
    uint h = uint(p.x * 374761393 + p.y * 668265263 + p.z * 1274126177);
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

// 2D Worley (cellular) distance for the moon's maria/crater shading — the
// disk-local sibling of VolumetricSky.metal's worley3.
static inline float nightWorley2(float2 p) {
    int2 ip = int2(floor(p));
    float2 fp = p - float2(ip);
    float minD2 = 1e9f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 cell = ip + int2(dx, dy);
            uint h = nightHash3(int3(cell.x, cell.y, 91));
            float2 jitter = float2(float(h & 0xFFu), float((h >> 8u) & 0xFFu)) * (1.0f / 255.0f);
            float2 d = (float2(dx, dy) + jitter) - fp;
            minD2 = min(minD2, dot(d, d));
        }
    }
    return clamp(sqrt(minD2), 0.0f, 1.0f);
}

// Procedural star field, evaluated at screen resolution. `pixAngle` is the
// angular size of one output pixel (radians); each star is a gaussian point
// ~1.5 px wide, so under TAA it resolves as a crisp spark instead of the
// dome-bake's magnified blob. Same hash → same sky as the dome version.
static inline float3 nightStarField(float3 rayDir, float brightness, float pixAngle) {
    if (brightness <= 0.0f) return float3(0.0f);

    float az = atan2(rayDir.z, rayDir.x);
    float el = asin(clamp(rayDir.y, -1.0f, 1.0f));
    const float cellsPerTurn = 450.0f;             // 0.8° cells (matches the dome grid)
    float2 uv = float2((az + M_PI_F) * (cellsPerTurn / (2.0f * M_PI_F)),
                       (el + M_PI_F * 0.5f) * (cellsPerTurn * 0.5f / M_PI_F));
    int2 ip = int2(floor(uv));
    float2 fp = uv - float2(ip);

    // One screen pixel in cell units (elevation cells are constant on the sphere).
    float cellAngle = (2.0f * M_PI_F) / cellsPerTurn;
    float pixCell = max(pixAngle / cellAngle, 1e-4f);
    // Point-spread: σ ≈ 0.75 px — a star's core lands on 1–2 pixels. The peak is
    // resolution-independent (a star is "a bright point", not a patch of sky).
    float sigma = 0.75f * pixCell;
    float invS2 = 1.0f / (2.0f * sigma * sigma);

    float3 result = float3(0.0f);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 cell = ip + int2(dx, dy);
            uint h = nightHash3(int3(cell.x, cell.y, 17));
            // ~4% of cells hold a star (matches the dome field).
            if ((h & 0xFFu) < 10u) {
                float2 starPos = float2(
                    float((h >> 8u)  & 0xFFu) / 255.0f,
                    float((h >> 16u) & 0xFFu) / 255.0f
                );
                float2 d  = fp - (float2(dx, dy) + starPos);
                float  d2 = dot(d, d);
                float  lum = exp(-d2 * invS2);
                // Magnitude: 4 bits → 0..1; brighter is rarer. Dim stars fade
                // fast so the field reads as sparkle over black, not noise.
                float  mag = 1.0f - float((h >> 28u) & 0xFu) / 15.0f;
                float3 col = mix(float3(1.0f, 0.92f, 0.72f),
                                 float3(0.78f, 0.87f, 1.0f), mag * mag);
                result += col * lum * (0.10f + 0.90f * mag * mag);
            }
        }
    }
    return result * brightness * 2.2f;
}

// Moon disk with a geometrically-correct phase terminator. Each disk pixel
// reconstructs the sphere normal at that point and lights it with the TRUE sun
// direction — so the phase (crescent → gibbous → full) and its orientation come
// straight from the real ephemeris, not a hand-tuned phase scalar. `angRadius`
// is the disk's angular radius in radians (real moon ≈ 0.0047; the default is
// modestly enlarged for a photographic read). `toSun` points from the scene
// toward the sun (below the horizon at night — exactly why the lit limb faces
// the sunset). Faint earthshine keeps the dark limb readable on a new-ish moon.
static inline float3 nightMoonDisk(float3 rayDir, float3 moonDir, float3 toSun,
                                   float angRadius, float intensity, float pixAngle) {
    if (intensity <= 0.0f || angRadius <= 0.0f) return float3(0.0f);
    float cosT = dot(rayDir, moonDir);
    if (cosT <= 0.0f) return float3(0.0f);

    // Disk-local frame + position in units of the angular radius.
    float3 upRef = fabs(moonDir.y) < 0.98f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(upRef, moonDir));
    float3 B = cross(moonDir, T);
    float  sinR = sin(angRadius);
    float2 q = float2(dot(rayDir, T), dot(rayDir, B)) / sinR;
    float  r2 = dot(q, q);
    float  r  = sqrt(r2);

    // Anti-aliased edge: one-pixel soft limb.
    float aaW = max(pixAngle / angRadius, 1e-3f);
    float disk = 1.0f - smoothstep(1.0f - aaW, 1.0f + aaW, r);
    if (disk <= 0.0f) return float3(0.0f);

    // Sphere normal at the visible point (the hemisphere facing the viewer).
    float nz = sqrt(max(0.0f, 1.0f - min(r2, 1.0f)));
    float3 n = q.x * T + q.y * B - nz * moonDir;

    // Geometric terminator (sun at infinity — parallax is negligible), softened a
    // touch: the regolith limb is not a hard lambert edge at this scale.
    float lit = max(0.0f, dot(n, normalize(toSun)));
    float shade = pow(lit, 0.75f);
    float earthshine = 0.03f;

    // Maria + crater speckle in DISK-LOCAL coords, so the moon always shows the
    // same face and the pattern doesn't swim as it crosses the sky.
    float mare   = 1.0f - 0.18f * nightWorley2(q * 2.5f + float2(3.7f, 1.3f));
    float crater = 1.0f - 0.08f * nightWorley2(q * 8.0f + float2(9.1f, 4.6f));

    float3 col = float3(0.92f, 0.93f, 1.0f) * (shade + earthshine) * mare * crater;
    return col * intensity * disk;
}

/// Stars + moon for one ray direction — the ONE composite both the primary sky
/// branch and every secondary ray call. Returns float3(0) when the params are
/// zeroed, and below the horizon (the celestials do not shine up out of the
/// ground; a ray refracted downward through a pane sees the yard, not the sky).
/// `pixAngle` is the angular size of one output pixel in radians — it sets the
/// star point-spread and the moon's limb anti-aliasing.
static inline float3 nightCelestials(float3 rayDir, NightSkyParams p, float pixAngle) {
    if (p.starBrightness <= 0.0f && p.moonIntensity <= 0.0f) return float3(0.0f);
    if (rayDir.y <= -0.05f) return float3(0.0f);
    return nightStarField(rayDir, p.starBrightness, pixAngle)
         + nightMoonDisk(rayDir, normalize(p.moonDir), p.toSun,
                         p.moonAngRadius, p.moonIntensity, pixAngle);
}
