#pragma once

// ── THE NIGHT SKY, IN ONE PLACE ───────────────────────────────────────────────
//
// Every way the engine draws the night sky calls THIS header: the equirect sky
// dome + IBL bake (`volSkyRender`), the in-view cloud march and its full-res
// upsample (`illumi_cloud_inview` / `illumi_cloud_upsample`) in VolumetricSky.metal,
// Illuminatorama's deferred sky branch, and every secondary ray (glass refraction /
// reflection, TLAS reflections — `sampleSky` in IlluminatoramaSecondary.h). One
// definition, so a window, a reflection, the dome and the open sky all show the same
// stars and the same moon.
//
// TWO MODELS, selected by `NightSkyParams.model`:
//
//   0 = LEGACY (the default, and what an all-zero param block means). Byte-identical to
//       the pre-2026-09-23 analytic sky: a 0.8° lat-long hashed star grid (`nightStarField`)
//       and a sphere-lit moon disk (`nightMoonDisk`). Every host that never opts in —
//       Daydream Home's house renders, every Visualizer scene but Superbloom Hills — keeps
//       exactly the pixels it had.
//
//   1 = PHYSICAL (opt-in). A night sky built from the real quantities rather than tuned
//       sprinkles:
//         • STARS on an EQUAL-AREA placement: Poisson points in a 3D lattice, kept inside a
//           thin spherical shell and projected radially. A uniform 3D density in a
//           constant-thickness shell projects to a uniform density per steradian — no
//           poles, no seams, no lat-long cells shrinking toward the zenith (the old grid's
//           density ∝ 1/cos(el) — the "clump at the zenith").
//         • A real MAGNITUDE DISTRIBUTION: cumulative counts ∝ 10^(0.5·m) (≈3.2× more stars
//           per magnitude fainter; ~23k stars to m ≈ 7.3 over the whole sky, ~20 brighter
//           than m = 1). Brightness is FLUX, not footprint: every star is the same ~1.5 px
//           point-spread carrying flux 10^(-0.4·m), so bright stars exceed 1.0 and bloom.
//         • BLACKBODY colours from a spectral-class-weighted temperature, desaturated to
//           what an eye / camera reads.
//         • AIR-MASS EXTINCTION (Kasten–Young air mass × per-channel extinction
//           coefficients): stars dim and redden toward the horizon — the same physics the
//           host already applies to the sun.
//         • SCINTILLATION: smooth (few-Hz) per-star intensity flicker whose amplitude grows
//           with air mass. Smooth in time so TAA tracks it instead of smearing a per-frame
//           recolour (see the TAA-smears-per-frame-recolour note in known issues).
//         • WASHOUT: a star is only visible against the sky BEHIND it — its peak radiance
//           over the local background radiance sets its visibility, so moonlight, twilight
//           and the lunar aureole hide the faint stars and keep the bright ones, with no
//           separate "moon on ⇒ fewer stars" switch.
//         • The MILKY WAY: the galactic plane in real J2000 coordinates, brighter and wider
//           toward the galactic centre (the bulge in Sagittarius), star-cloud structure,
//           and the Great Rift / dust lanes as absorption.
//         • The MOON at its true 0.26° angular radius (host-overridable), lit per pixel by
//           the real sun direction on a sphere (the terminator is the ellipse a lit sphere
//           shows, not a pie-slice), Lommel–Seeliger regolith reflectance (the flat full
//           moon, with only a mild darkening at the very limb), a procedural albedo map of
//           the real near side (the maria in their selenographic positions, Tycho and
//           Copernicus with rays, crater speckle) oriented by the celestial pole so it tilts
//           as it crosses the sky, earthshine on the night side (∝ the Earth's own phase as
//           seen from the Moon), and a Mie aureole around it that brightens with air mass.
//       Night radiance is in ONE physical unit (`radiance` = the scene radiance of a
//       magnitude-0 flux spread over one steradian), so stars, Milky Way, airglow and the
//       moonlit sky (VolumetricSky.metal) stay in their real proportions to each other; the
//       moon DISK keeps its own host gain because its true contrast (~10⁵–10⁶ × the sky
//       around it) cannot be displayed.
//
// The equatorial frame. Stars, the Milky Way and the moon's orientation live on the
// celestial sphere; `celestial` is the quaternion taking a WORLD direction into J2000
// equatorial coordinates (x → RA 0h, z → north celestial pole). The host builds it from
// latitude + local sidereal time (`NightSkyEphemeris` in Swift), so the sky turns about
// the pole as the hours pass. All-zero = identity.

#include <metal_stdlib>
using namespace metal;

/// Everything the celestials need. Built by the ONE packer each uniform block has
/// (`frameNightSky`, `glassNightSky`, `skyNightParams`) — never field-by-field at a call site.
struct NightSkyParams {
    float3 moonDir;        // unit, toward the moon
    float3 toSun;          // unit, toward the sun as seen from the moon — sets the phase
    float  starBrightness; // 0 = no stars
    float  moonIntensity;  // 0 = no moon
    float  moonAngRadius;  // radians (0 in the physical model ⇒ the real 0.00452)
    // ── physical model (all zero ⇒ legacy) ──────────────────────────────────
    float  model;          // 0 = legacy, 1 = physical
    float  moonHalo;       // aureole gain (1 = clear air); hosts raise it for haze / thin cloud
    float  earthshine;     // earthshine gain (1 = default)
    float  milkyWay;       // Milky Way gain (1 = physical surface brightness)
    float  twinkle;        // scintillation gain (1 = default, 0 = frozen)
    float  clock;          // seconds — drives scintillation
    float  radiance;       // scene radiance of a magnitude-0 flux over 1 sr
    float4 celestial;      // world → equatorial quaternion (xyz imaginary, w real); 0 ⇒ identity
};

/// Zeroed params — the exact no-op every non-night scene gets.
static inline NightSkyParams nightSkyOff() {
    NightSkyParams p;
    p.moonDir = float3(0, 1, 0); p.toSun = float3(0, -1, 0);
    p.starBrightness = 0; p.moonIntensity = 0; p.moonAngRadius = 0;
    p.model = 0; p.moonHalo = 0; p.earthshine = 0; p.milkyWay = 0;
    p.twinkle = 0; p.clock = 0; p.radiance = 0; p.celestial = float4(0);
    return p;
}

// ═════════════════════════════════════════════════════════════════════════════
// LEGACY MODEL (model = 0) — unchanged, byte-identical.
// ═════════════════════════════════════════════════════════════════════════════
//
// The star grid (cell layout, fill rate, magnitude/colour hashing) mirrors the dome's
// legacy `starField` in VolumetricSky.metal so the two agree on WHERE the stars are.

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


// ═════════════════════════════════════════════════════════════════════════════
// PHYSICAL MODEL (model = 1)
// ═════════════════════════════════════════════════════════════════════════════

constant float3 kNightLuma = float3(0.2126f, 0.7152f, 0.0722f);
constant float  kNightLog2Of10 = 3.3219281f;

// ── Star lattice ────────────────────────────────────────────────────────────
// Stars are Poisson points in a 3D integer lattice, kept only inside the shell
// |p| ∈ [R − h, R + h] and projected radially onto the sky. A uniform volume density
// in a shell of constant thickness is a uniform density per steradian — the
// placement is exactly equal-area, with no poles and no seams.
constant float kStarShellR     = 48.0f;    // shell radius in lattice cells (a cell ≈ 1.19°)
constant float kStarShellHalf  = 0.4f;     // shell half-thickness (cells)
constant float kStarLambda     = 1.0f;     // mean stars per lattice cell (before the galactic boost)
// Stars per steradian = λ·R²·2h ≈ 1843 ⇒ ~23 160 on the whole sky. Cumulative counts
// N(<m) ∝ 10^(0.5·m) (≈3.2× per magnitude, the naked-eye slope) — anchored so N(<6) ≈ 5000
// as on the real sky ⇒ the faintest hashed star is m ≈ 7.3. Brightest clamped at Sirius.
constant float kStarMagLimit   = 7.3f;
constant float kStarMagBright  = -1.46f;
// Mean star density boost toward the galactic plane — 1 + 2·exp(−(b/12°)²), divided
// by its sky average (1.368) so the TOTAL count is unchanged. Real faint-star counts
// rise ~2–5× into the plane; this is the resolved-star half of the Milky Way.
constant float kStarGalBoost   = 2.0f;
constant float kStarGalWidth   = 0.21f;    // rad (12°)
constant float kStarGalNorm    = 1.0f / 1.368f;

// Galactic frame in J2000 equatorial coordinates (IAU 1958 definition).
constant float3 kGalNorth  = float3(-0.86768f, -0.19804f,  0.45598f);  // NGP  (RA 192.86°, Dec +27.13°)
constant float3 kGalCentre = float3(-0.05487f, -0.87344f, -0.48387f);  // l=0 (RA 266.40°, Dec −28.94°)
constant float3 kGalL90    = float3( 0.49410f, -0.44486f,  0.74700f);  // l=90° (Cygnus)

// Physical surface brightnesses, in magnitude-0 fluxes per steradian ("F0/sr" — the
// unit `NightSkyParams.radiance` converts to scene radiance). mag/arcsec² → F0/sr:
// 10^(−0.4·(μ − 26.57)).
constant float kMilkyWayPeakF0  = 270.0f;  // ~20.5 mag/arcsec² — the brightest star clouds
constant float kDarkSkyF0       = 81.0f;   // ~21.8 mag/arcsec² — natural dark zenith (airglow)
constant float kFullMoonSkyF0   = 2690.0f; // ~18.0 mag/arcsec² — zenith under a full moon

// ── Hashing / noise ─────────────────────────────────────────────────────────

static inline uint nightPCG(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
/// Uniform in the OPEN interval (0, 1) from the top 24 bits.
static inline float nightU01(uint h) { return (float(h >> 8u) + 0.5f) * (1.0f / 16777216.0f); }

/// Lattice-cell hash (three chained PCG rounds — the star field reads several
/// independent numbers per cell, so it needs more than the cheap mix below).
static inline uint nightCellHash(int3 c) {
    uint h = nightPCG(uint(c.x) + 0x9E3779B9u);
    h = nightPCG(h ^ uint(c.y));
    return nightPCG(h ^ uint(c.z));
}

// Cheap single-round mix for value noise (the same mix as `nightHash3`, in uint).
static inline float nightNoiseHash(int3 c) {
    uint h = uint(c.x) * 374761393u + uint(c.y) * 668265263u + uint(c.z) * 1274126177u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= (h >> 16u);
    return float(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

/// Trilinear value noise in [0, 1].
static inline float nightValueNoise3(float3 p) {
    float3 i = floor(p);
    float3 f = p - i;
    float3 u = f * f * (3.0f - 2.0f * f);
    int3 c = int3(i);
    float n000 = nightNoiseHash(c);
    float n100 = nightNoiseHash(c + int3(1, 0, 0));
    float n010 = nightNoiseHash(c + int3(0, 1, 0));
    float n110 = nightNoiseHash(c + int3(1, 1, 0));
    float n001 = nightNoiseHash(c + int3(0, 0, 1));
    float n101 = nightNoiseHash(c + int3(1, 0, 1));
    float n011 = nightNoiseHash(c + int3(0, 1, 1));
    float n111 = nightNoiseHash(c + int3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

/// Fractal value noise, normalised to [0, 1].
static inline float nightFbm3(float3 p, int octaves) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * nightValueNoise3(p);
        norm += amp;
        p = p * 2.03f + float3(17.1f, 5.3f, 11.7f);
        amp *= 0.5f;
    }
    return sum / norm;
}

/// Smooth 1D value noise in [−1, 1] — the scintillation signal.
static inline float nightNoise1(float t) {
    float i = floor(t);
    float f = t - i;
    float a = nightU01(nightPCG(uint(int(i))));
    float b = nightU01(nightPCG(uint(int(i) + 1)));
    return mix(a, b, f * f * (3.0f - 2.0f * f)) * 2.0f - 1.0f;
}

// ── Frames ──────────────────────────────────────────────────────────────────

/// Rotate `v` by the unit quaternion `q` (xyz imaginary, w real). An all-zero `q`
/// (an unset uniform) is the identity.
static inline float3 nightQuatRotate(float4 q, float3 v) {
    if (dot(q, q) < 1e-8f) return v;
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
static inline float3 nightQuatRotateInverse(float4 q, float3 v) {
    return nightQuatRotate(float4(-q.xyz, q.w), v);
}

// ── Atmosphere along the line of sight ──────────────────────────────────────

/// Relative air mass for a line of sight at elevation asin(sinEl) — Kasten & Young
/// (1989), finite at the horizon (≈38). Below the horizon it is held at the horizon value.
static inline float nightAirmass(float sinEl) {
    float elDeg = max(asin(clamp(sinEl, -1.0f, 1.0f)) * 57.29578f, 0.0f);
    return 1.0f / (sin(elDeg * 0.01745329f) + 0.50572f * pow(elDeg + 6.07995f, -1.6364f));
}

/// Direct transmittance of a point source (star, moon) seen at elevation asin(sinEl):
/// 10^(−0.4·k·X) per channel, with clear-site extinction coefficients k ≈ 0.11 / 0.19 /
/// 0.32 mag per air mass in R / G / B (V ≈ 0.19, B ≈ 0.3). Stars fade ~1 magnitude and
/// redden by 10° elevation and are gone into the horizon murk (≈7 magnitudes at X ≈ 38).
static inline float3 nightExtinction(float sinEl) {
    float X = nightAirmass(sinEl);
    return exp2(-0.4f * kNightLog2Of10 * float3(0.11f, 0.19f, 0.32f) * X);
}

// ── Star colour ─────────────────────────────────────────────────────────────

/// Planckian-locus colour of temperature T (K) in LINEAR sRGB, normalised to unit
/// luminance. CIE 1931 xy from Kim et al. (2002)'s cubic fit (1667–25000 K), Y = 1.
static inline float3 nightBlackbody(float T) {
    T = clamp(T, 1667.0f, 25000.0f);
    float t = 1000.0f / T;
    float x = (T < 4000.0f)
        ? (-0.2661239f * t * t * t - 0.2343589f * t * t + 0.8776956f * t + 0.179910f)
        : (-3.0258469f * t * t * t + 2.1070379f * t * t + 0.2226347f * t + 0.240390f);
    float y = (T < 2222.0f) ? (-1.1063814f * x * x * x - 1.34811020f * x * x + 2.18555832f * x - 0.20219683f)
            : (T < 4000.0f) ? (-0.9549476f * x * x * x - 1.37418593f * x * x + 2.09137015f * x - 0.16748867f)
                            : ( 3.0817580f * x * x * x - 5.87338670f * x * x + 3.75112997f * x - 0.37001483f);
    float X = x / y, Z = (1.0f - x - y) / y;
    float3 rgb = float3( 3.2406f * X - 1.5372f - 0.4986f * Z,
                        -0.9689f * X + 1.8758f + 0.0415f * Z,
                         0.0557f * X - 0.2040f + 1.0570f * Z);
    rgb = max(rgb, 0.0f);
    return rgb / max(dot(rgb, kNightLuma), 1e-4f);
}

/// Surface temperature for a naked-eye star from a uniform variate — weighted by the
/// spectral classes that dominate the visible sky (lots of K giants and A/B dwarfs,
/// few M, a handful of hot B), so the field reads mostly white with orange and blue
/// minorities rather than a rainbow.
static inline float nightStarTemperature(float u) {
    if (u < 0.10f) return mix(3000.0f,  3700.0f,  u / 0.10f);             // M
    if (u < 0.45f) return mix(3700.0f,  5200.0f,  (u - 0.10f) / 0.35f);   // K
    if (u < 0.62f) return mix(5200.0f,  6000.0f,  (u - 0.45f) / 0.17f);   // G
    if (u < 0.75f) return mix(6000.0f,  7500.0f,  (u - 0.62f) / 0.13f);   // F
    if (u < 0.92f) return mix(7500.0f, 10000.0f,  (u - 0.75f) / 0.17f);   // A
    return mix(10000.0f, 25000.0f, (u - 0.92f) / 0.08f);                 // B
}

// ── Star field ──────────────────────────────────────────────────────────────

/// Galactic latitude (radians) of an equatorial direction.
static inline float nightGalacticLatitude(float3 dEq) {
    return asin(clamp(dot(dEq, kGalNorth), -1.0f, 1.0f));
}

/// Mean stars in lattice cell `c` — the base rate, raised toward the galactic plane.
static inline float nightStarCellLambda(int3 c) {
    float3 centre = normalize(float3(c) + 0.5f);
    float b = nightGalacticLatitude(centre);
    return kStarLambda * kStarGalNorm * (1.0f + kStarGalBoost * exp(-(b * b) / (kStarGalWidth * kStarGalWidth)));
}

/// One star, as the lattice defines it. `valid` is false when the hashed point
/// lies outside the shell (that sample is discarded — this is what makes the
/// projected density uniform). Shared by the renderer and the GPU unit tests.
struct NightStar {
    float3 dir;     // unit, equatorial
    float  mag;     // visual magnitude
    float  tempK;   // surface temperature
    uint   seed;    // per-star hash for scintillation
    bool   valid;
};

/// Number of stars hashed into cell `c` (Poisson, capped at 6). `h` = nightCellHash(c).
static inline int nightStarCount(int3 c, uint h) {
    float lam = nightStarCellLambda(c);
    float u = nightU01(h);
    float pk = exp(-lam), cdf = pk;
    int k = 0;
    while (u > cdf && k < 6) { k += 1; pk *= lam / float(k); cdf += pk; }
    return k;
}

/// Star `i` of cell `c` (`cellHash` = nightCellHash(c)). Magnitude and colour are only
/// hashed for a star inside the shell — most candidates are discarded before that.
static inline NightStar nightStarInCell(int3 c, uint cellHash, int i) {
    NightStar s;
    uint h = nightPCG(cellHash ^ (0x68E31DA4u * uint(i + 1)));
    uint h1 = nightPCG(h ^ 0x1B873593u);
    uint h2 = nightPCG(h1 ^ 0xCC9E2D51u);
    float3 p = float3(c) + float3(nightU01(h), nightU01(h1), nightU01(h2));
    float r = length(p);
    s.valid = fabs(r - kStarShellR) <= kStarShellHalf;
    s.dir = p / max(r, 1e-6f);
    s.mag = 99.0f; s.tempK = 5800.0f; s.seed = 0u;
    if (!s.valid) return s;
    uint h3 = nightPCG(h2 ^ 0x85EBCA6Bu);
    uint h4 = nightPCG(h3 ^ 0xC2B2AE35u);
    // Inverse CDF of N(<m) ∝ 10^(0.5·m) truncated at kStarMagLimit: m = m_lim + 2·log10(u).
    s.mag = max(kStarMagLimit + 2.0f * log10(nightU01(h3)), kStarMagBright);
    s.tempK = nightStarTemperature(nightU01(h4));
    s.seed = h4;
    return s;
}

/// Mean radiance of the whole star field in F0/sr — what a pixel too coarse to resolve
/// stars (a small IBL bake) should see instead of aliasing points. Σ flux over the
/// hashed population: density·F_lim·E[u^−0.8] = 1843·10^(−0.4·7.3)·5 ≈ 11 F0/sr.
constant float kStarMeanF0 = 11.0f;

/// Physical star field for one view direction, in scene radiance.
///   dEq        view direction in equatorial coordinates
///   sinEl      the view direction's WORLD elevation (air mass)
///   pixAngle   angular size of one output pixel (radians) — the point-spread width
///   gain       scene radiance per F0/sr (`radiance × starBrightness`)
///   bgLum      luminance of the sky BEHIND the star (washout)
static inline float3 nightStarsPhysical(float3 dEq, float sinEl, float pixAngle, float gain,
                                        float bgLum, float twinkle, float clock) {
    if (gain <= 0.0f || sinEl <= 0.0f) return float3(0.0f);
    float3 ext = nightExtinction(sinEl);
    // Point-spread: a gaussian of σ ≈ 0.75 px carries the star's whole flux, so the peak is
    // flux / (2πσ²) — brightness lives in the value (and blooms), not in the footprint.
    float sigma = max(0.75f * pixAngle, 1e-6f);
    float reach = 3.0f * sigma;
    float reachCells = reach * kStarShellR;
    // A pixel wider than ~0.4 cell (a small IBL bake, a coarse secondary ray) can't resolve
    // points — and would need up to 4³ cells: it gets the field's mean radiance instead.
    if (reachCells > 0.4f) return ext * (gain * kStarMeanF0);
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    float peakPerF = gain / (2.0f * M_PI_F * sigma * sigma);
    // Scintillation amplitude: a few % overhead, tens of % within ~10° of the horizon.
    float X = nightAirmass(sinEl);
    float amp = twinkle * (0.05f + 0.35f * saturate((X - 1.0f) / 9.0f));
    float chroma = saturate((X - 3.0f) / 8.0f);   // low stars also flash colours

    // Every lattice cell a star within `reach` of this ray can live in: the ray crosses
    // the shell over ±h·|d| per axis, plus the point-spread reach: 2 cells per axis typically,
    // 3 on the axis the ray runs along (≈8–12 cells a pixel).
    float3 p = dEq * kStarShellR;
    float3 halfExt = kStarShellHalf * fabs(dEq) + reachCells;
    int3 lo = int3(floor(p - halfExt));
    int3 hi = int3(floor(p + halfExt));
    float3 sum = float3(0.0f);
    for (int z = lo.z; z <= hi.z; ++z)
    for (int y = lo.y; y <= hi.y; ++y)
    for (int x = lo.x; x <= hi.x; ++x) {
        int3 c = int3(x, y, z);
        uint ch = nightCellHash(c);
        int n = nightStarCount(c, ch);
        for (int i = 0; i < n; ++i) {
            NightStar s = nightStarInCell(c, ch, i);
            if (!s.valid) continue;
            float3 d = s.dir - dEq;
            float a2 = dot(d, d);                     // chord² ≈ angle² at these scales
            if (a2 > reach * reach) continue;
            float flux = exp2(-0.4f * kNightLog2Of10 * s.mag);
            float peak = peakPerF * flux;
            // Visibility against the local sky: a star below ~4 % contrast is lost, full at 40 %.
            float vis = smoothstep(0.04f, 0.4f, peak * ext.g / max(bgLum, 1e-9f));
            if (vis <= 0.0f) continue;
            float ph = nightU01(s.seed) * 97.0f;
            float fr = 2.5f + 2.0f * nightU01(nightPCG(s.seed));
            float t = clock * fr + ph;
            float n0 = nightNoise1(t);
            float3 tw = 1.0f + amp * mix(float3(n0), float3(n0, nightNoise1(t + 31.7f), nightNoise1(t + 63.1f)), chroma);
            float3 col = mix(float3(1.0f), nightBlackbody(s.tempK), 0.65f);
            sum += col * max(tw, 0.0f) * (peak * vis * exp(-a2 * inv2s2));
        }
    }
    return sum * ext;
}

// ── Milky Way ───────────────────────────────────────────────────────────────

/// Diffuse light of the galactic plane, in F0/sr, for an equatorial direction. The
/// resolved stars come from the lattice's galactic density boost; this is the
/// unresolved glow: a disk ~6–10° thick that brightens and thickens toward the
/// galactic centre, the bulge in Sagittarius, star-cloud structure, and the dust —
/// a patchy absorbing lane along the plane that deepens into the Great Rift between
/// Cygnus and Scorpius (l ≈ +80° … −20°).
static inline float nightMilkyWayBound(float3 dEq) {
    // Upper bound of `nightMilkyWay`'s luminance (clouds = 1, no dust) — lets a caller skip
    // the ten octaves of noise where the band could not be seen anyway.
    float b = nightGalacticLatitude(dEq);
    if (fabs(b) > 0.7f) return 0.0f;
    float l = atan2(dot(dEq, kGalL90), dot(dEq, kGalCentre));
    float inner = exp(-(l * l) / (1.1f * 1.1f));
    float width = 0.10f + 0.07f * inner;
    float disk  = exp(-(b * b) / (width * width)) * (0.28f + 0.72f * inner);
    float bulge = exp(-((l * l) / (0.24f * 0.24f) + (b * b) / (0.16f * 0.16f)));
    return (disk * 1.7f * 1.2f + 0.9f * bulge * 1.3f) * kMilkyWayPeakF0 * 1.05f;
}

static inline float3 nightMilkyWay(float3 dEq) {
    float b = nightGalacticLatitude(dEq);
    if (fabs(b) > 0.7f) return float3(0.0f);   // |b| > 40°: nothing measurable
    float l = atan2(dot(dEq, kGalL90), dot(dEq, kGalCentre));   // (−π, π], 0 at the centre
    float inner = exp(-(l * l) / (1.1f * 1.1f));
    float width = 0.10f + 0.07f * inner;
    float disk  = exp(-(b * b) / (width * width)) * (0.28f + 0.72f * inner);
    float bulge = exp(-((l * l) / (0.24f * 0.24f) + (b * b) / (0.16f * 0.16f)));
    float clouds = nightFbm3(dEq * 7.0f + 11.0f, 4);
    float grain  = nightFbm3(dEq * 38.0f, 2);
    // Star clouds modulate a CONTINUOUS band (the disk is always there; the clouds are its
    // texture), so it reads as the arch across the sky rather than isolated patches.
    float light = disk * (0.55f + 0.9f * clouds) * (0.8f + 0.4f * grain)
                + 0.9f * bulge * (0.7f + 0.6f * clouds);
    float bl = b - 0.012f - 0.02f * sin(l * 2.3f);
    float lane = exp(-(bl * bl) / (0.05f * 0.05f));
    float rift = smoothstep(-0.45f, -0.05f, l) * (1.0f - smoothstep(1.0f, 1.5f, l));
    float dust = nightFbm3(dEq * 14.0f + 5.3f, 4);
    float tau = lane * ((0.35f + 1.8f * rift) * smoothstep(0.35f, 0.7f, dust) * 2.2f + 0.25f);
    float3 col = mix(float3(0.80f, 0.86f, 1.0f), float3(1.0f, 0.86f, 0.66f), saturate(bulge * 1.5f + inner * 0.3f));
    return col * (light * exp(-tau) * kMilkyWayPeakF0);
}

// ── The Moon ────────────────────────────────────────────────────────────────

/// Fraction of full-moon flux at phase angle α (cos α = toSun·(−moonDir)): Allen's
/// lunar phase law, m(α) = 0.026|α| + 4·10⁻⁹α⁴ (α in degrees) — a half moon is ~9 %
/// of a full one, not 50 % (the regolith's opposition surge).
static inline float nightMoonPhaseFlux(float cosAlpha) {
    float a = acos(clamp(cosAlpha, -1.0f, 1.0f)) * 57.29578f;
    float m = 0.026f * a + 4.0e-9f * a * a * a * a;
    return exp2(-0.4f * kNightLog2Of10 * m);
}

static inline float3 nightSelenographic(float latDeg, float lonDeg) {
    float la = latDeg * 0.01745329f, lo = lonDeg * 0.01745329f;
    return float3(cos(la) * sin(lo), sin(la), cos(la) * cos(lo));
}

// Near-side maria: selenographic (lat°, lon° — east positive), angular radius°, darkness.
// Big, irregular Oceanus Procellarum is four overlapping caps.
constant float4 kMoonMaria[] = {
    float4( 32.8f, -15.6f, 16.0f, 1.00f),   // Imbrium
    float4( 40.0f, -52.0f,  9.0f, 0.85f),   // Procellarum (N)
    float4( 25.0f, -60.0f, 11.0f, 0.90f),   // Procellarum
    float4( 10.0f, -57.0f, 11.0f, 0.90f),   // Procellarum
    float4( -3.0f, -48.0f,  9.0f, 0.80f),   // Procellarum (S)
    float4( 28.0f,  17.5f, 10.5f, 0.88f),   // Serenitatis
    float4(  8.5f,  31.4f, 12.0f, 1.00f),   // Tranquillitatis
    float4( 17.0f,  59.1f,  8.0f, 1.00f),   // Crisium
    float4( -7.8f,  51.3f, 10.0f, 0.85f),   // Fecunditatis
    float4(-15.2f,  35.5f,  5.5f, 0.85f),   // Nectaris
    float4(-21.3f, -16.6f, 11.0f, 0.80f),   // Nubium
    float4(-24.4f, -38.6f,  6.5f, 0.90f),   // Humorum
    float4(-10.0f, -23.0f,  5.5f, 0.75f),   // Cognitum
    float4(  7.5f, -30.9f,  7.5f, 0.70f),   // Insularum
    float4( 13.3f,   3.6f,  4.5f, 0.80f),   // Vaporum
    float4( 10.9f,  -8.8f,  3.5f, 0.70f),   // Sinus Aestuum
    float4( 56.0f, -10.0f,  6.0f, 0.70f),   // Frigoris
    float4( 57.0f,  15.0f,  5.0f, 0.65f),   // Frigoris (E)
    float4( 54.0f, -35.0f,  5.0f, 0.65f),   // Frigoris (W)
    float4(  1.3f,  87.0f,  6.0f, 0.70f),   // Smythii (limb)
    float4( 13.0f,  86.0f,  5.0f, 0.60f),   // Marginis (limb)
};
constant int kMoonMariaCount = 21;

// Bright young ray craters: (lat°, lon°, crater radius°, ray length°).
constant float4 kMoonRayCraters[] = {
    float4(-43.3f, -11.2f, 1.4f, 45.0f),   // Tycho
    float4(  9.6f, -20.1f, 1.6f, 22.0f),   // Copernicus
    float4(  8.1f, -38.0f, 0.9f, 14.0f),   // Kepler
    float4( 23.7f, -47.4f, 0.8f,  8.0f),   // Aristarchus (the brightest spot on the moon)
    float4( 16.1f,  46.8f, 0.6f, 10.0f),   // Proclus
};
constant int kMoonRayCraterCount = 5;

/// Normal albedo of the near side at selenographic unit vector `s` (x = east, y = north,
/// z = toward the Earth), normalised so the bright highlands are ~1: dark maria at ~0.5,
/// crater rims and ejecta rays brighter. Evaluated only on moon pixels.
static inline float nightMoonAlbedo(float3 s) {
    // Maria: lava plains that flooded old basins — the basins are round, the flooding is
    // not. Shorelines are pushed around by two octaves of warp (a large lobe + a ragged
    // edge) and fade over a fifth of the radius, so no mare reads as a stamped disc.
    float warp = (nightFbm3(s * 4.0f + 2.7f, 3) - 0.5f) + 0.45f * (nightFbm3(s * 13.0f + 8.1f, 2) - 0.5f);
    float mare = 0.0f;
    for (int i = 0; i < kMoonMariaCount; ++i) {
        float4 m = kMoonMaria[i];
        float d = acos(clamp(dot(s, nightSelenographic(m.x, m.y)), -1.0f, 1.0f)) * 57.29578f;
        float edge = d + warp * m.z * 1.1f;
        float soft = 0.2f * m.z + 1.0f;
        mare = max(mare, m.w * (1.0f - smoothstep(m.z - soft, m.z + soft, edge)));
    }
    float mottle = nightFbm3(s * 22.0f, 3);
    float flow = nightFbm3(s * 9.0f + 4.4f, 2);   // lava-flow tone differences within a mare
    float albedo = mix(0.90f + 0.18f * mottle, 0.40f + 0.12f * flow + 0.06f * mottle, mare);

    // Crater field: rims brighter, floors a touch darker; sparser on the smooth maria.
    float craters = 0.0f;
    for (int scale = 0; scale < 2; ++scale) {
        float fs = scale == 0 ? 16.0f : 40.0f;
        float3 q = s * fs;
        int3 ic = int3(floor(q));
        for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            int3 c = ic + int3(x, y, z) + int3(scale * 101);
            uint h = nightCellHash(c);
            float3 cp = float3(ic + int3(x, y, z)) + float3(nightU01(h), nightU01(nightPCG(h)), nightU01(nightPCG(h ^ 7u)));
            float rc = 0.12f + 0.3f * nightU01(nightPCG(h ^ 13u));
            float d = length(q - cp);
            float rim = exp(-((d - rc) * (d - rc)) / (0.02f * rc * rc + 1e-4f));
            float flr = (d < rc) ? (1.0f - d / rc) : 0.0f;
            craters += 0.07f * rim - 0.015f * flr;
        }
    }
    albedo *= 1.0f + craters * (1.0f - 0.6f * mare);

    // Ray craters: a bright halo and radial ejecta streaks.
    for (int i = 0; i < kMoonRayCraterCount; ++i) {
        float4 rc = kMoonRayCraters[i];
        float3 c = nightSelenographic(rc.x, rc.y);
        float d = acos(clamp(dot(s, c), -1.0f, 1.0f)) * 57.29578f;
        if (d > rc.w) continue;
        float3 e1 = normalize(cross(float3(0.0f, 1.0f, 0.0f), c));
        float3 e2 = cross(c, e1);
        float phi = atan2(dot(s, e2), dot(s, e1));
        float rays = nightValueNoise3(float3(cos(phi), sin(phi), float(i)) * 9.0f);
        rays = pow(saturate((rays - 0.45f) / 0.55f), 3.0f) * exp(-d / (0.35f * rc.w))
             * smoothstep(rc.z, rc.z * 2.0f, d);
        float spot = exp(-(d * d) / (rc.z * rc.z * 1.5f));
        albedo += 0.55f * rays + 0.7f * spot;
    }
    return albedo;
}

/// The physical moon disk: rgb = radiance, a = the disk's coverage (0…1) of this
/// pixel — stars and the Milky Way BEHIND the moon are hidden by it.
///   north   the celestial north pole in world space (orients the near side).
static inline float4 nightMoonPhysical(float3 rayDir, NightSkyParams p, float3 north, float pixAngle) {
    float3 f = normalize(p.moonDir);
    float angR = p.moonAngRadius > 0.0f ? p.moonAngRadius : 0.00452f;
    float cosT = dot(rayDir, f);
    float outer = angR * 1.2f + 2.0f * pixAngle;
    if (cosT < 1.0f - 0.5f * outer * outer) return float4(0.0f);

    // Disk frame: N = celestial north projected onto the disk, R = to the right of an
    // observer facing the moon with north up — lunar EAST (Mare Crisium's limb).
    float3 N = north - dot(north, f) * f;
    if (dot(N, N) < 1e-6f) N = float3(0.0f, 1.0f, 0.0f) - f.y * f;
    N = normalize(N);
    float3 R = cross(f, N);
    float sinR = sin(angR);
    float2 q = float2(dot(rayDir, R), dot(rayDir, N)) / sinR;
    float r = length(q);
    float aaW = max(pixAngle / angR, 1e-3f);
    float disk = 1.0f - smoothstep(1.0f - aaW, 1.0f + aaW, r);
    if (disk <= 0.0f) return float4(0.0f);
    float2 qc = q / max(r, 1.0f);
    float nz = sqrt(max(0.0f, 1.0f - dot(qc, qc)));
    float3 s = float3(qc, nz);                        // selenographic (east, north, earthward)
    float3 n = s.x * R + s.y * N - s.z * f;           // world-space surface normal

    float3 L = normalize(p.toSun);
    // Topography roughens the terminator: crater shadows eat into the lit side and peaks
    // catch the light past it, so the shadow line is ragged, not a clean ellipse.
    float mu0 = dot(n, L) + 0.06f * (nightFbm3(s * 45.0f, 3) - 0.5f);
    float mu = max(nz, 1e-3f);
    // Lommel–Seeliger (single-scattering regolith): 2μ0/(μ0+μ) — the FULL moon is a flat,
    // evenly bright disk (no Lambert ball), brightness falls toward the terminator. A
    // quarter Lambert in the mix gives the mild darkening a real photo shows at the limb.
    float ls = mu0 > 0.0f ? 2.0f * mu0 / (mu0 + mu) : 0.0f;
    float shade = mix(ls, max(mu0, 0.0f), 0.25f);
    // Earthshine: the night side lit by the Earth, whose phase seen from the Moon is the
    // complement of the Moon's (a new moon sees a full Earth).
    float cosA = dot(L, -f);
    // The Earth's own phase law is steep like the Moon's, so earthshine is a crescent-moon
    // phenomenon: ~1/300 of the sunlit limb on a thin crescent, gone by the quarter.
    float earthPhase = 0.5f - 0.5f * cosA;
    float es = p.earthshine * 0.004f * earthPhase * earthPhase * earthPhase * mu;

    float albedo = nightMoonAlbedo(s);
    // The disk's own highlight shoulder: its true contrast is compressed anyway, and a
    // Lommel–Seeliger crescent limb (up to 2× the disk centre) would otherwise clip the
    // lit edge flat and erase the maria along it. x/(1 + 0.4x)·1.4 — 1 → 1, 2 → 1.56.
    float x = albedo * (shade + es);
    x = 1.4f * x / (1.0f + 0.4f * x);
    float3 col = float3(1.0f, 0.975f, 0.93f) * x;
    return float4(col * (p.moonIntensity * disk) * nightExtinction(rayDir.y), disk);
}

/// Lunar aureole — moonlight forward-scattered by aerosols along the line of sight.
/// Expressed against the MOONLIT SKY (kFullMoonSkyF0 × the phase flux), not the disk:
/// the disk's host gain compresses a contrast no display can show, while the aureole
/// must sit in the right proportion to the sky around it. Physically ~5× the sky
/// background 2° from the limb, falling ~θ⁻² (the measured aureole slope), stronger
/// through more air. `moonHalo` scales it — hosts raise it for haze / thin cirrus.
static inline float3 nightMoonHalo(float3 rayDir, NightSkyParams p) {
    if (p.moonHalo <= 0.0f || p.radiance <= 0.0f || p.moonIntensity <= 0.0f) return float3(0.0f);
    float3 f = normalize(p.moonDir);
    float cosT = dot(rayDir, f);
    if (cosT < 0.866f) return float3(0.0f);   // > 30°: faded to nothing below
    float th2 = max(2.0f * (1.0f - cosT), 0.0f);   // θ² (small-angle); cosT can round past 1
    float sky = p.radiance * kFullMoonSkyF0 * nightMoonPhaseFlux(dot(normalize(p.toSun), -f));
    const float twoDeg2 = 0.0012185f;          // (2°)²
    const float core2   = 0.0002467f;          // (0.9°)² — softens the θ⁻² singularity at the limb
    float aureole = 2.5f * twoDeg2 / (th2 + core2);
    float wide = 0.08f * exp(-sqrt(th2) / 0.10f);   // 6° e-fold — clean air keeps it tight
    // Air mass: more aerosol along a low line of sight. Faded smoothly to zero by 30° so the
    // cut-off never prints as a ring.
    float X = nightAirmass(max(f.y, 0.0f));
    float fade = smoothstep(0.866f, 0.94f, cosT);
    return float3(1.0f, 0.97f, 0.92f) * (sky * p.moonHalo * (aureole + wide) * min(0.7f + 0.3f * X, 3.0f) * fade)
         * nightExtinction(rayDir.y);
}

/// Physical composite: stars + Milky Way behind the moon, moon + aureole in front.
/// `background` is the sky radiance behind the celestials (atmosphere + clouds) — the
/// washout reference.
static inline float3 nightCelestialsPhysical(float3 rayDir, NightSkyParams p, float pixAngle, float3 background) {
    if (rayDir.y <= 0.0f) return float3(0.0f);
    float3 halo = nightMoonHalo(rayDir, p);
    float4 moon = float4(0.0f);
    if (p.moonIntensity > 0.0f) {
        float3 north = nightQuatRotateInverse(p.celestial, float3(0.0f, 0.0f, 1.0f));
        moon = nightMoonPhysical(rayDir, p, north, pixAngle);
    }
    float3 behind = float3(0.0f);
    if (p.radiance > 0.0f && (p.starBrightness > 0.0f || p.milkyWay > 0.0f) && moon.a < 1.0f) {
        float3 dEq = nightQuatRotate(p.celestial, rayDir);
        float bgLum = dot(max(background, 0.0f) + halo, kNightLuma);
        if (p.starBrightness > 0.0f)
            behind += nightStarsPhysical(dEq, rayDir.y, pixAngle, p.radiance * p.starBrightness,
                                         bgLum, p.twinkle, p.clock);
        if (p.milkyWay > 0.0f) {
            // Visibility from the PHYSICAL band against the sky behind it (a gibbous moon's sky
            // is ~5× the Milky Way's brightest clouds — it vanishes, as it does outdoors); the
            // host gain is then a PERCEPTUAL lift: the dark-adapted eye sees the band by
            // summing over degrees, which a pixel does not.
            float3 ext = nightExtinction(rayDir.y);
            if (nightMilkyWayBound(dEq) * p.radiance * max(ext.r, ext.g) > 0.05f * bgLum) {
                float3 mw = nightMilkyWay(dEq) * p.radiance * ext;
                float vis = smoothstep(0.05f, 0.8f, dot(mw, kNightLuma) / max(bgLum, 1e-9f));
                behind += mw * (p.milkyWay * vis);
            }
        }
        behind *= 1.0f - moon.a;
    }
    return behind + moon.rgb + halo;
}

// ═════════════════════════════════════════════════════════════════════════════
// THE ONE ENTRY POINT
// ═════════════════════════════════════════════════════════════════════════════

/// Stars + moon (+ Milky Way + aureole in the physical model) for one ray direction —
/// what the primary sky branch, every secondary ray and the cloud upsample call.
/// Returns float3(0) when the params are zeroed, and below the horizon (the celestials
/// do not shine up out of the ground; a ray refracted downward through a pane sees the
/// yard, not the sky). `pixAngle` is the angular size of one output pixel in radians —
/// it sets the star point-spread and the moon's limb anti-aliasing. `background` is the
/// sky radiance behind this pixel (read by the physical model's washout only).
static inline float3 nightCelestials(float3 rayDir, NightSkyParams p, float pixAngle, float3 background) {
    if (p.model > 0.5f) return nightCelestialsPhysical(rayDir, p, pixAngle, background);
    if (p.starBrightness <= 0.0f && p.moonIntensity <= 0.0f) return float3(0.0f);
    if (rayDir.y <= -0.05f) return float3(0.0f);
    return nightStarField(rayDir, p.starBrightness, pixAngle)
         + nightMoonDisk(rayDir, normalize(p.moonDir), p.toSun,
                         p.moonAngRadius, p.moonIntensity, pixAngle);
}
