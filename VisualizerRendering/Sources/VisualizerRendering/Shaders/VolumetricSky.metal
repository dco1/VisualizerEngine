#include <metal_stdlib>
#include "IlluminatoramaNightSky.h"
using namespace metal;

// ── VolumetricSky.metal ──────────────────────────────────────────────────────
//
// Single equirect-projection compute kernel that renders the entire sky for a
// scene: a Rayleigh + Mie atmosphere gradient with a soft sun disk, and a
// volumetric cumulus deck raymarched through a finite altitude slab. Output is
// an RGBA16F texture in latitude–longitude layout (u = azimuth 0..2π, v =
// elevation +π/2..-π/2) that callers bind to `SCNScene.background.contents`
// and `SCNScene.lightingEnvironment.contents`.
//
// What it gives the scene
// ───────────────────────
//   • Sky:  three-band Rayleigh-ish gradient (zenith → mid-sky → horizon haze)
//           plus a Mie-like forward-scatter halo around the sun direction.
//   • Sun:  soft disk with a brighter inner core and an HDR clamp so SceneKit's
//           exposure-adaptation can drive the eye-bloom on it.
//   • Clouds: cumulus deck living between [cloudBaseY, cloudTopY] world units.
//             Density is FBM(Perlin) shaped by a Worley erosion mask and remapped
//             against a "coverage" weather field. Lit per-step by a short
//             4-sample light-march toward the sun, weighted by a dual-lobe
//             Henyey–Greenstein phase plus a "powder" term that darkens the
//             insides of clouds so the silhouette pops against the sky.
//
// Performance notes
// ─────────────────
//   • Aimed at ~1024×512 RGBA16F, ~60 cloud-march steps × 4 light steps per
//     pixel (skipped entirely when ray exits the slab without entering it, or
//     when the elevation is below the horizon — the kernel returns the
//     atmosphere only). No 3D textures: all noise is procedural hash-noise so
//     we don't ship a baked volume in the package bundle.
//   • Temporal reprojection is *not* implemented in V1. The cloud silhouette
//     is stable as long as `params.time` is the only thing changing, but
//     turning coverage / wind drives a re-march; that's expected.
//
// Coordinate conventions
// ──────────────────────
//   • World: right-handed, +Y up. Sun direction is "from the sun TO the scene"
//     (i.e. the direction sunlight travels), matching SceneKit's directional
//     light convention.
//   • Equirect uv: u = (x + 0.5) / W, v = (y + 0.5) / H. We map u → azimuth in
//     [0, 2π) and v → elevation in [+π/2, -π/2] (top of texture = looking up).
//   • Cloud world is in *the same scale as the host scene*. cloudBaseY/cloudTopY
//     are absolute world Y; horizontalScale stretches noise in XZ so a single
//     fbm period covers a believable cumulus-mass diameter.
//
// Swift mirror
// ────────────
// The `SkyUniforms` struct below must match the Swift mirror in
// VolumetricCloudRenderer.swift byte-for-byte. Pure float4 / float2 groupings
// keep the std-buffer alignment trivial; see the alignment-rule doc in
// PBDSolver.swift.

struct SkyUniforms {
    // ── Camera / framing ─────────────────────────────────────────────
    // World-space position of the viewer used as the ray origin. For an
    // at-infinity sky this barely matters, but it lets clouds drift relative
    // to a moving camera without re-binding the texture every frame.
    float4 cameraPos;            // xyz = position, w = unused

    // ── Sun ─────────────────────────────────────────────────────────
    float4 sunDir;               // xyz = direction sunlight TRAVELS (unit), w = unused
    float4 sunColor;             // xyz = HDR colour, w = disk intensity scalar

    // ── Sky gradient ────────────────────────────────────────────────
    float4 skyZenith;            // xyz = colour at the zenith, w = cloudAmbientGrey
                                 //   (0 = cloud underside lit by blue sky-ambient,
                                 //    1 = neutral overcast grey — DH-0585)
    float4 skyHorizon;           // xyz = colour at the horizon, w = haze power (1..8)
    float4 groundColor;          // xyz = colour for rays pointing down, w = horizon blend

    // ── Cloud slab geometry ─────────────────────────────────────────
    // cloudBaseY / cloudTopY define the altitude band the raymarch walks
    // through. The slab is treated as infinite in XZ.
    float4 cloudSlab;            // x = baseY, y = topY, z = horizontalScale, w = anvil

    // ── Cloud field shaping ─────────────────────────────────────────
    // x = coverage  (0 clear sky .. 1 overcast)
    // y = density   (mass multiplier on the raw FBM)
    // z = erosion   (Worley-Perlin mix, 0..1)
    // w = ambientTerm (sky-light fill on the unlit side of the cloud)
    float4 cloudShape;

    // ── Wind ─────────────────────────────────────────────────────────
    // x,y = unit direction in XZ; z = speed (units/sec); w = time (sec).
    float4 windTime;

    // ── Lighting tuning ─────────────────────────────────────────────
    // x = phase g_forward (HG, +0.2..+0.8)
    // y = phase g_back    (HG, -0.6..-0.1, summed at half weight)
    // z = light absorption (1..6 — bigger = darker cloud interiors)
    // w = powder strength (0..1)
    float4 lighting;

    // ── March budgets / burst lights ────────────────────────────────
    // x = cloudSteps (1..96), y = lightSteps (1..8). Stored as floats so the
    // Swift side can write to the same float4 with no alignment surprises.
    // z = burstLightCount — number of live entries in the BurstLight buffer
    //     bound at buffer(1) (0 = feature off; the renderer always binds at
    //     least a zeroed fallback so the pointer is never dangling).
    // w = burstLightGain — scalar on the burst in-scatter term so hosts can
    //     balance "fireworks glow in the clouds" against sun/ambient.
    float4 marchBudgets;

    // ── Night sky ────────────────────────────────────────────────────
    // moonParams: xyz = unit direction TO the moon (from scene), w = phase
    //   (0 = new moon dark, 0.5 = half, 1 = full moon).
    // nightParams: x = starBrightness (0..1), y = moonIntensity (0..1+),
    //   z = unused, w = nightBlend (0 = full day, 1 = full night — caller
    //   derives this from sunDir.y so the transition happens automatically
    //   as the sun crosses the horizon).
    float4 moonParams;
    float4 nightParams;

    // ── Atmosphere model ─────────────────────────────────────────────
    // x = mode (0 = faked three-band gradient, 1 = Nishita physical scatter)
    // y = Nishita sun-scatter intensity (unitless gain on the integral)
    // z = debug-neon background flag (>0.5 = on)
    // w = directional-moonlight gain (cloud form at night)
    float4 atmosphereParams;

    // ── Cloud detail ─────────────────────────────────────────────────
    // x = weatherScale — frequency multiplier on the large-scale weather map
    //     that decides WHERE cloud clusters form vs clear sky. 0 falls back to
    //     the legacy 0.125. Higher = smaller, more-separated cloud cells (the
    //     lever for "scattered individual clouds" vs "one continuous bed").
    // y = weatherSharp — extra contrast on the weather field so clusters have
    //     crisp edges with clear gaps between (0 = legacy linear).
    // z = fadeNear — clouds within this many metres of the camera fade out
    //     (clears foreground clutter right overhead). 0 = off.
    // w = fadeFar — clouds beyond this many metres fade out (kills the
    //     horizon pile-up an infinite slab makes at grazing angles). 0 = off.
    float4 cloudDetail;

    // ── Cloud field (world-anchored finite region) ───────────────────
    // xy = world XZ centre of the cloud region; z = inner radius (full
    // density); w = outer radius (faded to 0). Bounds the otherwise-infinite
    // slab to a finite disc around a fixed world point, so the deck reads the
    // SAME finite scattered field from every camera (a camera-relative fade
    // faded the deck out from far cameras). w = 0 disables (infinite slab).
    float4 cloudField;

    // ── Curl-noise domain warp ───────────────────────────────────────
    // Wispy distortion of the cloud field (issue #18 priority 2). Before the
    // body/erosion noise is sampled, the world position is displaced by a
    // divergence-free 2D curl-noise vector in XZ, so straight FBM mounds get
    // sheared into the swirled, torn-edged silhouette real cumulus has
    // (instead of round cotton balls). Divergence-free so it warps without
    // pumping density up/down.
    // x = curlStrength — displacement amplitude in WORLD units (0 = off).
    // y = curlScale    — frequency of the curl field (cycles per world unit;
    //                    0 falls back to a scale derived from horizontalScale).
    // z = moundMix     — weight (0..1) of a second, lower-frequency body sample
    //                    that supplies large coherent mound SHAPE over the detail
    //                    sample (0 = single-scale body, legacy/other scenes).
    // w = moundRatio   — frequency ratio of that low-freq sample (0 → 0.35).
    float4 cloudExtra;

    // ── Multiple-scattering approximation ────────────────────────────
    // x = multiScatter — strength (0..1+) of the multi-octave multiple-scatter
    //     approximation on the directional + burst lighting. 0 = single-scatter
    //     only (byte-identical to the legacy path; every non-fireworks scene).
    //     >0 adds 2 softer octaves (less extinction, flatter phase) so thick
    //     banks self-glow from a bright internal/behind source instead of
    //     reading as flat-lit shells.
    // y = celestialsInDome (>0.5 = stars/moon are baked into this equirect —
    //     the default). A host whose lighting pass evaluates the analytic
    //     night sky at SCREEN resolution (Illuminatorama's nightStarField /
    //     nightMoonDisk) sets this to 0 so the dome doesn't double-draw a
    //     blurry low-res copy underneath the crisp one. z,w reserved.
    float4 cloudExtra2;

    // ── Flat studio background (opt-in) ──────────────────────────────
    // Bypasses the ENTIRE dome (atmosphere, sun, clouds, stars/moon) with a flat,
    // direction-independent colour when enabled — see `volSkyRender`'s first branch.
    // xyz = colour (linear HDR), w = enable flag (>0.5 = on).
    float4 studioParams;

    // ── Sky grade ────────────────────────────────────────────────────
    // x = skySaturation: chroma of the physical sky about its own luma, 1 = the
    //     nishita march untouched, 0 = grey, >1 = deeper. The one lever a host has
    //     over the sky's COLOUR without re-grading the scene under it: a global
    //     saturation dial reaches the sky only through everything else (measured
    //     on Daydream's street render: the wall's blue channel collapsed to 4 before
    //     the sky got anywhere near a photograph's).
    // y = skyBlueLift: the blue channel scaled by this and the luma restored — a HUE pull toward
    //     blue. Saturation alone could not reach a photograph's sky: about its own luma the
    //     march goes cyan (red falls, blue barely rises). 1 = untouched.
    // z = cloudLightingFromAtmosphere flag (> 0.5 = the cloud march reads cloudLitSun /
    //     cloudLitAmbient, written on the GPU by `volSkyCloudLight`; skyHorizon.xyz then carries
    //     the ground ALBEDO for the bounce). w = unused.
    float4 skyGrade;
    // ── Cirrus veil (Params.cirrus*) — host-owned ───────────────────────────────────────────
    // cirrusA: x = coverage (0 = off, byte-identical), y = zenith optical depth of a full fibre,
    //          z = altitude (m), w = feature frequency (noise texels per metre)
    // cirrusB: xy = streak direction (xz, unit), z = along-streak stretch, w = drift multiplier
    float4 cirrusA;
    float4 cirrusB;
    // cirrusC: x = THREADS mode (>0.5), y = flow speed, z = frequency multiplier, w = unused
    // cirrusTint: rgb multiplies the veil's radiance (white = physical), w unused
    float4 cirrusC;
    float4 cirrusTint;
    // ── Lava-lamp sky (Params.lavaLamp*) — host-owned ───────────────────────────────────────
    // lavaA: x = on (>0.5), y = time (s, the dome's clock), z = blob scale, w = glow (radiance)
    // lavaBG / lavaBlobA / lavaBlobB: rgb palette; lavaBG.w = speed multiplier (0 → treated as 1)
    float4 lavaA;
    float4 lavaBG;
    float4 lavaBlobA;
    float4 lavaBlobB;
    // ── Night sky (Params.nightSkyModel / moon* / star* / nightRadiance…) — host-owned ──────
    // nightSkyA: x = model (0 legacy, 1 physical), y = moon angular radius (rad),
    //            z = aureole gain, w = earthshine gain
    // nightSkyB: x = Milky Way gain, y = scintillation gain, z = night radiance (scene radiance
    //            of a magnitude-0 flux per sr), w = airglow gain
    // nightSkyC: xyz = direction the moon is lit FROM (true sun, or the effective sun that
    //            gives `moonPhase`), w = zodiacal-light gain
    // nightSkyD: world → J2000-equatorial quaternion (xyz imaginary, w real)
    float4 nightSkyA;
    float4 nightSkyB;
    float4 nightSkyC;
    float4 nightSkyD;
    // ── Cloud lighting from the atmosphere — GPU-WRITTEN, never by the host ─────────────────
    // `volSkyCloudLight` fills these from the nishita march itself (the SAME `nishitaScatter`
    // the sky pixels use), so the deck is lit in the sky's own units. The host packs only the
    // struct PREFIX before these two, so its per-frame write never clobbers them.
    float4 cloudLitSun;      // xyz = solar irradiance at the deck base (intensity × T), w = 1
    float4 cloudLitAmbient;  // xyz = mean sky + ground-bounce radiance around the deck, w = 1
};

// The cloud kernel reads the noise volume's tile size from its own width
// (it's a cube — width == height == depth). Cheap, keeps the uniforms
// mirror simple, no new Swift-side field.

// ── Hash + noise primitives ─────────────────────────────────────────────────
//
// All inline / procedural. We avoid texture lookups so the renderer stays a
// single-kernel, single-texture-output thing without per-frame upload of a
// 3D Perlin/Worley volume.

inline uint hash3(int3 p) {
    uint h = uint(p.x * 374761393 + p.y * 668265263 + p.z * 1274126177);
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

inline float hashFloat(int3 p) {
    return float(hash3(p) & 0x00FFFFFFu) / float(0x01000000u);
}

inline float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Trilinear value noise on a unit lattice. Good enough as the low-frequency
// shaping field; we layer it into FBM below.
inline float valueNoise3(float3 p) {
    int3 i = int3(floor(p));
    float3 f = p - float3(i);
    float3 u = float3(fade(f.x), fade(f.y), fade(f.z));

    float n000 = hashFloat(i + int3(0,0,0));
    float n100 = hashFloat(i + int3(1,0,0));
    float n010 = hashFloat(i + int3(0,1,0));
    float n110 = hashFloat(i + int3(1,1,0));
    float n001 = hashFloat(i + int3(0,0,1));
    float n101 = hashFloat(i + int3(1,0,1));
    float n011 = hashFloat(i + int3(0,1,1));
    float n111 = hashFloat(i + int3(1,1,1));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

// 3D Worley noise — minimum distance to a Poisson-disk-ish point set on a
// jittered lattice. Returns ~0 at cell centres, ~1 at the cell boundary. We
// use 1 - worley as an erosion mask: it punches "gaps" out of the FBM body
// where cells meet, which is what shreds dense FBM into believable cumulus
// silhouettes (Schneider/Vos style).
inline float worley3(float3 p) {
    int3 ip = int3(floor(p));
    float3 fp = p - float3(ip);
    float minD2 = 1e9f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 cell = ip + int3(dx, dy, dz);
                uint h = hash3(cell);
                float3 jitter = float3(
                    float((h      ) & 0xFFu),
                    float((h >> 8u) & 0xFFu),
                    float((h >> 16u) & 0xFFu)
                ) * (1.0f / 255.0f);
                float3 pt = float3(dx, dy, dz) + jitter;
                float3 d = pt - fp;
                float d2 = dot(d, d);
                minD2 = min(minD2, d2);
            }
        }
    }
    return clamp(sqrt(minD2), 0.0f, 1.0f);
}

// 4-octave FBM over valueNoise3. Returns ~[0..1].
inline float fbm4(float3 p) {
    float a = 0.5f;
    float sum = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum  += a * valueNoise3(p);
        norm += a;
        p = p * 2.02f + float3(17.0f, 13.0f, 7.0f);
        a *= 0.5f;
    }
    return sum / max(norm, 1e-5f);
}

// ── Cloud field ─────────────────────────────────────────────────────────────

// Per-altitude shaping. cumulus base is dense and flat, the top tapers into a
// rounded crown. We bias the density curve toward "fat middle, sparse top
// and bottom" with an anvil parameter that lifts the upper portion outward
// (mimicking the anvil shelf of a congestus / cumulonimbus).
inline float heightShaping(float h, float anvil) {
    // h is 0 at the base of the slab, 1 at the top.
    float bottom = smoothstep(0.0f, 0.10f, h);
    float top    = 1.0f - smoothstep(0.55f, 1.0f, h);
    float core   = bottom * top;
    // Anvil — re-add a soft shoulder near h ≈ 0.85 weighted by `anvil`.
    float shelf  = smoothstep(0.75f, 0.92f, h) * (1.0f - smoothstep(0.92f, 1.0f, h));
    return clamp(core + anvil * shelf * 0.6f, 0.0f, 1.0f);
}

// Sampler for the baked 3D noise volume bound at [[texture(1)]] in the
// main kernel. Trilinear interpolation + repeat-wrap turns the tileable
// volume into a continuous noise field in world space.
constexpr sampler volSampler(filter::linear,
                              address::repeat,
                              coord::normalized);

// "Weather map" — large-scale 2D field driving local coverage. Drives where
// clouds form vs. clear sky. A second slow FBM in XZ is the simplest version
// that still feels like rolling weather. Runs at ~1/8 the fundamental cloud
// scale so the weather field changes only over many cloud diameters.
//
// Reads from the same baked 3D noise volume the cloud body uses — sampling
// the R channel (fbm) at a slow-moving uvw acts as a 2D-ish fbm in XZ.
inline float weatherCoverage(float2 xz,
                              float hScl,
                              float baseCoverage,
                              float t,
                              float tileSize,
                              float weatherScale,
                              float weatherSharp,
                              texture3d<float, access::sample> noiseVol) {
    // weatherScale multiplies the weather-map frequency. The legacy 0.125 is
    // tuned for far-away decks (large world pos); a close, small-scale deck
    // needs a much higher value or the weather field is spatially constant
    // across the whole scene → one continuous cloud bed. 0 → legacy.
    float wScl = (weatherScale > 0.0f) ? weatherScale : 0.125f;
    float2 q = xz * hScl * wScl + float2(0.13f, 0.27f) * t * 0.020f;
    float3 uvw = float3(q.x, 0.4f, q.y) / max(tileSize, 1.0f);
    float w = noiseVol.sample(volSampler, uvw).r;
    // Decorrelated second octave (only when weatherSharp > 0 — i.e. scenes that
    // opt into the crisp broken-cumulus weather; far decks at sharp 0 are
    // unchanged). A single fbm slice has long crossing RIDGES, which from above
    // read as axis-aligned channels / an "X", not scattered cumulus. Sampling a
    // SECOND octave at a rotated + offset coord and taking the PRODUCT keeps only
    // where BOTH peak → distributed blob maxima, killing the single-ridge
    // structure. Re-normalised (×2) so the product's range still spans ~0..1.
    if (weatherSharp > 0.0f) {
        float2 qr = float2(q.x * 0.80f - q.y * 0.60f,
                           q.x * 0.60f + q.y * 0.80f) * 1.7f + float2(5.2f, 1.3f);
        float3 uvw2 = float3(qr.x, 0.72f, qr.y) / max(tileSize, 1.0f);
        float w2 = noiseVol.sample(volSampler, uvw2).r;
        w = clamp(w * w2 * 2.0f, 0.0f, 1.0f);
    }
    float cov = clamp(w + (baseCoverage - 0.5f) * 1.8f, 0.0f, 1.0f);
    // weatherSharp pushes the field toward 0/1 so cloud CLUSTERS have crisp
    // boundaries with clear sky between, instead of a soft everywhere-haze.
    if (weatherSharp > 0.0f) {
        float c = smoothstep(0.0f, 1.0f, cov);
        c = c * c * (3.0f - 2.0f * c);              // extra S-curve
        cov = mix(cov, c, saturate(weatherSharp));
    }
    return cov;
}

// Divergence-free 2D curl-noise domain warp in the XZ plane. The curl of a
// scalar potential (here a value-noise field) is perpendicular to its
// gradient, so advecting the sample position along it shears the cloud body
// into swirled, torn billows WITHOUT inflating or deflating density (curl is
// divergence-free). This is the "curl distortion" element of issue #18's
// "good clouds use low-freq shape + high-freq erosion + curl" recipe — it
// turns straight FBM mounds into the wispy, non-spherical silhouette real
// cumulus has. Returns a world-space XZ displacement (caller scales by
// strength). Cheap: 4 inline value-noise taps; only called when curl is on.
inline float2 curlWarpXZ(float3 p, float scale) {
    float3 q = float3(p.x * scale, p.y * scale * 0.5f, p.z * scale);
    const float e = 0.35f;
    // ∂/∂x and ∂/∂z of a value-noise potential (central differences).
    float nx1 = valueNoise3(q + float3(e, 0.0f, 0.0f));
    float nx0 = valueNoise3(q - float3(e, 0.0f, 0.0f));
    float nz1 = valueNoise3(q + float3(0.0f, 0.0f, e));
    float nz0 = valueNoise3(q - float3(0.0f, 0.0f, e));
    float2 grad = float2(nx1 - nx0, nz1 - nz0) / (2.0f * e);
    // Perpendicular to the gradient → divergence-free flow field.
    return float2(grad.y, -grad.x);
}

// Sample cloud density at `pos` in world space. Returns post-shaped density in
// [0..1] (further scaled by `cloudShape.y` at call sites).
inline float cloudDensity(float3 pos,
                           constant SkyUniforms &u,
                           texture3d<float, access::sample> noiseVol) {
    float baseY = u.cloudSlab.x;
    float topY  = u.cloudSlab.y;
    float hScl  = u.cloudSlab.z;
    float anvil = u.cloudSlab.w;

    float h = clamp((pos.y - baseY) / max(topY - baseY, 1e-3f), 0.0f, 1.0f);
    float heightW = heightShaping(h, anvil);
    if (heightW <= 0.0f) return 0.0f;

    // Wind advection — drift the noise lookup along the wind vector so the
    // cloud field translates over time without re-baking. `hScl` maps
    // world units → noise-space; q is the noise-space position.
    float t = u.windTime.w;
    float2 wind = u.windTime.xy * u.windTime.z * t;
    float3 posW = float3(pos.x - wind.x, pos.y, pos.z - wind.y);

    // Curl-noise domain warp (issue #18). Shear the sample position along a
    // divergence-free flow field so the FBM body reads as wispy, torn cumulus
    // rather than round mounds. Gated by curlStrength so non-fireworks scenes
    // (curl off) are byte-identical to the pre-curl field.
    float curlStrength = u.cloudExtra.x;
    if (curlStrength > 0.0f) {
        float curlScale = (u.cloudExtra.y > 0.0f) ? u.cloudExtra.y : (hScl * 0.5f);
        float2 warp = curlWarpXZ(posW, curlScale);
        posW.x += warp.x * curlStrength;
        posW.z += warp.y * curlStrength;
    }

    float3 q = posW * hScl;

    // One trilinear sample fetches both fields:
    //   R = fbm4(q)         — low-frequency cloud body
    //   G = 1 - worley(q*4) — high-frequency erosion (4× density baked in)
    float tileSize = float(noiseVol.get_width());
    float4 noise = noiseVol.sample(volSampler, q / tileSize);
    float body    = noise.r;
    float erosion = noise.g;

    // Two-scale body (issue #18 "low-freq SHAPE + high-freq erosion"): the
    // baked volume only resolves ~1/hScl m per texel, so a single sample at the
    // detail frequency reads as high-frequency SPECKLE from above rather than
    // coherent cauliflower mounds (scene-reviewer round-3). Mixing in a SECOND
    // sample at a lower frequency (cloudExtra.w × q) supplies the large mound
    // SHAPE the detail sample lacks; `moundMix` weights it mound-dominant so the
    // turrets are big and carry fine erosion, not uniform noise. Gated (mix 0 =
    // single-sample, byte-identical for every other scene).
    float moundMix = u.cloudExtra.z;
    if (moundMix > 0.0f) {
        float moundRatio = (u.cloudExtra.w > 0.0f) ? u.cloudExtra.w : 0.35f;
        float moundBody = noiseVol.sample(volSampler, (q * moundRatio) / tileSize).r;
        body = mix(body, moundBody, moundMix);
    }

    // Mix body × erosion. erosionAmt controls how aggressively the cells
    // chew through the FBM body.
    float erosionAmt = u.cloudShape.z;
    float shaped = body * mix(1.0f, erosion, erosionAmt);

    // Weather-map coverage threshold: where the large-scale weather field is
    // high, clouds form; where it's low, clear sky. This (NOT hand-placed
    // shapes) is what gives a broken cumulus deck distinct banks with gaps.
    float cov = weatherCoverage(pos.xz, hScl, u.cloudShape.x, t,
                                tileSize, u.cloudDetail.x, u.cloudDetail.y, noiseVol);
    float thresh = 1.0f - cov;
    float d = max(0.0f, shaped - thresh) / max(1.0f - thresh, 1e-3f);

    // Density-contrast curve (nightParams.z; 1 = legacy linear). γ > 1
    // crushes the mid-density fringe toward sky while leaving cores near 1 —
    // the "either dense turret or empty gap" distribution real cumulus has.
    // Without it the linear ramp yields a wide mid-gray sheet whose optical
    // depth integrates to a structureless fog wash (FireworksUltra cloud
    // review). Stored in nightParams for layout-compat: the slot was unused.
    float gamma = u.nightParams.z;
    if (gamma > 1.001f) {
        d = pow(d, gamma);
    }

    return d * heightW;
}

// ── Phase + atmosphere ──────────────────────────────────────────────────────

// Henyey–Greenstein phase function. Forward-scatter for the bright fringe
// around the sun; we sum a dual-lobe (forward + faint back) so the cloud
// doesn't go pitch-black on the shadow side.
inline float hg(float cosT, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosT;
    return (1.0f - g2) / (4.0f * 3.14159265f * pow(max(denom, 1e-4f), 1.5f));
}

// Multiple-scattering approximation (Wrenninge / Hillaire multi-octave). A
// single-scatter `exp(-od·absK)·phase` term makes a thick cloud read as a
// flat-lit shell: all the light terminates at the surface and the interior is
// dark. Real cumulus glows because light bounces many times inside it. We
// approximate that by summing extra octaves whose extinction FALLS (light
// reaches deeper), contribution falls, and phase FLATTENS toward isotropic —
// so deep/back-lit regions pick up a soft self-glow without a second full
// multiple-scatter solve. Octave 0 is the exact single-scatter term, so
// `strength == 0` reproduces the legacy result; `strength > 0` adds the glow.
inline float msLight(float od, float cosT, float gF, float gB,
                     float absK, float strength) {
    float lum = exp(-od * absK) * (hg(cosT, gF) + 0.5f * hg(cosT, gB));
    float att = 0.5f, scat = 0.5f * strength, ph = 0.5f;
    for (int o = 1; o < 3; ++o) {
        float phase = hg(cosT, gF * ph) + 0.5f * hg(cosT, gB * ph);
        lum += scat * exp(-od * absK * att) * phase;
        att *= 0.5f; scat *= 0.5f; ph *= 0.5f;
    }
    return lum;
}

inline float3 atmosphereColor(float3 rayDir, constant SkyUniforms &u) {
    // Three-band gradient as a function of elevation = rayDir.y in [-1, +1].
    float t = clamp(rayDir.y, -1.0f, 1.0f);
    float skyBlend = smoothstep(-0.05f, 0.85f, t);
    float3 sky = mix(u.skyHorizon.xyz, u.skyZenith.xyz, pow(skyBlend, max(0.3f, u.skyHorizon.w)));

    // Below-horizon rays fade through the ground colour. Soft transition so
    // we don't get a hard horizon line.
    float groundBlend = smoothstep(-0.30f, 0.0f, t);
    sky = mix(u.groundColor.xyz, sky, groundBlend);

    // Mie-like forward scatter — brightens the sky toward the sun.
    float cosT = clamp(dot(rayDir, -u.sunDir.xyz), -1.0f, 1.0f);
    float mie = pow(max(0.0f, cosT) , 4.0f) * 0.6f;
    sky += u.sunColor.xyz * mie * 0.25f;

    return sky;
}

// ── Nishita physical atmosphere ──────────────────────────────────────────────
//
// A *real* single-scattering atmosphere (Nishita et al. 1993, "Display of the
// Earth Taking into account Atmospheric Scattering") replacing the faked
// three-band gradient above. Ported from BenSimonds/NishitaSky's OSL shader,
// which is itself the canonical scratchapixel formulation:
//
//   - Two nested ray-marches: a PRIMARY march of the view ray through the
//     atmosphere shell, and at each primary sample a SECONDARY march toward the
//     sun to measure how much sunlight survives the trip in (transmittance).
//   - Density falls off exponentially with altitude, with separate scale
//     heights for Rayleigh (air molecules, λ⁻⁴ blue scatter) and Mie (aerosols,
//     forward-biased haze). Multiplying by the per-channel β coefficients and
//     the angular phase functions and integrating gives the sky colour.
//
// This is "option 1" from the porting plan: BAKE the whole equirect once per
// sun move. The host throttles dome re-bakes (VolumetricCloudRenderer's
// minRenderInterval), and the cost is amortised because the result is reused
// as both the visible dome and the IBL environment. Sample budgets are kept
// modest (kPrimary × kLight) since we're filling a 2048×1024 (or 4K) target.
//
// Two heavier variants were deliberately NOT taken; sketches for later.
// (Both evaluated 2026-06-07; decision = ship the baked brute-force
// march above and keep these as notes, not code.)
//
//   • OPTION 2 — evaluate scattering PER-RAY, PER-FRAME inside volSkyRender
//     instead of baking. More accurate near a moving horizon / fast sun, and
//     removes the bake-staleness entirely, but it reruns the full nested march
//     for every cloud ray every frame. At 2048×1024 × (kPrimary·kLight) exp()
//     evals that blows the per-frame budget this kernel is tuned for — only
//     worth it if the sun is animated continuously and the dome cache visibly
//     lags. Implementation: drop the bake/throttle, call nishitaScatter()
//     directly where atmosphereColor() is called in the live cloud pass.
//
//     The way to make Option 2 actually affordable is Sean O'Neil's
//     "Accurate Atmospheric Scattering" (GPU Gems 2, ch. 16):
//     https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-16-accurate-atmospheric-scattering
//     It's the SAME single-scatter physics as nishitaScatter() but DELETES the
//     inner sun-ray loop. O'Neil observed the normalised optical-depth-vs-
//     altitude curves all collapse onto ~exp(-4x), so the whole inner integral
//     fits one polynomial — the `scale()` function:
//         scale(c) = 0.25 * exp(-0.00287 + c*(0.459 + c*(3.83 + c*(-6.80 + c*5.25))))
//     That takes it from ~3000 ops/sample to ~60 and just 5 OUTER samples, i.e.
//     it runs live in a vtx/frag skydome shader. Caveats that kept it out:
//     `scale()` is HARDCODED to scaleHeight=0.25 and atmosphere=2.5% of planet
//     radius (change those and the fit is wrong); O'Neil DISABLES the Rayleigh
//     phase (sky too dark near 90° without multi-scatter); and it needs split
//     camera-in-atmosphere vs in-space variants for precision. It is an
//     APPROXIMATION of the march above, not more accurate than it — the title's
//     "accurate" means "vs. the old faked gradient." To adopt: add a third
//     AtmosphereModel that swaps the inner loop here for scale(), then drop the
//     bake throttle so the dome updates every frame.
//
//   • OPTION 3 — PRECOMPUTED LUTs (Bruneton 2008 / Hillaire 2020, the modern
//     SOTA; Elek is the simple variant): bake a 2D transmittance table
//     (altitude × view-zenith), a multiple-scattering table, and a per-frame
//     sky-view LUT, then the per-pixel sky is a couple of texture fetches + the
//     phase function. Faster per-pixel than O'Neil AND more accurate (it adds
//     multiple scattering, which the single-scatter march above lacks; ozone
//     absorption is already in the march). The catch: it's a whole subsystem (extra kernels, LUT
//     textures, parameterisation) for what is currently one baked dome. This is
//     the right answer if many scenes want a live, cheap, physically-correct
//     sky — revisit if Nishita becomes the default across scenes.
//
// Constants are SI metres. The viewer sits ~1 m above the planet surface at the
// pole so world +Y == planet "up", matching the scene's up axis.

constant float  kEarthRadius = 6360e3;   // m
constant float  kAtmosRadius = 6420e3;   // m (60 km shell)
constant float  kRayleighH   = 7994.0;   // Rayleigh scale height (m)
constant float  kMieH        = 1200.0;   // Mie scale height (m)
constant float3 kBetaR       = float3(5.8e-6, 13.5e-6, 33.1e-6); // Rayleigh β (m⁻¹)
constant float3 kBetaM       = float3(21e-6);                    // Mie β (m⁻¹)
constant float  kMieG        = 0.76;     // Mie anisotropy (forward bias)
// Ozone absorption β (m⁻¹), Chappuis band — pure absorption, no scattering. The
// green/red-heavy cross-section (Hillaire 2020) is what gives a real twilight its
// BLUE zenith: over the long grazing sun-ray path at sunset ozone eats the green
// and red, leaving blue. Without it a single-scatter march reads greenish/olive
// at the twilight zenith.
constant float3 kBetaO       = float3(0.650e-6, 1.881e-6, 0.085e-6);
constant float  kOzoneCenter = 25000.0;  // m — ozone layer peak altitude
constant float  kOzoneWidth  = 15000.0;  // m — tent half-width (density → 0 by ~10/40 km)
constant int    kPrimary     = 16;       // view-ray samples
constant int    kLight       = 8;        // sun-ray samples per view sample

// Ozone number-density profile, normalised 0…1 — a linear tent peaking at
// kOzoneCenter. Coarse vs. a measured profile but the right shape for the
// Chappuis-band twilight tint, and free relative to the exp() calls beside it.
inline float ozoneDensity(float h) {
    return max(0.0f, 1.0f - fabs(h - kOzoneCenter) / kOzoneWidth);
}

// Smallest positive root of |orig + t·dir|² = radius² (dir is unit), or -1 if
// the ray misses the sphere. Used for both the atmosphere shell and the planet.
inline float raySphereExit(float3 orig, float3 dir, float radius) {
    float b = dot(orig, dir);               // a == 1 (dir is unit)
    float c = dot(orig, orig) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    return -b + sqrt(disc);                  // far root (we start inside the shell)
}

// Distance to the nearest planet intersection ahead of the ray, or -1 if the
// ray clears the planet (points at or above the horizon).
inline float rayGroundHit(float3 orig, float3 dir, float radius) {
    float b = dot(orig, dir);
    float c = dot(orig, orig) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    float t = -b - sqrt(disc);               // near root
    return (t > 0.0f) ? t : -1.0f;
}

inline float3 nishitaScatter(float3 rayDir, float3 sunDir, float intensity) {
    float3 orig = float3(0.0f, kEarthRadius + 1.0f, 0.0f); // viewer at the pole
    float3 toSun = -sunDir.xyz;                            // sunDir = travel dir

    // March only as far as the atmosphere shell, and stop early at the ground
    // so rays below the horizon don't integrate through the planet.
    float tShell = raySphereExit(orig, rayDir, kAtmosRadius);
    if (tShell <= 0.0f) return float3(0.0f);
    float tGround = rayGroundHit(orig, rayDir, kEarthRadius);
    float tMax = (tGround > 0.0f) ? min(tShell, tGround) : tShell;

    float segLen = tMax / float(kPrimary);
    float t = 0.0f;

    float3 sumR = float3(0.0f);
    float3 sumM = float3(0.0f);
    float odR = 0.0f;   // accumulated view-ray optical depth (Rayleigh)
    float odM = 0.0f;   // …(Mie)
    float odO = 0.0f;   // …(Ozone — absorption only)

    // Angular (phase) terms depend only on the view–sun angle, not position.
    float mu = dot(rayDir, toSun);
    float phaseR = 3.0f / (16.0f * 3.14159265f) * (1.0f + mu * mu);
    float g2 = kMieG * kMieG;
    float phaseM = 3.0f / (8.0f * 3.14159265f)
                 * ((1.0f - g2) * (1.0f + mu * mu))
                 / ((2.0f + g2) * pow(max(1.0f + g2 - 2.0f * kMieG * mu, 1e-4f), 1.5f));

    for (int i = 0; i < kPrimary; ++i) {
        float3 sp = orig + rayDir * (t + segLen * 0.5f);
        float h = length(sp) - kEarthRadius;
        float hr = exp(-h / kRayleighH) * segLen;
        float hm = exp(-h / kMieH) * segLen;
        odR += hr;
        odM += hm;
        odO += ozoneDensity(h) * segLen;

        // Secondary march toward the sun: how much air is between this sample
        // and space. If the sun ray dives into the planet we're in shadow and
        // this sample contributes nothing.
        float tLight = raySphereExit(sp, toSun, kAtmosRadius);
        float segLenL = tLight / float(kLight);
        float tl = 0.0f;
        float odLR = 0.0f;
        float odLM = 0.0f;
        float odLO = 0.0f;
        int j = 0;
        for (; j < kLight; ++j) {
            float3 spl = sp + toSun * (tl + segLenL * 0.5f);
            float hl = length(spl) - kEarthRadius;
            if (hl < 0.0f) break;          // sun ray is occluded by the planet
            odLR += exp(-hl / kRayleighH) * segLenL;
            odLM += exp(-hl / kMieH) * segLenL;
            odLO += ozoneDensity(hl) * segLenL;
            tl += segLenL;
        }

        if (j == kLight) {
            // Combined extinction along view-in + sun-out paths. Mie ×1.1 is
            // the standard extinction-vs-scattering fudge from the source; ozone
            // adds pure absorption (no scattering term, so it only attenuates).
            float3 tau = kBetaR * (odR + odLR) + kBetaM * 1.1f * (odM + odLM)
                       + kBetaO * (odO + odLO);
            float3 atten = exp(-tau);
            sumR += atten * hr;
            sumM += atten * hm;
        }
        t += segLen;
    }

    return intensity * (sumR * kBetaR * phaseR + sumM * kBetaM * phaseM);
}

// Wrapper matching atmosphereColor()'s role: physical sky above the horizon,
// horizon haze + art-directed ground colour below it.
//
// HORIZON CONTINUITY: a ray just below the horizon ends on distant ground
// seen through hundreds of km of grazing atmosphere — to the eye that is the
// SAME haze as the horizon band above it. The raw march stops at the planet
// and returns only a thin dark in-scatter, so the old groundColor mix left a
// near-BLACK stripe in the first degrees of depression — visible as a black
// band between any finite scene floor's far edge and the sky (a 400 m floor
// from a 3 m camera leaves ~1° of sky-below-horizon exposed). Sample the
// scattering at a grazing direction instead and fade to groundColor as the
// ray steepens; the band under the horizon now continues the horizon haze.
//
// 2026-08-11 — that fix corrected the VALUE and left the band FLAT, and the
// flatness is what the exterior hero has been showing ever since: a dead level
// warm plateau, +/-1 code value over 24 px, butting the lawn with a 25-code
// one-row step (PHOTOREALISM #10). Two things were wrong and both are here.
//
//  1. The haze was read 0.004 rad ABOVE the horizon, which is a measurably
//     BRIGHTER sample than the last above-horizon pixel: the band stepped UP
//     ~5 code values crossing the horizon and then held. It is read at the
//     true horizon now. A horizontal ray never meets the planet from a viewer
//     above it (`rayGroundHit`: b = 0, c = 2R + 1 > 0, disc < 0), so elevation
//     0 still gets the full-length march the clamp was protecting — and it is
//     continuous with the pixel above it by construction.
//
//  2. Below the horizon NO atmosphere sample can continue the gradient: the
//     march hits the planet, which is precisely what produced the near-black
//     stripe. The continuation that is physical is AIRLIGHT. A ray depressed
//     by theta ends on ground at distance ~ h/theta, and what comes back is
//     the ground's own colour attenuated by exp(-d/D) with scattered air
//     making up the rest — written in the one variable a sky shader has:
//
//         groundWeight = exp(-k / theta)
//
//     Pure haze at the horizon (theta -> 0), the ground's own colour a
//     fraction of a degree down. The `smoothstep(0, 0.35)` this replaces
//     spread that transition over 20 degrees, so the first degree — the ONLY
//     part a finite ground plane ever leaves visible — was 99.6 % haze, i.e.
//     one constant. `groundColor.w` (`Params.groundBlend`) scales k: it has
//     always been DOCUMENTED as exactly this control ("how fast rays below the
//     horizon fade to ground colour") and was read by neither atmosphere.
//
// Same march, same dome bake, no extra tap: zero per-frame cost.
//
// NOT a mirror of the ray. Mirroring is the other obvious way to "un-flatten"
// the band, and on this sky it is backwards: nishita DARKENS toward the
// horizon here (the exterior hero falls 182 -> 164 over the 13 px above it),
// so a mirrored band climbs back to ~197 and meets the 125 lawn with a
// ~72-code step — three times the seam it set out to remove.
/// The sky grade: chroma about luma, then a blue lift with the luma put back. Both preserve
/// the sky's exposure, so the grade never re-meters the scene under it.
inline float3 skyGraded(float3 s, float sat, float blueLift) {
    const float3 kLuma = float3(0.2126f, 0.7152f, 0.0722f);
    float l = dot(s, kLuma);
    float3 g = max(mix(float3(l), s, sat), 0.0f);
    g.z *= max(blueLift, 0.0f);
    float l2 = dot(g, kLuma);
    return (l2 > 1e-6f) ? g * (l / l2) : g;
}

// Night SKY GLOW (physical model) — the atmosphere's own light after dark, added to the
// nishita sky: (1) the MOONLIT SKY — the same single-scatter march with the moon as the
// light, normalised so a full moon's zenith is kFullMoonSkyF0 (≈18 mag/arcsec²) in the
// night's radiance unit and scaled by the lunar phase law; (2) AIRGLOW — the ~90 km
// emission layer, brighter toward the horizon by the van Rhijn path-length factor, dimmed
// by extinction; (3) ZODIACAL LIGHT — sunlight off interplanetary dust, a cone along the
// ecliptic that fades with elongation from the (below-horizon) sun.
constant float kNishitaZenithRef = 0.010f;   // nishitaScatter zenith luminance, sun at ~50°, intensity 1
inline float3 nightSkyGlow(float3 rayDir, constant SkyUniforms &u) {
    float rad = u.nightSkyB.z;
    if (u.nightSkyA.x < 0.5f || rad <= 0.0f || u.nightParams.w <= 0.0f) return float3(0.0f);
    float sinEl = max(rayDir.y, 0.0f);
    float3 g = float3(0.0f);
    if (u.nightParams.y > 0.0f) {
        float3 moonD = normalize(u.moonParams.xyz);
        float flux = nightMoonPhaseFlux(dot(normalize(u.nightSkyC.xyz), -moonD));
        // Under civil/nautical twilight the sun's own scatter is thousands of times brighter
        // than the moon's, so its march is skipped until the sun is ~6° down and cross-faded
        // in by ~10° (then the sun's march itself is skipped past 18°: never two at full cost).
        float w = smoothstep(0.10f, 0.17f, u.sunDir.y);
        if (moonD.y > -0.3f && w > 0.0f)
            g += nishitaScatter(rayDir, -moonD, 1.0f) * (w * kFullMoonSkyF0 * rad * flux / kNishitaZenithRef);
    }
    float3 ext = nightExtinction(sinEl);
    float ag = u.nightSkyB.w;
    if (ag > 0.0f) {
        float cos2 = 1.0f - sinEl * sinEl;
        float vanRhijn = rsqrt(max(1.0f - 0.9723f * cos2, 1e-4f));
        g += float3(0.78f, 1.0f, 0.80f) * (ag * rad * kDarkSkyF0 * vanRhijn) * ext;
    }
    float zl = u.nightSkyC.w;
    if (zl > 0.0f) {
        float3 toSun = -normalize(u.sunDir.xyz);
        float3 pole = nightQuatRotateInverse(u.nightSkyD, float3(0.0f, -0.39777f, 0.91748f));
        float elong = acos(clamp(dot(rayDir, toSun), -1.0f, 1.0f));
        float beta = asin(clamp(dot(rayDir, pole), -1.0f, 1.0f));
        float w = 0.17f + 0.2f * elong;
        float z = 2.5f * pow(max(elong, 0.35f) / 0.7f, -2.2f) * exp(-(beta * beta) / (w * w));
        g += float3(1.0f, 0.95f, 0.85f) * (zl * rad * kDarkSkyF0 * z) * ext;
    }
    return g;
}

inline float3 nishitaAtmosphereColor(float3 rayDir, constant SkyUniforms &u) {
    float intensity = max(0.0f, u.atmosphereParams.y);
    float sat = u.skyGrade.x;
    // Physical night sky: once the sun is 18° down (astronomical night) its single-scatter
    // march is exactly black — the whole 60 km shell is in the Earth's shadow — so skip it
    // and spend the budget on the moon's march instead (`nightSkyGlow`).
    bool sunMarch = !(u.nightSkyA.x > 0.5f && u.sunDir.y > 0.309f);
    if (rayDir.y >= 0.0f) {
        float3 s = (sunMarch ? nishitaScatter(rayDir, u.sunDir.xyz, intensity) : float3(0.0f))
                 + nightSkyGlow(rayDir, u);
        // Chroma about luma, luma preserved — the sky's exposure does not move with it.
        return skyGraded(s, sat, u.skyGrade.y);
    }
    // The horizontal component, guarded: at the NADIR (0, -1, 0) — which the full-sphere
    // dome bake does sample — x and z are both zero and `normalize` of a zero vector is NaN.
    // The old clamp could not hit this (its y was 0.004, never a zero vector); a NaN here
    // would poison the dome texel and, through it, the IBL convolution. The fallback
    // direction is arbitrary because at the nadir the ground weight is ~1 anyway.
    float2 hxz = float2(rayDir.x, rayDir.z);
    float hlen = length(hxz);
    float3 grazing = (hlen > 1e-5f) ? float3(hxz.x / hlen, 0.0f, hxz.y / hlen)
                                    : float3(1.0f, 0.0f, 0.0f);
    float3 horizonHaze = (sunMarch ? nishitaScatter(grazing, u.sunDir.xyz, intensity) : float3(0.0f))
                       + nightSkyGlow(grazing, u);
    horizonHaze = skyGraded(horizonHaze, sat, u.skyGrade.y);
    // Airlight scale in radians of depression: the 1/e point of the haze->ground
    // handover. 0.0035 rad (~0.2 deg) is one eye-height in ~3 km of haze, which is
    // where a real horizon hands over.
    const float kAirlightTheta = 0.0035f;
    float theta = max(-rayDir.y, 1e-6f);
    float ground = exp(-(kAirlightTheta / max(u.groundColor.w, 0.05f)) / theta);
    return mix(horizonHaze, u.groundColor.xyz, ground);
}

// ── Lava-lamp sky (Params.lavaLamp) ──────────────────────────────────────────────────────
// Metaball "wax" on the sky dome: blobs rise slowly up the sphere, wander sideways, swell and
// shrink, and merge where their fields overlap (Σ r²/d² over chord distance on the unit sphere),
// over a glowing vertical gradient — a lamp's warm base fading to a darker top. Emissive HDR, so
// it lights whatever the dome feeds (IBL) as well as filling the view.
inline float3 lavaLampSky(float3 rd, float tIn, constant SkyUniforms &u) {
    float t = tIn * (u.lavaBG.w > 0.0f ? u.lavaBG.w : 1.0f);
    float scale = max(u.lavaA.z, 0.2f);
    float field = 0.0f, wA = 0.0f;
    // Enough wax that any view of the sky holds several blobs: they live in the band a camera
    // actually sees (from just below the horizon to ~75°), rise through it and wrap around.
    // The wax lives where a camera looks: from just below the horizon to ~37° up (a flight or
    // eye-level camera sees 0–25°); spreading it to the zenith left ~1 blob per view.
    constexpr int N = 48;
    for (int i = 0; i < N; ++i) {
        float fi = float(i);
        float speed = 0.018f + 0.02f * fract(fi * 0.618f);
        float el = -0.10f + fract(fi * 0.371f + t * speed) * 0.75f;
        float az = fi * 2.3999f + 0.5f * sin(t * 0.05f + fi * 1.7f);
        float3 c = float3(cos(az) * cos(el), sin(el), sin(az) * cos(el));
        float R = scale * (0.21f + 0.07f * sin(t * 0.17f + fi * 2.1f)) * (0.6f + 0.4f * fract(fi * 0.83f));
        float3 d = rd - c;
        // COMPACT metaball kernel (Wyvill, zero beyond R): blobs stay distinct until they touch,
        // then merge. The 1/d² kernel's long tails summed over every blob flooded the whole
        // sky above threshold — one wax, no blobs (tuned against a CPU preview of this exact
        // field over 40 random headings × times: never an empty view — 36 blobs over 0–47° left
        // a 4 % minimum; 48 over −6…37° keeps the minimum near 15 %, the median near 40 %).
        float k = max(0.0f, 1.0f - dot(d, d) / (R * R));
        k *= k;
        field += k;
        wA += k * step(0.5f, fract(fi * 0.5f + 0.25f));   // alternate blobs take colour B
    }
    float blob = smoothstep(0.30f, 0.45f, field);
    float3 wax = mix(u.lavaBlobA.rgb, u.lavaBlobB.rgb, saturate(wA / max(field, 1e-4f)));
    // Hot core where the wax is thick or two blobs pile up.
    wax *= 1.0f + 0.5f * smoothstep(0.6f, 1.4f, field);
    float base = 1.0f - smoothstep(-0.2f, 0.9f, rd.y);
    // The fluid behind the wax: darker than the wax so the blobs read, glowing at the lamp's base.
    float3 bg = u.lavaBG.rgb * (0.12f + 0.55f * base);
    return mix(bg, wax, blob) * max(u.lavaA.w, 0.0f);
}

// ── Cirrus: a thin, high ice veil (Params.cirrusCoverage) ────────────────────────────────
// The cumulus march is a slab of rounded volumes; the fibrous "mares' tails" that streak a
// clear California sky are a different object — ice at 7–10 km, optically THIN (τ ≈ 0.05–0.5),
// sheared by the jet into long fibres. Thin means single scattering is exact enough and no march
// is needed: intersect the layer plane once, read wind-stretched fibre noise from the same baked
// volume the deck uses, and composite L = (E_sun·phase + fill)·(1 − e^−τ) over the sky with the
// path length through the layer growing toward the horizon (τ ∝ 1/sinθ). Ice's phase is sharply
// forward (g ≈ 0.75) with an isotropic share, and its fill is WHITE: the veil is lit by the whole
// bright hemisphere under it (haze, cloud, ground), not by the blue above.
inline float3 cloudSunIrradiance(constant SkyUniforms &u) {
    return (u.skyGrade.z > 0.5f) ? u.cloudLitSun.xyz : u.sunColor.xyz * u.sunColor.w;
}
inline float3 cloudAmbientBase(constant SkyUniforms &u) {
    return (u.skyGrade.z > 0.5f) ? u.cloudLitAmbient.xyz : u.skyZenith.xyz;
}

// One cirrus layer — the wind-streaked VEIL, or (threads) the dense curled filaments that flow.
inline float3 cirrusLayer(float3 sky, float3 ro, float3 rayDir, float time, constant SkyUniforms &u,
                          texture3d<float, access::sample> noiseVol, bool threads) {
    // cirrusC.w: the threads' coverage gain over the veil (0 = legacy: same coverage).
    float cov = threads ? min(1.0f, u.cirrusA.x * (u.cirrusC.w > 0.0f ? u.cirrusC.w : 1.0f)) : u.cirrusA.x;
    if (cov <= 0.0f || rayDir.y <= 0.004f) return sky;
    float t = (u.cirrusA.z - ro.y) / rayDir.y;
    if (t <= 0.0f) return sky;
    float2 p = ro.xz + rayDir.xz * t;
    // Same advection as the deck (the host folds any floating-origin shift into it), faster aloft.
    p -= u.windTime.xy * u.windTime.z * u.windTime.w * u.cirrusB.w;
    float2 along = normalize(u.cirrusB.xy + float2(1e-6f, 0.0f));
    float2 across = float2(-along.y, along.x);
    float tile = float(noiseVol.get_width());
    float stretch = max(u.cirrusB.z, 1.0f);
    // P: noise-texel space, x along the jet, y across it.
    // THREADS mode: denser, more strongly curled filaments that FLOW — the fibres stream along the
    // jet and the curl field itself drifts and morphs, so the threads undulate as if blown.
    float freq = threads ? max(u.cirrusC.z, 0.05f) : 1.0f;
    float2 P = float2(dot(p, along), dot(p, across)) * (u.cirrusA.w * freq);
    float flow = threads ? time * u.cirrusC.y : 0.0f;
    P.x -= flow * 1.6f;                                   // stream along the jet
    float2 warpShift = float2(flow * 0.45f, flow * 0.08f); // the curl field drifts slower …
    float morph = threads ? flow * 0.004f : 0.0f;          // … and slowly changes shape
    // Hooks and curls: a two-level warp ACROSS the streaks, slow then finer.
    float warpA = threads ? 6.0f : 4.0f, warpB = threads ? 1.8f : 1.0f;
    P.y += (noiseVol.sample(volSampler, float3((P + warpShift) * 0.18f, 0.37f + morph) / tile).r - 0.5f) * warpA;
    P.y += (noiseVol.sample(volSampler, float3((P + warpShift * 1.7f) * float2(0.35f, 0.9f), 0.91f + morph) / tile).r - 0.5f) * warpB;
    // BANDS: broad, very elongated sheets — the veils that sweep across a whole sky.
    float bands = noiseVol.sample(volSampler, float3(P.x / (stretch * 2.2f), P.y * 0.30f, 0.53f) / tile).r;
    bands = 0.7f * bands + 0.3f * noiseVol.sample(volSampler, float3(P.x / (stretch * 1.1f), P.y * 0.62f, 0.61f) / tile).r;
    // FIBRES within them: stretched mid-scale streaks, three octaves, finer = more stretched.
    float n = 0.0f, amp = 0.55f, sumA = 0.0f;
    float2 qq = float2(P.x / stretch, P.y);
    for (int o = 0; o < 3; ++o) {
        n += amp * noiseVol.sample(volSampler, float3(qq, 0.13f + 0.29f * float(o)) / tile).r;
        sumA += amp; amp *= 0.5f;
        qq = float2(qq.x * 1.7f, qq.y * 3.1f) + float2(17.3f, 5.1f);
    }
    n /= sumA;
    // Coverage over a WIDE soft band, so a sheet feathers out rather than ending on a line.
    float body = bands * 0.62f + n * 0.38f;
    float d = smoothstep(1.0f - cov, 1.0f - cov + 0.42f, body);
    // Striations: fine, hard-stretched texture along the jet, carving the sheet into fibres.
    float fib = noiseVol.sample(volSampler, float3(P.x * 0.9f, P.y * 9.0f, 0.71f) / tile).g;
    d *= mix(0.35f, 1.0f, smoothstep(0.25f, 0.85f, fib));
    // Thickness: slow variation — dense bright cores, thin translucent edges.
    float thick = noiseVol.sample(volSampler, float3(P * float2(0.06f, 0.4f) + 13.0f, 0.44f) / tile).r;
    d *= 0.35f + 1.3f * thick * thick;
    // Toward the horizon the plane's texels foreshorten to less than a pixel and the fibres alias
    // into dashes; the veil also sinks into the haze there. Fade it out over the last ~6°.
    d *= smoothstep(0.02f, 0.11f, rayDir.y);
    if (d <= 1e-4f) return sky;
    float tau = u.cirrusA.y * d / max(rayDir.y, 0.07f);
    float cosT = clamp(dot(rayDir, -u.sunDir.xyz), -1.0f, 1.0f);
    float phase = 0.6f * hg(cosT, 0.75f) + 0.4f * (1.0f / (4.0f * M_PI_F));
    float3 amb = cloudAmbientBase(u);
    float  ambL = dot(amb, float3(0.2126f, 0.7152f, 0.0722f));
    float3 L = (cloudSunIrradiance(u) * phase + float3(ambL) * 1.6f) * u.cirrusTint.rgb;
    // Physical night sky: the veil scatters MOONLIGHT too — in the moonlit sky's own units
    // (the irradiance whose Rayleigh single-scatter is `nightSkyGlow`'s moonlit zenith), with
    // the moonlit sky as its fill. Without it the ice after dark was lit by nothing and
    // printed as dark bands across the moonlit blue; with it the fibres glow silver around
    // the moon — the real aureole-through-cirrus.
    if (u.nightSkyA.x > 0.5f && u.nightParams.y > 0.0f && u.nightParams.w > 0.0f && u.nightSkyB.z > 0.0f) {
        float3 moonD = normalize(u.moonParams.xyz);
        if (moonD.y > -0.05f) {
            float E = kFullMoonSkyF0 * u.nightSkyB.z * nightMoonPhaseFlux(dot(normalize(u.nightSkyC.xyz), -moonD));
            float cosM = clamp(dot(rayDir, moonD), -1.0f, 1.0f);
            float phaseM = 0.6f * hg(cosM, 0.75f) + 0.4f * (1.0f / (4.0f * M_PI_F));
            L += (E / kNishitaZenithRef * phaseM * nightExtinction(moonD.y) + float3(1.6f * 1.5f * E))
               * u.cirrusTint.rgb;
        }
    }
    float a = 1.0f - exp(-tau);
    return sky * (1.0f - a) + L * a;
}

// The cirrus as the host asked for it: cirrusC.x (or the in-view per-frame override `threadsW`)
// is how much of it is THREADS. 0 or 1 draws one layer (byte-identical to the old switch);
// between, both are drawn and blended — only for the length of a fade.
inline float3 applyCirrus(float3 sky, float3 ro, float3 rayDir, float time, constant SkyUniforms &u,
                          texture3d<float, access::sample> noiseVol, float threadsW) {
    float w = saturate(threadsW);
    if (w <= 0.0f) return cirrusLayer(sky, ro, rayDir, time, u, noiseVol, false);
    if (w >= 1.0f) return cirrusLayer(sky, ro, rayDir, time, u, noiseVol, true);
    return mix(cirrusLayer(sky, ro, rayDir, time, u, noiseVol, false),
               cirrusLayer(sky, ro, rayDir, time, u, noiseVol, true), w);
}

// ── Cloud lighting from the atmosphere (Params.cloudLightingFromAtmosphere) ─────────────
// The cloud march's legacy sun (`sunColor·sunIntensity`) and fill (`skyZenith·ambient`) are
// hand-set numbers in units unrelated to the nishita sky's `intensity·∫β·phase·T` — under a
// physical sky that left every cloud several times DARKER than the blue behind it. This pass
// lights the deck in the sky's own units, on the GPU, from the same march the sky pixels use:
// the solar irradiance reaching the deck base through the air, and the mean radiance of the
// graded sky above + the sun- and sky-lit ground below. One 16-thread threadgroup per sky
// update; results land in `SkyUniforms.cloudLitSun/cloudLitAmbient`, which every later cloud
// dispatch on the queue reads.

inline float3 nishitaTransmittanceToSun(float altitude, float3 toSun) {
    float3 o = float3(0.0f, kEarthRadius + max(altitude, 1.0f), 0.0f);
    float tl = raySphereExit(o, toSun, kAtmosRadius);
    if (tl <= 0.0f) return float3(1.0f);
    constexpr int N = 16;
    float seg = tl / float(N);
    float3 od = float3(0.0f);
    for (int j = 0; j < N; ++j) {
        float3 p = o + toSun * ((float(j) + 0.5f) * seg);
        float h = length(p) - kEarthRadius;
        if (h < 0.0f) return float3(0.0f);          // the planet shades the deck
        od += float3(exp(-h / kRayleighH), exp(-h / kMieH), ozoneDensity(h)) * seg;
    }
    return exp(-(kBetaR * od.x + kBetaM * 1.1f * od.y + kBetaO * od.z));
}

kernel void volSkyCloudLight(device SkyUniforms &u [[buffer(0)]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float3 radiance[16];
    threadgroup float  weight[16];
    float intensity = max(0.0f, u.atmosphereParams.y);
    float3 s = float3(0.0f);
    float  w = 0.0f;
    // Thread 1 = zenith; 2…13 = two rings of six (low ring weighted up — it is most of
    // the solid angle a deck's flank sees). Threads 0, 14, 15 contribute nothing.
    if (tid >= 1u && tid <= 13u) {
        float3 d = float3(0.0f, 1.0f, 0.0f);
        w = 1.0f;
        if (tid >= 2u) {
            uint  k  = tid - 2u;
            float a  = float(k % 6u) / 6.0f * 2.0f * M_PI_F;
            float el = (k < 6u) ? 0.20f : 0.75f;
            d = normalize(float3(cos(a) * cos(el), sin(el), sin(a) * cos(el)));
            w = (k < 6u) ? 1.2f : 1.0f;
        }
        s = skyGraded(nishitaScatter(d, u.sunDir.xyz, intensity), u.skyGrade.x, u.skyGrade.y) * w;
    }
    radiance[tid] = s;
    weight[tid] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;
    float3 sum = float3(0.0f);
    float  wsum = 0.0f;
    for (uint i = 0u; i < 16u; ++i) { sum += radiance[i]; wsum += weight[i]; }
    float3 skyMean = sum / max(wsum, 1e-6f);
    float3 toSun = -u.sunDir.xyz;
    float3 sun = intensity * nishitaTransmittanceToSun(u.cloudSlab.x, toSun);
    // Lambertian ground under the deck (albedo in skyHorizon.xyz in this mode).
    float3 groundRad = u.skyHorizon.xyz * (sun * max(toSun.y, 0.0f) + M_PI_F * skyMean) / M_PI_F;
    u.cloudLitSun     = float4(sun, 1.0f);
    u.cloudLitAmbient = float4(0.5f * (skyMean + groundRad), 1.0f);
}

// Sun disk — a soft circular hotspot anchored to the sun's 3D direction.
//
// History note: an earlier revision drew this in equirect uv-space to dodge
// a "twin suns at opposite screen edges" bug. That bug was caused by
// SceneKit flat-mapping the equirect when bound as
// `scene.background.contents` — content near the u = 0 / u = 1 seam appeared
// at *both* left & right of the rendered viewport simultaneously. The fix
// is in the Swift side now: `VolumetricCloudRenderer.makeSkyDome()` wraps
// the texture onto an inside-facing SCNSphere instead, which is a true
// panoramic projection (u = 0 and u = 1 are physically the same point on
// the sphere). With the dome, dot-product 3D disks behave correctly across
// the seam, and we can shade the sun by ray direction directly.
inline float3 sunDisk(float3 rayDir, constant SkyUniforms &u) {
    float3 sp = -u.sunDir.xyz;               // direction TO the sun
    if (sp.y < -0.02f) return float3(0.0f);  // sun below horizon

    float cosT = clamp(dot(rayDir, sp), -1.0f, 1.0f);
    // Angular radii (cos thresholds):
    //   core ~3.6° wide  → cos(3.6°)  ≈ 0.998
    //   core inner ~1.5° → cos(1.5°)  ≈ 0.99966
    //   halo ~11° wide   → cos(11°)   ≈ 0.9816
    //   halo inner = core outer
    float core = smoothstep(0.998f,  0.99966f, cosT) * 8.0f;
    float halo = smoothstep(0.9816f, 0.998f,   cosT) * 0.6f;
    return u.sunColor.xyz * (core + halo) * u.sunColor.w;
}

// ── Night sky ────────────────────────────────────────────────────────────────

// Procedural star field. Each cell on a latitude–longitude grid has a small
// chance of containing a star; within the cell the star is placed at a
// random sub-cell offset so they read as a Poisson disk, not a lattice.
// `brightness` is nightParams.x — 0 for day, 1 for a clear dark sky.
inline float3 starField(float3 rayDir, float brightness) {
    if (brightness <= 0.0f) return float3(0.0f);

    // Map ray direction to latitude/longitude in degrees so one cell ≈ 0.4°.
    float az = atan2(rayDir.z, rayDir.x);
    float el = asin(clamp(rayDir.y, -1.0f, 1.0f));
    float2 uv = float2((az + M_PI_F) * (450.0f / (2.0f * M_PI_F)),
                        (el + M_PI_F * 0.5f) * (225.0f / M_PI_F));
    int2 ip = int2(floor(uv));
    float2 fp = uv - float2(ip);

    float3 result = float3(0.0f);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 cell = ip + int2(dx, dy);
            uint h = hash3(int3(cell.x, cell.y, 17));
            // ~4% chance of a star per cell (10/256 ≈ 3.9%).
            if ((h & 0xFFu) < 10u) {
                float2 starPos = float2(
                    float((h >> 8u)  & 0xFFu) / 255.0f,
                    float((h >> 16u) & 0xFFu) / 255.0f
                );
                float2 d   = fp - (float2(dx, dy) + starPos);
                float  d2  = dot(d, d);
                // Two-lobe profile: tight bright core + soft halo. Without
                // the halo, a sub-texel star either hits or misses a texel
                // and the equirect bilinear filter smears it into a chunky
                // square. The halo guarantees the star always footprints
                // several texels and reads as a sparkle, especially after
                // the camera's bloom pass hits it.
                float  core = exp(-d2 * 160.0f);
                float  halo = exp(-d2 * 6.0f) * 0.18f;
                float  lum  = core + halo;
                // Magnitude: 4 bits (h>>28) → 0..15 → 0..1; brighter is rarer.
                float  mag = 1.0f - float((h >> 28u) & 0xFu) / 15.0f;
                // Subtle colour temperature: faint blue tint to cooler stars.
                float3 col = mix(float3(1.0f, 0.92f, 0.72f),
                                 float3(0.78f, 0.87f, 1.0f), mag * mag);
                result += col * lum * (0.25f + 0.75f * mag);
            }
        }
    }
    return result * brightness * 2.2f;
}

// The dome's moon — the SHARED sphere-lit disk (IlluminatoramaNightSky.h `nightMoonDisk`),
// replacing a private copy whose phase shading used the ANGLE around the disc centre
// instead of position across it (a pie-slice "Pac-Man" terminator with a flat dot where
// the angle was undefined). Lit from nightSkyC.xyz: the effective sun that shows the
// host's `moonPhase` — so legacy hosts keep their phase and size, on a real sphere.
inline float3 moonDisk(float3 rayDir, constant SkyUniforms &u, float pixAngle) {
    float intensity = u.nightParams.y;
    if (intensity <= 0.0f) return float3(0.0f);
    return nightMoonDisk(rayDir, normalize(u.moonParams.xyz), u.nightSkyC.xyz,
                         u.nightSkyA.y, intensity, pixAngle);
}

/// This renderer's view of the shared night-sky currency — the ONE place that knows
/// SkyUniforms' packing (the twin of `frameNightSky` / `glassNightSky`).
inline NightSkyParams skyNightParams(constant SkyUniforms &u, float clock) {
    NightSkyParams p;
    p.moonDir        = normalize(u.moonParams.xyz);
    p.toSun          = u.nightSkyC.xyz;
    p.starBrightness = u.nightParams.x;
    p.moonIntensity  = u.nightParams.y;
    p.moonAngRadius  = u.nightSkyA.y;
    p.model          = u.nightSkyA.x;
    p.moonHalo       = u.nightSkyA.z;
    p.earthshine     = u.nightSkyA.w;
    p.milkyWay       = u.nightSkyB.x;
    p.twinkle        = u.nightSkyB.y;
    p.radiance       = u.nightSkyB.z;
    p.clock          = clock;
    p.celestial      = u.nightSkyD;
    return p;
}

/// Stars + moon for one sky ray, for every kernel in this file. Legacy: the dome's star
/// grid (or the screen-res grid when `sharpStars`) and the moon, faded by nightBlend.
/// Physical: the shared physical celestials — stars and the Milky Way only once the sun is
/// down (they wash out against the twilight on their own, so they are NOT scaled by
/// nightBlend), the moon always (a daytime moon is a pale disk, as it is outdoors).
/// `background` = the sky radiance behind the celestials (the washout reference).
inline float3 skyCelestials(float3 rayDir, constant SkyUniforms &u, float pixAngle,
                            float3 background, float clock, bool sharpStars) {
    float nightBlend = u.nightParams.w;
    if (u.nightSkyA.x < 0.5f) {
        if (nightBlend <= 0.0f || rayDir.y <= -0.05f) return float3(0.0f);
        float3 stars = sharpStars ? nightStarField(rayDir, u.nightParams.x, pixAngle)
                                  : starField(rayDir, u.nightParams.x);
        return (stars + moonDisk(rayDir, u, pixAngle)) * nightBlend;
    }
    NightSkyParams p = skyNightParams(u, clock);
    if (nightBlend <= 0.0f) { p.starBrightness = 0.0f; p.milkyWay = 0.0f; }
    return nightCelestials(rayDir, p, pixAngle, background);
}

// ── Light march ─────────────────────────────────────────────────────────────
//
// At each cloud step we shoot 4 short rays toward the sun and accumulate
// optical depth so we can shade the step by how much sunlight survives the
// trip through the cloud body. Distances grow geometrically so the kernel
// captures both fine detail near the sample and bulk extinction further in.

inline float lightMarch(float3 pos,
                         float3 L,              // unit vector TOWARD the light
                         float lightSteps,
                         constant SkyUniforms &u,
                         texture3d<float, access::sample> noiseVol) {
    float thickness = (u.cloudSlab.y - u.cloudSlab.x);
    int steps = max(1, int(lightSteps));
    float stride = thickness / float(steps) * 0.40f;

    float od = 0.0f;
    for (int i = 0; i < steps; ++i) {
        float t = stride * (float(i) + 0.5f) * (1.0f + 0.6f * float(i));
        float3 sp = pos + L * t;
        if (sp.y > u.cloudSlab.y || sp.y < u.cloudSlab.x) break;
        float d = cloudDensity(sp, u, noiseVol);
        od += d * stride;
    }
    return od;
}

// Optical depth from a march position TOWARD a finite point light (a burst /
// shell head), so the burst's in-scatter can be occluded by the cloud between
// the sample and the light. This is what turns a flat "glow added everywhere"
// into REAL volumetric lighting from within: a burst inside a bank lights the
// near interior bright and leaves its far side / the crevices shadowed (the
// backlit silver-lining rim + shafts), and a burst behind a thick bank is
// blocked instead of glowing through it. Distance-aware (unlike the directional
// `lightMarch`): marches toward the actual world position, capped at a couple
// of slab thicknesses so a distant burst's march stays local and cheap. Returns
// optical depth in the same UNSCALED units as `lightMarch` (caller applies absK).
inline float burstLightMarch(float3 pos, float3 lightPos, int steps,
                             constant SkyUniforms &u,
                             texture3d<float, access::sample> noiseVol) {
    float3 toL = lightPos - pos;
    float dist = length(toL);
    if (dist < 1e-3f) return 0.0f;
    float3 dir = toL / dist;
    float marchDist = min(dist, (u.cloudSlab.y - u.cloudSlab.x) * 2.0f);
    float stride = marchDist / float(max(1, steps));
    float od = 0.0f;
    for (int i = 0; i < steps; ++i) {
        float t = stride * (float(i) + 0.5f);
        od += cloudDensity(pos + dir * t, u, noiseVol) * stride;
    }
    return od;
}

// ── Burst-light in-scatter ──────────────────────────────────────────────────
//
// Point lights INSIDE / BELOW the cloud volume (fireworks bursts, rising
// shells). At each march step we sum single-scattered radiance from every
// live light: quadratic radial falloff inside the light's influence radius ×
// the burst's age-decay envelope. Isotropic phase — with many coloured
// lights at arbitrary positions a directional HG lobe per light costs more
// than it reads; isotropic single scattering is the honest cheap model and
// is what makes a burst below the deck glow the cloud BASE while a shell
// inside the slab lights the interior around itself.
//
// `VSBurstLight` is a byte-for-byte mirror of the shared 48-byte BurstLight
// struct in BurstLightField.metal / BurstLightField.swift (structs don't
// link across Metal translation units, so the mirror is local; the layout
// is the contract). Falloff math mirrors `sampleBurstField` there.

struct VSBurstLight {
    float4 positionIntensity;  // xyz = world position (m), w = peak intensity
    float4 colorAge;           // rgb = base colour, a = age (s) since detonate
    float4 radiusLifePad;      // x = max influence radius (m), y = lifespan (s)
};

// Equirect debug background — a tiled neon checker keyed on direction, painted
// BEHIND the clouds when atmosphereParams.z > 0.5. Cloud transmittance
// modulates how much shows through, so sky GAPS glow neon and cloud BODIES
// occlude it: a direct read of the deck's silhouette + density independent of
// the fireworks. (debug-containment-neon technique.)
inline float3 neonDebugBackground(float3 dir) {
    // ~18 cells around the azimuth, ~9 up the elevation.
    float az = atan2(dir.z, dir.x);                 // (-π, π]
    float el = asin(clamp(dir.y, -1.0f, 1.0f));     // (-π/2, π/2)
    float u = (az + 3.14159265f) / (2.0f * 3.14159265f);
    float v = (el + 1.5707963f) / 3.14159265f;
    int cx = int(floor(u * 18.0f));
    int cy = int(floor(v * 9.0f));
    int parity = (cx + cy) & 1;
    // Two saturated neons per parity, varied by cell so adjacent cells differ.
    float3 a = (parity == 0) ? float3(1.0f, 0.0f, 1.0f)   // magenta
                             : float3(0.0f, 1.0f, 0.4f);  // green
    float3 b = (parity == 0) ? float3(0.0f, 0.8f, 1.0f)   // cyan
                             : float3(1.0f, 0.55f, 0.0f); // orange
    float pick = float((cx * 7 + cy * 13) & 1);
    return mix(a, b, pick) * 0.6f;
}

// Per-RAY cap on simultaneously-marched lights. The cull below keeps only
// lights whose influence sphere intersects this ray's slab segment — for a
// typical fireworks frame that's 0–3 of the buffer's ≤24, so the per-step
// cost is near-zero instead of count×steps (the naive per-step loop over a
// 64-light buffer was ~8 billion light evaluations per 2 k dome render — the
// GPU-saturating mistake this cull exists to prevent).
constant constexpr uint kVSMaxRayLights = 8u;

// Evaluate one light's single-scatter contribution at a march position.
// Falloff mirrors `sampleBurstField` in BurstLightField.metal: quadratic
// radial inside maxR × the burst's flash-then-decay age envelope.
inline float3 burstLightEval(float3 worldPos, VSBurstLight L) {
    float intensity = L.positionIntensity.w;
    float maxR      = L.radiusLifePad.x;
    float life      = L.radiusLifePad.y;

    float3 d = worldPos - L.positionIntensity.xyz;
    float dist = length(d);
    if (dist >= maxR) return float3(0.0f);

    // Bounded ~inverse-square falloff. A quartic (rT⁴) collapsed to near-zero by
    // mid-radius, so a burst detonating in a clear GAP barely reached the cumulus
    // banks a few tens of metres away (scene-reviewer: burst colour not reaching
    // the deck). Quadratic (rT²) is broader and closer to a real 1/d² emitter, so
    // the burst lights the banks around it — and the burst-occlusion light-march
    // now shapes that reach realistically (near side bright, far side / crevices
    // shadowed), so the broad form no longer reads as a flat even wash.
    float rT = 1.0f - dist / maxR;
    float rFall = rT * rT;

    float ageT = L.colorAge.a / life;
    float ageScale;
    if (ageT < 0.08f) {
        ageScale = 1.0f;
    } else {
        float k = (ageT - 0.08f) / 0.92f;
        ageScale = (1.0f - k);
        ageScale = ageScale * ageScale;
    }

    return L.colorAge.rgb * (intensity * rFall * ageScale);
}

// ── Main kernel ─────────────────────────────────────────────────────────────

kernel void volSkyRender(
    texture2d<float, access::write>  outTex      [[texture(0)]],
    texture3d<float, access::sample> noiseVol    [[texture(1)]],
    constant SkyUniforms &u                      [[buffer(0)]],
    device const VSBurstLight* burstLights       [[buffer(1)]],
    uint2 gid                                    [[thread_position_in_grid]]
) {
    uint W = outTex.get_width();
    uint H = outTex.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // Flat studio background — bypass atmosphere/sun/clouds/stars entirely and write one
    // constant colour to every texel. Feeds BOTH the visible dome and the small IBL bake
    // (same kernel, two dispatch resolutions — see `VolumetricCloudRenderer.render(params:)`),
    // so this flattens ambient lighting too, not just what the camera sees behind the scene.
    if (u.studioParams.w > 0.5f) {
        outTex.write(float4(u.studioParams.xyz, 1.0f), gid);
        return;
    }

    // Equirect: u in [0, 2π), v in [+π/2, -π/2] so the texture's top row maps
    // to looking straight up. SceneKit wraps an equirect background such that
    // the +X direction in world space lines up with the texture's seam (u=0).
    float2 uv = (float2(gid) + 0.5f) / float2(W, H);
    float az = uv.x * (2.0f * 3.14159265f);
    float el = (0.5f - uv.y) * 3.14159265f;          // +π/2 at top, -π/2 at bottom

    float ce = cos(el);
    float3 rayDir = float3(ce * cos(az), sin(el), ce * sin(az));

    // ── Atmosphere + sun ─────────────────────────────────────────────
    bool debugBG = u.atmosphereParams.z > 0.5f;
    float3 sky;
    float lavaW = 0.0f;
    float3 lavaCol = float3(0.0f);
    if (debugBG) {
        // Neon tiled background BEHIND the clouds — isolates the deck
        // silhouette + density (sky gaps glow, cloud bodies occlude).
        sky = neonDebugBackground(rayDir);
    } else {
        // Mode selects the faked gradient (0) or the physical Nishita march (1).
        // lavaA.x = the lamp's weight: 1 = the lamp alone (the atmosphere is skipped); between 0
        // and 1 the full sky is rendered and cross-faded to the lamp at every write below.
        lavaW = saturate(u.lavaA.x);
        if (lavaW > 0.0f) lavaCol = lavaLampSky(rayDir, u.lavaA.y, u);
        if (lavaW >= 1.0f) {
            sky = lavaCol;   // the lamp replaces the atmosphere: skip it
        } else {
            sky = (u.atmosphereParams.x > 0.5f)
                ? nishitaAtmosphereColor(rayDir, u)
                : atmosphereColor(rayDir, u) + (rayDir.y >= 0.0f ? nightSkyGlow(rayDir, u) : float3(0.0f));
            sky += sunDisk(rayDir, u);
        }

        // ── Night sky: stars + moon (additive, gated by nightBlend) ──
        // Skipped when the host renders the analytic screen-res night sky
        // instead (cloudExtra2.y = 0) — the dome-baked copies would only be
        // magnified into blobs behind the crisp ones.
        // One dome texel's angular size sets the star point-spread / moon limb AA.
        if (u.cloudExtra2.y > 0.5f)
            sky += skyCelestials(rayDir, u, 2.0f * M_PI_F / float(W), sky, u.windTime.w, false);
        sky = applyCirrus(sky, u.cameraPos.xyz, rayDir, u.lavaA.y, u, noiseVol, u.cirrusC.x);
    }

    // Compute ray–slab entry and exit (slab is infinite in XZ). Rays
    // pointing down don't intersect the slab from below, so they short-
    // circuit to "no cloud" — but we don't apply a hard horizon cutoff:
    // a smoothstep on (yCos) blends the cloud contribution into the
    // atmosphere as the ray approaches grazing, which kills the hard
    // horizon-band artifact that an `if (yCos < eps)` early-out creates.
    float yCos = rayDir.y;
    float3 ro = u.cameraPos.xyz;
    float baseY = u.cloudSlab.x;
    float topY  = u.cloudSlab.y;

    // Slab intersection for all three camera altitudes. The original code
    // early-returned for any camera above the slab (`ro.y > topY`), which
    // made the deck vanish entirely from elevated / top-down cameras — the
    // documented "camera above slab" failure. Downward rays now march the
    // slab top→base; rays inside the slab march toward whichever boundary
    // they exit.
    float tEnter, tExit;
    if (ro.y < baseY) {
        // Below the deck: only upward rays can reach it.
        if (yCos < 0.01f) {
            outTex.write(float4(mix(sky, lavaCol, lavaW), 1.0f), gid);
            return;
        }
        tEnter = (baseY - ro.y) / yCos;
        tExit  = (topY  - ro.y) / yCos;
    } else if (ro.y <= topY) {
        // Inside the slab — march from the camera to the exit boundary.
        tEnter = 0.0f;
        if (yCos > 0.001f) {
            tExit = (topY - ro.y) / yCos;
        } else if (yCos < -0.001f) {
            tExit = (baseY - ro.y) / yCos;
        } else {
            // Horizontal ray inside the slab — capped by marchLen below.
            tExit = (topY - baseY) * 4.0f;
        }
    } else {
        // Above the deck: only downward rays can reach it.
        if (yCos > -0.01f) {
            outTex.write(float4(mix(sky, lavaCol, lavaW), 1.0f), gid);
            return;
        }
        tEnter = (topY  - ro.y) / yCos;
        tExit  = (baseY - ro.y) / yCos;
    }
    tEnter = max(tEnter, 0.0f);
    if (tExit <= tEnter) {
        outTex.write(float4(mix(sky, lavaCol, lavaW), 1.0f), gid);
        return;
    }

    // ── Cloud march ──────────────────────────────────────────────────
    int   steps   = max(1, int(u.marchBudgets.x));
    // Cap on march length so grazing-horizon rays don't traverse the
    // entire slab — long horizon paths over-integrate density and
    // produce a hard rectangular-tower band on the equator. Capping at
    // a few × the slab thickness lets clouds fade into the haze near
    // the horizon, matching real aerial perspective.
    float thickness = topY - baseY;
    float marchLen  = min(tExit - tEnter, thickness * 4.0f);
    float dt        = marchLen / float(steps);
    float3 step   = rayDir * dt;
    float3 pos    = ro + rayDir * tEnter;

    // Per-pixel jitter to break up step banding without true temporal accum.
    float jitter = fract(sin(dot(float2(gid), float2(12.9898f, 78.233f))) * 43758.5453f);
    pos += step * (jitter * 0.6f - 0.3f);

    float3 lum = float3(0.0f);
    float trans = 1.0f;

    float gF   = u.lighting.x;
    float gB   = u.lighting.y;
    float absK = max(0.05f, u.lighting.z);
    float powd = u.lighting.w;
    float lightSteps = u.marchBudgets.y;
    float densityScale = u.cloudShape.y;
    float ambient = u.cloudShape.w;
    float msStrength = u.cloudExtra2.x;   // 0 = single-scatter (legacy)

    float cosT = clamp(dot(rayDir, -u.sunDir.xyz), -1.0f, 1.0f);
    float phase = hg(cosT, gF) + 0.5f * hg(cosT, gB);

    // Sun is "active" only if it has colour — night decks (sunColor ≈ 0) skip
    // the sun light-march entirely (it would contribute nothing but cost a
    // full secondary march per dense step).
    // Atmosphere-lit deck (`cloudLightingFromAtmosphere`): the irradiance + fill the GPU
    // derived from the nishita march; otherwise the host's hand-set sun and zenith.
    bool   physLit = u.skyGrade.z > 0.5f;
    float3 sunIrr  = physLit ? u.cloudLitSun.xyz : u.sunColor.xyz * u.sunColor.w;
    float3 ambBase = physLit ? u.cloudLitAmbient.xyz : u.skyZenith.xyz;
    bool sunActive = (sunIrr.x + sunIrr.y + sunIrr.z) > 1e-3f;

    // Moon as the deck's DIRECTIONAL light at night: a real lit near-face /
    // shadowed far-face is what makes a cloud read as cumulus rather than a
    // flat ambient blob. moonParams.xyz points TOWARD the moon; the directional
    // light-march reuses the same geometry as the sun. Gain in atmosphereParams.w.
    float  moonGain = u.atmosphereParams.w;
    bool   moonActive = moonGain > 1e-3f;
    float3 moonL = normalize(u.moonParams.xyz);
    float  cosM = clamp(dot(rayDir, moonL), -1.0f, 1.0f);
    float  phaseM = hg(cosM, gF) + 0.5f * hg(cosM, gB);
    // Mild multiple-scattering floor: enough that deep back-scatter angles
    // aren't pitch black, but LOW so the directional self-shadow contrast (lit
    // crowns vs dark crevices/undersides) survives — that contrast is what
    // makes a cloud read as a real volume rather than a flat-lit balloon. (A
    // strong floor flattened them into fake-looking blobs.)
    phaseM = mix(phaseM, 0.5f, 0.25f);
    constexpr float3 kMoonTint = float3(0.62f, 0.70f, 0.92f);  // cool moonlight

    // ── Per-ray burst-light cull ─────────────────────────────────────
    // One ray–sphere test per light per PIXEL (not per step): keep only
    // lights whose influence sphere intersects this ray's slab segment.
    // Dead / expired entries are dropped here too, so the march loop's
    // inner evaluation does no liveness checks at all.
    uint  burstCount = uint(max(0.0f, u.marchBudgets.z));
    float burstGain  = u.marchBudgets.w;
    VSBurstLight rayLights[kVSMaxRayLights];
    uint rayLightCount = 0u;
    {
        float3 segStart = ro + rayDir * tEnter;
        for (uint i = 0u; i < burstCount && rayLightCount < kVSMaxRayLights; ++i) {
            VSBurstLight L = burstLights[i];
            if (L.positionIntensity.w <= 0.0f || L.radiusLifePad.x <= 0.0f) continue;
            if (L.colorAge.a >= L.radiusLifePad.y) continue;
            float3 toL  = L.positionIntensity.xyz - segStart;
            float tProj = clamp(dot(toL, rayDir), 0.0f, marchLen);
            float3 off  = toL - rayDir * tProj;
            float  r    = L.radiusLifePad.x;
            if (dot(off, off) < r * r) {
                rayLights[rayLightCount++] = L;
            }
        }
    }

    for (int i = 0; i < steps; ++i) {
        if (trans < 0.01f) break;

        float dRaw = cloudDensity(pos, u, noiseVol);
        float density = dRaw * densityScale;

        // World-anchored radial fade: bound the infinite slab to a finite disc
        // around a fixed world point (the burst zone), so the deck is the same
        // finite scattered field from every camera and never piles up at the
        // horizon. (Preferred over the camera-distance fade below, which faded
        // the deck out from far cameras like the top-down.)
        if (u.cloudField.w > 0.0f) {
            float dC = length(pos.xz - u.cloudField.xy);
            density *= 1.0f - smoothstep(u.cloudField.z, u.cloudField.w, dC);
        }
        // Camera-distance fade (legacy / optional; both gated, 0 = off).
        {
            float fadeNear = u.cloudDetail.z;
            float fadeFar  = u.cloudDetail.w;
            if (fadeNear > 0.0f || fadeFar > 0.0f) {
                float distCam = length(pos - ro);
                if (fadeNear > 0.0f) density *= smoothstep(0.0f, fadeNear, distCam);
                if (fadeFar  > 0.0f) density *= 1.0f - smoothstep(fadeFar, fadeFar * 1.8f, distCam);
            }
        }

        if (density > 0.001f) {
            // Powder — darken the lit side of dense regions so the cumulus
            // silhouette pops against the sky (Schneider).
            float powder = 1.0f - exp(-density * 2.0f * (1.0f + powd * 4.0f));

            // Sun directional (daytime decks only).
            float3 sunContrib = float3(0.0f);
            if (sunActive) {
                float od = lightMarch(pos, -u.sunDir.xyz, lightSteps, u, noiseVol);
                float lit = (msStrength > 0.0f)
                    ? msLight(od, cosT, gF, gB, absK, msStrength)
                    : exp(-od * absK) * phase;
                sunContrib = sunIrr * lit * powder;
            }

            // Moon directional (night decks) — gives the deck its form.
            float3 moonContrib = float3(0.0f);
            if (moonActive) {
                float odM = lightMarch(pos, moonL, lightSteps, u, noiseVol);
                // Multi-scatter lets the moonlit crowns bleed a soft glow down
                // into the bank instead of terminating at the lit surface.
                float lit = (msStrength > 0.0f)
                    ? msLight(odM, cosM, gF, gB, absK, msStrength)
                    : exp(-odM * absK) * phaseM;
                moonContrib = kMoonTint * moonGain * lit * powder;
            }

            // Cloud underside ambient fill. skyZenith is a saturated daytime
            // BLUE, so a thick opaque deck lit only by this reads as "deeper
            // blue sky" from below, not overcast. skyZenith.w (cloudAmbientGrey)
            // desaturates the fill toward a luminance-preserving neutral grey so
            // an overcast deck reads grey (DH-0585); 0 = the legacy blue fill.
            float  ambGrey = clamp(u.skyZenith.w, 0.0f, 1.0f);
            float  ambLum  = dot(ambBase, float3(0.2126f, 0.7152f, 0.0722f));
            float3 ambCol  = mix(ambBase, float3(ambLum), ambGrey);
            float3 ambientContrib = ambCol * ambient;

            // Fireworks: single-scattered radiance from the ray's culled
            // light list. Bursts below the slab glow the cloud base from
            // underneath ("reflection" read); bursts or shell heads inside
            // the slab light the interior around themselves. Isotropic
            // phase; radial+age falloff in burstLightEval.
            // Each burst is shaded by the SAME dual-lobe HG phase as the moon,
            // using the direction from this sample toward the burst: a thin
            // cloud edge between the camera and a bright burst lights up with a
            // forward-scatter silver lining (issue #18 priority 1), while the
            // isotropic floor keeps a burst below the deck pooling broadly on
            // the base. (cosB convention matches the sun/moon: view-march dir ·
            // toward-light dir, forward peak when looking toward the burst.)
            float3 burstContrib = float3(0.0f);
            for (uint li = 0u; li < rayLightCount; ++li) {
                float3 toL  = rayLights[li].positionIntensity.xyz - pos;
                float  toLn = length(toL);
                float3 Ldir = (toLn > 1e-4f) ? toL / toLn : rayDir;
                float  cosB = clamp(dot(rayDir, Ldir), -1.0f, 1.0f);
                float  phaseB = hg(cosB, gF) + 0.5f * hg(cosB, gB);
                phaseB = mix(phaseB, 0.5f, 0.45f);   // isotropic floor for the base pool
                // Occlude the burst by the cloud BETWEEN this sample and the
                // burst — the "light up from within" core. Without it the glow is
                // added everywhere a burst is in range (flat); with it a burst
                // inside a bank lights the near interior and shadows the far side
                // / crevices (backlit rim + shafts), and a burst behind a thick
                // bank is blocked. 3 steps is enough for the local occlusion that
                // carries the read; absK matches the moon/sun self-shadow scale.
                float odB = burstLightMarch(pos, rayLights[li].positionIntensity.xyz,
                                            3, u, noiseVol);
                // Multi-scatter so a burst INSIDE a bank glows the interior
                // (light bleeds past the first extinction) rather than only
                // lighting the thin near shell; falls back to single-scatter ×
                // the floored phase when MS is off.
                float litB = (msStrength > 0.0f)
                    ? msLight(odB, cosB, gF, gB, absK, msStrength)
                    : exp(-odB * absK) * phaseB;
                burstContrib += burstLightEval(pos, rayLights[li]) * litB;
            }
            burstContrib *= burstGain;

            // Beer-Lambert step extinction along the camera ray.
            float stepTrans = exp(-density * dt * absK * 0.4f);
            float3 stepLum  = (sunContrib + moonContrib + ambientContrib + burstContrib)
                            * (1.0f - stepTrans);

            lum += trans * stepLum;
            trans *= stepTrans;
        }

        pos += step;
        if (pos.y > topY + 0.5f || pos.y < baseY - 0.5f) break;
    }

    // Composite: cloud luminance over sky, weighted by transmittance.
    // Fade cloud contribution toward zero as the ray approaches the
    // horizon — kills the hard cloud-edge band that otherwise reads as
    // a row of distant buildings on the equator. |yCos| so the fade is
    // symmetric: downward rays from an above-deck camera get the same
    // grazing-angle treatment as upward rays from below.
    // Fade window host-tunable via cloudExtra2.zw (0 = the legacy 0.02/0.22).
    // A ground-level architectural camera sees sky ONLY near the horizon —
    // the legacy window (full clouds above ~13°) erased the whole deck there.
    float fadeLo = u.cloudExtra2.z > 0.0f ? u.cloudExtra2.z : 0.02f;
    float fadeHi = u.cloudExtra2.w > 0.0f ? u.cloudExtra2.w : 0.22f;
    float horizonFade = smoothstep(fadeLo, fadeHi, abs(yCos));
    float blend = mix(0.0f, 1.0f, horizonFade);
    float effTrans = mix(1.0f, trans, blend);
    float3 col = mix(sky * effTrans + lum * blend, lavaCol, lavaW);

    // Mild filmic-ish tone (just a soft Reinhard) so the HDR sun disk doesn't
    // bloom past representable float16 when bound as an environment map.
    col = col / (1.0f + col * 0.10f);

    // Physical night sky drawn ANALYTICALLY by the host (celestialsInDome = false): alpha
    // carries how much of what lies BEYOND the deck shows through this texel (the cloud
    // transmittance, and none under the lava lamp), so the per-pixel stars / moon are hidden
    // behind the clouds baked into this dome. Everyone else keeps alpha 1 (byte-identical).
    float alpha = (u.nightSkyA.x > 0.5f && u.cloudExtra2.y <= 0.5f) ? effTrans * (1.0f - lavaW) : 1.0f;
    outTex.write(float4(col, alpha), gid);
}

// ── In-view (perspective) cloud kernel ───────────────────────────────────────
//
// Same atmosphere + volumetric cumulus + burst-lit march as volSkyRender, but
// rays are reconstructed PER OUTPUT PIXEL from the host camera's inverse
// view-projection — so the deck renders crisp at the renderer's output
// resolution instead of being an upscaled crop of a 360° equirect (the
// fundamental blur of the dome path). Output is composited straight into the
// host's HDR scene-colour target (linear, NO Reinhard — the host tonemaps once
// downstream), behind the additive firework particles which draw later.
//
// This is gated/opt-in at the host (IlluminatoramaRenderer): the dome path
// (volSkyRender, above) is UNTOUCHED and still drives every other scene + the
// IBL environment. See issue #61.
//
// v2 — DEPTH-CLIPPED. This kernel overwrites the pixel it runs on (`outTex` is
// write-only, so it cannot blend), and every one of its early-outs writes the
// atmosphere colour. v1 dispatched over the WHOLE frame on the strength of its
// one opt-in scene (FireworksUltra) having no opaque G-buffer geometry — so in
// any scene that DOES have geometry it painted sky straight over the building.
// A house scene hid that fact well enough to be missed twice: below the horizon
// the kernel returns the host's `groundColor` fill, which an architectural host
// calibrates against its own lawn, so the lawn survived the overwrite looking
// almost right and only the building was destroyed.
//
// So the pass now takes the scene depth and writes ONLY where there is no
// geometry (cleared depth), leaving every opaque pixel exactly as the deferred
// pass shaded it. `depth >= 0.99999 ⇒ sky` is the same test the RT composite
// uses (IlluminatoramaRT.metal), against the same 1.0 clear.

struct CloudInViewUniforms {
    float4x4 invViewProjection;  // host clip → world (jittered VP inverse)
    float4   cameraWorldPos;     // xyz = ray origin, w = downsample factor (≤1 = full res)
    float4   extra;              // x = the renderer's frame time (s) — animates the lava lamp per frame
                                 // y = per-frame lava weight override (≥ 0), −1 = SkyUniforms.lavaA.x
                                 // z = cirrus clock override (≥ 0), −1 = x
                                 // w = cirrus threads weight override (≥ 0), −1 = SkyUniforms.cirrusC.x
    float4   night;              // x = the renderer's frame clock (s) — stellar scintillation; yzw reserved
};

/// Angular size (radians) of one output pixel of the camera the in-view pass reconstructs
/// rays for — the vertical neighbour's ray, unprojected the same way. Sets the star
/// point-spread and the moon's limb AA, as the deferred sky branch's `pixAngle` does.
inline float inViewPixelAngle(constant CloudInViewUniforms &cv, float2 uv, float2 size, float3 rd) {
    float2 uv1 = uv + float2(0.0f, 1.0f / size.y);
    float4 wp1 = cv.invViewProjection * float4(uv1.x * 2.0f - 1.0f, 1.0f - uv1.y * 2.0f, 1.0f, 1.0f);
    float3 rd1 = normalize(wp1.xyz / wp1.w - cv.cameraWorldPos.xyz);
    // Chord form, NOT acos(dot): two rays a telephoto pixel apart (~6e-5 rad at 6° FOV) have
    // a dot product within float32's last ulp of 1, so acos returned 0 or ~6× the true angle
    // pixel by pixel — a jagged moon limb and mis-sized stars. 2·asin(|a−b|/2) is exact.
    return 2.0f * asin(min(0.5f * length(rd - rd1), 1.0f));
}

kernel void illumi_cloud_inview(
    texture2d<float, access::write>  outTex      [[texture(0)]],
    texture3d<float, access::sample> noiseVol    [[texture(1)]],
    // Scene depth — the v2 clip. Cleared to 1.0, so anything below that is
    // opaque geometry this pass must not touch.
    depth2d<float,   access::read>   gDepth      [[texture(2)]],
    constant SkyUniforms &u                      [[buffer(0)]],
    constant CloudInViewUniforms &cv             [[buffer(1)]],
    device const VSBurstLight* burstLights       [[buffer(2)]],
    uint2 gid                                    [[thread_position_in_grid]]
) {
    uint W = outTex.get_width();
    uint H = outTex.get_height();
    if (gid.x >= W || gid.y >= H) return;
    // Downsampled march (`inViewCloudsResolutionScale` < 1): this texel covers an f×f block
    // of the depth buffer and marches if ANY of it is sky; an all-geometry block writes alpha 0
    // so the upsample never blends it in. At f = 1 this is the original full-res clip.
    uint f = max(1u, uint(cv.cameraWorldPos.w + 0.5f));
    if (f <= 1u) {
        // Opaque geometry owns this pixel — leave the deferred composite alone.
        if (gDepth.read(gid) < 0.99999f) return;
    } else {
        uint2 dmax = uint2(gDepth.get_width() - 1, gDepth.get_height() - 1);
        bool anySky = false;
        for (uint dy = 0u; dy < f && !anySky; ++dy)
            for (uint dx = 0u; dx < f && !anySky; ++dx)
                anySky = gDepth.read(min(gid * f + uint2(dx, dy), dmax)) >= 0.99999f;
        if (!anySky) { outTex.write(float4(0.0f), gid); return; }
    }

    // ── Reconstruct the world-space camera ray (the ONLY difference vs the
    // equirect kernel). NDC: x,y ∈ [-1,1] with +Y up (texture row 0 = top), z = 1
    // at the far plane (Metal clip z ∈ [0,1]). Unproject the far point, subtract
    // the eye, normalise.
    float2 uv = (float2(gid) + 0.5f) / float2(W, H);
    float4 ndc = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 1.0f, 1.0f);
    float4 wp  = cv.invViewProjection * ndc;
    float3 ro  = cv.cameraWorldPos.xyz;
    float3 rayDir = normalize(wp.xyz / wp.w - ro);

    // ── Atmosphere + sun (identical to volSkyRender) ─────────────────
    bool debugBG = u.atmosphereParams.z > 0.5f;
    float3 sky;
    float lavaW = 0.0f;
    float3 lavaCol = float3(0.0f);
    if (debugBG) {
        sky = neonDebugBackground(rayDir);
    } else {
        // The lava lamp animates on the RENDERER's frame clock here (every frame), not the
        // dome's (host-throttled) — its blobs would otherwise step visibly. It replaces the
        // atmosphere, so the nishita march is skipped rather than computed and discarded.
        // lavaA.x = the lamp's weight (see volSkyRender): < 1 renders both and cross-fades.
        // extra.y ≥ 0 = the host's per-frame weight (a fade must not step at the sky's rate).
        lavaW = cv.extra.y >= 0.0f ? saturate(cv.extra.y) : saturate(u.lavaA.x);
        if (lavaW > 0.0f) lavaCol = lavaLampSky(rayDir, cv.extra.x, u);
        if (lavaW >= 1.0f) {
            sky = lavaCol;
        } else {
            sky = (u.atmosphereParams.x > 0.5f)
                ? nishitaAtmosphereColor(rayDir, u)
                : atmosphereColor(rayDir, u) + (rayDir.y >= 0.0f ? nightSkyGlow(rayDir, u) : float3(0.0f));
            sky += sunDisk(rayDir, u);
        }
        // Same analytic-night-sky skip as volSkyRender (cloudExtra2.y).
        if (u.cloudExtra2.y > 0.5f)
            sky += skyCelestials(rayDir, u, inViewPixelAngle(cv, uv, float2(W, H), rayDir),
                                 sky, cv.night.x, false);
        sky = applyCirrus(sky, ro, rayDir, cv.extra.z >= 0.0f ? cv.extra.z : cv.extra.x, u, noiseVol,
                          cv.extra.w >= 0.0f ? cv.extra.w : u.cirrusC.x);   // w ≥ 0: per-frame threads weight
    }
    // Alpha for a texel with no deck in the way: fully transmissive (see the tail write).
    float clearA = f > 1u ? 2.0f : 1.0f;

    float yCos = rayDir.y;
    float baseY = u.cloudSlab.x;
    float topY  = u.cloudSlab.y;

    float tEnter, tExit;
    if (ro.y < baseY) {
        if (yCos < 0.01f) { outTex.write(float4(mix(sky, lavaCol, lavaW), clearA), gid); return; }
        tEnter = (baseY - ro.y) / yCos;
        tExit  = (topY  - ro.y) / yCos;
    } else if (ro.y <= topY) {
        tEnter = 0.0f;
        if (yCos > 0.001f)       tExit = (topY - ro.y) / yCos;
        else if (yCos < -0.001f) tExit = (baseY - ro.y) / yCos;
        else                     tExit = (topY - baseY) * 4.0f;
    } else {
        if (yCos > -0.01f) { outTex.write(float4(mix(sky, lavaCol, lavaW), clearA), gid); return; }
        tEnter = (topY  - ro.y) / yCos;
        tExit  = (baseY - ro.y) / yCos;
    }
    tEnter = max(tEnter, 0.0f);
    if (tExit <= tEnter) { outTex.write(float4(mix(sky, lavaCol, lavaW), clearA), gid); return; }

    // ── Cloud march (identical math to volSkyRender) ─────────────────
    int   steps     = max(1, int(u.marchBudgets.x));
    float thickness = topY - baseY;
    float marchLen  = min(tExit - tEnter, thickness * 4.0f);
    float dt        = marchLen / float(steps);
    float3 step     = rayDir * dt;
    float3 pos      = ro + rayDir * tEnter;
    float jitter = fract(sin(dot(float2(gid), float2(12.9898f, 78.233f))) * 43758.5453f);
    pos += step * (jitter * 0.6f - 0.3f);

    float3 lum = float3(0.0f);
    float trans = 1.0f;

    float gF   = u.lighting.x;
    float gB   = u.lighting.y;
    float absK = max(0.05f, u.lighting.z);
    float powd = u.lighting.w;
    float lightSteps = u.marchBudgets.y;
    float densityScale = u.cloudShape.y;
    float ambient = u.cloudShape.w;
    float msStrength = u.cloudExtra2.x;

    float cosT = clamp(dot(rayDir, -u.sunDir.xyz), -1.0f, 1.0f);
    float phase = hg(cosT, gF) + 0.5f * hg(cosT, gB);
    // Atmosphere-lit deck (`cloudLightingFromAtmosphere`): the irradiance + fill the GPU
    // derived from the nishita march; otherwise the host's hand-set sun and zenith.
    bool   physLit = u.skyGrade.z > 0.5f;
    float3 sunIrr  = physLit ? u.cloudLitSun.xyz : u.sunColor.xyz * u.sunColor.w;
    float3 ambBase = physLit ? u.cloudLitAmbient.xyz : u.skyZenith.xyz;
    bool sunActive = (sunIrr.x + sunIrr.y + sunIrr.z) > 1e-3f;

    float  moonGain = u.atmosphereParams.w;
    bool   moonActive = moonGain > 1e-3f;
    float3 moonL = normalize(u.moonParams.xyz);
    float  cosM = clamp(dot(rayDir, moonL), -1.0f, 1.0f);
    float  phaseM = hg(cosM, gF) + 0.5f * hg(cosM, gB);
    phaseM = mix(phaseM, 0.5f, 0.25f);
    constexpr float3 kMoonTint = float3(0.62f, 0.70f, 0.92f);

    uint  burstCount = uint(max(0.0f, u.marchBudgets.z));
    float burstGain  = u.marchBudgets.w;
    VSBurstLight rayLights[kVSMaxRayLights];
    uint rayLightCount = 0u;
    {
        float3 segStart = ro + rayDir * tEnter;
        for (uint i = 0u; i < burstCount && rayLightCount < kVSMaxRayLights; ++i) {
            VSBurstLight L = burstLights[i];
            if (L.positionIntensity.w <= 0.0f || L.radiusLifePad.x <= 0.0f) continue;
            if (L.colorAge.a >= L.radiusLifePad.y) continue;
            float3 toL  = L.positionIntensity.xyz - segStart;
            float tProj = clamp(dot(toL, rayDir), 0.0f, marchLen);
            float3 off  = toL - rayDir * tProj;
            float  r    = L.radiusLifePad.x;
            if (dot(off, off) < r * r) rayLights[rayLightCount++] = L;
        }
    }

    for (int i = 0; i < steps; ++i) {
        if (trans < 0.01f) break;
        float dRaw = cloudDensity(pos, u, noiseVol);
        float density = dRaw * densityScale;
        if (u.cloudField.w > 0.0f) {
            float dC = length(pos.xz - u.cloudField.xy);
            density *= 1.0f - smoothstep(u.cloudField.z, u.cloudField.w, dC);
        }
        {
            float fadeNear = u.cloudDetail.z;
            float fadeFar  = u.cloudDetail.w;
            if (fadeNear > 0.0f || fadeFar > 0.0f) {
                float distCam = length(pos - ro);
                if (fadeNear > 0.0f) density *= smoothstep(0.0f, fadeNear, distCam);
                if (fadeFar  > 0.0f) density *= 1.0f - smoothstep(fadeFar, fadeFar * 1.8f, distCam);
            }
        }
        if (density > 0.001f) {
            float powder = 1.0f - exp(-density * 2.0f * (1.0f + powd * 4.0f));
            float3 sunContrib = float3(0.0f);
            if (sunActive) {
                float od = lightMarch(pos, -u.sunDir.xyz, lightSteps, u, noiseVol);
                float lit = (msStrength > 0.0f)
                    ? msLight(od, cosT, gF, gB, absK, msStrength)
                    : exp(-od * absK) * phase;
                sunContrib = sunIrr * lit * powder;
            }
            float3 moonContrib = float3(0.0f);
            if (moonActive) {
                float odM = lightMarch(pos, moonL, lightSteps, u, noiseVol);
                float lit = (msStrength > 0.0f)
                    ? msLight(odM, cosM, gF, gB, absK, msStrength)
                    : exp(-odM * absK) * phaseM;
                moonContrib = kMoonTint * moonGain * lit * powder;
            }
            // Cloud underside ambient fill. skyZenith is a saturated daytime
            // BLUE, so a thick opaque deck lit only by this reads as "deeper
            // blue sky" from below, not overcast. skyZenith.w (cloudAmbientGrey)
            // desaturates the fill toward a luminance-preserving neutral grey so
            // an overcast deck reads grey (DH-0585); 0 = the legacy blue fill.
            float  ambGrey = clamp(u.skyZenith.w, 0.0f, 1.0f);
            float  ambLum  = dot(ambBase, float3(0.2126f, 0.7152f, 0.0722f));
            float3 ambCol  = mix(ambBase, float3(ambLum), ambGrey);
            float3 ambientContrib = ambCol * ambient;
            float3 burstContrib = float3(0.0f);
            for (uint li = 0u; li < rayLightCount; ++li) {
                float3 toL  = rayLights[li].positionIntensity.xyz - pos;
                float  toLn = length(toL);
                float3 Ldir = (toLn > 1e-4f) ? toL / toLn : rayDir;
                float  cosB = clamp(dot(rayDir, Ldir), -1.0f, 1.0f);
                float  phaseB = hg(cosB, gF) + 0.5f * hg(cosB, gB);
                phaseB = mix(phaseB, 0.5f, 0.45f);
                float odB = burstLightMarch(pos, rayLights[li].positionIntensity.xyz,
                                            3, u, noiseVol);
                float litB = (msStrength > 0.0f)
                    ? msLight(odB, cosB, gF, gB, absK, msStrength)
                    : exp(-odB * absK) * phaseB;
                burstContrib += burstLightEval(pos, rayLights[li]) * litB;
            }
            burstContrib *= burstGain;
            float stepTrans = exp(-density * dt * absK * 0.4f);
            float3 stepLum  = (sunContrib + moonContrib + ambientContrib + burstContrib)
                            * (1.0f - stepTrans);
            lum += trans * stepLum;
            trans *= stepTrans;
        }
        pos += step;
        if (pos.y > topY + 0.5f || pos.y < baseY - 0.5f) break;
    }

    // Fade window host-tunable via cloudExtra2.zw (0 = the legacy 0.02/0.22).
    // A ground-level architectural camera sees sky ONLY near the horizon —
    // the legacy window (full clouds above ~13°) erased the whole deck there.
    float fadeLo = u.cloudExtra2.z > 0.0f ? u.cloudExtra2.z : 0.02f;
    float fadeHi = u.cloudExtra2.w > 0.0f ? u.cloudExtra2.w : 0.22f;
    float horizonFade = smoothstep(fadeLo, fadeHi, abs(yCos));
    float blend = mix(0.0f, 1.0f, horizonFade);
    float effTrans = mix(1.0f, trans, blend);
    float3 col = mix(sky * effTrans + lum * blend, lavaCol, lavaW);
    // Full-res march + the PHYSICAL night sky drawn analytically (celestialsInDome = false):
    // this pass REPLACES the sky pixel, so it must draw the celestials itself — behind the
    // deck (× its transmittance), as the downsampled path's upsample does.
    if (f <= 1u && u.nightSkyA.x > 0.5f && u.cloudExtra2.y <= 0.5f && !debugBG && rayDir.y > -0.05f) {
        col += skyCelestials(rayDir, u, inViewPixelAngle(cv, uv, float2(W, H), rayDir), col, cv.night.x, true)
             * effTrans * (1.0f - lavaW);
    }
    // NO Reinhard here — the host tonemaps the HDR scene-colour once downstream.
    // Downsampled: alpha = 1 + transmittance, so the full-res upsample can put pixel-sharp
    // stars BEHIND the clouds (alpha ≥ 1 still marks "marched"). Full-res: alpha 1, as before.
    outTex.write(float4(col, f > 1u ? 1.0f + effTrans : 1.0f), gid);
}

// Depth-aware upsample of a downsampled in-view cloud march into the HDR composite. Sky pixels
// only; a manual bilinear over the four surrounding low-res texels weighted by their alpha
// (1 = marched, 0 = an all-geometry block), so a silhouette never pulls in unmarched texels.
// Clouds and the atmosphere are smooth at this scale — a quarter of the rays for the same sky.
kernel void illumi_cloud_upsample(
    texture2d<float, access::write> outTex [[texture(0)]],
    texture2d<float, access::read>  lowRes [[texture(1)]],
    depth2d<float,   access::read>  gDepth [[texture(2)]],
    constant CloudInViewUniforms &cv       [[buffer(0)]],
    constant SkyUniforms &u                [[buffer(1)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    if (gDepth.read(gid) < 0.99999f) return;
    float f = max(1.0f, cv.cameraWorldPos.w);
    int2 lmax = int2(lowRes.get_width() - 1, lowRes.get_height() - 1);
    float2 p = (float2(gid) + 0.5f) / f - 0.5f;
    int2 p0 = int2(floor(p));
    float2 t = p - float2(p0);
    float3 acc = float3(0.0f);
    float wsum = 0.0f, tr = 0.0f;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            int2 q = clamp(p0 + int2(i, j), int2(0), lmax);
            float4 s = lowRes.read(uint2(q));
            float marched = step(0.5f, s.a);
            float w = (i == 0 ? 1.0f - t.x : t.x) * (j == 0 ? 1.0f - t.y : t.y) * marched;
            acc += s.rgb * w;
            tr += max(s.a - 1.0f, 0.0f) * w;
            wsum += w;
        }
    }
    if (wsum <= 1e-5f) {
        // Every neighbour was geometry-only (a sky pixel inside a thin gap): the texel that
        // contains this pixel marched it (its block had sky), so read that one.
        float4 s = lowRes.read(uint2(clamp(int2(float2(gid) / f), int2(0), lmax)));
        acc = s.rgb; tr = max(s.a - 1.0f, 0.0f); wsum = 1.0f;
    }
    float3 col = acc / wsum;
    // Night sky at FULL resolution, behind the clouds (when the host keeps celestials out of the
    // dome): pixel-sharp stars and the moon disc, attenuated by the deck's transmittance.
    // (The physical model draws the moon by day too — a pale daytime moon — so it runs
    // whenever that model is on; the legacy one only after sunset, as before.)
    bool physicalNight = u.nightSkyA.x > 0.5f;
    if ((u.nightParams.w > 0.0f || physicalNight) && u.cloudExtra2.y <= 0.5f && u.atmosphereParams.z <= 0.5f) {
        uint W = outTex.get_width(), H = outTex.get_height();
        float2 uv = (float2(gid) + 0.5f) / float2(W, H);
        float4 wp = cv.invViewProjection * float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 1.0f, 1.0f);
        float3 rd = normalize(wp.xyz / wp.w - cv.cameraWorldPos.xyz);
        float pixelAngle = inViewPixelAngle(cv, uv, float2(W, H), rd);
        if (rd.y > -0.05f) {
            float trans = tr / wsum;
            // … and under the lava lamp's weight (the night sky fades out as the lamp fades in).
            // The washout background is the sky as marched (atmosphere + moonlit glow + deck).
            col += skyCelestials(rd, u, pixelAngle, col, cv.night.x, true) * trans
                 * (1.0f - (cv.extra.y >= 0.0f ? saturate(cv.extra.y) : saturate(u.lavaA.x)));
        }
    }
    outTex.write(float4(col, 1.0f), gid);
}
