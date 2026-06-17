// ── VOLUMETRIC SMOKE ────────────────────────────────────────────────────────
//
// Screen-space raymarched smoke renderer for the Fireworks+ scene. Sibling
// of `volSkyRender` (sky cumulus) — same toolkit (NoiseVolume sampling,
// Beer-Lambert march, single-light scatter) but rendered into a screen-space
// MTLTexture that gets composited via a camera-attached overlay quad
// instead of into a panoramic dome.
//
// The density field is an analytic union of fading Gaussian puffs anchored
// at every burst's centre and at every climbing-shell sample. The shape of
// each cloud comes from a low-frequency noise modulation sampled out of the
// shared `NoiseVolume` (the same 3D RGBA16F that drives the cloud kernel)
// — so the smoke has the same kind of fluffy/eroded shape as the cumulus
// renderer, not perfectly round bubbles.
//
// Lighting per march step:
//   - Sample the BurstLightField for "what colour is bouncing through here?"
//   - Short light-march toward each strong burst for self-shadow / silver-
//     lining transmittance
//   - Add a constant dim ambient sky tint so smoke between bursts is still
//     faintly visible against the dark sky
//
// The kernel runs at half-resolution (768×432) by default; the overlay
// material upscales bilinearly, which softens the per-pixel noise into the
// "look-up at a fluffy mass" feel and keeps the per-frame budget tight.
//
// 64-byte SmokePuff is float4-aligned per the project ALIGNMENT RULE.

#include <metal_stdlib>
using namespace metal;

struct SmokePuff {
    float4 positionAge;     // xyz = world centre, w = age (s)
    float4 paramsLife;      // x = startRadius (m), y = endRadius (m),
                            // z = baseDensity (0..1), w = lifespan (s)
};

struct BurstLight {
    float4 positionIntensity;
    float4 colorAge;
    float4 radiusLifePad;
};

struct SmokeUniforms {
    float4 cameraPos;       // xyz = world pos, w = pad
    float4 cameraRight;     // xyz = world right (unit), w = tan(halfFovX)
    float4 cameraUp;        // xyz = world up    (unit), w = tan(halfFovY)
    float4 cameraForward;   // xyz = world forward (unit), w = aspect (w/h)
    float4 puffsBursts;     // x = puff count (uint bits), y = burst count (uint bits),
                            // z = near plane, w = far plane (march cap)
    float4 ambient;         // rgb = ambient sky tint, a = global density gain
    float4 marchParams;     // x = step count (float), y = step size (m),
                            // z = light step count, w = noise scale (m per noise period)
    float4 timeAndPhase;    // x = wall time, y = HG phase forward, zw = pad
};

// PCG hash for cheap per-pixel jitter (dither the start of each ray so
// the march bands don't show as concentric circles around the camera).
inline uint smk_pcg(thread uint& s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
inline float smk_rand(thread uint& s) {
    return float(smk_pcg(s)) * (1.0f / 4294967296.0f);
}

// Henyey-Greenstein phase function — gives smoke that brightening
// silver-lining when viewed near the burst direction.
inline float hg(float cosA, float g) {
    float g2 = g * g;
    return (1.0f - g2) / (12.5663706f * pow(1.0f + g2 - 2.0f * g * cosA, 1.5f));
}

// Density at world-space point p, summed over every active puff.
// Each puff's contribution is a soft (1 - r²)² Gaussian-ish falloff
// scaled by an age-aware density curve, all modulated by NoiseVolume.
inline float smokeDensityAt(
    float3 p,
    device const SmokePuff* puffs,
    uint puffCount,
    texture3d<float, access::sample> noiseTex,
    float noiseScale
) {
    if (puffCount == 0u) return 0.0f;

    constexpr sampler s(coord::normalized, address::repeat, filter::linear);

    // Noise sample once per march step — read R (fbm) and G (1 - worley)
    // so the cloud surface has both base body and eroded edges.
    float4 n = noiseTex.sample(s, p * noiseScale);
    float fbm     = n.r;
    float erosion = n.g;
    // Same shaping as the cumulus kernel: dense body × erosion, biased
    // soft so smoke has thick centres + wispy edges.
    float noiseMass = clamp(fbm * 1.4f - 0.30f, 0.0f, 1.0f);
    noiseMass = mix(noiseMass, noiseMass * erosion, 0.45f);

    float total = 0.0f;
    for (uint i = 0u; i < puffCount; ++i) {
        SmokePuff puff = puffs[i];
        float age  = puff.positionAge.w;
        float life = puff.paramsLife.w;
        if (life <= 0.0f || age >= life) continue;

        // Radius grows with age (smoke disperses outward over time).
        float ageT = age / life;
        float radius = mix(puff.paramsLife.x, puff.paramsLife.y, ageT);
        float invRadius = 1.0f / max(radius, 0.001f);

        float3 d = p - puff.positionAge.xyz;
        float dist2 = dot(d, d);
        float r2 = dist2 * invRadius * invRadius;
        if (r2 >= 1.0f) continue;

        // Soft falloff to zero at r=1, with a peak at r=0. 1 - r² then
        // squared = quartic — peakier centre, softer edge.
        float radial = 1.0f - r2;
        radial = radial * radial;

        // Opacity profile: ramp in over first 15%, hold, fade out last 35%.
        float opacity;
        if (ageT < 0.15f) {
            opacity = ageT / 0.15f;
        } else if (ageT > 0.65f) {
            opacity = 1.0f - (ageT - 0.65f) / 0.35f;
        } else {
            opacity = 1.0f;
        }

        total += radial * opacity * puff.paramsLife.z;
    }

    // Bias the noise-mass by the analytic union — only chew detail into
    // regions that have meaningful density to begin with. Otherwise the
    // background noise everywhere would render the whole sky cloudy.
    return total * noiseMass;
}

// Sample the burst-light field — local copy of the same sampler used by
// the ocean / star kernels. Returns linear RGB additive contribution.
inline float3 smkSampleBurstField(
    float3 worldPos,
    device const BurstLight* lights,
    uint count
) {
    float3 sum = float3(0.0f);
    for (uint i = 0u; i < count; ++i) {
        BurstLight L = lights[i];
        float intensity = L.positionIntensity.w;
        float maxR      = L.radiusLifePad.x;
        float life      = L.radiusLifePad.y;
        if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

        float reach = maxR * 1.6f;
        float dist = distance(worldPos, L.positionIntensity.xyz);
        if (dist >= reach) continue;

        float age = L.colorAge.a;
        if (age >= life) continue;

        float rT = 1.0f - dist / reach;
        float rFall = rT * rT;
        float ageT = age / life;
        float ageScale;
        if (ageT < 0.08f) {
            ageScale = 1.0f;
        } else {
            float k = (ageT - 0.08f) / 0.92f;
            ageScale = (1.0f - k);
            ageScale = ageScale * ageScale;
        }
        sum += L.colorAge.rgb * (intensity * rFall * ageScale);
    }
    return sum;
}

// CHEAP per-burst scatter: skip the inner light-march. Use a fixed
// transmittance approximation × HG phase. The full per-burst per-step
// inner light march (4 light samples × N puffs per density eval) was
// O(steps × bursts × lightSteps × puffs) ≈ 10¹⁰ ops/frame at typical
// settings — guaranteed GPU saturation hang. Real production volumetric
// renderers do this only at ¼ resolution or with TAA reuse; we settle
// for a believable approximation that the visible smoke-catches-flash
// effect dominates anyway.
inline float3 smokeLightFromBursts(
    float3 p,
    float3 viewDir,
    device const BurstLight* lights,
    uint burstCount,
    float hgForward
) {
    if (burstCount == 0u) return float3(0.0f);

    float3 sum = float3(0.0f);
    for (uint i = 0u; i < burstCount; ++i) {
        BurstLight L = lights[i];
        float intensity = L.positionIntensity.w;
        float maxR      = L.radiusLifePad.x;
        float life      = L.radiusLifePad.y;
        if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

        float reach = maxR * 2.5f;
        float3 toBurst = L.positionIntensity.xyz - p;
        float lightDist = length(toBurst);
        if (lightDist >= reach) continue;
        float3 lightDir = toBurst / max(lightDist, 1e-4f);

        float age = L.colorAge.a;
        if (age >= life) continue;
        float ageT = age / life;
        float ageScale = ageT < 0.08f ? 1.0f
            : pow(max(0.0f, 1.0f - (ageT - 0.08f) / 0.92f), 2.0f);
        float distFall = 1.0f - lightDist / reach;
        distFall *= distFall;

        float cosA = dot(viewDir, -lightDir);
        float phase = hg(cosA, hgForward);

        sum += L.colorAge.rgb * (intensity * ageScale * distFall * phase * 0.55f);
    }
    return sum;
}

// ── Main kernel ─────────────────────────────────────────────────────────────

kernel void smokeVolumeMarch(
    texture2d<float, access::write>          output    [[ texture(0) ]],
    texture3d<float, access::sample>         noiseTex  [[ texture(1) ]],
    device   const SmokePuff*                puffs     [[ buffer(0)  ]],
    device   const BurstLight*               bursts    [[ buffer(1)  ]],
    constant SmokeUniforms&                  u         [[ buffer(2)  ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = output.get_width();
    uint h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Per-pixel ray direction from the camera basis.
    float fu = (float(gid.x) + 0.5f) / float(w);
    float fv = (float(gid.y) + 0.5f) / float(h);
    float ndcX = (fu * 2.0f - 1.0f) * u.cameraRight.w;
    float ndcY = (1.0f - fv * 2.0f) * u.cameraUp.w;  // SCN screen-space Y down
    float3 rayDir = normalize(
        u.cameraForward.xyz
        + u.cameraRight.xyz * ndcX
        + u.cameraUp.xyz    * ndcY
    );
    float3 rayPos = u.cameraPos.xyz;

    uint puffCount  = as_type<uint>(u.puffsBursts.x);
    uint burstCount = as_type<uint>(u.puffsBursts.y);
    if (puffCount == 0u) {
        output.write(float4(0.0f), gid);
        return;
    }

    float farPlane = u.puffsBursts.w;
    uint stepCount = uint(u.marchParams.x);
    float stepLen  = u.marchParams.y;
    uint lightSteps = uint(u.marchParams.z);
    float noiseScale = u.marchParams.w;
    float densityGain = u.ambient.a;
    float hgForward = u.timeAndPhase.y;

    // Per-pixel ray-jitter so march bands don't show. Use a hash of pixel
    // + time so the dither dances frame to frame (denoised by eye averaging).
    uint seed = (gid.x * 1973u) ^ (gid.y * 9277u) ^ uint(u.timeAndPhase.x * 1000.0f);
    float jitter = smk_rand(seed);
    float t = u.puffsBursts.z + jitter * stepLen;   // start from near, plus dither

    float3 scattered = float3(0.0f);
    float transmittance = 1.0f;

    for (uint i = 0u; i < stepCount; ++i) {
        if (t >= farPlane) break;
        if (transmittance < 0.01f) break;   // optical depth — rays saturated, stop

        float3 sp = rayPos + rayDir * t;
        float density = smokeDensityAt(sp, puffs, puffCount, noiseTex, noiseScale)
                      * densityGain;

        if (density > 0.005f) {
            // Beer-Lambert step.
            float extinction = density * stepLen * 0.6f;
            float stepT = exp(-extinction);

            // Per-step lighting: ambient sky + cheap burst scatter (no
            // inner light-march — that was the GPU-saturation hang cause).
            float3 ambient = u.ambient.rgb * 0.7f;
            float3 burstScatter = smokeLightFromBursts(
                sp, rayDir, bursts, burstCount, hgForward
            );
            float3 lit = ambient + burstScatter;

            // Energy-conserving scatter integration.
            float3 scatter = lit * (1.0f - stepT);
            scattered += scatter * transmittance;
            transmittance *= stepT;
        }

        t += stepLen;
    }

    float alpha = 1.0f - transmittance;
    // Premultiplied output — the overlay material uses regular alpha-over
    // blending, with the texture's RGB already scaled by alpha.
    output.write(float4(scattered, alpha), gid);
}
