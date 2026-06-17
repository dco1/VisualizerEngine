// ── VOLUMETRIC CLOUD 3D (depth-correct raymarched, full cumulus lighting) ──
//
// Screen-space raymarched cloud volume that READS the scene's depth buffer
// and terminates each pixel's march at scene depth — so shells, stars and
// other depth-writing elements properly occlude the cloud at their world-
// space depth.
//
// The cloud-march lighting model mirrors `VolumetricSky.metal` (proven for
// kilometre-scale sky domes): per-step Henyey-Greenstein phase, 4-tap
// light march toward the key direction for self-shadow, powder term so
// cumulus silhouettes pop against the sky, Beer-Lambert extinction along
// the camera ray. Without these the cloud reads as featureless cotton no
// matter how the density / coverage knobs are tuned.
//
// BURST INTEGRATION
//
// Bursts are out-of-loop: after the cloud march, the BurstLightField is
// sampled ONCE at the closest occupied cloud point along the ray and
// added in proportional to the cloud's transmittance (so glow concentrates
// where the cloud actually is). The earlier per-march-step burst sum
// painted vertical "search-light" stripes down through the cloud as the
// camera tracked across bursts; this approximation keeps burst illumination
// soft and confined to cloud-occupied pixels.

#include <metal_stdlib>
using namespace metal;

struct BurstLight {
    float4 positionIntensity;
    float4 colorAge;
    float4 radiusLifePad;
};

struct CloudVolume3DUniforms {
    float4 invViewProjRow0;       // inverse view-projection (row-major).
    float4 invViewProjRow1;
    float4 invViewProjRow2;
    float4 invViewProjRow3;
    float4 cameraPos;             // xyz = world camera, w = pad.
    float4 slabAndCounts;         // x = baseY, y = topY, z = hScl,
                                  // w = burst count (uint bits).
    float4 shapeAndTime;          // x = coverage, y = density,
                                  // z = erosion, w = wall time.
    float4 lightDirAndAnvil;      // xyz = unit direction TOWARD the key
                                  // light, w = anvil weight.
    float4 lightColorAndIntens;   // rgb = key colour, a = intensity.
    float4 ambientAndPhase;       // rgb = sky ambient, a = HG g.
    float4 marchParams;           // x = step count, y = max alpha,
                                  // z = light-march steps, w = powder.
    float4 absorptionAndBurst;    // x = absorption, y = burst gain,
                                  // z = zNear, w = zFar.
};

constexpr sampler volSampler(filter::linear, address::repeat, coord::normalized);

// ── Height shaping ─────────────────────────────────────────────────────────
//
// Real cumulus has a SHARP flat base (the lifting condensation level)
// and a ROUNDED rising top. The previous profile ramped the base softly
// from h=0..0.10 and the top from h=0.55..1.0, which gave a vertically
// symmetric blob shape that read as fog rather than a cumulus dome.
// New profile: ramp base in 5% of slab, hold density flat across the
// middle 60%, taper top across the upper 35% — gives a roughly
// hemispheric silhouette with the visible mass in the upper-middle of
// the slab where it belongs.

inline float vc3_heightShaping(float h, float anvil) {
    // RAZOR-FLAT base — real cumulus condenses at a sharp altitude
    // line (the LCL) and the underside is essentially a flat sheet.
    // Almost-step transition over a 1% band so the visible silhouette
    // IS a sheet, AA-only soft.
    float bottom = smoothstep(0.0f, 0.01f, h);
    // Cumulus body skews DENSER in the lower half (where condensation
    // is heaviest). Bias the density profile downward so the visible
    // mass lives in h≈0.05..0.55, with billowing breakup above that —
    // closer to real cumulus's "flat-bottomed, cauliflower-topped"
    // shape than the centred profile I had before.
    float top    = 1.0f - smoothstep(0.55f, 0.95f, h);
    float core   = bottom * top;
    // Anvil shelf near the top — slight outward bloom for congestus.
    float shelf  = smoothstep(0.75f, 0.92f, h) * (1.0f - smoothstep(0.92f, 1.0f, h));
    return clamp(core + anvil * shelf * 0.6f, 0.0f, 1.0f);
}

// Large-scale weather field driving coverage. Slow XZ FBM read from the
// R channel of the noise volume at a slow-moving uvw.
inline float vc3_weatherCoverage(
    float2 xz, float hScl, float baseCoverage, float t, float tileSize,
    texture3d<float, access::sample> noiseVol
) {
    float2 q = xz * hScl * 0.125f + float2(0.13f, 0.27f) * t * 0.020f;
    float3 uvw = float3(q.x, 0.4f, q.y) / max(tileSize, 1.0f);
    float w = noiseVol.sample(volSampler, uvw).r;
    return clamp(w + (baseCoverage - 0.5f) * 1.8f, 0.0f, 1.0f);
}

// ── Density sample ─────────────────────────────────────────────────────────
//
// Returns the SHAPED density at `pos` in [0..1]. Caller multiplies by
// the density scale (`u.shapeAndTime.y`) and uses the result as the
// extinction coefficient in Beer-Lambert.

inline float vc3_density(
    float3 pos,
    constant CloudVolume3DUniforms& u,
    texture3d<float, access::sample> noiseTex
) {
    float baseY = u.slabAndCounts.x;
    float topY  = u.slabAndCounts.y;
    float hScl  = u.slabAndCounts.z;
    float anvil = u.lightDirAndAnvil.w;

    float h = clamp((pos.y - baseY) / max(topY - baseY, 1e-3f), 0.0f, 1.0f);
    float heightW = vc3_heightShaping(h, anvil);
    if (heightW <= 0.0f) return 0.0f;

    float t = u.shapeAndTime.w;
    // Slow wind advection so the field evolves over time.
    float2 wind = float2(0.6f, 0.4f) * t * 0.05f;
    // ANISOTROPIC noise sampling — scale Y MORE than X/Z so the
    // baked FBM's natural features come out wider than tall. Without
    // this the noise field is isotropic and the resulting puffs read
    // as vertical stalagmites / wax drips / smoke columns instead of
    // buoyant cumulus. Real fair-weather cumulus has ~3:1 horizontal-
    // to-vertical aspect; multiplying Y by 1.7 gets us most of the
    // way there while keeping horizontal feature size correct.
    float3 q = float3(pos.x - wind.x, pos.y * 1.7f, pos.z - wind.y) * hScl;
    float tileSize = float(noiseTex.get_width());

    // ── SHAPE / EROSION SEPARATION ────────────────────────────────────
    //
    // Pre-ChatGPT-feedback this kernel did `shaped = body * (1 + erosion)`
    // and thresholded the result, which is "noise field, everywhere" —
    // the failure mode that gives you grey-hair-tangle clouds. The fix
    // is the standard production split:
    //
    //   1.  Define the MACRO BODY shape from a low-frequency field.
    //       This is what determines the cumulus silhouette.
    //   2.  Carve EROSION DETAIL only at the boundary, with a mask
    //       that goes to zero in the solid interior. Interior stays
    //       opaque + smooth; edges get wispy detail.
    //
    // Real cumulus has dense solid cores and wispy edges where the
    // condensation meets dry air — this is what gives the silhouette
    // its organic, NON-blobby character.

    // 1) Macro body: blend the noise volume's R (4-octave FBM) sampled
    //    at TWO scales — wider for shape + base scale for medium
    //    variation. Weighted toward the wider tap so low frequencies
    //    dominate the silhouette. Frequencies here drive the puff-
    //    break-up that gives cumulus its fluffy cauliflower read —
    //    don't soften them in pursuit of slice-pattern cleanup; the
    //    march-step count + jitter are the right levers for that.
    float macroLo = noiseTex.sample(volSampler, q * 0.4f / tileSize).r;
    float macroHi = noiseTex.sample(volSampler, q / tileSize).r;
    float body = mix(macroHi, macroLo, 0.65f);

    float cov = vc3_weatherCoverage(pos.xz, hScl, u.shapeAndTime.x, t,
                                     tileSize, noiseTex);
    float thresh = 1.0f - cov;

    // 2) Edge erosion: sample the G channel (1 - Worley) at HIGHER
    //    frequency for cell-shaped detail, but mask it to a thin
    //    shell around the puff boundary AND bias it toward the puff
    //    TOPS only. Real cumulus has turbulent cauliflower tops and
    //    smooth dense bases (the LCL); applying erosion uniformly
    //    leaves "stalactite drips" hanging off the undersides — the
    //    failure mode ChatGPT flagged.
    float worley = noiseTex.sample(volSampler, q * 3.0f / tileSize).g;

    // Pre-threshold "rough" density tells us where we are relative
    // to the cloud boundary. roughT in [0,1]: 0 = empty sky, 1 = deep
    // interior, ~0.5 = on the edge.
    float roughT = smoothstep(thresh - 0.05f, thresh + 0.12f, body);
    // Edge mask peaks at the boundary (roughT ≈ 0.5) and falls to
    // zero in both directions — `4 * x * (1-x)`. Carves only where
    // we'd see it.
    float edgeMask = 4.0f * roughT * (1.0f - roughT);
    // TOP bias — erosion strength grows from 0 at the cloud base to
    // full at the cloud top. Bottoms stay smooth and dense; tops get
    // the explosive cauliflower break-up real cumulus shows.
    float topBias = smoothstep(0.30f, 0.95f, h);
    edgeMask *= topBias;

    float erosionAmt = u.shapeAndTime.z;
    // `worley` is 1 - actual Worley distance, so HIGH where Worley
    // cells touch boundaries (the lines between cells). Inverting
    // here gives `(1 - worley)` = the cell CENTRES, which is what we
    // want to KEEP (carve away cell EDGES from the cloud body).
    // Bite STRENGTH backed off from 0.12 to 0.07 — at higher values
    // the high-freq detail competes with the macro shape and the
    // eye reads noise before cloud body (ChatGPT's "detail-forward,
    // silhouette-weak" failure mode). 0.07 keeps the wispy edges
    // characteristic of real cumulus while letting the macro shape
    // dominate the read.
    float erodedBody = body - (1.0f - worley) * edgeMask * 0.07f * (1.0f + erosionAmt);

    // Density edge band. Width tuned for "soft feathered cumulus
    // silhouette" — wider than the slice-ring-blur minimum (which
    // was ~0.07) so the edge has visible density falloff instead of
    // stepping sharply. 0.22 keeps the cauliflower break-up crisp
    // enough to read as cumulus (going wider here softens the
    // silhouette into a generic stratus blob). The dark-outline
    // artefact that this width USED to surface is now handled by
    // the density-gated powder term below.
    float d = smoothstep(thresh, thresh + 0.22f, erodedBody);

    return d * heightW;
}

// ── Phase function ─────────────────────────────────────────────────────────
//
// Dual-lobe Henyey-Greenstein — strong forward lobe at `g`, soft back lobe
// at `-g/3` so the cloud doesn't go pitch black on the shadow side.

inline float vc3_hg(float cosT, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosT;
    return (1.0f - g2) / (4.0f * 3.14159265f * pow(max(denom, 1e-4f), 1.5f));
}

inline float vc3_phase(float cosT, float g) {
    return vc3_hg(cosT, g) + 0.5f * vc3_hg(cosT, -g * 0.33f);
}

// ── Light march ────────────────────────────────────────────────────────────
//
// 4 short rays toward the key, geometric stride so we capture both fine
// nearby detail and bulk extinction further in. Returns accumulated
// optical depth; caller maps it to transmittance via exp(-od * absorption).

inline float vc3_lightMarch(
    float3 pos,
    int steps,
    constant CloudVolume3DUniforms& u,
    texture3d<float, access::sample> noiseTex
) {
    float3 L = u.lightDirAndAnvil.xyz;     // toward the light
    float thickness = u.slabAndCounts.y - u.slabAndCounts.x;
    float stride = thickness / float(max(steps, 1)) * 0.40f;
    float baseY = u.slabAndCounts.x;
    float topY  = u.slabAndCounts.y;
    float densityScale = u.shapeAndTime.y;

    float od = 0.0f;
    for (int i = 0; i < steps; ++i) {
        float t = stride * (float(i) + 0.5f) * (1.0f + 0.6f * float(i));
        float3 sp = pos + L * t;
        if (sp.y > topY || sp.y < baseY) break;
        float d = vc3_density(sp, u, noiseTex) * densityScale;
        od += d * stride;
    }
    return od;
}

// ── Burst contribution (out-of-loop, screen-space approximation) ───────────
//
// Sample the burst field once at `worldPos` and return additive HDR colour.
// Per-burst inverse-square falloff plus age scaling — same shape the per-
// step variant used, but called once per pixel from the main kernel
// AFTER the march, so neighbouring pixels don't disagree on which march
// step's burst contribution to integrate (the per-step variant's
// "search-light streak" artefact).

inline float3 vc3_burstAt(
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

        float reach = maxR * 3.0f;       // a bit more reach than the per-step variant
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

// ── Un-project helpers ─────────────────────────────────────────────────────

inline float3 vc3_unprojectToWorld(
    float2 ndcXY,
    float ndcZ,
    constant CloudVolume3DUniforms& u
) {
    float4 clip = float4(ndcXY, ndcZ, 1.0f);
    float4 r;
    r.x = dot(u.invViewProjRow0, clip);
    r.y = dot(u.invViewProjRow1, clip);
    r.z = dot(u.invViewProjRow2, clip);
    r.w = dot(u.invViewProjRow3, clip);
    return r.xyz / max(r.w, 1e-6f);
}

inline float vc3_sceneDepthDistance(
    float2 ndcXY,
    float depthSample,
    constant CloudVolume3DUniforms& u,
    float maxFallback
) {
    if (depthSample >= 0.9999f) return maxFallback;
    float3 sceneWorld = vc3_unprojectToWorld(ndcXY, depthSample, u);
    return distance(sceneWorld, u.cameraPos.xyz);
}

// ── Main kernel ────────────────────────────────────────────────────────────

kernel void volumetricCloud3D(
    texture2d<float, access::read_write>   output    [[ texture(0) ]],
    texture2d<float, access::sample>       sceneDepth [[ texture(1) ]],
    texture3d<float, access::sample>       noiseTex  [[ texture(2) ]],
    device   const BurstLight*             bursts    [[ buffer(0)  ]],
    constant CloudVolume3DUniforms&        u         [[ buffer(1)  ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = output.get_width();
    uint h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;


    // NDC: Metal x in [-1, +1], y flipped (compute grid Y goes DOWN).
    float2 ndcXY;
    ndcXY.x = (float(gid.x) + 0.5f) / float(w) * 2.0f - 1.0f;
    ndcXY.y = 1.0f - (float(gid.y) + 0.5f) / float(h) * 2.0f;

    // Build the world-space ray + read scene depth.
    float3 nearWorld = vc3_unprojectToWorld(ndcXY, 0.0f, u);
    float3 farWorld  = vc3_unprojectToWorld(ndcXY, 1.0f, u);
    float3 rayDir = normalize(farWorld - nearWorld);
    float3 rayPos = u.cameraPos.xyz;


    float depthSample = sceneDepth.read(gid).r;
    float maxMarch = 250.0f;
    float sceneDist = vc3_sceneDepthDistance(ndcXY, depthSample, u, maxMarch);
    sceneDist = min(sceneDist, maxMarch);

    // Slab traversal.
    float baseY = u.slabAndCounts.x;
    float topY  = u.slabAndCounts.y;
    float tEnter, tExit;
    if (abs(rayDir.y) < 1e-5f) {
        if (rayPos.y >= baseY && rayPos.y <= topY) {
            tEnter = 0.0f;
            tExit = sceneDist;
        } else {
            return;
        }
    } else {
        float t1 = (baseY - rayPos.y) / rayDir.y;
        float t2 = (topY  - rayPos.y) / rayDir.y;
        tEnter = min(t1, t2);
        tExit  = max(t1, t2);
        if (tExit <= 0.0f) return;
        tEnter = max(tEnter, 0.0f);
    }
    // Don't paint cloud in front of opaque scene elements.
    tExit = min(tExit, sceneDist);
    // Cap glancing-angle traversal — at a few × slab thickness clouds
    // fade into haze near the horizon rather than building up into a
    // solid white horizon-wall.
    float thickness = topY - baseY;
    tExit = min(tExit, tEnter + thickness * 6.0f);
    if (tExit <= tEnter) return;

    int   stepCount      = int(u.marchParams.x);
    float maxAlpha       = u.marchParams.y;
    int   lightSteps     = int(u.marchParams.z);
    float powderStrength = u.marchParams.w;
    float absorption     = u.absorptionAndBurst.x;
    float burstGain      = u.absorptionAndBurst.y;
    float densityScale   = u.shapeAndTime.y;
    uint  burstCount     = as_type<uint>(u.slabAndCounts.w);

    float stepLen = (tExit - tEnter) / float(stepCount);

    // ── BLUE-NOISE JITTER ────────────────────────────────────────────
    //
    // Hides the slice-ring quantisation artefact (visible contour
    // bands in low-density cloud patches caused by adjacent rays
    // making different in/out decisions at the same density step).
    // Standard fix: per-pixel + per-frame offset that's BLUE-NOISE
    // distributed (locally distinct, no white-noise clumping) rather
    // than the bog-standard hash we had before. Interleaved Gradient
    // Noise (IGN) — Jorge Jiménez's cheap blue-noise approximation —
    // is one ALU per pixel.
    //
    // Strength pushed to 0.85 of a full step (was 0.08). The original
    // throttle was a band-aid for the spider-web artefact that came
    // from WHITE-noise jitter at high strength on a noisy density
    // field; with the new edge-only erosion + low-freq macro shape,
    // neighbour rays are correlated and high jitter just smooths the
    // banding instead of surfacing as streaks.
    float ign = fract(52.9829189f * fract(0.06711056f * float(gid.x)
                                          + 0.00583715f * float(gid.y)));
    // Temporal animation — small frame-correlated offset cycles the
    // jitter pattern between frames, so the eye temporally averages
    // any remaining banding.
    ign = fract(ign + u.shapeAndTime.w * 0.42f);
    float3 step = rayDir * stepLen;
    float3 pos  = rayPos + rayDir * tEnter + step * (ign * 0.85f);

    // Phase precomputed once per pixel — depends only on (rayDir, lightDir).
    float cosT = clamp(dot(rayDir, u.lightDirAndAnvil.xyz), -1.0f, 1.0f);
    float phase = vc3_phase(cosT, u.ambientAndPhase.a);

    float3 lum = float3(0.0f);
    float trans = 1.0f;

    // Closest occupied cloud sample along the ray — used to evaluate
    // the burst contribution exactly once outside the loop.
    float3 firstHit = float3(0.0f);
    float  firstHitWeight = 0.0f;

    for (int i = 0; i < stepCount; ++i) {
        if (trans < 0.01f) break;

        float dRaw = vc3_density(pos, u, noiseTex);
        float density = dRaw * densityScale;

        if (density > 0.001f) {
            // Inner light march for self-shadow.
            float od = vc3_lightMarch(pos, lightSteps, u, noiseTex);
            float lightTrans = exp(-od * absorption);

            // Powder — darken the lit side of dense regions so the
            // cumulus silhouette pops against the sky (Schneider).
            //
            // GATED BY DENSITY: at thin/edge densities the raw Schneider
            // formula approaches zero, which makes cloud edges DARKER
            // than the sky behind them — alpha-blended against night
            // sky that reads as a hard black rim around every puff
            // (the "coloring-book outline" failure mode). The gate
            // ensures powder=1 (no darkening) at edges, ramping in to
            // the full Schneider effect only inside the dense core
            // where it does its job.
            float powderRaw  = 1.0f - exp(-density * 2.0f * (1.0f + powderStrength * 4.0f));
            float powderGate = smoothstep(0.08f, 0.35f, density);
            float powder     = mix(1.0f, powderRaw, powderGate);

            float3 keyContrib = u.lightColorAndIntens.rgb
                              * u.lightColorAndIntens.a
                              * lightTrans * phase * powder;
            float3 ambientContrib = u.ambientAndPhase.rgb;

            float stepTrans = exp(-density * stepLen * absorption * 0.4f);
            float3 stepLum  = (keyContrib + ambientContrib) * (1.0f - stepTrans);

            lum += trans * stepLum;
            trans *= stepTrans;

            // Capture the first-hit world position weighted by density
            // for burst illumination.
            if (firstHitWeight < 0.5f) {
                firstHit = pos;
                firstHitWeight = saturate(density * 4.0f);
            }
        }

        pos += step;
    }

    // Out-of-loop burst contribution — illuminates only where the cloud
    // actually exists (firstHit), weighted by how opaque the cloud is.
    if (firstHitWeight > 0.0f && burstCount > 0u && burstGain > 0.0f) {
        float3 burstColor = vc3_burstAt(firstHit, bursts, burstCount);
        float cloudOpacity = 1.0f - trans;
        lum += burstColor * burstGain * firstHitWeight * cloudOpacity;
    }

    // Soft horizon fade — fade out cloud contribution as the ray
    // approaches horizontal (|rayDir.y| → 0), regardless of whether
    // we're looking up or down at the slab. Stops grazing-angle rays
    // from over-integrating into a hard horizon band. `abs` because in
    // Fireworks+ the camera is ABOVE the slab (so cloud rays are
    // downward, yCos < 0), whereas VolumetricSky has the camera below
    // looking UP (yCos > 0). Same fade either way.
    float yCos = rayDir.y;
    float horizonFade = smoothstep(0.02f, 0.18f, abs(yCos));

    float alpha = (1.0f - trans) * maxAlpha * horizonFade;
    lum *= horizonFade;

    // ALPHA-OVER composite into the HDR target. The output already
    // contains shells / streaks / stars from earlier passes.
    float4 dst = output.read(gid);
    float3 final = lum + dst.rgb * (1.0f - alpha);
    output.write(float4(final, max(dst.a, alpha)), gid);
}
