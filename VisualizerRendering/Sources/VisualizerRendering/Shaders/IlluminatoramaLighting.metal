// ── ILLUMINATORAMA — DEFERRED LIGHTING ──────────────────────────────────────
//
// Cascaded shadow sampling, the BRDF lobes, LTC area lights, contact shadows,
// the deferred resolve kernel, and the subsurface blur/composite that follows it.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaDDGI.h"
using namespace metal;

static inline float distributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (M_PI_F * d * d + 1e-7);
}

static inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

static inline float3 worldPosFromDepth(float2 ndcXY, float depth, float4x4 invViewProj) {
    float4 clip = float4(ndcXY, depth, 1.0);
    float4 world = invViewProj * clip;
    return world.xyz / world.w;
}

// ── Cascaded shadow sampling (Phase 2.5) ─────────────────────────────────────
//
// Per pixel: pick a cascade from the view-space depth, project worldPos into
// that cascade's light space, then PCF-compare against the depth slice. The
// returned float is "sun visibility" in [0, 1] — 1 = lit, 0 = fully shadowed.

static inline uint pickCascade(float viewZ, float3 splits) {
    // viewZ here is a positive distance from the camera (i.e. -view.z), so
    // cascades fall into [0, splits.x), [splits.x, splits.y), [splits.y, splits.z).
    if (viewZ < splits.x) return 0u;
    if (viewZ < splits.y) return 1u;
    return 2u;
}

static inline float sampleCascade(
    depth2d_array<float, access::sample> shadowMap,
    sampler                              shadowSampler,
    float3 worldPos,
    float4x4 cascadeVP,
    uint cascade,
    float bias,
    uint pcfRadius
) {
    float4 lp = cascadeVP * float4(worldPos, 1.0);
    float3 ndc = lp.xyz / lp.w;
    // Outside the cascade's frustum → fall back to "lit" so we don't shadow
    // pixels that simply aren't covered by this cascade.
    if (any(abs(ndc.xy) > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    float refZ = ndc.z - bias;

    if (pcfRadius == 0u) {
        return shadowMap.sample_compare(shadowSampler, uv, cascade, refZ);
    }
    // 3×3 PCF (radius 1) or 5×5 (radius 2). The sample_compare hardware filter
    // gives us bilinear-filtered comparisons per tap, so a small grid here is
    // usually enough for nice soft contact shadows.
    int r = int(min(pcfRadius, 2u));
    float2 texel = 1.0 / float2(shadowMap.get_width(), shadowMap.get_height());
    float sum = 0.0;
    int count = 0;
    for (int j = -r; j <= r; ++j) {
        for (int i = -r; i <= r; ++i) {
            float2 off = float2(float(i), float(j)) * texel;
            sum += shadowMap.sample_compare(shadowSampler, uv + off, cascade, refZ);
            count += 1;
        }
    }
    return sum / float(count);
}

// ── Phase 4.42 — function-constant specialization of illumi_lighting (WWDC23 #10127) ──
//
// These gate the lighting kernel's STABLE per-config feature branches at COMPILE
// time. The Metal compiler constant-folds the branch and dead-code-eliminates the
// untaken path (and drops its register pressure) — the "Optimize GPU renderers with
// Metal" headline technique. The host (IlluminatoramaRenderer) sets them via
// MTLFunctionConstantValues and the shared cache memoises ONE pipeline variant per
// combination, so the lab's runtime A/B toggles still work (a toggle compiles its
// variant once, then hits the cache). Only flags that are constant across frames for
// a given config live here — per-frame-flipping flags (taa/ssao/ssr IsFirstFrame) and
// float intensities stay runtime in FrameUniforms.
//
// INVARIANT: each constant MUST equal the matching `frame.*Enabled` uniform the host
// still writes — both are derived from the same Swift Bool in `updateFrameUniforms`
// and `lightingFeatureConstants()`. The uniforms are kept (struct layout is shared
// Swift↔Metal) but no longer read by this kernel.
constant bool kLightingIBLEnabled          [[function_constant(0)]];
constant bool kLightingShadowEnabled       [[function_constant(1)]];
constant bool kLightingDFGLUTEnabled       [[function_constant(2)]];
constant bool kLightingDDGIEnabled         [[function_constant(3)]];
constant bool kLightingDDGIIrrCacheEnabled [[function_constant(4)]];

// One cascade's answer, everything that depends on the cascade index in one place.
// Factored out of `sunVisibility` for S1.4 (cascade-split blending), which needs to
// evaluate TWO cascades for the pixels inside a boundary's fade band — every one of
// the three cascade-dependent quantities below (matrix, depth bias, normal-offset
// push) has to travel with the index, and having them inline made a second
// evaluation impossible without duplicating them.
static inline float cascadeVisibility(
    depth2d_array<float, access::sample> shadowMap,
    sampler                              shadowSampler,
    constant FrameUniforms&              frame,
    float3 worldPos,
    float3 N,
    float  slope,
    uint   cascade
) {
    // Slope-scaled depth bias — surfaces nearly parallel to the light direction
    // need more bias to avoid acne, but biasing too much causes peter-panning.
    float bias = frame.shadowBias + frame.shadowSlopeBias * slope;
    // Outer cascades cover larger world extents, so a fixed bias becomes too
    // tight in light-NDC space — scale by cascade index.
    bias *= (1.0 + float(cascade) * 0.5);

    float4x4 cascadeVP;
    if (cascade == 0u)      cascadeVP = frame.shadowVP0;
    else if (cascade == 1u) cascadeVP = frame.shadowVP1;
    else                    cascadeVP = frame.shadowVP2;

    // Normal-offset bias. Depth bias along the light ray peter-pans grazing
    // surfaces (it scales with 1-NdotL, so walls detach from their contact
    // line — the classic offset artifact). Instead push the *receiver sample*
    // out along its own surface normal by ~1 shadow texel of WORLD size, so a
    // flat top no longer samples its own occluder (kills self-shadow acne)
    // while the in-plane shift barely moves the shadow edge. Texel world size
    // is derived from the cascade's ortho extent: the first matrix row's linear
    // length is 1/radius, and the map spans 2·radius across `width` texels.
    float3 row0 = float3(cascadeVP[0][0], cascadeVP[1][0], cascadeVP[2][0]);
    float  radius = 1.0 / max(length(row0), 1e-6);
    float  texelWorld = 2.0 * radius / float(shadowMap.get_width());
    // Widen mildly on grazing incidence (where acne is worst) but stay bounded
    // — 1–3 texels, never the runaway depth bias produced.
    // Reduced from (1.0 + 2.0*slope) — front-face culling (second-depth) is the primary acne
    // defence, so a smaller normal offset keeps acne away while the contact shadow stays tight
    // (the larger offset peter-panned shadows off object bases). Also capped so an outer-cascade
    // texel can't produce a runaway world push.
    float3 biasedPos = worldPos + N * min(texelWorld * (0.4 + 0.8 * slope), 0.006);

    return sampleCascade(shadowMap, shadowSampler, biasedPos,
                         cascadeVP, cascade, bias, frame.shadowPcfRadius);
}

// ── S1.4 — cascade-split blending ────────────────────────────────────────────
//
// Fraction of a cascade's own depth range, at its FAR end, over which its answer
// cross-fades into the next cascade's. `pickCascade` alone is a hard select, and
// the three quantities above ALL change with the index at once: the PCF kernel's
// world footprint (each cascade is fitted to its own sub-frustum sphere, so the
// outer ones are several times coarser), the depth bias, and the normal-offset
// push. A shadow edge crossing a split therefore kinks, and open ground picks up
// a line — the artefact this fixes.
//
// 0.10 is the usual figure and it costs what it looks like: only pixels inside
// the band take the second tap, i.e. ~10 % of the *depth* range of the cascades
// that have a successor (cascade 2 has none), and far less than 10 % of a typical
// frame's pixels. Outside the band `t == 0` and this is bit-identical to the hard
// select, so nothing but the seam moves.
constant float kCascadeBlendFraction = 0.10;

static inline float sunVisibility(
    depth2d_array<float, access::sample> shadowMap,
    sampler                              shadowSampler,
    constant FrameUniforms&              frame,
    float3 worldPos,
    float3 N,
    float  NdotL
) {
    if (!kLightingShadowEnabled) return 1.0;  // function_constant(1) — see decls above
    // Pick the cascade from view-space depth. We use -view.z (positive
    // distance) so the splits in frame.cascadeSplitsView are also positive.
    float4 vp = frame.view * float4(worldPos, 1.0);
    float viewDist = -vp.z;
    float3 splits = frame.cascadeSplitsView.xyz;
    uint cascade = pickCascade(viewDist, splits);
    float slope = clamp(1.0 - NdotL, 0.0, 1.0);

    float v = cascadeVisibility(shadowMap, shadowSampler, frame, worldPos, N, slope, cascade);

    // Inside the last `kCascadeBlendFraction` of this cascade's range, fade into the
    // next one. `cascade == 2` is the outermost, so it has nothing to fade into (its
    // far edge is `shadowMaxDistance`, where `sampleCascade` already falls back to
    // "lit" — a different boundary with a different fix).
    if (cascade < 2u) {
        float nearZ = (cascade == 0u) ? 0.0 : splits.x;
        float farZ  = (cascade == 0u) ? splits.x : splits.y;
        float band  = (farZ - nearZ) * kCascadeBlendFraction;
        float t = (band > 1e-4) ? saturate((viewDist - (farZ - band)) / band) : 0.0;
        if (t > 0.0) {
            float vNext = cascadeVisibility(shadowMap, shadowSampler, frame,
                                            worldPos, N, slope, cascade + 1u);
            v = mix(v, vNext, t);
        }
    }
    return v;
}

// ── Cloth sheen (Phase 7b, re-sited 2026-08-09) ──────────────────────────────────
//
// A woven fabric is not a rough dielectric with a GGX highlight. Its surface is a forest of
// near-vertical fibres, so the light it sends back is concentrated at GRAZING angles — the
// velvety bloom along the rim of a cushion, and the reason velvet reads as velvet. GGX has no
// such lobe; without a sheen term upholstery renders as painted plaster.
//
// **Estevez & Kulla 2017 ("Production Friendly Microfacet Sheen BRDF", SIGGRAPH talk)**
// replace GGX's Beckmann-like NDF with an inverted-Gaussian "Charlie" distribution, whose
// density is highest for microfacets standing PERPENDICULAR to the surface. Paired with
// Ashikhmin & Premoze / Neubelt & Pettineo's velvet visibility term (which stays finite as
// either cosine goes to zero, exactly where GGX's Smith term collapses), it is four lines and
// no texture.
//
// The `alpha → 0` guard matters: `invAlpha` becomes the exponent, so a zero roughness is an
// infinite power. Clamped at 1e-3.
static inline float clothSheenD(float alpha, float NdotH) {
    float invAlpha = 1.0 / max(alpha, 1e-3);
    float cos2h = NdotH * NdotH;
    float sin2h = max(1.0 - cos2h, 1e-7);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * M_PI_F);
}

/// Ashikhmin/Neubelt velvet visibility — finite at grazing, where Smith-GGX goes to zero and
/// takes the whole sheen lobe with it.
static inline float clothSheenV(float NdotV, float NdotL) {
    return 1.0 / max(4.0 * (NdotL + NdotV - NdotL * NdotV), 1e-5);
}

/// One fixed sheen roughness for the whole library. The strength already arrives per material
/// (packed as a negative `emission.alpha`); a per-material sheen ROUGHNESS would need a second
/// instance field, and the instance stride is not this change's to move. 0.30 is a soft,
/// broad nap — right for linen and wool, and velvet's much higher STRENGTH is what separates it.
constant float kClothSheenRoughness = 0.30;

/// Fibre-tip colour: cloth sheen is scattered by the pale tips of the nap, not by the dyed
/// core, so it is markedly whiter than the base albedo. Same mix the Phase-7b bolt-on used, so
/// this re-siting is a MECHANISM change and not a restyling of the fabric library.
static inline float3 clothSheenColor(float3 albedo) { return albedo * 0.4 + float3(0.6); }

/// Directional albedo of the Charlie/Neubelt lobe, for the environment (IBL/ambient) arm.
///
/// **A declared approximation**, in the same spirit as the area light's most-representative-
/// point specular above: the correct term is a fitted sheen DFG LUT (Estevez & Kulla §4), and
/// baking a second LUT is out of scope for this landing. What has to be right is the SHAPE —
/// the lobe returns very little at normal incidence and rises steeply toward grazing — and that
/// is what this reproduces. Replacing it with a LUT read later changes magnitude, not
/// behaviour.
static inline float clothSheenEnvAlbedo(float NdotV) {
    return 0.08 + 0.92 * pow(1.0 - saturate(NdotV), 4.0);
}

/// `sheenStrength` — the material's cloth-sheen scalar (velvet 0.85 / linen 0.30), unpacked
/// from the NEGATIVE `emission.alpha` the G-buffer writes. **0 is an exact early-out**: the
/// function returns the identical expression it always returned, so every non-fabric surface —
/// and every Visualizer scene, none of which set `sheen` — is byte-for-byte unchanged.
///
/// `sheenOut`, when non-null, receives ONLY the sheen part of this call's contribution. That is
/// what keeps `DebugTerm.clothSheen` an honest instrument now that the lobe is no longer a
/// separable bolt-on: the term can still be isolated without evaluating the light loops twice.
static inline float3 brdf(
    float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness, float3 lightColor,
    float anisotropy = 0.0, float3 grainT = float3(0.0),
    float sheenStrength = 0.0, thread float3 *sheenOut = nullptr
) {
    float3 H = normalize(V + L);
    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0) return float3(0);
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float HdotV = saturate(dot(H, V));

    float3 F0 = mix(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlick(HdotV, F0);
    float  G  = geometrySmith(NdotV, NdotL, roughness);
    float  D;
    if (anisotropy > 0.001 && dot(grainT, grainT) > 0.25) {
        // Anisotropic GGX (Burley): stretch the lobe along the in-plane grain tangent, compress
        // across it — a wood floor / brushed steel reads as a streaked highlight, not a round one.
        float3 B  = normalize(cross(N, grainT));
        float3 T  = normalize(cross(B, N));                 // re-orthogonalize into the plane
        float  a  = max(roughness * roughness, 1e-3);
        float  at = max(a * (1.0 + anisotropy), 1e-3);      // along grain (stretched)
        float  ab = max(a * (1.0 - 0.7 * anisotropy), 1e-3); // across grain (tight)
        float  ToH = dot(T, H), BoH = dot(B, H);
        float  d  = ToH * ToH / (at * at) + BoH * BoH / (ab * ab) + NdotH * NdotH;
        D = 1.0 / (M_PI_F * at * ab * d * d);
    } else {
        D = distributionGGX(NdotH, roughness);
    }

    float3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-7);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 diff = kd * albedo / M_PI_F;
    // THE EXACT EARLY-OUT. Everything above this line is untouched, and this is the expression
    // `brdf` has always returned; a surface with no cloth sheen therefore cannot move by a bit.
    if (!(sheenStrength > 0.0)) return (diff + spec) * lightColor * NdotL;

    float  Ds = clothSheenD(kClothSheenRoughness, NdotH);
    float  Vs = clothSheenV(NdotV, NdotL);
    float3 sheen = clothSheenColor(albedo) * (sheenStrength * Ds * Vs);
    if (sheenOut != nullptr) *sheenOut += sheen * lightColor * NdotL;
    return (diff + spec + sheen) * lightColor * NdotL;
}

// Issue #65 — the DIFFUSE-ONLY half of `brdf`, byte-for-byte the same diffuse
// term `brdf` adds to the composite (same Fresnel kd = (1-F)(1-metallic), same
// Lambert albedo/π · NdotL). The screen-space SSS pass uses it to peel the
// diffuse-lit irradiance off SSS-flagged pixels into a side buffer: because it
// reproduces `brdf`'s diffuse exactly, the composite (`hdr += s·(blur(D) − D)`)
// collapses to "replace the sharp diffuse with the blurred diffuse" at s = 1,
// leaving specular / emission / clearcoat untouched. No-op everywhere else.
static inline float3 brdfDiffuse(
    float3 N, float3 V, float3 L, float3 albedo, float metallic, float3 lightColor
) {
    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0) return float3(0);
    float3 H = normalize(V + L);
    float  HdotV = saturate(dot(H, V));
    float3 F0 = mix(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlick(HdotV, F0);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    return kd * albedo / M_PI_F * lightColor * NdotL;
}

// ── Rectangular area light (#60 task 5) ─────────────────────────────────────
// Replaces the 4.24 five-spot `.area` approximation. The DIFFUSE term is the
// EXACT closed-form polygon clamped-cosine integral — Linearly-Transformed-
// Cosines with M = identity (Heitz et al. 2016), i.e. a verifiable Lambert
// form-factor, not an approximation. The SPECULAR term is a most-representative-
// point sample (the rect point closest to the reflection ray) — a declared
// approximation pending the fitted GGX LTC specular LUT (increment 2).

// Vector irradiance of one polygon edge (clamped-cosine), v1/v2 normalised.
//
// Hill & Heitz's rational fit of θ/sin θ (the LTC reference implementation), NOT the
// literal acos form this shipped with first. The literal form is singular exactly where
// a WINDOW PORTAL lives: a fragment coplanar with the light (the wall the portal is cut
// into — every fragment of it) sees two corners in near-opposite directions, θ→π,
// sin θ→0, θ/sin θ→∞ while cross(v1,v2)→0 — 0·∞ = NaN, and the TAA settle smears one
// NaN into a fully black frame (measured: the first portal arm rendered 1 152 000 black
// pixels). A −0.9999 clamp tames the infinity but leaves acos's catastrophic
// cancellation at both ends — salt-and-pepper speckle across any coplanar floor or
// ceiling. The fit is stable at BOTH limits (the x → −1 branch pairs the 1/√(1−x²)
// growth against the cross's shrink analytically) and is what production LTC ships.
// Visualizer's softboxes float in open space and never exercised these limits; the
// sub-percent difference from the acos form elsewhere is the fit's documented accuracy.
static inline float3 ltcIntegrateEdge(float3 v1, float3 v2) {
    float x = clamp(dot(v1, v2), -1.0, 1.0);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;
    float thetaSinTheta = (x > 0.0)
        ? v
        : 0.5 * rsqrt(max(1.0 - x * x, 1e-7)) - v;
    return cross(v1, v2) * thetaSinTheta;
}

// Clamped-cosine form factor of the quad (corners p0..p3 CCW, relative to the
// shaded point) seen from a surface with normal N. Returns [0,1]; one-sided
// clamps the receiver to the front hemisphere, two-sided takes |·|.
static inline float ltcPolygonForm(float3 N, float3 p0, float3 p1, float3 p2, float3 p3,
                                   bool twoSided) {
    // Length-guarded normalise: a fragment AT a light corner (a portal's jamb pixel)
    // hands normalize() a zero vector — the remaining NaN seed once the edge integral
    // above went stable.
    float3 L0 = p0 * rsqrt(max(dot(p0, p0), 1e-8));
    float3 L1 = p1 * rsqrt(max(dot(p1, p1), 1e-8));
    float3 L2 = p2 * rsqrt(max(dot(p2, p2), 1e-8));
    float3 L3 = p3 * rsqrt(max(dot(p3, p3), 1e-8));
    float3 vsum = ltcIntegrateEdge(L0, L1) + ltcIntegrateEdge(L1, L2)
                + ltcIntegrateEdge(L2, L3) + ltcIntegrateEdge(L3, L0);
    float z = dot(vsum, N) * (1.0 / (2.0 * M_PI_F));
    return twoSided ? abs(z) : max(0.0, z);
}

// Full rectangular area-light contribution at a shaded point.
static inline float3 evalAreaLight(AreaLight al, float3 worldPos, float3 N, float3 V,
                                   float3 albedo, float metallic, float roughness,
                                   texture2d<float> ltcMat, texture2d<float> ltcMag,
                                   bool ltcEnabled) {
    float3 nL = cross(al.ex, al.ey);
    float  nLlen = length(nL);
    if (nLlen < 1e-8) return float3(0.0);
    nL /= nLlen;
    bool twoSided = al.twoSided > 0.5;

    // One-sided: the receiver must be on the emitting (+nL) face.
    float facing = dot(nL, worldPos - al.center);
    if (!twoSided && facing <= 0.0) return float3(0.0);

    // Distance falloff (centre distance) with a smooth radius window — keeps the
    // light local like the point/spot path rather than lighting the whole scene.
    float dist = length(al.center - worldPos);
    if (dist > al.radius) return float3(0.0);
    float window = saturate(1.0 - pow(dist / al.radius, 4.0));
    window *= window;
    if (window <= 0.0) return float3(0.0);

    // Quad corners relative to the shaded point — wound CLOCKWISE as seen from +nL,
    // which is COUNTER-clockwise as seen from a receiver ON the +nL side, the
    // orientation the edge integral's sign convention wants. The previous (CCW-from-
    // +nL) order made `z` NEGATIVE at every legitimately-lit receiver: two-sided
    // lights take |z| and never noticed (every Visualizer softbox is two-sided), but
    // the first ONE-sided light — a window portal — rendered `max(0, z)` = zero
    // diffuse everywhere, with normal-map jitter flickering individual pixels across
    // the clamp as salt-and-pepper speckle. Measured: a ×6 intensity change moved the
    // near-window floor probe 0.3 luma before this flip.
    float3 p0 = al.center - al.ex - al.ey - worldPos;
    float3 p1 = al.center - al.ex + al.ey - worldPos;
    float3 p2 = al.center + al.ex + al.ey - worldPos;
    float3 p3 = al.center + al.ex - al.ey - worldPos;

    // Diffuse — exact polygon clamped-cosine (LTC, M = identity).
    float  ff = ltcPolygonForm(N, p0, p1, p2, p3, twoSided);
    float3 diffuse = (1.0 - metallic) * albedo * ff;

    float3 spec = float3(0.0);
    float3 F0 = mix(float3(0.04), albedo, metallic);
    if (ltcEnabled) {
        // LTC specular — transform the light polygon by the per-(roughness, NdotV)
        // inverse matrix, then run the SAME clamped-cosine polygon integral. The
        // soft area light becomes a physically-shaped glossy reflection.
        constexpr sampler lutSamp(filter::linear, address::clamp_to_edge);
        float NoV = clamp(dot(N, V), 0.0, 1.0);
        float2 uv = float2(NoV, roughness);                  // (NdotV, roughness)
        float4 t1 = ltcMat.sample(lutSamp, uv);              // Minv (a,b,c,d)
        float2 t2 = ltcMag.sample(lutSamp, uv).xy;           // (scale, bias)
        // Tangent frame: T1 in the view-incidence plane (matches the fit's frame).
        // Guarded for V ∥ N — a fragment viewed dead-on (the wall straight ahead of a
        // walkthrough eye) makes V − N(V·N) a zero vector, and normalize(0) is the other
        // NaN this pass can seed. At normal incidence the LTC frame's azimuth is
        // irrelevant (the fit is isotropic there), so any tangent perpendicular to N is
        // exact, not approximate.
        float3 Tv = V - N * dot(V, N);
        float  TvLen = length(Tv);
        float3 T1 = (TvLen > 1e-5) ? Tv / TvLen
                  : normalize(cross(N, (fabs(N.y) < 0.99) ? float3(0, 1, 0) : float3(1, 0, 0)));
        float3 T2 = cross(N, T1);
        float3x3 worldToTan = transpose(float3x3(T1, T2, N)); // rows = T1,T2,N
        float3x3 Minv = float3x3(float3(t1.x, 0.0, t1.y),
                                 float3(0.0,  1.0, 0.0),
                                 float3(t1.z, 0.0, t1.w));
        // Same corner-fragment length guard as ltcPolygonForm — p ≈ 0 at a jamb pixel.
        float3 m0 = Minv * (worldToTan * p0), m1 = Minv * (worldToTan * p1);
        float3 m2 = Minv * (worldToTan * p2), m3 = Minv * (worldToTan * p3);
        float3 q0 = m0 * rsqrt(max(dot(m0, m0), 1e-8));
        float3 q1 = m1 * rsqrt(max(dot(m1, m1), 1e-8));
        float3 q2 = m2 * rsqrt(max(dot(m2, m2), 1e-8));
        float3 q3 = m3 * rsqrt(max(dot(m3, m3), 1e-8));
        float3 vsum = ltcIntegrateEdge(q0, q1) + ltcIntegrateEdge(q1, q2)
                    + ltcIntegrateEdge(q2, q3) + ltcIntegrateEdge(q3, q0);
        float ltcSpec = max(0.0, vsum.z / (2.0 * M_PI_F));
        // Firefly guard, measured not hypothetical: at grazing incidence on a low-roughness
        // texel the inverse LTC matrix stretches the polygon toward the horizon and the
        // integral spikes — white sparkle clusters on any wall flanking a window PORTAL
        // (an area light coplanar with real walls exercises grazing geometry a floating
        // softbox never does), and the worst spikes go non-finite and TAA smears them into
        // black holes. A clamped-cosine form factor is ≤ 1 by definition; 2.0 leaves the
        // fit slack while amputating the divergence, and the non-finite scrub is the last
        // line — one NaN pixel poisons the whole neighbourhood through bloom.
        if (!isfinite(ltcSpec)) ltcSpec = 0.0;
        ltcSpec = min(ltcSpec, 2.0);
        spec = ltcSpec * (F0 * t2.x + (1.0 - F0) * t2.y);
    } else {
        // Fallback — most-representative-point: the rect point closest to the
        // reflection ray, as a punctual GGX sample weighted by the form factor.
        float3 R = reflect(-V, N);
        float  denom = dot(R, nL);
        float  t = (abs(denom) > 1e-4) ? dot(al.center - worldPos, nL) / denom : 0.0;
        float3 hit = worldPos + R * max(t, 0.0);
        float3 d   = hit - al.center;
        float  ux  = length(al.ex), uy = length(al.ey);
        float3 exN = al.ex / max(ux, 1e-6), eyN = al.ey / max(uy, 1e-6);
        float3 rep = al.center + exN * clamp(dot(d, exN), -ux, ux)
                               + eyN * clamp(dot(d, eyN), -uy, uy);
        float3 Ls  = normalize(rep - worldPos);
        float  NdotLs = saturate(dot(N, Ls));
        if (NdotLs > 0.0) {
            float3 H     = normalize(V + Ls);
            float  NdotV = saturate(dot(N, V));
            float  NdotH = saturate(dot(N, H));
            float  HdotV = saturate(dot(H, V));
            float3 F     = fresnelSchlick(HdotV, F0);
            float  D     = distributionGGX(NdotH, roughness);
            float  G     = geometrySmith(NdotV, NdotLs, roughness);
            spec = (D * G * F) / (4.0 * NdotV * NdotLs + 1e-7) * NdotLs * ff;
        }
    }

    return (diffuse + spec) * al.color * window;
}

// Pull a fill colour toward its own luminance, ramped IN by how saturated it
// already is, scaled by `amount` (0 = no-op). Used to tame the monochromatic
// colour of indirect fill from a saturated environment (Phase 4.32).
static inline float3 desaturateFill(float3 c, float amount) {
    if (amount <= 0.0) return c;
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    float cmax = max(c.r, max(c.g, c.b));
    float sat = (cmax > 1e-4) ? saturate((cmax - lum) / cmax) : 0.0;
    float k = amount * smoothstep(0.35, 0.75, sat);
    return mix(c, float3(lum), k);
}

// ── Screen-space contact shadows (issue #65) ─────────────────────────────────
//
// A SHORT screen-space ray march toward the sun that catches the fine contact
// occlusion the cascaded shadow maps + RT soft shadows miss at object-base
// scale (a chip resting on felt, an egg on a floor, a prop on a table). We
// march a handful of steps in VIEW space from the shaded fragment toward the
// light, project each step back to screen space, and read the G-buffer depth
// there: if the scene surface sits in FRONT of the ray sample by more than a
// small self-shadow bias but less than `thickness` (a real thin occluder, not
// the far background), the sun is occluded at that step. Returns an occlusion
// fraction in [0,1] (0 = unoccluded) that the caller scales by
// `contactShadowStrength`; a hit nearer the fragment returns a stronger value
// so the darkening is densest right at the contact and fades along the ray
// (this also softens the discrete ray-step boundary). The caller skips this
// entirely when strength == 0, so it's an exact no-op by default.
static inline float illumiContactShadow(
    depth2d<float, access::read> gDepth,
    constant FrameUniforms&      frame,
    float3                       viewPos,    // fragment position, view space
    float3                       LviewN,     // normalized view-space dir TOWARD the light
    uint2                        dims,       // depth-buffer (width, height)
    float                        lengthWS,   // march reach, world units
    uint                         steps,
    float                        thickness,  // occluder depth window, world units
    float                        ndotl       // surface NdotL toward the sun
) {
    // Back-facing or grazing-to-sun fragments are already unlit by NdotL and
    // self-occlude trivially — skip them to avoid contact-shadow acne.
    if (ndotl <= 0.02 || steps == 0u || lengthWS <= 0.0) return 0.0;

    float  stepLen = lengthWS / float(steps);
    // Self-shadow guard: lift the ray off the originating surface and require
    // an occluder to be at least this far in front before it counts.
    float  bias    = max(lengthWS * 0.05, 1e-4);
    float3 rayPos  = viewPos + LviewN * bias;

    for (uint i = 0u; i < steps; ++i) {
        rayPos += LviewN * stepLen;
        float4 clip = frame.projection * float4(rayPos, 1.0);
        if (clip.w <= 1e-5) break;
        float3 ndc = clip.xyz / clip.w;
        float2 uv  = ndc.xy * float2(0.5, -0.5) + 0.5;
        if (any(uv < float2(0.0)) || any(uv > float2(1.0))) break;  // ray left the screen
        uint2  px  = min(uint2(uv * float2(dims)), dims - 1u);
        float  sceneDepth = gDepth.read(px);
        if (sceneDepth >= 0.99999) continue;                        // sky — no occluder

        // Reconstruct the scene surface's view-space Z at this pixel and compare
        // distances-into-scene (camera looks down -Z, so distance == -z).
        float4 sView      = frame.invProjection * float4(ndc.xy, sceneDepth, 1.0);
        float  sceneViewZ = sView.z / sView.w;
        float  diff       = (-rayPos.z) - (-sceneViewZ);  // >0: ray is BEHIND the surface
        if (diff > bias && diff < bias + thickness) {
            return 1.0 - float(i) / float(steps);         // densest at the contact
        }
    }
    return 0.0;
}

kernel void illumi_lighting(
    texture2d<half,  access::read>          gAlbedoMet      [[texture(0)]],
    texture2d<half,  access::read>          gNormalRgh      [[texture(1)]],
    texture2d<half,  access::read>          gEmission       [[texture(2)]],
    depth2d<float,   access::read>          gDepth          [[texture(3)]],
    texture2d<half,  access::write>         outHDR          [[texture(4)]],
    texture2d<half,  access::read>          aoTex           [[texture(5)]],
    texture2d<float, access::sample>        skyEquirect     [[texture(6)]],
    texturecube<half, access::sample>       irradianceCube  [[texture(7)]],
    texturecube<half, access::sample>       prefilteredCube [[texture(8)]],
    depth2d_array<float, access::sample>    shadowMap       [[texture(9)]],
    // Phase 3.2 — split-sum DFG LUT. RG channels: (scale, bias) such that
    // specular = prefilteredEnv * (F0 * scale + bias). Baked once on init;
    // `frame.dfgLUTEnabled == 0` falls back to Lagarde's roughness-Schlick.
    texture2d<half,  access::sample>        dfgLUT          [[texture(10)]],
    // Phase 3.1 — DDGI probe atlases. When ddgi.enabled == 0 these are
    // 1×1 dummies; the sampling is cheap and the enabled gate skips the loop.
    texture2d<half,  access::sample>        ddgiIrrAtlas    [[texture(11)]],
    texture2d<half,  access::sample>        ddgiDepthAtlas  [[texture(12)]],
    // Phase 4.10 — per-spot shadow depth atlas.
    depth2d_array<float, access::sample>    spotShadowAtlas [[texture(13)]],
    // Phase 3.4 — per-pixel DDGI irradiance EMA cache (ping-pong).
    texture2d<half,  access::sample>        irrCachePrev    [[texture(14)]],
    texture2d<half,  access::write>         irrCacheCur     [[texture(15)]],
    // #60 task 5 increment 2 — LTC area-light specular LUTs (Minv + magnitude).
    texture2d<float, access::sample>        ltcMat          [[texture(16)]],
    texture2d<float, access::sample>        ltcMag          [[texture(17)]],
    // Light-layer G-buffer target (R32Uint). Cleared to 0xFFFFFFFF for sky pixels;
    // each instance writes its `layer` bitfield here. A light contributes only when
    // (light.layerMask & fragLayer) != 0. When the host leaves every instance and
    // light at the default 0xFFFFFFFF the mask is always non-zero ⇒ no change.
    texture2d<uint,  access::read>          gLayer          [[texture(18)]],
    // Point-light cube depth atlas: 6 slices per shadowed light (slice = cube*6 + face).
    // Bound to a same-type placeholder when point shadows are off; never sampled then.
    depth2d_array<float, access::sample>    pointShadowAtlas [[texture(19)]],
    // Issue #65 — screen-space SSS diffuse side buffer. rgb = the diffuse-lit
    // irradiance of an SSS-flagged pixel; a = the SSS mask (1 = blur this pixel,
    // 0 = leave it). Written ONLY when frame.sssStrength > 0 (else the binding is
    // an unread dummy and the write is skipped, so non-SSS scenes pay nothing).
    // (texture(20): 18/19 were taken by gLayer/pointShadowAtlas in the merge.)
    texture2d<half,  access::write>         sssOut          [[texture(20)]],
    constant FrameUniforms&                 frame           [[buffer(0)]],
    const device PointLight*                pointLights     [[buffer(1)]],
    constant DDGIUniforms&                  ddgi            [[buffer(2)]],
    const device SpotLight*                 spotLights      [[buffer(3)]],
    const device AreaLight*                 areaLights      [[buffer(4)]],
    const device DirectionalLight*          extraDirectionals [[buffer(5)]],
    // Per-face view-projection matrices for shadowed point lights (6 per cube).
    const device float4x4*                  pointShadowFaces [[buffer(6)]],
    // Interior daylight apertures (S3.5 Stage D) — gates on the count/strength in
    // frame.interiorIrrDown.w / interiorIrrSide.w, both 0 unless the host opts in.
    const device InteriorAperture*          interiorApertures [[buffer(7)]],
    uint2                                   gid             [[thread_position_in_grid]]
) {
    uint w = outHDR.get_width();
    uint h = outHDR.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float depth = gDepth.read(gid);
    half4 emH = gEmission.read(gid);

    // Sky pixels (depth still cleared at 1.0). Reconstruct a world-space ray
    // through this pixel and sample the equirect HDR sky.
    float2 ndc = (float2(gid) + 0.5) / float2(w, h) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    if (depth >= 0.99999) {
        // Unproject a far-plane point and subtract the camera to get the
        // world-space view ray. (Cheaper than running invViewProj twice.)
        float4 farClip = float4(ndc, 1.0, 1.0);
        float4 farWorld = frame.invViewProjection * farClip;
        float3 dir = normalize(farWorld.xyz / farWorld.w - frame.cameraWorldPos);
        float3 sky = sampleSkyEquirect(skyEquirect, dir);
        // Analytic night sky (opt-in): pixel-sharp stars + a phase-correct moon
        // composited over the (celestial-free) dome. Zero params ⇒ exact no-op.
        // TAA's jittered projection supersamples the sub-pixel star cores.
        // Angular size of one pixel from the projection: tan(fovY/2) = 1/P[1][1].
        float pixAngle = (2.0f / frame.projection[1][1]) / float(h);
        sky += nightCelestials(dir, frameNightSky(frame), pixAngle);
        outHDR.write(half4(half3(sky), 1.0h), gid);
        // Issue #65 — sky is never SSS; clear its mask so the composite skips it.
        if (frame.sssStrength > 0.0) sssOut.write(half4(0.0h), gid);
        return;
    }

    half4 amH = gAlbedoMet.read(gid);
    half4 nrH = gNormalRgh.read(gid);

    float3 albedo    = float3(amH.rgb);
    float  metallic  = float(amH.a);
    float3 N         = octDecode(float2(nrH.rg));
    float  roughness = max(0.045, float(nrH.b));
    float3 emission  = float3(emH.rgb);
    // Phase 7b — the material's cloth-sheen strength, unpacked from the NEGATIVE emission
    // alpha the G-buffer writes (a surface is polished OR cloth, never both, so one channel
    // carries either). Declared HERE, above every light path, because that is the whole point
    // of the 2026-08-09 re-siting: it used to be read at the very bottom of this kernel, after
    // the point and spot loops had already closed, so a lamp-lit sofa received no sheen at all.
    // 0 for every non-cloth surface, and `brdf` early-outs exactly on 0.
    float  sheenStrength = (emH.a < -0.001h) ? float(-emH.a) : 0.0;
    // Sheen-only accumulator for `DebugTerm.clothSheen` (17). Every `brdf` call below is handed
    // `&clothSheen`, so the term can still be isolated even though it is no longer a separable
    // bolt-on. Untouched (and dead-stripped) when nothing in the scene is cloth.
    float3 clothSheen = float3(0.0);
    // Light-layer bitfield for this fragment. Default (host never set a layer)
    // reads 0xFFFFFFFF, so (light.layerMask & fragLayer) is always non-zero below
    // ⇒ every light contributes exactly as before.
    uint   fragLayer = gLayer.read(gid).r;

    float3 worldPos = worldPosFromDepth(ndc, depth, frame.invViewProjection);
    // Surface → eye. Under a PARALLEL projection the eye is not a point: every
    // pixel is viewed along the same direction, so `cameraWorldPos − worldPos`
    // (which fans out from a finite eye) would put a fake radial gradient in
    // every view-dependent term — specular highlights sliding across a flat wall
    // as if lit by a nearby lamp. The camera's forward is the constant view axis;
    // `invView`'s third column is −forward in this right-handed convention.
    float3 V = (frame.diagramParams.w > 0.5)
             ? normalize(frame.invView[2].xyz)
             : normalize(frame.cameraWorldPos - worldPos);

    // Issue #65 — screen-space SSS. SSS-flagged pixels carry ≈0.95h in
    // normalRoughness.w (vertex-colour alpha ∈ [0.90,0.98]; above the foliage 0.0
    // / plush 0.55 / casing 0.75 bands, below opaque 1.0). When the scene opts in
    // (frame.sssStrength > 0) we accumulate the DIFFUSE-only lit term alongside the
    // normal composite — sun·vis + points + spots + fill-directionals + diffuse-IBL
    // + ambient — and hand it to the separable blur via `sssOut`. The whole branch
    // is dead for non-SSS scenes (sssStrength == 0) and non-SSS pixels (mask 0).
    bool   sssBand  = (nrH.a > 0.90h && nrH.a < 0.99h);
    bool   sssForce = (frame.sssDebugForceAll > 0.5) && (nrH.a > 0.5h); // headless verify
    bool   isSSS    = (frame.sssStrength > 0.0) && (sssBand || sssForce);
    float3 sssDiffuse = float3(0.0);

    // Directional light, attenuated by the cascaded-shadow visibility term.
    // sample_compare returns 1.0 when `reference COMPARE_FUNC sample` holds.
    // We store front-most depths in the shadow map, so the surface is lit when
    // its light-space depth (minus bias) is ≤ the stored depth.
    constexpr sampler shadowSampler(filter::linear,
                                    compare_func::less_equal,
                                    address::clamp_to_edge);
    float3 Ld = normalize(frame.directionalLightDir);
    float NdotL_sun = saturate(dot(N, Ld));
    float visibility = sunVisibility(shadowMap, shadowSampler, frame,
                                     worldPos, N, NdotL_sun);
    // ── Screen-space contact shadows (issue #65) ────────────────────────────
    // Fold a short screen-space ray-march occlusion into the sun visibility so
    // EVERY sun-driven term below (direct BRDF, leaf/plush transmission, the
    // casing clearcoat, plush sheen) darkens consistently at object contacts.
    // Off-by-default: `contactShadowStrength == 0` skips the march entirely.
    if (frame.contactShadowStrength > 0.0 && visibility > 0.0) {
        float3 viewPos = (frame.view * float4(worldPos, 1.0)).xyz;
        float3 LviewN  = normalize((frame.view * float4(Ld, 0.0)).xyz);
        float  occ = illumiContactShadow(gDepth, frame, viewPos, LviewN,
                                         uint2(w, h),
                                         frame.contactShadowLength,
                                         frame.contactShadowSteps,
                                         frame.contactShadowThickness,
                                         NdotL_sun);
        visibility *= saturate(1.0 - frame.contactShadowStrength * occ);
    }
    // Terms kept separate so the per-term split-render (frame.debugTerm) can
    // isolate any one of them; the normal path just sums them at the end.
    // Phase 7c — grain anisotropy: normalRoughness.w carries (1 + aniso) for wood/brushed-metal
    // pixels (opaque is exactly 1.0). Reconstruct an in-plane grain tangent from a world reference
    // (plan-X for floors/ceilings, horizontal for walls) — approximate (per-instance, not per-
    // plank), but it's the highlight STRETCH that kills the plastic look, not the exact grain angle.
    float aniso = (nrH.a > 1.001h) ? float(nrH.a - 1.0h) : 0.0;
    float3 grainT = float3(0.0);
    if (aniso > 0.001) {
        float3 up = float3(0.0, 1.0, 0.0);
        grainT = (abs(dot(N, up)) > 0.95) ? normalize(float3(1.0, 0.0, 0.0) - N * N.x)
                                          : normalize(cross(N, up));
    }
    // NOTE the sheen accounting: `directSun` is scaled by `visibility` AFTER the call, so the
    // sheen `brdf` pushed into `clothSheen` has to be shadowed by hand to match. Every other
    // call site folds its attenuation into `lightColor`, so this is the only one that does.
    float3 directSunSheen = float3(0.0);
    float3 directSun = brdf(N, V, Ld, albedo, metallic, roughness,
                            frame.directionalLightColor, aniso, grainT,
                            sheenStrength, &directSunSheen) * visibility;
    clothSheen += directSunSheen * visibility;
    if (isSSS) sssDiffuse += brdfDiffuse(N, V, Ld, albedo, metallic,
                                         frame.directionalLightColor) * visibility;

    // ── Leaf thin-sheet transmission (issue #58 / #20 item 2) ───────────────
    // Leaves are flagged in normalRoughness.w (0 = foliage; opaque geometry is
    // 1). Opaque leaf cards in the G-buffer read "plastic" because they only
    // ever reflect; real foliage SCATTERS the sun through the thin blade, so a
    // back-lit leaf GLOWS. We add that transmitted term, driven by the actual
    // directional irradiance (`frame.directionalLightColor`) — ChatGPT's "tie
    // it to solar irradiance" — gated by the same cascade-shadow `visibility`
    // so a leaf buried in canopy shade doesn't light up. Zero for every other
    // surface (back == 0 when the sun is on the viewer's side of the leaf, and
    // the whole branch is skipped for non-foliage).
    float3 transmission = float3(0.0);
    if (nrH.a < 0.5h && frame.leafTransmission > 0.0) {   // foliage + scene opt-in
        float back    = saturate(dot(-N, Ld));           // sun on the FAR face
        float forward = pow(saturate(dot(V, -Ld)), 3.0); // viewing toward the sun
        // ROUND-6 POLISH #3: BACKLIT-ONLY GATE. The old `through = back·(0.40+0.60·
        // forward)` carried a 0.40 floor that fired whenever the sun touched the far
        // face — including FORWARD-LIT side-view leaves (sun on the camera's side,
        // V·(−Ld)≈0), which then bloomed pale khaki (the side-view bleach). Tie the
        // term to genuine backlight: a smoothstep on dot(V,−Ld) (== viewing toward
        // the sun) rolls the transmission to ~0 for forward-lit leaves while the
        // three-quarter rim (where V·(−Ld) is high) keeps the FULL term. Mirrored in
        // IlluminatoramaRT.metal so the deferred and RT sun paths agree.
        float backlit = smoothstep(0.02, 0.42, dot(V, -Ld));
        float through = back * backlit * (0.30 + 0.70 * forward);
        // ENERGY CLAMP: a leaf re-emits a FRACTION of the light reaching it, never
        // more. `leafTransmission` was an un-clamped (>1) gain, so under a low warm
        // BACKLIGHT the foliage blew past 1.0 and clipped to white while the bark
        // (no transmission) stayed dark — the connected canopy READ as bare sticks.
        // Clamp the transmitted radiance to the incident so backlit leaves glow but
        // never out-shine the sun. (Mirrored in IlluminatoramaRT.metal's sun pass.)
        float3 t = albedo * frame.directionalLightColor
                 * through * visibility * frame.leafTransmission;
        transmission = min(t, albedo * frame.directionalLightColor * visibility);
    }

    // ── Plush backlit thin-fabric SSS (Teddy Bear Press) ────────────────────
    // Plush pixels carry ≈0.55 in normalRoughness.w. A stuffed-fabric bear is a
    // thin shell over fibrefill, so a BACKLIT bear glows warmly through the fabric
    // — the same thin-sheet transmission as a leaf but softer/warmer and with less
    // forward bias (the fill diffuses it). Reuses the energy-clamped leaf machinery,
    // driven by `frame.plushTransmission` (0 ⇒ skipped). Mirrors the leaf gate so
    // it never fires forward-lit. Zero for every non-plush surface / scene.
    if (nrH.a > 0.5h && nrH.a < 0.6h && frame.plushTransmission > 0.0) {
        float back    = saturate(dot(-N, Ld));
        float backlit = smoothstep(0.02, 0.55, dot(V, -Ld));
        float through = back * backlit * (0.55 + 0.45 * pow(saturate(dot(V, -Ld)), 2.0));
        float3 warm   = frame.directionalLightColor * float3(1.06, 0.96, 0.82);  // warmer through fabric
        float3 t      = albedo * warm * through * visibility * frame.plushTransmission;
        transmission  = min(t, albedo * frame.directionalLightColor * visibility);
    }

    float3 pointSum = float3(0.0);
    float3 spotSum  = float3(0.0);

    // Point-light cube-shadow PCF sampler — hardware depth compare, same config as
    // the spot path. Declared once outside the loop.
    constexpr sampler pointShadowSampler(filter::linear,
                                         compare_func::less_equal,
                                         address::clamp_to_edge);

    // Point lights — inverse-square with smooth radius cutoff. (Includes the
    // synthesised emissive-as-light points from the extractor, Phase 4.27.)
    for (uint i = 0; i < frame.pointLightCount; ++i) {
        PointLight pl = pointLights[i];
        // Light-layer mask: skip lights that don't share a bit with this fragment's
        // layer. Default 0xFFFFFFFF on both sides ⇒ always passes (no change).
        // A SHADOWED light (castsShadow) ignores the mask entirely: the depth cube
        // is the real, geometry-correct occluder (blocks walls AND passes doorways),
        // so applying the coarse position-based room mask on top would double-darken
        // and re-introduce the doorway over-block the shadow exists to fix.
        if (pl.castsShadow == 0u && (pl.layerMask & fragLayer) == 0u) continue;
        float3 toLight = pl.position - worldPos;
        float  dist    = length(toLight);
        if (dist > pl.radius) continue;
        float3 L = toLight / max(dist, 1e-4);
        float atten = 1.0 / max(dist * dist, 1e-4);
        float window = saturate(1.0 - pow(dist / pl.radius, 4.0));
        atten *= window * window;

        // Cubemap shadow visibility. Only when this light opted in AND was assigned
        // a cube page this frame. Pick the cube face from the dominant axis of the
        // light→fragment direction (−L, pointing away from the bulb), project the
        // fragment through that face's VP, and PCF-compare — identical machinery to
        // the spot path, just with a face select in front.
        float visibility = 1.0;
        if (pl.castsShadow != 0u && pl.shadowCubeIndex >= 0) {
            float3 d = -L;                     // bulb → fragment
            float3 ad = abs(d);
            uint face;
            if (ad.x >= ad.y && ad.x >= ad.z)      face = (d.x > 0.0) ? 0u : 1u;  // ±X
            else if (ad.y >= ad.z)                 face = (d.y > 0.0) ? 2u : 3u;  // ±Y
            else                                   face = (d.z > 0.0) ? 4u : 5u;  // ±Z
            uint slice = uint(pl.shadowCubeIndex) * 6u + face;
            float4x4 faceVP = pointShadowFaces[slice];
            float4 lsPos = faceVP * float4(worldPos, 1.0);
            if (lsPos.w > 0.0) {
                float2 lsNDC = lsPos.xy / lsPos.w;
                float  lsZ   = lsPos.z  / lsPos.w;
                float2 shadowUV = float2(lsNDC.x * 0.5 + 0.5,
                                         -lsNDC.y * 0.5 + 0.5);
                bool inFrustum = lsZ > 0.0 && lsZ < 1.0
                              && shadowUV.x >= 0.0 && shadowUV.x <= 1.0
                              && shadowUV.y >= 0.0 && shadowUV.y <= 1.0;
                if (inFrustum) {
                    float ref = lsZ - frame.pointShadowBias;
                    float texel = 1.0 / 512.0;
                    float sum = 0.0;
                    for (int oy = -1; oy <= 1; ++oy) {
                        for (int ox = -1; ox <= 1; ++ox) {
                            sum += pointShadowAtlas.sample_compare(
                                pointShadowSampler,
                                shadowUV + float2(float(ox), float(oy)) * texel,
                                slice,
                                ref);
                        }
                    }
                    visibility = sum * (1.0 / 9.0);
                }
            }
        }
        if (visibility <= 0.0) continue;
        pointSum += brdf(N, V, L, albedo, metallic, roughness, pl.color * atten * visibility,
                         0.0, float3(0.0), sheenStrength, &clothSheen);
        if (isSSS) sssDiffuse += brdfDiffuse(N, V, L, albedo, metallic,
                                             pl.color * atten * visibility);
    }

    // Spot lights — same distance attenuation as point lights, multiplied
    // by a cone-attenuation term that smoothsteps between the outer and
    // inner cone cosines. Phase 4.10 adds an optional per-spot shadow
    // map: when `shadowSliceIndex >= 0`, the fragment's world position
    // is projected into the spot's light space and PCF-compared against
    // the stored depth. `visibility` modulates the contribution.
    constexpr sampler spotShadowSampler(filter::linear,
                                        compare_func::less_equal,
                                        address::clamp_to_edge);
    for (uint i = 0; i < frame.spotLightCount; ++i) {
        SpotLight sl = spotLights[i];
        // Light-layer mask (same rule as point lights). Default all-bits ⇒ passes.
        if ((sl.layerMask & fragLayer) == 0u) continue;
        float3 toLight = sl.position - worldPos;
        float  dist    = length(toLight);
        if (dist > sl.radius) continue;
        float3 L = toLight / max(dist, 1e-4);
        float coneCos   = dot(normalize(sl.direction), -L);
        float coneAtten = smoothstep(sl.outerCone, sl.innerCone, coneCos);
        if (coneAtten <= 0.0) continue;
        float atten = 1.0 / max(dist * dist, 1e-4);
        float window = saturate(1.0 - pow(dist / sl.radius, 4.0));
        atten *= window * window * coneAtten;

        // Shadow visibility. Skip cheaply when the host hasn't provided a
        // shadow slice for this spot — covers `spotShadowsEnabled = false`
        // (slice forced to -1 by `updateSpotShadows`) and any spot beyond
        // the atlas capacity.
        float visibility = 1.0;
        if (sl.shadowSliceIndex >= 0) {
            float4 lsPos = sl.shadowMatrix * float4(worldPos, 1.0);
            float2 lsNDC = lsPos.xy / lsPos.w;
            float  lsZ   = lsPos.z  / lsPos.w;
            // Flip Y to match texture UV convention.
            float2 shadowUV = float2(lsNDC.x * 0.5 + 0.5,
                                      -lsNDC.y * 0.5 + 0.5);
            // Out-of-frustum fragments stay fully lit — the cone attenuation
            // already drops their contribution to ~zero outside the
            // outer cone, so the shadow compare doesn't need to reject
            // them here.
            bool inFrustum = lsZ > 0.0 && lsZ < 1.0
                          && shadowUV.x >= 0.0 && shadowUV.x <= 1.0
                          && shadowUV.y >= 0.0 && shadowUV.y <= 1.0;
            if (inFrustum) {
                float ref = lsZ - frame.spotShadowBias;
                // 3×3 PCF using hardware bilinear compare. Each
                // `sample_compare` returns the linear-filtered fraction
                // of taps that pass; summing 9 gives a smooth penumbra.
                float w = 1.0 / 512.0; // shadow map texel size
                float sum = 0.0;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        sum += spotShadowAtlas.sample_compare(
                            spotShadowSampler,
                            shadowUV + float2(float(ox), float(oy)) * w,
                            uint(sl.shadowSliceIndex),
                            ref);
                    }
                }
                visibility = sum * (1.0 / 9.0);
            }
        }
        if (visibility <= 0.0) continue;
        spotSum += brdf(N, V, L, albedo, metallic, roughness,
                        sl.color * atten * visibility,
                        0.0, float3(0.0), sheenStrength, &clothSheen);
        if (isSSS) sssDiffuse += brdfDiffuse(N, V, L, albedo, metallic,
                                             sl.color * atten * visibility);
    }

    // Rectangular area lights (#60 task 5) — closed-form polygon diffuse + MRP
    // specular, replacing the old five-spot stand-in.
    float3 areaSum = float3(0.0);
    bool areaLTC = frame.areaLTCEnabled != 0u;
    for (uint i = 0; i < frame.areaLightCount; ++i) {
        // Light-layer mask — same rule as point/spot. An area light has no shadow map
        // and no visibility term, so the mask is its ONLY containment (a window portal
        // without it lights the yard through the back of its own wall).
        if ((areaLights[i].layerMask & fragLayer) == 0u) continue;
        areaSum += evalAreaLight(areaLights[i], worldPos, N, V, albedo, metallic, roughness,
                                 ltcMat, ltcMag, areaLTC);
    }

    // Secondary directional lights (#60 task 5) — fill / back lights that a
    // SCN 3-point rig ships alongside the key. Formerly collapsed into a flat
    // hemispheric ambient term (the 4.20 fold), which threw away their direction
    // (no NdotL shading gradient) and their specular entirely. Now each shades
    // with the SAME `brdf` the sun uses — real diffuse + GGX specular — just
    // without a shadow map (SCN fill/back lights are `castsShadow = false`; the
    // primary keeps the cascade rig). `dir` is pre-normalized toward the light.
    float3 dirFillSum = float3(0.0);
    for (uint i = 0; i < frame.directionalLightCount; ++i) {
        DirectionalLight dl = extraDirectionals[i];
        dirFillSum += brdf(N, V, dl.dir, albedo, metallic, roughness, dl.color,
                           0.0, float3(0.0), sheenStrength, &clothSheen);
        if (isSSS) sssDiffuse += brdfDiffuse(N, V, dl.dir, albedo, metallic, dl.color);
    }

    // SSAO (half-res, gid/2). Only the indirect term is modulated — direct
    // lights have their own shadowing pathway.
    uint aoW = aoTex.get_width();
    uint aoH = aoTex.get_height();
    uint2 aoCoord = min(gid / 2, uint2(aoW - 1, aoH - 1));
    float ao = float(aoTex.read(aoCoord).r);

    // ── Interior day-light separation (FrameUniforms.interiorMask) ──────────
    // Both factors stay exactly 1.0 unless the host opted in AND this fragment
    // carries a stamped interior room bit, so the multiplies below are exact
    // no-ops (×1.0) for every scene that never sets `interiorMask`.
    float interiorIBLK = 1.0;
    float interiorAmbK = 1.0;
    // Interior irradiance-band blend weight (FrameUniforms.interiorIrrUp.w). 0 for
    // every exterior fragment and every scene that never sets the bands — the diffuse
    // path below is then the exact cube sample it always was.
    float interiorBandW = 0.0;
    if (frame.interiorMask != 0u && fragLayer != 0xFFFFFFFFu &&
        (fragLayer & frame.interiorMask) != 0u) {
        interiorIBLK = mix(frame.interiorIBLSide, frame.interiorIBLUp, saturate(N.y));
        interiorAmbK = frame.interiorAmbient;
        interiorBandW = saturate(frame.interiorIrrUp.w);
    }
    // The DIFFUSE lobe's interior factor. As the host's irradiance bands take over
    // (w → 1) it folds to exactly 1.0: the bands already carry the up/side/down
    // weighting the `interiorIBLUp/Side` scalars faked, and scaling them again would
    // double-apply it. The SPECULAR lobe keeps `interiorIBLK` — it still samples the
    // outdoor prefiltered cube (RT reflections own the structured part of that story).
    float interiorIBLKd = mix(interiorIBLK, 1.0, interiorBandW);

    // ── Indirect (IBL + optional DDGI for diffuse) ──────────────────
    float3 indirect;
    // Debug accumulators (frame.debugTerm split-render) — populated below.
    float3 dbgDiffuseIBL = float3(0.0);
    float3 dbgSpecularIBL = float3(0.0);
    float3 dbgAmbient = float3(0.0);
    if (kLightingIBLEnabled) {  // function_constant(0)
        constexpr sampler cubeSampler(filter::linear, mip_filter::linear);

        float3 F0 = mix(float3(0.04), albedo, metallic);
        float NdotV = saturate(dot(N, V));
        float3 F  = fresnelSchlickRoughness(NdotV, F0, roughness);
        float3 kD = (1.0 - F) * (1.0 - metallic);

        // Diffuse: use DDGI probe irradiance when available (one-bounce GI),
        // otherwise fall back to the sky-probe irradiance cube.
        // Phase 3.4 — EMA irradiance cache cuts the per-pixel 8-probe blend to
        // 1 texture read in steady state. Fresh probe lookup only on cache miss
        // (first frame, cache disabled, or DDGI off).
        float3 irradianceSrc;
        if (kLightingDDGIEnabled) {  // function_constant(3) — agrees with ddgi.enabled
            // Phase 3.4b — velocity-reproject the cache read. Sampling the
            // history at the *current* UV means a moving camera reads the EMA
            // that belongs to whatever world point used to sit under this pixel,
            // so a fast move smears the cache (it reconverges over ~20 frames).
            // Project THIS pixel's world position by the previous frame's VP to
            // fetch the history for the SAME surface point: fast moves reconverge
            // in ~1 frame. Off-screen reprojection (disocclusion) → fresh lookup.
            float4 prevClip  = frame.previousViewProjection * float4(worldPos, 1.0);
            float2 reprojUV  = (prevClip.xy / prevClip.w) * float2(0.5, -0.5) + 0.5;
            bool   reprojOK  = prevClip.w > 1e-5
                            && all(reprojUV >= float2(0.0))
                            && all(reprojUV <= float2(1.0));
            constexpr sampler cacheSmp(filter::linear, address::clamp_to_edge);
            float3 history  = reprojOK
                            ? float3(irrCachePrev.sample(cacheSmp, reprojUV).rgb)
                            : float3(0.0);
            bool   useCache = kLightingDDGIIrrCacheEnabled  // function_constant(4)
                           && reprojOK
                           && any(history > float3(1e-6f));
            if (useCache) {
                float3 fresh = sampleDDGIIrradiance(worldPos, N,
                                                    ddgiIrrAtlas, ddgiDepthAtlas, ddgi);
                irradianceSrc = mix(history, fresh, frame.ddgiIrrCacheBlend);
            } else {
                irradianceSrc = sampleDDGIIrradiance(worldPos, N,
                                                     ddgiIrrAtlas, ddgiDepthAtlas, ddgi);
            }
            irrCacheCur.write(half4(half3(irradianceSrc), 1.0h), gid);
        } else {
            irradianceSrc = float3(irradianceCube.sample(cubeSampler, N).rgb);
            irrCacheCur.write(half4(0.0h), gid);
        }
        // Phase 4.15 — IBL diffuse saturation boost. Procedural-gradient
        // backdrops that the extractor reuses as the IBL probe integrate
        // down to a near-grey irradiance, and `irradiance * albedo` then
        // multiplies grey × albedo → a muted tint on every indirect-lit
        // surface. We push the IBL diffuse term's chroma outward from
        // its own luminance before composing with albedo, which keeps
        // hue but lifts saturation back toward what the SCN baseline
        // shows. 1.0 disables the boost (no-op); >1.0 boosts.
        //
        // Mathematically: this saturates THE LIGHTING (the irradiance
        // probe), not the surface. The surface's albedo still drives
        // the hue; we're just refusing to let the integration's grey
        // average wash it out.
        // Phase 4.29 — ADAPTIVE saturation. The Phase 4.15 boost amplifies
        // whatever chroma the irradiance probe carries; that rescues grey /
        // pastel procedural-gradient IBL (Forest, FloatingFlowers+), but on
        // an already-warm IBL (Pizza's red oven environment) it pushes the
        // red even further, flooding the frame. Roll the boost off toward
        // 1.0 (no-op) as the source irradiance's own saturation rises, so
        // low-chroma probes get the full lift while saturated probes are
        // left alone.
        float irrLum = dot(irradianceSrc, float3(0.2126, 0.7152, 0.0722));
        float irrMax = max(irradianceSrc.r, max(irradianceSrc.g, irradianceSrc.b));
        float srcSat = (irrMax > 1e-4) ? saturate((irrMax - irrLum) / irrMax) : 0.0;
        float satBoost = mix(frame.iblDiffuseSaturation, 1.0,
                             smoothstep(0.18, 0.5, srcSat));
        // Phase 4.32 — hue-balance DESATURATION for saturated probes. The 4.15 /
        // 4.29 boost only ever lifts chroma (≥1.0); a colored-environment scene
        // (Pizza's red broiler used as IBL) needs the opposite: pull the
        // monochromatic-red irradiance toward its luminance so a cream surface
        // isn't washed pure red. Ramps in only as the probe's own saturation
        // rises, so neutral / pastel IBL (Forest, FloatingFlowers+) is untouched.
        float desat = frame.iblDiffuseDesaturation * smoothstep(0.35, 0.75, srcSat);
        float effBoost = satBoost * (1.0 - desat);
        float3 irradianceSat = max(mix(float3(irrLum), irradianceSrc, effBoost), 0.0);
        // ── Interior irradiance bands (see FrameUniforms.interiorIrr*) ───────
        // Replace the OUTDOOR cube's irradiance with the room's own three-band
        // environment on interior fragments: a ceiling's hemisphere sees the
        // floor's bounce, not the lawn on the far side of it. Host-authored
        // values — the grey-probe saturation rescue above exists for baked
        // probes and must not distort them, so the blend happens after it.
        if (interiorBandW > 0.0) {
            float3 band = mix(frame.interiorIrrSide.xyz,
                              N.y >= 0.0 ? frame.interiorIrrUp.xyz
                                         : frame.interiorIrrDown.xyz,
                              abs(N.y));
            // ── S3.5 Stage D — the bands GRADE with daylight-aperture proximity ──
            // The bands (like the scalars before them) are functions of the surface
            // NORMAL alone, so a wall point 0.5 m from its window and one 8 m away
            // read identically (R2wall ≈ 0.98, measured — the flatness no dial could
            // fix). Grade them by summed daylight PROXIMITY over the room's glazed
            // apertures:
            //     Ω_i = A / (d² + A)
            // — the finite-rect solid-angle form WITHOUT foreshortening, on purpose.
            // A cosine-weighted (sky-visibility) form cancels along a side wall: the
            // far end sees the window more squarely exactly as it sees it farther,
            // and the gradient this term exists to produce vanishes. What actually
            // brightens the wall beside a window in a photograph is the bounce glow
            // of the lit zone in front of it, which is isotropic — proximity, not
            // aspect. The aperture's inward normal is used only for CONTAINMENT
            // (nothing behind the window plane — the outdoors, the next room on the
            // far side of the same wall — may brighten). Cross-room leak through an
            // interior partition is accepted the way pre-mask point lights accepted
            // it: 1/d² bounds it, and the clamp bounds its worst case.
            //
            // Normalized so Ω_ref (a 3 m² window at ~3.5 m — mid-room) maps to 1.0,
            // then root-compressed and clamped: this is an AMBIENT gradient, not a
            // spotlight — the direct sun + window cones carry the hard directional
            // story. Count/strength ride the band clusters' spare .w lanes — both 0
            // unless the host opts in, keeping the factor exactly 1.0 (and the loop
            // unentered) for every other scene.
            uint  apCount = uint(frame.interiorIrrDown.w);
            float apStrength = saturate(frame.interiorIrrSide.w);
            if (apCount > 0u && apStrength > 0.0) {
                float omega = 0.0;
                for (uint i = 0u; i < apCount; ++i) {
                    InteriorAperture ap = interiorApertures[i];
                    float3 toP = worldPos - ap.center;
                    if (dot(ap.inward, toP) < -0.05) continue;   // outdoor side
                    float d2   = max(dot(toP, toP), 1e-3);
                    float area = ap.width * ap.height;
                    omega += area / (d2 + area);
                }
                const float omegaRef = 0.20;
                float factor = clamp(sqrt(omega / omegaRef), 0.55, 1.65);
                band *= mix(1.0, factor, apStrength);
            }
            irradianceSat = mix(irradianceSat, band, interiorBandW);
        }
        float3 diffuseIBL = kD * irradianceSat * albedo;

        // Specular IBL: sample prefilteredCube along the world reflection
        // vector at the roughness-derived mip, then apply the split-sum DFG
        // LUT (Phase 3.2) for the F0-weighted environment BRDF integral:
        //   specular = prefilteredEnv * (F0 * dfg.x + dfg.y).
        // The LUT is baked once at startup (`illumi_dfg_bake`). When
        // `frame.dfgLUTEnabled == 0` we fall back to Lagarde's roughness-
        // Schlick (`F` above) — visually decent on dielectrics, drops energy
        // at high roughness / grazing on metals. Toggle preserved for A/B.
        float3 R = reflect(-V, N);
        float mipCount = float(max(frame.iblPrefilteredMipCount, 1u));
        float lod = roughness * (mipCount - 1.0);
        float3 specEnv = float3(prefilteredCube.sample(cubeSampler, R, level(lod)).rgb);
        float3 specularIBL;
        if (kLightingDFGLUTEnabled) {  // function_constant(2)
            constexpr sampler dfgSampler(filter::linear, address::clamp_to_edge);
            float2 dfg = float2(dfgLUT.sample(dfgSampler, float2(NdotV, roughness)).rg);
            float3 FssEss = F0 * dfg.x + dfg.y;      // the split-sum single-scatter result
            // ── S1.3b — multi-scatter GGX energy compensation ────────────────
            // Single-scattering GGX DROPS the light that would have bounced a
            // second time between microfacets, and the loss grows with roughness:
            // `dfg.x + dfg.y` IS the directional albedo of the lobe, so
            // `Ems = 1 − (dfg.x + dfg.y)` is exactly the energy going missing.
            // On a dielectric that is invisible (F0 = 0.04 ⇒ the returned light is
            // ~0.5 %), but on a metal it is a colour shift as well as a darkening,
            // because what is dropped is Fresnel-weighted — rough gold renders
            // grey-brown instead of gold.
            //
            // Fdez-Agüera 2019 ("A Multiple-Scattering Microfacet Model for
            // Real-Time Image-Based Lighting", JCGT 8.1) closes it with ONE extra
            // term built from the LUT already sampled above: the missing energy is
            // re-emitted having undergone an average Fresnel `F_avg`, which for
            // Schlick integrates to `F0 + (1 − F0)/21`, and the geometric series of
            // further bounces sums to `1 / (1 − F_avg·Ems)`. No new texture, no new
            // sample — four lines on top of the split sum.
            float  Ems    = saturate(1.0 - (dfg.x + dfg.y));
            float3 Favg   = F0 + (1.0 - F0) * (1.0 / 21.0);
            float3 FmsEms = (Ems * FssEss * Favg) / max(1.0 - Favg * Ems, 1e-4);
            specularIBL = specEnv * (FssEss + FmsEms);
        } else {
            // No LUT ⇒ no `Ems` to compensate with. Lagarde's roughness-Schlick
            // fallback has no energy budget to speak of; leaving it alone keeps this
            // toggle a clean A/B of the split sum itself.
            specularIBL = specEnv * F;
        }

        // ── S1.3a — specular occlusion ───────────────────────────────────────
        // `ao` is a DIFFUSE visibility estimate: it answers "how much of the whole
        // hemisphere can this point see?", which is the right question for a
        // Lambertian lobe and the wrong one for a specular one. A near-mirror
        // integrates a narrow cone around the reflection vector, and in a shallow
        // cavity that cone is still looking at open sky, so multiplying it by raw
        // `ao` over-darkens every glossy surface near a wall, a corner or a crease.
        //
        // Frostbite's horizon-based approximation (Lagarde & de Rousiers, *Moving
        // Frostbite to PBR*, §4.10.2) reconstructs a specular visibility from the
        // three things that decide it: how much of the hemisphere is open (`ao`),
        // how far the reflection lobe leans toward the horizon (`NdotV`), and how
        // wide the lobe is (`roughness`, through the exponent). The exponent
        // `exp2(−16·roughness − 1)` is what carries the physics: at roughness 1 it
        // is 7.6e−6, `pow(x, e) → 1`, and the whole expression collapses to exactly
        // `ao` — a matte surface is occluded exactly as much as a diffuse one, which
        // is correct and also makes this an arithmetic no-op for most of the
        // material library. It only opens up as the surface polishes.
        float specOcc = saturate(pow(NdotV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);

        // Diffuse and the ambient supplement below keep RAW `ao`; only the specular
        // lobe takes `specOcc`.
        indirect = (diffuseIBL * ao * interiorIBLKd + specularIBL * specOcc * interiorIBLK)
                 * frame.iblIntensity;
        dbgDiffuseIBL = diffuseIBL * frame.iblIntensity * ao * interiorIBLKd;
        dbgSpecularIBL = specularIBL * frame.iblIntensity * specOcc * interiorIBLK;
        // ── Cloth sheen, environment arm ─────────────────────────────────────
        // The direct arm lives in `brdf`; this is the other half. A cushion in a room the sun
        // never reaches is lit almost entirely by the environment, and the Phase-7b bolt-on
        // never consulted it — `ambSheen` read `frame.ambientColor` alone, which is 0 in every
        // Daydream Home scene, so an indoor sofa had no sheen from ANY source. Driven by the
        // SAME irradiance the diffuse lobe uses, through the same `iblIntensity × interiorIBLK`
        // and raw `ao`, so it dims with the room exactly as the diffuse does.
        if (sheenStrength > 0.0) {
            float3 sheenEnv = clothSheenColor(albedo)
                            * (sheenStrength * clothSheenEnvAlbedo(NdotV))
                            * irradianceSat * ao * frame.iblIntensity * interiorIBLKd;
            indirect += sheenEnv;
            clothSheen += sheenEnv;
        }
        // `ambientColor` is now a TRUE ambient term — only SCN `.ambient`
        // lights (uniform, no NdotL) feed it. As of #60 task 5 the secondary
        // SCN directionals (fill, back) are NO LONGER folded in here; they
        // shade with a real BRDF in `dirFillSum` above. The remaining ambient
        // supplement is upness-weighted (40% at the nadir → 100% at the
        // zenith), × albedo, AO-gated. `ambientColor == 0` is the default no-op.
        float upness = saturate(N.y * 0.5 + 0.5);
        float3 ambCol = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation);
        float3 ambSupp = mix(ambCol * 0.4, ambCol, upness) * albedo;
        indirect += ambSupp * ao * interiorAmbK;
        dbgAmbient = ambSupp * ao * interiorAmbK;
        // Ambient sheen — the same lobe against the flat ambient term, so a scene that lights
        // its interior with `ambientColor` rather than an IBL probe still gets fabric. Uses
        // `interiorAmbK` for the same reason the supplement above does. Exact no-op when
        // `ambientColor` is 0 (every Daydream Home scene today) or the surface is not cloth.
        if (sheenStrength > 0.0) {
            float3 ambSheen = clothSheenColor(albedo)
                            * (sheenStrength * clothSheenEnvAlbedo(NdotV))
                            * mix(ambCol * 0.4, ambCol, upness) * ao * interiorAmbK;
            indirect += ambSheen;
            clothSheen += ambSheen;
        }
    } else {
        // Legacy hemispheric ambient — only the diffuse term, no spec.
        float upness = saturate(N.y * 0.5 + 0.5);
        float3 ambCol = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation);
        float3 amb = mix(ambCol * 0.4, ambCol, upness) * albedo;
        indirect = amb * ao * interiorAmbK;
        dbgAmbient = amb * ao * interiorAmbK;
    }

    // Issue #65 — fold the indirect DIFFUSE (diffuse-IBL irradiance + ambient
    // supplement) into the SSS side buffer. `dbgSpecularIBL` is deliberately
    // excluded — specular must stay sharp under SSS. (In the no-IBL branch
    // dbgDiffuseIBL is 0 and dbgAmbient carries the hemispheric diffuse.)
    if (isSSS) sssDiffuse += dbgDiffuseIBL + dbgAmbient;

    // ── Phase 7 — per-material clearcoat (terrazzo/marble/lacquered wood) ──────
    // Clearcoat strength is packed into emission.alpha in the G-buffer (default 0).
    // A second dielectric GGX lobe (F0=0.04, roughness=0.08) is added for
    // polished surfaces. Strength 0 → no cost (branch exits immediately).
    float3 clearcoat = float3(0.0);
    float houseCC = float(emH.a);
    if (houseCC > 0.001f) {
        const float ccRough = 0.08;    // tight polish (terrazzo/marble)
        const float ccF0    = 0.04;    // dielectric IOR 1.5
        float ccNdotV = saturate(dot(N, V));
        if (NdotL_sun > 0.0) {
            float3 Hcc = normalize(V + Ld);
            float  Dcc = distributionGGX(saturate(dot(N, Hcc)), ccRough);
            float  Gcc = geometrySmith(ccNdotV, NdotL_sun, ccRough);
            float  Fcc = ccF0 + (1.0 - ccF0) * pow(1.0 - saturate(dot(Hcc, V)), 5.0);
            float  spec = (Dcc * Gcc * Fcc) / max(4.0 * ccNdotV * NdotL_sun, 1e-4);
            clearcoat += frame.directionalLightColor * spec * NdotL_sun * visibility;
        }
        if (kLightingIBLEnabled) {
            constexpr sampler ccSampler(filter::linear, mip_filter::linear);
            float3 Rcc  = reflect(-V, N);
            float  mips = float(max(frame.iblPrefilteredMipCount, 1u));
            float3 env  = float3(prefilteredCube.sample(ccSampler, Rcc,
                                                        level(ccRough * (mips - 1.0))).rgb);
            float  Fcc  = ccF0 + (1.0 - ccF0) * pow(1.0 - ccNdotV, 5.0);
            // Interior separation: a lacquered interior floor's clearcoat environment
            // reflection is the same open-top skylight the diffuse IBL is — dim it by
            // the same factor (×1.0 no-op when the feature is off).
            // NOTE: the clearcoat lobes deliberately keep RAW `ao`, unlike the base
            // specular IBL above (S1.3a). They are specular and physically want the
            // same treatment — a 0.08-roughness coat is about as narrow a lobe as this
            // renderer has, so specular occlusion would open them up the most of
            // anything — but every clearcoated material in the library (terrazzo,
            // marble, polished concrete, lacquered wood) would shift at once, and
            // S1.3 is scoped to the specular IBL term. Left as a follow-up rather than
            // an unmeasured side effect.
            clearcoat += env * Fcc * frame.iblIntensity * ao * interiorIBLK;
        }
        clearcoat *= houseCC;
        // Energy conservation: clearcoat layer attenuates the base for grazing V.
        float baseAtten = 1.0 - houseCC * (ccF0 + (1.0 - ccF0) * pow(1.0 - saturate(dot(N, V)), 5.0));
        directSun    *= baseAtten;
        indirect     *= baseAtten;
    }

    // ── Sausage-casing clearcoat (HotdogDropUltra) ──────────────────────────
    // Casing pixels carry 0.75 in normalRoughness.w (foliage = 0, opaque = 1).
    // HotdogDrop+'s frank is a SATIN base (roughness mottle mean ≈ 0.48) under
    // a thin wet-glaze clearcoat (SCNMaterial clearCoat 0.55, ccRoughness 0.18)
    // — the glaze is what carries the tight "just off the grill" glint. A
    // single GGX lobe can't be both satin and glinting (rounds 30↔31 oscillated
    // between silicone and foam-rubber trying), so this is a real second lobe:
    // fixed-F0 dielectric GGX on the sun + a tight prefiltered-IBL sample,
    // weighted by Drop+'s 0.55 coat strength. No-op for every other scene.
    float3 hotdogCC = float3(0.0);
    if (nrH.a > 0.6h && nrH.a < 0.9h) {
        const float ccRough    = 0.18;
        const float ccF0       = 0.04;
        const float ccStrength = 0.55;
        float ccNdotV = saturate(dot(N, V));
        if (NdotL_sun > 0.0) {
            float3 Hcc = normalize(V + Ld);
            float  Dcc = distributionGGX(saturate(dot(N, Hcc)), ccRough);
            float  Gcc = geometrySmith(ccNdotV, NdotL_sun, ccRough);
            float  Fcc = ccF0 + (1.0 - ccF0) * pow(1.0 - saturate(dot(Hcc, V)), 5.0);
            float  spec = (Dcc * Gcc * Fcc) / max(4.0 * ccNdotV * NdotL_sun, 1e-4);
            hotdogCC += frame.directionalLightColor * spec * NdotL_sun * visibility;
        }
        if (kLightingIBLEnabled) {
            constexpr sampler ccSampler(filter::linear, mip_filter::linear);
            float3 Rcc  = reflect(-V, N);
            float  mips = float(max(frame.iblPrefilteredMipCount, 1u));
            float3 env  = float3(prefilteredCube.sample(ccSampler, Rcc,
                                                        level(ccRough * (mips - 1.0))).rgb);
            float  Fcc  = ccF0 + (1.0 - ccF0) * pow(1.0 - ccNdotV, 5.0);
            hotdogCC += env * Fcc * frame.iblIntensity * ao;
        }
        hotdogCC *= ccStrength;
    }

    // ── Plush fur sheen rim (Teddy Bear Press) ───────────────────────────────
    // Plush pixels carry ≈0.55 in normalRoughness.w. Short-fibre fur is retro-
    // reflective at grazing angles — the velvety bright edge a real teddy catches.
    // A cheap Fresnel-weighted, NdotL-modulated sheen on the sun + ambient, gated
    // by `frame.plushSheen` (0 ⇒ skipped). No-op for every non-plush surface.
    float3 plushSheenTerm = float3(0.0);
    if (nrH.a > 0.5h && nrH.a < 0.6h && frame.plushSheen > 0.0) {
        float fres = pow(1.0 - saturate(dot(N, V)), 3.0);     // grazing-angle edge
        float3 sheenTint = albedo * 0.5 + float3(0.5);        // warm-white fibre tip
        // Sun-lit sheen + a soft ambient sheen so back/side fur still fuzzes.
        float3 sunSheen = frame.directionalLightColor * (NdotL_sun * visibility);
        float3 ambSheen = desaturateFill(frame.ambientColor, frame.iblDiffuseDesaturation) * ao;
        plushSheenTerm = sheenTint * fres * frame.plushSheen * (sunSheen + ambSheen);
    }

    // ── Phase 7b cloth sheen — NO LONGER SUMMED HERE (2026-08-09) ─────────────
    // The lobe used to be a bolt-on at exactly this point: one grazing-Fresnel term built from
    // `frame.directionalLightColor · NdotL_sun · visibility` plus `frame.ambientColor · ao`,
    // added to `color` after the point and spot loops had closed. It could therefore see only
    // the sun — a lamp-lit sofa got nothing, and neither did one lit purely by the sky, because
    // `ambientColor` is 0 in every Daydream Home scene and the IBL was never consulted. That is
    // photorealism gap #8, "upholstery reads as plaster", measured at
    // `HouseRenderBridgeGPUTests_ClothSheen.testB0ClothSheenTermAcrossFourLightingConfigs`:
    // night-with-lamps and night-with-everything-off returned the SAME number to the last digit.
    //
    // It now lives inside `brdf` (Charlie D + Ashikhmin/Neubelt V), so every light type picks it
    // up from ONE place with no per-light-type code, plus an environment arm in the IBL block.
    // `clothSheen` above is only the DEBUG accumulator — the energy is already inside
    // `directSun` / `pointSum` / `spotSum` / `dirFillSum` / `indirect`, so adding it again here
    // would double-count it.
    float3 color = directSun + transmission + dirFillSum + pointSum + spotSum + areaSum + indirect + emission + clearcoat + hotdogCC + plushSheenTerm;

    // Per-term split-render: isolate ONE contribution so a flooded/flat scene
    // can be decomposed. Surfaces only — sky already returned above.
    switch (frame.debugTerm) {
        case 1u: color = directSun + transmission + dirFillSum; break; // direct sun + leaf SSS + fill directionals
        case 2u: color = pointSum;        break;  // point + synthesised emissive lights
        case 3u: color = spotSum + areaSum; break; // spot + area lights
        case 4u: color = dbgDiffuseIBL;   break;  // diffuse IBL (× iblIntensity × ao)
        case 5u: color = dbgSpecularIBL;  break;  // specular IBL (× iblIntensity × ao)
        case 6u: color = emission;        break;  // G-buffer emissive surface
        case 7u: color = dbgAmbient;      break;  // ambient supplement
        // 17 — the per-material cloth sheen lobe on its own. NOT a G-buffer channel: the
        // tonemap's `debugTerm >= 10` branch has no case for 17, so it falls through and this
        // value is tonemapped exactly like cases 1–7. Zero on every non-cloth surface.
        case 17u: color = clothSheen;     break;  // cloth sheen (velvet/linen/wool)
        default: break;                           // 0 = full composite
    }
    outHDR.write(half4(half3(color), 1.0h), gid);

    // Issue #65 — hand the diffuse-lit term to the separable SSS blur. rgb = the
    // diffuse irradiance peeled off above; a = the SSS mask (1 = blur, 0 = leave).
    // Skipped entirely when the scene hasn't opted in (sssStrength == 0), so the
    // binding is an unread dummy and non-SSS scenes pay nothing here.
    if (frame.sssStrength > 0.0) {
        sssOut.write(half4(half3(sssDiffuse), isSSS ? 1.0h : 0.0h), gid);
    }
}
