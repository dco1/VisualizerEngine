#include <metal_stdlib>
#include <metal_raytracing>
// The ONE secondary-ray surface shader (mirror structs, RNG, scatter-cone policy,
// surface-cache read, and `shadeSecondarySurface`), shared with the AAA glass pass
// in IlluminatoramaGlassRT.metal. A GI bounce and a glossy reflection ray are
// secondary rays like any other — they shade their hits with the same body, so a
// term added for one path cannot go missing on the other. Nothing declared there
// may be re-declared here.
#include "IlluminatoramaSecondary.h"
using namespace metal;
using namespace raytracing;

// ── ILLUMINATORAMA INSTANCED RAY TRACING (TLAS) ──────────────────────────────
//
// The TLAS variant of `illumi_rt_lighting`. Where the room path traces a single
// world-space triangle soup (one primitive AS, rebuilt whenever anything
// moves), this traces an INSTANCE acceleration structure: per-mesh BLAS built
// once in object space, a TLAS of per-instance transforms refit each frame.
// That removes the per-frame CPU vertex transform + geometry-AS rebuild, so RT
// generalises to ANIMATED extracted scenes cheaply.
//
// At a hit the intersector returns `instance_id` (→ per-instance albedo +
// normal matrix + the instance's slot in the concatenated object-normal
// buffer) and `primitive_id` (→ the triangle within that mesh). The world
// geometric normal is `normalize(normalMatrix · objNormal[triBase+prim])`.

struct RTInstUniforms {
    float4x4 invViewProjection;
    float3 cameraWorldPos;   float _pad0;
    float3 sunDir;           float sunSoftnessRad;
    float3 sunColor;         float giStrength;
    float3 skyAmbient;       float specStrength;
    uint  width;   uint height;
    uint  shadowRays; uint giRays;
    uint  frameSeed;  float rayTMin;  float maxGIDist;  uint _pad1;
    float reflStrength; float reflMaxDist; float reflRoughnessCutoff;
    uint  reflRays;  uint reflEnabled;
    // Surface cache (P1c): read cached multi-bounce radiance at GI/reflection
    // hits when enabled. A TLAS hit's (instance_id, primitive_id) resolves to a
    // GLOBAL soup triangle via `soupTriBase[instance_id] + primitive_id`, which
    // indexes the same per-triangle card buffers the soup path uses.
    uint  surfCacheEnabled;
    uint  surfTileSize; uint surfTilesPerRow; uint surfAtlasW; uint surfAtlasH;
    uint  surfTriCount;   // bound for soupTriBase[iid]+prim — OOB skips the read
    // Debug isolation (DebugTerm.surfaceCacheGI). When 1, the kernel writes ONLY
    // the surface-cache-derived term (GI + reflection cache reads), REPLACING the
    // lit composite, so a moving object's stale-vs-fresh cache is pixel-obvious
    // (the cache contribution is otherwise a weak secondary term — see the
    // surface-cache-incremental-invalidation design note).
    uint  debugSurfCacheGI;
    // Curve primitives (#60 item 7): TLAS instance ids >= curveInstanceBase are
    // curve sets (id - base = index into the RTCurveSetData buffer). Unread by
    // the base (curve-free) pipeline variant.
    uint  curveInstanceBase;
    uint  curveSetCount;
    // Debug isolation (DebugTerm.surfaceCacheVariance, Phase 5 / B0). When 1,
    // REPLACES the composite with the per-texel cache variance (E[L²] − μ²)
    // sampled at GI/reflection cache hits — a heatmap of how converged the cache
    // is. Converged surfaces read dark; freshly-reset / cold cards read bright.
    // The signal B1's à-trous denoiser targets. TLAS path only (like term 8).
    uint  debugSurfCacheVar;
    // Phase 5 / A (streaming) — residency feedback. When 1, a GI/reflection cache
    // hit marks `cardRequested[hitCard] = 1` (the working-set signal A1's residency
    // pass keys off). Marks the HIT card, not the viewing pixel's surface (a
    // directly-viewed surface samples its NEIGHBOURS' caches, never its own). Plain
    // store — races are benign (every writer writes 1). Default 0 ⇒ zero cost.
    uint  surfFeedbackEnabled;
    // ── Secondary-hit shading parity with the deferred pass ──────────────────
    // A GI / reflection hit used to be shaded by a strictly poorer model than the
    // same surface one pixel away: the instance's MEAN albedo (no texture tap, so
    // a plank floor and a tiled wall each collapsed to one colour), a flat
    // exterior-strength `albedo * skyAmbient` fill, no local lights and no
    // emission at all. These feed the SHARED `SecondaryShadeParams` — see
    // IlluminatoramaSecondary.h — so this path and the AAA glass pass shade a hit
    // with one body. All default to off/neutral: a host that sets none of them
    // renders exactly as before, except for the texture tap.
    float skyIntensity;         // scales the irradiance cube (== iblIntensity)
    uint  interiorMask;         // interior day-light separation (0 = off)
    float interiorIBLUp;
    float interiorIBLSide;
    float interiorAmbient;
    uint  albedoAtlasEnabled;   // 1 ⇒ objUV (buffer 15) + albedoAtlas (texture 7) live
    uint  objUVCount;           // bound of objUV in float2 entries
    uint  pointLightCount;      // local lights at buffers 16 / 17
    uint  spotLightCount;
    // ── C1: who computes the sun's DIRECT term ───────────────────────────────
    // 0 ⇒ the DEFERRED lighting pass already shaded the sun at this pixel, so this
    // kernel must NOT shade it again. This pass is ADDITIVE on top of a complete
    // deferred frame (`outHDR.write(prev.rgb + …)` at the bottom), and the frame
    // graph does not suppress deferred lighting when RT is on — it only picks
    // WHICH RT pass runs. With this at 0 and `giStrength`/`reflStrength` also 0,
    // this kernel adds EXACTLY zero and an RT-on frame is bit-identical to an
    // RT-off one. That property is the gate; see `IlluminatoramaRenderer
    // .RTSunOwnership`.
    // 1 ⇒ the host handed the sun to RT (deferred directional + cascades forced
    // off), and the soft-disc penumbra below is the only sun in the frame.
    uint  directSunEnabled;
    // ── C3: which TLAS instances stop a TRANSPORT ray ────────────────────────
    // Mask for shadow + GI rays. 0x01 = opaque; 0x04 = "invisible occluder" — a
    // slab that is real to LIGHT but never drawn (Daydream's lighting-only
    // `ceilshadow.*` ceilings over a roofless dollhouse). Without 0x04 a GI ray
    // leaving an interior surface through the open top hits nothing and returns
    // FULL SKY, flooding the room with the exact patio light the ceiling exists
    // to stop. Camera-visible rays (glass refraction, and the glossy reflections
    // below) deliberately do NOT carry 0x04 — an undrawn slab must not appear in
    // the picture. Host default 0x05.
    uint  transportRayMask;
    uint  _padIrr0; uint _padIrr1; uint _padIrr2;   // align the float4 cluster below
    // ── Interior irradiance bands (mirror of FrameUniforms.interiorIrr*) ─────
    // A GI bounce or reflection landing on an interior ceiling must see the
    // FLOOR's bounce, not the outdoor cube's lawn — same fix, same values, as
    // the deferred pass. xyz = irradiance, `interiorIrrUp.w` = blend weight
    // (0 = the exact cube sample — the default, byte-identical).
    float4 interiorIrrUp;
    float4 interiorIrrSide;
    float4 interiorIrrDown;
};

// ── Curve primitives (#60 item 7) ────────────────────────────────────────────
// `kRTCurvesEnabled` specializes the kernel for a TLAS that contains curve
// BLAS instances (round Catmull-Rom). The base variant (constant undefined →
// false) keeps the original triangle-only intersector contract — curve-free
// scenes run the exact code they always did.
constant bool kRTCurvesEnabledFC [[function_constant(30)]];
constant bool kRTCurvesEnabled = is_function_constant_defined(kRTCurvesEnabledFC) && kRTCurvesEnabledFC;

// Mirror of the Swift `RTCurveSetData` (112 B). m0..m3 = the set's
// object→world matrix columns; meta.x = the set's first segment index into the
// pooled segment buffer.
struct RTCurveSetData {
    float4 m0; float4 m1; float4 m2; float4 m3;
    float4 albedoRoughness;   // xyz albedo, w roughness
    float4 emissionPad;       // xyz emission
    uint4  meta;              // x = segment base
};

// Catmull-Rom point at t for one curve segment (Metal's RT convention: the
// segment spans P1..P2; P0/P3 steer the tangents — uniform CR, tension 0.5).
static inline float3 crPoint(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * t
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

// World normal of a round-curve hit: the exact radial direction from the
// curve axis at the hit parameter to the hit point (rigid set transforms —
// the registry's contract — keep this exact under the instance matrix).
static inline float3 curveHitNormal(uint setIdx, uint prim, float t,
                                    float3 hitP,             // world
                                    const device RTCurveSetData* curveSets,
                                    const device packed_float3* curvePts,
                                    const device uint* curveSegs) {
    RTCurveSetData cs = curveSets[setIdx];
    uint i0 = curveSegs[cs.meta.x + prim];
    float3 p0 = float3(curvePts[i0 + 0u]), p1 = float3(curvePts[i0 + 1u]);
    float3 p2 = float3(curvePts[i0 + 2u]), p3 = float3(curvePts[i0 + 3u]);
    float3 axisObj = crPoint(p0, p1, p2, p3, t);
    float4x4 M = float4x4(cs.m0, cs.m1, cs.m2, cs.m3);
    float3 axisW = (M * float4(axisObj, 1.0)).xyz;
    float3 n = hitP - axisW;
    float nl = length(n);
    return nl > 1e-8 ? n / nl : float3(0.0, 1.0, 0.0);
}

// `SurfCard`, `PrimUV`, `RTInstanceData` and `sampleSurfCacheRT` are the shared
// mirror structs / cache read — see IlluminatoramaSecondary.h.

// Phase 5 / B0 — variance readout for the cache-variance debug term. Same
// card/UV addressing as sampleSurfCacheRT, but returns the texel's tracked
// variance E[L²] − μ² (μ² from the stored RGB luminance, E[L²] from the atlas
// .w channel the update kernel now EMAs). Only sampled when the variance debug
// term is on, so it adds no cost to the normal composite.
static inline float sampleSurfCacheVarRT(
    texture2d<float, access::sample> atlas, uint prim, float2 bary,
    const device uint* triCard, const device float4* triUVa,
    const device float4* triUVc,
    const device void* primData,
    const device float4* cardRect, uint atlasW, uint atlasH)
{
    uint card = triCard[prim];
    float4 rect = cardRect[card];
    float2 uvA, uvB, uvC;
    if (primData != nullptr) {
        const device PrimUV* pd = (const device PrimUV*)primData;
        uvA = pd->uvA; uvB = pd->uvB; uvC = pd->uvC;
    } else {
        float4 a = triUVa[prim], c = triUVc[prim];
        uvA = a.xy; uvB = a.zw; uvC = c.xy;
    }
    float w0 = 1.0 - bary.x - bary.y;
    float2 uv = saturate(w0 * uvA + bary.x * uvB + bary.y * uvC);
    float2 inset = 0.5 / max(float2(1.0), rect.zw);
    uv = clamp(uv, inset, 1.0 - inset);
    float2 px = rect.xy + uv * rect.zw;
    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float4 t = atlas.sample(samp, px / float2(atlasW, atlasH));
    float mu = dot(t.rgb, float3(0.2126, 0.7152, 0.0722));
    return max(0.0, t.a - mu * mu);   // E[L²] − μ²
}

// The RNG (`pcgHash`/`rnd`), the sampling bases (`onb`/`cosineSample`/`coneSample`)
// and `dirToEquirectUV` are shared — see IlluminatoramaSecondary.h. Only the two
// G-buffer decoders below are specific to this deferred kernel.
static inline float3 octDecode(float2 e) {
    e = e * 2.0 - 1.0; float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) { float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); n.xy = (1.0 - abs(n.yx)) * s; }
    return normalize(n);
}
static inline float3 worldPosFromDepth(float2 ndcXY, float depth, float4x4 invVP) {
    float4 w = invVP * float4(ndcXY, depth, 1.0); return w.xyz / w.w;
}

// ── This kernel's two secondary-ray parameterisations ────────────────────────
//
// Both call the SAME `shadeSecondarySurface`; they differ only in which terms
// they ask for, which is the whole reason that function takes a params block.

/// FULL outgoing radiance at a hit — emission, textured albedo, sky IBL + ambient
/// with the interior split, local lights, direct sun. Used by the GLOSSY
/// REFLECTION path, because a specular reflection REPLACES what the eye would
/// otherwise see at that pixel, and by both surface-cache fallbacks, because a
/// resident card returns full radiance and a non-resident one must not be darker
/// than the card that was evicted.
static inline SecondaryShadeParams fullRadianceParams(constant RTInstUniforms& u) {
    SecondaryShadeParams p;
    p.sunDir = u.sunDir;
    // A HARD shadow ray at a secondary hit, not a cone: this kernel's own budget
    // is one reflection ray per pixel by default, and softening the sun at the
    // reflected hit on top of that would spend the sample budget on penumbra noise
    // instead of on the reflection. (`coneSample` with theta 0 returns the
    // direction unchanged, so this is exactly the single hard ray the path traced
    // before.) The PRIMARY surface still gets `u.shadowRays` soft rays above.
    p.sunSoftnessRad = 0.0;
    p.sunColor = u.sunColor;
    p.skyIntensity = u.skyIntensity;
    p.skyAmbient = u.skyAmbient;
    p.shadowRays = 1u;
    p.interiorMask = u.interiorMask;
    p.interiorIBLUp = u.interiorIBLUp; p.interiorIBLSide = u.interiorIBLSide;
    p.interiorAmbient = u.interiorAmbient;
    p.interiorIrrUp = u.interiorIrrUp.xyz;     p.interiorIrrW = u.interiorIrrUp.w;
    p.interiorIrrSide = u.interiorIrrSide.xyz;
    p.interiorIrrDown = u.interiorIrrDown.xyz;
    p.albedoAtlasEnabled = u.albedoAtlasEnabled;
    p.objUVCount = u.objUVCount;
    p.pointLightCount = u.pointLightCount;
    p.spotLightCount = u.spotLightCount;
    // C3 — a secondary hit found by this kernel is on a TRANSPORT path, so its own
    // sun shadow ray honours invisible occluders too. Without this a GI bounce that
    // lands on a floor under a lighting-only ceiling would come back full-sunlit and
    // put the light straight back that the transport mask on the bounce ray removed.
    p.occluderMask = u.transportRayMask;
    return p;
}

/// A hit found by a GI BOUNCE. Same body, two terms suppressed by zeroing their
/// params: the sky IBL and the ambient supplement. A GI bounce supplies INCOMING
/// radiance for the receiving surface's diffuse integral, and that surface already
/// carries its own IBL + ambient from the deferred pass — folding them in again at
/// the bounce would double-count the sky. Emission, texture, local lights and the
/// sun are all kept, because none of them is represented anywhere else.
static inline SecondaryShadeParams giBounceParams(constant RTInstUniforms& u) {
    SecondaryShadeParams p = fullRadianceParams(u);
    p.skyIntensity = 0.0;               // no sky IBL at a GI bounce …
    p.skyAmbient = float3(0.0);         // … and no ambient supplement (double count)
    return p;
}

kernel void illumi_rt_lighting_tlas(
    texture2d<float, access::read>        gDepth      [[texture(0)]],
    texture2d<half,  access::read>        gNormalRgh  [[texture(1)]],
    texture2d<half,  access::read>        gAlbedoMet  [[texture(2)]],
    texture2d<half,  access::read_write>  outHDR      [[texture(3)]],
    texture2d<float, access::sample>      skyEquirect [[texture(4)]],
    texture2d<float, access::sample>      surfAtlas   [[texture(5)]],
    instance_acceleration_structure       accel       [[buffer(0)]],
    const device RTInstanceData*          insts       [[buffer(1)]],
    const device float4*                  objNormal   [[buffer(2)]],
    constant RTInstUniforms&              u           [[buffer(3)]],
    const device uint*                    triCard     [[buffer(4)]],
    const device float4*                  triUVa      [[buffer(5)]],
    const device float4*                  triUVc      [[buffer(6)]],
    const device uint*                    soupTriBase [[buffer(7)]],
    const device float4*                  surfCardRect [[buffer(8)]],   // surface-cache per-card atlas rect
    const device SurfCard*                surfCards    [[buffer(9)]],   // per-card material (albedo/emission) for L_out reconstruction
    // Curve primitives (#60 item 7) — dummies bound for the base variant
    // (kRTCurvesEnabled false ⇒ never read).
    const device RTCurveSetData*          curveSets   [[buffer(10)]],
    const device packed_float3*           curvePts    [[buffer(11)]],
    const device float*                   curveRadii  [[buffer(12)]],
    const device uint*                    curveSegs   [[buffer(13)]],
    device uint*                          cardRequested [[buffer(14)]],  // Phase 5 / A residency feedback (gated)
    // Secondary-hit shading parity (shared with the AAA glass pass): the
    // per-triangle mesh UVs + albedo atlas so a hit samples the SAME texel the
    // G-buffer would, the local lights, and the cosine-convolved irradiance cube.
    // Dummies keep the bindings valid when a term is off (`u.albedoAtlasEnabled`
    // / the light counts gate every read).
    const device float2*                  objUV       [[buffer(15)]],
    const device float2*                  albedoUVScale [[buffer(16)]],
    const device RTPointLight*            pointLights [[buffer(17)]],
    const device RTSpotLight*             spotLights  [[buffer(18)]],
    texturecube<float, access::sample>    irrCube     [[texture(6)]],
    texture2d_array<float, access::sample> albedoAtlas [[texture(7)]],
    // C2 — the noisy diffuse (soft shadow + 1-bounce GI) goes OUT to its own
    // buffer for the temporal accumulator + SVGF/bilateral to clean, exactly as
    // the soup kernel has always done. It used to be composited inline here, and
    // that is the whole reason neither denoiser ran on this path: they consume
    // THIS texture, and nothing was writing it. With this app's TAA also off,
    // 4 GI rays reached the screen raw — the "dark noisy blotches".
    texture2d<half, access::write>        rtDiffuse   [[texture(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    float depth = gDepth.read(gid).r;
    if (depth >= 0.99999) return;

    float2 ndc = (float2(gid) + 0.5) / float2(u.width, u.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 P = worldPosFromDepth(ndc, depth, u.invViewProjection);
    half4 nrH = gNormalRgh.read(gid);
    half4 amH = gAlbedoMet.read(gid);
    float3 N = octDecode(float2(nrH.rg));
    // LOAD-BEARING FLOOR, not a cosmetic clamp: the glossy-reflection cone below is
    // `roughness² · k` gated on a threshold, and this 0.045 is what keeps that cone
    // (≥ 2.4e-3 rad) above the ~1e-3 rad at which a cone is sub-pixel and a single
    // stochastic sample of it is pure per-pixel NOISE rather than blur. The AAA glass
    // pass had no such floor and shipped visible edge jitter because of it. Lower this
    // and read `secondaryConeVisible` in IlluminatoramaSecondary.h first.
    float roughness = max(0.045, float(nrH.b));
    float3 albedo = float3(amH.rgb);
    float3 Pofs = P + N * max(u.rayTMin, 1e-3);

    constexpr sampler skySamp(filter::linear, address::repeat);
    // The curve_data tag is compile-time; the base variant keeps the original
    // triangle-only traversal contract (assume default), the curve variant
    // widens it to match a TLAS that holds curve instances (#60 item 7).
    intersector<triangle_data, instancing, curve_data> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    if (kRTCurvesEnabled) {
        isect.assume_geometry_type(geometry_type::triangle | geometry_type::curve);
        isect.assume_curve_basis(curve_basis::catmull_rom);
        isect.assume_curve_type(curve_type::round);
        isect.assume_curve_control_point_count(4);
    }
    uint seed = pcgHash(gid.x + gid.y * u.width + u.frameSeed * 9781u);
    float3 Ld = normalize(u.sunDir);

    // The shared secondary-ray currency. Filled ONCE; the GI and reflection hit
    // shading below are the same function with two parameterisations.
    SecondaryScene sec;
    sec.insts = insts;        sec.objNormal = objNormal;
    sec.objUV = objUV;        sec.uvScale = albedoUVScale;
    sec.pointLights = pointLights; sec.spotLights = spotLights;
    SecondaryShadeParams secFull   = fullRadianceParams(u);   // reflections + cache fallbacks
    SecondaryShadeParams secBounce = giBounceParams(u);       // GI bounces (no sky double-count)

    // ── Direct sun (soft shadows) ─────────────────────────────────
    //
    // C1: gated on `directSunEnabled`, NOT on `shadowRays > 0`. `shadowRays`
    // defaults to 4, so the old gate meant "always" — and this pass is additive
    // over a deferred frame that already shaded the sun. Two sun terms is what
    // made turning RT on look WORSE. Now exactly one of the two runs, and which
    // one is the renderer's stated `rtSunOwnership`.
    float3 direct = float3(0.0);
    float NdotL = saturate(dot(N, Ld));
    if (u.directSunEnabled != 0 && NdotL > 0.0 && u.shadowRays > 0) {
        isect.accept_any_intersection(true);
        uint hits = 0;
        for (uint s = 0; s < u.shadowRays; ++s) {
            ray r; r.origin = Pofs; r.direction = coneSample(Ld, u.sunSoftnessRad, rnd(seed), rnd(seed));
            r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = 1e4;
            // Any occluder counts — triangles always; curves when enabled.
            // `transportRayMask` = 0x01 opaque + curve | 0x04 invisible occluder
            // (C3). Glass (mask 0x02, #60 AAA glass) is excluded either way so a
            // clear pane doesn't cast a solid shadow.
            if (isect.intersect(r, accel, u.transportRayMask).type != intersection_type::none) hits++;
        }
        float vis = 1.0 - float(hits) / float(u.shadowRays);
        float3 V = normalize(u.cameraWorldPos - P);
        float3 H = normalize(Ld + V);
        float spec = pow(saturate(dot(N, H)), mix(8.0, 90.0, 1.0 - roughness)) * (1.0 - roughness) * u.specStrength;
        direct = u.sunColor * NdotL * vis * (albedo * (1.0 / M_PI_F) + spec);
    }

    // ── One-bounce indirect (GI) ──────────────────────────────────
    float3 indirect = float3(0.0);
    // B0 — variance accumulated across cache hits when the variance debug term
    // is on (zero cost otherwise; the sample sites guard on u.debugSurfCacheVar).
    float surfVarAcc = 0.0; uint surfVarN = 0u;
    if (u.giRays > 0 && u.giStrength > 0.0) {
        isect.accept_any_intersection(false);
        for (uint g = 0; g < u.giRays; ++g) {
            float3 dir = cosineSample(N, rnd(seed), rnd(seed));
            ray r; r.origin = Pofs; r.direction = dir;
            r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = u.maxGIDist;
            // C3: transport mask — opaque + curve + INVISIBLE OCCLUDER. Without
            // 0x04 a GI ray leaving an interior through the roofless dollhouse's
            // open top misses everything and returns full sky.
            auto res = isect.intersect(r, accel, u.transportRayMask);
            if (res.type == intersection_type::triangle) {
                if (u.surfCacheEnabled != 0) {
                    // Cached path: reconstruct the hit's full outgoing radiance
                    // (albedo·irradiance + emission, MULTI-bounce, accumulated over
                    // frames) from the cache — one atlas read + a card lookup, no
                    // shadow ray. A TLAS hit is per-instance-local, so resolve the
                    // global soup triangle first. Cheaper AND richer than re-shade.
                    uint gp = soupTriBase[res.instance_id] + res.primitive_id;
                    uint hitCard = (gp < u.surfTriCount) ? triCard[gp] : 0xFFFFFFFFu;
                    // Phase 5 / A — feedback marks the HIT card regardless of residency
                    // (a non-resident-but-visible card must be discoverable so streaming
                    // can promote it next frame), so this is OUTSIDE the residency gate.
                    if (hitCard != 0xFFFFFFFFu && u.surfFeedbackEnabled != 0) cardRequested[hitCard] = 1u;
                    // A0 — residency. A non-resident card (budget streaming) has a zero
                    // atlas rect; reading it would sample (0,0). Resident ⇒ cache read;
                    // non-resident ⇒ emission + ambient fallback (never black).
                    bool resident = (hitCard != 0xFFFFFFFFu) && (surfCardRect[hitCard].z > 0.0);
                    if (resident) {
                        indirect += sampleSurfCacheRT(surfAtlas, gp,
                            res.triangle_barycentric_coord, surfCards, triCard, triUVa, triUVc,
                            res.primitive_data, surfCardRect, u.surfAtlasW, u.surfAtlasH);
                        if (u.debugSurfCacheVar != 0) {
                            surfVarAcc += sampleSurfCacheVarRT(surfAtlas, gp,
                                res.triangle_barycentric_coord, triCard, triUVa, triUVc,
                                res.primitive_data, surfCardRect, u.surfAtlasW, u.surfAtlasH);
                            surfVarN++;
                        }
                    } else if (hitCard != 0xFFFFFFFFu) {
                        // Known card, not resident (budget streaming zeroed its
                        // rect). `secFull`, NOT `secBounce`: this stands in for a
                        // cache read, and a resident card returns full radiance —
                        // an evicted one must not turn a lit surface black. Shared
                        // fallback, so it fills as a re-shade would instead of with
                        // the flat `albedo * skyAmbient` slab it used to use.
                        float3 cN = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
                        if (dot(cN, dir) > 0.0) cN = -cN;
                        indirect += secondaryCardFallback(
                            surfCards[hitCard], cN,
                            secondaryLayerBits(res.instance_id, insts), secFull, irrCube);
                    }
                    continue;
                }
                // Re-shade through the ONE secondary-ray surface shader (see
                // IlluminatoramaSecondary.h) — the same body the glass pass uses.
                // `giBounceParams` suppresses the sky IBL + ambient (they would
                // double-count the receiving surface's own fill); emission, the
                // TEXTURED albedo, local lights and the sun all carry.
                SecondaryHit h;
                h.P = r.origin + dir * res.distance;
                h.N = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
                if (dot(h.N, dir) > 0.0) h.N = -h.N;      // face the incoming ray
                h.bary = res.triangle_barycentric_coord;
                h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
                indirect += shadeSecondarySurface(isect, accel, h, secBounce, sec,
                                                  irrCube, albedoAtlas, seed);
            } else if (kRTCurvesEnabled && res.type == intersection_type::curve) {
                // Curve hit (#60 item 7): no surface-cache card — re-shade with
                // the set's material (same sun + visibility shape as the
                // triangle re-shade above, plus the set's emission).
                uint setIdx = res.instance_id - u.curveInstanceBase;
                if (setIdx < u.curveSetCount) {
                    float3 hitP = r.origin + dir * res.distance;
                    float3 hitN = curveHitNormal(setIdx, res.primitive_id,
                                                 res.curve_parameter, hitP,
                                                 curveSets, curvePts, curveSegs);
                    RTCurveSetData cs = curveSets[setIdx];
                    float3 hitRad = cs.emissionPad.xyz;
                    float hN = saturate(dot(hitN, Ld));
                    if (hN > 0.0) {
                        isect.accept_any_intersection(true);
                        ray sr; sr.origin = hitP + hitN * 2e-3; sr.direction = Ld;
                        sr.min_distance = 2e-3; sr.max_distance = 1e4;
                        float sv = (isect.intersect(sr, accel, u.transportRayMask).type != intersection_type::none) ? 0.0 : 1.0;
                        hitRad += cs.albedoRoughness.xyz * (1.0 / M_PI_F) * u.sunColor * hN * sv;
                        isect.accept_any_intersection(false);
                    }
                    indirect += hitRad;
                }
            } else {
                indirect += skyEquirect.sample(skySamp, dirToEquirectUV(dir)).rgb;
            }
        }
        indirect = (indirect / float(u.giRays)) * albedo * u.giStrength;
    }

    // ── Glossy reflections (RT) ───────────────────────────────────
    float3 reflection = float3(0.0);
    if (u.reflEnabled != 0 && u.reflStrength > 0.0 && roughness <= u.reflRoughnessCutoff) {
        float3 V = normalize(u.cameraWorldPos - P);
        float3 R = reflect(-V, N);
        float NdotV = saturate(dot(N, V));
        // Metalness-aware Fresnel. Dielectrics keep F0 ≈ 0.04 (a faint glossy
        // sheen — unchanged from before), but a metal uses its albedo as F0, so a
        // chrome surface (metalness 1) reflects the room at ~full strength and
        // tinted by its own colour. This is what turns the metal spheres into
        // mirrors instead of a 4 % gloss over the deferred specular-IBL base.
        float metalness = float(amH.a);
        float3 F0 = mix(float3(0.04), albedo, metalness);
        // Plain Schlick at the macro NdotV — identical to the sibling RT path
        // (IlluminatoramaRT.metal). The roughness spread is already captured by
        // the cone sampling below, so a roughness-aware grazing floor here would
        // double-count roughness. The roughness-aware `fresnelSchlickRoughness`
        // variant is for the prefiltered-IBL deferred path (one env sample that
        // needs the grazing-energy fudge), not for a cone-traced reflection.
        float3 fres = F0 + (float3(1.0) - F0) * pow(1.0 - NdotV, 5.0);
        // Glossy cone widens with roughness² (GGX α scales as roughness²); mirror
        // surfaces stay tight. Harmonised with the soup-path kernel in
        // IlluminatoramaRT.metal (#60 task 5) — was the linear `roughness * 0.5`,
        // which over-blurred shiny surfaces and under-blurred rough ones. The
        // width coefficient now lives next to the glass path's in
        // IlluminatoramaSecondary.h (`kReflConeK`), so the two numbers can be
        // compared instead of hunted for.
        //
        // ── The sub-pixel-cone trap, and why this path never fell into it ──────
        // The old gate here was `coneTheta > 1e-4` — an order of magnitude below
        // the ~1e-3 rad at which a cone is still sub-pixel, i.e. the exact shape
        // of the bug that made polished GLAZING jitter. This path escaped it only
        // because `roughness` is CLAMPED to ≥ 0.045 up at the G-buffer read
        // (search `max(0.045`, ~170 lines above), which floors the cone at
        // 2.4e-3 rad — always above the threshold. The glass pass was uniquely
        // exposed because it alone reads raw per-instance roughness with no floor.
        // The shared `secondaryConeVisible` therefore changes NOTHING here today —
        // but if that clamp is ever lowered, this site is protected by the same
        // rule as glass instead of quietly going live.
        uint rrays = max(1u, u.reflRays);
        // The glass path's OTHER half — "a cone worth tracing is a cone worth averaging" —
        // deliberately does NOT apply here, and the reason is the roughness floor above.
        // `secondaryConeSamples` returns 1 only when the cone is sub-pixel; the floor of 0.045
        // puts every surface above that, so wiring it in here would raise EVERY reflection to at
        // least 2 rays unconditionally — a flat 2× on this path for scenes that never asked for
        // it, with no measurement showing the second sample buys anything at a 0.14° cone. The
        // count stays the host's to choose (`reflRays`, default 1). If a Visualizer scene wants
        // resolved glossy reflections it should raise its own budget, and if that budget is ever
        // made cone-aware the place to do it is the host, where the cost is visible.
        //
        // Kept as a comment rather than deleted because the asymmetry with glass is deliberate
        // and someone will otherwise "fix" it back.
        float3 acc = float3(0.0);
        isect.accept_any_intersection(false);
        for (uint i = 0; i < rrays; ++i) {
            float3 dir = secondaryConeVisible(roughness, kReflConeK)
                ? coneSample(R, secondaryConeRad(roughness, kReflConeK), rnd(seed), rnd(seed)) : R;
            if (dot(dir, N) <= 0.0) continue;
            ray r; r.origin = Pofs; r.direction = dir;
            r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = u.reflMaxDist;
            // C3 — DELIBERATELY the opaque mask, NOT `transportRayMask`. A
            // reflection is a CAMERA-VISIBLE ray: whatever it hits is drawn into
            // the picture. An invisible occluder (0x04) is a slab that exists for
            // light and is deliberately not drawn, so showing one in a glossy floor
            // would put a ceiling in the frame that is nowhere else in it. Flagged
            // for Danny; measured in `testRTReflectionRaysDoNotShowInvisibleCeilings`.
            auto res = isect.intersect(r, accel, 0x01u);
            if (kRTCurvesEnabled && res.type == intersection_type::curve) {
                // Curve reflection hit (#60 item 7) — same shading shape as the
                // triangle re-shade below (ambient + sun + the set's emission).
                uint setIdx = res.instance_id - u.curveInstanceBase;
                if (setIdx < u.curveSetCount) {
                    float3 hitP = r.origin + dir * res.distance;
                    float3 hitN = curveHitNormal(setIdx, res.primitive_id,
                                                 res.curve_parameter, hitP,
                                                 curveSets, curvePts, curveSegs);
                    RTCurveSetData cs = curveSets[setIdx];
                    float3 cA = cs.albedoRoughness.xyz;
                    float3 hitRad = cs.emissionPad.xyz + cA * u.skyAmbient;
                    float hN = saturate(dot(hitN, Ld));
                    if (hN > 0.0) {
                        isect.accept_any_intersection(true);
                        ray sr; sr.origin = hitP + hitN * 2e-3; sr.direction = Ld;
                        sr.min_distance = 2e-3; sr.max_distance = 1e4;
                        float sv = (isect.intersect(sr, accel, u.transportRayMask).type != intersection_type::none) ? 0.0 : 1.0;
                        isect.accept_any_intersection(false);
                        hitRad += cA * (1.0 / M_PI_F) * u.sunColor * hN * sv;
                    }
                    acc += hitRad;
                }
                continue;
            }
            if (res.type != intersection_type::triangle) continue;
            if (u.surfCacheEnabled != 0) {
                // Cached path: reflect the surface's full cached radiance
                // (multi-bounce), one atlas read instead of a re-shade.
                uint gp = soupTriBase[res.instance_id] + res.primitive_id;
                uint hitCard = (gp < u.surfTriCount) ? triCard[gp] : 0xFFFFFFFFu;
                // Phase 5 / A — feedback marks the hit card regardless of residency
                // (see the GI path); OUTSIDE the residency gate.
                if (hitCard != 0xFFFFFFFFu && u.surfFeedbackEnabled != 0) cardRequested[hitCard] = 1u;
                // A0 — residency (same as the GI path): resident ⇒ cache read;
                // non-resident ⇒ emission + ambient fallback (never black).
                bool resident = (hitCard != 0xFFFFFFFFu) && (surfCardRect[hitCard].z > 0.0);
                if (resident) {
                    acc += sampleSurfCacheRT(surfAtlas, gp,
                        res.triangle_barycentric_coord, surfCards, triCard, triUVa, triUVc,
                            res.primitive_data, surfCardRect, u.surfAtlasW, u.surfAtlasH);
                    if (u.debugSurfCacheVar != 0) {
                        surfVarAcc += sampleSurfCacheVarRT(surfAtlas, gp,
                            res.triangle_barycentric_coord, triCard, triUVa, triUVc,
                            res.primitive_data, surfCardRect, u.surfAtlasW, u.surfAtlasH);
                        surfVarN++;
                    }
                } else if (hitCard != 0xFFFFFFFFu) {
                    // Known card, not resident — same shared fallback as the GI path.
                    float3 cN = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
                    if (dot(cN, dir) > 0.0) cN = -cN;
                    acc += secondaryCardFallback(
                        surfCards[hitCard], cN,
                        secondaryLayerBits(res.instance_id, insts), secFull, irrCube);
                }
                continue;
            }
            // Re-shade through the ONE secondary-ray surface shader (see
            // IlluminatoramaSecondary.h). This is where this path used to shade a
            // reflected surface with the instance's MEAN albedo under a flat
            // exterior-strength `albedo * skyAmbient` — no texture, no local
            // lights, no emission, no interior split. Fixed once, for both paths.
            SecondaryHit h;
            h.P = r.origin + dir * res.distance;
            h.N = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
            if (dot(h.N, dir) > 0.0) h.N = -h.N;          // face the incoming ray
            h.bary = res.triangle_barycentric_coord;
            h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
            acc += shadeSecondarySurface(isect, accel, h, secFull, sec,
                                         irrCube, albedoAtlas, seed);
        }
        reflection = (acc / float(rrays)) * fres * u.reflStrength;
    }

    half4 prev = outHDR.read(gid);
    if (u.debugSurfCacheGI != 0) {
        // Isolation view: show ONLY the surface-cache-derived radiance. With the
        // cache on, `indirect` and `reflection` ARE the atlas reads (the non-cache
        // re-shade branches `continue` past), so this is the cache contribution in
        // isolation — replacing the lit composite makes the stale-pose ghost on a
        // moved object visible (it's sub-grain in the normal additive composite).
        outHDR.write(half4(half3(indirect + reflection), prev.a), gid);
        rtDiffuse.write(half4(0.0h), gid);   // isolation view owns the pixel; add nothing
        return;
    }
    if (u.debugSurfCacheVar != 0) {
        // Isolation view: per-texel cache variance (E[L²] − μ²) averaged over the
        // GI + reflection cache hits this pixel made. Replaces the composite so the
        // cache's convergence state is visible — a freshly-reset / cold card lights
        // up, a long-static card is near-black. Same per-hit-of-secondary-rays
        // caveat as term 8: it shows the variance of whatever the GI/reflection rays
        // landed on, not the primary surface. This is what B1's filter will drive.
        float v = surfVarN > 0u ? surfVarAcc / float(surfVarN) : 0.0;
        outHDR.write(half4(half3(half(v)), prev.a), gid);
        rtDiffuse.write(half4(0.0h), gid);   // isolation view owns the pixel; add nothing
        return;
    }
    // C2 — split by FREQUENCY, matching `illumi_rt_lighting` (the soup kernel):
    //   • reflection is sharp and varies across a flat surface that shares
    //     depth+normal, so a depth+normal bilateral would smear it → composite
    //     straight in;
    //   • direct + indirect is the low-frequency Monte-Carlo grain → out to
    //     `rtDiffuse`, where `encodeRTGITemporalAccum` and then SVGF (or the
    //     fixed-radius bilateral) clean it before it reaches the composite.
    // Sky pixels early-out at the top, so `rtDiffuse` is left untouched there and
    // the denoise pass guards on the same depth test.
    outHDR.write(half4(prev.rgb + half3(reflection), prev.a), gid);
    rtDiffuse.write(half4(half3(direct + indirect), 1.0h), gid);
}
