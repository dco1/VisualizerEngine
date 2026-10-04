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
    // 1 ⇒ `diffSky` holds the deferred composite's diffuse SKY share and the traced GI,
    // whose misses are the sky, REPLACES it (was `_pad0` — same 4 bytes, stride unchanged).
    float3 cameraWorldPos;   uint giReplacesDiffuseSky;
    float3 sunDir;           float sunSoftnessRad;
    float3 sunColor;         float giStrength;
    float3 skyAmbient;       float specStrength;
    uint  width;   uint height;
    uint  shadowRays; uint giRays;
    uint  frameSeed;  float rayTMin;  float maxGIDist;
    // DH-0896 — 1 ⇒ `specIBL` holds the deferred composite's specular-IBL share and a
    // reflection HIT replaces it (was the `_pad1` slot — same 4 bytes, stride unchanged).
    uint  reflReplacesIBL;
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
    // ── DH-0653 surface-cache instruments (were _padIrr0/_padIrr1 — same bytes) ──
    // `surfStatsEnabled` 1 ⇒ every GI / reflection TRIANGLE hit adds itself to
    // `surfHitStats` (buffer 19): [0..3] GI cached / fallback / noCard / reshade,
    // [4..7] the same for reflections. Set only while the stats sidecar is.
    // `debugSurfCacheCharts` 1 ⇒ DebugTerm.surfaceCacheCharts: REPLACE the composite
    // with the PRIMARY surface's card, one hashed colour per card.
    // DH-0872 — scotopic (Purkinje) night desaturation for the GI sky-MISS sample
    // only (see the miss branch below). Mirrors `FrameUniforms.scotopicDesaturation`
    // / `IlluminatoramaRenderer.scotopicDesaturation`; 0 (day, the default) ⇒ the
    // branch below never runs ⇒ byte-identical. Repurposes `_padIrr2` — same 4
    // bytes, stride unchanged; still aligns the float4 cluster below.
    uint  surfStatsEnabled; uint debugSurfCacheCharts; float scotopicDesaturation;
    // ── Interior irradiance bands (mirror of FrameUniforms.interiorIrr*) ─────
    // A GI bounce or reflection landing on an interior ceiling must see the
    // FLOOR's bounce, not the outdoor cube's lawn — same fix, same values, as
    // the deferred pass. xyz = irradiance, `interiorIrrUp.w` = blend weight
    // (0 = the exact cube sample — the default, byte-identical).
    float4 interiorIrrUp;
    float4 interiorIrrSide;
    float4 interiorIrrDown;
    // Per-room band level (S3.5 Stage E) — mirror of FrameUniforms.interiorRoomGain*.
    // Carried here so a room seen THROUGH a pane is scaled like the room beside it.
    float4 interiorRoomGain[8];
    float4 interiorRoomGainMeta;
    // Daydream DH-0887 — the GI directions' progressive index: +1 per TLAS lighting dispatch,
    // CONTIGUOUS (unlike `frameSeed`, which the glass pass also advances). Ray `g` of a dispatch is
    // point `giProgressiveIndex·giRays + g` of the pixel's own Owen-scrambled Sobol sequence.
    uint  giProgressiveIndex;
    // Daydream DH-0715 — the traced-origin snap's depth-ULP reach (`illumiSnapToVisibleSurface`;
    // 0 ⇒ off). Was `_padProg0` — same 4 bytes, stride unchanged.
    float surfaceSnapULPs;
    // Daydream DH-0989 / DH-0642 — the PATH-TRACED GI lane (see `illumiPathTraceIncoming`).
    // `pathBounces` > 0 ⇒ each GI ray is a whole path of up to that many bounces that owns the
    // pixel's ENTIRE diffuse indirect (the deferred pass hands its sky + bands + ambient + portal
    // share over in `diffSky`, `kFrameFlagPathOwnsIndirect`). 0 ⇒ the one-bounce estimator, exactly
    // as before. `pathClamp` caps one path's luminance (0 ⇒ no clamp). Were `_padProg1/_padProg2`.
    uint pathBounces; float pathClamp;
    // Daydream DH-1011 — how a path vertex samples its LIGHTS (`kPathFlag*` below). 0 ⇒ the
    // DH-0989 estimator exactly (byte-identical): apertures an exact partition with the cosine
    // continuation (a continuation through a portal is dropped), lamps an unshadowed deterministic
    // sum, every area emitter traced with `areaShadowRays` white-noise rays.
    uint pathFlags; uint _padPath0; uint _padPath1; uint _padPath2;
};

/// DH-1011 — `RTInstUniforms.pathFlags` bits.
/// `kPathFlagLightSampling`: (1) the aperture estimator and the cosine continuation are COMBINED
/// by multiple importance sampling (power heuristic) instead of partitioning the hemisphere — a
/// continuation that leaves through a portal is kept, weighted, rather than dropped; (2) the area
/// EMITTERS (light strips, skylight lenses) at a vertex are reduced to ONE pick ∝ their unshadowed
/// contribution and ONE Sobol shadow ray, instead of `areaShadowRays` white-noise rays each.
/// `kPathFlagLocalShadows`: the lamps / cans / sconces / pendants that cast a shadow in the
/// deferred pass (`castsShadow`) get one at a path vertex too — joined into the same single pick,
/// so a vertex traces ONE light ray however many fixtures the room holds. Unshadowed lights
/// (bounce fills, string lights) stay unshadowed, as they are in the deferred pass.
/// `kPathFlagDebugPrimaryLocal`: an INSTRUMENT — the path returns only the local-light NEE at
/// the primary (whose lamps the deferred pass normally owns), so the traced lamp visibility can be
/// checked against the shadow maps on open floor. Never set by a host.
#define kPathFlagLightSampling     1u
#define kPathFlagLocalShadows      2u
#define kPathFlagDebugPrimaryLocal 128u
/// INSTRUMENT: a continuation that hits an emissive surface adds nothing — the share of a still's
/// noise the glowing shades are, measured by its absence. Never set by a host.
#define kPathFlagDebugNoEmissionHits 64u
/// INSTRUMENTS for the noise budget by light source (never set by a host): drop one source class
/// from the path at its vertices — 32 the local lights (lamps + area emitters), 16 the sun, 8 the
/// window apertures (both the aperture sample and a continuation that leaves through a portal).
#define kPathFlagDebugNoLocal     32u
#define kPathFlagDebugNoSun       16u
#define kPathFlagDebugNoApertures 8u

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

// `SurfCard`, `PrimUV`, `RTInstanceData` and `sampleSurfCacheIndirectRT` are the shared
// mirror structs / cache read — see IlluminatoramaSecondary.h.

// Phase 5 / B0 — variance readout for the cache-variance debug term. Same
// card/UV addressing as sampleSurfCacheIndirectRT, but returns the texel's tracked
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

// DH-0653 — hit/miss counter slots in `surfHitStats` (GI slots; reflection = +4).
// Every column is shaded by `shadeSecondarySurface`; they differ only in the indirect term (DH-0622).
#define SURF_STAT_CACHED   0u   // resident card → its atlas indirect term
#define SURF_STAT_FALLBACK 1u   // known card, not resident → the fill estimate stands in for the read
#define SURF_STAT_NOCARD   2u   // cache on, triangle has no card → shaded as uncached
#define SURF_STAT_RESHADE  3u   // cache off → shaded as uncached
static inline void surfStat(device atomic_uint* s, uint enabled, uint slot) {
    if (enabled != 0u) atomic_fetch_add_explicit(&s[slot], 1u, memory_order_relaxed);
}

// DH-0653 — chart-assignment overlay colour. One stable hashed hue per card index,
// so two neighbouring cards differ in hue AND brightness and a coplanar chart reads
// as one flat patch. A NON-resident card (budget streaming) keeps its hue but drops
// to near-black, so residency is visible on the same picture.
static inline float3 surfChartColour(uint card, bool resident) {
    uint h = pcgHash(card * 2654435761u + 0x9E3779B9u);
    float hue = float(h & 0xFFFFu) / 65535.0;
    float sat = 0.55 + 0.40 * float((h >> 16) & 0xFFu) / 255.0;
    float val = (resident ? 0.50 : 0.08) + (resident ? 0.45 : 0.04) * float((h >> 24) & 0xFFu) / 255.0;
    float3 k = saturate(abs(fract(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return val * mix(float3(1.0), k, sat);
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
    p.interiorRoomGains = &u.interiorRoomGain[0];
    p.interiorRoomGainEnabled = u.interiorRoomGainMeta.x;
    p.interiorIrrUp = u.interiorIrrUp.xyz;     p.interiorIrrW = u.interiorIrrUp.w;
    p.interiorIrrSide = u.interiorIrrSide.xyz;
    p.interiorIrrDown = u.interiorIrrDown.xyz;
    p.albedoAtlasEnabled = u.albedoAtlasEnabled;
    p.objUVCount = u.objUVCount;
    p.pointLightCount = u.pointLightCount;
    p.spotLightCount = u.spotLightCount;
    // DH-0718 — the window portals (area lights), carried in the room-gain meta's spare
    // lanes (y = count, z = shadow rays per portal) so the uniform layout is unchanged.
    // 0 ⇒ none, the pre-DH-0718 behaviour.
    p.areaLightCount = uint(max(0.0, u.interiorRoomGainMeta.y));
    p.areaShadowRays = uint(max(0.0, u.interiorRoomGainMeta.z));
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

// ── THE PATH-TRACED GI LANE (DH-0989; the DH-0642 reference is the same code at a deep budget) ──
//
// Returns an estimate of (1/π)·∫ L_in cosθ dω at the primary point — the same quantity the
// one-bounce loop's cosine-sampled mean estimates, so the caller's `× albedo` is unchanged. Unlike
// that loop it is the WHOLE diffuse indirect: no interior bands, no ambient supplement, no
// "sunlit-room stand-in" in the portals — every one of those was a stand-in for light this walks.
//
// Per vertex v (T = product of the albedos met so far; T = 1 at the primary):
//   • next-event estimation, v ≥ 1 only (the deferred pass shades the primary's sun and lamps):
//     the sun through its cone (one shadow ray) and the local lights (the shared unshadowed fill);
//   • next-event estimation through ONE window aperture, every vertex including the primary:
//     the frame's area lights are, in this host, the house's window/skylight PORTALS — rectangles
//     over real openings. One is picked ∝ its unshadowed form factor × level (DH-0951 item 2: one
//     ray per vertex instead of one per portal), a point is drawn on it, and a single ray is traced
//     THROUGH it: blocked short of the portal ⇒ 0; past it ⇒ the sky, or the shaded outdoor hit,
//     along that exact direction (not the portal's fitted constant colour) × the pane;
//   • a cosine-sampled continuation. A continuation that leaves THROUGH a portal (crosses its
//     rectangle on the emitting side before any hit) is DROPPED: those directions belong to the
//     aperture estimator above. The two are a partition of the hemisphere, so nothing is counted
//     twice and nothing is lost — no MIS weights needed. Any other miss is the sky.
// Russian roulette from the third vertex; every random number is a padded, Owen-scrambled Sobol
// dimension pair (DH-0951 item 1), so a still's frames stratify every bounce, not just the first.
//
// DH-1011 (`pathFlags`, see `kPathFlag*`) — what changes when the host asks for light sampling:
//   • the aperture estimator and the continuation are no longer a partition. BOTH strategies cover
//     the portal's solid angle and each is weighted by the power heuristic against the other's
//     density there: the aperture sample by p_L² / (p_L² + p_B²), a continuation that crosses a
//     portal (first, before any hit) by p_B² / (p_B² + p_L²), where p_L = P(pick that portal at
//     this vertex) · d² / (A cosθ_L) and p_B = cosθ / π. The weights sum to one for every
//     direction, so the integrand — and therefore the converged mean — is the partition's; only
//     the variance moves: the area sample is a poor density near grazing (the wall the window is
//     cut into, a floor right under the sill) and for sky radiance that varies across the
//     opening, and the continuation covers exactly that. The crossing continuation reads the same
//     radiance the aperture sample does (the sky or the shaded outdoor hit along the ray, × the
//     pane) and ends there, as the aperture sample does.
//   • because the continuation now covers every portal direction, the aperture PICK no longer has
//     to: it is restricted to the portals of the vertex's own room (`pathPortalWeightAt`; the
//     primary's room comes from one short bracketing ray). A portal left out has p_L = 0 there and
//     its light arrives through the continuation at weight 1 — unbiased by construction. The
//     partition could not do this: a portal outside the pick was light it simply lost.
//   • the vertex's local emitters — area emitters always, and with `kPathFlagLocalShadows` every
//     point / spot light too — are ONE estimator: one light picked ∝ its own unshadowed
//     contribution at this vertex (power × distance falloff × cone × N·L for a lamp, form factor ×
//     range window for a rect — the exact quantity the deterministic sum adds), one Sobol shadow ray
//     to it. Picking ∝ the unshadowed term means an unoccluded vertex's estimate EQUALS the sum (up
//     to the lights' chroma spread), so the only variance left is visibility's — and the ray count
//     is one however many fixtures the room holds.
constant uint kPathDimsPerVertex = 4u;
// DH-1011's dimensions live under their own salt so a flag-off path draws EXACTLY the numbers it
// always did (byte-identical): .x of the first pair picks the local light, the second pair is the
// point on a picked rect.
constant uint kPathLightDimsPerVertex = 2u;
// A lamp shadow ray stops this short of the light — the FIXTURE CLEARANCE. Every cone this host
// emits sits INSIDE its own fixture: a bulb in a shade over a lamp base, a sconce's source in its
// cylinder, a can's apex 10 cm under the ceiling. The cones are authored wider than the
// housings' geometric openings (a sconce throws a 110° fan out of a short tube), and the deferred
// pass never sees the housing (most of these cones get no shadow slice; the rest store back faces
// behind a near plane). A ray that stopped 3 cm short found the housing: measured on the Debug
// Tester's many-lights bay, the traced visibility of the sconces' own wall fans was ~0 and of a
// table lamp's floor pool 0.40. 25 cm clears every shipped housing (a shade's radius, a sconce's
// half-height, a can's well) and still leaves the furniture a lamp stands on or hangs over — a
// nightstand top half a metre under its bulb, a dining table under a pendant — as the occluders.
constant float kPathLampShadowGap = 0.25;
// The one-pick's per-vertex weight cache (thread memory), in lights.
constant uint kPathLightCache = 64u;
constant uint kPathPortalCache = 32u;   // per-vertex aperture-weight cache (thread memory)

static inline float pathLuma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

/// Distance along (o, d) to area light `al`'s rectangle when the ray leaves its EMITTING side
/// through it (the side whose receivers it lights), or −1.
static inline float pathPortalCrossing(RTAreaLight al, float3 o, float3 d) {
    if (al.isAperture <= 0.5) return -1.0;               // an emitter, not an opening
    float3 nL = cross(al.ex, al.ey);
    float  nLlen = length(nL);
    if (nLlen < 1e-8) return -1.0;
    nL /= nLlen;
    float side = dot(nL, o - al.center);
    if (al.twoSided <= 0.5 && side <= 0.0) return -1.0;   // behind a one-sided portal
    float denom = dot(d, nL);
    if (abs(denom) < 1e-6) return -1.0;
    float t = -side / denom;
    if (t <= 1e-4) return -1.0;
    float3 q = o + d * t - al.center;
    float u = dot(q, al.ex) / max(dot(al.ex, al.ex), 1e-12);
    float v = dot(q, al.ey) / max(dot(al.ey, al.ey), 1e-12);
    return (abs(u) <= 1.0 && abs(v) <= 1.0) ? t : -1.0;
}

/// The vertex's selection weight for portal `al`: unshadowed clamped-cosine form factor × level.
/// Zero exactly where no continuation from this vertex could cross the portal (behind it, or
/// wholly below this vertex's horizon), which is what keeps the partition above exact.
static inline float pathPortalWeight(RTAreaLight al, float3 P, float3 N) {
    if (al.isAperture <= 0.5) return 0.0;                // an emitter, not an opening
    float3 nL = cross(al.ex, al.ey);
    if (length(nL) < 1e-8) return 0.0;
    bool twoSided = al.twoSided > 0.5;
    if (!twoSided && dot(normalize(nL), P - al.center) <= 0.0) return 0.0;
    float3 p0 = al.center - al.ex - al.ey - P;
    float3 p1 = al.center - al.ex + al.ey - P;
    float3 p2 = al.center + al.ex + al.ey - P;
    float3 p3 = al.center + al.ex - al.ey - P;
    float ff = ltcPolygonForm(N, p0, p1, p2, p3, twoSided);
    // A floor on the level so a portal the host emits at all is always selectable.
    return ff * max(pathLuma(al.color), 1e-4);
}

/// DH-1011 — the aperture PICK weight at a path vertex. Under MIS a portal may be left out of the
/// pick without bias (a continuation through it then carries weight 1), so the pick is restricted
/// to the portals of the vertex's own ROOM (`layerMask` — the same per-room bit the deferred pass
/// confines a portal by): the windows of the bays on the other side of a wall have a real
/// unshadowed form factor, and every sample spent on them is a ray that hits the wall. Without
/// light sampling the partition needs every portal covered, so the old weight stands.
static inline float pathPortalWeightAt(RTAreaLight al, float3 P, float3 N, uint layerBits, bool restrictToRoom) {
    if (restrictToRoom && (al.layerMask & layerBits) == 0u) return 0.0;
    return pathPortalWeight(al, P, N);
}

// ── DH-1011 — the vertex's local emitters as ONE pick ───────────────────────────────────────
//
// Each helper returns the light's UNSHADOWED contribution at (P, N) — radiance-per-albedo, the
// exact term `secondaryLocalLightFill` / `secondaryAreaLightFill` add for it — and is the light's
// selection weight (by luminance) too. Same layer mask, range window, cone and N·L tests as those
// sums, so a light the sum skips has weight 0 here and a light the sum counts can be picked.

static inline float3 pathPointContribution(RTPointLight pl, float3 P, float3 N, uint layerBits) {
    if (pl.giVisible == 0u || (pl.layerMask & layerBits) == 0u) return float3(0.0);
    float3 toL = pl.position - P;
    float dist = length(toL);
    if (dist > pl.radius) return float3(0.0);
    float nl = saturate(dot(N, toL / max(dist, 1e-4)));
    if (nl <= 0.0) return float3(0.0);
    float atten = 1.0 / max(dist * dist + pl.softRadius * pl.softRadius, 1e-4);
    float window = saturate(1.0 - pow(dist / pl.radius, 4.0));
    return pl.color * (atten * window * window * nl * (1.0 / M_PI_F));
}

static inline float3 pathSpotContribution(RTSpotLight sl, float3 P, float3 N, uint layerBits) {
    if (sl.giVisible == 0u || (sl.layerMask & layerBits) == 0u) return float3(0.0);
    float3 toL = sl.position - P;
    float dist = length(toL);
    if (dist > sl.radius) return float3(0.0);
    float3 L = toL / max(dist, 1e-4);
    float coneAtten = smoothstep(sl.outerCone, sl.innerCone, dot(normalize(sl.direction), -L));
    if (coneAtten <= 0.0) return float3(0.0);
    float nl = saturate(dot(N, L));
    if (nl <= 0.0) return float3(0.0);
    float atten = 1.0 / max(dist * dist + sl.softRadius * sl.softRadius, 1e-4);
    float window = saturate(1.0 - pow(dist / sl.radius, 4.0));
    return sl.color * (atten * window * window * coneAtten * nl * (1.0 / M_PI_F));
}

/// An area EMITTER's (not an aperture's) unshadowed term — `secondaryAreaLightFill`'s per-light
/// body before its visibility rays, including its `kSecondaryAreaSkip` cut.
static inline float3 pathEmitterContribution(RTAreaLight al, float3 P, float3 N, uint layerBits) {
    if (al.isAperture > 0.5 || (al.layerMask & layerBits) == 0u) return float3(0.0);
    float3 nL = cross(al.ex, al.ey);
    float  nLlen = length(nL);
    if (nLlen < 1e-8) return float3(0.0);
    nL /= nLlen;
    bool twoSided = al.twoSided > 0.5;
    if (!twoSided && dot(nL, P - al.center) <= 0.0) return float3(0.0);
    float dist = length(al.center - P);
    if (dist > al.radius) return float3(0.0);
    float window = saturate(1.0 - pow(dist / al.radius, 4.0));
    window *= window;
    float3 p0 = al.center - al.ex - al.ey - P;
    float3 p1 = al.center - al.ex + al.ey - P;
    float3 p2 = al.center + al.ex + al.ey - P;
    float3 p3 = al.center + al.ex - al.ey - P;
    float3 L = al.color * (ltcPolygonForm(N, p0, p1, p2, p3, twoSided) * window);
    return pathLuma(L) <= kSecondaryAreaSkip ? float3(0.0) : L;
}

/// Light index space for the one pick: [0, nPoint) points, then spots, then area emitters.
/// `lamps` false ⇒ points and spots are left out (the caller adds their deterministic sum).
static inline float3 pathLocalContribution(uint k, bool lamps, float3 P, float3 N, uint layerBits,
                                           SecondaryShadeParams p, SecondaryScene sec) {
    uint nP = lamps ? p.pointLightCount : 0u;
    uint nS = lamps ? p.spotLightCount : 0u;
    if (k < nP) return pathPointContribution(sec.pointLights[k], P, N, layerBits);
    k -= nP;
    if (k < nS) return pathSpotContribution(sec.spotLights[k], P, N, layerBits);
    k -= nS;
    return pathEmitterContribution(sec.areaLights[k], P, N, layerBits);
}

/// ONE light from the vertex's local emitters, picked ∝ its unshadowed contribution, with ONE
/// visibility ray when the light casts a shadow: an unbiased estimate of Σ_i c_i·V_i. The weights
/// are recomputed on the second pass rather than cached — an ALU-only pass over the room's lights
/// is noise next to one traced ray, and it keeps the cost flat in light count for thread memory.
template <typename Isect>
static inline float3 pathSampleLocalLights(thread Isect& isect, instance_acceleration_structure accel,
                                           float3 P, float3 N, float3 Pofs, uint layerBits,
                                           bool lamps, bool lampVisibility, uint areaCount,
                                           SecondaryShadeParams p, SecondaryScene sec,
                                           constant RTInstUniforms& u, uint sampleIndex, uint lightSalt,
                                           uint2 gid)
{
    uint nP = lamps ? p.pointLightCount : 0u;
    uint nS = lamps ? p.spotLightCount : 0u;
    uint n = nP + nS + areaCount;
    // Weights evaluated ONCE and reused by the pick (the first `kPathLightCache` lights; a room past
    // that re-evaluates the rest on the pick pass) — measured, the second full pass cost as much
    // as the shadow ray itself.
    float wc[kPathLightCache];
    float W = 0.0;
    for (uint k = 0u; k < n; ++k) {
        float wk = pathLuma(pathLocalContribution(k, lamps, P, N, layerBits, p, sec));
        if (k < kPathLightCache) wc[k] = wk;
        W += wk;
    }
    if (W <= 0.0) return float3(0.0);
    float2 sel = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, lightSalt + 0u));
    float pick = sel.x * W, acc = 0.0;
    uint j = n; float wj = 0.0;
    for (uint k = 0u; k < n; ++k) {
        float wk = (k < kPathLightCache) ? wc[k]
                                          : pathLuma(pathLocalContribution(k, lamps, P, N, layerBits, p, sec));
        acc += wk;
        if (wk > 0.0 && (pick < acc || k + 1u == n)) { j = k; wj = wk; break; }
    }
    if (j >= n || wj <= 0.0) return float3(0.0);
    float3 c = pathLocalContribution(j, lamps, P, N, layerBits, p, sec);
    float pj = wj / W;

    // Visibility — the same rule the deferred pass / the old sums apply per light class.
    bool castsShadow; float3 target; float gap;
    if (j < nP) {
        RTPointLight pl = sec.pointLights[j];
        castsShadow = lampVisibility && pl.castsShadow != 0u; target = pl.position; gap = kPathLampShadowGap;
    } else if (j < nP + nS) {
        RTSpotLight sl = sec.spotLights[j - nP];
        castsShadow = lampVisibility && sl.castsShadow != 0; target = sl.position; gap = kPathLampShadowGap;
    } else {
        RTAreaLight al = sec.areaLights[j - nP - nS];
        castsShadow = al.castsShadow != 0 && p.areaShadowRays > 0u;
        float2 q = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, lightSalt + 1u));
        target = al.center + al.ex * (q.x * 2.0 - 1.0) + al.ey * (q.y * 2.0 - 1.0);
        gap = 0.02;                                        // never the pane / lens itself
    }
    if (castsShadow) {
        // Off the surface on the light's side — the area fill's convention (a back-facing rect
        // is skipped by its weight, so this is N for every light that can be picked).
        float3 offN = (dot(N, target - P) >= 0.0) ? N : -N;
        float3 o = P + offN * 2e-3;                        // `secondaryAreaLightFill`'s origin
        float3 d = target - o;
        float len = length(d);
        if (len > gap + 1e-3) {
            isect.accept_any_intersection(true);
            ray sr; sr.origin = o; sr.direction = d / len;
            sr.min_distance = 2e-3; sr.max_distance = len - gap;
            bool blocked = isect.intersect(sr, accel, p.occluderMask).type != intersection_type::none;
            isect.accept_any_intersection(false);
            if (blocked) return float3(0.0);
        }
    }
    return c / pj;
}

/// The radiance the aperture estimator reads along a ray that has crossed a portal — the sky, or
/// the shaded outdoor hit — for a ray ALREADY traced (the continuation's own `res`). Must stay the
/// body of `pathSkyOrOutdoor` below, or the two MIS strategies would estimate different integrals.
template <typename Isect, typename Res>
static inline float3 pathOutdoorFromResult(thread Isect& isect, instance_acceleration_structure accel,
                                           thread Res& res, float3 o, float3 d,
                                           constant RTInstUniforms& u, SecondaryShadeParams pOut,
                                           SecondaryScene sec,
                                           texture2d<float, access::sample> skyEquirect,
                                           texturecube<float, access::sample> irrCube,
                                           texture2d_array<float, access::sample> albedoAtlas,
                                           thread uint& seed)
{
    if (res.type == intersection_type::triangle) {
        SecondaryHit h;
        h.P = o + d * res.distance;
        h.N = hitWorldNormal(res.instance_id, res.primitive_id, sec.insts, sec.objNormal);
        if (dot(h.N, d) > 0.0) h.N = -h.N;
        h.bary = res.triangle_barycentric_coord;
        h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
        return shadeSecondarySurface(isect, accel, h, pOut, sec, irrCube, albedoAtlas, seed);
    }
    if (res.type != intersection_type::none) return float3(0.0);
    constexpr sampler skySamp(filter::linear, address::repeat);
    float3 sky = skyEquirect.sample(skySamp, dirToEquirectUV(d)).rgb;
    if (u.scotopicDesaturation > 0.0) sky = mix(sky, float3(pathLuma(sky)), u.scotopicDesaturation);
    return sky * u.skyIntensity;
}

template <typename Isect>
static inline float3 pathSkyOrOutdoor(thread Isect& isect, instance_acceleration_structure accel,
                                      float3 o, float3 d, float tMin,
                                      constant RTInstUniforms& u, SecondaryShadeParams pOut,
                                      SecondaryScene sec,
                                      texture2d<float, access::sample> skyEquirect,
                                      texturecube<float, access::sample> irrCube,
                                      texture2d_array<float, access::sample> albedoAtlas,
                                      thread uint& seed, thread float& hitDist)
{
    constexpr sampler skySamp(filter::linear, address::repeat);
    isect.accept_any_intersection(false);
    ray r; r.origin = o; r.direction = d; r.min_distance = tMin; r.max_distance = 1e4;
    auto res = isect.intersect(r, accel, u.transportRayMask);
    if (res.type == intersection_type::triangle) {
        hitDist = res.distance;
        SecondaryHit h;
        h.P = o + d * res.distance;
        h.N = hitWorldNormal(res.instance_id, res.primitive_id, sec.insts, sec.objNormal);
        if (dot(h.N, d) > 0.0) h.N = -h.N;
        h.bary = res.triangle_barycentric_coord;
        h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
        return shadeSecondarySurface(isect, accel, h, pOut, sec, irrCube, albedoAtlas, seed);
    }
    if (res.type != intersection_type::none) { hitDist = res.distance; return float3(0.0); }
    hitDist = INFINITY;
    float3 sky = skyEquirect.sample(skySamp, dirToEquirectUV(d)).rgb;
    if (u.scotopicDesaturation > 0.0) sky = mix(sky, float3(pathLuma(sky)), u.scotopicDesaturation);
    return sky * u.skyIntensity;
}

template <typename Isect>
static inline float3 illumiPathTraceIncoming(thread Isect& isect, instance_acceleration_structure accel,
                                             float3 P0, float3 N0, uint primaryLayerBits,
                                             uint sampleIndex, uint2 gid,
                                             constant RTInstUniforms& u, SecondaryScene sec,
                                             SecondaryShadeParams pFull,
                                             texture2d<float, access::sample> skyEquirect,
                                             texturecube<float, access::sample> irrCube,
                                             texture2d_array<float, access::sample> albedoAtlas,
                                             thread uint& seed)
{
    uint portalCount = uint(max(0.0, u.interiorRoomGainMeta.y));
    // Outdoor radiance seen THROUGH an aperture: full radiance of whatever is out there, with
    // no area lights of its own (they are this lane's apertures, not outdoor sources).
    SecondaryShadeParams pOut = pFull;
    pOut.areaLightCount = 0u;
    // A path vertex's own direct terms: sun + local lights, never the fitted fills.
    SecondaryShadeParams pVert = pFull;
    pVert.skyIntensity = 0.0; pVert.skyAmbient = float3(0.0); pVert.areaLightCount = 0u;
    float  tMin = max(u.rayTMin, 1e-3);
    float3 Ld = normalize(u.sunDir);

    float3 L = float3(0.0);
    float3 T = float3(1.0);
    float3 P = P0, N = N0;
    uint layerBits = primaryLayerBits;   // the vertex's light layers (the local-light / room mask)
    uint maxB = min(u.pathBounces, 32u);
    // DH-1011 — light sampling (see the lane's header comment and `kPathFlag*`).
    bool lightSampling = (u.pathFlags & kPathFlagLightSampling) != 0u;
    bool lampShadows   = (u.pathFlags & kPathFlagLocalShadows) != 0u;
    if ((u.pathFlags & kPathFlagDebugPrimaryLocal) != 0u) {
        // INSTRUMENT: the primary's own local-light NEE and nothing else — the traced lamp
        // visibility, readable against the deferred pass's shadow-mapped lamps (DH-1011).
        return pathSampleLocalLights(isect, accel, P0, N0, P0 + N0 * tMin, layerBits, true, lampShadows,
                                     0u, pVert, sec, u, sampleIndex, 0x4C495445u, gid);
    }
    for (uint v = 0u; v <= maxB; ++v) {
        uint dimSalt = 0x50415448u + v * kPathDimsPerVertex;   // 'PATH'
        uint lightSalt = 0x4C495445u + v * kPathLightDimsPerVertex;   // 'LITE' (DH-1011)
        float3 Pofs = P + N * tMin;
        float3 Lv = float3(0.0);

        // ── NEE: sun + local lights (secondary vertices) ──
        if (v > 0u) {
            float nl = saturate(dot(N, Ld));
            if (nl > 0.0 && any(u.sunColor > 0.0) && (u.pathFlags & kPathFlagDebugNoSun) == 0u) {
                float2 sq = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, dimSalt + 3u));
                isect.accept_any_intersection(true);
                ray sr; sr.origin = Pofs;
                sr.direction = coneSample(Ld, u.sunSoftnessRad, sq.x, sq.y);
                sr.min_distance = tMin; sr.max_distance = 1e4;
                bool blocked = isect.intersect(sr, accel, u.transportRayMask).type != intersection_type::none;
                isect.accept_any_intersection(false);
                if (!blocked) Lv += u.sunColor * nl * (1.0 / M_PI_F);
            }
            if ((u.pathFlags & kPathFlagDebugNoLocal) != 0u) {
                // instrument: no local lights at path vertices
            } else if (!lightSampling) {
                // Unshadowed (the shared fill), so the layer mask is what keeps a lamp in its room.
                Lv += secondaryLocalLightFill(P, N, layerBits, pVert, sec);
                // Area EMITTERS (light strips, skylight lenses) — the apertures are sampled below.
                if (portalCount > 0u) {
                    SecondaryShadeParams pA = pVert; pA.areaLightCount = portalCount;
                    Lv += secondaryAreaLightFill(isect, accel, P, N, layerBits, pA, sec, seed, true);
                }
            } else {
                // DH-1011 — the lamps stay the deterministic unshadowed sum unless they are to be
                // shadowed; the emitters (and then the lamps) are ONE pick + ONE ray.
                if (!lampShadows) Lv += secondaryLocalLightFill(P, N, layerBits, pVert, sec);
                Lv += pathSampleLocalLights(isect, accel, P, N, Pofs, layerBits, lampShadows, true,
                                            portalCount, pVert, sec, u, sampleIndex, lightSalt, gid);
            }
        }

        // ── NEE: one window aperture ──
        // Weights are evaluated ONCE per vertex (an LTC form factor each) and reused by the pick;
        // a scene with more apertures than the cache re-evaluates for the pick.
        float wCache[kPathPortalCache];
        float wSum = 0.0;
        if ((u.pathFlags & kPathFlagDebugNoApertures) == 0u)
        for (uint i = 0u; i < portalCount; ++i) {
            float w = pathPortalWeightAt(sec.areaLights[i], P, N, layerBits, lightSampling);
            if (i < kPathPortalCache) wCache[i] = w;
            wSum += w;
        }
        if (wSum > 0.0) {
            float2 sel = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, dimSalt + 2u));
            float pick = sel.x * wSum, acc = 0.0, pj = 0.0;
            uint j = portalCount;
            for (uint i = 0u; i < portalCount; ++i) {
                float w = (i < kPathPortalCache) ? wCache[i] : pathPortalWeightAt(sec.areaLights[i], P, N, layerBits, lightSampling);
                acc += w;
                if (w > 0.0 && (pick < acc || i + 1u == portalCount)) { j = i; pj = w / wSum; break; }
            }
            if (j < portalCount && pj > 0.0) {
                RTAreaLight al = sec.areaLights[j];
                float2 pq = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, dimSalt + 1u));
                float3 X = al.center + al.ex * (pq.x * 2.0 - 1.0) + al.ey * (pq.y * 2.0 - 1.0);
                float3 toX = X - Pofs;
                float dist = length(toX);
                float3 d = toX / max(dist, 1e-6);
                float cosR = dot(N, d);
                float3 nL = cross(al.ex, al.ey);
                float area4 = 4.0 * length(nL);                     // |2ex × 2ey|
                float cosL = abs(dot(normalize(nL), d));
                if (cosR > 0.0 && cosL > 1e-4 && dist > 1e-4) {
                    float hitDist;
                    float3 Lo = pathSkyOrOutdoor(isect, accel, Pofs, d, tMin, u, pOut, sec,
                                                 skyEquirect, irrCube, albedoAtlas, seed, hitDist);
                    if (hitDist > dist - 2e-3) {
                        // The aperture's OPACITY (1 − pane transmittance): 0 ⇒ an open hole.
                        float tPane = 1.0 - saturate(al.apertureOpacity);
                        float pdfW = dist * dist / (area4 * cosL);          // area → solid angle
                        // DH-1011 — power-heuristic weight against the continuation's density.
                        float misW = 1.0;
                        if (lightSampling) {
                            float pL = pj * pdfW, pB = cosR * (1.0 / M_PI_F);   // both per steradian
                            misW = (pL * pL) / max(pL * pL + pB * pB, 1e-30);
                        }
                        Lv += Lo * tPane * cosR * misW / (M_PI_F * pj * pdfW);
                    }
                }
            }
        }
        L += T * Lv;
        if (v == maxB) break;

        // ── Continuation (cosine) ──
        float2 bq = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, dimSalt + 0u));
        float3 dir = illumiCosineHemisphereAxisSafe(N, bq);
        isect.accept_any_intersection(false);
        ray r; r.origin = Pofs; r.direction = dir; r.min_distance = tMin; r.max_distance = 1e4;
        auto res = isect.intersect(r, accel, u.transportRayMask);
        float tHit = (res.type != intersection_type::none) ? res.distance : INFINITY;
        if (lightSampling) {
            // DH-1011 — a continuation that leaves through a portal is the OTHER strategy for the
            // aperture integral: keep it, MIS-weighted against the aperture sample's density for
            // this direction, then end the path there exactly as the aperture sample does.
            uint kc = portalCount; float tcMin = tHit;
            for (uint i = 0u; i < portalCount; ++i) {
                float tc = pathPortalCrossing(sec.areaLights[i], Pofs, dir);
                if (tc > 0.0 && tc < tcMin) { tcMin = tc; kc = i; }
            }
            if (kc < portalCount && (u.pathFlags & kPathFlagDebugNoApertures) != 0u) break;
            if (kc < portalCount) {
                RTAreaLight al = sec.areaLights[kc];
                float wk = (kc < kPathPortalCache) ? wCache[kc] : pathPortalWeightAt(al, P, N, layerBits, lightSampling);
                float pL = 0.0;
                float3 nL = cross(al.ex, al.ey);
                float area4 = 4.0 * length(nL);
                float cosL = abs(dot(normalize(nL), dir));
                if (wSum > 0.0 && wk > 0.0 && cosL > 1e-4 && area4 > 0.0)
                    pL = (wk / wSum) * (tcMin * tcMin) / (area4 * cosL);
                float pB = max(dot(N, dir), 0.0) * (1.0 / M_PI_F);
                float misW = (pB * pB) / max(pB * pB + pL * pL, 1e-30);
                float tPane = 1.0 - saturate(al.apertureOpacity);
                float3 Lo = pathOutdoorFromResult(isect, accel, res, Pofs, dir, u, pOut, sec,
                                                  skyEquirect, irrCube, albedoAtlas, seed);
                L += T * Lo * tPane * misW;
                break;
            }
        } else {
            bool throughPortal = false;
            for (uint i = 0u; i < portalCount && !throughPortal; ++i) {
                float tc = pathPortalCrossing(sec.areaLights[i], Pofs, dir);
                throughPortal = tc > 0.0 && tc < tHit;
            }
            if (throughPortal) break;             // the aperture estimator owns that direction
        }
        if (res.type == intersection_type::none) {
            constexpr sampler skySamp(filter::linear, address::repeat);
            float3 sky = skyEquirect.sample(skySamp, dirToEquirectUV(dir)).rgb;
            if (u.scotopicDesaturation > 0.0) sky = mix(sky, float3(pathLuma(sky)), u.scotopicDesaturation);
            L += T * sky * u.skyIntensity;
            break;
        }
        if (res.type != intersection_type::triangle) break;   // curves: opaque, unlit (rare)
        SecondaryHit h;
        h.P = Pofs + dir * res.distance;
        h.N = hitWorldNormal(res.instance_id, res.primitive_id, sec.insts, sec.objNormal);
        if (dot(h.N, dir) > 0.0) h.N = -h.N;
        h.bary = res.triangle_barycentric_coord;
        h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
        if ((u.pathFlags & kPathFlagDebugNoEmissionHits) == 0u) L += T * secondaryEmission(h.instanceID, sec.insts);
        float3 A = saturate(secondaryAlbedo(h, pFull, sec, albedoAtlas));
        T *= A;
        P = h.P; N = h.N;
        layerBits = secondaryLayerBits(h.instanceID, sec.insts);
        if (v >= 1u) {
            float q = clamp(max(T.r, max(T.g, T.b)), 0.05, 0.95);
            float2 rr = illumiSobolOwen2DShuffled(sampleIndex, illumiPixelSeed(gid, dimSalt + 2u));
            if (rr.y >= q) break;
            T /= q;
        }
    }
    if (u.pathClamp > 0.0) {
        float l = pathLuma(L);
        if (l > u.pathClamp) L *= u.pathClamp / l;
    }
    return L;
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
    // DH-0718 — the frame's area lights (window portals); read only when
    // `interiorRoomGainMeta.y` > 0, a dummy otherwise.
    const device RTAreaLight*             areaLights  [[buffer(20)]],
    texturecube<float, access::sample>    irrCube     [[texture(6)]],
    texture2d_array<float, access::sample> albedoAtlas [[texture(7)]],
    // C2 — the noisy diffuse (soft shadow + 1-bounce GI) goes OUT to its own
    // buffer for the temporal accumulator + SVGF/bilateral to clean, exactly as
    // the soup kernel has always done. It used to be composited inline here, and
    // that is the whole reason neither denoiser ran on this path: they consume
    // THIS texture, and nothing was writing it. With this app's TAA also off,
    // 4 GI rays reached the screen raw — the "dark noisy blotches".
    texture2d<half, access::write>        rtDiffuse   [[texture(8)]],
    device atomic_uint*                   surfHitStats [[buffer(19)]],  // DH-0653 hit/miss counters (gated)
    // DH-0896 — the deferred pass's specular-IBL share of `outHDR` (read only when
    // `u.reflReplacesIBL`; a 1×1 dummy otherwise).
    texture2d<half, access::read>         specIBL     [[texture(9)]],
    texture2d<half, access::read>         diffSky     [[texture(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    float depth = gDepth.read(gid).r;
    if (depth >= 0.99999) return;

    float2 ndc = (float2(gid) + 0.5) / float2(u.width, u.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    // Rays leave from the surface the camera SEES (a substrate's biased depth buries the
    // rebuilt point — see `illumiSnapToVisibleSurface`).
    float3 P = illumiSnapToVisibleSurface(accel, worldPosFromDepth(ndc, depth, u.invViewProjection),
                                          u.cameraWorldPos, ndc, depth, u.invViewProjection,
                                          u.surfaceSnapULPs);
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
    sec.areaLights = areaLights;
    SecondaryShadeParams secFull   = fullRadianceParams(u);   // reflections + cache fallbacks
    SecondaryShadeParams secBounce = giBounceParams(u);       // GI bounces (no sky double-count)

    // ── DH-0653: chart-assignment overlay (DebugTerm.surfaceCacheCharts) ─────
    // Terms 8/9 show what the SECONDARY rays read; this shows the PRIMARY surface's
    // own card, so the chart decomposition (`VIZ_SURFCACHE_CHARTS`) and any residual
    // seam are seen directly instead of eyeballed off the composite. The G-buffer
    // stores no triangle id, so the pixel's surface is re-found with one short ray
    // that brackets the depth-reconstructed point (5 cm either side, opaque mask —
    // the drawn surface, never an invisible occluder or glass). Near-black grey = no
    // card here (cache off / skipped for this scene, a curve, a miss).
    if (u.debugSurfCacheCharts != 0) {
        float3 chart = float3(0.02);
        if (u.surfCacheEnabled != 0) {
            float3 toP = P - u.cameraWorldPos;
            float dist = length(toP);
            float3 dir = toP / max(dist, 1e-6);
            float bracket = min(0.05, 0.5 * dist);
            isect.accept_any_intersection(false);
            ray pr; pr.origin = P - dir * bracket; pr.direction = dir;
            pr.min_distance = 0.0; pr.max_distance = 2.0 * bracket + 1e-3;
            auto pres = isect.intersect(pr, accel, 0x01u);
            if (pres.type == intersection_type::triangle) {
                uint gp = soupTriBase[pres.instance_id] + pres.primitive_id;
                uint card = (gp < u.surfTriCount) ? triCard[gp] : 0xFFFFFFFFu;
                if (card != 0xFFFFFFFFu) chart = surfChartColour(card, surfCardRect[card].z > 0.0);
            }
        }
        outHDR.write(half4(half3(chart), outHDR.read(gid).a), gid);
        rtDiffuse.write(half4(0.0h), gid);   // isolation view owns the pixel; add nothing
        return;
    }

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
    // DH-0622 — debug term 8: the cache's share of GI + reflection alone, same weights.
    float3 cacheGI = float3(0.0), cacheRefl = float3(0.0), cacheTerms = float3(0.0);
    if (u.giRays > 0 && u.giStrength > 0.0) {
        isect.accept_any_intersection(false);
        uint giSobolSeed = illumiPixelSeed(gid, 0x4749u);
        if (u.pathBounces > 0u) {
            // DH-0989 — the path-traced lane: each GI ray is a whole path owning the pixel's
            // entire diffuse indirect (see `illumiPathTraceIncoming`). The one-bounce body
            // below is skipped; its cache / curve / stats instruments belong to that estimator.
            // DH-1011 — the primary surface's ROOM (light-layer bits), which the G-buffer does not
            // carry: one short ray bracketing the depth-reconstructed point along the view ray
            // finds the drawn triangle (the chart overlay's probe). Only the aperture pick reads
            // it, and only under MIS, so a miss (all bits) costs nothing but efficiency.
            uint primaryLayer = 0xFFFFFFFFu;
            if ((u.pathFlags & kPathFlagLightSampling) != 0u) {
                float3 toP = P - u.cameraWorldPos;
                float dist = length(toP);
                float3 vdir = toP / max(dist, 1e-6);
                float bracket = min(0.05, 0.5 * dist);
                ray pr; pr.origin = P - vdir * bracket; pr.direction = vdir;
                pr.min_distance = 0.0; pr.max_distance = 2.0 * bracket + 1e-3;
                auto pres = isect.intersect(pr, accel, 0x01u);
                if (pres.type == intersection_type::triangle) primaryLayer = secondaryLayerBits(pres.instance_id, insts);
            }
            for (uint g = 0; g < u.giRays; ++g) {
                indirect += illumiPathTraceIncoming(isect, accel, Pofs - N * max(u.rayTMin, 1e-3), N, primaryLayer,
                                                    u.giProgressiveIndex * u.giRays + g, gid, u, sec,
                                                    secFull, skyEquirect, irrCube, albedoAtlas, seed);
            }
        } else
        for (uint g = 0; g < u.giRays; ++g) {
            float2 gq = illumiSobolOwen2DShuffled(u.giProgressiveIndex * u.giRays + g, giSobolSeed);
            float3 dir = illumiCosineHemisphereAxisSafe(N, gq);
            ray r; r.origin = Pofs; r.direction = dir;
            r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = u.maxGIDist;
            // C3: transport mask — opaque + curve + INVISIBLE OCCLUDER. Without
            // 0x04 a GI ray leaving an interior through the roofless dollhouse's
            // open top misses everything and returns full sky.
            auto res = isect.intersect(r, accel, u.transportRayMask);
            if (res.type == intersection_type::triangle) {
                // Every opaque hit — cached or not — is shaded by the ONE secondary-ray
                // surface shader (see IlluminatoramaSecondary.h), the same body the glass
                // pass uses. `giBounceParams` suppresses the sky IBL + ambient (they would
                // double-count the receiving surface's own fill); emission, the TEXTURED
                // albedo, local lights and the sun all carry.
                SecondaryHit h;
                h.P = r.origin + dir * res.distance;
                h.N = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
                if (dot(h.N, dir) > 0.0) h.N = -h.N;      // face the incoming ray
                h.bary = res.triangle_barycentric_coord;
                h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
                // DH-0622 — a surface-cache card supplies ONLY the hit's multi-bounce
                // indirect term; it never stands in for the shading. A TLAS hit is
                // per-instance-local, so resolve the global soup triangle first.
                uint cardState = SURF_STAT_RESHADE;
                float3 cacheIrr = float3(0.0);
                if (u.surfCacheEnabled != 0) {
                    uint gp = soupTriBase[res.instance_id] + res.primitive_id;
                    uint hitCard = (gp < u.surfTriCount) ? triCard[gp] : 0xFFFFFFFFu;
                    // Phase 5 / A — feedback marks the HIT card regardless of residency (a
                    // non-resident-but-visible card must be discoverable so streaming can
                    // promote it next frame).
                    if (hitCard != 0xFFFFFFFFu && u.surfFeedbackEnabled != 0) cardRequested[hitCard] = 1u;
                    if (hitCard == 0xFFFFFFFFu) {
                        cardState = SURF_STAT_NOCARD;       // no card: shaded as uncached
                    } else if (surfCardRect[hitCard].z > 0.0) {
                        cardState = SURF_STAT_CACHED;
                        cacheIrr = sampleSurfCacheIndirectRT(surfAtlas, gp, h.bary, triCard, triUVa, triUVc,
                                                             res.primitive_data, surfCardRect,
                                                             u.surfAtlasW, u.surfAtlasH);
                        if (u.debugSurfCacheVar != 0) {
                            surfVarAcc += sampleSurfCacheVarRT(surfAtlas, gp, h.bary, triCard, triUVa, triUVc,
                                                               res.primitive_data, surfCardRect,
                                                               u.surfAtlasW, u.surfAtlasH);
                            surfVarN++;
                        }
                    } else {
                        // A0 — known card, not resident (budget streaming zeroed its rect, so
                        // the atlas holds nothing for it). The full fill estimate of the same
                        // arriving light (`secFull`, NOT `secBounce`) stands in for the read,
                        // so an evicted card is never darker than its resident neighbour.
                        cardState = SURF_STAT_FALLBACK;
                        cacheIrr = secondaryIndirectFill(h.N, secondaryLayerBits(res.instance_id, insts),
                                                         secFull, irrCube);
                    }
                }
                surfStat(surfHitStats, u.surfStatsEnabled, cardState);
                float3 cacheTerm;
                indirect += shadeSecondarySurface(isect, accel, h, secBounce, sec, irrCube, albedoAtlas, seed,
                                                  cardState == SURF_STAT_CACHED || cardState == SURF_STAT_FALLBACK,
                                                  cacheIrr, cacheTerm);
                cacheGI += cacheTerm;
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
                // DH-0872 — a raw, un-convolved single sample of the shared environment
                // texture, unlike the CONVOLVED bake the deferred diffuse/specular IBL terms
                // read (and unlike the interior-irradiance-band substitution above, which is
                // day-only and only ever runs for geometry HITS, never for a sky miss). The
                // below-horizon band deliberately carries a saturated "moonlit lawn" green
                // tint at night (`groundFillRadiance`/N3) that the tonemap's own
                // `scotopicDesaturation` is meant to neutralise — but that global pass runs
                // on the FINAL composite, gated below its display-luma knee, and a
                // ceiling-corner normal near a window sends most of its cosine-weighted rays
                // straight out through the glass into exactly this band: undiluted, then
                // amplified by `giStrength`, landing bright enough post-composite to dodge the
                // knee entirely. Apply the same scotopic coefficient directly to the raw
                // source sample instead — this is exterior night-ambient light gathered
                // through one bounce, exactly the domain the scotopic model targets.
                float3 sky = skyEquirect.sample(skySamp, dirToEquirectUV(dir)).rgb;
                if (u.scotopicDesaturation > 0.0) {
                    float skyLum = dot(sky, float3(0.2126, 0.7152, 0.0722));
                    sky = mix(sky, float3(skyLum), u.scotopicDesaturation);
                }
                // Replacing the deferred sky: a miss carries the sky at the SAME intensity
                // the deferred diffuse term used, so the host's sky-fill dial stays live.
                indirect += (u.giReplacesDiffuseSky != 0u) ? sky * u.skyIntensity : sky;
            }
        }
        indirect = (indirect / float(u.giRays)) * albedo * u.giStrength;
        cacheTerms += (cacheGI / float(u.giRays)) * albedo * u.giStrength;
        if (u.pathBounces > 0u && (u.pathFlags & kPathFlagDebugPrimaryLocal) != 0u) {
            // DH-1011 INSTRUMENT — isolation view: the primary's traced local-light estimate ALONE
            // (albedo × it), replacing the composite, so traced lamp visibility reads as a ratio
            // of two frames (shadowed ÷ unshadowed) with nothing else in the pixel.
            outHDR.write(half4(half3(indirect), outHDR.read(gid).a), gid);
            rtDiffuse.write(half4(0.0h), gid);
            return;
        }
    }

    // ── Glossy reflections (RT) ───────────────────────────────────
    float3 reflection = float3(0.0);
    // DH-0896 — how much of the deferred SKY reflection this pixel's rays proved wrong.
    // `outHDR` already carries the specular IBL — the sky cube along R — at the full
    // Fresnel of the surface. A reflection ray that MISSES leaves it (the sky is what that
    // direction sees). A ray that HITS the scene saw a wall, not the sky, so the sky that
    // direction contributed must come OUT as the room goes IN. Adding the hit on top of it
    // (as this pass did) left a mirror showing the sunset horizon with the room ghosted
    // faintly through it — the whole of the Primary Bathroom report.
    float reflHitFraction = 0.0;
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
        uint hits = 0u;
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
                    hits += 1u;
                }
                continue;
            }
            if (res.type != intersection_type::triangle) continue;
            hits += 1u;
            // Re-shade through the ONE secondary-ray surface shader (see
            // IlluminatoramaSecondary.h), cached or not. This is where this path used to
            // shade a reflected surface with the instance's MEAN albedo under a flat
            // exterior-strength `albedo * skyAmbient` — no texture, no local lights, no
            // emission, no interior split. Fixed once, for both paths.
            SecondaryHit h;
            h.P = r.origin + dir * res.distance;
            h.N = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
            if (dot(h.N, dir) > 0.0) h.N = -h.N;          // face the incoming ray
            h.bary = res.triangle_barycentric_coord;
            h.instanceID = res.instance_id; h.primitiveID = res.primitive_id;
            // DH-0622 — same card resolution as the GI path: the cache supplies only the
            // indirect term the shader would otherwise estimate with its fill.
            uint cardState = SURF_STAT_RESHADE;
            float3 cacheIrr = float3(0.0);
            if (u.surfCacheEnabled != 0) {
                uint gp = soupTriBase[res.instance_id] + res.primitive_id;
                uint hitCard = (gp < u.surfTriCount) ? triCard[gp] : 0xFFFFFFFFu;
                if (hitCard != 0xFFFFFFFFu && u.surfFeedbackEnabled != 0) cardRequested[hitCard] = 1u;
                if (hitCard == 0xFFFFFFFFu) {
                    cardState = SURF_STAT_NOCARD;
                } else if (surfCardRect[hitCard].z > 0.0) {
                    cardState = SURF_STAT_CACHED;
                    cacheIrr = sampleSurfCacheIndirectRT(surfAtlas, gp, h.bary, triCard, triUVa, triUVc,
                                                         res.primitive_data, surfCardRect,
                                                         u.surfAtlasW, u.surfAtlasH);
                    if (u.debugSurfCacheVar != 0) {
                        surfVarAcc += sampleSurfCacheVarRT(surfAtlas, gp, h.bary, triCard, triUVa, triUVc,
                                                           res.primitive_data, surfCardRect,
                                                           u.surfAtlasW, u.surfAtlasH);
                        surfVarN++;
                    }
                } else {
                    cardState = SURF_STAT_FALLBACK;
                    cacheIrr = secondaryIndirectFill(h.N, secondaryLayerBits(res.instance_id, insts),
                                                     secFull, irrCube);
                }
            }
            surfStat(surfHitStats, u.surfStatsEnabled, 4u + cardState);
            float3 cacheTerm;
            acc += shadeSecondarySurface(isect, accel, h, secFull, sec, irrCube, albedoAtlas, seed,
                                         cardState == SURF_STAT_CACHED || cardState == SURF_STAT_FALLBACK,
                                         cacheIrr, cacheTerm);
            cacheRefl += cacheTerm;
        }
        reflection = (acc / float(rrays)) * fres * u.reflStrength;
        reflHitFraction = saturate(float(hits) / float(rrays) * u.reflStrength);
        cacheTerms += (cacheRefl / float(rrays)) * fres * u.reflStrength;
    }

    half4 prev = outHDR.read(gid);
    if (u.debugSurfCacheGI != 0) {
        // Isolation view: ONLY the surface cache's share — `albedo · cached indirect` at
        // every GI and reflection hit, carried through the same weights (DH-0622). What an
        // uncached hit also has is left out, so this is exactly what the cache adds;
        // replacing the lit composite makes the stale-pose ghost on a moved object
        // visible (it's sub-grain in the normal additive composite).
        outHDR.write(half4(half3(cacheTerms), prev.a), gid);
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
    float3 skySeenThrough = float3(0.0);
    if (u.reflReplacesIBL != 0u && reflHitFraction > 0.0) {
        skySeenThrough = float3(specIBL.read(gid).rgb) * reflHitFraction;
    }
    // The traced GI owns the diffuse sky wherever it ran: the deferred share comes out.
    float3 diffSkyReplaced = float3(0.0);
    if (u.giReplacesDiffuseSky != 0u && u.giRays > 0 && u.giStrength > 0.0) {
        diffSkyReplaced = float3(diffSky.read(gid).rgb);
    }
    outHDR.write(half4(max(prev.rgb - half3(skySeenThrough) - half3(diffSkyReplaced), half3(0.0h)) + half3(reflection), prev.a), gid);
    rtDiffuse.write(half4(half3(direct + indirect), 1.0h), gid);
}

// ── RAY-TRACED AMBIENT OCCLUSION (RTAO) — photo/export lane (DH-0528) ─────────
//
// Tier 2 of DH-0440's AO research, greenlit by Danny. The screen-space GTAO march
// (`illumi_ssao`) draws a foamy crust at wall/ceiling junctions because it estimates
// occlusion from the DEPTH buffer, which only knows the front-most surface — at a
// concave junction there is no geometry behind the front face to sample, so the
// horizon integral fills a noisy, half-res, per-pixel-rotated point cloud. RTAO
// answers the same question against the REAL geometry: short cosine-weighted
// hemisphere rays cast at the TLAS the RT sun-shadow / reflection passes already
// build. The occlusion radius is a TRUE world-space distance (the ray
// `max_distance`), which is what dissolves the DH-0441 fork — one physically
// meaningful metre grounds a sofa foot AND stays tight at a drywall junction,
// where the single screen-space band width could only do one.
//
// It is a DROP-IN raw-AO producer: it writes the same half-res AO texture
// `illumi_ssao` does, in the same `1.0 = unoccluded` contract, so the existing
// bilateral + temporal denoiser and the lighting kernel's AO read consume it
// unchanged. Cosine-importance sampling makes the estimator simply the fraction of
// occluded rays (the cosine weight is folded into the sample distribution), matching
// GTAO's final `ao = 1 - occlusion·intensity`. Per-frame jitter (`frameSeed`) is the
// same contract GTAO and the RT sun-shadow pass honour, so the photo lane's 32-frame
// temporal accumulator integrates the Monte-Carlo noise out.
//
// PHOTO/EXPORT LANE ONLY. The per-frame ray budget is Monte-Carlo and only the
// still's accumulator converges it; an interactive frame would show ray banding.
// The renderer gates the dispatch on `rtaoActive` (photo lane + a live TLAS); a
// frame with no TLAS yet falls back to GTAO.
struct RTAOUniforms {
    float4x4 invViewProjection;   // depth → world position
    float     radius;             // occlusion reach, WORLD metres (ray max_distance)
    float     intensity;          // 0..1 AO strength (matches ssaoIntensity semantics)
    uint      rayCount;           // cosine-hemisphere rays per pixel per frame
    uint      frameSeed;          // per-frame jitter walk (0 = frozen)
    float     rayTMin;            // self-intersection guard, world metres
    uint      transportRayMask;   // 0x01 opaque | 0x04 invisible occluder (glass excluded)
    uint      fullWidth;          // full-res G-buffer dims (AO is half-res)
    uint      fullHeight;
};

kernel void illumi_rtao_tlas(
    depth2d<float, access::read>      gDepth   [[texture(0)]],   // full-res
    texture2d<half,  access::read>    gNormal  [[texture(1)]],   // full-res oct-normal.xy
    texture2d<half,  access::write>   outAO    [[texture(2)]],   // half-res AO out
    instance_acceleration_structure   accel    [[buffer(0)]],
    constant RTAOUniforms&            u        [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint outW = outAO.get_width();
    uint outH = outAO.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    // Disabled ⇒ neutral 1.0, so the lighting kernel reads unconditionally.
    if (u.intensity <= 0.0) { outAO.write(half4(1.0h), gid); return; }

    // Half-res gid → full-res G-buffer texel (2×2 top-left; matches illumi_ssao).
    uint fullW = u.fullWidth, fullH = u.fullHeight;
    uint2 fullGid = min(gid * 2, uint2(fullW - 1, fullH - 1));

    float depth = gDepth.read(fullGid);
    if (depth >= 0.99999) { outAO.write(half4(1.0h), gid); return; }   // sky — no surface

    float2 ndc = (float2(fullGid) + 0.5) / float2(fullW, fullH) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 P = worldPosFromDepth(ndc, depth, u.invViewProjection);
    float3 N = octDecode(float2(gNormal.read(fullGid).rg));
    float3 Pofs = P + N * max(u.rayTMin, 1e-3);

    // Triangle-only traversal against the instance AS. NO curve assume block — foliage
    // (curve geometry) is not a near-field contact occluder, and omitting it keeps this
    // kernel free of the kRTCurvesEnabled function constant (one un-specialized pipeline).
    intersector<triangle_data, instancing, curve_data> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.accept_any_intersection(true);   // occlusion-only

    // STRATIFIED, not white: the i-th ray takes the i-th point of the R2 low-discrepancy
    // sequence, rotated per pixel (Cranley–Patterson) by a hash and walked per frame by the
    // seed. White noise at 8–12 rays left a crust the bilateral could not clean (the DH-0528
    // gate measured 1.15 in-band against GTAO's 0.47); a stratified estimator's error falls as
    // ~1/N rather than 1/√N, and the per-pixel rotation keeps neighbouring texels decorrelated
    // so the spatial filter still averages independent estimates.
    uint seed = pcgHash(gid.x + gid.y * outW + u.frameSeed * 9781u);
    float2 cp = float2(rnd(seed), rnd(seed));                 // per-pixel/per-frame rotation
    uint rays = max(1u, u.rayCount);
    float radius = max(1e-3, u.radius);
    uint hits = 0u;
    const float g = 1.32471795724474602596;                    // plastic constant (R2)
    const float a1 = 1.0 / g, a2 = 1.0 / (g * g);
    for (uint i = 0; i < rays; ++i) {
        float2 q = fract(cp + float2(a1, a2) * float(i + 1));
        ray r;
        r.origin = Pofs;
        r.direction = cosineSample(N, q.x, q.y);
        r.min_distance = max(u.rayTMin, 1e-3);
        r.max_distance = radius;   // world-space reach — beyond this is not an occluder
        if (isect.intersect(r, accel, u.transportRayMask).type != intersection_type::none) hits++;
    }
    // Cosine weighting is in the sampling, so the occlusion estimate is just the hit
    // fraction; visibility = 1 − occlusion. Same final form as illumi_ssao's `ao`.
    float occlusion = float(hits) / float(rays);
    float ao = 1.0 - occlusion * u.intensity;
    outAO.write(half4(half(clamp(ao, 0.0, 1.0))), gid);
}

// ── VOLUMETRIC LIGHT IN THE ROOM'S AIR (Daydream DH-0990, photo lane) ─────────────────────────
//
// Single scattering in a faint homogeneous haze, marched along each view ray against the REAL
// geometry — the traced sibling of `illumi_volumetric`, whose sun test is one hard-coded window
// box. At each sample of the march:
//   • the SUN: one shadow ray toward it (transport mask, so glass passes and the house's walls,
//     roof and invisible ceilings stop it) — the shafts are whatever the window openings cut, by
//     construction; Henyey–Greenstein phase about the sun;
//   • the SKY through ONE window aperture, chosen ∝ its level × solid angle from the sample, a
//     point on it, one ray through it (blocked short of it ⇒ 0; past it, the sky along that exact
//     direction × the pane; an outdoor hit beyond is left dark — a faint term on a faint term);
//   • the medium exists only in ROOFED air: a short ray straight up must meet the house (drawn or
//     lighting-only ceiling) within `roofReach`. Outdoors the far-field haze is aerial
//     perspective's job (#10c); without this an exterior view would fog over.
// Every random number is a progressive Owen-scrambled Sobol pair per march step, so a still's
// frames stratify the march. The background is attenuated by the roofed path's transmittance.
//
// SUNLIT DUST (DH-0999). The air above is molecular-faint: at a density that never greys a
// sunless room, a sunbeam crossing it barely reads. What makes a beam visible in a real room is
// DUST — large particles, strongly forward-scattering — and dust is only seen where direct sun
// lights it. So a second, sun-only scatterer rides on top, and its amount is tied to how much
// direct sun actually enters the view's air:
//   • `illumi_volumetric_rt_admittance` marches a coarse fixed grid of the SAME view rays first
//     (same roof test, same sun shadow ray) and counts roofed samples and the sunlit ones among
//     them; A = sunlit / roofed is the measured direct-sun admittance of the room's air in view.
//   • σ_dust = dustDensity · saturate(A / dustAdmittanceRef): full dust once A reaches the
//     reference fraction, proportionally less below it, and EXACTLY none when no sun gets in
//     (night, overcast-dark sun, a north room, the sun on the other façade) — no grey fog.
//   • the dust in-scatters the SUN ONLY (HG g = dustAnisotropy). Dust lit by the sky alone is a
//     low-contrast veil the faint air term already stands for; carrying it at dust density
//     would be exactly the grey fog the gate exists to prevent (a deliberate departure).
//   • it does NOT dim the view. Dust is a conservative scatterer (albedo ≈ 1), and in the room's
//     near-uniform DIFFUSE field radiance is invariant along a ray (the equilibrium result:
//     what the dust scatters out of the view ray it scatters back in from the same field), so
//     its extinction and its diffuse in-scatter cancel and are both left out. The one part of
//     the field that is NOT uniform is the collimated sunbeam — that is the shaft, and it is
//     the dust's only net term. (Measured before this: carrying the dust's extinction alone
//     darkened the sunny Primary Bedroom by −0.13 stops, −4.9 codes outside the beam.)
struct VolRTUniforms {
    float4x4 invViewProjection;
    float3 cameraWorldPos; float density;          // σ, per metre (scattering = extinction)
    float3 sunDir;         float anisotropy;       // sunDir toward the sun; HG g
    float3 sunColor;       float skyIntensity;     // irradiance units of the deferred sun; dome scale
    uint width; uint height; uint steps; uint progressiveIndex;
    float maxDist; float roofReach; uint areaLightCount; uint transportMask;
    float sunScatter; float skyScatter; float scotopicDesaturation; float rayTMin;
    float dustDensity; float dustAnisotropy; float dustAdmittanceRef; uint admittanceGrid;
};

/// The view ray through pixel `gid` and its march end (depth-terminated, capped at maxDist).
static inline float volRTViewRay(texture2d<float, access::read> gDepth, constant VolRTUniforms& u,
                                 uint2 gid, thread float3& rd) {
    float2 ndc = (float2(gid) + 0.5) / float2(u.width, u.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 fw = u.invViewProjection * float4(ndc, 1.0, 1.0);
    rd = normalize(fw.xyz / fw.w - u.cameraWorldPos);
    float depth = gDepth.read(gid).r;
    float tEnd = u.maxDist;
    if (depth < 0.99999) {
        float4 w = u.invViewProjection * float4(ndc, depth, 1.0);
        tEnd = min(u.maxDist, length(w.xyz / w.w - u.cameraWorldPos));
    }
    return tEnd;
}

static inline bool volRTSunOn(constant VolRTUniforms& u) {
    return u.sunScatter > 0.0 && any(u.sunColor > 0.0) && normalize(u.sunDir).y > -0.05;
}

/// DH-0999 — the direct-sun ADMITTANCE of the air in view (see the block comment above): a
/// fixed `admittanceGrid` × `admittanceGrid·h/w` lattice of the view's own rays, each marched at
/// the main pass's step count with mid-step samples (deterministic: the same View measures the
/// same A every frame), counting roofed samples → counts[0] and the sun-seeing ones → counts[1].
kernel void illumi_volumetric_rt_admittance(
    texture2d<float, access::read>        gDepth      [[texture(0)]],
    instance_acceleration_structure       accel       [[buffer(0)]],
    constant VolRTUniforms&               u           [[buffer(1)]],
    device atomic_uint*                   counts      [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint gx = max(u.admittanceGrid, 1u);
    uint gy = max(1u, uint(float(gx) * float(u.height) / float(max(u.width, 1u))));
    if (gid.x >= gx || gid.y >= gy) return;
    uint2 px = min(uint2((float2(gid) + 0.5) / float2(gx, gy) * float2(u.width, u.height)),
                   uint2(u.width - 1u, u.height - 1u));
    float3 rd;
    float tEnd = volRTViewRay(gDepth, u, px, rd);
    if (tEnd <= 0.01) return;
    intersector<triangle_data, instancing> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.accept_any_intersection(true);
    float3 toSun = normalize(u.sunDir);
    bool sunOn = volRTSunOn(u);
    uint N = clamp(u.steps, 1u, 64u);
    float dt = tEnd / float(N);
    uint roofed = 0u, sunlit = 0u;
    for (uint i = 0u; i < N; ++i) {
        float3 X = u.cameraWorldPos + rd * ((float(i) + 0.5) * dt);
        ray up; up.origin = X; up.direction = float3(0.0, 1.0, 0.0);
        up.min_distance = 0.0; up.max_distance = u.roofReach;
        if (isect.intersect(up, accel, u.transportMask).type == intersection_type::none) continue;
        roofed++;
        if (!sunOn) continue;
        ray sr; sr.origin = X; sr.direction = toSun; sr.min_distance = max(u.rayTMin, 1e-3); sr.max_distance = 1e4;
        if (isect.intersect(sr, accel, u.transportMask).type == intersection_type::none) sunlit++;
    }
    if (roofed > 0u) atomic_fetch_add_explicit(&counts[0], roofed, memory_order_relaxed);
    if (sunlit > 0u) atomic_fetch_add_explicit(&counts[1], sunlit, memory_order_relaxed);
}

static inline float volRTPhaseHG(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * M_PI_F * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

/// Aperture weight seen from a point in the AIR (no surface normal): level × the rectangle's
/// projected solid angle, softened at the near field. Zero behind a one-sided aperture.
static inline float volRTApertureWeight(RTAreaLight al, float3 X) {
    if (al.isAperture <= 0.5) return 0.0;
    float3 nL = cross(al.ex, al.ey);
    float area4 = 4.0 * length(nL);
    if (area4 < 1e-8) return 0.0;
    nL = normalize(nL);
    float3 toX = X - al.center;
    if (al.twoSided <= 0.5 && dot(nL, toX) <= 0.0) return 0.0;
    float d2 = dot(toX, toX);
    float cosL = abs(dot(nL, toX)) * rsqrt(max(d2, 1e-8));
    return max(pathLuma(al.color), 1e-4) * area4 * cosL / (d2 + area4);
}

kernel void illumi_volumetric_rt(
    texture2d<float, access::read>        gDepth      [[texture(0)]],
    texture2d<half,  access::read_write>  outHDR      [[texture(1)]],
    texture2d<float, access::sample>      skyEquirect [[texture(2)]],
    instance_acceleration_structure       accel       [[buffer(0)]],
    constant VolRTUniforms&               u           [[buffer(1)]],
    const device RTAreaLight*             areaLights  [[buffer(2)]],
    const device uint*                    admittance  [[buffer(3)]],   // counts from the pre-pass
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height || u.density <= 0.0) return;
    float3 ro = u.cameraWorldPos;
    float3 rd;
    float tEnd = volRTViewRay(gDepth, u, gid, rd);
    if (tEnd <= 0.01) return;

    // DH-0999 — sunlit dust, in proportion to the direct sun the pre-pass measured entering.
    float sigmaDust = 0.0;
    if (u.dustDensity > 0.0 && admittance[0] > 0u) {
        float A = float(admittance[1]) / float(admittance[0]);
        sigmaDust = u.dustDensity * saturate(A / max(u.dustAdmittanceRef, 1e-4));
    }
    float gDust = clamp(u.dustAnisotropy, -0.95, 0.95);
    float sigmaT = u.density;   // the dust's extinction cancels its diffuse in-scatter (above)

    intersector<triangle_data, instancing> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    constexpr sampler skySamp(filter::linear, address::repeat);
    float3 Ls = float3(0.0);
    float3 toSun = normalize(u.sunDir);
    float  cosSun = dot(rd, toSun);
    // σ·phase of everything the sun lights: the air, plus the dust (density folded in here, so
    // the in-scatter line below multiplies the air's σ only into the sky term).
    float  sunSigmaPhase = u.density * volRTPhaseHG(cosSun, u.anisotropy)
                         + sigmaDust * volRTPhaseHG(cosSun, gDust);
    bool   sunOn = volRTSunOn(u);
    uint   N = clamp(u.steps, 1u, 64u);
    float  dt = tEnd / float(N);
    float  roofedLen = 0.0;
    float  tMin = max(u.rayTMin, 1e-3);

    for (uint i = 0u; i < N; ++i) {
        uint salt = 0x564F4C00u + i * 2u;   // 'VOL'
        float2 q0 = illumiSobolOwen2DShuffled(u.progressiveIndex, illumiPixelSeed(gid, salt));
        float t = (float(i) + q0.x) * dt;
        float3 X = ro + rd * t;

        // Roofed air only.
        isect.accept_any_intersection(true);
        ray up; up.origin = X; up.direction = float3(0.0, 1.0, 0.0);
        up.min_distance = 0.0; up.max_distance = u.roofReach;
        if (isect.intersect(up, accel, u.transportMask).type == intersection_type::none) continue;
        roofedLen += dt;
        float Tcam = exp(-sigmaT * t);
        float3 Lx = float3(0.0);

        if (sunOn) {
            ray sr; sr.origin = X; sr.direction = toSun; sr.min_distance = tMin; sr.max_distance = 1e4;
            if (isect.intersect(sr, accel, u.transportMask).type == intersection_type::none) {
                Lx += u.sunColor * (sunSigmaPhase * u.sunScatter);
            }
        }

        if (u.skyScatter > 0.0 && u.areaLightCount > 0u) {
            float wSum = 0.0;
            for (uint k = 0u; k < u.areaLightCount; ++k) wSum += volRTApertureWeight(areaLights[k], X);
            if (wSum > 0.0) {
                float pick = q0.y * wSum, acc = 0.0, pj = 0.0;
                uint j = u.areaLightCount;
                for (uint k = 0u; k < u.areaLightCount; ++k) {
                    float w = volRTApertureWeight(areaLights[k], X);
                    acc += w;
                    if (w > 0.0 && (pick < acc || k + 1u == u.areaLightCount)) { j = k; pj = w / wSum; break; }
                }
                if (j < u.areaLightCount) {
                    RTAreaLight al = areaLights[j];
                    float2 pq = illumiSobolOwen2DShuffled(u.progressiveIndex, illumiPixelSeed(gid, salt + 1u));
                    float3 P = al.center + al.ex * (pq.x * 2.0 - 1.0) + al.ey * (pq.y * 2.0 - 1.0);
                    float3 toP = P - X;
                    float dist = length(toP);
                    float3 d = toP / max(dist, 1e-6);
                    float3 nL = cross(al.ex, al.ey);
                    float area4 = 4.0 * length(nL);
                    float cosL = abs(dot(normalize(nL), d));
                    if (cosL > 1e-4 && dist > 1e-3) {
                        isect.accept_any_intersection(false);
                        ray pr; pr.origin = X; pr.direction = d; pr.min_distance = tMin; pr.max_distance = 1e4;
                        auto h = isect.intersect(pr, accel, u.transportMask);
                        if (h.type == intersection_type::none) {
                            float3 sky = skyEquirect.sample(skySamp, dirToEquirectUV(d)).rgb * u.skyIntensity;
                            if (u.scotopicDesaturation > 0.0) sky = mix(sky, float3(pathLuma(sky)), u.scotopicDesaturation);
                            float pdfW = dist * dist / (area4 * cosL);
                            Lx += sky * (1.0 - saturate(al.apertureOpacity))
                                * (u.density * volRTPhaseHG(dot(rd, d), u.anisotropy)) * u.skyScatter / (pj * pdfW);
                        }
                    }
                }
            }
        }
        Ls += Lx * (Tcam * dt);   // σ is inside Lx (air on both terms, dust on the sun's)
    }
    if (roofedLen <= 0.0) return;
    half4 prev = outHDR.read(gid);
    float Tbg = exp(-sigmaT * roofedLen);
    outHDR.write(half4(half3(float3(prev.rgb) * Tbg + Ls), prev.a), gid);
}
