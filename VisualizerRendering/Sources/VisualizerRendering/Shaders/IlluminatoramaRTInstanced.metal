#include <metal_stdlib>
#include <metal_raytracing>
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

// DUPLICATED from IlluminatoramaRT.metal / IlluminatoramaSurfaceCache.metal —
// `static inline` is file-local, so the soup kernel's copy isn't visible here.
// Keep in lockstep. Maps a global triangle (soup prim + barycentric) to its
// card-UV and samples the radiance atlas tile (UV inset half a texel so
// bilinear can't bleed across tiles).
// Field-for-field mirror of `SurfCard` in IlluminatoramaSurfaceCache.metal (Metal
// has no cross-file linkage; keep in lockstep). Only albedo/emission/normal.w are
// read here, but the full layout is needed so `cards[card]` strides correctly.
struct SurfCard {
    float4 origin; float4 uAxis; float4 vAxis; float4 normal;
    float4 albedo; float4 emission;
    float4 originB; float4 uAxisB; float4 vAxisB; float4 normalB;
    float4 albedoB; float4 emissionB;
};

// #60 item 6 — per-triangle card-frame UVs baked into the BLAS `primitive_data`,
// read straight off the hit (`res.primitive_data`) instead of the per-instance
// global `triUVa`/`triUVc` side buffers. Field-for-field mirror of Swift
// `IlluminatoramaRenderer.IlluminatoramaPrimUV` (3×float2, 24 B). Mesh-invariant,
// so one table per BLAS serves every instance.
struct PrimUV { float2 uvA; float2 uvB; float2 uvC; };

// The atlas stores albedo-free IRRADIANCE (#60 task 2). Reconstruct the hit's
// outgoing radiance L_out = albedo·irr + emission from its card, so a card seam
// isn't amplified by the albedo multiply. Returns the same outgoing radiance the
// callers always consumed — no caller math changes.
//
// `primData` is the hit's `res.primitive_data`: non-null ⇒ the BLAS carries baked
// per-mesh card UVs (#60 item 6), so read uvA/uvB/uvC from it and SKIP the
// `triUVa[prim]`/`triUVc[prim]` dependent loads (32 B/hit). Null (charts path, or
// the fold disabled) ⇒ fall back to the side buffers. `prim` (== the global soup
// triangle `soupTriBase[iid]+primitive_id`) still indexes the per-instance
// `triCard` for the card slot, which is genuinely per-instance and not foldable.
static inline float3 sampleSurfCacheRT(
    texture2d<float, access::sample> atlas, uint prim, float2 bary,
    const device SurfCard* cards,
    const device uint* triCard, const device float4* triUVa,
    const device float4* triUVc,
    const device void* primData,
    const device float4* cardRect, uint atlasW, uint atlasH)
{
    uint card = triCard[prim];
    float4 rect = cardRect[card];          // (x, y, w, h) in atlas px
    float2 uvA, uvB, uvC;
    if (primData != nullptr) {
        const device PrimUV* pd = (const device PrimUV*)primData;  // AS-baked, no side-buffer load
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
    float3 irr = atlas.sample(samp, px / float2(atlasW, atlasH)).rgb;
    SurfCard sc = cards[card];
    // Frame-B membership: frame A has uvA=(0,0) (sum 0), frame B has uvA=(1,1) (sum 2).
    bool useB = sc.normal.w > 0.5 && (uvA.x + uvA.y > 1.0);
    float3 albedo   = useB ? sc.albedoB.xyz   : sc.albedo.xyz;
    float3 emission = useB ? sc.emissionB.xyz : sc.emission.xyz;
    return albedo * irr + emission;
}

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

// Compact per-instance RT data (grouped order == TLAS instance_id). nrm0..2 are
// the columns of the 3×3 normal matrix; albedoTriBase.xyz = albedo, .w = the
// instance's base offset into `objNormal` (as float; cast back to uint).
struct RTInstanceData {
    float4 nrm0; float4 nrm1; float4 nrm2;
    float4 albedoTriBase;
};

static inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline float rnd(thread uint& seed) { seed = pcgHash(seed); return float(seed) * (1.0 / 4294967296.0); }
static inline void basis(float3 n, thread float3& t, thread float3& b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0; float a = -1.0 / (s + n.z); float d = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}
static inline float3 cosineSample(float3 n, float u1, float u2) {
    float r = sqrt(u1); float phi = 2.0 * M_PI_F * u2; float3 t, b; basis(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}
static inline float3 coneSample(float3 dir, float theta, float u1, float u2) {
    float cosT = mix(cos(theta), 1.0, u1); float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 2.0 * M_PI_F * u2; float3 t, b; basis(dir, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + dir * cosT);
}
static inline float2 dirToEquirectUV(float3 d) {
    return float2(atan2(d.z, d.x) * (1.0 / (2.0 * M_PI_F)) + 0.5,
                  acos(clamp(d.y, -1.0, 1.0)) * (1.0 / M_PI_F));
}
static inline float3 octDecode(float2 e) {
    e = e * 2.0 - 1.0; float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) { float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); n.xy = (1.0 - abs(n.yx)) * s; }
    return normalize(n);
}
static inline float3 worldPosFromDepth(float2 ndcXY, float depth, float4x4 invVP) {
    float4 w = invVP * float4(ndcXY, depth, 1.0); return w.xyz / w.w;
}

// World normal of the hit triangle from the per-instance normal matrix.
static inline float3 hitWorldNormal(uint iid, uint prim,
                                    const device RTInstanceData* insts,
                                    const device float4* objNormal) {
    RTInstanceData d = insts[iid];
    uint triBase = uint(d.albedoTriBase.w);
    float3 nObj = objNormal[triBase + prim].xyz;
    float3x3 nm = float3x3(d.nrm0.xyz, d.nrm1.xyz, d.nrm2.xyz);
    float3 n = nm * nObj;
    float len = length(n);
    return len > 1e-8 ? n / len : float3(0.0, 1.0, 0.0);
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

    // ── Direct sun (soft shadows) ─────────────────────────────────
    float3 direct = float3(0.0);
    float NdotL = saturate(dot(N, Ld));
    if (NdotL > 0.0 && u.shadowRays > 0) {
        isect.accept_any_intersection(true);
        uint hits = 0;
        for (uint s = 0; s < u.shadowRays; ++s) {
            ray r; r.origin = Pofs; r.direction = coneSample(Ld, u.sunSoftnessRad, rnd(seed), rnd(seed));
            r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = 1e4;
            // Any occluder counts — triangles always; curves when enabled. Mask
            // 0x01 = opaque + curve instances only; glass (mask 0x02, #60 AAA
            // glass) is excluded so a clear pane doesn't cast a solid shadow.
            if (isect.intersect(r, accel, 0x01u).type != intersection_type::none) hits++;
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
            auto res = isect.intersect(r, accel, 0x01u);  // opaque+curve only (exclude glass mask 0x02)
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
                        SurfCard sc = surfCards[hitCard];
                        indirect += sc.emission.xyz + sc.albedo.xyz * u.skyAmbient;
                    }
                    continue;
                }
                float3 hitN = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
                float3 hitA = insts[res.instance_id].albedoTriBase.xyz;
                float hN = saturate(dot(hitN, Ld));
                float3 hitRad = float3(0.0);
                if (hN > 0.0) {
                    isect.accept_any_intersection(true);
                    float3 hitP = r.origin + dir * res.distance;
                    ray sr; sr.origin = hitP + hitN * 2e-3; sr.direction = Ld;
                    sr.min_distance = 2e-3; sr.max_distance = 1e4;
                    float sv = (isect.intersect(sr, accel, 0x01u).type != intersection_type::none) ? 0.0 : 1.0;
                    hitRad = hitA * (1.0 / M_PI_F) * u.sunColor * hN * sv;
                    isect.accept_any_intersection(false);
                }
                indirect += hitRad;
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
                        float sv = (isect.intersect(sr, accel, 0x01u).type != intersection_type::none) ? 0.0 : 1.0;
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
        // which over-blurred shiny surfaces and under-blurred rough ones.
        float coneTheta = roughness * roughness * 1.2;
        uint rrays = max(1u, u.reflRays);
        float3 acc = float3(0.0);
        isect.accept_any_intersection(false);
        for (uint i = 0; i < rrays; ++i) {
            float3 dir = (coneTheta > 1e-4) ? coneSample(R, coneTheta, rnd(seed), rnd(seed)) : R;
            if (dot(dir, N) <= 0.0) continue;
            ray r; r.origin = Pofs; r.direction = dir;
            r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = u.reflMaxDist;
            auto res = isect.intersect(r, accel, 0x01u);  // opaque+curve only (exclude glass mask 0x02)
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
                        float sv = (isect.intersect(sr, accel, 0x01u).type != intersection_type::none) ? 0.0 : 1.0;
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
                    SurfCard sc = surfCards[hitCard];
                    acc += sc.emission.xyz + sc.albedo.xyz * u.skyAmbient;
                }
                continue;
            }
            float3 hitN = hitWorldNormal(res.instance_id, res.primitive_id, insts, objNormal);
            float3 hitA = insts[res.instance_id].albedoTriBase.xyz;
            float3 hitRad = hitA * u.skyAmbient;
            float hN = saturate(dot(hitN, Ld));
            if (hN > 0.0) {
                isect.accept_any_intersection(true);
                float3 hitP = r.origin + dir * res.distance;
                ray sr; sr.origin = hitP + hitN * 2e-3; sr.direction = Ld;
                sr.min_distance = 2e-3; sr.max_distance = 1e4;
                float sv = (isect.intersect(sr, accel, 0x01u).type != intersection_type::none) ? 0.0 : 1.0;
                isect.accept_any_intersection(false);
                hitRad += hitA * (1.0 / M_PI_F) * u.sunColor * hN * sv;
            }
            acc += hitRad;
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
        return;
    }
    float3 add = direct + indirect + reflection;
    outHDR.write(half4(prev.rgb + half3(add), prev.a), gid);
}
