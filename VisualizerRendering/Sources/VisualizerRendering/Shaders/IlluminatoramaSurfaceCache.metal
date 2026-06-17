#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// ── ILLUMINATORAMA SURFACE CACHE ─────────────────────────────────────────────
//
// A Lumen-style on-surface lit-radiance cache. Each static RT "card" (a planar
// parallelogram — the room's floor / walls / ceiling / window jambs) owns a
// tile in a ping-pong radiance atlas. Once per frame the update kernel re-lights
// every atlas texel and EMA-blends it into the current atlas; the RT GI +
// reflection rays then READ this cached radiance at their hit point instead of
// re-shading sun-only. Two wins:
//
//   • Multi-bounce — the update's indirect term samples the PREVIOUS frame's
//     atlas at its own ray hits, so light bounces accumulate frame over frame
//     (direct sun on the floor this frame → indirect on the wall next frame →
//     indirect on the ceiling the frame after …).
//   • Cheaper hits — a GI/reflection ray hit becomes one atlas read instead of
//     a re-shade + shadow ray; the shadow work is done once per atlas texel and
//     amortised across every ray that lands on that surface.
//
// The cache is COARSE on purpose (low-res per-card tiles) — it only feeds
// INDIRECT light (GI bounces + reflections of distant surfaces); primary
// visible surfaces are still fully deferred-shaded. This is the standard
// surface-cache trade.
//
// STRUCT / SAMPLER DUPLICATION: the `SurfCard` array is OWNED here (only the
// update kernel reads card frames). The consumers — IlluminatoramaRT.metal and
// IlluminatoramaRTInstanced.metal — carry only the per-triangle card-UV layout +
// a `sampleSurfCacheRT()` copy that reads `triCard`/`triUVa`/`triUVc` + the
// atlas. Because a hit's barycentric→UV routes to the correct half of a packed
// tile, the samplers are AGNOSTIC to the A/B split and don't change for P3 — but
// keep the UV-layout + atlas-indexing math in lockstep across all three files,
// and the Swift `SurfCard` mirror in IlluminatoramaRenderer.swift in step with
// this struct. All structs are SIMD4-aligned (no packed_float3 / SIMD3 trap).

// One atlas tile, carrying up to TWO planar triangle frames (P3 — 2-triangles-
// per-tile packing). Frame A occupies the lower-left half (texel u+v ≤ 1); frame
// B the upper-right half (u+v > 1), split on the diagonal. `normal.w` is the
// PACKED flag: 1 ⇒ split A/B; 0 ⇒ frame A fills the WHOLE tile (the room's quad
// cards — both of a quad's triangles map across the full tile — and a lone odd
// per-triangle card). Field-for-field mirror of the Swift `SurfCard`.
struct SurfCard {
    float4 origin;    // frame A corner (uv 0,0)
    float4 uAxis;     // frame A: origin + uAxis = uv (1,0)
    float4 vAxis;     // frame A: origin + vAxis = uv (0,1)
    float4 normal;    // frame A outward normal; .w = PACKED flag (1 = split A/B)
    float4 albedo;    // frame A; .w unused
    float4 emission;  // frame A; .w unused
    float4 originB;    // frame B corner (uv 1,1, mirrored); valid iff packed
    float4 uAxisB;     // frame B
    float4 vAxisB;     // frame B
    float4 normalB;    // frame B outward normal; .w unused
    float4 albedoB;    // frame B; .w unused
    float4 emissionB;  // frame B; .w unused
};

struct SurfCacheUniforms {
    float4 sunDir;       // xyz = toward sun (normalised); .w unused
    float4 sunColor;     // xyz = premultiplied radiance; .w unused
    float4 skyAmbient;   // xyz = flat fill for indirect-ray misses w/o sky; .w unused
    uint   atlasW;   uint atlasH;   uint tileSize;  uint tilesPerRow;
    uint   cardCount; uint triangleCount; uint indirectRays; uint frameSeed;
    float  alpha;    float rayTMin; float maxDist;  uint incrementalEnabled;
};

// ── shared helpers (kept local; Metal has no cross-file linkage) ──────────────

// Curve primitives (#60 item 7) — specializes `illumi_surfcache_update_tlas`
// for a TLAS that contains curve BLAS instances (round Catmull-Rom). Same
// constant index as IlluminatoramaRTInstanced.metal's kRTCurvesEnabled; the
// base variant (undefined → false) keeps the original triangle-only contract.
constant bool kSCCurvesEnabledFC [[function_constant(30)]];
constant bool kSCCurvesEnabled = is_function_constant_defined(kSCCurvesEnabledFC) && kSCCurvesEnabledFC;

static inline uint sc_pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline float sc_rnd(thread uint& seed) {
    seed = sc_pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}
static inline void sc_basis(float3 n, thread float3& t, thread float3& b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float d = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}
static inline float3 sc_cosineSample(float3 n, float u1, float u2) {
    float r = sqrt(u1);
    float phi = 2.0 * M_PI_F * u2;
    float3 t, b; sc_basis(n, t, b);
    float x = r * cos(phi), y = r * sin(phi), z = sqrt(max(0.0, 1.0 - u1));
    return normalize(t * x + b * y + n * z);
}
static inline float2 sc_dirToEquirectUV(float3 d) {
    float u = atan2(d.z, d.x) * (1.0 / (2.0 * M_PI_F)) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / M_PI_F);
    return float2(u, v);
}

// Phase 5 / B0 — per-texel variance tracking for the cache-domain denoiser.
// Rec.709 luminance. The atlas EMA is linear and per-channel, so
// luminance(EMA(rgb)) == EMA(luminance(rgb)) — i.e. the stored RGB already IS the
// mean-luminance EMA μ. We additionally EMA the luminance SQUARED into the atlas
// .w channel (previously a constant 1.0, read by nobody — consumers take .rgb),
// so any reader recovers variance = E[L²] − μ² with no extra buffer and no atlas
// format change. This is the signal B1's à-trous filter keys its step width off.
static inline float sc_luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// B3-spline à-trous tap weight (the SVGF spatial kernel) for an offset in
// [-2,2]; 0 outside, so the filter loop is safe for any radius ≤ 2.
static inline float sc_b3(int o) {
    switch (o) {
        case -2: case 2: return 1.0 / 16.0;
        case -1: case 1: return 4.0 / 16.0;
        case  0:         return 6.0 / 16.0;
        default:         return 0.0;
    }
}

// Map a triangle hit (primitive id + barycentric) to its card-UV, then sample
// the radiance atlas at that card's tile. `triCard[prim]` = card index;
// `triUVa[prim]` = (uvA.xy, uvB.xy); `triUVc[prim]` = (uvC.xy, _, _). UV is
// inset half a texel from the tile edge so bilinear can't bleed across tiles.
// The atlas stores albedo-free IRRADIANCE (#60 task 2): the radiance LEAVING a
// hit surface is reconstructed here as `albedo·irr + emission`, using the hit
// card's own material. Keeping albedo out of the stored value means a card seam
// (A=0.80 vs 0.82) no longer gets amplified into a coloured stripe by the albedo
// multiply, and the RT-hit path agrees with the cache path on the stored basis.
// The return value is the same outgoing radiance the consumers always read, so
// no caller math changes — only the stored basis + this reconstruction.
static inline float3 sampleSurfCache(
    texture2d<float, access::sample> atlas,
    uint prim,
    float2 bary,                       // intersector (u,v): w0=1-u-v, w1=u, w2=v
    const device SurfCard* cards,
    const device uint*   triCard,
    const device float4* triUVa,
    const device float4* triUVc,
    const device float4* cardRect, uint atlasW, uint atlasH)
{
    uint card = triCard[prim];
    float4 rect = cardRect[card];          // (x, y, w, h) in atlas px
    float4 a = triUVa[prim];
    float4 c = triUVc[prim];
    float2 uvA = a.xy, uvB = a.zw, uvC = c.xy;
    float w0 = 1.0 - bary.x - bary.y;
    float2 uv = saturate(w0 * uvA + bary.x * uvB + bary.y * uvC);
    float2 inset = 0.5 / max(float2(1.0), rect.zw);
    uv = clamp(uv, inset, 1.0 - inset);
    float2 px = rect.xy + uv * rect.zw;
    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 irr = atlas.sample(samp, px / float2(atlasW, atlasH)).rgb;
    // Reconstruct L_out from the hit card's material. Pick the packed tile's
    // frame the same way the UV blend does (frame B's reference uvA ≈ (1,1)).
    SurfCard sc = cards[card];
    bool useB = sc.normal.w > 0.5 && (a.x + a.y > 1.0);
    float3 albedo   = useB ? sc.albedoB.xyz   : sc.albedo.xyz;
    float3 emission = useB ? sc.emissionB.xyz : sc.emission.xyz;
    return albedo * irr + emission;
}

// ── update kernel ────────────────────────────────────────────────────────────
//
// One thread per atlas texel. Reconstructs the texel's world position + normal
// from its card frame, re-lights it (emission + direct sun + one-bounce indirect
// sampled from the PREVIOUS atlas), and EMA-blends into the current atlas.

kernel void illumi_surfcache_update(
    texture2d<float, access::write>   outAtlas    [[texture(0)]],
    texture2d<float, access::read>    prevAtlasR  [[texture(1)]],
    texture2d<float, access::sample>  prevAtlasS  [[texture(2)]],
    texture2d<float, access::sample>  skyEquirect [[texture(3)]],
    primitive_acceleration_structure  accel       [[buffer(0)]],
    const device SurfCard*            cards       [[buffer(1)]],
    const device uint*                triCard     [[buffer(2)]],
    const device float4*              triUVa      [[buffer(3)]],
    const device float4*              triUVc      [[buffer(4)]],
    constant SurfCacheUniforms&       u           [[buffer(5)]],
    const device float4*              cardRect    [[buffer(6)]],   // x,y,w,h px
    const device uint*                texelCard   [[buffer(7)]],   // texel→card (0xFFFFFFFF gap)
    const device uint*                cardDirty   [[buffer(8)]],   // incremental: 1 = re-light fresh (α=1)
    // Curve primitives live in their OWN primitive AS (#60 item 7 incr. 2). When
    // kSCCurvesEnabled, traced alongside `accel` as an occluder (canopy darkens the
    // cards beneath). Bound to `accel` as a harmless dummy for the base variant.
    primitive_acceleration_structure  curveAccel  [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.atlasW || gid.y >= u.atlasH) return;

    // P2 — per-card RECT addressing (shelf-packed atlas). The texel→card map
    // tells us which card owns this texel; its rect gives the local [0,1]² UV.
    uint card = texelCard[gid.y * u.atlasW + gid.x];
    if (card == 0xFFFFFFFFu || card >= u.cardCount) return;   // unused gap

    SurfCard sc = cards[card];
    float4 rect = cardRect[card];                 // (x, y, w, h) in atlas px

    // Texel-centre UV inside this card's rect.
    float2 local = (float2(gid) - rect.xy + 0.5) / rect.zw;

    // P3 — pick the tile's frame. Packed (normal.w==1): the lower-left half
    // (u+v ≤ 1) is frame A; the upper-right half (u+v > 1) is frame B, whose
    // texels mirror back into a lower-left parameterisation (lb = 1 − local) so
    // `origin + lb·axes` reconstructs frame B's plane — matching the B reference
    // UVs A(1,1) B(0,1) C(1,0) the sampler uses. Unpacked: frame A fills the tile.
    bool packed = sc.normal.w > 0.5;
    bool useB = packed && (local.x + local.y > 1.0);
    // Albedo/emission are NOT needed here — the stored value is albedo-free
    // irradiance (#60 task 2), so the update only reconstructs the texel's frame.
    float3 frO, frU, frV, N;
    float2 luv;
    if (useB) {
        frO = sc.originB.xyz; frU = sc.uAxisB.xyz; frV = sc.vAxisB.xyz;
        N = normalize(sc.normalB.xyz);
        luv = float2(1.0 - local.x, 1.0 - local.y);
    } else {
        frO = sc.origin.xyz; frU = sc.uAxis.xyz; frV = sc.vAxis.xyz;
        N = normalize(sc.normal.xyz);
        luv = local;
    }
    float3 P = frO + luv.x * frU + luv.y * frV;
    float3 Pofs = P + N * max(u.rayTMin, 1e-3);

    constexpr sampler skySamp(filter::linear, address::repeat);
    // Curve primitives (#60 item 7, incr. 2): when this soup primitive AS holds a
    // curve geometry descriptor, widen the intersector to match. NO `instancing`
    // tag (primitive AS). A curve hit OCCLUDES the sun ray (canopy darkens the
    // cards beneath) and TERMINATES an indirect ray with no contribution (curves
    // carry no card — the declared black-bounce fallback).
    intersector<triangle_data, curve_data> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    if (kSCCurvesEnabled) {
        isect.assume_geometry_type(geometry_type::triangle | geometry_type::curve);
        isect.assume_curve_basis(curve_basis::catmull_rom);
        isect.assume_curve_type(curve_type::round);
        isect.assume_curve_control_point_count(4);
    }

    uint seed = sc_pcgHash(gid.x + gid.y * u.atlasW + u.frameSeed * 9781u);
    float3 Ld = normalize(u.sunDir.xyz);

    // ── Direct sun (one shadow ray) ──────────────────────────────
    float3 direct = float3(0.0);
    float NdotL = saturate(dot(N, Ld));
    if (NdotL > 0.0) {
        isect.accept_any_intersection(true);
        ray sr;
        sr.origin = Pofs; sr.direction = Ld;
        sr.min_distance = max(u.rayTMin, 1e-3); sr.max_distance = 1e4;
        bool occ = isect.intersect(sr, accel).type != intersection_type::none;
        if (kSCCurvesEnabled && !occ) occ = isect.intersect(sr, curveAccel).type != intersection_type::none;
        float vis = occ ? 0.0 : 1.0;   // curve or triangle occludes
        // Store albedo-free IRRADIANCE (#60 task 2): the Lambertian kernel
        // (1/π · E·N·L) without the surface albedo. Albedo is re-applied per card
        // at read in sampleSurfCache, so material boundaries don't amplify cache
        // discontinuities.
        direct = (1.0 / M_PI_F) * u.sunColor.xyz * NdotL * vis;
    }

    // ── One-bounce indirect (reads PREVIOUS atlas at hits → multi-bounce) ──
    // sampleSurfCache reconstructs the hit's outgoing radiance (albedo·irr +
    // emission); summed/averaged it is the irradiance arriving at THIS texel.
    float3 indirect = float3(0.0);
    uint rays = max(1u, u.indirectRays);
    isect.accept_any_intersection(false);
    for (uint i = 0; i < rays; ++i) {
        float3 dir = sc_cosineSample(N, sc_rnd(seed), sc_rnd(seed));
        ray r;
        r.origin = Pofs; r.direction = dir;
        r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = u.maxDist;
        auto res = isect.intersect(r, accel);   // triangle soup AS
        float triDist = (res.type == intersection_type::triangle) ? res.distance : 3.0e38;
        // Curves are a SEPARATE primitive AS (#60 item 7 incr. 2). A curve nearer
        // than the triangle hit OCCLUDES the bounce (no card → no contribution —
        // the declared black-bounce fallback; a canopy darkens the cards beneath).
        if (kSCCurvesEnabled) {
            auto cres = isect.intersect(r, curveAccel);
            if (cres.type == intersection_type::curve && cres.distance < triDist) continue;
        }
        if (res.type == intersection_type::triangle) {
            uint prim = res.primitive_id;
            if (prim >= u.triangleCount) continue;
            indirect += sampleSurfCache(prevAtlasS, prim,
                                        res.triangle_barycentric_coord,
                                        cards, triCard, triUVa, triUVc,
                                        cardRect, u.atlasW, u.atlasH);
        } else {
            indirect += skyEquirect.sample(skySamp, sc_dirToEquirectUV(dir)).rgb;
        }
    }
    indirect = indirect / float(rays);   // arriving irradiance (no albedo — applied at read)

    // Stored value is irradiance only — emission is added per card at read time.
    float3 newIrr = direct + indirect;

    // Temporal EMA so the multi-bounce builds up smoothly and the Monte-Carlo
    // indirect noise is denoised across frames. Incremental invalidation: a card
    // flagged dirty this frame (its instance moved) uses α=1 to DISCARD its now-
    // stale-position history and re-converge from the new pose, while stationary
    // cards keep their small α and their accumulated multi-bounce irradiance.
    float alpha = clamp(u.alpha, 0.02, 1.0);
    if (u.incrementalEnabled != 0u && cardDirty[card] != 0u) alpha = 1.0;
    float4 prevTexel = prevAtlasR.read(gid);
    float3 outIrr = mix(prevTexel.rgb, newIrr, alpha);
    // B0 — EMA the per-texel luminance² into .w alongside the radiance, so
    // variance = E[L²] − μ² is free for the denoiser. The SAME α applies to both
    // moments (incl. the dirty-card α=1 reset), so a re-framed card's variance
    // resets in lockstep with its mean and re-converges from its new pose.
    float newL = sc_luminance(newIrr);
    float outM2 = mix(prevTexel.a, newL * newL, alpha);
    outAtlas.write(float4(outIrr, outM2), gid);
}

// ── update kernel — TLAS-traced variant (§3 endpoint) ─────────────────────────
//
// Identical to `illumi_surfcache_update` EXCEPT it traces the per-frame-refit
// INSTANCE acceleration structure (the same TLAS the lighting pass uses) instead
// of the static t₀ soup primitive-AS. Why it matters for incremental
// invalidation: the soup AS bakes object positions at topology time and is never
// refit, so a moved object's shadow/bounce rays trace its OLD pose — its
// self-shadow + bounce-occlusion lag its (correctly re-framed) surface. The TLAS
// is refit every frame to current transforms, so tracing it makes occlusion track
// the motion. A TLAS hit is per-instance-local, so the global soup triangle (for
// the card-UV lookup) is `soupTriBase[instance_id] + primitive_id` — exactly the
// mapping `illumi_rt_lighting_tlas` uses.
//
// KEEP IN LOCKSTEP with `illumi_surfcache_update` above: the texel→world
// reconstruction, direct-sun, EMA, and incremental-α logic are line-for-line the
// same; only the intersector type (+`instancing`), the AS, and the bounce-hit
// prim resolution differ.
kernel void illumi_surfcache_update_tlas(
    texture2d<float, access::write>   outAtlas    [[texture(0)]],
    texture2d<float, access::read>    prevAtlasR  [[texture(1)]],
    texture2d<float, access::sample>  prevAtlasS  [[texture(2)]],
    texture2d<float, access::sample>  skyEquirect [[texture(3)]],
    instance_acceleration_structure   accel       [[buffer(0)]],
    const device SurfCard*            cards       [[buffer(1)]],
    const device uint*                triCard     [[buffer(2)]],
    const device float4*              triUVa      [[buffer(3)]],
    const device float4*              triUVc      [[buffer(4)]],
    constant SurfCacheUniforms&       u           [[buffer(5)]],
    const device float4*              cardRect    [[buffer(6)]],
    const device uint*                texelCard   [[buffer(7)]],
    const device uint*                cardDirty   [[buffer(8)]],
    const device uint*                soupTriBase [[buffer(9)]],   // TLAS (inst,prim) → global soup tri
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.atlasW || gid.y >= u.atlasH) return;

    uint card = texelCard[gid.y * u.atlasW + gid.x];
    if (card == 0xFFFFFFFFu || card >= u.cardCount) return;

    SurfCard sc = cards[card];
    float4 rect = cardRect[card];
    float2 local = (float2(gid) - rect.xy + 0.5) / rect.zw;

    bool packed = sc.normal.w > 0.5;
    bool useB = packed && (local.x + local.y > 1.0);
    // Albedo/emission are NOT needed here — the stored value is albedo-free
    // irradiance (#60 task 2), so the update only reconstructs the texel's frame.
    float3 frO, frU, frV, N;
    float2 luv;
    if (useB) {
        frO = sc.originB.xyz; frU = sc.uAxisB.xyz; frV = sc.vAxisB.xyz;
        N = normalize(sc.normalB.xyz);
        luv = float2(1.0 - local.x, 1.0 - local.y);
    } else {
        frO = sc.origin.xyz; frU = sc.uAxis.xyz; frV = sc.vAxis.xyz;
        N = normalize(sc.normal.xyz);
        luv = local;
    }
    float3 P = frO + luv.x * frU + luv.y * frV;
    float3 Pofs = P + N * max(u.rayTMin, 1e-3);

    constexpr sampler skySamp(filter::linear, address::repeat);
    // Curve primitives (#60 item 7): when the TLAS contains curve instances
    // this kernel runs as the `kSCCurvesEnabled` variant so the traversal
    // contract matches the AS. Curve hits OCCLUDE the sun ray (a canopy
    // darkens the cards beneath it) and TERMINATE an indirect ray without
    // contribution (curves carry no cache card; a black bounce is the
    // conservative honest fallback — declared in the design note). The base
    // variant keeps the original triangle-only contract.
    intersector<triangle_data, instancing, curve_data> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    if (kSCCurvesEnabled) {
        isect.assume_geometry_type(geometry_type::triangle | geometry_type::curve);
        isect.assume_curve_basis(curve_basis::catmull_rom);
        isect.assume_curve_type(curve_type::round);
        isect.assume_curve_control_point_count(4);
    }

    uint seed = sc_pcgHash(gid.x + gid.y * u.atlasW + u.frameSeed * 9781u);
    float3 Ld = normalize(u.sunDir.xyz);

    // ── Direct sun (one shadow ray, traces CURRENT geometry) ─────
    float3 direct = float3(0.0);
    float NdotL = saturate(dot(N, Ld));
    if (NdotL > 0.0) {
        isect.accept_any_intersection(true);
        ray sr;
        sr.origin = Pofs; sr.direction = Ld;
        sr.min_distance = max(u.rayTMin, 1e-3); sr.max_distance = 1e4;
        // Mask 0x01 = opaque + curve only; AAA glass (mask 0x02) is excluded so
        // a clear pane doesn't occlude the cache's sun visibility / GI gather.
        auto sres = isect.intersect(sr, accel, 0x01u);
        float vis = (sres.type != intersection_type::none) ? 0.0 : 1.0;
        // Albedo-free irradiance (#60 task 2) — albedo applied per card at read.
        direct = (1.0 / M_PI_F) * u.sunColor.xyz * NdotL * vis;
    }

    // ── One-bounce indirect (reads PREVIOUS atlas at hits) ───────
    float3 indirect = float3(0.0);
    uint rays = max(1u, u.indirectRays);
    isect.accept_any_intersection(false);
    for (uint i = 0; i < rays; ++i) {
        float3 dir = sc_cosineSample(N, sc_rnd(seed), sc_rnd(seed));
        ray r;
        r.origin = Pofs; r.direction = dir;
        r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = u.maxDist;
        auto res = isect.intersect(r, accel, 0x01u);  // opaque+curve only (exclude glass mask 0x02)
        if (res.type == intersection_type::triangle) {
            uint prim = soupTriBase[res.instance_id] + res.primitive_id;
            if (prim >= u.triangleCount) continue;
            indirect += sampleSurfCache(prevAtlasS, prim,
                                        res.triangle_barycentric_coord,
                                        cards, triCard, triUVa, triUVc,
                                        cardRect, u.atlasW, u.atlasH);
        } else if (kSCCurvesEnabled && res.type == intersection_type::curve) {
            continue;   // occluded by a curve — no card, no sky (see above)
        } else {
            indirect += skyEquirect.sample(skySamp, sc_dirToEquirectUV(dir)).rgb;
        }
    }
    indirect = indirect / float(rays);   // arriving irradiance (no albedo)

    float3 newIrr = direct + indirect;   // emission added per card at read

    float alpha = clamp(u.alpha, 0.02, 1.0);
    if (u.incrementalEnabled != 0u && cardDirty[card] != 0u) alpha = 1.0;
    float4 prevTexel = prevAtlasR.read(gid);
    float3 outIrr = mix(prevTexel.rgb, newIrr, alpha);
    // B0 — luminance² EMA in .w (see the soup variant above; kept line-for-line).
    float newL = sc_luminance(newIrr);
    float outM2 = mix(prevTexel.a, newL * newL, alpha);
    outAtlas.write(float4(outIrr, outM2), gid);
}

// ── Phase 5 / B1: cache-domain à-trous denoiser ──────────────────────────────
//
// A spatial variance-guided filter over the radiance atlas, run AFTER the update
// kernel and BEFORE the GI/reflection consumers read it. It accelerates the
// cache's convergence: the temporal EMA alone resolves a freshly-reset (moved /
// cold) card's Monte-Carlo noise over ~20 frames; the spatial filter cleans it in
// ~1.
//
// THE ATLAS-NEIGHBOUR TRAP. Atlas texels adjacent in (x,y) are NOT world
// neighbours — the atlas is shelf-packed, so the tile next door is an unrelated
// surface. A naive 2D blur would bleed one card's radiance into another's. The
// guard is `texelCard`: a tap is taken ONLY when it belongs to the SAME card as
// the centre. Within one card every texel is coplanar (a single planar frame), so
// the sole feature to preserve is genuine radiance variation on the card (e.g. a
// shadow edge); the luminance edge-stop — loosened where the tracked variance is
// high — does that without over-blurring converged detail.
//
// FEEDBACK IS LEFT RAW. The multi-bounce feedback (the update kernel sampling the
// previous atlas) keeps reading the un-denoised EMA atlas; only the per-frame
// CONSUMER reads this filtered copy. Filter-then-feed-back would risk temporal
// energy drift; display-only denoise cannot change the accumulation's fixed
// point. So the filter speeds convergence without altering what it converges to.
struct SurfAtrousUniforms {
    uint  atlasW; uint atlasH; uint stepSize; uint radius;
    float phi;    float _ap0; float _ap1; float _ap2;   // 16-B clusters (stride 32)
};

kernel void illumi_surfcache_atrous(
    texture2d<float, access::write>  outAtlas  [[texture(0)]],
    texture2d<float, access::read>   inAtlas   [[texture(1)]],
    const device uint*               texelCard [[buffer(0)]],   // texel→card (0xFFFFFFFF gap)
    constant SurfAtrousUniforms&     u         [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.atlasW || gid.y >= u.atlasH) return;
    float4 c = inAtlas.read(gid);
    uint center = texelCard[gid.y * u.atlasW + gid.x];
    if (center == 0xFFFFFFFFu) { outAtlas.write(c, gid); return; }   // atlas gap — pass through

    // B1b — intra-card variance-guided à-trous. One B3-spline pass over a
    // (2·radius+1)² footprint at `stepSize`, edge-stopped by (1) SAME-CARD
    // membership — the only thing that keeps shelf-packed neighbours from bleeding
    // — and (2) a luminance edge-stop whose width scales with the local NOISE.
    //
    // Noise estimate = max(temporal, spatial) variance. The B0 temporal variance
    // (E[L²]−μ²) is the right signal for a slowly-converging static card, but it
    // reads ≈0 for a card RESET every frame (α=1: a single sample has E[L²]=L²,
    // μ=L ⇒ var=0) — which is exactly the moving / cold card whose Monte-Carlo
    // grain is worst. So we ALSO estimate variance SPATIALLY over the footprint
    // (independent per-texel noise ⇒ high spatial variance) and take the larger.
    // A converged flat card has both ≈0 ⇒ tight stop ⇒ preserved (no over-blur);
    // a noisy card gets a loose stop ⇒ the grain is averaged out. A genuine
    // luminance edge keeps a large |ΔL| ⇒ low weight ⇒ survives regardless.
    float lumC = sc_luminance(c.rgb);
    float tVar = max(0.0, c.a - lumC * lumC);       // B0 temporal variance
    int   R = min(2, int(u.radius));
    int   S = int(max(1u, u.stepSize));
    int   W = int(u.atlasW), H = int(u.atlasH);

    // Pass 1 — spatial luminance mean/variance over same-card taps.
    float sL = 0.0, sL2 = 0.0, sn = 0.0;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int tx = int(gid.x) + dx * S, ty = int(gid.y) + dy * S;
            if (tx < 0 || ty < 0 || tx >= W || ty >= H) continue;
            if (texelCard[ty * W + tx] != center) continue;
            float lumT = sc_luminance(inAtlas.read(uint2(uint(tx), uint(ty))).rgb);
            sL += lumT; sL2 += lumT * lumT; sn += 1.0;
        }
    }
    float sMean = sn > 0.0 ? sL / sn : lumC;
    float sVar  = max(0.0, (sn > 0.0 ? sL2 / sn : 0.0) - sMean * sMean);
    float sigma = u.phi * sqrt(max(tVar, sVar)) + 1e-4;   // edge-stop width

    // Pass 2 — B3 à-trous with the variance-scaled luminance edge-stop.
    float3 sumRGB = float3(0.0);
    float  sumM2  = 0.0;
    float  wsum   = 0.0;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int tx = int(gid.x) + dx * S, ty = int(gid.y) + dy * S;
            if (tx < 0 || ty < 0 || tx >= W || ty >= H) continue;
            if (texelCard[ty * W + tx] != center) continue;   // SAME-CARD guard
            float4 t    = inAtlas.read(uint2(uint(tx), uint(ty)));
            float  lumT = sc_luminance(t.rgb);
            float  wl   = exp(-fabs(lumT - lumC) / sigma);
            float  w    = sc_b3(dx) * sc_b3(dy) * wl;
            sumRGB += w * t.rgb;
            sumM2  += w * t.a;
            wsum   += w;
        }
    }
    // wsum ≥ the centre's own weight (same card, wl=1), so this branch is taken;
    // the fallback only guards a degenerate zero.
    outAtlas.write(wsum > 1e-6 ? float4(sumRGB / wsum, sumM2 / wsum) : c, gid);
}

// ── GPU-side incremental invalidation (#60 item 1 + Phase D) ─────────────────
//
// The per-frame moved-set detection + card re-frame used to be a CPU loop in
// `applyIncrementalSurfaceCacheUpdate` (diff every instance's model matrix,
// re-transform every moved triangle on the CPU). These kernels move that onto
// the frame's command buffer: each thread diffs its OWNER instance's
// current-vs-previous model matrix inline (32 floats — trivially bandwidth-
// bound) and, when it moved, re-frames its card (half) in place + flags it
// dirty, exactly mirroring the CPU math. Stationary cards get dirty=0, which
// is the memset the CPU path did.
//
// Phase D — deforming (DynamicMesh / GPU-fed) meshes are GPU-side too:
// `illumi_surfcache_reframe_deform` re-frames a deforming instance's cards
// straight from the LIVE packed_float3 position buffer the solver writes (the
// same buffer the BLAS refit reads), every frame, with NO CPU readback. The
// topology (vertex-index triples) is constant per bake, so it rides a baked
// side buffer. The tri kernel skips deform-flagged triangles entirely — one
// path owns each card's dirty flag.

// Field-for-field mirror of Illuminatorama.metal's `Instance` (stride 208).
// Only `modelMatrix` is read here; the rest fixes the stride.
struct SCInstance {
    float4x4 modelMatrix;
    float4x4 normalMatrix;
    float3   albedo;    float metallic;
    float3   emission;  float roughness;
    int      albedoTextureSlice;  int metallicTextureSlice;
    int      roughnessTextureSlice; int normalTextureSlice;
    int      emissionTextureSlice;
    float    emissionIntensity;   int _padSlice1;
};

struct SCReframeUniforms {
    uint  count;          // grid size: soup triangles (tri kernel) / cards (chart kernel)
    uint  cardCount;
    uint  instanceCount;
    float epsilon;        // matrix-diff threshold (mirrors the CPU 1e-5)
    // Phase D — 0 when rigid incremental invalidation is OFF but a deforming
    // mesh still drives the per-frame pass: rigid cards then just get dirty=0
    // (the CPU memset semantics) with no diff / re-frame. Chart kernel ignores.
    uint  rigidEnabled;
    uint  _pad0; uint _pad1; uint _pad2;   // close on 16 B (stride 32 both sides)
};

// Per-dispatch uniforms for the deform kernel (one dispatch per deforming
// instance; thread = LOCAL triangle index).
struct SCDeformUniforms {
    uint triBase;        // instance's first GLOBAL soup-triangle index
    uint triCount;       // local (appended) triangle count for this instance
    uint instanceIndex;  // into the instance buffer (model matrix)
    uint vertexCount;    // live position-buffer length (defensive bound)
};

// `triFlags` bit layout (tri kernel + deform kernel):
#define SC_FLAG_FRAME_B   1u   // this triangle is the frame-B half of its card
#define SC_FLAG_DEFORMING 2u   // owner is a deforming kind → deform kernel owns it

static inline bool sc_instanceMoved(const device SCInstance* cur,
                                    const device SCInstance* prev,
                                    uint i, float e) {
    float4x4 a = cur[i].modelMatrix, b = prev[i].modelMatrix;
    for (int c = 0; c < 4; ++c) {
        float4 d = abs(a[c] - b[c]);
        if (d.x > e || d.y > e || d.z > e || d.w > e) return true;
    }
    return false;
}

// One thread per grouped-soup triangle (per-triangle card path). The frame-A
// thread OWNS its card's dirty flag (every card has exactly one frame-A
// triangle by construction — pairs are A+B, lone cards are A — so there is no
// write race); the frame-B thread only rewrites its own half's frame fields
// when moved (same owner instance as its A partner — pairing never crosses an
// instance block — so the A thread has already set dirty=1). `stats` is read
// back by the CPU NEXT frame (maxFramesInFlight == 1) for the diagnostic
// sidecar: [0] moved instances, [1] dirtied cards.
kernel void illumi_surfcache_reframe_tri(
    device SurfCard*            cards     [[buffer(0)]],
    device uint*                dirty     [[buffer(1)]],
    const device SCInstance*    curInst   [[buffer(2)]],
    const device SCInstance*    prevInst  [[buffer(3)]],
    const device uint*          triOwner  [[buffer(4)]],   // instance per soup tri
    const device uint*          triCard   [[buffer(5)]],
    const device uint*          triFlags  [[buffer(6)]],   // SC_FLAG_* bits
    const device float4*        objVerts  [[buffer(7)]],   // 3 object-space verts per tri
    constant SCReframeUniforms& u         [[buffer(8)]],
    device atomic_uint*         stats     [[buffer(9)]],
    uint gt [[thread_position_in_grid]])
{
    if (gt >= u.count) return;
    uint card = triCard[gt];
    if (card >= u.cardCount) return;
    uint flags = triFlags[gt];
    // Phase D — deform-flagged triangles belong to `_reframe_deform` (which
    // re-frames from the LIVE vertex buffer and sets dirty=1 every frame);
    // this kernel must not touch their cards' frames OR dirty flags.
    if ((flags & SC_FLAG_DEFORMING) != 0u) return;
    bool frameB = (flags & SC_FLAG_FRAME_B) != 0u;
    uint owner = triOwner[gt];
    if (owner >= u.instanceCount) {            // sentinel / excluded instance
        if (!frameB) dirty[card] = 0u;
        return;
    }
    bool didMove = u.rigidEnabled != 0u
        && sc_instanceMoved(curInst, prevInst, owner, u.epsilon);
    if (!frameB) {
        dirty[card] = didMove ? 1u : 0u;
        if (didMove) {
            atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
            // First soup triangle of its instance counts the instance once.
            if (gt == 0u || triOwner[gt - 1u] != owner)
                atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
        }
    }
    if (!didMove) return;
    float4x4 m = curInst[owner].modelMatrix;
    float3 w0 = (m * float4(objVerts[3u * gt + 0u].xyz, 1.0)).xyz;
    float3 w1 = (m * float4(objVerts[3u * gt + 1u].xyz, 1.0)).xyz;
    float3 w2 = (m * float4(objVerts[3u * gt + 2u].xyz, 1.0)).xyz;
    float3 n = cross(w1 - w0, w2 - w0);
    float len = length(n);
    n = len > 1e-8 ? n / len : float3(0.0, 1.0, 0.0);
    if (frameB) {
        cards[card].originB = float4(w0, 0.0);
        cards[card].uAxisB  = float4(w1 - w0, 0.0);
        cards[card].vAxisB  = float4(w2 - w0, 0.0);
        cards[card].normalB = float4(n, cards[card].normalB.w);
    } else {
        cards[card].origin = float4(w0, 0.0);
        cards[card].uAxis  = float4(w1 - w0, 0.0);
        cards[card].vAxis  = float4(w2 - w0, 0.0);
        cards[card].normal = float4(n, cards[card].normal.w);
    }
}

// One thread per card (coplanar-chart path). A chart card's frame is
// affine-covariant: worldFrame = M_now · objectFrame (origin as point, axes as
// vectors), UVs invariant — same math as the CPU
// `applyIncrementalChartSurfaceCacheUpdate`. `cardOwner[c] < 0` marks a static
// or degenerate-bake-pose card (never re-framed, mirrors the CPU skip). Owners
// are contiguous card ranges, so the first card of a run counts its instance.
kernel void illumi_surfcache_reframe_chart(
    device SurfCard*            cards     [[buffer(0)]],
    device uint*                dirty     [[buffer(1)]],
    const device SCInstance*    curInst   [[buffer(2)]],
    const device SCInstance*    prevInst  [[buffer(3)]],
    const device int*           cardOwner [[buffer(4)]],   // -1 = static
    const device float4*        objFrame  [[buffer(5)]],   // origin,uAxis,vAxis,normal per card
    constant SCReframeUniforms& u         [[buffer(6)]],
    device atomic_uint*         stats     [[buffer(7)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= u.count) return;
    int owner = cardOwner[c];
    if (owner < 0 || uint(owner) >= u.instanceCount) { dirty[c] = 0u; return; }
    if (!sc_instanceMoved(curInst, prevInst, uint(owner), u.epsilon)) {
        dirty[c] = 0u;
        return;
    }
    float4x4 m = curInst[uint(owner)].modelMatrix;
    float3 o  = (m * float4(objFrame[4u * c + 0u].xyz, 1.0)).xyz;
    float3 ua = (m * float4(objFrame[4u * c + 1u].xyz, 0.0)).xyz;
    float3 va = (m * float4(objFrame[4u * c + 2u].xyz, 0.0)).xyz;
    float3 n  = (m * float4(objFrame[4u * c + 3u].xyz, 0.0)).xyz;
    float nl = length(n);
    n = nl > 1e-8 ? n / nl : float3(0.0, 1.0, 0.0);
    cards[c].origin = float4(o, 0.0);
    cards[c].uAxis  = float4(ua, 0.0);
    cards[c].vAxis  = float4(va, 0.0);
    cards[c].normal = float4(n, cards[c].normal.w);   // preserve packed flag (0 for charts)
    dirty[c] = 1u;
    atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
    if (c == 0u || cardOwner[c - 1u] != owner)
        atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
}

// Phase D — deforming-mesh card re-frame, fully GPU-resident. One dispatch per
// deforming INSTANCE (they're rare — a fluid sheet, a flag); one thread per
// LOCAL soup triangle. Reads the instance's CURRENT model matrix, the baked
// vertex-index triple for the triangle (topology is constant per bake — a
// topology change re-bakes everything), and the LIVE packed_float3 position
// buffer the solver writes (the same buffer the BLAS refit traces), then
// rewrites the card half + marks the card dirty (α=1) — deforming surfaces
// re-light every frame by definition. Replaces the Phase-C CPU path's
// per-frame readback + transform of the whole live soup.
//
// Dirty ownership: the tri kernel skips SC_FLAG_DEFORMING triangles, so this
// kernel is the only writer of its cards' dirty flags. Both halves of a packed
// card write the same value (1u) — benign.
kernel void illumi_surfcache_reframe_deform(
    device SurfCard*             cards     [[buffer(0)]],
    device uint*                 dirty     [[buffer(1)]],
    const device SCInstance*     curInst   [[buffer(2)]],
    const device uint*           triCard   [[buffer(3)]],
    const device uint*           triFlags  [[buffer(4)]],   // SC_FLAG_* bits
    const device uint4*          triIdx    [[buffer(5)]],   // vertex-index triple per GLOBAL tri (.w unused)
    const device packed_float3*  livePos   [[buffer(6)]],   // solver-written positions (object space)
    constant SCDeformUniforms&   u         [[buffer(7)]],
    device atomic_uint*          stats     [[buffer(8)]],   // [1] dirtied cards, [2] deformed instances
    uint t [[thread_position_in_grid]])
{
    if (t >= u.triCount) return;
    uint gt = u.triBase + t;
    uint card = triCard[gt];
    uint flags = triFlags[gt];
    bool frameB = (flags & SC_FLAG_FRAME_B) != 0u;
    uint4 idx = triIdx[gt];
    if (idx.x >= u.vertexCount || idx.y >= u.vertexCount || idx.z >= u.vertexCount) {
        if (!frameB) dirty[card] = 0u;   // defensive — bake guards make this unreachable
        return;
    }
    if (t == 0u)
        atomic_fetch_add_explicit(&stats[2], 1u, memory_order_relaxed);
    float4x4 m = curInst[u.instanceIndex].modelMatrix;
    float3 w0 = (m * float4(float3(livePos[idx.x]), 1.0)).xyz;
    float3 w1 = (m * float4(float3(livePos[idx.y]), 1.0)).xyz;
    float3 w2 = (m * float4(float3(livePos[idx.z]), 1.0)).xyz;
    float3 n = cross(w1 - w0, w2 - w0);
    float len = length(n);
    n = len > 1e-8 ? n / len : float3(0.0, 1.0, 0.0);
    if (frameB) {
        cards[card].originB = float4(w0, 0.0);
        cards[card].uAxisB  = float4(w1 - w0, 0.0);
        cards[card].vAxisB  = float4(w2 - w0, 0.0);
        cards[card].normalB = float4(n, cards[card].normalB.w);
    } else {
        cards[card].origin = float4(w0, 0.0);
        cards[card].uAxis  = float4(w1 - w0, 0.0);
        cards[card].vAxis  = float4(w2 - w0, 0.0);
        cards[card].normal = float4(n, cards[card].normal.w);
        dirty[card] = 1u;                 // deforming cards re-light every frame
        atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
    }
}
