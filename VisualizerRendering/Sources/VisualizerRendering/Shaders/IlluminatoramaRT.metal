#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// ── ILLUMINATORAMA RAY TRACING ───────────────────────────────────────────────
//
// Hardware-ray-traced sun lighting for the Illuminatorama renderer. Runs as a
// compute pass AFTER the deferred lighting + SSR composite. When enabled, the
// host turns the deferred directional light (and cascaded shadows) OFF and lets
// this pass own ALL sun lighting:
//
//   • Soft shadows — `shadowRays` rays sampled across the sun's angular disc
//     give a true penumbra (contact-hard, soft far from the occluder).
//   • One-bounce GI — `giRays` cosine-weighted hemisphere rays gather indirect
//     radiance: each ray that hits room geometry is shaded with the sun (one
//     shadow ray) × the hit surface's albedo; rays that escape the window
//     sample the sky. This throws warm light off the wood floor onto the
//     walls + ceiling that the screen-space passes can't produce.
//
// The result is added into the HDR composite, so the renderer's TAA history
// denoises the Monte-Carlo noise across frames for free, and bloom + tonemap
// process it on the same path as everything else.
//
// The acceleration structure is built once from the room's static world-space
// triangles (see IlluminatoramaRenderer.setRTGeometry). Per-triangle albedo +
// normal travel in parallel buffers indexed by `primitive_id`.

struct RTUniforms {
    float4x4 invViewProjection;
    float3 cameraWorldPos;   float _pad0;
    float3 sunDir;           float sunSoftnessRad;   // sunDir = toward the sun
    float3 sunColor;         float giStrength;       // sunColor = premultiplied intensity
    float3 skyAmbient;       float specStrength;     // flat sky term for GI ray misses (fallback)
    uint  width;   uint height;
    uint  shadowRays; uint giRays;
    uint  frameSeed;  float rayTMin;  float maxGIDist; uint triangleCount;
    // ── Reflections (geometry-hit only; sky misses left to IBL) ──
    float reflStrength;       // master scale; 0 = reflection block skipped
    float reflMaxDist;        // max reflection ray length, metres
    float reflRoughnessCutoff;// surfaces rougher than this skip reflections
    uint  reflRays;           // glossy samples (1 = mirror-sharp)
    uint  reflEnabled;        // 0/1 master toggle
    // Surface cache: read cached radiance at GI/reflection hits when enabled.
    uint  surfCacheEnabled;
    uint  surfTileSize; uint surfTilesPerRow; uint surfAtlasW; uint surfAtlasH;
    // Analytic point emitters (particle fields) — entries in DDGIPointEmitter[].
    uint  emitterCount;
    // Leaf thin-sheet transmission strength (issue #58). 0 = OFF (default) so the
    // foliage-flag branch is a no-op unless the scene opts in. Repurposes _padRT0
    // — same 4 bytes, stride unchanged. Keep in lockstep with FrameUniforms.
    float leafTransmission;
    // Curve primitives (#60 item 7, increment 2): 1 ⇒ this soup primitive-AS holds
    // a curve geometry descriptor (round Catmull-Rom wood), so curve hits occlude
    // + shade. Repurposes _padRT1 (same 4 bytes). Unread by the base (curve-free)
    // pipeline variant. Keep in lockstep with the Swift RTUniforms mirror.
    uint  curvesEnabled; uint _padRT2;           // keep stride 16-byte aligned
};

// Analytic point emitter — mirrored from Illuminatorama.metal. Keep in lockstep.
struct DDGIPointEmitter {
    float3 position;
    float  radius;
    float3 color;
    float  _pad;
};

// DUPLICATED from IlluminatoramaSurfaceCache.metal — keep in lockstep. Maps a
// triangle hit (prim + barycentric) to its card-UV and samples the radiance
// atlas tile (UV inset half a texel so bilinear can't bleed across tiles).
// Field-for-field mirror of `SurfCard` in IlluminatoramaSurfaceCache.metal (Metal
// has no cross-file linkage; keep in lockstep). Only albedo/emission/normal.w are
// read here, but the full layout is needed so `cards[card]` strides correctly.
struct SurfCard {
    float4 origin; float4 uAxis; float4 vAxis; float4 normal;
    float4 albedo; float4 emission;
    float4 originB; float4 uAxisB; float4 vAxisB; float4 normalB;
    float4 albedoB; float4 emissionB;
};

// ── Curve primitives (#60 item 7, increment 2) ───────────────────────────────
// `kRTCurvesEnabled` specializes this SOUP kernel for a primitive AS that holds a
// curve geometry descriptor (round Catmull-Rom wood) alongside the triangle soup.
// Same function-constant index (30) as the TLAS kernels in
// IlluminatoramaRTInstanced.metal. The base variant (undefined → false) keeps the
// original triangle-only intersector contract — curve-free soup scenes run the
// exact code they always did. The soup case is single-set + identity-placed
// (the Forest soup is authored in world space), so the helpers below carry the
// same struct/math as the TLAS file (kept in lockstep) and the matrix multiply is
// just identity — no per-set transform indirection.
constant bool kRTCurvesEnabledFC [[function_constant(30)]];
constant bool kRTCurvesEnabled = is_function_constant_defined(kRTCurvesEnabledFC) && kRTCurvesEnabledFC;

// Mirror of the Swift `RTCurveSetData` (112 B, SIMD4-aligned). m0..m3 = the set's
// object→world matrix columns (identity for the world-space soup set); meta.x =
// the set's first segment index in the pooled segment buffer. Keep in lockstep
// with IlluminatoramaRTInstanced.metal + the Swift mirror.
struct RTCurveSetData {
    float4 m0; float4 m1; float4 m2; float4 m3;
    float4 albedoRoughness;   // xyz albedo, w roughness
    float4 emissionPad;       // xyz emission
    uint4  meta;              // x = segment base
};

// Catmull-Rom point at t for one curve segment (Metal's RT convention: the
// segment spans P1..P2; P0/P3 steer the tangents — uniform CR, tension 0.5).
static inline float3 crPointRT(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * t
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

// World normal of a round-curve hit: the radial direction from the curve axis at
// the hit parameter to the hit point. Single soup set ⇒ setIdx is always 0.
static inline float3 curveHitNormalRT(uint prim, float t,
                                      float3 hitP,
                                      const device RTCurveSetData* curveSets,
                                      const device packed_float3* curvePts,
                                      const device uint* curveSegs) {
    RTCurveSetData cs = curveSets[0];
    uint i0 = curveSegs[cs.meta.x + prim];
    float3 p0 = float3(curvePts[i0 + 0u]), p1 = float3(curvePts[i0 + 1u]);
    float3 p2 = float3(curvePts[i0 + 2u]), p3 = float3(curvePts[i0 + 3u]);
    float3 axisObj = crPointRT(p0, p1, p2, p3, t);
    float4x4 M = float4x4(cs.m0, cs.m1, cs.m2, cs.m3);
    float3 axisW = (M * float4(axisObj, 1.0)).xyz;
    float3 n = hitP - axisW;
    float nl = length(n);
    return nl > 1e-8 ? n / nl : float3(0.0, 1.0, 0.0);
}

// The atlas stores albedo-free IRRADIANCE (#60 task 2). Reconstruct the hit's
// outgoing radiance L_out = albedo·irr + emission from its card, so a card seam
// isn't amplified by the albedo multiply. Returns the same outgoing radiance the
// callers always consumed — no caller math changes.
static inline float3 sampleSurfCacheRT(
    texture2d<float, access::sample> atlas, uint prim, float2 bary,
    const device SurfCard* cards,
    const device uint* triCard, const device float4* triUVa,
    const device float4* triUVc,
    const device float4* cardRect, uint atlasW, uint atlasH)
{
    uint card = triCard[prim];
    float4 rect = cardRect[card];          // (x, y, w, h) in atlas px
    float4 a = triUVa[prim]; float4 c = triUVc[prim];
    float2 uvA = a.xy, uvB = a.zw, uvC = c.xy;
    float w0 = 1.0 - bary.x - bary.y;
    float2 uv = saturate(w0 * uvA + bary.x * uvB + bary.y * uvC);
    float2 inset = 0.5 / max(float2(1.0), rect.zw);
    uv = clamp(uv, inset, 1.0 - inset);
    float2 px = rect.xy + uv * rect.zw;
    constexpr sampler samp(filter::linear, address::clamp_to_edge);
    float3 irr = atlas.sample(samp, px / float2(atlasW, atlasH)).rgb;
    SurfCard sc = cards[card];
    bool useB = sc.normal.w > 0.5 && (a.x + a.y > 1.0);
    float3 albedo   = useB ? sc.albedoB.xyz   : sc.albedo.xyz;
    float3 emission = useB ? sc.emissionB.xyz : sc.emission.xyz;
    return albedo * irr + emission;
}

// ── helpers ──────────────────────────────────────────────────────────────────

static inline float3 octDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return normalize(n);
}

static inline float3 worldPosFromDepth(float2 ndcXY, float depth, float4x4 invVP) {
    float4 clip = float4(ndcXY, depth, 1.0);
    float4 world = invVP * clip;
    return world.xyz / world.w;
}

// PCG hash → [0,1)
static inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline float rnd(thread uint& seed) {
    seed = pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

// Build an orthonormal basis around n.
static inline void basis(float3 n, thread float3& t, thread float3& b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float d = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere sample about n.
static inline float3 cosineSample(float3 n, float u1, float u2) {
    float r = sqrt(u1);
    float phi = 2.0 * M_PI_F * u2;
    float3 t, b;
    basis(n, t, b);
    float x = r * cos(phi), y = r * sin(phi), z = sqrt(max(0.0, 1.0 - u1));
    return normalize(t * x + b * y + n * z);
}

// A direction within a cone of half-angle `theta` about `dir`.
static inline float3 coneSample(float3 dir, float theta, float u1, float u2) {
    float cosT = mix(cos(theta), 1.0, u1);
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 2.0 * M_PI_F * u2;
    float3 t, b;
    basis(dir, t, b);
    return normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + dir * cosT);
}

static inline float2 dirToEquirectUV(float3 d) {
    float u = atan2(d.z, d.x) * (1.0 / (2.0 * M_PI_F)) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / M_PI_F);
    return float2(u, v);
}

// ── kernel ───────────────────────────────────────────────────────────────────

kernel void illumi_rt_lighting(
    texture2d<float, access::read>        gDepth      [[texture(0)]],
    texture2d<half,  access::read>        gNormalRgh  [[texture(1)]],
    texture2d<half,  access::read>        gAlbedoMet  [[texture(2)]],
    texture2d<half,  access::read_write>  outHDR      [[texture(3)]],
    texture2d<float, access::sample>      skyEquirect [[texture(4)]],
    texture2d<float, access::sample>      surfAtlas   [[texture(5)]],
    texture2d<half,  access::write>       rtDiffuse   [[texture(6)]],
    primitive_acceleration_structure      accel       [[buffer(0)]],
    const device float4*                  triAlbedo   [[buffer(1)]],
    const device float4*                  triNormal   [[buffer(2)]],
    constant RTUniforms&                  u           [[buffer(3)]],
    const device uint*                    triCard     [[buffer(4)]],
    const device float4*                  triUVa      [[buffer(5)]],
    const device float4*                  triUVc      [[buffer(6)]],
    const device DDGIPointEmitter*        emitters    [[buffer(7)]],
    const device float4*                  surfCardRect [[buffer(8)]],   // surface-cache per-card atlas rect
    const device SurfCard*                surfCards    [[buffer(9)]],   // per-card material (albedo/emission) for L_out reconstruction
    // Curve primitives (#60 item 7) — dummies bound for the base variant
    // (kRTCurvesEnabled false ⇒ never read). curveRadii is bound for symmetry
    // with the BLAS-build / TLAS path; the shading path reads only the axis.
    const device RTCurveSetData*          curveSets   [[buffer(10)]],
    const device packed_float3*           curvePts    [[buffer(11)]],
    const device float*                   curveRadii  [[buffer(12)]],
    const device uint*                    curveSegs   [[buffer(13)]],
    // Curve primitives live in their OWN primitive AS (#60 item 7 incr. 2 —
    // Metal forbids mixed-type geometry in one AS). When kRTCurvesEnabled, this
    // is traced alongside `accel` (any-hit occludes; nearest-hit compares
    // distance). Bound to `accel` as a harmless dummy for the base variant.
    primitive_acceleration_structure      curveAccel  [[buffer(14)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;

    float depth = gDepth.read(gid).r;
    if (depth >= 0.99999) return;   // sky pixel — leave the composite alone

    float2 ndc = (float2(gid) + 0.5) / float2(u.width, u.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 P = worldPosFromDepth(ndc, depth, u.invViewProjection);
    half4 nrH = gNormalRgh.read(gid);
    half4 amH = gAlbedoMet.read(gid);
    float3 N = octDecode(float2(nrH.rg));
    float roughness = max(0.045, float(nrH.b));
    float3 albedo = float3(amH.rgb);
    float metalness = float(amH.a);

    float3 Pofs = P + N * max(u.rayTMin, 1e-3);

    constexpr sampler skySamp(filter::linear, address::repeat);

    // The curve_data tag is compile-time; the base variant keeps the original
    // triangle-only traversal contract, the curve variant widens it to match a
    // primitive AS that holds a curve geometry descriptor (#60 item 7). NO
    // `instancing` tag — this is a primitive AS, not a TLAS.
    intersector<triangle_data, curve_data> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    if (kRTCurvesEnabled) {
        isect.assume_geometry_type(geometry_type::triangle | geometry_type::curve);
        isect.assume_curve_basis(curve_basis::catmull_rom);
        isect.assume_curve_type(curve_type::round);
        isect.assume_curve_control_point_count(4);
    }

    uint seed = pcgHash(gid.x + gid.y * u.width + u.frameSeed * 9781u);

    // ── Direct sun (soft shadows) ─────────────────────────────────
    float3 Ld = normalize(u.sunDir);
    float NdotL = saturate(dot(N, Ld));
    float3 direct = float3(0.0);
    if (NdotL > 0.0 && u.shadowRays > 0) {
        isect.accept_any_intersection(true);   // occlusion-only
        uint hits = 0;
        for (uint s = 0; s < u.shadowRays; ++s) {
            float3 L = coneSample(Ld, u.sunSoftnessRad, rnd(seed), rnd(seed));
            ray r;
            r.origin = Pofs;
            r.direction = L;
            r.min_distance = max(u.rayTMin, 1e-3);
            r.max_distance = 1e4;
            bool occ = isect.intersect(r, accel).type != intersection_type::none;
            if (!occ && kRTCurvesEnabled) occ = isect.intersect(r, curveAccel).type != intersection_type::none;
            if (occ) hits++;   // any occluder (triangle soup OR curve AS)
        }
        float vis = 1.0 - float(hits) / float(u.shadowRays);
        // Lambert diffuse + a soft Blinn spec for the floor sheen.
        float3 V = normalize(u.cameraWorldPos - P);
        float3 H = normalize(Ld + V);
        float spec = pow(saturate(dot(N, H)), mix(8.0, 90.0, 1.0 - roughness));
        spec *= (1.0 - roughness) * u.specStrength;
        direct = u.sunColor * NdotL * vis * (albedo * (1.0 / M_PI_F) + spec);
    }

    // ── Leaf thin-sheet transmission (issue #58 / #20 item 2) ─────
    // Foliage is flagged in normalRoughness.w (0 = leaf). Under RT this pass
    // owns the sun, so the back-light term lives here too — and crucially it
    // runs for BACK-LIT leaves, where NdotL == 0 and the direct block above
    // never executes. Only back-lit foliage (sun on the far face) enters, so
    // the extra shadow rays are bounded to that subset. The transmitted light
    // arrives through the leaf from the sun side, so the occlusion test is cast
    // from a point nudged toward the sun (`P + Ld·ε`) — offsetting along +N as
    // the front-face test does would self-shadow the leaf's own triangle.
    if (nrH.a < 0.5h && u.leafTransmission > 0.0 && u.shadowRays > 0) {
        float3 V = normalize(u.cameraWorldPos - P);
        float back = saturate(dot(-N, Ld));              // sun on the FAR face
        if (back > 0.0) {
            float forward = pow(saturate(dot(V, -Ld)), 3.0); // viewing toward the sun
            // ROUND-6 POLISH #3: BACKLIT-ONLY GATE (mirror of Illuminatorama.metal).
            // The 0.40 floor used to fire on forward-lit side-view leaves (sun on the
            // camera's side, V·(−Ld)≈0), blooming them pale. Gate on genuine backlight
            // — smoothstep(dot(V,−Ld)) — so forward-lit leaves roll to ~0 while the
            // three-quarter backlit rim keeps the full transmitted term.
            float backlit = smoothstep(0.02, 0.42, dot(V, -Ld));
            float through = back * backlit * (0.30 + 0.70 * forward);
            isect.accept_any_intersection(true);         // occlusion-only
            float3 PofsT = P + Ld * 3e-3;
            uint th = 0;
            for (uint s = 0; s < u.shadowRays; ++s) {
                float3 L = coneSample(Ld, u.sunSoftnessRad, rnd(seed), rnd(seed));
                ray r;
                r.origin = PofsT;
                r.direction = L;
                r.min_distance = 1e-3;
                r.max_distance = 1e4;
                bool occ = isect.intersect(r, accel).type != intersection_type::none;
                if (!occ && kRTCurvesEnabled) occ = isect.intersect(r, curveAccel).type != intersection_type::none;
                if (occ) th++;   // any occluder (triangle soup OR curve AS)
            }
            float tvis = 1.0 - float(th) / float(u.shadowRays);
            // ENERGY CLAMP: a leaf transmits a FRACTION of the light reaching it —
            // it cannot re-emit more than it receives. `leafTransmission` was a raw
            // (>1) gain with no ceiling, so under Forest's low warm BACKLIGHT the
            // fill cards blew past 1.0 and clipped to white while the bark (no
            // transmission) stayed dark — the connected canopy READ as bare sticks.
            // Clamp the transmitted radiance to the incident (`albedo·sun·tvis`) so
            // backlit leaves still glow but never out-shine the light source.
            float3 t = albedo * u.sunColor * through * tvis * u.leafTransmission;
            direct += min(t, albedo * u.sunColor * tvis);
        }
    }

    // ── One-bounce indirect (GI) ──────────────────────────────────
    float3 indirect = float3(0.0);
    if (u.giRays > 0 && u.giStrength > 0.0) {
        isect.accept_any_intersection(false);   // need nearest hit + ids
        for (uint g = 0; g < u.giRays; ++g) {
            float3 dir = cosineSample(N, rnd(seed), rnd(seed));
            ray r;
            r.origin = Pofs;
            r.direction = dir;
            r.min_distance = max(u.rayTMin, 1e-3);
            r.max_distance = u.maxGIDist;
            auto res = isect.intersect(r, accel);   // triangle soup AS
            float triDist = (res.type == intersection_type::triangle) ? res.distance : 3.0e38;
            // Curves are a SEPARATE primitive AS (#60 item 7 incr. 2); trace it too
            // and take the nearer hit. A curve hit carries no surface-cache card →
            // re-shade with the set's material (sun + emission).
            if (kRTCurvesEnabled) {
                auto cres = isect.intersect(r, curveAccel);
                if (cres.type == intersection_type::curve && cres.distance < triDist) {
                    float3 hitP = r.origin + dir * cres.distance;
                    float3 hitN = curveHitNormalRT(cres.primitive_id, cres.curve_parameter,
                                                   hitP, curveSets, curvePts, curveSegs);
                    RTCurveSetData cs = curveSets[0];
                    float3 hitRad = cs.emissionPad.xyz;
                    float hN = saturate(dot(hitN, Ld));
                    if (hN > 0.0) {
                        isect.accept_any_intersection(true);
                        ray sr; sr.origin = hitP + hitN * 2e-3; sr.direction = Ld;
                        sr.min_distance = 2e-3; sr.max_distance = 1e4;
                        bool occ = isect.intersect(sr, accel).type != intersection_type::none;
                        if (!occ) occ = isect.intersect(sr, curveAccel).type != intersection_type::none;
                        hitRad += cs.albedoRoughness.xyz * (1.0 / M_PI_F) * u.sunColor * hN * (occ ? 0.0 : 1.0);
                        isect.accept_any_intersection(false);
                    }
                    indirect += hitRad;
                    continue;
                }
            }
            if (res.type == intersection_type::triangle) {
                uint prim = res.primitive_id;
                if (prim >= u.triangleCount) continue;
                if (u.surfCacheEnabled != 0) {
                    // Cached path: reconstruct the hit's full outgoing radiance
                    // (albedo·irradiance + emission, MULTI-bounce, accumulated over
                    // frames) from the cache — one atlas read + a card lookup, no
                    // shadow ray. Both cheaper and richer than the sun-only
                    // re-shade below.
                    // Phase 5 / A0 — residency: a non-resident card (budget streaming)
                    // has a zero atlas rect; resident ⇒ cache read; non-resident ⇒
                    // emission + ambient fallback (never black). Soup-path twin of the
                    // TLAS residency gate.
                    uint card = triCard[prim];
                    if (surfCardRect[card].z > 0.0) {
                        indirect += sampleSurfCacheRT(surfAtlas, prim,
                            res.triangle_barycentric_coord, surfCards, triCard, triUVa, triUVc,
                            surfCardRect, u.surfAtlasW, u.surfAtlasH);
                    } else {
                        SurfCard sc = surfCards[card];
                        indirect += sc.emission.xyz + sc.albedo.xyz * u.skyAmbient;
                    }
                    continue;
                }
                float3 hitP = r.origin + dir * res.distance;
                float3 hitN = normalize(triNormal[prim].xyz);
                float3 hitA = triAlbedo[prim].xyz;
                // Direct sun at the hit (one shadow ray; curves occlude too).
                float hN = saturate(dot(hitN, Ld));
                float3 hitRad = float3(0.0);
                if (hN > 0.0) {
                    isect.accept_any_intersection(true);
                    ray sr;
                    sr.origin = hitP + hitN * 2e-3;
                    sr.direction = Ld;
                    sr.min_distance = 2e-3;
                    sr.max_distance = 1e4;
                    bool occ = isect.intersect(sr, accel).type != intersection_type::none;
                    if (kRTCurvesEnabled && !occ) occ = isect.intersect(sr, curveAccel).type != intersection_type::none;
                    float sv = occ ? 0.0 : 1.0;
                    hitRad = hitA * (1.0 / M_PI_F) * u.sunColor * hN * sv;
                    isect.accept_any_intersection(false);
                }
                // Emitter contribution at the GI bounce hit.
                for (uint e = 0; e < u.emitterCount; ++e) {
                    float3 toE  = emitters[e].position - hitP;
                    float  d    = length(toE);
                    float  rr   = max(emitters[e].radius, 0.001f);
                    float  atten = max(0.0f, 1.0f - d / rr);
                    atten       *= atten;
                    if (atten < 0.001f) continue;
                    float NdotE = max(0.0f, dot(hitN, toE / max(d, 0.001f)));
                    hitRad += emitters[e].color * atten * NdotE * hitA * (1.0f / M_PI_F);
                }
                indirect += hitRad;
            } else {
                // Ray escaped the room (through the window) → sky.
                float3 sky = skyEquirect.sample(skySamp, dirToEquirectUV(dir)).rgb;
                indirect += sky;
            }
        }
        indirect = (indirect / float(u.giRays)) * albedo * u.giStrength;
    }

    // ── Glossy reflections (RT) ───────────────────────────────────
    // Traces the mirror/glossy direction against the SAME static AS the GI
    // pass uses. Only GEOMETRY hits are added: a hit reflects scene content
    // the sky-cube IBL fundamentally cannot contain (the room's own walls /
    // floor / props), with correct OFF-screen support that SSR lacks. On a
    // MISS the ray would see the sky — but the deferred IBL specular already
    // reflects the sky cube, so we add nothing there to avoid double-counting
    // the environment. Host is expected to zero `ssrIntensity` when this is
    // on, so RT owns scene-geometry reflections (no SSR double-count).
    float3 reflection = float3(0.0);
    if (u.reflEnabled != 0 && u.reflStrength > 0.0 &&
        roughness <= u.reflRoughnessCutoff) {
        float3 V = normalize(u.cameraWorldPos - P);
        float NdotV = saturate(dot(N, V));
        float3 R = reflect(-V, N);
        // Glossy cone widens with roughness; mirror surfaces stay tight.
        float coneTheta = roughness * roughness * 1.2;
        // Schlick Fresnel with metal-aware F0.
        float3 F0 = mix(float3(0.04), albedo, metalness);
        float3 fres = F0 + (float3(1.0) - F0) * pow(1.0 - NdotV, 5.0);

        uint rrays = max(1u, u.reflRays);
        float3 acc = float3(0.0);
        isect.accept_any_intersection(false);
        for (uint i = 0; i < rrays; ++i) {
            float3 dir = (coneTheta > 1e-4)
                ? coneSample(R, coneTheta, rnd(seed), rnd(seed)) : R;
            if (dot(dir, N) <= 0.0) continue;   // reflected below the surface
            ray r;
            r.origin = Pofs;
            r.direction = dir;
            r.min_distance = max(u.rayTMin, 1e-3);
            r.max_distance = u.reflMaxDist;
            auto res = isect.intersect(r, accel);   // triangle soup AS
            float triDist = (res.type == intersection_type::triangle) ? res.distance : 3.0e38;
            // Curve reflection hit (separate AS) — take it when nearer than the
            // triangle hit. Ambient fill + sun + the set's emission.
            if (kRTCurvesEnabled) {
                auto cres = isect.intersect(r, curveAccel);
                if (cres.type == intersection_type::curve && cres.distance < triDist) {
                    float3 hitP = r.origin + dir * cres.distance;
                    float3 hitN = curveHitNormalRT(cres.primitive_id, cres.curve_parameter,
                                                   hitP, curveSets, curvePts, curveSegs);
                    RTCurveSetData cs = curveSets[0];
                    float3 cA = cs.albedoRoughness.xyz;
                    float3 hitRad = cs.emissionPad.xyz + cA * u.skyAmbient;
                    float hN = saturate(dot(hitN, Ld));
                    if (hN > 0.0) {
                        isect.accept_any_intersection(true);
                        ray sr; sr.origin = hitP + hitN * 2e-3; sr.direction = Ld;
                        sr.min_distance = 2e-3; sr.max_distance = 1e4;
                        bool occ = isect.intersect(sr, accel).type != intersection_type::none;
                        if (!occ) occ = isect.intersect(sr, curveAccel).type != intersection_type::none;
                        isect.accept_any_intersection(false);
                        hitRad += cA * (1.0 / M_PI_F) * u.sunColor * hN * (occ ? 0.0 : 1.0);
                    }
                    acc += hitRad;
                    continue;
                }
            }
            if (res.type != intersection_type::triangle) continue;  // sky → IBL
            uint prim = res.primitive_id;
            if (prim >= u.triangleCount) continue;
            if (u.surfCacheEnabled != 0) {
                // Cached path: reflect the surface's full cached radiance
                // (multi-bounce), one atlas read instead of a re-shade.
                // Phase 5 / A0 — residency gate (same as the GI path above).
                uint card = triCard[prim];
                if (surfCardRect[card].z > 0.0) {
                    acc += sampleSurfCacheRT(surfAtlas, prim,
                        res.triangle_barycentric_coord, surfCards, triCard, triUVa, triUVc,
                            surfCardRect, u.surfAtlasW, u.surfAtlasH);
                } else {
                    SurfCard sc = surfCards[card];
                    acc += sc.emission.xyz + sc.albedo.xyz * u.skyAmbient;
                }
                continue;
            }
            float3 hitP = r.origin + dir * res.distance;
            float3 hitN = normalize(triNormal[prim].xyz);
            float3 hitA = triAlbedo[prim].xyz;
            // Outgoing radiance of the reflected surface = sky-ambient fill +
            // sun-direct (one shadow ray). The ambient term is what keeps RT
            // reflections from reading darker than reality: every surface in
            // the deferred pass also gets this fill, so a reflection that drops
            // it looks dim. Sun term is added only when the hit faces the sun.
            float3 hitRad = hitA * u.skyAmbient;
            float hN = saturate(dot(hitN, Ld));
            if (hN > 0.0) {
                isect.accept_any_intersection(true);
                ray sr;
                sr.origin = hitP + hitN * 2e-3;
                sr.direction = Ld;
                sr.min_distance = 2e-3;
                sr.max_distance = 1e4;
                bool occ = isect.intersect(sr, accel).type != intersection_type::none;
                if (kRTCurvesEnabled && !occ) occ = isect.intersect(sr, curveAccel).type != intersection_type::none;
                float sv = occ ? 0.0 : 1.0;   // curve or triangle occludes
                isect.accept_any_intersection(false);
                hitRad += hitA * (1.0 / M_PI_F) * u.sunColor * hN * sv;
            }
            acc += hitRad;
        }
        reflection = (acc / float(rrays)) * fres * u.reflStrength;
    }

    // Split the RT contribution by frequency:
    //   • Reflection/specular is sharp (mirror-ish, ~1 spp) and varies across a
    //     flat surface that shares depth+normal — a depth+normal bilateral would
    //     smear it. Composite it straight into the HDR target.
    //   • Direct (soft shadow) + indirect (1-bounce GI) is the low-frequency
    //     Monte-Carlo grain the user sees. Write it to a dedicated buffer for
    //     `illumi_rt_denoise` to bilateral-filter before it reaches the composite
    //     (and TAA). Sky pixels early-out above, so `rtDiffuse` is left untouched
    //     there — the denoise pass guards on the same depth test.
    half4 prev = outHDR.read(gid);
    outHDR.write(half4(prev.rgb + half3(reflection), prev.a), gid);
    rtDiffuse.write(half4(half3(direct + indirect), 1.0h), gid);
}

// ── Curve sway displacement (#60 item 7, increment 2) ────────────────────────
// One thread per curve control point. Reads the REST (un-swayed) control points
// + a per-point wind attribute, applies the SAME wind displacement the G-buffer /
// shadow vertex shaders apply to the rastered wood (so RT curve shadows track the
// swaying tubes), and writes the displaced points into the AS-referenced control-
// point buffer. The host then refits the curve geometry in the primitive AS on
// the frame command buffer.
//
// `applyCurveWind` is a VERBATIM transcription of `applyTreeWind` in
// Illuminatorama.metal (Metal has no cross-file linkage). **KEEP IN LOCKSTEP** —
// any divergence detaches the RT shadow from the rastered tube. windAttr =
// (swayWeight, treePhase, flutter, woodMark) — same encoding as the vertex tangent.
struct CurveWindUniforms {
    float time;
    float windStrength;
    float windHeading;
    uint  pointCount;
};

static inline float3 applyCurveWind(float3 wp, float4 windAttr, float time,
                                    float windStrength, float windHeading) {
    float sway = windAttr.x;
    if (sway < 0.0001 || windStrength <= 0.0) return wp;
    float phase   = windAttr.y;
    float flutter = windAttr.z;
    float2 wdir = float2(cos(windHeading), sin(windHeading));
    float gust  = 0.55 + 0.45 * sin(time * 0.43 + dot(wp.xz, wdir) * 0.07);
    float macro = sin(time * 0.9 + phase);
    float bend  = sway * windStrength * (0.5 + gust) * macro;
    wp.xz += wdir * bend;
    wp.y  -= sway * windStrength * 0.18 * fabs(macro);
    if (flutter > 0.0001) {
        wp.x += flutter * windStrength * 0.05 * sin(time * 6.5 + phase * 4.0 + wp.y * 1.1);
        wp.z += flutter * windStrength * 0.05 * sin(time * 5.7 + phase * 3.0 + wp.x * 1.1);
        wp.y += flutter * windStrength * 0.035 * sin(time * 6.0 + phase * 3.5);
    }
    return wp;
}

kernel void illumi_curve_wind_displace(
    const device packed_float3*  restPts   [[buffer(0)]],   // un-swayed control points
    const device float4*         windAttr  [[buffer(1)]],   // per-point (sway,phase,flutter,woodMark)
    device packed_float3*        outPts     [[buffer(2)]],   // AS-referenced control points (written)
    constant CurveWindUniforms&  u          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.pointCount) return;
    float3 wp = float3(restPts[gid]);
    wp = applyCurveWind(wp, windAttr[gid], u.time, u.windStrength, u.windHeading);
    outPts[gid] = packed_float3(wp);
}
