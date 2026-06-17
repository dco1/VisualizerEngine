#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// ── ILLUMINATORAMA GLASS CAUSTICS (#60) ──────────────────────────────────────
//
// Photon-traced caustics for the AAA RT glass, reusable on ANY receiver (not a
// flat plane like the legacy water `CausticsRT`). The receiver store is the
// surface-cache CARD atlas: every opaque triangle already owns a card + atlas
// rect (the surface cache builds them), so a focused photon lands at a unique
// texel of whatever geometry it hits — floor, wall, another object. Requires the
// surface cache to be on (that's where the card layout comes from).
//
// Three kernels:
//   1. `illumi_caustic_decay`   — temporal decay of the caustic atlas (EMA, so
//      the accumulation converges over frames instead of flickering).
//   2. `illumi_caustic_photons` — one thread per photon: emit from the sun toward
//      the glass AABB, refract through the glass (entry+exit, Beer–Lambert), trace
//      the exit ray to an opaque receiver, resolve its card+texel, atomic-add the
//      focused energy. (Mask 0x02 = glass; 0x01 = opaque receivers.)
//   3. `illumi_caustic_primary` — per primary opaque pixel, trace one camera ray
//      to find the pixel's OWN triangle (the surface cache is read only by
//      secondary rays, so the directly-viewed receiver can't read its own card —
//      this pass supplies that read), resolve its card+texel, and add the caustic
//      (× albedo) to the lit HDR composite.
//
// The atlas is `device atomic_float*`, 3 floats (RGB) per texel, index
// `(py*atlasW + px)*3 + c`.

struct RTGlassData { float4 tintIor; float4 rdrf; float4 dispersionPad; };

struct RTInstanceData {
    float4 nrm0; float4 nrm1; float4 nrm2;
    float4 albedoTriBase;
};

struct PrimUV { float2 uvA; float2 uvB; float2 uvC; };

struct CausticUniforms {
    float4x4 invViewProjection;
    float3 cameraWorldPos;  float rayTMin;
    float3 sunDir;          float photonEnergy;   // toward-sun unit dir; per-photon flux scale
    float3 sunColor;        float decay;          // EMA decay per frame
    float3 aabbMin;         float strength;       // glass AABB (world) + composite strength
    float3 aabbMax;         float _pad0;
    uint   width;  uint height;                   // primary pass target size
    uint   glassInstanceBase;
    uint   photonCount;
    uint   surfTriCount;  uint surfAtlasW;  uint surfAtlasH;  uint frameSeed;
    uint   maxGlassBounces; uint glassDiscCount; uint _pad2; uint _pad3;
};

// ── helpers (duplicated per Metal file convention) ───────────────────────────
static inline uint c_pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
static inline float c_rnd(thread uint& seed) { seed = c_pcg(seed); return float(seed) * (1.0 / 4294967296.0); }
static inline void c_basis(float3 n, thread float3& t, thread float3& b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0; float a = -1.0 / (s + n.z); float d = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * d, -s * n.x);
    b = float3(d, s + n.y * n.y * a, -n.y);
}

static inline float3 c_hitNormal(uint iid, uint prim,
                                 const device RTInstanceData* insts,
                                 const device float4* objNormal) {
    RTInstanceData d = insts[iid];
    uint base = uint(d.albedoTriBase.w);
    float3 n = float3x3(d.nrm0.xyz, d.nrm1.xyz, d.nrm2.xyz) * objNormal[base + prim].xyz;
    float l = length(n);
    return l > 1e-8 ? n / l : float3(0, 1, 0);
}

// Resolve a receiver triangle hit → its caustic-atlas texel (integer px). Mirror
// of the surface-cache card/UV addressing. Returns false if the hit has no card.
static inline bool c_resolveTexel(uint iid, uint prim, float2 bary,
                                  constant CausticUniforms& u,
                                  const device uint* soupTriBase,
                                  const device uint* triCard,
                                  const device float4* triUVa,
                                  const device float4* triUVc,
                                  const device float4* cardRect,
                                  thread uint2& outPx, thread float4& outRect)
{
    uint gp = soupTriBase[iid] + prim;
    if (gp >= u.surfTriCount) return false;
    uint card = triCard[gp];
    float4 rect = cardRect[card];               // (x,y,w,h) px
    if (rect.z <= 0.0) return false;            // non-resident card
    float4 a = triUVa[gp], c = triUVc[gp];
    float2 uvA = a.xy, uvB = a.zw, uvC = c.xy;
    float w0 = 1.0 - bary.x - bary.y;
    float2 uv = saturate(w0 * uvA + bary.x * uvB + bary.y * uvC);
    float2 inset = 0.5 / max(float2(1.0), rect.zw);
    uv = clamp(uv, inset, 1.0 - inset);
    float2 px = rect.xy + uv * rect.zw;
    outPx = uint2(clamp(px, float2(0.0), float2(u.surfAtlasW - 1, u.surfAtlasH - 1)));
    outRect = rect;
    return true;
}

// Read the caustic atlas at `px` with an NxN box blur CLAMPED to the hit's card
// rect — denoises the photon speckle without bleeding across surface boundaries
// (each card is a contiguous atlas region for one surface). `radius` in texels.
static inline float3 c_readBlurred(const device float* atlas, uint atlasW,
                                   uint2 px, float4 rect, int radius)
{
    int x0 = int(rect.x), y0 = int(rect.y);
    int x1 = x0 + int(rect.z) - 1, y1 = y0 + int(rect.w) - 1;
    float3 sum = float3(0.0); float wsum = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = clamp(int(px.x) + dx, x0, x1);
            int sy = clamp(int(px.y) + dy, y0, y1);
            uint idx = (uint(sy) * atlasW + uint(sx)) * 3u;
            sum += float3(atlas[idx + 0u], atlas[idx + 1u], atlas[idx + 2u]);
            wsum += 1.0;
        }
    }
    return wsum > 0.0 ? sum / wsum : float3(0.0);
}

static inline void c_splat(device atomic_float* atlas, uint2 px, uint atlasW, float3 e) {
    uint idx = (px.y * atlasW + px.x) * 3u;
    atomic_fetch_add_explicit(&atlas[idx + 0u], e.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&atlas[idx + 1u], e.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&atlas[idx + 2u], e.z, memory_order_relaxed);
}

// ── Kernel 1 — temporal decay ────────────────────────────────────────────────
kernel void illumi_caustic_decay(
    device atomic_float*       atlas [[buffer(0)]],
    constant CausticUniforms&  u     [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = u.surfAtlasW * u.surfAtlasH * 3u;
    if (gid >= n) return;
    float v = atomic_load_explicit(&atlas[gid], memory_order_relaxed);
    atomic_store_explicit(&atlas[gid], v * u.decay, memory_order_relaxed);
}

// ── Kernel 2 — emit + refract + trace + splat ────────────────────────────────
kernel void illumi_caustic_photons(
    device atomic_float*           atlas       [[buffer(0)]],
    constant CausticUniforms&      u           [[buffer(1)]],
    instance_acceleration_structure accel      [[buffer(2)]],
    const device RTInstanceData*   insts       [[buffer(3)]],
    const device float4*           objNormal   [[buffer(4)]],
    const device RTGlassData*      glassData   [[buffer(5)]],
    const device uint*             soupTriBase [[buffer(6)]],
    const device uint*             triCard     [[buffer(7)]],
    const device float4*           triUVa      [[buffer(8)]],
    const device float4*           triUVc      [[buffer(9)]],
    const device float4*           cardRect    [[buffer(10)]],
    const device float4*           glassDiscs  [[buffer(11)]],  // xyz centre, w radius (per glass instance)
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.photonCount) return;
    uint seed = c_pcg(gid * 9277u + u.frameSeed * 65537u);

    float3 Ld = normalize(u.sunDir);                 // toward the sun
    float eps = max(u.rayTMin, 1e-3);
    // Emission: aim each photon at a specific glass instance's sun-facing disc, so
    // ~every photon hits glass (vs sampling the whole AABB volume, where most miss
    // and waste a full BVH walk). Near-100% hit rate = cheaper + denser caustics.
    float3 origin, dir = -Ld;
    if (u.glassDiscCount > 0u) {
        float4 disc = glassDiscs[gid % u.glassDiscCount];   // xyz centre, w radius
        float3 t, b; c_basis(Ld, t, b);
        float rr = sqrt(c_rnd(seed)) * disc.w * 1.3;        // 1.3× to cover the silhouette
        float ang = 2.0 * M_PI_F * c_rnd(seed);
        float3 onDisc = disc.xyz + (t * cos(ang) + b * sin(ang)) * rr;
        origin = onDisc + Ld * 60.0;
    } else {
        float3 p = mix(u.aabbMin, u.aabbMax, float3(c_rnd(seed), c_rnd(seed), c_rnd(seed)));
        origin = p + Ld * 60.0;
    }

    intersector<triangle_data, instancing> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.accept_any_intersection(false);

    // First glass hit (entry).
    ray r; r.origin = origin; r.direction = dir; r.min_distance = eps; r.max_distance = 1e4;
    auto res = isect.intersect(r, accel, 0x02u);     // glass only
    if (res.type != intersection_type::triangle) return;

    float3 throughput = u.sunColor * u.photonEnergy;
    float3 ro = origin + dir * res.distance;
    float3 rd = dir;
    bool inside = false;
    float curIOR = 1.0;
    float3 tint = float3(1.0); float density = 0.0;

    // Refract through the glass (entry + exit + any nested boundary).
    uint maxB = min(u.maxGlassBounces, 8u);
    bool escaped = false;
    for (uint b = 0u; b < maxB; ++b) {
        uint iid = res.instance_id;
        if (iid < u.glassInstanceBase) break;        // safety
        RTGlassData gd = glassData[iid - u.glassInstanceBase];
        float hitIOR = max(1.0, gd.tintIor.w);
        float3 hn = c_hitNormal(iid, res.primitive_id, insts, objNormal);
        bool exiting = dot(rd, hn) > 0.0;
        float3 n = exiting ? -hn : hn;
        if (inside) throughput *= exp(-density * (1.0 - clamp(tint, 0.0, 1.0)) *
                                      length(ro - r.origin));
        float n1 = curIOR, n2 = exiting ? 1.0 : hitIOR;
        float3 t2 = refract(rd, n, n1 / n2);
        if (dot(t2, t2) < 1e-8) { return; }          // TIR — drop this photon
        rd = t2;
        if (exiting) { inside = false; curIOR = 1.0; tint = float3(1.0); density = 0.0; }
        else         { inside = true;  curIOR = hitIOR; tint = gd.tintIor.xyz; density = gd.rdrf.y; }
        float3 next = ro - n * eps;
        // Trace next segment: another glass boundary, or escape to receivers.
        ray rr; rr.origin = next; rr.direction = normalize(rd);
        rr.min_distance = eps; rr.max_distance = 1e4;
        r = rr;
        res = isect.intersect(rr, accel, 0x03u);     // glass + opaque
        if (res.type != intersection_type::triangle) return;   // escaped to sky
        ro = rr.origin + rr.direction * res.distance;
        if (res.instance_id < u.glassInstanceBase) { escaped = true; break; }  // hit a receiver
    }
    if (!escaped) return;

    // Focused photon hit an opaque receiver → splat into its card texel.
    uint2 px; float4 prect;
    if (!c_resolveTexel(res.instance_id, res.primitive_id, res.triangle_barycentric_coord,
                        u, soupTriBase, triCard, triUVa, triUVc, cardRect, px, prect)) return;
    // Lambert receive: weight by N·L so grazing hits don't over-brighten.
    float3 rn = c_hitNormal(res.instance_id, res.primitive_id, insts, objNormal);
    float ndl = max(0.0, dot(rn, -normalize(rd)));
    c_splat(atlas, px, u.surfAtlasW, throughput * ndl);
}

// ── Kernel 3 — primary-surface caustic composite ─────────────────────────────
kernel void illumi_caustic_primary(
    texture2d<float, access::read>       gDepth     [[texture(0)]],
    texture2d<half,  access::read>       gAlbedoMet [[texture(1)]],
    texture2d<half,  access::read_write> outHDR     [[texture(2)]],
    constant CausticUniforms&            u          [[buffer(0)]],
    const device float*                  causticAtlas [[buffer(1)]],
    instance_acceleration_structure      accel      [[buffer(2)]],
    const device uint*                   soupTriBase [[buffer(3)]],
    const device uint*                   triCard     [[buffer(4)]],
    const device float4*                 triUVa      [[buffer(5)]],
    const device float4*                 triUVc      [[buffer(6)]],
    const device float4*                 cardRect    [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    float depth = gDepth.read(gid).r;
    if (depth >= 0.99999) return;                    // sky

    float2 ndc = (float2(gid) + 0.5) / float2(u.width, u.height) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 wp = u.invViewProjection * float4(ndc, depth, 1.0);
    float3 P = wp.xyz / wp.w;

    // Trace one camera ray to find THIS pixel's own triangle (opaque only).
    intersector<triangle_data, instancing> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    isect.accept_any_intersection(false);
    float3 dir = normalize(P - u.cameraWorldPos);
    ray r; r.origin = u.cameraWorldPos; r.direction = dir;
    r.min_distance = max(u.rayTMin, 1e-3); r.max_distance = 1e4;
    auto res = isect.intersect(r, accel, 0x01u);
    if (res.type != intersection_type::triangle) return;

    uint2 px; float4 rect;
    if (!c_resolveTexel(res.instance_id, res.primitive_id, res.triangle_barycentric_coord,
                        u, soupTriBase, triCard, triUVa, triUVc, cardRect, px, rect)) return;
    // Box-blur the caustic within the card to denoise the photon speckle.
    float3 caustic = c_readBlurred(causticAtlas, u.surfAtlasW, px, rect, 2);
    if (dot(caustic, caustic) < 1e-9) return;

    half4 am = gAlbedoMet.read(gid);
    float3 albedo = float3(am.rgb);
    float3 add = albedo * caustic * u.strength;
    half4 prev = outHDR.read(gid);
    outHDR.write(half4(prev.rgb + half3(add), prev.a), gid);
}
