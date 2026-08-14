#include <metal_stdlib>
#include <metal_raytracing>
// The ONE secondary-ray surface shader (mirror structs, RNG, scatter-cone policy,
// surface-cache read, and `shadeSecondarySurface`), shared with the deferred TLAS
// kernel in IlluminatoramaRTInstanced.metal. Nothing declared there may be
// re-declared here — that is the point.
#include "IlluminatoramaSecondary.h"
using namespace metal;
using namespace raytracing;

// ── ILLUMINATORAMA AAA RAY-TRACED GLASS (#60) ────────────────────────────────
//
// True ray-traced dielectric glass, traced against the SAME instance
// acceleration structure (TLAS) the deferred RT lighting pass uses. This is the
// "Approach B" the old screen-space `illumi_glass_fs` documented as a TODO:
// instead of sampling the sky/background along a single refracted ray, this
// rasterises the glass FRONT surface (so depth-test + occlusion are free) and
// then, per fragment, fires real rays into the scene:
//
//   • REFRACTION — refract at the entry surface (Snell, η = 1/IOR), march the ray
//     through the glass to the BACK surface, refract again on exit (or total-
//     internal-reflect past the critical angle), and keep bouncing through any
//     further dielectric boundaries (lens-through-lens) until the ray escapes to
//     an opaque surface or the sky. Beer–Lambert absorption attenuates the ray
//     over the path length INSIDE glass, so thick/coloured glass tints and
//     darkens correctly.
//   • REFLECTION — reflect at the entry surface and trace that ray against the
//     TLAS too, so glass reflects REAL scene geometry (not just the sky).
//   • FRESNEL — Schlick blend of refraction vs reflection from the per-material
//     IOR (F0 = ((ior−1)/(ior+1))²).
//   • ROUGHNESS — frosted glass jitters the refraction/reflection rays in a cone
//     whose width grows with roughness²; one sample per frame, denoised by the
//     renderer's existing TAA accumulation (the glass pass runs before TAA).
//   • DISPERSION — optional per-wavelength IOR: the refraction path is traced
//     three times (R/G/B) with split IORs, giving prism rainbows. Gated per
//     instance (dispersion == 0 ⇒ single trace, no cost).
//
// Hit shading is surface-cache-when-available, else re-shade: an opaque hit that
// resolves to a resident surface-cache card returns its full multi-bounce
// radiance (one atlas read); otherwise the hit is re-shaded with direct sun
// (shadow ray) + sky ambient — exactly the contract the TLAS reflection path in
// IlluminatoramaRTInstanced.metal uses, so glass and the rest of the RT pipeline
// agree.
//
// Glass instances are appended to the TLAS after the opaque (and curve)
// instances; `u.glassInstanceBase` is the first glass instance_id. Opaque
// instances carry mask bit 0x01, glass carries 0x02; the deferred RT lighting
// pass traces with mask 0x01 (so it never sees glass), while this pass traces
// with 0x03 (opaque + glass) so refraction/reflection see everything. A hit with
// instance_id >= base is glass: its IOR/tint/roughness come from `glassData`.

// ── Shared mirror structs (Metal has no cross-file linkage — keep in lockstep
// with the Swift side and the sibling RT kernels) ────────────────────────────

// Per-instance glass currency for the raster pass (transform + material).
// Field-for-field mirror of Swift `IlluminatoramaGlassInstance` (stride 176).
struct GlassInstance {
    float4x4 modelMatrix;
    float4x4 normalMatrix;
    float4   tintIor;        // xyz = Beer–Lambert tint (1 = clear), w = IOR
    float4   rdrf;           // x = roughness, y = density (absorption/m),
                             // z = reflectivity, w = fresnelPower (unused; reserved)
    float4   dispersionPad;  // x = dispersion (0 = off), yzw reserved
};

// Per-glass-TLAS-instance material the bounce loop reads at a glass hit.
// Mirror of Swift `IlluminatoramaRTGlassData` (stride 48).
struct RTGlassData {
    float4 tintIor;
    float4 rdrf;
    float4 dispersionPad;
};

// `RTInstanceData`, `SurfCard` and `PrimUV` are the shared mirror structs — see
// IlluminatoramaSecondary.h.

struct GlassRTUniforms {
    float3 cameraWorldPos;  float rayTMin;
    float3 sunDir;          float sunSoftnessRad;
    float3 sunColor;        float reflStrength;
    float3 skyAmbient;      float skyIntensity;
    uint   glassInstanceBase;   // first glass instance_id in the TLAS
    uint   maxBounces;          // dielectric bounce cap (entry+exit ≈ 2)
    uint   shadowRays;          // sun shadow rays at re-shaded opaque hits
    uint   frameSeed;
    uint   surfCacheEnabled;
    uint   surfTriCount;        // bound for soupTriBase[iid]+prim
    uint   surfAtlasW;          uint surfAtlasH;
    uint   dispersionEnabled;   // global gate; per-instance value still required
    uint   cheapGlassMode;      // 0 = plain fallback, 1 = synthetic, 2 = screen-space
    float  viewW; float viewH;  // viewport px (mode 2: clipPos.xy → backdrop UV)
    // ── Thin-film iridescence (soap bubbles — Bubble Lab) ────────────────────
    float  time;                // swirl-animation clock (seconds)
    float  thinFilmStrength;    // 0 = OFF (exact no-op for every other scene)
    float  filmThicknessNm;     // base film thickness (nm) at the equator
    float  filmIOR;             // soap-film refractive index (~1.33)
    // ── Oscillation-mode surface undulation (soap bubbles) ───────────────────
    float  wobbleAmp;           // 0 = OFF (no vertex displacement); radial mode amplitude
    float  wobbleFreq;          // global rate multiplier for the beating modes
    // ── Through-glass parity with the deferred pass ──────────────────────────
    // All zero/1 by default ⇒ exactly the previous behaviour.
    uint   pointLightCount;     // local lights at fragment buffers 14 / 15
    uint   spotLightCount;
    uint   interiorMask;        // interior day-light separation (0 = off)
    float  interiorIBLUp;
    float  interiorIBLSide;
    float  interiorAmbient;
    uint   albedoAtlasEnabled;  // 1 ⇒ objUV (buffer 12) + albedoAtlas (texture 4) live
    uint   objUVCount;          // bound of objUV in float2 entries
    // ── Analytic night sky through the glass ─────────────────────────────────
    // Mirrors FrameUniforms' nightSkyParams / nightMoonDir / nightSunDir. The sky
    // dome is baked WITHOUT celestials, so without these a night window renders a
    // flat black pane while the sky beside it is full of stars. All-zero ⇒ exact
    // no-op (see `sampleSky` / `nightCelestials`).
    float4 nightSkyParams;      // x = starBrightness, y = moonIntensity, z = moonAngRadius
    float4 nightMoonDir;        // xyz = unit direction toward the moon
    float4 nightSunDir;         // xyz = unit direction toward the (below-horizon) sun
    float  nightPixAngle;       // angular size of one output pixel (radians)
    float  _nightPad0; float _nightPad1; float _nightPad2;
};

/// This pass's view of the shared night-sky currency — the mirror of
/// `frameNightSky` in Illuminatorama.metal, reading the same packing order.
static inline NightSkyParams glassNightSky(constant GlassRTUniforms& u) {
    NightSkyParams p;
    p.starBrightness = u.nightSkyParams.x;
    p.moonIntensity  = u.nightSkyParams.y;
    p.moonAngRadius  = u.nightSkyParams.z;
    p.moonDir        = u.nightMoonDir.xyz;
    p.toSun          = u.nightSunDir.xyz;
    return p;
}

// The local-light currency (`RTPointLight` / `RTSpotLight`) is shared — see
// IlluminatoramaSecondary.h.

// Mirror of FrameUniforms' leading fields the glass VS needs (viewProjection is
// first). Only viewProjection is read; the rest pad to the right offset is
// unnecessary because we bind the same FrameUniforms buffer and read .x of the
// first matrix. We declare just the matrix we use.
struct GlassFrameUniforms {
    float4x4 viewProjection;
};

// RNG (`pcgHash`/`rnd`), `onb`/`coneSample`, the sky sample and the scatter-cone
// policy (`secondaryConeRad`/`secondaryConeVisible`/`secondaryConeSamples`, with
// `kGlassConeK` as this path's width coefficient) are all shared — see
// IlluminatoramaSecondary.h.

// ── THIN-FILM INTERFERENCE (soap-bubble iridescence) ─────────────────────────
// Reflected colour from a single thin dielectric film over a higher-index
// boundary. The reflected spectrum is modulated by the optical path difference
//   OPD = 2·n·d·cosθt        (θt = refraction angle INSIDE the film)
// with a half-wave (π) shift added for the "hard" reflection off the denser
// film. We sample the interference comb at R/G/B reference wavelengths and build
// an RGB approximation — the same realtime trick Belcour & Barla (2017) and the
// Filament / Standard-Surface thin-film coats use. This is REAL interference
// math, not a scrolled rainbow texture.
static inline float3 thinFilmIridescence(float thicknessNm, float cosI, float filmIOR) {
    float sin2 = (1.0 - cosI * cosI) / (filmIOR * filmIOR);     // Snell
    float cosT = sqrt(max(0.0, 1.0 - sin2));                    // cosθ inside film
    float opd  = 2.0 * filmIOR * thicknessNm * cosT;           // nm
    const float3 lambda = float3(680.0, 550.0, 440.0);         // R, G, B (nm)
    float3 phase = (2.0 * M_PI_F) * (opd / lambda) + M_PI_F;   // + half-wave shift
    float3 c = 0.5 + 0.5 * cos(phase);                         // two-beam intensity
    // Soften toward the pastel magenta-gold-cyan-green of a real DRAINING film:
    // ease the band contrast (no c² over-sharpening → less "oil-slick") and pull
    // ~15% toward luminance so the rings read pastel, not full-primary.
    c = c * (0.6 + 0.4 * c);                                    // gentle contrast
    float lum = dot(c, float3(0.299, 0.587, 0.114));
    return mix(float3(lum), c, 0.85);
}

// Cheap smooth value noise (3D) + 2-octave fbm for the swirling drainage /
// turbulence of the film thickness. Animation comes from advecting the sample
// point by `time` (NOT reseeding), so it stays stable under TAA.
static inline float vnHash3(float3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
static inline float valueNoise3(float3 x) {
    float3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = vnHash3(i + float3(0,0,0)), n100 = vnHash3(i + float3(1,0,0));
    float n010 = vnHash3(i + float3(0,1,0)), n110 = vnHash3(i + float3(1,1,0));
    float n001 = vnHash3(i + float3(0,0,1)), n101 = vnHash3(i + float3(1,0,1));
    float n011 = vnHash3(i + float3(0,1,1)), n111 = vnHash3(i + float3(1,1,1));
    float nx00 = mix(n000, n100, f.x), nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x), nx11 = mix(n011, n111, f.x);
    return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}
static inline float filmFbm(float3 p) {
    return 0.65 * valueNoise3(p) + 0.35 * valueNoise3(p * 2.7 + 11.3);
}

// `hitWorldNormal`, `secondaryLayerBits`, the textured-albedo tap
// (`secondaryAlbedo` + `secondaryAtlasSample`) and `sampleSurfCacheRT` are shared —
// see IlluminatoramaSecondary.h. The albedo tap in particular is the term that
// stopped the world behind a pane reading as flat slabs, and it is now the SAME
// code the deferred reflection path runs.

/// This pass's uniforms → the shared `SecondaryShadeParams`. ONE place where the
/// glass block is translated into the secondary-shading contract; every field of
/// the glass block that participates in hit shading is listed here, so a term
/// added to one side cannot be silently dropped on the other.
static inline SecondaryShadeParams glassSecondaryParams(constant GlassRTUniforms& u) {
    SecondaryShadeParams p;
    p.sunDir = u.sunDir;               p.sunSoftnessRad = u.sunSoftnessRad;
    p.sunColor = u.sunColor;           p.skyIntensity = u.skyIntensity;
    p.skyAmbient = u.skyAmbient;       p.shadowRays = u.shadowRays;
    p.interiorMask = u.interiorMask;
    p.interiorIBLUp = u.interiorIBLUp; p.interiorIBLSide = u.interiorIBLSide;
    p.interiorAmbient = u.interiorAmbient;
    p.albedoAtlasEnabled = u.albedoAtlasEnabled;
    p.objUVCount = u.objUVCount;
    p.pointLightCount = u.pointLightCount;
    p.spotLightCount = u.spotLightCount;
    // C3 — the glass pass keeps the PLAIN opaque mask on purpose. Its rays are
    // camera-visible (a refraction ray is what the eye sees through the pane), and
    // the lighting-only ceiling slabs were excluded from RT precisely because this
    // path refracted into one and re-shaded it as a flat slab. The invisible-
    // occluder bit is for TRANSPORT rays only.
    p.occluderMask = 0x01u;
    return p;
}

// ── Vertex stage ─────────────────────────────────────────────────────────────

// Interleaved vertex the renderer's mesh buffers use (IlluminatoramaVertex,
// stride 112). Only position + normal are read here. Mirror of `Vertex` in
// Illuminatorama.metal.
struct GlassVertex {
    float3 position; float _padPos;
    float3 normal;   float _padNrm;
    float2 uv;       float2 _padUv;
    float4 tangent;
    float4 color;
};

struct GlassVSOut {
    float4 clipPos [[position]];
    float3 worldPos;
    float3 worldNormal;
    uint   instanceID [[flat]];
};

// ── OSCILLATION-MODE SURFACE UNDULATION (wobbling soap bubble / droplet) ──────
// Radial displacement of a unit-sphere surface point `d` (|d|=1) by a sum of
// angular modes that beat against each other at different rates — the normal
// modes of an oscillating liquid drop (Rayleigh drop oscillations): low orders
// stretch the bubble prolate/oblate, higher orders push triangular and star
// lobes. This is the per-VERTEX deformation an affine model-matrix transform
// physically cannot produce (a linear map only makes ellipsoids). Returns a
// roughly [-1,1] radial offset; the host scales it by `wobbleAmp`.
static inline float bubbleWobbleField(float3 d, float t, float ph, float freq) {
    // Real Rayleigh drop oscillation is dominated by LOW-order modes — big, smooth
    // lobes that beat against each other — with only a whisper of finer ripple.
    // Heavy high-order content reads as lumpy cauliflower, not a wobbling bubble.
    float s = 0.0;
    // l≈2 — the bubble breathes between prolate and oblate (the dominant motion).
    s += 0.85 * sin(M_PI_F * d.y + t * 1.30 * freq + ph);
    s += 0.70 * sin(M_PI_F * d.x - t * 1.10 * freq + ph * 1.7);
    s += 0.60 * sin(M_PI_F * d.z + t * 0.95 * freq + ph * 2.3);
    // l≈3 — gentle triangular lobes.
    s += 0.34 * sin(1.5 * M_PI_F * (d.x + d.y) + t * 1.55 * freq + ph * 0.7);
    s += 0.30 * sin(1.5 * M_PI_F * (d.z - d.y) - t * 1.40 * freq + ph * 1.9);
    // l≈4 — just a hint of finer ripple.
    s += 0.12 * sin(2.5 * M_PI_F * d.x + 2.0 * M_PI_F * d.z + t * 1.80 * freq + ph);
    return s * 0.39;
}

vertex GlassVSOut illumi_glass_rt_vs(
    uint                         vid       [[vertex_id]],
    uint                         iid       [[instance_id]],
    const device GlassVertex*    verts     [[buffer(0)]],
    constant GlassFrameUniforms& frame     [[buffer(1)]],
    const device GlassInstance*  instances [[buffer(2)]],
    constant GlassRTUniforms&    gu        [[buffer(3)]])
{
    GlassVertex v = verts[vid];
    GlassInstance inst = instances[iid];

    float3 objPos = v.position;
    float3 objNrm = v.normal;
    // Oscillation-mode undulation (soap bubbles). OFF (exact passthrough) unless
    // the host sets wobbleAmp > 0 — every other glass scene is unaffected.
    if (gu.wobbleAmp > 0.0) {
        float3 d  = normalize(v.position);          // unit-sphere surface dir
        float  ph = inst.dispersionPad.z;           // per-bubble phase
        float  fr = gu.wobbleFreq * max(0.2, inst.dispersionPad.w);  // per-bubble rate
        float  amp = gu.wobbleAmp;
        // Displaced radius at d and at two tangent neighbours → finite-difference
        // the perturbed normal so shading/Fresnel follow the lobes.
        float3 t1, t2; onb(d, t1, t2);
        float  e  = 0.06;
        float3 dA = normalize(d + t1 * e);
        float3 dB = normalize(d + t2 * e);
        float  r0 = 1.0 + amp * bubbleWobbleField(d,  gu.time, ph, fr);
        float  rA = 1.0 + amp * bubbleWobbleField(dA, gu.time, ph, fr);
        float  rB = 1.0 + amp * bubbleWobbleField(dB, gu.time, ph, fr);
        float3 p0 = d  * r0;
        float3 pA = dA * rA;
        float3 pB = dB * rB;
        objPos = p0;
        float3 n = normalize(cross(pA - p0, pB - p0));
        objNrm = (dot(n, d) < 0.0) ? -n : n;
    }

    float4 worldP = inst.modelMatrix * float4(objPos, 1.0);
    GlassVSOut o;
    o.clipPos     = frame.viewProjection * worldP;
    o.worldPos    = worldP.xyz;
    o.worldNormal = (inst.normalMatrix * float4(objNrm, 0.0)).xyz;
    o.instanceID  = iid;
    return o;
}

// ── Fragment stage ───────────────────────────────────────────────────────────
//
// Buffer indices (fragment):
//   0  acceleration structure (setFragmentAccelerationStructure index 0)
//   1  GlassRTUniforms
//   2  GlassInstance[]            (this fragment's material via instanceID)
//   3  RTInstanceData[]           (opaque+glass instance data: hit normals/albedo)
//   4  objNormal float4[]         (concatenated per-mesh object normals)
//   5  RTGlassData[]              (glass material per glass TLAS instance)
//   6  triCard uint[]             (surface cache)
//   7  triUVa float4[]
//   8  triUVc float4[]
//   9  soupTriBase uint[]
//   10 surfCardRect float4[]
//   11 surfCards SurfCard[]
// Fragment textures: 0 = sky equirect, 1 = surface-cache atlas.

/// Buffers this pass reads. Everything the SHARED secondary shading needs lives in
/// `sec` (one `SecondaryScene`, filled identically by every RT path); the rest is
/// genuinely glass-specific (the dielectric material table) or surface-cache
/// plumbing this pass addresses itself.
struct GlassRTState {
    SecondaryScene               sec;
    const device RTGlassData*    glassData;
    const device uint*           triCard;
    const device float4*         triUVa;
    const device float4*         triUVc;
    const device uint*           soupTriBase;
    const device float4*         surfCardRect;
    const device SurfCard*       surfCards;
};

// Re-shade or cache-read an OPAQUE triangle hit's outgoing radiance.
//
// The re-shade itself is `shadeSecondarySurface` — the ONE secondary-ray surface
// shader (IlluminatoramaSecondary.h), shared with the deferred TLAS reflection
// path. What is left here is only what is genuinely glass-pass-specific: this
// pass's own surface-cache addressing (no residency feedback, no variance readout)
// and facing the hit normal against the ray that found it.
static float3 shadeOpaqueHit(thread intersector<triangle_data, instancing>& isect,
                             instance_acceleration_structure accel,
                             uint iid, uint prim, float2 bary, float3 hitP, float3 rd,
                             constant GlassRTUniforms& u,
                             GlassRTState st,
                             texture2d<float, access::sample> surfAtlas,
                             texturecube<float, access::sample> irrCube,
                             texture2d_array<float, access::sample> albedoAtlas,
                             thread uint& seed,
                             intersection_result<triangle_data, instancing> res)
{
    SecondaryShadeParams p = glassSecondaryParams(u);
    float3 hitN = hitWorldNormal(iid, prim, st.sec.insts, st.sec.objNormal);
    if (dot(hitN, rd) > 0.0) hitN = -hitN;             // face the incoming ray
    // Surface-cache-when-available: a resident card returns full multi-bounce L_out.
    if (u.surfCacheEnabled != 0) {
        uint gp = st.soupTriBase[iid] + prim;
        uint hitCard = (gp < u.surfTriCount) ? st.triCard[gp] : 0xFFFFFFFFu;
        bool resident = (hitCard != 0xFFFFFFFFu) && (st.surfCardRect[hitCard].z > 0.0);
        if (resident) {
            return sampleSurfCacheRT(surfAtlas, gp, bary, st.surfCards, st.triCard,
                                     st.triUVa, st.triUVc, res.primitive_data,
                                     st.surfCardRect, u.surfAtlasW, u.surfAtlasH);
        }
        if (hitCard != 0xFFFFFFFFu) {
            return secondaryCardFallback(st.surfCards[hitCard], hitN,
                                         secondaryLayerBits(iid, st.sec.insts), p, irrCube);
        }
    }
    SecondaryHit h;
    h.P = hitP; h.N = hitN; h.bary = bary;
    h.instanceID = iid; h.primitiveID = prim;
    return shadeSecondarySurface(isect, accel, h, p, st.sec, irrCube, albedoAtlas, seed);
}

// Beer–Lambert attenuation over a path of length L inside glass of tint `tint`
// and density `density` (per-metre absorption). Clear glass (tint = 1) ⇒ no
// absorption; a coloured/dense glass absorbs the complementary channels.
static inline float3 beerLambert(float3 tint, float density, float L) {
    float3 sigma = density * (float3(1.0) - clamp(tint, 0.0, 1.0));
    return exp(-sigma * max(0.0, L));
}

// Trace one refraction path from the entry surface and return the radiance that
// reaches the eye through the glass. `iorEntry` is the wavelength-specific IOR
// (for dispersion the caller calls this 3× with split IORs).
static float3 traceRefractionPath(
    thread intersector<triangle_data, instancing>& isect,
    instance_acceleration_structure accel,
    float3 P, float3 V, float3 Ng, float iorEntry, float3 tint, float density,
    float roughness, constant GlassRTUniforms& u, GlassRTState st,
    texture2d<float, access::sample> sky,
    texture2d<float, access::sample> surfAtlas,
    texturecube<float, access::sample> irrCube,
    texture2d_array<float, access::sample> albedoAtlas,
    thread uint& seed)
{
    float eps = max(u.rayTMin, 1e-3);
    // Refract into the glass at the entry surface.
    float3 rd = refract(-V, Ng, 1.0 / max(1.0, iorEntry));
    if (dot(rd, rd) < 1e-8) rd = reflect(-V, Ng);      // grazing guard
    // Frosted: jitter the transmitted ray in a cone ∝ roughness². A sub-pixel cone is
    // skipped outright — see `secondaryConeVisible`; on a polished pane it was pure noise.
    if (secondaryConeVisible(roughness, kGlassConeK))
        rd = coneSample(normalize(rd), secondaryConeRad(roughness, kGlassConeK), rnd(seed), rnd(seed));
    float3 ro = P - Ng * eps;                           // start just inside
    float3 throughput = float3(1.0);
    bool inside = true;                                 // inside the entry medium
    float curIOR = iorEntry;

    isect.accept_any_intersection(false);
    uint maxB = min(u.maxBounces, 10u);
    for (uint b = 0u; b < maxB; ++b) {
        ray r; r.origin = ro; r.direction = normalize(rd);
        r.min_distance = eps; r.max_distance = 1e4;
        auto res = isect.intersect(r, accel, 0x03u);    // opaque + glass
        if (res.type != intersection_type::triangle) {
            // Escaped to the sky.
            return throughput * sampleSky(sky, rd, u.skyIntensity, glassNightSky(u), u.nightPixAngle);
        }
        uint iid = res.instance_id;
        float t = res.distance;
        float3 hitP = r.origin + r.direction * t;
        if (inside) throughput *= beerLambert(tint, density, t);

        if (iid >= u.glassInstanceBase) {
            // Dielectric boundary (exit, or enter a nested/adjacent glass).
            uint gi = iid - u.glassInstanceBase;
            RTGlassData gd = st.glassData[gi];
            float hitIOR = max(1.0, gd.tintIor.w);
            float3 hn = hitWorldNormal(iid, res.primitive_id, st.sec.insts, st.sec.objNormal);
            bool exiting = dot(r.direction, hn) > 0.0;  // leaving the medium we're in
            float3 n = exiting ? -hn : hn;              // normal against the ray
            float n1 = curIOR;
            float n2 = exiting ? 1.0 : hitIOR;
            float gr = gd.rdrf.x;
            float3 t2 = refract(r.direction, n, n1 / n2);
            if (dot(t2, t2) < 1e-8) {
                // Total internal reflection — bounce inside, same medium.
                rd = reflect(r.direction, n);
                ro = hitP + n * eps;                    // nudge to the incoming side
                continue;
            }
            if (secondaryConeVisible(gr, kGlassConeK))
                t2 = coneSample(normalize(t2), secondaryConeRad(gr, kGlassConeK), rnd(seed), rnd(seed));
            rd = t2;
            ro = hitP - n * eps;                        // cross the boundary
            // Update medium state. Exiting ⇒ now in air; entering ⇒ in the new glass.
            if (exiting) { inside = false; curIOR = 1.0; tint = float3(1.0); density = 0.0; }
            else         { inside = true;  curIOR = hitIOR; tint = gd.tintIor.xyz; density = gd.rdrf.y; }
            continue;
        }
        // Opaque surface: shade it and stop.
        float3 rad = shadeOpaqueHit(isect, accel, iid, res.primitive_id,
                                    res.triangle_barycentric_coord, hitP, r.direction,
                                    u, st, surfAtlas, irrCube, albedoAtlas, seed, res);
        return throughput * rad;
    }
    // Bounce budget exhausted — return the accumulated sky as a fallback.
    return throughput * sampleSky(sky, rd, u.skyIntensity, glassNightSky(u), u.nightPixAngle);
}

// Trace one reflection ray off the entry surface (single bounce; a glass hit
// gets a cheap sky-tinted approximation rather than recursing).
static float3 traceReflection(
    thread intersector<triangle_data, instancing>& isect,
    instance_acceleration_structure accel,
    float3 P, float3 Ng, float3 R, float roughness,
    constant GlassRTUniforms& u, GlassRTState st,
    texture2d<float, access::sample> sky,
    texture2d<float, access::sample> surfAtlas,
    texturecube<float, access::sample> irrCube,
    texture2d_array<float, access::sample> albedoAtlas,
    thread uint& seed)
{
    float eps = max(u.rayTMin, 1e-3);
    float3 dir = R;
    if (secondaryConeVisible(roughness, kGlassConeK))
        dir = coneSample(normalize(R), secondaryConeRad(roughness, kGlassConeK), rnd(seed), rnd(seed));
    if (dot(dir, Ng) <= 0.0) dir = R;                   // keep it above the surface
    ray r; r.origin = P + Ng * eps; r.direction = normalize(dir);
    r.min_distance = eps; r.max_distance = 1e4;
    isect.accept_any_intersection(false);
    auto res = isect.intersect(r, accel, 0x03u);
    if (res.type != intersection_type::triangle) {
        return sampleSky(sky, dir, u.skyIntensity, glassNightSky(u), u.nightPixAngle);
    }
    if (res.instance_id >= u.glassInstanceBase) {
        // Reflection landing on another glass surface: approximate with the sky
        // behind it tinted by the glass colour (avoids a second full path).
        uint gi = res.instance_id - u.glassInstanceBase;
        float3 tint = st.glassData[gi].tintIor.xyz;
        return sampleSky(sky, dir, u.skyIntensity, glassNightSky(u), u.nightPixAngle) * tint;
    }
    float3 hitP = r.origin + r.direction * res.distance;
    return shadeOpaqueHit(isect, accel, res.instance_id, res.primitive_id,
                          res.triangle_barycentric_coord, hitP, r.direction,
                          u, st, surfAtlas, irrCube, albedoAtlas, seed, res);
}

fragment float4 illumi_glass_rt_fs(
    GlassVSOut                       in          [[stage_in]],
    constant GlassRTUniforms&        u           [[buffer(1)]],
    const device GlassInstance*      instances   [[buffer(2)]],
    instance_acceleration_structure  accel       [[buffer(0)]],
    const device RTInstanceData*     insts       [[buffer(3)]],
    const device float4*             objNormal   [[buffer(4)]],
    const device RTGlassData*        glassData   [[buffer(5)]],
    const device uint*               triCard     [[buffer(6)]],
    const device float4*             triUVa      [[buffer(7)]],
    const device float4*             triUVc      [[buffer(8)]],
    const device uint*               soupTriBase [[buffer(9)]],
    const device float4*             surfCardRect[[buffer(10)]],
    const device SurfCard*           surfCards   [[buffer(11)]],
    const device float2*             objUV       [[buffer(12)]],
    const device float2*             albedoUVScale [[buffer(13)]],
    const device RTPointLight*       pointLights [[buffer(14)]],
    const device RTSpotLight*        spotLights  [[buffer(15)]],
    texture2d<float, access::sample> sky         [[texture(0)]],
    texture2d<float, access::sample> surfAtlas   [[texture(1)]],
    texturecube<float, access::sample> irrCube   [[texture(3)]],
    texture2d_array<float, access::sample> albedoAtlas [[texture(4)]])
{
    GlassInstance gi = instances[in.instanceID];
    float ior        = max(1.0, gi.tintIor.w);
    float3 tint      = gi.tintIor.xyz;
    float roughness  = saturate(gi.rdrf.x);
    float density    = max(0.0, gi.rdrf.y);
    float reflMul    = gi.rdrf.z;
    float dispersion = gi.dispersionPad.x;

    float3 N = normalize(in.worldNormal);
    float3 V = normalize(u.cameraWorldPos - in.worldPos);
    if (dot(N, V) < 0.0) N = -N;                        // face the viewer (front surface)

    GlassRTState st;
    st.sec.insts = insts;        st.sec.objNormal = objNormal;
    st.sec.objUV = objUV;        st.sec.uvScale = albedoUVScale;
    st.sec.pointLights = pointLights; st.sec.spotLights = spotLights;
    st.glassData = glassData;
    st.triCard = triCard; st.triUVa = triUVa; st.triUVc = triUVc;
    st.soupTriBase = soupTriBase; st.surfCardRect = surfCardRect; st.surfCards = surfCards;

    intersector<triangle_data, instancing> isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);

    // Per-fragment seed for the stochastic cones; varies per frame so TAA averages.
    uint seed = pcgHash(uint(in.clipPos.x) + uint(in.clipPos.y) * 9781u + u.frameSeed * 6151u);

    // ── Fresnel: a pane is a SLAB, not one interface ─────────────────────────
    // `Fs` is Schlick for ONE air–glass surface. A window has TWO of them, and the
    // light that gets past the front surface meets the back one on its way out —
    // some of it reflects there, comes back, partially exits toward the eye, and so
    // on. Summing that geometric series (incoherent, non-absorbing) is the standard
    // slab result:
    //
    //     R_slab = 2·Fs / (1 + Fs)
    //
    // At normal incidence on float glass that is 0.0769 rather than 0.0400 — the
    // sheen a real window carries is very nearly TWICE what a single interface
    // gives. It still goes to 1 at grazing, so the silhouette is unchanged; what
    // changes is the head-on read, which is exactly where a pane was disappearing
    // and the window was reading as an open hole.
    //
    // The complement matters just as much. The transmitted lobe below is weighted
    // `1 − R_slab` = 0.923 head-on; before this it was `1 − Fs` = 0.960, because the
    // entry surface's Fresnel was charged and the EXIT surface's never was
    // (`traceRefractionPath` refracts across the exit boundary without a
    // transmission factor). The pane was passing more light than two surfaces can.
    //
    // ASSUMPTION, stated: the series is summed with no absorption between the
    // surfaces (a = 1). Exact for clear glass; for a deeply body-tinted pane the
    // internal bounce is attenuated, so the true R_slab sits between Fs and this
    // value and we slightly over-state the sheen. Beer–Lambert still attenuates the
    // transmitted path itself inside `traceRefractionPath`, so only the second-order
    // internal bounce is idealised.
    float cosI = saturate(dot(N, V));
    float f0s = (ior - 1.0) / (ior + 1.0);
    float F0 = f0s * f0s;
    float Fs = F0 + (1.0 - F0) * pow(1.0 - cosI, 5.0);
    float F = saturate(2.0 * Fs / (1.0 + Fs));

    float3 P = in.worldPos;

    // ── Refraction (with optional chromatic dispersion) ──────────────────────
    float3 refr;
    if (u.dispersionEnabled != 0u && dispersion > 1e-4) {
        // Normal dispersion: blue bends more (higher IOR) than red. Split the IOR
        // and trace the path once per primary; recombine as RGB. Three full
        // paths — gated, so only dispersive glass pays it.
        float spread = dispersion * 0.04 * ior;         // ±IOR offset
        uint s0 = seed;
        float r = traceRefractionPath(isect, accel, P, V, N, ior - spread, tint, density,
                                      roughness, u, st, sky, surfAtlas, irrCube, albedoAtlas, seed).r;
        seed = s0 ^ 0x1234u;
        float g = traceRefractionPath(isect, accel, P, V, N, ior, tint, density,
                                      roughness, u, st, sky, surfAtlas, irrCube, albedoAtlas, seed).g;
        seed = s0 ^ 0x9abcu;
        float bb = traceRefractionPath(isect, accel, P, V, N, ior + spread, tint, density,
                                       roughness, u, st, sky, surfAtlas, irrCube, albedoAtlas, seed).b;
        refr = float3(r, g, bb);
    } else {
        // Frosted glass jitters the refraction in a cone, so a single sample is
        // noisy (TAA cleans it when static but it shimmers under motion). Average
        // a few stochastic path traces — count scales with roughness, so polished
        // glass pays for exactly one and only frosted glass pays more.
        uint nRefr = secondaryConeSamples(roughness, kGlassConeK, 3.0, 4u);
        if (nRefr <= 1u) {
            refr = traceRefractionPath(isect, accel, P, V, N, ior, tint, density,
                                       roughness, u, st, sky, surfAtlas, irrCube, albedoAtlas, seed);
        } else {
            float3 acc = float3(0.0);
            for (uint s = 0u; s < nRefr; ++s) {
                acc += traceRefractionPath(isect, accel, P, V, N, ior, tint, density,
                                           roughness, u, st, sky, surfAtlas, irrCube, albedoAtlas, seed);
            }
            refr = acc / float(nRefr);
        }
    }

    // ── Reflection ───────────────────────────────────────────────────────────
    // Same roughness-scaled averaging for glossy/frosted reflection.
    float3 R = reflect(-V, N);
    float3 refl;
    {
        uint nRefl = secondaryConeSamples(roughness, kGlassConeK, 2.0, 3u);
        float3 acc = float3(0.0);
        for (uint s = 0u; s < nRefl; ++s) {
            acc += traceReflection(isect, accel, P, N, R, roughness, u, st, sky, surfAtlas, irrCube, albedoAtlas, seed);
        }
        refl = acc / float(nRefl);
    }
    refl *= max(0.0, reflMul) * max(0.0001, u.reflStrength);

    // ── Fresnel blend. Refraction folds the whole background in, so the glass is
    // opaque (alpha = 1) — same contract as the old screen-space refraction
    // branch (blending the background again would double-count it). ────────────
    float3 color = mix(refr, refl, F);
    return float4(color, 1.0);
}

// ── Non-RT fallback ──────────────────────────────────────────────────────────
//
// When the device has no ray-tracing hardware (or `rtGlassEnabled` is off / the
// TLAS isn't live this frame), glass falls back to single-surface Fresnel + sky
// reflection + a tinted sky refraction — the same class of look the original
// screen-space pass gave, reading the SAME `GlassInstance` currency so there is
// only one glass system. Alpha-blends (src-alpha) so the lit scene shows through
// head-on. (The RT path returns alpha 1, so the same blended pipeline renders
// both: RT replaces, fallback blends.)
fragment float4 illumi_glass_fallback_fs(
    GlassVSOut                       in        [[stage_in]],
    constant GlassRTUniforms&        u         [[buffer(1)]],
    const device GlassInstance*      instances [[buffer(2)]],
    texture2d<float, access::sample> sky       [[texture(0)]],
    texture2d<float, access::sample> backdrop  [[texture(2)]])
{
    GlassInstance gi = instances[in.instanceID];
    float ior     = max(1.0, gi.tintIor.w);
    float3 tint   = gi.tintIor.xyz;
    float reflMul = max(0.0, gi.rdrf.z);
    float3 N = normalize(in.worldNormal);
    float3 V = normalize(u.cameraWorldPos - in.worldPos);
    if (dot(N, V) < 0.0) N = -N;
    float cosI = saturate(dot(N, V));
    float f0s = (ior - 1.0) / (ior + 1.0);
    float F0 = f0s * f0s;
    float F = F0 + (1.0 - F0) * pow(1.0 - cosI, 5.0);
    float3 R = reflect(-V, N);

    // ── Cheap semi-transparent glass (no ray tracing) ────────────────────────
    // A host (Hot Dog Press #64) opts in via `cheapGlassMode` when it wants
    // "semi-transparent glass" without paying a BLAS-per-deforming-body TLAS.
    if (u.cheapGlassMode != 0u) {
        // A sharp specular glint where the eye-reflected key sun aligns — the
        // single strongest "this surface is glass" cue against a dark void. Shared
        // by both cheap modes. `reflStrength` keeps it consistent with the RT path.
        float specA = pow(saturate(dot(R, normalize(u.sunDir))), 220.0);
        float3 glint = u.sunColor * (specA * 1.6 * max(0.05, u.reflStrength));
        // Sky reflection (usually a dark void here) scaled by reflectivity, plus a
        // faint tinted Fresnel rim so the silhouette/edges read even with no sky.
        float3 refl = sampleSky(sky, R, u.skyIntensity, glassNightSky(u), u.nightPixAngle) * reflMul;
        float3 rim  = tint * (F * 0.5 + 0.04);

        if (u.cheapGlassMode == 2u && u.viewW > 0.5) {
            // SCREEN-SPACE refraction: sample the pre-glass scene behind the pane,
            // offset along the screen-projected surface normal (grows toward
            // grazing). The contents must pass through CLEAR — a frank seen through
            // the wall should keep the orange it has above the rim (the reviewer's
            // "smoked amber" failure was over-absorption). So absorption is near-
            // neutral head-on and only lightly tints toward grazing.
            float2 uv = in.clipPos.xy / float2(u.viewW, u.viewH);
            float refrPx = gi.rdrf.w;                       // per-pane px strength
            float2 dir = N.xy;                              // view-space-ish screen push
            float dl = length(dir);
            if (dl > 1e-4) dir /= dl;
            float graze = 1.0 - cosI;                       // 0 head-on … 1 grazing
            float2 off = dir * (refrPx * graze) / float2(u.viewW, u.viewH);
            float3 behind = backdrop.sample(
                sampler(filter::linear, address::clamp_to_edge), uv + off).rgb;
            // Beer–Lambert, but a LOW coefficient + short head-on path so the
            // contents read essentially clear; the tint only bites near grazing
            // (thick edge) where it reads as a cool glass rim, not a brown wash.
            float L = (0.08 + graze * 0.9) * max(0.0, gi.rdrf.y);
            float3 absorb = exp(-(1.0 - clamp(tint, 0.0, 1.0)) * L);
            float3 transmitted = behind * absorb;
            // Bright glass edges — the legibility the Fresnel mode gets for free.
            // A broad grazing specular BAR (the cylinder's bright highlight) + a
            // tinted Fresnel rim, both gated BY FRESNEL (`* Fp`) so they brighten
            // ONLY the grazing edges and never wash the contents in the body of the
            // pane (the reviewer's interior-desaturation tell). The sharp sun glint
            // is a narrow lobe, so it can stay additive without washing anything.
            float bar = pow(saturate(dot(R, normalize(u.sunDir))), 40.0);
            float3 edge = tint * (F * 0.7) + u.sunColor * (bar * 0.8 * max(0.05, u.reflStrength));
            // Keep transmission-dominant on the body (contents stay clear) but let
            // the Fresnel edge fold in strongly at grazing so the silhouette pops.
            float Fp = saturate(F * 1.3 + 0.02);
            float3 color = mix(transmitted, refl, Fp) + edge * Fp + glint;

            // ── SOAP-BUBBLE thin-film shell ─────────────────────────────────
            // OFF unless a host opts in (`thinFilmStrength > 0`) → the opaque
            // fold-in path below is the EXACT prior behaviour for every other
            // cheap-glass scene (Hot Dog Press, …). When on, the shell is a
            // TRANSMISSIVE film: the body returns a low alpha so the scene AND
            // bubbles behind it show through (real bubble centres are windows),
            // and the iridescence is SILHOUETTE-WEIGHTED so it concentrates at
            // the rim and fades to a whisper through the centre. Composited via
            // straight alpha over the back-to-front-sorted framebuffer.
            if (u.thinFilmStrength > 0.0) {
                // Film thickness over the shell: gravity DRAINAGE thins the top
                // (high N.y → thinner) + a fine swirling turbulence advected by
                // time (NOT reseeded → TAA-stable). Per-bubble scale in dispPad.y.
                float drain = mix(1.30, 0.40, saturate(N.y * 0.5 + 0.5));
                float perBubble = max(0.2, gi.dispersionPad.y);
                float3 sp = in.worldPos * 15.0
                          + float3(0.0, -u.time * 0.5, u.time * 0.18);
                float swirl = filmFbm(sp);
                float thick = u.filmThicknessNm * perBubble * drain
                            * (0.6 + 0.8 * swirl);
                float3 irid = thinFilmIridescence(thick, cosI, max(1.05, u.filmIOR));

                float coatW = pow(graze, 2.2);              // silhouette weight
                // Bright, thin, near-white Fresnel rim — a HOT LIP that brightens
                // steeply only in the last few degrees toward the silhouette (high
                // exponent), the "film catching light" cue against the dark tank.
                float rimW = pow(graze, 9.0);
                // The shell's own contribution: a faint refracted scene that bites
                // toward the rim (the lensing), the interference film (whisper in
                // the centre, strong at the rim), a sky reflection, the rim, glint.
                float3 srcRGB = behind * (F * 0.9)
                              + irid * (u.thinFilmStrength * (0.04 + coatW))
                              + refl * (0.10 + 0.5 * F)
                              + float3(1.0) * (rimW * 1.15)
                              + glint;
                float iridLum = dot(irid, float3(0.299, 0.587, 0.114));
                // Mostly TRANSPARENT in the centre (~0.05), opaque toward the rim.
                float a = saturate(0.05 + coatW * 1.25 + rimW * 1.0
                                   + specA * 0.7
                                   + iridLum * u.thinFilmStrength * 0.04);
                return float4(srcRGB, a);
            }
            return float4(color, 1.0);                       // opaque (folds scene in)
        }

        // mode 1 — SYNTHETIC: no scene read. A tinted translucent sheet (alpha-
        // blended so the bodies behind still show) + Fresnel rim + key glint. The
        // tint pushes the bodies behind toward the glass colour; the rim/glint
        // give the pane a legible surface and silhouette.
        float3 sheet = mix(tint * 0.10, refl + rim, F) + glint;
        float alpha = clamp(0.14 + F * 0.78, 0.0, 1.0) + specA * 0.6;
        return float4(sheet, clamp(alpha, 0.0, 1.0));
    }

    // ── Plain Fresnel + sky fallback (unchanged; non-RT hardware) ─────────────
    float3 refl = sampleSky(sky, R, u.skyIntensity, glassNightSky(u), u.nightPixAngle) * reflMul;
    float3 T = refract(-V, N, 1.0 / ior);
    if (dot(T, T) < 1e-8) T = R;
    float3 refr = sampleSky(sky, T, u.skyIntensity, glassNightSky(u), u.nightPixAngle) * tint;
    float3 color = mix(refr, refl, F);
    float alpha = clamp(0.22 + F * 0.72, 0.0, 1.0);   // see-through head-on, opaque at grazing
    return float4(color, alpha);
}
