// ── ILLUMINATORAMA — SUPERQUADRIC IMPOSTORS ─────────────────────────────────
//
// Ray-marched superquadric proxies drawn as screen-space impostors, writing the
// same G-buffer attachments as a real mesh.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Perfect analytic superquadric (hero primitive) ───────────────────────────
//
// A superquadric is rendered the CORRECT way — by ray–surface intersection in
// the fragment, NOT as a tessellated mesh — so its silhouette is mathematically
// exact at any zoom (no facets, ever). One logical object is drawn as two
// instances (see IlluminatoramaSuperquadric.swift): this IMPOSTOR (a [-1,1]
// bounding box whose fragments ray-trace the analytic surface and write the
// G-buffer + analytic depth + analytic motion vectors) and a moderate-tessellation
// triangle PROXY (raster-skipped, lives only in the TLAS so the object still
// casts RT shadows / appears in RT GI & reflections — the RT path is hardware-
// triangle-only).
//
// KEY TRICK: extents are folded into the instance's modelMatrix SCALE, so the
// object-space shape is always a UNIT superquadric (a=b=c=1, bounded by [-1,1]³).
// That makes step sizes scale-independent and lets the ellipsoid case reduce to a
// closed-form ray–UNIT-SPHERE intersection — the ellipsoid-ness (and the correct
// `normalize(localHit/extents²)` normal) falls straight out of the existing
// modelMatrix / normalMatrix, no special-casing.
//
// Convention (matched byte-for-byte by the proxy mesh + the SDF below):
//   F(x,y,z) = (|x|^(2/e2) + |y|^(2/e2))^(e2/e1) + |z|^(2/e1)
//   surface = F==1 ; inside F<1 ; e1==e2==1 → unit sphere.

struct SQParam {
    float4x4 invModel;   // world → object (precomputed CPU-side per instance)
    float4   shape;      // x=e1 (vertical squareness), y=e2 (horizontal), z=isEllipsoid(1/0), w=0
};

// Superquadric inside–outside function F (object space, unit shape a=b=c=1).
// <1 inside, >1 outside, ==1 on the surface.
static inline float sqField(float3 p, float e1, float e2) {
    float ax = pow(abs(p.x), 2.0 / e2);
    float ay = pow(abs(p.y), 2.0 / e2);
    float az = pow(abs(p.z), 2.0 / e1);
    return pow(ax + ay, e2 / e1) + az;
}

// Slab intersection with the object-space AABB [-1,1]³. Returns the entry/exit
// ray parameters; t1 < t0 means a miss. (oo,od) need not be normalized — t is in
// object-direction units, which is fine because every hit point is reconstructed
// as oo + od*t consistently.
static inline bool sqBoxRange(float3 oo, float3 od, thread float& t0, thread float& t1) {
    float3 inv = 1.0 / od;
    float3 a = (float3(-1.0) - oo) * inv;
    float3 b = (float3( 1.0) - oo) * inv;
    float3 tmin = min(a, b), tmax = max(a, b);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 >= max(t0, 0.0);
}

// Object-space ray–superquadric intersection. tHit is the near hit; oN is the
// outward object-space surface normal (central-difference gradient of F — robust
// at the axis/pole singularities where the analytic derivative of |x|^p (p<1)
// blows up). Returns false on a miss.
static inline bool sqIntersect(float3 oo, float3 od, float4 shape,
                               thread float& tHit, thread float3& oN) {
    float e1 = shape.x, e2 = shape.y;

    // ── Ellipsoid fast path: object space is the UNIT SPHERE. Closed-form. ──
    if (shape.z > 0.5) {
        float A = dot(od, od);
        float B = 2.0 * dot(oo, od);
        float C = dot(oo, oo) - 1.0;
        float disc = B * B - 4.0 * A * C;
        if (disc < 0.0) return false;
        float sq = sqrt(disc);
        float tn = (-B - sq) / (2.0 * A);
        float tf = (-B + sq) / (2.0 * A);
        float t = (tn > 1e-5) ? tn : tf;        // camera-outside → near root; inside → far
        if (t <= 1e-5) return false;
        tHit = t;
        oN = oo + od * t;                        // unit-sphere normal == position
        return true;
    }

    // ── General superquadric: sphere-trace the bounded field inside [-1,1]³. ──
    float t0, t1;
    if (!sqBoxRange(oo, od, t0, t1)) return false;
    float t = max(t0, 0.0);
    float th = -1.0;
    float startF = sqField(oo + od * t, e1, e2);
    if (startF <= 1.0) {
        // BOXY shapes nearly fill their AABB, so a face-on ray enters the solid
        // right at the box boundary (F ≤ 1 already). The entry point IS the
        // surface — without this, the near face is missed and the primitive
        // renders see-through (the green near-cube bug).
        th = t;
    } else {
        // 128 steps: stiff boxy fields (e≈0.1 → |x|^20) cross F=1 in a very thin
        // band, especially for face-grazing rays — too few steps step over it and
        // punch a see-through hole. A hero-count primitive can afford the density.
        const int   STEPS = 128;
        const float dt = (t1 - t) / float(STEPS);
        if (dt <= 0.0) return false;
        // Walk until F first drops to/below 1 (outside→inside), then bisect.
        for (int i = 0; i < STEPS; ++i) {
            float tn = t + dt;
            float f = sqField(oo + od * tn, e1, e2);
            if (f <= 1.0) {
                float lo = t, hi = tn;
                for (int b = 0; b < 12; ++b) {
                    float mid = 0.5 * (lo + hi);
                    if (sqField(oo + od * mid, e1, e2) > 1.0) lo = mid; else hi = mid;
                }
                th = 0.5 * (lo + hi);
                break;
            }
            t = tn;
        }
        if (th < 0.0) return false;
    }
    if (th <= 1e-5) return false;
    // Central-difference gradient of F → outward object-space normal.
    float3 ph = oo + od * th;
    const float h = 2e-3;
    float gx = sqField(ph + float3(h,0,0), e1, e2) - sqField(ph - float3(h,0,0), e1, e2);
    float gy = sqField(ph + float3(0,h,0), e1, e2) - sqField(ph - float3(0,h,0), e1, e2);
    float gz = sqField(ph + float3(0,0,h), e1, e2) - sqField(ph - float3(0,0,h), e1, e2);
    tHit = th;
    oN = float3(gx, gy, gz);
    return true;
}

struct SQImpostorVSOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   iid      [[flat]];
};

// The impostor draws the instance's [-1,1] bounding box; the fragment does the
// real intersection. The box is just there to generate fragments over the
// silhouette's screen footprint — back faces are rendered (cull .front host-side)
// so the primitive survives the camera entering the bounding box.
vertex SQImpostorVSOut illumi_superquadric_impostor_vs(
    uint                       vid       [[vertex_id]],
    uint                       iid       [[instance_id]],
    const device Vertex*       verts     [[buffer(0)]],
    constant FrameUniforms&    frame     [[buffer(1)]],
    const device Instance*     instances [[buffer(2)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    SQImpostorVSOut o;
    o.clipPos  = frame.viewProjection * worldP;   // jittered current VP (matches illumi_vs)
    o.worldPos = worldP.xyz;
    o.iid      = iid;
    return o;
}

// Writes the SAME G-buffer layout as illumi_fs, plus analytic [[depth]]. Renders
// into the same G-buffer encoder/pass. `[[depth(less)]]`: with back faces, the
// analytic surface is always at or in front of the rasterized far-wall depth, so
// the written depth is ≤ the interpolated one — a valid promise (no early-Z gain
// under the `.less` compare, negligible for a hero-count primitive).
struct SQImpostorFSOut {
    half4 albedoMetallic   [[color(0)]];
    half4 normalRoughness  [[color(1)]];
    half4 emission         [[color(2)]];
    half2 velocity         [[color(3)]];
    uint  layer            [[color(4)]];   // light-layer bitfield (default 0xFFFFFFFF)
    float depth            [[depth(less)]];
};

fragment SQImpostorFSOut illumi_superquadric_impostor_fs(
    SQImpostorVSOut            in            [[stage_in]],
    constant FrameUniforms&    frame         [[buffer(1)]],
    const device Instance*     instances     [[buffer(2)]],
    const device Instance*     prevInstances [[buffer(4)]],
    const device SQParam*      params        [[buffer(6)]]
) {
    Instance inst = instances[in.iid];
    SQParam  p    = params[in.iid];

    // World-space view ray through this fragment, taken into object space. od is
    // intentionally NOT renormalized after the inverse-model transform — the
    // hit point is reconstructed in object space as oo + od*t, so units cancel.
    float3 ro  = frame.cameraWorldPos;
    float3 rdW = normalize(in.worldPos - ro);
    float3 oo  = (p.invModel * float4(ro,  1.0)).xyz;
    float3 od  = (p.invModel * float4(rdW, 0.0)).xyz;

    float  tHit;
    float3 oN;
    if (!sqIntersect(oo, od, p.shape, tHit, oN)) discard_fragment();

    float3 oHit  = oo + od * tHit;
    float3 wHit  = (inst.modelMatrix * float4(oHit, 1.0)).xyz;
    float3 wN    = normalize((inst.normalMatrix * float4(oN, 0.0)).xyz);

    // Analytic depth — exact ray-hit, NOT the box. Every downstream pass
    // (lighting, SSR, SSAO, DOF, volumetric, TAA) reads this.
    float4 curClip = frame.viewProjection * float4(wHit, 1.0);

    // Motion vector from the analytic hit under the current vs previous instance
    // transform (rigid-motion assumption: the surface point sticks to the object).
    // The proxy-box vertices' motion is meaningless for the impostor, so velocity
    // MUST come from here or TAA smears every moving instance.
    Instance prevInst = prevInstances[in.iid];
    float3 prevW    = (prevInst.modelMatrix * float4(oHit, 1.0)).xyz;
    float4 prevClip = frame.previousViewProjection * float4(prevW, 1.0);
    float2 curNDC  = curClip.xy  / curClip.w;
    float2 prevNDC = prevClip.xy / prevClip.w;
    float2 velUV   = (curNDC - prevNDC) * float2(0.5, -0.5);

    SQImpostorFSOut o;
    o.albedoMetallic  = half4(half3(inst.albedo), half(inst.metallic));
    float2 oct = octEncode(wN);
    // .w tag: 1.0 = opaque (default); 1.0 + anisotropy (>1) carries the grain-highlight stretch
    // for the deferred pass (all existing tag tests are < 1.0, so this reads as opaque to them).
    // The SIGN of anisotropy authors the grain AXIS the same way the mesh G-buffer does: positive
    // = horizontal grain (1+a, unchanged), negative = vertical grain (2+|a|). Impostors never set
    // vertical grain today, so the positive path stays byte-identical.
    half sqW = (inst.anisotropy < 0.0f)
             ? 2.0h + half(clamp(-inst.anisotropy, 0.0f, 1.0f))
             : 1.0h + half(inst.anisotropy);
    o.normalRoughness = half4(half(oct.x), half(oct.y), half(inst.roughness), sqW);
    // DH-0081 — same sheen encoding the mesh G-buffer uses: fold the roughness band into the
    // integer part, strength into the fraction. Band 0 (default nap) packs `-sheen` unchanged.
    float sqSheenAlpha = (inst.sheen > 0.0f)
        ? float(clothSheenBandForRoughness(inst.sheenRoughness)) + min(inst.sheen, 0.98f)
        : 0.0f;
    o.emission        = half4(half3(inst.emission), half(inst.clearcoat > 0.0 ? inst.clearcoat : -sqSheenAlpha));
    o.velocity        = half2(velUV);
    o.layer           = inst.layer;   // light-layer bitfield (default 0xFFFFFFFF)
    o.depth           = curClip.z / curClip.w;   // Metal NDC z ∈ [0,1]
    return o;
}
