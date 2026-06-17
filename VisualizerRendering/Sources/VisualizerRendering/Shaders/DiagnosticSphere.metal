// ── DIAGNOSTIC 3D SPHERE (RAY-TRACED, METAL COMPUTE) ────────────────────────
//
// A real 3D sphere ray-traced per-pixel into the compositor's HDR target,
// rendered through the SAME Metal compositor pipeline that the burst
// particles use. This exists to settle the "can the pipeline even draw a
// 3D-looking sphere?" question — if YES, the answer for the particle
// shader is to do whatever this kernel does, scaled down to a particle.
//
// Pipeline:
//   1. For each pixel of the HDR target, compute the world-space ray from
//      camera through that pixel using the inverse view-projection.
//   2. Analytic ray-sphere intersection. Closed-form, no march steps.
//   3. On hit, compute the hit point's surface normal — that's the line
//      from sphere centre to hit point, normalised. This is the GEOMETRIC
//      normal of a real 3D sphere at the hit point in world space.
//   4. Shade with Lambert (diffuse) + Blinn-Phong specular using a fixed
//      directional light. Lambert produces the lit/unlit hemisphere
//      gradient that reads as 3D depth; the specular highlight, OFFSET
//      from the screen-space centre toward the light direction, is the
//      single biggest cue that makes the eye read it as 3D rather than
//      as a flat radially-shaded disc.
//   5. Additive write to the HDR target. The compositor's tonemap +
//      luminance-preserving ACES handles everything downstream.

#include <metal_stdlib>
using namespace metal;

// Uniform layout — kept simple and explicit so the diagnostic can be
// added/removed without touching other passes.
struct DiagnosticSphereUniforms {
    // Camera basis — inverse view-projection rows so we can unproject
    // screen-pixel NDC coordinates back to world space.
    float4 invViewProjRow0;
    float4 invViewProjRow1;
    float4 invViewProjRow2;
    float4 invViewProjRow3;
    // xyz = camera world position, w = pad
    float4 cameraPos;
    // xyz = sphere centre world position, w = sphere radius
    float4 sphereCenterRadius;
    // xyz = direction TO the light (unit), w = pad
    float4 lightDir;
    // rgb = surface diffuse albedo, a = pad
    float4 albedo;
};

// Unproject NDC + depth → world space using the inverse view-projection.
inline float3 ds_unprojectToWorld(
    float2 ndcXY,
    float ndcZ,
    constant DiagnosticSphereUniforms& u
) {
    float4 clip = float4(ndcXY, ndcZ, 1.0f);
    float4 r;
    r.x = dot(u.invViewProjRow0, clip);
    r.y = dot(u.invViewProjRow1, clip);
    r.z = dot(u.invViewProjRow2, clip);
    r.w = dot(u.invViewProjRow3, clip);
    return r.xyz / max(r.w, 1e-6f);
}

kernel void diagnosticSphere(
    texture2d<float, access::read_write> hdrOut [[ texture(0) ]],
    constant DiagnosticSphereUniforms& u        [[ buffer(0)  ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = hdrOut.get_width();
    uint h = hdrOut.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Pixel centre → NDC. Metal NDC y goes UP, compute thread y goes DOWN,
    // so flip.
    float2 ndcXY;
    ndcXY.x = (float(gid.x) + 0.5f) / float(w) * 2.0f - 1.0f;
    ndcXY.y = 1.0f - (float(gid.y) + 0.5f) / float(h) * 2.0f;

    // World-space ray for this pixel.
    float3 nearW = ds_unprojectToWorld(ndcXY, 0.0f, u);
    float3 farW  = ds_unprojectToWorld(ndcXY, 1.0f, u);
    float3 rayDir = normalize(farW - nearW);
    float3 rayOrigin = u.cameraPos.xyz;

    // ── ANALYTIC RAY-SPHERE INTERSECTION ────────────────────────────
    // |ray(t) - centre|² = r²  ⇒ quadratic in t.
    float3 centre = u.sphereCenterRadius.xyz;
    float  R      = u.sphereCenterRadius.w;
    float3 oc = rayOrigin - centre;
    float  b  = dot(oc, rayDir);
    float  c  = dot(oc, oc) - R * R;
    float  disc = b * b - c;
    if (disc < 0.0f) return;   // ray misses the sphere entirely
    float sqrtDisc = sqrt(disc);
    float t = -b - sqrtDisc;   // near intersection
    if (t < 0.0f) {
        t = -b + sqrtDisc;     // try far intersection (camera inside sphere)
        if (t < 0.0f) return;  // sphere is behind camera
    }

    // Hit point + surface normal in world space.
    float3 hit = rayOrigin + rayDir * t;
    float3 normal = (hit - centre) / R;

    // ── LAMBERT + BLINN-PHONG SHADING ───────────────────────────────
    //
    // This is the part that makes the result read as 3D — NOT a radial
    // falloff curve, NOT a Gaussian, NOT a "soft glowing ball" profile.
    // It's directional shading: brightness depends on the angle between
    // the surface normal and the light direction.
    //
    //   • Lambert: max(0, dot(N, L)) — gradient bright on the lit side,
    //     dark on the unlit side. This creates the lit/unlit hemisphere
    //     visible on every real photographed sphere.
    //
    //   • Blinn-Phong specular: a bright spot OFFSET from the screen-
    //     space centre toward the half-vector direction. The OFFSET is
    //     the visual cue that screams "3D" — a real sphere's specular
    //     highlight sits on the side facing the light, not in the centre
    //     of the silhouette. Any centred radial shape, no matter how
    //     soft, lacks this offset and reads as flat.
    //
    //   • Ambient: a small constant so the unlit side isn't pure black —
    //     real environments always have some sky/bounce light.
    float3 L = normalize(u.lightDir.xyz);
    float NdotL = max(0.0f, dot(normal, L));
    float3 viewDir = -rayDir;
    float3 H = normalize(L + viewDir);
    float NdotH = max(0.0f, dot(normal, H));
    float spec = pow(NdotH, 48.0f);

    float3 diffuse = u.albedo.rgb * (0.10f + 0.9f * NdotL);
    float3 highlight = float3(1.4f, 1.35f, 1.2f) * spec;
    float3 shaded = diffuse + highlight;

    // Additive composite into the HDR target. The downstream tonemap
    // handles bloom / ACES; this kernel just writes the lit colour and
    // marks the pixel as opaque so the compositor's premultiplied-alpha
    // tonemap covers the SCN backdrop here.
    float4 dst = hdrOut.read(gid);
    float3 final = shaded + dst.rgb;
    hdrOut.write(float4(final, 1.0f), gid);
}
