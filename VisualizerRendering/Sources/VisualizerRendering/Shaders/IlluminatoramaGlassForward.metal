// ── ILLUMINATORAMA — FORWARD GLASS ──────────────────────────────────────────
//
// The forward-rendered transparent pass drawn after the deferred resolve.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Transparent reflective glass pane (forward, alpha-blended) ────────────────
//
// One glass sheet rendered AFTER the opaque HDR composite and BEFORE TAA/bloom,
// so its reflections bloom + tonemap naturally. Reuses the Vertex / Instance /
// FrameUniforms structs and the equirect helpers above. Repurposed Instance
// fields (so the host needs no new upload path):
//   albedo     = glass tint (multiplies the reflection + a faint base sheen)
//   metallic   = reflectivity (0..1 strength of the env reflection)
//   roughness  = base alpha (min opacity head-on; edges add fresnel on top)
//   emission.x = fresnel power (higher → reflection only at grazing angles)
// Depth-TESTED (coins in front occlude the glass) but no depth WRITE; src-alpha
// blended over the lit pile already in the composite — the pile shows through.

// Per-glass-pass uniforms: the positionable key light (a controllable specular
// glint, since the forward glass pass otherwise only reflects the sky) plus the
// screen-space refraction knobs. xyz/w packed so the Swift struct round-trips
// without SIMD3 stride surprises. Stride 48 — matches GlassKeyLightUniform.
struct GlassKeyLight {
    float4 positionIntensity;   // xyz = world position, w = intensity
    float4 colorShininess;      // rgb = colour, w = shininess (glint tightness)
    uint   keyEnabled;
    uint   refractEnabled;      // screen-space refraction on/off
    float  ior;                 // index of refraction (glass ≈ 1.5)
    float  refractStrength;     // artistic scale on the screen-space UV offset
};

struct GlassVSOut {
    float4 clipPos [[position]];
    float3 worldPos;
    float3 worldNormal;
    // Per-instance index, flat-interpolated so the fragment shader can read
    // its own instance's material. `instance_id` is vertex-stage only, so we
    // forward it here — this is what lets the glass pass draw MANY panes
    // (orbiting lenses) in one instanced draw instead of a single sheet.
    uint   instanceID [[flat]];
};

vertex GlassVSOut illumi_glass_vs(
    uint                    vid       [[vertex_id]],
    uint                    iid       [[instance_id]],
    const device Vertex*    verts     [[buffer(0)]],
    constant FrameUniforms& frame     [[buffer(1)]],
    const device Instance*  instances [[buffer(2)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    GlassVSOut o;
    o.clipPos     = frame.viewProjection * worldP;
    o.worldPos    = worldP.xyz;
    o.worldNormal = (inst.normalMatrix * float4(v.normal, 0.0)).xyz;
    o.instanceID  = iid;
    return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// APPROACH B (later — GG3·17 "Robust Multiple Specular Reflections & Refractions",
// tracked on issue #57): true ray-traced refraction through the dielectric.
// Instead of the screen-space grab below, put the lens geometry in the RT TLAS
// and, per primary ray: refract at the FRONT surface (Snell, η = 1/IOR), march
// through the glass to the BACK surface, refract again on exit (or total-internal-
// reflect when past the critical angle), then continue to sky/scene. This gives
// physically-correct magnification/inversion, real lens-through-lens stacking, and
// caustics — none of which the screen-space approximation can do. It needs a
// dielectric BSDF + multi-bounce loop added to the RT kernel (which today only
// does diffuse GI + soft shadows) and the lenses moved from this forward pass into
// the RT scene representation. Big lift; keep the screen-space path as the
// realtime default and gate B behind its own toggle when built.
// ─────────────────────────────────────────────────────────────────────────────
fragment float4 illumi_glass_fs(
    GlassVSOut              in        [[stage_in]],
    constant FrameUniforms& frame     [[buffer(1)]],
    const device Instance*  instances [[buffer(2)]],
    constant GlassKeyLight& key       [[buffer(3)]],
    texture2d<float, access::sample> sky        [[texture(0)]],
    texture2d<float, access::sample> background [[texture(1)]]
) {
    Instance inst = instances[in.instanceID];
    float3 N = normalize(in.worldNormal);
    float3 V = normalize(frame.cameraWorldPos - in.worldPos);
    if (dot(N, V) < 0.0) N = -N;                 // two-sided: face the viewer
    float  ndv     = max(dot(N, V), 0.0);
    float  fresPow = max(inst.emission.x, 1.0);
    float  fresnel = pow(1.0 - ndv, fresPow);    // 0 head-on → 1 grazing
    float3 R       = reflect(-V, N);
    float3 refl    = sampleSkyEquirect(sky, R) * inst.metallic;
    float3 tint    = inst.albedo;
    // Visible glass over a DARK interior: the env reflection alone is too dim to
    // read, so add a FRESNEL-WEIGHTED tint sheen — the pane's grazing edges catch a
    // cool highlight while it stays mostly see-through head-on. Sells "there's glass
    // here" without occluding the pile.
    float  sheen   = 0.03 + fresnel * 0.55;

    float3 color;
    float  alpha;
    if (key.refractEnabled != 0u) {
        // ── APPROACH A — screen-space refraction (GG2·19, issue #47) ──────────
        // Sample the pre-glass composite (sky + opaque scene, blitted into
        // `background` before this pass) along the REFRACTED view ray. We perturb
        // the screen UV by the refracted direction expressed in view space; the
        // lens's analytic concave normals make the offset DIVERGE outward, so the
        // background reads minified/upright — a real biconcave (diverging) lens,
        // not a fake warp. With refraction on the lens is opaque: the background
        // is already folded in here, so alpha = 1 (alpha-blending it again would
        // double-count the background).
        //
        // Limitation (the reason Approach B exists): the grab is frozen BEFORE the
        // glass pass, so a lens does not refract OTHER lenses behind it — only the
        // sky/opaque scene. Overlapping lenses simply occlude front-over-back.
        constexpr sampler bgS(filter::linear, address::clamp_to_edge);
        float2 screenUV = in.clipPos.xy /
                          float2(background.get_width(), background.get_height());
        float  eta = 1.0 / max(1.0, key.ior);
        float3 T   = refract(-V, N, eta);                  // world-space refracted ray
        float3 Tv  = (frame.view * float4(T, 0.0)).xyz;    // → view space
        float2 uvOff = Tv.xy * key.refractStrength;
        uvOff.y = -uvOff.y;                                // view +y is up, UV +y is down
        float2 ruv = clamp(screenUV + uvOff, 0.0, 1.0);
        float3 bg  = background.sample(bgS, ruv).rgb;
        float3 transmit = bg * tint;                       // glass tints transmitted light
        // Fresnel mix: see-through (refracted) head-on, sky reflection at grazing.
        color = mix(transmit, refl + tint * sheen, fresnel);
        alpha = 1.0;
    } else {
        // Translucent reflective glass (no refraction): see-through pane that
        // catches the sky + fresnel sheen, alpha-blended over the scene.
        color = refl * tint + tint * sheen;
        alpha = clamp(inst.roughness + fresnel * 0.85, 0.0, 1.0);
    }

    // ── Positionable key light: a Blinn specular glint that sweeps across the
    // lens as it tumbles. Pure highlight (no diffuse — glass has none); it both
    // brightens the colour and locally firms up alpha so the glint reads as a
    // crisp hotspot rather than a faint wash.
    if (key.keyEnabled != 0u) {
        float3 Lp        = key.positionIntensity.xyz;
        float  intensity = key.positionIntensity.w;
        float3 kColor    = key.colorShininess.rgb;
        float  shininess = max(1.0, key.colorShininess.w);
        float3 L = normalize(Lp - in.worldPos);
        float3 H = normalize(L + V);
        float  spec = pow(max(dot(N, H), 0.0), shininess);
        // Tint the glint by the lens colour so a coloured lens (e.g. Clock mode's
        // hour/minute/second hands) keeps its hue under the key light at night,
        // instead of washing to the light's own colour. Clear glass (tint ≈ white)
        // is unaffected.
        float3 glint = kColor * tint * (intensity * spec);
        color += glint;
        alpha  = clamp(alpha + spec * intensity * 0.5, 0.0, 1.0);
    }

    return float4(color, alpha);                 // straight (non-premultiplied) src-alpha blend
}
