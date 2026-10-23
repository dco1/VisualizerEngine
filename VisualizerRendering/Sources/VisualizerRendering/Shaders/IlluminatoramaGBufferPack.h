// ── ILLUMINATORAMA — G-BUFFER OUTPUT PACKING (shared) ─────────────────────────────────────────
//
// The G-buffer's attachment layout (`GBufferOut`) and the last step of the material evaluation:
// a finished surface (normal, albedo, metallic, roughness) packed into the targets — the material
// class tag in normalRoughness.w, emission + clearcoat/sheen, the photo-lane material payload and
// the motion vector. Split out of IlluminatoramaGBufferShade.h (Signack S4, Daydream DH-1461); the
// body is unchanged.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
#include "IlluminatoramaGBufferDeform.h"
#include "IlluminatoramaLeafOptics.h"
using namespace metal;

struct GBufferOut {
    half4 albedoMetallic   [[color(0)]];
    half4 normalRoughness  [[color(1)]];
    half4 emission         [[color(2)]];
    // Phase 2.7 — screen-space motion vector (current_uv - previous_uv), in
    // UV units (range typically ±0.5). RG16Float gives plenty of precision.
    half2 velocity         [[color(3)]];
    // Light-layer bitfield (R32Uint target). Read by the deferred lighting kernel
    // to mask per-room lights. Cleared to 0xFFFFFFFF for sky/no-geometry pixels, so
    // a scene that never sets a layer (all instances default 0xFFFFFFFF) writes
    // 0xFFFFFFFF everywhere ⇒ every light passes the mask ⇒ byte-identical.
    uint  layer            [[color(4)]];
    // DH-0140 — photo-lane material payload (RGBA16F), present only when kExtendedGBuffer:
    //   .r  thin-sheet translucency (DH-0140 slice 3: milky canopy, woven shade — 0 = opaque)
    //   .gb per-texel grain tangent (reserved: DH-0478 — 0,0 = use the per-instance axis)
    //   .a  clearcoat GGX roughness (DH-0478; 0 = "not written" → the lighting kernel's 0.08)
    // A pipeline that does not write this target (impostors, any legacy PSO) leaves the clear
    // value, and every reader treats 0 as "fall back to the live-lane behaviour".
    half4 material         [[color(5), function_constant(kExtendedGBuffer)]];
    // (color 6 is the visibility buffer's memoryless id target — IlluminatoramaVisBuffer.metal.)
    // Daydream DH-1300 — the material-layer target (R32Uint): this fragment's INSTANCE SLOT when
    // the instance carries per-material layer parameters (specular IOR / weight / colour …), else
    // kMaterialLayerNone — the clear value, and what every instance of a host that never sets one
    // writes. The lighting and RT kernels read the parameters out of the instance buffer by it
    // (IlluminatoramaMaterialLayers.h). Every lane, live and photo.
    uint  materialLayer    [[color(7)]];
};

static inline __attribute__((always_inline)) GBufferOut illumiGBufferPack(
    VSOut in, Instance inst, constant FrameUniforms& frame, InteriorMapParams imap,
    float3 n, float3 albedo, float metallic, float roughness, float2 matUV,
    float2 duvdx, float2 duvdy,
    const device float2* albedoUVScale, const device float2* nonColorUVScale,
    const device float4* albedoSliceMean, const device float4* nonColorSliceMean,
    texture2d_array<float, access::sample> albedoAtlas,
    texture2d_array<float, access::sample> nonColorAtlas, sampler texSampler
) {
    GBufferOut o;
    // DH-1301 — a pure dielectric's diffuse roughness rides the sign of the metalness channel
    // (IlluminatoramaDiffuseRoughness.h); 0 (every material that never sets one) packs `metallic`.
    o.albedoMetallic  = half4(half3(albedo), illumiPackMetallicDiffuseRoughness(metallic, inst.materialExt.x));
    float2 oct = octEncode(n);
    // normalRoughness.w is otherwise unused (always 1) — repurpose it as a
    // foliage flag. Scenes that want leaf thin-sheet transmission tag their
    // leaf vertices with colour ALPHA 0 (opaque geometry keeps alpha 1); the
    // deferred lighting pass reads w < 0.5 to add the back-light term. No-op
    // for every existing scene (they all ship vertex alpha 1 → w stays 1).
    // Signack S5 (DH-1461 / DH-1361) — a foliage vertex alpha in (0, 0.45] is the leaf's relative
    // THICKNESS (IlluminatoramaLeafOptics.h `leafThickness`), carried in the tag itself: still < 0.5,
    // so every foliage test is unchanged, and 0 (every legacy foliage writer) is the reference blade.
    half foliageFlag = (in.vertexColor.a < 0.5) ? half(clamp(in.vertexColor.a, 0.0f, kLeafThicknessCodeMax)) : 1.0h;
    // Sausage casing (vertexColor.a = 0.75 from pbdTubeExpand) forwards its
    // flag into normalRoughness.w so the deferred lighting pass can add the
    // clearcoat glaze lobe. Every foliage test is `w < 0.5` and the matRough
    // cap below is `< 0.5h`, so 0.75 behaves as ordinary opaque geometry
    // everywhere except the clearcoat branch.
    if (in.vertexColor.a > 0.66 && in.vertexColor.a < 0.9) foliageFlag = 0.75h;
    // Lamp-shade fabric flag (DH-0458): colour alpha in (0.60, 0.66] → 0.62h. Carved
    // from the casing band's low end (raised to >0.66 above), so a paper/linen drum
    // shade is tagged for thin-sheet transmission WITHOUT the sausage clearcoat or the
    // plush fuzz. 0.62 is not < 0.5 (roughness cap left alone) and outside every other
    // band; the deferred lighting pass reads 0.60h<w<0.66h to add the fabric back-light.
    // No-op for every other scene (no other mesh ships colour alpha in this band).
    if (in.vertexColor.a > 0.60 && in.vertexColor.a <= 0.66) foliageFlag = 0.62h;
    // Plush flag (Teddy Bear Press): colour alpha ≈ 0.55 (band 0.5–0.6) → 0.55h.
    // 0.55 is not < 0.5 (so the foliage roughness cap below leaves plush's high
    // roughness alone) and not in (0.6,0.9) (so it skips the casing clearcoat); the
    // deferred lighting pass reads 0.5h<w<0.6h to add the fur sheen + SSS. No-op for
    // every other scene (no other mesh ships colour alpha in this band).
    if (in.vertexColor.a >= 0.5 && in.vertexColor.a <= 0.6) foliageFlag = 0.55h;
    // Screen-space SSS flag (issue #65 / DH-0138): colour alpha ∈ [0.90,0.98] → 0.95h,
    // for skin / wax / marble / food. This is the rung the lighting pass's SSS band
    // (`nrH.a > 0.90h && nrH.a < 0.99h`) and every comment in Common.h / Types /
    // Lighting already assumed existed — without it a 0.95-alpha mesh fell through to
    // the 1.0h opaque tag and was reclassified as anisotropy, so SSS was unreachable
    // outside the headless `sssDebugForceAll` escape. 0.95h lands BELOW 1.0h, so the
    // `wTag >= 1.0h` anisotropy branch does NOT fire on it (it forgoes anisotropy, as
    // plush/casing already do); 0.95 is not < 0.5h so the foliage roughness cap is left
    // alone. No-op for every scene that ships opaque alpha 1.0, and dead unless the
    // scene opts in via `frame.sssStrength > 0`.
    if (in.vertexColor.a >= 0.9 && in.vertexColor.a <= 0.98) foliageFlag = 0.95h;
    // Waxy-leaf sheen (#58): a real leaf cuticle is markedly smoother than the
    // matte moss / bark / stone around it, so a single soup roughness reads
    // every surface as the same dry matte and leaves never catch a glint.
    // Cap foliage roughness so backlit / edge-lit leaves pick up a soft
    // specular highlight (the wet-canopy cue) while staying well short of
    // plastic. Gated by the foliage flag (w < 0.5 ⇒ foliage) → exact no-op for
    // every non-foliage surface and every scene that ships vertex alpha 1.
    float matRough = (foliageFlag < 0.5h) ? min(roughness, 0.46) : roughness;
    // Phase 7c — grain anisotropy, and the reason it had never done anything on this
    // path. The deferred lighting pass reads `(w - 1)` as the anisotropy amount when
    // `w > 1.001h` (IlluminatoramaLighting.metal), and the superquadric impostor path
    // has written it that way since Phase 7c shipped — but THIS shader, which every
    // ordinary mesh goes through, only ever wrote the class tag. So `inst.anisotropy`
    // reached the instance buffer and stopped: every wood floor and brushed-metal
    // surface in a deferred scene rendered with a round, isotropic highlight.
    //
    // The tag and the anisotropy share one channel, which is why this is a branch and
    // not an add. `foliageFlag` is one of four DISCRETE class tags (0.0 foliage / 0.55
    // plush / 0.75 casing / 1.0 ordinary opaque) and every consumer reads them as bands
    // bounded ABOVE (< 0.5h, (0.5,0.6), (0.6,0.9), (0.90,0.99)) — so adding anisotropy
    // to a tagged pixel would silently reclassify it into the next band up. Only the
    // ordinary-opaque tag carries it; a leaf or a plush surface keeps its class and
    // forgoes anisotropy, which costs nothing (neither ships a grain tangent).
    //
    // Exactly identity when `inst.anisotropy >= 0`: the write is `1.0h + clamp(a)`, the same
    // bits the old line produced, so every scene that never sets the field — which is
    // every Visualizer scene — is bit-identical. The clamp is insurance the field has
    // nowhere else in the pipeline: `brdf()` splits GGX as `ab = a * (1 - 0.7 * aniso)`,
    // which goes NEGATIVE past aniso ≈ 1.43.
    //
    // AUTHORABLE grain axis: the SIGN of `inst.anisotropy` chooses the band the lighting pass
    // reads the orientation from — POSITIVE = horizontal grain (1+|a|, the original), NEGATIVE
    // = vertical grain (2+|a|). A brushed fridge/dishwasher door ships negative anisotropy so its
    // highlight runs UP the panel; a range ships positive so it runs across. See the decode in
    // IlluminatoramaLighting.metal.
    half wTag = foliageFlag;
    if (wTag >= 1.0h) {
        wTag = (inst.anisotropy < 0.0f)
             ? 2.0h + half(clamp(-inst.anisotropy, 0.0f, 1.0f))
             : 1.0h + half(clamp(inst.anisotropy, 0.0f, 1.0f));
    }
    o.normalRoughness = half4(half(oct.x), half(oct.y), half(matRough), wTag);
    // Phase 4.9 — emission can be a texture (used heavily by Plus scenes
    // for glow effects on rails / lamps / fire) or a scalar. Both are
    // additive on top of the lit colour. Sampling from the sRGB-decoded
    // albedo atlas means the bake gets linear RGB directly.
    float3 emission = inst.emission;
    if (inst.emissionTextureSlice >= 0) {
        // Domain-warped sampling UV. Exact no-op when `uvWarp.x == 0`.
        float2 emUV = warpUV(in.uv, inst.uvWarp, frame.time);
        float4 tx = sampleAtlasAspect(albedoAtlas, texSampler, emUV,
                                      uint(inst.emissionTextureSlice), albedoUVScale, duvdx, duvdy);
        // Phase 4.27b — scale the emission texture by the material's
        // `emission.intensity` so a texture-driven glow reads at its tuned
        // HDR brightness (Pizza's heat coils were flat at intensity 1).
        emission += tx.rgb * inst.emissionIntensity;
    }
    // Interior mapping: the pane's emission is the lamp-lit room behind it (inst.emission = the
    // lamp's colour × the room's luminance), seen along this fragment's camera ray.
    if (kInteriorMap) {
        emission = inst.emission * interiorMapRadiance(in.worldPos, n, in.worldTangent, in.uv,
                                                       frame.cameraWorldPos, imap);
    }
    // Glow of host-TAGGED vertices (Instance.tagGlow; same tangent.w ≈ 1 tag as the hue cycle):
    // the petal emits its own final albedo, so a hue-cycling flower glows in its current colour.
    // Static per-instance weight × the frame's live gain.
    float tagGlow = inst.tagGlow.x * frame.tagGlowGain;
    if (tagGlow > 0.0f && in.worldTangent.w > 0.5f && in.worldTangent.w < 1.5f) {
        emission += albedo * tagGlow;
    }
    // Phase 7 — pack clearcoat (≥0) OR cloth sheen (<0) into emission.alpha (was always 1.0,
    // unused). A surface is polished OR cloth, never both, so one channel carries either: > 0 =
    // polished/lacquered second GGX lobe; < 0 = velvet/wool grazing-Fresnel sheen.
    // DH-0081 — the sheen value folds the roughness BAND into its integer part and the STRENGTH
    // into its fraction: `-(band + strength)`. Band 0 (default 0.30 nap) makes this `-strength`,
    // byte-for-byte the pre-band encoding, so every non-opting material is unchanged.
    // A carpet's sheen swells with the pile lay (same field as the tone swell above): the nap that
    // leans toward the eye is lighter and glossier, the nap that leans away is darker and flatter.
    float sheenStrength = inst.sheen * carpetLaySheenScale(in.uv, inst.carpetMacro);
    float sheenAlpha = (inst.sheen > 0.0f)
        ? float(clothSheenBandForRoughness(inst.sheenRoughness)) + min(sheenStrength, 0.98f)
        : 0.0f;
    o.emission        = half4(half3(emission), half(inst.clearcoat > 0.0 ? inst.clearcoat : -sheenAlpha));
    if (kExtendedGBuffer) {
        // DH-0478 — per-texel grain DIRECTION for the anisotropy lobe. The material's map holds
        // (cos 2θ, sin 2θ), θ from its U axis in tangent space; here it becomes a world-space angle
        // φ against the SAME base tangent the lighting kernel builds (`anisoBaseTangent`, on the
        // normal this fragment writes), and is stored doubled again — (cos 2φ, sin 2φ) in .gb — so
        // it is sign-agnostic and wrap-free. Raw (0,0) is "no per-texel grain": it decodes to a
        // vector of length √2, off the unit circle, and the kernel keeps its base tangent.
        float2 grainEnc = float2(0.0);
        if (wTag > 1.001h && inst.grainTangentTextureSlice >= 0 &&
            length_squared(in.worldTangent.xyz) > 1e-4) {
            float4 gs = sampleAtlasHex(nonColorAtlas, texSampler, matUV,
                                       uint(inst.grainTangentTextureSlice), nonColorUVScale,
                                       nonColorSliceMean,
                                       frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
            float2 d2 = gs.xy * 2.0 - 1.0;
            float theta = 0.5 * atan2(d2.y, d2.x);
            float3 nGeo = normalize(in.worldNormal);
            float3 T = normalize(in.worldTangent.xyz);
            float3 B = cross(nGeo, T) * in.worldTangent.w;
            float3 Gw = normalize(cos(theta) * T + sin(theta) * B);
            float3 baseT = anisoBaseTangent(n, inst.anisotropy < 0.0f);
            float3 baseB = normalize(cross(n, baseT));
            float phi = atan2(dot(Gw, baseB), dot(Gw, baseT));
            grainEnc = float2(cos(2.0 * phi), sin(2.0 * phi)) * 0.5 + 0.5;
        }
        // .r thin-sheet translucency (slice 3): the kernel adds the light reaching the FAR face
        //    times this fraction; 0 = opaque, the live-lane behaviour.
        // .a per-material clearcoat roughness (slice 1); 0 = "not written" → the kernel's 0.08.
        o.material = half4(half(saturate(inst.thinTransmission)), half(grainEnc.x), half(grainEnc.y),
                           half(inst.clearcoat > 0.0 ? inst.clearcoatRoughness : 0.0));
    }
    // Screen-space motion vector. NDC.y is up, UV.y is down → Y is flipped.
    // The result is (currentUV - previousUV), so history reprojection in the
    // TAA kernel is `historyUV = currentUV - velocity`.
    float2 currNDC = in.currentClip.xy  / in.currentClip.w;
    float2 prevNDC = in.previousClip.xy / in.previousClip.w;
    // Both clip positions come from JITTERED matrices — this frame's VP and, via
    // `previousViewProjection`, last frame's — so their difference carries the
    // difference of two sample offsets on top of the surface's real motion. Take
    // it back out (`taaJitterDelta`), or the resolve realigns the history onto
    // the current sub-pixel sample and jitter supersampling cancels itself. Zero
    // whenever jitter is off ⇒ an IEEE no-op ⇒ byte-identical velocity.
    float2 velocityUV = ((currNDC - prevNDC) - frame.taaJitterDelta.xy) * float2(0.5, -0.5);
    o.velocity        = half2(velocityUV);
    o.layer           = in.layer;   // light-layer bitfield (default 0xFFFFFFFF)
    // DH-1300 — the slot of an instance with layer parameters (IlluminatoramaMaterialLayers.h).
    o.materialLayer   = illumiHasMaterialLayers(inst.materialExt, inst.specularTint)
                      ? in.instanceID : kMaterialLayerNone;
    return o;
    return o;
}
