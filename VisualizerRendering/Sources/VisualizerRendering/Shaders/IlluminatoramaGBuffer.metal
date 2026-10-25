// ── ILLUMINATORAMA — G-BUFFER PASS ──────────────────────────────────────────
//
// Vertex transform (wind + sway), the shadow-map vertex stage, and the G-buffer
// fragment shader that writes albedo / normal / material / velocity.

#include <metal_stdlib>
#include "IlluminatoramaGBufferShade.h"
#include "IlluminatoramaLeafCutout.h"
using namespace metal;

// The shading itself (vertex deformation + material evaluation + their helpers) lives in
// IlluminatoramaGBufferShade.h, shared with the Signack visibility buffer (DH-1461). This file is
// the conventional raster path's entry points.

vertex VSOut illumi_vs(
    uint                       vid           [[vertex_id]],
    uint                       iid           [[instance_id]],
    const device Vertex*       verts         [[buffer(0)]],
    constant FrameUniforms&    frame         [[buffer(1)]],
    const device Instance*     instances     [[buffer(2)]],
    const device Instance*     prevInstances [[buffer(4)]],
    // Only bound (and only exists) when kUsePrevVerts — marking the argument with
    // the function constant means the false variant declares no buffer(5) binding
    // at all, so the base pipeline never has to bind it (avoids the dead-binding
    // validation abort documented in metal_dead_buffer_binding_validation_abort).
    const device packed_float3* prevPositions [[buffer(5), function_constant(kUsePrevVerts)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    Instance prevInst = prevInstances[iid];

    // Tree wind displacement. THREE conditions, and the per-instance one is what keeps this
    // off ordinary geometry: `inst.windScale` is 0 unless the host declared this draw
    // vegetation, so a mesh carrying an honest TBN tangent (every normal-mapped surface —
    // a floor tangent of (1,0,0,1) reads as `sway = 1.0`, a canopy tip's weight) can no
    // longer be swung by the gust. `v.tangent.x` then selects WHICH vertices of a
    // vegetation mesh bend and by how much, which is all it was ever able to mean.
    //
    // Current uses `frame.time`; the PREVIOUS position uses `frame.time − windPrevDelta`
    // so the gust contributes a real screen-space velocity TAA can reproject (DH-0492) — without
    // it the sub-pixel canopy/blade edges crawled between pixels instead of reading as a
    // continuous bend. `windPrevDelta` is 0 for a SETTLED capture (its clock does not advance
    // between accumulation frames) ⇒ prevWindTime == frame.time ⇒ the static pose is unchanged,
    // byte-for-byte; it is the real per-frame clock delta only on the live, advancing canvas.
    // `windScale` is static per instance, so both arms read `inst` — taking it off `prevInst`
    // would spike the motion vector on the frame a draw first opts in.
    //
    // Previous-frame world position uses the previous-frame model matrix — captures
    // per-instance motion (spin, translation) on top of camera motion. The renderer
    // ping-pongs an instance buffer so prevInstances holds last frame's data at the same
    // instance index. For deforming GPU meshes (kUsePrevVerts) the previous OBJECT-space
    // position is the per-vertex value from last frame, not v.position, so the vertex motion
    // is captured too. The tangent goes to world through `modelMatrix` (not normalMatrix): it
    // lies along the surface and tracks non-uniform scale linearly. Drag/impact sway rotates
    // worldN/worldT with the lean; the previous frame gets LAST frame's sway. All of it lives
    // in `illumiDeform` (IlluminatoramaGBufferShade.h), shared with the visibility buffer.
    float3 prevObjP = v.position;
    if (kUsePrevVerts) {
        prevObjP = float3(prevPositions[vid]);
    }
    IllumiDeformed d = illumiDeform(v, prevObjP, inst, prevInst, frame);
    float4 worldP = d.worldP;
    float4 prevWorldP = d.prevWorldP;
    float3 worldN = d.worldN;
    float3 worldT = d.worldT;

    VSOut o;
    o.clipPos      = frame.viewProjection * worldP;
    o.worldPos     = worldP.xyz;
    o.worldNormal  = worldN;
    o.uv           = v.uv;
    o.instanceID   = iid;
    o.layer        = inst.layer;
    o.currentClip  = o.clipPos;
    o.previousClip = frame.previousViewProjection * prevWorldP;
    o.worldTangent = float4(worldT, v.tangent.w);
    o.vertexColor  = v.color;
    o.cardUV       = v.cardUV;
    return o;
}
// ── Shadow depth pass (Phase 2.5) ────────────────────────────────────────────
//
// Depth-only vertex shader, run once per cascade. The host binds the cascade's
// light-space view-projection matrix at buffer(3) — we don't read frame
// uniforms because the matrix changes per draw call. No fragment shader: the
// rasterizer writes depth into the assigned slice of the cascade array
// automatically.
static inline float4 illumiShadowClip(Vertex v, Instance inst, constant float4x4& lightVP, constant float4& shadowClock) {
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    // Match the visible pose: a swaying object casts its leaned shadow (no-op unless
    // swayMode != 0). Position-only — the depth pass needs no normal/tangent.
    // `shadowClock` mirrors frame.time + the frame's wind so a self-oscillating pendulum
    // (swayMode 2) or a wind-lifted flap (3) casts its swung shadow in phase with the mesh.
    float3 nDummy = float3(0.0), tDummy = float3(0.0);
    worldP.xyz = applySway(worldP.xyz, inst.modelMatrix, inst.swayMode,
                           inst.swayLean, inst.swayJostle, shadowClock.x, shadowClock.y, shadowClock.z, nDummy, tDummy);
    worldP.y -= fieldSinkOffset(inst.tagGlow.y, shadowClock.w, inst.tagGlow.z, inst.modelMatrix);
    return lightVP * worldP;
}

vertex float4 illumi_shadow_vs(
    uint                        vid       [[vertex_id]],
    uint                        iid       [[instance_id]],
    const device Vertex*        verts     [[buffer(0)]],
    const device Instance*      instances [[buffer(2)]],
    constant float4x4&          lightVP   [[buffer(3)]],
    constant float4&            shadowClock [[buffer(4)]]   // x = time, y = wind strength, z = wind heading, w = field sink
) {
    return illumiShadowClip(verts[vid], instances[iid], lightVP, shadowClock);
}

// DH-1487 — the leaf-cutout shadow stage: the same depth, plus a fragment that drops what lies
// outside the card's species leaf, so a cutout leaf casts its own outline (meshes flagged
// `leafCutout` only; every other caster keeps the fragment-less pipeline).
struct ShadowCutoutOut {
    float4 position [[position]];
    float2 cardUV;
};

vertex ShadowCutoutOut illumi_shadow_cutout_vs(
    uint                        vid       [[vertex_id]],
    uint                        iid       [[instance_id]],
    const device Vertex*        verts     [[buffer(0)]],
    const device Instance*      instances [[buffer(2)]],
    constant float4x4&          lightVP   [[buffer(3)]],
    constant float4&            shadowClock [[buffer(4)]]
) {
    Vertex v = verts[vid];
    ShadowCutoutOut o;
    o.position = illumiShadowClip(v, instances[iid], lightVP, shadowClock);
    o.cardUV = v.cardUV;
    return o;
}

fragment void illumi_shadow_cutout_fs(ShadowCutoutOut in [[stage_in]]) {
    if (!illumiLeafCutoutKeep(in.cardUV)) discard_fragment();
}

fragment GBufferOut illumi_fs(
    VSOut                                       in              [[stage_in]],
    bool                                        frontFacing     [[front_facing]],
    const device Instance*                      instances       [[buffer(2)]],
    // Phase 7 — per-frame uniforms, bound to the fragment stage so the G-buffer
    // shader can read `antiTilingStrength` (the opt-in hex-stochastic de-repeat
    // knob). Same buffer the vertex stage reads at buffer(1); default strength 0
    // makes every hex sample fall through to the plain single texture read, so
    // scenes that never opt in are byte-for-byte unchanged.
    constant FrameUniforms&                     frame           [[buffer(1)]],
    // Phase 4.25 — per-slice UV scale for the two atlases (letterbox fill
    // fraction; (1,1) = square). Indexed by the instance's slice id; drives
    // `sampleAtlasAspect`. `float2` (8-byte stride) matches the host's
    // `SIMD2<Float>` exactly.
    const device float2*                        albedoUVScale   [[buffer(3)]],
    const device float2*                        nonColorUVScale [[buffer(5)]],
    // S2.5 — per-slice mean (rgb) + validity (w) for each atlas, the μ the
    // variance-preserving hex blend rescales around. Same indexing as the UV-scale
    // tables; `float4` (16-byte stride) matches the host's `SIMD4<Float>`. A zeroed
    // entry (w == 0) means "unknown" and the blend stays linear, so a scene that
    // never registers a slice through the host path is unaffected.
    const device float4*                        albedoSliceMean [[buffer(7)]],
    const device float4*                        nonColorSliceMean [[buffer(8)]],
    // Interior mapping (only the kInteriorMap pipeline declares it).
    constant InteriorMapParams&                 imap            [[buffer(9), function_constant(kInteriorMap)]],
    // DH-1300 — the draw group's first record in the frame's instance ring (the ring is bound at the
    // group's offset, so `instance_id` is group-LOCAL), or 0xFFFFFFFF when the group's records are not
    // in the ring (a GPU-compacted batch, Signack S6): the material-layer slot is base + instance_id.
    constant uint&                              layerSlotBase   [[buffer(10)]],
    // Phase 4.0 — atlas of diffuse-albedo textures. Each slice is the
    // 512×512 BGRA8-sRGB upload from `IlluminatoramaTextureAtlas`; the
    // texture-format's sRGB→linear decode happens automatically inside
    // `sample()`, so the value we feed into albedo is already linear.
    texture2d_array<float, access::sample>      albedoAtlas     [[texture(0)]],
    // Phase 4.1 — atlas of linear, non-colour material maps. Same shape
    // as the albedo atlas but BGRA8-Unorm (no sRGB decode), so values
    // round-trip as the GPU's exact 0–1 linear quantity. R channel =
    // metallic, G channel = roughness (matches SCN's PBR convention of
    // packing both into one image, e.g. `roughnessMetallicAO.png`).
    texture2d_array<float, access::sample>      nonColorAtlas   [[texture(1)]]
) {
    // S1.1 — ONE screen-space UV derivative pair for the whole fragment, taken here in UNIFORM
    // control flow from the interpolated uv BEFORE any hex-hash offset, fract() wrap or
    // detail-frequency scale. Every atlas read hands these to gradient2d rather than letting the
    // texture unit infer a LOD from an offset uv. (The visibility-buffer resolve reconstructs the
    // same pair analytically.)
    float2 duvdx = dfdx(in.uv);
    float2 duvdy = dfdy(in.uv);
    float3 worldPosFwidth = fwidth(in.worldPos);
    // DH-1487 — the cutout variant: outside the card's species leaf, nothing is drawn.
    if (kLeafCutout && !illumiLeafCutoutKeep(in.cardUV)) {
        discard_fragment();
        GBufferOut none;
        return none;
    }
    InteriorMapParams ip = {};
    if (kInteriorMap) { ip = imap; }
    GBufferOut o = illumiGBufferShade(in, frontFacing, duvdx, duvdy, worldPosFwidth, instances, frame,
                              albedoUVScale, nonColorUVScale, albedoSliceMean, nonColorSliceMean,
                              ip, albedoAtlas, nonColorAtlas);
    // DH-1300 — make the material-layer slot global (see `layerSlotBase`).
    if (o.materialLayer != kMaterialLayerNone)
        o.materialLayer = (layerSlotBase == kMaterialLayerNone) ? kMaterialLayerNone : layerSlotBase + o.materialLayer;
    return o;
}
