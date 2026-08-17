// ── ILLUMINATORAMA — G-BUFFER PASS ──────────────────────────────────────────
//
// Vertex transform (wind + sway), the shadow-map vertex stage, and the G-buffer
// fragment shader that writes albedo / normal / material / velocity.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaMaterial.h"
using namespace metal;

// TEMP DIAGNOSTIC (forest-tex session): set to 1 to flat-colour Forest wood by
// species marker (oak=red birch=white maple=blue log=green cap=yellow). REVERT
// to 0 before counting any fix; this is a marker-contract probe only.
#define FOREST_SPECIES_DEBUG 0
// ── Hierarchical tree wind (#58 #1) ──────────────────────────────────────────
// Vertex-shader vegetation wind (GG3-style): each tree vertex carries
// (swayWeight, phase, flutter) in its tangent (packed by ForestGeometry). The
// trunk base has swayWeight 0 so it stays planted while the canopy sways
// (cantilever). Layered: macro height-weighted bend + a coarse TRAVELLING GUST
// envelope (wind moves across the stand in waves) + high-freq leaf flutter.
// Gated by windStrength > 0 → an exact no-op for every scene that doesn't set it;
// windHeading (frame._padPhase2B) is the dir. `windStrength` reaching here is the
// frame's global (repurposed frame._padPhase2A) TIMES the per-instance `windScale`,
// which is 0 unless the host declared the draw vegetation. That per-instance term is
// load-bearing: `windAttr` IS `v.tangent`, and a tangent is the TBN basis every
// normal-mapped surface carries, so on ordinary geometry `windAttr.x` is a surface
// direction, not a sway weight. Without the instance gate this function swings any
// mesh whose tangents happen to point +X.
static inline float3 applyTreeWind(float3 wp, float4 windAttr, float time,
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
    wp.y  -= sway * windStrength * 0.18 * fabs(macro);   // slight droop as it bends
    // Leaf flutter: high-freq shimmer on foliage. Kept SMALL (sub-card amplitude)
    // and at LOW spatial frequency so neighbouring outer-rind cards flutter
    // COHERENTLY — at the prior 0.15 coefficient the per-vertex chaotic jitter
    // (~3 cm vs ~8.5 cm cards) slid adjacent rind cards apart and re-opened the
    // sealed crown silhouette (side/top "flake cloud" regression). The macro sway
    // above stays coherent (shared tree phase), so the crown leans as one mass.
    if (flutter > 0.0001) {
        wp.x += flutter * windStrength * 0.05 * sin(time * 6.5 + phase * 4.0 + wp.y * 1.1);
        wp.z += flutter * windStrength * 0.05 * sin(time * 5.7 + phase * 3.0 + wp.x * 1.1);
        wp.y += flutter * windStrength * 0.035 * sin(time * 6.0 + phase * 3.5);
    }
    return wp;
}

// Issue #65 — per-vertex motion vectors for animated GPU meshes (fluid /
// marching-cubes / MLS-MPM surfaces). When false (the default for every
// ordinary instanced draw) `illumi_vs` is BYTE-FOR-BYTE its old self: the
// previous-frame object-space position is just `v.position`, so velocity comes
// purely from the per-instance model matrix delta (rigid spin/translation).
// When true (the dedicated G-buffer GPU-mesh draw specializes this on), the
// previous object-space position is read from a side `prevPositions` buffer the
// host blits each frame — so a deforming surface whose model matrix is identity
// still writes a real screen-space velocity (curr vertex − prev vertex). Both
// cases compose: a rigid GPU mesh has prevPos == currPos so velocity falls back
// to the matrix delta, exactly as before.
constant bool kUsePrevVerts [[function_constant(10)]];

// ── Drag/impact sway (generic rigid secondary motion) ────────────────────────
// The non-foliage sibling of applyTreeWind: a placed object the host is dragging
// (or that just knocked into something) leans + hops, driven entirely by the
// per-instance swayMode/swayLean/swayJostle the host's DragSwayTracker fills.
//
// Applied in WORLD space about the instance's bottom-pivot so it composes cleanly
// with the per-instance non-uniform scale already baked into modelMatrix (rotating
// unit-box object space then scaling would shear). Pivot = base centre of the
// box; lean axis = the object's local +Z in world (modelMatrix column 2). swayMode
// 0 ⇒ identity, so every non-swaying instance is an exact no-op.
//
// swayMode reference:
//   0 · none                — hard no-op (all vegetation instances that don't opt in).
//   1 · bottom-pivot lean   — rigid rotation about the box BASE (object y=-0.5) by the
//                             host-supplied static `lean` angle (books, upright shelf
//                             contents, dragged/knocked props). `lean` IS the angle.
//   2 · top-pivot pendulum  — rigid rotation about the model ORIGIN (object y=0), for a
//                             HANGING object (ceiling pendant). The pendant mesh is
//                             authored with its ceiling anchor at object y=0 and its body
//                             hanging DOWN into −Y, and it's placed by an unscaled matrix,
//                             so object y=0 is exactly the ceiling attach point — the
//                             pivot. Self-oscillates in the shader: angle = `lean` *
//                             sin(time·ω + phase), so `lean` here is the AMPLITUDE (max
//                             swing, radians) and the host sets it ONCE (static
//                             per-instance) — no per-frame drive. Phase is derived from
//                             the pivot's world XZ so neighbouring pendants swing out of
//                             step. Displacement is 0 at the top anchor and grows toward
//                             the hanging bottom (pivot about the top) — the inverse of
//                             mode 1 (which pivots at the box base y=-0.5).
//
// Returns the rotated world position; `worldN`/`worldT` are rotated in place by the
// same rigid rotation so lighting/normal-mapping track the lean.
static inline float3 applySway(float3 wp, float4x4 model, int mode,
                               float lean, float jostle, float time,
                               thread float3& worldN, thread float3& worldT) {
    if (mode == 0) return wp;
    // Mode 1 pivots at the box base (y=-0.5) and applies `lean` directly. Mode 2 pivots
    // at the model origin (y=0) — the hanging pendant's ceiling anchor — and
    // self-oscillates `lean` (as amplitude) from `time`.
    float  pivotY = (mode == 2) ? 0.0 : -0.5;
    float3 pivot  = (model * float4(0.0, pivotY, 0.0, 1.0)).xyz;
    float3 axis   = normalize((model * float4(0.0, 0.0, 1.0, 0.0)).xyz);  // local +Z in world
    float  angle  = lean;
    if (mode == 2) {
        // Gentle pendulum ~0.42 Hz (ω ≈ 2.65 rad/s); per-instance phase from the top
        // anchor's world XZ so a row of pendants doesn't swing in lock-step.
        float phase = pivot.x * 1.7 + pivot.z * 2.3;
        angle = lean * sin(time * 2.65 + phase);
    }
    float  c = cos(angle), s = sin(angle);
    // Rodrigues rotation of (wp - pivot) about `axis`, then restore pivot.
    float3 r = wp - pivot;
    float3 rot = r * c + cross(axis, r) * s + axis * dot(axis, r) * (1.0 - c);
    worldN = worldN * c + cross(axis, worldN) * s + axis * dot(axis, worldN) * (1.0 - c);
    worldT = worldT * c + cross(axis, worldT) * s + axis * dot(axis, worldT) * (1.0 - c);
    return pivot + rot + float3(0.0, jostle, 0.0);
}

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

    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    // Tree wind displacement. THREE conditions, and the per-instance one is what keeps this
    // off ordinary geometry: `inst.windScale` is 0 unless the host declared this draw
    // vegetation, so a mesh carrying an honest TBN tangent (every normal-mapped surface —
    // a floor tangent of (1,0,0,1) reads as `sway = 1.0`, a canopy tip's weight) can no
    // longer be swung by the gust. `v.tangent.x` then selects WHICH vertices of a
    // vegetation mesh bend and by how much, which is all it was ever able to mean.
    //
    // Same time for current + previous below: during a settled headless capture time is
    // frozen so the pose is static (no TAA smear); in the live app the per-frame delta is
    // tiny (gentle wind), matching the other deforming-geometry scenes. `windScale` is
    // static per instance, so both arms read `inst` — taking it off `prevInst` would spike
    // the motion vector on the frame a draw first opts in.
    float windStrength = frame._padPhase2A * inst.windScale;
    worldP.xyz = applyTreeWind(worldP.xyz, v.tangent, frame.time,
                               windStrength, frame._padPhase2B);
    float3 worldN = (inst.normalMatrix * float4(v.normal, 0.0)).xyz;
    // Previous-frame world position uses the previous-frame model matrix —
    // captures per-instance motion (spin, translation) on top of camera
    // motion. The renderer ping-pongs an instance buffer so prevInstances
    // holds last frame's data at the same instance index. For deforming GPU
    // meshes (kUsePrevVerts) the previous OBJECT-space position is the per-vertex
    // value from last frame, not v.position, so the vertex motion is captured too.
    float3 prevObjP = v.position;
    if (kUsePrevVerts) {
        prevObjP = float3(prevPositions[vid]);
    }
    float4 prevWorldP = prevInst.modelMatrix * float4(prevObjP, 1.0);
    prevWorldP.xyz = applyTreeWind(prevWorldP.xyz, v.tangent, frame.time,
                                   windStrength, frame._padPhase2B);
    // Phase 4.5 — transform the object-space tangent into world. Using
    // `modelMatrix` (not normalMatrix) because a tangent is along the
    // surface — it should track non-uniform scale linearly, unlike a
    // normal. Handedness in .w rides along unchanged.
    float3 worldT = (inst.modelMatrix * float4(v.tangent.xyz, 0.0)).xyz;

    // Drag/impact sway — rigid lean+hop driven by the host DragSwayTracker, no-op
    // unless swayMode != 0. worldN/worldT rotate with it so lighting + normal maps
    // track the lean. The previous frame gets LAST frame's sway (prevInst) so the
    // motion vector captures the swing (TAA/motion-blur correctness).
    worldP.xyz = applySway(worldP.xyz, inst.modelMatrix, inst.swayMode,
                           inst.swayLean, inst.swayJostle, frame.time, worldN, worldT);
    float3 prevN = worldN, prevT = worldT;   // throwaway: prev normal/tangent unused
    // Prev frame uses the same `frame.time` as current (matching applyTreeWind above):
    // during a settled headless capture time is frozen (static pose, no TAA smear); in
    // the live app the per-frame swing delta is tiny, so the motion vector stays clean.
    prevWorldP.xyz = applySway(prevWorldP.xyz, prevInst.modelMatrix, prevInst.swayMode,
                               prevInst.swayLean, prevInst.swayJostle, frame.time, prevN, prevT);

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
    return o;
}

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
};

// ── Shadow depth pass (Phase 2.5) ────────────────────────────────────────────
//
// Depth-only vertex shader, run once per cascade. The host binds the cascade's
// light-space view-projection matrix at buffer(3) — we don't read frame
// uniforms because the matrix changes per draw call. No fragment shader: the
// rasterizer writes depth into the assigned slice of the cascade array
// automatically.
vertex float4 illumi_shadow_vs(
    uint                        vid       [[vertex_id]],
    uint                        iid       [[instance_id]],
    const device Vertex*        verts     [[buffer(0)]],
    const device Instance*      instances [[buffer(2)]],
    constant float4x4&          lightVP   [[buffer(3)]],
    constant float&             shadowTime [[buffer(4)]]
) {
    Vertex v = verts[vid];
    Instance inst = instances[iid];
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    // Match the visible pose: a swaying object casts its leaned shadow (no-op unless
    // swayMode != 0). Position-only — the depth pass needs no normal/tangent.
    // `shadowTime` mirrors frame.time so a self-oscillating pendulum (swayMode 2) casts
    // its swung shadow in phase with the visible mesh.
    float3 nDummy = float3(0.0), tDummy = float3(0.0);
    worldP.xyz = applySway(worldP.xyz, inst.modelMatrix, inst.swayMode,
                           inst.swayLean, inst.swayJostle, shadowTime, nDummy, tDummy);
    return lightVP * worldP;
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
    Instance inst = instances[in.instanceID];
    float3 n = normalize(in.worldNormal);
    // Two-sided meshes (cull .none) rasterize back faces too; flip the normal
    // so they're lit as a real surface instead of going dark/hollow. For
    // single-sided meshes back faces are culled, so this is always front (no-op).
    if (!frontFacing) { n = -n; }

    GBufferOut o;
    constexpr sampler texSampler(filter::linear,
                                 mip_filter::linear,
                                 address::repeat,
                                 // S1.1 — 8x. The archviz camera's whole problem is the
                                 // grazing floor, where the footprint is a long thin
                                 // ellipse and trilinear picks the mip fitting its MAJOR
                                 // axis, over-blurring the minor one by the same ratio.
                                 // 8x recovers ~3 mip levels along the minor axis. Not
                                 // 16x: the hex blend already triples every atlas read.
                                 // Apple GPUs spend taps proportional to the ACTUAL
                                 // anisotropy, not the cap, so head-on surfaces pay
                                 // nothing.
                                 max_anisotropy(8));

    // S1.1 — ONE screen-space UV derivative pair for the whole fragment, taken here in
    // UNIFORM control flow from the interpolated uv BEFORE any hex-hash offset, fract()
    // wrap or detail-frequency scale. Every atlas read below hands these to gradient2d
    // rather than letting the texture unit infer a LOD from an offset uv. Taking them
    // inside the `if (inst.*TextureSlice >= 0)` branches instead would be undefined:
    // quad-mate lanes can be inactive there.
    float2 duvdx = dfdx(in.uv);
    float2 duvdy = dfdy(in.uv);

    // Phase 4.5 — tangent-space normal-map sampling. The atlas is the
    // same `bgra8Unorm` non-colour atlas as metallic/roughness; the
    // normal map encodes tangent-space (x,y,z) into [0,1] via
    // `0.5*(N+1)`. Decode with `2*sample - 1`, then transform into
    // world via TBN. When no normal map is bound, fall through to the
    // geometric normal `n`.
    // ── S2.4 — the detail band's diffuse-visible companion ────────────────────
    // Micro-occlusion sampled from the detail slice's BLUE channel and applied to the
    // albedo/roughness THIS fragment is about to write. Declared out here (not inside the
    // normal-map branch) so the value survives to the albedo and roughness blocks below.
    // 1.0 = fully open = the exact pre-S2.4 result; every path that doesn't bind a detail
    // slice leaves it at 1.0, and ×1.0 / +0.0 are IEEE no-ops.
    float detailOcc = 1.0;
    const bool wantsDetailRelief = (inst.detailOcclusionStrength > 0.0 ||
                                    inst.detailRoughnessStrength > 0.0);

    if (inst.normalTextureSlice >= 0 &&
        length_squared(in.worldTangent.xyz) > 1e-4) {
        float3 T = normalize(in.worldTangent.xyz);
        float3 B = cross(n, T) * in.worldTangent.w;
        // Hex-stochastic normal map — eliminates the regular tiling pattern
        // on large flat surfaces (walls, floors) by blending three stochastically-
        // offset lattice samples. The blend is linear in encoded space (before
        // decode), which slightly overshoots tangent-space magnitude at boundaries
        // but gives correct appearance after renormalize below.
        float4 nmSample = sampleAtlasHex(nonColorAtlas, texSampler, in.uv,
                                         uint(inst.normalTextureSlice), nonColorUVScale,
                                         nonColorSliceMean,
                                         frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
        float3 tangentN = normalize(nmSample.xyz * 2.0 - 1.0);
        // Phase 7 detail normal — blended on top of the macro normal at
        // higher UV frequency (pores, weave, grain). Uses overlay-normal
        // blend: partial-derivative add in tangent space, then renormalize.
        if (inst.detailNormalTextureSlice >= 0) {
            float2 detailUV = in.uv * inst.detailNormalUVScale;
            // S1.1 — the detail band is handed the UNSCALED gradient pair ON PURPOSE, and
            // this is a deliberate, measured exception to the rule the rest of this shader
            // follows.
            //
            // The mip-correct thing is to scale by `detailNormalUVScale` (8×) to match the
            // frequency actually being sampled. I did that first, and it BROKE the band:
            // `testDetailNormalAddsGrazingMicroRelief` collapsed from ~1.4× to 1.0002× on
            // aluminum and 0.97× on brushed-nickel — i.e. detail normals stopped doing
            // anything at all. Correct LOD for an 8×-frequency band means selecting a mip
            // ~3 levels coarser, which averages the pore/weave/grain perturbation to flat.
            // The band exists precisely to survive there, and per the 2026-08-02
            // whole-library A/B it is the ONE thing that measurably moves metals.
            //
            // Passing the unscaled pair makes this band sample about as sharply as it did
            // when the atlas had no mip chain at all — so it is the STATUS QUO for detail
            // normals, not a new under-filtering defect, while every other atlas read gets
            // proper mips. The honest cost is that this band can still alias at grazing
            // angles.
            //
            // S2.4 UPDATE (2026-08-06): that companion now exists (see just below), and it
            // does NOT retire this exception — it explains it. The companion survives
            // filtering by running at a resolvable frequency; this band does not, because at
            // 8x on a 2 m macro tile a texel is ~0.5 mm. So what "correct LOD kills the
            // effect" actually means here is that **this band's contribution is sub-pixel
            // aliasing** — which is also why it reads on mirror-like metal and on nothing
            // else. Retiring the exception means moving the detail NORMAL to a resolvable
            // frequency too, which is a look change, not a filtering fix.
            float4 dnSample = sampleAtlasAspect(nonColorAtlas, texSampler, detailUV,
                                                uint(inst.detailNormalTextureSlice), nonColorUVScale,
                                                duvdx, duvdy);
            float3 dn = dnSample.xyz * 2.0 - 1.0;
            // NOTE: `dn.z` is deliberately not used — the blend is a partial-derivative add
            // in tangent space, so blue was always dead weight here. S2.4 puts the detail
            // occlusion in it (see below).
            tangentN = normalize(float3(tangentN.xy + dn.xy, tangentN.z));

            // S2.4 — the occlusion is a SECOND tap of the same slice, at its own frequency
            // and with MIP-CORRECT gradients. Both differences are load-bearing:
            //
            //  * Mip-correct, because occlusion is a non-negative quantity: a coarser mip
            //    averages it to its MEAN, so it degrades into a slight uniform darkening
            //    rather than vanishing (the normal's failure mode) or aliasing (the
            //    normal's cost — and the whole reason S1.1 had to hand THIS band unscaled
            //    gradients). Reading it off `dnSample` would inherit that defect.
            //  * Its own frequency, because mip-correct sampling at the NORMAL's 8× shows
            //    nothing: measured, that band is ~0.5 mm per texel on a 2 m macro tile, so
            //    correct filtering is doing exactly its job when it returns a flat mean.
            //    Occlusion has to run where the camera can resolve it to read as relief.
            if (wantsDetailRelief) {
                float occScale = inst.detailOcclusionUVScale > 0.0
                               ? inst.detailOcclusionUVScale : inst.detailNormalUVScale;
                float2 occUV = in.uv * occScale;
                float4 occSample = sampleAtlasAspect(nonColorAtlas, texSampler, occUV,
                                                     uint(inst.detailNormalTextureSlice), nonColorUVScale,
                                                     duvdx * occScale, duvdy * occScale);
                detailOcc = occSample.z;   // [0,1], 1 = open. 255-blue slices ⇒ exactly 1.
            }
        }
        // World normal = T*x + B*y + N*z. Equivalent to a TBN matrix
        // multiply but cheaper to spell out as a single dot per axis.
        n = normalize(T * tangentN.x + B * tangentN.y + n * tangentN.z);
    }

    float3 albedo = inst.albedo;
    if (inst.albedoTextureSlice >= 0) {
        // Phase 7 hex-stochastic anti-tiling: floor/wall surfaces tile seamlessly but
        // don't repeat visibly — three blended samples from stochastically-offset
        // lattice cells. `in.uv` is already in tile-space (world ÷ uvMetres), so
        // the hex cell size equals one texture tile. sampleAtlasAspect handles
        // letterbox padding per-slice; the offset just shifts into a neighbouring
        // infinitely-tiling region, so aspect is preserved.
        float4 tx = sampleAtlasHex(albedoAtlas, texSampler, in.uv,
                                    uint(inst.albedoTextureSlice), albedoUVScale,
                                    albedoSliceMean,
                                    frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
        albedo = tx.rgb;
        // S2.5 half 2 — per-PATTERN-CELL value jitter, the COHERENT categories' de-repeat
        // (full contract on `Instance.patternCells` in IlluminatoramaCommon.h). The hex
        // blend above is invalid for a tile grid or a plank run (it superimposes misaligned
        // copies), so those materials ship antiTilingScale = 0 and, until this line, had no
        // de-repeat at all. `in.uv` counts up across the whole surface — only the atlas
        // LOOKUP wraps — so `floor(uv * patternCells)` indexes each physical tile / plank /
        // course uniquely over the entire floor, and a ± tone multiplier hashed from it
        // breaks the macro repeat with no UV displacement (grout and seams stay put) and no
        // extra texture tap. A bonded axis ships patternCells = 0 there: floor(uv·0) = 0
        // keeps that axis's index constant, so a running-bond COURSE varies as a whole and
        // a cell boundary never cuts through an offset tile mid-body.
        // `patternJitter == 0` — the default, and every scene that never opts in — is an
        // exact no-op, so Visualizer is byte-identical.
        if (inst.patternJitter != 0.0f) {
            albedo *= 1.0f + inst.patternJitter * patternCellHash(floor(in.uv * inst.patternCells));
        }
    }
    // Phase 4.17 — modulate albedo by per-vertex color (default white,
    // so a no-op for meshes that ship no .color source). This is where
    // pattern detail painted into the vertex stream (chevron stripes,
    // candy gradients, anything fed via SCNGeometrySource(semantic:
    // .color)) actually paints the fragment. SceneKit convention is
    // multiplicative: vertex colour and material diffuse compose.
    albedo *= in.vertexColor.rgb;

    float metallic = inst.metallic;
    if (inst.metallicTextureSlice >= 0) {
        // SCN-style packed metallic-roughness usually puts metallic in B
        // (Blue) and roughness in G — but the exact channel depends on
        // how the asset was authored. Falling back to R keeps single-
        // channel grayscale metallic maps working without per-scene wiring.
        float4 tx = sampleAtlasAspect(nonColorAtlas, texSampler, in.uv,
                                      uint(inst.metallicTextureSlice), nonColorUVScale, duvdx, duvdy);
        metallic = tx.r;
    }

    float roughness = inst.roughness;
    if (inst.roughnessTextureSlice >= 0) {
        // Hex-stochastic roughness — same three-cell blend as albedo so
        // roughness variation doesn't lag the albedo tile seam.
        float4 tx = sampleAtlasHex(nonColorAtlas, texSampler, in.uv,
                                    uint(inst.roughnessTextureSlice), nonColorUVScale,
                                    nonColorSliceMean,
                                    frame.antiTilingStrength * inst.antiTilingScale, duvdx, duvdy);
        roughness = tx.g;
    }

    // ── S2.4 detail-band relief: apply the micro-occlusion ────────────────────
    // Placed here because both terms land on quantities the G-buffer already writes —
    // albedo (color 0) and roughness (color 1.z). NOTHING new is carried to the lighting
    // pass. That was the design constraint: `normalRoughness.w` is already a
    // material-class tag (foliage / SSS band / plush / 1+anisotropy) and has no room, and
    // a fifth G-buffer target would cost every pixel of every frame for a close-range
    // effect. In a deferred renderer a micro-AO on albedo IS a diffuse-only term for a
    // dielectric (F0 is the fixed 0.04, not the albedo), which is exactly the lobe the
    // detail normal could not reach.
    //
    // `detailOcc` is 1.0 unless a detail slice was bound AND the instance opted in, so
    // this is an exact no-op for every existing scene.
    if (wantsDetailRelief) {
        // Micro-shadowing: the pits between pores/weave/granules lose part of their
        // hemisphere to their own rim.
        albedo *= mix(1.0, detailOcc, saturate(inst.detailOcclusionStrength));
        // Micro-cavity roughening: the same pits scatter more widely than the open
        // surface between them. Survives mip filtering (see the sample site above).
        roughness = saturate(roughness + inst.detailRoughnessStrength * (1.0 - detailOcc));
    }

    // ── Procedural soil material (#58 #11/#12/#13) ───────────────────────────
    // Ground vertices pack soil data into uv as a NEGATIVE-x marker
    // (uv.x = -(roughness + 0.01), uv.y = wetness). Real UVs are never negative
    // and the branch is gated to UNTEXTURED instances, so this is an exact no-op
    // for every other surface and every other scene. Delivers per-region
    // roughness (#12), wetness darkening/smoothing (#12 — wet = darker, smoother,
    // more reflective), and world-space macro/micro normal relief (#13, #11 —
    // world-space so no UV stretching on slopes).
    if (inst.albedoTextureSlice < 0 && inst.roughnessTextureSlice < 0 &&
        inst.normalTextureSlice < 0 && in.uv.x < -0.005) {
        float soilRough = saturate(-in.uv.x - 0.01);
        float wet = saturate(in.uv.y);
        roughness = mix(soilRough, soilRough * 0.45, wet);   // wet → smoother + reflective
        albedo *= mix(1.0, 0.82, wet);                       // gentle extra darken (vertex colour already cools hollows)
        float2 grad = soilNormalGrad(in.worldPos.xz);
        // Dry ground takes more micro-relief; wet (pooled) ground reads smoother.
        float relief = 0.32 * (1.0 - 0.55 * wet);
        n = normalize(n + float3(grad.x, 0.0, grad.y) * relief);
        // ── GRASS-BASE MOTTLE (round-5 polish): a 20–40 cm value/hue mottle on the
        // ground ALBEDO everywhere (clearing + far) so the mossy floor between blades
        // stops reading as one flat green. Two octaves: ~33 cm patches + ~17 cm. Pulls
        // value ±18 % and shifts a fraction toward yellow-green / damp-dark so the
        // sward base has patchy life under the blades. No-op off the soil marker.
        {
            float2 gp = in.worldPos.xz;
            float mo0 = soilValueNoise(float2(gp.x * 3.0, gp.y * 3.0));   // ≈33 cm
            float mo1 = soilValueNoise(float2(gp.x * 6.0 + 17.0, gp.y * 6.0 - 9.0)); // ≈17 cm
            float mottleV = (mo0 * 0.65 + mo1 * 0.35);
            albedo *= 0.84 + 0.34 * mottleV;                             // value patches
            float3 damp = albedo * float3(0.78, 0.92, 0.70);             // damp-dark green
            float3 dryG = albedo * float3(1.12, 1.08, 0.74);             // yellow-green
            albedo = mix(albedo, damp, smoothstep(0.62, 0.30, mo0) * 0.35);
            albedo = mix(albedo, dryG, smoothstep(0.66, 0.88, mo1) * 0.30);
        }
        // ── FOREST-FLOOR DUFF (round 3, target 4): per-PIXEL leaf-litter ─────────
        // The vertex-baked duff (ForestGeometry.groundColor) can only carry ≈0.6 m
        // features (192-res / 120 m grid), so beyond the grass ring the floor read as
        // FLAT GREEN FELT at the hero/side cameras. Add per-pixel litter HERE where it
        // survives distance: BIG (≈8–15 cm) high-contrast tan/brown/dull-orange oval
        // blotches (x-stretched domain ⇒ leaf-shaped, not round dots), gated to the
        // out-of-clearing band (radius > ≈12 m from the clearing centre at z=−3) so
        // the mossy clearing floor stays clean. World-space → no swim, no UV stretch.
        float2 fp = in.worldPos.xz;
        float rad = length(float2(fp.x, fp.y + 3.0));
        // ROUND-5 floor fix: the band between the grass ring and the treeline
        // (≈8–16 m) read as FLAT SATURATED GREEN — the litter started too far out
        // (10→18 m) and was too weak to survive against the bright mossy-green
        // vertex base there. Start the duff sooner (the mossy CLEARING CENTRE at
        // z=−3 stays clean because the clearing radius is ≈9 m and the litter only
        // reaches full past ≈14 m), and make it strong enough to read.
        // ROUND-8: the 8→14 ramp left a HALF-treated annulus (litter at 20–60%)
        // exactly where the hero camera sees substrate past the grass ring, and it
        // still read as green felt (the top view proves duff fully lands past
        // 14 m). Tighten the ramp so full duff arrives where the dense sward ends.
        float farBand = smoothstep(7.0, 10.0, rad);    // 0 in clearing → 1 past 10 m
        if (farBand > 0.001) {
            // First, DESATURATE the bright-green vertex base toward a duff floor so
            // the litter has a neutral ground to sit on (the green felt was the base
            // colour, not the litter). Pull the saturated green toward a dull
            // olive-brown forest-floor tone, scaled by the band.
            float baseLum = dot(albedo, float3(0.299, 0.587, 0.114));
            // ROUND-8b: the old near-constant duffBase (0.07–0.10 regardless of
            // incoming value) compressed the vertex-baked MACRO patches (2–6 m
            // damp/dry swings — the only floor variation that survives 25 m +
            // atmosphere) down to ~22% of their range. Preserve incoming
            // LUMINANCE and shift only the HUE toward duff-brown, so the macro
            // value patchiness rides through the desaturation.
            float3 duffBase = float3(0.118, 0.096, 0.056) * saturate(baseLum * 3.0);
            albedo = mix(albedo, duffBase, farBand * 0.78);
            // Oval litter mask: x-stretched noise domain so blotches read as leaves.
            // Wider duty (0.42–0.78) so litter blotches are COMMON, not sparse.
            float lit = soilValueNoise(float2(fp.x * 9.0, fp.y * 4.5));   // ≈11/22 cm
            float lit2 = soilValueNoise(float2(fp.x * 3.2 + 50.0, fp.y * 3.2 - 20.0)); // coarse drifts
            float litMask = smoothstep(0.42, 0.74, lit) * (0.55 + 0.45 * lit2) * farBand;
            float litHue = soilValueNoise(float2(fp.x * 5.0 + 13.0, fp.y * 5.0 + 7.0));
            float3 litTan    = float3(0.170, 0.120, 0.058);
            float3 litOrange = float3(0.205, 0.105, 0.034);
            float3 litBrown  = float3(0.085, 0.058, 0.032);
            float3 litCol = (litHue < 0.4)
                ? mix(litBrown, litTan, litHue / 0.4)
                : mix(litTan, litOrange, (litHue - 0.4) / 0.6);
            albedo = mix(albedo, litCol, litMask * 0.80);
            // A few surviving green moss patches break the brown (not pure duff).
            float moss = smoothstep(0.74, 0.90, lit2) * farBand;
            albedo = mix(albedo, float3(0.070, 0.115, 0.045), moss * 0.5);
            // Twig/duff micro-normal: a coarser bump on the littered patches.
            float2 lg = soilNormalGrad(fp * 6.0);
            n = normalize(n + float3(lg.x, 0.0, lg.y) * (0.45 * litMask));
            roughness = mix(roughness, 0.92, litMask);
        }
    }

    // ── Procedural SPECIES-CORRECT bark + mossy-log surface (#58 / round 3) ──
    // ForestGeometry marks WOOD in worldTangent.w with a SPECIES/material code:
    //   2 = oak (deep furrowed PLATES)   3 = birch (pale + lenticel dashes)
    //   4 = maple (shaggy lifting strips) 5 = mossy fallen LOG (moss velvet)
    // Leaves are 1, everything else 0/±1, and NO other scene emits w > 1.5 (every
    // non-Forest mesh ships tangent .zero or, from this same file, wind w==1/2 on
    // the *same* Forest scene only). So `w > 1.5` gates cleanly to Forest wood and
    // is an exact no-op for foliage, ground, every prop without a marker, and every
    // other scene. The detail is COMPUTED (the soup has no texture atlas) and is far
    // crisper than the coarse 18-radial trunk mesh could carry.
    if (inst.normalTextureSlice < 0 && in.worldTangent.w > 1.5) {
        float species = in.worldTangent.w;                 // 2/3/4/5
        float ang = atan2(n.z, n.x);                       // sweeps around the trunk
        float3 tang = normalize(cross(n, float3(0.0, 1.0, 0.0)) + float3(1e-4, 0, 0));
        float yW = in.worldPos.y;
        float e  = 0.14;

#if FOREST_SPECIES_DEBUG
        // TEMP marker probe: flat colour per species marker so the contract
        // (geometry stamp vs shader gate) can be read off one render.
        // oak2=red, birch3=white, maple4=blue, log5=green, cap6=yellow.
        if (species < 2.5)      albedo = float3(1.0, 0.0, 0.0);
        else if (species < 3.5) albedo = float3(1.0, 1.0, 1.0);
        else if (species < 4.5) albedo = float3(0.0, 0.3, 1.0);
        else if (species < 5.5) albedo = float3(0.0, 1.0, 0.0);
        else                    albedo = float3(1.0, 1.0, 0.0);
#else
        if (species < 2.5) {
            // ── OAK — deep furrows broken into discrete shedding PLATE CELLS ────
            // ROUND-3 RE-ARCHITECT (the previous furrow-only field still read as a
            // smooth airbrushed gradient in hero_oakbark_tight — no plate structure
            // survived to 3–6 m). The fix the brief asks for is a real CELL field:
            // a Worley F1 lattice (barkCellF1) carves the trunk into discrete
            // plates, the inter-cell DISTANCE drives a dark furrow crack between
            // them, and each cell takes a COHERENT ridge tint by its per-cell ID.
            // The lattice is ANISOTROPIC (ang×3.4 around, yW×1.1 up ⇒ ≈18 cm wide ×
            // ≈30 cm tall plate cells — oak-correct vertical-ish plates) so it reads
            // as stacked vertical bark scales, not bubbles. A vertical furrow octave
            // grooves WITHIN each plate; a fine octave adds the close-up grain.
            float2 cellP = float2(ang * 3.4, yW * 1.1);
            float2 cell = barkCellF1(cellP);
            float crackDist = cell.x;                 // small ⇒ near a plate crack
            float cellID    = cell.y;                 // coherent per-plate random
            // Crack mask: 1 deep in a plate, 0 in the dark furrow border between
            // plates. Tight band so plates have flat lit faces + sharp dark grooves.
            float plate = smoothstep(0.06, 0.24, crackDist);
            // Vertical grooving within the plate (oak's lengthwise fissures).
            float2 vc  = float2(ang * 9.0, yW * 0.7);
            float vf   = soilValueNoise(vc);
            float vfa  = soilValueNoise(vc + float2(e, 0));
            float vfy  = soilValueNoise(vc + float2(0, e));
            float vGroove = smoothstep(0.38, 0.60, vf);
            float fine = soilValueNoise(float2(ang * 38.0, yW * 9.0)) - 0.5;
            // ── ROUND-5 STRUCTURAL #1: DOMAIN-WARPED BRICK-OFFSET HORIZONTAL CRACKS ─
            // The round-4 field (`hPhase = yW*7 + lowFreqAngNoise`) used a phase that
            // varied only SLOWLY with `ang` (×1.7), so adjacent furrow columns shared
            // nearly the same crack height — the horizontal cracks LINED UP across the
            // trunk into continuous rows. Combined with the regular ×9 vertical groove
            // that reads as a KNITTED MESH / SNAKESKIN grid (the basket-weave defect).
            //
            // The fix the reviewer prescribes: per-furrow-COLUMN brick-offset so plate
            // seams stagger between adjacent columns (never align into a grid), plus
            // low-freq jitter on BOTH crack spacing AND amplitude so some plates are
            // tall, some squat, and some cracks fade out entirely — cells read as
            // IRREGULAR polygons. Derive a discrete column index from `ang`, hash it
            // for an independent phase OFFSET (the brick stagger) and a per-column row
            // SPACING + a per-column crack DEPTH, so no two columns share a row line.
            float colW = 6.5;                                  // ~6–7 furrow columns/m circumference
            float colF = ang * colW;
            float colIdx = floor(colF);                        // furrow-column index
            // Brick stagger: each column gets its own phase offset in [0,1) so the
            // crack rows of neighbouring columns never align into a continuous course.
            float colPhase = soilHash(float2(colIdx, 3.1));
            // Per-column row SPACING jitter (±35 %): some columns stack squat plates,
            // some tall — so the horizontal seams don't form a regular ladder.
            float colSpace = 5.5 + 4.0 * soilHash(float2(colIdx, 11.7));   // 5.5–9.5 rows/m
            // Per-column crack DEPTH jitter: some columns' cracks fade out (shallow),
            // so a few plates merge vertically — irregular polygon cells, not a weave.
            // ROUND-5: raise the FLOOR (0.62) so EVERY column shows real horizontal
            // cracks — the prior 0.45 floor let too many columns merge vertically, so
            // the trunk read as continuous combed strands (shaggy maple) instead of
            // discrete stacked OAK plates. A few still merge (0.62→1.0), keeping the
            // irregular-polygon read without losing the plate-row structure.
            float colDepth = 0.62 + 0.38 * soilHash(float2(colIdx, 23.9));
            // Smoothly blend the per-column phase/spacing across the column boundary so
            // the brick offset doesn't itself read as a hard vertical seam line.
            float colMix = smoothstep(0.0, 1.0, fract(colF));
            float colPhaseB = soilHash(float2(colIdx + 1.0, 3.1));
            float colSpaceB = 5.5 + 4.0 * soilHash(float2(colIdx + 1.0, 11.7));
            float colDepthB = 0.45 + 0.55 * soilHash(float2(colIdx + 1.0, 23.9));
            float phase = mix(colPhase, colPhaseB, colMix);
            float space = mix(colSpace, colSpaceB, colMix);
            float crackDepthCol = mix(colDepth, colDepthB, colMix);
            // Low-freq AMPLITUDE jitter on top (slow in y) so a plate is taller here,
            // squatter there even within a column run.
            float ampJ = 0.7 + 0.6 * soilValueNoise(float2(colIdx * 0.7, yW * 0.6));
            float hPhase = (yW * space) * ampJ + phase * 1.31;
            float hTri = abs(fract(hPhase) - 0.5) * 2.0;        // 0 at crack → 1 mid-cell
            // crackDepthCol scales the crack WIDTH: shallow columns barely groove (the
            // "some cracks fade out" cue) so plates merge into irregular polygons.
            // ROUND-5: WIDEN the crack band (0.10→0.34) so the dark horizontal seam is
            // fat enough to read at hero range and physically separate the plate rows.
            float hCrack = mix(1.0, smoothstep(0.12, 0.34, hTri), crackDepthCol);
            // Combined "is this lit ridge-top vs dark damp floor": a pixel reads as
            // ridge only when inside a plate AND on a vertical ridge AND not in a
            // horizontal crack — so the ridges break into discrete blocky plate cells.
            // ROUND-6 STRUCTURAL #2 (oak side): the round-5 field STILL read as soft
            // painted streaks at hero distance — the horizontal cracks barely resolved
            // into discrete stacked plate CELLS. Two reasons: (a) the `ridge` value
            // floored at plate·0.45·0.18≈0.08 so the cracks never reached a true dark
            // floor, and (b) the albedo split (crackTint 0.22 → ridgeTint ~1.3) was a
            // ~6× spread but applied to a non-zero `ridge` ⇒ washed grey, not damp-dark
            // vs grey-lifted. Push BOTH: zero the floors so cracks cut to a true dark
            // furrow floor, and split harder (darker damp floor, brighter ridge top),
            // so each plate reads as a discrete lit cell with sharp dark seams.
            // The horizontal crack now FULLY gates the ridge (multiplicative, no floor)
            // so a horizontal seam carves the vertical ridge into stacked plate cells —
            // the oak-vs-maple tell the reviewer keeps flagging.
            float ridge = plate * (0.20 + 0.80 * vGroove) * hCrack;
            // ≈5× luminance split: weathered grey-lifted ridge tops ↔ DARK damp furrow
            // floors. Per-plate value jitter so adjacent plates differ.
            float plateVal = 0.78 + 0.5 * cellID;          // 0.78..1.28 per plate
            float3 ridgeTint = float3(1.44, 1.37, 1.22) * plateVal;  // bright weathered grey
            float3 crackTint = float3(0.135, 0.105, 0.075);          // dark damp furrow floor
            albedo *= mix(crackTint, ridgeTint, ridge) * (0.90 + 0.22 * fine);
            // Sparse lichen blotches — pale grey-green, slightly SMOOTHER — only on
            // the lit plate tops, keyed to whole plates (per-cell) so they blotch a
            // plate, not a pixel.
            float lichSel = soilHash(floor(cellP) + 5.1);
            float lich = soilValueNoise(float2(ang * 3.5 + 31.0, yW * 0.9 + 9.0));
            float lichM = step(0.82, lichSel) * smoothstep(0.55, 0.85, lich) * plate;
            albedo = mix(albedo, float3(0.42, 0.47, 0.36), lichM * 0.6);
            roughness = mix(roughness, roughness * 0.82, lichM);
            // Lift plate edges: push the dark cracks IN (recessed grooves) and tilt
            // the vertical grooving within plates.
            float gA = (vfa - vf) / e, gY = (vfy - vf) / e;
            float edge = 1.0 - plate;                       // 1 at the vertical plate crack
            float hEdge = 1.0 - hCrack;                     // 1 at the horizontal crack
            // ROUND-6: deepen the recess on BOTH crack axes so the plate edges lift in
            // relief (discrete cells under the rake light), and tilt the lit ridge top
            // OUTWARD along +n so the grey ridge faces the camera (plate-top read).
            n = normalize(n + tang * (gA * 1.1 + fine * 0.45)
                            + float3(0,1,0) * (gY * 0.30)
                            + n * (ridge * 0.22)                  // ridge tops lift toward camera
                            - n * (edge * 0.55 + hEdge * 0.55));  // recess BOTH crack axes deeper
        } else if (species < 3.5) {
            // ── BIRCH — pale paper bark, DARK HORIZONTAL lenticel dashes ────────
            // ROUND-3 RETUNE: at 3–10 m the old dashes (ang×22, smoothstep 0.62–0.80
            // — a narrow ~18 % duty) were too sparse + too thin to survive, and the
            // warm paper tint pushed the trunk pinkish-tan. Now: FATTER, ~3× DENSER,
            // higher-CONTRAST near-black dashes (lower angular freq ≈10–14 cm dashes,
            // wider duty), a brighter NEUTRAL-COOL paper white, and a dark peel band.
            float2 lc  = float2(ang * 2.4, yW * 0.9);          // band selector (big)
            float band = soilValueNoise(lc);
            // Dash field: MODERATE angular freq (fat dashes) broken into short
            // segments by a vertical mask. Wider smoothstep windows → ~3× the duty
            // cycle of the old field so dashes are common, not rare.
            // ROUND-5 polish #5: hero-facing birch dashes read too FAINT vs the
            // side-view birch that passes — at hero distance the moderate duty + 0.94
            // mix wasn't surviving. Widen the duty windows (denser dashes), and raise
            // the band coverage so dashes appear over more of the trunk, so hero
            // birches match the side one.
            float dashRun = soilValueNoise(float2(ang * 12.0, yW * 0.5));
            float dashGap = soilValueNoise(float2(ang * 4.5,  yW * 5.5));
            float dash = smoothstep(0.40, 0.60, dashRun) * smoothstep(0.34, 0.56, dashGap);
            // Lenticels concentrate in horizontal bands but appear broadly — wider
            // band coverage so dashes carry across more of the hero-facing trunk.
            float lent = dash * smoothstep(0.22, 0.52, band);
            // Occasional darker peel band / triangular scar (low freq, rare).
            float peel = smoothstep(0.76, 0.90, soilValueNoise(float2(ang * 1.6 + 7.0, yW * 0.5)));
            // NEUTRAL-COOL paper — but DON'T push >1.0 hard: the hero/foreground
            // birches were blowing to a featureless WHITE PAINT-DRIP smear (the
            // 1.18–1.22 multiplier compounded with the golden-hour sun term and
            // clipped, washing out the dashes). Hold the paper just under unity so
            // the trunk reads as bright bark with VISIBLE structure, not a clipped
            // white column. A faint per-trunk grey value noise breaks the flat sheet.
            float papGrain = soilValueNoise(float2(ang * 1.3 + 4.0, yW * 0.4)) - 0.5;
            float3 paper = float3(0.96, 0.97, 0.99) * (1.0 + 0.10 * papGrain);
            albedo *= mix(paper, float3(0.50, 0.45, 0.42), peel * 0.65);
            // DARK, high-contrast lenticel — drive nearly to black so the dash reads
            // even against the bright paper (full-weight mix so hero dashes hold).
            albedo = mix(albedo, float3(0.035, 0.032, 0.032), lent * 0.98);
            // Paper-smooth roughness; peel edges lift the normal slightly.
            roughness = clamp(roughness * 0.85, 0.30, 0.70);
            n = normalize(n + float3(0,1,0) * (lent * 0.22) + tang * (peel - 0.5) * 0.14);
        } else if (species < 4.5) {
            // ── MAPLE — narrow BROKEN vertical shaggy strips, lifted/curling edges ─
            // ROUND-5 STRUCTURAL #2: the round-3 tuning read as a smooth tan pole at
            // hero distance — its strips were too wide + too low-contrast to survive,
            // so no foreground trunk read as maple (it looked like generic oak/birch
            // wood). The reviewer wants maple UNMISTAKABLY DISTINCT from oak: NARROW,
            // BROKEN vertical shaggy strips with strong tangential normal at the strip
            // borders (lifted/curling edges) and NO horizontal cracking — it must NOT
            // share the oak crack field. Build a vertical-strip field, BREAK each strip
            // vertically into shaggy segments (a separate vertical break mask), drive a
            // wide albedo split (dark damp furrow ↔ pale grey lifted strip), and lift
            // the curling edges HARD (strong tangential tilt at the strip-vertical
            // borders, away from the strip — the peeling-up shag of maple bark).
            // ROUND-6 STRUCTURAL #2 (maple side): the reviewer says maple's strips
            // don't read NARROWER / more BROKEN than oak. Make the difference
            // unmistakable: narrow the strips further (ang×18 ≈ 6–7 cm vs oak's plate
            // cells), break them harder vertically into short shaggy SEGMENTS, push the
            // albedo split MORE (pure-black furrow ↔ pale lifted strip), and lift the
            // strip BORDERS hard with tangential normal (the peeling-up shag of maple),
            // with ZERO horizontal cracking (no oak crack field touches this branch).
            float2 sc  = float2(ang * 18.0, yW * 0.40);        // narrower strips (~6–7 cm)
            float2 sc2 = float2(ang * 38.0, yW * 1.8);
            float f   = soilValueNoise(sc) * 0.70 + soilValueNoise(sc2) * 0.30;
            float fa  = soilValueNoise(sc + float2(e,0)) * 0.70 + soilValueNoise(sc2 + float2(e,0)) * 0.30;
            float fy  = soilValueNoise(sc + float2(0,e)) * 0.70 + soilValueNoise(sc2 + float2(0,e)) * 0.30;
            float fine = soilValueNoise(float2(ang * 50.0, yW * 14.0)) - 0.5;
            // Vertical BREAK: strips snap into short shaggy SEGMENTS (the maple tell vs
            // oak's stacked plate cells). Stronger break (higher vertical freq, wider
            // gap window) so the strips are visibly BROKEN, not continuous combed lines.
            float vbreak = soilValueNoise(float2(ang * 7.0, yW * 4.2));
            float segMask = smoothstep(0.34, 0.50, vbreak);    // 0 at a strip break gap
            // Sharper, NARROWER strip mask → the shaggy lift concentrates at the borders.
            float strip = smoothstep(0.44, 0.52, f) * segMask;
            float border = (1.0 - abs(smoothstep(0.44, 0.52, f) - 0.5) * 2.0) * segMask;
            // WIDER albedo split, pulled DARKER/greyer than oak so the two species never
            // read alike: near-BLACK damp furrow ↔ pale weathered grey lifted strip.
            albedo *= mix(0.18, 1.22, strip) * (0.80 + 0.32 * fine);
            float gA = (fa - f) / e, gY = (fy - f) / e;
            // CURLING-EDGE lift: very strong tangential tilt at the vertical strip
            // borders (the bark peeling up), plus an outward push at the segment break
            // so the shaggy ends catch the rake light.
            float segEdge = (1.0 - segMask);                   // 1 at a vertical break
            n = normalize(n + tang * (gA * 2.7 + border * 1.5 + fine * 0.60)
                            + float3(0,1,0) * (gY * 0.30)
                            + n * (border * 0.35 - segEdge * 0.22));
        } else if (species < 5.5) {
            // ── MOSSY FALLEN LOG — BARREL (damp moss velvet + length-aligned bark) ─
            // ROUND-4 MATERIAL-GATING FIX: the concentric end-grain RING field was
            // firing on the BARREL circumference as alternating pale/dark candy
            // stripes (the "striped wicker basket" read). Rings now live ONLY on the
            // cut-end cap (marker 6, the `species >= 5.5` branch below) — this branch
            // is the BARREL only: moss velvet on top, bark on the underside, with
            // bark RIDGES running ALONG the log's length (not around it).
            float3 ref = (abs(n.y) > 0.9) ? float3(1,0,0) : float3(0,1,0);
            float3 lt = normalize(cross(n, ref));
            float3 lb = cross(n, lt);
            float3 wp = in.worldPos;
            float mu = dot(wp, lt), mv = dot(wp, lb);
            // ── ROUND-6 STRUCTURAL #1 — BARREL MATERIAL REBUILT ──────────────────
            // DIAGNOSIS (why the previous branch rendered a smooth tan dowel even
            // though the marker-5 gate provably fired): the moss was gated to the
            // GEOMETRIC up component `n.y` via `topness = smoothstep(-0.20,0.50,n.y)`.
            // A fallen log is HORIZONTAL, and the hero/low camera sees its SIDE, where
            // n.y≈0 ⇒ topness≈0.29 ⇒ `surf = mix(bark, velvet, 0.29·0.85+0.15≈0.40)`
            // — i.e. the visible side was 60 % BARK. The moss velvet DID survive to the
            // G-buffer, but only on the thin n.y≈1 crest the camera barely grazes. The
            // tan dowel WAS the bark underside, shown on the side. Plus `bareBark`
            // (coverage<0.46) punched even more bark through.
            // REBUILD: dense moss velvet over the whole UPPER HEMISPHERE (top crest AND
            // both upper flanks the camera actually sees), bark only on the genuine
            // UNDERSIDE (n.y<0), smooth blend at the equator. The split is on `topness`
            // but with a much LOWER, wider ramp so the sides read mossy; bark
            // breakthrough is confined to the underside and a few small dry clumps.
            float topness = smoothstep(-0.55, 0.10, n.y);   // moss covers upper hemi + sides
            // Velvet mottle: COARSE clumps dominate (≈12–30 cm), one fine octave for
            // the close fuzz. Weighted toward the low freq so it survives distance.
            float m0 = soilValueNoise(float2(mu * 4.5,  mv * 4.5));   // ≈20 cm patches
            float m1 = soilValueNoise(float2(mu * 11.0, mv * 11.0));  // ≈9 cm
            float m2 = soilValueNoise(float2(mu * 40.0, mv * 40.0));  // close fuzz
            float mottle = m0 * 0.58 + m1 * 0.28 + m2 * 0.14;
            // HIGH-CONTRAST damp velvet: dark olive-green floor ↔ bright yellow-green
            // tip clumps. Floor lifted off black (a damp forest moss is dark olive).
            float3 mossDamp = float3(0.040, 0.072, 0.030);
            float3 mossTip  = float3(0.245, 0.300, 0.082);
            float3 velvet = mix(mossDamp, mossTip, smoothstep(0.34, 0.70, mottle));
            // Bark breakthrough now SPARSE on the moss: only small DRY clumps on the
            // upper surface (rare), but the underside is fully bark. Tightened so the
            // barrel reads UNMISTAKABLY mossy from the side, not 60 % bark.
            float cover = soilValueNoise(float2(mu * 2.4 + 13.0, mv * 2.4 + 5.0));
            float dryClump = smoothstep(0.30, 0.14, cover);  // rare small dry patch
            // LENGTH-ALIGNED bark ridges on the exposed bark: the bark coordinate
            // `mv` runs roughly along the bole, so high-freq in `mv`, low-freq in `mu`
            // makes ridges that run ALONG the log — not stripes that wrap it.
            float barkRidge = soilValueNoise(float2(mu * 2.2, mv * 16.0));
            float3 barkUnder = float3(0.21, 0.135, 0.090) * (0.6 + 0.7 * m1)
                             * (0.78 + 0.5 * barkRidge);   // ridges run lengthwise
            // Moss dominates the upper hemisphere; bark shows on the underside (low
            // topness) and a few small dry clumps on top. The blend is on `topness`
            // (smooth at the equator) with only a sparse dry-clump break on top.
            float3 mossSurf = mix(velvet, barkUnder, dryClump * 0.5 * topness);
            float3 surf = mix(barkUnder, mossSurf, topness);
            albedo = surf;
            // Bark coverage (for the normal/roughness) is high only where bark wins.
            float barkAmt = (1.0 - topness) + dryClump * 0.5 * topness;
            // Fine fuzz normal (where moss is) so velvet catches the rake light as a
            // soft fur. Coarse mottle bumps the silhouette. Bark grooves on underside.
            float ep = 0.05;
            float gU = (soilValueNoise(float2((mu+ep)*40.0, mv*40.0)) - m2);
            float gV = (soilValueNoise(float2(mu*40.0, (mv+ep)*40.0)) - m2);
            float gC0 = (soilValueNoise(float2((mu+ep)*3.2, mv*3.2)) - m0);   // coarse bump
            float gC1 = (soilValueNoise(float2(mu*3.2, (mv+ep)*3.2)) - m0);
            float bg = (soilValueNoise(float2(mu*2.2 + ep, mv*16.0)) - barkRidge);
            n = normalize(n + (lt * (gU + gC0 * 2.6 + bg * barkAmt * 1.2)
                               + lb * (gV + gC1 * 2.6)) * (0.85 * topness + 0.20 * barkAmt)
                            + float3(0,1,0) * (mottle - 0.5) * 0.20);
            // Velvet is matte (high roughness); damp bark a touch glossier.
            roughness = mix(0.97, 0.90, barkAmt);
        } else {
            // ── MOSSY FALLEN LOG — CUT END-GRAIN CAP (marker 6) ─────────────────
            // ROUND-4: rings fire ONLY here. This face is the sawn/snapped END of the
            // bole (addMossyLog stamps materialMark = 6 on the cut-end disc; no other
            // surface emits w ≈ 6). Build a polar radial coordinate centred on the
            // cap's pith and draw concentric growth rings as a luminance alternation —
            // so rings appear on the cut face and NEVER wrap the barrel circumference.
            // The cap's geometric normal is ≈ axial, so the radial frame is built in
            // the cap plane (perpendicular to n).
            float3 ref = (abs(n.y) > 0.9) ? float3(1,0,0) : float3(0,1,0);
            float3 ct = normalize(cross(n, ref));
            float3 cb = cross(n, ct);
            float3 wp = in.worldPos;
            // Radial distance from the local pith. The cap is small (≈0.3 m), so use a
            // fract-free world projection; the baked vertex gradient handles absolute
            // centring, this adds the crisp ring alternation on top.
            float cu = dot(wp, ct), cv = dot(wp, cb);
            // A coherent per-cap centre from the low-freq field so the rings are
            // roughly concentric without packing the true pith position.
            float2 pith = (soilValueNoise(float2(cu, cv)) - 0.5) * 0.05;
            float rr = length(float2(cu, cv) - pith);
            // ≈5 rings over the cap radius: alternate light sap / dark latewood.
            float ringPhase = rr * 22.0;
            float ring = 0.5 + 0.5 * cos(ringPhase * 2.0 * 3.14159265);
            // Heartwood pith → paler sapwood radial base (the baked vertex colour
            // already trends this way; reinforce + add the ring alternation).
            // Warm wood, not black: the cut face is sap-bright with a darker (but not
            // black) latewood line so the END-GRAIN rings read as warm wood, not a
            // hole punched in the log.
            float3 latewood = float3(0.30, 0.20, 0.115);   // darker growth-ring line
            float3 earlywood = max(albedo, float3(0.34, 0.24, 0.15)) * 1.18;  // bright sap
            albedo = mix(latewood, earlywood, smoothstep(0.30, 0.70, ring));
            // Faint radial checks / rays + slight recess at each dark ring line.
            float ray = soilValueNoise(float2(atan2(cv, cu) * 6.0, rr * 3.0));
            albedo *= 0.88 + 0.24 * ray;
            roughness = mix(0.62, 0.85, ring);             // latewood a touch glossier
            n = normalize(n - n * (1.0 - ring) * 0.06);    // dark rings recess slightly
        }
#endif
    }

    // ── Procedural sausage-casing micro-detail (HotdogDropUltra) ──────────────
    // Gates on vertexColor.a ∈ (0.6, 0.9) — set to 0.75 by pbdTubeExpand.
    // All sausage-casing vertices share alpha=0.75; normal geometry is 1.0,
    // foliage is 0.0. This is a no-op for every other scene.
    //
    // PORT OF THE HotdogDrop+ SAUSAGE MATERIAL (HotdogPlusGeometry.swift):
    // Drop+ drives the frank with (a) a SUBTLE wrinkle/pore normal map
    // (octaves 8/24/64 over a 256px tile that wraps the ~2-unit circumference
    // ⇒ ~4/12/32 cycles per world unit, intensity 1.1 — skin texture, not
    // blisters) and (b) a low-frequency ROUGHNESS mottle map in [0.18, 0.78]
    // (octaves 4/10/28 per tile ⇒ ~2/5/14 c/u) so wet/dry patches catch
    // highlights independently — that spatial gloss variance, not albedo
    // mottle, is where the "just off the grill" glisten lives. Albedo stays
    // a solid per-instance colour, exactly like Drop+'s flat diffuse.
    // Drop+'s clearcoat (0.55 / cc-rough 0.18) has no Illuminatorama analog;
    // the 0.18 glossy floor of the roughness mottle approximates the glaze.
    if (in.vertexColor.a > 0.6 && in.vertexColor.a < 0.9) {
        // Build a CONTINUOUS tangent frame from the surface normal.
        // Previous version used a hard branch on abs(n.y): as a horizontal frank's
        // normal swept around the tube it crossed the 0.9 threshold, snapping the
        // reference vector (0,1,0)→(1,0,0) and rotating the entire noise-input
        // space discontinuously → visible diagonal two-tone seam on the franks.
        // Fix: blend the two candidate reference axes smoothly on n.y so the
        // tangent frame transitions without a step-discontinuity.
        float yBlend    = smoothstep(0.75f, 0.95f, abs(n.y));
        float3 blendRef = normalize(float3(yBlend, 1.0f - yBlend, 0.0f));
        float3 casingTang = normalize(cross(n, blendRef));
        float3 casingBtan = cross(n, casingTang);

        float3 wp = in.worldPos;
        float tu = dot(wp, casingTang);
        float tv = dot(wp, casingBtan);
        // Wrinkle/pore normal — Drop+'s sausageNormal equivalent in world space.
        // Gentle: combined peak tilt ≈ 4-5°, reads as skin texture in specular.
        const float sn0 = 4.0f, sn1 = 12.0f, sn2 = 32.0f;
        const float ep = 0.04f;
        float w0   = soilValueNoise(float2(tu * sn0, tv * sn0));
        float w0dx = soilValueNoise(float2((tu + ep) * sn0, tv * sn0));
        float w0dy = soilValueNoise(float2(tu * sn0, (tv + ep) * sn0));
        float w1   = soilValueNoise(float2(tu * sn1, tv * sn1));
        float w1dx = soilValueNoise(float2((tu + ep) * sn1, tv * sn1));
        float w1dy = soilValueNoise(float2(tu * sn1, (tv + ep) * sn1));
        float w2   = soilValueNoise(float2(tu * sn2, tv * sn2));
        float w2dx = soilValueNoise(float2((tu + ep) * sn2, tv * sn2));
        float w2dy = soilValueNoise(float2(tu * sn2, (tv + ep) * sn2));
        // Drop+ octave weights 0.6 / 0.25 / 0.15 at SUBTLE amplitude. The
        // round-31 1.7× lift made the casing read lumpy under a matte lobe;
        // with the clearcoat now supplying a real glint, faint wrinkle is
        // enough — the coat highlight picks it up (round-31 review).
        float gU = (w0dx - w0) * 0.050f + (w1dx - w1) * 0.022f + (w2dx - w2) * 0.012f;
        float gV = (w0dy - w0) * 0.050f + (w1dy - w1) * 0.022f + (w2dy - w2) * 0.012f;
        n = normalize(n + casingTang * gU + casingBtan * gV);

        // Roughness mottle — Drop+'s sausageRoughness map verbatim: big wet/dry
        // blotches, weights 0.65 / 0.25 / 0.10, range [0.18, 0.78], mean ≈ 0.48.
        // This is the SATIN base layer; the tight glint lives in the clearcoat
        // lobe the deferred lighting pass adds for casing-flagged pixels (see
        // illumi_lighting — flag 0.75 in normalRoughness.w). Folding the glaze
        // into this one lobe instead provably oscillates: [0.16, 0.62] read as
        // lacquered silicone (round 30), [0.30, 0.62] as foam rubber (round 31).
        // Overrides the per-instance scalar for casing pixels only.
        const float sr0 = 2.0f, sr1 = 5.0f, sr2 = 14.0f;
        float rN = soilValueNoise(float2(tu * sr0, tv * sr0)) * 0.65f
                 + soilValueNoise(float2(tu * sr1, tv * sr1)) * 0.25f
                 + soilValueNoise(float2(tu * sr2, tv * sr2)) * 0.10f;
        roughness = 0.18f + saturate(rN) * 0.60f;
    }

    // ── Procedural plush fuzz micro-normal (Teddy Bear Press) ─────────────────
    // Gates on vertexColor.a ∈ [0.5, 0.6] (≈0.55 from TeddyBearGeometry). A fine,
    // high-frequency normal break-up so the silhouette + specular read as short
    // fur fuzz rather than smooth rubber, and the per-instance high roughness stays
    // matte. No-op for every other scene (no other mesh ships alpha in this band).
    if (in.vertexColor.a >= 0.5f && in.vertexColor.a <= 0.6f) {
        float yBlend    = smoothstep(0.75f, 0.95f, abs(n.y));
        float3 blendRef = normalize(float3(yBlend, 1.0f - yBlend, 0.0f));
        float3 fuzzTang = normalize(cross(n, blendRef));
        float3 fuzzBtan = cross(n, fuzzTang);
        float3 wp = in.worldPos;
        float tu = dot(wp, fuzzTang);
        float tv = dot(wp, fuzzBtan);
        // High-frequency fuzz (≈ 60 / 150 cycles per world unit) at small amplitude
        // — a soft micro-tilt, not blisters. The matte roughness scatters it into
        // the velvety sheen the deferred plush lobe rim-lights.
        // Two scales of fur fuzz: a coarse tuft (~28 c/u, reads at the camera distance
        // the bears are seen from) + a fine fibre (~90 c/u). Amplitudes lifted so the
        // matte sheen visibly breaks up the silhouette instead of staying glassy-smooth.
        const float fn0 = 28.0f, fn1 = 90.0f, ep = 0.012f;
        float f0   = soilValueNoise(float2(tu * fn0, tv * fn0));
        float f0dx = soilValueNoise(float2((tu + ep) * fn0, tv * fn0));
        float f0dy = soilValueNoise(float2(tu * fn0, (tv + ep) * fn0));
        float f1   = soilValueNoise(float2(tu * fn1, tv * fn1));
        float f1dx = soilValueNoise(float2((tu + ep) * fn1, tv * fn1));
        float f1dy = soilValueNoise(float2(tu * fn1, (tv + ep) * fn1));
        float gU = (f0dx - f0) * 0.11f + (f1dx - f1) * 0.05f;
        float gV = (f0dy - f0) * 0.11f + (f1dy - f1) * 0.05f;
        n = normalize(n + fuzzTang * gU + fuzzBtan * gV);
    }

    o.albedoMetallic  = half4(half3(albedo), half(metallic));
    float2 oct = octEncode(n);
    // normalRoughness.w is otherwise unused (always 1) — repurpose it as a
    // foliage flag. Scenes that want leaf thin-sheet transmission tag their
    // leaf vertices with colour ALPHA 0 (opaque geometry keeps alpha 1); the
    // deferred lighting pass reads w < 0.5 to add the back-light term. No-op
    // for every existing scene (they all ship vertex alpha 1 → w stays 1).
    half foliageFlag = (in.vertexColor.a < 0.5) ? 0.0h : 1.0h;
    // Sausage casing (vertexColor.a = 0.75 from pbdTubeExpand) forwards its
    // flag into normalRoughness.w so the deferred lighting pass can add the
    // clearcoat glaze lobe. Every foliage test is `w < 0.5` and the matRough
    // cap below is `< 0.5h`, so 0.75 behaves as ordinary opaque geometry
    // everywhere except the clearcoat branch.
    if (in.vertexColor.a > 0.6 && in.vertexColor.a < 0.9) foliageFlag = 0.75h;
    // Plush flag (Teddy Bear Press): colour alpha ≈ 0.55 (band 0.5–0.6) → 0.55h.
    // 0.55 is not < 0.5 (so the foliage roughness cap below leaves plush's high
    // roughness alone) and not in (0.6,0.9) (so it skips the casing clearcoat); the
    // deferred lighting pass reads 0.5h<w<0.6h to add the fur sheen + SSS. No-op for
    // every other scene (no other mesh ships colour alpha in this band).
    if (in.vertexColor.a >= 0.5 && in.vertexColor.a <= 0.6) foliageFlag = 0.55h;
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
    // Exactly identity when `inst.anisotropy == 0`: the write is `1.0h + 0.0h`, the same
    // bits the old line produced, so every scene that never sets the field — which is
    // every Visualizer scene — is bit-identical. The clamp is insurance the field has
    // nowhere else in the pipeline: `brdf()` splits GGX as `ab = a * (1 - 0.7 * aniso)`,
    // which goes NEGATIVE past aniso ≈ 1.43.
    half wTag = foliageFlag;
    if (wTag >= 1.0h) wTag = 1.0h + half(clamp(inst.anisotropy, 0.0f, 1.0f));
    o.normalRoughness = half4(half(oct.x), half(oct.y), half(matRough), wTag);
    // Phase 4.9 — emission can be a texture (used heavily by Plus scenes
    // for glow effects on rails / lamps / fire) or a scalar. Both are
    // additive on top of the lit colour. Sampling from the sRGB-decoded
    // albedo atlas means the bake gets linear RGB directly.
    float3 emission = inst.emission;
    if (inst.emissionTextureSlice >= 0) {
        float4 tx = sampleAtlasAspect(albedoAtlas, texSampler, in.uv,
                                      uint(inst.emissionTextureSlice), albedoUVScale, duvdx, duvdy);
        // Phase 4.27b — scale the emission texture by the material's
        // `emission.intensity` so a texture-driven glow reads at its tuned
        // HDR brightness (Pizza's heat coils were flat at intensity 1).
        emission += tx.rgb * inst.emissionIntensity;
    }
    // Phase 7 — pack clearcoat (≥0) OR cloth sheen (<0) into emission.alpha (was always 1.0,
    // unused). A surface is polished OR cloth, never both, so one channel carries either: > 0 =
    // polished/lacquered second GGX lobe; < 0 = velvet/wool grazing-Fresnel sheen (strength = -a).
    o.emission        = half4(half3(emission), half(inst.clearcoat > 0.0 ? inst.clearcoat : -inst.sheen));
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
    return o;
}
