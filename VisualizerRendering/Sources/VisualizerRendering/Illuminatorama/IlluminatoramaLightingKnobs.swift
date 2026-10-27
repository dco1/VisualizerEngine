import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - LightingKnobs

/// Stored state for the renderer's `lighting` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `LightingKnobs` (`kLighting`) and forwards each name to it from
/// `IlluminatoramaRenderer+LightingKnobs.swift`; callers see no difference.
@MainActor
final class LightingKnobs {
    typealias InstanceRef = IlluminatoramaRenderer.InstanceRef
    typealias MeshKind = IlluminatoramaRenderer.MeshKind
    typealias RTUniforms = IlluminatoramaRenderer.RTUniforms
    typealias StaticSkipStats = IlluminatoramaRenderer.StaticSkipStats

    var instances: [InstanceRef] = []
    var staticSkipStats = StaticSkipStats()
    /// Opt-in FRAME-COMPUTE hook (additive, nil elsewhere — every other host is byte-identical).
    /// Called once per frame that actually renders — never for a tick `render()` drops because
    /// `maxFramesInFlight` frames are already queued — with the frame's own command buffer, before
    /// any pass (and before the RT acceleration-structure refit) reads a mesh. A host that writes
    /// its own GPU geometry each frame (skinning a `LinearBlendSkinMesh`, an `ArcTubeBatch`) encodes
    /// it here instead of committing a command buffer of its own every tick: at a 30 fps GPU frame
    /// and a 60 Hz tick a per-tick side buffer is encoded twice for every frame drawn, queued behind
    /// the frames in flight, and holds command-buffer slots on the renderer's queue (Digital Clock,
    /// VZ-0233). Same ordering on the GPU as a side buffer committed just before the frame.
    var onEncodeFrameCompute: ((_ cb: MTLCommandBuffer) -> Void)?
    /// Opt-in AFTER-G-BUFFER hook (Daydream DH-1466; nil elsewhere): called with the frame's command
    /// buffer right after the G-buffer pass, before glass/particles touch `gbufferDepthTexture` — a
    /// host reads this frame's opaque depth here (Signack's Hi-Z pyramid).
    var onEncodeAfterGBuffer: ((_ cb: MTLCommandBuffer) -> Void)?
    var pointLights: [IlluminatoramaPointLight] = []
    var spotLights: [IlluminatoramaSpotLight] = []
    /// Rectangular LTC area lights (#60 task 5). Populated by the extractor from
    /// SCN `.area` lights; consumed by `illumi_lighting` (buffer 4).
    var areaLights: [IlluminatoramaAreaLight] = []
    /// Secondary directional lights (#60 task 5 — retires the 4.20 ambient fold).
    /// The PRIMARY directional is `directionalLightDirection`/`directionalLightColor`
    /// (cascade-rigged); these are the fill/back directionals a SCN 3-point rig
    /// ships alongside it. Populated by the extractor; consumed by `illumi_lighting`
    /// (buffer 5) with a real BRDF (no shadow).
    var extraDirectionals: [IlluminatoramaDirectionalLight] = []
    var directionalLightDirection: SIMD3<Float> = simd_normalize(SIMD3(0.5, 1.0, 0.3))
    var directionalLightColor: SIMD3<Float> = SIMD3(1.0, 0.97, 0.92) * 3.0
    var ambientColor: SIMD3<Float> = SIMD3(0.10, 0.13, 0.18)
    /// Leaf thin-sheet transmission strength (issue #58). 0 = OFF (default) — the
    /// foliage back-light term is skipped entirely unless a scene opts in. Drives
    /// both the deferred lighting kernel (`FrameUniforms.leafTransmission`) and the
    /// RT sun pass (`RTUniforms.leafTransmission`, which owns the sun under RT).
    /// Foliage fragments are flagged by tagging leaf vertices with colour alpha 0.
    var leafTransmission: Float = 0
    /// Signack S5 (Daydream DH-1461 / DH-1361) — τ/ρ (at the pigment peak; 0.6–1.0 for real
    /// broadleaves) of the PHYSICAL thin-leaf model (`IlluminatoramaLeafOptics.h`), applied to every
    /// foliage pixel whose vertex alpha carries a THICKNESS (0 < alpha ≤ 0.45): the transmitted
    /// colour comes from the leaf's own albedo + thickness, the sun behind it transmits through a
    /// Lambert + mild forward lobe, and the sky behind it transmits too. Foliage at alpha 0 (grass,
    /// every legacy writer) keeps the #58 `leafTransmission` backlight. 0 (default) = off: every host
    /// that never sets it is unchanged. Deferred kernel only (the RT soup sun pass is legacy).
    var leafTransmittanceRatio: Float = 0
    /// Plush material strengths (Teddy Bear Press). `plushSheen` = fur retro-
    /// reflective edge sheen; `plushTransmission` = backlit thin-fabric SSS. Both
    /// 0 = OFF (default) → exact no-op for every scene. Only the plush-flagged
    /// teddy mesh (`normalRoughness.w` ≈ 0.55h) reads them. Mirror the deferred
    /// `FrameUniforms.plushSheen/plushTransmission`.
    var plushSheen: Float = 0
    var plushTransmission: Float = 0
    /// Lamp-shade fabric translucency strength (DH-0458). 0 = OFF (default) → the
    /// shade thin-sheet transmission branch is skipped entirely. A paper/linen drum
    /// shade is a thin translucent sheet; this drives the back-lit transmission that
    /// lets external light (a sunset through the glazing, the room behind the shade)
    /// scatter through the fabric rather than landing flat on an opaque cone. Only
    /// the shade-flagged mesh (`normalRoughness.w` ≈ 0.62) reads it. Mirrors the
    /// deferred `FrameUniforms.shadeTransmission`.
    var shadeTransmission: Float = 0
    /// Opt-in INTERIOR MAPPING (van Dongen 2008) per mesh kind: every instance of a listed kind is
    /// drawn with a G-buffer variant whose emission is the radiance of a lamp-lit room box behind
    /// the pane, ray-traced analytically per fragment along the camera ray (parallax-correct room
    /// depth, one draw, no extra geometry) — `inst.emission` is the lamp's colour × the room's
    /// luminance. The mesh must carry UVs across the pane's face (u along its width from
    /// `pane.z`, v up from 0) and a tangent along u. Empty (default) ⇒ the variant is never built
    /// or bound ⇒ byte-identical.
    var interiorMappedMeshKinds: [MeshKind: IlluminatoramaInteriorMapping] = [:]
    /// Interior day-light separation (opt-in; pairs with light-layer masking).
    /// `interiorLayerMask` = OR of every instance-layer bit the host stamped on
    /// INTERIOR rooms; 0 (default) disables the feature entirely (byte-identical).
    /// Fragments carrying an intersecting room stamp get their sky-IBL indirect
    /// scaled by mix(`interiorIBLSide`, `interiorIBLUp`, saturate(N.y)) — up-facing
    /// floors lose the open-top skylight hardest — and their ambient supplement
    /// scaled by `interiorAmbient` (>1 = a warm indoor fill). Rationale: an open-top
    /// dollhouse room is otherwise lit like a patio (full sky irradiance), several
    /// stops over a believable interior.
    var interiorLayerMask: UInt32 = 0
    var interiorIBLUp: Float = 1
    var interiorIBLSide: Float = 1
    var interiorAmbient: Float = 1
    /// Interior irradiance BANDS — the room's own three-band diffuse environment
    /// (the lawn-green-ceiling fix). The irradiance cube is baked from the outdoor
    /// sky dome, so an interior ceiling (N.y = −1) otherwise samples the GROUND
    /// half and renders in the lawn's colour — the `interiorIBL*` scalars can dim
    /// that, never recolour it. Interior fragments blend their diffuse irradiance
    /// toward these host-authored values by `interiorIrradianceWeight`
    /// (0 = the exact cube sample, the default — byte-identical for every scene
    /// that never opts in; hosts ramp the weight with daylight so night keeps the
    /// cube). As the weight → 1 the DIFFUSE interior scalar folds to 1.0 (the
    /// bands already carry the up/side weighting it faked); specular keeps the
    /// scalar and the prefiltered cube. Units match the cube sample they replace
    /// (pre-`iblIntensity` irradiance). Applied by the deferred kernel AND by
    /// secondary-hit shading (glass refraction / RT reflections / GI), so the room
    /// seen through a pane agrees with the room beside it.
    var interiorIrradianceUp: SIMD3<Float> = .zero
    var interiorIrradianceSide: SIMD3<Float> = .zero
    var interiorIrradianceDown: SIMD3<Float> = .zero
    var interiorIrradianceWeight: Float = 0
    /// Daylight apertures for the band GRADATION (S3.5 Stage D): the host's glazed
    /// openings, one rect each. With `interiorApertureGradation` > 0, interior fragments
    /// scale their BAND irradiance by an approximate summed solid angle of these — a wall
    /// point near its window reads brighter than the same wall 8 m away, which no
    /// normal-only term can produce. Empty list or strength 0 (the defaults) ⇒ factor is
    /// exactly 1.0 ⇒ byte-identical. Primary (deferred) pass only: the secondary paths
    /// carry the flat bands — a reflection of a wall does not re-derive its gradation.
    var interiorApertures: [IlluminatoramaInteriorAperture] = []
    /// 0 = off (default); 1 = the full gradation. Fractional = partial (a look dial).
    var interiorApertureGradation: Float = 0
    /// Per-room BAND LEVEL (S3.5 Stage E): one multiplier per light-layer bit, applied
    /// to a fragment's interior irradiance bands (diffuse and specular alike) according
    /// to the room its `layer` bits name.
    ///
    /// The bands are otherwise ONE environment for the whole frame, so their level knows
    /// the time of day and nothing about the room: a hall of glass and an interior closet
    /// rendered the same ceiling. The host measures how much daylight each room's own
    /// glazing admits and hands the levels here. A fragment on a shared wall carries both
    /// rooms' bits and takes their mean (there is no per-face signal at this point in the
    /// pipeline). Bits beyond `InteriorRoomGains.capacity`, an empty table, or
    /// `interiorRoomGainsEnabled == false` (all defaults) ⇒ every resolve is exactly 1.0
    /// ⇒ byte-identical. Carried by the deferred kernel AND both secondary paths, so a
    /// room through a pane or in a reflection is scaled like the room seen directly.
    var interiorRoomGains: [Float] = []
    /// Master enable for `interiorRoomGains`. False (default) ⇒ exact no-op.
    var interiorRoomGainsEnabled: Bool = false
    /// A/B DEBUG: overrides the LTC-specular arm of the area-light eval (nil = the
    /// baked LUT's own validation decides, the shipped behaviour). `false` forces the
    /// most-representative-point fallback — the bisect lever for separating "the LTC
    /// transform misbehaves" from "the polygon integral misbehaves" on a live scene.
    var areaLTCOverride: Bool?
    // ── Phase 3 IBL knobs ────────────────────────────────────────────
    /// Master scale on diffuse + specular IBL contribution. 1 = unscaled.
    var iblIntensity: Float = 1.0
    /// Toggle the IBL path. When `false`, the lighting kernel falls back to
    /// the legacy hemispheric ambient; sky pixels still come from
    /// `equirectSky` when one is supplied.
    var iblEnabled: Bool = true
    /// External equirect HDR sky source — typically
    /// `VolumetricCloudRenderer.outputTexture`. When `nil`, the renderer
    /// uses a 1×1 black fallback (sky pixels render black, IBL contributes
    /// nothing). The texture is sampled, so it must be `.shaderRead`-capable.
    var equirectSky: MTLTexture?
    var iblNeedsRebake: Bool = true
    /// How many seconds may elapse between IBL bakes even when `iblNeedsRebake`
    /// stays false. Default 5 s — enough to keep slow cloud drift from going
    /// completely stale without paying full 60fps bake cost.
    var iblRebakeInterval: Float = 5.0
    // ── Phase 3.2 split-sum DFG LUT ──────────────────────────────────
    /// Toggle the pre-integrated DFG LUT for the F0-weighted IBL specular
    /// response. Off = Lagarde's roughness-Schlick approximation (Phase 3.0
    /// behaviour). The LUT itself is baked once at startup and is
    /// view/material/scene-independent, so there is no rebake cost.
    var dfgLUTEnabled: Bool = true
    /// **DH-1014 — the layered BSDF's energy bookkeeping** (IlluminatoramaBSDF.h). `true`:
    /// multiple-scattering GGX compensation on EVERY specular lobe (sun, lamps, fills, area
    /// portals — the IBL arm has had it since S1.3b) and the diffuse base weighted by what the
    /// specular layer did not reflect, in the deferred pass AND in the ray-traced lighting
    /// kernel's GI / reflection weights, so a white surface under a uniform environment returns
    /// 1.0 at every roughness on the live lane and on Maximum alike. Needs `dfgLUTEnabled`.
    /// `false` (the default) is every lobe's historical expression, byte for byte — a look
    /// change is the host's call (Daydream Home opts in).
    var layeredMultiscatter: Bool = false
    /// **DH-1301 — diffuse roughness (OpenPBR `base_diffuse_roughness`, the EON lobe).** The master
    /// switch for every lane: with it set, an instance's `diffuseRoughness` (> 0, pure dielectric)
    /// shades with the energy-preserving Oren-Nayar lobe (IlluminatoramaDiffuseRoughness.h) — the full
    /// lobe under every light with a direction (sun, points, spots, fills, the traced primary sun), its
    /// directional albedo under every integrated source (area / portal lights, the diffuse IBL and
    /// ambient, the traced GI, every secondary and path vertex). `true` (the default) costs nothing and
    /// changes nothing for a material that never sets one — the per-material value defaults to 0, which
    /// is Lambert bit for bit. `false` makes every lane Lambert whatever the materials carry (the A/B lever).
    var diffuseRoughnessEnabled: Bool = true
    /// **DH-1300 — the per-material specular layer (OpenPBR `specular_ior` / `specular_weight` /
    /// `specular_color`).** The master switch for every lane: with it set, an instance's
    /// `specularIOR` / `specularWeight` / `specularColor` set its dielectric F0 under every light, in
    /// the environment, in the traced reflection's Fresnel and the traced GI's weight, and at every path
    /// vertex. `true` (the default) changes nothing for a material that never sets them — the instance
    /// then carries no layer parameters and every lane keeps the literal F0 0.04 — so it costs one
    /// G-buffer texel read per pixel. `false` makes every dielectric F0 0.04 whatever the materials carry.
    var layeredSpecularEnabled: Bool = true
    /// **DH-1309 — the coat as a real layer.** `true`: a clearcoated instance's coat is a second GGX lobe
    /// under EVERY light (sun, points, spots, fills, area / portal lights) and the environment, at its
    /// own roughness on every lane, the base below attenuated by the coat lobe's own directional albedo
    /// (DFG LUT) instead of `1 − cc·Schlick(N·V)`, and the traced reflection takes the coat's share. A
    /// coat can then share a pixel with a cloth sheen (DH-1299) and carries its IOR / tint / darkening
    /// (DH-1298). `false` (the default) is the Phase-7 clearcoat, byte-identical — it changes every
    /// coated material, so a host opts in.
    var layeredCoatEnabled: Bool = false
    /// **DH-1014 / DH-0597 — cloth sheen as a layer.** `true`: the sheen's environment arms read
    /// the fitted Charlie albedo from the DFG LUT (`.a`) instead of the hand-fitted curve, and
    /// the base under the sheen loses what the sheen reflected (albedo scaling), in the deferred
    /// pass and the RT weights. `false` (the default) is byte-identical.
    var layeredClothSheen: Bool = false
    /// **Daydream DH-1316 — the cloth sheen takes the dye's colour.** `true`: the sheen lobe's colour
    /// (every light + the environment, bolted-on or layered) is the fibres' single-scattering albedo,
    /// `0.85·4ρ/(1+ρ)² + 0.15` (`clothSheenColorDyed`, IlluminatoramaBSDF.h) — a dark velvet's rim is a
    /// lighter version of its own colour instead of a grey veil. With `layeredClothSheen` the base's
    /// albedo scaling and the RT kernel's weights follow. Exactly the old colour for a white cloth
    /// (furnace unchanged). `false` (the default) is byte-identical.
    var layeredDyedSheen: Bool = false
    /// **Daydream DH-1534 — a metal takes no flat ambient.** `true`: the ambient supplement (a stand-in
    /// for diffuse room bounce) is scaled by `1 - metallic`, so a metal's pixel is its environment
    /// reflection + lights instead of ~76% one uniform constant. `false` (the default) is byte-identical.
    var metalNoFlatAmbient: Bool = false
    /// Phase-by-phase breakdown of how long `init` took (see
    /// `IlluminatoramaInitTiming`). Recorded unconditionally so a host can fold it
    /// into its own launch timeline without a special build; `VIZ_ILLUMI_INIT_TIMING`
    /// additionally emits signposts / a sidecar file.
    var initTiming = IlluminatoramaInitTiming()
}
