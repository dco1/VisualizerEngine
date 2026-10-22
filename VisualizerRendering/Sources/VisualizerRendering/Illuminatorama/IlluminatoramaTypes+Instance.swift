import Metal
import simd

/// Per-instance draw data. Bound as a structured buffer; vertex stage reads
/// `instances[instance_id]`.
public struct IlluminatoramaInstance {
    public var modelMatrix: simd_float4x4
    public var normalMatrix: simd_float4x4   // upper-3x3 inverse-transpose, padded
    public var albedo: SIMD3<Float>
    public var metallic: Float
    // Phase 7 — clearcoat: a second GGX lobe for polished/lacquered surfaces
    // (terrazzo, marble, glazed tile, lacquered wood). Sits in the 12-byte
    // padding gap between `metallic` and `emission` (offsets 148-159); stride
    // stays 208. Default 0 = no clearcoat (no change to existing materials).
    public var clearcoat: Float = 0          // lobe strength [0, 1]
    /// GGX roughness of the clearcoat lobe. 0.08 IS the constant the lighting kernel used for
    /// every material until DH-0140, so the default renders exactly as before; a material that
    /// sets its own value is honoured only where the photo-lane G-buffer target carries it
    /// (`IlluminatoramaRenderer.extendedGBufferEnabled`) — the live lane keeps the constant.
    public var clearcoatRoughness: Float = 0.08
    // Phase 7b — cloth sheen lobe strength [0,1] (velvet/wool/linen). Repurposes the former
    // `_padClearcoat` slot (same offset 156, stride stays 208). Packed as a NEGATIVE
    // emission.alpha in the G-buffer (a surface is polished OR cloth, never both).
    public var sheen: Float = 0
    public var emission: SIMD3<Float>
    public var roughness: Float
    /// Phase 4.0 — slice index into the diffuse albedo atlas. `< 0` means
    /// "no texture, use `albedo` colour".
    public var albedoTextureSlice: Int32 = -1
    /// Phase 4.1 — slice index into the non-colour material atlas (R/G/B
    /// channels each readable independently). `< 0` means "no texture,
    /// use `metallic` scalar".
    public var metallicTextureSlice: Int32 = -1
    /// Phase 4.1 — same atlas as metallic; `< 0` falls back to `roughness`.
    public var roughnessTextureSlice: Int32 = -1
    /// Phase 4.5 — slice index for the tangent-space normal map in the
    /// non-colour atlas. `< 0` means "no normal map, use geometric normal".
    public var normalTextureSlice: Int32 = -1
    /// Phase 4.9 — slice index for the emission map in the sRGB albedo
    /// atlas (emission is colour data, so it shares the sRGB-decoding
    /// atlas with diffuse). `< 0` falls back to `emission` scalar.
    public var emissionTextureSlice: Int32 = -1
    /// Phase 4.27b — multiplier applied to the emission TEXTURE sample so a
    /// texture-driven glow (Pizza's heat coils = emission texture × intensity)
    /// renders at its tuned HDR brightness instead of flat intensity-1. Solid
    /// emission already folds intensity into the `emission` scalar in the
    /// extractor, and has no texture to multiply, so this is a no-op there.
    /// Repurposes the former `_padSlice0` slot — same 4 bytes, stride stays 208.
    public var emissionIntensity: Float = 1.0
    /// Former trailing pad, repurposed (#60 item 7). How this instance appears to
    /// the ray tracer — THREE states, mapped to TLAS instance masks in
    /// `rebuildRTAccel`:
    ///
    ///   `0` (default) — ordinary traced geometry (mask `0x01`).
    ///   `rtExcludeFully` (1) — raster-only: mask `0x00`, so RT rays never
    ///     intersect it and the RT representation comes from elsewhere (a
    ///     registered `IlluminatoramaCurveSet` twin, a superquadric proxy).
    ///   `rtInvisibleOccluder` (2) — real to LIGHT, never drawn (mask `0x04`).
    ///     Shadow + GI rays are stopped by it; camera-visible rays (glass
    ///     refraction, glossy reflections) pass straight through. This is the
    ///     lighting-only ceiling over a roofless dollhouse: it must stop the sky
    ///     from flooding an interior GI bounce, while never appearing in the
    ///     picture. Value 1 could not express that — it is the whole reason
    ///     interiors flooded when RT GI was enabled.
    ///
    /// Same 4 bytes; kept as `Int32` rather than an enum so the struct layout the
    /// Metal mirror depends on is untouched.
    public var rtExclude: Int32 = 0
    /// `rtExclude` value: raster-only, never intersected by any ray.
    public static let rtExcludeFully: Int32 = 1
    /// `rtExclude` value: invisible occluder — stops transport rays (shadow, GI),
    /// invisible to camera-visible rays (glass refraction, glossy reflections).
    public static let rtInvisibleOccluder: Int32 = 2
    // Phase 7 — detail-normal path for close-range pores/weave/grain.
    // Sampled at `detailNormalUVScale × in.uv` and blended into the
    // macro normal map result. Stride grows from 208 → 224 (next 16-byte
    // boundary) to maintain float4x4 natural alignment.
    /// Slice index for the detail normal map in the non-colour atlas.
    /// `< 0` = no detail normal (pass-through). Default -1.
    public var detailNormalTextureSlice: Int32 = -1
    /// Tile frequency of the detail normal relative to the macro UV.
    /// 8 = eight tiles per macro tile = fine grain/pore detail.
    public var detailNormalUVScale: Float = 8.0
    /// Phase 7c — anisotropy [0,1] (was `_padDetail0`, same offset/stride). Stretches the specular
    /// highlight along the surface grain; packed as (1 + anisotropy) into normalRoughness.w.
    public var anisotropy: Float = 0
    /// Highlight mode for the screen-space halo pass: 0 none · 1 selected (blue
    /// halo) · 2 hover (yellow halo). Only honoured on bounding-box proxies fed
    /// through `highlightMaskInstances`; the detailed scene meshes never set it.
    public var highlight: Int32 = 0
    // ── Drag/impact sway (vertex-shader secondary motion) ─────────────────────
    // A generic, GPU-side rigid bend driven by the host `DragSwayTracker`: the
    // sibling of `applyTreeWind` for non-foliage objects. The shader rotates the
    // vertex about the instance's world bottom-pivot + local-Z axis (both read off
    // `modelMatrix`, so no extra per-instance pivot data), then lifts it by jostle.
    // `swayMode 0` is a hard no-op, so every instance that doesn't opt in is
    // untouched. New 16-byte cluster (offsets 224-239): stride 224 → 240.
    /// Sway kind: 0 none · 1 bottom-pivot lean (books, upright shelf contents) ·
    /// 2 top-pivot pendulum (a hanging pendant: rigid rotation about the model origin —
    /// the ceiling anchor — self-oscillating from the frame time; see `applySway` in
    /// Illuminatorama.metal) · 3 wind hinge flap (rotation about the model origin's local +Z,
    /// one-sided and gusty, scaled on the GPU by the frame's `treeWindStrength` — a toilet seat
    /// or lid the wind lifts off its rest).
    public var swayMode: Int32 = 0
    /// Mode 1: static lean angle (radians) about local-Z, pivoting at the base.
    /// Mode 2: pendulum AMPLITUDE (max swing, radians) — the shader animates the swing
    /// from the frame time, so the host sets this once (no per-frame drive).
    /// Mode 3: the flap at full wind (radians), met square-on by it; its SIGN is the direction it
    /// lifts (> 0 lifts the free edge up and back — it catches wind blowing into that edge).
    public var swayLean: Float = 0
    /// Vertical pop (metres) added in world space — a knock hops the object up.
    public var swayJostle: Float = 0
    /// Light-layer bitfield (mirrors the Metal `Instance.layer`, was `_padSway0`).
    /// Written into the `gLayer` G-buffer target; the deferred lighting kernel keeps
    /// a light only when `(light.layerMask & layer) != 0`. Default `0xFFFFFFFF` ⇒
    /// every light affects this instance (byte-identical to prior behaviour). Same
    /// 4 bytes as the former float pad, so the stride stays 240.
    public var layer: UInt32 = 0xFFFF_FFFF
    // ── Anti-tiling suitability (per instance) ────────────────────────────────
    // Multiplier on the frame's global `antiTilingStrength` for THIS instance's
    // hex-stochastic samples. The de-repeat blend is only valid for STOCHASTIC
    // textures — on a coherent pattern it superimposes misaligned copies: a tile
    // grid double-prints its grout lines and directional wood grain dices into a
    // patchwork quilt (both Danny-reported). Hosts set 0 for regular/directional
    // materials (tile, wallpaper, wood planks, brick) and leave 1 elsewhere.
    // NEW 16-byte cluster (offsets 240-255): stride 240 → 256; three float pads.
    public var antiTilingScale: Float = 1
    // ── S2.4 — the detail band's diffuse-visible companions ────────────────────
    // Reinterpret the former `_padAntiTiling0/1` (same 4 bytes each, stride stays 256).
    //
    // **The measured problem.** Before this, the detail band carried a NORMAL and nothing
    // else. A whole-library A/B on the real Metal path (2026-08-02) put the grazing
    // micro-contrast ratio at matte-black 3.44 / aluminum 1.43 / brushed-nickel 1.41 and
    // **~1.00 on all 38 dielectrics** — velvet 1.01, cork 1.02, paint 1.00, linen 1.00.
    // A fine normal perturbation is a specular-band effect; a matte surface's broad diffuse
    // lobe plus rough specular lobe average it straight out. More frequency didn't help
    // (a 5× coarser sweep moved velvet to 1.06).
    //
    // **What these read.** The detail-normal slice's BLUE channel, which the G-buffer shader
    // has always fetched and never used (it blends `tangentN.xy + dn.xy` and reconstructs
    // nothing from z). Hosts bake a micro-occlusion there from the same height field the
    // detail normal comes from — so no new atlas slice, no extra texture fetch for the value
    // itself, and no G-buffer channel (both terms are applied in the G-buffer FRAGMENT, on
    // the albedo and roughness it was going to write anyway; `normalRoughness.w` stays the
    // material-class tag it already is).

    /// How strongly the detail band's micro-occlusion multiplies **albedo**, i.e. the
    /// DIFFUSE lobe — the half of the BRDF a matte dielectric can actually show. 0 (default)
    /// ⇒ `mix(1, occ, 0)` is exactly 1 ⇒ no change. 1 ⇒ the baked occlusion at full depth.
    public var detailOcclusionStrength: Float = 0
    /// How strongly occluded (pit) texels are additionally **roughened** — micro-cavities
    /// scatter more widely than the open surface between them. Added as
    /// `roughness + strength × (1 − occ)`. 0 (default) ⇒ exact no-op.
    ///
    /// Unlike the normal, this survives mip filtering: occlusion is a non-negative quantity
    /// whose coarse levels converge to its MEAN, so a distant surface keeps the roughening
    /// (it just loses the contrast), which is the Toksvig-correct behaviour.
    public var detailRoughnessStrength: Float = 0
    /// Tile frequency for the OCCLUSION tap, independent of `detailNormalUVScale`
    /// (reinterprets the former `_padAntiTiling2`). `0` ⇒ fall back to `detailNormalUVScale`.
    ///
    /// **Why it can't just share the normal's frequency — measured, not assumed.** At the
    /// normal's 8× on a 2 m macro tile, one detail texel is ~0.5 mm and one relief feature is
    /// ~2 mm: an order of magnitude below a pixel at any archviz camera. Sampled mip-correctly
    /// that band returns a flat mean and shows nothing (the first S2.4 build measured velvet
    /// 1.09 → 1.12, cork 1.01 → 1.04 — i.e. nothing). Sampling it sharp instead would only
    /// reproduce the detail normal's own defect: its 1.4–3.4× on metals is *sub-pixel aliasing*,
    /// which is why S1.1 found that mip-correct gradients collapse aluminum to 1.0002×.
    ///
    /// So the occlusion runs at a coarser, actually resolvable band (cm-scale — plaster tooth,
    /// cork granules, weave), while the normal keeps its fine one. Same baked tile, two
    /// frequencies: no extra memory, one extra fetch.
    public var detailOcclusionUVScale: Float = 0

    // ── S2.5 half 2 — per-PATTERN-CELL value jitter ───────────────────────────
    // New 16-byte cluster (offsets 256-271): stride 256 → 272.
    //
    // The de-repeat for the COHERENT categories, which the hex blend cannot serve. A hex
    // tap offsets the UV, which superimposes misaligned copies of a regular pattern (grout
    // double-prints, plank grain dices), so those materials ship with `antiTilingScale = 0`
    // and had no de-repeat at all. The right one for them is what a real tiled floor has:
    // unit-to-unit VALUE variation.
    //
    // `uv` counts up across the whole surface — only the atlas LOOKUP wraps — so
    // `floor(uv * patternCells)` is a unique index per physical tile/plank over the entire
    // floor even though the texture repeats every UV unit. Hashing that index to a small
    // tone multiplier breaks the macro repeat **without displacing any UV** and **without a
    // single extra texture tap**.

    /// Pattern cells per macro UV tile, per axis — e.g. a 4×4 tile grid baked into one UV
    /// tile is `(4, 4)`; a 16-plank floor whose planks run unbroken along v is `(16, 0)`.
    ///
    /// A component of **0** means "no subdivision on this axis": the cell index there is
    /// constant, so a plank keeps ONE tone down its whole length and varies board to board.
    /// `(0, 0)` (the default) disables the effect entirely.
    ///
    /// It must be the pattern's real cell count and nothing else. A count of 1 would be a
    /// per-UV-TILE hash — a ~2 m checkerboard with hard seams, i.e. one visible grid swapped
    /// for a coarser and uglier one. Hosts take this from the generator argument that drew
    /// the grid; deriving it a second time downstream is how the two drift apart.
    public var patternCells: SIMD2<Float> = .zero
    /// Half-amplitude of the per-cell multiplicative tone jitter (0.05 ⇒ ±5 %). 0 (default)
    /// is an exact no-op, so Visualizer and any host that never opts in are byte-identical.
    /// Real tile/plank batch variation lives around 3–6 %.
    public var patternJitter: Float = 0
    // ── Vegetation-wind opt-in (per instance) ─────────────────────────────────
    // Reinterprets the former `_padPattern0` trailing pad — same 4 bytes, stride stays 272.
    //
    // **The defect this closes.** `applyTreeWind` runs in `illumi_vs`, the vertex stage of
    // EVERY ordinary instanced draw, and its only per-vertex gate was `v.tangent.x >= 0.0001`.
    // That is not a vegetation marker: `tangent` is the TBN basis every normal-mapped surface
    // carries, and a host that builds tangents from world axes — Daydream Home's
    // `HouseScene.vertex` emits `(1,0,0,1)` for every floor/ceiling and ±Z wall face — hands
    // the shader `sway = 1.0`, the weight of an outermost canopy tip. The whole house then
    // swings on the gust, while its ±X faces (tangent `(0,0,1,1)`, `sway = 0`) stay planted,
    // so meshes visibly tear along their own face-orientation boundaries. The bug was latent
    // for as long as the feature was inert (nothing set `treeWindStrength`) and shipped the
    // day a host finally switched it on.
    //
    // A vertex attribute could never have been the gate, because `tangent` has a prior,
    // universal meaning. The opt-in has to live where the intent does: the draw.
    //
    /// Per-instance multiplier on the frame's global vegetation wind strength
    /// (`renderer.treeWindStrength`). **0 (the default) is an exact shader no-op**, so wind
    /// reaches only the draws whose host declared them vegetation — foliage meshes that
    /// deliberately pack `(swayWeight, phase, flutter)` into their vertex tangents
    /// (`ForestGeometry`/`ForestTreeGeometry` canopies, `GrassFillerMesh` cards). Everything
    /// else — house geometry, furniture, props, and every host that never opts in — is
    /// untouched no matter what its tangents contain.
    ///
    /// It is a scale, not a flag, so a background stand can sway less than a hero tree
    /// without a second global.
    public var windScale: Float = 0

    /// **World-space wood knots** — see `sampleWoodKnots` in `IlluminatoramaMaterial.h` for the
    /// full contract and for why a knot cannot be a baked texel or a decal quad.
    ///
    ///   x = lattice cells per UV unit (0 = disabled — the default, and an exact no-op)
    ///   y = knot radius in UV units
    ///   z = darkness of the core against the wood it grew through, [0, 1]
    ///   w = fraction of lattice cells that carry a knot
    public var woodKnots: SIMD4<Float> = .zero

    // ── Per-INSTANCE (per-object) macro material variation ─────────────────────
    // NEW 16-byte cluster (offsets 288-303): two floats + 8 bytes pad, stride 288 -> 304.
    //
    // The MACRO tier of the material system. Two placed pieces that wear the SAME baked
    // material id share ONE atlas slice and ONE registered mesh, so without a per-instance
    // channel they render the identical texels through the identical UVs — "two nightstands
    // are the same pixels twice". The meso/micro bands (grain, pores, detail relief) live in
    // the baked texture and so are shared too; only a per-INSTANCE term can break the
    // object-to-object repeat.
    //
    // `macroTone` is an ACHROMATIC multiplier on the FINAL albedo (AFTER the atlas sample and
    // the per-vertex colour), so it works whether the material is textured or flat and shifts
    // the piece LIGHTER/DARKER without ever changing hue (a hue shift would read as a different
    // finish, not the same finish cut from a different board). `macroRoughnessDelta` is added
    // to the resolved roughness. Both identity by default — 1 and 0 — so Visualizer and every
    // instance that never opts in is byte-identical. Hosts seed these off the element's own
    // stable id (never `hashValue`, which reseeds per process), so the variation is document
    // state: stable across save/load and identical in the app and in every headless capture.
    public var macroTone: Float = 1
    public var macroRoughnessDelta: Float = 0

    /// **Per-INSTANCE UV phase** (DH-0475) — a translation, in tile-UV units, added to the UV that
    /// every MATERIAL sample reads (the same `matUV` the wood-knot warp displaces), so two placed
    /// pieces that share ONE baked slice and ONE mesh don't show the same GRAIN figure in the same
    /// place. `macroTone`/`macroRoughnessDelta` shift a twin lighter/darker or rougher; this shifts
    /// WHICH part of the tiling material it's cut from — the same finish, a different board.
    ///
    /// It rides in the 8 bytes that were trailing padding, so the struct stride is UNCHANGED (304)
    /// and `.zero` (the default) is an exact shader no-op: every non-opting instance — every
    /// Visualizer scene — is byte-identical. Only translation, never rotation: furniture wood grain
    /// is directional, so an offset relocates the figure while keeping the grain running along the
    /// board, whereas a rotation would twist it off-axis and read as a defect. Because the material
    /// tiles seamlessly under the hardware `repeat` sampler, any offset is seam-free.
    public var uvPhase: SIMD2<Float> = .zero

    /// **Per-material cloth-sheen ROUGHNESS** (DH-0081) — the WIDTH of the sheen nap, the sibling of
    /// `sheen` (its strength). Low = a crisp grazing highlight (sateen); high = a broad soft glow
    /// (velvet, wool). `0.30` (the default) is the single constant the lobe shipped with, so an
    /// instance that never sets it — every Visualizer scene, every non-cloth surface — renders
    /// exactly as before. The G-buffer snaps it to the nearest of a few curated bands and folds the
    /// band into the INTEGER part of the sign-multiplexed `sheen` emission.alpha channel (band 0 ⇒
    /// the value packs `-sheen` byte-for-byte as it did before this existed). Only read where
    /// `sheen > 0`. NEW 16-byte cluster (offsets 304-319): stride 304 → 320.
    public var sheenRoughness: Float = 0.30

    /// **Carpet pile-lay tone bands** (DH-0472) — a low-frequency achromatic value field the
    /// G-buffer draws on the UNWRAPPED uv (`sampleCarpetMacro` in `IlluminatoramaMaterial.h`), the
    /// metre-scale nap variation a ~0.30 m carpet tile physically cannot carry. Rides in the 8 bytes
    /// of trailing pad after `sheenRoughness` (a float2 aligns to offset 312), so the struct stride
    /// stays 320 and `.zero` — the default and every non-carpet instance — is an exact shader no-op.
    ///   x = band lattice cells per UV unit (0 = disabled)
    ///   y = ± tone amplitude
    /// Mirrors `Instance.carpetMacro` in `IlluminatoramaCommon.h`.
    public var carpetMacro: SIMD2<Float> = .zero

    /// DH-0478 — slice index (non-colour atlas) of the per-texel grain-direction map, encoded as
    /// (cos 2θ, sin 2θ) in RG, θ from the material's U axis. `< 0` = none: the anisotropy lobe keeps
    /// its per-instance axis. Read ONLY by the photo-lane G-buffer variant (`kExtendedGBuffer`),
    /// which re-expresses it as a world-space angle in `material.gb` for the lighting kernel. NEW
    /// 16-byte cluster (offsets 320-335): stride 320 → 336. Every host that never sets it — every
    /// Visualizer scene — is byte-identical (the field is never read on the live lane).
    public var grainTangentTextureSlice: Int32 = -1
    /// DH-0140 slice 3 — thin-sheet translucency [0,1]: the fraction of light reaching the FAR face
    /// that comes through diffusely (milky polycarbonate ≈ 0.5, a woven shade ≈ 0.3). Honoured only
    /// in the still (the photo-lane G-buffer carries it in `material.r`; the lighting kernel adds
    /// `albedo × transmitted incident × t`); the live canvas renders the sheet opaque as it always
    /// has. 0 = opaque — the default, and byte-identical for every host that never sets it.
    public var thinTransmission: Float = 0
    /// DH-0947 — the per-pixel leaf VENATION the G-buffer draws from a leaf's own flat blade
    /// coordinates, which the host packs into `uv` as (10.5 + x, y) in blade lengths
    /// (VisualizerVegetation `LeafSheet.bladeUVOrigin` / `PlantStyle.leafVenation`). 0 = none — the
    /// default, and an exact no-op for every host that never sets it; 1 = a pinnate, looping fig.
    /// Was `_padGrain1`: same 4 bytes, the stride is unchanged. Mirrors `Instance.leafVenation`.
    public var leafVenation: Int32 = 0
    /// **DH-1275 — surface wear**: amplitude of a fine, world-space scuff/streak field drawn per
    /// pixel in the G-buffer (thin streaks a few mm across, cm long; faded out analytically when
    /// the pixel footprint can no longer resolve them, so it never aliases and never becomes a
    /// blotch). It darkens albedo a few % and roughens along the streak. 0 (default) ⇒ exact
    /// no-op. Was `_padGrain2` — same 4 bytes, stride unchanged. Mirrors `Instance.surfaceWear`.
    public var surfaceWear: Float = 0

    // ── Animated UV DOMAIN WARP (per instance) ────────────────────────────────
    // NEW 16-byte cluster (offsets 336-351): stride 336 → 352. (Landed at 288 on a
    // parallel line; moved after the DH macro/sheen/grain clusters in the merge.)
    //
    // Offsets the UV the material is sampled at, as a smooth function of the UV
    // itself and of time — a domain warp, not a scroll. A scroll translates the
    // whole map rigidly; warping the DOMAIN makes the pattern breathe and
    // wobble in place, which is what "the emission slowly morphs" asks for.
    //
    // The alternative in a host without this is to cross-dissolve two pre-baked
    // variants, which can only ever swap between states it baked in advance.
    // A warp is continuous and costs no atlas slices.
    //
    // Applied ONLY where it is asked for, and `x = 0` is an exact no-op, so
    // every existing instance is untouched.
    //
    //   x = amplitude in UV units (0 = disabled — the default)
    //   y = spatial frequency: how many wobbles across one UV unit
    //   z = temporal speed in radians/second
    //   w = target: 0 = emission only · 1 = albedo + emission
    public var uvWarp: SIMD4<Float> = .zero

    /// **Animated hue cycle of TAGGED vertex colour** (offsets 352–367; stride 352 → 368). The
    /// G-buffer rotates the vertex colour's hue on vertices whose `tangent.w` is ≈ 1 (0.5…1.5) — a host tag;
    /// foliage wind meshes pack (sway, phase, flutter, tag) in tangent — so a flower's petals
    /// change colour while leaves and stems do not, animated on the GPU by frame time: set once,
    /// no per-frame instance upload.
    ///   x = strength 0…1 (0 = off: the default and an exact no-op for every existing host)
    ///   y = phase (radians) · z = speed (radians/second)
    ///   w = spatial wave (radians per metre along world (x, z)·(0.6, 0.8)) — a travelling band
    public var hueCycle: SIMD4<Float> = .zero

    /// **Glow of TAGGED vertex colour** (offsets 368–383; stride 368 → 384). Same host tag as
    /// `hueCycle` (vertex `tangent.w` ≈ 1): a tagged vertex adds its own final albedo (after the
    /// hue cycle) × `x` × the renderer's `tagGlowGain` to its emission, so a flower's petals glow
    /// in their own — possibly cycling — colour while its leaves and stems do not. `x` is a
    /// STATIC weight set once at assembly; the live amount is the frame-level gain, so gliding
    /// or modulating a glow costs one float per frame, not an instance re-upload.
    ///   x = glow weight (0 = off: the default and an exact no-op for every existing host)
    ///   y = FIELD-SINK depth (m): how far the instance slides below ground when the renderer's
    ///       `fieldSinkProgress` is 1 (0 = never sinks — the default, an exact no-op)
    ///   z = field-sink stagger key (> 0: its fraction is the delay, shared by an object's parts
    ///       placed with different matrices; 0 = hash the instance's origin) · w reserved
    public var tagGlow: SIMD4<Float> = .zero

    /// Signack S6 (DH-1461) — radians added to a vegetation mesh's baked wind phase (trunk sway,
    /// flutter and every limb's φ), so the many instances of ONE baked plant in a planting sway out
    /// of step. Rides in `tagGlow.w` (unused until now); 0 — every existing instance — is a no-op.
    public var windPhaseOffset: Float {
        get { tagGlow.w }
        set { tagGlow.w = newValue }
    }

    public init(
        modelMatrix: simd_float4x4,
        albedo: SIMD3<Float> = SIMD3(0.8, 0.8, 0.8),
        metallic: Float = 0.0,
        roughness: Float = 0.5,
        emission: SIMD3<Float> = .zero,
        albedoTextureSlice: Int32 = -1,
        metallicTextureSlice: Int32 = -1,
        roughnessTextureSlice: Int32 = -1,
        normalTextureSlice: Int32 = -1,
        emissionTextureSlice: Int32 = -1,
        emissionIntensity: Float = 1.0
    ) {
        self.modelMatrix = modelMatrix
        self.normalMatrix = Self.normalMatrix(from: modelMatrix)
        self.albedo = albedo
        self.metallic = metallic
        self.roughness = roughness
        self.emission = emission
        self.albedoTextureSlice = albedoTextureSlice
        self.metallicTextureSlice = metallicTextureSlice
        self.roughnessTextureSlice = roughnessTextureSlice
        self.normalTextureSlice = normalTextureSlice
        self.emissionTextureSlice = emissionTextureSlice
        self.emissionIntensity = emissionIntensity
    }

    public mutating func setTransform(_ m: simd_float4x4) {
        self.modelMatrix = m
        self.normalMatrix = Self.normalMatrix(from: m)
    }

    /// Compile-time guard: Swift and Metal structs must agree on 384 bytes.
    /// If this fires, either a Swift field was added without the matching Metal
    /// field (or vice versa), or alignment changed unexpectedly.
    static let _assertStride240: Void = { assert(MemoryLayout<IlluminatoramaInstance>.stride == 384, "IlluminatoramaInstance stride must be 384") }()

    // ── Perfect analytic superquadric impostor — per-instance GPU param ────────
    //
    // Rides in a buffer PARALLEL to the instance buffer (NOT widened into
    // `IlluminatoramaInstance` — that would re-trigger the SIMD3 stride-16 vs
    // packed_float3 hazard and grow the 208 B stride for every instance engine-
    // wide). Indexed by the same grouped instance id as `instances[iid]`. Mirrors
    // `SQParam` in Illuminatorama.metal byte-for-byte (stride 80, 16-aligned).
    struct SuperquadricParam {
        var invModel: simd_float4x4    // world → object (= inverse(modelMatrix))
        var shape: SIMD4<Float>        // x=e1, y=e2, z=isEllipsoid(1/0), w=0
    }

    /// Inverse-transpose of the upper 3×3, padded out to float4x4 so we can
    /// memcpy into a shader struct without a separate `float3x3` codepath.
    /// Metal's `float3x3` has tricky alignment; float4x4 is unambiguous.
    static func normalMatrix(from model: simd_float4x4) -> simd_float4x4 {
        let upper = simd_float3x3(
            SIMD3(model.columns.0.x, model.columns.0.y, model.columns.0.z),
            SIMD3(model.columns.1.x, model.columns.1.y, model.columns.1.z),
            SIMD3(model.columns.2.x, model.columns.2.y, model.columns.2.z)
        )
        let inv = upper.inverse.transpose
        return simd_float4x4(
            SIMD4(inv.columns.0, 0),
            SIMD4(inv.columns.1, 0),
            SIMD4(inv.columns.2, 0),
            SIMD4(0, 0, 0, 1)
        )
    }
}

// ── AAA ray-traced glass (#60) ───────────────────────────────────────────────


// ── Equatable (scene-content stability gate) ────────────────────────────────
//
// The renderer keeps last frame's `instances` / flattened glass and compares
// per frame to detect a static scene (camera-only motion): when nothing
// changed it skips the instance-buffer regroup/rewrite, the RT TLAS refit,
// and (when light params also held) the spot/point shadow re-renders. Both
// structs are plain SIMD/scalar bags, so the synthesized memberwise `==`
// compiles to tight vector compares — cheap enough to run on every frame.
extension IlluminatoramaInstance: Equatable {}
